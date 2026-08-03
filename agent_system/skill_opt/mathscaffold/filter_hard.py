"""Stage 2 — the hard-problem filter that defines D, still BEFORE any scaffold.

OC-GRPO (arXiv 2607.19313, App. G):

    "We identify problems on which the base model pi_ref receives no learning signal under
     standard GRPO. For each problem x, we generate M = 64 independent rollouts from the
     model under training using nucleus sampling, and retain those problems for which EVERY
     rollout fails. This filter yields 595 hard problems for the 7B model. Similarly, for the
     3B and 1.5B models, we repeat this filtering process to identify model-specific hard
     problems."

So D is model-specific: run this once per base model. A problem is kept iff pass@64 == 0 —
these are exactly the problems where every GRPO group is all-fail, the group reward variance
is zero, and the gradient vanishes. That is the learning cliff our scaffold exists to break.

We also write the pass rate for EVERY problem, not just the kept ones. p(x) = pass@64 is the
quantity the Teacher needs to reason about difficulty per instance, and keeping it lets the
hard-set threshold be revisited without re-running 64 x N generations.

Resumable: results are appended per shard, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os

from .verify import score

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_chat(tok, problem):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}],
        tokenize=False, add_generation_prompt=True)


def pass_rate(completions, gold):
    """Fraction of the M rollouts whose boxed answer verifies against `gold`."""
    if not completions:
        return 0.0
    return sum(score(c, gold) for c in completions) / len(completions)


def load_done(path):
    """uids already scored in a previous (possibly interrupted) run."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["uid"]] = r
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/mnt/data1/zha00175/math_scaffold_data/math_l345_pool.parquet")
    ap.add_argument("--out-dir", default="/mnt/data1/zha00175/math_scaffold_data")
    ap.add_argument("--model", default="/mnt/data1/zha00175/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tag", default=None, help="output suffix; defaults to the model dir name")
    ap.add_argument("--m", type=int, default=64, help="rollouts per problem (paper: 64)")
    ap.add_argument("--temperature", type=float, default=0.7)   # paper's rollout temperature
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpus", default="2", help="CUDA_VISIBLE_DEVICES for this filter run")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N problems")
    ap.add_argument("--chunk", type=int, default=128, help="problems per generate() call")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    os.environ.setdefault("HF_HOME", "/mnt/data1/zha00175/hf_home")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tag = a.tag or os.path.basename(a.model.rstrip("/"))
    os.makedirs(a.out_dir, exist_ok=True)
    scored_path = f"{a.out_dir}/passrate_{tag}.jsonl"

    rows = pd.read_parquet(a.pool).to_dict("records")
    if a.limit:
        rows = rows[:a.limit]
    done = load_done(scored_path)
    todo = [r for r in rows if r["uid"] not in done]
    print(f"[filter] {len(rows)} problems | already scored {len(done)} | to do {len(todo)} "
          f"| M={a.m} model={tag}", flush=True)

    if todo:
        tok = AutoTokenizer.from_pretrained(a.model)
        llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
                  max_model_len=4096, enforce_eager=False, seed=0)
        sp = SamplingParams(n=a.m, temperature=a.temperature, top_p=a.top_p,
                            max_tokens=a.max_tokens)
        with open(scored_path, "a") as out:
            for i in range(0, len(todo), a.chunk):
                batch = todo[i:i + a.chunk]
                prompts = [build_chat(tok, r["problem"]) for r in batch]
                gens = llm.generate(prompts, sp)
                for r, g in zip(batch, gens):
                    comps = [o.text for o in g.outputs]
                    pr = pass_rate(comps, r["answer"])
                    out.write(json.dumps({"uid": r["uid"], "pass_rate": pr,
                                          "n": len(comps)}) + "\n")
                out.flush()
                print(f"[filter] {min(i + a.chunk, len(todo))}/{len(todo)}", flush=True)

    # ---- assemble D (pass@M == 0) -------------------------------------------------
    scored = load_done(scored_path)
    pool = pd.read_parquet(a.pool)
    pool["pass_rate"] = pool["uid"].map(lambda u: scored.get(u, {}).get("pass_rate"))
    covered = pool.dropna(subset=["pass_rate"])
    hard = covered[covered["pass_rate"] == 0.0].drop(columns=["pass_rate"])

    hard.to_parquet(f"{a.out_dir}/math_hard_{tag}.parquet", index=False)
    covered.to_parquet(f"{a.out_dir}/math_passrate_{tag}.parquet", index=False)

    from collections import Counter
    print(f"\n[filter] scored {len(covered)}/{len(pool)} problems")
    print(f"[filter] HARD (pass@{a.m} == 0): {len(hard)}   <- this is D, before any scaffold")
    print(f"[filter] hard by level: {dict(Counter(hard['level']))}")
    print(f"[filter] hard by type : {dict(Counter(hard['type']))}")
    print(f"[filter] wrote {a.out_dir}/math_hard_{tag}.parquet")


if __name__ == "__main__":
    main()
