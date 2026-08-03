"""pass@k evaluation for Search-R1 checkpoints -- the CAPABILITY-BOUNDARY probe.

Motivated by the pass@k analysis of "Does RL Really Incentivize Reasoning Capacity
Beyond the Base Model?": RL can lift pass@1 (sharpen) while pass@k at large k stays at
the base model's level (no boundary expansion). For an "auto-SCAFFOLD" claim we must show
the scaffold expands the boundary, i.e. pass@128(scaffolded-RL) > pass@128(plain RL) >=
pass@128(base). This script draws K samples per question (temperature sampling) and reports
the UNBIASED pass@k for k in {1,2,4,...,128}, overall and per data_source.

STANDALONE by default (no scaffold injected) -- the objective is the bare model's boundary.
Reuses the batched multi-turn rollout from capture_search.py.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")
from agent_system.skill_opt.capture_search import run_pass, cover_em


def pass_at_k(n, c, k):
    """Unbiased estimator (Chen et al. 2021), numerically stable product form.
    n = samples drawn, c = # correct, k = budget."""
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)              # hf dir or base model path
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)         # distinct questions
    ap.add_argument("--k", type=int, default=128)         # samples per question
    ap.add_argument("--temp", type=float, default=1.0)    # high temp -> coverage (boundary)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--chunk_q", type=int, default=16)    # questions per batch (bounds retriever load: chunk_q*k concurrent)
    ap.add_argument("--split", default="train")           # train | test
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--scaffold", default="")             # optional: inject this scaffold (WITH-scaffold pass@k)
    args = ap.parse_args()

    import pandas as pd
    path = ("/home/zha00175/data/searchR1_processed_direct/train.parquet" if args.split == "train"
            else "/mnt/data1/zha00175/searchR1_data/test_subset.parquet")
    df = pd.read_parquet(path).sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    def get_q(row):
        ei = row.get("extra_info")
        if isinstance(ei, dict) and ei.get("question"):
            return ei["question"]
        for m in reversed(list(row["prompt"])):
            if m.get("role") == "user":
                return m["content"]
        return row["prompt"][-1]["content"]

    def get_gold(row):
        rm = row["reward_model"]
        t = rm["ground_truth"]["target"] if isinstance(rm, dict) else None
        return [] if t is None else (list(t) if not isinstance(t, str) else [t])

    questions = [(get_q(r), get_gold(r), r.get("data_source", "?")) for _, r in df.iterrows()]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    llm = LLM(model=args.ckpt, gpu_memory_utilization=0.6, dtype="bfloat16", max_model_len=8192, enforce_eager=True)
    sp = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=512, stop=["</search>", "</answer>"])

    skill_of = None
    if args.scaffold:
        from agent_system.skill.skill_store import SkillStore
        store = SkillStore.from_json(args.scaffold, mode="full")
        block = {ds: store.render(store.search_key(ds)) for _, _, ds in questions}
        skill_of = lambda ds: block.get(ds, "")

    # c[qid] = # correct out of k; ds_of[qid] = data_source
    c = defaultdict(int); ds_of = {}
    for qi, chunk in enumerate(chunks(questions, args.chunk_q)):
        flat, qids = [], []
        for j, (q, gold, ds) in enumerate(chunk):
            qid = qi * args.chunk_q + j
            ds_of[qid] = ds
            for _ in range(args.k):
                flat.append((q, gold, ds)); qids.append(qid)
        eps = run_pass(flat, llm, tok, sp, args.max_steps, skill_of=skill_of)
        for e, qid in zip(eps, qids):
            if cover_em(e["final"], e["gold"]):
                c[qid] += 1
        print(f"  chunk {qi+1}: {len(chunk)} q x{args.k} done", flush=True)

    ks = [x for x in [1, 2, 4, 8, 16, 32, 64, 128, 256] if x <= args.k]
    qids = list(ds_of.keys())

    def agg(subset):
        return {k: round(float(np.mean([pass_at_k(args.k, c[q], k) for q in subset])), 4) for k in ks} if subset else {}

    per_source = {}
    for ds in sorted(set(ds_of.values())):
        per_source[ds] = agg([q for q in qids if ds_of[q] == ds])

    out = {"ckpt": args.ckpt, "n_questions": len(qids), "k_samples": args.k, "temp": args.temp,
           "with_scaffold": bool(args.scaffold), "pass_at_k": agg(qids), "per_source": per_source}
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"PASSK_DONE n={len(qids)} k={args.k} temp={args.temp} "
          f"pass@1={out['pass_at_k'].get(1)} pass@{max(ks)}={out['pass_at_k'].get(max(ks))} -> {args.out}")


if __name__ == "__main__":
    main()
