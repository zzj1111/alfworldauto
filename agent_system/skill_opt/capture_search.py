"""Capture standalone (no-scaffold) rollouts for the search agent, for the auto-scaffold
controller. BATCHED: all questions advance turn-by-turn together (vllm batch generate +
concurrent retriever calls).

Emits the controller's OBSERVATION (no priors, just measurements):
  - per-data_source standalone accuracy (the objective, at finest granularity so the
    controller can bucket freely)
  - full FAILURE trajectories, balanced per source (not first-N global)
  - failure-mode counts + a few successes for contrast
  - (optional --counterfactual) with-scaffold vs no-scaffold per-source accuracy, by
    running the SAME questions a second time with the CURRENT scaffold injected.
"""
import argparse, json, re, sys, string
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")

RETRIEVE_URL = "http://127.0.0.1:8010/retrieve"


def retrieve(query, topk=3):
    try:
        body = json.dumps({"query": query, "topk": topk, "return_scores": False}).encode()
        req = urllib.request.Request(RETRIEVE_URL, data=body, headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        docs = r["result"][0]
        return [d["document"]["contents"] if isinstance(d, dict) and "document" in d else str(d) for d in docs]
    except Exception as e:
        return [f"(retriever error: {e})"]


def norm(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the")).strip()


def cover_em(pred, golds):
    p = norm(pred)
    return any(norm(g) and norm(g) in p for g in golds)


def build_prompt(question, history, skill_block=""):
    """Base search prompt; when skill_block is non-empty, splice it EXACTLY as training
    does (splice_skill), so the with-scaffold pass matches the injected-training prompt."""
    from agent_system.environments.prompts.search import SEARCH_TEMPLATE_NO_HIS, SEARCH_TEMPLATE
    from agent_system.skill.skill_store import splice_skill
    if not history:
        p = SEARCH_TEMPLATE_NO_HIS.format(task_description=question)
    else:
        mem = "\n".join(f"<search>{h['q']}</search>\n<information>{h['info']}</information>" for h in history)
        p = SEARCH_TEMPLATE.format(task_description=question, memory_context=mem, step_count=len(history))
    return splice_skill(p, skill_block) if skill_block else p


def parse_action(text):
    ms = re.search(r"<search>(.*?)</search>", text, re.DOTALL | re.IGNORECASE)
    ma = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if ms and (not ma or ms.start() < ma.start()):
        return ("search", ms.group(1).strip())
    if ma:
        return ("answer", ma.group(1).strip())
    return (None, "")


def run_pass(items, llm, tok, sp, max_steps, skill_of=None):
    """Advance a fresh set of episodes to completion. `skill_of(ds)->block` injects the
    scaffold (with-scaffold pass); None -> standalone. Returns per-episode records."""
    eps = [{"q": q, "gold": g, "ds": ds, "history": [], "final": "",
            "turns": [], "done": False, "reason": "budget_exhausted_no_answer"}
           for (q, g, ds) in items]

    def chat(p):
        return tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful and harmless assistant."},
             {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    for t in range(max_steps):
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        prompts = [chat(build_prompt(e["q"], e["history"], skill_of(e["ds"]) if skill_of else "")) for e in active]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        to_search = []
        for e, o in zip(active, outs):
            txt = o.outputs[0].text
            if "<search>" in txt and "</search>" not in txt: txt += "</search>"
            if "<answer>" in txt and "</answer>" not in txt: txt += "</answer>"
            act, arg = parse_action(txt)
            if act == "search":
                to_search.append((e, arg, txt))
            elif act == "answer":
                e["final"] = arg; e["done"] = True; e["reason"] = "answered"
                e["turns"].append({"turn": t + 1, "model_output": txt[:600], "answer": arg})
            else:
                e["done"] = True; e["reason"] = "malformed_no_action"
                e["turns"].append({"turn": t + 1, "malformed_output": txt[:600]})
        if to_search:
            with ThreadPoolExecutor(max_workers=32) as ex:
                docs_list = list(ex.map(lambda x: retrieve(x[1]), to_search))
            for (e, arg, txt), docs in zip(to_search, docs_list):
                info = " || ".join(d[:400] for d in docs)
                e["history"].append({"q": arg, "info": info})
                e["turns"].append({"turn": t + 1, "model_output": txt[:600],
                                   "search_query": arg, "retrieved": info[:900]})
    return eps


def per_source_acc(eps):
    agg = defaultdict(lambda: [0, 0])  # ds -> [correct, n]
    for e in eps:
        agg[e["ds"]][1] += 1
        if cover_em(e["final"], e["gold"]):
            agg[e["ds"]][0] += 1
    return {ds: {"n": n, "correct": c, "acc": round(c / max(1, n), 4)} for ds, (c, n) in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)          # questions to run (batched)
    ap.add_argument("--max_fail", type=int, default=84)    # total failures kept (balanced per source)
    ap.add_argument("--max_ok", type=int, default=8)       # successes kept for contrast
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--counterfactual", default="")        # scaffold JSON path -> run with-scaffold pass too
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet("/home/zha00175/data/searchR1_processed_direct/train.parquet")
    df = df.sample(n=min(args.n, len(df)), random_state=13).reset_index(drop=True)

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
        if t is None: return []
        return list(t) if not isinstance(t, str) else [t]

    items = [(get_q(r), get_gold(r), r.get("data_source", "?")) for _, r in df.iterrows()]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    llm = LLM(model=args.ckpt, gpu_memory_utilization=0.5, dtype="bfloat16", max_model_len=8192, enforce_eager=True)
    sp = SamplingParams(temperature=0.4, max_tokens=512, stop=["</search>", "</answer>"])

    # ---- Pass 1: standalone (the objective) ----
    eps = run_pass(items, llm, tok, sp, args.max_steps, skill_of=None)

    # ---- Pass 2 (optional): with the CURRENT scaffold injected -> counterfactual ----
    counterfactual = {}
    if args.counterfactual:
        try:
            from agent_system.skill.skill_store import SkillStore
            store = SkillStore.from_json(args.counterfactual, mode="full")
            block = {ds: store.render(store.search_key(ds)) for ds in {i[2] for i in items}}
            scaf_eps = run_pass(items, llm, tok, sp, args.max_steps, skill_of=lambda ds: block.get(ds, ""))
            base, scaf = per_source_acc(eps), per_source_acc(scaf_eps)
            counterfactual = {ds: {"no_scaffold_acc": base.get(ds, {}).get("acc"),
                                   "with_scaffold_acc": scaf.get(ds, {}).get("acc")} for ds in base}
        except Exception as e:
            counterfactual = {"error": str(e)[:200]}

    # ---- Aggregate: per-source acc, balanced failures, successes, modes ----
    n_sources = len({e["ds"] for e in eps}) or 1
    cap_per = max(1, args.max_fail // n_sources)
    per_src_fail = defaultdict(list)
    oks, n_correct = [], 0
    for e in eps:
        rec = {"data_source": e["ds"], "question": e["q"], "gold": e["gold"],
               "final_answer": e["final"], "n_search": len(e["history"]),
               "reason": e["reason"], "trajectory": e["turns"]}
        if cover_em(e["final"], e["gold"]):
            n_correct += 1
            if len(oks) < args.max_ok: oks.append(rec)
        elif len(per_src_fail[e["ds"]]) < cap_per:
            per_src_fail[e["ds"]].append(rec)
    failures = [f for lst in per_src_fail.values() for f in lst]
    modes = Counter(e["reason"] for e in eps if not cover_em(e["final"], e["gold"]))

    json.dump({"n_run": len(eps), "n_correct": n_correct, "acc": round(n_correct / max(1, len(eps)), 4),
               "per_source": per_source_acc(eps), "counterfactual": counterfactual,
               "failure_modes": dict(modes), "n_fail_collected": len(failures),
               "failures": failures, "successes_for_contrast": oks},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"CAPTURE_DONE n_run={len(eps)} correct={n_correct} acc={n_correct/max(1,len(eps)):.3f} "
          f"failures_kept={len(failures)}/{n_sources}src modes={dict(modes)} "
          f"cf={'yes' if args.counterfactual else 'no'} -> {args.out}")


if __name__ == "__main__":
    main()
