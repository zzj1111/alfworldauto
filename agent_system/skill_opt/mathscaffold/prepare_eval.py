"""Stage 3 — the held-out evaluation sets, matched to OC-GRPO (arXiv 2607.19313).

They report Pass@1 and Pass@16 on AIME (1983-2026), Gaokao2023 and OmniMath, with 16
trajectories per problem and the unbiased pass@k estimator. Using the same three benchmarks
is what makes our numbers directly comparable to their Tables 2-4.

Each source is tried in order of preference and skipped (loudly) if unavailable, so a missing
mirror never blocks the rest. Output is one parquet per benchmark with a uniform schema:
    uid, problem, answer, source
"""
from __future__ import annotations

import argparse
import json
import os

from .prepare_math import SYSTEM_PROMPT, uid_of
from .verify import extract_boxed

# (benchmark, [candidate HF repos], field guesses)
SOURCES = {
    "aime": (["gneubig/aime-1983-2024", "di-zhang-fdu/AIME_1983_2024",
              "HuggingFaceH4/aime_2024"],
             ("Question", "problem", "question"), ("Answer", "answer", "solution")),
    "gaokao2023": (["MARIO-Math-Reasoning/Gaokao2023-Math-En", "gaokao2023en"],
                   ("question", "problem"), ("answer", "final_answer")),
    "omnimath": (["KbsdJames/Omni-MATH"],
                 ("problem", "question"), ("answer", "final_answer", "solution")),
}


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def normalize_rows(rows, qkeys, akeys, source):
    """Uniform {uid, problem, answer, source}; answers given as full solutions are reduced to
    their boxed content so the verifier compares answer-to-answer, not answer-to-prose."""
    out, dropped = [], 0
    for r in rows:
        q, a = _pick(r, qkeys), _pick(r, akeys)
        if not q or a is None:
            dropped += 1
            continue
        a = str(a).strip()
        if "\\boxed" in a:
            a = extract_boxed(a) or a
        out.append({"uid": uid_of(q), "problem": q, "answer": a, "source": source})
    return out, dropped


def to_verl(row):
    return {
        "data_source": row["source"],
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": row["problem"]}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["answer"]},
        "extra_info": {"uid": row["uid"], "source": row["source"], "split": "test"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/data1/zha00175/math_scaffold_data/eval")
    ap.add_argument("--only", default="", help="comma-separated subset of: aime,gaokao2023,omnimath")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/mnt/data1/zha00175/hf_home")
    import datasets
    import pandas as pd

    os.makedirs(a.out_dir, exist_ok=True)
    want = [x for x in (a.only.split(",") if a.only else SOURCES) if x in SOURCES]
    summary = {}
    for name in want:
        repos, qkeys, akeys = SOURCES[name]
        rows = None
        for repo in repos:
            try:
                d = datasets.load_dataset(repo, trust_remote_code=True)
                split = "test" if "test" in d else list(d.keys())[0]
                rows, dropped = normalize_rows(list(d[split]), qkeys, akeys, name)
                print(f"[{name}] {repo} split={split}: {len(rows)} kept, {dropped} dropped")
                break
            except Exception as e:
                print(f"[{name}] {repo} unavailable: {type(e).__name__}: {str(e)[:120]}")
        if not rows:
            print(f"[{name}] SKIPPED — no source available")
            continue
        pd.DataFrame(rows).to_parquet(f"{a.out_dir}/{name}.parquet", index=False)
        pd.DataFrame([to_verl(r) for r in rows]).to_parquet(f"{a.out_dir}/{name}_verl.parquet",
                                                            index=False)
        summary[name] = len(rows)
    print("\nwrote:", json.dumps(summary, indent=2))
    print(f"dir: {a.out_dir}")


if __name__ == "__main__":
    main()
