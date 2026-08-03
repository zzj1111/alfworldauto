"""Stage 1 — build the base MATH pool, BEFORE any scaffold/guidance is attached.

Mirrors the dataset construction in OC-GRPO (arXiv 2607.19313, App. G):

    "We construct our training set from the MATH dataset (Hendrycks et al., 2021),
     restricting to problems at difficulty Levels 3-5 (intermediate through advanced),
     which yields 5,586 problems from the training split."

Difference from the copy already on disk (`~/data/math_drgrpo/train.parquet`, also 5,586
rows): that one kept only the final answer. We KEEP THE FULL REFERENCE SOLUTION, because the
whole reason to move the auto-scaffold to math is that ground-truth solutions exist here —
they are what makes `Domain(has_reference_solutions=True, instance_scope=True)` real.

Output columns (one row per problem, no guidance attached):
    problem, solution, answer, level, type, uid
Plus a verl-style `prompt`/`reward_model`/`extra_info` view so the same parquet can be fed
straight to the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from .verify import extract_boxed

HF_DATASET = "DigitalLearningGmbH/MATH-lighteval"
LEVELS = ("Level 3", "Level 4", "Level 5")
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def uid_of(problem):
    """Stable per-problem id: lets guidance, hard-filter results and scaffold text be joined
    across files without depending on row order."""
    return hashlib.sha1(problem.strip().encode("utf-8")).hexdigest()[:16]


def build_rows(split_rows, levels=LEVELS):
    """Filter to `levels` and attach the gold answer parsed out of the reference solution.

    Rows whose reference solution has no parsable `\\boxed{}` are dropped: without a gold
    answer the verifier cannot score them, so they would silently contribute zero-reward
    noise to every group.
    """
    out, dropped = [], 0
    for r in split_rows:
        if r.get("level") not in levels:
            continue
        sol = r.get("solution") or ""
        ans = extract_boxed(sol)
        if not ans:
            dropped += 1
            continue
        prob = r["problem"]
        out.append({
            "uid": uid_of(prob),
            "problem": prob,
            "solution": sol,                 # full reference solution — the privileged signal
            "answer": ans,
            "level": r.get("level"),
            "type": r.get("type"),
        })
    return out, dropped


def to_verl(row, data_source="math_scaffold"):
    """verl-trainer view of one row. `extra_info` carries everything the scaffold layer needs
    (uid to key per-instance text, solution as the privileged signal, level/type as structure)."""
    return {
        "data_source": data_source,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": row["problem"]}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["answer"]},
        "extra_info": {"uid": row["uid"], "level": row["level"], "type": row["type"],
                       "solution": row["solution"], "split": "train"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/data1/zha00175/math_scaffold_data")
    ap.add_argument("--hf-dataset", default=HF_DATASET)
    ap.add_argument("--expect", type=int, default=5586,
                    help="paper's count for MATH L3-5 train; 0 disables the check")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/mnt/data1/zha00175/hf_home")
    import datasets
    import pandas as pd

    ds = datasets.load_dataset(a.hf_dataset, trust_remote_code=True)
    rows, dropped = build_rows(list(ds["train"]))
    print(f"MATH train: {len(ds['train'])} -> L3-5 with parsable answer: {len(rows)} "
          f"(dropped {dropped} with no \\boxed in the reference solution)")
    if a.expect and len(rows) + dropped != a.expect:
        print(f"WARNING: L3-5 count {len(rows) + dropped} != paper's {a.expect}")

    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f"{a.out_dir}/math_l345_pool.parquet", index=False)
    pd.DataFrame([to_verl(r) for r in rows]).to_parquet(f"{a.out_dir}/math_l345_verl.parquet",
                                                        index=False)
    with open(f"{a.out_dir}/math_l345_sample.json", "w") as f:
        json.dump(rows[:3], f, indent=2, ensure_ascii=False)

    from collections import Counter
    print("levels:", dict(Counter(r["level"] for r in rows)))
    print("types :", dict(Counter(r["type"] for r in rows)))
    print(f"wrote {a.out_dir}/math_l345_pool.parquet  (+ _verl view, + _sample.json)")


if __name__ == "__main__":
    main()
