"""QuestA data pipeline, replicated exactly, plus the fields our Teacher needs.

Upstream is three tiny scripts (QuestA/AReaL/datasets/): `add_prefix.py` builds the hinted
prompt, `process.py` renames fields, `convert2hf.py` saves to disk. This module reproduces
`add_prefix.py` BYTE-FOR-BYTE (verified against the original in the tests) and emits verl
parquet instead of AReaL's on-disk HF format.

Three quirks of the original that are easy to "fix" by accident and must NOT be:
  1. The cut is by CHARACTER, with no word-boundary snapping. (The paper says the ratio is
     over tokens; the code is characters. We follow the code.)
  2. The hint marker is exactly ``'## Hint.'`` concatenated with no separator — not the
     ``## Hint: Partial Solution`` shown in the paper figure.
  3. Rows whose final answer string does not literally occur in the solution block are
     DROPPED, and a prefix shorter than 10 characters degrades the row to no hint at all.

What we add on top (never changing the above): a stable `uid`, and the raw `problem` and
`solution` carried through, so a per-instance alpha decided by the Teacher can rebuild the
prompt at training time instead of the ratio being frozen into the dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

SYSTEM_HINT_MARKER = "## Hint."


def uid_of(problem):
    """Stable per-problem id, so alpha decisions join to rows across files and cycles."""
    return hashlib.sha1(problem.strip().encode("utf-8")).hexdigest()[:16]


def solution_block(generation):
    """The graded solution, i.e. what follows the reasoning trace.

    Mirrors add_prefix.py: strip one layer of surrounding double quotes if the field was
    serialized that way, then keep everything after the LAST `</think>`.
    """
    text = generation or ""
    if text[:1] == '"':
        text = text[1:-1]
    return text.split("</think>")[-1]


def split_prefix(text, scale):
    """add_prefix.py::split_prefix — raw character cut, no word-boundary snapping."""
    length = len(text)
    length *= scale
    return text[: int(length)], text[int(length):]


def build_prompt(problem, prefix):
    """add_prefix.py's prompt assembly, including the <10-character degradation."""
    if len(prefix) < 10:
        return problem + "\n\n"
    return problem + "\n\n" + SYSTEM_HINT_MARKER + prefix


def build_rows(records, ratio):
    """Replicate add_prefix.py over parsed jsonl records. -> (rows, stats).

    ratio is an INTEGER percentage (50 / 25), matching the original CLI.
    """
    rows, dropped_answer, degraded = [], 0, 0
    for i, d in enumerate(records):
        sol = solution_block(d.get("generation"))
        final_answer = d.get("answer")
        if final_answer is None or str(final_answer) not in sol:
            dropped_answer += 1                      # original: `continue`
            continue
        prefix, _ = split_prefix(sol, ratio / 100)
        if len(prefix) < 10:
            degraded += 1
        rows.append({
            "query_id": str(i),                      # original keys the row by line index
            "uid": uid_of(d["problem"]),             # ours: stable across ratios and files
            "problem": d["problem"],
            "solution": sol,                         # ours: lets alpha be chosen at train time
            "answer": str(final_answer),
            "prompt": build_prompt(d["problem"], prefix),
            "prefix": prefix,
            "ratio": ratio,
        })
    return rows, {"in": len(records), "kept": len(rows),
                  "dropped_answer_not_in_solution": dropped_answer,
                  "degraded_prefix_lt_10_chars": degraded}


def read_jsonl(path):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue                              # original prints and skips
    return out


# ------------------------------- verl views ------------------------------- #
def to_verl(row, bare=False, data_source="questa_math"):
    """One verl training row.

    bare=True emits the UNHINTED prompt (the no-guidance arm and the standalone eval).
    `extra_info` always carries uid / problem / solution / answer so the scaffold layer can
    rebuild the prompt for any alpha without touching this file again.
    """
    content = row["problem"] + "\n\n" if bare else row["prompt"]
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["answer"]},
        "extra_info": {
            "uid": row["uid"], "query_id": row["query_id"],
            "problem": row["problem"], "solution": row["solution"],
            "answer": row["answer"], "ratio": 0 if bare else row["ratio"],
            "split": "train",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="/mnt/data1/zha00175/questa_data")
    ap.add_argument("--out-dir", default="/mnt/data1/zha00175/questa_scaffold_data")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="held-out fraction carved off the p=50 pool for the standalone anchor")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import pandas as pd

    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}
    pools = {}
    for tag, fname, ratio in [("p50", "OpenR1-50-0-4.jsonl", 50),
                              ("p25", "OpenR1-25-0-4.jsonl", 25)]:
        recs = read_jsonl(f"{a.src_dir}/{fname}")
        rows, stats = build_rows(recs, ratio)
        pools[tag] = rows
        summary[tag] = stats
        print(f"[{tag}] {stats}")

    # UNION pool. The two released files are difficulty-filtered at DIFFERENT ratios (p=50
    # keeps 1,853 problems, p=25 keeps 10,653, and 96% of the former sit inside the latter),
    # so neither alone is the right problem set for an arm whose ratio is decided per instance.
    # We key by uid, keep the first occurrence, and prefer the p50 copy when a problem is in
    # both — identical `solution`, but it keeps provenance stable.
    union = {}
    for tag in ("p50", "p25"):
        for r in pools[tag]:
            union.setdefault(r["uid"], {**r, "pool": tag})
    union_rows = list(union.values())

    # Held-out split BY UID over the UNION. The pools contain the same problem up to twice, so
    # a row-wise split leaks: one copy lands in val while its twin stays in train.
    import random
    rng = random.Random(a.seed)
    uids = sorted(union)
    rng.shuffle(uids)
    n_val = max(64, int(len(uids) * a.val_frac))
    val_uids = set(uids[:n_val])
    val_rows = [union[u] for u in uids[:n_val]]

    train_union = [r for r in union_rows if r["uid"] not in val_uids]
    train50 = [r for r in pools["p50"] if r["uid"] not in val_uids]
    train25 = [r for r in pools["p25"] if r["uid"] not in val_uids]

    def at_ratio(rows, ratio):
        """Re-cut every row's prefix at a FIXED ratio (used for the controlled A1 arm, where
        all arms must draw from the same union pool and only the schedule differs)."""
        out = []
        for r in rows:
            pre, _ = split_prefix(r["solution"], ratio / 100)
            out.append({**r, "prefix": pre, "ratio": ratio,
                        "prompt": build_prompt(r["problem"], pre)})
        return out

    out = {
        # --- faithful QuestA arms: the released per-ratio pools, ratio frozen in ---------
        "train_p50.parquet":  [to_verl(r) for r in train50],
        "train_p25.parquet":  [to_verl(r) for r in train25],
        # --- union pool: the SAME problems for every arm, only the schedule differs ------
        "union_bare.parquet": [to_verl(r, bare=True) for r in train_union],
        "union_p50.parquet":  [to_verl(r) for r in at_ratio(train_union, 50)],
        "union_p25.parquet":  [to_verl(r) for r in at_ratio(train_union, 25)],
        # --- standalone anchor: ALWAYS unhinted, held out of everything above ------------
        "val.parquet":        [to_verl(r, bare=True) for r in val_rows],
    }
    for name, rows in out.items():
        pd.DataFrame(rows).to_parquet(f"{a.out_dir}/{name}", index=False)
        print(f"  wrote {name:22s} {len(rows):6d} rows")

    pd.DataFrame(train_union).to_parquet(f"{a.out_dir}/pool_union.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{a.out_dir}/pool_val.parquet", index=False)
    summary["union"] = {"unique_uids": len(union), "train": len(train_union), "val": len(val_rows)}
    with open(f"{a.out_dir}/stats.json", "w") as f:
        json.dump({**summary, "n_val": len(val_rows), "seed": a.seed}, f, indent=2)
    print(f"\nunion: {len(union)} unique problems | train {len(train_union)} | val {len(val_rows)}")
    print(f"dir: {a.out_dir}")


if __name__ == "__main__":
    main()
