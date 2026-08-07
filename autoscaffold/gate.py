"""The A/B gate: accept or reject a proposed TEXT change. Pure function on a
measurement; no I/O, no GPU.

The rule is candidate mean > current mean over the touched categories, strict, no
margin (locked 2026-08-05). ARM_AB_NOISE_K restores a standard-error margin for runs
where the false-accept rate matters more than proposal throughput.

The bare condition does not enter the rule, but it is measured on the same games and
can say what the rule cannot: an accepted candidate scoring BELOW bare is text that
loses to no text, made permanent (nothing rewinds a scaffold). Flagged, not vetoed.
"""
from __future__ import annotations

import os

NOISE_K = float(os.environ.get("ARM_AB_NOISE_K", "0"))


def _weighted_mean(per_task, tasks):
    num = den = 0
    for t in tasks:
        if t in per_task:
            s, n = per_task[t]
            num += s * n
            den += n
    return (num / den if den else 0.0), den


def ab_gate(measure, tasks):
    """measure: {"bare"|"current"|"candidate": {task: (success_rate, n)}} — the frozen
    policy on held-out games, three ways, SAME games across conditions (paired).
    Returns {accept, reason, cand_mean, cur_mean, bare_mean, below_bare, margin, n}."""
    if not tasks:
        return {"accept": False, "reason": "no touched categories (nothing to A/B)", "n": 0}
    cur_mean, cur_n = _weighted_mean(measure.get("current", {}), tasks)
    cand_mean, cand_n = _weighted_mean(measure.get("candidate", {}), tasks)
    bare_mean, _ = _weighted_mean(measure.get("bare", {}), tasks)
    if not cur_n or not cand_n:
        return {"accept": False, "reason": "missing A/B samples -> reject (keep current)",
                "cand_mean": round(cand_mean, 3), "cur_mean": round(cur_mean, 3),
                "bare_mean": round(bare_mean, 3), "below_bare": False, "n": 0}
    margin = NOISE_K * ((cur_mean * (1 - cur_mean) / cur_n)
                        + (cand_mean * (1 - cand_mean) / cand_n)) ** 0.5
    accept = cand_mean > cur_mean + margin
    below_bare = bool(accept and cand_mean < bare_mean)
    reason = (f"candidate {cand_mean:.3f} vs current {cur_mean:.3f} "
              f"(bare {bare_mean:.3f}, margin {margin:.3f}, n {cur_n}+{cand_n}) "
              f"-> {'ACCEPT' if accept else 'reject'}"
              + ("  [WARNING: accepted text scores BELOW the no-text condition]"
                 if below_bare else ""))
    return {"accept": accept, "reason": reason, "cand_mean": round(cand_mean, 3),
            "cur_mean": round(cur_mean, 3), "bare_mean": round(bare_mean, 3),
            "below_bare": below_bare, "margin": round(margin, 4), "n": cur_n + cand_n}


def update_best(best, best_step, sr, step):
    if sr is not None and (best is None or sr > best):
        return sr, step
    return best, best_step
