"""The A/B gate: accept or reject a proposed TEXT change. Pure function on a
measurement; no I/O, no GPU.

The rule is candidate mean > current mean over the touched categories, strict, no
margin (locked 2026-08-05). ARM_AB_NOISE_K restores a standard-error margin for runs
where the false-accept rate matters more than proposal throughput.

The bare condition is a FLOOR for acceptance (user decision 2026-08-07): a candidate
must beat the current scaffold AND score no lower than no text at all. Text that loses
to nothing has no theory of benefit — injecting it makes injected groups worse, which
is the opposite of the mechanism. When bare also outscores the CURRENT scaffold, that
is recorded as bare_beats_current: the measured best edit is deletion, which is the
Teacher's to make (the A/B samples only the touched categories, so the harness cannot
justify clearing scopes the measurement never saw).
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
                "bare_mean": round(bare_mean, 3), "blocked_by_bare_floor": False,
                "bare_beats_current": False, "n": 0}
    margin = NOISE_K * ((cur_mean * (1 - cur_mean) / cur_n)
                        + (cand_mean * (1 - cand_mean) / cand_n)) ** 0.5
    beats_current = cand_mean > cur_mean + margin
    blocked_by_bare_floor = bool(beats_current and cand_mean < bare_mean)
    accept = beats_current and not blocked_by_bare_floor
    bare_beats_current = bool(bare_mean > cur_mean)
    reason = (f"candidate {cand_mean:.3f} vs current {cur_mean:.3f} "
              f"(bare {bare_mean:.3f}, margin {margin:.3f}, n {cur_n}+{cand_n}) "
              f"-> {'ACCEPT' if accept else 'reject'}")
    if blocked_by_bare_floor:
        reason += ("  [bare floor: beats the current scaffold but loses to NO TEXT; "
                   "such text never enters training]")
    if bare_beats_current and not accept:
        reason += ("  [note: the no-text condition outscored the CURRENT scaffold on the "
                   "touched categories — deletion is the measured best edit]")
    return {"accept": accept, "reason": reason, "cand_mean": round(cand_mean, 3),
            "cur_mean": round(cur_mean, 3), "bare_mean": round(bare_mean, 3),
            "blocked_by_bare_floor": blocked_by_bare_floor,
            "bare_beats_current": bare_beats_current,
            "margin": round(margin, 4), "n": cur_n + cand_n}


def update_best(best, best_step, sr, step):
    if sr is not None and (best is None or sr > best):
        return sr, step
    return best, best_step
