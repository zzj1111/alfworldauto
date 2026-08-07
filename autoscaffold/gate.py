"""The A/B gate: accept or reject a proposed TEXT change. Pure function on a
measurement; no I/O, no GPU.

The rule is candidate mean > current mean over the touched categories, strict, no
margin (locked 2026-08-05). ARM_AB_NOISE_K restores a standard-error margin for runs
where the false-accept rate matters more than proposal throughput.

Three-way tournament (user decision 2026-08-07, second revision): the measurement
covers the UNION of the categories the current scaffold reaches and the proposal
touches, so every category holding text is measured, and:

  1. ACCEPT the candidate iff it beats the current scaffold and scores no lower
     than no text at all (text losing to nothing has no benefit path);
  2. else REVERT TO BARE iff no-text strictly beats BOTH current and candidate —
     the union scope is what makes clearing all text a measurement-supported act;
  3. else REJECT and keep the current scaffold.
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
        return {"accept": False, "revert_to_bare": False,
                "reason": "no touched categories (nothing to A/B)", "n": 0}
    cur_mean, cur_n = _weighted_mean(measure.get("current", {}), tasks)
    cand_mean, cand_n = _weighted_mean(measure.get("candidate", {}), tasks)
    bare_mean, _ = _weighted_mean(measure.get("bare", {}), tasks)
    if not cur_n or not cand_n:
        return {"accept": False, "revert_to_bare": False,
                "reason": "missing A/B samples -> reject (keep current)",
                "cand_mean": round(cand_mean, 3), "cur_mean": round(cur_mean, 3),
                "bare_mean": round(bare_mean, 3),
                "bare_beats_current": False, "n": 0}
    margin = NOISE_K * ((cur_mean * (1 - cur_mean) / cur_n)
                        + (cand_mean * (1 - cand_mean) / cand_n)) ** 0.5
    beats_current = cand_mean > cur_mean + margin
    accept = bool(beats_current and cand_mean >= bare_mean)
    revert_to_bare = bool(not accept and bare_mean > cur_mean and bare_mean > cand_mean)
    bare_beats_current = bool(bare_mean > cur_mean)
    action = "ACCEPT" if accept else ("REVERT-TO-BARE" if revert_to_bare else "reject")
    reason = (f"candidate {cand_mean:.3f} vs current {cur_mean:.3f} "
              f"(bare {bare_mean:.3f}, margin {margin:.3f}, n {cur_n}+{cand_n}) "
              f"-> {action}")
    if revert_to_bare:
        reason += ("  [no text strictly beats both the current scaffold and the "
                   "candidate over the measured union of categories; all items are "
                   "cleared]")
    elif not accept and beats_current:
        reason += ("  [bare floor: beats the current scaffold but loses to NO TEXT "
                   "without NO TEXT beating current — kept as is]")
    # bool()/float() everywhere: the measurement often arrives as numpy scalars, and
    # a verdict that cannot be json.dumps'd kills the journal write of the very cycle
    # that paid for the A/B
    return {"accept": bool(accept), "revert_to_bare": bool(revert_to_bare),
            "reason": reason,
            "cand_mean": round(float(cand_mean), 3),
            "cur_mean": round(float(cur_mean), 3),
            "bare_mean": round(float(bare_mean), 3),
            "bare_beats_current": bool(bare_beats_current),
            "margin": round(float(margin), 4), "n": int(cur_n + cand_n)}


def update_best(best, best_step, sr, step):
    if sr is not None and (best is None or sr > best):
        return sr, step
    return best, best_step
