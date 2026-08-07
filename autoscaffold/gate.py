"""The A/B gate: accept or reject a proposed TEXT change. Pure function on a
measurement; no I/O, no GPU.

The rule is candidate mean > current mean over the touched categories, strict, no
margin (locked 2026-08-05). ARM_AB_NOISE_K restores a standard-error margin for runs
where the false-accept rate matters more than proposal throughput.

Per-category tournaments (user decision 2026-08-07, third revision): the measurement
covers the UNION of the categories the current scaffold reaches and the proposal
touches, and EACH category is judged on its own numbers (~30 episodes per condition
at full breadth — the noise cost of that n is accepted by decision):

  per category: ACCEPT the candidate's edits iff candidate > current and >= bare;
                else REVERT (clear that category's own items) iff bare strictly
                beats both; else KEEP.

General-scoped text reaches every category, so general edits and general items are
judged by the same rule on the AGGREGATE over the union — the general text's actual
audience. The aggregate fields also remain the headline record.
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
    # Per-category breakdown INTO THE RECORD. The decision stays aggregate (per-
    # category n is ~30 at full breadth and three-way verdicts flip on 2-3 games at
    # that n), but a mixed pattern — candidate best in one category, bare best in
    # another — must be visible to the Teacher, or it cannot target the losing
    # category with a free precise delete next cycle.
    per_category = {}
    for t in tasks:
        b = measure.get("bare", {}).get(t)
        c = measure.get("current", {}).get(t)
        k = measure.get("candidate", {}).get(t)
        if not (b and c and k):
            continue
        m_t = NOISE_K * ((c[0] * (1 - c[0]) / c[1]) + (k[0] * (1 - k[0]) / k[1])) ** 0.5             if c[1] and k[1] else 0.0
        acc_t = k[0] > c[0] + m_t and k[0] >= b[0]
        rev_t = (not acc_t) and b[0] > c[0] and b[0] > k[0]
        best = max(("bare", b[0]), ("current", c[0]), ("candidate", k[0]),
                   key=lambda x: x[1])[0]
        per_category[t] = {"bare": round(float(b[0]), 3),
                           "current": round(float(c[0]), 3),
                           "candidate": round(float(k[0]), 3),
                           "n": int(b[1]), "best": best,
                           "verdict": "accept" if acc_t else ("revert" if rev_t else "keep")}
    mixed = len({v["verdict"] for v in per_category.values()}) > 1
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
    if mixed:
        reason += ("  [per-category verdicts differ — each category's edits and items "
                   "are acted on by its own verdict; general text follows the aggregate]")
    # bool()/float() everywhere: the measurement often arrives as numpy scalars, and
    # a verdict that cannot be json.dumps'd kills the journal write of the very cycle
    # that paid for the A/B
    return {"accept": bool(accept), "revert_to_bare": bool(revert_to_bare),
            "per_category": per_category, "mixed_verdict": bool(mixed),
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
