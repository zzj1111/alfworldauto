"""Free signals: everything the Teacher sees, read from the rollouts training already
wrote. No measurement pass, no GPU.

The recorder (scaffold_env_manager.py) appends one JSONL row per training episode:
  {uid, task_type, gamefile, injected, success, steps: [{a, o, v}]}
`injected` is the coin the manager actually flipped, never reconstructed from (seed, p).
`a` is the EXECUTED action (the projection's output), not the raw generation.

The window is one cycle: byte offsets taken around that cycle's training. Counts are
raw episode counts; nothing is carried across cycles and nothing is smoothed. The
prompt (prompts.py) states exactly this, and test_alignment keeps the two in step.
"""
from __future__ import annotations

import collections
import json
import os
import re

from .scaffold import CATEGORIES

TRACES_PER_SIDE = 3       # 3 failures + 3 successes per category
MAX_TRACE_STEPS = 50      # an episode is capped at 50 env steps anyway


def log_offset(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_rows(path, offset=0):
    """Rows appended since `offset`. A truncated final line (training mid-write) is
    dropped rather than raising."""
    rows = []
    try:
        with open(path) as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def _groups(rows):
    """uid -> {task, injected, gamefile, rows}. Injection is per group by construction;
    a mixed group would mean the recorder or the coin is broken, so it is surfaced."""
    g = collections.OrderedDict()
    mixed = set()
    for r in rows:
        uid = r.get("uid")
        if uid is None or r.get("success") is None:
            continue
        e = g.setdefault(uid, {"task": r.get("task_type"), "injected": bool(r.get("injected")),
                               "gamefile": r.get("gamefile"), "rows": []})
        if bool(r.get("injected")) != e["injected"]:
            mixed.add(uid)
        e["rows"].append(r)
    return g, mixed


def per_task_gap(groups, scaffold=None):
    """Success split by injected vs bare, per category, RAW counts from this window."""
    acc = {c: {"bare": [0, 0], "injected": [0, 0]} for c in CATEGORIES}
    for e in groups.values():
        cat = e["task"]
        if cat not in acc:
            continue
        side = "injected" if e["injected"] else "bare"
        for r in e["rows"]:
            acc[cat][side][0] += 1 if float(r.get("success") or 0) > 0 else 0
            acc[cat][side][1] += 1
    out = {}
    for cat, d in acc.items():
        b_s, b_n = d["bare"]
        i_s, i_n = d["injected"]
        row = {"bare": round(b_s / b_n, 4) if b_n else None, "n_bare": b_n,
               "injected": round(i_s / i_n, 4) if i_n else None, "n_injected": i_n,
               "gap": round(i_s / i_n - b_s / b_n, 4) if (b_n and i_n) else None}
        if not i_n and scaffold is not None:
            from .scaffold import items_of
            has_text = bool(items_of(scaffold, "general") or items_of(scaffold, cat))
            p = float((scaffold.get("p_task") or {}).get(cat, 0.0) or 0.0)
            if not has_text:
                row["no_injection_reason"] = "no scaffold text applies to this category"
            elif p <= 0:
                row["no_injection_reason"] = f"p_task[{cat}]=0"
            else:
                row["no_injection_reason"] = (f"p_task[{cat}]={p} but no group fired "
                                              f"this window")
        out[cat] = row
    return out


def zero_gradient_groups(groups, rollout_n):
    """Groups contributing NO gradient, per category. The advantage is reward minus the
    group mean, so the condition is no variance — all fail OR all succeed. Complete
    groups only: a partial group (truncated log) says nothing about variance."""
    out = {}
    for e in groups.values():
        cat = e["task"]
        if cat not in CATEGORIES or len(e["rows"]) != rollout_n:
            continue
        z = out.setdefault(cat, {"total": 0, "zero_gradient": 0,
                                 "all_fail": 0, "all_succeed": 0,
                                 "rollout_n": rollout_n, "unit": "one game's rollouts"})
        outs = [1 if float(r.get("success") or 0) > 0 else 0 for r in e["rows"]]
        z["total"] += 1
        if len(set(outs)) == 1:
            z["zero_gradient"] += 1
            z["all_fail" if outs[0] == 0 else "all_succeed"] += 1
    return out


def _trim_steps(row):
    steps = (row.get("steps") or [])[:MAX_TRACE_STEPS]
    return {"gamefile": row.get("gamefile"), "success": row.get("success"),
            "n_steps": len(row.get("steps") or []),
            "steps": [{"a": s.get("a"), "o": s.get("o"), "v": s.get("v", True)}
                      for s in steps]}


def contrastive_traces(groups, rollout_n, per_side=TRACES_PER_SIDE):
    """Per category: failures from all-fail groups beside the category's shortest
    successes. All-fail groups are the zero-gradient mass the scaffold exists to fix;
    they have no in-group success to pair with, so the contrast is cross-category.

    Failure entry = the longest FAILED rollout of the group — never a successful one,
    even when topping up from low-success groups near convergence."""
    by_cat = collections.defaultdict(list)
    for e in groups.values():
        if e["task"] in CATEGORIES:
            by_cat[e["task"]].append(e)
    out = {}
    for cat, es in by_cat.items():
        complete = [e for e in es if len(e["rows"]) == rollout_n]
        rate = lambda e: sum(1 for r in e["rows"] if float(r.get("success") or 0) > 0) / len(e["rows"])
        all_fail = [e for e in complete if rate(e) == 0]
        pool = all_fail + sorted((e for e in complete if rate(e) > 0), key=rate)

        def failure_of(e):
            losses = [r for r in e["rows"]
                      if r.get("steps") and float(r.get("success") or 0) <= 0]
            if not losses:
                return None
            return _trim_steps(max(losses, key=lambda r: len(r["steps"])))

        fails, skipped_no_failures = [], 0
        for e in pool:
            if len(fails) >= per_side:
                break
            f = failure_of(e)
            if f is None:
                skipped_no_failures += 1
                continue
            fails.append(f)

        successes = [r for e in es for r in e["rows"]
                     if float(r.get("success") or 0) > 0 and r.get("steps")]
        succs = [_trim_steps(r) for r in
                 sorted(successes, key=lambda r: len(r["steps"]))[:per_side]]

        entry = {"n_groups": len(complete), "n_all_fail_groups": len(all_fail),
                 "zero_gradient_failures": fails, "successes_same_category": succs}
        if not fails and skipped_no_failures:
            entry["no_failures_to_show"] = ("every group in this category succeeded at "
                                            "least once; nothing failing to display")
        out[cat] = entry
    return out


_WS = re.compile(r"\s+")


def failure_patterns(rows):
    """Rule-computed counters over ALL failed rollouts — not an impression from the few
    traces that fit in the prompt, and they survive any trimming those traces get."""
    per_cat = collections.defaultdict(lambda: {"n_failed": 0, "repeated_action": 0,
                                               "looped_observation": 0, "invalid_heavy": 0,
                                               "example_repeated": None})
    for r in rows:
        if float(r.get("success") or 0) > 0:
            continue
        steps = r.get("steps") or []
        cat = r.get("task_type")
        if cat not in CATEGORIES or not steps:
            continue
        d = per_cat[cat]
        d["n_failed"] += 1
        acts = [_WS.sub(" ", str(s.get("a") or "")).strip().lower() for s in steps]
        counts = collections.Counter(a for a in acts if a)
        if counts:
            top, n = counts.most_common(1)[0]
            if n >= 3 and n >= len(steps) / 2:
                d["repeated_action"] += 1
                d["example_repeated"] = d["example_repeated"] or top
        obs = [str(s.get("o") or "") for s in steps]
        run = best = 1
        for a, b in zip(obs, obs[1:]):
            run = run + 1 if a == b and a else 1
            best = max(best, run)
        if best >= 3:
            d["looped_observation"] += 1
        invalid = sum(1 for s in steps if not s.get("v", True))
        if invalid >= 0.2 * len(steps):
            d["invalid_heavy"] += 1
    return dict(per_cat)


def signals_from_rows(rows, rollout_n, scaffold=None):
    """The full packet the observation builder consumes."""
    groups, mixed = _groups(rows)
    out = {
        "per_task_gap": per_task_gap(groups, scaffold),
        "zero_gradient_groups": zero_gradient_groups(groups, rollout_n),
        "contrastive_traces": contrastive_traces(groups, rollout_n),
        "failure_patterns": failure_patterns(rows),
        "n_episodes": sum(len(e["rows"]) for e in groups.values()),
        "n_groups": len(groups),
    }
    if mixed:
        # Injection is decided once per group; a split group means the recorder or the
        # coin is broken and every gap number above is suspect. Loud, not fatal.
        out["mixed_injection_groups"] = sorted(mixed)[:10]
    return out
