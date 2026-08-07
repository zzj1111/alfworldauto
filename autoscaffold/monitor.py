"""Per-cycle snapshots to three sinks, and the status report.

wandb is the convenient view and also the sink most likely to BE the broken thing (no
credential, no network, wrong project) — and a run that is healthy-but-invisible looks
exactly like a dead one from outside. status.json (atomic, latest) and metrics.jsonl
(append, complete history) depend on nothing.

warnings_for encodes the known ways a run looks healthy while producing nothing.
"""
from __future__ import annotations

import json
import os
import time

from . import config as C
from . import scaffold as S


def cycle_snapshot(state, cfg):
    out = {}
    step = int(state.get("step") or 0)
    target = int(cfg.get("target_step") or 0)
    out["progress/step"] = step
    out["progress/cycle"] = int(state.get("cycle") or 0)
    if target:
        out["progress/target_step"] = target
        out["progress/frac"] = round(step / target, 4)

    sr_hist = state.get("sr_history") or []
    if sr_hist:
        out["valid_seen/success_rate"] = sr_hist[-1]
        if len(sr_hist) >= 2:
            out["valid_seen/delta"] = round(sr_hist[-1] - sr_hist[-2], 4)
    ev = state.get("last_eval") or {}
    draws = ev.get("draws") or []
    if len(draws) >= 2:
        out["valid_seen/draw_spread"] = round(max(draws) - min(draws), 4)
        out["valid_seen/n_draws"] = len(draws)
    for t, v in (ev.get("per_task") or {}).items():
        out[f"valid_seen/{t}"] = v
    if state.get("best") is not None:
        out["valid_seen/best"] = state["best"]
        out["valid_seen/best_step"] = state.get("best_step")

    tr = (state.get("train_rollouts") or [{}])[-1].get("by_category") or {}
    n_all = sum(v.get("n", 0) for v in tr.values())
    if n_all:
        out["train/success_rate"] = round(
            sum(v.get("n_correct", 0) for v in tr.values()) / n_all, 4)
        out["train/episodes"] = n_all
        for cat, v in tr.items():
            if v.get("n"):
                out[f"train/{cat}"] = round(v["n_correct"] / v["n"], 4)

    zg = cfg.get("_last_zero_gradient") or {}
    tot = sum(v.get("total", 0) for v in zg.values())
    if tot:
        silent = sum(v.get("zero_gradient", 0) for v in zg.values())
        out["zero_grad/groups"] = tot
        out["zero_grad/silent"] = silent
        out["zero_grad/frac"] = round(silent / tot, 4)
        out["zero_grad/all_fail"] = sum(v.get("all_fail", 0) for v in zg.values())
        out["zero_grad/all_succeed"] = sum(v.get("all_succeed", 0) for v in zg.values())
        for cat, v in zg.items():
            if v.get("total"):
                out[f"zero_grad/{cat}_frac"] = round(v["zero_gradient"] / v["total"], 4)

    scaf = state.get("scaffold") or {}
    p_task = {k: float(v or 0) for k, v in (scaf.get("p_task") or {}).items()}
    if p_task:
        out["scaffold/p_mean"] = round(sum(p_task.values()) / len(p_task), 4)
        out["scaffold/p_max"] = round(max(p_task.values()), 4)
    items = scaf.get("items") or {}
    out["scaffold/items"] = sum(len(v or []) for v in items.values())
    out["scaffold/chars"] = sum(len(it.get("text", "")) for v in items.values()
                                for it in (v or []))

    hist = state.get("decision_history") or []
    if hist:
        last = hist[-1]
        out["teacher/verdict"] = {"accepted": 2, "rejected": 1, "p_only": 3,
                                  "reverted_to_bare": 4}.get(last.get("verdict"), 0)
        ab = last.get("ab") or {}
        for src, dst in (("cand_mean", "candidate"), ("cur_mean", "current"),
                         ("bare_mean", "bare"), ("n", "n")):
            if ab.get(src) is not None:
                out[f"ab/{dst}"] = ab[src]
        if ab:
            out["ab/accept"] = int(bool(ab.get("accept")))
            out["ab/revert_to_bare"] = int(bool(ab.get("revert_to_bare")))
            out["ab/bare_beats_current"] = int(bool(ab.get("bare_beats_current")))
        out["teacher/accepted_total"] = sum(1 for e in hist if e.get("verdict") == "accepted")
        out["teacher/cycles_recorded"] = len(hist)
    out["teacher/unreachable_cycles"] = int(state.get("teacher_unreachable_cycles") or 0)

    used, limit = C.container_memory()
    if used is not None:
        out["mem/used_gb"] = used
        if limit:
            out["mem/limit_gb"] = limit
            out["mem/frac"] = round(used / limit, 3)

    out["heartbeat/unix"] = int(time.time())
    return out


def publish(state, cfg, wandb_run=None):
    """Never raises: monitoring must not end a cycle that already cost GPU hours."""
    log = cfg.get("log", lambda *a: None)
    try:
        snap = cycle_snapshot(state, cfg)
    except Exception as e:
        log(f"[monitor] snapshot failed: {type(e).__name__}: {e}")
        return
    root = cfg["state_dir"]
    try:
        stamped = dict(snap, exp=cfg.get("exp"), iso=time.strftime("%F %T"))
        S.persist(stamped, os.path.join(root, "status.json"))
        with open(os.path.join(root, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(stamped, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[monitor] disk sinks failed: {type(e).__name__}: {e}")
    if wandb_run is not None:
        try:
            wandb_run.log({k: v for k, v in snap.items() if isinstance(v, (int, float))},
                          step=int(state.get("step") or 0))
        except Exception as e:
            log(f"[monitor] wandb failed: {type(e).__name__}: {e}")
    log(f"[monitor] step={snap.get('progress/step')} "
        f"valid_seen={snap.get('valid_seen/success_rate')} "
        f"train={snap.get('train/success_rate')} "
        f"zero_grad={snap.get('zero_grad/silent')}/{snap.get('zero_grad/groups')} "
        f"items={snap.get('scaffold/items')} p_max={snap.get('scaffold/p_max')} "
        f"unreachable={snap.get('teacher/unreachable_cycles')}")


def warnings_for(s, now=None):
    """The states in which every number looks fine and the run is producing nothing."""
    now = int(now if now is not None else time.time())
    g = s.get
    out = []
    if g("teacher/unreachable_cycles", 0):
        out.append(f"TEACHER UNREACHABLE for {g('teacher/unreachable_cycles')} cycles — "
                   f"no scaffold can be proposed; this run is a plain-RL control, not a result")
    if not g("scaffold/items", 0) and (g("progress/cycle") or 0) >= 4:
        out.append(f"scaffold still EMPTY at cycle {g('progress/cycle')} — nothing accepted yet")
    if g("scaffold/items", 0) and not (g("scaffold/p_max") or 0):
        out.append("scaffold holds text but p=0 everywhere — it reaches no rollout")
    if g("ab/bare_beats_current", 0) and g("teacher/verdict") != 4:
        out.append("the no-text condition outscored the CURRENT scaffold in the last "
                   "A/B (candidate still won or data was short of a revert) — watch "
                   "per_task_gap for this text")
    if (g("mem/frac") or 0) > 0.85:
        out.append(f"container memory at {g('mem/frac', 0):.0%} of its limit "
                   f"({g('mem/used_gb')}G / {g('mem/limit_gb')}G) — an OOM kill at a "
                   f"checkpoint save took a node down once")
    age = now - int(g("heartbeat/unix", 0) or 0)
    if age > 3 * 3600:
        out.append(f"snapshot is {age // 3600}h old — the run may have stopped")
    return out


def render_status(s, now=None):
    now = int(now if now is not None else time.time())
    g = s.get
    L = []
    age = now - int(g("heartbeat/unix", 0) or 0)
    tgt = g("progress/target_step")
    L.append(f"=== {g('exp')} ===  snapshot {g('iso', '?')}  ({age // 60}m ago)")
    L.append(f"  step        {g('progress/step', 0)}" + (f" / {tgt}" if tgt else "")
             + f"   cycle {g('progress/cycle', '?')}")
    L.append(f"  valid_seen  {g('valid_seen/success_rate', '-')}   "
             f"delta {g('valid_seen/delta', '-')}   best {g('valid_seen/best', '-')} "
             f"@ {g('valid_seen/best_step', '-')}"
             + (f"   draw spread {g('valid_seen/draw_spread')}"
                if g("valid_seen/draw_spread") is not None else ""))
    if g("train/episodes"):
        L.append(f"  train       {g('train/success_rate')}   {g('train/episodes')} episodes")
    if g("zero_grad/groups"):
        L.append(f"  zero grad   {g('zero_grad/silent')}/{g('zero_grad/groups')} "
                 f"({g('zero_grad/frac', 0):.0%}; all_fail {g('zero_grad/all_fail')}, "
                 f"all_succeed {g('zero_grad/all_succeed')})")
    L.append(f"  scaffold    {g('scaffold/items', 0)} items, {g('scaffold/chars', 0)} chars, "
             f"p_max {g('scaffold/p_max', '-')}")
    if g("teacher/cycles_recorded"):
        L.append(f"  teacher     accepted {g('teacher/accepted_total', 0)} of "
                 f"{g('teacher/cycles_recorded')} cycles"
                 + (f"   last A/B cand {g('ab/candidate')} vs cur {g('ab/current')} "
                    f"(bare {g('ab/bare')}, n {g('ab/n')})" if g("ab/n") else ""))
    if g("mem/frac") is not None:
        L.append(f"  memory      {g('mem/used_gb')}G / {g('mem/limit_gb')}G "
                 f"({g('mem/frac', 0):.0%})")
    warn = warnings_for(s, now)
    L.append("")
    L.append("!! " + "\n!! ".join(warn) if warn else "   nothing anomalous")
    return "\n".join(L)


def main():
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    with open(os.path.join(d, "status.json")) as f:
        print(render_status(json.load(f)))


if __name__ == "__main__":
    main()
