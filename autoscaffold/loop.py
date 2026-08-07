"""The control loop. Pure control flow: every side effect is an injected function, so
the whole loop runs under test with mocks.

One cycle: train K steps -> standalone eval on held-out (VAL_N draws) -> free signals
from the cycle's own rollouts -> Teacher proposes -> A/B on text -> apply -> persist.
There is no revert: a regression stays in the curve.

Injected functions (fns dict):
  train_fn(scaffold, from_step, to_step) -> checkpoint path
  eval_fn(checkpoint)                    -> {"avg", "per_task", "draws"}
  signals_fn(checkpoint, scaffold)       -> signals packet (signals.signals_from_rows)
  teacher_fn(obs, scaffold)              -> (action, note)
  measure_ab_fn(checkpoint, current, candidate, tasks) -> gate.ab_gate's measure dict
  persist_fn(scaffold), state_fn(state), journal_fn(history), log(msg)
  snapshot_fn(state)                     -> None (monitoring; optional)
"""
from __future__ import annotations

import copy

from . import gate
from . import prompts
from . import scaffold as S
from .teacher import UNREACHABLE_NOTE


def new_state(step0=0, scaffold=None):
    return {
        "cycle": 0,
        "step": step0,
        "scaffold": scaffold if scaffold is not None else S.empty_scaffold(),
        "sr_history": [],
        "best": None,
        "best_step": step0,
        "decision_history": [],
        "last_eval": None,
        "train_rollouts": [],              # last cycles' per-category outcomes (trend)
        "teacher_unreachable_cycles": 0,
    }


# Everything a restart must carry. Derived from new_state so the created set and the
# persisted set cannot drift apart — the previous implementation lost two fields to
# exactly that drift.
STATE_KEYS = tuple(new_state())


def _call(fns, name, *a, default=None):
    fn = fns.get(name)
    return fn(*a) if fn else default


def _save(state, fns):
    try:
        _call(fns, "state_fn", state)
    except Exception as e:
        _call(fns, "log", f"[warn] state save failed: {type(e).__name__}: {e}")
    try:
        _call(fns, "snapshot_fn", state)
    except Exception as e:
        _call(fns, "log", f"[warn] snapshot failed: {type(e).__name__}: {e}")
    return state


def _journal(state, fns):
    try:
        _call(fns, "journal_fn", state["decision_history"])
    except Exception as e:
        _call(fns, "log", f"[warn] journal write failed: {type(e).__name__}: {e}")
    return _save(state, fns)


def _backfill(history, sr_after):
    for entry in reversed(history):
        if entry.get("sr_after") is None:
            entry["sr_after"] = sr_after
            break


def _summary(action):
    """The record of a proposal. Keeps the ACTUAL text: the wording is what the A/B
    judged, and the Teacher must see it next cycle to avoid re-proposing a loser."""
    items = list(action.get("item_ops") or [])
    proposed = [{k: v for k, v in op.items() if v is not None} for op in items]
    return {"text_edits": [f"{op.get('op')}:{op.get('scope') or op.get('id')}" for op in items],
            "p_edits": {op["task"]: op["p"] for op in action.get("p_ops") or []},
            "text_proposed": proposed,
            "diagnosis": action.get("diagnosis", "")}


def _mark_unreachable(state, note, fns, cyc):
    n = state.get("teacher_unreachable_cycles", 0) + 1 if UNREACHABLE_NOTE in (note or "") else 0
    state["teacher_unreachable_cycles"] = n
    if n:
        bar = "=" * 72
        _call(fns, "log",
              f"[c{cyc}] {bar}\n"
              f"[c{cyc}] TEACHER UNREACHABLE for {n} consecutive cycle(s): {note}\n"
              f"[c{cyc}] Training and eval continue, but NO scaffold can be proposed while "
              f"this lasts. A run that finishes this way is a plain-RL control, NOT evidence "
              f"that text does not help. Check OPENAI_API_KEY / quota / network.\n"
              f"[c{cyc}] {bar}")
    return n


def run_cycle(state, fns, cfg):
    log = fns.get("log", lambda *a: None)
    K = int(cfg.get("steps_per_cycle", 10))
    from_step, to_step = state["step"], state["step"] + K
    cyc = state["cycle"] + 1

    # 1) train with the CURRENT scaffold
    ckpt = fns["train_fn"](state["scaffold"], from_step, to_step)
    state["step"], state["cycle"] = to_step, cyc

    # 1b) this cycle's per-category training outcomes, kept as a series so "stuck" is
    # visible as a trend (populated by the runner via cfg after train_fn)
    tr = cfg.get("_last_train_rollouts")
    if tr:
        state["train_rollouts"] = (state["train_rollouts"] + [
            {"cycle": cyc, "step": to_step, "by_category": tr}])[-12:]

    # 2) standalone eval (bare, held-out); save immediately so a crash later in the
    # cycle does not lose a measured number
    ev = fns["eval_fn"](ckpt) or {}
    sr = ev.get("avg")
    if sr is not None:
        _backfill(state["decision_history"], sr)
        state["sr_history"] = state["sr_history"] + [sr]
        state["last_eval"] = ev
        state["best"], state["best_step"] = gate.update_best(
            state["best"], state["best_step"], sr, to_step)
    log(f"[c{cyc}] step {to_step} valid_seen={sr} draws={ev.get('draws')}")
    _save(state, fns)

    # 3) signals -> Teacher
    sig = fns["signals_fn"](ckpt, state["scaffold"]) or {}
    obs = prompts.assemble_observation(state["scaffold"], sig, state["decision_history"],
                                       to_step, cyc, valid_seen=ev)
    action, note = fns["teacher_fn"](obs, state["scaffold"])
    log(f"[c{cyc}] teacher: {note}; edits={len(action.get('item_ops') or [])} "
        f"p={len(action.get('p_ops') or [])}")
    _mark_unreachable(state, note, fns, cyc)

    if S.is_noop(action):
        state["decision_history"].append({
            "cycle": cyc, "step": to_step,
            "summary": {"noop": True, "note": note,
                        "diagnosis": action.get("diagnosis", "")},
            "sr_before": sr, "draws_before": ev.get("draws"),
            "sr_after": None, "verdict": "noop"})
        return _journal(state, fns)

    entry = {"cycle": cyc, "step": to_step, "summary": _summary(action),
             "sr_before": sr, "draws_before": ev.get("draws"), "sr_after": None}

    # 4) text changes go through the A/B; p-only changes do not
    if action.get("item_ops"):
        try:
            candidate, notes = S.apply_item_ops(state["scaffold"], action["item_ops"])
        except ValueError as e:
            entry["summary"]["invalid"] = str(e)
            entry["verdict"] = "noop"
            state["decision_history"].append(entry)
            log(f"[c{cyc}] item_ops failed validation late: {e}")
            return _journal(state, fns)
        tasks = S.touched_categories(action["item_ops"], state["scaffold"])
        measure = fns["measure_ab_fn"](ckpt, state["scaffold"], candidate, tasks)
        verdict = gate.ab_gate(measure, tasks)
        entry["ab"] = verdict
        log(f"[c{cyc}] A/B: {verdict['reason']}")
        if verdict["accept"]:
            with_p, p_notes = S.apply_p_ops(candidate, action.get("p_ops"))
            state["scaffold"] = with_p
            entry["verdict"] = "accepted"
            entry["p_applied"] = bool(action.get("p_ops"))
            entry["apply_notes"] = notes + p_notes
            _call(fns, "persist_fn", state["scaffold"])
        else:
            # p-veto: a p change arriving with rejected text is discarded with it — the
            # A/B judged the pair, and applying half of it would credit the accept rule
            # with a change it never saw.
            entry["verdict"] = "rejected"
            entry["p_applied"] = False
            entry["p_vetoed_with_text"] = bool(action.get("p_ops"))
    else:
        with_p, p_notes = S.apply_p_ops(state["scaffold"], action.get("p_ops"))
        state["scaffold"] = with_p
        entry["verdict"] = "p_only"
        entry["p_applied"] = True
        entry["apply_notes"] = p_notes
        _call(fns, "persist_fn", state["scaffold"])

    state["decision_history"].append(entry)
    return _journal(state, fns)


def run(state, fns, cfg, n_cycles):
    """Up to n_cycles, or until cfg['target_step'] is reached. target_step is the
    absolute finish line and survives restarts; n_cycles only caps this process."""
    target = int(cfg.get("target_step", 0) or 0)
    for _ in range(n_cycles):
        if target and state.get("step", 0) >= target:
            break
        state = run_cycle(state, fns, cfg)
    return state
