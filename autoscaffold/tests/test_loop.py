import copy
import json

import pytest

from autoscaffold import loop as L
from autoscaffold import scaffold as S
from autoscaffold import teacher as T

ADD = {"op": "add", "scope": "pick_and_place", "kind": "skill", "text": "look first"}
POP = {"task": "pick_and_place", "p": 0.2}


def _fns(teacher=None, ab_accept=True, calls=None):
    calls = calls if calls is not None else []

    def rec(name, ret=None):
        def f(*a, **k):
            calls.append(name)
            return ret(*a, **k) if callable(ret) else ret
        return f

    measure_ok = {"bare": {"pick_and_place": (0.1, 60)},
                  "current": {"pick_and_place": (0.2, 60)},
                  "candidate": {"pick_and_place": (0.5 if ab_accept else 0.1, 60)}}
    return {
        "train_fn": rec("train", "/ck/global_step_10"),
        "eval_fn": rec("eval", {"avg": 0.3, "per_task": {}, "draws": [0.3, 0.31, 0.29]}),
        "signals_fn": rec("signals", {"per_task_gap": {}, "zero_gradient_groups": {}}),
        "teacher_fn": rec("teacher", teacher or (lambda o, sc: (dict(T.NOOP), "ok"))),
        "measure_ab_fn": rec("ab", lambda ck, cur, cand, tasks: measure_ok),
        "persist_fn": rec("persist"),
        "state_fn": rec("state"),
        "journal_fn": rec("journal"),
        "log": lambda m: calls.append("log"),
    }, calls


def test_cycle_order_and_noop_skips_the_ab():
    fns, calls = _fns()
    st = L.run_cycle(L.new_state(0), fns, {"steps_per_cycle": 10})
    order = [c for c in calls if c != "log"]
    assert order.index("train") < order.index("eval") < order.index("signals") \
        < order.index("teacher")
    assert "ab" not in order, "a no-op must not spend the A/B"
    assert st["step"] == 10 and st["decision_history"][-1]["verdict"] == "noop"


def test_accepted_text_applies_with_its_p():
    fns, _ = _fns(teacher=lambda o, sc: (
        {"diagnosis": "d", "item_ops": [ADD], "p_ops": [POP]}, "ok"), ab_accept=True)
    st = L.run_cycle(L.new_state(0), fns, {})
    assert S.items_of(st["scaffold"], "pick_and_place")
    assert st["scaffold"]["p_task"]["pick_and_place"] == 0.2
    assert st["decision_history"][-1]["verdict"] == "accepted"


def test_rejected_text_vetoes_its_p_and_keeps_the_scaffold():
    fns, _ = _fns(teacher=lambda o, sc: (
        {"diagnosis": "d", "item_ops": [ADD], "p_ops": [POP]}, "ok"), ab_accept=False)
    before = L.new_state(0)
    frozen = copy.deepcopy(before["scaffold"])
    st = L.run_cycle(before, fns, {})
    e = st["decision_history"][-1]
    assert e["verdict"] == "rejected" and e["p_vetoed_with_text"] is True
    assert st["scaffold"] == frozen, "a rejected proposal changes nothing"
    assert st["scaffold"]["p_task"]["pick_and_place"] == 0.0


def test_p_only_skips_the_ab_and_applies():
    fns, calls = _fns(teacher=lambda o, sc: (
        {"diagnosis": "", "item_ops": [], "p_ops": [POP]}, "ok"))
    st = L.run_cycle(L.new_state(0), fns, {})
    assert "ab" not in calls
    assert st["scaffold"]["p_task"]["pick_and_place"] == 0.2
    assert st["decision_history"][-1]["verdict"] == "p_only"


def test_unreachable_counter_climbs_banners_and_resets():
    seen = []
    fns, _ = _fns(teacher=lambda o, sc: (dict(T.NOOP), f"{T.UNREACHABLE_NOTE} (401)"))
    fns["log"] = seen.append
    st = L.new_state(0)
    for _ in range(3):
        st = L.run_cycle(st, fns, {})
    assert st["teacher_unreachable_cycles"] == 3
    assert any("TEACHER UNREACHABLE" in m for m in seen)
    fns2, _ = _fns()
    st = L.run_cycle(st, fns2, {})
    assert st["teacher_unreachable_cycles"] == 0, "a reachable Teacher clears the count"


def test_sr_after_backfills_onto_the_decision_that_caused_it():
    fns, _ = _fns(teacher=lambda o, sc: (
        {"diagnosis": "", "item_ops": [], "p_ops": [POP]}, "ok"))
    st = L.run_cycle(L.new_state(0), fns, {})
    assert st["decision_history"][-1]["sr_after"] is None
    fns2, _ = _fns()
    fns2["eval_fn"] = lambda ck: {"avg": 0.4, "per_task": {}, "draws": [0.4]}
    st = L.run_cycle(st, fns2, {})
    assert st["decision_history"][0]["sr_after"] == 0.4


def test_every_state_field_survives_a_round_trip():
    """STATE_KEYS is derived from new_state, so created and persisted sets cannot
    drift; this asserts the derivation stays total."""
    st = L.new_state(0)
    assert set(L.STATE_KEYS) == set(st)
    dumped = json.loads(json.dumps({k: st[k] for k in L.STATE_KEYS}))
    assert set(dumped) == set(st)


def test_target_step_stops_the_run():
    fns, calls = _fns()
    st = L.run(L.new_state(0), fns, {"steps_per_cycle": 10, "target_step": 20}, n_cycles=99)
    assert st["step"] == 20
    assert calls.count("train") == 2


def test_revert_to_bare_clears_items_and_vetoes_p():
    """When no-text strictly beats both sides over the union, the harness clears
    every item, keeps p (inert without text), vetoes the co-submitted p, and
    journals what was removed so the Teacher's memory shows the cleared wording."""
    revert_measure = {"bare": {"pick_and_place": (0.5, 60)},
                      "current": {"pick_and_place": (0.3, 60)},
                      "candidate": {"pick_and_place": (0.4, 60)}}
    st0 = L.new_state(0)
    st0["scaffold"], _ = S.apply_item_ops(
        st0["scaffold"], [{"op": "add", "scope": "pick_and_place", "kind": "skill",
                           "text": "old text that turned harmful"}])
    st0["scaffold"], _ = S.apply_p_ops(st0["scaffold"], [{"task": "pick_and_place", "p": 0.2}])
    fns, _ = _fns(teacher=lambda o, sc: (
        {"diagnosis": "d", "item_ops": [ADD], "p_ops": [{"task": "pick_and_place", "p": 0.4}]},
        "ok"))
    fns["measure_ab_fn"] = lambda ck, cur, cand, tasks: revert_measure
    st = L.run_cycle(st0, fns, {})
    e = st["decision_history"][-1]
    assert e["verdict"] == "reverted_to_bare"
    assert e["p_vetoed_with_text"] is True
    assert e["cleared_items"] and "old text" in e["cleared_items"][0]["text"]
    assert sum(len(v) for v in st["scaffold"]["items"].values()) == 0
    assert st["scaffold"]["p_task"]["pick_and_place"] == 0.2, "p survives, inert"


def test_ab_scope_is_the_union_of_touched_and_reached():
    """A proposal touching one category while general text exists must measure ALL
    categories — that union is what makes a revert measurement-honest."""
    seen = {}
    st0 = L.new_state(0)
    st0["scaffold"], _ = S.apply_item_ops(
        st0["scaffold"], [{"op": "add", "scope": "general", "kind": "skill",
                           "text": "general text reaching every category"}])
    fns, _ = _fns(teacher=lambda o, sc: (
        {"diagnosis": "d", "item_ops": [ADD], "p_ops": []}, "ok"), ab_accept=True)
    real_measure = {"bare": {c: (0.1, 30) for c in S.CATEGORIES},
                    "current": {c: (0.2, 30) for c in S.CATEGORIES},
                    "candidate": {c: (0.5, 30) for c in S.CATEGORIES}}
    def measure(ck, cur, cand, tasks):
        seen["tasks"] = tasks
        return real_measure
    fns["measure_ab_fn"] = measure
    L.run_cycle(st0, fns, {})
    assert set(seen["tasks"]) == set(S.CATEGORIES), seen["tasks"]


def test_late_validation_failure_is_a_noop_not_a_crash():
    # teacher_fn bypassing normalize (a replayed journal, a harness bug) must not
    # take the cycle down
    fns, _ = _fns(teacher=lambda o, sc: (
        {"diagnosis": "", "item_ops": [{"op": "add", "scope": "nope", "kind": "skill",
                                        "text": "x"}], "p_ops": []}, "ok"))
    st = L.run_cycle(L.new_state(0), fns, {})
    e = st["decision_history"][-1]
    assert e["verdict"] == "noop" and "invalid" in e["summary"]
