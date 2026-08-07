import copy
import json
import os
import tempfile

import pytest

from autoscaffold import scaffold as S


def _with_items(n=2, scope="general"):
    sc = S.empty_scaffold()
    for i in range(n):
        sc, _ = S.apply_item_ops(sc, [{"op": "add", "scope": scope, "kind": "skill",
                                       "text": f"rule number {i}"}])
    return sc


def test_apply_never_mutates_its_argument():
    sc = S.empty_scaffold()
    frozen = copy.deepcopy(sc)
    S.apply_item_ops(sc, [{"op": "add", "scope": "general", "kind": "skill", "text": "x"}])
    S.apply_p_ops(sc, [{"task": "pick_and_place", "p": 0.3}])
    assert sc == frozen


def test_budget_is_three_changes_per_cycle():
    sc = S.empty_scaffold()
    ops = [{"op": "add", "scope": "general", "kind": "skill", "text": f"t{i}"}
           for i in range(4)]
    ok, reason = S.validate_item_ops(ops, sc)
    assert not ok and "budget" in reason
    assert S.validate_item_ops(ops[:3], sc)[0]


def test_capacity_requires_delete_before_add():
    sc = S.empty_scaffold()
    for i in range(S.MAX_ITEMS_PER_SCOPE):
        sc, _ = S.apply_item_ops(sc, [{"op": "add", "scope": "general", "kind": "skill",
                                       "text": f"filler {i}"}])
    ok, reason = S.validate_item_ops(
        [{"op": "add", "scope": "general", "kind": "skill", "text": "one more"}], sc)
    assert not ok and "delete" in reason
    victim = S.items_of(sc, "general")[0]["id"]
    ok, _ = S.validate_item_ops(
        [{"op": "delete", "id": victim},
         {"op": "add", "scope": "general", "kind": "skill", "text": "one more"}], sc)
    assert ok, "delete-then-add within one action must fit"


def test_duplicate_text_is_rejected_whitespace_and_case_insensitive():
    sc = _with_items(1)
    ok, reason = S.validate_item_ops(
        [{"op": "add", "scope": "general", "kind": "skill", "text": "  RULE   number 0 "}], sc)
    assert not ok and "duplicate" in reason


def test_update_and_delete_resolve_ids_and_unknown_ids_fail():
    sc = _with_items(2)
    an_id = S.items_of(sc, "general")[0]["id"]
    nxt, _ = S.apply_item_ops(sc, [{"op": "update", "id": an_id, "text": "rewritten"}])
    assert any(it["text"] == "rewritten" for it in S.items_of(nxt, "general"))
    ok, reason = S.validate_item_ops([{"op": "delete", "id": "i999"}], sc)
    assert not ok and "unknown id" in reason


def test_p_is_capped_and_rate_limited_by_clamping():
    sc = S.empty_scaffold()
    nxt, notes = S.apply_p_ops(sc, [{"task": "pick_and_place", "p": 0.9}])
    # both limits bind from 0.0: the cap says 0.5, the step limit says 0.2
    assert nxt["p_task"]["pick_and_place"] == S.P_MAX_DELTA
    assert notes, "a clamp must be recorded, not silent"
    nxt2, _ = S.apply_p_ops(nxt, [{"task": "pick_and_place", "p": 0.9}])
    assert nxt2["p_task"]["pick_and_place"] == pytest.approx(2 * S.P_MAX_DELTA)
    nxt3, _ = S.apply_p_ops(nxt2, [{"task": "pick_and_place", "p": 0.9}])
    assert nxt3["p_task"]["pick_and_place"] == S.P_MAX, "the hard cap holds"


def test_render_composes_general_then_category_and_leaks_nothing_across():
    sc = S.empty_scaffold()
    sc, _ = S.apply_item_ops(sc, [
        {"op": "add", "scope": "general", "kind": "skill", "text": "GENERAL-MARK"},
        {"op": "add", "scope": "pick_two_obj_and_place", "kind": "skill", "text": "TWO-MARK"}])
    for cat in S.CATEGORIES:
        r = S.render(sc, cat)
        assert "GENERAL-MARK" in r
        assert ("TWO-MARK" in r) == (cat == "pick_two_obj_and_place")
    assert r.startswith("- ")


def test_injects_nothing_needs_both_text_and_p():
    sc = S.empty_scaffold()
    assert S.injects_nothing(sc)
    with_text, _ = S.apply_item_ops(sc, [{"op": "add", "scope": "general",
                                          "kind": "skill", "text": "x"}])
    assert S.injects_nothing(with_text), "text with p=0 everywhere reaches no rollout"
    with_p, _ = S.apply_p_ops(with_text, [{"task": "pick_and_place", "p": 0.2}])
    assert not S.injects_nothing(with_p)


def test_touched_categories_general_touches_all():
    sc = _with_items(1, scope="pick_and_place")
    an_id = S.items_of(sc, "pick_and_place")[0]["id"]
    assert S.touched_categories([{"op": "update", "id": an_id}], sc) == ["pick_and_place"]
    assert S.touched_categories(
        [{"op": "add", "scope": "general", "kind": "skill", "text": "y"}], sc) \
        == list(S.CATEGORIES)


def test_persist_is_atomic_and_load_round_trips():
    sc = _with_items(3)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scaffold.json")
    S.persist(sc, p)
    assert S.load(p) == sc
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    with open(p, "w") as f:
        f.write("{ torn")
    assert S.load(p) is None, "a torn file reads as absent, never as a crash"


def test_category_of_gamefile():
    g = "/data/json_2.1.1/train/pick_two_obj_and_place-Mug-None-Desk-308/trial_x/game.tw-pw"
    assert S.category_of_gamefile(g) == "pick_two_obj_and_place"
    g2 = "/data/json_2.1.1/train/pick_and_place_simple-Pen-None-Shelf-1/trial_y/game.tw-pw"
    assert S.category_of_gamefile(g2) == "pick_and_place"
    assert S.category_of_gamefile("/nothing/here") is None


def test_duplicate_delete_is_rejected_and_apply_never_crashes():
    """A Teacher sending the same delete twice passed validation (the pending set
    collapsed the duplicate) and then crashed apply_item_ops with a KeyError on
    items[None] — an uncaught exception inside the training cycle. Found by review."""
    sc = _with_items(2)
    victim = S.items_of(sc, "general")[0]["id"]
    ops = [{"op": "delete", "id": victim}, {"op": "delete", "id": victim}]
    ok, reason = S.validate_item_ops(ops, sc)
    assert not ok and "duplicate delete" in reason
    # belt: even if a double-delete reaches apply, it degrades to a note
    nxt, notes = S.apply_item_ops(sc, [{"op": "delete", "id": victim}])
    assert victim not in [it["id"] for it in S.items_of(nxt, "general")]


def test_update_duplicating_another_item_is_rejected():
    sc = _with_items(2)
    ids = [it["id"] for it in S.items_of(sc, "general")]
    ok, reason = S.validate_item_ops(
        [{"op": "update", "id": ids[0], "text": "  Rule NUMBER 1 "}], sc)
    assert not ok and "duplicates" in reason


def test_touched_categories_resolves_the_real_scope_over_a_spurious_one():
    """An update/delete carries an id; the item's REAL scope decides which categories
    the edit can change. A spurious 'scope' key on the op must not steer the A/B to
    the wrong games — a general-item edit measured on one category would be judged on
    a sixth of the population it changes."""
    sc = _with_items(1, scope="general")
    an_id = S.items_of(sc, "general")[0]["id"]
    op = {"op": "update", "id": an_id, "text": "x", "scope": "pick_and_place"}
    assert S.touched_categories([op], sc) == list(S.CATEGORIES)
