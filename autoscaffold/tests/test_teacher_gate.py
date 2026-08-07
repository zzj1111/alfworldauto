import pytest

from autoscaffold import gate as G
from autoscaffold import scaffold as S
from autoscaffold import teacher as T


# ---------------- teacher ----------------

def test_malformed_output_degrades_to_noop_never_raises():
    sc = S.empty_scaffold()
    for raw in (None, "text", [], {"item_ops": "not a list"},
                {"item_ops": [{"op": "update", "id": "i99", "text": "x"}]},
                {"p_ops": [{"task": "bogus_category", "p": 0.2}]},
                {"p_ops": [{"task": "pick_and_place", "p": "high"}]}):
        action, note = T.normalize(raw, sc)
        assert action == T.NOOP and "no-op" in note, raw


def test_valid_action_passes_through():
    sc = S.empty_scaffold()
    raw = {"diagnosis": "d",
           "item_ops": [{"op": "add", "scope": "general", "kind": "skill", "text": "t"}],
           "p_ops": [{"task": "pick_and_place", "p": 0.2}]}
    action, note = T.normalize(raw, sc)
    assert note == "ok" and action["item_ops"] == raw["item_ops"]


def test_unreachable_is_marked_distinctly_from_transient():
    sc = S.empty_scaffold()

    def dead(system, user):
        raise RuntimeError("Error code: 401 - invalid_api_key")

    def flaky(system, user):
        raise ValueError("500 internal server error")

    _, note_dead = T.propose("s", "u", sc, call_fn=dead)
    _, note_flaky = T.propose("s", "u", sc, call_fn=flaky)
    assert T.UNREACHABLE_NOTE in note_dead
    assert T.UNREACHABLE_NOTE not in note_flaky, \
        "a transient 500 marked unreachable makes the outage banner cry wolf"


# ---------------- gate ----------------

def _m(bare, cur, cand, n=180):
    return {"bare": {"t": (bare, n)}, "current": {"t": (cur, n)},
            "candidate": {"t": (cand, n)}}


def test_accept_rule_is_strict_greater():
    assert G.ab_gate(_m(0.1, 0.25, 0.26), ["t"])["accept"] is True
    assert G.ab_gate(_m(0.1, 0.25, 0.25), ["t"])["accept"] is False, "ties reject"
    assert G.ab_gate(_m(0.1, 0.25, 0.24), ["t"])["accept"] is False


def test_below_bare_is_flagged_only_on_accept():
    r = G.ab_gate(_m(0.122, 0.100, 0.111), ["t"])
    assert r["accept"] and r["below_bare"] and "BELOW" in r["reason"]
    r2 = G.ab_gate(_m(0.9, 0.8, 0.7), ["t"])
    assert not r2["accept"] and not r2["below_bare"], \
        "below_bare describes an ACCEPTED candidate; a rejected one is never applied"


def test_missing_samples_reject_rather_than_invent():
    r = G.ab_gate({"bare": {}, "current": {}, "candidate": {"t": (0.5, 60)}}, ["t"])
    assert not r["accept"] and "missing" in r["reason"]
    assert not G.ab_gate(_m(0.1, 0.2, 0.9), [])["accept"], "no touched categories"


def test_aggregation_covers_exactly_the_touched_categories():
    measure = {"bare": {"a": (0.1, 100), "b": (0.9, 100)},
               "current": {"a": (0.2, 100), "b": (0.9, 100)},
               "candidate": {"a": (0.5, 100), "b": (0.1, 100)}}
    only_a = G.ab_gate(measure, ["a"])
    assert only_a["accept"] and only_a["cand_mean"] == 0.5, \
        "category b's collapse must not leak into a proposal that only touched a"
    both = G.ab_gate(measure, ["a", "b"])
    assert not both["accept"]
