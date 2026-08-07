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


def test_bare_is_a_floor_for_acceptance():
    """Text that loses to NO TEXT has no benefit path — injecting it makes injected
    groups worse. It had been accepted twice under the old candidate-vs-current-only
    rule (0.111 over 0.100 with bare 0.122; 0.883 over 0.867 with bare 0.917)."""
    r = G.ab_gate(_m(0.122, 0.100, 0.111), ["t"])   # beats current, loses to bare
    assert not r["accept"] and r["blocked_by_bare_floor"]
    assert "bare floor" in r["reason"]
    r2 = G.ab_gate(_m(0.100, 0.100, 0.150), ["t"])  # beats both
    assert r2["accept"] and not r2["blocked_by_bare_floor"]
    r3 = G.ab_gate(_m(0.150, 0.100, 0.150), ["t"])  # ties bare exactly -> allowed
    assert r3["accept"]


def test_bare_beating_current_is_reported_as_a_deletion_signal():
    """The harness never auto-clears (the A/B samples only the touched categories, so
    clearing unmeasured scopes would exceed the measurement); the Teacher gets the
    fact and owns the delete."""
    r = G.ab_gate(_m(0.9, 0.8, 0.7), ["t"])         # plain reject; bare on top
    assert not r["accept"] and not r["blocked_by_bare_floor"]
    assert r["bare_beats_current"] and "deletion" in r["reason"]
    r2 = G.ab_gate(_m(0.1, 0.2, 0.3), ["t"])        # healthy accept
    assert r2["accept"] and not r2["bare_beats_current"]


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


def test_verdict_survives_json_with_numpy_inputs():
    """The A/B measurement arrives as numpy scalars from the rollout harness; the
    verdict feeds json sinks (journal, status.json), and a numpy bool in it killed
    the journal write of the first real end-to-end A/B."""
    import json
    import numpy as np
    m = {"bare": {"t": (np.float64(0.3333), np.int64(12))},
         "current": {"t": (np.float64(0.3333), np.int64(12))},
         "candidate": {"t": (np.float64(0.5), np.int64(12))}}
    r = G.ab_gate(m, ["t"])
    dumped = json.dumps(r)
    assert json.loads(dumped)["accept"] is True
