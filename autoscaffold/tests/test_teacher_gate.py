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


def test_three_way_rule_accept_revert_reject():
    """The tournament over the measured union (2026-08-07, second revision):
    accept iff cand > cur and cand >= bare; revert to bare iff bare strictly beats
    BOTH; else keep. The two historical bad accepts (0.111 over 0.100 with bare
    0.122; 0.883 over 0.867 with bare 0.917) both land in the revert branch now."""
    r = G.ab_gate(_m(0.122, 0.100, 0.111), ["t"])   # bare beats both -> revert
    assert not r["accept"] and r["revert_to_bare"]
    assert "REVERT-TO-BARE" in r["reason"]
    r2 = G.ab_gate(_m(0.917, 0.867, 0.883), ["t"])  # the other historical case
    assert r2["revert_to_bare"]
    r3 = G.ab_gate(_m(0.100, 0.100, 0.150), ["t"])  # candidate beats both -> accept
    assert r3["accept"] and not r3["revert_to_bare"]
    r4 = G.ab_gate(_m(0.150, 0.100, 0.150), ["t"])  # ties bare, beats cur -> accept
    assert r4["accept"]
    r5 = G.ab_gate(_m(0.5, 0.55, 0.45), ["t"])      # current best -> plain reject
    assert not r5["accept"] and not r5["revert_to_bare"]
    r6 = G.ab_gate(_m(0.9, 0.8, 0.7), ["t"])        # bare beats both -> revert
    assert r6["revert_to_bare"] and r6["bare_beats_current"]


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
    assert json.loads(dumped)["revert_to_bare"] is False


def test_per_category_verdicts_split_a_mixed_measure():
    """Candidate best in A while bare best in B: A gets verdict accept, B gets
    verdict revert, and the record itemizes both — each category is acted on by its
    own numbers (user decision: per-category at n~30)."""
    measure = {"bare": {"a": (0.10, 90), "b": (0.60, 90)},
               "current": {"a": (0.20, 90), "b": (0.40, 90)},
               "candidate": {"a": (0.50, 90), "b": (0.30, 90)}}
    r = G.ab_gate(measure, ["a", "b"])
    assert r["per_category"]["a"]["verdict"] == "accept"
    assert r["per_category"]["b"]["verdict"] == "revert"
    assert r["mixed_verdict"] and "per-category verdicts differ" in r["reason"]
    # aggregate fields remain the record (and rule general-scoped text)
    assert r["accept"] is True     # cand 0.40 > cur 0.30, >= bare 0.35
    import json
    json.dumps(r)                  # must survive the journal's JSON sinks


def test_per_category_keep_when_no_side_wins():
    measure = {"bare": {"a": (0.30, 30)},
               "current": {"a": (0.35, 30)},
               "candidate": {"a": (0.32, 30)}}
    r = G.ab_gate(measure, ["a"])
    assert r["per_category"]["a"]["verdict"] == "keep"
    assert not r["accept"] and not r["revert_to_bare"]
