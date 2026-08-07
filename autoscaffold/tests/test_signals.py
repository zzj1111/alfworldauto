import collections
import json
import os
import tempfile

from autoscaffold import scaffold as S
from autoscaffold import signals as G

CAT = "pick_two_obj_and_place"
N = 4  # rollout_n in these fixtures


def _row(uid, success, injected=False, cat=CAT, steps=3, action="go to desk 1", valid=True):
    return {"uid": uid, "task_type": cat, "gamefile": f"/g/{cat}-X/{uid}",
            "injected": injected, "success": 1.0 if success else 0.0,
            "steps": [{"a": action, "o": f"obs {i}", "v": valid} for i in range(steps)]}


def _group(uid, outcomes, injected=False, cat=CAT, steps_of=None):
    return [_row(uid, oc, injected, cat, steps=(steps_of(i) if steps_of else 3))
            for i, oc in enumerate(outcomes)]


def test_zero_gradient_matches_an_independent_recount():
    rows = (_group("g1", [0, 0, 0, 0]) + _group("g2", [1, 1, 1, 1])
            + _group("g3", [0, 1, 0, 0]) + _group("g4", [0, 0, 0, 0]))
    zg = G.signals_from_rows(rows, N)["zero_gradient_groups"][CAT]
    # recount sharing no code with the implementation
    by = collections.defaultdict(list)
    for r in rows:
        by[r["uid"]].append(r["success"] > 0)
    total = sum(1 for v in by.values() if len(v) == N)
    silent = sum(1 for v in by.values() if len(v) == N and len(set(v)) == 1)
    fails = sum(1 for v in by.values() if len(v) == N and not any(v))
    assert (zg["total"], zg["zero_gradient"], zg["all_fail"], zg["all_succeed"]) \
        == (total, silent, fails, silent - fails) == (4, 3, 2, 1)


def test_partial_groups_are_excluded_from_zero_gradient():
    rows = _group("g1", [0, 0, 0, 0]) + _group("half", [0, 0])
    zg = G.signals_from_rows(rows, N)["zero_gradient_groups"][CAT]
    assert zg["total"] == 1, "a truncated group says nothing about variance"


def test_gap_counts_are_raw_episode_counts_with_reasons():
    rows = _group("b1", [1, 0, 0, 0]) + _group("i1", [1, 1, 0, 0], injected=True)
    sc = S.empty_scaffold()
    gap = G.signals_from_rows(rows, N, scaffold=sc)["per_task_gap"][CAT]
    assert (gap["n_bare"], gap["n_injected"]) == (4, 4)
    assert gap["gap"] == 0.25
    # a category with no injected side names WHY, and the reasons differ
    other = "look_at_obj_in_light"
    rows2 = _group("b2", [1, 0, 0, 0], cat=other)
    g0 = G.signals_from_rows(rows2, N, scaffold=sc)["per_task_gap"][other]
    assert "no scaffold text" in g0["no_injection_reason"]
    sc2, _ = S.apply_item_ops(sc, [{"op": "add", "scope": other, "kind": "skill", "text": "t"}])
    g1 = G.signals_from_rows(rows2, N, scaffold=sc2)["per_task_gap"][other]
    assert g1["no_injection_reason"] == f"p_task[{other}]=0"
    sc3, _ = S.apply_p_ops(sc2, [{"task": other, "p": 0.2}])
    g2 = G.signals_from_rows(rows2, N, scaffold=sc3)["per_task_gap"][other]
    assert "no group fired" in g2["no_injection_reason"]


def test_failure_side_never_shows_a_successful_trajectory():
    # every group succeeds at least once; the top-up must not surface a success
    rows = _group("g1", [1, 1, 1, 1]) + _group("g2", [1, 1, 1, 0])
    ct = G.signals_from_rows(rows, N)["contrastive_traces"][CAT]
    for f in ct["zero_gradient_failures"]:
        assert f["success"] <= 0
    # g2's single failure is the only legitimate entry
    assert len(ct["zero_gradient_failures"]) == 1
    # and a category where nothing failed says so instead of showing something wrong
    rows_all_win = _group("w1", [1, 1, 1, 1]) + _group("w2", [1, 1, 1, 1])
    ct2 = G.signals_from_rows(rows_all_win, N)["contrastive_traces"][CAT]
    assert ct2["zero_gradient_failures"] == []
    assert "no_failures_to_show" in ct2


def test_traces_pair_all_fail_groups_with_shortest_successes():
    rows = (_group("f1", [0, 0, 0, 0], steps_of=lambda i: 5 + i)
            + _group("f2", [0, 0, 0, 0]) + _group("f3", [0, 0, 0, 0])
            + _group("f4", [0, 0, 0, 0])
            + _group("s1", [1, 1, 1, 1], steps_of=lambda i: 10 - i))
    ct = G.signals_from_rows(rows, N)["contrastive_traces"][CAT]
    assert ct["n_all_fail_groups"] == 4
    assert len(ct["zero_gradient_failures"]) == G.TRACES_PER_SIDE
    # the failure entry is the LONGEST failed rollout of its group
    assert ct["zero_gradient_failures"][0]["n_steps"] == 8
    # successes are the shortest in the category
    lens = [s["n_steps"] for s in ct["successes_same_category"]]
    assert lens == sorted(lens) and lens[0] == 7


def test_failure_patterns_rules():
    looped = {"uid": "p1", "task_type": CAT, "gamefile": "/g", "injected": False,
              "success": 0.0,
              "steps": [{"a": "open drawer 1", "o": "locked", "v": True}] * 6}
    invalid = {"uid": "p2", "task_type": CAT, "gamefile": "/g", "injected": False,
               "success": 0.0,
               "steps": [{"a": f"a{i}", "o": f"o{i}", "v": i > 0} for i in range(4)]}
    healthy_fail = {"uid": "p3", "task_type": CAT, "gamefile": "/g", "injected": False,
                    "success": 0.0,
                    "steps": [{"a": f"go to shelf {i}", "o": f"o{i}", "v": True}
                              for i in range(6)]}
    won = dict(healthy_fail, uid="p4", success=1.0)
    fp = G.failure_patterns([looped, invalid, healthy_fail, won])[CAT]
    assert fp["n_failed"] == 3, "successes are not failures"
    assert fp["repeated_action"] == 1 and fp["example_repeated"] == "open drawer 1"
    assert fp["looped_observation"] == 1
    assert fp["invalid_heavy"] == 1


def test_offset_windowing_and_torn_final_line():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "rollouts.jsonl")
    with open(p, "w") as f:
        for r in _group("old", [1, 0, 0, 0]):
            f.write(json.dumps(r) + "\n")
    off = G.log_offset(p)
    with open(p, "a") as f:
        for r in _group("new", [0, 0, 0, 0]):
            f.write(json.dumps(r) + "\n")
        f.write('{"uid": "torn"')          # training mid-write
    rows = G.read_rows(p, off)
    assert {r["uid"] for r in rows} == {"new"}, "the window is one cycle, no rereads"


def test_mixed_injection_group_is_surfaced():
    rows = _group("m1", [0, 0, 0, 0])
    rows[2]["injected"] = True
    out = G.signals_from_rows(rows, N)
    assert out["mixed_injection_groups"] == ["m1"]
