"""Unit + mocked-integration tests for the auto-scaffold harness. No GPU, no API.
Run: python -m agent_system.skill_opt.autoscaffold.tests.test_autoscaffold"""
import copy
import json
import sys
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")

from agent_system.skill_opt.autoscaffold import scaffold as S
from agent_system.skill_opt.autoscaffold import gates
from agent_system.skill_opt.autoscaffold import teacher
from agent_system.skill_opt.autoscaffold import loop as L
from agent_system.skill_opt.autoscaffold import adapters as A

FAIL = []
def make_usable_ckpt(path, world_size=1):
    """Create a checkpoint dir that ckpt_is_usable() accepts.

    It requires the COMPLETE shard set (model/optim/extra_state for every rank, plus the
    trainer's data.pt), not just an `actor/` directory — verl creates that dir minutes before
    the shards finish landing, and treating the half-written state as done makes the loop skip
    training and report a stale eval as the new step's score.
    """
    import os
    os.makedirs(os.path.join(path, "actor"), exist_ok=True)
    open(os.path.join(path, "data.pt"), "w").close()
    for kind in ("model", "optim", "extra_state"):
        for rank in range(world_size):
            open(os.path.join(path, "actor", f"{kind}_world_size_{world_size}_rank_{rank}.pt"), "w").close()
    return path


def ok(c, m):
    """Assert, and say what was checked either way.

    This used to only append to FAIL and print. Nothing ever inspected FAIL, so every test
    written with ok() passed under pytest no matter what it measured — 19 of them, including
    the revert tests. Found 2026-07-29 while removing the revert gate: test_loop_revert was
    asserting a mechanism that no longer existed and still reported green.
    """
    print(("PASS" if c else "FAIL"), m)
    if not c:
        FAIL.append(m)
    assert c, m


# ----------------------------- scaffold ----------------------------- #
def test_scaffold():
    sc = S.empty_scaffold()
    ok(sc["general_skill"] == "" and all(sc["skills"][t] == "" for t in S.TASKS),
       "empty_scaffold is truly empty (真空)")
    ok(all(sc["p_task"][t] == 0.0 for t in S.TASKS),
       "empty_scaffold injects nothing until the Teacher raises p")
    ok(len(sc["skills"]) == 6 and len(S.TASKS) == 6, "6 task categories")

    # validate_action: good
    a = {"text_ops": [{"target": "look_at_obj_in_light", "text": "take it first"}],
         "p_ops": [{"task": "pick_and_place", "p": 0.5}]}
    ok(S.validate_action(a)[0], "valid action accepted")
    # bad target / empty text / bad p / bad task / overlong
    ok(not S.validate_action({"text_ops": [{"target": "nope", "text": "x"}], "p_ops": []})[0],
       "unknown text target rejected")
    ok(not S.validate_action({"text_ops": [{"target": "general", "text": "  "}], "p_ops": []})[0],
       "empty text rejected")
    ok(not S.validate_action({"text_ops": [], "p_ops": [{"task": "pick_and_place", "p": 1.5}]})[0],
       "p out of range rejected")
    ok(not S.validate_action({"text_ops": [], "p_ops": [{"task": "nope", "p": 0.5}]})[0],
       "unknown p task rejected")
    ok(not S.validate_action({"text_ops": [{"target": "general", "text": "z" * 2000}], "p_ops": []})[0],
       "overlong text rejected (crash guard)")

    ok(S.is_noop({"text_ops": [], "p_ops": []}), "empty action is no-op")
    ok(not S.is_noop(a), "non-empty action is not no-op")

    ok(S.touched_tasks([{"target": "general", "text": "g"}]) == sorted(S.TASKS),
       "general edit touches all tasks")
    ok(S.touched_tasks([{"target": "look_at_obj_in_light", "text": "x"}]) == ["look_at_obj_in_light"],
       "per-task edit touches only that task")

    # immutability + version bump
    sc2 = S.apply_text_ops(sc, [{"target": "look_at_obj_in_light", "text": "hint A"}])
    ok(sc["skills"]["look_at_obj_in_light"] == "", "apply_text_ops did NOT mutate input")
    ok(sc2["skills"]["look_at_obj_in_light"] == "hint A" and sc2["version"] == 1,
       "apply_text_ops set new text + bumped version")
    sc3 = S.apply_p_ops(sc2, [{"task": "pick_two_obj_and_place", "p": 0.15}])
    ok(sc2["p_task"]["pick_two_obj_and_place"] == 0.0, "apply_p_ops did NOT mutate input")
    ok(sc3["p_task"]["pick_two_obj_and_place"] == 0.15 and sc3["version"] == 2,
       "apply_p_ops set p + bumped (0.15 is within the per-cycle step limit)")

    ok(S.validate_scaffold(sc3)[0], "assembled scaffold validates")
    broken = dict(sc3); broken["skills"] = {k: v for k, v in sc3["skills"].items() if k != S.TASKS[0]}
    ok(not S.validate_scaffold(broken)[0], "scaffold missing a task rejected")


# ------------------------------- gates ------------------------------ #
def test_ab_gate():
    tasks = ["look_at_obj_in_light"]
    m_win = {"bare": {"look_at_obj_in_light": (0.30, 30)},
             "current": {"look_at_obj_in_light": (0.40, 30)},
             "candidate": {"look_at_obj_in_light": (0.65, 30)}}
    ok(gates.ab_gate(m_win, tasks)["accept"], "A/B accepts a clear candidate win (+0.25)")
    m_tiny = {"bare": {"look_at_obj_in_light": (0.40, 30)},
              "current": {"look_at_obj_in_light": (0.50, 30)},
              "candidate": {"look_at_obj_in_light": (0.51, 30)}}
    ok(gates.ab_gate(m_tiny, tasks)["accept"], "A/B accepts ANY strict improvement (+0.01, no margin)")
    m_worse = {"bare": {"look_at_obj_in_light": (0.40, 30)},
               "current": {"look_at_obj_in_light": (0.50, 30)},
               "candidate": {"look_at_obj_in_light": (0.50, 30)}}
    ok(not gates.ab_gate(m_worse, tasks)["accept"], "A/B rejects when candidate is not greater (==)")
    ok(not gates.ab_gate({"current": {}, "candidate": {}}, tasks)["accept"],
       "A/B rejects when samples are missing (keep current)")
    ok(not gates.ab_gate(m_win, [])["accept"], "A/B rejects when no tasks touched")

    # general edit -> aggregate over all tasks
    allt = S.TASKS
    cur = {t: (0.4, 30) for t in allt}
    cand = {t: (0.6, 30) for t in allt}
    g = gates.ab_gate({"bare": {t: (0.3, 30) for t in allt}, "current": cur, "candidate": cand}, allt)
    ok(g["accept"] and g["n"] == 30 * 6 * 2, "A/B aggregates a general edit over all 6 tasks")



def test_teacher_normalize():
    a, note = teacher.normalize({"diagnosis": "d", "text_ops": [{"target": "general", "text": "g"}], "p_ops": []})
    ok(a["text_ops"] and note == "ok", "normalize passes a valid action")
    a2, n2 = teacher.normalize({"text_ops": [{"target": "bad", "text": "x"}]})
    ok(S.is_noop(a2), "normalize -> no-op on invalid action")
    a3, n3 = teacher.normalize("garbage")
    ok(S.is_noop(a3), "normalize -> no-op on non-dict")
    a4, n4 = teacher.propose({}, call_fn=lambda s, u: (_ for _ in ()).throw(RuntimeError("boom")))
    ok(S.is_noop(a4) and "failed" in n4, "propose -> no-op when the GPT call throws")


# --------------------- mocked full-loop scenarios ------------------- #
def _mock_fns(eval_seq, teacher_action, measure, calls):
    it = iter(eval_seq)
    def eval_fn(ckpt, val_n):
        sr = next(it)
        return {"avg": sr, "per_task": {}, "draws": [sr]}
    def train_fn(sc, a, b):
        calls["train"] += 1
        return f"ckpt_{b}"
    def signals_fn(ckpt, sc):
        calls["signals"] += 1
        return {"per_task_gap": {}, "all_fail_groups": {}, "failures": []}
    def measure_ab_fn(ckpt, cur, cand, tasks):
        calls["measure"] += 1
        return measure
    def teacher_fn(obs):
        calls["teacher"] += 1
        return dict(teacher_action), "mock"
    def persist_fn(sc):
        calls["persist"] = sc
    return {"train_fn": train_fn, "eval_fn": eval_fn, "signals_fn": signals_fn,
            "measure_ab_fn": measure_ab_fn, "teacher_fn": teacher_fn,
            "persist_fn": persist_fn, "log": lambda *a: None}


def test_loop_accept():
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    action = {"diagnosis": "look_at weak", "p_ops": [],
              "text_ops": [{"target": "look_at_obj_in_light", "text": "take the object first"}]}
    measure = {"bare": {"look_at_obj_in_light": (0.3, 30)},
               "current": {"look_at_obj_in_light": (0.4, 30)},
               "candidate": {"look_at_obj_in_light": (0.7, 30)}}
    st = L.new_state()
    st = L.run_cycle(st, _mock_fns([0.42], action, measure, calls), {"val_n": 1})
    ok(st["scaffold"]["skills"]["look_at_obj_in_light"] == "take the object first",
       "ACCEPT: winning text applied to scaffold")
    ok(calls["persist"]["skills"]["look_at_obj_in_light"] == "take the object first",
       "ACCEPT: updated scaffold persisted to disk")
    ok(st["decision_history"][-1]["accepted_text"] and st["decision_history"][-1]["verdict"] == "accepted",
       "ACCEPT: decision logged as accepted")


def test_loop_reject_takes_the_p_edits_down_with_it():
    """A rejected text proposal now voids the p edits submitted with it.

    The Teacher proposes text and p as ONE action behind ONE diagnosis, and p is the half no
    measurement ever sees. Observed in cycle 1 of alf_scratch150_pcap: the text scored 0.078
    against a bare 0.128 and was rejected, yet p still went 0 -> 0.35/0.5. To get a p change
    judged on its own the Teacher must submit it WITHOUT text (covered below)."""
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    action = {"diagnosis": "d", "p_ops": [{"task": "pick_and_place", "p": 0.2}],
              "text_ops": [{"target": "pick_and_place", "text": "worse wording"}]}
    measure = {"bare": {"pick_and_place": (0.30, 30)},
               "current": {"pick_and_place": (0.40, 30)},
               "candidate": {"pick_and_place": (0.20, 30)}}      # candidate < current -> reject
    fns = _mock_fns([0.50], action, measure, calls)
    st = L.run_cycle(L.new_state(), fns, {"val_n": 1})
    ok(st["scaffold"]["p_task"]["pick_and_place"] == 0.0,
       "text rejected -> the p edit that rode with it is discarded too")
    ok(st["scaffold"]["skills"]["pick_and_place"] == "", "rejected text was not applied either")
    d = st["decision_history"][-1]
    ok(d["verdict"] == "rejected" and d["p_vetoed_with_text"] is True,
       "the veto is recorded, not silent")

    # a p-only action has no A/B verdict to be vetoed by, so it applies
    calls2 = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    fns2 = _mock_fns([0.50], {"diagnosis": "d", "text_ops": [],
                              "p_ops": [{"task": "pick_and_place", "p": 0.2}]}, {}, calls2)
    st2 = L.run_cycle(L.new_state(), fns2, {"val_n": 1})
    ok(st2["scaffold"]["p_task"]["pick_and_place"] == 0.2, "p-only proposal applies on its own")
    ok(calls2["measure"] == 0, "no text change -> no A/B measurement paid for")


def test_there_is_no_revert():
    """The revert gate was removed 2026-07-29. Pinned here because its failure mode was silent:
    measured cycle volatility (~+-0.08) exceeded any usable margin, so it anchored on whichever
    cycle got lucky and reverted forever. A sustained drop must now simply stay in the curve."""
    ok(not hasattr(gates, "revert_gate"), "gates.revert_gate is gone")
    from agent_system.skill_opt.autoscaffold import adapters as A
    ok(not hasattr(A, "revert_to_step"), "adapters.revert_to_step is gone")
    ok("best_checkpoint" not in L.new_state() and "best_scaffold" not in L.new_state(),
       "restore-only state is gone")

    seq = [0.60, 0.40, 0.38]                       # would have tripped the old gate twice over
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    fns = _mock_fns(seq, {"diagnosis": "d", "text_ops": [], "p_ops": []}, {}, calls)
    st = L.new_state()
    for _ in range(3):
        st = L.run_cycle(st, fns, {"val_n": 1})
    ok(calls["restore"] == 0, "nothing rewinds the weights")
    ok(st["step"] == 30, "the step counter keeps advancing through the drop")
    ok(st["sr_history"] == seq, "every eval stays in the history; none is truncated away")
    ok(st["best"] == 0.60 and st["best_step"] == 10, "best is still tracked, for reporting only")
    ok(all(e["verdict"] != "reverted" for e in st["decision_history"]), "no cycle is a revert")


def test_loop_noop():
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    st = L.new_state()
    st = L.run_cycle(st, _mock_fns([0.5], {"text_ops": [], "p_ops": []}, {}, calls), {"val_n": 1})
    ok(calls["measure"] == 0, "NO-OP: A/B not invoked when Teacher declines")
    ok(st["scaffold"]["version"] == 0 and st["decision_history"][-1]["verdict"] == "noop",
       "NO-OP: scaffold unchanged, logged as noop")



def test_adapters_pure():
    from agent_system.skill_opt.autoscaffold import adapters as A
    import json, tempfile, os
    tr_bare = [{"task_type": "look_at_obj_in_light", "success": False}] * 3 + \
              [{"task_type": "look_at_obj_in_light", "success": True}]        # 1/4 = 0.25
    tr_cand = [{"task_type": "look_at_obj_in_light", "success": True}] * 3 + \
              [{"task_type": "look_at_obj_in_light", "success": False}]        # 3/4 = 0.75
    agg = A.agg_per_task(tr_bare, ["look_at_obj_in_light"])
    ok(agg["look_at_obj_in_light"] == (0.25, 4), "agg_per_task computes rate + n")
    m = A.assemble_measure(tr_bare, tr_bare, tr_cand, ["look_at_obj_in_light"])
    ok(m["bare"]["look_at_obj_in_light"][0] == 0.25 and m["candidate"]["look_at_obj_in_light"][0] == 0.75,
       "assemble_measure -> ab_gate-shaped dict")
    ok(gates.ab_gate(m, ["look_at_obj_in_light"])["accept"], "assembled measure feeds ab_gate (accept)")

    groups = [{"task": "look_at_obj_in_light", "outcomes": [False, False, False]},   # all-fail
              {"task": "look_at_obj_in_light", "outcomes": [False, True, False]},     # not all-fail
              {"task": "pick_two_obj_and_place", "outcomes": [False, False]}]         # all-fail
    sig = A.assemble_signals(tr_bare, tr_cand, groups, ["look_at_obj_in_light", "pick_two_obj_and_place"])
    ok(sig["per_task_gap"]["look_at_obj_in_light"]["gap"] == 0.5, "assemble_signals computes the gap (0.75-0.25)")
    ok(sig["all_fail_groups"]["look_at_obj_in_light"] == {"all_fail": 1, "total": 2},
       "assemble_signals counts all-fail groups")
    ok(len(sig["failures"]) == 3, "assemble_signals collects the failed bare transcripts")

    # group_by_gamefile: several rollouts of the same game -> one group; all-fail detection
    flat = [{"gamefile": "gA", "task_type": "look_at_obj_in_light", "success": False},
            {"gamefile": "gA", "task_type": "look_at_obj_in_light", "success": False},
            {"gamefile": "gB", "task_type": "look_at_obj_in_light", "success": False},
            {"gamefile": "gB", "task_type": "look_at_obj_in_light", "success": True}]
    grps = A.group_by_gamefile(flat)
    ok(len(grps) == 2 and grps[0]["outcomes"] == [False, False], "group_by_gamefile groups per game")
    sg = A.assemble_signals(flat, flat, grps, ["look_at_obj_in_light"])
    ok(sg["all_fail_groups"]["look_at_obj_in_light"] == {"all_fail": 1, "total": 2},
       "grouped -> 1 all-fail group (gA) of 2 games")

    d = tempfile.mkdtemp(); p = os.path.join(d, "scaf.json")
    A.persist_scaffold(S.empty_scaffold(), p)
    ok(json.load(open(p))["version"] == 0 and not os.path.exists(p + ".tmp"),
       "persist_scaffold writes atomically (no leftover .tmp)")



def test_graceful_kill():
    from agent_system.skill_opt.autoscaffold import adapters as A
    import subprocess
    p = subprocess.Popen(["sleep", "120"])
    ok(A._alive(p.pid), "_alive: true for a running process")
    A._graceful_kill([p.pid], grace=6)
    ok(not A._alive(p.pid), "_graceful_kill: SIGTERM stops a normal process within grace (no SIGKILL needed)")
    ok(not A._alive(999999999), "_alive: false for a nonexistent pid")



def test_journal_memory():
    """Rejected/accepted proposals are journaled WITH their wording, and replayed as memory."""
    from agent_system.skill_opt.autoscaffold import observation as O
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    action = {"diagnosis": "look_at is weakest", "p_ops": [],
              "text_ops": [{"target": "look_at_obj_in_light", "text": "TAKE THE OBJECT FIRST"}]}
    measure = {"bare": {"look_at_obj_in_light": (0.3, 30)},      # candidate LOSES -> rejected
               "current": {"look_at_obj_in_light": (0.5, 30)},
               "candidate": {"look_at_obj_in_light": (0.2, 30)}}
    fns = _mock_fns([0.4], action, measure, calls)
    seen = {}
    fns["journal_fn"] = lambda hist: seen.update({"hist": hist})
    st = L.run_cycle(L.new_state(), fns, {"val_n": 1})
    ok(st["decision_history"][-1]["verdict"] == "rejected", "journal: losing proposal marked rejected")
    prop = st["decision_history"][-1]["summary"]["text_proposed"]
    ok(prop.get("look_at_obj_in_light") == "TAKE THE OBJECT FIRST",
       "journal: the REJECTED wording is recorded (not just the target name)")
    ok(seen.get("hist") is st["decision_history"], "journal_fn called with the decision history")

    # memory replay: recent entries keep the text; older ones are compacted away
    hist = [{"cycle": i, "summary": {"text_proposed": {"general": "old %d" % i},
                                     "diagnosis": "d" * 500}, "verdict": "rejected"}
            for i in range(10)]
    comp = O.compact_history(hist, recent=2)
    ok("text_proposed" in comp[-1]["summary"], "memory: newest cycle keeps wording verbatim")
    ok("text_proposed" not in comp[0]["summary"], "memory: old cycle drops wording (prompt bound)")
    ok(len(comp[0]["summary"]["diagnosis"]) <= 200, "memory: old diagnosis truncated")
    obs = O.assemble_observation(S.empty_scaffold(), {}, hist, step=1)
    ok("text_proposed" in O.render_user_prompt(obs), "memory: proposals reach the Teacher prompt")
    ok("YOUR OWN MEMORY" in O.render_system_prompt(), "memory: prompt explains the history as memory")



def test_signals_contrast_and_computed():
    """Gap fixes: successes are kept as contrast, and each trajectory carries computed signals."""
    from agent_system.skill_opt.autoscaffold import adapters as A
    mk = lambda ok, acts, invalid=0: {
        "task_type": "look_at_obj_in_light", "gamefile": f"g{ok}{len(acts)}", "success": ok,
        "steps": [{"obs": "o", "action": a, "valid": (i >= invalid)} for i, a in enumerate(acts)]}
    bare = [mk(False, ["go to drawer 1", "open drawer 1", "look"], invalid=1),
            mk(True, ["take cd from desk", "use desklamp"])]
    sig = A.assemble_signals(bare, bare, [], ["look_at_obj_in_light"])
    ok(len(sig["successes"]) == 1, "successes_for_contrast is now populated")
    ok(len(sig["failures"]) == 1, "failures still collected")
    c = sig["failures"][0]["computed"]
    ok(c["opens"] == 1 and c["takes"] == 0, "computed: opens/takes counted")
    ok(c["held_an_object"] is False, "computed: held_an_object False when never took")
    ok(c["invalid_actions"] == 1, "computed: invalid actions counted")
    ok(sig["successes"][0]["computed"]["held_an_object"] is True, "computed: success held an object")


def test_domain_driven_prompt():
    """The prompt gives STRUCTURE + a content vocabulary, never the granularity to use."""
    from agent_system.skill_opt.autoscaffold import observation as O
    p = O.render_system_prompt()                       # ALFWorld default
    ok("one skill per task category" not in p, "no prior: does not tell it to write per-category skills")
    ok("category labels" in p and "pick_two_obj_and_place" in p, "structure: category labels + meanings given")
    ok(all(w in p for w in ["skills:", "hints:", "examples:", "rubrics:"]), "content vocabulary offered")
    ok("not modes to select" in p, "vocabulary framed as descriptive, not a mode switch")
    ok("Reference (known-good) solutions available per instance: NO" in p,
       "ALFWorld: told there are NO reference solutions (so hints are not possible)")

    math = S.Domain(name="math", episode_desc="one problem, boxed answer",
                    has_reference_solutions=True, instance_scope=True)
    pm = O.render_system_prompt(math)
    ok("Reference (known-good) solutions available per instance: YES" in pm,
       "math domain: told reference solutions EXIST")
    ok("attached to an individual instance: YES" in pm, "math domain: per-instance scope advertised")
    ok("Instances carry no category labels" in pm, "math domain: no categories -> stated")
    ok(math.scopes() == ["general"], "math domain scopes = general only (no categories)")
    ok(S.ALF_DOMAIN.scopes() == ["general"] + S.TASKS, "ALFWorld scopes = general + 6 categories")


def test_failed_step_is_loud():
    """A training subprocess that dies must NOT look like a completed cycle. Regression test
    for the 2026-07-25 run where Ray failed to start, no checkpoint was written, and the loop
    still logged '[c1] step=10 valid_seen avg=None' and marched on to ask the Teacher for edits."""
    import os
    import tempfile
    from agent_system.skill_opt.autoscaffold import adapters as A
    with tempfile.TemporaryDirectory() as d:
        ok(not A.ckpt_is_usable(f"{d}/global_step_10"), "missing ckpt dir -> not usable")
        os.makedirs(f"{d}/global_step_10/hf_model")
        ok(not A.ckpt_is_usable(f"{d}/global_step_10"),
           "hf_model-only ckpt -> not usable (would silently resume from base)")
        os.makedirs(f"{d}/global_step_10/actor")
        ok(not A.ckpt_is_usable(f"{d}/global_step_10"),
           "actor/ dir but no shards -> torn checkpoint, NOT usable")
        make_usable_ckpt(f"{d}/global_step_10")
        ok(A.ckpt_is_usable(f"{d}/global_step_10"), "complete shard set -> usable")

    # the loop must propagate the failure, not swallow it into another cycle
    st = L.new_state()

    def boom(scaf, frm, to):
        raise A.StepFailed("no checkpoint")

    raised = False
    try:
        L.run(st, {"train_fn": boom, "eval_fn": lambda *_: {"avg": 0.5},
                   "signals_fn": lambda *_: {}, "measure_ab_fn": lambda *_: {},
                   "teacher_fn": lambda o: ({}, "")}, {}, 3)
    except A.StepFailed:
        raised = True
    ok(raised, "StepFailed propagates out of the loop (run aborts loudly)")
    ok(st["decision_history"] == [], "no decision recorded for the failed cycle")


def test_resume_after_crash():
    """A crash mid-cycle must not cold-start the next run. Regression test for the 2026-07-26
    vLLM-timeout crash: main() unconditionally rebuilt an EMPTY state and re-persisted an empty
    scaffold, so a restart would (a) destroy the live scaffold and (b) set step=0 while
    global_step_10..40 still existed on disk -> ckpt_is_usable() passes on a checkpoint from
    four cycles earlier and the loop evaluates stale weights."""
    import json
    import tempfile
    from agent_system.skill_opt.autoscaffold import run_arm as R

    with tempfile.TemporaryDirectory() as d:
        cfg = {"state_path": f"{d}/state.json"}
        ok(R.load_state(cfg) is None, "no state file -> cold start (None)")

        # simulate: crash right after cycle 4's eval, with a non-empty scaffold
        live = S.apply_text_ops(S.empty_scaffold(), [{"target": "general", "text": "PROC"}])
        live = S.apply_p_ops(live, [{"task": "pick_and_place", "p": 0.0}])
        st = L.new_state(scaffold=live)
        st.update({"cycle": 4, "step": 40, "sr_history": [0.0993, 0.1693, 0.289, 0.3203],
                   "best": 0.3203, "best_step": 40,
                   "best_checkpoint": "/ckpts/exp/global_step_40",
                   })
        fns = {"state_fn": lambda s: A.persist_scaffold(
            {k: s[k] for k in R.STATE_KEYS if k in s}, cfg["state_path"])}
        L._save(st, fns)

        back = R.load_state(cfg)
        ok(back["step"] == 40, "resume restores step (not 0)")
        ok(back["cycle"] == 4, "resume restores cycle counter")
        ok(back["best"] == 0.3203 and back["best_step"] == 40, "resume restores the revert anchor")
        ok(back["scaffold"]["general_skill"] == "PROC", "resume restores the LIVE scaffold text")
        ok(back["scaffold"]["p_task"]["pick_and_place"] == 0.0, "resume restores per-task p")
        ok(back["sr_history"] == [0.0993, 0.1693, 0.289, 0.3203], "resume restores sr_history")

        merged = {**L.new_state(), **back}
        ok(merged["step"] == 40 and merged["scaffold"]["general_skill"] == "PROC",
           "merge over new_state keeps resumed values (defaults do not clobber)")

    # the eval result must be saved BEFORE the teacher/A-B stage, so a crash there keeps it
    saves = []
    st = L.new_state()
    fns = {"train_fn": lambda *a: "/ckpts/global_step_10",
           "eval_fn": lambda *a: {"avg": 0.42, "draws": [0.42]},
           "signals_fn": lambda *a: (_ for _ in ()).throw(RuntimeError("crash after eval")),
           "state_fn": lambda s: saves.append(copy.deepcopy(s)),
           "log": lambda *a: None}
    try:
        L.run_cycle(st, fns, {})
    except RuntimeError:
        pass
    ok(len(saves) >= 1, "state saved before the post-eval stages")
    ok(saves[0]["sr_history"] == [0.42] and saves[0]["step"] == 10,
       "the saved state already carries the expensive eval result")


def test_domain_parameterised_action_space():
    """Regression for the 2026-07-26 math probe: the Teacher correctly proposed per-subject
    text for MATH, and validate_action rejected every op because the ALFWorld task list was
    hardcoded — the whole proposal silently degraded to a no-op with an empty diagnosis."""
    from agent_system.skill_opt.mathscaffold.domain import MATH_DOMAIN

    act = {"diagnosis": "d", "p_ops": [],
           "text_ops": [{"target": "Intermediate Algebra", "text": "check the discriminant"}]}
    ok(not S.validate_action(act)[0], "math target rejected under the ALFWorld default (as before)")
    ok(S.validate_action(act, MATH_DOMAIN)[0], "math target ACCEPTED under the math domain")
    ok(not S.validate_action(
        {"text_ops": [{"target": "pick_and_place", "text": "x"}], "p_ops": []}, MATH_DOMAIN)[0],
       "ALFWorld target rejected under the math domain (domains do not leak)")

    a2, note = teacher.normalize(act, MATH_DOMAIN)
    ok(a2["text_ops"] and note == "ok", "teacher.normalize honours the domain")

    sc = S.empty_scaffold(MATH_DOMAIN)
    ok(set(sc["skills"]) == set(MATH_DOMAIN.categories) and len(sc["skills"]) == 7,
       "empty_scaffold builds the 7 MATH subject slots, not ALFWorld's 6 tasks")
    ok(S.validate_scaffold(sc, MATH_DOMAIN)[0], "math scaffold validates under its own domain")
    ok(not S.validate_scaffold(sc)[0], "math scaffold fails ALFWorld validation (slots differ)")

    sc2 = S.apply_text_ops(sc, act["text_ops"])
    ok(sc2["skills"]["Intermediate Algebra"] == "check the discriminant", "text applied to subject")
    ok(S.touched_tasks([{"target": "general", "text": "g"}], MATH_DOMAIN)
       == sorted(MATH_DOMAIN.categories), "general edit touches all 7 subjects")

    # ALFWorld behaviour must be byte-identical to before the change
    alf = S.empty_scaffold()
    ok(set(alf["skills"]) == set(S.TASKS), "default (no domain) is still ALFWorld")



def test_cold_start_inherits_ckpt_step():
    """Regression for the 2026-07-27 from-step150 launch. Clearing state.json to reset the
    scaffold and Teacher memory also reset the STEP to 0, so cycle 1 asked verl to train
    "to step 10" while the seeded weights were already at 150. verl resumed, ran one step,
    exited, and the loop looked for global_step_10 — which was never written."""
    import os
    import tempfile
    from agent_system.skill_opt.autoscaffold import adapters as A

    with tempfile.TemporaryDirectory() as root:
        ok(A.existing_ckpt_step(root) == 0, "empty dir -> step 0 (a genuine cold start)")
        make_usable_ckpt(f"{root}/global_step_150")
        make_usable_ckpt(f"{root}/global_step_40")
        ok(A.existing_ckpt_step(root) == 150, "picks the HIGHEST existing step, not the first")
        os.makedirs(f"{root}/global_step_200/hf_model")          # no actor/ -> unusable
        ok(A.existing_ckpt_step(root) == 150,
           "an unusable checkpoint is ignored (hf_model alone would resume from base)")


def main():
    for fn in [test_scaffold, test_ab_gate, test_revert_gate, test_teacher_normalize,
               test_adapters_pure, test_graceful_kill, test_journal_memory,
               test_signals_contrast_and_computed, test_domain_driven_prompt,
               test_failed_step_is_loud, test_resume_after_crash,
               test_domain_parameterised_action_space,
               test_revert_repoints_tracker_and_step, test_revert_can_be_disabled,
               test_cold_start_inherits_ckpt_step,
               test_loop_accept, test_loop_reject_but_apply_p, test_loop_noop, test_loop_revert]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()


def test_eval_and_train_agree_on_world_size(monkeypatch, tmp_path):
    """Eval loads the FSDP shards training wrote; the filenames encode the world size
    (model_world_size_4_rank_*.pt), so a GPU-count mismatch makes verl look for shards that
    were never written. Regression: eval_adapter once omitted these and defaulted to 2 GPUs
    against a 4-GPU checkpoint."""
    seen = {}

    class _P:
        returncode = 0

    def fake_run(cmd, log, env):
        seen["none" if " none " in cmd else "train"] = dict(env)
        return _P()

    monkeypatch.setattr(A, "_run", fake_run)
    # False on the pre-train probe (so training actually runs), True on the post-train check.
    seen_probe = {"n": 0}

    def usable(path):
        seen_probe["n"] += 1
        return seen_probe["n"] > 1

    monkeypatch.setattr(A, "ckpt_is_usable", usable)
    monkeypatch.setattr("agent_system.skill_opt.push98_loop.parse_val",
                        lambda log, with_counts=False: (0.9, {"pick_and_place": 0.9},
                                                        {"pick_and_place": 19})
                        if with_counts else (0.9, {"pick_and_place": 0.9}))
    cfg = {"exp": "t", "gpus": "0,1,6,7", "model": "/m", "ray_tmp": "/dev/shm/t",
           "scaffold_path": str(tmp_path / "s.json"), "train_log": str(tmp_path / "t.log"),
           "log_dir": str(tmp_path), "n_gpus": 4, "tp_size": 2, "gpu_mem": 0.35, "val_bs": 64}

    A.train_adapter(cfg["scaffold_path"], 150, 160, cfg)
    A.eval_adapter("/ckpt/global_step_160", 1, cfg)

    for key in ("N_GPUS", "TP_SIZE", "GPU_MEM", "VAL_BS"):
        assert seen["train"][key] == seen["none"][key], (
            f"{key}: train={seen['train'][key]} eval={seen['none'][key]}")
    assert seen["none"]["N_GPUS"] == "4"


def _prime_fns(calls):
    def train_fn(scaf, frm, to):
        calls.append(("train", copy.deepcopy(scaf)))
        return f"/ckpt/global_step_{to}"

    return {"train_fn": train_fn,
            "eval_fn": lambda c, n: {"avg": 0.9, "per_task": {}, "draws": [0.9]},
            "signals_fn": lambda c, s: {},
            # {arm: {task: (success_rate, n)}} — same games across arms (paired comparison)
            "measure_ab_fn": lambda c, cur, cand, t: {
                "bare": {x: (0.10, 30) for x in t},
                "current": {x: (0.20, 30) for x in t},
                "candidate": {x: (0.50, 30) for x in t}},
            "teacher_fn": lambda obs: (
                {"text_ops": [{"target": "pick_two_obj_and_place", "text": "two objects: place one, then fetch the second"}],
                 "p_ops": [{"task": "pick_two_obj_and_place", "p": 0.7}], "diagnosis": "weakest"},
                "wrote pick_two"),
            "persist_fn": lambda s: calls.append(("persist", copy.deepcopy(s))),
            "log": lambda m: None}


def test_prime_teacher_writes_before_first_training():
    """A cycle is train -> eval -> Teacher, so without priming the FIRST K steps train against
    the start scaffold — empty on a seeded start, i.e. pure-RL steps wearing the experiment's
    name. Regression: alf_from150 burned step 160->170 that way."""
    calls = []
    fns = _prime_fns(calls)
    cfg = {"exp": "t", "steps_per_cycle": 10, "val_n": 1,
           "domain": S.ALF_DOMAIN}
    state = L.new_state(step0=160, scaffold=S.empty_scaffold())
    state.update(sr_history=[0.8957], best=0.8957, best_step=160,
                 best_checkpoint="/ckpt/global_step_160",
                 last_eval={"avg": 0.8957, "per_task": {"pick_two_obj_and_place": 0.73}})

    L.run(state, fns, cfg, n_cycles=1)

    first_train = next(c for c in calls if c[0] == "train")
    assert first_train[1]["skills"]["pick_two_obj_and_place"], \
        "first training stretch still ran with an empty scaffold"
    # the Teacher asked for 0.7; from a cold start P_MAX_DELTA is the binding limit (P_MAX would
    # allow 0.5, but one cycle may only move p by P_MAX_DELTA), so training sees 0.2, not 0.7.
    assert first_train[1]["p_task"]["pick_two_obj_and_place"] == S.P_MAX_DELTA


def test_prime_is_noop_when_teacher_already_spoke():
    """A mid-run restart must not decide twice off the same eval."""
    calls = []
    fns = _prime_fns(calls)
    cfg = {"exp": "t", "domain": S.ALF_DOMAIN}
    state = L.new_state(step0=160)
    state.update(sr_history=[0.9], last_eval={"avg": 0.9},
                 decision_history=[{"cycle": 1, "verdict": "accepted"}])
    out = L.prime(state, fns, cfg)
    assert len(out["decision_history"]) == 1
    assert not [c for c in calls if c[0] == "persist"]


def test_prime_is_noop_without_an_eval():
    """True cold start: nothing measured yet, so the Teacher has nothing to read."""
    calls = []
    state = L.prime(L.new_state(step0=0), _prime_fns(calls), {"exp": "t", "domain": S.ALF_DOMAIN})
    assert state["decision_history"] == []
    assert calls == []


def test_injects_nothing():
    assert S.injects_nothing(S.empty_scaffold())
    assert S.injects_nothing(None)
    sc = S.empty_scaffold()
    sc["skills"]["pick_two_obj_and_place"] = "   "        # whitespace still renders to ""
    assert S.injects_nothing(sc)
    sc["skills"]["pick_two_obj_and_place"] = "leave the first object where you put it"
    assert not S.injects_nothing(sc)
    g = S.empty_scaffold()
    g["general_skill"] = "search receptacles in order"     # general alone is enough to inject
    assert not S.injects_nothing(g)


def test_ab_skips_redundant_current_arm_when_empty(monkeypatch):
    """An empty current scaffold splices to the identical prompt, so running it as its own arm
    is a second independently-noisy sample of the SAME condition as bare. Measured once, those
    two arms differed by 0.05 on a difference that is zero by construction — 3x the 0.016
    margin the gate then ruled on."""
    seen = {}

    def fake_serve(checkpoint, cfg, passes, seed):
        seen["labels"] = [p[0] for p in passes]
        return {lbl: [{"task_type": "pick_and_place", "success": True}] for lbl, *_ in passes}

    monkeypatch.setattr(A, "_serve_and_rollout", fake_serve)
    cfg = {"n_per_task": 30}
    empty = S.empty_scaffold()
    cand = S.apply_text_ops(empty, [{"target": "pick_and_place", "text": "open closed drawers"}])

    m = A.measure_ab_adapter("/ckpt", empty, cand, ["pick_and_place"], cfg, seed=1)
    assert seen["labels"] == ["bare", "candidate"], "redundant 'current' arm was still run"
    assert m["current"] == m["bare"], "current must reuse bare when it injects nothing"

    m2 = A.measure_ab_adapter("/ckpt", cand, cand, ["pick_and_place"], cfg, seed=1)
    assert "current" in seen["labels"], "a non-empty current scaffold must get its own arm"


def test_partial_eval_warns_and_reports_sample_size(monkeypatch, tmp_path):
    """A draw can die on a Ray collision with the training run tearing down. Averaging the
    survivors is right, but the result must carry its sample size: this eval anchors every
    accept/revert decision. Regression: step 190 silently reported a 2-draw mean as if it were 3."""
    seen = []
    vals = iter([(0.938, {"pick_and_place": 0.9}), (None, {}), (0.922, {"pick_and_place": 0.9})])

    class _P:
        returncode = 0

    monkeypatch.setattr(A, "_run", lambda cmd, log, env: _P())
    monkeypatch.setattr("agent_system.skill_opt.push98_loop.parse_val",
                        lambda log, with_counts=False: (lambda v: (*v, {}) if with_counts else v)(next(vals)))
    cfg = {"exp": "t", "gpus": "0,1", "model": "/m", "ray_tmp": "/dev/shm/t",
           "scaffold_path": str(tmp_path / "s.json"), "log_dir": str(tmp_path),
           "n_gpus": 2, "tp_size": 2, "gpu_mem": 0.6, "val_bs": 64,
           "log": lambda m: seen.append(m)}

    ev = A.eval_adapter("/ckpt/global_step_190", 3, cfg)

    assert ev["n_draws"] == 2 and ev["n_draws_requested"] == 3
    assert ev["avg"] == round((0.938 + 0.922) / 2, 4)
    assert any("only 2/3 draws" in m for m in seen), f"no warning emitted: {seen}"


def test_priors_are_off_by_default_and_opt_in():
    """The zero-prior prompt is a scientific position, not an oversight: what the Teacher works
    out from signals alone is a result, and priors turn it into an instruction. Keep both
    runnable so they can be compared."""
    from agent_system.skill_opt.autoscaffold import observation as O
    plain = O.render_system_prompt(S.ALF_DOMAIN)
    withp = O.render_system_prompt(S.ALF_DOMAIN, priors=True)
    assert "PRIOR KNOWLEDGE" not in plain
    assert "PRIOR KNOWLEDGE" in withp
    assert len(withp) > len(plain)
    # the descriptive core must be identical in both
    for chunk in ("THE INJECTION MECHANISM", "WHAT THE SIGNALS MEAN", "DOMAIN STRUCTURE"):
        assert chunk in plain and chunk in withp
    # the claims the priors make must be the ones this run actually measured
    for claim in ("37.6", "42.9", "+0.167", "UNGUARDED"):
        assert claim in withp


def test_priors_reach_the_teacher_through_cfg(monkeypatch):
    seen = {}

    def fake_call(system, user):
        seen["system"] = system
        return '{"diagnosis": "d", "text_ops": [], "p_ops": []}'

    teacher.propose({}, call_fn=fake_call, domain=S.ALF_DOMAIN, priors=True)
    assert "PRIOR KNOWLEDGE" in seen["system"]
    teacher.propose({}, call_fn=fake_call, domain=S.ALF_DOMAIN)
    assert "PRIOR KNOWLEDGE" not in seen["system"]


def test_both_ray_registration_timeouts_are_set():
    """Ray has two independent registration deadlines. Only raising the agent one leaves the
    worker one at its 60s default, and on a loaded box that turns the ALFWorld eval into a
    spawn/reap thrash loop that pins the machine without ever erroring out."""
    env = A._ray_env() if hasattr(A, "_ray_env") else None
    if env is None:                       # helper is inlined; assert on the module constants
        assert A.RAY_AGENT_REGISTER_TIMEOUT_MS and A.RAY_WORKER_REGISTER_TIMEOUT_S
        return
    assert env["RAY_agent_register_timeout_ms"] == A.RAY_AGENT_REGISTER_TIMEOUT_MS
    assert env["RAY_worker_register_timeout_seconds"] == A.RAY_WORKER_REGISTER_TIMEOUT_S
    assert int(A.RAY_WORKER_REGISTER_TIMEOUT_S) > 60


def test_train_is_idempotent_when_checkpoint_exists(monkeypatch, tmp_path):
    """Each to_step is trained exactly once, so an existing checkpoint means training already
    finished and the loop died afterwards (typically during the eval). Retraining would ask verl
    to advance to a step it is already at -- it resumes, does nothing, writes nothing, and the
    cycle dies with StepFailed. Observed at the step-200 boundary; unattended restarts need this."""
    ran = []
    monkeypatch.setattr(A, "_run", lambda cmd, log, env: ran.append(cmd))
    monkeypatch.setattr(A, "ckpt_is_usable", lambda p: p.endswith("global_step_160"))
    logged = []
    cfg = {"exp": "t", "gpus": "0,1", "model": "/m", "ray_tmp": "/dev/shm/t",
           "train_log": str(tmp_path / "t.log"), "n_gpus": 4, "tp_size": 2,
           "gpu_mem": 0.35, "val_bs": 64, "log": logged.append}

    ckpt = A.train_adapter(str(tmp_path / "s.json"), 150, 160, cfg)

    assert ckpt.endswith("global_step_160")
    assert ran == [], "retrained despite the checkpoint already being on disk"
    assert any("skipping retrain" in m for m in logged)


def _write_ckpt(root, step, world_size=4, kinds=("model", "optim", "extra_state"),
                ranks=None, data_pt=True):
    import os
    d = os.path.join(str(root), f"global_step_{step}", "actor")
    os.makedirs(d, exist_ok=True)
    for kind in kinds:
        for r in (range(world_size) if ranks is None else ranks):
            open(os.path.join(d, f"{kind}_world_size_{world_size}_rank_{r}.pt"), "w").close()
    if data_pt:
        open(os.path.join(str(root), f"global_step_{step}", "data.pt"), "w").close()
    return os.path.join(str(root), f"global_step_{step}")


def test_torn_checkpoint_is_not_usable(tmp_path):
    """verl mkdirs actor/ before writing ~19 GB of shards and updates the tracker only at the
    end. A crash in that window leaves a directory that looks complete. Calling it usable makes
    train_adapter skip retraining, and the eval then resumes from the TRACKER -- an older step --
    and the loop records that score as this step's. No training, no error, wrong number."""
    assert A.ckpt_is_usable(_write_ckpt(tmp_path, 10))                    # complete

    # actor/ exists, nothing written yet
    import os
    os.makedirs(os.path.join(str(tmp_path), "global_step_20", "actor"), exist_ok=True)
    assert not A.ckpt_is_usable(os.path.join(str(tmp_path), "global_step_20"))

    # model shards for every rank, but the optim/extra half never landed
    assert not A.ckpt_is_usable(_write_ckpt(tmp_path, 30, kinds=("model",)))

    # one rank missing
    assert not A.ckpt_is_usable(_write_ckpt(tmp_path, 40, ranks=[0, 1, 2]))

    # shards complete but the trainer's data.pt (written after the actor save) is absent
    assert not A.ckpt_is_usable(_write_ckpt(tmp_path, 50, data_pt=False))

    # hf_model only -- the original verl resume gotcha
    os.makedirs(os.path.join(str(tmp_path), "global_step_60", "actor", "huggingface"), exist_ok=True)
    assert not A.ckpt_is_usable(os.path.join(str(tmp_path), "global_step_60"))

    # existing_ckpt_step must inherit the stricter check, not just pick the highest directory
    _write_ckpt(tmp_path, 70, kinds=("model",))          # torn, higher number
    assert A.existing_ckpt_step(str(tmp_path)) == 10


def test_failure_trim_is_balanced_and_declared():
    """A 50-step ALFWorld failure serialises to ~8k chars, so 40 of them blow a 160k budget and a
    tail trim deletes whole task types picked by list order alone. The Teacher then cannot tell
    "no failures here" from "trimmed away", and it reasons from exactly this field."""
    from agent_system.skill_opt.autoscaffold import observation as O
    tasks = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light"]
    # Realistic size: a 50-step ALFWorld failure carries the full observation text per step and
    # serialises to roughly 8k chars, so 36 of them are ~290k against a 160k budget.
    obs_text = "You arrive at cabinet 1. The cabinet 1 is closed. " * 3
    fails = [{"task_type": t,
              "steps": [{"action": "go to cabinet 1", "obs": obs_text, "valid": True}] * 50}
             for t in tasks for _ in range(12)]          # grouped by task, 36 total, oversized
    obs = {"failure_trajectories": fails, "per_task_gap": {}, "valid_seen": {"avg": 0.3}}

    body = O.render_user_prompt(obs)
    packet = json.loads(body.split("\n", 1)[1])

    kept = packet["failure_trajectories"]
    assert len(kept) < len(fails), "test data was not big enough to trigger a trim"
    seen = {f["task_type"] for f in kept}
    assert seen == set(tasks), f"a task type was trimmed away entirely: kept {seen}"
    counts = [sum(1 for f in kept if f["task_type"] == t) for t in tasks]
    assert max(counts) - min(counts) <= 1, f"unbalanced across task types: {counts}"

    d = packet["failure_trajectories_dropped"]
    assert d["kept"] == len(kept) and d["of"] == len(fails)
    assert set(d["of_by_task"]) == set(tasks)


def test_no_trim_key_when_everything_fits():
    from agent_system.skill_opt.autoscaffold import observation as O
    obs = {"failure_trajectories": [{"task_type": "pick_and_place", "steps": []}],
           "per_task_gap": {}}
    packet = json.loads(O.render_user_prompt(obs).split("\n", 1)[1])
    assert "failure_trajectories_dropped" not in packet
    assert len(packet["failure_trajectories"]) == 1


def test_p_is_capped_at_p_max():
    """Always injecting is the documented failure mode (matched ablation elsewhere: never
    withdrawing landed BELOW the no-scaffold baseline, 37.6 vs 42.9). The cap guarantees at least
    half of every category's groups see the bare prompt regardless of what the Teacher proposes."""
    assert S.P_MAX == 0.5
    sc = S.empty_scaffold()
    assert all(v == 0.0 for v in sc["p_task"].values()), "cold start must inject nothing"
    assert sc["default_p"] == 0.0, "injection is opt-in, not opt-out"

    # The cap binds on the DESTINATION. Start close enough that P_MAX_DELTA is not the limiting
    # factor, or the step limit would mask the cap being tested here (see the delta test below).
    near = S.apply_p_ops(sc, [{"task": "pick_two_obj_and_place", "p": 0.2}])
    near = S.apply_p_ops(near, [{"task": "pick_two_obj_and_place", "p": 0.4}])
    out = S.apply_p_ops(near, [{"task": "pick_two_obj_and_place", "p": 1.0},
                               {"task": "look_at_obj_in_light", "p": 0.3}])
    assert out["p_task"]["pick_two_obj_and_place"] == S.P_MAX, "over-cap p was not clamped"
    assert out["p_task"]["look_at_obj_in_light"] == 0.2, "0 -> 0.3 exceeds the per-cycle step"
    assert abs(out["p_clamped"]["requested"]["pick_two_obj_and_place"] - 0.6) < 1e-9, \
        "clamp records the post-step-limit request"
    assert sc["p_task"]["pick_two_obj_and_place"] == 0.0, "input was mutated"


def test_p_moves_at_most_one_step_per_cycle():
    """p is the lever with no measurement behind it: text must clear a paired A/B, p never does.
    Bounding the STEP (not just the destination) means every intermediate value gets a cycle of
    held-out evidence before the next move, so no single unverified judgment can swing a category
    from fully withdrawn to the cap. Symmetric on purpose — an abrupt withdrawal is equally
    unmeasured."""
    sc = S.empty_scaffold()
    up = S.apply_p_ops(sc, [{"task": "pick_two_obj_and_place", "p": 0.5}])
    assert up["p_task"]["pick_two_obj_and_place"] == S.P_MAX_DELTA, "raise was not step-limited"
    assert up["p_rate_limited"]["ops"]["pick_two_obj_and_place"]["requested"] == 0.5, \
        "step limit not recorded"

    down = S.apply_p_ops(up, [{"task": "pick_two_obj_and_place", "p": 0.0}])
    assert down["p_task"]["pick_two_obj_and_place"] == 0.0, \
        "0.2 -> 0 is within one step and must pass"
    far = S.apply_p_ops(S.apply_p_ops(up, [{"task": "pick_two_obj_and_place", "p": 0.4}]),
                        [{"task": "pick_two_obj_and_place", "p": 0.0}])
    assert abs(far["p_task"]["pick_two_obj_and_place"] - 0.2) < 1e-9, "cut was not step-limited"

    ok = S.apply_p_ops(sc, [{"task": "pick_two_obj_and_place", "p": 0.15}])
    assert ok["p_task"]["pick_two_obj_and_place"] == 0.15, "in-range request must pass unchanged"
    assert "p_rate_limited" not in ok, "nothing was limited, so nothing should be recorded"

    # a scaffold from before the cap (p=1.0 everywhere) is pulled down on load
    legacy = S.empty_scaffold()
    legacy["p_task"] = {t: 1.0 for t in legacy["p_task"]}
    legacy["default_p"] = 1.0
    fixed = S.clamp_p(legacy)
    assert all(v == S.P_MAX for v in fixed["p_task"].values())
    assert fixed["default_p"] == S.P_MAX
    assert fixed["p_clamped"]["cap"] == S.P_MAX

    # p=0 (full withdrawal) is still reachable
    assert S.apply_p_ops(sc, [{"task": "pick_and_place", "p": 0.0}])["p_task"]["pick_and_place"] == 0.0


def test_teacher_prompt_states_the_cap():
    from agent_system.skill_opt.autoscaffold import observation as O
    sys_prompt = O.render_system_prompt(S.ALF_DOMAIN)
    assert f"[0, {S.P_MAX}]" in sys_prompt
    assert "HARD CAP" in sys_prompt


def test_cold_start_injects_nothing_until_p_is_raised():
    """p is opt-in. A cold-start scaffold trains fully bare no matter what text is attached, so
    the first cycle measures the policy itself and any later injection is a deliberate, recorded
    act rather than a default nobody chose."""
    sc = S.empty_scaffold()
    assert sc["default_p"] == 0.0 and all(v == 0.0 for v in sc["p_task"].values())

    # writing text alone changes nothing about what training sees
    withtext = S.apply_text_ops(sc, [{"target": "pick_two_obj_and_place", "text": "leave the first one"}])
    assert all(v == 0.0 for v in withtext["p_task"].values()), "text alone must not enable injection"

    # the Teacher has to raise p in the same action for the text to be used, and one cycle can
    # only move it by P_MAX_DELTA, so reaching a high rate from cold start takes several cycles
    live = S.apply_p_ops(withtext, [{"task": "pick_two_obj_and_place", "p": 0.4}])
    assert live["p_task"]["pick_two_obj_and_place"] == S.P_MAX_DELTA
    assert live["p_task"]["look_at_obj_in_light"] == 0.0

    # and both limits still bind on a large ask
    assert S.apply_p_ops(withtext, [{"task": "pick_two_obj_and_place", "p": 0.9}]
                         )["p_task"]["pick_two_obj_and_place"] == S.P_MAX_DELTA


def test_teacher_prompt_says_p_is_opt_in():
    from agent_system.skill_opt.autoscaffold import observation as O
    t = O.render_system_prompt(S.ALF_DOMAIN)
    assert "p starts at 0" in t and "inert while its category's p is 0" in t


def test_prompt_states_the_optimization_mechanism_and_the_loss_change():
    """The Teacher cannot reason about what a good scaffold is without knowing two things the
    harness does but never used to say: that a group whose rollouts all fail contributes no
    gradient, and that the loss is recomputed on the BARE prompt. The second is the unusual one --
    it means the text is an exploration device, not a context the student is trained to keep."""
    import os
    from agent_system.skill_opt.autoscaffold import observation as O
    prev = os.environ.get("ARM_BARE_LOSS")
    try:
        os.environ["ARM_BARE_LOSS"] = "True"
        t = O.render_system_prompt(S.ALF_DOMAIN)
        for claim in ("all_fail_groups",
                      "advantage is computed WITHIN the group",
                      "THE LOSS IS COMPUTED ON THE BARE PROMPT",
                      "never trained to condition on your text",
                      "exploration device",
                      "Evaluation is always bare"):
            assert claim in t, f"missing from the Teacher prompt: {claim!r}"
    finally:
        os.environ.pop("ARM_BARE_LOSS", None)
        if prev is not None:
            os.environ["ARM_BARE_LOSS"] = prev
    # Mechanism, not advice: the DESCRIPTIVE sections must not tell the Teacher what to do.
    # _WHEN_TO_INTERVENE is exempt because it is a declared directive (see the test below);
    # the guard is scoped rather than deleted so the rest of the prompt stays descriptive and a
    # future prior cannot leak in unannounced.
    descriptive = t.split("WHEN TO INTERVENE")[0]
    for banned in ("you should", "we recommend", "the best strategy"):
        assert banned.lower() not in descriptive.lower(), \
            f"prescriptive language leaked into a descriptive section: {banned!r}"


def test_when_to_intervene_is_a_declared_directive():
    """Timing guidance is a deliberate PRIOR, added 2026-07-29 at the user's instruction, and the
    paper has to say so. Cycle 1 of alf_scratch150_pcap is the evidence behind it: a broad
    scaffold written at the first opportunity measured 0.078 against a bare 0.128 and was
    rejected. The Teacher can only act on that if it knows both the cost of injecting and what
    signal indicates the policy has stopped improving on its own."""
    from agent_system.skill_opt.autoscaffold import observation as O
    t = O.render_system_prompt(S.ALF_DOMAIN)
    assert "WHEN TO INTERVENE" in t, "the directive section is missing"
    assert "IS a directive" in t, "the section must announce that it prescribes, not describes"
    for claim in ("PREFER TO DECLINE",          # the default while the curve climbs
                  "FLATTENS or REGRESSES",      # the condition for stepping in
                  "eval_trajectory",            # the signal it must read to tell them apart
                  "Declining is a first-class action",
                  "Exploration.",               # what an injected group costs
                  "Transfer risk."):            # why off-prompt behaviour may not survive
        assert claim in t, f"missing from the intervention directive: {claim!r}"
    # the two enforced p limits must be stated, or the Teacher will plan moves it cannot make
    assert f"at most {S.P_MAX_DELTA} per cycle" in t
    assert "discarded" in t and "fails its A/B" in t


def test_loss_section_follows_the_trainer_flag():
    """The Teacher's whole model of what text can buy hangs off what the update is conditioned
    on, so the prompt must describe the loss the trainer is ACTUALLY running. Both come from
    ARM_BARE_LOSS (run_arm.py passes it to algorithm.bare_prompt_loss.enable) precisely so the
    two cannot drift apart. Telling the Teacher the wrong loss is worse than telling it nothing:
    with bare-prompt loss it should treat text as a disposable exploration device, and without it
    withdrawal is the only thing that turns injected behaviour into standalone ability."""
    import os
    from agent_system.skill_opt.autoscaffold import observation as O
    prev = os.environ.get("ARM_BARE_LOSS")
    try:
        os.environ["ARM_BARE_LOSS"] = "True"
        on = O.render_system_prompt(S.ALF_DOMAIN)
        os.environ["ARM_BARE_LOSS"] = "False"
        off = O.render_system_prompt(S.ALF_DOMAIN)
    finally:
        os.environ.pop("ARM_BARE_LOSS", None)
        if prev is not None:
            os.environ["ARM_BARE_LOSS"] = prev

    assert "THE LOSS IS COMPUTED ON THE BARE PROMPT" in on
    assert "exploration device" in on
    assert "THE LOSS IS COMPUTED ON THE BARE PROMPT" not in off, \
        "standard-loss run must not claim the loss is re-conditioned on the bare prompt"
    assert "PROMPT THAT WAS ACTUALLY USED" in off
    assert "WITHDRAWAL IS THEREFORE THE TRANSFER MECHANISM" in off, \
        "without bare-prompt loss, withdrawal is what makes behaviour transfer; say so"
    assert "IS trained to condition on your text" in off

    # the mode-independent mechanism must survive in both
    for claim in ("all_fail_groups", "advantage is computed WITHIN the group",
                  "Evaluation is always bare", "WHEN TO INTERVENE"):
        assert claim in on and claim in off, f"{claim!r} must not depend on the loss mode"
    for t in (on, off):
        assert "{loss_section}" not in t and "{p_max" not in t, "unsubstituted placeholder"


def test_invalid_scaffold_rollback_clears_every_applied_flag():
    """When validate_scaffold rejects the assembled result, NOTHING was applied — so the record
    must not claim otherwise. It used to clear accepted_text only, leaving p_applied True and the
    verdict 'accepted' on a cycle where the scaffold never changed. decision_history is the
    Teacher's memory as well as the audit trail, so a false 'that edit landed' would teach it
    from an edit that does not exist.

    validate_scaffold is stubbed rather than provoked: every apply_* helper clamps and validates
    its own input, so the normal path cannot produce a structurally invalid scaffold. That is the
    right design — and it is exactly why this rollback branch needs a test of its own, since
    nothing else reaches it."""
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    action = {"diagnosis": "d",
              "text_ops": [{"target": "pick_and_place", "text": "some rule"}],
              "p_ops": [{"task": "pick_and_place", "p": 0.2}]}
    measure = {"bare": {"pick_and_place": (0.2, 30)},
               "current": {"pick_and_place": (0.2, 30)},
               "candidate": {"pick_and_place": (0.9, 30)}}          # A/B would ACCEPT the text
    fns = _mock_fns([0.50], action, measure, calls)

    real = S.validate_scaffold
    S.validate_scaffold = lambda sc, domain=None: (False, "forced invalid")
    try:
        st = L.run_cycle(L.new_state(), fns, {"val_n": 1})
    finally:
        S.validate_scaffold = real

    ok(st["scaffold"]["skills"]["pick_and_place"] == "", "rolled back: text not applied")
    ok(st["scaffold"]["p_task"]["pick_and_place"] == 0.0, "rolled back: p not applied")
    d = st["decision_history"][-1]
    ok(d["accepted_text"] is False and d["p_applied"] is False and d["prefix_applied"] is False,
       "no applied-flag survives a rollback")
    ok(d["verdict"] == "rejected", "a cycle that changed nothing is not 'accepted'")


def test_absolute_target_step_stops_the_loop():
    """n_cycles counts cycles in one PROCESS, so on its own it cannot express a total. A restart
    that resumes at step 100 and is handed n_cycles=15 would run to step 250. ARM_TARGET_STEP is
    the absolute finish line, checked before every cycle, so the total is the same no matter how
    many times the watchdog had to relaunch."""
    calls = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    fns = _mock_fns([0.5] * 30, {"diagnosis": "d", "text_ops": [], "p_ops": []}, {}, calls)
    st = L.new_state(step0=100)
    cfg = {"val_n": 1, "steps_per_cycle": 10, "stop_fn": lambda s: s.get("step", 0) >= 130}
    st = L.run(st, fns, cfg, 15)                      # generous per-process budget
    ok(st["step"] == 130, f"stopped at the absolute target, not 100+15*10 (got {st['step']})")
    ok(calls["train"] == 3, f"ran exactly the 3 cycles needed (got {calls['train']})")

    # already past the target -> no cycle at all
    calls2 = {"train": 0, "signals": 0, "measure": 0, "teacher": 0, "restore": 0, "persist": None}
    fns2 = _mock_fns([0.5] * 5, {"diagnosis": "d", "text_ops": [], "p_ops": []}, {}, calls2)
    st2 = L.run(L.new_state(step0=200), fns2, cfg, 15)
    ok(calls2["train"] == 0 and st2["step"] == 200, "at/over target -> trains nothing")


def test_per_task_counts_reach_the_triage_view():
    """A per-task rate without its denominator is unreadable, and on ALFWorld that is not a
    hypothetical: the sampler draws an uneven task mix, so a rare category lands a handful of
    episodes and reads 0.000 or 1.000 while meaning nothing. The counts now travel with the rates
    from the eval log all the way into the cheap triage observation, because the decision they
    inform — is the total flat because everything is at ceiling, or because one category is stuck
    — cannot be made from the aggregate alone."""
    from agent_system.skill_opt.autoscaffold import observation as O
    traj = O.eval_trajectory([], 60, 0.5763, [0.593, 0.593, 0.543])
    obs = O.assemble_triage_observation(
        S.empty_scaffold(), traj, [], 60,
        per_task={"look_at_obj_in_light": 0.333, "pick_and_place": 0.947},
        per_task_n={"look_at_obj_in_light": 3, "pick_and_place": 57})
    pt = obs["valid_seen_per_task"]
    ok(pt["look_at_obj_in_light"] == {"success": 0.333, "n_episodes": 3},
       "rate and count travel together")
    ok(pt["pick_and_place"]["n_episodes"] == 57, "the readable category keeps its larger n")
    ok("per_task_gap" not in obs and "all_fail_groups" not in obs,
       "triage stays cheap: no train-side measurement leaked in")

    # absent counts must not crash or fabricate a denominator
    obs2 = O.assemble_triage_observation(S.empty_scaffold(), traj, [], 60,
                                         per_task={"pick_and_place": 0.9})
    ok(obs2["valid_seen_per_task"]["pick_and_place"]["n_episodes"] is None,
       "unknown n is None, not zero or invented")

    t = O.render_triage_prompt()
    ok("valid_seen_per_task" in t and "n_episodes" in t, "the prompt names the field")
    ok("one or two are stuck" in t, "the prompt says why the breakdown matters")
