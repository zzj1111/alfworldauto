"""Unit tests for the math auto-scaffold data layer. No GPU, no network.
Run: python -m agent_system.skill_opt.mathscaffold.tests.test_mathscaffold"""
import sys
sys.path.insert(0, "/mnt/data1/zha00175/verl-agent")

from agent_system.skill_opt.mathscaffold import verify as V
from agent_system.skill_opt.mathscaffold import prepare_math as P
from agent_system.skill_opt.mathscaffold import questa as Q
from agent_system.skill_opt.mathscaffold import prefix as PX

FAIL = []
def ok(c, m):
    print(("PASS" if c else "FAIL"), m)
    if not c:
        FAIL.append(m)


def test_extract_boxed():
    ok(V.extract_boxed(r"so the answer is \boxed{42}.") == "42", "plain boxed")
    ok(V.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}",
       "NESTED braces kept whole (a lazy regex would truncate to \\frac{1)")
    ok(V.extract_boxed(r"\boxed{1} then later \boxed{7}") == "7",
       "LAST boxed wins (the final answer, not an intermediate one)")
    ok(V.extract_boxed(r"\boxed {  3 }") == "3", "space after \\boxed, trimmed")
    ok(V.extract_boxed("no answer here") is None, "no boxed -> None")
    ok(V.extract_boxed(r"\boxed{oops") is None, "unbalanced braces -> None, not a crash")
    ok(V.extract_boxed("") is None and V.extract_boxed(None) is None, "empty/None -> None")


def test_answers_equal():
    ok(V.answers_equal("42", "42"), "identical")
    ok(V.answers_equal(r"\left(3\right)", "(3)"), r"\left/\right stripped")
    ok(V.answers_equal(r"\dfrac{1}{2}", r"\frac{1}{2}"), "dfrac == frac")
    ok(V.answers_equal("3.0", "3"), "3.0 == 3")
    ok(V.answers_equal(r"\text{even}", "even"), r"\text{} unwrapped")
    ok(not V.answers_equal("42", "43"), "different numbers differ")
    ok(not V.answers_equal(None, "42") and not V.answers_equal("42", None), "None never matches")
    ok(V.score(r"... therefore \boxed{7}", "7") == 1.0, "score 1 on match")
    ok(V.score("I give up", "7") == 0.0, "score 0 when nothing is boxed")
    ok(V.score(r"\boxed{8}", "7") == 0.0, "score 0 on wrong answer")


def test_build_rows():
    raw = [
        {"problem": "p1", "solution": r"work \boxed{5}", "level": "Level 3", "type": "Algebra"},
        {"problem": "p2", "solution": r"work \boxed{6}", "level": "Level 1", "type": "Algebra"},
        {"problem": "p3", "solution": "no boxed at all", "level": "Level 5", "type": "Geometry"},
        {"problem": "p4", "solution": r"\boxed{\frac{1}{2}}", "level": "Level 4", "type": "Number Theory"},
    ]
    rows, dropped = P.build_rows(raw)
    ok([r["problem"] for r in rows] == ["p1", "p4"], "keeps only L3-5 with a parsable answer")
    ok(dropped == 1, "counts the L5 row whose reference solution has no \\boxed")
    ok(rows[0]["answer"] == "5" and rows[1]["answer"] == r"\frac{1}{2}", "answer parsed from solution")
    ok(rows[0]["solution"] == r"work \boxed{5}",
       "REFERENCE SOLUTION retained (the privileged signal math has and ALFWorld does not)")
    ok(P.uid_of("p1") == P.uid_of(" p1 ") and P.uid_of("p1") != P.uid_of("p4"),
       "uid is stable under whitespace and distinct per problem")

    v = P.to_verl(rows[0])
    ok(v["reward_model"]["ground_truth"] == "5", "verl view carries the gold answer")
    ok(v["extra_info"]["uid"] == rows[0]["uid"] and v["extra_info"]["solution"],
       "verl view carries uid + solution so per-instance scaffold can be keyed later")
    ok(v["prompt"][0]["role"] == "system" and "boxed" in v["prompt"][0]["content"],
       "prompt asks for a boxed final answer (verifier depends on it)")


def test_questa_replication():
    """QuestA's add_prefix.py has three behaviours that look like bugs and must be preserved,
    because the released datasets and the paper's numbers were produced with them."""
    sol = "AAAA BBBB CCCC DDDD"                       # 19 chars
    ok(Q.split_prefix(sol, 0.5)[0] == sol[:9],
       "cut is by RAW CHARACTER (int(19*0.5)=9), NOT snapped to a word boundary")
    ok(Q.split_prefix(sol, 1.0)[0] == sol, "scale 1.0 -> whole text")

    ok(Q.solution_block("<think>reasoning</think>THE SOLUTION") == "THE SOLUTION",
       "solution block = text AFTER </think>, not the CoT")
    ok(Q.solution_block('"<think>x</think>Y"') == "Y",
       "one layer of wrapping double quotes is stripped first")
    ok(Q.solution_block("<think>a</think>mid<think>b</think>Z") == "Z", "LAST </think> wins")

    ok(Q.build_prompt("P", "0123456789") == "P\n\n## Hint.0123456789",
       "marker is exactly '## Hint.' concatenated with NO separator")
    ok(Q.build_prompt("P", "short") == "P\n\n",
       "a prefix under 10 chars degrades the row to NO hint (not a short hint)")

    recs = [
        # solution must be long enough that a 50% prefix clears the 10-char floor, otherwise
        # build_prompt correctly degrades the row to no hint and the assertion below is vacuous
        {"problem": "p1", "answer": "42",
         "generation": "<think>t</think>Step one, expand. Step two, simplify to get 42 at the end."},
        {"problem": "p2", "answer": "99",
         "generation": "<think>t</think>Step one, expand. Step two, simplify to get 7 at the end."},
    ]
    rows, stats = Q.build_rows(recs, 50)
    ok([r["problem"] for r in rows] == ["p1"],
       "row is DROPPED when the gold answer does not literally occur in the solution block")
    ok(stats["dropped_answer_not_in_solution"] == 1, "drop is counted")
    ok(rows[0]["query_id"] == "0", "query_id is the input line index, as upstream")
    ok(rows[0]["solution"] and rows[0]["uid"],
       "solution + uid carried through so alpha can be chosen at train time")

    v = Q.to_verl(rows[0])
    ok("## Hint." in v["prompt"][0]["content"], "verl view: hinted prompt by default")
    ok(v["reward_model"]["ground_truth"] == "42", "verl view: gold answer in reward_model")
    b = Q.to_verl(rows[0], bare=True)
    ok("## Hint." not in b["prompt"][0]["content"] and b["extra_info"]["ratio"] == 0,
       "bare view has NO hint and records ratio 0 (the standalone/eval condition)")
    ok(b["extra_info"]["solution"] == rows[0]["solution"],
       "bare view still carries the solution — the alpha layer needs it")


def test_prefix_levels():
    sol = "x" * 100
    ok(PX.solution_prefix(sol, 0.0) == "", "alpha 0 discloses nothing (our added level)")
    ok(PX.solution_prefix(sol, 1.0) == sol, "alpha 1 discloses everything")
    ok(PX.snap_level(0.37) == 0.4 and PX.snap_level(0.9) == 0.8,
       "arbitrary Teacher alpha snaps to the nearest supported level")
    ok(PX.snap_level(-5) == 0.0 and PX.snap_level(7) == 1.0, "out-of-range alpha is clamped")
    ok(PX.snap_level("abc") is None, "non-numeric alpha rejected, not silently coerced")
    ok(PX.render_guided_problem("Q", "S", 0.0) == "Q", "alpha 0 -> the bare question")


def main():
    for fn in [test_extract_boxed, test_answers_equal, test_build_rows,
               test_questa_replication, test_prefix_levels]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
