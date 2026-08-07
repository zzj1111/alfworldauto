"""The prompt and the machinery must not disagree.

Every defect this guards against has the same shape: the Teacher is told something
untrue of the system it operates, nothing crashes, and it reasons correctly about a
fiction. The previous implementation was corrected three separate times for this
(a smoothing that did not exist, another experiment's assignment mechanism, retired
fields still described). These tests are structural where possible so new fields are
covered the day they are added.
"""
import json

from autoscaffold import prompts as P
from autoscaffold import scaffold as S
from autoscaffold import signals as G
from autoscaffold import teacher as T
from autoscaffold.tests.test_signals import _group, CAT, N


def _obs(sig=None, hist=None, scaffold=None):
    return P.assemble_observation(scaffold or S.empty_scaffold(),
                                  sig or {"per_task_gap": {}, "zero_gradient_groups": {},
                                          "contrastive_traces": {}, "failure_patterns": {}},
                                  hist or [], 10, 1, valid_seen={"avg": 0.3})


def test_prompt_promises_no_mechanism_the_code_lacks():
    p = P.render_system_prompt()
    # "nothing is smoothed" is a true denial and allowed; what must never appear are
    # the affirmative claims of mechanisms this code does not have
    for claim in ("geometrically", "down-weight", "EFFECTIVE count", "effective count",
                  "hash of the task"):
        assert claim not in p, f"prompt promises {claim!r}"
    assert "THE LAST CYCLE ONLY" in p
    assert "RAW episode counts" in p
    assert "nothing is smoothed" in p


def test_prompt_constants_match_the_code():
    p = P.render_system_prompt()
    for value in (str(S.BUDGET_CHANGES), str(S.MAX_ITEMS_PER_SCOPE),
                  str(S.MAX_ITEM_CHARS), str(S.P_MAX), str(S.P_MAX_DELTA)):
        assert value in p, f"constant {value} missing from the prompt"
    for cat in S.CATEGORIES:
        assert cat in p


def test_every_packet_field_is_described_and_vice_versa():
    rows = _group("g1", [0, 0, 0, 0]) + _group("g2", [1, 0, 0, 0])
    sig = G.signals_from_rows(rows, N, scaffold=S.empty_scaffold())
    obs = _obs(sig=sig)
    p = P.render_system_prompt()
    structural = {"step", "cycle", "scaffold", "n_episodes", "n_groups"}
    for field in (set(obs) | set(sig)) - structural:
        assert field in p, f"packet carries {field!r}; the prompt never describes it"
    for named in ("per_task_gap", "zero_gradient_groups", "contrastive_traces",
                  "failure_patterns", "valid_seen", "decision_history"):
        assert named in obs or named in obs.get("signals", {}), \
            f"prompt describes {named!r}; the packet does not carry it"


def test_action_spec_in_prompt_is_what_normalize_accepts():
    p = P.render_system_prompt()
    for token in ('"item_ops"', '"p_ops"', '"diagnosis"', '"op"', '"scope"',
                  '"kind"', '"id"', '"text"', '"task"'):
        assert token in p
    for kind in S.KINDS:
        assert kind in p
    action, note = T.normalize(
        {"diagnosis": "d",
         "item_ops": [{"op": "add", "scope": "general", "kind": "skill", "text": "t"}],
         "p_ops": []}, S.empty_scaffold())
    assert note == "ok"


def test_losing_wording_reaches_the_next_prompt_with_its_verdict():
    hist = [{"cycle": 1, "step": 10,
             "summary": {"text_proposed": [{"op": "add", "scope": "general",
                                            "kind": "skill",
                                            "text": "MARKER: always look before taking"}],
                         "p_edits": {}, "diagnosis": "d"},
             "ab": {"accept": False, "reason": "candidate 0.1 vs current 0.2 -> reject"},
             "verdict": "rejected", "sr_before": 0.2, "sr_after": 0.25}]
    user = P.render_user_prompt(_obs(hist=hist))
    assert "MARKER: always look before taking" in user
    assert "rejected" in user


def test_trimming_keeps_every_category_represented_and_patterns_untouched(monkeypatch):
    rows = []
    for i, cat in enumerate(S.CATEGORIES):
        for g in range(3):
            rows += _group(f"{cat}-f{g}", [0, 0, 0, 0], cat=cat,
                           steps_of=lambda i: 40)
        rows += _group(f"{cat}-s", [1, 1, 1, 1], cat=cat, steps_of=lambda i: 40)
    sig = G.signals_from_rows(rows, N)
    obs = _obs(sig=sig)
    full = len(json.dumps(obs, default=str))
    monkeypatch.setattr(P, "PROMPT_BUDGET", full // 3)
    user = P.render_user_prompt(obs)
    assert len(user) <= full // 3
    packet = json.loads(user)
    kept = packet["signals"]["contrastive_traces"]
    represented = [c for c, v in kept.items()
                   if v.get("zero_gradient_failures") or v.get("successes_same_category")]
    assert len(represented) == len(S.CATEGORIES), \
        "the trim must shrink the largest category, never delete whole categories"
    assert packet["signals"]["failure_patterns"] == json.loads(
        json.dumps(sig["failure_patterns"])), "failure_patterns are never trimmed"
    assert "contrastive_traces_dropped" in packet["signals"], "a trim must say so"


def test_untrimmed_packet_is_passed_through_verbatim():
    obs = _obs()
    assert json.loads(P.render_user_prompt(obs)) == json.loads(json.dumps(obs, default=str))
