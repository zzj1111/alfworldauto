"""Drive the real ScaffoldAlfWorldEnvironmentManager against a fake ALFWorld env.

These are the experiment's two premises, tested by execution:
- the coin is per group, hits its rate, and eval (vanilla manager, chosen by the
  upstream hook) never sees scaffold text;
- text_bare differs from the training prompt by exactly the spliced block.
Plus the recorder's row schema, which everything in signals.py consumes.
"""
import collections
import json
import math
import os
import types

import numpy as np
import pytest

pytest.importorskip("agent_system.environments.env_manager",
                    reason="needs the verl-agent tree on sys.path")

from autoscaffold import scaffold as S
from autoscaffold import signals as G
from autoscaffold.scaffold_env_manager import ScaffoldAlfWorldEnvironmentManager

GROUP_N = 4
CAT = "pick_two_obj_and_place"


class FakeAlfEnvs:
    """The surface AlfWorldEnvironmentManager actually touches."""

    def __init__(self, n_groups, cat=CAT):
        self.n = n_groups * GROUP_N
        self.games = [f"/data/{cat}-Mug-None-Desk-{g}/trial/game.tw-pw"
                      for g in range(n_groups) for _ in range(GROUP_N)]
        self.t = 0

    @property
    def get_admissible_commands(self):
        return [["go to desk 1", "take mug 1 from desk 1", "help"] for _ in range(self.n)]

    def reset(self):
        obs = [f"You are in a room. Your task is to: put two mugs on the desk. (env {i})"
               for i in range(self.n)]
        infos = [{"extra.gamefile": self.games[i]} for i in range(self.n)]
        return obs, [None] * self.n, infos

    def step(self, actions):
        self.t += 1
        obs = [f"Nothing happens. (t={self.t})" for _ in range(self.n)]
        rewards = [0.0] * self.n
        dones = [self.t >= 2] * self.n
        infos = [{"extra.gamefile": self.games[i], "won": False} for i in range(self.n)]
        return obs, [None] * self.n, rewards, dones, infos


def _config():
    env = types.SimpleNamespace(rollout=types.SimpleNamespace(n=GROUP_N), history_length=2)
    return types.SimpleNamespace(env=env)


def _projection(text_actions, admissible):
    """Mimics alfworld_projection: extract the <action> body; invalid when missing."""
    acts, valids = [], []
    for a in text_actions:
        lo = a.lower()
        i, j = lo.find("<action>"), lo.find("</action>")
        if i != -1 and j != -1:
            acts.append(lo[i + len("<action>"):j].strip())
            valids.append(1)
        else:
            acts.append("nothing")
            valids.append(0)
    return acts, valids


def _manager(n_groups=8, scaffold=None, record=None, monkeypatch=None, seed="0"):
    monkeypatch.setenv("AUTOSCAFFOLD_SEED", seed)
    if record:
        monkeypatch.setenv("AUTOSCAFFOLD_ROLLOUT_LOG", record)
    else:
        monkeypatch.delenv("AUTOSCAFFOLD_ROLLOUT_LOG", raising=False)
    monkeypatch.delenv("AUTOSCAFFOLD_SCAFFOLD", raising=False)
    m = ScaffoldAlfWorldEnvironmentManager(FakeAlfEnvs(n_groups), _projection, _config())
    if scaffold is not None:
        m._scaffold_override = scaffold
    return m


def _scaffold(p=0.5, text="MARK-look before you take"):
    sc = S.empty_scaffold()
    sc, _ = S.apply_item_ops(sc, [{"op": "add", "scope": CAT, "kind": "skill", "text": text}])
    sc["p_task"][CAT] = p
    return sc


def test_coin_is_per_group_and_hits_its_rate(monkeypatch):
    hits = tot = 0
    for trial in range(60):
        m = _manager(8, _scaffold(p=0.3), monkeypatch=monkeypatch, seed=str(trial))
        m.reset({})
        for g0 in range(0, len(m._inject), GROUP_N):
            block = m._inject[g0:g0 + GROUP_N]
            assert len(set(block)) == 1, "a group split across arms breaks the pairing"
            tot += 1
            hits += bool(block[0])
    rate = hits / tot
    z = (rate - 0.3) / math.sqrt(0.3 * 0.7 / tot)
    assert abs(z) < 4, f"realized {rate:.3f} over {tot} groups against p=0.3 (z={z:+.1f})"


def test_bare_prompt_differs_by_exactly_the_block(monkeypatch):
    m = _manager(4, _scaffold(p=1.0), monkeypatch=monkeypatch)
    obs, _ = m.reset({})
    assert any(m._inject), "p=1 with text must fire"
    for i, fired in enumerate(m._inject):
        spliced, bare = obs["text"][i], obs["text_bare"][i]
        assert "MARK-look" not in bare, "the bare copy leaked scaffold text"
        if fired:
            assert "MARK-look" in spliced
            # the strict check: removing the injected block reproduces the bare prompt
            from autoscaffold.scaffold_env_manager import SPLICE_ANCHOR
            start = spliced.find("Hints for this task type")
            end = spliced.find(SPLICE_ANCHOR)
            assert 0 <= start < end, "block must sit ahead of the final instruction"
            assert spliced[:start] + spliced[end:] == bare
        else:
            assert spliced == bare


def test_empty_scaffold_or_p_zero_is_a_strict_noop(monkeypatch):
    for sc in (S.empty_scaffold(), _scaffold(p=0.0)):
        m = _manager(4, sc, monkeypatch=monkeypatch)
        obs, _ = m.reset({})
        assert not any(m._inject)
        assert obs["text"] == obs["text_bare"], "no coin, no difference"


def test_anchor_is_never_touched(monkeypatch):
    m = _manager(4, _scaffold(p=1.0), monkeypatch=monkeypatch)
    obs, _ = m.reset({})
    for a in obs["anchor"]:
        assert "MARK-look" not in a and "Hints for" not in a, \
            "GiGPO groups on the anchor; splicing it changes the grouping"
    nxt, _, _, _ = m.step(["<action>go to desk 1</action>"] * (4 * GROUP_N))
    for a in nxt["anchor"]:
        assert "MARK-look" not in a


def test_recorder_rows_feed_signals_end_to_end(monkeypatch, tmp_path):
    log = str(tmp_path / "rollouts.jsonl")
    m = _manager(4, _scaffold(p=0.5), record=log, monkeypatch=monkeypatch, seed="7")
    m.reset({})
    n = 4 * GROUP_N
    for _ in range(2):
        m.step([f"<action>take mug 1 from desk 1</action>"] * n)
    # what the rollout loop would hand success_evaluator
    total_batch_list = [[{"uid": f"u{i // GROUP_N}", "active_masks": True},
                         {"uid": f"u{i // GROUP_N}", "active_masks": True}]
                        for i in range(n)]
    total_infos = [[{"won": False, "extra.gamefile": m.gamefile[i]},
                    {"won": i % GROUP_N == 0, "extra.gamefile": m.gamefile[i]}]
                   for i in range(n)]
    m.success_evaluator(total_infos=total_infos, total_batch_list=total_batch_list,
                        episode_rewards=np.zeros(n), episode_lengths=np.full(n, 2))
    rows = G.read_rows(log)
    assert len(rows) == n
    r = rows[0]
    assert set(r) == {"uid", "task_type", "gamefile", "injected", "success", "steps"}
    assert r["task_type"] == CAT and len(r["steps"]) == 2
    assert r["steps"][0]["a"] == "take mug 1 from desk 1", "the EXECUTED action, not raw text"
    # and the whole packet builds from these rows without error
    sig = G.signals_from_rows(rows, GROUP_N)
    assert sig["n_groups"] == 4
    assert "mixed_injection_groups" not in sig, "recorded coins must be group-constant"
    zg = sig["zero_gradient_groups"][CAT]
    assert zg["total"] == 4 and zg["all_fail"] == 0 and zg["all_succeed"] == 0


def test_history_never_accumulates_scaffold(monkeypatch):
    m = _manager(2, _scaffold(p=1.0), monkeypatch=monkeypatch)
    m.reset({})
    n = 2 * GROUP_N
    m.step(["<action>go to desk 1</action>"] * n)
    nxt, _, _, _ = m.step(["<action>open drawer 1</action>"] * n)
    for i, fired in enumerate(m._inject):
        prompt = nxt["text"][i]
        assert prompt.count("MARK-look") == (1 if fired else 0), \
            "the block must appear once per prompt, not once per remembered step"
