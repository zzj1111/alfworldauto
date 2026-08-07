"""The scaffold-injecting ALFWorld env manager: the only training-side component.

Selected by the gated hook in agent_system/environments/env_manager.py when
AUTOSCAFFOLD_ALFWORLD is set; the validation manager is always the vanilla class, so
the eval path cannot inject regardless of anything here.

What it does, per rollout:
- reset: reload scaffold.json (hot-reload — the orchestrator rewrites it between
  cycles), draw ONE Bernoulli(p_task[category]) coin per GROUP (consecutive blocks of
  env.rollout.n envs; uid minting in the rollout loop uses the same block layout).
- build_text_obs: capture the vanilla prompt as text_bare, splice the scaffold block
  into coin-heads envs' prompts. The anchor obs is untouched (GiGPO groups on it),
  and memory stores raw observations upstream, so history never accumulates scaffold.
- step: buffer (executed action, observation, validity) per env — the projected
  action is invisible outside this method.
- success_evaluator: join buffers with uid/won/gamefile from the rollout loop and
  append one JSONL row per episode to AUTOSCAFFOLD_ROLLOUT_LOG.

Env vars: AUTOSCAFFOLD_SCAFFOLD (path to scaffold.json), AUTOSCAFFOLD_ROLLOUT_LOG
(recorder target; absent = recorder off), AUTOSCAFFOLD_SEED (coin RNG).
"""
from __future__ import annotations

import json
import os

import numpy as np

from agent_system.environments.env_manager import AlfWorldEnvironmentManager

from . import scaffold as S

# Kept well under data.max_prompt_length headroom: the ALFWorld template + history is
# ~1400 tokens of the 2048 budget with truncation='error', so the block must stay
# small. 8 items x 500 chars would blow it; the render is capped and the cap noted.
MAX_BLOCK_CHARS = int(os.environ.get("AUTOSCAFFOLD_MAX_BLOCK_CHARS", "1200"))

_HEADER = ("Hints for this task type (guidance, not gospel — you must still follow the "
           "required output format):\n")


SPLICE_ANCHOR = "Now it's your turn to take an action."


def _splice(prompt, block):
    """Insert the scaffold block ahead of the final instruction block (present in both
    upstream ALFWorld templates), so guidance is read before the output contract, not
    after it. Falls back to prepending when the anchor is missing — a template change
    must not silently drop injection."""
    text = _HEADER + block + "\n\n"
    i = prompt.find(SPLICE_ANCHOR)
    if i == -1:
        return text + prompt
    return prompt[:i] + text + prompt[i:]


class ScaffoldAlfWorldEnvironmentManager(AlfWorldEnvironmentManager):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self._scaffold_path = os.environ.get("AUTOSCAFFOLD_SCAFFOLD", "")
        self._record_path = os.environ.get("AUTOSCAFFOLD_ROLLOUT_LOG", "")
        self._rng = np.random.default_rng(int(os.environ.get("AUTOSCAFFOLD_SEED", "0")) + 12345)
        self._scaffold = S.empty_scaffold()
        self._inject = []          # per-env bool, decided at reset, constant all episode
        self._categories = []      # per-env category slug or None
        self._step_buf = []        # per-env [{a, o, v}]
        # A/B harness controls: an explicit scaffold overrides the file, and force-all
        # makes every category-bearing group heads (text forced onto held-out games on
        # purpose — the one sanctioned exception to train-only injection).
        self._scaffold_override = None
        self._force_all = False
        group_n = int(self.config.env.rollout.n)
        assert group_n > 0, "autoscaffold requires env grouping (env.rollout.n > 0)"
        try:
            fg_enabled = bool(self.config.algorithm.filter_groups.enable)
        except Exception:
            fg_enabled = False
        if fg_enabled:
            # The recorder flushes once per rollout via success_evaluator; dynamic
            # sampling reruns the loop pre-filter and would double-record groups that
            # never reach the gradient.
            raise NotImplementedError("autoscaffold recorder assumes filter_groups.enable=False")

    # ---------------- injection ----------------

    def _reload_scaffold(self):
        if self._scaffold_override is not None:
            self._scaffold = self._scaffold_override
            return
        sc = S.load(self._scaffold_path) if self._scaffold_path else None
        self._scaffold = sc if sc is not None else S.empty_scaffold()

    def _draw_coins(self):
        n = len(self.gamefile)
        group_n = int(self.config.env.rollout.n)
        assert n % group_n == 0, f"{n} envs do not tile into groups of {group_n}"
        self._categories = [S.category_of_gamefile(g) for g in self.gamefile]
        inject = [False] * n
        for g0 in range(0, n, group_n):
            cat = self._categories[g0]
            # the rollout loop mints one uid per consecutive block; the whole block is
            # one game, so one coin
            assert all(c == cat for c in self._categories[g0:g0 + group_n]), \
                "a group spans two categories; the block layout assumption broke"
            if cat is None:
                continue
            p = float((self._scaffold.get("p_task") or {}).get(cat, 0.0) or 0.0)
            block = S.render(self._scaffold, cat)
            if self._force_all:
                fire = bool(block)
            else:
                fire = bool(block) and p > 0 and bool(self._rng.random() < p)
            for i in range(g0, g0 + group_n):
                inject[i] = fire
        self._inject = inject

    def reset(self, kwargs):
        obs, infos = super().reset(kwargs)
        self._reload_scaffold()
        self._draw_coins()
        self._step_buf = [[] for _ in range(len(self.gamefile))]
        # super().reset built prompts before the coins existed; rebuild through our
        # build_text_obs so injection and text_bare apply from step one
        full = self.build_text_obs(self.pre_text_obs, self.envs.get_admissible_commands,
                                   init=True)
        return {**obs, "text": full, "text_bare": list(self._last_bare)}, infos

    def build_text_obs(self, text_obs, admissible_actions, init=False):
        vanilla = super().build_text_obs(text_obs, admissible_actions, init)
        self._last_bare = list(vanilla)
        if not getattr(self, "_inject", None) or not any(self._inject):
            return vanilla
        out = []
        for i, prompt in enumerate(vanilla):
            if i < len(self._inject) and self._inject[i]:
                block = S.render(self._scaffold, self._categories[i])[:MAX_BLOCK_CHARS]
                out.append(_splice(prompt, block) if block else prompt)
            else:
                out.append(prompt)
        return out

    # ---------------- recorder ----------------

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        raw_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        if self._record_path:
            for i, a in enumerate(actions):
                self._step_buf[i].append({
                    "a": str(a),
                    "o": str(raw_obs[i])[:400],
                    "v": bool(np.asarray(valids[i]).item() if hasattr(valids[i], "item")
                              else valids[i]),
                })
        self.pre_text_obs = raw_obs
        full = self.build_text_obs(raw_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            from agent_system.environments.env_manager import set_gamefile
            infos = set_gamefile(infos, self.gamefile)
        for i, info in enumerate(infos):
            info["is_action_valid"] = np.asarray(valids[i])
        next_obs = {"text": full, "image": image_obs, "anchor": raw_obs,
                    "text_bare": list(self._last_bare)}
        return next_obs, np.asarray(rewards), np.asarray(dones), infos

    def success_evaluator(self, *args, **kwargs):
        success = super().success_evaluator(*args, **kwargs)
        if self._record_path:
            try:
                self._flush(kwargs["total_batch_list"], kwargs["total_infos"])
            except Exception as e:
                # A recorder failure must never take down the training step it annotates.
                print(f"[autoscaffold] recorder flush failed: {type(e).__name__}: {e}")
        return success

    def _flush(self, total_batch_list, total_infos):
        rows = []
        for i in range(len(total_batch_list)):
            steps_meta = total_batch_list[i]
            if not steps_meta:
                continue
            last_active = None
            for j in reversed(range(len(steps_meta))):
                if steps_meta[j]["active_masks"]:
                    last_active = j
                    break
            if last_active is None:
                continue
            info = total_infos[i][last_active]
            rows.append({
                "uid": str(steps_meta[0]["uid"]),
                "task_type": self._categories[i] if i < len(self._categories) else None,
                "gamefile": str(info.get("extra.gamefile") or
                                (self.gamefile[i] if i < len(self.gamefile) else "")),
                "injected": bool(self._inject[i]) if i < len(self._inject) else False,
                "success": float(info.get("won") or 0.0),
                "steps": self._step_buf[i][:last_active + 1] if i < len(self._step_buf) else [],
            })
        os.makedirs(os.path.dirname(os.path.abspath(self._record_path)), exist_ok=True)
        with open(self._record_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
