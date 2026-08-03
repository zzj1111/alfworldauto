"""Per-instance alpha injection for the Teacher-scheduled math arm.

Wire it in with:
    data.custom_cls.path=agent_system/skill_opt/mathscaffold/dataset.py
    data.custom_cls.name=AlphaRLHFDataset

The dataset on disk is always the UNHINTED union pool. This class rewrites the user turn at
`__getitem__` time from `extra_info.solution` using the alpha the Teacher chose for that
problem, so the disclosure schedule lives in one small JSON file instead of being frozen into
the parquet. That is the whole point: QuestA has to build two datasets and run two trainings
to change its ratio, and it applies one global ratio to every problem.

Injection is done by overriding `_build_messages` only. Tokenization, truncation, padding and
position ids all stay with verl's RLHFDataset — rebuilding those tensors by hand (as the
OC-GRPO fork does) is where silent prompt corruption comes from.

Hot reload: the scaffold file's mtime is checked on every access, so a cycle that rewrites
alphas takes effect on the next epoch without restarting training.

The rendered text is byte-identical to QuestA's `add_prefix.py` at the same ratio — same
character cut, same `'## Hint.'` marker, same <10-character degradation — so an alpha of 0.5
here and QuestA's p=50 produce the same prompt.
"""
from __future__ import annotations

import json
import os
import threading

from verl.utils.dataset.rl_dataset import RLHFDataset

from .questa import build_prompt, split_prefix

ENV_PATH = "QUESTA_SCAFFOLD_PATH"


class _AlphaStore:
    """{uid: alpha} reloaded whenever the file's mtime moves. Missing file / unreadable JSON
    means "no alpha for anyone", i.e. every prompt stays bare — a broken scaffold degrades to
    the no-hint arm rather than to a crash or, worse, to a stale schedule."""

    def __init__(self, path):
        self.path = path
        self._mtime = None
        self._alpha = {}
        self._default = 0.0
        self._lock = threading.Lock()

    def _maybe_reload(self):
        if not self.path:
            return
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return
        if m == self._mtime:
            return
        with self._lock:
            try:
                with open(self.path) as f:
                    d = json.load(f)
                self._alpha = {str(k): float(v) for k, v in (d.get("alpha") or {}).items()}
                self._default = float(d.get("default_alpha", 0.0) or 0.0)
                self._mtime = m
            except Exception:
                return                       # keep the last good schedule; do not half-apply

    def get(self, uid):
        self._maybe_reload()
        return self._alpha.get(str(uid), self._default)

    def stats(self):
        self._maybe_reload()
        return {"n_alpha": len(self._alpha), "default_alpha": self._default,
                "path": self.path, "loaded": self._mtime is not None}


class AlphaRLHFDataset(RLHFDataset):
    """RLHFDataset whose user turn is re-rendered at the Teacher's per-problem alpha."""

    def __init__(self, data_files, tokenizer, config, processor=None):
        path = config.get("scaffold_path", None) or os.environ.get(ENV_PATH, "")
        self.alpha_store = _AlphaStore(path)
        super().__init__(data_files, tokenizer, config, processor)
        print(f"[AlphaRLHFDataset] scaffold={path or '(none)'} -> {self.alpha_store.stats()}",
              flush=True)

    def _build_messages(self, example):
        extra = example.get("extra_info") or {}
        uid = extra.get("uid")
        problem = extra.get("problem")
        solution = extra.get("solution")
        # Without uid/problem/solution there is nothing to re-render; fall through to whatever
        # the parquet already holds rather than guessing.
        if uid is None or not problem or not solution:
            return super()._build_messages(example)

        alpha = self.alpha_store.get(uid)
        if alpha and alpha > 0.0:
            prefix, _ = split_prefix(solution, float(alpha))
            content = build_prompt(problem, prefix)
        else:
            content = problem + "\n\n"       # alpha 0 == the bare prompt, as on disk

        messages = example.pop(self.prompt_key)
        out = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "user":
                m["content"] = content
            out.append(m)
        return out


def render_for_alpha(problem, solution, alpha):
    """Pure helper mirroring what the dataset does, for tests and for the A/B measurement."""
    if not alpha or float(alpha) <= 0.0:
        return problem + "\n\n"
    prefix, _ = split_prefix(solution, float(alpha))
    return build_prompt(problem, prefix)
