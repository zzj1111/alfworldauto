"""DynamicHintDataset — our dynamic scaffold for math, as a verl custom dataset.

QuestA prepends a FIXED first-p% of the reference solution to each problem (static
add_prefix.py, 50%→25% two-stage). We make that DYNAMIC: the hint fraction f is read live
from a hot-reloadable JSON (keyed by query_id), so a controller can withdraw / target it
during training. The reference solution is carried per-row in extra_info["solution"]
(built by math_prep.py), so the hint = solution[:f%] — real GT, mechanical truncation,
zero distillation.

Register via:  data.custom_cls.path=.../math_hint_dataset.py  data.custom_cls.name=DynamicHintDataset
Scaffold file: env MATH_SCAFFOLD_PATH (JSON: {"default_fraction": f, "instances": {qid: f}}).
"""
import json
import os
import threading
import time

from verl.utils.dataset.rl_dataset import RLHFDataset

_HINT_HEADER = "\n\n## Hint."


class DynamicHintDataset(RLHFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scaffold_path = os.environ.get("MATH_SCAFFOLD_PATH", "")
        self._default_f = float(os.environ.get("MATH_HINT_FRACTION", "0.5"))
        self._max_hint_chars = int(os.environ.get("MATH_HINT_MAX_CHARS", "6000"))
        self._lock = threading.Lock()
        self._inst = {}
        self._mtime = 0.0
        self._last_stat = 0.0
        self._reload(force=True)

    # ---- live scaffold (hot-reload by mtime, throttled) ----
    def _reload(self, force=False):
        if not self._scaffold_path:
            return
        now = time.time()
        if not force and now - self._last_stat < 1.0:
            return
        self._last_stat = now
        try:
            m = os.path.getmtime(self._scaffold_path)
            if force or m > self._mtime:
                d = json.load(open(self._scaffold_path))
                with self._lock:
                    self._default_f = float(d.get("default_fraction", self._default_f))
                    self._inst = {str(k): float(v) for k, v in (d.get("instances") or {}).items()}
                    self._mtime = m
        except Exception:
            pass

    def _fraction_for(self, qid):
        with self._lock:
            return self._inst.get(str(qid), self._default_f)

    # ---- inject the truncated-solution hint into the prompt ----
    def _build_messages(self, example: dict):
        self._reload()
        ei = example.get("extra_info") or {}
        f = self._fraction_for(ei.get("query_id", ""))
        sol = ei.get("solution") or ""
        if f > 0 and sol:
            hint = sol[: int(len(sol) * f)][: self._max_hint_chars]
            if len(hint) >= 10:
                msgs = list(example[self.prompt_key])
                for i in range(len(msgs) - 1, -1, -1):
                    if msgs[i].get("role") == "user":
                        m = dict(msgs[i])
                        m["content"] = m["content"] + _HINT_HEADER + hint
                        msgs[i] = m
                        break
                example[self.prompt_key] = msgs
        return super()._build_messages(example)
