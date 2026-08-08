"""One wandb publish per cycle, and it is the one that carries the decision.

The trainer shares the wandb run and owns its step counter; a resumed run starts at
last+1, so every publish consumes a step index the trainer then cannot use. Two saves
per cycle silently cost two of its ten per-step diagnostic points (measured on two
live runs: steps 11,12 / 21,22 / 31,32 ... missing). Disk sinks are unaffected and
still run on every save.
"""
import json

from autoscaffold import monitor as M


class FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, data, step=None):
        self.calls.append((dict(data), step))


def _state(cycle, step, decided):
    return {
        "cycle": cycle, "step": step, "sr_history": [0.5],
        "scaffold": {"items": {}, "p_task": {}},
        "decision_history": ([{"cycle": cycle, "verdict": "p_only"}] if decided else []),
        "last_eval": {"draws": [0.5, 0.5]},
    }


def _cfg(tmp_path):
    return {"state_dir": str(tmp_path), "exp": "e", "log": lambda *a: None}


def test_the_post_eval_save_does_not_publish(tmp_path):
    run, cfg = FakeRun(), _cfg(tmp_path)
    M.publish(_state(3, 30, decided=False), cfg, run)
    assert run.calls == [], "publishing before the Teacher decides reports no verdict"


def test_the_decision_save_publishes_once(tmp_path):
    run, cfg = FakeRun(), _cfg(tmp_path)
    M.publish(_state(3, 30, decided=False), cfg, run)
    M.publish(_state(3, 30, decided=True), cfg, run)
    M.publish(_state(3, 30, decided=True), cfg, run)   # a retry must not double-log
    assert len(run.calls) == 1
    assert run.calls[0][0]["progress/cycle"] == 3


def test_each_cycle_publishes_again(tmp_path):
    run, cfg = FakeRun(), _cfg(tmp_path)
    for c in (1, 2, 3):
        M.publish(_state(c, c * 10, decided=False), cfg, run)
        M.publish(_state(c, c * 10, decided=True), cfg, run)
    assert [c[0]["progress/cycle"] for c in run.calls] == [1, 2, 3]


def test_disk_sinks_still_run_on_every_save(tmp_path):
    run, cfg = FakeRun(), _cfg(tmp_path)
    M.publish(_state(3, 30, decided=False), cfg, run)
    M.publish(_state(3, 30, decided=True), cfg, run)
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl") if l.strip()]
    assert len(rows) == 2, "the complete record on disk must not thin out with wandb"
