"""The experiment name owns the wandb run id.

env.sh exports WANDB_RUN_ID, so a fallback that honored an existing one made the id
sticky per shell: source it for experiment A, launch experiment B from that same
shell, and B appended to A's run under a different experiment name — silently, since
every other path (checkpoints, state dir, ray tmp) is namespaced correctly.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(exp, env_extra):
    env = dict(os.environ, ARM_ROOT=ROOT, ARM_EXP=exp, ARM_WANDB="1", **env_extra)
    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/autoscaffold/env.sh" >/dev/null 2>&1; '
                       'printf "%s\\n%s\\n" "$WANDB_RUN_ID" "$WANDB_PROJECT"'],
        env=env, capture_output=True, text=True, timeout=120)
    lines = out.stdout.strip().splitlines()
    return (lines + ["", ""])[:2]


def test_a_stale_run_id_from_a_previous_experiment_is_ignored():
    run_id, _ = _resolve("experiment_b", {"WANDB_RUN_ID": "experiment_a"})
    assert run_id == "experiment_b", \
        f"experiment_b appended to {run_id!r} — the sticky-id bug is back"


def test_a_stale_project_from_a_previous_experiment_is_ignored():
    _, project = _resolve("e", {"WANDB_PROJECT": "someone_elses_project",
                                "ARM_WANDB_PROJECT": "chosen_project"})
    assert project == "chosen_project"


def test_the_default_project_is_where_the_baselines_live():
    """New runs must land beside the AutoScaffold runs they are compared against."""
    # No site file: this asserts the built-in default, which is what a fresh clone
    # gets. A site file legitimately overrides it (see the precedence test above).
    env = {k: v for k, v in os.environ.items() if k != "ARM_WANDB_PROJECT"}
    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/autoscaffold/env.sh" >/dev/null 2>&1; '
                       'printf "%s" "$ARM_WANDB_PROJECT"'],
        env=dict(env, ARM_ROOT=ROOT, ARM_EXP="e", ARM_WANDB="1",
                 ARM_ENV_FILE="/nonexistent/.autoscaffold.env"),
        capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "verl_agent_alfworld_inspect"


def test_the_explicit_override_still_wins():
    run_id, _ = _resolve("experiment_b", {"ARM_WANDB_RUN_ID": "resume_this_one",
                                          "WANDB_RUN_ID": "experiment_a"})
    assert run_id == "resume_this_one"


def test_the_python_publisher_derives_the_same_id_as_the_shell():
    from autoscaffold import run_arm as R
    os.environ.pop("ARM_WANDB_RUN_ID", None)
    stale = dict(os.environ, WANDB_RUN_ID="experiment_a")
    old, os.environ = os.environ, stale
    try:
        pub = R._WandbPublisher({"exp": "experiment_b"})
    finally:
        os.environ = old
    shell_id, _ = _resolve("experiment_b", {"WANDB_RUN_ID": "experiment_a"})
    assert pub.run_id == shell_id == "experiment_b", \
        "the orchestrator and the trainer subprocesses must land in the same run"


def test_the_slug_matches_between_shell_and_python():
    exp = "exp.with/odd chars"
    shell_id, _ = _resolve(exp, {})
    assert shell_id == re.sub(r"[^A-Za-z0-9_-]", "_", exp)
