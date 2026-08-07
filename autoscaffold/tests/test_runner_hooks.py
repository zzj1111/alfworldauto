"""GPU-free coverage of runner.py plumbing plus structural checks that the four
upstream hooks stay shaped the way DESIGN.md promises: gated, inert by default, and
with the eval path physically unable to inject."""
import json
import os
import socket

from autoscaffold import config as C
from autoscaffold import runner as R
from autoscaffold import scaffold as S

REPO = C.repo_root()


# ---------------- runner plumbing ----------------

def _shards(root, world=2, complete=True, with_data=True):
    actor = os.path.join(root, "actor")
    os.makedirs(actor, exist_ok=True)
    for r in range(world):
        for kind in ("model", "optim", "extra_state"):
            if not complete and r == world - 1 and kind == "extra_state":
                continue
            open(os.path.join(actor, f"{kind}_world_size_{world}_rank_{r}.pt"), "w").close()
    if with_data:
        open(os.path.join(root, "data.pt"), "w").close()


def test_ckpt_usable_requires_every_shard_and_data_pt(tmp_path):
    good = str(tmp_path / "global_step_10")
    _shards(good)
    assert R.ckpt_is_usable(good)
    torn = str(tmp_path / "global_step_20")
    _shards(torn, complete=False)
    assert not R.ckpt_is_usable(torn), "a mid-save crash leaves a torn dir; it must read as absent"
    no_data = str(tmp_path / "global_step_30")
    _shards(no_data, with_data=False)
    assert not R.ckpt_is_usable(no_data), "data.pt is written after the actor save"
    assert not R.ckpt_is_usable(str(tmp_path / "never_existed"))


def test_step_of():
    assert R.step_of("/x/ckpts/exp/global_step_120") == 120
    assert R.step_of("/x/global_step_120/") == 120
    assert R.step_of("garbage") == 0


def test_parse_val_reads_the_last_block_by_key_presence(tmp_path):
    log = tmp_path / "t.log"
    log.write_text(
        "step:0 - val/success_rate:0.100 - val/pick_and_place_success_rate:0.200\n"
        "step:5 - actor/entropy:1.2 - training/loss:0.3\n"
        # val metrics merged into a training line — key presence, not line shape
        "step:10 - actor/lr:1e-6 - val/success_rate:0.350 - "
        "val/look_at_obj_in_light_success_rate:0.500 - training/other:1\n")
    overall, per_task, found = R.parse_val(str(log))
    assert found and overall == 0.350
    assert per_task == {"look_at_obj_in_light": 0.500}
    assert R.parse_val(str(tmp_path / "missing.log")) == (None, {}, False)


def test_free_port_skips_a_busy_one():
    base = 8910
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", base))
        s.listen(1)
        assert R.free_port(base) == base + 1
    assert R.free_port(base) == base


def test_task_id_covers_every_category_exactly():
    assert set(R.TASK_ID) == set(S.CATEGORIES)
    assert len(set(R.TASK_ID.values())) == len(S.CATEGORIES)


def test_cat_config_filters_to_one_task_type(tmp_path):
    out = R._cat_config("look_at_obj_in_light", str(tmp_path))
    import yaml
    conf = yaml.safe_load(open(out))
    assert conf["env"]["task_types"] == [2]


# ---------------- the upstream hooks, structurally ----------------

def _text(rel):
    with open(os.path.join(REPO, rel)) as f:
        return f.read()


def test_env_manager_hook_is_gated_and_val_stays_vanilla():
    src = _text("agent_system/environments/env_manager.py")
    i = src.index('os.environ.get("AUTOSCAFFOLD_ALFWORLD")')
    block = src[i - 400: i + 600]
    assert "manager_cls = AlfWorldEnvironmentManager" in block, \
        "the default must be the upstream class, chosen before the env-var check"
    assert "val_envs = AlfWorldEnvironmentManager(_val_envs" in block, \
        "the VAL manager must be the vanilla class unconditionally — this line is " \
        "the physical lock that keeps scaffold text out of evaluation"


def test_rollout_loop_carries_text_bare_with_anchor_fallback():
    src = _text("agent_system/multi_turn_rollout/rollout_loop.py")
    assert "obs.get('text_bare', None)" in src
    assert "obs_text_bares[item] if obs_text_bares is not None else _obs_anchor" in src, \
        "vanilla managers return no text_bare; the field must fall back, not KeyError"


def test_trainer_swap_is_gated_and_ordered_before_old_log_prob():
    src = _text("verl/trainer/ppo/ray_trainer.py")
    gate_i = src.index('_bpl = self.config.algorithm.get("bare_prompt_loss", None)')
    swap_i = src.index("swap_to_bare_prompt(batch, self.tokenizer, self.config)")
    old_lp_i = src.index('with _timer("old_log_prob"')
    mask_i = src.index('batch.batch["response_mask"] = compute_response_mask(batch)')
    assert gate_i < swap_i < mask_i < old_lp_i, \
        "mode=both requires the swap before old_log_prob (and before response_mask)"
    helper = src[src.index("def swap_to_bare_prompt"):src.index("def compute_response_mask")]
    assert 'batch.batch["prompts"].shape[-1]' in helper, \
        "width must be pinned to the existing prompts tensor or prompt_length consumers mis-index"
    assert "multi_modal_inputs" in helper, "must refuse multimodal batches"


def test_yaml_key_defaults_off():
    src = _text("verl/trainer/config/ppo_trainer.yaml")
    i = src.index("bare_prompt_loss:")
    block = src[i:i + 200]
    assert "enable: False" in block, "the hook must be inert by default"


def test_train_script_mode_none_sets_no_scaffold_flags():
    src = _text("autoscaffold/train_alfworld.sh")
    i = src.index('if [[ "$MODE" == "scaffold" ]]')
    assert "AUTOSCAFFOLD_ALFWORLD" in src[i:], \
        "the enabling export must sit inside the scaffold branch"
    before = src[:i]
    assert "export AUTOSCAFFOLD_ALFWORLD" not in before
    assert "bare_prompt_loss.enable=${" not in before, \
        "mode=none must be byte-identical upstream behavior (the baseline)"


def test_no_machine_specific_path_in_the_package():
    """A default naming a directory that exists on one machine is the portability bug
    class that broke the previous implementation 47 files at a time."""
    import re
    bad = re.compile(r"/(?:home|mnt|scratch|checkpoints|data\d)/[A-Za-z0-9_.-]+/")
    offenders = []
    pkg = os.path.join(REPO, "autoscaffold")
    for root, _, files in os.walk(pkg):
        if "tests" in root or "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith((".py", ".sh")):
                continue
            for ln, line in enumerate(open(os.path.join(root, name)), 1):
                if line.strip().startswith("#"):
                    continue
                if bad.search(line):
                    offenders.append(f"{name}:{ln}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)
