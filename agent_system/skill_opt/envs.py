"""Build ALFWorld train/eval managers for the inference-time harness.

Reuses AlfWorldEnvironmentManager (prompt building + skill injection + task-type
detection) but with group_n=1 (each env a distinct game). The Actor is generated
via a vllm server, so this process does NO GPU work.
"""
import os
import yaml
from omegaconf import OmegaConf

import agent_system.environments.env_manager as em
from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
from agent_system.environments.env_manager import AlfWorldEnvironmentManager

ALF_CFG = os.path.join(os.path.dirname(em.__file__),
                       "env_package/alfworld/configs/config_tw.yaml")

# config task_type id -> the task-type string detect_task_type() returns (gamefile substring)
TYPE_ID_TO_NAME = {1: "pick_and_place", 2: "look_at_obj_in_light",
                   3: "pick_clean_then_place_in_recep", 4: "pick_heat_then_place_in_recep",
                   5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place"}
_CFG_DIR = "/mnt/data1/zha00175/gigpo_helper_skillopt/_alf_cfgs"


def _type_config(task_id):
    """Write a config_tw.yaml variant restricted to a single task_type id; return its path."""
    os.makedirs(_CFG_DIR, exist_ok=True)
    with open(ALF_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg["env"]["task_types"] = [int(task_id)]
    out = os.path.join(_CFG_DIR, f"config_tw_t{task_id}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def _cfg(seed, history_length):
    return OmegaConf.create({
        "env": {
            "env_name": "alfworld/AlfredTWEnv",
            "seed": int(seed),
            "history_length": int(history_length),
            "rollout": {"n": 1},
        },
    })


def build_manager(n_games, seed, is_train, eval_dataset="eval_out_of_distribution",
                  history_length=2, task_id=None):
    """eval_dataset only matters when is_train=False:
    'eval_in_distribution' (valid_seen) or 'eval_out_of_distribution' (valid_unseen).
    task_id (1..6): restrict this manager to a single task type (for balanced sampling)."""
    res = {"num_cpus": 0.1}
    cfg_path = _type_config(task_id) if task_id is not None else ALF_CFG
    envs = build_alfworld_envs(cfg_path, seed, n_games, 1, res, is_train,
                               {"eval_dataset": eval_dataset})
    mgr = AlfWorldEnvironmentManager(envs, alfworld_projection, _cfg(seed, history_length))
    return mgr


def build_balanced_managers(n_per_type, seed, is_train,
                            eval_dataset="eval_out_of_distribution", history_length=2,
                            only_types=None):
    """One manager per task type (n_per_type games each) -> balanced sampling.
    Returns a list of (task_name, manager). Total games per rollout = 6 * n_per_type.
    only_types: optional list of task-type names; if set, build ONLY those (targeted
    sampling on weak tasks, more games each)."""
    managers = []
    for tid in sorted(TYPE_ID_TO_NAME):
        if only_types is not None and TYPE_ID_TO_NAME[tid] not in only_types:
            continue
        mgr = build_manager(n_per_type, seed + tid, is_train, eval_dataset,
                            history_length, task_id=tid)
        managers.append((TYPE_ID_TO_NAME[tid], mgr))
    return managers


def attach_skill(manager, skill_store):
    """Point the manager at a shared SkillStore and force injection (skill always on)."""
    manager.skill_store = skill_store
    manager.skill_force = True
    return manager
