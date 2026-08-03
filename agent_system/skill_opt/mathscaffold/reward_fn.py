"""verl-compatible reward function for the math auto-scaffold arm.

Wire it in with:
    custom_reward_function.path=agent_system/skill_opt/mathscaffold/reward_fn.py
    custom_reward_function.name=compute_score

The reward is exactly the one OC-GRPO uses: binary, from the extracted `\\boxed{}` answer
compared by symbolic equivalence. Nothing about the reasoning text is graded — which is what
makes a scaffold that merely gets copied into the output worthless at eval time, and is why
the standalone (no-scaffold) number is the only one worth reporting.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from agent_system.skill_opt.mathscaffold.verify import score  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """1.0 iff the last \\boxed{} in `solution_str` matches `ground_truth`, else 0.0."""
    return score(solution_str, ground_truth)
