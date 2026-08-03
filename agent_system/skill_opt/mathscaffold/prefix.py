"""Ground-truth solution prefixes as the privileged signal (the option-3 mechanism).

The Teacher does NOT write mathematical content here. It only decides HOW MUCH of the known
reference solution to disclose for each problem — a fraction alpha in [0,1] — and the prefix
text is then derived from the ground truth by a pure, deterministic string operation.

Why this shape: OC-GRPO (arXiv 2607.19313) measured both. Solution prefixes gave +13.8% over
vanilla GRPO on 7B; LLM-generated hints gave -2.8% (their Table 5, i.e. WORSE than no
guidance at all). So the content should come from the ground truth, and what is left to
decide is the disclosure schedule.

That schedule is where we differ from the paper:
  - OC-GRPO-Fixed:    alpha*(x) = smallest level with >=1 success under pi_ref, chosen ONCE
                      before training by a mechanical rule, then frozen.
  - OC-GRPO-Adaptive: same rule, recomputed every step.
  - ours:             the Teacher chooses alpha per problem from measured signals, keeps a
                      memory of what happened, and can LOWER alpha again (withdraw) as the
                      policy improves. No rule, no priors.

`solution_prefix` is a byte-for-byte reimplementation of their
`HintGenerator.get_solution_prefix`, verified against the original in the unit tests, so
prefixes are identical to the ones behind their published numbers.
"""
from __future__ import annotations

import re

# The cascade the paper instantiates. 0.0 = fully withdrawn (nothing disclosed); we add it
# because withdrawal is an action our Teacher has and theirs does not.
LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

_ASY = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL)


def strip_asymptote(text):
    """Drop [asy]...[/asy] diagram blocks: they are figure source, not reasoning, and a
    character-fraction cut through one yields unparsable garbage."""
    return _ASY.sub("", text or "").strip()


def solution_prefix(solution_text, fraction):
    """First `fraction` of the reference solution by character count, snapped BACK to a word
    boundary. fraction<=0 -> "" (nothing disclosed); fraction>=1 -> the whole solution."""
    text = strip_asymptote(solution_text)
    if fraction >= 1.0:
        return text
    if fraction <= 0.0 or not text:
        return ""
    target_len = max(1, int(fraction * len(text)))
    if target_len >= len(text):
        return text
    cut = target_len
    while cut > 0 and text[cut] not in " \n\t":
        cut -= 1
    if cut == 0:                      # no boundary before target -> cut at target
        cut = target_len
    return text[:cut].rstrip()


# --- prompt templates (kept identical to the paper's, so the guided distribution matches) ---
PARTIAL_INSTRUCTION = ("Here is a partial reference solution to the problem above. "
                       "Complete the rest of the solution step by step and output the "
                       "final answer within \\boxed{}.")
FULL_INSTRUCTION = ("Here is the full reference solution. Verify it, show your work "
                    "step by step, and output the final answer within \\boxed{}.")


def render_guided_problem(question, solution, alpha):
    """The user-turn content for a guided rollout, or the bare question when alpha<=0.

    Note the two framings are deliberately different in the paper: a partial prefix asks the
    model to CONTINUE (so the trajectory contains its own reasoning, which is what the policy
    gradient is computed over), while the full solution asks it to VERIFY/reproduce (which
    only anchors a guaranteed-correct trajectory).
    """
    pre = solution_prefix(solution, alpha)
    if not pre:
        return question
    if alpha >= 1.0:
        return f"Problem: {question}\n\nReference solution: {pre}\n\n{FULL_INSTRUCTION}"
    return f"Problem: {question}\n\nPartial reference solution: {pre}\n\n{PARTIAL_INSTRUCTION}"


def snap_level(alpha, levels=LEVELS):
    """Round a Teacher-proposed alpha to the nearest level in the cascade. The Teacher may
    return any float; the mechanism only supports the discrete cascade."""
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return None
    a = min(1.0, max(0.0, a))
    return min(levels, key=lambda L: abs(L - a))
