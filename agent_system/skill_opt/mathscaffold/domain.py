"""The math training domain, described to the Teacher as STRUCTURE FACTS only.

This is the whole point of the Domain-descriptor pattern: the ALFWorld prompt and this one
run through the identical `render_system_prompt`. Nothing here says "use hints" or "write one
skill per subject". We only state what is true about the domain, and the Teacher works out
for itself what scaffolding the structure affords:

    ALFWorld : has_reference_solutions=False, instance_scope=False
               -> the Teacher read those two facts and ruled hints out on its own
    math     : has_reference_solutions=True,  instance_scope=True
               -> the same two facts now say per-instance, solution-derived guidance is possible

The subject labels come from MATH's own `type` field, so they are a property of the dataset,
not a taxonomy we invented.
"""
from __future__ import annotations

from ..autoscaffold.scaffold import Domain

# MATH's seven subject labels (the dataset's own `type` field; counts from L3-5 train).
SUBJECTS = [
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Precalculus",
    "Geometry",
    "Number Theory",
    "Counting & Probability",
]

# Factual descriptions of what each subject label denotes — never how to solve it.
SUBJECT_INFO = {
    "Algebra": "Equations, inequalities, functions, sequences and algebraic manipulation.",
    "Intermediate Algebra": "Polynomials, complex numbers, conic sections, functional equations, inequalities.",
    "Prealgebra": "Arithmetic, fractions, percentages, ratios, basic geometry and elementary word problems.",
    "Precalculus": "Trigonometry, vectors, matrices, parametric and polar forms, complex plane.",
    "Geometry": "Plane and solid geometry: lengths, areas, volumes, angles, similarity, circles.",
    "Number Theory": "Divisibility, modular arithmetic, primes, base representations, Diophantine equations.",
    "Counting & Probability": "Combinatorics, permutations and combinations, expected value, discrete probability.",
}

MATH_DOMAIN = Domain(
    name="math",
    episode_desc=("One instance is a single competition-style mathematics problem. The model "
                  "produces a step-by-step derivation and must put its final answer inside "
                  "\\boxed{}. Reward is 1 if the extracted answer is symbolically equivalent "
                  "to the ground-truth answer, else 0. There is exactly one attempt per "
                  "rollout and no interaction with an environment."),
    categories=SUBJECTS,
    category_info=SUBJECT_INFO,
    action_primitives=(),            # free-form text; no restricted action set
    has_reference_solutions=True,    # MATH ships a full worked solution for every problem
    instance_scope=True,             # text can be attached to an individual problem (keyed by uid)
    extra_facts=(
        "Every instance carries exactly one of the subject labels above, and a difficulty "
        "label in {Level 3, Level 4, Level 5}.",
        "Every instance has a written reference solution that derives the ground-truth answer.",
        "The training set is filtered to instances the base model failed on all 64 sampled "
        "attempts, so under the unguided prompt every rollout group starts with zero successes.",
        "Only the final boxed answer is scored; the reasoning text itself is not graded.",
        "For an individual instance you may disclose a leading FRACTION alpha of its reference "
        "solution during training. The disclosed text is taken verbatim from the reference "
        "solution by a deterministic cut; you choose alpha, not the wording. alpha=0 discloses "
        "nothing, alpha=1 discloses the whole solution. Available levels: 0, 0.2, 0.4, 0.6, "
        "0.8, 1.0. alpha is per instance and can be raised or lowered at any cycle.",
    ),
)
