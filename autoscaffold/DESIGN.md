# AutoScaffold — port contract

This package rebuilds the AutoScaffold experiment on a fresh verl-agent clone. It is a
rewrite against this design document, not a copy of the previous implementation
(`/mnt/data1/zha00175/verl-agent`, now frozen). Every locked decision below was settled
during that project's runs; the rationale lives in that repo's history and is not
re-litigated here.

## The experiment

A GPT Teacher writes short scaffold text that is injected into TRAINING prompts only.
The policy is always evaluated on the bare prompt. The objective is standalone
(no-scaffold) success on held-out games; the mechanism being tested is whether
training-time text converts zero-gradient groups into learnable ones.

Zero-gradient group: GRPO/GiGPO advantage is reward minus the group mean, so a group
whose `rollout_n` episodes all score the same (all fail OR all succeed) contributes no
gradient. At ALFWorld step 10 with the 1.5B base, 58% of groups were silent.

## Locked decisions (constants and rules)

Scaffold state
- Addressable items with ids; scopes = `general` + the six ALFWorld task categories.
- Item kinds: `skill`, `example`. (No `rubric`, no `hint`/alpha reveal, no instance
  scope — dropped; ALFWorld does not use them.)
- Caps: 3 edits (add+update) per cycle; 8 items per scope; dedup on normalized text.
- p_task per category: hard cap P_MAX = 0.5 (at least half of every category's groups
  always see the bare prompt); per-cycle change clamped to ±0.2. Clamping, not rejection.
- Persisted atomically (tmp+rename). scaffold.json is hot-reloaded by training.

Injection (training side)
- One Bernoulli(p_task[cat]) draw PER GROUP (all rollout_n episodes of one game share the
  outcome). A split group would put the scaffold on both sides of the advantage.
- Training split only. Evaluation and validation are bare, always, with two independent
  locks: the eval path runs with injection mode off, AND the manager refuses to inject
  when the env is not the training split. `skill_force` is the single exception, used by
  the A/B to force text onto held-out games on purpose.
- The bare prompt copy (`text_bare`) is captured BEFORE splicing, so the two texts differ
  by exactly the injected block.

bare_prompt_loss
- `algorithm.bare_prompt_loss.enable` + `mode=both`: swap the prompt to the bare text
  before old_log_prob, so the loss conditions on the prompt the policy is evaluated
  under. Config-gated, default off; upstream behavior identical when off.

Free signals (no measurement pass; read from the rollouts training already wrote)
- Recorder appends one JSONL row per training episode: uid (group id), task_type,
  gamefile, injected (the coin the manager actually flipped — never replayed from
  (seed,p)), success, steps [{a, o, v}] with the EXECUTED action (not the raw generation;
  raw text truncates from the front and loses the action tail).
- Window = byte offsets taken around ONE cycle's training. No cross-cycle accumulation,
  no smoothing; counts are raw. The prompt must say exactly this.
- per_task_gap: success split by injected vs bare, per category, with n_bare/n_injected.
  gap=null when a side is empty, with a reason field (no text / p=0 / no group fired).
- zero_gradient_groups: complete groups only (len == rollout_n); condition is NO reward
  variance; report total / all_fail / all_succeed (opposite remedies).
- contrastive_traces: per category, 3 failures from all-fail groups (the longest FAILED
  rollout of each; never a successful trajectory) + 3 shortest same-category successes.
  When a category has no all-fail group, top up from lowest-success-rate groups; if a
  group has no failures at all, skip it and note `no_failures_to_show`.
- failure_patterns: rule-computed over ALL failed rollouts (repeated command >=3 and
  >=half the trajectory; looped observation >=3; >=20% unparseable actions).

Teacher
- GPT-5.5 (`AUTOSCAFFOLD_TEACHER_MODEL`), JSON response, validated; any malformed output
  degrades to a no-op, never a crash.
- Unreachable (auth/quota/network) is marked distinctly from a decline, counted across
  consecutive cycles (persisted), and bannered in the log: a run finished this way is a
  plain-RL control, not evidence that text does not help.
- Memory = decision_history: every proposal's exact text, verdict, A/B numbers, and
  diagnosis; replayed into the next observation so losing wording is not re-proposed.
- No triage pre-check. Signals are free; propose() is the only decision point.

A/B gate (text changes only)
- Held-out split (same split as the headline eval), frozen checkpoint, three conditions:
  bare / current / candidate. `current` is skipped when the current scaffold injects
  nothing (it would duplicate bare).
- Fixed episode budget per condition: 180 (ARM_AB_EPISODES), split across touched
  categories — resolution must not shrink when the Teacher narrows.
- Conditions are paired: same games, env order re-seeded before each condition.
- Accept iff candidate mean > current mean. Strict; no margin (decided 2026-08-05).
- below_bare flag when an accepted candidate scores under the bare condition (recorded,
  not vetoed; it has happened twice).
- Log the distinct-game pool size and replay factor per category (valid_seen pools are
  28–43 games; 180 episodes over 1–2 categories replays 3–6x).
- p-only proposals skip the A/B. A p edit co-submitted with text that fails its A/B is
  discarded with the text (p-veto), and the journal records p_vetoed_with_text.
- No revert gate. A regression stays in the curve.

Loop
- One cycle: train K=10 steps -> standalone eval on the held-out split (VAL_N=3 draws)
  -> free signals -> Teacher proposes -> A/B on text -> apply -> persist state+journal.
- Resume: state.json holds every key new_state() creates (enforced by a structural
  test); ARM_TARGET_STEP is the absolute finish line; a checkpoint counts as usable only
  with every rank's model/optim/extra shards plus data.pt; training a step whose usable
  checkpoint exists is skipped (idempotent).
- Relaunch chain: restart the arm until the target step is reached; three failures
  under 60s each = broken environment, stop. Keepalive (plain-rl baseline / none) after.

Monitoring (every cycle, after eval and at cycle exit)
- Snapshot -> three sinks: wandb (same run id as the trainer: WANDB_RUN_ID = sanitized
  exp name, resume=allow), status.json (atomic), metrics.jsonl (append).
- Snapshot contents: progress, valid_seen + per-task + draw spread, train success per
  category, zero-gradient fractions, scaffold items/chars/p, teacher verdicts + A/B
  numbers, teacher_unreachable_cycles, container memory (cgroup, not /proc/meminfo).
- status.sh renders one screen with warnings for the known silent-failure modes:
  unreachable teacher; scaffold empty past cycle 4; text present but p=0 everywhere;
  accepted text below bare; stale heartbeat; memory > 85% of the cgroup limit.

Operational hardening (each item corresponds to a real incident)
- dataloader_num_workers=0 (the dataset is one batch of short prompts; 8 workers fork
  the trainer for nothing and were the OOM killer's first victim).
- Memory checks read the cgroup limit (v2 then v1); /proc/meminfo inside a container
  reports the host and passes exactly when it should fail.
- ALFWORLD_DATA must be node-local (18412 small files on NFS took the source machine to
  load 414); preflight checks the filesystem type.
- vLLM serve: scan up from ARM_VLLM_PORT for a free port; after the health wait, require
  our own subprocess to still be alive (a healthy endpoint alone may be another run's
  server). Graceful teardown scoped by GPU + uid.
- Every log file stamped per launch (ARM_RUN_ID); cumulative orch.log kept as history.
- One site file (.autoscaffold.env), parsed identically by shell and python (trailing
  comments stripped; caller env > site file > defaults), with an agreement test.
- No hardcoded machine paths anywhere (portability test scans for them).

## Integration footprint (hard limit)

At most 4 upstream files touched, each with one marked, env/config-gated block that is
inert by default:
1. env manager selection (choose the scaffold subclass when enabled) — ~5 lines
2. rollout loop: carry text_bare into the batch; recorder hook if the subclass cannot
   see episode success — few lines each, no-op when disabled
3. verl/trainer/ppo/ray_trainer.py: bare_prompt_loss block
4. verl/trainer/config/ppo_trainer.yaml: the bare_prompt_loss key

Everything else lives under `autoscaffold/` (this package), including the training
launch script (a copy adapted from the upstream example — the upstream example itself is
not edited).

## Explicitly dropped from the previous implementation

triage pre-check + intervene floor; hint/alpha partial reveal; instance-scope texts;
multi-domain abstraction (the CUDA/Triton experiment stays in StitchCUDA, unmodified);
the paid signals fallback (ARM_SIGNALS_PASS); the offline wandb reconstruction tool;
teacher priors flag. If any of these is wanted later it is a new feature, not a port.
