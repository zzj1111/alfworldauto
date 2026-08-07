# AutoScaffold × verl-agent Integration Map
Repo: `/mnt/data1/zha00175/verl-agent-rebuild` (branch `autoscaffold-rebuild`, upstream base `20bd331` + local `78eafe5` = `autoscaffold/DESIGN.md` only). Every anchor below was verified by opening the file at the stated lines. Entry point covered: `verl/trainer/main_ppo.py` only (recipe/GraphGPO and recipe/hgpo carry their own `make_envs`/trainer copies and are NOT covered — do not launch through them).

---

## (1) Upstream files to touch — exactly 4, each one gated block, inert by default

### 1a. `agent_system/environments/env_manager.py` — manager selection (~4 lines)
Anchor: the alfworld branch of `make_envs`, lines 645–648 (verified verbatim):
```python
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
```
This is the single choke point for the main_ppo path (both train and val managers). `import os` already exists at line 21. Replace with:
```python
        projection_f = partial(alfworld_projection)
        manager_cls = AlfWorldEnvironmentManager
        if os.environ.get("AUTOSCAFFOLD_ALFWORLD"):
            from autoscaffold.scaffold_env_manager import ScaffoldAlfWorldEnvironmentManager as manager_cls
        envs = manager_cls(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
```
**Val stays vanilla on purpose** — this is the stronger of DESIGN.md's two injection locks (eval path physically cannot inject; the A/B's `skill_force` runs in the orchestrator's own vLLM harness, not through the trainer val path). Safe when disabled: env var unset → `manager_cls` is the upstream class, `val_envs` line is behaviorally identical to upstream.

Env-var propagation caveat: `make_envs` runs inside the `@ray.remote(num_cpus=1) TaskRunner` actor (`main_ppo.py:54, 70–71`). When the driver starts the local Ray cluster itself (`run_ppo`, `main_ppo.py:36–48`) shell exports are inherited; when attaching to a pre-existing cluster (`RAY_ADDRESS` set) they are not. Belt-and-suspenders: also pass `+ray_init.runtime_env.env_vars.AUTOSCAFFOLD_ALFWORLD=1` — verified that `config.ray_init.runtime_env` is merged over the defaults at `main_ppo.py:41–48`.

### 1b. `agent_system/multi_turn_rollout/rollout_loop.py` — `text_bare` carriage (2–3 lines); recorder hook NOT needed
The subclass alone cannot get a new field into the PPO batch: `preprocess_single_sample` reads only `obs['text'|'image'|'anchor']` (lines 67–69); extra obs keys are silently dropped. Overloading `anchor` is forbidden — it is GiGPO's step-grouping key (`ray_trainer.py:350`).

Add next to line 69:
```python
        obs_text_bares = obs.get('text_bare', None)
```
and inside the `row_dict.update({...})` at lines 175–183, beside `'anchor_obs'` (line 180):
```python
            'text_bare': obs_text_bares[item] if obs_text_bares is not None else _obs_anchor,
```
Safe when disabled: vanilla managers return no `text_bare` key → the field falls back to the anchor value; it is a metrics-free non-tensor passenger nobody reads unless `bare_prompt_loss` is on.

Survival path (verified): key is absent from the generation pop lists (`rollout_loop.py:337–348` pops only `input_ids/attention_mask/position_ids` + `raw_prompt_ids/multi_modal_data/raw_prompt/tools_kwargs`) and from the trainer pop lists (`ray_trainer.py:1054–1063`); flows `union(361) → to_list_of_dict(391) → total_batch_list → gather_rollout_data collate(280–282) → del batch; batch = gen_batch_output (ray_trainer.py:1108–1109) → adjust_batch(1118)` and is present at the swap point.

**Recorder hook: not required.** The rollout loop already calls `envs.success_evaluator(total_infos=..., total_batch_list=..., episode_rewards=..., episode_lengths=...)` at `rollout_loop.py:407–412`, and each step-dict in `total_batch_list` carries `uid`, `traj_uid`, `active_masks`, `rewards`, `is_action_valid` (attached at 358–359, 374–388); `total_infos` carries `won` and `extra.gamefile`. The subclass override of `success_evaluator` is the recorder flush point. Reserve line 413 (between the `success_evaluator` call and `return` at 414) as the fallback insertion point for an `on_rollout_end` hook only if post-filter group membership ever needs recording — we run `filter_groups.enable=False`, so it is not part of this port.

### 1c. `verl/trainer/ppo/ray_trainer.py` — `bare_prompt_loss` block (~3 lines + one module-level helper)
Insert between `batch = adjust_batch(self.config, batch)` (line 1118) and `batch.batch["response_mask"] = compute_response_mask(batch)` (line 1120):
```python
                    bpl = self.config.algorithm.get('bare_prompt_loss', None)
                    if bpl is not None and bpl.get('enable', False):
                        batch = swap_to_bare_prompt(batch, self.tokenizer, self.config)
```
This is strictly before the `old_log_prob` recompute (`with _timer("old_log_prob", ...)` at 1141–1143), so old_log_probs, entropy (1144–1148), ref_log_prob (1177–1184), and `update_actor` (1250) all condition on the same swapped tensors — DESIGN.md's `mode=both`. (A `numerator` mode, if ever revived, would insert before `if self.use_reference_policy:` at 1177.)

`swap_to_bare_prompt` lives as a module-level helper next to `compute_response_mask` (line 226), per sample:
1. read `batch.non_tensor_batch['text_bare']` (full bare templated prompt, pre-chat-template — NOT `anchor_obs`, which is the raw untemplated observation);
2. wrap as one user message, `tokenizer.apply_chat_template(add_generation_prompt=True, tokenize=False, **config.data.get('apply_chat_template_kwargs', {}))` — mirror of `rollout_loop.py:90–101`;
3. `verl_F.tokenize_and_postprocess_data(max_length=config.data.max_prompt_length, left_pad=True, truncation=config.data.truncation)` — same width, so `apply_invalid_action_penalty` stays correct (it derives `prompt_length` from `batch['prompts'].shape[-1]` and indexes `attention_mask[prompt_length:]`, verified at `ray_trainer.py:200–217`);
4. new prompt position ids via `compute_position_id_with_mask`; keep `responses` and the response slice `attention_mask[:, -response_length:]` verbatim; response position ids = `new_prompt_pos[..., -1:] + arange(1..response_length)` (the `vllm_rollout_spmd.py` stitch pattern);
5. overwrite `batch.batch['prompts', 'input_ids', 'attention_mask', 'position_ids']`; gate off when `'multi_modal_inputs' in batch.non_tensor_batch` (mrope not supported).

Safe when disabled: `algorithm.get('bare_prompt_loss')` returns the yaml default with `enable: False` → block never executes; zero behavior change.

### 1d. `verl/trainer/config/ppo_trainer.yaml` — the config key
The `algorithm:` block spans 234–257 (verified; `gigpo:` at 250–254, `filter_groups:` at 255–257). Append after line 257 (before `trainer:` at 259):
```yaml
  bare_prompt_loss:
    enable: False
    mode: both
```
Required so hydra struct mode accepts `algorithm.bare_prompt_loss.enable=True` as a CLI override (precedent: `algorithm.gigpo.*` reads at `ray_trainer.py:1232–1235`, `.get` pattern at 1219). Safe when disabled: `enable: False` default; nothing reads it otherwise.

Everything else (subclass, scaffold store, recorder, Teacher, A/B, launch script copy) lives under `autoscaffold/` — zero further upstream diff.

---

## (2) Manager-subclass surface

`ScaffoldAlfWorldEnvironmentManager(AlfWorldEnvironmentManager)` in `autoscaffold/`, overriding four methods:

| Method | Override does | Upstream anchor |
|---|---|---|
| `reset(kwargs)` | Reset recorder buffers; after super's reset flow, gamefiles/tasks are populated; draw one Bernoulli(p_task[cat]) coin **per group** (consecutive blocks of `config.env.rollout.n` envs — layout guaranteed by uid minting `i % n == 0` at `rollout_loop.py:317` and worker seeding `seed + i // group_n` at `envs.py:105`); return a **new** obs dict with added `'text_bare'` key | `env_manager.py:138–148` |
| `build_text_obs(text_obs, admissible_actions, init)` | `vanilla = super().build_text_obs(...)`; stash as the `text_bare` list for this pass; splice scaffold block per index i keyed on category derived from `self.gamefile[i]` (only for envs whose group coin was heads); return spliced list. Both `reset` (line 147) and `step` (line 156) funnel through this one method. `self.gamefile`/`self.tasks` are set before the reset-time call (lines 140/145 precede 147) | `env_manager.py:180–212` |
| `step(text_actions)` | Reimplement the 18-line body (kept in autoscaffold/, not upstream) so the projected `actions, valids` (line 151) can be buffered per env: `(executed_action, raw_obs, valid, done)` — the executed action is otherwise invisible from outside; attach `'text_bare'` to the returned obs dict | `env_manager.py:150–168` |
| `success_evaluator(**kwargs)` | `success = super().success_evaluator(**kwargs)`; join buffered per-step records with per-step `uid` from `total_batch_list[i][j]['uid']`, `won` from the last active step's info, `extra.gamefile`, and the manager's own `injected` coin; append one JSONL row per episode; return `success` unchanged | `base.py:114–133`, called at `rollout_loop.py:407–412` |

Readable state: `self.gamefile` (per-env full gamefile path, from `parse_gamefile` at `env_manager.py:27–34`; re-stamped into step infos at 157–158), `self.tasks` (parsed from "Your task is to: "), `self.memory` (`SimpleMemory` — stores RAW `pre_text_obs` + executed action at line 153, so scaffold never leaks into history and each step's splice must be self-contained), `self.pre_text_obs`, `self.envs.get_admissible_commands`, `self.config` (gives `env.rollout.n`, `env.history_length`), `self.projection_f`. Task category = substring match against the six slugs, exactly as `_process_gamefile` (`env_manager.py:229–242`).

Invariants the subclass must keep: `'anchor'` stays the raw `text_obs` (GiGPO grouping key); batch index == env worker index for the whole loop (assert at `rollout_loop.py:312`; never reorder); scaffold wording must not suggest any output format other than `<think>/<action>` (the projection lowercases, requires both tags, and zeroes validity otherwise).

---

## (3) Data path: rollout → recorder / trainer

Per-step, inside the manager (recorder side):
- **executed action + validity**: `step()` line 151 (`self.projection_f(...)`) — buffer here, per env index.
- **raw observation, done**: same scope (lines 152–166).
- **gamefile / category**: cached at reset (line 140), re-stamped into infos (157–158).
- **injected flag**: the manager's own coin — never reconstructed from (seed, p).

Episode end (recorder flush = `success_evaluator`, fired once per rollout at `rollout_loop.py:407–412`):
- **uid (group id)** and **traj_uid**: minted at `rollout_loop.py:314–325`, attached to every step-dict at 358–359; reach the manager only here, inside `total_batch_list` — never during `step()`. Join buffered records by env index (safe: index-stable throughout).
- **success**: `info['won']` of the last active step (`_process_batch`, `env_manager.py:214–227`); `episode_rewards` = 10.0 × won for the text env (`envs.py:48–53`).

Trainer side (what the PPO batch does and does not contain): the dataloader batch is discarded wholesale (`del batch; batch = gen_batch_output`, `ray_trainer.py:1108–1109`). `gather_rollout_data` drops inactive steps (line 266) and broadcasts only batch-**mean** success rates into rows (257–259, 274–275) — per-env success never reaches the training batch, which is why the recorder must live in the manager. `text_bare` rides `non_tensor_batch` end-to-end (path verified in 1b) and is consumed only by the swap at `ray_trainer.py:1118+`.

---

## (4) Val-log parsing and checkpoint completeness

**Console format** (`LocalLogger` → `concat_dict_to_str`, `verl/utils/logger/aggregate_logger.py:23–29`): one line per log call, `step:{N}` then `" - "`-joined `{key}:{value:.3f}`, numeric values only.

**Keys** (built at `ray_trainer.py:814–826`; success keys from `env_manager.py:214–242` averaged in `rollout_loop.py:257–259`):
- `val/success_rate` — the official number (mean over one draw's episodes).
- `val/{task}_success_rate` for the six slugs: `pick_and_place`, `pick_two_obj_and_place`, `look_at_obj_in_light`, `pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`, `pick_clean_then_place_in_recep`.
- `val/text/test_score` — mean total episode reward = 10 × success (data_source is literally `'text'` from `prepare.py`). Do not confuse with success rate.
- `val/text/tool_call_count/mean` — irrelevant here.

**Draw shape**: step-0 (`val_before_train`) appears twice — a `pprint "Initial validation metrics: {...}"` dict AND a `step:0 - val/...` logger line (`ray_trainer.py:1032–1036`). In-training draws (`test_freq=5` in the script) are `metrics.update(val_metrics)`-merged into the single big per-step metrics line (verified at the validate block + single `logger.log` in fit). **Parse by key presence (`val/success_rate:`), never by line shape.** One draw = `data.val_batch_size` (128) sampled episodes on `eval_in_distribution` (valid_seen), temperature 0.4, `do_sample=True`, `val_kwargs.n=1`; DESIGN's VAL_N=3 draws is orchestrator-level repetition.

**Checkpoint completeness criterion** — a usable `global_step_N` under `trainer.default_local_dir` (default `checkpoints/<project>/<experiment>`, RELATIVE to launch cwd — set it absolute) contains, with W = nnodes × n_gpus_per_node:
- `actor/model_world_size_{W}_rank_{r}.pt`, `actor/optim_world_size_{W}_rank_{r}.pt`, `actor/extra_state_world_size_{W}_rank_{r}.pt` for **every** r ∈ 0..W−1 (`fsdp_checkpoint_manager.py:176–185`);
- `actor/config.json` + tokenizer/generation-config files (rank-0, lines 193–203);
- `data.pt` (dataloader state, `ray_trainer.py:934–936`);
- sibling top-level `latest_checkpointed_iteration.txt` containing N (939–941) — written last, so it doubles as the completion marker, but the robust check is all 3·W shards + `data.pt`.
- NO `critic/` (GiGPO is critic-free); NO `actor/huggingface/` unless `'hf_model'` is added to `actor_rollout_ref.actor.checkpoint.contents` (default `['model','optimizer','extra']`, yaml ~line 70). Known gotcha (memory): save model+optimizer+extra for resume; both example scripts default `save_freq=-1` = **no checkpoints** — must override `trainer.save_freq>0`.

---

## (5) Version / toolchain notes (uv env; this machine + B200)

**This machine**: 3+× H200 (sm_90a), driver 575.57.08 (CUDA ≤12.9 runtime OK), `/usr/local/cuda-12.4` and `cuda-12.9` present (system `nvcc` is 11.5 — do not use it; put cuda-12.9/bin first for any source builds), system Python 3.10.12, `uv 0.11.6` at `~/.local/bin/uv`.

**Repo pins** (verified): `requirements.txt` (dev lockfile-ish): `transformers==4.51.1`, `tensordict<=0.6.2`, `flash-attn`, vllm commented out (`# vllm==0.8.4`), no torch pin. `setup.py` install_requires: `ray[default]>=2.41.0,<=2.50.0`, `tensordict>=0.8.0,<=0.10.0,!=0.9.0`, `transformers<=4.57.3`; extras: `vllm>=0.8.5,<=0.11.0` (vllm extra), `torch==2.8.0` (sglang extra only). The tensordict conflict between the two files is inherited upstream, not drift — follow setup.py (vllm ≥0.8.5 requires tensordict ≥0.8).

**One env spec that works on both H200 and B200** (B200 = Blackwell sm_100 → needs torch built with cu128 and a Blackwell-capable vllm):
- Python 3.10 (matches the machine; 3.11 also fine).
- `torch==2.8.0` from the cu128 index (satisfies the sglang-extra pin; sm_90a and sm_100 both in the cu128 wheels).
- `vllm` 0.10.x (within the `<=0.11.0` bound; first-class Blackwell support; will co-resolve torch — install vllm first and let it drive, then verify torch==2.8.0+cu128 landed).
- `tensordict>=0.8,<=0.10,!=0.9.0`; `ray[default]` 2.41–2.50; `transformers`: start at 4.51.1 (the tested pin) and only lift toward ≤4.57.3 if vllm 0.10 demands it — resolve in the spike.
- `flash-attn`: prebuilt wheel matching torch 2.8/cu12x ABI, else `uv pip install flash-attn --no-build-isolation` with cuda-12.9 nvcc.
- Editable install of the repo with `--no-deps` (or `uv pip install -e .` after the above, letting the already-installed pins satisfy it) to avoid the requirements.txt/setup.py tensordict fight.
- **ALFWorld runtime deps are undeclared**: the vendored `env_package/alfworld` imports `textworld` (verified in `alfred_tw_env.py`) and there is no requirements file for it anywhere in the package. The dependency set (textworld + its pddl toolchain, opencv, etc.) must be lifted from the old fork's working conda env (`pip list`) — do not guess from the repo (surveyor-5 risk, confirmed).
- **B200-specific**: the launch script's line 3 `export VLLM_ATTENTION_BACKEND=XFORMERS` is an H100/H200-era choice; xformers kernels for sm_100 are not a given — make the backend conditional in our autoscaffold launch copy (drop the export / use FLASH_ATTN or FLASHINFER on B200). Known from prior runs: vLLM wake_up OOM is fixed with actor `param_offload=True`, and `expandable_segments` conflicts with vLLM cumem — keep it unset.
- Ops (from prior incidents, in DESIGN): `ALFWORLD_DATA` must be node-local (the ~18k-file scan runs twice at startup inside TaskRunner; NFS = minutes of silent hang), unset `ALFWORLD_DATA` does NOT fail fast (vendored `info.py` defaults to `~/.cache/alfworld` → "Overall we have 0 games"); per-run fresh `RAY_TMPDIR`; no `RAY_ADDRESS` in the shell (attach-guard at `main_ppo.py:36`); 256 persistent env actors × 0.1 CPU must fit under `ray_init.num_cpus`; `+data.dataloader_num_workers=0` (hidden default is 8, needs the `+` prefix — key absent from the yaml).

---

## (6) Risks / unknowns needing a code spike before implementation

1. **Prompt-length budget (highest-risk)**: ALFWorld's `build_text_obs` has no length guard (unlike Webshop's 13k-char fallback), and the example script sets `data.truncation='error'` with `max_prompt_length=2048` — an over-long spliced prompt raises `RuntimeError` and kills the run (`rollout_loop.py:171–172` / `tokenize_and_postprocess_data`). Spike: tokenize worst-case scaffold (8 items × cap) + `ALFWORLD_TEMPLATE` + history_length=2 observations; decide between a scaffold char budget in the subclass or raising `max_prompt_length`.
2. **Env-var visibility inside TaskRunner**: confirm `AUTOSCAFFOLD_ALFWORLD` is visible in the Ray actor in both launch modes (fresh local cluster vs. anything pre-existing); ship the `+ray_init.runtime_env.env_vars.*` override in the launch copy regardless.
3. **Flag-off equivalence**: with all four blocks applied and flags off, 5 training steps on H200 must reproduce a known-good run's metrics; with the flag on but empty scaffold / p=0 everywhere, the spliced prompt must be byte-identical to the vanilla prompt (splice is a strict no-op) and `text_bare == text`.
4. **bare_prompt_loss semantics under KL**: the 1.5B script uses `use_kl_loss=True, kl_loss_coef=0.01` — after the swap, ref_log_prob and entropy condition on the bare prompt too, and `training/rollout_probs_diff_*` (ray_trainer.py:1153–1175) becomes a full-vs-bare divergence metric (large by construction, loss-inert since `rollout_log_probs` is not in `dp_actor` select_keys). Spike: one swapped step; sanity-check loss/KL magnitudes and accept the metric discontinuity explicitly.
5. **Truncation of bare prompts in the swap**: `text_bare` is shorter than the spliced prompt but not guaranteed under `max_prompt_length` with `truncation='error'`; the swap helper must tokenize-check before overwrite (or the spike proves it can't overflow given the budget from item 1).
6. **transformers/vllm co-resolution**: 4.51.1 vs vllm 0.10's floor — resolve in the env-build spike on H200 before touching a B200 node; then a 1-step smoke on B200 to validate the attention-backend choice and cu128 wheels.
7. **ALFWorld dep excavation**: enumerate the old conda env's textworld/alfworld dependency closure and freeze it into `autoscaffold/` (uv-installable); the repo declares none of it.
8. **Recorder-vs-filtering assumption**: the recorder design assumes `filter_groups.enable=False` (vanilla loop, success_evaluator once per training step). Add an assert in the subclass so a future DAPO experiment fails loudly instead of double-recording.
9. **Group-layout coupling**: injection coins, uid minting (`i % env.rollout.n`), and worker seeding (`seed + i // group_n`) all assume consecutive-block layout with val `group_n=1` — encode this as a startup assert in the subclass (`len(envs) % group_n == 0`, val manager is vanilla class).
10. **`prompts` tensor consumers**: `apply_invalid_action_penalty` is ON in the script (`use_invalid_action_penalty=True`); the swap must overwrite `prompts` and re-pad to the same width or `valid_response_length` silently mis-indexes — covered by the helper contract, verify in the spike with a deliberate shape assert.
