# RoboSynChallenge submission — `robosyn_team`

**Team PeaceVLA** — Wenhao Li, Kai Kang, Yurong Liu (SDU, SLAI, CUHK-SZ, NUS). Contact: wenhao.li@mail.sdu.edu.cn

This directory is the evaluation package we hand to the organisers
(GitHub repository + Hugging Face checkpoints, see the competition FAQ).
It plugs into the official `RoboSynChallenge` repository as
`policy/robosyn_team/` and follows the required adapter interface:

```
policy/robosyn_team/
├── __init__.py            # exports get_model / eval / reset_model
├── deploy_policy.py       # per-task router -> official family adapters (act / dp / smolvla / pi05)
├── deploy_policy.yml      # evaluator configuration (expert check on, videos on, 20 episodes)
├── eval.sh                # entry point:  bash policy/robosyn_team/eval.sh <task> <setting> [gpu] [extra...]
├── download_checkpoints.sh  # fetch each distinct repo@revision once (offline evaluation afterwards)
├── setup_env.sh           # builds policy/pi05/.venv: official uv.lock + dexsim-engine 0.4.3 (see below)
├── pi05_runtime.py        # self-contained π0.5 loader/runner (see "π0.5 tasks")
└── task_checkpoints.json  # task -> family, Hugging Face repo/revision, rollout parameters
```

## Current submission: multi-task π0.5

Every task routes to a π0.5 checkpoint (family `pi05`) fine-tuned by the team on the
official EmbodiChain synthetic data of all ten tasks; the task is selected by the
evaluator's language instruction.  `task_checkpoints.json` names the checkpoint of
each task (one multi-task model, possibly a second one for individual tasks, each
pinned to a Hugging Face revision).

```bash
cd RoboSynChallenge                                        # keep this checkout under a short path, see below
bash policy/robosyn_team/setup_env.sh                      # once: builds policy/pi05/.venv
bash policy/robosyn_team/download_checkpoints.sh           # once; evaluation is offline afterwards
bash policy/robosyn_team/eval.sh click_bell random 0 --max_episodes 20 --seed 0
```

Notes for the evaluation machine:

* **The evaluation code never downloads checkpoints** (competition rule).
  Run `download_checkpoints.sh` once before evaluating. It places the pinned
  Hugging Face revision under `policy/robosyn_team/checkpoints/<org>__<repo>@<rev>`,
  or under `$ROBOSYN_CHECKPOINTS_ROOT` if that is set. If the checkpoint is
  missing, evaluation stops with a clear error.
* **Environment (`setup_env.sh`):** the official `policy/pi05` project installed from
  its `uv.lock` (`uv sync --frozen`), except `dexsim-engine`, which is installed at
  **0.4.3** instead of the locked 0.4.1: EmbodiChain v0.2.4 requires 0.4.3 and fails
  to import with 0.4.1.  torch stays at the locked 2.7.1+cu126.  On an RTX 5090
  (sm_120) that build cannot run CUDA kernels, but the evaluation never needs them:
  the simulator's environment runs on CPU and π0.5 runs in JAX 0.5.3, which uses the
  RTX 5090.  A cu128 torch is therefore not needed; it was not validated (it also
  upgrades the CUDA libraries that JAX shares), so please keep the locked build.
  This setup was verified closed loop on RTX 5090 (driver 580), RTX 4090D (driver
  580) and RTX 3090 (driver 535).
* **Slow networks:** `setup_env.sh` raises uv's download timeout to 600 s (`UV_HTTP_TIMEOUT`) and
  retries `uv sync` up to three times; the large CUDA wheels otherwise time out on slow links.
  `download_checkpoints.sh` likewise uses longer Hugging Face timeouts, retries interrupted
  downloads and checks every file against the sha256 list shipped with the checkpoint.
* **ffmpeg:** the official evaluator records episode videos through `ffmpeg`
  (`eval_video_log: true` in `deploy_policy.yml`, as in the official baselines), so
  `ffmpeg` must be on `PATH`.
* **Short paths:** DexSim 0.4.3 aborts in `OptiXMaterialMgr::CreateMaterial`
  (`DFString.h:94`) when the checkout sits under a long directory prefix (we hit it
  with a 44-character prefix).  Keep `RoboSynChallenge/` and `EmbodiChain/` under a
  short path such as `/opt/rsc/`.
* **GPU self-check with CPU fallback:** each checkpoint ships a fixed test input,
  `robosyn_golden.npz`, whose reference actions were computed on CPU. At load
  time the runtime runs this input on the GPU in a subprocess. If the GPU run
  crashes, or its actions differ from the reference by more than the tolerance
  stored with the checkpoint (0.2), the task is evaluated on CPU instead. Measured
  GPU deviations: 0.015 on RTX 5090, 0.008-0.02 on RTX 3090, 0.04 on H100; one
  π0.5 action chunk takes about 0.12 s on an RTX 5090. This guards against a JAX build that does not
  support, or miscompiles on, a new GPU generation. The CPU fallback uses the
  same model and is only slower per call. Set `ROBOSYN_PI05_DEVICE=gpu` or `cpu`
  to force a choice.
* **Inference-time retry (manipulate_pipette only):** `task_checkpoints.json` sets
  `usr_args.watchdog.retry_at` for this task.  The evaluator ends an episode as soon as the
  task succeeds, so if the policy is still being called 320 steps into an attempt, that attempt
  has failed; the runtime then replays its own executed joint commands in reverse (every 4th
  command) back to the pose of step 175, before the pressing phase, and lets the policy try
  again, at most 4 times per episode.  Only the policy's own step count and command history
  are used (no simulator state, no success flag).  All other tasks run the plain policy.
* **Simulator crash guard:** DexSim 0.4.3 occasionally segfaults while it loads the
  robot, before the first episode, and sometimes in its teardown after the results
  are written. `eval.sh` restarts the evaluator (up to `ROBOSYN_EVAL_RETRIES`, default
  2, more times) only when it died from a signal *before* printing
  `Metrics saved to:`; a completed run is never repeated, and a teardown crash after
  the metrics were saved is reported as success. Python errors are not retried.
* `usr_args.gripper_scale` (per task, default 1) rescales the two gripper
  commands, in case an environment's gripper convention changes.

The per-task router described below (ACT / DP / SmolVLA families) is still
supported, but this submission does not use it.

## How the organisers run it

```bash
# 1. Official workspace (EmbodiChain v0.2.4 + RoboSynChallenge main), then
git clone <our repository> RoboSynChallenge_ws/robosyn_team_submission
cp -r RoboSynChallenge_ws/robosyn_team_submission/policy/robosyn_team RoboSynChallenge/policy/

# 2. Family environments exactly as in the official tutorials
(cd RoboSynChallenge/policy/act && uv sync --frozen)      # ACT and DP tasks
(cd RoboSynChallenge/policy/dp  && uv sync --frozen)
bash RoboSynChallenge/policy/smolvla/setup_lerobot.sh      # SmolVLA tasks
bash RoboSynChallenge/policy/robosyn_team/setup_env.sh    # pi05 tasks: official uv.lock + dexsim-engine 0.4.3

# 3. Fetch every checkpoint once, so the evaluation itself needs no network
cd RoboSynChallenge
bash policy/robosyn_team/download_checkpoints.sh

# 4. Evaluate one task
bash policy/robosyn_team/eval.sh click_bell random 0 --max_episodes 20 --seed 0
```

`eval.sh` selects the interpreter from the task's family (`policy/<family>/.venv`),
passes `--task_manifest policy/robosyn_team/task_checkpoints.json` to
`scripts/eval_policy.py`, and everything else (expert feasibility pre-check,
`evaluation_metrics.json`, inference timing) is the untouched official code path.
`ROBOSYN_TASK_MANIFEST` overrides the manifest, `PYTHON_BIN` the interpreter,
`SMOLVLA_PYTHON` / `SMOLVLA_LEROBOT_ROOT` the SmolVLA worker.

## π0.5 tasks

`pi05_runtime.py` loads OpenPI checkpoints directly (config + `create_trained_policy`
with the checkpoint's own normalization asset, flat `observation/*` keys,
truncation-safe chunk execution, synchronized timing excluding `env.step`), so the
package does not depend on the official `policy/pi05/pi_model.py`, which still calls
a removed OpenPI keyword and hard-codes its checkpoint path.

A π0.5 entry needs `local_dir` (or `hf_repo`) pointing at the checkpoint directory
that contains `params/` and `assets/`, plus
`usr_args.train_config_name` (default `pi05_base_robosynchallenge_full`) and
optionally `pi0_step` (open-loop actions executed per inference, default 50).
`eval.sh` passes the physical GPU index to DexSim (`--gpu_id`) and leaves CUDA
unmasked, because DexSim's OptiX denoiser opens that index through the CUDA driver;
JAX is confined to the same card inside the process (`ROBOSYN_JAX_GPU`), so any
card works.  Exported checkpoints carry the PaliGemma tokenizer
(`openpi_cache/big_vision/`), so inference never needs network access.

## Checkpoint table

`task_checkpoints.json` is the single source of truth.  Each entry records the
family, the Hugging Face repository and the exact revision that was evaluated
locally; `released_random_sr` keeps the organisers' 100-episode `random`
result of that checkpoint for reference.  Replace an entry with a team
checkpoint by pointing `hf_repo`/`hf_revision` (or `local_dir`) at it and
re-running the local protocol (`scripts/eval.sh`) before submitting.

Checkpoint directories are resolved from the manifest only: `local_dir` if
given, otherwise `checkpoints/<org>__<repo>@<revision[:12]>`, which
`download_checkpoints.sh` fills ahead of time (the evaluation itself never downloads;
a missing checkpoint stops with an error that names this script).

For the team's single multi-task π0.5 model, generate the whole manifest so
all ten tasks route to one exported checkpoint (the task is selected by the
evaluator's language instruction):

```bash
python scripts/make_pi05_manifest.py --hf-repo <org>/<repo> --hf-revision <sha> \
    --out submission/policy/robosyn_team/task_checkpoints.json
python scripts/validate_submission.py --strict
```

## Local dry run of the submission package

```bash
cd /path/to/RoboSynVLA
bash scripts/submission_dry_run.sh click_bell random 0 --max_episodes 2
```
