#!/bin/bash
# ----------------------------------------------------------------------------
# RoboSynChallenge submission entry point (robosyn_team per-task router)
#
#   bash policy/robosyn_team/eval.sh <task_name> <setting> [gpu_id] [extra_opts...]
#   bash policy/robosyn_team/eval.sh click_bell random 0 --max_episodes 20 --seed 0
#
# The Python interpreter is chosen from the policy family selected for the
# task in task_checkpoints.json (ACT/DP -> policy/act|dp/.venv, SmolVLA ->
# simulator python + SMOLVLA_PYTHON worker, pi0.5 -> policy/pi05/.venv); set
# PYTHON_BIN to override.  Evaluation never downloads anything: fetch the
# checkpoints once beforehand with policy/robosyn_team/download_checkpoints.sh
# (see README.md for the exact repo/revision per task).
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
POLICY_NAME=robosyn_team

TASK_NAME="${1:?Usage: bash policy/robosyn_team/eval.sh <task_name> <setting> [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash policy/robosyn_team/eval.sh <task_name> <setting> [gpu_id] [extra_opts...]}"
GPU_ID="${3:-0}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

MANIFEST="${ROBOSYN_TASK_MANIFEST:-$SCRIPT_DIR/task_checkpoints.json}"
FAMILY="$(python3 - "$MANIFEST" "$TASK_NAME" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
spec = dict(manifest.get("defaults", {})); spec.update(manifest["tasks"][sys.argv[2]])
print(spec["family"])
PY
)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    case "$FAMILY" in
        act|dp|pi0|pi05) PYTHON_BIN="$REPO_ROOT/policy/$FAMILY/.venv/bin/python" ;;
        smolvla) PYTHON_BIN="${ROBOSYN_SIM_PYTHON:-python}" ;;
        *) echo "unknown family $FAMILY" >&2; exit 1 ;;
    esac
fi
[[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$REPO_ROOT/policy/$FAMILY:$REPO_ROOT/policy/$FAMILY/src:$REPO_ROOT/policy/$FAMILY/packages/openpi-client/src:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
# DexSim renders on the PHYSICAL card --gpu_id (Vulkan ignores the CUDA mask)
# and opens the same id through the CUDA driver for OptiX (which honours it).
# Torch policies get a one-card mask so both address the same GPU as ordinal 0
# of the policy; for π0.5 the simulator shares the process, so the mask would
# make --gpu_id>0 invalid for OptiX: leave it unset and pin JAX in-process.
EXTRA_ARGS+=(--gpu_id "$GPU_ID")
if [[ "$FAMILY" == pi0 || "$FAMILY" == pi05 ]]; then
    unset CUDA_VISIBLE_DEVICES
    export ROBOSYN_JAX_GPU="$GPU_ID"
    EXTRA_ARGS+=(--pytorch_device cpu)   # π0.5 runs on JAX; torch only formats actions
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
if [[ "$FAMILY" == smolvla ]]; then
    # The SmolVLA worker is a separate process; it gets its own mask while the
    # simulator parent stays unmasked so DexSim can address the physical card.
    unset CUDA_VISIBLE_DEVICES
    EXTRA_ARGS+=(--smolvla_cuda_visible_devices "$GPU_ID")
    [[ -n "${SMOLVLA_PYTHON:-}" ]] && EXTRA_ARGS+=(--smolvla_python "$SMOLVLA_PYTHON")
    [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]] && EXTRA_ARGS+=(--lerobot_root "$SMOLVLA_LEROBOT_ROOT")
fi

echo "========================================="
echo "  robosyn_team Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)  family=$FAMILY"
echo "  GPU:        $GPU_ID   Python: $PYTHON_BIN"
echo "  Manifest:   $MANIFEST"
echo "========================================="

cd "$REPO_ROOT"
# DexSim 0.4.3 occasionally segfaults while it loads the robot, before the first
# episode (8 times in about 300 evaluation runs on our RTX 3090s, also seen on an
# RTX 5090), and
# the simulator can also crash in its teardown after the results are written.
# Restart the evaluator when it dies from a signal BEFORE writing its metrics
# (at most ROBOSYN_EVAL_RETRIES more times, default 2), so such a crash does not
# cost the task; a run that already wrote "Metrics saved to:" is never repeated.
# Ordinary errors (Python exceptions, exit status < 128) are not retried.
attempts=$(( ${ROBOSYN_EVAL_RETRIES:-2} + 1 ))
for (( attempt = 1; attempt <= attempts; attempt++ )); do
    run_log="$(mktemp "${TMPDIR:-/tmp}/robosyn_team_eval.XXXXXX")"
    set +e
    PYTHONWARNINGS=ignore::UserWarning \
    EMBODICHAIN_SIM_EXIT_PROCESS=0 \
    "$PYTHON_BIN" scripts/eval_policy.py \
        --config policy/$POLICY_NAME/deploy_policy.yml \
        --overrides \
        --task_name "$TASK_NAME" \
        --setting "$SETTING" \
        --task_manifest "$MANIFEST" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$run_log"
    status=${PIPESTATUS[0]}
    set -e
    saved=false
    grep -aq "Metrics saved to:" "$run_log" && saved=true
    rm -f "$run_log"
    (( status == 0 )) && exit 0
    if $saved; then
        echo "[robosyn_team] evaluator exited with status $status after saving its metrics (simulator teardown); the run is complete" >&2
        exit 0
    fi
    if (( status < 128 || attempt == attempts )); then
        exit "$status"
    fi
    echo "[robosyn_team] evaluator died from signal $(( status - 128 )) before producing results (attempt $attempt/$attempts); restarting" >&2
    sleep 5
done
