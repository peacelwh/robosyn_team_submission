#!/bin/bash
# ----------------------------------------------------------------------------
# Build the evaluation environment for policy/robosyn_team (π0.5 tasks).
# Run once from the RoboSynChallenge root, with EmbodiChain v0.2.4 checked out
# next to it exactly as in the official setup instructions:
#
#   bash policy/robosyn_team/setup_env.sh
#
# The resulting package set (official lock + dexsim-engine 0.4.3) was verified end to
# end -- official evaluator, closed loop, GPU self-check passing -- on RTX 5090
# (driver 580), RTX 4090D (driver 580) and RTX 3090 (driver 535).
#
# Why not just `cd policy/pi05 && uv sync`: policy/pi05/uv.lock pins
# dexsim-engine==0.4.1, but EmbodiChain v0.2.4 requires dexsim-engine==0.4.3 (it
# imports dexsim.types.ToneMappingType, which 0.4.1 lacks).  A plain `uv sync`
# either fails to resolve or installs 0.4.1, and importing EmbodiChain then fails.
# Everything else is installed exactly as locked (torch 2.7.1+cu126, JAX 0.5.3,
# warp-lang 1.14.0, ...).  No tracked file of the official repository is changed;
# the result is the usual policy/pi05/.venv that policy/robosyn_team/eval.sh uses.
#
# Notes:
#   * torch stays at the locked 2.7.1+cu126.  That build cannot run CUDA kernels
#     on sm_120 (RTX 50xx), but the evaluation never needs them: the simulator's
#     environment runs on CPU and π0.5 runs in JAX, which works on sm_120.  A cu128
#     torch is not needed and was not validated (it also upgrades the nvidia-* CUDA
#     libraries that JAX shares), so do not swap it in.
#   * Keep the checkout under a short path (e.g. /opt/rsc/RoboSynChallenge):
#     DexSim 0.4.3 aborts in OptiXMaterialMgr::CreateMaterial (DFString.h:94)
#     when the repository sits under a long prefix (seen with 44 characters).
#
# Overrides: UV (uv binary), UV_HTTP_TIMEOUT (default 600 s here), ROBOSYN_DEXSIM_INDEX, or
# ROBOSYN_DEXSIM_FIND_LINKS (a local directory holding the dexsim_engine-0.4.3 wheel).
# ----------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PI05="$ROOT/policy/pi05"
UV="${UV:-uv}"
DEXSIM_INDEX="${ROBOSYN_DEXSIM_INDEX:-http://pyp.open3dv.site:2345/simple/}"
[[ -f "$PI05/uv.lock" ]] || { echo "run from the RoboSynChallenge root (policy/pi05/uv.lock not found)" >&2; exit 1; }
[[ -d "$ROOT/../EmbodiChain" ]] || { echo "EmbodiChain v0.2.4 must be checked out next to RoboSynChallenge" >&2; exit 1; }
(( ${#ROOT} <= 40 )) || echo "[robosyn_team] WARNING: $ROOT is ${#ROOT} characters long; DexSim 0.4.3 may abort under long paths (see notes)" >&2

# The locked pymeshlab 2025.7.post1 only has manylinux_2_35 wheels; on older glibc
# (e.g. Ubuntu 20.04) install 2023.12.post3 instead (DexSim needs >= 2023.12.post3).
skip=(--no-install-package dexsim-engine)
glibc="$(ldd --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
legacy_glibc=false
if [[ -n "$glibc" ]] && [[ "$(printf '%s\n2.35\n' "$glibc" | sort -V | head -n 1)" != 2.35 ]]; then
    legacy_glibc=true; skip+=(--no-install-package pymeshlab)
fi

# Large wheels (jax-cuda12-plugin, torch, nvidia-*) can time out on slow links with uv's 30 s default.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
echo "[robosyn_team] 1/3 policy/pi05: official uv.lock, without dexsim-engine (UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT)"
for attempt in 1 2 3; do
    (cd "$PI05" && GIT_LFS_SKIP_SMUDGE=1 "$UV" sync --frozen "${skip[@]}") && break
    (( attempt < 3 )) || { echo "[robosyn_team] uv sync failed 3 times" >&2; exit 1; }
    echo "[robosyn_team] uv sync failed (attempt $attempt/3); retrying in 30 s (downloaded wheels are reused)" >&2
    sleep 30
done
PY="$PI05/.venv/bin/python"
if $legacy_glibc; then
    echo "[robosyn_team]     glibc $glibc < 2.35: pymeshlab 2023.12.post3 instead of the locked 2025.7.post1"
    "$UV" pip install --python "$PY" --no-deps "pymeshlab==2023.12.post3"
fi

echo "[robosyn_team] 2/3 dexsim-engine 0.4.3 (required by EmbodiChain v0.2.4)"
if [[ -n "${ROBOSYN_DEXSIM_FIND_LINKS:-}" ]]; then
    "$UV" pip install --python "$PY" --no-deps --no-index --find-links "$ROBOSYN_DEXSIM_FIND_LINKS" "dexsim-engine==0.4.3"
else
    "$UV" pip install --python "$PY" --no-deps --index-url "$DEXSIM_INDEX" \
        --allow-insecure-host pyp.open3dv.site:2345 "dexsim-engine==0.4.3"
fi

echo "[robosyn_team] 3/3 import and device check"
"$PY" - <<'PY'
import importlib.metadata as md
import jax, jax.numpy as jnp, torch
import embodichain, dexsim  # noqa: F401  (fails here if the dexsim/EmbodiChain pair is wrong)
from openpi.training import config as _config  # noqa: F401
print("dexsim-engine", md.version("dexsim-engine"), "| embodichain", md.version("embodichain"),
      "| torch", torch.__version__, "| warp-lang", md.version("warp-lang"))
assert md.version("dexsim-engine") == "0.4.3", "dexsim-engine must be 0.4.3"
x = jnp.ones((256, 256))
print("jax", jax.__version__, jax.devices()[0], "matmul ok:", float((x @ x)[0, 0]) == 256.0)
PY
echo "[robosyn_team] environment ready: $PY"
