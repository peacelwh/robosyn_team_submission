#!/bin/bash
# ----------------------------------------------------------------------------
# Pre-download every checkpoint referenced by task_checkpoints.json so that the
# evaluation itself never needs network access (organiser notice 2026-09-09:
# network availability during inference is not guaranteed).
#
#   bash policy/robosyn_team/download_checkpoints.sh            # all tasks
#   bash policy/robosyn_team/download_checkpoints.sh click_bell # one task
#
# Each distinct repo@revision is downloaded once into
# policy/robosyn_team/checkpoints/<org>__<repo>@<revision>, which is exactly
# where deploy_policy.ensure_checkpoint() looks.  Set HF_ENDPOINT for a mirror.
# ----------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Interpreter: $PYTHON_BIN, else the policy/pi05 environment built by setup_env.sh (it ships
# huggingface_hub), else python3.  The download needs huggingface_hub; fail early with a hint.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/policy/pi05/.venv/bin/python" ]]; then PYTHON_BIN="$REPO_ROOT/policy/pi05/.venv/bin/python"; else PYTHON_BIN=python3; fi
fi
"$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null || {
    echo "download_checkpoints.sh: $PYTHON_BIN cannot import huggingface_hub; run policy/robosyn_team/setup_env.sh first," >&2
    echo "or set PYTHON_BIN to a Python with huggingface_hub (pip install huggingface_hub)." >&2
    exit 1
}
echo "download_checkpoints.sh: using $PYTHON_BIN"
# Slow or distant links (e.g. a mirror) need longer timeouts than huggingface_hub's 10 s defaults.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
MANIFEST="${ROBOSYN_TASK_MANIFEST:-$SCRIPT_DIR/task_checkpoints.json}"
"$PYTHON_BIN" - "$SCRIPT_DIR" "$MANIFEST" "$@" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
policy_root, manifest_path, only = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3:])
sys.path.insert(0, str(policy_root.parent.parent))
manifest = json.loads(manifest_path.read_text())
from huggingface_hub import snapshot_download

import os
root = Path(os.environ.get("ROBOSYN_CHECKPOINTS_ROOT") or policy_root / "checkpoints")

def target(spec, task):
    rev = (spec.get("hf_revision") or "main")[:12]
    return root / f"{spec['hf_repo'].replace('/', '__')}@{rev}"

seen = {}
for task, entry in manifest["tasks"].items():
    if only and task not in only:
        continue
    spec = dict(manifest.get("defaults", {})); spec.update(entry)
    if spec.get("local_dir") and not spec.get("hf_repo"):
        print(f"{task}: local_dir only ({spec['local_dir']}), nothing to download")
        continue
    key = (spec["hf_repo"], spec.get("hf_revision"))
    dest = target(spec, task)
    if key in seen:
        print(f"{task}: shares {key[0]}@{key[1] or 'main'} with {seen[key]}")
        continue
    seen[key] = task
    print(f"{task}: {key[0]}@{key[1] or 'main'} -> {dest}")
    # Retry interrupted downloads (snapshot_download resumes the files it already has).
    for attempt in range(1, 7):
        try:
            snapshot_download(repo_id=key[0], revision=key[1], repo_type="model", local_dir=str(dest))
            break
        except Exception as exc:  # network timeouts, mirror hiccups
            if attempt == 6:
                raise
            wait = min(60, 10 * attempt)
            print(f"  download attempt {attempt}/6 failed ({type(exc).__name__}: {str(exc)[:120]}); retrying in {wait} s")
            time.sleep(wait)
    # Integrity: every file listed in the export manifest must match its sha256.
    manifest_file = dest / "robosyn_export_manifest.json"
    if manifest_file.is_file():
        files = json.loads(manifest_file.read_text()).get("files", {})
        items = list(files.items()) if isinstance(files, dict) else [(f["path"], f) for f in files]
        bad = []
        for rel, info in items:
            info = info if isinstance(info, dict) else {"sha256": info}
            digest = hashlib.sha256()
            with open(dest / rel, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    digest.update(chunk)
            if digest.hexdigest() != info.get("sha256"):
                bad.append(rel)
        if bad:
            sys.exit(f"{dest}: {len(bad)} file(s) do not match robosyn_export_manifest.json, e.g. {bad[:3]}; delete them and re-run")
        print(f"  verified {len(items)} files against robosyn_export_manifest.json")
print(f"downloaded {len(seen)} distinct checkpoint(s)")
PY
