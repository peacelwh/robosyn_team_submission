# ----------------------------------------------------------------------------
# RoboSynChallenge submission adapter: per-task policy router.
#
# The organisers run scripts/eval_policy.py with this adapter for every task.
# ``task_checkpoints.json`` maps each task to a policy family (act / dp /
# smolvla / pi05), a Hugging Face checkpoint (repo + revision) and the rollout
# parameters that were validated locally.  The family adapters shipped with the
# official repository do the actual inference, so the evaluation interface and
# the synchronized inference timing stay byte-identical to the baselines.
#
# Interface required by scripts/eval_policy.py:
#   get_model(usr_args) -> model
#   eval(env, model, obs) -> (obs, info, truncated, inference_times_s)
#   reset_model(model) -> None
# ----------------------------------------------------------------------------
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

POLICY_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = POLICY_ROOT / "task_checkpoints.json"
SUPPORTED_FAMILIES = {"act", "dp", "smolvla", "pi05", "pi0"}


def load_manifest(path: str | os.PathLike | None = None) -> dict:
    manifest_path = Path(path or os.environ.get("ROBOSYN_TASK_MANIFEST") or DEFAULT_MANIFEST)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "robosyn.submission.v1":
        raise ValueError(f"unsupported task manifest schema in {manifest_path}")
    return manifest


def resolve_task_spec(task_name: str, manifest: dict) -> dict:
    tasks = manifest.get("tasks", {})
    if task_name not in tasks:
        raise KeyError(f"task {task_name!r} is not covered by the submission manifest; tasks={sorted(tasks)}")
    spec = dict(manifest.get("defaults", {}))
    spec.update(tasks[task_name])
    family = spec.get("family")
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"task {task_name!r}: unsupported policy family {family!r}")
    return spec


def default_checkpoint_dir(spec: dict, task_name: str) -> Path:
    """Where a Hub checkpoint lives locally: one directory per repo@revision.

    Tasks that share a checkpoint (e.g. one multi-task π0.5 model for all ten)
    share the directory, so it is downloaded once.  The path is derived from the
    manifest alone - never from a per-task directory - so changing a task's
    repo/revision can never silently keep loading a previously cached model.
    """
    repo = spec.get("hf_repo")
    if not repo:
        raise ValueError(f"task {task_name!r}: manifest entry needs hf_repo or local_dir")
    revision = (spec.get("hf_revision") or "main")[:12]
    return checkpoints_root() / f"{repo.replace('/', '__')}@{revision}"


def checkpoints_root() -> Path:
    """policy/robosyn_team/checkpoints, or ROBOSYN_CHECKPOINTS_ROOT if the organisers keep them elsewhere."""
    return Path(os.environ.get("ROBOSYN_CHECKPOINTS_ROOT") or POLICY_ROOT / "checkpoints")


def ensure_checkpoint(spec: dict, task_name: str) -> Path:
    """Return the local checkpoint directory; never downloads anything.

    Competition rule (organiser Q&A, 2026-09): checkpoints may not be downloaded by
    the evaluation code.  They are fetched beforehand with download_checkpoints.sh
    (Hugging Face repo + pinned revision from task_checkpoints.json) into
    checkpoints/<org>__<repo>@<revision[:12]>, where this function looks.
    """
    local_dir = Path(spec.get("local_dir") or default_checkpoint_dir(spec, task_name))
    if not local_dir.is_absolute():
        local_dir = POLICY_ROOT / local_dir
    marker = spec.get("checkpoint_marker", "config.json")
    if (local_dir / marker).exists() or (local_dir / "params").is_dir():
        return local_dir
    source = f"{spec.get('hf_repo')}@{spec.get('hf_revision') or 'main'}" if spec.get("hf_repo") else "local_dir"
    raise FileNotFoundError(
        f"task {task_name!r}: checkpoint not found at {local_dir} ({source}). "
        "Fetch it before evaluating:  bash policy/robosyn_team/download_checkpoints.sh"
    )


def _family_module(family: str):
    if family in {"pi0", "pi05"}:
        # The official pi_model.py calls an OpenPI API that no longer exists and
        # hard-codes its checkpoint path; ship our own equivalent runtime instead.
        from . import pi05_runtime

        return pi05_runtime
    # policy/ is on sys.path (scripts/eval_policy.py prepends REPO_ROOT and policy/).
    return importlib.import_module(f"policy.{family}.deploy_policy")


def get_model(usr_args):
    task_name = usr_args.get("task_name")
    if not task_name:
        raise ValueError("task_name is required")
    manifest = load_manifest(usr_args.get("task_manifest"))
    spec = resolve_task_spec(task_name, manifest)
    family = spec["family"]
    checkpoint = ensure_checkpoint(spec, task_name)

    family_args = dict(usr_args)
    family_args.update(spec.get("usr_args", {}))
    if family in {"act", "dp"}:
        family_args["checkpoint_path"] = str(checkpoint)
    elif family == "smolvla":
        pretrained = checkpoint / "pretrained_model"
        family_args["checkpoint_dir"] = str(pretrained if pretrained.is_dir() else checkpoint)
        family_args.setdefault("smolvla_cuda_visible_devices", str(usr_args.get("gpu_id", 0)))
    impl = _family_module(family)
    if family in {"pi0", "pi05"}:
        model = impl.get_model(family_args, checkpoint)
        model._robosyn_family = family
        model._robosyn_impl = impl
        model._robosyn_task = task_name
        print(f"[robosyn_team] task={task_name} family={family} checkpoint={checkpoint}")
        return model
    if family == "smolvla" and os.environ.get("ROBOSYN_SMOLVLA_FLOAT_IMAGES", "0") == "1":
        # Newer LeRobot preprocessors (>= 0.6) resize with F.interpolate, which
        # rejects uint8 tensors; hand the worker float32 [0, 1] CHW images instead.
        _orig_extract = impl._extract_image

        def _extract_float(obs, sensor_name, env_index=0):
            image = _orig_extract(obs, sensor_name, env_index=env_index)
            return (image.astype("float32") / 255.0).copy()

        impl._extract_image = _extract_float
        print("[robosyn_team] smolvla images converted to float32 [0,1] for the worker")
    model = impl.get_model(family_args)
    try:
        model._robosyn_family = family
        model._robosyn_impl = impl
        model._robosyn_task = task_name
    except AttributeError:
        model = _Wrapped(model, family, impl, task_name)
    if family == "smolvla":
        _install_infer_timer(_unwrap(model))
    print(f"[robosyn_team] task={task_name} family={family} checkpoint={checkpoint}")
    return model


class _Wrapped:
    """Fallback holder for model objects that refuse attribute assignment."""

    def __init__(self, inner, family, impl, task):
        self.inner = inner
        self._robosyn_family = family
        self._robosyn_impl = impl
        self._robosyn_task = task

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _unwrap(model):
    return model.inner if isinstance(model, _Wrapped) else model


def _install_infer_timer(model):
    """Time every worker round trip so the 4-tuple evaluator contract is met.

    The upstream SmolVLA adapter still returns ``(obs, info, truncated)`` while
    scripts/eval_policy.py expects ``(obs, info, truncated, inference_times_s)``.
    ``model.infer`` covers observation transfer, worker preprocessing, model
    inference and action return, i.e. the same synchronized boundary the other
    adapters report (env.step excluded).
    """
    import time

    original = model.infer
    model._robosyn_times = []

    def timed_infer(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        model._robosyn_times.append(time.perf_counter() - started)
        return result

    model.infer = timed_infer


def eval(env, model, obs):
    inner = _unwrap(model)
    if hasattr(inner, "_robosyn_times"):
        inner._robosyn_times = []
    result = model._robosyn_impl.eval(env, inner, obs)
    if len(result) == 3:  # legacy adapter signature (SmolVLA)
        final_obs, info, truncated = result
        return final_obs, info, truncated, list(getattr(inner, "_robosyn_times", []))
    return result


def reset_model(model):
    model._robosyn_impl.reset_model(_unwrap(model))
