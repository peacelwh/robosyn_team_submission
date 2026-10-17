"""Self-contained π0/π0.5 runtime for the RoboSynChallenge submission adapter.

The official ``policy/pi05/pi_model.py`` cannot be used as-is: it calls
``create_trained_policy(..., robotwin_repo_id=...)``, a keyword the vendored
OpenPI no longer accepts, hard-codes ``policy/pi05/checkpoints/...`` as the only
checkpoint location, and feeds the policy a nested ``{"images": {...}}``
observation although ``create_trained_policy`` only installs the model-side
transforms (which expect flat ``observation/image`` keys).

This module implements the same behaviour without those defects and keeps the
timing boundary identical to the other official adapters: timing starts at the
raw environment observation and stops once executable actions exist, excluding
``env.step``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _test_image_perturbation(frame: np.ndarray) -> np.ndarray:
    """Diagnostics only (off unless the environment variable is set; never set by the organisers).

    ROBOSYN_TEST_IMAGE_BLUR=<sigma px> applies a Gaussian blur and ROBOSYN_TEST_IMAGE_JPEG=<quality>
    a JPEG round trip, to measure how sensitive the policy is to renderer / driver differences
    (the RTX 4090D with driver 580 renders slightly softer frames than the RTX 3090 with driver 535).
    """
    sigma = float(os.environ.get("ROBOSYN_TEST_IMAGE_BLUR", "0") or 0)
    quality = int(os.environ.get("ROBOSYN_TEST_IMAGE_JPEG", "0") or 0)
    if sigma <= 0 and quality <= 0:
        return frame
    import io
    from PIL import Image, ImageFilter

    image = Image.fromarray(frame)
    if sigma > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if quality > 0:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def encode_obs(obs, env_index: int = 0):
    """Return (images[cam_high, cam_right_wrist, cam_left_wrist], state).

    Frames are returned at NATIVE resolution (480x640), exactly like the
    official adapter.  Do not resize here: the model transforms apply
    ``ResizeImages(224, 224)``, i.e. ``resize_with_pad`` (aspect-preserving
    letterbox), which is what the checkpoint saw during training.  A
    pre-stretch to 224x224 would make that step a no-op and feed the vision
    tower a 1.33x vertically distorted, unpadded image on every step.
    """
    images = []
    for name in ("cam_high", "cam_right_wrist", "cam_left_wrist"):
        frame = _to_numpy(obs["sensor"][name]["color"])
        if frame.ndim == 4:
            frame = frame[env_index]
        frame = frame[..., :3]
        if np.issubdtype(frame.dtype, np.floating):
            if float(np.nanmax(frame)) <= 1.5:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255)
        images.append(_test_image_perturbation(np.ascontiguousarray(frame.astype(np.uint8))))
    state = _to_numpy(obs["robot"]["qpos"])
    if state.ndim > 1:
        state = state[env_index]
    return images, np.asarray(state, dtype=np.float32).reshape(-1)


def format_action(action, env):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    env_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action.shape[0] < env_dim:
        raise ValueError(f"policy action dim {action.shape[0]} < env action dim {env_dim}")
    tensor = torch.as_tensor(action[:env_dim], dtype=torch.float32, device=env.unwrapped.device)
    return tensor.unsqueeze(0)


def _resolve_checkpoint(base: Path, train_config: str, model_name: str, checkpoint_id: int) -> Path:
    """Accept both the deployment name (30000) and OpenPI's zero-based dir (29999)."""
    requested = base / train_config / model_name / str(checkpoint_id)
    if requested.is_dir():
        return requested
    zero_based = base / train_config / model_name / str(checkpoint_id - 1)
    if checkpoint_id > 0 and zero_based.is_dir():
        print(f"[robosyn_team] checkpoint {checkpoint_id} resolved to {zero_based}")
        return zero_based
    raise FileNotFoundError(f"π0.5 checkpoint not found: {requested}")


TOKENIZER_RELPATH = Path("big_vision") / "paligemma_tokenizer.model"
GRIPPER_DIMS = [6, 13]   # left / right gripper in the 14-D joint action


def _pin_jax_to_gpu() -> None:
    """Restrict JAX to the evaluator's physical GPU without a CUDA mask.

    DexSim opens CUDA device ``--gpu_id`` through the driver API for OptiX, so
    masking with CUDA_VISIBLE_DEVICES makes every id except 0 invalid
    (CUDA_ERROR_INVALID_DEVICE).  eval.sh therefore leaves the mask unset and
    exports ROBOSYN_JAX_GPU; JAX is limited to that card here, before its
    backend initialises, so it neither lands on GPU 0 nor preallocates memory
    on every card.

    Only the configuration is changed here: the backend must NOT be initialised
    before OpenPI is imported (on some hosts, e.g. RTX 3090 / driver 535.230,
    importing openpi.models.model after the CUDA backend is up segfaults), so
    callers import OpenPI first and check the device with _check_jax_pin().
    """
    gpu = os.environ.get("ROBOSYN_JAX_GPU")
    if gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    import jax

    jax.config.update("jax_cuda_visible_devices", gpu)


def _check_jax_pin() -> None:
    gpu = os.environ.get("ROBOSYN_JAX_GPU")
    if gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("JAX_PLATFORMS") == "cpu":
        return
    import jax

    devices = jax.devices()
    if [device.id for device in devices] != [int(gpu)]:
        print(f"[robosyn_team] warning: JAX devices {devices} do not match ROBOSYN_JAX_GPU={gpu}")


def _use_bundled_openpi_cache(checkpoint_dir: Path) -> None:
    """Load the PaliGemma tokenizer from the checkpoint instead of gs://big_vision.

    The organisers do not guarantee network access at inference; OpenPI only
    downloads the tokenizer when it is missing from OPENPI_DATA_HOME.
    """
    bundled = checkpoint_dir / "openpi_cache"
    if not (bundled / TOKENIZER_RELPATH).is_file():
        return
    current = os.environ.get("OPENPI_DATA_HOME")
    if current and (Path(current).expanduser() / TOKENIZER_RELPATH).is_file():
        return
    os.environ["OPENPI_DATA_HOME"] = str(bundled)


GOLDEN_FILE = "robosyn_golden.npz"


def golden_observation(golden) -> dict:
    """Flat OpenPI observation for the stored self-check input."""
    return {
        "observation/image": golden["images"][0],
        "observation/right_wrist_image": golden["images"][1],
        "observation/left_wrist_image": golden["images"][2],
        "observation/state": golden["state"],
        "prompt": str(golden["prompt"]),
    }


def golden_check(policy, checkpoint_dir: Path) -> tuple[bool, str]:
    """Compare the policy on this machine against the CPU reference stored with the export.

    The evaluation machine (RTX 5090, Blackwell) differs from the GPUs we validated on,
    and JAX/XLA on sm_120 has known crash and miscompilation bugs.  A deterministic
    input (fixed images, state, prompt and flow-matching noise) is run once at load
    time; a crash or a deviation beyond the stored tolerance means the GPU path cannot
    be trusted.  It also moves the one-time JIT compilation out of the first episode.
    """
    path = checkpoint_dir / GOLDEN_FILE
    if not path.is_file():
        return True, "no golden reference in the checkpoint (check skipped)"
    golden = np.load(path, allow_pickle=False)
    # The reference uses OpenPI's default sampler settings; a policy configured with
    # other ones (e.g. more flow-matching steps) is checked at the defaults.
    saved_kwargs = getattr(policy, "_sample_kwargs", None)
    if saved_kwargs:
        policy._sample_kwargs = {}
    try:
        actions = np.asarray(policy.infer(golden_observation(golden), noise=golden["noise"])["actions"])
    except Exception as error:  # noqa: BLE001 - any backend failure means "do not trust this device"
        return False, f"inference raised {type(error).__name__}: {error}"
    finally:
        if saved_kwargs:
            policy._sample_kwargs = saved_kwargs
    reference = golden["actions"]
    if actions.shape != reference.shape or not np.all(np.isfinite(actions)):
        return False, f"bad output shape/values {actions.shape}"
    deviation = float(np.max(np.abs(actions - reference)))
    tolerance = float(golden["tolerance"])
    return deviation <= tolerance, f"max |a - a_ref| = {deviation:.4f} (tolerance {tolerance:.4f})"


def _jax_initialised() -> bool:
    if "jax" not in sys.modules:
        return False
    try:
        from jax._src import xla_bridge

        return bool(getattr(xla_bridge, "_backends", {}))
    except Exception:  # noqa: BLE001
        return True


def choose_device(checkpoint_dir: Path, train_config_name: str) -> str:
    """Decide between the accelerator and CPU BEFORE JAX initialises in this process.

    ROBOSYN_PI05_DEVICE=gpu|cpu forces a choice.  In the default "auto" mode the golden
    self-check runs in a short-lived subprocess on the accelerator: a crash there (e.g.
    an unsupported GPU architecture) cannot take the evaluator down, and a failed or
    deviating result selects CPU, which is then set up with JAX_PLATFORMS=cpu so that
    OpenPI really places the parameters on the host.
    """
    mode = os.environ.get("ROBOSYN_PI05_DEVICE", "auto").lower()
    if mode in {"gpu", "cpu"}:
        return mode
    if not (checkpoint_dir / GOLDEN_FILE).is_file():
        return "gpu"
    command = [sys.executable, str(Path(__file__).resolve()), "--golden-probe", str(checkpoint_dir), train_config_name]
    env = dict(os.environ, ROBOSYN_PI05_DEVICE="gpu")
    try:
        probe = subprocess.run(command, env=env, capture_output=True, text=True,
                               timeout=float(os.environ.get("ROBOSYN_PI05_PROBE_TIMEOUT", "900")))
    except subprocess.TimeoutExpired:
        print("[robosyn_team] π0.5 accelerator self-check timed out; using CPU")
        return "cpu"
    lines = [line for line in probe.stdout.splitlines() if "[robosyn_team]" in line or "GOLDEN_PROBE" in line]
    for line in lines[-3:]:
        print(line)
    if probe.returncode == 0 and any("GOLDEN_PROBE_OK" in line for line in lines):
        return "gpu"
    print(f"[robosyn_team] π0.5 accelerator self-check failed (exit {probe.returncode}); using CPU. "
          f"stderr tail: {probe.stderr.strip().splitlines()[-1:] if probe.stderr.strip() else ''}")
    return "cpu"


class PI05Model:
    """Loads a trained OpenPI checkpoint and serves action chunks."""

    def __init__(self, checkpoint_dir: Path, train_config_name: str, pi0_step: int,
                 pytorch_device: str = "cuda", gripper_scale: float = 1.0, flow_steps: int | None = None,
                 noise_temp: float = 1.0, speed: float = 1.0, noise_seed: int = 0, watchdog: dict | None = None,
                 rtc: dict | None = None, vote: dict | None = None):
        checkpoint_dir = Path(checkpoint_dir)
        device = choose_device(checkpoint_dir, train_config_name)
        if device == "cpu":
            if _jax_initialised():
                print("[robosyn_team] warning: JAX already initialised; cannot switch this process to CPU")
            os.environ["JAX_PLATFORMS"] = "cpu"
        _use_bundled_openpi_cache(checkpoint_dir)
        # Import OpenPI before the JAX backend initialises (see _pin_jax_to_gpu).
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi.training import checkpoints as _checkpoints

        if device != "cpu":
            _pin_jax_to_gpu()

        assets_root = checkpoint_dir / "assets"
        norm_files = sorted(assets_root.rglob("norm_stats.json")) if assets_root.is_dir() else []
        if len(norm_files) != 1:
            raise ValueError(
                f"expected exactly one norm_stats.json below {assets_root}, found {len(norm_files)}"
            )
        asset_id = norm_files[0].parent.relative_to(assets_root).as_posix()
        norm_stats = _checkpoints.load_norm_stats(assets_root, asset_id)
        config = _config.get_config(train_config_name)

        # flow_steps: number of flow-matching integration steps per action chunk
        # (OpenPI default 10).  More steps cost inference time only.
        sample_kwargs = {"num_steps": int(flow_steps)} if flow_steps else None
        self.policy = _policy_config.create_trained_policy(
            config, checkpoint_dir, norm_stats=norm_stats, pytorch_device=pytorch_device,
            sample_kwargs=sample_kwargs,
        )
        _check_jax_pin()
        import jax

        self.jax_device = str(jax.devices()[0])
        # Warm-up with the golden input: compiles the sampler before the first timed
        # episode and re-checks the numerics in this process.
        trusted, detail = golden_check(self.policy, checkpoint_dir)
        print(f"[robosyn_team] π0.5 self-check on {self.jax_device}: {'ok' if trusted else 'MISMATCH'}; {detail}")
        if sample_kwargs and (checkpoint_dir / GOLDEN_FILE).is_file():
            golden = np.load(checkpoint_dir / GOLDEN_FILE, allow_pickle=False)
            self.policy.infer(golden_observation(golden))   # compile the configured sampler too
        self.flow_steps = int(flow_steps) if flow_steps else None
        self.train_config_name = train_config_name
        self.checkpoint_dir = checkpoint_dir
        self.pytorch_device = pytorch_device
        self.pi0_step = int(pi0_step)
        # Per-task gripper command scale (action dims 6 and 13).  The organisers
        # announced (2026-09) that table_rearrangement's gripper actions (0-0.05)
        # will be aligned with the other tasks (0-1); if an updated environment
        # interprets the command differently from the training data, the manifest
        # sets usr_args.gripper_scale for that task instead of retraining.
        self.gripper_scale = float(gripper_scale)
        # Optional inference-time knobs (defaults reproduce plain OpenPI sampling and execution):
        # noise_temp scales the variance of the initial flow-matching noise (tau < 1 = less random
        # chunks); speed > 1 replays each chunk faster by resampling it in time (fewer env steps).
        self.noise_temp = float(noise_temp)
        self.speed = float(speed)
        if not (0.0 < self.noise_temp <= 1.0) or not (1.0 <= self.speed <= 2.0):
            raise ValueError(f"noise_temp must be in (0, 1] and speed in [1, 2]: {noise_temp}, {speed}")
        self.noise_shape = (int(config.model.action_horizon), int(config.model.action_dim))
        self._noise_rng = np.random.default_rng(int(noise_seed))
        self.watchdog = Watchdog(**watchdog) if watchdog else None
        # Real-time chunking (Black et al. 2025, training-free): every chunk after the first is guided
        # towards the not-yet-executed tail of the previous chunk, so short execution horizons
        # (pi0_step < 50) replan often without discontinuities between chunks.
        self.rtc = None
        if rtc:
            self.rtc = dict(beta=float(rtc.get("beta", 5.0)), schedule=str(rtc.get("schedule", "exp")))
            _install_rtc_sampler(self.policy, flow_steps)
        self._prev_raw = None
        # Consensus selection (training-free test-time scaling, cf. majority voting over sampled
        # chunks in RoboMonkey / KeyStone): sample ``n`` chunks with independent noise and execute
        # the medoid, i.e. the chunk with the smallest summed distance (mean L2 over its first
        # ``horizon`` actions) to the other candidates.  Off by default; not combined with rtc.
        self.vote = None
        if vote and int(vote.get("n", 1)) > 1:
            self.vote = dict(n=int(vote["n"]), horizon=int(vote.get("horizon", 25)))
        self.instruction = None
        self.observation_window = None
        print(f"[robosyn_team] π0.5 loaded: {checkpoint_dir} (asset {asset_id}, JAX device {self.jax_device})")

    def set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, images, state):
        base, right_wrist, left_wrist = images
        self.observation_window = {
            "observation/image": base,
            "observation/left_wrist_image": left_wrist,
            "observation/right_wrist_image": right_wrist,
            "observation/state": state,
            "prompt": self.instruction,
        }

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError("update_observation_window must be called first")
        if self.rtc is not None:
            horizon, dim = self.noise_shape
            prev = np.zeros((horizon, dim), np.float32)
            weights = np.zeros((horizon,), np.float32)
            if self._prev_raw is not None:
                s = min(self.pi0_step, horizon)
                tail = self._prev_raw[s:]
                prev[: len(tail)] = tail
                prev[len(tail):] = tail[-1] if len(tail) else 0.0
                weights = _prefix_weights(horizon - s, horizon, self.rtc["schedule"])
            self.policy._sample_kwargs.update(prev_chunk=prev, weights=weights, beta=np.float32(self.rtc["beta"]))
        if self.vote is not None and self.rtc is None:
            actions = self._consensus_chunk()
        elif self.noise_temp != 1.0:
            noise = np.sqrt(self.noise_temp) * self._noise_rng.standard_normal(self.noise_shape)
            actions = self.policy.infer(self.observation_window, noise=noise.astype(np.float32))["actions"]
        else:
            actions = self.policy.infer(self.observation_window)["actions"]
        if self.rtc is not None:
            self._prev_raw = np.asarray(self.policy._rtc_last_raw)[0]
        if self.gripper_scale != 1.0:
            actions = np.array(actions, dtype=np.float32, copy=True)
            actions[..., GRIPPER_DIMS] *= self.gripper_scale
        return actions

    def _consensus_chunk(self):
        cands = []
        for _ in range(self.vote["n"]):
            noise = np.sqrt(self.noise_temp) * self._noise_rng.standard_normal(self.noise_shape)
            cands.append(np.asarray(self.policy.infer(self.observation_window, noise=noise.astype(np.float32))["actions"]))
        return cands[consensus_index(cands, self.vote["horizon"])]

    def reset(self):
        self.instruction = None
        self.observation_window = None
        self._prev_raw = None
        if self.watchdog is not None:
            self.watchdog.reset()


class Watchdog:
    """Inference-time failure recovery (no privileged information: only the robot's own joint state
    and the commands already sent).  Three triggers, all computed from proprioception or step counts:
      * missed grasp: a monitored gripper has been commanded closed (< 0.2) for ``closed_steps``
        consecutive steps but its measured opening is below that gripper's threshold, i.e. it
        closed on nothing (holding widths come from the task's demonstrations);
      * stall: the arm joints moved less than ``stall_eps`` over ``stall_chunks`` whole chunks;
      * deadline (``retry_at`` > 0): the episode is still running ``retry_at`` steps into the current
        attempt.  The evaluator ends an episode as soon as the task succeeds, so still being called
        means the attempt failed (only the policy's own step count is used).  The arms are replayed
        back along their own path, every ``rewind_stride``-th command, to the command executed at
        step ``rewind_to`` (grippers held as they are), and the policy attempts the phase again.
    Recovery replays the last ``backoff`` executed commands in reverse (back along the approach
    path; for a missed grasp with that gripper commanded open) and lets the next inference start
    from the new state with fresh noise.  At most ``max_retries`` recoveries per episode."""

    ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

    def __init__(self, grippers: str = "", closed_steps: int = 15, stall_chunks: int = 2,
                 stall_eps: float = 0.02, backoff: int = 25, max_retries: int = 3,
                 retry_at: int = 0, rewind_to: int = 150, rewind_stride: int = 4):
        # grippers: "dim:threshold,..." e.g. "6:0.15,13:0.15"; empty = no missed-grasp check
        self.grippers = {int(d): float(t) for d, t in (x.split(":") for x in str(grippers).split(",") if x)}
        self.closed_steps, self.stall_chunks = int(closed_steps), int(stall_chunks)
        self.stall_eps, self.backoff, self.max_retries = float(stall_eps), int(backoff), int(max_retries)
        self.retry_at, self.rewind_to, self.rewind_stride = int(retry_at), int(rewind_to), max(1, int(rewind_stride))
        self.reset()

    def reset(self):
        trace_dir = os.environ.get("ROBOSYN_WD_TRACE_DIR")
        if trace_dir and getattr(self, "history", None):
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            out = Path(trace_dir) / f"wd_{os.getpid()}_{time.time_ns()}.npz"
            np.savez_compressed(out, actions=np.stack(self.history), states=np.stack(self.states),
                                events=np.array([f"{k}:{g}:{n}" for k, g, n in self.events]))
        self.history = []          # executed commands, oldest first
        self.states = []
        self.closed_run = {d: 0 for d in self.grippers}
        self.chunk_starts = []     # arm joint state at the start of each chunk
        self.retries = 0
        self.events = []
        self.next_retry = self.retry_at   # deadline trigger: step count at which a retry is due

    def on_chunk_start(self, state):
        self.chunk_starts.append(np.asarray(state, dtype=np.float32)[self.ARM_DIMS])

    def after_step(self, action, state):
        """Record one executed step; return the gripper dim of a missed grasp, else None."""
        self.history.append(np.asarray(action, dtype=np.float32))
        self.states.append(np.asarray(state, dtype=np.float32))
        if self.retries >= self.max_retries:
            return None
        for d, thr in self.grippers.items():
            if action[d] < 0.2:
                self.closed_run[d] += 1
                if self.closed_run[d] >= self.closed_steps and state[d] < thr:
                    return d
            else:
                self.closed_run[d] = 0
        return None

    def deadline_due(self):
        return (self.retry_at > 0 and self.retries < self.max_retries
                and len(self.history) >= self.next_retry > self.rewind_to)

    def rewind(self):
        """Replay the commands since step ``rewind_to`` in reverse (every ``rewind_stride``-th, ending
        exactly on the command of step ``rewind_to``), grippers held at their current command."""
        self.retries += 1
        back = list(reversed(self.history[self.rewind_to:]))
        plan = [a.copy() for a in back[self.rewind_stride - 1::self.rewind_stride]]
        if not plan or not np.array_equal(plan[-1], back[-1]):
            plan.append(back[-1].copy())
        held = self.history[-1][GRIPPER_DIMS]
        for a in plan:
            a[GRIPPER_DIMS] = held
        self.next_retry = len(self.history) + len(plan) + (self.retry_at - self.rewind_to)
        for d in self.closed_run:
            self.closed_run[d] = 0
        self.chunk_starts.clear()
        self.events.append(("deadline", None, len(self.history)))
        print(f"[robosyn_team] watchdog: deadline after {len(self.history)} steps -> rewind to step {self.rewind_to} "
              f"in {len(plan)} steps (retry {self.retries}/{self.max_retries}, next check at {self.next_retry})")
        return plan

    def stalled(self, state):
        if self.retries >= self.max_retries or len(self.chunk_starts) <= self.stall_chunks:
            return False
        ref = self.chunk_starts[-1 - self.stall_chunks]
        return float(np.max(np.abs(np.asarray(state, dtype=np.float32)[self.ARM_DIMS] - ref))) < self.stall_eps

    def recovery(self, kind, gripper=None):
        """Reverse replay of recent commands (skipping the stalled stretch for a stall)."""
        self.retries += 1
        hist = self.history
        if kind == "stall":
            hist = hist[: max(0, len(hist) - 50 * self.stall_chunks)] or self.history
        plan = [a.copy() for a in reversed(hist[-self.backoff:])]
        if gripper is not None:
            for a in plan:
                a[gripper] = 1.0
        for d in self.closed_run:
            self.closed_run[d] = 0
        self.chunk_starts.clear()
        self.events.append((kind, gripper, len(self.history)))
        print(f"[robosyn_team] watchdog: {kind}{'' if gripper is None else f' (gripper {gripper})'} after {len(self.history)} steps -> back off {len(plan)} steps (retry {self.retries}/{self.max_retries})")
        return plan


def consensus_index(cands, horizon: int) -> int:
    """Index of the medoid chunk: smallest summed mean-L2 distance (first ``horizon`` steps) to the rest."""
    a = np.stack([np.asarray(c, dtype=np.float32)[:horizon] for c in cands])          # n x h x d
    dist = np.linalg.norm(a[:, None] - a[None, :], axis=-1).mean(-1)                    # n x n
    return int(np.argmin(dist.sum(1)))


def _prefix_weights(end: int, total: int, schedule: str = "exp") -> np.ndarray:
    """RTC soft mask for zero inference delay: weight decays from ~1 at the first new step to 0 at
    ``end`` (= horizon - executed steps); steps beyond the previous chunk get 0."""
    i = np.arange(total, dtype=np.float32)
    if schedule == "ones":
        w = np.ones(total, np.float32)
    else:
        w = np.clip((-1.0 - i) / (end + 1.0) + 1.0, 0.0, 1.0)
        if schedule == "exp":
            w = w * np.expm1(w) / (np.e - 1.0)
    return np.where(i >= end, 0.0, w).astype(np.float32)


def _install_rtc_sampler(policy, flow_steps) -> None:
    """Replace the policy's jitted ``sample_actions`` by a guided version (same inputs plus
    prev_chunk / weights / beta) and keep the raw (normalized, padded) chunk for the next call."""
    import types
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared import nnx_utils

    model = policy._model

    def rtc_sample_actions(self, rng, observation, *, num_steps=10, noise=None, prev_chunk=None,
                           weights=None, beta=5.0):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps            # OpenPI: t = 1 is noise, t = 0 is data
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def velocity(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond])
            return self.action_out_proj(suffix_out[:, -self.action_horizon:])

        y = jnp.asarray(prev_chunk, dtype=noise.dtype)[None]
        w = jnp.asarray(weights, dtype=noise.dtype)[None, :, None]

        def step(carry):
            x_t, time = carry

            def denoise(x):
                v = velocity(x, time)
                return x - time * v, v      # predicted clean chunk at t = 0

            x0, vjp_fn, v_t = jax.vjp(denoise, x_t, has_aux=True)
            correction = vjp_fn((y - x0) * w)[0]
            tau = 1.0 - time                 # PI convention: tau = 0 noise, 1 data
            inv_r2 = (tau ** 2 + (1.0 - tau) ** 2) / jnp.maximum((1.0 - tau) ** 2, 1e-12)
            c = jnp.nan_to_num((1.0 - tau) / tau, posinf=beta)
            gw = jnp.minimum(c * inv_r2, beta)
            v_t = v_t - gw * correction      # OpenPI velocity points from data to noise
            return x_t + dt * v_t, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    bound = types.MethodType(rtc_sample_actions, model)
    jitted = nnx_utils.module_jit(bound)

    def sampler(rng, observation, **kwargs):
        out = jitted(rng, observation, **kwargs)
        policy._rtc_last_raw = out
        return out

    policy._sample_actions = sampler
    policy._sample_kwargs = dict(policy._sample_kwargs)
    if flow_steps:
        policy._sample_kwargs["num_steps"] = int(flow_steps)


def get_model(usr_args, checkpoint_dir: Path):
    train_config = usr_args.get("train_config_name") or "pi05_base_robosynchallenge_full"
    checkpoint_dir = Path(checkpoint_dir)
    # A manifest may point either at the checkpoint itself or at a checkpoint base.
    if not (checkpoint_dir / "params").is_dir() and usr_args.get("model_name"):
        checkpoint_dir = _resolve_checkpoint(
            checkpoint_dir, train_config, usr_args["model_name"], int(usr_args.get("checkpoint_id", 30000))
        )
    device = usr_args.get("pytorch_device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return PI05Model(
        checkpoint_dir,
        train_config_name=train_config,
        pi0_step=int(usr_args.get("pi0_step", usr_args.get("n_action_steps", 50))),
        pytorch_device=str(device),
        gripper_scale=float(usr_args.get("gripper_scale", 1.0)),
        flow_steps=usr_args.get("flow_steps"),
        noise_temp=float(usr_args.get("noise_temp", 1.0)),
        speed=float(usr_args.get("speed", 1.0)),
        noise_seed=int(usr_args.get("noise_seed", 0)),
        watchdog=usr_args.get("watchdog") or None,
        rtc=usr_args.get("rtc") or None,
        vote=usr_args.get("vote") or None,
    )


def resample_chunk(actions, speed: float):
    """Replay an action chunk ``speed`` times faster: linear interpolation of the absolute joint
    targets at t = 0, speed, 2*speed, ... (in source steps).  speed == 1 returns the chunk."""
    actions = np.asarray(actions, dtype=np.float32)
    if speed == 1.0 or len(actions) < 2:
        return actions
    t = np.arange(0.0, len(actions) - 1 + 1e-6, speed)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(actions) - 1)
    w = (t - lo)[:, None].astype(np.float32)
    return (1.0 - w) * actions[lo] + w * actions[hi]


def _as_done(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def eval(env, model, obs):
    """One inference cycle plus open-loop execution of the predicted chunk."""
    from policy.inference_timing import finish_inference, start_inference

    if model.observation_window is None:
        model.set_language(getattr(env, "_current_instruction", None))

    inference_times_s = []
    started_at = start_inference(model.pytorch_device)
    images, state = encode_obs(obs)
    model.update_observation_window(images, state)
    actions = resample_chunk(model.get_action()[: model.pi0_step], getattr(model, "speed", 1.0))
    action_tensors = [format_action(action, env) for action in actions]
    finish_inference(started_at, inference_times_s, model.pytorch_device)

    # Open-loop chunk execution; the next call re-encodes the latest observation,
    # so intermediate frames are not converted.
    wd = getattr(model, "watchdog", None)
    if wd is not None:
        wd.on_chunk_start(state)
    final_obs, info, truncated = obs, None, False
    recovery = None
    for action, action_tensor in zip(actions, action_tensors):
        final_obs, _reward, _terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            return final_obs, info, truncated, inference_times_s
        if _as_done(truncated):
            return final_obs, info, truncated, inference_times_s
        if wd is not None:
            missed = wd.after_step(action, _qpos(final_obs))
            if missed is not None:
                recovery = wd.recovery("missed_grasp", missed)
                break
            if wd.deadline_due():
                recovery = wd.rewind()
                break
    if recovery is None and wd is not None and wd.stalled(_qpos(final_obs)):
        recovery = wd.recovery("stall")
    for action in recovery or ():
        final_obs, _reward, _terminated, truncated, info = env.step(format_action(action, env))
        wd.history.append(action)
        wd.states.append(_qpos(final_obs))
        if env.get_wrapper_attr("is_task_success")() or _as_done(truncated):
            break
    return final_obs, info, truncated, inference_times_s


def _qpos(obs, env_index: int = 0):
    state = _to_numpy(obs["robot"]["qpos"])
    if state.ndim > 1:
        state = state[env_index]
    return np.asarray(state, dtype=np.float32).reshape(-1)


def reset_model(model):
    model.reset()


def _golden_probe(checkpoint_dir: str, train_config_name: str) -> int:
    """Subprocess entry point used by choose_device (accelerator self-check)."""
    checkpoint = Path(checkpoint_dir)
    _use_bundled_openpi_cache(checkpoint)
    from openpi.policies import policy_config as _policy_config   # before the backend initialises
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config

    _pin_jax_to_gpu()

    assets_root = checkpoint / "assets"
    norm_file = sorted(assets_root.rglob("norm_stats.json"))[0]
    norm_stats = _checkpoints.load_norm_stats(assets_root, norm_file.parent.relative_to(assets_root).as_posix())
    policy = _policy_config.create_trained_policy(
        _config.get_config(train_config_name), checkpoint, norm_stats=norm_stats, pytorch_device="cpu")
    import jax

    trusted, detail = golden_check(policy, checkpoint)
    print(f"[robosyn_team] π0.5 accelerator self-check on {jax.devices()[0]}: {'ok' if trusted else 'FAILED'}; {detail}")
    print("GOLDEN_PROBE_OK" if trusted else "GOLDEN_PROBE_FAIL", flush=True)
    return 0 if trusted else 1


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--golden-probe":
    sys.exit(_golden_probe(sys.argv[2], sys.argv[3]))
