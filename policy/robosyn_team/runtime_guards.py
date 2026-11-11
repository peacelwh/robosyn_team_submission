"""Execution guards for the real-robot final (RoboSynChallenge 2026, finals FAQ Q13-Q17).

On the real robot every command is sent to the arms through the SDK as it is, at 10 Hz, with no
interpolation, rate limit or filtering, and an episode is cut at 1000 commands.  Rules, corrections and
smoothing are allowed inside policy/ as long as they only read the observation.  This module adds them
for the π0.5 runtime; everything here uses only the commands the policy has sent itself and the joint
state in the observation.

Nothing is active unless a task's ``usr_args`` carry a ``runtime`` dict, so manifests without it behave
exactly as before.  Keys (all optional):

  step_budget       commands the evaluator allows per episode (real robot: 1000).  Only used to decide
                    whether a recovery is still worth starting.
  min_attempt       commands a fresh attempt needs after a recovery motion (default 60); a recovery is
                    started only if ``steps + recovery + min_attempt <= step_budget``.
  seam_steps        chunk-seam smoothing: the jump between the last command sent and the first command of
                    a new chunk is spread evenly over this many steps (0 = off).  The new chunk is anchored
                    to the measured state, which trails the last command on the real arm, so without
                    blending every seam starts with a step backwards.
  seam_dims         "arm" (default; gripper commands stay as predicted) or "all".
  max_joint_step    per-step limit on |command - previous command| for the 12 arm joints, rad (0 = off).
                    Limits the step, not the joint value: J5/J6 keep the real robot's range (FAQ Q21).
  max_gripper_step  the same for the two gripper commands (0 = off).
  joint_limits      absolute clip of arm joint commands to the evaluation robot's soft limits, a dict
                    {joint: limit} (absent = off).  ``joint`` is a joint name (waist, shoulder, elbow,
                    forearm_roll, wrist_angle, wrist_rotate -> both arms) or an action index 0-13; ``limit``
                    is L for [-L, L] or [lo, hi], rad.  Meant for J5/J6 with the real robot's own limits
                    (the released demos reach J6 = 2.767 rad, so the simulation URDF's 120 deg must NOT be
                    used here, FAQ Q21): it only removes commands the arm could not execute anyway.
  stall             stall watchdog, a dict (absent = off):
      steps          window length in commands; a stall is a window in which every arm joint command stayed
                     within ``eps`` rad (max - min) and both gripper commands within ``grip_eps`` times
                     the largest value that gripper was commanded so far
      eps            default 0.01 rad
      grip_eps       default 0.1
      backoff        commands to replay back along the path executed before the stall (default 40)
      stride         replay every stride-th of them, so the way back is faster (default 2)
      max_retries    default 2
      min_motion     the arms must have moved this far from the first state (max over joints, rad) before
                     a stall counts (default 0.2)
      release_guard  a gripper release (its command rose again by ``release_frac`` of its maximum after
                     having dropped by as much) within this many commands before the stall, or during it,
                     means the task has probably just been finished (place-and-release tasks end with a
                     release): the watchdog then stays idle for the rest of the episode instead of moving
                     the arm back through the placed objects (0 = off, default 30)
      release_frac   default 0.25

The recovery motion goes back along the arm's own executed path (the way it came, so it is collision-free
up to object motion) with both gripper commands held, then the policy attempts the phase again from the
new observation with fresh flow-matching noise.
"""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPER_DIMS = [6, 13]
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]   # arm joint i -> dims i, i + 7

STALL_DEFAULTS = dict(steps=50, eps=0.01, grip_eps=0.1, backoff=40, stride=2, max_retries=2,
                      min_motion=0.2, release_guard=30, release_frac=0.25)


def _parse_joint_limits(spec, action_dim):
    """[(dims, lo, hi)] from {joint name | action index: L | [lo, hi]}; anything else fails loudly."""
    out = []
    for key, value in dict(spec).items():
        name = str(key)
        if name in JOINT_NAMES:
            i = JOINT_NAMES.index(name)
            dims = [d for d in (i, i + 7) if d < action_dim]
        elif name.isdigit() and int(name) < action_dim:
            dims = [int(name)]
        else:
            raise ValueError(f"joint_limits: unknown joint {key!r} (use one of {JOINT_NAMES} or an action index)")
        if np.isscalar(value):
            lo, hi = -float(value), float(value)
        elif len(value) == 2:
            lo, hi = float(value[0]), float(value[1])
        else:
            raise ValueError(f"joint_limits[{key!r}]: expected L or [lo, hi], got {value!r}")
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"joint_limits[{key!r}]: need L > 0 or lo < hi, got {value!r}")
        out.append((dims, lo, hi))
    return out


class _ReleaseDetector:
    """Peak -> valley -> peak detector on one gripper command, thresholds relative to its largest value."""

    def __init__(self, frac: float):
        self.frac = float(frac)
        self.gmax = 0.0
        self.low = False
        self.peak = 0.0
        self.valley = 0.0

    def update(self, value: float) -> bool:
        value = float(value)
        self.gmax = max(self.gmax, value)
        step = self.frac * max(self.gmax, 1e-6)
        if not self.low:
            self.peak = max(self.peak, value)
            if value < self.peak - step:
                self.low, self.valley = True, value
            return False
        self.valley = min(self.valley, value)
        if value > self.valley + step:
            self.low, self.peak = False, value
            return True
        return False


class RuntimeGuard:
    def __init__(self, step_budget: int = 1000, min_attempt: int = 60, seam_steps: int = 0,
                 seam_dims: str = "arm", max_joint_step: float = 0.0, max_gripper_step: float = 0.0,
                 stall: dict | None = None, joint_limits: dict | None = None, action_dim: int = 14):
        self.step_budget, self.min_attempt = int(step_budget), int(min_attempt)
        self.seam_steps = int(seam_steps)
        if seam_dims not in ("arm", "all"):
            raise ValueError(f"seam_dims must be 'arm' or 'all': {seam_dims!r}")
        self.seam_dims = ARM_DIMS if seam_dims == "arm" else list(range(action_dim))
        self.step_limit = np.full(action_dim, np.inf, np.float32)
        if float(max_joint_step) > 0:
            self.step_limit[ARM_DIMS] = float(max_joint_step)
        if float(max_gripper_step) > 0:
            self.step_limit[GRIPPER_DIMS] = float(max_gripper_step)
        self.joint_lo = np.full(action_dim, -np.inf, np.float32)
        self.joint_hi = np.full(action_dim, np.inf, np.float32)
        for dims, lo, hi in _parse_joint_limits(joint_limits or {}, action_dim):
            self.joint_lo[dims], self.joint_hi[dims] = lo, hi
        self.stall = None
        if stall:
            unknown = set(stall) - set(STALL_DEFAULTS)
            if unknown:
                raise ValueError(f"unknown stall keys {sorted(unknown)}")
            self.stall = {k: type(v)(stall.get(k, v)) for k, v in STALL_DEFAULTS.items()}
        self.action_dim = int(action_dim)
        self.episode = 0
        self._log_path = os.environ.get("ROBOSYN_RT_LOG")
        self._clear()

    # ------------------------------------------------------------------ episode bookkeeping
    def _clear(self):
        self.steps = 0
        self.last_cmd = None            # last command sent (after all guards)
        self.history = []               # every command sent this episode
        self.floor = 0                  # a recovery never replays commands before this index
        self.q0 = None
        self.max_motion = 0.0
        self.window = deque(maxlen=self.stall["steps"] if self.stall else 1)
        frac = self.stall["release_frac"] if self.stall else 0.25
        self.releases = {d: _ReleaseDetector(frac) for d in GRIPPER_DIMS}
        self.release_steps = []
        self.retries = 0
        self.idle = False               # watchdog disarmed for the rest of the episode
        self.stats = dict(chunks=0, seam_jump_max=0.0, limited_steps=0, raw_step_max=0.0, clipped_steps=0,
                          clip_max=0.0, stalls=0, recoveries=0, holds=0, budget_denied=0, events=[])

    def reset(self):
        if self.steps:
            summary = dict(episode=self.episode, steps=self.steps, **self.stats)
            print(f"[robosyn_team] runtime: {json.dumps(summary)}")
            if self._log_path:
                try:
                    with open(self._log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(summary) + "\n")
                except OSError:
                    pass
            self.episode += 1
        self._clear()

    def budget_allows(self, cost: int) -> bool:
        return self.steps + int(cost) + self.min_attempt <= self.step_budget

    # ------------------------------------------------------------------ command shaping
    def _reference(self, state):
        """Previous command for the seam/limiter; at episode start the measured arm joints (grippers free)."""
        if self.last_cmd is not None:
            return self.last_cmd.copy(), np.ones(self.action_dim, bool)
        ref = np.zeros(self.action_dim, np.float32)
        valid = np.zeros(self.action_dim, bool)
        if state is not None:
            state = np.asarray(state, np.float32).reshape(-1)
            if state.shape[0] >= self.action_dim:
                ref[ARM_DIMS] = state[ARM_DIMS]
                valid[ARM_DIMS] = True
        return ref, valid

    def prepare(self, actions, state=None, seam: bool = True) -> np.ndarray:
        """Shape the commands about to be sent: seam blending (new chunks only), the per-step limit and the
        joint limits (a clip, hence a contraction: the per-step limit still holds after it).

        Pure with respect to the guard's state, so it can run inside the timed inference block; the
        commands actually sent are registered afterwards with ``record``."""
        actions = np.array(actions, dtype=np.float32, copy=True).reshape(len(actions), -1)
        if not len(actions):
            return actions
        ref, valid = self._reference(state)
        if seam:
            self.stats["chunks"] += 1
            dims = [d for d in self.seam_dims if valid[d]]
            if dims:
                jump = actions[0, dims] - ref[dims]
                self.stats["seam_jump_max"] = max(self.stats["seam_jump_max"],
                                                  round(float(np.max(np.abs(jump))), 4))
                if self.seam_steps > 0:
                    k = min(self.seam_steps, len(actions))
                    remaining = 1.0 - np.arange(1, k + 1, dtype=np.float32) / (k + 1)
                    actions[:k, dims] -= remaining[:, None] * jump[None, :]
        if np.isfinite(self.step_limit).any():
            prev, have = ref, valid.copy()
            limited = 0
            for i in range(len(actions)):
                delta = actions[i] - prev
                bound = np.where(have, self.step_limit, np.inf)
                clipped = np.clip(delta, -bound, bound)
                self.stats["raw_step_max"] = max(self.stats["raw_step_max"],
                                                 round(float(np.max(np.abs(np.where(have, delta, 0.0))[ARM_DIMS])), 4))
                if np.any(clipped != delta):
                    limited += 1
                    actions[i] = prev + clipped
                prev, have = actions[i], np.ones_like(have)
            self.stats["limited_steps"] += limited
        if np.isfinite(self.joint_hi).any() or np.isfinite(self.joint_lo).any():
            clipped = np.clip(actions, self.joint_lo, self.joint_hi)
            over = np.abs(clipped - actions)
            if over.any():
                self.stats["clipped_steps"] += int(np.count_nonzero(over.max(1)))
                self.stats["clip_max"] = max(self.stats["clip_max"], round(float(over.max()), 4))
                actions = clipped
        return actions

    # ------------------------------------------------------------------ after every env.step
    def record(self, action, state=None):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.steps += 1
        self.last_cmd = action.copy()
        self.history.append(self.last_cmd)
        if self.stall is None:
            return
        if self.q0 is None:
            self.q0 = (np.asarray(state, np.float32).reshape(-1)[ARM_DIMS]
                       if state is not None else action[ARM_DIMS].copy())
        self.max_motion = max(self.max_motion, float(np.max(np.abs(action[ARM_DIMS] - self.q0))))
        for d, detector in self.releases.items():
            if detector.update(action[d]):
                self.release_steps.append(self.steps)
        self.window.append(action)

    def _stalled(self) -> bool:
        if len(self.window) < self.window.maxlen:
            return False
        win = np.stack(self.window)
        spread = win.max(0) - win.min(0)
        if np.any(spread[ARM_DIMS] >= self.stall["eps"]):
            return False
        for d in GRIPPER_DIMS:
            if spread[d] >= self.stall["grip_eps"] * max(self.releases[d].gmax, 1e-3):
                return False
        return True

    def check_stall(self):
        """Recovery plan (commands to replay) if the arms have stalled and a retry is allowed, else None."""
        if self.stall is None or self.idle or not self._stalled():
            return None
        cfg = self.stall
        start = self.steps - cfg["steps"]
        self.stats["stalls"] += 1
        if self.max_motion < cfg["min_motion"]:
            self.window.clear()             # never really started: nothing to go back along
            self._event("stall_no_motion")
            return None
        if cfg["release_guard"] > 0 and any(s > start - cfg["release_guard"] for s in self.release_steps):
            self.idle = True
            self.stats["holds"] += 1
            self._event("hold_after_release")
            return None
        if self.retries >= cfg["max_retries"]:
            self.idle = True
            self._event("max_retries")
            return None
        path = self.history[self.floor:max(self.floor, start)]
        back = list(reversed(path[-cfg["backoff"]:]))
        if not back:
            self.window.clear()
            self._event("stall_no_path")
            return None
        stride = max(1, cfg["stride"])
        plan = [a.copy() for a in back[stride - 1::stride]]
        if not plan or not np.array_equal(plan[-1], back[-1]):
            plan.append(back[-1].copy())
        held = self.last_cmd[GRIPPER_DIMS]
        for a in plan:
            a[GRIPPER_DIMS] = held
        if not self.budget_allows(len(plan)):
            self.idle = True
            self.stats["budget_denied"] += 1
            self._event("budget")
            return None
        self.retries += 1
        self.stats["recoveries"] += 1
        self.floor = self.steps + len(plan)       # the next recovery does not undo this one
        self.window.clear()
        self._event("stall_rewind", len(plan))
        print(f"[robosyn_team] runtime: stall for {cfg['steps']} commands at step {self.steps} -> back along "
              f"the path in {len(plan)} commands (retry {self.retries}/{cfg['max_retries']})")
        return plan

    def _event(self, kind, n=None):
        self.stats["events"].append([kind, self.steps] + ([n] if n is not None else []))
