from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from arena_humansim.core import interaction_classes
from arena_humansim.core.agents.types import MAX_HARMONICS
from arena_humansim.core.pool import AgentPool, PoolAware

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent

MODE_NORMAL = 0
MODE_ROTATE = 1
MODE_REVERSE = 2

REVERSE_SPEED = 0.3
ROTATE_DONE_RAD = 0.15
COOLDOWN_S = 2.0
RESTALL_AFTER_S = 1.0
IMPROVE_M = 0.05
GOAL_NEAR_M = 0.3
MIN_SPEED_FACTOR = 0.05
_TWO_PI = 2.0 * math.pi
_HARMONIC_K = np.arange(1, MAX_HARMONICS + 1, dtype=np.float64)


class LocomotionExtension(PoolAware):
    """Gait phase, speed and sway profiles, kinematics modes and stall recovery for the pool's active rows."""

    _pool: AgentPool

    def attach(self, pool: AgentPool) -> None:
        self._pool = pool
        self._alloc(pool.capacity)
        pool.register_extension(self)

    def _alloc(self, capacity: int) -> None:
        f64 = np.float64
        self.active = np.zeros(capacity, dtype=np.bool_)
        self.phase = np.zeros(capacity, dtype=f64)
        self.cadence = np.zeros(capacity, dtype=f64)
        self.kinematics = np.zeros(capacity, dtype=np.uint8)
        self.cadence_base = np.zeros(capacity, dtype=f64)
        self.cadence_per_speed = np.zeros(capacity, dtype=f64)
        self.cadence_min = np.zeros(capacity, dtype=f64)
        self.cadence_max = np.zeros(capacity, dtype=f64)
        self.warp_split = np.full(capacity, 0.5, dtype=f64)
        self.speed_amp = np.zeros((capacity, MAX_HARMONICS), dtype=f64)
        self.speed_phase = np.zeros((capacity, MAX_HARMONICS), dtype=f64)
        self.speed_scale = np.zeros(capacity, dtype=f64)
        self.speed_bias = np.zeros(capacity, dtype=f64)
        self.lateral_bias = np.zeros(capacity, dtype=f64)
        self.lateral_amp = np.zeros((capacity, MAX_HARMONICS), dtype=f64)
        self.lateral_phase = np.zeros((capacity, MAX_HARMONICS), dtype=f64)
        self.stall_after = np.zeros(capacity, dtype=f64)
        self.reverse_m = np.zeros(capacity, dtype=f64)
        self.mode = np.zeros(capacity, dtype=np.uint8)
        self.stall_timer = np.zeros(capacity, dtype=f64)
        self.best_goal_dist = np.full(capacity, np.inf, dtype=f64)
        self.goal_seen = np.zeros((capacity, 2), dtype=f64)
        self.reversed_m = np.zeros(capacity, dtype=f64)
        self.cooldown = np.zeros(capacity, dtype=f64)
        self.overlay_stored = np.zeros(capacity, dtype=np.bool_)
        self.overlay_factor = np.ones(capacity, dtype=f64)
        self.overlay_sway = np.zeros((capacity, 2), dtype=f64)
        self.overlay_vel = np.zeros((capacity, 2), dtype=f64)
        self.pre_theta = np.zeros(capacity, dtype=f64)

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.active,
            self.phase,
            self.cadence,
            self.kinematics,
            self.cadence_base,
            self.cadence_per_speed,
            self.cadence_min,
            self.cadence_max,
            self.warp_split,
            self.speed_amp,
            self.speed_phase,
            self.speed_scale,
            self.speed_bias,
            self.lateral_bias,
            self.lateral_amp,
            self.lateral_phase,
            self.stall_after,
            self.reverse_m,
            self.mode,
            self.stall_timer,
            self.best_goal_dist,
            self.goal_seen,
            self.reversed_m,
            self.cooldown,
            self.overlay_stored,
            self.overlay_factor,
            self.overlay_sway,
            self.overlay_vel,
            self.pre_theta,
        )

    def on_pool_grow(self, new_capacity: int, old_capacity: int) -> None:
        old = self._arrays()
        self._alloc(new_capacity)
        for src, dst in zip(old, self._arrays(), strict=True):
            dst[:old_capacity] = src[:old_capacity]

    def on_pool_swap(self, idx: int, last: int) -> None:
        for arr in self._arrays():
            arr[idx] = arr[last]

    def on_pool_add(self, idx: int, agent: BaseAgent) -> None:
        loc = agent.params.locomotion
        self.active[idx] = loc.active
        self.phase[idx] = (agent.state.agent_id % 360) * math.pi / 180.0
        self.cadence[idx] = 0.0
        self.kinematics[idx] = loc.kinematics
        self.cadence_base[idx] = loc.cadence_base
        self.cadence_per_speed[idx] = loc.cadence_per_speed
        self.cadence_min[idx] = loc.cadence_min
        self.cadence_max[idx] = loc.cadence_max
        self.warp_split[idx] = min(max(loc.phase_warp_split, 1e-3), 1.0 - 1e-3)
        self.speed_amp[idx] = 0.0
        self.speed_phase[idx] = 0.0
        for k, (amp, phase) in enumerate(loc.speed_profile):
            self.speed_amp[idx, k] = amp
            self.speed_phase[idx, k] = phase
        self.speed_scale[idx] = loc.speed_amplitude_scale
        self.lateral_amp[idx] = 0.0
        self.lateral_phase[idx] = 0.0
        for k, (amp, phase) in enumerate(loc.lateral_profile):
            self.lateral_amp[idx, k] = amp
            self.lateral_phase[idx, k] = phase
        self.speed_bias[idx] = self._time_mean(idx, self.speed_amp, self.speed_phase)
        self.lateral_bias[idx] = self._time_mean(idx, self.lateral_amp, self.lateral_phase)
        self.stall_after[idx] = loc.recovery_stall_after_s
        self.reverse_m[idx] = loc.recovery_reverse_m
        self.mode[idx] = MODE_NORMAL
        self.stall_timer[idx] = 0.0
        self.best_goal_dist[idx] = np.inf
        self.goal_seen[idx] = np.nan
        self.reversed_m[idx] = 0.0
        self.cooldown[idx] = 0.0
        self.overlay_stored[idx] = False
        self.overlay_factor[idx] = 1.0
        self.overlay_sway[idx] = 0.0
        self.overlay_vel[idx] = 0.0
        pool = self._pool
        self.pre_theta[idx] = pool.theta[idx]
        pool.axial_offset[idx] = max(0.0, loc.footprint_length / 2.0 - agent.params.agent_radius)
        pool.interaction_class[idx] = interaction_classes.index(agent.params.interaction_class)

    def _rows(self) -> np.ndarray:
        return np.flatnonzero(self.active[: self._pool.n])

    def warped_phase(self, rows: np.ndarray) -> np.ndarray:
        """Phase in [0, 2 pi) with the first half-cycle stretched to warp_split of the period."""
        u = np.mod(self.phase[rows], _TWO_PI) / _TWO_PI
        split = self.warp_split[rows]
        first = u < split
        return np.where(first, math.pi * u / split, math.pi + math.pi * (u - split) / (1.0 - split))

    def _profile(self, rows: np.ndarray, amp: np.ndarray, phase: np.ndarray, bias: np.ndarray) -> np.ndarray:
        w = self.warped_phase(rows)
        return np.sum(amp[rows] * np.sin(_HARMONIC_K[None, :] * w[:, None] + phase[rows]), axis=1) - bias[rows]

    def _time_mean(self, idx: int, amp: np.ndarray, phase: np.ndarray) -> float:
        """Mean over one cycle of the warped profile, zero for an unwarped one."""
        lobe = (np.cos(phase[idx]) - np.cos(_HARMONIC_K * math.pi + phase[idx])) / (_HARMONIC_K * math.pi)
        return float((2.0 * self.warp_split[idx] - 1.0) * np.sum(amp[idx] * lobe))

    def heading_owned_mask(self, n: int) -> np.ndarray:
        """Rows over [:n] whose heading must not follow the velocity in the integrator: along_heading rows and rows in recovery."""
        return self.active[:n] & ((self.kinematics[:n] == 1) | (self.mode[:n] != MODE_NORMAL))

    def restore(self, pool: AgentPool) -> None:
        """Take the previous tick's overlay back out of pool.vel and note each active row's heading before the planners run."""
        active = self._rows()
        self.pre_theta[active] = pool.theta[active]
        rows = np.flatnonzero(self.overlay_stored[: pool.n])
        if rows.size == 0:
            return
        vel = pool.vel[rows]
        untouched = np.all(vel == self.overlay_vel[rows], axis=1)
        sway = np.where(untouched[:, None], self.overlay_sway[rows], 0.0)
        pool.vel[rows] = (vel - sway) / self.overlay_factor[rows][:, None]
        self.overlay_stored[rows] = False
        self.overlay_factor[rows] = 1.0
        self.overlay_sway[rows] = 0.0

    def constrain(self, pool: AgentPool, dt: float, provides_heading: np.ndarray) -> None:
        rows = self._rows()
        if rows.size == 0:
            return
        self._recover(pool, rows, dt)
        mode = self.mode[rows]
        theta = pool.theta[rows]
        vel = pool.vel[rows]
        along = (self.kinematics[rows] == 1) & (mode == MODE_NORMAL)
        turning = along & ~provides_heading[rows] & ~pool.robot_halted[rows]  # held still for a robot, it keeps facing its way, not the drift walls and people push it at
        if turning.any():
            t_rows = rows[turning]
            v = pool.vel[t_rows]
            speed = np.hypot(v[:, 0], v[:, 1])
            moving = speed > 1e-9
            target = np.arctan2(v[:, 1], v[:, 0])
            cur = pool.theta[t_rows]
            delta = np.arctan2(np.sin(target - cur), np.cos(target - cur))
            max_d = np.maximum(pool.pivot_angular_velocity[t_rows], speed / np.maximum(pool.min_turning_radius[t_rows], 1e-9)) * dt
            pool.theta[t_rows] = cur + np.where(moving, np.clip(delta, -max_d, max_d), 0.0)
            theta = pool.theta[rows]
        f = np.column_stack([np.cos(theta), np.sin(theta)])
        if along.any():
            a_rows = rows[along]
            forward = np.maximum(np.sum(vel[along] * f[along], axis=1), 0.0)
            pool.vel[a_rows] = forward[:, None] * f[along]
        rotate = mode == MODE_ROTATE
        if rotate.any():
            pool.vel[rows[rotate]] = 0.0
        reverse = mode == MODE_REVERSE
        if reverse.any():
            pool.vel[rows[reverse]] = -REVERSE_SPEED * f[reverse]

    def _recover(self, pool: AgentPool, rows: np.ndarray, dt: float) -> None:
        rec = rows[self.stall_after[rows] > 0.0]
        if rec.size == 0:
            return
        self.cooldown[rec] = np.maximum(self.cooldown[rec] - dt, 0.0)
        goal = pool.goal_pos[rec]
        delta = goal - pool.pos[rec]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        tracked = pool.has_goal[rec] & (pool.desired_vel[rec] > 0.0) & (dist > GOAL_NEAR_M) & ~pool.robot_halted[rec]  # waiting for a robot is not being stuck
        new_goal = np.any(goal != self.goal_seen[rec], axis=1)
        self.goal_seen[rec] = goal
        improved = dist < self.best_goal_dist[rec] - IMPROVE_M
        reset = ~tracked | new_goal | improved
        self.best_goal_dist[rec] = np.where(reset, dist, self.best_goal_dist[rec])
        self.stall_timer[rec] = np.where(reset, 0.0, self.stall_timer[rec] + dt)
        self.mode[rec] = np.where(tracked, self.mode[rec], MODE_NORMAL)
        mode = self.mode[rec]

        in_reverse = mode == MODE_REVERSE
        self.reversed_m[rec] = np.where(in_reverse, self.reversed_m[rec] + REVERSE_SPEED * dt, 0.0)
        reverse_done = in_reverse & (self.reversed_m[rec] >= self.reverse_m[rec])
        mode = np.where(reverse_done, MODE_ROTATE, mode)

        heading = self.pre_theta[rec]
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        err = np.arctan2(np.sin(bearing - heading), np.cos(bearing - heading))
        in_rotate = mode == MODE_ROTATE
        rotate_done = in_rotate & (np.abs(err) < ROTATE_DONE_RAD)
        mode = np.where(rotate_done, MODE_NORMAL, mode)
        self.cooldown[rec] = np.where(rotate_done, COOLDOWN_S, self.cooldown[rec])
        self.stall_timer[rec] = np.where(rotate_done, 0.0, self.stall_timer[rec])

        threshold = np.where(self.cooldown[rec] > 0.0, np.minimum(self.stall_after[rec], RESTALL_AFTER_S), self.stall_after[rec])
        stalled = tracked & (mode == MODE_NORMAL) & ~rotate_done & (self.stall_timer[rec] >= threshold)
        to_reverse = stalled & (self.cooldown[rec] > 0.0) & (self.reverse_m[rec] > 0.0)
        mode = np.where(to_reverse, MODE_REVERSE, np.where(stalled, MODE_ROTATE, mode))
        self.stall_timer[rec] = np.where(stalled, 0.0, self.stall_timer[rec])
        self.reversed_m[rec] = np.where(to_reverse, 0.0, self.reversed_m[rec])
        self.mode[rec] = mode

        rotating = in_rotate | (mode == MODE_ROTATE)
        owned = rotating | (mode == MODE_REVERSE)
        max_d = pool.pivot_angular_velocity[rec] * dt
        pool.theta[rec[owned]] = (heading + np.where(rotating, np.clip(err, -max_d, max_d), 0.0))[owned]

    def overlay(self, pool: AgentPool, dt: float, standing: np.ndarray | None = None) -> None:
        """Advance the gait phase and lay the speed and sway profiles over pool.vel for this tick's integration only, `standing` rows gaiting as at rest and moved unprofiled."""
        rows = self._rows()
        if rows.size == 0:
            return
        vel = pool.vel[rows]
        speed = np.hypot(vel[:, 0], vel[:, 1])
        still = np.zeros(rows.size, dtype=np.bool_) if standing is None else standing[rows]
        cadence = np.clip(self.cadence_base[rows] + self.cadence_per_speed[rows] * np.where(still, 0.0, speed), self.cadence_min[rows], self.cadence_max[rows])
        self.cadence[rows] = cadence
        self.phase[rows] += _TWO_PI * cadence * dt
        normal = (self.mode[rows] == MODE_NORMAL) & ~still
        factor = np.where(normal, np.maximum(1.0 + self.speed_scale[rows] * self._profile(rows, self.speed_amp, self.speed_phase, self.speed_bias), MIN_SPEED_FACTOR), 1.0)
        lateral = np.where(normal & (self.kinematics[rows] == 0), speed * self._profile(rows, self.lateral_amp, self.lateral_phase, self.lateral_bias), 0.0)
        theta = pool.theta[rows]
        sway = lateral[:, None] * np.column_stack([-np.sin(theta), np.cos(theta)])
        changed = (factor != 1.0) | np.any(sway != 0.0, axis=1)
        if not changed.any():
            return
        rows = rows[changed]
        out = vel[changed] * factor[changed][:, None] + sway[changed]
        pool.vel[rows] = out
        self.overlay_stored[rows] = True
        self.overlay_factor[rows] = factor[changed]
        self.overlay_sway[rows] = sway[changed]
        self.overlay_vel[rows] = out

    def frame_arrays(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        act = self.active[rows]
        return np.where(act, self.phase[rows], 0.0), np.where(act, self.cadence[rows], 0.0)
