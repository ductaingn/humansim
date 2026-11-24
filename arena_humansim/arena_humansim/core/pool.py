from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.utils.types import Pose2D

_DEFAULT_CAPACITY = 512

KIND_HUMAN = 0
KIND_ROBOT = 1

# Robot sound follows arena_auditory's drivetrain (sources/drivetrain/spec.py): the motor level grows with speed over 1 m/s and stops
# below 0.02 m/s, and a level falls 6 dB per doubling of distance, so a robot is heard out to a distance in proportion to its speed.
ROBOT_SOUND_REF_SPEED = 1.0
ROBOT_SOUND_MIN_SPEED = 0.02


def human_mask(pool: AgentPool) -> np.ndarray:
    return pool.kind[: pool.n] == KIND_HUMAN


def is_human(pool: AgentPool, idx: int) -> bool:
    return int(pool.kind[idx]) == KIND_HUMAN


class PoolAware:
    def attach(self, pool: AgentPool) -> None:
        pass

    def on_pool_grow(self, new_capacity: int, old_capacity: int) -> None:
        pass

    def on_pool_add(self, idx: int, agent: BaseAgent) -> None:
        pass

    def on_pool_update(self, idx: int, agent: BaseAgent) -> None:
        """The agent at `idx` has new params (`AgentPool.update_agent`); re-read what you cache from them."""
        pass

    def on_pool_swap(self, idx: int, last: int) -> None:
        pass

    def on_pool_reset(self) -> None:
        pass


class AgentPool:
    def __init__(self, capacity: int = _DEFAULT_CAPACITY):
        self.capacity = capacity
        self.n: int = 0
        self._id_to_idx: dict[int, int] = {}
        self._extensions: list[PoolAware] = []
        self._late_attach: bool = False

        self.agent_ids = np.zeros(capacity, dtype=np.int32)
        self.pos = np.zeros((capacity, 2), dtype=np.float64)
        self.vel = np.zeros((capacity, 2), dtype=np.float64)
        self.prev_vel = np.zeros((capacity, 2), dtype=np.float64)
        self.theta = np.zeros(capacity, dtype=np.float64)
        self.desired_vel = np.zeros(capacity, dtype=np.float64)

        self.agent_radius = np.zeros(capacity, dtype=np.float64)
        self.max_velocity = np.zeros(capacity, dtype=np.float64)
        self.max_acceleration = np.zeros(capacity, dtype=np.float64)
        self.max_deceleration = np.zeros(capacity, dtype=np.float64)
        self.min_turning_radius = np.zeros(capacity, dtype=np.float64)
        self.pivot_angular_velocity = np.zeros(capacity, dtype=np.float64)

        self.vision_range = np.zeros(capacity, dtype=np.float64)
        self.vision_fov = np.zeros(capacity, dtype=np.float64)
        self.proximity_sense = np.zeros(capacity, dtype=np.float64)
        self.vision_occlusion = np.ones(capacity, dtype=np.bool_)
        self.robot_range_offset = np.zeros(capacity, dtype=np.float64)
        self.robot_fov_offset = np.zeros(capacity, dtype=np.float64)
        self.robot_repulsion_gain = np.ones(capacity, dtype=np.float64)
        self.robot_reaction_gain = np.ones(capacity, dtype=np.float64)
        self.robot_blind = np.zeros(capacity, dtype=np.bool_)
        self.robot_halt = np.zeros(capacity, dtype=np.bool_)
        self.robot_halted = np.zeros(capacity, dtype=np.bool_)  # set each tick by the local planner: held still for a robot ahead
        self.robot_yields = np.ones(capacity, dtype=np.bool_)
        self.hearing_range = np.zeros(capacity, dtype=np.float64)
        self.cane_reach = np.zeros(capacity, dtype=np.float64)
        self.cane_width = np.zeros(capacity, dtype=np.float64)
        self.detection_delay = np.zeros(capacity, dtype=np.float64)
        # Scales the distance every robot is heard at: 0 is a silent robot, 1 the robot the agent types' hearing_range assumes at 1 m/s.
        self.robot_loudness = 1.0
        # (observer id, robot id) -> perception clock when the observer started sensing that robot, for detection_delay.
        self.robot_sensed_since: dict[tuple[int, int], float] = {}
        self.perception_clock = 0.0

        self.goal_pos = np.zeros((capacity, 2), dtype=np.float64)
        self.has_goal = np.zeros(capacity, dtype=np.bool_)
        self.terminal_pos = np.zeros((capacity, 2), dtype=np.float64)
        self.has_terminal = np.zeros(capacity, dtype=np.bool_)
        self.goal_theta = np.zeros(capacity, dtype=np.float64)
        self.has_goal_theta = np.zeros(capacity, dtype=np.bool_)
        self.latched = np.zeros(capacity, dtype=np.bool_)

        self.kind = np.zeros(capacity, dtype=np.uint8)
        self.interaction_class = np.zeros(capacity, dtype=np.int32)
        self.axial_offset = np.zeros(capacity, dtype=np.float64)
        self.policy_idx = np.full(capacity, -1, dtype=np.int32)
        self.animation_state = np.zeros(capacity, dtype=np.uint8)

        self.neighbor_indptr = np.zeros(1, dtype=np.int32)
        self.neighbor_indices = np.empty(0, dtype=np.int32)
        self.ignored_pairs = np.empty((0, 2), dtype=np.int64)  # row pairs contact separation leaves alone this tick (core/carry.py)

    def register_extension(self, ext: PoolAware) -> None:
        assert self.n == 0 or self._late_attach, "register_extension must be called before agents are added"
        self._extensions.append(ext)

    def unregister_extension(self, ext: PoolAware) -> None:
        if ext in self._extensions:
            self._extensions.remove(ext)

    def attach_late(self, ext: PoolAware, agents: Iterable[BaseAgent]) -> None:
        """Attach a PoolAware after agents exist and back-fill its rows for every pooled agent."""
        self._late_attach = True
        try:
            ext.attach(self)
        finally:
            self._late_attach = False
        if ext not in self._extensions:
            return
        for i, agent in enumerate(agents):
            if i >= self.n:
                break
            ext.on_pool_add(i, agent)

    def idx(self, agent_id: int) -> int:
        return self._id_to_idx[agent_id]

    def add_agent(self, agent: BaseAgent) -> int:
        i = self.n
        if i >= self.capacity:
            self._grow(i + 1)
        self.n = i + 1

        aid = agent.state.agent_id
        self._id_to_idx[aid] = i
        self.agent_ids[i] = aid

        self.pos[i, 0] = agent.state.pose.x
        self.pos[i, 1] = agent.state.pose.y
        self.theta[i] = agent.state.pose.theta
        self.vel[i, 0] = agent.state.velocity[0]
        self.vel[i, 1] = agent.state.velocity[1]
        self.agent_radius[i] = agent.params.agent_radius
        self._write_params(i, agent)

        self.has_goal[i] = False
        self.has_terminal[i] = False
        self.has_goal_theta[i] = False
        self.latched[i] = False
        self.prev_vel[i] = self.vel[i]
        self.robot_halted[i] = False
        self.kind[i] = 0
        self.interaction_class[i] = 0
        self.axial_offset[i] = 0.0
        self.policy_idx[i] = -1
        self.animation_state[i] = agent.state.animation_state

        for ext in self._extensions:
            ext.on_pool_add(i, agent)
        return i

    def update_agent(self, agent: BaseAgent) -> int:
        """Rewrite a pooled agent's param rows in place; position, velocity, goals and the posture radius stay."""
        i = self._id_to_idx[agent.state.agent_id]
        self._write_params(i, agent)
        for ext in self._extensions:
            ext.on_pool_update(i, agent)
        return i

    def _write_params(self, i: int, agent: BaseAgent) -> None:
        self.desired_vel[i] = agent.state.desired_velocity
        p = agent.params
        self.max_velocity[i] = p.max_velocity
        self.max_acceleration[i] = p.max_acceleration
        self.max_deceleration[i] = p.max_deceleration
        self.min_turning_radius[i] = p.min_turning_radius
        self.pivot_angular_velocity[i] = p.pivot_angular_velocity
        perc = p.perception
        self.vision_range[i] = perc.vision_range
        self.vision_fov[i] = perc.vision_fov
        self.proximity_sense[i] = perc.proximity_sense
        self.vision_occlusion[i] = perc.vision_occlusion
        self.robot_range_offset[i] = perc.robot_range_offset
        self.robot_fov_offset[i] = perc.robot_fov_offset
        self.robot_repulsion_gain[i] = perc.robot_repulsion_gain
        self.robot_reaction_gain[i] = perc.robot_reaction_gain
        self.robot_blind[i] = perc.robot_blind
        self.robot_halt[i] = perc.robot_halt
        self.robot_yields[i] = perc.robot_yields
        self.hearing_range[i] = perc.hearing_range
        self.cane_reach[i] = perc.cane_reach
        self.cane_width[i] = perc.cane_width
        self.detection_delay[i] = perc.detection_delay

    def swap_remove(self, agent_id: int) -> int | None:
        idx = self._id_to_idx.pop(agent_id)
        last = self.n - 1
        if idx != last:
            swapped_id = int(self.agent_ids[last])
            self._id_to_idx[swapped_id] = idx
            for arr in (
                self.agent_ids,
                self.theta,
                self.desired_vel,
                self.agent_radius,
                self.max_velocity,
                self.max_acceleration,
                self.max_deceleration,
                self.min_turning_radius,
                self.pivot_angular_velocity,
                self.vision_range,
                self.vision_fov,
                self.proximity_sense,
                self.vision_occlusion,
                self.robot_range_offset,
                self.robot_fov_offset,
                self.robot_repulsion_gain,
                self.robot_reaction_gain,
                self.robot_blind,
                self.robot_halt,
                self.robot_halted,
                self.robot_yields,
                self.hearing_range,
                self.cane_reach,
                self.cane_width,
                self.detection_delay,
                self.has_goal,
                self.has_terminal,
                self.goal_theta,
                self.has_goal_theta,
                self.latched,
                self.kind,
                self.interaction_class,
                self.axial_offset,
                self.policy_idx,
                self.animation_state,
            ):
                arr[idx] = arr[last]
            for arr in (self.pos, self.vel, self.prev_vel, self.goal_pos, self.terminal_pos):
                arr[idx] = arr[last]
            for ext in self._extensions:
                ext.on_pool_swap(idx, last)
        else:
            swapped_id = None
        self.n = last
        return swapped_id

    def robot_hearing_scale(self, rows: np.ndarray) -> np.ndarray:
        """Factor on hearing_range out to which each row is heard: robot_loudness times speed over 1 m/s for robots, 0 below 0.02 m/s, 1 for people."""
        speed = np.hypot(self.vel[rows, 0], self.vel[rows, 1])
        robot = np.where(speed < ROBOT_SOUND_MIN_SPEED, 0.0, speed / ROBOT_SOUND_REF_SPEED) * self.robot_loudness
        return np.where(self.kind[rows] == KIND_ROBOT, robot, 1.0)

    def reset(self) -> None:
        self.n = 0
        self._id_to_idx.clear()
        self.robot_sensed_since.clear()
        for ext in self._extensions:
            ext.on_pool_reset()

    def sync_back(self, agents: Iterable[BaseAgent]) -> None:
        for i, agent in enumerate(agents):
            agent.state.pose.x = float(self.pos[i, 0])
            agent.state.pose.y = float(self.pos[i, 1])
            agent.state.pose.theta = float(self.theta[i])
            agent.state.velocity = (float(self.vel[i, 0]), float(self.vel[i, 1]))
            agent.state.animation_state = int(self.animation_state[i])

    def set_goals(self, goals: dict[int, Pose2D]) -> None:
        self.has_goal[: self.n] = False
        for aid, goal in goals.items():
            idx = self._id_to_idx.get(aid)
            if idx is None:
                continue
            self.goal_pos[idx, 0] = goal.x
            self.goal_pos[idx, 1] = goal.y
            self.has_goal[idx] = True

    def set_terminals(self, terminals: dict[int, Pose2D]) -> None:
        self.has_terminal[: self.n] = False
        for aid, pose in terminals.items():
            idx = self._id_to_idx.get(aid)
            if idx is None:
                continue
            self.terminal_pos[idx, 0] = pose.x
            self.terminal_pos[idx, 1] = pose.y
            self.has_terminal[idx] = True

    def set_heading_goals(self, headings: dict[int, float]) -> None:
        self.has_goal_theta[: self.n] = False
        for aid, theta in headings.items():
            idx = self._id_to_idx.get(aid)
            if idx is None:
                continue
            self.goal_theta[idx] = theta
            self.has_goal_theta[idx] = True

    def store_prev_vel(self) -> None:
        n = self.n
        self.prev_vel[:n] = self.vel[:n]

    def set_neighbor_csr(self, indptr: np.ndarray, indices: np.ndarray) -> None:
        self.neighbor_indptr = indptr
        self.neighbor_indices = indices

    def visible_agent_ids(self, agent_id: int) -> set[int]:
        idx = self._id_to_idx.get(agent_id)
        if idx is None or idx + 1 >= len(self.neighbor_indptr):
            return set()
        start = int(self.neighbor_indptr[idx])
        stop = int(self.neighbor_indptr[idx + 1])
        if stop <= start:
            return set()
        return {int(self.agent_ids[i]) for i in self.neighbor_indices[start:stop]}

    def _grow(self, min_capacity: int) -> None:
        new_cap = max(min_capacity, self.capacity * 2)
        old = self.capacity

        def _resize_1d(arr: np.ndarray) -> np.ndarray:
            out = np.zeros(new_cap, dtype=arr.dtype)
            out[:old] = arr[:old]
            return out

        def _resize_2d(arr: np.ndarray) -> np.ndarray:
            out = np.zeros((new_cap, arr.shape[1]), dtype=arr.dtype)
            out[:old] = arr[:old]
            return out

        self.agent_ids = _resize_1d(self.agent_ids)
        self.pos = _resize_2d(self.pos)
        self.vel = _resize_2d(self.vel)
        self.prev_vel = _resize_2d(self.prev_vel)
        self.theta = _resize_1d(self.theta)
        self.desired_vel = _resize_1d(self.desired_vel)
        self.agent_radius = _resize_1d(self.agent_radius)
        self.max_velocity = _resize_1d(self.max_velocity)
        self.max_acceleration = _resize_1d(self.max_acceleration)
        self.max_deceleration = _resize_1d(self.max_deceleration)
        self.min_turning_radius = _resize_1d(self.min_turning_radius)
        self.pivot_angular_velocity = _resize_1d(self.pivot_angular_velocity)
        self.vision_range = _resize_1d(self.vision_range)
        self.vision_fov = _resize_1d(self.vision_fov)
        self.proximity_sense = _resize_1d(self.proximity_sense)
        occ_new = np.ones(new_cap, dtype=self.vision_occlusion.dtype)
        occ_new[:old] = self.vision_occlusion[:old]
        self.vision_occlusion = occ_new
        self.robot_range_offset = _resize_1d(self.robot_range_offset)
        self.robot_fov_offset = _resize_1d(self.robot_fov_offset)
        self.robot_repulsion_gain = _resize_1d(self.robot_repulsion_gain)
        self.robot_reaction_gain = _resize_1d(self.robot_reaction_gain)
        self.robot_blind = _resize_1d(self.robot_blind)
        self.robot_halt = _resize_1d(self.robot_halt)
        self.robot_halted = _resize_1d(self.robot_halted)
        yields_new = np.ones(new_cap, dtype=self.robot_yields.dtype)
        yields_new[:old] = self.robot_yields[:old]
        self.robot_yields = yields_new
        self.hearing_range = _resize_1d(self.hearing_range)
        self.cane_reach = _resize_1d(self.cane_reach)
        self.cane_width = _resize_1d(self.cane_width)
        self.detection_delay = _resize_1d(self.detection_delay)
        self.goal_pos = _resize_2d(self.goal_pos)
        self.has_goal = _resize_1d(self.has_goal)
        self.terminal_pos = _resize_2d(self.terminal_pos)
        self.has_terminal = _resize_1d(self.has_terminal)
        self.goal_theta = _resize_1d(self.goal_theta)
        self.has_goal_theta = _resize_1d(self.has_goal_theta)
        self.latched = _resize_1d(self.latched)
        kind_new = np.zeros(new_cap, dtype=self.kind.dtype)
        kind_new[:old] = self.kind[:old]
        self.kind = kind_new
        self.interaction_class = _resize_1d(self.interaction_class)
        self.axial_offset = _resize_1d(self.axial_offset)
        pidx_new = np.full(new_cap, -1, dtype=self.policy_idx.dtype)
        pidx_new[:old] = self.policy_idx[:old]
        self.policy_idx = pidx_new
        anim_new = np.zeros(new_cap, dtype=self.animation_state.dtype)
        anim_new[:old] = self.animation_state[:old]
        self.animation_state = anim_new
        self.capacity = new_cap

        for ext in self._extensions:
            ext.on_pool_grow(new_cap, old)
