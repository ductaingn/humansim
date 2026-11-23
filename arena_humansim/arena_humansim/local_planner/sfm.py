from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core import interaction_classes
from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.utils.types import Pose2D, Segments
from arena_humansim.utils.wall_grid import WallGrid, query_walls

from . import LocalPlanner

if TYPE_CHECKING:
    from std_msgs.msg import ColorRGBA

    from arena_humansim.core.pool import AgentPool
    from arena_humansim.core.viz import MarkerPublisher, MarkerView

_EPS = 1e-6

_DEFAULT_WALL_REPULSION_STRENGTH = 3.0
_DEFAULT_WALL_REPULSION_RANGE = 0.1

_DEFAULT_CLASS_GAINS: dict[tuple[str, str], tuple[float, float]] = {
    ("human", "robot"): (1.5, 1.3),
    ("human", "wheelchair"): (1.3, 1.4),
    ("wheelchair", "robot"): (1.5, 1.3),
}

# robot_response "halt": share of the desired speed kept while a perceived robot is beside or behind (assumed, not measured).
HALT_SLOW_FACTOR = 0.5

_WALL_CUTOFF_RANGES = 28.0
_WALL_QUERY_CAPACITY = 64


def _resize_1d(arr: np.ndarray, new_capacity: int, old_capacity: int) -> np.ndarray:
    out = np.zeros(new_capacity, dtype=arr.dtype)
    out[:old_capacity] = arr[:old_capacity]
    return out


def _grow_gains(mat: np.ndarray, size: int) -> np.ndarray:
    out = np.ones((size, size), dtype=np.float64)
    k = mat.shape[0]
    out[:k, :k] = mat
    return out


@njit(cache=True)
def _wall_force_at(
    x: float,
    y: float,
    r: float,
    p1: np.ndarray,
    d: np.ndarray,
    len_sq: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    strength: float,
    rng: float,
) -> tuple[float, float]:
    query_r = r + rng * _WALL_CUTOFF_RANGES
    buf = np.empty(_WALL_QUERY_CAPACITY, dtype=np.int64)
    k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, buf)
    if k > buf.shape[0]:
        buf = np.empty(k, dtype=np.int64)
        k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, buf)
    fx = 0.0
    fy = 0.0
    for c in range(k):
        w = buf[c]
        dx = d[w, 0]
        dy = d[w, 1]
        t = ((x - p1[w, 0]) * dx + (y - p1[w, 1]) * dy) / max(len_sq[w], _EPS)
        t = min(max(t, 0.0), 1.0)
        ox = x - (p1[w, 0] + t * dx)
        oy = y - (p1[w, 1] + t * dy)
        dist = max(math.hypot(ox, oy), _EPS)
        mag = strength * math.exp((r - dist) / rng)
        fx += mag * (ox / dist)
        fy += mag * (oy / dist)
    return fx, fy


@njit(cache=True, parallel=True)
def _wall_forces_kernel(
    pos: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
    axial_offset: np.ndarray,
    p1: np.ndarray,
    d: np.ndarray,
    len_sq: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    strength: float,
    rng: float,
    out: np.ndarray,
) -> None:
    for i in prange(pos.shape[0]):
        x = pos[i, 0]
        y = pos[i, 1]
        r = radius[i]
        off = axial_offset[i]
        if off > 0.0:
            ax = off * math.cos(theta[i])
            ay = off * math.sin(theta[i])
            f1x, f1y = _wall_force_at(x + ax, y + ay, r, p1, d, len_sq, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, strength, rng)
            f2x, f2y = _wall_force_at(x - ax, y - ay, r, p1, d, len_sq, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, strength, rng)
            out[i, 0] = f1x + f2x
            out[i, 1] = f1y + f2y
        else:
            query_r = r + rng * _WALL_CUTOFF_RANGES
            buf = np.empty(_WALL_QUERY_CAPACITY, dtype=np.int64)
            k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, buf)
            if k > buf.shape[0]:
                buf = np.empty(k, dtype=np.int64)
                k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, buf)
            fx = 0.0
            fy = 0.0
            for c in range(k):
                w = buf[c]
                dx = d[w, 0]
                dy = d[w, 1]
                t = ((x - p1[w, 0]) * dx + (y - p1[w, 1]) * dy) / max(len_sq[w], _EPS)
                t = min(max(t, 0.0), 1.0)
                ox = x - (p1[w, 0] + t * dx)
                oy = y - (p1[w, 1] + t * dy)
                dist = max(math.hypot(ox, oy), _EPS)
                mag = strength * math.exp((r - dist) / rng)
                fx += mag * (ox / dist)
                fy += mag * (oy / dist)
            out[i, 0] = fx
            out[i, 1] = fy


def _closest_disk_diff(pos: np.ndarray, theta: np.ndarray, offsets: np.ndarray, pair_obs: np.ndarray, pair_nbr: np.ndarray) -> np.ndarray:
    axis = offsets[:, None] * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    obs_centers = np.stack((pos[pair_obs] + axis[pair_obs], pos[pair_obs] - axis[pair_obs]), axis=1)
    nbr_centers = np.stack((pos[pair_nbr] + axis[pair_nbr], pos[pair_nbr] - axis[pair_nbr]), axis=1)
    cand = obs_centers[:, :, None, :] - nbr_centers[:, None, :, :]
    cand = cand.reshape(len(pair_obs), 4, 2)
    best = np.argmin(np.hypot(cand[:, :, 0], cand[:, :, 1]), axis=1)
    return cand[np.arange(len(pair_obs)), best]


class SFMPlanner(LocalPlanner):
    supports_pool: bool = True

    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "relaxation_time": ParamDist(0.5, 0.05),
        "repulsion_strength": ParamDist(2.1, 0.2),
        "repulsion_range": ParamDist(0.3, 0.03),
        "anisotropy": ParamDist(0.5, 0.0),
    }

    def __init__(
        self,
        wall_repulsion_strength: float = _DEFAULT_WALL_REPULSION_STRENGTH,
        wall_repulsion_range: float = _DEFAULT_WALL_REPULSION_RANGE,
    ):
        self.wall_repulsion_strength = wall_repulsion_strength
        self.wall_repulsion_range = wall_repulsion_range
        self._wall_segments: Segments = []
        self._wall_segments_np: np.ndarray = np.empty((0, 2, 2), dtype=np.float64)
        self._wall_p1: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._wall_d: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._wall_len_sq: np.ndarray = np.empty(0, dtype=np.float64)
        self._wall_grid = WallGrid(np.empty((0, 4), dtype=np.float64))
        self._last_forces: dict[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {}
        self._last_force_arrays: tuple | None = None
        self._last_agents: Sequence[BaseAgent] | None = None

        self._gain_strength_scale = np.ones((0, 0), dtype=np.float64)
        self._gain_range_scale = np.ones((0, 0), dtype=np.float64)
        self._ensure_gain_classes(interaction_classes.ROBOT + 1)

        self._relaxation_time = np.zeros(0, dtype=np.float64)
        self._repulsion_strength = np.zeros(0, dtype=np.float64)
        self._repulsion_range = np.zeros(0, dtype=np.float64)
        self._anisotropy = np.zeros(0, dtype=np.float64)
        self._warmup()

    def _warmup(self) -> None:
        grid = WallGrid(np.array([[0.0, 0.0, 1.0, 0.0]]))
        p1 = np.ascontiguousarray(grid.segments[:, :2])
        d = np.ascontiguousarray(grid.segments[:, 2:] - p1)
        self._run_wall_kernel(np.zeros((2, 2), dtype=np.float64), np.full(2, 0.3), np.zeros(2), np.array([0.0, 0.5]), grid, p1, d, np.sum(d**2, axis=1))

    def _ensure_gain_classes(self, count: int) -> None:
        old = self._gain_strength_scale.shape[0]
        count = max(count, interaction_classes.count())
        if count <= old:
            return
        self._gain_strength_scale = _grow_gains(self._gain_strength_scale, count)
        self._gain_range_scale = _grow_gains(self._gain_range_scale, count)
        names = {interaction_classes.name(i): i for i in range(min(count, interaction_classes.count()))}
        for (a, b), (s, r) in _DEFAULT_CLASS_GAINS.items():
            i = names.get(a)
            j = names.get(b)
            if i is None or j is None or (i < old and j < old):
                continue
            self._gain_strength_scale[i, j] = s
            self._gain_range_scale[i, j] = r

    def _run_wall_kernel(self, pos: np.ndarray, radius: np.ndarray, theta: np.ndarray, axial_offset: np.ndarray, grid: WallGrid, p1: np.ndarray, d: np.ndarray, len_sq: np.ndarray) -> np.ndarray:
        out = np.empty((pos.shape[0], 2), dtype=np.float64)
        _wall_forces_kernel(
            np.ascontiguousarray(pos, dtype=np.float64),
            np.ascontiguousarray(radius, dtype=np.float64),
            np.ascontiguousarray(theta, dtype=np.float64),
            np.ascontiguousarray(axial_offset, dtype=np.float64),
            p1,
            d,
            len_sq,
            grid.cell_start,
            grid.cell_walls,
            grid.wall_cx0,
            grid.wall_cy0,
            grid.origin_x,
            grid.origin_y,
            grid.cell,
            grid.nx,
            grid.ny,
            float(self.wall_repulsion_strength),
            float(self.wall_repulsion_range),
            out,
        )
        return out

    def attach(self, pool: AgentPool) -> None:
        self._allocate_soa(pool.capacity)
        pool.register_extension(self)

    def _allocate_soa(self, cap: int) -> None:
        self._relaxation_time = np.zeros(cap, dtype=np.float64)
        self._repulsion_strength = np.zeros(cap, dtype=np.float64)
        self._repulsion_range = np.zeros(cap, dtype=np.float64)
        self._anisotropy = np.zeros(cap, dtype=np.float64)

    def on_pool_grow(self, new_capacity: int, old_capacity: int) -> None:
        self._relaxation_time = _resize_1d(self._relaxation_time, new_capacity, old_capacity)
        self._repulsion_strength = _resize_1d(self._repulsion_strength, new_capacity, old_capacity)
        self._repulsion_range = _resize_1d(self._repulsion_range, new_capacity, old_capacity)
        self._anisotropy = _resize_1d(self._anisotropy, new_capacity, old_capacity)

    def _param(self, lp: dict, key: str) -> float:
        try:
            return float(lp[key])
        except (KeyError, TypeError):
            return float(self.PARAM_DEFAULTS[key].mean)

    def on_pool_add(self, idx: int, agent: BaseAgent) -> None:
        lp = agent.params.local_planner_params
        self._relaxation_time[idx] = self._param(lp, "relaxation_time")
        self._repulsion_strength[idx] = self._param(lp, "repulsion_strength")
        self._repulsion_range[idx] = self._param(lp, "repulsion_range")
        self._anisotropy[idx] = self._param(lp, "anisotropy")

    def on_pool_update(self, idx: int, agent: BaseAgent) -> None:
        self.on_pool_add(idx, agent)

    def on_pool_swap(self, idx: int, last: int) -> None:
        self._relaxation_time[idx] = self._relaxation_time[last]
        self._repulsion_strength[idx] = self._repulsion_strength[last]
        self._repulsion_range[idx] = self._repulsion_range[last]
        self._anisotropy[idx] = self._anisotropy[last]

    def on_pool_reset(self) -> None:
        pass

    def apply_policy_params(self, params_json: str) -> None:
        if not params_json:
            return
        try:
            blob = json.loads(params_json)
        except (ValueError, TypeError):
            return
        if not isinstance(blob, dict):
            return
        gains = blob.get("kind_gains")
        if not isinstance(gains, dict):
            return
        for key, entry in gains.items():
            if not isinstance(entry, dict) or "_" not in key:
                continue
            a, b = key.split("_", 1)
            i = interaction_classes.index(a.lower())
            j = interaction_classes.index(b.lower())
            self._ensure_gain_classes(max(i, j) + 1)
            s = entry.get("strength_scale")
            r = entry.get("range_scale")
            if isinstance(s, (int, float)):
                self._gain_strength_scale[i, j] = float(s)
            if isinstance(r, (int, float)):
                self._gain_range_scale[i, j] = float(r)

    def set_walls(self, segments: Segments) -> None:
        self._wall_segments = list(segments)
        if segments:
            self._wall_segments_np = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
        else:
            self._wall_segments_np = np.empty((0, 2, 2), dtype=np.float64)
        self._wall_p1 = np.ascontiguousarray(self._wall_segments_np[:, 0, :])
        self._wall_d = np.ascontiguousarray(self._wall_segments_np[:, 1, :] - self._wall_p1)
        self._wall_len_sq = np.sum(self._wall_d**2, axis=1)
        self._wall_grid = WallGrid(self._wall_segments_np.reshape(-1, 4))
        self._logger.info(f"Loaded {len(segments)} wall segment(s)")

    def _compute_forces_pool(self, pool: AgentPool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = pool.n
        pos = pool.pos[:n]
        vel = pool.vel[:n]
        goal = pool.goal_pos[:n]
        has_goal = pool.has_goal[:n]

        desired_v = pool.desired_vel[:n]
        relax = self._relaxation_time[:n]
        rep_str = self._repulsion_strength[:n]
        rep_rng = self._repulsion_range[:n]
        radii = pool.agent_radius[:n]

        if relax.shape[0] != n:
            raise RuntimeError(f"{type(self).__name__}: parameter arrays hold {relax.shape[0]} rows for {n} pooled agents, this planner was never attached to the pool (AgentPool.attach_late)")

        d_goal = goal - pos
        dist_goal = np.hypot(d_goal[:, 0], d_goal[:, 1])[:, None]
        dist_goal_safe = np.maximum(dist_goal, _EPS)
        e_goal = d_goal / dist_goal_safe

        speed_scale = np.ones(n, dtype=np.float64)
        f_rep = np.zeros((n, 2), dtype=np.float64)
        indptr = pool.neighbor_indptr
        indices = pool.neighbor_indices

        aniso = self._anisotropy[:n]

        if len(indices) > 0:
            pair_obs = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
            pair_nbr = indices

            offsets = pool.axial_offset[:n]
            if np.any(offsets > 0.0):
                diff = _closest_disk_diff(pos, pool.theta[:n], offsets, pair_obs, pair_nbr)
            else:
                diff = pos[pair_obs] - pos[pair_nbr]
            dists = np.hypot(diff[:, 0], diff[:, 1])
            dists = np.maximum(dists, _EPS)
            normals = diff / dists[:, None]

            cls_arr = pool.interaction_class[:n]
            self._ensure_gain_classes(int(cls_arr.max()) + 1)
            n_cls = self._gain_strength_scale.shape[0]
            obs_cls = np.clip(cls_arr[pair_obs].astype(np.int64), 0, n_cls - 1)
            nbr_cls = np.clip(cls_arr[pair_nbr].astype(np.int64), 0, n_cls - 1)
            s_scale = self._gain_strength_scale[obs_cls, nbr_cls]
            r_scale = self._gain_range_scale[obs_cls, nbr_cls]

            r_ij = radii[pair_obs] + radii[pair_nbr]
            eff_strength = rep_str[pair_obs] * s_scale
            eff_range = rep_rng[pair_obs] * r_scale
            nbr_is_robot = pool.kind[pair_nbr] == KIND_ROBOT
            if np.any(nbr_is_robot):
                eff_strength = eff_strength.copy()
                eff_range = eff_range.copy()
                robot_pairs = pair_obs[nbr_is_robot]
                eff_strength[nbr_is_robot] *= pool.robot_repulsion_gain[robot_pairs]
                eff_range[nbr_is_robot] *= pool.robot_reaction_gain[robot_pairs]
                halting = nbr_is_robot & pool.robot_halt[pair_obs]
                if np.any(halting):
                    eff_strength[halting] = 0.0  # no steering around the robot
                    obs = pair_obs[halting]
                    heading = pool.theta[obs]
                    ahead = normals[halting, 0] * np.cos(heading) + normals[halting, 1] * np.sin(heading) < 0.0
                    np.minimum.at(speed_scale, obs, np.where(ahead, 0.0, HALT_SLOW_FACTOR))
            eff_range = np.maximum(eff_range, 1e-4)
            magnitudes = eff_strength * np.exp((r_ij - dists) / eff_range)

            lam = aniso[pair_obs]
            cos_phi = np.sum(-normals * e_goal[pair_obs], axis=1)
            w = lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)
            magnitudes *= w

            pair_forces = magnitudes[:, None] * normals
            np.add.at(f_rep, pair_obs, pair_forces)

        f_att = ((desired_v * speed_scale)[:, None] * e_goal - vel) / relax[:, None]
        at_goal = (dist_goal[:, 0] < _EPS) | ~has_goal
        f_att[at_goal] = 0.0

        f_wall = self._compute_wall_forces_vectorized(pos, radii, pool.theta[:n], pool.axial_offset[:n])

        return f_att, f_rep, f_wall, at_goal

    def _stash_force_arrays(self, pool: AgentPool, n: int, f_att: np.ndarray, f_rep: np.ndarray, f_wall: np.ndarray) -> None:
        self._last_force_arrays = (
            pool.agent_ids[:n].copy(),
            pool.pos[:n].copy(),
            f_att.copy(),
            f_rep.copy(),
            f_wall.copy(),
        )

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            self._last_forces = {}
            self._last_force_arrays = None
            return

        f_att, f_rep, f_wall, at_goal = self._compute_forces_pool(pool)
        vel = pool.vel[:n]
        max_v = pool.max_velocity[:n]

        total_f = f_att + f_rep + f_wall
        new_vel = vel + total_f * dt

        speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
        too_fast = speed > max_v
        if np.any(too_fast):
            new_vel[too_fast] *= (max_v[too_fast] / speed[too_fast])[:, None]

        new_vel[at_goal] = 0.0
        pool.vel[:n] = new_vel

        if store_forces:
            self._stash_force_arrays(pool, n, f_att, f_rep, f_wall)
        else:
            self._last_force_arrays = None

    def _compute_wall_forces_vectorized(
        self,
        agent_pos: np.ndarray,
        agent_radius: np.ndarray,
        theta: np.ndarray | None = None,
        axial_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        n: int = agent_pos.shape[0]
        if self._wall_segments_np.shape[0] == 0:
            return np.zeros((n, 2), dtype=np.float64)
        if theta is None:
            theta = np.zeros(n, dtype=np.float64)
        if axial_offset is None:
            axial_offset = np.zeros(n, dtype=np.float64)

        return self._run_wall_kernel(agent_pos, agent_radius, theta, axial_offset, self._wall_grid, self._wall_p1, self._wall_d, self._wall_len_sq)

    def _compute_forces_scalar(
        self,
        agent: BaseAgent,
        goal: Pose2D,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        params = agent.params
        belief = agent.belief

        lp = params.local_planner_params
        agent_radius: float = params.agent_radius

        cur_x, cur_y = agent.state.pose.x, agent.state.pose.y
        cur_vx, cur_vy = agent.state.velocity

        dx_goal = goal.x - cur_x
        dy_goal = goal.y - cur_y
        dist_goal = np.hypot(dx_goal, dy_goal)
        if dist_goal < _EPS:
            return None

        e_goal_x = dx_goal / dist_goal
        e_goal_y = dy_goal / dist_goal

        speed_scale = 1.0
        f_rep_x = 0.0
        f_rep_y = 0.0
        if belief is not None:
            neighbors = [ag for ag in belief.observed_agents if ag.agent_id != agent.state.agent_id]
            if neighbors:
                other_pos = np.empty((len(neighbors), 2), dtype=np.float64)
                nbr_cls = np.empty(len(neighbors), dtype=np.int64)
                for i, ag in enumerate(neighbors):
                    other_pos[i, 0] = ag.pose.x
                    other_pos[i, 1] = ag.pose.y
                    nbr_cls[i] = ag.kind

                diff = np.array([cur_x, cur_y]) - other_pos
                dists = np.maximum(np.linalg.norm(diff, axis=1), _EPS)
                normals = diff / dists[:, np.newaxis]

                obs_cls = int(agent.state.kind)
                self._ensure_gain_classes(max(obs_cls, int(nbr_cls.max())) + 1)
                n_cls = self._gain_strength_scale.shape[0]
                obs_cls = min(max(obs_cls, 0), n_cls - 1)
                nbr_cls = np.clip(nbr_cls, 0, n_cls - 1)
                s_scale = self._gain_strength_scale[obs_cls, nbr_cls]
                r_scale = self._gain_range_scale[obs_cls, nbr_cls]

                r_ij = 2.0 * agent_radius
                eff_s = lp["repulsion_strength"] * s_scale
                eff_r = lp["repulsion_range"] * r_scale
                rep_gain = agent.params.perception.robot_repulsion_gain
                react_gain = agent.params.perception.robot_reaction_gain
                is_robot_nbr = nbr_cls == KIND_ROBOT
                if np.any(is_robot_nbr):
                    eff_s = eff_s.copy()
                    eff_r = eff_r.copy()
                    eff_s[is_robot_nbr] *= rep_gain
                    eff_r[is_robot_nbr] *= react_gain
                    if agent.params.perception.robot_halt:
                        eff_s[is_robot_nbr] = 0.0  # no steering around the robot
                        heading = agent.state.pose.theta
                        ahead = diff[is_robot_nbr, 0] * math.cos(heading) + diff[is_robot_nbr, 1] * math.sin(heading) < 0.0
                        speed_scale = 0.0 if np.any(ahead) else HALT_SLOW_FACTOR
                eff_r = np.maximum(eff_r, 1e-4)
                magnitudes = eff_s * np.exp((r_ij - dists) / eff_r)

                cos_phi = -normals[:, 0] * e_goal_x + -normals[:, 1] * e_goal_y
                w = lp["anisotropy"] + (1.0 - lp["anisotropy"]) * 0.5 * (1.0 + cos_phi)
                magnitudes *= w

                forces = magnitudes[:, np.newaxis] * normals
                f_rep_x = float(forces[:, 0].sum())
                f_rep_y = float(forces[:, 1].sum())

        f_att_x = (params.desired_velocity * speed_scale * e_goal_x - cur_vx) / lp["relaxation_time"]
        f_att_y = (params.desired_velocity * speed_scale * e_goal_y - cur_vy) / lp["relaxation_time"]

        f_wall_x = 0.0
        f_wall_y = 0.0
        if self._wall_segments:
            f_wall_x, f_wall_y = self._compute_wall_forces_scalar(cur_x, cur_y, agent_radius)

        return (f_att_x, f_att_y), (f_rep_x, f_rep_y), (f_wall_x, f_wall_y), (e_goal_x, e_goal_y)

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        self._last_agents = agents
        if not agents:
            self._last_forces = {}
            return {}

        velocities: dict[int, tuple[float, float]] = {}
        last_forces: dict[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {}

        for agent in agents:
            agent_id = agent.state.agent_id
            goal = global_goals.get(agent_id)
            if goal is None:
                velocities[agent_id] = (0.0, 0.0)
                continue

            forces = self._compute_forces_scalar(agent, goal)
            if forces is None:
                velocities[agent_id] = (0.0, 0.0)
                continue
            (f_att_x, f_att_y), (f_rep_x, f_rep_y), (f_wall_x, f_wall_y), _ = forces

            cur_vx, cur_vy = agent.state.velocity
            total_fx = f_att_x + f_rep_x + f_wall_x
            total_fy = f_att_y + f_rep_y + f_wall_y
            new_vx = cur_vx + total_fx * dt
            new_vy = cur_vy + total_fy * dt

            max_velocity = agent.params.max_velocity
            speed = np.hypot(new_vx, new_vy)
            if speed > max_velocity:
                scale = max_velocity / speed
                new_vx *= scale
                new_vy *= scale

            velocities[agent_id] = (float(new_vx), float(new_vy))
            last_forces[agent_id] = (
                (f_att_x, f_att_y),
                (f_rep_x, f_rep_y),
                (f_wall_x, f_wall_y),
            )

        self._last_forces = last_forces
        return velocities

    def publish_markers(self, pub: MarkerPublisher) -> None:
        from visualization_msgs.msg import Marker

        from arena_humansim.core.viz import rgba

        c_goal = rgba(0.2, 0.9, 0.2, 0.7)
        c_social = rgba(1.0, 0.2, 0.2, 0.7)
        c_obstacle = rgba(1.0, 0.6, 0.0, 0.7)
        scale = 0.3

        goal_view = pub.view("f_goal", Marker.ARROW)
        social_view = pub.view("f_social", Marker.ARROW)
        obstacle_view = pub.view("f_obstacle", Marker.ARROW)

        if self._last_force_arrays is not None:
            ids, pos, f_att, f_rep, f_wall = self._last_force_arrays
            for i in range(len(ids)):
                aid = int(ids[i])
                x, y = float(pos[i, 0]), float(pos[i, 1])
                fg = (float(f_att[i, 0]), float(f_att[i, 1]))
                fs = (float(f_rep[i, 0]), float(f_rep[i, 1]))
                fo = (float(f_wall[i, 0]), float(f_wall[i, 1]))
                self._emit_force(goal_view, aid, x, y, fg, scale, c_goal)
                self._emit_force(social_view, aid, x, y, fs, scale, c_social)
                self._emit_force(obstacle_view, aid, x, y, fo, scale, c_obstacle)
        elif self._last_forces and self._last_agents is not None:
            for agent in self._last_agents:
                aid = agent.state.agent_id
                forces = self._last_forces.get(aid)
                if forces is None:
                    continue
                x, y = agent.state.pose.x, agent.state.pose.y
                fg, fs, fo = forces
                self._emit_force(goal_view, aid, x, y, fg, scale, c_goal)
                self._emit_force(social_view, aid, x, y, fs, scale, c_social)
                self._emit_force(obstacle_view, aid, x, y, fo, scale, c_obstacle)

    @staticmethod
    def _emit_force(view: MarkerView, aid: int, x: float, y: float, f: tuple[float, float], scale: float, color: ColorRGBA) -> None:
        from geometry_msgs.msg import Point

        if abs(f[0]) < 1e-4 and abs(f[1]) < 1e-4:
            return
        m, new = view.get(aid)
        if new:
            m.scale.x, m.scale.y, m.scale.z = 0.02, 0.04, 0.04
            m.color = color
            m.points = [Point(), Point()]
        m.points[0].x, m.points[0].y, m.points[0].z = x, y, 0.15  # pyright: ignore[reportIndexIssue]
        m.points[1].x = x + f[0] * scale  # pyright: ignore[reportIndexIssue]
        m.points[1].y = y + f[1] * scale  # pyright: ignore[reportIndexIssue]
        m.points[1].z = 0.15  # pyright: ignore[reportIndexIssue]

    def _compute_wall_forces_scalar(self, px: float, py: float, agent_radius: float) -> tuple[float, float]:
        f_wall_x = 0.0
        f_wall_y = 0.0

        for seg in self._wall_segments:
            (x1, y1), (x2, y2) = seg
            sx, sy = x2 - x1, y2 - y1
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < _EPS:
                cp_x, cp_y = x1, y1
            else:
                t = ((px - x1) * sx + (py - y1) * sy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                cp_x = x1 + t * sx
                cp_y = y1 + t * sy

            dx = px - cp_x
            dy = py - cp_y
            dist = np.hypot(dx, dy)
            if dist < _EPS:
                continue

            nx = dx / dist
            ny = dy / dist

            magnitude = self.wall_repulsion_strength * np.exp((agent_radius - dist) / self.wall_repulsion_range)
            f_wall_x += magnitude * nx
            f_wall_y += magnitude * ny

        return f_wall_x, f_wall_y
