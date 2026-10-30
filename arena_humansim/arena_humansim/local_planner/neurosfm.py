"""NeuroSFM, the neural social force model of HumanoidsBonn/NeuroSFM_Pybullet (MIT, see NEUROSFM_LICENSE): the released force nets run frozen in numpy, net1 for the goal force, net3 for the force from each neighbor, net5 for the force from a robot, after social_gym_v2/envs/pedestrian.py, the gap between two walkers taken between ellipses fitted to the reference male and female human meshes, each walker's body drawn like the reference's random.choice of URDF."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core import interaction_classes
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.pool import AgentPool

from .force import GOAL, SOCIAL, WALL, ForcePlanner, WallGridArgs, near_walls, wall_point

WEIGHTS = Path(__file__).with_name("neurosfm.npz")
CONTACT_STRENGTH = 2.0
CONTACT_RANGE = 0.89
ROBOT_GAIN = 4.0
ROBOT_RANGE = 2.5
MALE_BODY = (0.175, 0.3275)
FEMALE_BODY = (0.1475, 0.3125)
BODY_SEED = 0x5EED
_EPS = 1e-9

_Layers = tuple[tuple[str, str], ...]
_Net = tuple[tuple[tuple[tuple[int, ...], _Layers], ...], _Layers]

NETS: dict[str, _Net] = {
    "net1": ((((0, 1), (("vel_branch.0", "sigmoid"), ("vel_branch.2", ""))), ((2, 3), (("goal_branch.0", "tanh"), ("goal_branch.2", "")))), (("combiner.0", "relu"), ("combiner.2", ""))),
    "net2": ((((0, 1, 2), (("mlp1.0", "sigmoid"), ("mlp2.0", "relu"), ("mlp2.2", "relu"), ("mlp2.4", ""))),), ()),
    "net3": ((((0, 1, 2), (("branch1.0", "relu"), ("branch1.2", ""))), ((0, 3, 4), (("branch2.0", "relu"), ("branch2.2", "")))), (("combiner.0", "relu"), ("combiner.2", "relu"), ("combiner.4", ""))),
    "net4": ((((0, 1), (("vis_branch.0", "relu"), ("vis_branch.2", ""))), ((2, 3), (("att_branch.0", "relu"), ("att_branch.2", "")))), (("combiner.0", "relu"), ("combiner.2", "relu"), ("combiner.4", ""))),
    "net5": ((((0, 1, 2), (("mlp1.0", "sigmoid"), ("mlp2.0", "relu"), ("mlp2.2", "relu"), ("mlp2.4", ""))),), ()),
}


def _activate(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if kind == "tanh":
        return np.tanh(x)
    return x


def _dense(weights: dict[str, np.ndarray], prefix: str, layers: _Layers, x: np.ndarray) -> np.ndarray:
    for name, kind in layers:
        x = _activate(x @ weights[f"{prefix}.{name}.weight"].T + weights[f"{prefix}.{name}.bias"], kind)
    return x


def forward(weights: dict[str, np.ndarray], net: str, x: np.ndarray) -> np.ndarray:
    """One released net on a batch of input rows."""
    branches, combiner = NETS[net]
    hidden = np.concatenate([_dense(weights, net, layers, x[:, list(cols)]) for cols, layers in branches], axis=1)
    return _dense(weights, net, combiner, hidden)


def male(agent_ids: np.ndarray) -> np.ndarray:
    """A fair, fixed draw per agent id from a splitmix64 hash."""
    z = (agent_ids.astype(np.uint64) + np.uint64(BODY_SEED)) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((z ^ (z >> np.uint64(31))) >> np.uint64(63)) == 1


def support(heading: np.ndarray, toward: np.ndarray, is_male: np.ndarray) -> np.ndarray:
    """Extent of each walker's body ellipse along a world direction, for bodies facing their heading."""
    phi = toward - heading
    depth = np.where(is_male, MALE_BODY[0], FEMALE_BODY[0])
    width = np.where(is_male, MALE_BODY[1], FEMALE_BODY[1])
    return np.sqrt((depth * np.cos(phi)) ** 2 + (width * np.sin(phi)) ** 2)


def load_weights(path: Path = WEIGHTS) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float64) for key in data.files}


@njit(cache=True, parallel=True)
def _contact_kernel(pos: np.ndarray, radius: np.ndarray, seg: np.ndarray, grid: WallGridArgs, out: np.ndarray) -> None:
    for i in prange(pos.shape[0]):
        x = pos[i, 0]
        y = pos[i, 1]
        r = radius[i]
        fx = 0.0
        fy = 0.0
        for k in near_walls(grid, x, y, r):
            px, py = wall_point(seg, k, x, y)
            rx = x - px
            ry = y - py
            d = math.sqrt(rx * rx + ry * ry)
            if d < 1e-6 or d >= r:
                continue
            mag = CONTACT_STRENGTH * math.exp((r - d) / CONTACT_RANGE)
            fx += mag * rx / d
            fy += mag * ry / d
        out[i, 0] = fx
        out[i, 1] = fy


class NeuroSFMPlanner(ForcePlanner):
    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "drive_gain": ParamDist(0.7, 0.0, clip_low=0.0),
        "social_gain": ParamDist(1.0, 0.0, clip_low=0.0),
    }

    def __init__(self, wall_grid_cell: float = 1.0) -> None:
        self._weights = load_weights()
        super().__init__(wall_grid_cell)

    def _step(self, pool: AgentPool, params: np.ndarray, dt: float, store_forces: bool) -> None:
        n = pool.n
        pos = np.ascontiguousarray(pool.pos[:n])
        vel = np.ascontiguousarray(pool.prev_vel[:n])
        radius = np.ascontiguousarray(pool.agent_radius[:n])
        to_goal = pool.goal_pos[:n] - pos
        dist = np.hypot(to_goal[:, 0], to_goal[:, 1])
        moving = pool.has_goal[:n] & (dist >= 1e-6)
        e = np.where(moving[:, None], to_goal / np.maximum(dist, _EPS)[:, None], 0.0)
        out = np.zeros((n, 3, 2), dtype=np.float64)
        out[:, GOAL] = params[:, 0:1] * forward(self._weights, "net1", np.concatenate([vel, e], axis=1))

        indptr = pool.neighbor_indptr if pool.neighbor_indptr.shape[0] == n + 1 else np.zeros(n + 1, dtype=np.int32)
        observer = np.repeat(np.arange(n), np.diff(indptr))
        other = pool.neighbor_indices[: indptr[-1]].astype(np.int64)
        robot = pool.interaction_class[:n][other] == interaction_classes.ROBOT
        center = pos[observer] - pos[other]
        center_d = np.hypot(center[:, 0], center[:, 1])
        unit = center / np.maximum(center_d, _EPS)[:, None]

        peds = ~robot & (center_d > 0.0)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.where(speed > 1e-3, np.arctan2(vel[:, 1], vel[:, 0]), pool.theta[:n])
        bearing = np.arctan2(-center[:, 1], -center[:, 0])
        body = male(pool.agent_ids[:n])
        gap = center_d - support(heading[observer], bearing, body[observer]) - support(heading[other], bearing + np.pi, body[other])
        surface = peds & (gap <= CONTACT_RANGE)
        direction = np.where(surface[:, None], unit * gap[:, None], center)
        d = np.hypot(direction[:, 0], direction[:, 1])
        peds &= d > 0.0
        if peds.any():
            u = direction[peds] / d[peds, None]
            rows = np.column_stack([d[peds], u, e[observer[peds]]])
            np.add.at(out[:, SOCIAL], observer[peds], forward(self._weights, "net3", rows))
        out[:, SOCIAL] *= params[:, 1:2]

        near = robot & (center_d > 0.0) & (center_d <= ROBOT_RANGE)
        if near.any():
            rows = np.column_stack([center_d[near], unit[near]])
            np.add.at(out[:, SOCIAL], observer[near], ROBOT_GAIN * forward(self._weights, "net5", rows))

        grid = self._grid
        wall = np.zeros((n, 2), dtype=np.float64)
        _contact_kernel(pos, radius, self._seg, (grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny), wall)
        out[:, WALL] = wall

        new_vel = vel + out.sum(axis=1) * dt
        speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
        max_v = pool.max_velocity[:n]
        too_fast = speed > max_v
        new_vel[too_fast] *= (max_v[too_fast] / speed[too_fast])[:, None]
        new_vel[~moving] = 0.0
        pool.vel[:n] = new_vel
        self._last_force_arrays = (pool.agent_ids[:n].copy(), pool.pos[:n].copy(), out) if store_forces else None
