from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from arena_humansim.collision import CollisionResolver
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Segments
from arena_humansim.utils.wall_grid import WallGrid, query_walls

_QUERY_CAPACITY = 64
_CONTACT_PASSES = 3


@njit(cache=True)
def _contact(segments: np.ndarray, w: int, x: float, y: float) -> tuple[float, float, float]:
    ax = segments[w, 0]
    ay = segments[w, 1]
    abx = segments[w, 2] - ax
    aby = segments[w, 3] - ay
    ab_sq = max(abx * abx + aby * aby, 1e-12)
    t = min(max(((x - ax) * abx + (y - ay) * aby) / ab_sq, 0.0), 1.0)
    dx = x - (ax + t * abx)
    dy = y - (ay + t * aby)
    dist = math.sqrt(dx * dx + dy * dy)
    safe_dist = dist if dist > 1e-9 else 1e-9
    return dist, dx / safe_dist, dy / safe_dist


@njit(cache=True, parallel=True)
def _resolve_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    segments: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    margin: float,
    offset: np.ndarray,
    theta: np.ndarray,
    corrected: np.ndarray,
) -> None:
    for i in prange(pos.shape[0]):
        corrected[i] = False
        threshold = radius[i] + margin
        buf = np.empty(_QUERY_CAPACITY, dtype=np.int64)
        if offset[i] > 0.0:
            ox = offset[i] * math.cos(theta[i])
            oy = offset[i] * math.sin(theta[i])
            for _ in range(3):
                x = pos[i, 0]
                y = pos[i, 1]
                vx = vel[i, 0]
                vy = vel[i, 1]
                hit = False
                cx = 0.0
                cy = 0.0
                prev_ids = np.empty(0, dtype=np.int64)
                prev_depths = np.empty(0)
                for end in range(2):
                    sign = 2.0 * end - 1.0
                    px = x + sign * ox
                    py = y + sign * oy
                    k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, px, py, threshold, buf)
                    if k > buf.shape[0]:
                        buf = np.empty(k, dtype=np.int64)
                        k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, px, py, threshold, buf)
                    ids = np.empty(k, dtype=np.int64)
                    depths = np.empty(k)
                    n = 0
                    for c in range(k):
                        dist, ux, uy = _contact(segments, buf[c], px, py)
                        if dist < threshold - 1e-9:
                            hit = True
                            depth = threshold - dist
                            push = depth
                            for j in range(prev_ids.shape[0]):
                                if prev_ids[j] == buf[c]:
                                    push = max(depth - prev_depths[j], 0.0)
                                    break
                            cx += ux * push
                            cy += uy * push
                            ids[n] = buf[c]
                            depths[n] = depth
                            n += 1
                            proj = vx * ux + vy * uy
                            if proj < 0:
                                vx -= proj * ux
                                vy -= proj * uy
                    prev_ids = ids[:n]
                    prev_depths = depths[:n]
                if not hit:
                    break
                corrected[i] = True
                pos[i, 0] = x + cx
                pos[i, 1] = y + cy
                vel[i, 0] = vx
                vel[i, 1] = vy
            continue
        for _ in range(3):
            x = pos[i, 0]
            y = pos[i, 1]
            k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, threshold, buf)
            if k > buf.shape[0]:
                buf = np.empty(k, dtype=np.int64)
                k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, threshold, buf)

            hit = False
            cx = 0.0
            cy = 0.0
            for c in range(k):
                dist, ux, uy = _contact(segments, buf[c], x, y)
                if dist < threshold - 1e-9:
                    hit = True
                    cx += ux * (threshold - dist)
                    cy += uy * (threshold - dist)
            if not hit:
                break

            corrected[i] = True
            pos[i, 0] = x + cx
            pos[i, 1] = y + cy

            vx = vel[i, 0]
            vy = vel[i, 1]
            for c in range(k):
                dist, ux, uy = _contact(segments, buf[c], x, y)
                if dist < threshold - 1e-9:
                    proj = vx * ux + vy * uy
                    if proj < 0:
                        vx -= proj * ux
                        vy -= proj * uy
            vel[i, 0] = vx
            vel[i, 1] = vy


@njit(cache=True)
def _nearest_ends(pos: np.ndarray, offset: np.ndarray, theta: np.ndarray, i: int, j: int) -> tuple[float, float]:
    oix = offset[i] * math.cos(theta[i])
    oiy = offset[i] * math.sin(theta[i])
    ojx = offset[j] * math.cos(theta[j])
    ojy = offset[j] * math.sin(theta[j])
    best = math.inf
    bx = 0.0
    by = 0.0
    for a in range(2):
        sa = 2.0 * a - 1.0
        for b in range(2):
            sb = 2.0 * b - 1.0
            dx = pos[i, 0] + sa * oix - pos[j, 0] - sb * ojx
            dy = pos[i, 1] + sa * oiy - pos[j, 1] - sb * ojy
            d_sq = dx * dx + dy * dy
            if d_sq < best:
                best = d_sq
                bx = dx
                by = dy
    return bx, by


@njit(cache=True)
def _separate_kernel(pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, movable: np.ndarray, pairs: np.ndarray, offset: np.ndarray, theta: np.ndarray) -> None:
    for _ in range(_CONTACT_PASSES):
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]
            mi = 1.0 if movable[i] else 0.0
            mj = 1.0 if movable[j] else 0.0
            total = mi + mj
            if total == 0.0:
                continue
            if offset[i] > 0.0 or offset[j] > 0.0:
                dx, dy = _nearest_ends(pos, offset, theta, i, j)
            else:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            pen = radius[i] + radius[j] - d
            if pen <= 0.0:
                continue
            if d > 1e-9:
                nx = dx / d
                ny = dy / d
            else:
                nx = 1.0
                ny = 0.0
            if total == 1.0:
                m = i if mi == 1.0 else j
                s = 1.0 if mi == 1.0 else -1.0
                vn = (vel[m, 0] * nx + vel[m, 1] * ny) * s
                if vn < 0.0:
                    vel[m, 0] -= vn * s * nx
                    vel[m, 1] -= vn * s * ny
                continue
            pos[i, 0] += 0.5 * pen * nx
            pos[i, 1] += 0.5 * pen * ny
            pos[j, 0] -= 0.5 * pen * nx
            pos[j, 1] -= 0.5 * pen * ny
            rv = (vel[i, 0] - vel[j, 0]) * nx + (vel[i, 1] - vel[j, 1]) * ny
            if rv < 0.0:
                vel[i, 0] -= 0.5 * rv * nx
                vel[i, 1] -= 0.5 * rv * ny
                vel[j, 0] += 0.5 * rv * nx
                vel[j, 1] += 0.5 * rv * ny


def _resolve_grid(grid: WallGrid, pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, margin: float, offset: np.ndarray, theta: np.ndarray) -> np.ndarray:
    corrected = np.empty(pos.shape[0], dtype=np.bool_)
    _resolve_kernel(pos, vel, radius, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, margin, offset, theta, corrected)
    return corrected


class WallProjectionResolver(CollisionResolver):
    def __init__(self, margin: float = 0.01):
        self._margin = margin
        self._grid = WallGrid([])
        self._warmup()

    def _warmup(self) -> None:
        grid = WallGrid([((0.0, 0.0), (1.0, 0.0))])
        pairs = np.array([[0, 1]], dtype=np.int64)
        for offset in (0.0, 0.2):
            pos = np.array([[0.1, 0.1]], dtype=np.float64)
            vel = np.array([[-1.0, -1.0]], dtype=np.float64)
            radius = np.array([0.3], dtype=np.float64)
            _resolve_grid(grid, pos, vel, radius, self._margin, np.full(1, offset), np.zeros(1))
            _separate_kernel(np.zeros((2, 2)), np.zeros((2, 2)), np.full(2, 0.3), np.ones(2, dtype=np.bool_), pairs, np.full(2, offset), np.zeros(2))

    def set_walls(self, segments: Segments) -> None:
        self._grid = WallGrid(segments)

    def resolve(self, pool: AgentPool) -> set[int]:
        self._separate(pool)
        return self._project(pool)

    def _separate(self, pool: AgentPool) -> None:
        n = pool.n
        if n < 2:
            return
        pos = pool.pos[:n]
        radius = pool.agent_radius[:n]
        offset = pool.axial_offset[:n]
        pairs = cKDTree(pos).query_pairs(2.0 * float(radius.max()) + 2.0 * float(offset.max()), output_type="ndarray")
        if len(pairs) == 0:
            return
        i, j = pairs[:, 0], pairs[:, 1]
        pairs = pairs[np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) < radius[i] + radius[j] + offset[i] + offset[j]]
        ids = pool.agent_ids[:n][pairs]
        pairs = pairs[np.lexsort((ids.max(axis=1), ids.min(axis=1)))]
        _separate_kernel(pos, pool.vel[:n], radius, pool.policy_idx[:n] != -1, pairs.astype(np.int64), offset, pool.theta[:n])

    def _project(self, pool: AgentPool) -> set[int]:
        n = pool.n
        if n == 0 or self._grid.segments.shape[0] == 0:
            return set()

        corrected = _resolve_grid(self._grid, pool.pos[:n], pool.vel[:n], pool.agent_radius[:n], self._margin, pool.axial_offset[:n], pool.theta[:n])

        if not corrected.any():
            return set()
        return {int(aid) for aid in pool.agent_ids[:n][corrected]}
