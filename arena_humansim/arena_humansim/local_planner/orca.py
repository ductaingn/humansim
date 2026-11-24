from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.utils.types import Pose2D, Segments
from arena_humansim.utils.wall_grid import WallGrid, query_walls

from . import LocalPlanner

_RVO_EPS = 1e-5


@njit(cache=True)
def _det(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


@njit(cache=True)
def _dist_sq_point_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    abx = bx - ax
    aby = by - ay
    r = ((cx - ax) * abx + (cy - ay) * aby) / (abx * abx + aby * aby)
    if r < 0.0:
        dx = cx - ax
        dy = cy - ay
    elif r > 1.0:
        dx = cx - bx
        dy = cy - by
    else:
        dx = cx - (ax + r * abx)
        dy = cy - (ay + r * aby)
    return dx * dx + dy * dy


@njit(cache=True)
def _push_line(lines: np.ndarray, n_lines: int, px: float, py: float, dx: float, dy: float) -> int:
    lines[n_lines, 0] = px
    lines[n_lines, 1] = py
    lines[n_lines, 2] = dx
    lines[n_lines, 3] = dy
    return n_lines + 1


@njit(cache=True)
def _linear_program1(lines: np.ndarray, line_no: int, radius: float, opt_x: float, opt_y: float, direction_opt: bool) -> tuple[bool, float, float]:
    px = lines[line_no, 0]
    py = lines[line_no, 1]
    dx = lines[line_no, 2]
    dy = lines[line_no, 3]
    dot = px * dx + py * dy
    discriminant = dot * dot + radius * radius - (px * px + py * py)
    if discriminant < 0.0:
        return False, 0.0, 0.0

    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc

    for i in range(line_no):
        denominator = _det(dx, dy, lines[i, 2], lines[i, 3])
        numerator = _det(lines[i, 2], lines[i, 3], px - lines[i, 0], py - lines[i, 1])
        if abs(denominator) <= _RVO_EPS:
            if numerator < 0.0:
                return False, 0.0, 0.0
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, 0.0, 0.0

    if direction_opt:
        t = t_right if opt_x * dx + opt_y * dy > 0.0 else t_left
    else:
        t = min(max(dx * (opt_x - px) + dy * (opt_y - py), t_left), t_right)
    return True, px + t * dx, py + t * dy


@njit(cache=True)
def _linear_program2(lines: np.ndarray, n_lines: int, radius: float, opt_x: float, opt_y: float, direction_opt: bool) -> tuple[int, float, float]:
    opt_sq = opt_x * opt_x + opt_y * opt_y
    if direction_opt:
        rx = opt_x * radius
        ry = opt_y * radius
    elif opt_sq > radius * radius:
        opt_len = math.sqrt(opt_sq)
        rx = opt_x / opt_len * radius
        ry = opt_y / opt_len * radius
    else:
        rx = opt_x
        ry = opt_y

    for i in range(n_lines):
        if _det(lines[i, 2], lines[i, 3], lines[i, 0] - rx, lines[i, 1] - ry) > 0.0:
            ok, nx, ny = _linear_program1(lines, i, radius, opt_x, opt_y, direction_opt)
            if not ok:
                return i, rx, ry
            rx = nx
            ry = ny
    return n_lines, rx, ry


@njit(cache=True)
def _linear_program3(lines: np.ndarray, n_lines: int, n_obst_lines: int, begin_line: int, radius: float, rx: float, ry: float) -> tuple[float, float]:
    proj = np.empty((n_lines, 4), dtype=np.float64)
    distance = 0.0
    for i in range(begin_line, n_lines):
        pix = lines[i, 0]
        piy = lines[i, 1]
        dix = lines[i, 2]
        diy = lines[i, 3]
        if _det(dix, diy, pix - rx, piy - ry) <= distance:
            continue

        proj[:n_obst_lines] = lines[:n_obst_lines]
        n_proj = n_obst_lines
        for j in range(n_obst_lines, i):
            pjx = lines[j, 0]
            pjy = lines[j, 1]
            djx = lines[j, 2]
            djy = lines[j, 3]
            determinant = _det(dix, diy, djx, djy)
            if abs(determinant) <= _RVO_EPS:
                if dix * djx + diy * djy > 0.0:
                    continue
                qx = 0.5 * (pix + pjx)
                qy = 0.5 * (piy + pjy)
            else:
                s = _det(djx, djy, pix - pjx, piy - pjy) / determinant
                qx = pix + s * dix
                qy = piy + s * diy
            ex = djx - dix
            ey = djy - diy
            e_len = math.sqrt(ex * ex + ey * ey)
            n_proj = _push_line(proj, n_proj, qx, qy, ex / e_len, ey / e_len)

        fail, nx, ny = _linear_program2(proj, n_proj, radius, -diy, dix, True)
        if fail >= n_proj:
            rx = nx
            ry = ny
        distance = _det(dix, diy, pix - rx, piy - ry)
    return rx, ry


@njit(cache=True)
def _obstacle_line(lines: np.ndarray, n_lines: int, edge: np.ndarray, x: float, y: float, vx: float, vy: float, ra: float, inv_t: float) -> int:
    ax = edge[0]
    ay = edge[1]
    bx = edge[2]
    by = edge[3]
    ux = edge[4]
    uy = edge[5]
    r1x = ax - x
    r1y = ay - y
    r2x = bx - x
    r2y = by - y

    for j in range(n_lines):
        c1 = _det(inv_t * r1x - lines[j, 0], inv_t * r1y - lines[j, 1], lines[j, 2], lines[j, 3]) - inv_t * ra
        c2 = _det(inv_t * r2x - lines[j, 0], inv_t * r2y - lines[j, 1], lines[j, 2], lines[j, 3]) - inv_t * ra
        if c1 >= -_RVO_EPS and c2 >= -_RVO_EPS:
            return n_lines

    dist_sq1 = r1x * r1x + r1y * r1y
    dist_sq2 = r2x * r2x + r2y * r2y
    r_sq = ra * ra
    ovx = bx - ax
    ovy = by - ay
    s = (-r1x * ovx - r1y * ovy) / (ovx * ovx + ovy * ovy)
    dlx = -r1x - s * ovx
    dly = -r1y - s * ovy
    dist_sq_line = dlx * dlx + dly * dly

    if s < 0.0 and dist_sq1 <= r_sq:
        d = math.sqrt(dist_sq1)
        return _push_line(lines, n_lines, 0.0, 0.0, -r1y / d, r1x / d)
    if s >= 1.0 and dist_sq2 <= r_sq:
        if _det(r2x, r2y, -ux, -uy) >= 0.0:
            d = math.sqrt(dist_sq2)
            return _push_line(lines, n_lines, 0.0, 0.0, -r2y / d, r2x / d)
        return n_lines
    if s >= 0.0 and s < 1.0 and dist_sq_line <= r_sq:
        return _push_line(lines, n_lines, 0.0, 0.0, -ux, -uy)

    o1x = ax
    o1y = ay
    o1dx = ux
    o1dy = uy
    o2x = bx
    o2y = by
    o2dx = -ux
    o2dy = -uy
    same = False
    if s < 0.0 and dist_sq_line <= r_sq:
        o2x = o1x
        o2y = o1y
        o2dx = o1dx
        o2dy = o1dy
        same = True
        leg = math.sqrt(dist_sq1 - r_sq)
        llx = (r1x * leg - r1y * ra) / dist_sq1
        lly = (r1x * ra + r1y * leg) / dist_sq1
        rlx = (r1x * leg + r1y * ra) / dist_sq1
        rly = (-r1x * ra + r1y * leg) / dist_sq1
    elif s > 1.0 and dist_sq_line <= r_sq:
        o1x = o2x
        o1y = o2y
        o1dx = o2dx
        o1dy = o2dy
        same = True
        leg = math.sqrt(dist_sq2 - r_sq)
        llx = (r2x * leg - r2y * ra) / dist_sq2
        lly = (r2x * ra + r2y * leg) / dist_sq2
        rlx = (r2x * leg + r2y * ra) / dist_sq2
        rly = (-r2x * ra + r2y * leg) / dist_sq2
    else:
        leg1 = math.sqrt(dist_sq1 - r_sq)
        llx = (r1x * leg1 - r1y * ra) / dist_sq1
        lly = (r1x * ra + r1y * leg1) / dist_sq1
        leg2 = math.sqrt(dist_sq2 - r_sq)
        rlx = (r2x * leg2 + r2y * ra) / dist_sq2
        rly = (-r2x * ra + r2y * leg2) / dist_sq2

    left_foreign = False
    right_foreign = False
    if _det(llx, lly, o1dx, o1dy) >= 0.0:
        llx = o1dx
        lly = o1dy
        left_foreign = True
    if _det(rlx, rly, o2dx, o2dy) <= 0.0:
        rlx = o2dx
        rly = o2dy
        right_foreign = True

    lcx = inv_t * (o1x - x)
    lcy = inv_t * (o1y - y)
    rcx = inv_t * (o2x - x)
    rcy = inv_t * (o2y - y)
    cvx = rcx - lcx
    cvy = rcy - lcy

    t = 0.5 if same else ((vx - lcx) * cvx + (vy - lcy) * cvy) / (cvx * cvx + cvy * cvy)
    t_left = (vx - lcx) * llx + (vy - lcy) * lly
    t_right = (vx - rcx) * rlx + (vy - rcy) * rly

    if (t < 0.0 and t_left < 0.0) or (same and t_left < 0.0 and t_right < 0.0):
        wx = vx - lcx
        wy = vy - lcy
        w_len = math.sqrt(wx * wx + wy * wy)
        uwx = wx / w_len
        uwy = wy / w_len
        return _push_line(lines, n_lines, lcx + ra * inv_t * uwx, lcy + ra * inv_t * uwy, uwy, -uwx)
    if t > 1.0 and t_right < 0.0:
        wx = vx - rcx
        wy = vy - rcy
        w_len = math.sqrt(wx * wx + wy * wy)
        uwx = wx / w_len
        uwy = wy / w_len
        return _push_line(lines, n_lines, rcx + ra * inv_t * uwx, rcy + ra * inv_t * uwy, uwy, -uwx)

    if t < 0.0 or t > 1.0 or same:
        dist_sq_cutoff = np.inf
    else:
        ex = vx - (lcx + t * cvx)
        ey = vy - (lcy + t * cvy)
        dist_sq_cutoff = ex * ex + ey * ey
    if t_left < 0.0:
        dist_sq_left = np.inf
    else:
        ex = vx - (lcx + t_left * llx)
        ey = vy - (lcy + t_left * lly)
        dist_sq_left = ex * ex + ey * ey
    if t_right < 0.0:
        dist_sq_right = np.inf
    else:
        ex = vx - (rcx + t_right * rlx)
        ey = vy - (rcy + t_right * rly)
        dist_sq_right = ex * ex + ey * ey

    if dist_sq_cutoff <= dist_sq_left and dist_sq_cutoff <= dist_sq_right:
        dx = -o1dx
        dy = -o1dy
        return _push_line(lines, n_lines, lcx - ra * inv_t * dy, lcy + ra * inv_t * dx, dx, dy)
    if dist_sq_left <= dist_sq_right:
        if left_foreign:
            return n_lines
        return _push_line(lines, n_lines, lcx - ra * inv_t * lly, lcy + ra * inv_t * llx, llx, lly)
    if right_foreign:
        return n_lines
    dx = -rlx
    dy = -rly
    return _push_line(lines, n_lines, rcx - ra * inv_t * dy, rcy + ra * inv_t * dx, dx, dy)


@njit(cache=True)
def _agent_line(lines: np.ndarray, n_lines: int, rpx: float, rpy: float, rvx: float, rvy: float, vx: float, vy: float, cr: float, inv_tau: float, inv_dt: float) -> int:
    dist_sq = rpx * rpx + rpy * rpy
    cr_sq = cr * cr
    if dist_sq > cr_sq:
        wx = rvx - inv_tau * rpx
        wy = rvy - inv_tau * rpy
        w_len_sq = wx * wx + wy * wy
        dot1 = wx * rpx + wy * rpy
        if dot1 < 0.0 and dot1 * dot1 > cr_sq * w_len_sq:
            w_len = math.sqrt(w_len_sq)
            uwx = wx / w_len
            uwy = wy / w_len
            dx = uwy
            dy = -uwx
            ux = (cr * inv_tau - w_len) * uwx
            uy = (cr * inv_tau - w_len) * uwy
        else:
            leg = math.sqrt(dist_sq - cr_sq)
            if _det(rpx, rpy, wx, wy) > 0.0:
                dx = (rpx * leg - rpy * cr) / dist_sq
                dy = (rpx * cr + rpy * leg) / dist_sq
            else:
                dx = -(rpx * leg + rpy * cr) / dist_sq
                dy = -(-rpx * cr + rpy * leg) / dist_sq
            dot2 = rvx * dx + rvy * dy
            ux = dot2 * dx - rvx
            uy = dot2 * dy - rvy
    else:
        wx = rvx - inv_dt * rpx
        wy = rvy - inv_dt * rpy
        w_len = math.sqrt(wx * wx + wy * wy)
        if w_len > 0.0:
            uwx = wx / w_len
            uwy = wy / w_len
        elif dist_sq > 0.0:
            dist = math.sqrt(dist_sq)
            uwx = -rpx / dist
            uwy = -rpy / dist
        else:
            return n_lines
        dx = uwy
        dy = -uwx
        ux = (cr * inv_dt - w_len) * uwx
        uy = (cr * inv_dt - w_len) * uwy
    return _push_line(lines, n_lines, vx + 0.5 * ux, vy + 0.5 * uy, dx, dy)


@njit(cache=True, parallel=True)
def _orca_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    max_speed: np.ndarray,
    desired_speed: np.ndarray,
    goal: np.ndarray,
    has_goal: np.ndarray,
    nbr_idx: np.ndarray,
    max_neighbors: int,
    edges: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    goal_radius: float,
    time_horizon: float,
    time_horizon_obst: float,
    wall_clearance: float,
    inv_dt: float,
    out: np.ndarray,
    kind: np.ndarray | None = None,
    robot_range: np.ndarray | None = None,
    robot_repulsion_gain: np.ndarray | None = None,
    vision_range: np.ndarray | None = None,
    robot_reaction_gain: np.ndarray | None = None,
) -> None:
    n = pos.shape[0]
    n_walls = edges.shape[0] // 2
    n_agent_lines = min(max_neighbors, nbr_idx.shape[1])
    inv_tau = 1.0 / time_horizon
    inv_tau_obst = 1.0 / time_horizon_obst
    for i in prange(n):
        out[i, 0] = 0.0
        out[i, 1] = 0.0
        if not has_goal[i]:
            continue

        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        ra = radius[i]
        vmax = max_speed[i]

        gx = goal[i, 0] - x
        gy = goal[i, 1] - y
        dist = math.sqrt(gx * gx + gy * gy)
        if dist < goal_radius:
            pref_x = 0.0
            pref_y = 0.0
        else:
            pref_x = gx / dist * desired_speed[i]
            pref_y = gy / dist * desired_speed[i]

        ra_obst = ra + wall_clearance
        range_sq = (time_horizon_obst * vmax + ra_obst) ** 2
        query_r = math.sqrt(range_sq)
        walls = np.empty(64, dtype=np.int64)
        n_hit = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, walls)
        if n_hit > walls.shape[0]:
            walls = np.empty(n_hit, dtype=np.int64)
            n_hit = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, walls)
        cand = np.empty(2 * n_hit, dtype=np.int64)
        cand_d = np.empty(2 * n_hit, dtype=np.float64)
        n_cand = 0
        for h in range(n_hit):
            for side in range(2):
                e = walls[h] + side * n_walls
                ax = edges[e, 0]
                ay = edges[e, 1]
                bx = edges[e, 2]
                by = edges[e, 3]
                if _det(ax - x, ay - y, bx - ax, by - ay) >= 0.0:
                    continue
                d = _dist_sq_point_segment(ax, ay, bx, by, x, y)
                if d >= range_sq:
                    continue
                k = n_cand
                while k > 0 and (d < cand_d[k - 1] or (d == cand_d[k - 1] and e < cand[k - 1])):
                    cand[k] = cand[k - 1]
                    cand_d[k] = cand_d[k - 1]
                    k -= 1
                cand[k] = e
                cand_d[k] = d
                n_cand += 1

        lines = np.empty((n_cand + n_agent_lines, 4), dtype=np.float64)
        n_lines = 0
        for c in range(n_cand):
            n_lines = _obstacle_line(lines, n_lines, edges[cand[c]], x, y, vx, vy, ra_obst, inv_tau_obst)
        n_obst_lines = n_lines

        n_found = 0
        for c in range(nbr_idx.shape[1]):
            if n_found >= n_agent_lines:
                break
            j = nbr_idx[i, c]
            if j >= n:
                break
            if j == i:
                continue
            combined_r = ra + radius[j]
            inv_tau_j = inv_tau
            if kind is not None and robot_range is not None and robot_repulsion_gain is not None and vision_range is not None:
                if kind[j] == 1 and kind[i] != 1:
                    if robot_range[i] <= 0.0:
                        continue
                    if robot_range[i] != vision_range[i]:
                        dx = pos[j, 0] - x
                        dy = pos[j, 1] - y
                        if dx * dx + dy * dy > robot_range[i] * robot_range[i]:
                            continue
                    eff_r_robot = max(0.05, radius[j] * max(0.85, robot_repulsion_gain[i]))
                    combined_r = ra + eff_r_robot
                    if robot_reaction_gain is not None:
                        inv_tau_j = inv_tau / max(0.05, robot_reaction_gain[i])
            n_found += 1
            n_lines = _agent_line(lines, n_lines, pos[j, 0] - x, pos[j, 1] - y, vx - vel[j, 0], vy - vel[j, 1], vx, vy, combined_r, inv_tau_j, inv_dt)

        fail, rx, ry = _linear_program2(lines, n_lines, vmax, pref_x, pref_y, False)
        if fail < n_lines:
            rx, ry = _linear_program3(lines, n_lines, n_obst_lines, fail, vmax, rx, ry)
        out[i, 0] = rx
        out[i, 1] = ry


class ORCAPlanner(LocalPlanner):
    supports_pool: bool = True

    def __init__(
        self,
        time_horizon: float = 5.0,
        max_neighbors: int = 10,
        neighbor_dist: float = 5.0,
        goal_radius: float = 1e-6,
        time_horizon_obst: float = 2.0,
        wall_clearance: float = 0.05,
        wall_grid_cell: float = 1.0,
    ):
        self.time_horizon = time_horizon
        self.max_neighbors = max_neighbors
        self.neighbor_dist = neighbor_dist
        self.goal_radius = goal_radius
        self.time_horizon_obst = time_horizon_obst
        self.wall_clearance = wall_clearance
        self.wall_grid_cell = wall_grid_cell
        self._edges = np.empty((0, 6), dtype=np.float64)
        self._grid = WallGrid(np.empty((0, 4), dtype=np.float64), cell=wall_grid_cell)
        self._warmup()

    def _warmup(self) -> None:
        pool = AgentPool(capacity=2)
        pool.n = 2
        pool.pos[1] = (1.0, 0.0)
        pool.has_goal[:2] = True
        self.compute_pool(pool)

    def set_walls(self, segments: Segments) -> None:
        seg = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
        d = seg[:, 2:] - seg[:, :2]
        length = np.hypot(d[:, 0], d[:, 1])
        keep = length > 0.0
        seg = seg[keep]
        u = d[keep] / length[keep, None]
        forward = np.hstack([seg, u])
        backward = np.hstack([seg[:, 2:], seg[:, :2], -u])
        self._edges = np.ascontiguousarray(np.vstack([forward, backward]))
        self._grid = WallGrid(seg, cell=self.wall_grid_cell)

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            return
        pos = pool.pos[:n]
        k = min(self.max_neighbors + 1, n)
        _, idx = cKDTree(pos).query(pos, k=k, distance_upper_bound=self.neighbor_dist)
        idx = np.ascontiguousarray(idx.reshape(n, k), dtype=np.int64)
        out = np.empty((n, 2), dtype=np.float64)
        grid = self._grid
        vision = pool.vision_range[:n]
        sight = np.minimum(vision, np.maximum(0.0, vision + pool.robot_range_offset[:n]))
        # robot_range == vision_range means no cut: full sight keeps robots out to neighbor_dist, as for people. Shortened or
        # no sight cuts at the farthest of sight, hearing and cane reach.
        cut = (sight < vision) | (vision <= 0.0)
        # One cut per pedestrian covers every robot, so hearing takes the loudest robot (loudness times speed).
        loudest = float(pool.robot_hearing_scale(np.flatnonzero(pool.kind[:n] == KIND_ROBOT)).max(initial=0.0))
        robot_range = np.where(
            pool.robot_blind[:n] | ~pool.robot_yields[:n],
            0.0,
            np.where(cut, np.maximum(sight, np.maximum(pool.hearing_range[:n] * loudest, pool.cane_reach[:n])), vision),
        )
        _orca_kernel(
            pos,
            pool.prev_vel[:n],
            pool.agent_radius[:n],
            pool.max_velocity[:n],
            pool.desired_vel[:n],
            pool.goal_pos[:n],
            pool.has_goal[:n],
            idx,
            self.max_neighbors,
            self._edges,
            grid.cell_start,
            grid.cell_walls,
            grid.wall_cx0,
            grid.wall_cy0,
            grid.origin_x,
            grid.origin_y,
            grid.cell,
            grid.nx,
            grid.ny,
            self.goal_radius,
            self.time_horizon,
            self.time_horizon_obst,
            self.wall_clearance,
            1.0 / dt,
            out,
            pool.kind[:n],
            robot_range,
            pool.robot_repulsion_gain[:n],
            pool.vision_range[:n],
            pool.robot_reaction_gain[:n],
        )
        pool.vel[:n] = out

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}
        pool = AgentPool(capacity=len(agents))
        for agent in agents:
            pool.add_agent(agent)
        pool.set_goals(global_goals)
        self.compute_pool(pool, dt=dt)
        return {int(pool.agent_ids[i]): (float(pool.vel[i, 0]), float(pool.vel[i, 1])) for i in range(pool.n)}
