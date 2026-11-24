from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from . import Perception

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.agents.types import SampledPerception
    from arena_humansim.core.pool import AgentPool
    from arena_humansim.occlusion import Occluder
    from arena_humansim.utils.types import AgentState, BeliefState, WorldState

from arena_humansim.core.pool import KIND_ROBOT


class DefaultPerception(Perception):
    supports_pool: bool = True

    def __init__(self, occluder: Occluder | None = None, small_n_threshold: int = 64) -> None:
        self._occluder = occluder
        self._small_n_threshold = small_n_threshold
        self._shared_tree: cKDTree | None = None
        self._shared_ids: list[int] | None = None
        self._shared_positions: list[tuple[float, float]] | None = None
        self._dt: float | None = None
        self._pool: AgentPool | None = None

    def attach(self, pool: AgentPool) -> None:
        self._pool = pool

    def _heard_scale(self, agent_id: int) -> float:
        """Hearing factor of one robot for the per-agent path: robot_loudness times its speed over 1 m/s."""
        pool = self._pool
        row = None if pool is None else pool._id_to_idx.get(agent_id)
        if pool is None or row is None:
            return 1.0 if pool is None else pool.robot_loudness
        return float(pool.robot_hearing_scale(np.array([row]))[0])

    def prepare_tick(self, all_agents: dict[int, AgentState]) -> None:
        ids: list[int] = []
        positions: list[tuple[float, float]] = []
        for aid, ag in all_agents.items():
            ids.append(aid)
            positions.append((ag.pose.x, ag.pose.y))
        self._shared_ids = ids
        self._shared_positions = positions
        if len(positions) > 1:
            self._shared_tree = cKDTree(np.array(positions, dtype=np.float64))
        else:
            self._shared_tree = None

    @property
    def shared_tree(self) -> cKDTree | None:
        return self._shared_tree

    @property
    def shared_ids(self) -> list[int] | None:
        return self._shared_ids

    @property
    def shared_positions(self) -> list[tuple[float, float]] | None:
        return self._shared_positions

    def compute_pool(self, pool: AgentPool, dt: float | None = None) -> None:
        if dt is not None:
            pool.perception_clock += dt
        self._dt = dt
        n = pool.n
        if n <= 1:
            pool.robot_sensed_since.clear()
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        positions = pool.pos[:n]

        if n <= self._small_n_threshold:
            self._compute_pool_dense(pool, positions, n)
        else:
            self._compute_pool_kdtree(pool, positions, n)

    def _compute_pool_dense(self, pool: AgentPool, positions: np.ndarray, n: int) -> None:
        diff = positions[None, :, :] - positions[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(dists, np.inf)

        vision_range = pool.vision_range[:n]
        mask = dists <= vision_range[:, None]

        proximity_sense = pool.proximity_sense[:n]
        prox_mask = dists <= proximity_sense[:, None] if np.any(proximity_sense > 0.0) else None

        vision_fov = pool.vision_fov[:n]
        needs_fov = vision_fov < 360.0
        if np.any(needs_fov):
            fov_pairs = mask & needs_fov[:, None]
            if prox_mask is not None:
                fov_pairs &= ~prox_mask
            r_fov, c_fov = np.nonzero(fov_pairs)
            if len(r_fov) > 0:
                heading = pool.theta[:n]
                pair_fov = _pair_fov(pool, r_fov, c_fov)
                cos_half = np.cos(np.radians(np.clip(pair_fov, 0.0, 360.0) * 0.5))
                dot = diff[r_fov, c_fov, 0] * np.cos(heading)[r_fov] + diff[r_fov, c_fov, 1] * np.sin(heading)[r_fov]
                outside = (pair_fov < 360.0) & (dot < dists[r_fov, c_fov] * cos_half)
                mask[r_fov[outside], c_fov[outside]] = False

        if prox_mask is not None:
            mask |= prox_mask

        if self._occluder is not None:
            r_los, c_los = np.nonzero(mask & pool.vision_occlusion[:n][:, None])
            if len(r_los) > 0:
                blocked = ~_clear_unordered(self._occluder, positions, r_los, c_los, n)
                mask[r_los[blocked], c_los[blocked]] = False

        # Robots are gated by each observer's blind flag, range offset and cone offset.
        is_robot = pool.kind[:n] == KIND_ROBOT
        if np.any(is_robot):
            robot_cols = np.flatnonzero(is_robot)
            human_rows = np.flatnonzero(~is_robot)
            if len(human_rows) > 0 and len(robot_cols) > 0:
                h_idx = human_rows[:, None]
                r_idx = robot_cols[None, :]
                blind_mask = pool.robot_blind[human_rows, None]
                mask[h_idx, r_idx] &= ~blind_mask

                partial = (pool.robot_range_offset[human_rows, None] != 0.0) | (pool.robot_fov_offset[human_rows, None] != 0.0)
                partial = np.broadcast_to(partial, (len(human_rows), len(robot_cols)))
                if np.any(partial):
                    p_rows, p_cols = np.nonzero(partial)
                    orig_r = human_rows[p_rows]
                    orig_c = robot_cols[p_cols]
                    pair_dists = dists[orig_r, orig_c]
                    eff_range = _robot_range(pool.vision_range[orig_r], pool.robot_range_offset[orig_r])
                    eff_prox = np.minimum(pool.proximity_sense[orig_r], eff_range)
                    beyond = pair_dists > np.maximum(eff_range, eff_prox)
                    mask[orig_r[beyond], orig_c[beyond]] = False

                    surviving = ~beyond & (pair_dists > eff_prox)
                    if np.any(surviving):
                        sr_r = orig_r[surviving]
                        sr_c = orig_c[surviving]
                        eff_fov = _robot_fov(pool.vision_fov[sr_r], pool.robot_fov_offset[sr_r])
                        needs_fov_check = eff_fov < 360.0
                        cos_half = np.cos(np.radians(eff_fov * 0.5))
                        heading = pool.theta[sr_r]
                        dot = diff[sr_r, sr_c, 0] * np.cos(heading) + diff[sr_r, sr_c, 1] * np.sin(heading)
                        outside_fov = needs_fov_check & (dot < pair_dists[surviving] * cos_half)
                        mask[sr_r[outside_fov], sr_c[outside_fov]] = False

        rows = _channel_rows(pool, n)
        if len(rows) > 0:
            c_row = np.repeat(rows, n)
            c_col = np.tile(np.arange(n), len(rows))
            hit = _channel_pairs(pool, positions, c_row, c_col, n, self._occluder)
            mask[c_row[hit], c_col[hit]] = True

        row, col = np.where(mask)
        if self._dt is not None and np.any(pool.detection_delay[:n] > 0.0):
            keep = _noticed(pool, row, col)
            row, col = row[keep], col[keep]
            mask = np.zeros_like(mask)
            mask[row, col] = True
        else:
            pool.robot_sensed_since.clear()
        if len(row) == 0:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        counts = mask.sum(axis=1)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])
        col = col[np.lexsort((pool.agent_ids[col], row))]
        pool.set_neighbor_csr(indptr, col.astype(np.int32))

    def _compute_pool_kdtree(self, pool: AgentPool, positions: np.ndarray, n: int) -> None:
        tree = cKDTree(positions)
        self._shared_tree = tree

        max_range = float(max(pool.vision_range[:n].max(), pool.proximity_sense[:n].max()))
        chan_rows = _channel_rows(pool, n)
        if len(chan_rows) > 0:
            # Farthest anyone is heard (robots scaled by loudness and speed) or touched (the cane's far corner plus the largest disc).
            heard = float(pool.hearing_range[chan_rows].max()) * max(1.0, float(pool.robot_hearing_scale(np.arange(n)).max()))
            touched = float(np.hypot(pool.cane_reach[chan_rows], 0.5 * pool.cane_width[chan_rows]).max() + pool.agent_radius[:n].max())
            max_range = max(max_range, heard, touched)
        sdm = tree.sparse_distance_matrix(tree, max_range, output_type="coo_matrix")

        row = sdm.row
        col = sdm.col
        dist = sdm.data

        mask = row != col
        row = row[mask]
        col = col[mask]
        dist = dist[mask]

        if len(row) == 0:
            pool.robot_sensed_since.clear()
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        prox_ok = dist <= pool.proximity_sense[row]
        range_ok = dist <= pool.vision_range[row]
        keep = range_ok | prox_ok

        vision_fov = pool.vision_fov[:n]
        fov_idx = np.flatnonzero(range_ok & ~prox_ok & (vision_fov[row] < 360.0))
        if len(fov_idx) > 0:
            heading = pool.theta[:n]
            hx = np.cos(heading)
            hy = np.sin(heading)
            r_fov = row[fov_idx]
            c_fov = col[fov_idx]
            pair_fov = _pair_fov(pool, r_fov, c_fov)
            cos_half = np.cos(np.radians(np.clip(pair_fov, 0.0, 360.0) * 0.5))
            dx = positions[c_fov, 0] - positions[r_fov, 0]
            dy = positions[c_fov, 1] - positions[r_fov, 1]
            outside = (pair_fov < 360.0) & (dx * hx[r_fov] + dy * hy[r_fov] < dist[fov_idx] * cos_half)
            keep[fov_idx[outside]] = False

        if self._occluder is not None:
            los_idx = np.flatnonzero(keep & pool.vision_occlusion[row])
            if len(los_idx) > 0:
                blocked = ~_clear_unordered(self._occluder, positions, row[los_idx], col[los_idx], n)
                keep[los_idx[blocked]] = False

        # Robots are gated by each observer's blind flag, range offset and cone offset.
        robot_nbr = (pool.kind[col] == KIND_ROBOT) & (pool.kind[row] != KIND_ROBOT)
        if np.any(robot_nbr):
            robot_nbr_idx = np.flatnonzero(robot_nbr)
            keep[robot_nbr_idx[pool.robot_blind[row[robot_nbr]]]] = False

            partial_sub = (pool.robot_range_offset[row[robot_nbr]] != 0.0) | (pool.robot_fov_offset[row[robot_nbr]] != 0.0)
            if np.any(partial_sub):
                partial_idx = robot_nbr_idx[partial_sub]
                p_row = row[partial_idx]
                p_col = col[partial_idx]
                p_dist = dist[partial_idx]
                eff_range = _robot_range(pool.vision_range[p_row], pool.robot_range_offset[p_row])
                eff_prox = np.minimum(pool.proximity_sense[p_row], eff_range)
                beyond = p_dist > np.maximum(eff_range, eff_prox)
                keep[partial_idx[beyond]] = False

                surviving = ~beyond & (p_dist > eff_prox)
                if np.any(surviving):
                    s_idx = partial_idx[surviving]
                    s_row = p_row[surviving]
                    s_col = p_col[surviving]
                    eff_fov = _robot_fov(pool.vision_fov[s_row], pool.robot_fov_offset[s_row])
                    needs_fov_check = eff_fov < 360.0
                    cos_half = np.cos(np.radians(eff_fov * 0.5))
                    heading = pool.theta[s_row]
                    dx = positions[s_col, 0] - positions[s_row, 0]
                    dy = positions[s_col, 1] - positions[s_row, 1]
                    outside = needs_fov_check & (dx * np.cos(heading) + dy * np.sin(heading) < p_dist[surviving] * cos_half)
                    keep[s_idx[outside]] = False

        if len(chan_rows) > 0:
            is_chan = np.zeros(n, dtype=np.bool_)
            is_chan[chan_rows] = True
            cand = np.flatnonzero(is_chan[row])
            keep[cand[_channel_pairs(pool, positions, row[cand], col[cand], n, self._occluder)]] = True

        row = row[keep]
        col = col[keep]
        if self._dt is not None and np.any(pool.detection_delay[:n] > 0.0):
            keep = _noticed(pool, row, col)
            row, col = row[keep], col[keep]
        else:
            pool.robot_sensed_since.clear()

        counts = np.bincount(row, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        order = np.lexsort((pool.agent_ids[col], row))
        indices = col[order].astype(np.int32)

        pool.set_neighbor_csr(indptr, indices)

    def compute(
        self,
        agent: BaseAgent,
        all_agents: dict[int, AgentState],
        world_state: WorldState,
        belief: BeliefState,
    ) -> BeliefState:
        agent_id = agent.state.agent_id
        vision_range: float = agent.params.perception.vision_range
        vision_fov: float = agent.params.perception.vision_fov
        proximity_sense: float = agent.params.perception.proximity_sense
        vision_occlusion: bool = agent.params.perception.vision_occlusion

        if agent_id not in all_agents:
            return belief

        if self._shared_tree is None or self._shared_ids is None or self._shared_positions is None:
            return belief

        observer = all_agents[agent_id]
        ox, oy = observer.pose.x, observer.pose.y

        if len(all_agents) <= 1:
            return belief

        heading = observer.pose.theta
        perception = agent.params.perception
        robot_range = min(vision_range, max(0.0, vision_range + perception.robot_range_offset))
        robot_prox = min(proximity_sense, robot_range)
        robot_fov = min(360.0, max(0.0, vision_fov + perception.robot_fov_offset))

        p = self._pool
        loudest = 1.0 if p is None else float(p.robot_hearing_scale(np.flatnonzero(p.kind[: p.n] == KIND_ROBOT)).max(initial=1.0))
        query_radius = max(vision_range, proximity_sense, perception.hearing_range * loudest, float(np.hypot(perception.cane_reach, 0.5 * perception.cane_width)))
        neighbor_indices = self._shared_tree.query_ball_point([ox, oy], query_radius)

        candidates: list[int] = []
        for idx in neighbor_indices:
            nid = self._shared_ids[idx]
            if nid == agent_id:
                continue
            nbr_is_robot = all_agents[nid].kind == KIND_ROBOT
            if nbr_is_robot and perception.robot_blind:
                continue
            eff_range = robot_range if nbr_is_robot else vision_range
            eff_prox = robot_prox if nbr_is_robot else proximity_sense
            eff_fov = robot_fov if nbr_is_robot else vision_fov

            nx, ny = self._shared_positions[idx]
            d = float(np.hypot(nx - ox, ny - oy))
            if _channel_hit(perception, nx - ox, ny - oy, heading, d, self._heard_scale(nid) if nbr_is_robot else 1.0):
                candidates.append(idx)
                continue
            within_prox = d <= eff_prox
            if not within_prox:
                if d > eff_range:
                    continue
                if eff_fov < 360.0:
                    bearing = np.arctan2(ny - oy, nx - ox)
                    angle_diff = abs(_wrap_angle(bearing - heading))
                    if angle_diff > np.radians(eff_fov * 0.5):
                        continue
            candidates.append(idx)

        if self._occluder is not None and vision_occlusion and candidates:
            p_a = np.array([[ox, oy]] * len(candidates), dtype=np.float64)
            p_b = np.array([self._shared_positions[idx] for idx in candidates], dtype=np.float64)
            los = self._occluder.clear(p_a, p_b)
            candidates = [idx for idx, clear in zip(candidates, los, strict=True) if clear]

        for idx in candidates:
            nid = self._shared_ids[idx]
            belief.observed_agents.append(all_agents[nid])

        return belief


def _robot_range(vision_range: np.ndarray, range_offset: np.ndarray) -> np.ndarray:
    return np.clip(vision_range + range_offset, 0.0, vision_range)


def _robot_fov(vision_fov: np.ndarray, fov_offset: np.ndarray) -> np.ndarray:
    return np.clip(vision_fov + fov_offset, 0.0, 360.0)


def _pair_fov(pool: AgentPool, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    robot_target = (pool.kind[cols] == KIND_ROBOT) & (pool.kind[rows] != KIND_ROBOT)
    return np.where(robot_target, _robot_fov(pool.vision_fov[rows], pool.robot_fov_offset[rows]), pool.vision_fov[rows])


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _clear_unordered(occluder: Occluder, positions: np.ndarray, row: np.ndarray, col: np.ndarray, n: int) -> np.ndarray:
    lo = np.minimum(row, col).astype(np.int64)
    hi = np.maximum(row, col).astype(np.int64)
    keys, inverse = np.unique(lo * n + hi, return_inverse=True)
    return occluder.clear(positions[keys // n], positions[keys % n])[inverse]


def _channel_rows(pool: AgentPool, n: int) -> np.ndarray:
    """Pool rows of the people that sense by hearing or by cane touch, besides sight."""
    return np.flatnonzero(((pool.hearing_range[:n] > 0.0) | (pool.cane_reach[:n] > 0.0)) & (pool.kind[:n] != KIND_ROBOT))


def _channel_pairs(pool: AgentPool, positions: np.ndarray, row: np.ndarray, col: np.ndarray, n: int, occluder: Occluder | None) -> np.ndarray:
    """Keep mask over candidate (row, col) pairs: whether row senses col by hearing (any bearing, out to hearing_range, robots
    scaled by pool.robot_hearing_scale: loudness times speed) or by touching it with the cane (a disc reaching into the rectangle cane_reach ahead by
    cane_width across). Walls block both for people with vision_occlusion; robot_blind people sense no robot."""
    dx = positions[col, 0] - positions[row, 0]
    dy = positions[col, 1] - positions[row, 1]
    is_robot = pool.kind[col] == KIND_ROBOT
    hearing = pool.hearing_range[row]
    sensed = (hearing > 0.0) & (np.hypot(dx, dy) <= hearing * pool.robot_hearing_scale(np.arange(n))[col])
    cos_h = np.cos(pool.theta[row])
    sin_h = np.sin(pool.theta[row])
    ahead = dx * cos_h + dy * sin_h
    across = np.abs(dy * cos_h - dx * sin_h)
    reach = pool.cane_reach[row]
    gap_x = np.maximum(np.maximum(-ahead, ahead - reach), 0.0)
    gap_y = np.maximum(across - 0.5 * pool.cane_width[row], 0.0)
    radius = pool.agent_radius[col]
    sensed |= (reach > 0.0) & (gap_x * gap_x + gap_y * gap_y <= radius * radius)
    sensed &= (row != col) & ~(pool.robot_blind[row] & is_robot)
    if occluder is not None:
        los = np.flatnonzero(sensed & pool.vision_occlusion[row])
        if len(los) > 0:
            sensed[los[~_clear_unordered(occluder, positions, row[los], col[los], n)]] = False
    return sensed


def _noticed(pool: AgentPool, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Keep mask over the sensed (row, col) pairs: a robot enters an observer's list once the observer has sensed it without a
    break for its detection_delay. Sensing start times live on the pool, keyed by agent ids, and pairs no longer sensed drop. Ticks that skip this check clear them all, so no delay carries over a
    gap."""
    keep = np.ones(len(row), dtype=np.bool_)
    delayed = np.flatnonzero((pool.detection_delay[row] > 0.0) & (pool.kind[col] == KIND_ROBOT) & (pool.kind[row] != KIND_ROBOT))
    now = pool.perception_clock
    since: dict[tuple[int, int], float] = {}
    for k in delayed:
        key = (int(pool.agent_ids[row[k]]), int(pool.agent_ids[col[k]]))
        start = pool.robot_sensed_since.get(key, now)
        since[key] = start
        keep[k] = now - start >= pool.detection_delay[row[k]] - 1e-9
    pool.robot_sensed_since = since
    return keep


def _channel_hit(perception: SampledPerception, dx: float, dy: float, heading: float, d: float, scale: float) -> bool:
    """Single-observer form of _channel_pairs for the per-agent fallback path: targets are points, and there is no detection delay."""
    if perception.hearing_range > 0.0 and d <= perception.hearing_range * scale:
        return True
    if perception.cane_reach <= 0.0:
        return False
    ahead = dx * np.cos(heading) + dy * np.sin(heading)
    across = abs(dy * np.cos(heading) - dx * np.sin(heading))
    return 0.0 <= ahead <= perception.cane_reach and across <= 0.5 * perception.cane_width
