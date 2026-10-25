from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.collision.wall_projection import _QUERY_CAPACITY, WallProjectionResolver
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.types import Segments

_MARGIN = 0.01


def _reference_resolve(pool: AgentPool, segments: Segments, margin: float) -> set[int]:
    wall_segments_np = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
    n = pool.n
    if n == 0 or wall_segments_np.shape[0] == 0:
        return set()

    pos = pool.pos[:n]
    radii = pool.agent_radius[:n]
    vel = pool.vel[:n]

    A = wall_segments_np[:, 0, :]
    AB = wall_segments_np[:, 1, :] - A
    ab_sq = np.einsum("ij,ij->i", AB, AB)

    corrected = np.zeros(n, dtype=bool)

    for _ in range(3):
        AP = pos[:, np.newaxis, :] - A[np.newaxis, :, :]
        t = np.einsum("nwj,wj->nw", AP, AB) / np.maximum(ab_sq[np.newaxis, :], 1e-12)
        t = np.clip(t, 0.0, 1.0)
        closest = A[np.newaxis, :, :] + t[:, :, np.newaxis] * AB[np.newaxis, :, :]
        diff = pos[:, np.newaxis, :] - closest
        dist = np.linalg.norm(diff, axis=2)

        threshold = radii[:, np.newaxis] + margin
        penetrating = dist < threshold - 1e-9

        if not penetrating.any():
            break

        corrected |= penetrating.any(axis=1)

        safe_dist = np.where(dist > 1e-9, dist, 1e-9)
        normal = diff / safe_dist[:, :, np.newaxis]
        overlap = (threshold - dist) * penetrating
        correction = (normal * overlap[:, :, np.newaxis]).sum(axis=1)
        pos += correction

        for i in range(n):
            if not penetrating[i].any():
                continue
            wall_normals = normal[i, penetrating[i]]
            v = vel[i]
            for wn in wall_normals:
                proj = v[0] * wn[0] + v[1] * wn[1]
                if proj < 0:
                    vel[i] -= proj * wn

    if not corrected.any():
        return set()
    return {int(aid) for aid in pool.agent_ids[:n][corrected]}


def _maze_segments() -> Segments:
    return [((float(x0), float(y0)), (float(x1), float(y1))) for (x0, y0), (x1, y1) in generate_maze(10, seed=42).walls]


def _scene_pool(seed: int, n: int, pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> AgentPool:
    rng = np.random.default_rng(seed)
    uniform = rng.uniform(-0.5, 20.5, size=(n // 2, 2))
    near_nodes = 2.0 * rng.integers(0, 11, size=(n - n // 2, 2)) + rng.uniform(-0.4, 0.4, size=(n - n // 2, 2))
    positions = np.concatenate([uniform, near_nodes])
    pool = pool_empty(capacity=n)
    for i, (x, y) in enumerate(positions):
        pool.add_agent(agent_factory(i + 1, x=float(x), y=float(y)))
    pool.pos[:n] = positions
    pool.agent_radius[:n] = rng.uniform(0.2, 0.45, size=n)
    pool.vel[:n] = rng.normal(0.0, 1.5, size=(n, 2))
    return pool


def test_inside_corner_cancels_velocity_into_both_walls(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((0.0, 0.0), (2.0, 0.0)), ((0.0, 0.0), (0.0, 2.0))])
    pool = pool_empty(capacity=2)
    pool.add_agent(agent_factory(7, x=0.1, y=0.1))
    pool.pos[0] = (0.1, 0.1)
    pool.agent_radius[0] = 0.3
    pool.vel[0] = (-1.0, -2.0)

    corrected = resolver.resolve(pool)

    assert corrected == {7}
    assert pool.pos[0] == pytest.approx((0.31, 0.31), abs=1e-12)
    assert pool.vel[0, 0] == 0.0
    assert pool.vel[0, 1] == 0.0


def test_inside_corner_keeps_velocity_pointing_out_of_one_wall(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((0.0, 0.0), (2.0, 0.0)), ((0.0, 0.0), (0.0, 2.0))])
    pool = pool_empty(capacity=2)
    pool.add_agent(agent_factory(7, x=0.1, y=0.1))
    pool.pos[0] = (0.1, 0.1)
    pool.agent_radius[0] = 0.3
    pool.vel[0] = (-1.0, 0.5)

    resolver.resolve(pool)

    assert pool.vel[0, 0] == 0.0
    assert pool.vel[0, 1] == 0.5


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_resolve_matches_sequential_reference_on_maze(seed: int, pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    segments = _maze_segments()
    pool = _scene_pool(seed, 300, pool_empty, agent_factory)
    ref_pool = _scene_pool(seed, 300, pool_empty, agent_factory)
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls(segments)

    corrected = resolver._project(pool)
    ref_corrected = _reference_resolve(ref_pool, segments, _MARGIN)

    assert len(ref_corrected) > 100
    assert corrected == ref_corrected
    assert np.allclose(pool.pos, ref_pool.pos, rtol=0.0, atol=1e-12)
    assert np.allclose(pool.vel, ref_pool.vel, rtol=0.0, atol=1e-12)


def test_resolve_matches_sequential_reference_over_repeated_ticks(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    segments = _maze_segments()
    pool = _scene_pool(11, 300, pool_empty, agent_factory)
    ref_pool = _scene_pool(11, 300, pool_empty, agent_factory)
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls(segments)
    rng = np.random.default_rng(11)

    for _ in range(5):
        corrected = resolver._project(pool)
        ref_corrected = _reference_resolve(ref_pool, segments, _MARGIN)
        assert corrected == ref_corrected
        assert np.allclose(pool.pos, ref_pool.pos, rtol=0.0, atol=1e-12)
        assert np.allclose(pool.vel, ref_pool.vel, rtol=0.0, atol=1e-12)
        step = rng.normal(0.0, 0.3, size=(300, 2))
        pool.pos[:300] += step
        ref_pool.pos[:300] += step


def test_resolve_matches_sequential_reference_on_random_long_walls(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    rng = np.random.default_rng(7)
    starts = rng.uniform(0.0, 20.0, size=(400, 2))
    ends = starts + rng.uniform(-8.0, 8.0, size=(400, 2))
    segments: Segments = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in zip(starts, ends, strict=True)]
    pool = _scene_pool(7, 300, pool_empty, agent_factory)
    ref_pool = _scene_pool(7, 300, pool_empty, agent_factory)
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls(segments)

    corrected = resolver._project(pool)
    ref_corrected = _reference_resolve(ref_pool, segments, _MARGIN)

    assert len(ref_corrected) > 100
    assert corrected == ref_corrected
    assert np.allclose(pool.pos, ref_pool.pos, rtol=0.0, atol=1e-12)
    assert np.allclose(pool.vel, ref_pool.vel, rtol=0.0, atol=1e-12)


def test_resolve_matches_reference_when_candidates_exceed_query_buffer(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    n_walls = 3 * _QUERY_CAPACITY
    angles = np.linspace(0.0, np.pi, n_walls, endpoint=False)
    segments: Segments = [((5.0 - 3.0 * float(np.cos(a)), 5.0 - 3.0 * float(np.sin(a))), (5.0 + 3.0 * float(np.cos(a)), 5.0 + 3.0 * float(np.sin(a)))) for a in angles]

    def crossing_pool() -> AgentPool:
        pool = pool_empty(capacity=2)
        pool.add_agent(agent_factory(3, x=5.05, y=5.02))
        pool.pos[0] = (5.05, 5.02)
        pool.agent_radius[0] = 0.3
        pool.vel[0] = (0.7, -0.4)
        return pool

    pool = crossing_pool()
    ref_pool = crossing_pool()
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls(segments)

    corrected = resolver._project(pool)
    ref_corrected = _reference_resolve(ref_pool, segments, _MARGIN)

    assert ref_corrected == {3}
    assert corrected == ref_corrected
    assert np.allclose(pool.pos, ref_pool.pos, rtol=0.0, atol=1e-12)
    assert np.allclose(pool.vel, ref_pool.vel, rtol=0.0, atol=1e-12)


def _contact_pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], positions: list[tuple[float, float]], autonomous: list[bool]) -> AgentPool:
    pool = pool_empty(capacity=max(2, len(positions)))
    for i, (x, y) in enumerate(positions):
        pool.add_agent(agent_factory(i + 1, x=x, y=y))
        pool.agent_radius[i] = 0.25
        pool.policy_idx[i] = 0 if autonomous[i] else -1
    return pool


def test_overlapping_pair_separates_and_stops_closing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _contact_pool(pool_empty, agent_factory, [(0.0, 0.0), (0.3, 0.0)], [True, True])
    pool.vel[0] = (1.0, 0.4)
    pool.vel[1] = (-1.0, -0.2)

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.pos[0] == pytest.approx((-0.1, 0.0), abs=1e-12)
    assert pool.pos[1] == pytest.approx((0.4, 0.0), abs=1e-12)
    assert pool.vel[0] == pytest.approx((0.0, 0.4), abs=1e-12)
    assert pool.vel[1] == pytest.approx((0.0, -0.2), abs=1e-12)


def test_separating_pair_keeps_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _contact_pool(pool_empty, agent_factory, [(0.0, 0.0), (0.3, 0.0)], [True, True])
    pool.vel[0] = (-1.0, 0.0)
    pool.vel[1] = (1.0, 0.0)

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.vel[0] == pytest.approx((-1.0, 0.0), abs=1e-12)
    assert pool.vel[1] == pytest.approx((1.0, 0.0), abs=1e-12)


def test_contact_with_a_non_autonomous_agent_moves_neither(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _contact_pool(pool_empty, agent_factory, [(0.0, 0.0), (0.3, 0.0)], [True, False])
    pool.vel[0] = (1.0, 0.0)
    pool.vel[1] = (-0.5, 0.0)

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.pos[1] == pytest.approx((0.3, 0.0), abs=1e-12)
    assert pool.vel[1] == pytest.approx((-0.5, 0.0), abs=1e-12)
    assert pool.pos[0] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert pool.vel[0] == pytest.approx((0.0, 0.0), abs=1e-12)


def test_two_non_autonomous_agents_stay_overlapping(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _contact_pool(pool_empty, agent_factory, [(0.0, 0.0), (0.3, 0.0)], [False, False])

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert pool.pos[0] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert pool.pos[1] == pytest.approx((0.3, 0.0), abs=1e-12)


def test_coincident_agents_split_apart(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _contact_pool(pool_empty, agent_factory, [(1.0, 1.0), (1.0, 1.0)], [True, True])

    WallProjectionResolver(margin=_MARGIN).resolve(pool)

    assert np.all(np.isfinite(pool.pos[:2]))
    assert np.hypot(*(pool.pos[0] - pool.pos[1])) == pytest.approx(0.5, abs=1e-12)


def test_walls_win_over_agent_contact(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((-2.0, 0.0), (2.0, 0.0))])
    pool = _contact_pool(pool_empty, agent_factory, [(0.0, 0.27), (0.0, 0.6)], [True, True])

    resolver.resolve(pool)

    assert pool.pos[0, 1] >= 0.25 + _MARGIN - 1e-12


def test_crowd_overlap_converges_over_ticks_and_is_deterministic(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    rng = np.random.default_rng(3)
    positions = [(float(x), float(y)) for x, y in rng.uniform(0.0, 10.0, size=(200, 2))]

    def worst_overlap(pool: AgentPool) -> float:
        p = pool.pos[: pool.n]
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        np.fill_diagonal(d, np.inf)
        return float(0.5 - d.min())

    def run(ticks: int) -> AgentPool:
        pool = _contact_pool(pool_empty, agent_factory, positions, [True] * 200)
        resolver = WallProjectionResolver(margin=_MARGIN)
        for _ in range(ticks):
            resolver.resolve(pool)
        return pool

    overlaps = [worst_overlap(run(ticks)) for ticks in range(5)]
    again = run(4)

    assert overlaps[0] > 0.4
    assert all(later < earlier for earlier, later in zip(overlaps, overlaps[1:], strict=False))
    assert overlaps[4] < 0.05
    assert np.array_equal(run(4).pos, again.pos)
    assert np.array_equal(run(4).vel, again.vel)


def test_far_robot_leaves_crowd_separation_unchanged(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    rng = np.random.default_rng(3)
    positions = [(float(x), float(y)) for x, y in rng.uniform(0.0, 3.0, size=(40, 2))]

    def separated(robot: bool) -> dict[int, tuple[float, float, float, float]]:
        pool = _contact_pool(pool_empty, agent_factory, positions, [True] * len(positions))
        if robot:
            idx = pool.add_agent(agent_factory(-1, x=50.0, y=50.0))
            pool.agent_radius[idx] = 0.6
            pool.policy_idx[idx] = -1
        WallProjectionResolver(margin=_MARGIN).resolve(pool)
        return {int(pool.agent_ids[i]): (*pool.pos[i], *pool.vel[i]) for i in range(pool.n) if pool.agent_ids[i] > 0}

    assert separated(robot=True) == separated(robot=False)
