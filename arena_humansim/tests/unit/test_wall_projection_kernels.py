from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.collision.wall_projection import (
    WallProjectionResolver,
    _contact,
    _nearest_ends,
    _resolve_kernel,
    _separate_kernel,
)
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.wall_grid import WallGrid

_MARGIN = 0.01


def _resolve(kernel: Callable[..., None], segments: list, pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, offset: np.ndarray, theta: np.ndarray) -> np.ndarray:
    grid = WallGrid(segments)
    corrected = np.empty(pos.shape[0], dtype=np.bool_)
    kernel(pos, vel, radius, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, _MARGIN, offset, theta, corrected)
    return corrected


def _resolve_both(segments: list, pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, offset: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jit_pos, jit_vel = pos.copy(), vel.copy()
    jit_corrected = _resolve(_resolve_kernel, segments, jit_pos, jit_vel, radius, offset, theta)
    corrected = _resolve(_resolve_kernel.py_func, segments, pos, vel, radius, offset, theta)
    assert np.array_equal(corrected, jit_corrected)
    assert np.array_equal(pos, jit_pos)
    assert np.array_equal(vel, jit_vel)
    return corrected, pos, vel


def _separate_both(pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, movable: np.ndarray, pairs: np.ndarray, offset: np.ndarray, theta: np.ndarray) -> None:
    jit_pos, jit_vel = pos.copy(), vel.copy()
    _separate_kernel(jit_pos, jit_vel, radius, movable, pairs, offset, theta)
    _separate_kernel.py_func(pos, vel, radius, movable, pairs, offset, theta)
    assert np.array_equal(pos, jit_pos)
    assert np.array_equal(vel, jit_vel)


def _pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], positions: list[tuple[float, float]]) -> AgentPool:
    pool = pool_empty(capacity=max(2, len(positions)))
    for i, (x, y) in enumerate(positions):
        pool.add_agent(agent_factory(i + 1, x=x, y=y))
        pool.agent_radius[i] = 0.25
        pool.policy_idx[i] = 0
    return pool


@pytest.mark.parametrize(
    ("segment", "point", "expected"),
    [
        (((0.0, 0.0), (2.0, 0.0)), (1.0, 0.5), (0.5, 0.0, 1.0)),
        (((0.0, 0.0), (2.0, 0.0)), (3.0, 0.0), (1.0, 1.0, 0.0)),
        (((0.0, 0.0), (2.0, 0.0)), (-0.6, -0.8), (1.0, -0.6, -0.8)),
        (((1.0, 1.0), (1.0, 1.0)), (1.0, 2.0), (1.0, 0.0, 1.0)),
        (((0.0, 0.0), (2.0, 0.0)), (0.5, 0.0), (0.0, 0.0, 0.0)),
    ],
    ids=["interior", "past_end", "before_start", "zero_length", "on_wall"],
)
def test_contact_returns_distance_and_outward_unit_normal(segment: tuple, point: tuple[float, float], expected: tuple[float, float, float]) -> None:
    seg = np.array([[*segment[0], *segment[1]]], dtype=np.float64)
    got = _contact.py_func(seg, 0, *point)
    assert got == pytest.approx(expected, abs=1e-12)
    assert got == _contact(seg, 0, *point)


def test_resolve_kernel_pushes_disc_to_threshold_and_cancels_inward_velocity() -> None:
    pos = np.array([[1.0, 0.1], [1.0, 3.0]])
    vel = np.array([[0.5, -2.0], [0.0, -1.0]])
    corrected, pos, vel = _resolve_both([((0.0, 0.0), (2.0, 0.0))], pos, vel, np.array([0.3, 0.3]), np.zeros(2), np.zeros(2))
    assert corrected.tolist() == [True, False]
    assert pos[0] == pytest.approx((1.0, 0.31), abs=1e-12)
    assert vel[0] == pytest.approx((0.5, 0.0), abs=1e-12)
    assert pos[1].tolist() == [1.0, 3.0]
    assert vel[1].tolist() == [0.0, -1.0]


def test_resolve_kernel_keeps_velocity_already_leaving_the_wall() -> None:
    pos = np.array([[1.0, 0.1]])
    vel = np.array([[0.3, 0.7]])
    corrected, pos, vel = _resolve_both([((0.0, 0.0), (2.0, 0.0))], pos, vel, np.array([0.3]), np.zeros(1), np.zeros(1))
    assert corrected.tolist() == [True]
    assert vel[0].tolist() == [0.3, 0.7]


def test_resolve_kernel_capsule_pushes_the_penetrating_end() -> None:
    pos = np.array([[0.1, 0.0]])
    vel = np.array([[1.0, 0.5]])
    corrected, pos, vel = _resolve_both([((0.5, -2.0), (0.5, 2.0))], pos, vel, np.array([0.25]), np.array([0.2]), np.zeros(1))
    assert corrected.tolist() == [True]
    assert pos[0] == pytest.approx((0.04, 0.0), abs=1e-12)
    assert vel[0] == pytest.approx((0.0, 0.5), abs=1e-12)


def test_resolve_kernel_capsule_clear_of_walls_is_untouched() -> None:
    pos = np.array([[-1.0, 0.0]])
    vel = np.array([[1.0, 0.5]])
    corrected, pos, vel = _resolve_both([((0.5, -2.0), (0.5, 2.0))], pos, vel, np.array([0.25]), np.array([0.2]), np.zeros(1))
    assert corrected.tolist() == [False]
    assert pos[0].tolist() == [-1.0, 0.0]
    assert vel[0].tolist() == [1.0, 0.5]


@pytest.mark.parametrize(("offset", "theta", "y"), [(0.0, 0.0, 0.305), (0.2, np.pi / 2, 0.505)], ids=["disc", "capsule"])
def test_resolve_kernel_more_walls_than_query_buffer_still_sees_all(offset: float, theta: float, y: float) -> None:
    walls = [((0.0, 0.0), (2.0, 0.0))] * 70
    pos = np.array([[1.0, y]])
    vel = np.array([[0.0, -1.0]])
    corrected, pos, vel = _resolve_both(walls, pos, vel, np.array([0.3]), np.array([offset]), np.array([theta]))
    assert corrected.tolist() == [True]
    assert pos[0, 1] == pytest.approx(y + 70 * 0.005, abs=1e-9)
    assert vel[0] == pytest.approx((0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize("offset", [0.0, 0.2], ids=["disc", "capsule"])
def test_resolve_kernel_body_squeezed_between_walls_stays_centered_and_stops(offset: float) -> None:
    pos = np.array([[1.0, 0.0]])
    vel = np.array([[0.4, 1.0]])
    corrected, pos, vel = _resolve_both([((0.0, -0.2), (2.0, -0.2)), ((0.0, 0.2), (2.0, 0.2))], pos, vel, np.array([0.3]), np.array([offset]), np.zeros(1))
    assert corrected.tolist() == [True]
    assert pos[0] == pytest.approx((1.0, 0.0), abs=1e-12)
    assert vel[0] == pytest.approx((0.4, 0.0), abs=1e-12)


def test_resolve_kernel_capsule_parallel_to_wall_lands_on_threshold() -> None:
    pos = np.array([[1.0, 0.3]])
    vel = np.zeros((1, 2))
    corrected, pos, _ = _resolve_both([((0.0, 0.0), (2.0, 0.0))], pos, vel, np.array([0.3]), np.array([0.2]), np.zeros(1))
    assert corrected.tolist() == [True]
    assert pos[0, 1] == pytest.approx(0.3 + _MARGIN, abs=1e-12)


def test_resolve_kernel_matches_compiled_on_maze_scene() -> None:
    walls = [((float(x0), float(y0)), (float(x1), float(y1))) for (x0, y0), (x1, y1) in generate_maze(6, seed=3).walls]
    rng = np.random.default_rng(4)
    n = 40
    pos = 2.0 * rng.integers(0, 6, size=(n, 2)) + rng.uniform(-0.4, 0.4, size=(n, 2))
    vel = rng.normal(0.0, 1.0, size=(n, 2))
    radius = rng.uniform(0.2, 0.45, n)
    offset = np.where(np.arange(n) % 2 == 0, 0.0, rng.uniform(0.05, 0.3, n))
    theta = rng.uniform(-np.pi, np.pi, n)
    corrected, _, _ = _resolve_both(walls, pos, vel, radius, offset, theta)
    assert corrected.any()
    assert not corrected.all()


def test_nearest_ends_picks_the_closest_pair_of_capsule_ends() -> None:
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    offset = np.array([0.5, 0.5])
    assert _nearest_ends.py_func(pos, offset, np.zeros(2), 0, 1) == pytest.approx((-1.0, 0.0), abs=1e-12)
    crossed = np.array([0.0, np.pi / 2])
    assert _nearest_ends.py_func(pos, offset, crossed, 0, 1) == pytest.approx((-1.5, 0.5), abs=1e-12)
    assert _nearest_ends.py_func(pos, offset, crossed, 1, 0) == pytest.approx((1.5, -0.5), abs=1e-12)


def test_separate_kernel_splits_capsule_ends_and_stops_closing() -> None:
    pos = np.array([[0.0, 0.0], [1.3, 0.0]])
    vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _separate_both(pos, vel, np.full(2, 0.25), np.ones(2, dtype=np.bool_), np.array([[0, 1]], dtype=np.int64), np.full(2, 0.5), np.zeros(2))
    assert pos == pytest.approx(np.array([[-0.1, 0.0], [1.4, 0.0]]), abs=1e-12)
    assert vel == pytest.approx(np.zeros((2, 2)), abs=1e-12)


def test_separate_kernel_coincident_discs_split_along_x() -> None:
    pos = np.array([[1.0, 1.0], [1.0, 1.0]])
    vel = np.zeros((2, 2))
    _separate_both(pos, vel, np.full(2, 0.25), np.ones(2, dtype=np.bool_), np.array([[0, 1]], dtype=np.int64), np.zeros(2), np.zeros(2))
    assert pos == pytest.approx(np.array([[1.25, 1.0], [0.75, 1.0]]), abs=1e-12)


def test_separate_kernel_moves_only_movable_agents_and_keeps_separating_velocity() -> None:
    pos = np.array([[0.0, 0.0], [0.3, 0.0], [5.0, 0.0], [5.1, 0.0], [9.0, 0.0], [9.9, 0.0]])
    vel = np.array([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    movable = np.array([True, True, False, False, True, True])
    pairs = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    _separate_both(pos, vel, np.full(6, 0.25), movable, pairs, np.zeros(6), np.zeros(6))
    assert pos[:2] == pytest.approx(np.array([[-0.1, 0.0], [0.4, 0.0]]), abs=1e-12)
    assert vel[:2].tolist() == [[-1.0, 0.0], [1.0, 0.0]]
    assert pos[2:].tolist() == [[5.0, 0.0], [5.1, 0.0], [9.0, 0.0], [9.9, 0.0]]
    assert vel[2:].tolist() == [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]


def test_separate_kernel_never_moves_an_agent_off_an_immovable_partner() -> None:
    pos = np.array([[0.0, 0.0], [0.3, 0.0], [5.0, 0.0], [5.3, 0.0]])
    vel = np.array([[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    movable = np.array([True, False, True, False])
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    _separate_both(pos, vel, np.full(4, 0.25), movable, pairs, np.zeros(4), np.zeros(4))
    assert pos.tolist() == [[0.0, 0.0], [0.3, 0.0], [5.0, 0.0], [5.3, 0.0]]
    assert vel == pytest.approx(np.array([[0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]), abs=1e-12)


def test_resolve_far_apart_agents_away_from_walls_reports_nothing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((-5.0, -2.0), (5.0, -2.0))])
    pool = _pool(pool_empty, agent_factory, [(0.0, 0.0), (3.0, 0.0)])
    before = pool.pos[:2].copy()
    assert resolver.resolve(pool) == set()
    assert np.array_equal(pool.pos[:2], before)


def test_resolve_single_agent_without_walls_reports_nothing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    pool = _pool(pool_empty, agent_factory, [(0.0, 0.0)])
    assert resolver.resolve(pool) == set()
    assert resolver.resolve(pool_empty(capacity=2)) == set()


def test_resolve_reports_only_agents_pushed_off_a_wall(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls([((-5.0, 0.0), (5.0, 0.0))])
    pool = _pool(pool_empty, agent_factory, [(0.0, 0.1), (3.0, 1.0)])
    assert resolver.resolve(pool) == {1}
    assert pool.pos[0] == pytest.approx((0.0, 0.26), abs=1e-12)


def test_resolve_separates_overlapping_capsules_in_the_pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    resolver = WallProjectionResolver(margin=_MARGIN)
    pool = _pool(pool_empty, agent_factory, [(0.0, 0.0), (1.3, 0.0)])
    pool.axial_offset[:2] = 0.5
    assert resolver.resolve(pool) == set()
    assert pool.pos[:2] == pytest.approx(np.array([[-0.1, 0.0], [1.4, 0.0]]), abs=1e-12)
