from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import GOAL, INF, SOCIAL, WALL, near_walls, ray_circle_ttc, wall_point
from arena_humansim.local_planner.helbing import HelbingPlanner
from arena_humansim.utils.types import BeliefState, Pose2D
from arena_humansim.utils.wall_grid import WallGrid


def _grid_args(segments: list) -> tuple:
    grid = WallGrid(segments)
    return (grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny)


def _with_params(agent: BaseAgent, **params: float) -> BaseAgent:
    agent.params = attrs.evolve(agent.params, local_planner_params=params)
    return agent


def _no_neighbors(pool: AgentPool) -> None:
    pool.set_neighbor_csr(np.zeros(pool.n + 1, dtype=np.int32), np.empty(0, dtype=np.int32))


@pytest.fixture
def marker_publisher(rclpy_context: object) -> Iterator[object]:
    if rclpy_context is None:
        pytest.skip("rclpy unavailable")
    from arena_rclpy_mixins.Time import TimeNode

    from arena_humansim.core.viz import MarkerPublisher

    node = TimeNode("force_planner_markers_test")
    try:
        yield MarkerPublisher(node, "map")
    finally:
        node.destroy_node()


def test_near_walls_lists_walls_in_range_including_beyond_the_query_buffer() -> None:
    walls = [((float(k) * 0.01, -1.0), (float(k) * 0.01, 1.0)) for k in range(70)] + [((50.0, 50.0), (51.0, 50.0))]
    found = near_walls.py_func(_grid_args(walls), 0.3, 0.0, 1.0)
    assert found.tolist() == list(range(70))
    assert np.array_equal(found, near_walls(_grid_args(walls), 0.3, 0.0, 1.0))
    assert near_walls.py_func(_grid_args(walls), 50.5, 50.2, 0.5).tolist() == [70]


@pytest.mark.parametrize(
    ("point", "expected"),
    [((1.0, 2.0), (1.0, 0.0)), ((-3.0, 1.0), (0.0, 0.0)), ((5.0, -1.0), (2.0, 0.0))],
    ids=["interior", "before_start", "past_end"],
)
def test_wall_point_is_the_closest_point_on_the_segment(point: tuple[float, float], expected: tuple[float, float]) -> None:
    seg = np.array([[0.0, 0.0, 2.0, 0.0]])
    assert wall_point.py_func(seg, 0, *point) == pytest.approx(expected)


def test_wall_point_of_zero_length_wall_is_its_endpoint() -> None:
    seg = np.array([[1.5, -0.5, 1.5, -0.5]])
    assert wall_point.py_func(seg, 0, 4.0, 4.0) == (1.5, -0.5)


@pytest.mark.parametrize(
    ("direction", "center", "radius", "expected"),
    [
        ((0.0, 0.0), (0.2, 0.0), 0.5, 0.0),
        ((0.0, 0.0), (2.0, 0.0), 0.5, INF),
        ((1.0, 0.0), (5.0, 0.0), 1.0, 4.0),
        ((2.0, 0.0), (5.0, 0.0), 1.0, 2.0),
        ((1.0, 0.0), (5.0, 3.0), 1.0, INF),
        ((1.0, 0.0), (0.5, 0.0), 1.0, 0.0),
        ((1.0, 0.0), (-5.0, 0.0), 1.0, INF),
        ((1.0, 0.0), (5.0, 1.0), 1.0, 5.0),
    ],
    ids=["resting_inside", "resting_outside", "head_on", "faster_head_on", "miss", "inside", "behind", "tangent"],
)
def test_ray_circle_ttc_known_cases(direction: tuple[float, float], center: tuple[float, float], radius: float, expected: float) -> None:
    got = ray_circle_ttc.py_func(*direction, *center, radius)
    assert got == pytest.approx(expected)
    assert got == ray_circle_ttc(*direction, *center, radius)


def test_ray_circle_ttc_is_invariant_under_rotation() -> None:
    base = ray_circle_ttc.py_func(1.3, 0.0, 4.0, 0.4, 0.6)
    for angle in np.linspace(0.0, 2.0 * math.pi, 9):
        c, s = math.cos(angle), math.sin(angle)
        assert ray_circle_ttc.py_func(1.3 * c, 1.3 * s, 4.0 * c - 0.4 * s, 4.0 * s + 0.4 * c, 0.6) == pytest.approx(base)


def test_isolated_agent_relaxes_toward_preferred_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = pool_empty(capacity=2)
    planner.attach(pool)
    pool.add_agent(_with_params(agent_factory(agent_id=1), relaxation_time=0.25))
    pool.vel[0] = (0.2, 0.1)
    pool.store_prev_vel()
    pool.set_goals({1: Pose2D(x=0.0, y=10.0)})
    _no_neighbors(pool)
    planner.compute_pool(pool, store_forces=True, dt=0.1)
    pref = np.array([0.0, pool.desired_vel[0]])
    expected = np.array([0.2, 0.1]) + (pref - np.array([0.2, 0.1])) / 0.25 * 0.1
    assert pool.vel[0] == pytest.approx(expected)
    assert planner._last_force_arrays is not None
    ids, pos, out = planner._last_force_arrays
    assert ids.tolist() == [1]
    assert pos[0].tolist() == [0.0, 0.0]
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]
    assert out[0, WALL].tolist() == [0.0, 0.0]


def test_step_clamps_to_max_velocity_and_stops_agents_without_goal(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = pool_empty(capacity=3)
    planner.attach(pool)
    for k in range(3):
        pool.add_agent(_with_params(agent_factory(agent_id=k + 1, x=10.0 * k), relaxation_time=0.05))
    pool.vel[:3] = (1.0, 0.0)
    pool.store_prev_vel()
    pool.set_goals({1: Pose2D(x=100.0, y=0.0), 3: Pose2D(x=20.0, y=0.0)})
    _no_neighbors(pool)
    planner.compute_pool(pool, dt=1.0)
    assert np.hypot(*pool.vel[0]) == pytest.approx(pool.max_velocity[0])
    assert pool.vel[1].tolist() == [0.0, 0.0]
    assert pool.vel[2].tolist() == [0.0, 0.0]
    assert planner._last_force_arrays is None


def test_parameter_rows_follow_pool_growth_and_swap_remove(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = pool_empty(capacity=1)
    planner.attach(pool)
    pool.add_agent(_with_params(agent_factory(agent_id=1), mass=60.0))
    pool.add_agent(_with_params(agent_factory(agent_id=2), mass=90.0, friction=1.0))
    pool.add_agent(agent_factory(agent_id=3))
    assert pool.capacity >= 3
    assert planner._params.shape[0] == pool.capacity
    assert planner._params[:3, 1].tolist() == [60.0, 90.0, planner._defaults[1]]
    assert planner._params[1, 6] == 1.0
    pool.swap_remove(1)
    assert planner._params[0, 1] == planner._defaults[1]
    assert planner._params[1, 1] == 90.0


def test_compute_pool_without_attach_raises(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = pool_empty(capacity=2)
    pool.add_agent(agent_factory(agent_id=1))
    with pytest.raises(RuntimeError, match="never attached"):
        planner.compute_pool(pool)


def test_compute_pool_on_empty_pool_drops_stored_forces(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = pool_empty(capacity=2)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1))
    _no_neighbors(pool)
    planner.compute_pool(pool, store_forces=True)
    assert planner._last_force_arrays is not None
    pool.swap_remove(1)
    planner.compute_pool(pool, store_forces=True)
    assert planner._last_force_arrays is None


def test_stale_neighbor_table_is_treated_as_no_neighbors(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = pool_empty(capacity=2)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0))
    pool.add_agent(agent_factory(agent_id=2, x=0.3))
    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=-5.0, y=0.0)})
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    assert np.array_equal(planner._last_force_arrays[2][:, SOCIAL], np.zeros((2, 2)))
    assert np.all(planner._last_force_arrays[2][:, GOAL, 0] * [1, -1] > 0.0)


def test_compute_without_agents_is_empty() -> None:
    assert HelbingPlanner().compute([], {}) == {}


def test_compute_uses_only_perceived_pooled_neighbors(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    goals = {1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=-5.0, y=0.0)}
    a, b = agent_factory(agent_id=1, x=0.0), agent_factory(agent_id=2, x=0.45)
    alone_a = planner.compute([a], {1: goals[1]}, dt=0.05)[1]
    stranger = agent_factory(agent_id=77, x=0.1)
    a.belief = BeliefState(observed_agents=[a.state, stranger.state])
    b.belief = None
    blind = planner.compute([a, b], goals, dt=0.05)
    assert blind[1] == pytest.approx(alone_a)
    a.belief = BeliefState(observed_agents=[b.state])
    seen = planner.compute([a, b], goals, dt=0.05)
    assert seen[1][0] < blind[1][0]
    assert seen[2] == blind[2]


def test_publish_markers_draws_one_arrow_per_nonzero_force(marker_publisher: object, pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    planner.publish_markers(marker_publisher)
    assert marker_publisher._scene == {}
    planner.set_walls([((-5.0, -0.5), (5.0, -0.5))])
    pool = pool_empty(capacity=2)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=4, x=1.0, y=0.0))
    pool.set_goals({4: Pose2D(x=6.0, y=0.0)})
    _no_neighbors(pool)
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    planner.publish_markers(marker_publisher)
    assert set(marker_publisher._scene) == {("f_goal", 4), ("f_obstacle", 4)}
    out = planner._last_force_arrays[2]
    goal = marker_publisher._scene[("f_goal", 4)].points
    assert (goal[0].x, goal[0].y) == (1.0, 0.0)
    assert (goal[1].x, goal[1].y) == pytest.approx((1.0 + 0.3 * out[0, GOAL, 0], 0.3 * out[0, GOAL, 1]))
    wall = marker_publisher._scene[("f_obstacle", 4)].points
    assert wall[1].y > wall[0].y
