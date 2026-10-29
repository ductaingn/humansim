from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena_humansim.core import interaction_classes
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.local_planner.force import SOCIAL, WALL
from arena_humansim.local_planner.neurosfm import ROBOT_GAIN, NeuroSFMPlanner, forward, load_weights

REFERENCE = Path(__file__).resolve().parent / "golden" / "neurosfm_reference.npz"
DT = 0.067
RADIUS = 0.3
MAX_SPEED = 1.34


def _attached(pos: np.ndarray, goals: np.ndarray, classes: tuple[int, ...] | None = None) -> tuple[AgentPool, NeuroSFMPlanner]:
    n = len(pos)
    pool = AgentPool(capacity=max(n, 2))
    planner = NeuroSFMPlanner()
    planner.attach(pool)
    pool.n = n
    pool.agent_ids[:n] = np.arange(1, n + 1)
    pool.pos[:n] = pos
    pool.goal_pos[:n] = goals
    pool.has_goal[:n] = True
    pool.desired_vel[:n] = 1.0
    pool.max_velocity[:n] = MAX_SPEED
    pool.max_acceleration[:n] = 10.0
    pool.agent_radius[:n] = RADIUS
    if classes is not None:
        pool.interaction_class[:n] = classes
    indptr = np.arange(0, n * (n - 1) + 1, n - 1, dtype=np.int32)
    indices = np.array([j for i in range(n) for j in range(n) if j != i], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    return pool, planner


def test_registry_creates_the_planner_with_only_its_two_gains_tunable() -> None:
    assert isinstance(LocalPlanner.create("neurosfm"), NeuroSFMPlanner)
    assert {k: d.mean for k, d in NeuroSFMPlanner.PARAM_DEFAULTS.items()} == {"drive_gain": 0.7, "social_gain": 1.0}


@pytest.mark.parametrize("net", [1, 2, 3, 4, 5])
def test_each_released_net_matches_the_reference_forward_pass(net: int) -> None:
    reference = np.load(REFERENCE)
    np.testing.assert_allclose(forward(load_weights(), f"net{net}", reference[f"in{net}"].astype(np.float64)), reference[f"out{net}"], rtol=1e-5, atol=1e-6)


def test_closed_loop_rollout_follows_the_reference_update_step_by_step() -> None:
    reference = np.load(REFERENCE)
    track = reference["loop_track"]
    pool, planner = _attached(track[0].astype(np.float64), reference["loop_goal"].astype(np.float64))
    for step in range(1, len(track)):
        planner.compute_pool(pool, dt=DT)
        pool.prev_vel[: pool.n] = pool.vel[: pool.n]
        pool.pos[: pool.n] += pool.vel[: pool.n] * DT
        np.testing.assert_allclose(pool.pos[: pool.n], track[step], atol=1e-4, err_msg=f"step {step}")


def test_a_robot_within_range_pushes_through_net5_and_one_beyond_does_not() -> None:
    weights = load_weights()
    for gap, expected in ((2.0, True), (3.0, False)):
        pool, planner = _attached(np.array([[0.0, 0.0], [gap, 0.0]]), np.array([[-5.0, 0.0], [gap, 5.0]]), (interaction_classes.HUMAN, interaction_classes.ROBOT))
        planner.compute_pool(pool, store_forces=True, dt=DT)
        assert planner._last_force_arrays is not None
        social = planner._last_force_arrays[2][0, SOCIAL]
        push = ROBOT_GAIN * forward(weights, "net5", np.array([[gap, -1.0, 0.0]]))[0] if expected else np.zeros(2)
        np.testing.assert_allclose(social, push, atol=1e-12)


def test_walls_push_only_on_contact() -> None:
    for y, touching in ((0.2, True), (0.5, False)):
        pool, planner = _attached(np.array([[0.0, y], [9.0, 9.0]]), np.array([[5.0, y], [9.0, 5.0]]))
        planner.set_walls([((-5.0, 0.0), (5.0, 0.0))])
        planner.compute_pool(pool, store_forces=True, dt=DT)
        assert planner._last_force_arrays is not None
        wall = planner._last_force_arrays[2][0, WALL]
        assert (wall[1] > 0.0 and wall[0] == pytest.approx(0.0)) if touching else np.allclose(wall, 0.0)
