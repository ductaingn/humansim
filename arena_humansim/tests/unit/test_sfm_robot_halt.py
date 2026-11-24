from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim.core.agents.types import sample_agent_type
from arena_humansim.core.animation_kinds import AnimationType, locomotion_states
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.local_planner.sfm import HALT_SLOW_FACTOR, SFMPlanner
from arena_humansim.utils.types import Pose2D

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "agent_types"


def _pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], halt: bool, robot: tuple[float, float], robot_vel: tuple[float, float] = (-0.5, 0.0)) -> tuple[AgentPool, SFMPlanner]:
    """A walker at the origin heading +x to (10, 0), and a robot it perceives, driving at robot_vel."""
    pool = pool_empty(capacity=8)
    planner = SFMPlanner()
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.theta[0] = 0.0
    pool.robot_halt[0] = halt
    pool.policy_idx[0] = 0
    pool.add_agent(agent_factory(agent_id=2, x=robot[0], y=robot[1]))
    pool.kind[1] = KIND_ROBOT
    pool.policy_idx[1] = 0
    pool.vel[1] = robot_vel
    pool.set_goals({1: Pose2D(x=10.0, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 1], dtype=np.int32), np.array([1], dtype=np.int32))
    return pool, planner


def _target_speed(pool: AgentPool, planner: SFMPlanner) -> tuple[float, np.ndarray]:
    """The speed the attraction pulls towards along +x, and the robot repulsion on the walker."""
    f_att, f_rep, _, _ = planner._compute_forces_pool(pool)
    relax = planner._relaxation_time[0]
    return float(f_att[0, 0] * relax + pool.vel[0, 0]), f_rep[0]


def test_a_halting_walker_stops_for_a_robot_ahead_without_steering(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory, halt=True, robot=(0.8, 0.3))
    target, f_rep = _target_speed(pool, planner)
    assert target == pytest.approx(0.0)
    assert np.allclose(f_rep, 0.0)


def test_a_halting_walker_slows_for_a_moving_robot_ahead_but_out_of_its_path(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory, halt=True, robot=(2.0, 1.6))
    target, f_rep = _target_speed(pool, planner)
    assert target == pytest.approx(HALT_SLOW_FACTOR * pool.desired_vel[0])
    assert np.allclose(f_rep, 0.0)


def test_a_halting_walker_steers_around_a_robot_that_stands_still(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory, halt=True, robot=(0.8, 0.3), robot_vel=(0.0, 0.0))
    target, f_rep = _target_speed(pool, planner)
    assert target == pytest.approx(pool.desired_vel[0]), "a parked robot may never move off, so it is not waited for"
    assert f_rep[1] < 0.0, "pushed away from the robot on its left, like any obstacle"
    assert not pool.robot_halted[0]


def test_a_halting_walker_slows_for_a_robot_behind(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory, halt=True, robot=(-0.8, 0.3))
    target, f_rep = _target_speed(pool, planner)
    assert target == pytest.approx(HALT_SLOW_FACTOR * pool.desired_vel[0])
    assert np.allclose(f_rep, 0.0)


@pytest.mark.parametrize(("robot", "halted"), [((0.8, 0.3), True), ((-0.8, 0.3), False)])
def test_a_walker_held_still_for_a_robot_ahead_stands_idle(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], robot: tuple[float, float], halted: bool) -> None:
    pool, planner = _pool(pool_empty, agent_factory, halt=True, robot=robot)
    planner._compute_forces_pool(pool)
    assert pool.robot_halted[0] is np.bool_(halted)
    state = locomotion_states(pool.vel[:2], pool.has_goal[:2], pool.desired_vel[:2], pool.robot_halted[:2])[0]
    assert state == (AnimationType.IDLE if halted else AnimationType.WALKING)  # at rest, a goal-holder otherwise reports walking


def test_an_avoiding_walker_keeps_its_speed_and_steers(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory, halt=False, robot=(0.8, 0.3))
    target, f_rep = _target_speed(pool, planner)
    assert target == pytest.approx(pool.desired_vel[0])
    assert f_rep[1] < 0.0  # pushed away from the robot on its left


@pytest.mark.parametrize(("name", "halt"), [("blind_white_cane", True), ("blind_rolling_cane", True), ("patient_walking_frame", True), ("adult", False), ("elder_rollator", False), ("patient_iv_pole", False)])
def test_the_cane_and_frame_types_halt_for_robots(name: str, halt: bool) -> None:
    sampled = sample_agent_type(load_agent_type_from_file(CONFIG_DIR / f"{name}.yaml"), np.random.default_rng(0))
    assert sampled.perception.robot_halt is halt


def test_an_unknown_robot_response_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nextends: adult\nperception: {robot_response: freeze}\n")
    with pytest.raises((ValueError, TypeError)):
        load_agent_type_from_file(bad)


@pytest.mark.parametrize("name", ["blind_white_cane", "blind_rolling_cane", "patient_walking_frame", "elder_rollator", "patient_iv_pole"])
def test_cane_and_aid_users_do_not_turn_to_look_around_when_standing(name: str) -> None:
    sampled = sample_agent_type(load_agent_type_from_file(CONFIG_DIR / f"{name}.yaml"), np.random.default_rng(0))
    assert sampled.idle_gaze_rate_hz == 0.0
