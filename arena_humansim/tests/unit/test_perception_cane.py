from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest
from arena_humansim.core.agents import BUILTIN_AGENTS
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import (
    AWARENESS_PRESETS,
    AgentType,
    PerceptionDist,
    sample_agent_type,
    sample_awareness,
    with_robot_awareness,
)
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.perception.default import DefaultPerception

DT = 0.1


def _neighbors(pool: AgentPool, idx: int) -> list[int]:
    return [int(i) for i in pool.neighbor_indices[int(pool.neighbor_indptr[idx]) : int(pool.neighbor_indptr[idx + 1])]]


def _cane_user(pool: AgentPool, agent_factory: Callable[..., BaseAgent], delay: float = 0.0) -> None:
    """A blind cane user at the origin facing +x: no sight, cane 1.2 m ahead and 1 m across."""
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.theta[0] = 0.0
    pool.vision_range[0] = 0.0
    pool.proximity_sense[0] = 0.0
    pool.cane_reach[0] = 1.2
    pool.cane_width[0] = 1.0
    pool.detection_delay[0] = delay


def _add(pool: AgentPool, agent_factory: Callable[..., BaseAgent], aid: int, x: float, y: float, kind: int, radius: float = 0.3) -> int:
    pool.add_agent(agent_factory(agent_id=aid, x=x, y=y))
    idx = pool.n - 1
    pool.kind[idx] = kind
    pool.agent_radius[idx] = radius
    return idx


def _pad(pool: AgentPool, agent_factory: Callable[..., BaseAgent], count: int) -> None:
    for k in range(count):
        _add(pool, agent_factory, 100 + k, 1000.0 + 10.0 * k, 1000.0, KIND_HUMAN)


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_a_blind_cane_user_senses_nothing_beyond_cane_reach(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    _cane_user(pool, agent_factory)
    far = _add(pool, agent_factory, 2, 2.0, 0.0, KIND_ROBOT)
    walker = _add(pool, agent_factory, 3, -1.0, 0.0, KIND_HUMAN)
    _pad(pool, agent_factory, far_humans)

    DefaultPerception().compute_pool(pool)
    assert far not in _neighbors(pool, 0)
    assert walker not in _neighbors(pool, 0), "hearing is not modelled: a walker behind is not sensed"

    pool.pos[far] = (1.4, 0.0)
    DefaultPerception().compute_pool(pool)
    assert far in _neighbors(pool, 0), "the robot's disc reaches the cane tip"


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_the_cane_touches_only_ahead_and_within_its_sweep(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    _cane_user(pool, agent_factory)
    beside = _add(pool, agent_factory, 2, 0.6, 0.9, KIND_ROBOT)
    behind = _add(pool, agent_factory, 3, -0.5, 0.0, KIND_ROBOT)
    ahead = _add(pool, agent_factory, 4, 0.8, 0.6, KIND_ROBOT)
    _pad(pool, agent_factory, far_humans)

    DefaultPerception().compute_pool(pool)
    neighbors = _neighbors(pool, 0)
    assert beside not in neighbors
    assert behind not in neighbors
    assert ahead in neighbors


def test_robot_blind_people_sense_no_robot_with_the_cane(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    _cane_user(pool, agent_factory)
    pool.robot_blind[0] = True
    robot = _add(pool, agent_factory, 2, 1.0, 0.0, KIND_ROBOT)

    DefaultPerception().compute_pool(pool)
    assert robot not in _neighbors(pool, 0)


def test_a_robot_enters_the_list_after_the_detection_delay_and_the_clock_restarts_when_it_is_lost(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    _cane_user(pool, agent_factory, delay=0.5)
    robot = _add(pool, agent_factory, 2, 1.0, 0.0, KIND_ROBOT)
    walker = _add(pool, agent_factory, 3, 0.8, 0.6, KIND_HUMAN)
    perception = DefaultPerception()

    listed = []
    for _ in range(8):
        perception.compute_pool(pool, dt=DT)
        listed.append(robot in _neighbors(pool, 0))
        assert walker in _neighbors(pool, 0), "the delay is for robots only"
    assert listed == [False] * 5 + [True] * 3

    pool.pos[robot] = (10.0, 0.0)
    perception.compute_pool(pool, dt=DT)
    pool.pos[robot] = (1.0, 0.0)
    perception.compute_pool(pool, dt=DT)
    assert robot not in _neighbors(pool, 0)


@pytest.mark.parametrize("small_n_threshold", [64, 1], ids=["dense", "kdtree"])
def test_the_detection_delay_restarts_after_a_tick_with_nobody_in_range(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], small_n_threshold: int) -> None:
    pool = pool_empty(capacity=8)
    observer = _add(pool, agent_factory, 1, 0.0, 0.0, KIND_HUMAN)
    pool.vision_range[observer] = 5.0
    pool.vision_fov[observer] = 360.0
    pool.detection_delay[observer] = 0.5
    robot = _add(pool, agent_factory, 2, 2.0, 0.0, KIND_ROBOT)
    perception = DefaultPerception(small_n_threshold=small_n_threshold)
    for _ in range(6):
        perception.compute_pool(pool, dt=DT)
    assert robot in _neighbors(pool, observer)

    pool.pos[robot] = (50.0, 0.0)
    perception.compute_pool(pool, dt=DT)
    pool.pos[robot] = (2.0, 0.0)
    perception.compute_pool(pool, dt=DT)
    assert robot not in _neighbors(pool, observer)


def test_dense_and_kdtree_paths_agree_with_cane_users_in_occlusion_free_crowds(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    rng = np.random.default_rng(7)
    pool = pool_empty(capacity=128)
    for aid in range(1, 81):
        _add(pool, agent_factory, aid, *rng.uniform(-6.0, 6.0, size=2), KIND_ROBOT if aid % 20 == 0 else KIND_HUMAN)
        pool.theta[pool.n - 1] = rng.uniform(-np.pi, np.pi)
    for idx in range(0, 80, 7):
        pool.vision_range[idx] = 0.0
        pool.proximity_sense[idx] = 0.0
        pool.cane_reach[idx] = 1.2
        pool.cane_width[idx] = 1.0

    DefaultPerception(small_n_threshold=1000).compute_pool(pool)
    dense = (pool.neighbor_indptr.copy(), pool.neighbor_indices.copy())
    DefaultPerception(small_n_threshold=1).compute_pool(pool)
    np.testing.assert_array_equal(dense[0], pool.neighbor_indptr)
    np.testing.assert_array_equal(dense[1], pool.neighbor_indices)


def test_not_yielding_people_still_sense_the_robot_but_drop_its_repulsion() -> None:
    perception = with_robot_awareness(
        sample_agent_type(AgentType(name="adult"), np.random.default_rng(0)).perception,
        sample_awareness(AWARENESS_PRESETS["not_yielding"], np.random.default_rng(0)),
    )
    assert perception.robot_yields is False
    assert perception.robot_blind is False
    assert perception.robot_repulsion_gain == 0.0


def test_declaring_the_new_perception_fields_leaves_every_other_sampled_value_unchanged() -> None:
    plain = AgentType(name="walker")
    with_cane = attrs.evolve(
        plain,
        perception=PerceptionDist(
            cane_reach={"mean": 1.2, "std": 0.05},
            cane_width={"mean": 1.0, "std": 0.1},
            detection_delay={"mean": 0.6, "std": 0.1},
        ),
    )
    a = sample_agent_type(plain, np.random.default_rng(42))
    b = sample_agent_type(with_cane, np.random.default_rng(42))
    assert attrs.evolve(b.perception, cane_reach=0.0, cane_width=0.0, detection_delay=0.0) == a.perception
    assert attrs.evolve(b, perception=a.perception) == a
    assert b.perception.cane_reach > 0.0 and b.perception.detection_delay > 0.0


def _notice_distance(agent_type: AgentType, seed: int = 3) -> float:
    """Distance at which a person of `agent_type`, facing +x, first lists a robot driving at them from 8 m at 1 m/s."""
    from typing import Any, cast

    from arena_humansim.utils.types import AgentState, Pose2D

    def agent(aid: int, x: float, params: object) -> BaseAgent:
        state = AgentState(agent_id=aid, pose=Pose2D(x=x, y=0.0, theta=0.0), velocity=(0.0, 0.0), desired_velocity=0.0)
        return BaseAgent(state=state, params=cast(Any, params), global_planner=cast(Any, None), local_planner=cast(Any, None), animation=cast(Any, None))

    rng = np.random.default_rng(seed)
    pool = AgentPool(capacity=4)
    pool.add_agent(agent(1, 0.0, sample_agent_type(agent_type, rng)))
    pool.add_agent(agent(2, 8.0, sample_agent_type(AgentType(name="robot"), rng)))
    pool.kind[1] = KIND_ROBOT
    pool.agent_radius[1] = 0.3
    pool.vel[1] = (-1.0, 0.0)
    perception = DefaultPerception()
    for _ in range(200):
        perception.compute_pool(pool, dt=DT)
        if 1 in _neighbors(pool, 0):
            return float(pool.pos[1, 0])
        pool.pos[1, 0] -= 1.0 * DT
    return 0.0


def test_in_a_head_on_encounter_the_cane_user_notices_the_robot_later_than_the_adult() -> None:
    builtin = BUILTIN_AGENTS
    adult = _notice_distance(builtin["adult"])
    cane = _notice_distance(builtin["blind_white_cane"])
    assert adult > 4.0, "the adult sees the robot at its vision range"
    assert 0.7 < cane < 0.95, "the cane finds the robot at 1.05 m plus its radius and lists it half a second later"
    seeds = range(10)
    rollator_mean = np.mean([_notice_distance(builtin["elder_rollator"], seed) for seed in seeds])
    adult_mean = np.mean([_notice_distance(builtin["adult"], seed) for seed in seeds])
    assert rollator_mean < adult_mean, "older adults see a robot later on average (4 m against 5 m), one draw can tie"


def test_the_per_agent_path_senses_a_robot_ahead_with_the_cane_and_not_one_behind() -> None:
    from typing import Any, cast

    from arena_humansim.utils.types import AgentState, BeliefState, Pose2D

    def agent(state: AgentState, params: object) -> BaseAgent:
        return BaseAgent(state=state, params=cast(Any, params), global_planner=cast(Any, None), local_planner=cast(Any, None), animation=cast(Any, None))

    rng = np.random.default_rng(5)
    observer = agent(AgentState(agent_id=1, pose=Pose2D(x=0.0, y=0.0, theta=0.0)), sample_agent_type(BUILTIN_AGENTS["blind_white_cane"], rng))
    robot_state = AgentState(agent_id=2, pose=Pose2D(x=0.8, y=0.0, theta=0.0), kind=KIND_ROBOT)
    states = {1: observer.state, 2: robot_state}
    perception = DefaultPerception()

    def senses(x: float) -> bool:
        robot_state.pose.x = x
        perception.prepare_tick(states)
        return robot_state in perception.compute(observer, states, cast(Any, None), BeliefState(agent_id=1)).observed_agents

    assert senses(0.8), "the robot 0.8 m ahead is within the 1.05 m cane"
    assert not senses(-0.8), "the cane does not reach behind"
