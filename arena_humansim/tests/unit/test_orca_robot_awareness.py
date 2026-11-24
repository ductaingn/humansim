from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.local_planner.orca import ORCAPlanner
from arena_humansim.utils.types import Pose2D


def test_orca_blind_pedestrian_ignores_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = pool_empty(capacity=8)
    planner = ORCAPlanner()

    # Agent 1: Human at (0, 0), goal at (5, 0), blind to robot
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_blind[0] = True
    pool.desired_vel[0] = 1.0

    # Agent 2: Robot at (1.0, 0.0) right in front
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0

    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})

    planner.compute_pool(pool, dt=0.1)

    # Blind pedestrian keeps preferred forward velocity directly toward goal without swerving
    assert pool.vel[0, 0] > 0.95
    assert abs(pool.vel[0, 1]) < 1e-4


def test_orca_aware_pedestrian_swerves_or_decelerates_for_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = pool_empty(capacity=8)
    planner = ORCAPlanner()

    # Agent 1: Human at (0, 0), goal at (5, 0), fully aware of robot
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.desired_vel[0] = 1.0

    # Agent 2: Robot at (1.0, 0.0) blocking path
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0

    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})

    planner.compute_pool(pool, dt=0.1)

    # Aware pedestrian either swerves (lateral velocity != 0) or decelerates
    assert abs(pool.vel[0, 1]) > 1e-3 or pool.vel[0, 0] < 0.9


def test_orca_default_awareness_preserves_neighbor_dist(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = pool_empty(capacity=8)
    planner = ORCAPlanner()

    # Agent 1: Human at (0, 0), vision_range=3.0, default awareness (no robot range offset)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.vision_range[0] = 3.0
    pool.desired_vel[0] = 1.0

    # Agent 2: Robot at (4.0, 0.0) -- beyond vision_range (3.0m) but within ORCA neighbor_dist (5.0m)
    pool.add_agent(agent_factory(agent_id=2, x=4.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0

    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=4.0, y=0.0)})
    planner.compute_pool(pool, dt=0.1)

    # No range offset means no distance cut, so the robot at 4 m stays in the ORCA constraints
    assert pool.vel[0, 0] < 1.0


def test_orca_robot_range_offset_drops_robots_beyond_the_shrunk_range(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = pool_empty(capacity=8)
    planner = ORCAPlanner()

    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.vision_range[0] = 3.0
    pool.robot_range_offset[0] = -2.5
    pool.desired_vel[0] = 1.0

    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0

    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})
    planner.compute_pool(pool, dt=0.1)

    # The robot range shrinks to 0.5 m, so the robot 1.0 m ahead is ignored
    assert pool.vel[0, 0] > 0.95
    assert abs(pool.vel[0, 1]) < 1e-4


def test_orca_robot_repulsion_gain_widens_the_clearance_around_the_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    def speed_toward_goal(gain: float) -> float:
        pool = pool_empty(capacity=8)
        planner = ORCAPlanner()
        pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
        pool.kind[0] = KIND_HUMAN
        pool.robot_repulsion_gain[0] = gain
        pool.desired_vel[0] = 1.0
        pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
        pool.kind[1] = KIND_ROBOT
        pool.desired_vel[1] = 0.0
        pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})
        planner.compute_pool(pool, dt=0.1)
        return float(pool.vel[0, 0])

    assert speed_toward_goal(2.0) < speed_toward_goal(1.0)


def test_orca_robot_reaction_gain_slows_down_earlier_for_higher_reaction_gain(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    def speed_toward_goal(reaction_gain: float) -> float:
        pool = pool_empty(capacity=8)
        planner = ORCAPlanner()
        pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
        pool.kind[0] = KIND_HUMAN
        pool.robot_reaction_gain[0] = reaction_gain
        pool.desired_vel[0] = 1.0
        pool.add_agent(agent_factory(agent_id=2, x=3.0, y=0.0))
        pool.kind[1] = KIND_ROBOT
        pool.desired_vel[1] = 0.0
        pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=3.0, y=0.0)})
        planner.compute_pool(pool, dt=0.1)
        return float(pool.vel[0, 0])

    assert speed_toward_goal(2.0) < speed_toward_goal(1.0)


def test_orca_robot_repulsion_gain_is_floored_at_085_of_the_robot_radius(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    def speed_toward_goal(gain: float) -> float:
        pool = pool_empty(capacity=8)
        planner = ORCAPlanner()
        pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
        pool.kind[0] = KIND_HUMAN
        pool.robot_repulsion_gain[0] = gain
        pool.desired_vel[0] = 1.0
        pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
        pool.kind[1] = KIND_ROBOT
        pool.desired_vel[1] = 0.0
        pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})
        planner.compute_pool(pool, dt=0.1)
        return float(pool.vel[0, 0])

    assert speed_toward_goal(0.0) == pytest.approx(speed_toward_goal(0.85))


def _head_on(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], robot_x: float) -> AgentPool:
    pool = pool_empty(capacity=8)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.desired_vel[0] = 1.0
    pool.add_agent(agent_factory(agent_id=2, x=robot_x, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0
    pool.set_goals({1: Pose2D(x=8.0, y=0.0), 2: Pose2D(x=robot_x, y=0.0)})
    return pool


def test_orca_not_yielding_pedestrian_keeps_its_course_at_the_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = _head_on(pool_empty, agent_factory, 1.0)
    pool.robot_yields[0] = False
    ORCAPlanner().compute_pool(pool, dt=0.1)
    assert pool.vel[0, 0] > 0.95
    assert abs(pool.vel[0, 1]) < 1e-4


def test_orca_hearing_and_cane_reach_extend_the_robot_cutoff_beyond_sight(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = _head_on(pool_empty, agent_factory, 1.0)
    pool.vision_range[0] = 0.0
    pool.robot_loudness = 0.0
    ORCAPlanner().compute_pool(pool, dt=0.1)
    assert pool.vel[0, 0] > 0.95, "no sight, a silent robot and no cane: nothing to avoid"

    pool = _head_on(pool_empty, agent_factory, 1.0)
    pool.vision_range[0] = 0.0
    pool.robot_loudness = 0.0
    pool.cane_reach[0] = 1.2
    ORCAPlanner().compute_pool(pool, dt=0.1)
    assert pool.vel[0, 0] < 0.95 or abs(pool.vel[0, 1]) > 1e-4, "the cane reaches the robot"

    pool = _head_on(pool_empty, agent_factory, 1.0)
    pool.vision_range[0] = 0.0
    pool.hearing_range[0] = 2.0
    ORCAPlanner().compute_pool(pool, dt=0.1)
    assert pool.vel[0, 0] > 0.95, "a standing robot is silent"

    pool = _head_on(pool_empty, agent_factory, 1.0)
    pool.vision_range[0] = 0.0
    pool.hearing_range[0] = 2.0
    pool.vel[1] = (-1.0, 0.0)
    ORCAPlanner().compute_pool(pool, dt=0.1)
    assert pool.vel[0, 0] < 0.95 or abs(pool.vel[0, 1]) > 1e-4, "a robot driving at 1 m/s is heard out to 2 m"


def test_orca_hearing_never_shortens_the_robot_range_of_a_seeing_pedestrian(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    plain = _head_on(pool_empty, agent_factory, 1.0)
    plain.vision_range[0] = 0.5
    ORCAPlanner().compute_pool(plain, dt=0.1)
    assert plain.vel[0, 0] < 0.95 or abs(plain.vel[0, 1]) > 1e-4, "full sight keeps the robot out to neighbor_dist"

    hearing = _head_on(pool_empty, agent_factory, 1.0)
    hearing.vision_range[0] = 0.5
    hearing.hearing_range[0] = 0.8
    ORCAPlanner().compute_pool(hearing, dt=0.1)
    np.testing.assert_array_equal(hearing.vel[:2], plain.vel[:2])
