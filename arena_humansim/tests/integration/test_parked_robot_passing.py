"""Walkers that cannot sidestep get past a robot parked on their line: they neither wait for it nor get turned back at it."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg

PERSON = 1
TICKS = 2000


def _parked_robot(tick: int, dt: float) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    ms = round(tick * dt * 1000)
    msg.header.stamp.sec, msg.header.stamp.nanosec = ms // 1000, ms % 1000 * 1_000_000
    robot = AgentStateMsg(agent_id=424242, radius=0.3)
    robot.pose = Pose2DMsg(x=3.0, y=0.0, theta=math.pi)
    msg.agents.append(robot)
    return msg


@pytest.mark.parametrize("agent_type", ["patient_walking_frame", "elder_rollator", "blind_white_cane"])
def test_a_walker_gets_past_a_robot_parked_on_its_line(manager_factory: Callable[..., AgentManager], agent_type: str) -> None:
    scenario = ScenarioConfig(
        name="parked_robot_passing",
        simulation=SimulationParams(seed=3, dt=0.05, bt_tick_interval=5, max_ticks=TICKS),
        modules=ModuleConfig(global_planner="navmesh"),
        agents=[AgentConfig(agent_id=PERSON, agent_type=agent_type, spawn_pose=Pose2D(x=0.0, y=0.0, theta=0.0), goal_sequence=[Pose2D(x=8.0, y=0.0)])],
    )
    mgr = manager_factory(scenario, node_name=f"test_parked_robot_{agent_type}")
    for tick in range(TICKS):
        mgr._world_state_callback(_parked_robot(tick, mgr._dt))
        mgr.tick()
        if mgr._agents[PERSON].state.pose.x > 6.0:
            return
    pytest.fail(f"stuck at x={mgr._agents[PERSON].state.pose.x:.2f} behind the robot at x=3")
