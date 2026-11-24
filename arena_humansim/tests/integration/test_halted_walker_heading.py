"""A walking-frame user held still for a robot keeps facing its way: the drift a wall pushes it at does not turn it, and
waiting does not count as being stuck, so the stall recovery does not turn it either."""

from __future__ import annotations

import math
from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams, WallConfig
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg

PERSON = 1
TICKS = 400


def _robot_pose(t: float) -> AgentStatesMsg:
    """Drives up the corridor at 0.4 m/s towards the person, swerving to the right while it passes."""
    y = 8.0 + 0.4 * t
    x = 10.0 + 0.9 * max(0.0, min(1.0, (y - 10.5) / 1.5, (17.5 - y) / 1.5))
    msg = AgentStatesMsg()
    ms = round(t * 1000)
    msg.header.stamp.sec, msg.header.stamp.nanosec = ms // 1000, ms % 1000 * 1_000_000
    robot = AgentStateMsg(agent_id=424242, radius=0.3)
    robot.pose = Pose2DMsg(x=x, y=y, theta=math.pi / 2)
    msg.agents.append(robot)
    return msg


def test_a_halted_walking_frame_user_keeps_its_heading(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(
        name="halted_walker_heading",
        simulation=SimulationParams(seed=3, dt=0.05, bt_tick_interval=5, max_ticks=TICKS),
        modules=ModuleConfig(global_planner="navmesh"),
        agents=[AgentConfig(agent_id=PERSON, agent_type="patient_walking_frame", spawn_pose=Pose2D(x=10.3, y=15.0, theta=-math.pi / 2), goal_sequence=[Pose2D(x=10.3, y=3.0)])],
        walls=[  # a corridor, its west wall 0.7 m from the person's left shoulder
            WallConfig(name="west", start=Pose2D(x=9.6, y=0.0), end=Pose2D(x=9.6, y=30.0)),
            WallConfig(name="east", start=Pose2D(x=11.9, y=0.0), end=Pose2D(x=11.9, y=30.0)),
        ],
    )
    mgr = manager_factory(scenario, node_name="test_halted_walker_heading")
    headings = []
    for tick in range(TICKS):
        mgr._world_state_callback(_robot_pose(tick * mgr._dt))
        mgr.tick()
        if mgr._pool.robot_halted[mgr._pool.idx(PERSON)]:
            headings.append(mgr._agents[PERSON].state.pose.theta)
    assert len(headings) > 100, "the robot never held the person still"
    spread = max(headings) - min(headings)
    assert spread < math.radians(5.0), f"turned {math.degrees(spread):.0f} deg while held still"
