"""The frame_id param stamps every published pose and wall."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from arena_humansim_msgs.msg import WorldGeometry
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile

from tests.ros._helpers import AddWalls, RosTestSystem, SpawnAgents, make_add_walls_request, make_spawn_request, make_system

pytestmark = pytest.mark.ros

FRAME = "env_3/map"


@pytest.fixture(scope="module")
def system(rclpy_context) -> Iterator[RosTestSystem]:
    sys_ = make_system(client_node_name="frame_id_client", parameter_overrides=[Parameter("frame_id", value=FRAME)])
    try:
        yield sys_
    finally:
        sys_.shutdown()


def test_agent_states_carry_the_configured_frame(system: RosTestSystem) -> None:
    system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 1.0, "y": 2.0}]))
    msg = system.wait_for_agent_states(timeout=20.0, tick=True)
    assert msg.header.frame_id == FRAME


def test_world_geometry_carries_the_configured_frame(system: RosTestSystem) -> None:
    received: list[WorldGeometry] = []
    system.client_node.create_subscription(WorldGeometry, "world_geometry", received.append, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    system.call(AddWalls, "add_walls", make_add_walls_request([("w_frame", (-1.0, 0.0), (1.0, 0.0))]))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not any("w_frame" in msg.wall_names for msg in received):
        system.executor.spin_once(timeout_sec=0.05)
    assert [msg.header.frame_id for msg in received if "w_frame" in msg.wall_names][-1:] == [FRAME]
