"""Engine side of the world-mechanism hooks: semantic stimuli as world state, external teleports, holding in a cabin."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("py_trees")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import NeedsState, NeedState

from tests.integration._helpers import build_manager

ATOM = "elev_a.cabin_door==open"


def _scenario(global_planner: str = "navmesh") -> ScenarioConfig:
    return ScenarioConfig(
        name="mechanism_hooks",
        simulation=SimulationParams(seed=3, dt=0.05, bt_tick_interval=1, max_ticks=10),
        modules=ModuleConfig(global_planner=global_planner),
    )


@pytest.fixture
def manager(rclpy_context: object, request: pytest.FixtureRequest) -> Iterator[AgentManager]:
    mgr = build_manager(_scenario(), node_name=f"test_mech_{request.node.name}"[:60])
    yield mgr
    mgr.destroy_node()


def _spawn(mgr: AgentManager, x: float, y: float, waypoints: list[tuple[float, float]] = (), name: str = "") -> int:
    from arena_humansim_msgs.msg import AgentState as AgentStateMsg
    from arena_humansim_msgs.msg import Waypoint, Waypoints
    from arena_humansim_msgs.srv import SpawnAgents
    from geometry_msgs.msg import Pose2D

    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.name = name
    msg.pose = Pose2D(x=x, y=y, theta=0.0)
    msg.desired_velocity = 1.2
    msg.agent_type = "adult"
    msg.waypoints = Waypoints(mode=Waypoints.MODE_ONCE, points=[Waypoint(pose=Pose2D(x=wx, y=wy, theta=0.0)) for wx, wy in waypoints])
    resp = mgr._spawn_agents_callback(SpawnAgents.Request(agents=[msg]), SpawnAgents.Response())
    assert resp.success
    return int(resp.spawned_ids[0])


def _notify(mgr: AgentManager, agent_id: int, stimulus: str, intensity: float) -> None:
    from arena_humansim_msgs.srv import NotifyStimulus

    resp = mgr._notify_stimulus_callback(NotifyStimulus.Request(agent_id=agent_id, stimulus=stimulus, intensity=intensity), NotifyStimulus.Response())
    assert resp.success


def _run_until_due(mgr: AgentManager) -> None:
    for _ in range(max((int(round(a.params.reaction_time / mgr._dt)) for a in mgr._agents.values()), default=0) + 2):
        mgr.tick()


def _door_need() -> NeedsState:
    return NeedsState(needs={ATOM: NeedState(value=0.0, decay_rate=0.0)})


def _pos(mgr: AgentManager, aid: int) -> tuple[float, float]:
    idx = mgr._pool.idx(aid)
    return float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1])


def test_zero_intensity_lowers_need_without_event(manager: AgentManager) -> None:
    aid = _spawn(manager, 0.0, 0.0)
    manager._agents[aid].needs = _door_need()
    _notify(manager, aid, ATOM, 1.0)
    _run_until_due(manager)
    assert manager._agents[aid].needs.needs[ATOM].value == pytest.approx(100.0)

    fired: list[str] = []
    orig = manager._event_bus.fire
    manager._event_bus.fire = lambda name, agent_id: (fired.append(name), orig(name, agent_id))  # type: ignore[method-assign]
    _notify(manager, aid, ATOM, 0.0)
    _run_until_due(manager)
    assert manager._agents[aid].needs.needs[ATOM].value == pytest.approx(0.0)
    assert fired == []


def test_broadcast_level_seeds_agents_spawned_later(manager: AgentManager) -> None:
    first = _spawn(manager, 0.0, 0.0)
    _notify(manager, -1, ATOM, 1.0)
    _notify(manager, first, "private", 1.0)
    assert manager._stimulus_levels == {ATOM: 1.0}

    import attrs
    from arena_humansim.core.agents import BUILTIN_AGENTS
    from arena_humansim.core.agents.types import NeedDist, ParamDist

    adult = BUILTIN_AGENTS["adult"]
    manager._agent_types["door_watcher"] = attrs.evolve(adult, name="door_watcher", needs={ATOM: NeedDist(initial=ParamDist(0.0, clip_low=0.0), decay_rate=ParamDist(0.0, clip_low=0.0))})

    from arena_humansim_msgs.msg import AgentState as AgentStateMsg
    from arena_humansim_msgs.srv import SpawnAgents
    from geometry_msgs.msg import Pose2D

    msg = AgentStateMsg(agent_id=0, pose=Pose2D(x=2.0, y=0.0), desired_velocity=1.0, agent_type="door_watcher")
    late = int(manager._spawn_agents_callback(SpawnAgents.Request(agents=[msg]), SpawnAgents.Response()).spawned_ids[0])
    assert manager._agents[late].needs.needs[ATOM].value == pytest.approx(100.0)

    _notify(manager, -1, ATOM, 0.0)
    msg.pose = Pose2D(x=4.0, y=0.0)
    later = int(manager._spawn_agents_callback(SpawnAgents.Request(agents=[msg]), SpawnAgents.Response()).spawned_ids[0])
    assert manager._agents[later].needs.needs[ATOM].value == pytest.approx(0.0)


def test_full_reset_forgets_broadcast_levels(manager: AgentManager) -> None:
    from arena_humansim_msgs.srv import ResetSimulation

    _spawn(manager, 0.0, 0.0)
    _notify(manager, -1, ATOM, 1.0)
    manager._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response())
    assert manager._stimulus_levels == {ATOM: 1.0}
    manager._reset_callback(ResetSimulation.Request(), ResetSimulation.Response())
    assert manager._stimulus_levels == {}


def _add_walls(mgr: AgentManager, walls: list[tuple[tuple[float, float], tuple[float, float]]]) -> None:
    from arena_humansim_msgs.srv import AddWalls
    from geometry_msgs.msg import Point32

    req = AddWalls.Request()
    for i, ((sx, sy), (ex, ey)) in enumerate(walls):
        req.names.append(f"w{i}")
        req.starts.append(Point32(x=sx, y=sy, z=0.0))
        req.ends.append(Point32(x=ex, y=ey, z=0.0))
    assert mgr._add_walls_callback(req, AddWalls.Response()).success


def _box(x0: float, y0: float, x1: float, y1: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


LEVEL_A = _box(0.0, 0.0, 8.0, 6.0)
LEVEL_B = _box(20.0, 0.0, 28.0, 6.0)
CABIN_A = (7.0, 3.0)
CABIN_B = (21.0, 3.0)
EXIT_B = (26.0, 3.0)


def _run(mgr: AgentManager, seconds: float) -> None:
    for _ in range(int(round(seconds / mgr._dt))):
        mgr.tick()


def test_unreachable_next_waypoint_makes_the_agent_leave_the_cabin(manager: AgentManager) -> None:
    """A next-level waypoint handed to the engine drags the agent off the cabin."""
    _add_walls(manager, [*LEVEL_A, *LEVEL_B])
    aid = _spawn(manager, *CABIN_A, waypoints=[CABIN_A, EXIT_B])
    _run(manager, 6.0)
    x, y = _pos(manager, aid)
    assert math.hypot(x - CABIN_A[0], y - CABIN_A[1]) > 0.5


def test_cabin_hold_then_teleport_continues_with_autonomy(manager: AgentManager) -> None:
    from arena_humansim_msgs.msg import Waypoint, Waypoints
    from arena_humansim_msgs.srv import SetWaypoints, TeleportAgents
    from geometry_msgs.msg import Pose2D

    _add_walls(manager, [*LEVEL_A, *LEVEL_B])
    aid = _spawn(manager, 2.0, 3.0, waypoints=[CABIN_A], name="env_0/rider")
    _run(manager, 12.0)
    x, y = _pos(manager, aid)
    assert math.hypot(x - CABIN_A[0], y - CABIN_A[1]) < 0.5
    _run(manager, 5.0)
    x, y = _pos(manager, aid)
    assert math.hypot(x - CABIN_A[0], y - CABIN_A[1]) < 0.5, "the agent must wait at its last (cabin) waypoint"

    policy = int(manager._pool.policy_idx[manager._pool.idx(aid)])
    radius = float(manager._pool.agent_radius[manager._pool.idx(aid)])
    resp = manager._teleport_agents_callback(TeleportAgents.Request(agent_id=[aid], pose=[Pose2D(x=CABIN_B[0], y=CABIN_B[1], theta=0.0)]), TeleportAgents.Response())
    assert resp.success
    idx = manager._pool.idx(aid)
    assert _pos(manager, aid) == pytest.approx(CABIN_B)
    assert int(manager._pool.policy_idx[idx]) == policy
    assert float(manager._pool.agent_radius[idx]) == pytest.approx(radius)

    req = SetWaypoints.Request(agent_id=aid, waypoints=Waypoints(mode=Waypoints.MODE_ONCE, points=[Waypoint(pose=Pose2D(x=EXIT_B[0], y=EXIT_B[1]))]))
    assert manager._set_waypoints_callback(req, SetWaypoints.Response()).success
    _run(manager, 10.0)
    x, y = _pos(manager, aid)
    assert math.hypot(x - EXIT_B[0], y - EXIT_B[1]) < 0.5
