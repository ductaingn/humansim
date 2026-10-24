from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import attrs
import py_trees
import pytest
import yaml
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.teleport_zones import HOLD_RADIUS, WAIT_RADIUS, Phase
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.scenario import (
    AgentConfig,
    AgentTemplateModel,
    FlowScenarioConfig,
    ModuleConfig,
    RateKeyframeModel,
    ScenarioConfig,
    ShapeModel,
    SimulationParams,
    SinkAffinityModel,
    SinkScenarioConfig,
    SourceScenarioConfig,
    WallConfig,
    load_scenario,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import TeleportZone
from arena_humansim_msgs.srv import ResetSimulation, SetTeleportZones, SetZoneAccess, TeleportAgents
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.parameter import Parameter

ROOM = 6.0
ROOM_B_X = 20.0
SINK = Pose2D(x=ROOM_B_X + 1.0, y=3.0)
PLANNERS = {
    "navmesh": [],
    "dijkstra": [Parameter("global_planner.resolution", Parameter.Type.DOUBLE, 0.1)],
}


def _room(name: str, x0: float) -> list[WallConfig]:
    corners = [(x0, 0.0), (x0 + ROOM, 0.0), (x0 + ROOM, ROOM), (x0, ROOM)]
    return [WallConfig(name=f"{name}{i}", start=Pose2D(x=ax, y=ay), end=Pose2D(x=bx, y=by)) for i, ((ax, ay), (bx, by)) in enumerate(zip(corners, corners[1:] + corners[:1], strict=True))]


def _scenario(name: str, planner: str, riders: int = 1) -> ScenarioConfig:
    flow = FlowScenarioConfig(
        sources=[
            SourceScenarioConfig(
                pose=Pose2D(x=1.0, y=3.0),
                shape=ShapeModel(type="circle", radius=0.1),
                type="poisson",
                rate_profile=[RateKeyframeModel(t=0.0, rate=20.0)],
                max_concurrent=riders,
                max_total=riders,
                agent_template=AgentTemplateModel(desired_velocity_min=1.2, desired_velocity_max=1.2, agent_radius=0.25, sink_affinity=[SinkAffinityModel(sink_idx=0, weight=1.0)]),
            ),
        ],
        sinks=[SinkScenarioConfig(pose=SINK, shape=ShapeModel(type="circle", radius=0.5), absorption_radius=0.5)],
    )
    return ScenarioConfig(
        name=name,
        simulation=SimulationParams(seed=3, dt=0.05),
        modules=ModuleConfig(global_planner=planner),
        walls=_room("a", 0.0) + _room("b", ROOM_B_X),
        flow=flow,
    )


def _zone(name: str, destination: str, x0: float, y: float, is_open: bool = False, half: float = 0.6, delay: float = -1.0) -> TeleportZone:
    z = TeleportZone()
    z.name = name
    z.destinations = [destination] if destination else []
    z.hold = Pose2DMsg(x=x0 + 4.6 + half, y=y, theta=0.0)
    z.entry = Pose2DMsg(x=x0 + 4.0, y=y, theta=0.0)
    z.wait = Pose2DMsg(x=x0 + 3.4, y=y, theta=0.0)
    z.footprint = [Point32(x=x0 + px, y=y + py, z=0.0) for px, py in ((4.6, -half), (4.6 + 2 * half, -half), (4.6 + 2 * half, half), (4.6, half))]
    z.open = is_open
    z.delay = delay
    return z


def _set_zones(mgr: AgentManager, zones: list[TeleportZone]) -> SetTeleportZones.Response:
    return mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=zones), SetTeleportZones.Response())


def _set_access(mgr: AgentManager, name: str, is_open: bool) -> SetZoneAccess.Response:
    return mgr._set_zone_access_callback(SetZoneAccess.Request(name=[name], open=[is_open]), SetZoneAccess.Response())


def _teleport(mgr: AgentManager, ids: list[int], poses: list[tuple[float, float]]) -> TeleportAgents.Response:
    request = TeleportAgents.Request(agent_id=ids, pose=[Pose2DMsg(x=x, y=y, theta=0.0) for x, y in poses])
    return mgr._teleport_agents_callback(request, TeleportAgents.Response())


def _tick_until(mgr: AgentManager, done: Callable[[], bool], limit: int) -> int:
    for n in range(limit):
        if done():
            return n
        mgr.tick()
    raise AssertionError(f"condition not met in {limit} ticks")


def _rider(mgr: AgentManager) -> int:
    _tick_until(mgr, lambda: bool(mgr._agents), 40)
    return next(iter(mgr._agents))


def _zone_pair(name: str, delay: float = -1.0) -> list[TeleportZone]:
    return [_zone(f"{name}_a", f"{name}_b", 0.0, 4.6, delay=delay), _zone(f"{name}_b", f"{name}_a", ROOM_B_X, 4.6, delay=delay)]


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_pedestrian_waits_at_closed_zone_holds_inside_and_reaches_sink_in_other_room_after_teleport(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    mgr = manager_factory(_scenario(f"zone_pass_{planner}", planner), node_name=f"test_zone_pass_{planner}", extra_params=PLANNERS[planner])
    assert _set_zones(mgr, _zone_pair("pair")).success
    aid = _rider(mgr)
    pool = mgr._pool

    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    assert mgr._zones.transfers[aid].zone == "pair_a"
    for _ in range(60):
        mgr.tick()
    i = pool.idx(aid)
    assert mgr._zones.transfers[aid].phase == Phase.WAIT
    assert math.hypot(pool.pos[i, 0] - 3.4, pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3
    assert math.hypot(*pool.vel[i]) < 0.05

    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: mgr._zones.transfers[aid].phase == Phase.HOLD, 200)
    for _ in range(40):
        mgr.tick()
    i = pool.idx(aid)
    assert mgr._zones.transfers[aid].phase == Phase.HOLD
    assert math.hypot(*pool.vel[i]) < 0.05

    x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
    assert _teleport(mgr, [aid], [(x + ROOM_B_X, y)]).success
    assert aid not in mgr._zones.transfers
    i = pool.idx(aid)
    assert (pool.pos[i, 0], pool.pos[i, 1]) == (x + ROOM_B_X, y)
    assert not pool.vel[i].any()

    last: list[tuple[float, float]] = []

    def absorbed() -> bool:
        if aid not in mgr._agents:
            return True
        j = pool.idx(aid)
        last[:] = [(float(pool.pos[j, 0]), float(pool.pos[j, 1]))]
        assert aid not in mgr._zones.transfers
        return False

    _tick_until(mgr, absorbed, 400)
    assert math.hypot(last[0][0] - SINK.x, last[0][1] - SINK.y) < 1.0


def _riders(mgr: AgentManager, n: int) -> list[int]:
    _tick_until(mgr, lambda: len(mgr._agents) >= n, 200)
    return sorted(mgr._agents)[:n]


def test_second_rider_waits_clear_of_the_opening_while_the_only_spot_is_taken(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_queue", "navmesh", riders=2), node_name="test_zone_queue")
    assert _set_zones(mgr, _zone_pair("pair")).success
    assert len(mgr._zones.zones["pair_a"].slots) == 1
    first, second = _riders(mgr, 2)
    transfers = mgr._zones.transfers

    _tick_until(mgr, lambda: all(a in transfers and transfers[a].phase == Phase.WAIT for a in (first, second)), 600)
    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: any(transfers[a].phase == Phase.HOLD for a in (first, second)), 300)
    holder, queued = (first, second) if transfers[first].phase == Phase.HOLD else (second, first)
    for _ in range(40):
        mgr.tick()
    assert transfers[queued].phase == Phase.WAIT
    j = mgr._pool.idx(queued)
    assert math.hypot(mgr._pool.pos[j, 0] - 3.4, mgr._pool.pos[j, 1] - 4.6) <= WAIT_RADIUS + 0.3

    i = mgr._pool.idx(holder)
    assert _teleport(mgr, [holder], [(float(mgr._pool.pos[i, 0]) + ROOM_B_X, float(mgr._pool.pos[i, 1]))]).success
    _tick_until(mgr, lambda: transfers[queued].phase == Phase.HOLD, 300)


def test_riders_of_one_zone_hold_on_distinct_spots(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_spots", "navmesh", riders=2), node_name="test_zone_spots")
    zones = [_zone("big_a", "big_b", 0.0, 3.0, half=1.0), _zone("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]
    assert _set_zones(mgr, zones).success
    assert len(mgr._zones.zones["big_a"].slots) == 4
    riders = _riders(mgr, 2)
    transfers = mgr._zones.transfers

    _tick_until(mgr, lambda: all(a in transfers and transfers[a].phase == Phase.WAIT for a in riders), 600)
    assert _set_access(mgr, "big_a", True).success
    _tick_until(mgr, lambda: all(transfers[a].phase == Phase.HOLD for a in riders), 400)
    assert transfers[riders[0]].slot != transfers[riders[1]].slot


def test_holding_rider_pushed_out_of_an_open_zone_boards_again(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_reboard", "navmesh"), node_name="test_zone_reboard")
    assert _set_zones(mgr, _zone_pair("pair")).success
    aid = _rider(mgr)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: aid in transfers and transfers[aid].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)

    i = mgr._pool.idx(aid)
    mgr._pool.pos[i] = (3.9, 4.6)
    mgr.tick()
    assert transfers[aid].phase == Phase.BOARD
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_wait_timeout_moves_waiting_rider_to_the_next_zone_then_back_to_the_first(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    params = [*PLANNERS[planner], Parameter("zone_wait_timeout", Parameter.Type.DOUBLE, 1.0)]
    mgr = manager_factory(_scenario(f"zone_timeout_{planner}", planner), node_name=f"test_zone_timeout_{planner}", extra_params=params)
    zones = _zone_pair("a") + [_zone("b_a", "b_b", 0.0, 1.4), _zone("b_b", "b_a", ROOM_B_X, 1.4)]
    assert _set_zones(mgr, zones).success
    aid = _rider(mgr)

    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    assert mgr._zones.transfers[aid].zone == "a_a"
    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].zone == "b_a", 60)
    _tick_until(mgr, lambda: mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    i = mgr._pool.idx(aid)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 1.4) <= WAIT_RADIUS + 0.3
    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].zone == "a_a", 60)
    _tick_until(mgr, lambda: mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    i = mgr._pool.idx(aid)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_wait_timeout_at_the_only_zone_requeues_there_and_never_targets_the_other_room(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    params = [*PLANNERS[planner], Parameter("zone_wait_timeout", Parameter.Type.DOUBLE, 1.0)]
    mgr = manager_factory(_scenario(f"zone_requeue_{planner}", planner), node_name=f"test_zone_requeue_{planner}", extra_params=params)
    assert _set_zones(mgr, _zone_pair("only")).success
    aid = _rider(mgr)
    transfers = mgr._zones.transfers
    global_planner = mgr._agents[aid].global_planner

    _tick_until(mgr, lambda: aid in transfers and transfers[aid].phase == Phase.WAIT, 300)
    first = transfers[aid]
    requeued = False
    for _ in range(120):
        mgr.tick()
        transfer = transfers.get(aid)
        if transfer is not None:
            assert transfer.zone == "only_a"
            requeued = requeued or transfer is not first
        planned = global_planner._path_cache.get(aid)
        assert planned is None or planned[0][0] < ROOM
        subgoal = mgr._cached_intermediate_goals.get(aid)
        assert subgoal is None or subgoal.x < ROOM
        i = mgr._pool.idx(aid)
        assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3
    assert requeued


def _boarding_manager(manager_factory: Callable[..., AgentManager], name: str, delay: float) -> tuple[AgentManager, int]:
    mgr = manager_factory(_scenario(name, "navmesh"), node_name=f"test_{name}")
    zones = [_zone("hop_a", "hop_b", 0.0, 4.6, is_open=True, delay=delay), _zone("hop_b", "hop_a", ROOM_B_X, 4.6, delay=delay)]
    assert _set_zones(mgr, zones).success
    aid = _rider(mgr)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: aid in transfers and transfers[aid].phase == Phase.BOARD, 300)
    return mgr, aid


def _pos(mgr: AgentManager, aid: int) -> tuple[float, float]:
    i = mgr._pool.idx(aid)
    return float(mgr._pool.pos[i, 0]), float(mgr._pool.pos[i, 1])


def test_zone_delay_teleports_a_holding_rider_to_the_destination_at_the_same_offset_after_the_delay(manager_factory: Callable[..., AgentManager]) -> None:
    mgr, aid = _boarding_manager(manager_factory, "zone_delay_two", 2.0)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)

    held = 0
    before = _pos(mgr, aid)
    while aid in transfers:
        assert transfers[aid].phase == Phase.HOLD
        assert held <= 41
        before = _pos(mgr, aid)
        mgr.tick()
        held += 1
    assert abs(held - 40) <= 1
    x, y = _pos(mgr, aid)
    assert math.hypot(x - before[0] - ROOM_B_X, y - before[1]) < 0.05
    assert not mgr._pool.vel[mgr._pool.idx(aid)].any()


def test_zero_zone_delay_teleports_a_rider_on_its_first_holding_tick(manager_factory: Callable[..., AgentManager]) -> None:
    mgr, aid = _boarding_manager(manager_factory, "zone_delay_zero", 0.0)
    transfers = mgr._zones.transfers
    spot = mgr._zones.zones["hop_a"].slots[transfers[aid].slot]

    before, at_spot = _pos(mgr, aid), False
    for _ in range(200):
        before = _pos(mgr, aid)
        at_spot = math.hypot(before[0] - spot.x, before[1] - spot.y) <= HOLD_RADIUS
        mgr.tick()
        if aid not in transfers:
            break
        assert transfers[aid].phase == Phase.BOARD
        assert not at_spot
    else:
        raise AssertionError("rider never teleported")
    assert at_spot
    x, y = _pos(mgr, aid)
    assert math.hypot(x - before[0] - ROOM_B_X, y - before[1]) < 0.05


def test_negative_zone_delay_never_teleports_a_holding_rider(manager_factory: Callable[..., AgentManager]) -> None:
    mgr, aid = _boarding_manager(manager_factory, "zone_delay_never", -1.0)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)
    for _ in range(200):
        mgr.tick()
        assert transfers[aid].phase == Phase.HOLD
        assert _pos(mgr, aid)[0] < ROOM


def test_set_teleport_zones_replaces_and_clears_the_zone_set_and_survives_soft_reset(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_set", "navmesh"), node_name="test_zone_set")
    assert _set_zones(mgr, _zone_pair("x") + [_zone("solo", "", 0.0, 1.4, is_open=True)]).success
    assert sorted(mgr._zones.zones) == ["solo", "x_a", "x_b"]
    assert mgr._zones.open == {"x_a": False, "x_b": False, "solo": True}

    assert _set_zones(mgr, _zone_pair("y")).success
    assert sorted(mgr._zones.zones) == ["y_a", "y_b"]
    assert not _set_access(mgr, "x_a", True).success
    assert _set_access(mgr, "y_b", True).success
    assert mgr._zones.open["y_b"]

    rejected = _set_zones(mgr, [_zone("z", "nowhere", 0.0, 1.4), _zone("z", "", 0.0, 1.4)])
    assert not rejected.success
    assert "nowhere" in rejected.message and "'z'" in rejected.message
    assert sorted(mgr._zones.zones) == ["y_a", "y_b"]

    assert mgr._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response()).success
    assert sorted(mgr._zones.zones) == ["y_a", "y_b"]

    assert _set_zones(mgr, []).success
    assert mgr._zones.zones == {}

    assert _set_zones(mgr, _zone_pair("y")).success
    assert mgr._reset_callback(ResetSimulation.Request(soft=False), ResetSimulation.Response()).success
    assert mgr._zones.zones == {}


def test_teleport_agents_rejects_unknown_ids_and_moves_nobody(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_teleport", "navmesh"), node_name="test_zone_teleport")
    aid = _rider(mgr)
    i = mgr._pool.idx(aid)
    before = mgr._pool.pos[i].copy()

    response = _teleport(mgr, [aid, 999], [(2.0, 2.0), (3.0, 3.0)])
    assert not response.success
    assert "999" in response.message
    assert (mgr._pool.pos[i] == before).all()
    assert not _teleport(mgr, [aid], []).success
    assert _teleport(mgr, [aid], [(2.0, 2.0)]).success
    assert tuple(mgr._pool.pos[mgr._pool.idx(aid)]) == (2.0, 2.0)


def _go_to_scenario(tmp_path: Path, planner: str) -> ScenarioConfig:
    walls = [{"name": w.name, "start": {"x": w.start.x, "y": w.start.y}, "end": {"x": w.end.x, "y": w.end.y}} for w in _room("a", 0.0) + _room("b", ROOM_B_X)]
    raw = {
        "name": f"zone_go_to_{planner}",
        "simulation": {"seed": 42, "dt": 0.05, "bt_tick_interval": 1},
        "modules": {"global_planner": planner},
        "walls": walls,
        "agent_types": {
            "go_to_walker": {
                "extends": "adult",
                "mode": "behavior_tree",
                "initial_sequence": "walk",
                "sequences": {"walk": {"steps": {"to_other_room": {"kind": "go_to", "target_pose": {"x": SINK.x, "y": SINK.y}, "patience": {"mean": 120.0, "std": 0.0, "clip_low": 120.0, "clip_high": 120.0}}}}},
            },
        },
        "agents": [{"agent_id": 1, "agent_type": "go_to_walker", "spawn_pose": {"x": 1.0, "y": 3.0, "theta": 0.0}, "desired_velocity": 1.2}],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_scenario(str(path))


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_behavior_tree_go_to_rider_is_teleported_and_reaches_target_in_other_room(manager_factory: Callable[..., AgentManager], tmp_path: Path, planner: str) -> None:
    mgr = manager_factory(_go_to_scenario(tmp_path, planner), node_name=f"test_zone_go_to_{planner}", extra_params=PLANNERS[planner])
    assert _set_zones(mgr, _zone_pair("pair")).success
    bt = mgr._behavior_trees[1]
    pool = mgr._pool

    _tick_until(mgr, lambda: 1 in mgr._zones.transfers and mgr._zones.transfers[1].phase == Phase.WAIT, 300)
    assert mgr._zones.transfers[1].zone == "pair_a"
    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: mgr._zones.transfers[1].phase == Phase.HOLD, 200)
    i = pool.idx(1)
    assert _teleport(mgr, [1], [(float(pool.pos[i, 0]) + ROOM_B_X, float(pool.pos[i, 1]))]).success

    _tick_until(mgr, lambda: bt.root.status == py_trees.common.Status.SUCCESS, 600)
    assert 1 not in mgr._zones.transfers
    pose = mgr._agents[1].state.pose
    assert math.hypot(pose.x - SINK.x, pose.y - SINK.y) <= DISTANCE_TOLERANCE


def test_robots_with_a_goal_in_the_other_room_never_transfer(manager_factory: Callable[..., AgentManager]) -> None:
    goal = [Pose2D(x=SINK.x, y=SINK.y)]
    agents = [
        AgentConfig(agent_id=-1, kind=1, policy="sfm", agent_type="robot", spawn_pose=Pose2D(x=1.0, y=1.5), goal_sequence=goal, desired_velocity=1.0),
        AgentConfig(agent_id=-2, kind=1, policy="", agent_type="robot", spawn_pose=Pose2D(x=1.0, y=4.5), goal_sequence=goal, desired_velocity=1.0),
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=goal, desired_velocity=1.2),
    ]
    scenario = attrs.evolve(_scenario("zone_robots", "navmesh"), flow=None, agents=agents)
    mgr = manager_factory(scenario, node_name="test_zone_robots")
    assert _set_zones(mgr, _zone_pair("pair")).success
    assert {-1, -2, 1} <= set(mgr._agents)

    phases: set[Phase] = set()
    for _ in range(300):
        mgr.tick()
        assert not {-1, -2} & mgr._zones.transfers.keys()
        assert all(mgr._high_level_cmds[r].target_pose.x == SINK.x for r in (-1, -2))
        if 1 in mgr._zones.transfers:
            phases.add(mgr._zones.transfers[1].phase)
    assert Phase.WAIT in phases


def _robot_in_zone_manager(manager_factory: Callable[..., AgentManager], name: str) -> AgentManager:
    center = Pose2D(x=5.6, y=3.0)
    agents = [
        AgentConfig(agent_id=-1, kind=1, policy="", agent_type="robot", agent_radius=0.267, spawn_pose=center, goal_sequence=[center], desired_velocity=1.0),
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=[Pose2D(x=SINK.x, y=SINK.y)], desired_velocity=1.2, agent_radius=0.25),
    ]
    mgr = manager_factory(attrs.evolve(_scenario(name, "navmesh"), flow=None, agents=agents), node_name=f"test_{name}")
    zones = [_zone("big_a", "big_b", 0.0, 3.0, half=1.0), _zone("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]
    assert _set_zones(mgr, zones).success
    return mgr


def test_rider_waits_outside_while_a_robot_in_the_zone_covers_every_spot(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _robot_in_zone_manager(manager_factory, "zone_robot_inside")
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: 1 in transfers and transfers[1].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "big_a", True).success
    for _ in range(100):
        mgr.tick()
        assert transfers[1].phase == Phase.WAIT
    i = mgr._pool.idx(1)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 3.0) <= WAIT_RADIUS + 0.3

    assert _teleport(mgr, [-1], [(5.6 + ROOM_B_X, 3.0)]).success
    _tick_until(mgr, lambda: transfers[1].phase == Phase.HOLD, 300)


def test_boarding_rider_steps_back_to_waiting_when_a_robot_takes_the_zone(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _robot_in_zone_manager(manager_factory, "zone_robot_enters")
    assert _teleport(mgr, [-1], [(5.6 + ROOM_B_X, 3.0)]).success
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: 1 in transfers and transfers[1].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "big_a", True).success
    _tick_until(mgr, lambda: transfers[1].phase == Phase.BOARD, 50)

    assert _teleport(mgr, [-1], [(5.6, 3.0)]).success
    mgr.tick()
    assert transfers[1].phase == Phase.WAIT
    assert transfers[1].slot == -1


def test_teleport_zones_from_scenario_yaml_reach_the_manager_open_with_zero_delay(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    walls = [{"name": w.name, "start": {"x": w.start.x, "y": w.start.y}, "end": {"x": w.end.x, "y": w.end.y}} for w in _room("a", 0.0) + _room("b", ROOM_B_X)]
    footprint = [{"x": 4.6, "y": 4.0}, {"x": 5.8, "y": 4.0}, {"x": 5.8, "y": 5.2}, {"x": 4.6, "y": 5.2}]
    raw = {
        "name": "zone_yaml",
        "simulation": {"seed": 3, "dt": 0.05},
        "modules": {"global_planner": "navmesh"},
        "walls": walls,
        "teleport_zones": [
            {"name": "up", "destinations": ["down"], "entry": {"x": 4.0, "y": 4.6}, "wait": {"x": 3.4, "y": 4.6}, "hold": {"x": 5.2, "y": 4.6, "theta": 1.5}, "footprint": footprint},
            {"name": "down", "entry": {"x": ROOM_B_X + 4.0, "y": 4.6}, "wait": {"x": ROOM_B_X + 3.4, "y": 4.6}, "hold": {"x": ROOM_B_X + 5.2, "y": 4.6}, "open": False, "delay": -1.0},
        ],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    mgr = manager_factory(load_scenario(str(path)), node_name="test_zone_yaml")

    zones = mgr._zones.zones
    assert sorted(zones) == ["down", "up"]
    assert mgr._zones.open == {"up": True, "down": False}
    assert (zones["up"].delay, zones["down"].delay) == (0.0, -1.0)
    assert (zones["up"].destinations, zones["down"].destinations) == (("down",), ())
    assert zones["up"].entry == Pose2D(x=4.0, y=4.6)
    assert zones["up"].hold == Pose2D(x=5.2, y=4.6, theta=1.5)
    assert zones["up"].footprint == ((4.6, 4.0), (5.8, 4.0), (5.8, 5.2), (4.6, 5.2))
    assert zones["down"].footprint == ()
