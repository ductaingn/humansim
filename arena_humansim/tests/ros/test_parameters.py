from __future__ import annotations

from collections.abc import Iterator

import pytest
from rclpy.parameter import Parameter

from arena_humansim.global_planner.dijkstra import DijkstraPlanner
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from tests.ros._helpers import (
    AddWalls,
    RemoveAgents,
    RemoveWalls,
    ResetSimulation,
    RosTestSystem,
    SpawnAgents,
    make_add_walls_request,
    make_agent_msg,
    make_remove_request,
    make_remove_walls_request,
)

pytestmark = pytest.mark.ros

_RESOLUTION = "global_planner.resolution"
_COMFORT = "global_planner.comfort_radius"
_RELAXATION = "local_planner.relaxation_time"
_RESTORED = (
    _RESOLUTION,
    _COMFORT,
    _RELAXATION,
    "global_planner",
    "local_planner",
    "waypoint_threshold",
    "arrival_r_enter",
    "publish_markers",
    "rtf",
    "seed",
)


def _soft_reset(system: RosTestSystem) -> None:
    assert system.call(ResetSimulation, "reset", ResetSimulation.Request(soft=True)).success


@pytest.fixture
def system(ros_system: RosTestSystem) -> Iterator[RosTestSystem]:
    before = [ros_system.manager.get_parameter(name) for name in _RESTORED]
    yield ros_system
    ros_system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    ros_system.call(RemoveWalls, "remove_walls", make_remove_walls_request())
    assert all(r.successful for r in ros_system.manager.set_parameters(before))
    _soft_reset(ros_system)


def _set(system: RosTestSystem, name: str, value: object):
    (result,) = system.manager.set_parameters([Parameter(name, value=value)])
    return result


def _spawn(system: RosTestSystem, agent_id: int, policy: str = "", relaxation_time: float = 0.0):
    msg = make_agent_msg(agent_id=agent_id)
    msg.policy = policy
    msg.relaxation_time = relaxation_time
    resp = system.call(SpawnAgents, "spawn_agents", SpawnAgents.Request(agents=[msg]))
    assert resp.success
    return system.manager._agents[agent_id]


def _policy(system: RosTestSystem, agent_id: int):
    manager = system.manager
    return manager._policies[int(manager._pool.policy_idx[manager._pool.idx(agent_id)])]


def test_set_waits_for_the_reset(system: RosTestSystem) -> None:
    before = system.manager._global_planner._comfort_radius
    assert _set(system, _COMFORT, 0.5).successful
    assert system.manager._global_planner._comfort_radius == pytest.approx(before)
    _soft_reset(system)
    assert system.manager._global_planner._comfort_radius == pytest.approx(0.5)


def test_full_reset_applies_too(system: RosTestSystem) -> None:
    assert _set(system, "waypoint_threshold", 0.3).successful
    assert system.call(ResetSimulation, "reset", ResetSimulation.Request()).success
    assert system.manager._waypoint_threshold == pytest.approx(0.3)


def test_soft_reset_keeps_agents_and_walls(system: RosTestSystem) -> None:
    system.call(AddWalls, "add_walls", make_add_walls_request([("w", (-5.0, 2.0), (5.0, 2.0))]))
    _spawn(system, 70)
    _soft_reset(system)
    assert 70 in system.manager._agents
    assert "w" in system.manager._walls


def test_integer_set_on_double_param_is_widened(system: RosTestSystem) -> None:
    assert _set(system, _RESOLUTION, 1).successful
    param = system.manager.get_parameter(_RESOLUTION)
    assert param.type_ == Parameter.Type.DOUBLE
    assert param.value == pytest.approx(1.0)


def test_out_of_range_values_are_rejected(system: RosTestSystem) -> None:
    before = system.manager.get_parameter(_RESOLUTION).value
    assert not _set(system, _RESOLUTION, 0.0).successful
    assert not _set(system, "publish_markers", 3).successful
    assert system.manager.get_parameter(_RESOLUTION).value == pytest.approx(before)


@pytest.mark.parametrize(("name", "value"), [("dt", 0.1), ("collision", "noop"), ("mode", "master"), ("bt_tick_interval", 3)])
def test_static_param_is_rejected(system: RosTestSystem, name: str, value: object) -> None:
    before = system.manager.get_parameter(name).value
    result = _set(system, name, value)
    assert not result.successful
    assert "not reconfigurable" in result.reason
    assert system.manager.get_parameter(name).value == before


def test_arrival_radii_stay_ordered(system: RosTestSystem) -> None:
    r_exit = system.manager.get_parameter("arrival_r_exit").value
    assert not _set(system, "arrival_r_enter", r_exit).successful
    assert _set(system, "arrival_r_enter", r_exit / 2).successful
    _soft_reset(system)
    assert system.manager._arrival_r_enter == pytest.approx(r_exit / 2)


def test_atomic_batch_takes_planner_keys_in_any_order(system: RosTestSystem) -> None:
    batch = [Parameter("local_planner.lateral_gain", value=1.5), Parameter("local_planner", value="hsfm")]
    assert system.manager.set_parameters_atomically(batch).successful
    assert system.manager.get_parameter("local_planner.lateral_gain").value == pytest.approx(1.5)
    assert system.manager.set_parameters_atomically([Parameter("local_planner.lateral_gain", value=0.0)]).successful


def test_atomic_batch_is_all_or_nothing(system: RosTestSystem) -> None:
    before = system.manager.get_parameter(_RESOLUTION).value
    batch = [Parameter(_RESOLUTION, value=0.1), Parameter("dt", value=0.1)]
    result = system.manager.set_parameters_atomically(batch)
    assert not result.successful
    assert "dt" in result.reason
    assert system.manager.get_parameter(_RESOLUTION).value == pytest.approx(before)


def _named_params(system: RosTestSystem, name: str) -> object:
    msg = make_agent_msg()
    msg.name = name
    resp = system.call(SpawnAgents, "spawn_agents", SpawnAgents.Request(agents=[msg]))
    assert resp.success
    return system.manager._agents[resp.spawned_ids[0]].params


def test_seed_takes_effect_at_the_next_remove_all(system: RosTestSystem) -> None:
    before = system.manager.get_parameter("seed").value
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    base = _named_params(system, "p")
    assert _set(system, "seed", before + 7).successful
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    reseeded = _named_params(system, "p")
    assert _set(system, "seed", before).successful
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    assert _named_params(system, "p") == base
    assert reseeded != base


def test_unknown_module_is_rejected(system: RosTestSystem) -> None:
    assert "available" in _set(system, "local_planner", "nope").reason
    assert "available" in _set(system, "global_planner", "nope").reason


def test_local_planner_switch_reseats_live_agents_and_prunes(system: RosTestSystem) -> None:
    follower = _spawn(system, 71)
    assert isinstance(_policy(system, 71), SFMPlanner)
    assert _set(system, "local_planner", "hsfm").successful
    _soft_reset(system)
    assert isinstance(follower.local_planner, HSFMPlanner)
    assert isinstance(_policy(system, 71), HSFMPlanner)
    assert "lateral_gain" in follower.params.local_planner_params
    assert [type(p) for p in system.manager._policies] == [HSFMPlanner]
    assert not any(type(w) is SFMPlanner for w in system.manager._wall_aware)
    assert isinstance(_spawn(system, 72).local_planner, HSFMPlanner)
    system.tick_manager(3)


def test_pinned_policy_survives_the_switch_and_keeps_its_planner_alive(system: RosTestSystem) -> None:
    _spawn(system, 73, policy="sfm")
    assert _set(system, "local_planner", "hsfm").successful
    _soft_reset(system)
    assert type(_policy(system, 73)) is SFMPlanner
    assert {type(p) for p in system.manager._policies} == {SFMPlanner, HSFMPlanner}
    system.call(RemoveAgents, "remove_agents", make_remove_request([73]))
    _soft_reset(system)
    assert [type(p) for p in system.manager._policies] == [HSFMPlanner]
    system.tick_manager(3)


def test_global_planner_switch_reseats_prunes_and_inherits_grid(system: RosTestSystem) -> None:
    system.call(AddWalls, "add_walls", make_add_walls_request([("w", (-5.0, 2.0), (5.0, 2.0))]))
    agent = _spawn(system, 74)
    assert _set(system, _RESOLUTION, 0.1).successful
    assert _set(system, "global_planner", "dijkstra").successful
    _soft_reset(system)
    planner = system.manager._global_planner
    assert isinstance(planner, DijkstraPlanner)
    assert planner._resolution == pytest.approx(0.1)
    assert planner._occupancy_grid is not None
    assert agent.global_planner is planner
    assert not any(isinstance(w, NavMeshPlanner) for w in system.manager._wall_aware)
    assert "navmesh" not in system.manager._module_pool["global_planner"]
    system.tick_manager(3)


def test_local_planner_mean_shifts_live_and_new_agents(system: RosTestSystem) -> None:
    live = _spawn(system, 75)
    pinned = _spawn(system, 76, relaxation_time=0.7)
    sampled = live.params.local_planner_params["relaxation_time"]
    type_mean = system.manager._agent_types["adult"].local_planner_params["relaxation_time"].mean

    assert _set(system, _RELAXATION, 2.0).successful
    _soft_reset(system)
    assert live.params.local_planner_params["relaxation_time"] == pytest.approx(sampled + 2.0 - type_mean)
    assert _policy(system, 75)._relaxation_time[system.manager._pool.idx(75)] == pytest.approx(sampled + 2.0 - type_mean)
    assert pinned.params.local_planner_params["relaxation_time"] == pytest.approx(0.7)
    assert _spawn(system, 77).params.local_planner_params["relaxation_time"] == pytest.approx(2.0, abs=0.5)

    assert _set(system, _RELAXATION, 0.0).successful
    _soft_reset(system)
    assert live.params.local_planner_params["relaxation_time"] == pytest.approx(sampled)


def test_marker_level_toggles(system: RosTestSystem) -> None:
    assert _set(system, "publish_markers", 2).successful
    _soft_reset(system)
    assert system.manager._marker_pub is not None
    _spawn(system, 78)
    system.tick_manager(2)
    assert _set(system, "publish_markers", 0).successful
    _soft_reset(system)
    assert system.manager._publish_markers == 0
    system.tick_manager(2)


def test_rtf_retimes_the_master_timer(system: RosTestSystem) -> None:
    assert _set(system, "rtf", 2.0).successful
    _soft_reset(system)
    assert system.manager._timer.timer_period_ns == int(1e9 * system.manager._dt / 2.0)
