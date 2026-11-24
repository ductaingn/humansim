from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ScenarioConfig, SimulationParams, load_scenario
from arena_humansim.utils.types import Pose2D

TYPED, PINNED, ROBOT = 1, 2, 3


def _write_scenario(path: Path) -> Path:
    data = {
        "name": "agent_type_params",
        "simulation": {"seed": 5, "dt": 0.05, "max_ticks": 1},
        "agent_types": {"stroller": {"extends": "adult", "desired_velocity": {"mean": 0.8, "std": 0.0}, "agent_radius": {"mean": 0.21, "std": 0.0}}},
        "agents": [
            {"agent_id": TYPED, "agent_type": "stroller", "spawn_pose": {"x": 0.0, "y": 0.0}, "goal_sequence": [{"x": 5.0, "y": 0.0}]},
            {"agent_id": PINNED, "agent_type": "stroller", "spawn_pose": {"x": 0.0, "y": 2.0}, "goal_sequence": [{"x": 5.0, "y": 2.0}], "desired_velocity": 1.4, "agent_radius": 0.4},
            {"agent_id": ROBOT, "kind": 1, "policy": "straight", "agent_type": "robot", "spawn_pose": {"x": 0.0, "y": -2.0}, "goal_sequence": [{"x": 5.0, "y": -2.0}]},
        ],
    }
    path.write_text(yaml.safe_dump(data))
    return path


def _spawned(mgr: AgentManager, aid: int) -> tuple[float, float, float, float]:
    agent = mgr._agents[aid]
    return agent.params.desired_velocity, agent.state.desired_velocity, agent.params.agent_radius, float(mgr._pool.agent_radius[mgr._pool.idx(aid)])


def test_scenario_agents_without_explicit_values_take_their_type_sample(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    scenario = load_scenario(str(_write_scenario(tmp_path / "scenario.yaml")))
    mgr = manager_factory(scenario, node_name="test_scenario_agent_type_sample")
    assert _spawned(mgr, TYPED) == pytest.approx((0.8, 0.8, 0.21, 0.21))
    assert _spawned(mgr, ROBOT) == pytest.approx((1.1, 1.1, 0.3, 0.3))


def test_scenario_agents_with_explicit_values_keep_them(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    scenario = load_scenario(str(_write_scenario(tmp_path / "scenario.yaml")))
    assert (scenario.agents[1].desired_velocity, scenario.agents[1].agent_radius) == (1.4, 0.4)
    mgr = manager_factory(scenario, node_name="test_scenario_agent_explicit_params")
    assert _spawned(mgr, PINNED) == pytest.approx((1.4, 1.4, 0.4, 0.4))


def test_scenario_agent_of_an_unknown_type_falls_back_to_walking_defaults(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(
        name="unknown_type",
        simulation=SimulationParams(seed=5, dt=0.05, max_ticks=1),
        agents=[AgentConfig(agent_id=TYPED, agent_type="ghost", spawn_pose=Pose2D(x=0.0, y=0.0), goal_sequence=[Pose2D(x=5.0, y=0.0)])],
    )
    mgr = manager_factory(scenario, node_name="test_scenario_agent_unknown_type")
    assert _spawned(mgr, TYPED) == pytest.approx((1.3, 1.3, 0.35, 0.35))


def test_flow_spawns_take_the_type_sample_unless_the_template_sets_values(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    data = yaml.safe_load(_write_scenario(tmp_path / "scenario.yaml").read_text())
    source = {"shape": {"type": "circle", "radius": 0.5}, "type": "max", "max_concurrent": 2, "max_total": 2}
    data["agents"] = []
    data["flow"] = {
        "sources": [
            {**source, "pose": {"x": -5.0, "y": 0.0}, "agent_template": {"agent_type": "stroller"}},
            {**source, "pose": {"x": 5.0, "y": 0.0}, "agent_template": {"agent_type": "stroller", "desired_velocity_min": 1.4, "desired_velocity_max": 1.4, "agent_radius": 0.4}},
        ],
        "sinks": [],
    }
    path = tmp_path / "flow.yaml"
    path.write_text(yaml.safe_dump(data))
    mgr = manager_factory(load_scenario(str(path)), node_name="test_flow_agent_type_params")
    for _ in range(20):
        mgr.tick()
    spawned = sorted(_spawned(mgr, aid) for aid in mgr._agents)
    assert spawned == [pytest.approx((0.8, 0.8, 0.21, 0.21))] * 2 + [pytest.approx((1.4, 1.4, 0.4, 0.4))] * 2


def test_explicit_speed_above_the_type_max_raises_the_max(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    data = yaml.safe_load(_write_scenario(tmp_path / "scenario.yaml").read_text())
    data["simulation"]["max_ticks"] = 200
    data["agents"] = [{"agent_id": ROBOT, "kind": 1, "policy": "straight", "agent_type": "robot", "spawn_pose": {"x": 0.0, "y": 0.0}, "goal_sequence": [{"x": 50.0, "y": 0.0}], "desired_velocity": 1.3}]
    path = tmp_path / "fast_robot.yaml"
    path.write_text(yaml.safe_dump(data))
    mgr = manager_factory(load_scenario(str(path)), node_name="test_explicit_speed_raises_max")
    pool = mgr._pool
    assert mgr._agents[ROBOT].params.max_velocity == pytest.approx(1.3)
    assert float(pool.max_velocity[pool.idx(ROBOT)]) == pytest.approx(1.3)
    for _ in range(100):
        mgr.tick()
    assert float((pool.vel[pool.idx(ROBOT)] ** 2).sum()) ** 0.5 > 1.2


def _flow_with_awareness(tmp_path: Path, awareness: str = "blind") -> Path:
    data = yaml.safe_load(_write_scenario(tmp_path / "scenario.yaml").read_text())
    data["agents"] = []
    data["flow"] = {
        "sources": [
            {
                "shape": {"type": "circle", "radius": 0.5},
                "type": "max",
                "max_concurrent": 2,
                "max_total": 2,
                "pose": {"x": -5.0, "y": 0.0},
                "agent_template": {"agent_type": "stroller", "robot_awareness": awareness},
            }
        ],
        "sinks": [],
    }
    path = tmp_path / "flow_awareness.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_flow_spawns_take_the_template_robot_awareness(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = manager_factory(load_scenario(str(_flow_with_awareness(tmp_path))), node_name="test_flow_robot_awareness")
    for _ in range(20):
        mgr.tick()
    pool = mgr._pool
    assert len(mgr._agents) == 2
    for aid, agent in mgr._agents.items():
        idx = pool.idx(aid)
        assert bool(pool.robot_blind[idx]) is True
        assert agent.params.perception.robot_blind is True


def test_flow_spawns_take_the_template_not_yielding_awareness(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = manager_factory(load_scenario(str(_flow_with_awareness(tmp_path, "not_yielding"))), node_name="test_flow_not_yielding")
    for _ in range(20):
        mgr.tick()
    pool = mgr._pool
    assert len(mgr._agents) == 2
    for aid in mgr._agents:
        idx = pool.idx(aid)
        assert bool(pool.robot_yields[idx]) is False, "ORCA reads this column"
        assert bool(pool.robot_blind[idx]) is False
        assert pool.robot_repulsion_gain[idx] == 0.0


def test_removing_a_flow_agent_drops_its_awareness_override(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = manager_factory(load_scenario(str(_flow_with_awareness(tmp_path))), node_name="test_flow_override_removal")
    for _ in range(20):
        mgr.tick()
    aids = list(mgr._agents)
    assert set(aids) == set(mgr._scenario_awareness_overrides)
    mgr._remove_agent(aids[0])
    assert set(mgr._scenario_awareness_overrides) == set(aids[1:])
