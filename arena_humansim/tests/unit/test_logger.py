from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from arena_humansim.core.agents.types import (
    SampledParams,
    SampledPerception,
)
from arena_humansim.core.logger import SimulationLogger
from arena_humansim.utils.types import (
    AgentState,
    CommandType,
    HighLevelCommand,
    InteractionContract,
    InteractionState,
    Pose2D,
)


def _params(name: str = "adult") -> SampledParams:
    return SampledParams(
        name=name,
        desired_velocity=1.1,
        agent_radius=0.25,
        max_velocity=1.5,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        reaction_time=0.4,
        personal_space_min=0.6,
        perception=SampledPerception(vision_range=5.0, vision_fov=180.0),
        local_planner_params={
            "relaxation_time": 0.5,
            "repulsion_strength": 2.1,
            "repulsion_range": 0.3,
            "anisotropy": 0.5,
        },
        perception_stack=("default",),
        local_planner="sfm",
        global_planner="dijkstra",
        animation="noop",
    )


def _read_lines(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_session_jsonl_created_and_closed(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=7, config={"seed": 7})
    log_path = tmp_path / "session.jsonl"
    assert log_path.is_file()
    assert not log._log_file.closed
    log.close()
    assert log._log_file.closed


def test_config_snapshot_header_and_schema(tmp_path: Path) -> None:
    cfg = {"seed": 7, "dt": 0.05, "modules": {"navigation": "default"}}
    log = SimulationLogger(str(tmp_path), seed=7, config=cfg)
    log.close()

    snap = tmp_path / "config_snapshot.yaml"
    assert snap.is_file()
    text = snap.read_text()
    assert text.startswith("# Config snapshot - arena_humansim\n")
    assert "# Generated at:" in text

    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    parsed = yaml.safe_load(body)
    assert parsed == cfg


def test_config_snapshot_accepts_none(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config=None)
    log.close()
    snap = tmp_path / "config_snapshot.yaml"
    assert snap.is_file()


def test_record_spawn_emits_expected_keys(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config={})
    log.record_agent_spawn(agent_id=42, params=_params("adult"), agent_type_name="adult")
    log.close()

    records = _read_lines(tmp_path / "session.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "spawn"
    assert rec["agent_id"] == 42
    assert rec["agent_type"] == "adult"
    params = rec["params"]
    assert params["name"] == "adult"
    assert params["desired_velocity"] == pytest.approx(1.1)
    assert params["agent_radius"] == pytest.approx(0.25)
    assert params["perception"] == {
        "vision_range": pytest.approx(5.0),
        "vision_fov": pytest.approx(180.0),
        "proximity_sense": pytest.approx(1.0),
        "vision_occlusion": True,
        "vision_range_sd": pytest.approx(0.0),
        "vision_fov_sd": pytest.approx(0.0),
        "repulsion_rel_sd": pytest.approx(0.0),
        "reaction_rel_sd": pytest.approx(0.0),
        "robot_range_offset": pytest.approx(0.0),
        "robot_fov_offset": pytest.approx(0.0),
        "robot_repulsion_gain": pytest.approx(1.0),
        "robot_reaction_gain": pytest.approx(1.0),
        "robot_blind": False,
        "robot_halt": False,
        "robot_yields": True,
        "hearing_range": pytest.approx(0.0),
        "cane_reach": pytest.approx(0.0),
        "cane_width": pytest.approx(0.0),
        "detection_delay": pytest.approx(0.0),
    }
    assert params["local_planner_params"]["relaxation_time"] == pytest.approx(0.5)
    assert params["local_planner_params"]["repulsion_strength"] == pytest.approx(2.1)
    assert params["local_planner_params"]["repulsion_range"] == pytest.approx(0.3)
    assert params["local_planner_params"]["anisotropy"] == pytest.approx(0.5)
    assert params["local_planner"] == "sfm"
    assert params["global_planner"] == "dijkstra"
    assert params["animation"] == "noop"


def test_record_despawn_schema(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config={})
    log.record_agent_despawn(agent_id=9, reason="sink", tick=123)
    log.close()

    records = _read_lines(tmp_path / "session.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "despawn"
    assert rec["agent_id"] == 9
    assert rec["reason"] == "sink"
    assert rec["tick"] == 123


def test_record_tick_serializes_agents_interactions_commands(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config={})

    agents = {
        1: AgentState(agent_id=1, pose=Pose2D(x=1.25, y=-0.5, theta=0.3), velocity=(0.4, -0.2)),
        2: AgentState(agent_id=2, pose=Pose2D(x=3.0, y=2.0, theta=1.57), velocity=(0.0, 0.5)),
    }
    interactions = {
        10: InteractionState(
            id=10,
            type=1,
            contract=InteractionContract(),
            participants=[1, 2],
            state={"phase": "active"},
        ),
    }
    commands = {
        1: HighLevelCommand(
            agent_id=1,
            type=CommandType.NAVIGATE,
            target_pose=Pose2D(x=5.0, y=0.0, theta=0.0),
            interaction_target=-1,
        ),
    }

    log.record_tick(tick=5, timestamp=1.25, agents=agents, interactions=interactions, commands=commands)
    log.close()

    records = _read_lines(tmp_path / "session.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["tick"] == 5
    assert rec["timestamp"] == pytest.approx(1.25)

    assert set(rec["agents"].keys()) == {"1", "2"}
    a1 = rec["agents"]["1"]
    assert a1["pose"] == {"x": pytest.approx(1.25), "y": pytest.approx(-0.5), "theta": pytest.approx(0.3)}
    assert a1["velocity"] == {"vx": pytest.approx(0.4), "vy": pytest.approx(-0.2)}

    i = rec["interactions"]["10"]
    assert i["type"] == 1
    assert i["participants"] == [1, 2]
    assert i["state"] == {"phase": "active"}

    c = rec["commands"]["1"]
    assert c["type"] == 0
    assert c["target_pose"] == {"x": pytest.approx(5.0), "y": pytest.approx(0.0), "theta": pytest.approx(0.0)}
    assert c["interaction_target"] == -1


def test_close_is_idempotent(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config={})
    log.close()
    log.close()
    assert log._log_file.closed


def test_records_are_line_delimited_json(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config={})
    log.record_agent_spawn(agent_id=1, params=_params(), agent_type_name="adult")
    log.record_tick(tick=0, timestamp=0.0, agents={}, interactions={}, commands={})
    log.record_agent_despawn(agent_id=1, reason="ttl", tick=1)
    log.close()

    raw = (tmp_path / "session.jsonl").read_text()
    lines = raw.split("\n")
    assert lines[-1] == ""
    records = [json.loads(line) for line in lines[:-1]]
    assert [r.get("event", "tick") for r in records] == ["spawn", "tick", "despawn"]


def test_record_seed_logs_the_seed_streams_are_re_rooted_on(tmp_path: Path) -> None:
    log = SimulationLogger(str(tmp_path), seed=1, config={})
    log.record_seed(77)
    log.close()

    assert _read_lines(tmp_path / "session.jsonl") == [{"event": "seed", "seed": 77}]
