from __future__ import annotations

import json
from pathlib import Path

import attrs
import pytest

from . import _wheelchair_suite as suite

pytestmark = pytest.mark.slow

_GOLDEN = Path(__file__).resolve().parent / "golden" / "wheelchair_baseline.json"

SUCCESS_RATE_SLACK = 0.1
MIN_CLEARANCE_M = -0.05
MAX_WALL_PENETRATION_M = 0.05
GOLDEN_TOLERANCE = 1e-6


@attrs.frozen
class Gate:
    """Thresholds from golden/wheelchair_baseline.json: r_time = 1.25 x baseline_time_ratio, s_stuck_s = baseline stuck difference + 2.0, e_efficiency = baseline efficiency difference + 0.05."""

    baseline_time_ratio: float
    r_time: float
    s_stuck_s: float
    e_efficiency: float


GATES = {
    "doorway": Gate(baseline_time_ratio=1.0990, r_time=1.3737, s_stuck_s=2.0250, e_efficiency=0.0508),
    "corridor_head_on_adult": Gate(baseline_time_ratio=1.1055, r_time=1.3818, s_stuck_s=2.0250, e_efficiency=0.0387),
    "corridor_head_on_robot": Gate(baseline_time_ratio=1.1515, r_time=1.4393, s_stuck_s=2.5000, e_efficiency=0.0150),
    "crowd_crossing": Gate(baseline_time_ratio=1.0967, r_time=1.3708, s_stuck_s=2.0500, e_efficiency=0.0701),
    "corridor_u_turn": Gate(baseline_time_ratio=1.2009, r_time=1.5012, s_stuck_s=2.8750, e_efficiency=0.0918),
    "robot_parked": Gate(baseline_time_ratio=1.0442, r_time=1.3053, s_stuck_s=2.0250, e_efficiency=0.0309),
}


@pytest.mark.parametrize("scene", suite.SCENES)
def test_wheelchair_stays_within_the_scene_thresholds_relative_to_the_adult(scene: str) -> None:
    gate = GATES[scene]
    runs = suite.scene_runs(scene)
    assert {r.planner for r in runs["wheelchair_manual"]} == {"hsfm"}
    assert {r.planner for r in runs["adult"]} == {"sfm"}
    per_type = suite.scene_aggregates(scene)
    chair = per_type["wheelchair_manual"]
    adult = per_type["adult"]
    assert chair["success_rate"] >= adult["success_rate"] - SUCCESS_RATE_SLACK
    assert chair["time_to_goal_s"] <= gate.r_time * adult["time_to_goal_s"]
    assert chair["time_stuck_s"] <= adult["time_stuck_s"] + gate.s_stuck_s
    assert chair["path_efficiency"] >= adult["path_efficiency"] - gate.e_efficiency
    if chair["min_clearance_worst_m"] is not None:
        assert chair["min_clearance_worst_m"] >= MIN_CLEARANCE_M
    assert chair["max_wall_penetration_worst_m"] <= MAX_WALL_PENETRATION_M


@pytest.mark.parametrize("scene", suite.SCENES)
def test_aggregates_reproduce_the_recorded_baseline_on_its_machine_class(scene: str) -> None:
    golden = json.loads(_GOLDEN.read_text())
    if golden["machine"] != suite.machine_class():
        pytest.skip(f"baseline recorded on {golden['machine']!r}, running on {suite.machine_class()!r}")
    assert golden["seeds"] == list(suite.SEEDS)
    assert golden["budgets_s"][scene] == suite.budget_s(scene)
    fresh = suite.scene_aggregates(scene)
    assert set(fresh) == set(golden["scenes"][scene])
    for subject_type, expected in golden["scenes"][scene].items():
        assert fresh[subject_type] == pytest.approx(expected, rel=0.0, abs=GOLDEN_TOLERANCE), subject_type
