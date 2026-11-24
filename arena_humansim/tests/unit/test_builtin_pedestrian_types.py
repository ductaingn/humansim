from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from arena_humansim.core.agents import BUILTIN_AGENTS, loader
from arena_humansim.core.agents.types import STAIRS_KINDS, AgentType, sample_agent_type
from arena_humansim.utils.scenario_loader import converter

KINEMATIC = ("adult", "elder", "child", "hurried", "distracted")
FIELDS = (
    "desired_velocity",
    "agent_radius",
    "max_velocity",
    "max_acceleration",
    "max_deceleration",
    "min_turning_radius",
    "pivot_angular_velocity",
)


@pytest.fixture(scope="module")
def types() -> dict[str, AgentType]:
    raw = loader.load_agent_types_raw_from_dir(Path(loader.__file__).resolve().parents[3] / "config" / "agent_types")
    resolved = loader.resolve_extends({name: r for name, (r, _) in raw.items()}, {})
    return {name: converter.structure(r, AgentType) for name, r in resolved.items()}


def test_every_kinematic_type_loads_as_simple_mode(types: dict[str, AgentType]) -> None:
    for name in KINEMATIC:
        assert types[name].mode == "simple", name


def test_extends_gives_every_type_the_full_field_set(types: dict[str, AgentType]) -> None:
    for name in KINEMATIC:
        t = types[name]
        for field in FIELDS:
            assert getattr(t, field) is not None, f"{name}.{field}"
        assert t.perception.vision_range is not None and t.perception.vision_fov is not None
        assert {"relaxation_time", "repulsion_strength", "repulsion_range"} <= set(t.local_planner_params)


def test_distributions_are_well_formed(types: dict[str, AgentType]) -> None:
    for name in KINEMATIC:
        t = types[name]
        for field in FIELDS:
            d = getattr(t, field)
            assert d.clip_low <= d.mean <= d.clip_high, f"{name}.{field}: {d}"
            assert d.std >= 0


def test_the_types_differ_where_they_are_meant_to(types: dict[str, AgentType]) -> None:
    radius = {n: types[n].agent_radius.mean for n in KINEMATIC}
    speed = {n: types[n].desired_velocity.mean for n in KINEMATIC}
    assert radius["child"] < radius["adult"]
    assert speed["elder"] < speed["distracted"] < speed["adult"] < speed["hurried"]
    assert types["distracted"].perception.vision_range.mean < types["adult"].perception.vision_range.mean


def test_every_type_samples(types: dict[str, AgentType]) -> None:
    rng = np.random.default_rng(3)
    for name in KINEMATIC:
        p = sample_agent_type(types[name], rng)
        t = types[name]
        assert t.desired_velocity.clip_low <= p.desired_velocity <= t.desired_velocity.clip_high
        assert t.agent_radius.clip_low <= p.agent_radius <= t.agent_radius.clip_high


def test_walkers_climb_every_stair_kind_and_the_manual_wheelchair_none(types: dict[str, AgentType]) -> None:
    rng = np.random.default_rng(3)
    for name in KINEMATIC:
        assert types[name].climbs == STAIRS_KINDS, name
    assert types["wheelchair_manual"].climbs == ()
    assert sample_agent_type(types["wheelchair_manual"], rng).climbs == ()
    assert sample_agent_type(types["adult"], rng).climbs == STAIRS_KINDS


def test_climbs_rejects_an_unknown_stair_kind() -> None:
    assert converter.structure({"name": "x", "climbs": ["stairs"]}, AgentType).climbs == ("stairs",)
    with pytest.raises(ValueError, match="ladder"):
        converter.structure({"name": "x", "climbs": ["ladder"]}, AgentType)


def test_stair_effort_is_drawn_per_walker_from_its_type_within_its_clip(types: dict[str, AgentType]) -> None:
    efforts = [sample_agent_type(types["adult"], np.random.default_rng(seed)).stair_effort for seed in range(40)]
    assert all(1.0 <= e <= 3.0 for e in efforts) and len(set(efforts)) > 10
    assert sample_agent_type(types["adult"], np.random.default_rng(7)).stair_effort == efforts[7]


def test_a_stair_effort_mapping_is_clipped_to_1_and_3_unless_it_sets_its_own_clips() -> None:
    wide = converter.structure({"name": "x", "stair_effort": {"mean": 1.6, "std": 5.0}}, AgentType)
    efforts = [sample_agent_type(wide, np.random.default_rng(seed)).stair_effort for seed in range(40)]
    assert min(efforts) == 1.0 and max(efforts) == 3.0
    own = converter.structure({"name": "x", "stair_effort": {"mean": 4.0, "clip_low": 3.5, "clip_high": 5.0}}, AgentType)
    assert sample_agent_type(own, np.random.default_rng(0)).stair_effort == 4.0
    assert sample_agent_type(converter.structure({"name": "x", "stair_effort": 2.0}, AgentType), np.random.default_rng(0)).stair_effort == 2.0


@pytest.mark.parametrize("name", sorted(BUILTIN_AGENTS))
def test_no_sampled_top_speed_is_below_the_walking_speed(name: str) -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = sample_agent_type(BUILTIN_AGENTS[name], rng)
        assert p.max_velocity >= p.desired_velocity
