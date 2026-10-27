"""Clip sets: each two-person interaction draws one set, members play their role's clip at the stance it was made in."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.core.formation.dyad import DyadFormation
from arena_humansim.core.interaction_kinds import ClipSet, InteractionType
from arena_humansim.core.interaction_manager import CONTACT_ENABLED, InteractionManager
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, HighLevelCommand, Pose2D, SeekSpec

DT = 0.1


@dataclass
class _State:
    agent_id: int
    pose: Pose2D
    desired_velocity: float = 1.0


@dataclass
class _Agent:
    state: _State
    movement: BehaviorTreeMovement = field(default_factory=BehaviorTreeMovement)


def _bound(itype: InteractionType, n: int = 2, seed: int = 0) -> tuple[InteractionManager, dict[int, _Agent]]:
    agents = {i: _Agent(_State(agent_id=i, pose=Pose2D(x=1.5 * (i - 1), y=0.3 * (i % 2)))) for i in range(1, n + 1)}
    mgr = InteractionManager(RNG(seed))
    mgr.set_context(world_knowledge=WorldKnowledge(), agent_lookup=agents.get, visibility_lookup=lambda aid: set(agents) - {aid})
    mgr.set_contact_mode(CONTACT_ENABLED)
    for aid in agents:  # one at a time, so the join order is the agent id order
        mgr.update({aid: HighLevelCommand(agent_id=aid, type=CommandType.SEEK, spec=SeekSpec(interaction_type=itype))}, dt=DT)
    return mgr, agents


def _wrap(a: float) -> float:
    return math.remainder(a, math.tau)


def test_pair_stance_reproduces_the_partner_offset() -> None:
    """Slots of a ClipSet.pair stance put b where role a's annotation has it: (dx, dy) in a's frame, turned dyaw."""
    dx, dy, dyaw = 0.46, -0.34, 2.38
    clips = ClipSet.pair("a", "b", dx, dy, dyaw)
    assert clips.separation == pytest.approx(math.hypot(dx, dy))
    agents = {1: _Agent(_State(1, Pose2D(x=-1.0, y=0.2))), 2: _Agent(_State(2, Pose2D(x=1.0, y=0.9)))}
    dyad = DyadFormation(PoseAnchor(Pose2D(x=3.0, y=-1.0)), agents.get, separation=clips.separation, heading_a=clips.headings[0], heading_b=clips.headings[1])
    dyad.on_join(1)
    dyad.on_join(2)
    slots = dyad.tick(DT)
    a, b = slots[1], slots[2]
    c, s = math.cos(a.theta), math.sin(a.theta)
    local = ((b.x - a.x) * c + (b.y - a.y) * s, -(b.x - a.x) * s + (b.y - a.y) * c)
    assert local == pytest.approx((dx, dy))
    assert _wrap(b.theta - a.theta) == pytest.approx(dyaw)


def test_hug_members_play_their_roles_at_the_sets_stance() -> None:
    mgr, _ = _bound(InteractionType.HUG)
    interaction = next(iter(mgr.interactions.values()))
    clips = mgr.clip_set(interaction)
    assert clips in InteractionType.HUG.kind.clips
    assert interaction.participants == [1, 2]
    assert (mgr.role_clip(interaction, 1), mgr.role_clip(interaction, 2)) == (clips.role(0), clips.role(1))
    assert mgr.role_clip(interaction, 99) is None
    a, b = mgr.formation_target(1), mgr.formation_target(2)
    assert a is not None and b is not None
    assert clips.separation is not None and math.hypot(b.x - a.x, b.y - a.y) == pytest.approx(clips.separation)
    axis = math.atan2(b.y - a.y, b.x - a.x)
    assert _wrap(a.theta - axis) == pytest.approx(clips.headings[0])
    assert _wrap(b.theta - axis - math.pi) == pytest.approx(clips.headings[1])


def test_clip_set_is_drawn_once_and_reproducibly() -> None:
    picks = []
    for seed in range(12):
        mgr, _ = _bound(InteractionType.HUG, seed=seed)
        interaction = next(iter(mgr.interactions.values()))
        first = mgr.clip_set(interaction)
        assert all(mgr.clip_set(interaction) is first for _ in range(3))
        again, _ = _bound(InteractionType.HUG, seed=seed)
        assert mgr.clip_set(next(iter(again.interactions.values()))) == first
        picks.append(first)
    assert len(set(picks)) > 1, "every seed drew the same hug"


def test_group_members_cycle_the_roles() -> None:
    mgr, _ = _bound(InteractionType.GROUP_CONVERSATION, n=3)
    interaction = next(iter(mgr.interactions.values()))
    clips = mgr.clip_set(interaction)
    assert clips is not None and len(clips.roles) == 2
    assert [mgr.role_clip(interaction, aid) for aid in interaction.participants] == [clips.roles[0], clips.roles[1], clips.roles[0]]


def test_one_role_set_plays_the_same_clip_on_both() -> None:
    clips = ClipSet(roles=("shake",), separation=0.88)
    assert clips.role(0) == clips.role(1) == "shake"
    assert InteractionType.SHAKE_HAND.kind.clip == InteractionType.SHAKE_HAND.kind.clips[0].roles[0]
    assert InteractionType.BLOCK.kind.clip == ""
