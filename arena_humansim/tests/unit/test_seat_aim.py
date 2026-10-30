from __future__ import annotations

import math
from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.behavior.nodes import GoToNode, ResolveObjectNode
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import BehaviorTreeMovement, Pose2D, SeekSpec

# A three-seat sofa at the origin facing +y, its sitters standing 0.6 m ahead of a seat before they sit.
SEATS = [Pose2D(x=x, y=0.0, theta=math.pi / 2) for x in (-0.55, 0.0, 0.55)]
APPROACH = 0.6


def _stand(seat: int) -> tuple[float, float]:
    return SEATS[seat].x, SEATS[seat].y + APPROACH


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def world() -> WorldKnowledge:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="sofa", type="sofa", pose=Pose2D(), seats=list(SEATS), seat_approach=[APPROACH] * 3))
    return wk


def _setup(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, poses: dict[int, tuple[float, float]]) -> tuple[dict[int, BaseAgent], InteractionManager]:
    agents = {aid: agent_factory(agent_id=aid, x=x, y=y) for aid, (x, y) in poses.items()}
    for agent in agents.values():
        agent.movement = BehaviorTreeMovement()
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world, agent_lookup=lambda aid: agents.get(aid))
    return agents, mgr


def _walk(agent: BaseAgent, world: WorldKnowledge, mgr: InteractionManager, interaction: str | None = "SIT_ON") -> tuple[ResolveObjectNode, GoToNode, StepContext]:
    ctx = StepContext(im=mgr)
    resolve = ResolveObjectNode("resolve", agent, world, target="sofa", ctx=ctx, interaction_name=interaction)
    go = GoToNode("go", agent, ctx=ctx, world=world)
    resolve.tick_once()
    return resolve, go, ctx


def _target(ctx: StepContext) -> tuple[float, float]:
    assert ctx.target_pose is not None
    return pytest.approx(ctx.target_pose.x), pytest.approx(ctx.target_pose.y)


def _sit(mgr: InteractionManager, agent_id: int) -> int | None:
    return mgr.seek(agent_id, SeekSpec(interaction_type=InteractionType.SIT_ON, target="sofa"))


def test_a_sitter_walks_to_the_nearest_free_seats_standing_point(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agents, mgr = _setup(agent_factory, world, {1: (-0.4, 1.4)})
    _, _, ctx = _walk(agents[1], world, mgr)
    assert _target(ctx) == _stand(0)


def test_aiming_reserves_nothing_the_first_to_seek_takes_the_seat_and_the_other_retargets(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agents, mgr = _setup(agent_factory, world, {1: (-0.6, 30.0), 2: (-0.5, 1.0)})
    _, far_go, far = _walk(agents[1], world, mgr)
    assert _target(far) == _stand(0)
    assert mgr.held_seats("sofa") == set()
    _sit(mgr, 2)
    assert mgr.held_seats("sofa") == {0}
    far_go.tick_once()
    assert _target(far) == _stand(1)


def test_the_seat_a_sitter_aims_at_is_the_seat_it_is_given(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agents, mgr = _setup(agent_factory, world, {1: (0.5, 1.4)})
    _, _, ctx = _walk(agents[1], world, mgr)
    iid = _sit(mgr, 1)
    assert iid is not None
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert _target(ctx) == _stand(formation.seat_index_of(1))


def test_a_seat_overlapping_a_held_one_is_not_aimed_at(agent_factory: Callable[..., BaseAgent]) -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="sofa", type="sofa", pose=Pose2D(), seats=list(SEATS), seat_approach=[APPROACH] * 3, seat_conflicts=[frozenset({1}), frozenset({0, 2}), frozenset({1})]))
    agents, mgr = _setup(agent_factory, wk, {1: (0.0, 1.4), 2: (0.1, 1.4)})
    _sit(mgr, 1)
    assert mgr.held_seats("sofa") == {1}
    _, _, ctx = _walk(agents[2], wk, mgr)
    assert _target(ctx) == (0.0, 0.0)  # seats 0 and 2 overlap the held seat 1: none free, the object


def test_without_a_free_seat_the_sitter_walks_to_the_object(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agents, mgr = _setup(agent_factory, world, {1: (-0.5, 1.0), 2: (0.0, 1.0), 3: (0.5, 1.0), 4: (0.0, 2.0)})
    for aid in (1, 2, 3):
        _sit(mgr, aid)
    assert mgr.held_seats("sofa") == {0, 1, 2}
    _, _, ctx = _walk(agents[4], world, mgr)
    assert _target(ctx) == (0.0, 0.0)


@pytest.mark.parametrize("interaction", [None, "LIE_ON"])
def test_only_a_sitter_aims_at_seats(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, interaction: str | None) -> None:
    agents, mgr = _setup(agent_factory, world, {1: (-0.4, 1.4)})
    _, _, ctx = _walk(agents[1], world, mgr, interaction)
    assert _target(ctx) == (0.0, 0.0)
