from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("arena_humansim_msgs")
pytest.importorskip("rclpy")

from std_msgs.msg import Header

from arena_humansim.core.agent_manager import _interaction_frame
from arena_humansim.core.interaction_kinds import InteractionType, MembershipRole
from arena_humansim.core.interaction_manager import InteractionEdge, InteractionManager, InteractionRow, LifecycleEdge
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import AgentKind, AgentState, BehaviorTreeMovement, InteractionOutcome, Pose2D, SeekSpec


class _Params:
    reaction_time = 0.4
    personal_space_min = 0.6


class _Agent:
    def __init__(self, agent_id: int) -> None:
        self.state = AgentState(agent_id=agent_id, desired_velocity=1.0, pose=Pose2D(), kind=int(AgentKind.HUMAN))
        self.movement = BehaviorTreeMovement()
        self.params = _Params()


def _manager(n_agents: int) -> InteractionManager:
    agents: dict[int, Any] = {aid: _Agent(aid) for aid in range(1, n_agents + 1)}
    world = WorldKnowledge()
    world.add_object(WorldObject(object_id="counter", type="counter", pose=Pose2D()))
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world, agent_lookup=agents.get, visibility_lookup=lambda aid: set(agents) - {aid})
    return mgr


def _create(mgr: InteractionManager, creator: int, itype: InteractionType, target: str | None = None) -> int:
    return mgr._create_interaction(creator_id=creator, spec=SeekSpec(interaction_type=itype, target=target)).id


def test_membership_table_lists_participants_then_queue_in_fifo_order() -> None:
    mgr = _manager(4)
    iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="counter")
    for aid in (3, 2, 4):
        assert mgr.accept(aid, iid)
    (row,) = mgr.membership_table()
    assert (row.interaction_id, row.interaction_type, row.object_id) == (iid, int(InteractionType.QUEUE_USE), "counter")
    assert row.outcome == int(mgr.interactions[iid].outcome)
    assert row.participants == (1,)
    assert row.queue == (3, 2, 4)
    assert not row.holding


def test_membership_table_keeps_an_agent_in_every_interaction_it_belongs_to() -> None:
    mgr = _manager(3)
    queue_iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="counter")
    talk_iid = _create(mgr, 3, InteractionType.GROUP_CONVERSATION)
    assert mgr.accept(2, queue_iid)
    assert mgr.accept(2, talk_iid)
    rows = {row.interaction_id: row for row in mgr.membership_table()}
    assert rows[queue_iid].queue == (2,)
    assert rows[talk_iid].participants == (3, 2)
    assert rows[talk_iid].object_id == ""


def test_membership_table_carries_forming_and_active_and_drops_ended() -> None:
    mgr = _manager(3)
    forming = _create(mgr, 1, InteractionType.GROUP_CONVERSATION)
    active = _create(mgr, 2, InteractionType.GROUP_CONVERSATION)
    ended = _create(mgr, 3, InteractionType.GROUP_CONVERSATION)
    mgr.interactions[active].outcome = InteractionOutcome.ACTIVE
    mgr.interactions[forming].outcome = InteractionOutcome.FORMING
    mgr.interactions[ended].outcome = InteractionOutcome.CANCELED
    outcomes = {row.interaction_id: row.outcome for row in mgr.membership_table()}
    assert outcomes == {forming: int(InteractionOutcome.FORMING), active: int(InteractionOutcome.ACTIVE)}


def test_membership_table_changes_when_a_queued_agent_leaves_and_empties_on_reset() -> None:
    mgr = _manager(3)
    iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="counter")
    assert mgr.accept(2, iid)
    assert mgr.accept(3, iid)
    before = mgr.membership_table()
    mgr.force_stop(2)
    after = mgr.membership_table()
    assert after != before
    assert after[0].queue == (3,)
    mgr.reset()
    assert mgr.membership_table() == ()


def test_interaction_frame_flattens_members_with_offsets_roles_and_edges() -> None:
    table = (
        InteractionRow(4, int(InteractionType.QUEUE_USE), int(InteractionOutcome.ACTIVE), "counter", (1,), (3, 2), True, True, 5.0, (2,)),
        InteractionRow(7, int(InteractionType.GROUP_CONVERSATION), int(InteractionOutcome.FORMING), "", (2, 5), (), False, False, -1.0, (-1, -1)),
    )
    edges = (InteractionEdge(4, int(InteractionType.QUEUE_USE), LifecycleEdge.HOLD_ONSET, (1,)),)
    header = Header(frame_id="map")
    msg = _interaction_frame(header, table, edges)
    assert msg.header.frame_id == "map"
    assert list(msg.interaction_id) == [4, 7]
    assert list(msg.interaction_type) == [int(InteractionType.QUEUE_USE), int(InteractionType.GROUP_CONVERSATION)]
    assert list(msg.outcome) == [int(InteractionOutcome.ACTIVE), int(InteractionOutcome.FORMING)]
    assert list(msg.object_id) == ["counter", ""]
    assert list(msg.arrived) == [True, False]
    assert list(msg.holding) == [True, False]
    assert list(msg.duration) == [5.0, -1.0]
    assert list(msg.member_offset) == [0, 3, 5]
    assert list(msg.member_id) == [1, 3, 2, 2, 5]
    assert list(msg.member_role) == [MembershipRole.PARTICIPANT, MembershipRole.QUEUED, MembershipRole.QUEUED, MembershipRole.PARTICIPANT, MembershipRole.PARTICIPANT]
    assert list(msg.member_seat) == [2, -1, -1, -1, -1]
    (event,) = msg.events
    assert (event.interaction_id, event.interaction_type, event.event, list(event.participants)) == (4, int(InteractionType.QUEUE_USE), int(LifecycleEdge.HOLD_ONSET), [1])


def test_membership_table_reports_the_seat_of_each_participant() -> None:
    agents: dict[int, Any] = {aid: _Agent(aid) for aid in (1, 2)}
    world = WorldKnowledge()
    world.add_object(WorldObject(object_id="bench", type="bench", pose=Pose2D(), seats=[Pose2D(x=-1.0), Pose2D(x=1.0)]))
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world, agent_lookup=agents.get, visibility_lookup=lambda aid: set(agents) - {aid})
    agents[1].state.pose = Pose2D(x=1.0, y=-2.0)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="bench")
    assert mgr.accept(2, iid)
    (row,) = mgr.membership_table()
    assert row.participants == (1,)
    assert row.queue == (2,)
    assert row.seats == (1,)


def test_interaction_frame_of_an_empty_table_has_a_single_zero_offset() -> None:
    msg = _interaction_frame(Header(), ())
    assert list(msg.interaction_id) == []
    assert list(msg.member_offset) == [0]
    assert list(msg.member_id) == []
    assert list(msg.events) == []
