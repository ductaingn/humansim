from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation import (
    AgentAnchor,
    CentroidAnchor,
    ObjectAnchor,
    PoseAnchor,
)
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import (
    CommandType,
    InteractionManager,
    _make_contract,
)
from arena_humansim.core.world_knowledge import AnchorKind, FormationSpec, WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    BehaviorTreeMovement,
    InteractionOutcome,
    InteractionState,
    Pose2D,
    SeatPose,
    SeekSpec,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


@dataclass
class _FakeState:
    agent_id: int = 0
    pose: Pose2D = field(default_factory=Pose2D)
    desired_velocity: float = 1.2
    kind: int = 0


@dataclass
class _FakeAgent:
    state: _FakeState
    params: _FakeParams = field(default_factory=_FakeParams)
    movement: BehaviorTreeMovement = field(default_factory=BehaviorTreeMovement)


def _mk_manager(agents: dict[int, _FakeAgent], world: WorldKnowledge | None = None) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world or WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid))
    return mgr


def _create(
    mgr: InteractionManager,
    agent_id: int,
    itype: InteractionType,
    *,
    target: str | int | None = None,
    offer: bool = False,
    duration: float | None = None,
    formation_spec: FormationSpec | None = None,
    min_participants: int | None = None,
    max_participants: int | None = None,
    queueable: bool | None = None,
) -> int:
    spec = SeekSpec(
        interaction_type=itype,
        target=target,
        offer=offer,
        duration=duration,
        formation_spec=formation_spec,
        min_participants=min_participants,
        max_participants=max_participants,
        queueable=queueable,
    )
    interaction = mgr._create_interaction(creator_id=agent_id, spec=spec)
    return interaction.id


def test_resolve_formation_uses_object_metadata_override() -> None:
    wk = WorldKnowledge()
    wk.add_object(
        WorldObject(
            object_id="counter",
            type="counter",
            pose=Pose2D(x=10.0, y=0.0, theta=0.0),
            formation=FormationSpec(type="line", params={"base_step": 0.5}),
        )
    )
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="counter")
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "LineFormation"
    assert formation.base_step == 0.5


def test_resolve_formation_falls_back_to_type_default() -> None:
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents)
    iid = _create(mgr, 1, InteractionType.GROUP_CONVERSATION)
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "FFormation"


def test_anchor_from_spec_object_kind() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=3.0, y=4.0)))
    wk.add_object(
        WorldObject(
            object_id="counter",
            type="counter",
            pose=Pose2D(x=10.0, y=0.0),
            formation=FormationSpec(type="line", anchor_kind=AnchorKind.OBJECT, anchor_ref="atm"),
        )
    )
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents, world=wk)
    contract = _make_contract(InteractionType.QUEUE_USE)
    inter = InteractionState(contract=contract, object_id="counter", participants=[1])
    counter = wk.get("counter")
    assert counter is not None and counter.formation is not None
    anchor = mgr._anchor_from_spec(counter.formation, inter)
    assert isinstance(anchor, ObjectAnchor)
    assert anchor.pose() == Pose2D(x=3.0, y=4.0)


def test_anchor_from_spec_pose_kind() -> None:
    pose = Pose2D(x=2.0, y=2.0, theta=0.3)
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.POSE, anchor_pose=pose)
    agents: dict[int, _FakeAgent] = {}
    mgr = _mk_manager(agents)
    inter = InteractionState(contract=_make_contract(InteractionType.USE))
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, PoseAnchor)
    assert anchor.pose() == pose


def test_anchor_from_spec_agent_kind() -> None:
    agents = {7: _FakeAgent(state=_FakeState(agent_id=7, pose=Pose2D(x=9.0, y=9.0)))}
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.AGENT, anchor_ref="7")
    mgr = _mk_manager(agents)
    inter = InteractionState(contract=_make_contract(InteractionType.USE))
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, AgentAnchor)
    assert anchor.pose() == Pose2D(x=9.0, y=9.0)


def test_anchor_from_spec_centroid_kind() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=4.0, y=0.0))),
    }
    mgr = _mk_manager(agents)
    spec = FormationSpec(type="f_formation", anchor_kind=AnchorKind.CENTROID)
    inter = InteractionState(
        contract=_make_contract(InteractionType.GROUP_CONVERSATION),
        participants=[1, 2],
    )
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, CentroidAnchor)
    assert anchor.pose().x == pytest.approx(2.0)


def test_anchor_from_spec_provider_kind() -> None:
    agents = {5: _FakeAgent(state=_FakeState(agent_id=5, pose=Pose2D(x=7.0, y=0.0)))}
    mgr = _mk_manager(agents)
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.PROVIDER, params={"base_step": 0.8})
    inter = InteractionState(
        contract=_make_contract(InteractionType.SERVICE),
        participants=[5],
        state={"provider": 5},
    )
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, AgentAnchor)
    assert anchor.pose() == Pose2D(x=7.0, y=0.0)


def test_anchor_from_spec_object_with_no_world_knowledge_returns_none() -> None:
    mgr = InteractionManager(RNG(0))  # no world_knowledge wired
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.OBJECT, anchor_ref="atm")
    inter = InteractionState(contract=_make_contract(InteractionType.QUEUE_USE))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_anchor_from_spec_agent_with_invalid_ref_returns_none() -> None:
    mgr = _mk_manager({})
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.AGENT, anchor_ref="not_an_int")
    inter = InteractionState(contract=_make_contract(InteractionType.USE))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_anchor_from_spec_provider_without_provider_returns_none() -> None:
    mgr = _mk_manager({})
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.PROVIDER)
    inter = InteractionState(contract=_make_contract(InteractionType.SERVICE))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_tick_formations_writes_navigate_to_member() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=5.0, y=0.0, theta=0.0)))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D()))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="atm")
    _, formation_targets, _ = mgr.update({}, dt=0.05)
    assert 1 in formation_targets
    assert formation_targets[1] == Pose2D(x=5.0, y=0.0, theta=0.0)
    cmd = agents[1].movement.command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose == Pose2D(x=5.0, y=0.0, theta=0.0)
    assert cmd.desired_velocity == agents[1].state.desired_velocity
    assert mgr.interactions[iid].contract.formation is not None


def test_tick_formations_skips_when_agent_lookup_returns_none() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    agents: dict[int, _FakeAgent] = {}  # empty
    mgr = _mk_manager(agents, world=wk)
    _create(mgr, 99, InteractionType.QUEUE_USE, target="atm")
    mgr.update({}, dt=0.05)
    assert 99 in mgr._formation_targets


def test_service_provider_anchored_formation_via_spec() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=1.0, y=0.0))),
    }
    mgr = _mk_manager(agents)
    fs = FormationSpec(type="line", anchor_kind=AnchorKind.PROVIDER, params={"base_step": 0.8})
    iid = _create(mgr, 1, InteractionType.SERVICE, target="water", offer=True, formation_spec=fs)
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "LineFormation"
    assert isinstance(formation.anchor, AgentAnchor)
    assert formation.anchor.agent_id == 1


def test_group_conversation_releases_formation_on_duration_expiry() -> None:
    # For 2-member F-formation with defaults: radius = base_radius + radius_per_member*(n-1) = 0.82.
    # Placing agents on that circle makes arrived() true so _tick_durations actually ticks.
    wk = WorldKnowledge()
    wk.add_object(
        WorldObject(
            object_id="gathering",
            type="gathering_area",
            pose=Pose2D(x=0.0, y=0.0),
            formation=FormationSpec(type="f_formation", anchor_kind=AnchorKind.POSE, anchor_pose=Pose2D(x=0.0, y=0.0)),
        )
    )
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.82, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=-0.82, y=0.0))),
    }
    mgr = _mk_manager(agents, world=wk)
    iid = _create(
        mgr,
        1,
        InteractionType.GROUP_CONVERSATION,
        duration=0.2,
        formation_spec=FormationSpec(type="f_formation", anchor_kind=AnchorKind.POSE, anchor_pose=Pose2D()),
    )
    assert mgr.accept(2, iid) is True
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE

    _, targets, departed = mgr.update({}, dt=0.15)
    assert iid in mgr.interactions
    assert set(targets.keys()) == {1, 2}
    assert departed == set()

    _, targets, departed = mgr.update({}, dt=0.15)
    assert iid not in mgr.interactions
    assert departed == {1, 2}
    assert targets == {}
    assert mgr._formation_targets == {}

    _, targets, departed = mgr.update({}, dt=0.05)
    assert targets == {}
    assert departed == set()


def test_object_seats_become_cluster_slots() -> None:
    wk = WorldKnowledge()
    seat = Pose2D(x=5.4, y=0.2, theta=1.5)
    wk.add_object(WorldObject(object_id="chair", type="chair", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=[seat]))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D()))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="chair")
    _, formation_targets, _ = mgr.update({}, dt=0.05)
    assert formation_targets[1] == seat
    assert mgr.interactions[iid].contract.formation is not None


def test_seated_agent_parks_on_its_seat_from_reach() -> None:
    wk = WorldKnowledge()
    seat = Pose2D(x=5.0, y=0.0, theta=-1.5)
    wk.add_object(WorldObject(object_id="chair", type="chair", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=[seat]))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=5.0, y=-2.0)))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="chair")
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    mgr.update({}, dt=0.05)
    assert mgr.parked() == {}
    agents[1].state.pose = Pose2D(x=5.0, y=-0.7)
    assert mgr.parked() == {1: seat}


def test_posture_waits_for_arrival() -> None:
    """A participant is ACTIVE while still walking up, the widened footprint must not apply yet."""
    wk = WorldKnowledge()
    seat = Pose2D(x=5.0, y=0.0, theta=0.0)
    wk.add_object(WorldObject(object_id="bed", type="bed", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=[seat]))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=5.0, y=-8.0)))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.LIE_ON, target="bed")
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    mgr.update({}, dt=0.05)
    assert mgr.posture_of(1) == "standing"
    agents[1].state.pose = Pose2D(x=5.0, y=-0.5)
    assert mgr.posture_of(1) == "prone"


def test_support_height_is_the_seat_surface_once_arrived() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="bed", type="bed", pose=Pose2D(x=5.0, y=0.0), seats=[SeatPose(x=5.0, y=0.0, height=0.62)]))
    wk.add_object(WorldObject(object_id="stool", type="stool", pose=Pose2D(x=-5.0, y=0.0), seats=[SeatPose(x=-5.0, y=0.0)]))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=5.0, y=-8.0))), 2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=-5.0, y=0.1)))}
    mgr = _mk_manager(agents, world=wk)
    for agent_id, itype, target in ((1, InteractionType.LIE_ON, "bed"), (2, InteractionType.SIT_ON, "stool")):
        mgr.interactions[_create(mgr, agent_id, itype, target=target)].outcome = InteractionOutcome.ACTIVE
    mgr.update({}, dt=0.05)
    assert mgr.support_height(1) is None  # still walking to the bed
    agents[1].state.pose = Pose2D(x=5.0, y=-0.5)
    assert mgr.support_height(1) == 0.62
    assert mgr.support_height(2) is None  # seated, but the stool's surface was never measured


def test_posture_of_follows_the_active_interaction_kind() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="bed", type="bed", pose=Pose2D(x=2.0, y=0.0, theta=0.0)))
    # on its ring slot, so the posture gate sees it as arrived
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.8, y=0.0))), 2: _FakeAgent(state=_FakeState(agent_id=2))}
    mgr = _mk_manager(agents, world=wk)
    assert mgr.posture_of(1) == "standing"
    iid = _create(mgr, 1, InteractionType.LIE_ON, target="bed")
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.posture_of(1) == "prone"
    assert mgr.posture_of(2) == "standing"


def test_offset_seat_does_not_trip_drift_eviction() -> None:
    """The seat can lie beyond the drift radius, but the pinned agent is exactly where the interaction wants it."""
    wk = WorldKnowledge()
    seat = Pose2D(x=7.5, y=0.0, theta=3.1)
    wk.add_object(WorldObject(object_id="bench", type="bench", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=[seat]))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=5.0, y=0.3)))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="bench")
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    for x in (5.0, 6.0, 7.0, seat.x):
        agents[1].state.pose = Pose2D(x=x, y=0.3 if x == 5.0 else 0.0)
        mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].state["_drift_arrived"] == {1}
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert mgr.parked() == {1: seat}


def test_two_kinds_on_one_object_do_not_share_a_seat() -> None:
    wk = WorldKnowledge()
    near = Pose2D(x=5.0, y=1.0, theta=0.0)
    far = Pose2D(x=5.0, y=-1.0, theta=0.0)
    wk.add_object(WorldObject(object_id="bench", type="bench", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=[near, far]))
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=5.0, y=3.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=5.0, y=3.0))),
    }
    mgr = _mk_manager(agents, world=wk)
    for agent_id, itype in ((1, InteractionType.SIT_ON), (2, InteractionType.LIE_ON)):
        iid = _create(mgr, agent_id, itype, target="bench")
        mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    _, formation_targets, _ = mgr.update({}, dt=0.05)
    assert formation_targets == {1: near, 2: far}


def test_queued_member_gets_no_seat() -> None:
    """Seats are for sitters: a queued agent must not be routed onto the furniture."""
    wk = WorldKnowledge()
    seats = [Pose2D(x=5.0, y=1.0, theta=0.0), Pose2D(x=5.0, y=-1.0, theta=0.0)]
    wk.add_object(WorldObject(object_id="bench", type="bench", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=seats))
    agents = {aid: _FakeAgent(state=_FakeState(agent_id=aid, pose=Pose2D(x=5.0, y=6.0))) for aid in (1, 2)}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="bench")
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue
    _, formation_targets, _ = mgr.update({}, dt=0.05)
    assert 2 not in formation_targets
    assert mgr.interactions[iid].contract.formation.occupied_slots() == [formation_targets[1]]


def test_promotion_from_the_queue_claims_the_freed_seat() -> None:
    wk = WorldKnowledge()
    seat = Pose2D(x=5.0, y=1.0, theta=0.0)
    wk.add_object(WorldObject(object_id="chair", type="chair", pose=Pose2D(x=5.0, y=0.0, theta=0.0), seats=[seat]))
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=seat)),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=5.0, y=6.0))),
    }
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="chair", duration=0.5)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    formation = mgr.interactions[iid].contract.formation
    assert formation.slot_of(2) is None
    mgr.update({}, dt=0.6)
    assert mgr.interactions[iid].participants == [2]
    assert formation.slot_of(2) == seat


def test_generated_ring_slot_does_not_trip_drift_eviction() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="bench", type="bench", pose=Pose2D(x=5.0, y=0.0, theta=0.0)))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=5.3, y=0.0)))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.SIT_ON, target="bench")
    slot = mgr.interactions[iid].contract.formation.slot_of(1)
    assert slot is not None
    for frac in (0.0, 0.5, 1.0):
        agents[1].state.pose = Pose2D(x=5.3 + frac * (slot.x - 5.3), y=frac * slot.y)
        mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert mgr.interactions[iid].state["_drift_arrived"] == {1}


def test_line_front_slot_does_not_trip_drift_eviction() -> None:
    wk = WorldKnowledge()
    line = FormationSpec(type="line", params={"base_step": 0.8, "front_offset": 0.8}, anchor_kind=AnchorKind.OBJECT)
    wk.add_object(WorldObject(object_id="fountain", type="fountain", pose=Pose2D(x=4.0, y=0.0, theta=3.14159), formation=line))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=4.2, y=0.0)))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.USE, target="fountain")
    for x in (4.2, 4.5, 4.8):
        agents[1].state.pose = Pose2D(x=x, y=0.0)
        mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert mgr.interactions[iid].participants == [1]


def test_requeued_member_gets_a_fresh_drift_latch() -> None:
    wk = WorldKnowledge()
    line = FormationSpec(type="line", params={"base_step": 0.8, "front_offset": 0.8}, anchor_kind=AnchorKind.OBJECT)
    wk.add_object(WorldObject(object_id="fountain", type="fountain", pose=Pose2D(x=4.0, y=0.0, theta=3.14159), formation=line))
    front = Pose2D(x=4.8, y=0.0)
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=front)),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=5.6, y=0.0))),
    }
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.USE, target="fountain", duration=0.5)
    assert mgr.accept(2, iid) is True
    mgr.update({}, dt=0.05)
    mgr.update({}, dt=1.0)
    assert mgr.interactions[iid].participants == [2]
    agents[1].state.pose = Pose2D(x=9.0, y=0.0)
    agents[2].state.pose = front
    assert mgr.accept(1, iid) is True
    mgr.update({}, dt=0.05)
    mgr.update({}, dt=1.0)
    assert mgr.interactions[iid].participants == [1]
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE


def test_queue_rotation_keeps_the_service_provider() -> None:
    line = FormationSpec(type="line", params={"base_step": 0.8}, anchor_kind=AnchorKind.PROVIDER)
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=-0.8, y=0.0))),
        3: _FakeAgent(state=_FakeState(agent_id=3, pose=Pose2D(x=-1.6, y=0.0))),
    }
    mgr = _mk_manager(agents)
    iid = _create(mgr, 1, InteractionType.SERVICE, target="desk", offer=True, duration=0.5, formation_spec=line, min_participants=2, max_participants=2, queueable=True)
    assert mgr.accept(2, iid) is True
    assert mgr.accept(3, iid) is True
    mgr.update({}, dt=0.05)
    mgr.update({}, dt=1.0)
    interaction = mgr.interactions[iid]
    assert interaction.participants == [1, 3]
    assert interaction.outcome == InteractionOutcome.ACTIVE
    assert agents[2].movement.last_outcome == InteractionOutcome.COMPLETED
    assert agents[1].movement.last_outcome is None


def _fountain() -> WorldKnowledge:
    wk = WorldKnowledge()
    line = FormationSpec(type="line", params={"base_step": 0.8, "front_offset": 0.8}, anchor_kind=AnchorKind.OBJECT)
    wk.add_object(WorldObject(object_id="fountain", type="fountain", pose=Pose2D(x=4.0, y=0.0, theta=3.14159), capacity=2, formation=line))
    return wk


def test_promoted_member_closing_on_the_front_slot_is_not_evicted_and_keeps_its_turn() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=4.8, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=5.6, y=0.0))),
    }
    mgr = _mk_manager(agents, world=_fountain())
    iid = _create(mgr, 1, InteractionType.USE, target="fountain", duration=0.5)
    assert mgr.accept(2, iid) is True
    mgr.update({}, dt=0.05)
    mgr.update({}, dt=1.0)
    assert mgr.interactions[iid].participants == [2]
    for x in (5.39, 5.42, 5.40):
        agents[2].state.pose = Pose2D(x=x, y=0.0)
        mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].participants == [2]
    assert mgr.accept(1, iid) is True
    assert mgr.interactions[iid].contract.queue == [1]
    agents[2].state.pose = Pose2D(x=4.8, y=0.0)
    mgr.update({}, dt=0.05)
    agents[2].state.pose = Pose2D(x=9.0, y=0.0)
    mgr.update({}, dt=0.05)
    assert agents[2].movement.last_outcome == InteractionOutcome.INTERRUPTED


def test_latched_member_whose_slot_moves_up_is_not_evicted() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=4.8, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=5.6, y=0.0))),
    }
    mgr = _mk_manager(agents, world=_fountain())
    iid = _create(mgr, 1, InteractionType.USE, target="fountain", max_participants=2)
    assert mgr.accept(2, iid) is True
    mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].participants == [1, 2]
    assert mgr.interactions[iid].state["_drift_arrived"] == {1, 2}
    mgr.stop(1, iid, reason=InteractionOutcome.COMPLETED)
    mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].participants == [2]
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE


def test_mobile_provider_line_paces_the_provider_to_its_slowest_passenger() -> None:
    line = FormationSpec(type="line", params={"base_step": 0.8}, anchor_kind=AnchorKind.PROVIDER)
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0), desired_velocity=1.1)),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=-0.8, y=0.0), desired_velocity=0.8)),
    }
    mgr = _mk_manager(agents)
    iid = _create(mgr, 1, InteractionType.SERVICE, target="ride", offer=True, formation_spec=line, min_participants=2, max_participants=2)
    assert mgr.accept(2, iid) is True
    mgr.update({}, dt=0.05)
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert mgr.formation_speeds() == {1: pytest.approx(0.8)}
    agents[2].state.desired_velocity = 1.4
    mgr.update({}, dt=0.05)
    assert mgr.formation_speeds() == {1: pytest.approx(1.1)}
