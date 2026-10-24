from __future__ import annotations

import enum
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import attrs
import numpy as np
from numba import njit

from arena_humansim.core.access import AcceptResult, AccessPolicy
from arena_humansim.core.access.fifo_queue import FIFOQueue
from arena_humansim.core.formation import (
    AgentAnchor,
    Anchor,
    CentroidAnchor,
    Formation,
    ObjectAnchor,
    PoseAnchor,
)
from arena_humansim.core.interaction_kinds import AccessKind, HandleKind, InteractionType, MembershipRole
from arena_humansim.utils import RNG
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import (
    AnchorKind,
    BehaviorTreeMovement,
    CommandType,
    FormationSpec,
    HighLevelCommand,
    InteractionContract,
    InteractionOutcome,
    InteractionState,
    Pose2D,
    SeatPose,
    SeekSpec,
    pose_distance,
)

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.formation.clearance import Clearance
    from arena_humansim.core.world_knowledge import WorldKnowledge


AgentLookup = Callable[[int], "BaseAgent | None"]
VisibilityLookup = Callable[[int], set[int]]


_ENDED_OUTCOMES: frozenset[int] = frozenset(
    {
        int(InteractionOutcome.COMPLETED),
        int(InteractionOutcome.INTERRUPTED),
        int(InteractionOutcome.CANCELED),
    }
)

_UNSET: Any = object()

_HOLDING = "_holding"
_SEATED = "_seated"
_DRIFT_LATCH_FRACTION = 0.5


class LifecycleEdge(enum.IntEnum):
    """Mirrors arena_humansim_msgs/InteractionEvent."""

    ACTIVATED = 0
    HOLD_ONSET = 1
    RELEASED = 2
    INTERRUPTED = 3
    CANCELED = 4


_TERMINAL_EDGES: dict[int, LifecycleEdge] = {
    int(InteractionOutcome.COMPLETED): LifecycleEdge.RELEASED,
    int(InteractionOutcome.INTERRUPTED): LifecycleEdge.INTERRUPTED,
    int(InteractionOutcome.CANCELED): LifecycleEdge.CANCELED,
}


CONTACT_ENABLED = "enabled"
CONTACT_LOCOMOTION_ONLY = "locomotion_only"
CONTACT_MODES = (CONTACT_ENABLED, CONTACT_LOCOMOTION_ONLY)
GESTURE_ENABLED = "enabled"
GESTURE_DISABLED = "disabled"
GESTURE_MODES = (GESTURE_ENABLED, GESTURE_DISABLED)
DEFAULT_STANDING_DISTANCE = 1.2


def check_gesture_mode(mode: str) -> str:
    if mode not in GESTURE_MODES:
        raise ValueError(f"gesture_mode must be one of {GESTURE_MODES}, got {mode!r}")
    return mode


@attrs.frozen
class InteractionEdge:
    interaction_id: int
    interaction_type: int
    edge: LifecycleEdge
    participants: tuple[int, ...]


class InteractionRow(NamedTuple):
    interaction_id: int
    interaction_type: int
    outcome: int
    object_id: str
    participants: tuple[int, ...]
    queue: tuple[int, ...]
    arrived: bool
    holding: bool
    duration: float  # seconds, -1 = unbounded
    seats: tuple[int, ...]  # object seat index per participant, -1 = none


@njit(cache=True)
def _nearest_peer(xy: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest other point of its group, groups delimited by ptr."""
    out = np.full(xy.shape[0], np.inf)
    for g in range(ptr.shape[0] - 1):
        for i in range(ptr[g], ptr[g + 1]):
            for j in range(ptr[g], ptr[g + 1]):
                if i != j:
                    out[i] = min(out[i], math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
    return out


def _make_contract(
    interaction_type: InteractionType,
    min_participants: int | None = None,
    max_participants: int | None = None,
    queueable: bool | None = None,
) -> InteractionContract:
    defaults = interaction_type.kind.contract_defaults
    min_p = defaults.min_participants if min_participants is None else min_participants
    max_p = defaults.max_participants if max_participants is None else max_participants
    queue_p = defaults.queueable if queueable is None else queueable
    contract = InteractionContract(
        type=int(interaction_type),
        min_participants=min_p,
        max_participants=max_p,
        queueable=queue_p,
    )
    if queue_p and defaults.access is AccessKind.FIFO:
        contract.access = FIFOQueue()
    return contract


class InteractionManager(Loggable):
    def __init__(
        self,
        rng_manager: RNG,
        world_knowledge: WorldKnowledge | None = None,
        agent_lookup: AgentLookup | None = None,
        formation_scale: float = 1.0,
        cohesion_multiplier: float = 1.2,
    ):
        self.rng_manager = rng_manager
        self.interactions: dict[int, InteractionState] = {}
        self.next_interaction_id: int = 0
        self._agent_membership: dict[int, dict[int, MembershipRole]] = {}
        self._interaction_by_object_type: dict[tuple[str, int], int] = {}
        self._interactions_by_type: dict[int, set[int]] = {}
        self._world_knowledge = world_knowledge
        self._agent_lookup: AgentLookup = agent_lookup or (lambda _aid: None)
        self._visibility_lookup: VisibilityLookup | None = None
        self._formation_scale = formation_scale
        self._cohesion_multiplier = cohesion_multiplier
        self._formation_targets: dict[int, Pose2D] = {}
        self._formation_speeds: dict[int, float] = {}
        self._formation_dt = 0.0
        self._clearance: Clearance | None = None
        self._warmup()
        self._current_departed: set[int] = set()
        self._edges: list[InteractionEdge] = []
        self._contact_mode = CONTACT_ENABLED
        self._standing_distance = DEFAULT_STANDING_DISTANCE

    def set_contact_mode(self, mode: str, standing_distance: float | None = None) -> None:
        """Applies to contact interactions created from now on, live ones keep their formation."""
        if mode not in CONTACT_MODES:
            raise ValueError(f"contact mode {mode!r} not in {CONTACT_MODES}")
        self._contact_mode = mode
        if standing_distance is not None:
            if standing_distance <= 0.0:
                raise ValueError(f"standing distance must be positive, got {standing_distance}")
            self._standing_distance = standing_distance

    @property
    def contact_mode(self) -> str:
        return self._contact_mode

    @staticmethod
    def is_contact(interaction: InteractionState) -> bool:
        return InteractionType(interaction.type).kind.render_pose_override

    def shows_contact(self, interaction: InteractionState) -> bool:
        """Contact clip and render override are shown: contact enabled and the pair is holding."""
        return self.is_contact(interaction) and self._contact_mode == CONTACT_ENABLED and self.is_holding(interaction)

    def _contact_params(self, interaction: InteractionState, formation_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if formation_type == "dyad" and self.is_contact(interaction) and self._contact_mode == CONTACT_LOCOMOTION_ONLY:
            return {**params, "separation": self._standing_distance}
        return params

    def _emit(self, interaction: InteractionState, edge: LifecycleEdge, participants: list[int] | None = None) -> None:
        members = interaction.participants if participants is None else participants
        self._edges.append(InteractionEdge(interaction.id, interaction.type, edge, tuple(members)))

    def drain_edges(self) -> list[InteractionEdge]:
        """Lifecycle edges fired since the last drain, in firing order."""
        edges, self._edges = self._edges, []
        return edges

    def all_arrived(self, interaction: InteractionState) -> bool:
        formation = interaction.contract.formation
        if formation is None or not interaction.participants:
            return True
        return all(formation.arrived(pid) for pid in interaction.participants)

    def is_holding(self, interaction: InteractionState) -> bool:
        return bool(interaction.state.get(_HOLDING))

    def _restart_hold(self, interaction: InteractionState) -> None:
        interaction.state[_HOLDING] = False
        interaction.contract.elapsed = 0.0

    def membership_table(self) -> tuple[InteractionRow, ...]:
        """One row per FORMING or ACTIVE interaction, sorted by id, participants then the queue in FIFO order."""
        rows = []
        for iid, interaction in sorted(self.interactions.items()):
            if interaction.outcome not in (InteractionOutcome.FORMING, InteractionOutcome.ACTIVE):
                continue
            contract = interaction.contract
            rows.append(
                InteractionRow(
                    iid,
                    int(interaction.type),
                    int(interaction.outcome),
                    interaction.object_id or "",
                    tuple(interaction.participants),
                    tuple(contract.queue),
                    self.all_arrived(interaction),
                    self.is_holding(interaction),
                    float(contract.duration) if contract.duration is not None else -1.0,
                    tuple(self._seat_index(interaction, pid) for pid in interaction.participants),
                )
            )
        return tuple(rows)

    @staticmethod
    def _seat_index(interaction: InteractionState, agent_id: int) -> int:
        formation = interaction.contract.formation
        seat = formation.seat_index_of(agent_id) if formation is not None else None
        return -1 if seat is None else seat

    def _warmup(self) -> None:
        _nearest_peer(np.zeros((2, 2)), np.array([0, 2], dtype=np.int64))

    def _pose_lookup(self, agent_id: int) -> Pose2D | None:
        agent = self._agent_lookup(agent_id)
        return agent.state.pose if agent is not None else None

    def reset(self) -> None:
        """Drop all interaction state between episodes. Clearing `interactions` alone leaves
        dangling ids in the indices the scan helpers walk; `next_interaction_id` stays monotonic
        so ids remain unique across episodes."""
        for interaction in self.interactions.values():
            if interaction.outcome == InteractionOutcome.ACTIVE:
                self._emit(interaction, LifecycleEdge.CANCELED)
        self.interactions.clear()
        self._agent_membership.clear()
        self._interaction_by_object_type.clear()
        self._interactions_by_type.clear()
        self._formation_targets.clear()
        self._formation_speeds.clear()
        self._formation_dt = 0.0
        self._current_departed.clear()

    def set_context(
        self,
        world_knowledge: WorldKnowledge | None = None,
        agent_lookup: AgentLookup | None = None,
        formation_scale: float | None = None,
        visibility_lookup: VisibilityLookup | None = None,
        clearance: Clearance | None = None,
    ) -> None:
        if world_knowledge is not None:
            self._world_knowledge = world_knowledge
        if agent_lookup is not None:
            self._agent_lookup = agent_lookup
        if formation_scale is not None:
            self._formation_scale = formation_scale
        if visibility_lookup is not None:
            self._visibility_lookup = visibility_lookup
        if clearance is not None:
            self._clearance = clearance

    def _update_bt_movement(
        self,
        agent_id: int,
        *,
        interaction_id: Any = _UNSET,  # noqa: ANN401
        clear_command: bool = False,
        last_outcome: Any = _UNSET,  # noqa: ANN401
    ) -> None:
        agent = self._agent_lookup(agent_id)
        if agent is None or not isinstance(agent.movement, BehaviorTreeMovement):
            return
        mv = agent.movement
        changed = False
        if interaction_id is not _UNSET and mv.interaction_id != interaction_id:
            changed = True
        if clear_command and mv.command is not None:
            changed = True
        if last_outcome is not _UNSET and mv.last_outcome != last_outcome:
            changed = True
        if not changed:
            return
        if interaction_id is not _UNSET:
            mv.interaction_id = interaction_id
        if clear_command:
            mv.command = None
        if last_outcome is not _UNSET:
            mv.last_outcome = last_outcome

    def _add_membership(self, agent_id: int, interaction_id: int, role: MembershipRole) -> None:
        self._agent_membership.setdefault(agent_id, {})[interaction_id] = role

    def _drop_membership(self, agent_id: int, interaction_id: int) -> None:
        roles = self._agent_membership.get(agent_id)
        if roles is None:
            return
        roles.pop(interaction_id, None)
        if not roles:
            del self._agent_membership[agent_id]

    def _iter_membership(self, agent_id: int, role: MembershipRole | None = None) -> list[int]:
        roles = self._agent_membership.get(agent_id)
        if roles is None:
            return []
        if role is None:
            return list(roles.keys())
        return [iid for iid, r in roles.items() if r == role]

    def is_bound(self, agent_id: int) -> bool:
        roles = self._agent_membership.get(agent_id)
        if roles is None:
            return False
        for iid, role in roles.items():
            if role != MembershipRole.PARTICIPANT:
                continue
            interaction = self.interactions.get(iid)
            if interaction is not None and interaction.outcome == InteractionOutcome.ACTIVE:
                return True
        return False

    def accept(self, agent_id: int, interaction_id: int) -> bool:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return False
        target = interaction.target_agent
        if target is not None and target != agent_id:
            return False
        if agent_id in interaction.participants:
            return True
        if agent_id in interaction.contract.queue:
            return True

        contract = interaction.contract
        access: AccessPolicy | None = contract.access
        if access is not None:
            result = access.on_accept(interaction, agent_id)
            if result == AcceptResult.REJECTED:
                return False
            if result == AcceptResult.QUEUED:
                self._add_membership(agent_id, interaction_id, MembershipRole.QUEUED)
                self._update_bt_movement(agent_id, interaction_id=interaction_id)
                self._on_formation_join(interaction, agent_id)
                return True
            self._add_membership(agent_id, interaction_id, MembershipRole.PARTICIPANT)
            if interaction.outcome == InteractionOutcome.ACTIVE:
                self._restart_hold(interaction)
            self._maybe_activate(interaction)
            self._update_bt_movement(agent_id, interaction_id=interaction_id)
            self._on_formation_join(interaction, agent_id)
            return True

        if contract.is_full:
            if contract.queueable:
                if contract.max_queue == -1 or len(contract.queue) < contract.max_queue:
                    contract.queue.append(agent_id)
                    self._add_membership(agent_id, interaction_id, MembershipRole.QUEUED)
                    self._update_bt_movement(agent_id, interaction_id=interaction_id)
                    self._on_formation_join(interaction, agent_id)
                    return True
            return False

        interaction.participants.append(agent_id)
        contract.current_participants.append(agent_id)
        self._add_membership(agent_id, interaction_id, MembershipRole.PARTICIPANT)
        self._maybe_activate(interaction)
        self._update_bt_movement(agent_id, interaction_id=interaction_id)
        self._on_formation_join(interaction, agent_id)
        return True

    def _on_formation_join(self, interaction: InteractionState, agent_id: int) -> None:
        formation: Formation | None = interaction.contract.formation
        if formation is None:
            return
        formation.on_join(agent_id, participant=agent_id in interaction.participants)

    def _on_formation_leave(self, interaction: InteractionState, agent_id: int) -> None:
        formation: Formation | None = interaction.contract.formation
        if formation is None:
            return
        formation.on_leave(agent_id)
        self._formation_targets.pop(agent_id, None)
        self._formation_speeds.pop(agent_id, None)

    def stop(
        self,
        agent_id: int,
        interaction_id: int,
        reason: InteractionOutcome = InteractionOutcome.INTERRUPTED,
    ) -> InteractionState | None:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return None

        contract = interaction.contract
        was_involved = agent_id in contract.queue or agent_id in interaction.participants
        if was_involved:
            self._current_departed.add(agent_id)

        access: AccessPolicy | None = contract.access
        if access is not None:
            access.on_stop(interaction, agent_id)
        elif agent_id in contract.queue:
            contract.queue.remove(agent_id)

        self._drop_membership(agent_id, interaction_id)

        members = list(interaction.participants)
        was_participant = agent_id in interaction.participants
        if was_participant:
            interaction.participants.remove(agent_id)
        if agent_id in contract.current_participants:
            contract.current_participants.remove(agent_id)

        if was_involved:
            interaction.member_durations.pop(agent_id, None)
            self._on_formation_leave(interaction, agent_id)
            self._update_bt_movement(agent_id, interaction_id=None, clear_command=True, last_outcome=reason)

        if len(interaction.participants) < contract.min_participants:
            self._teardown(interaction_id, reason, members)
            return None
        if was_participant and interaction.outcome == InteractionOutcome.ACTIVE:
            self._restart_hold(interaction)
        return interaction

    def update(
        self,
        high_level_commands: dict[int, HighLevelCommand],
        dt: float = 0.0,
        extra_commands: list[HighLevelCommand] | None = None,
        formations: bool = True,
    ) -> tuple[dict[int, InteractionState], dict[int, Pose2D], set[int]]:
        """Formations and drift checks advance only when `formations` is set, by the time since their last advance."""
        self._current_departed = set()
        self._prune_ended_interactions()
        self._tick_access(dt)
        self._tick_durations(dt)

        interaction_cmds: list[HighLevelCommand] = []
        for agent_id in sorted(high_level_commands.keys()):
            cmd = high_level_commands[agent_id]
            if not isinstance(cmd, HighLevelCommand):
                continue
            if cmd.type == CommandType.NAVIGATE:
                continue
            interaction_cmds.append(cmd)

        if extra_commands:
            for cmd in extra_commands:
                if isinstance(cmd, HighLevelCommand) and cmd.type != CommandType.NAVIGATE:
                    interaction_cmds.append(cmd)

        interaction_cmds.sort(key=lambda c: self.rng_manager.get_agent_substream(c.agent_id, "interaction_order").random())
        # Providers create before seekers match, so same-tick offers become visible.
        interaction_cmds.sort(key=lambda c: 0 if (c.spec is not None and c.spec.offer) else 1)

        for cmd in interaction_cmds:
            self._process_command(cmd)

        self._formation_dt += dt
        if formations:
            self._tick_drift_eviction()
            self._tick_formations(self._formation_dt)
            self._formation_dt = 0.0
        self._prune_dead_interactions()
        self._prune_ended_interactions()
        return self.interactions, dict(self._formation_targets), set(self._current_departed)

    def _tick_drift_eviction(self) -> None:
        # Same threshold as request-time proximity (x cohesion_multiplier)
        victims: list[tuple[int, int]] = []
        loose: list[tuple[int, float, set[int], list[tuple[int, Pose2D]]]] = []
        for iid, interaction in self.interactions.items():
            if interaction.outcome != InteractionOutcome.ACTIVE:
                continue
            if not interaction.participants:
                continue
            kind = InteractionType(interaction.type).kind
            radius = kind.interaction_radius * self._cohesion_multiplier
            latched: set[int] = interaction.state.setdefault("_drift_arrived", set())
            latched.intersection_update(interaction.participants)
            if kind.is_object_bound:
                object_id = interaction.object_id
                if object_id is None or self._world_knowledge is None:
                    continue
                anchor = self._world_knowledge.object_pose(object_id)
                if anchor is None:
                    continue
                formation = interaction.contract.formation
                latched_at: dict[int, Pose2D] = interaction.state.setdefault("_drift_latched_at", {})
                for aid in interaction.participants:
                    pose = self._pose_lookup(aid)
                    if pose is None:
                        continue
                    # an explicit slot can lie further from the object origin than the radius, so drift is measured from it
                    slot = formation.slot_of(aid) if formation is not None else None
                    target = anchor if slot is None else slot
                    if aid in latched and pose_distance(target, latched_at[aid]) > radius * _DRIFT_LATCH_FRACTION:
                        latched.discard(aid)
                    distance = pose_distance(pose, target)
                    if distance <= radius * _DRIFT_LATCH_FRACTION:
                        latched.add(aid)
                        latched_at[aid] = target
                    elif distance > radius and aid in latched:
                        victims.append((aid, iid))
            else:
                poses = [(aid, p) for aid in interaction.participants if (p := self._pose_lookup(aid)) is not None]
                if len(poses) >= 2:
                    loose.append((iid, radius, latched, poses))
        if loose:
            ptr = np.zeros(len(loose) + 1, dtype=np.int64)
            np.cumsum([len(poses) for *_, poses in loose], out=ptr[1:])
            nearest = iter(_nearest_peer(np.array([(p.x, p.y) for *_, poses in loose for _, p in poses], dtype=np.float64), ptr).tolist())
            for iid, radius, latched, poses in loose:
                for aid, _ in poses:
                    if next(nearest) <= radius:
                        latched.add(aid)
                    elif aid in latched:
                        victims.append((aid, iid))
        for aid, iid in victims:
            self.stop(aid, iid, reason=InteractionOutcome.INTERRUPTED)

    def _tick_formations(self, dt: float) -> dict[int, Pose2D]:
        targets: dict[int, Pose2D] = {}
        speeds: dict[int, float] = {}
        by_type: dict[type[Formation], list[Formation]] = {}
        for interaction in self.interactions.values():
            formation: Formation | None = interaction.contract.formation
            if interaction.outcome == InteractionOutcome.ACTIVE and formation is not None:
                by_type.setdefault(type(formation), []).append(formation)
        for kind, formations in by_type.items():
            for formation, per_formation in zip(formations, kind.tick_all(formations, dt), strict=True):
                speeds.update(formation.speeds())
                for aid, pose in per_formation.items():
                    targets[aid] = pose
                    agent = self._agent_lookup(aid)
                    if agent is None or not isinstance(agent.movement, BehaviorTreeMovement):
                        continue
                    agent.movement.command = HighLevelCommand(
                        agent_id=aid,
                        type=CommandType.NAVIGATE,
                        target_pose=pose,
                        desired_velocity=agent.state.desired_velocity,
                    )
        self._formation_targets = targets
        self._formation_speeds = speeds
        return targets

    def formation_target(self, agent_id: int) -> Pose2D | None:
        return self._formation_targets.get(agent_id)

    def formation_speeds(self) -> dict[int, float]:
        return self._formation_speeds

    def is_in_interaction(self, agent_id: int) -> bool:
        return any(role == MembershipRole.PARTICIPANT for role in self._agent_membership.get(agent_id, {}).values())

    def posture_of(self, agent_id: int) -> str:
        """Posture the agent's active interaction kind imposes, ``standing`` until it has arrived."""
        for iid, role in self._agent_membership.get(agent_id, {}).items():
            interaction = self.interactions.get(iid)
            if role != MembershipRole.PARTICIPANT or interaction is None or interaction.outcome != InteractionOutcome.ACTIVE:
                continue
            formation = interaction.contract.formation
            if formation is not None and not formation.arrived(agent_id):
                continue
            return InteractionType(interaction.type).kind.posture
        return "standing"

    def support_height(self, agent_id: int) -> float | None:
        """Height of the surface the arrived agent sits or lies on, above its floor, None while on its feet.

        Only explicit seats know their surface: an agent on a generated slot or an unmeasured seat stays on its feet.
        """
        for iid, role in self._agent_membership.get(agent_id, {}).items():
            interaction = self.interactions.get(iid)
            if role != MembershipRole.PARTICIPANT or interaction is None or interaction.outcome != InteractionOutcome.ACTIVE:
                continue
            if InteractionType(interaction.type).kind.posture == "standing" or interaction.contract.formation is None:
                continue
            seat = interaction.contract.formation.seat_of(agent_id)
            if isinstance(seat, SeatPose) and seat.height is not None:
                return seat.height
        return None

    def active_interaction(self, agent_id: int) -> tuple[int, int] | None:
        """(interaction_id, interaction_type) of the agent's ACTIVE participant interaction, or None."""
        for iid in self._iter_membership(agent_id, MembershipRole.PARTICIPANT):
            interaction = self.interactions.get(iid)
            if interaction is not None and interaction.outcome == InteractionOutcome.ACTIVE:
                return iid, interaction.type
        return None

    def postures(self) -> dict[int, str]:
        """posture_of for every agent it does not leave standing."""
        out: dict[int, str] = {}
        for interaction in self.interactions.values():
            if interaction.outcome != InteractionOutcome.ACTIVE or InteractionType(interaction.type).kind.posture == "standing":
                continue
            for pid in interaction.participants:
                if pid not in out and (posture := self.posture_of(pid)) != "standing":
                    out[pid] = posture
        return out

    def parked(self) -> dict[int, Pose2D]:
        """Agents held on an explicit seat by a posture-imposing interaction, and the seat pose."""
        out: dict[int, Pose2D] = {}
        for interaction in self.interactions.values():
            if interaction.outcome != InteractionOutcome.ACTIVE or InteractionType(interaction.type).kind.posture == "standing":
                continue
            formation = interaction.contract.formation
            if formation is None:
                continue
            for pid in interaction.participants:
                seat = formation.seat_of(pid)
                if seat is not None:
                    out[pid] = seat
        return out

    def is_in_queue(self, agent_id: int) -> bool:
        return any(role == MembershipRole.QUEUED for role in self._agent_membership.get(agent_id, {}).values())

    def force_stop(self, agent_id: int, reason: InteractionOutcome = InteractionOutcome.INTERRUPTED) -> None:
        had_memberships = agent_id in self._agent_membership
        for iid in self._iter_membership(agent_id, MembershipRole.PARTICIPANT):
            self.stop(agent_id, iid, reason=reason)
        for iid in self._iter_membership(agent_id, MembershipRole.QUEUED):
            interaction = self.interactions.get(iid)
            if interaction is None:
                continue
            contract = interaction.contract
            access: AccessPolicy | None = contract.access
            if access is not None:
                access.on_stop(interaction, agent_id)
            elif agent_id in contract.queue:
                contract.queue.remove(agent_id)
            self._on_formation_leave(interaction, agent_id)
        self._agent_membership.pop(agent_id, None)
        if had_memberships:
            self._update_bt_movement(agent_id, interaction_id=None, clear_command=True, last_outcome=reason)

    def stop_object(self, object_id: str | None = None) -> None:
        """Interrupt the live interactions bound to object_id, every object-bound one when None."""
        for iid, interaction in list(self.interactions.items()):
            if interaction.object_id is None or interaction.outcome in _ENDED_OUTCOMES:
                continue
            if object_id is None or interaction.object_id == object_id:
                self._teardown(iid, InteractionOutcome.INTERRUPTED, list(interaction.participants))

    def queue_length_for_object(self, object_id: str) -> int:
        total = 0
        for interaction in self.interactions.values():
            if interaction.object_id == object_id:
                total += interaction.contract.queue_length
        return total

    def _tick_access(self, dt: float) -> None:
        self._seat_queued_on_free_seats()
        for interaction in self.interactions.values():
            contract = interaction.contract
            access: AccessPolicy | None = contract.access

            if access is not None:
                promoted = access.tick(interaction, dt)
                for next_agent in promoted:
                    self._add_membership(next_agent, interaction.id, MembershipRole.PARTICIPANT)
                if promoted:
                    self._restart_hold(interaction)
                    self._maybe_activate(interaction)
                    for next_agent in promoted:
                        self._update_bt_movement(next_agent, interaction_id=interaction.id)
                        self._on_formation_join(interaction, next_agent)
                continue

            if not contract.queueable or not contract.queue:
                continue
            while not contract.is_full and contract.queue:
                next_agent = contract.queue.pop(0)
                interaction.participants.append(next_agent)
                contract.current_participants.append(next_agent)
                self._add_membership(next_agent, interaction.id, MembershipRole.PARTICIPANT)
                self._restart_hold(interaction)
                self._maybe_activate(interaction)
                self._update_bt_movement(next_agent, interaction_id=interaction.id)
                self._on_formation_join(interaction, next_agent)

    def _process_command(self, cmd: HighLevelCommand) -> None:
        ctype = cmd.type
        if ctype == CommandType.SEEK:
            self._handle_seek(cmd)
        elif ctype == CommandType.STOP:
            reason = cmd.reason
            target_id = cmd.interaction_target
            if target_id >= 0:
                self.stop(cmd.agent_id, target_id, reason=reason)
            else:
                self.force_stop(cmd.agent_id, reason=reason)

    def seek(self, agent_id: int, spec: SeekSpec) -> int | None:
        handle_kind = spec.interaction_type.kind.handle.kind
        # Symmetric FORMING must not short-circuit: a solo 1p owner must still scan so a
        # better-populated peer's interaction can take over.
        own_forming_iid: int | None = None
        for iid in self._iter_membership(agent_id):
            interaction = self.interactions.get(iid)
            if interaction is None or not self._matches(iid, spec):
                continue
            if interaction.outcome == InteractionOutcome.ACTIVE:
                return iid
            if interaction.outcome == InteractionOutcome.FORMING:
                if handle_kind != HandleKind.NONE:
                    return iid
                own_forming_iid = iid

        for iid in list(self._iter_membership(agent_id)):
            if iid == own_forming_iid:
                continue
            interaction = self.interactions.get(iid)
            if interaction is None or interaction.outcome in _ENDED_OUTCOMES:
                continue
            if self._matches(iid, spec):
                continue
            self.stop(agent_id, iid, reason=InteractionOutcome.INTERRUPTED)

        strategy = spec.interaction_type.kind.handle.strategy

        if not spec.offer:
            existing_iid = strategy.find(self, spec, agent_id)
            if existing_iid is not None and existing_iid != own_forming_iid:
                if own_forming_iid is not None:
                    self._detach_quiet(own_forming_iid, agent_id)
                if self.accept(agent_id, existing_iid):
                    existing = self.interactions[existing_iid]
                    if spec.duration is not None and spec.duration > 0:
                        existing.member_durations[agent_id] = spec.duration
                    if spec.formation_spec is not None and spec.formation_spec != existing.state.get("formation_spec"):
                        self._logger.warning(f"interaction {existing_iid} keeps the formation spec of its creator, agent {agent_id} asked for {spec.formation_spec}")
                return existing_iid

        if own_forming_iid is not None:
            return own_forming_iid

        if not strategy.can_create(self, spec, agent_id):
            return None

        if spec.interaction_type.kind.handle.kind == HandleKind.OBJECT and isinstance(spec.target, str) and self._world_knowledge is not None:
            obj = self._world_knowledge.resolve(spec.target, self._pose_lookup(agent_id), exclude_full=False)
            resolved_target = obj.object_id if obj is not None else None
            spec = attrs.evolve(spec, target=resolved_target)
        interaction = self._create_interaction(creator_id=agent_id, spec=spec)
        if spec.duration is not None and spec.duration > 0:
            interaction.member_durations[agent_id] = spec.duration
        return interaction.id

    def _detach_quiet(self, interaction_id: int, agent_id: int) -> None:
        # Silent counterpart to stop(): used by seek() mid-migration, when the agent is moving
        # from its own 1p FORMING into a peer's - emitting INTERRUPTED would fail the SeekNode.
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return
        contract = interaction.contract
        if agent_id in interaction.participants:
            interaction.participants.remove(agent_id)
        if agent_id in contract.current_participants:
            contract.current_participants.remove(agent_id)
        self._drop_membership(agent_id, interaction_id)
        self._on_formation_leave(interaction, agent_id)
        interaction.member_durations.pop(agent_id, None)
        if not interaction.participants and not contract.queue:
            interaction.outcome = InteractionOutcome.INTERRUPTED
            self._unmap_object_bound(interaction)

    def is_bound_matching(self, agent_id: int, spec: SeekSpec) -> bool:
        """True iff the agent is currently a PARTICIPANT in an ACTIVE interaction that matches
        the spec (same interaction_type + the handle strategy's matches() check passes). Used by
        BT nodes to decide whether to re-emit a seek or return SUCCESS."""
        for iid in self._iter_membership(agent_id, MembershipRole.PARTICIPANT):
            interaction = self.interactions.get(iid)
            if interaction is not None and interaction.outcome == InteractionOutcome.ACTIVE and self._matches(iid, spec):
                return True
        return False

    def _handle_seek(self, cmd: HighLevelCommand) -> None:
        assert cmd.spec is not None, "SEEK command requires spec"
        self.seek(cmd.agent_id, cmd.spec)

    def _matches(self, interaction_id: int, spec: SeekSpec) -> bool:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return False
        if interaction.type != int(spec.interaction_type):
            return False
        return spec.interaction_type.kind.handle.strategy.matches(interaction, spec)

    def _scan_tag(self, spec: SeekSpec, agent_id: int) -> int | None:
        visible = self._visibility_lookup(agent_id) if self._visibility_lookup is not None else None
        agent_pose = self._pose_lookup(agent_id)
        best_iid: int | None = None
        best_d = float("inf")
        for iid in self._interactions_by_type.get(int(spec.interaction_type), ()):
            interaction = self.interactions[iid]
            if interaction.outcome in _ENDED_OUTCOMES or not interaction.contract.can_admit:
                continue
            if spec.target is not None and interaction.service_tag != spec.target:
                continue
            provider_id = interaction.provider
            if provider_id is None or provider_id == agent_id:
                continue
            if visible is not None and provider_id not in visible:
                continue
            provider_pose = self._pose_lookup(provider_id)
            d = 0.0 if agent_pose is None or provider_pose is None else pose_distance(agent_pose, provider_pose)
            if d < best_d:
                best_d = d
                best_iid = iid
        return best_iid

    def _scan_agent(self, spec: SeekSpec, _agent_id: int) -> int | None:
        for iid in self._interactions_by_type.get(int(spec.interaction_type), ()):
            interaction = self.interactions[iid]
            if interaction.outcome in _ENDED_OUTCOMES or not interaction.contract.can_admit:
                continue
            if interaction.target_agent == spec.target:
                return iid
        return None

    def _scan_symmetric(self, spec: SeekSpec, agent_id: int) -> int | None:
        visible = self._visibility_lookup(agent_id) if self._visibility_lookup is not None else None
        agent_pose = self._pose_lookup(agent_id)
        best_iid: int | None = None
        best_d = float("inf")
        for iid in self._interactions_by_type.get(int(spec.interaction_type), ()):
            interaction = self.interactions[iid]
            if interaction.outcome in _ENDED_OUTCOMES or not interaction.contract.can_admit:
                continue
            peer_ids = [pid for pid in interaction.participants if pid != agent_id]
            if not peer_ids:
                continue
            if visible is not None and not any(pid in visible for pid in peer_ids):
                continue
            best_peer_d = float("inf")
            for pid in peer_ids:
                p = self._pose_lookup(pid)
                if p is None or agent_pose is None:
                    continue
                pd = pose_distance(agent_pose, p)
                if pd < best_peer_d:
                    best_peer_d = pd
            d = 0.0 if agent_pose is None or best_peer_d == float("inf") else best_peer_d
            if d < best_d:
                best_d = d
                best_iid = iid
        return best_iid

    def _find_object_bound(self, spec: SeekSpec, agent_id: int) -> int | None:
        if not isinstance(spec.target, str):
            return None
        itype_int = int(spec.interaction_type)
        iid = self._interaction_by_object_type.get((spec.target, itype_int))
        if iid is not None:
            interaction = self.interactions.get(iid)
            if interaction is not None and interaction.outcome not in _ENDED_OUTCOMES:
                if interaction.contract.can_admit and not self._seat_left_for_another(interaction):
                    return iid
        if self._world_knowledge is None:
            return None
        obj = self._world_knowledge.resolve(spec.target, self._pose_lookup(agent_id), exclude_full=False)
        if obj is None or obj.object_id == spec.target:
            return None
        iid = self._interaction_by_object_type.get((obj.object_id, itype_int))
        if iid is None:
            return None
        interaction = self.interactions.get(iid)
        if interaction is None or interaction.outcome in _ENDED_OUTCOMES:
            return None
        if not interaction.contract.can_admit or self._seat_left_for_another(interaction):
            return None
        return iid

    def _seat_left_for_another(self, interaction: InteractionState) -> bool:
        """A full seated interaction sends a seeker to a new interaction while its object has a seat nobody waits for."""
        if not interaction.state.get(_SEATED) or not interaction.contract.is_full or interaction.object_id is None:
            return False
        seated = self._seated_on(interaction.object_id)
        if any(other.contract.queue or self._unseated(other) for other in seated):
            return False
        return bool(self._fitting_seats(interaction.object_id))

    def _seat_queued_on_free_seats(self) -> None:
        """Move the head of a full seated interaction's queue into a new interaction while its object has a seat nobody waits for."""
        for object_id in sorted({i.object_id for i in self.interactions.values() if i.object_id is not None and i.state.get(_SEATED)}):
            while True:
                seated = self._seated_on(object_id)
                if len(self._fitting_seats(object_id)) <= sum(self._unseated(i) for i in seated):
                    break
                source = next((i for i in seated if i.contract.queue and len(i.contract.current_participants) >= i.contract.max_participants != -1), None)
                if source is None:
                    break
                agent_id = source.contract.queue.pop(0)
                duration = source.member_durations.pop(agent_id, None)
                self._drop_membership(agent_id, source.id)
                spec = SeekSpec(interaction_type=InteractionType(source.type), target=object_id, duration=duration)
                interaction = self._create_interaction(creator_id=agent_id, spec=spec)
                if duration is not None and duration > 0:
                    interaction.member_durations[agent_id] = duration

    def _create_interaction(
        self,
        creator_id: int,
        spec: SeekSpec,
    ) -> InteractionState:
        itype = spec.interaction_type
        contract = _make_contract(
            itype,
            min_participants=spec.min_participants,
            max_participants=spec.max_participants,
            queueable=spec.queueable,
        )
        if spec.duration is not None and spec.duration > 0:
            contract.duration = spec.duration

        iid = self.next_interaction_id
        self.next_interaction_id += 1

        handle = itype.kind.handle
        state_dict: dict[str, Any] = {}
        object_id = handle.strategy.populate_state(state_dict, spec.target, creator_id)
        if handle.kind == HandleKind.OBJECT and object_id is not None and self._world_knowledge is not None:
            obj = self._world_knowledge.get(object_id)
            if obj is not None:
                state_dict["object_type"] = obj.type
        if spec.offer:
            state_dict["provider"] = creator_id

        itype_int = int(itype)
        interaction = InteractionState(
            id=iid,
            type=itype_int,
            contract=contract,
            participants=[creator_id],
            state=state_dict,
            object_id=object_id,
        )
        contract.current_participants.append(creator_id)
        self._add_membership(creator_id, iid, MembershipRole.PARTICIPANT)
        self.interactions[iid] = interaction
        self._interactions_by_type.setdefault(itype_int, set()).add(iid)
        if object_id is not None:
            self._interaction_by_object_type[(object_id, itype_int)] = iid

        if spec.formation_spec is not None:
            interaction.state["formation_spec"] = spec.formation_spec
        contract.formation = self._resolve_formation(interaction)
        if contract.formation is not None:
            contract.formation.clearance = self._clearance
        self._on_formation_join(interaction, creator_id)

        self._maybe_activate(interaction)
        self._update_bt_movement(creator_id, interaction_id=iid)
        self._logger.debug(f"Interaction {iid} created: type={itype.name}, creator={creator_id}")
        return interaction

    def _resolve_formation(self, interaction: InteractionState) -> Formation | None:
        step_spec = interaction.formation_spec
        if isinstance(step_spec, FormationSpec):
            spec = step_spec
            anchor = self._anchor_from_spec(spec, interaction)
            if anchor is None:
                return None
            try:
                formation = Formation.create(
                    spec.type,
                    anchor=anchor,
                    agent_lookup=self._agent_lookup,
                    formation_scale=self._formation_scale,
                    **self._contact_params(interaction, spec.type, dict(spec.params or {})),
                )
            except (KeyError, TypeError) as e:
                self._logger.warning(f"Formation '{spec.type}' instantiation failed for interaction {interaction.id}: {e}")
                return None
            interaction.state["_active_formation_spec"] = spec
            return formation

        obj_spec = self._object_formation_spec(interaction.object_id)
        if obj_spec is not None:
            active_spec = obj_spec
        else:
            default_spec = InteractionType(interaction.type).kind.formation_default
            if not isinstance(default_spec, FormationSpec):
                return None
            active_spec = default_spec

        anchor = self._anchor_from_spec(active_spec, interaction)
        if anchor is None:
            return None

        params = self._contact_params(interaction, active_spec.type, dict(active_spec.params or {}))
        seat_object: str | None = None
        if interaction.object_id is not None and active_spec.type == "cluster" and self._seat_count(interaction.object_id):
            seat_object = interaction.object_id
            assert self._world_knowledge is not None
            obj = self._world_knowledge.get(seat_object)
            assert obj is not None
            params["slot_poses"] = list(obj.seats)
            params["admissible"] = self._seat_admissible(seat_object)
        try:
            formation = Formation.create(
                active_spec.type,
                anchor=anchor,
                agent_lookup=self._agent_lookup,
                formation_scale=self._formation_scale,
                **params,
            )
        except (KeyError, TypeError) as e:
            self._logger.warning(f"Formation '{active_spec.type}' instantiation failed for interaction {interaction.id}: {e}")
            return None
        interaction.state["_active_formation_spec"] = active_spec
        if seat_object is not None:
            interaction.state[_SEATED] = True
            interaction.contract.seat_gate = self._seat_gate(seat_object, interaction)
        return formation

    def _seat_admissible(self, object_id: str) -> Callable[[int], bool]:
        return lambda seat: self._seat_fits(object_id, seat)

    def _seat_gate(self, object_id: str, interaction: InteractionState) -> Callable[[], bool]:
        return lambda: len(self._fitting_seats(object_id)) > self._unseated(interaction)

    def _seat_count(self, object_id: str) -> int:
        obj = self._world_knowledge.get(object_id) if self._world_knowledge is not None else None
        return len(obj.seats) if obj is not None else 0

    def held_seats(self, object_id: str) -> set[int]:
        """Seat indices of the object held by any live interaction on it."""
        held: set[int] = set()
        for interaction in self.interactions.values():
            if interaction.object_id != object_id or interaction.outcome in _ENDED_OUTCOMES or interaction.contract.formation is None:
                continue
            held.update(interaction.contract.formation.held_seats())
        return held

    def _seat_fits(self, object_id: str, seat: int, held: set[int] | None = None) -> bool:
        """The seat is free and conflicts with no held seat of its object."""
        held = self.held_seats(object_id) if held is None else held
        if seat in held:
            return False
        obj = self._world_knowledge.get(object_id) if self._world_knowledge is not None else None
        conflicts = obj.seat_conflicts[seat] if obj is not None and seat < len(obj.seat_conflicts) else frozenset()
        return not (conflicts & held)

    def _fitting_seats(self, object_id: str) -> list[int]:
        held = self.held_seats(object_id)
        return [seat for seat in range(self._seat_count(object_id)) if self._seat_fits(object_id, seat, held)]

    @staticmethod
    def _unseated(interaction: InteractionState) -> int:
        formation = interaction.contract.formation
        return len(interaction.participants) - (len(formation.held_seats()) if formation is not None else 0)

    def _seated_on(self, object_id: str) -> list[InteractionState]:
        return [i for _, i in sorted(self.interactions.items()) if i.object_id == object_id and i.state.get(_SEATED) and i.outcome not in _ENDED_OUTCOMES]

    def _object_formation_spec(self, object_id: str | None) -> FormationSpec | None:
        if object_id is None or self._world_knowledge is None:
            return None
        obj = self._world_knowledge.get(object_id)
        if obj is None:
            return None
        return obj.formation

    def _anchor_from_spec(self, spec: FormationSpec, interaction: InteractionState) -> Anchor | None:
        kind = spec.anchor_kind
        if kind is AnchorKind.OBJECT:
            ref = spec.anchor_ref or interaction.object_id
            if ref and self._world_knowledge is not None:
                return ObjectAnchor(world_knowledge=self._world_knowledge, object_id=ref)
            return None
        if kind is AnchorKind.AGENT:
            if spec.anchor_ref is None:
                return None
            try:
                aid = int(spec.anchor_ref)
            except (TypeError, ValueError):
                return None
            if aid == 0:
                return None
            return AgentAnchor(pose_lookup=self._pose_lookup, agent_id=aid)
        if kind is AnchorKind.PROVIDER:
            provider_id = interaction.provider
            if provider_id is not None:
                return AgentAnchor(pose_lookup=self._pose_lookup, agent_id=provider_id)
            return None
        if kind is AnchorKind.LEADER:
            return AgentAnchor(pose_lookup=self._pose_lookup, agent_id=interaction.participants[0])
        if kind is AnchorKind.POSE:
            return PoseAnchor(fixed=spec.anchor_pose or Pose2D())
        if kind is AnchorKind.CENTROID:
            members_ref = interaction.participants
            return CentroidAnchor(
                pose_lookup=self._pose_lookup,
                members_fn=lambda: list(members_ref),
            )
        return None

    def _teardown(
        self,
        interaction_id: int,
        outcome: int = InteractionOutcome.INTERRUPTED,
        participants: list[int] | None = None,
    ) -> None:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return
        if interaction.outcome == InteractionOutcome.ACTIVE and (edge := _TERMINAL_EDGES.get(int(outcome))) is not None:
            self._emit(interaction, edge, participants)
        interaction.outcome = outcome
        self._unmap_object_bound(interaction)
        self._logger.debug(f"Interaction {interaction_id} torn down (outcome={InteractionOutcome(outcome).name}, participants={interaction.participants})")
        for agent_id in list(interaction.participants):
            self._current_departed.add(agent_id)
            self._drop_membership(agent_id, interaction_id)
            self._on_formation_leave(interaction, agent_id)
        for agent_id in list(interaction.contract.queue):
            self._current_departed.add(agent_id)
            self._drop_membership(agent_id, interaction_id)
            self._on_formation_leave(interaction, agent_id)
            self._update_bt_movement(agent_id, interaction_id=None, clear_command=True, last_outcome=outcome)
        for agent_id in list(interaction.participants):
            self._update_bt_movement(agent_id, interaction_id=None, clear_command=True, last_outcome=outcome)
        interaction.member_durations.clear()

    def _unmap_object_bound(self, interaction: InteractionState) -> None:
        key = (interaction.object_id or "", interaction.type)
        if interaction.object_id is not None and self._interaction_by_object_type.get(key) == interaction.id:
            del self._interaction_by_object_type[key]

    def _tick_durations(self, dt: float) -> None:
        for iid, interaction in list(self.interactions.items()):
            contract = interaction.contract
            if interaction.outcome != InteractionOutcome.ACTIVE or not self.all_arrived(interaction):
                continue
            if interaction.participants and not self.is_holding(interaction):
                interaction.state[_HOLDING] = True
                self._emit(interaction, LifecycleEdge.HOLD_ONSET)
            if contract.duration is None:
                continue
            contract.elapsed += dt
            if contract.elapsed < contract.duration:
                continue
            if contract.access is not None and contract.queue:
                released = [pid for pid in interaction.participants if pid != interaction.provider]
                if released:
                    self._emit(interaction, LifecycleEdge.RELEASED, released)
                interaction.state[_HOLDING] = False
                for pid in released:
                    self._release_participant(interaction, pid)
                promoted = contract.access.tick(interaction, 0.0)
                for next_agent in promoted:
                    self._add_membership(next_agent, iid, MembershipRole.PARTICIPANT)
                    self._update_bt_movement(next_agent, interaction_id=iid)
                    self._on_formation_join(interaction, next_agent)
                if promoted:
                    stored = interaction.member_durations.get(promoted[0])
                    if stored is not None and stored > 0:
                        contract.duration = stored
                contract.elapsed = 0.0
                continue
            self._teardown(iid, InteractionOutcome.COMPLETED)

    def _release_participant(self, interaction: InteractionState, agent_id: int) -> None:
        self._current_departed.add(agent_id)
        contract = interaction.contract
        if agent_id in interaction.participants:
            interaction.participants.remove(agent_id)
        if agent_id in contract.current_participants:
            contract.current_participants.remove(agent_id)
        self._drop_membership(agent_id, interaction.id)
        interaction.member_durations.pop(agent_id, None)
        self._on_formation_leave(interaction, agent_id)
        self._update_bt_movement(agent_id, interaction_id=None, clear_command=True, last_outcome=InteractionOutcome.COMPLETED)

    def _maybe_activate(self, interaction: InteractionState) -> None:
        if interaction.outcome != InteractionOutcome.FORMING:
            return
        if len(interaction.participants) >= interaction.contract.min_participants:
            interaction.outcome = InteractionOutcome.ACTIVE
            self._emit(interaction, LifecycleEdge.ACTIVATED)
            for pid in interaction.participants:
                self._update_bt_movement(pid, interaction_id=interaction.id)

    def _prune_ended_interactions(self) -> None:
        to_remove = [iid for iid, interaction in self.interactions.items() if interaction.outcome in _ENDED_OUTCOMES]
        for iid in to_remove:
            interaction = self.interactions[iid]
            bucket = self._interactions_by_type.get(interaction.type)
            if bucket is not None:
                bucket.discard(iid)
                if not bucket:
                    del self._interactions_by_type[interaction.type]
            del self.interactions[iid]

    def _prune_dead_interactions(self) -> None:
        to_remove = [iid for iid, interaction in self.interactions.items() if interaction.outcome == InteractionOutcome.ACTIVE and len(interaction.participants) < interaction.contract.min_participants]
        for iid in to_remove:
            self._teardown(iid)
