"""Core data structures for arena_humansim."""

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    from collections.abc import Callable

    from arena_humansim.core.access import AccessPolicy
    from arena_humansim.core.formation import Formation
    from arena_humansim.core.interaction_kinds import InteractionType
    from arena_humansim.utils.scenario import FormationConfig


Segment = tuple[tuple[float, float], tuple[float, float]]
Segments = list[Segment]


class WallAware:
    def set_walls(self, segments: Segments) -> None:
        pass


class WaypointMode(enum.IntEnum):
    REPEAT = 0
    REVERSE = 1
    ONCE = 2
    RANDOM = 3


class CommandType(enum.IntEnum):
    NAVIGATE = 0
    STOP = 1
    SEEK = 2


class InteractionOutcome(enum.IntEnum):
    FORMING = 0
    ACTIVE = 1
    COMPLETED = 2
    INTERRUPTED = 3
    CANCELED = 4


@attrs.define
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@attrs.define
class SeatPose(Pose2D):
    """An explicit slot on a piece of furniture (seat, bed): where the agent is parked, plus the surface it rests on."""

    height: float | None = None  # surface height above the object's floor, m, None = not measured


def pose_distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


Goal = Pose2D | tuple[float, float]


class AnchorKind(enum.StrEnum):
    OBJECT = "object"
    AGENT = "agent"
    PROVIDER = "provider"
    POSE = "pose"
    CENTROID = "centroid"
    LEADER = "leader"


_VALID_FORMATION_TYPES = ("line", "cluster", "f_formation", "dyad", "walk")


@attrs.define
class FormationSpec:
    """Runtime representation of a scenario FormationConfig on a WorldObject."""

    type: str
    params: dict[str, float] = attrs.Factory(dict)
    anchor_kind: AnchorKind = AnchorKind.OBJECT
    anchor_ref: str | None = None
    anchor_pose: Pose2D | None = None

    @classmethod
    def from_config(cls, cfg: FormationConfig | None) -> FormationSpec | None:
        if cfg is None or not cfg.type:
            return None
        if cfg.type not in _VALID_FORMATION_TYPES:
            raise ValueError(f"Unknown formation type '{cfg.type}'. Valid: {_VALID_FORMATION_TYPES}")
        anchor_kind: AnchorKind = AnchorKind.OBJECT
        anchor_ref: str | None = None
        anchor_pose: Pose2D | None = None
        if cfg.anchor is not None:
            try:
                anchor_kind = AnchorKind(cfg.anchor.kind)
            except ValueError:
                raise ValueError(f"Unknown anchor kind '{cfg.anchor.kind}'. Valid: {[k.value for k in AnchorKind]}") from None
            anchor_ref = cfg.anchor.ref
            if cfg.anchor.pose is not None:
                anchor_pose = Pose2D(
                    x=float(cfg.anchor.pose.x),
                    y=float(cfg.anchor.pose.y),
                    theta=float(cfg.anchor.pose.theta),
                )
        return cls(
            type=cfg.type,
            params=dict(cfg.params),
            anchor_kind=anchor_kind,
            anchor_ref=anchor_ref,
            anchor_pose=anchor_pose,
        )


@attrs.define
class AgentState:
    agent_id: int = 0
    pose: Pose2D = attrs.Factory(Pose2D)
    velocity: tuple[float, float] = (0.0, 0.0)
    desired_velocity: float = 1.3
    kind: int = 0
    animation_state: int = 0


@attrs.define
class WorldAgentState:
    pose: Pose2D = attrs.Factory(Pose2D)
    velocity: tuple[float, float] = (0.0, 0.0)


WorldState = dict[int, WorldAgentState]


@attrs.define
class BeliefState:
    """Perceived world state for an agent (not ground truth)."""

    agent_id: int = 0
    observed_agents: list[AgentState] = attrs.Factory(list)
    extra: dict[str, object] = attrs.Factory(dict)


class AcceptResult(enum.IntEnum):
    BECAME_PARTICIPANT = 0
    QUEUED = 1
    REJECTED = 2


@attrs.define
class InteractionContract:
    type: int = 0
    min_participants: int = 2
    max_participants: int = 2  # -1 for unbounded
    current_participants: list[int] = attrs.Factory(list)
    queueable: bool = False
    max_queue: int = -1  # -1 = unbounded
    queue: list[int] = attrs.Factory(list)
    duration: float | None = None  # seconds; None = no contract-level timeout
    elapsed: float = 0.0
    access: AccessPolicy | None = None
    formation: Formation | None = None
    seat_gate: Callable[[], bool] | None = None  # None = no seats to wait for

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def remaining_slots(self) -> int:
        if self.max_participants == -1:
            return -1
        return self.max_participants - len(self.current_participants)

    @property
    def is_full(self) -> bool:
        if self.max_participants != -1 and len(self.current_participants) >= self.max_participants:
            return True
        return self.seat_gate is not None and not self.seat_gate()

    @property
    def can_admit(self) -> bool:
        return not self.is_full or self.queueable


@attrs.define
class InteractionState:
    id: int = 0
    type: int = 0
    contract: InteractionContract = attrs.Factory(InteractionContract)
    participants: list[int] = attrs.Factory(list)
    state: dict[str, Any] = attrs.Factory(dict)
    object_id: str | None = None
    outcome: int = 0  # InteractionOutcome.FORMING
    member_durations: dict[int, float] = attrs.Factory(dict)

    @property
    def provider(self) -> int | None:
        return self.state.get("provider")

    @property
    def target_agent(self) -> int | None:
        return self.state.get("target_agent")

    @property
    def service_tag(self) -> str | None:
        return self.state.get("service_tag")

    @property
    def object_type(self) -> str | None:
        return self.state.get("object_type")

    @property
    def formation_spec(self) -> FormationSpec | None:
        return self.state.get("formation_spec")


@attrs.frozen
class SeekSpec:
    interaction_type: InteractionType
    target: str | int | None = None
    offer: bool = False
    min_participants: int | None = None
    max_participants: int | None = None
    queueable: bool | None = None
    formation_spec: FormationSpec | None = None
    duration: float | None = None


@attrs.define
class HighLevelCommand:
    agent_id: int = 0
    type: CommandType = CommandType.NAVIGATE
    target_pose: Pose2D = attrs.Factory(Pose2D)
    desired_velocity: float = 1.3
    interaction_target: int = -1
    reason: InteractionOutcome = InteractionOutcome.INTERRUPTED
    spec: SeekSpec | None = None


@attrs.define
class WaypointMovement:
    waypoints: list[Pose2D] = attrs.Factory(list)
    radii: list[float] = attrs.Factory(list)
    index: int = 0
    mode: WaypointMode = WaypointMode.REPEAT
    forward: bool = True


@attrs.frozen
class GestureIntent:
    slot: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    clip: str = ""  # body slot: animation clip name
    hand: str = ""  # arm slot: dominant hand l | r, empty = default
    render_pose_override: bool = False  # body slot: (x, y) is a render-only position override


@attrs.define
class BehaviorTreeMovement:
    command: HighLevelCommand | None = None
    last_outcome: int | None = None
    interaction_id: int | None = None
    heading_goal: float | None = None
    gestures: tuple[GestureIntent, ...] = ()
    posture: str = ""  # authored posture while a step holds one (standing | seated | prone), empty = none


class ShapeType(enum.Enum):
    CIRCLE = "circle"
    POLYGON = "polygon"


class SourceType(enum.Enum):
    POISSON = "poisson"  # arrivals at rate_profile, lost over max_concurrent
    MAX = "max"  # holds max_concurrent alive, refills on despawn


@attrs.define
class Shape:
    type: ShapeType = ShapeType.POLYGON
    radius: float = 0.0
    vertices: list[Pose2D] = attrs.field(factory=list)


@attrs.define
class RateKeyframe:
    t: float = 0.0  # seconds since sim start
    rate: float = 0.0  # agents/sec (lambda for Poisson)


@attrs.define
class SinkAffinity:
    sink_name: str = ""  # name of the target sink
    weight: float = 1.0  # relative weight (normalized at runtime)


class AgentKind(enum.IntEnum):
    HUMAN = 0
    ROBOT = 1


@attrs.define
class AgentTemplate:
    desired_velocity_min: float = 0.0
    desired_velocity_max: float = 0.0
    agent_radius: float = 0.0
    agent_type: str = "adult"
    sink_affinity: list[SinkAffinity] = attrs.Factory(list)
    kind: AgentKind = AgentKind.HUMAN
    policy: str = ""
    policy_params: str = ""


@attrs.define
class SourceConfig:
    name: str = ""
    pose: Pose2D = attrs.Factory(Pose2D)
    shape: Shape = attrs.Factory(Shape)
    type: SourceType = SourceType.POISSON
    rate_profile: list[RateKeyframe] = attrs.Factory(list)
    max_concurrent: int = -1  # -1 = unlimited
    max_total: int = -1  # -1 = unlimited
    agent: AgentTemplate = attrs.Factory(AgentTemplate)


@attrs.define
class SinkConfig:
    name: str = ""
    pose: Pose2D = attrs.Factory(Pose2D)
    shape: Shape = attrs.Factory(Shape)
    absorption_radius: float = 0.5
    capacity: int = -1  # -1 = unlimited simultaneous


@attrs.define
class FlowConfig:
    sources: list[SourceConfig] = attrs.Factory(list)
    sinks: list[SinkConfig] = attrs.Factory(list)


@attrs.define
class AgentLifetime:
    agent_id: int = 0
    source_name: str = ""  # which source spawned this agent ("" if explicit)
    source_seq: int = 0
    spawn_tick: int = 0  # tick when agent was created
    max_lifetime_s: float = -1  # -1 = no TTL
    target_sink_name: str = ""  # "" = no sink target
    pending_despawn: bool = False  # True if reached sink but in interaction


@attrs.define
class SpawnRequest:
    """Lightweight spawn descriptor - AgentManager materializes into BaseAgent."""

    agent_id: int = 0  # 0 = manager allocates
    pose: Pose2D = attrs.Factory(Pose2D)
    desired_velocity: float = 1.3
    agent_radius: float = 0.35
    agent_type: str = "adult"
    waypoints: list[Pose2D] = attrs.Factory(list)
    lifetime: AgentLifetime = attrs.Factory(AgentLifetime)


@attrs.define
class DespawnRequest:
    agent_id: int = 0
    force: bool = False  # True = TTL hard cutoff, force-STOP interactions
    reason: str = "sink"  # "sink" | "ttl" | "deferred"


@attrs.define
class NeedState:
    """Runtime state for a single need."""

    value: float = 100.0
    decay_rate: float = 0.5


@attrs.define
class NeedsState:
    """Per-agent mutable needs container."""

    needs: dict[str, NeedState] = attrs.Factory(dict)

    def decay(self, dt: float, modifiers: dict[str, float] | None = None) -> None:
        """Tick all needs downward. modifiers: need_name -> decay multiplier."""
        for name, need in self.needs.items():
            mod = modifiers.get(name, 1.0) if modifiers else 1.0
            need.value = max(0.0, need.value - need.decay_rate * dt * mod)

    def satisfy(self, deltas: dict[str, float]) -> None:
        """Apply satisfaction from a completed action."""
        for name, amount in deltas.items():
            if name in self.needs:
                self.needs[name].value = min(100.0, self.needs[name].value + amount)

    def set(self, name: str, value: float) -> None:
        """Overwrite one need, clamped to [0, 100]. Unknown names are ignored."""
        need = self.needs.get(name)
        if need is not None:
            need.value = min(100.0, max(0.0, value))
