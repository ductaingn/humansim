"""InteractionType enum and per-type registry: handle addressing, contract defaults, formation defaults."""

from __future__ import annotations

import enum
import functools
import math
from typing import TYPE_CHECKING

import attrs

from arena_humansim.utils.const import DISTANCE_TOLERANCE
from arena_humansim.utils.types import AnchorKind, FormationSpec

if TYPE_CHECKING:
    from arena_humansim.core.interaction_manager import InteractionManager
    from arena_humansim.utils.types import InteractionState, SeekSpec


POSTURES = ("standing", "seated", "prone")


class InteractionType(enum.IntEnum):
    # Values are persisted in logs/replays - never reassign or reuse an index.
    TALK_TO = 0
    GROUP_CONVERSATION = 1
    SIT_ON = 3
    LIE_ON = 4
    USE = 5
    QUEUE_USE = 6
    WAVE_AT = 7
    BLOCK = 8
    SERVICE = 9
    HUG = 10
    SHAKE_HAND = 11
    GROUP_WALK = 12

    @property
    def kind(self) -> InteractionKind:
        return _registry()[self]

    @property
    def is_object_bound(self) -> bool:
        return self.kind.is_object_bound


class HandleKind(enum.Enum):
    NONE = "none"
    TAG = "tag"
    AGENT = "agent"
    OBJECT = "object"


class AccessKind(enum.StrEnum):
    FIFO = "fifo"


class MembershipRole(enum.IntEnum):
    PARTICIPANT = 0
    QUEUED = 1


class HandleStrategy:
    target_shape: tuple[type, ...] = ()
    target_required_for_create: bool = False
    offer_required_for_create: bool = False

    def matches(self, interaction: InteractionState, spec: SeekSpec) -> bool:
        raise NotImplementedError

    def find(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> int | None:
        raise NotImplementedError

    def can_create(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> bool:
        if self.offer_required_for_create and not spec.offer:
            return False
        if self.target_required_for_create and not isinstance(spec.target, self.target_shape):
            return False
        return True

    def populate_state(self, state: dict, target: str | int | None, creator_id: int) -> str | None:
        return None


@attrs.frozen
class HandleSpec:
    kind: HandleKind
    state_key: str | None = None
    strategy: HandleStrategy = attrs.field(factory=lambda: HandleStrategy())


@attrs.frozen
class ContractDefaults:
    min_participants: int
    max_participants: int
    queueable: bool = False
    access: AccessKind | None = None


@attrs.frozen
class ClipSet:
    """Clips one interaction plays, one per role, and the dyad stance they were made in.

    Members take the roles in join order, cycling when there are more members than roles, so a one-role set plays the
    same clip on everyone. ``separation`` (m) and ``headings`` (rad, each role's turn off facing its partner) replace
    the kind's default dyad placement, contact clips only touch where they were made to.
    """

    roles: tuple[str, ...]
    separation: float | None = None
    headings: tuple[float, float] = (0.0, 0.0)

    def role(self, index: int) -> str:
        return self.roles[index % len(self.roles)]

    @classmethod
    def pair(cls, a: str, b: str, dx: float, dy: float, dyaw: float) -> ClipSet:
        """Two role clips from the ``pair.partner_offset`` of role a's annotation: b at (dx, dy) in a's frame (x
        forward, y left, m), turned dyaw (rad) from a."""
        axis = math.atan2(dy, dx)
        turn_b = math.remainder(dyaw - axis - math.pi, math.tau)
        return cls(roles=(a, b), separation=math.hypot(dx, dy), headings=(-axis, turn_b))


@attrs.frozen
class InteractionKind:
    label: str
    handle: HandleSpec
    contract_defaults: ContractDefaults
    formation_default: FormationSpec | None = None
    allows_offer: bool = False
    interaction_radius: float = DISTANCE_TOLERANCE
    clips: tuple[ClipSet, ...] = ()  # one drawn per interaction, shown while a participant, authored attention.clip overrides
    posture: str = "standing"  # standing | seated | prone while a participant
    render_pose_override: bool = False  # display the formation target instead of the physics pose (contact kinds)

    @property
    def is_object_bound(self) -> bool:
        return self.handle.kind == HandleKind.OBJECT

    @property
    def clip(self) -> str:
        """First role of the first clip set, what a member shows before its interaction drew a set."""
        return self.clips[0].roles[0] if self.clips else ""


def _fs(type_: str, anchor_kind: AnchorKind, params: dict[str, float] | None = None) -> FormationSpec:
    return FormationSpec(type=type_, params=dict(params or {}), anchor_kind=anchor_kind)


def _one(*names: str) -> tuple[ClipSet, ...]:
    """One-role sets: each interaction plays one of the clips on every member."""
    return tuple(ClipSet(roles=(name,)) for name in names)


# Clip sets of the two-person kinds. Contact pairs carry the stance their clips were made in (hug_v7_* and
# shake_hand_v7_*: the separation of their annotation, facing each other; intermask_*: pair.partner_offset of role a,
# see ClipSet.pair).
TALK_CLIPS = (*_one("talk_with_arm_gesture", "talk_with_arm_gesture_2", "talk_with_arm_gesture_3", "talk_with_arm_gesture_4", "talk_with_arm_gesture_v1"), ClipSet(roles=("intermask_talk_a", "intermask_talk_b")))
GROUP_CLIPS = (ClipSet(roles=("intermask_group_conversation_a", "intermask_group_conversation_b")),)
WAVE_CLIPS = (*_one("wave"), ClipSet(roles=("intermask_wave_at_a", "intermask_wave_at_b")))
HUG_CLIPS = (
    ClipSet(roles=("hug_v7_air_w", "hug_v7_air_m"), separation=0.3126),
    ClipSet(roles=("hug_v7_hero_w", "hug_v7_hero_m"), separation=0.2838),
)
SHAKE_CLIPS = (ClipSet(roles=("shake_hand_v7_doctor_pair",), separation=0.88),)


class _NoneStrategy(HandleStrategy):
    # Target must be omitted for symmetric interactions (TALK_TO / GROUP_CONVERSATION / WAVE_AT).
    target_shape = ()
    target_required_for_create = False
    offer_required_for_create = False

    def matches(self, interaction: InteractionState, spec: SeekSpec) -> bool:
        return interaction.type == int(spec.interaction_type)

    def find(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> int | None:
        return im._scan_symmetric(spec, agent_id)


class _TagStrategy(HandleStrategy):
    target_shape = (str,)
    target_required_for_create = False  # target presence is enforced only for offer=True (checked by loader)
    offer_required_for_create = True

    def matches(self, interaction: InteractionState, spec: SeekSpec) -> bool:
        if spec.target is not None and interaction.service_tag != spec.target:
            return False
        return True

    def find(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> int | None:
        return im._scan_tag(spec, agent_id)

    def populate_state(self, state: dict, target: str | int | None, creator_id: int) -> str | None:
        if isinstance(target, str):
            state["service_tag"] = target
        return None


class _AgentStrategy(HandleStrategy):
    target_shape = (int,)
    target_required_for_create = True
    offer_required_for_create = False

    def matches(self, interaction: InteractionState, spec: SeekSpec) -> bool:
        return interaction.target_agent == spec.target

    def find(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> int | None:
        return im._scan_agent(spec, agent_id)

    def can_create(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> bool:
        del im, agent_id
        return isinstance(spec.target, int) and spec.target != 0

    def populate_state(self, state: dict, target: str | int | None, creator_id: int) -> str | None:
        del creator_id
        if isinstance(target, int) and target != 0:
            state["target_agent"] = target
        return None


class _ObjectStrategy(HandleStrategy):
    target_shape = (str,)
    target_required_for_create = True
    offer_required_for_create = False

    def matches(self, interaction: InteractionState, spec: SeekSpec) -> bool:
        if interaction.object_id is None or spec.target is None:
            return False
        if interaction.object_id == spec.target:
            return True
        return interaction.object_type == spec.target

    def find(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> int | None:
        return im._find_object_bound(spec, agent_id)

    def can_create(self, im: InteractionManager, spec: SeekSpec, agent_id: int) -> bool:
        if not isinstance(spec.target, str) or im._world_knowledge is None:
            return False
        return im._world_knowledge.resolve(spec.target, im._pose_lookup(agent_id), exclude_full=False) is not None

    def populate_state(self, state: dict, target: str | int | None, creator_id: int) -> str | None:
        del state, creator_id
        return target if isinstance(target, str) else None


_SYMMETRIC_HANDLE = HandleSpec(kind=HandleKind.NONE, strategy=_NoneStrategy())
# OBJECT writes to InteractionState.object_id directly, not into state[]; no state_key.
_OBJECT_HANDLE = HandleSpec(kind=HandleKind.OBJECT, strategy=_ObjectStrategy())
_AGENT_HANDLE = HandleSpec(kind=HandleKind.AGENT, state_key="target_agent", strategy=_AgentStrategy())
_TAG_HANDLE = HandleSpec(kind=HandleKind.TAG, state_key="service_tag", strategy=_TagStrategy())


@functools.cache
def _registry() -> dict[InteractionType, InteractionKind]:
    return {
        InteractionType.TALK_TO: InteractionKind(
            label="TALK",
            handle=_SYMMETRIC_HANDLE,
            contract_defaults=ContractDefaults(min_participants=2, max_participants=2, queueable=False),
            formation_default=_fs("dyad", AnchorKind.CENTROID),
            interaction_radius=2.0,
            clips=TALK_CLIPS,
        ),
        InteractionType.GROUP_CONVERSATION: InteractionKind(
            label="GROUP",
            handle=_SYMMETRIC_HANDLE,
            contract_defaults=ContractDefaults(min_participants=2, max_participants=-1, queueable=False),
            formation_default=_fs("f_formation", AnchorKind.CENTROID),
            interaction_radius=3.0,
            clips=GROUP_CLIPS,
        ),
        InteractionType.SIT_ON: InteractionKind(
            label="SIT",
            handle=_OBJECT_HANDLE,
            contract_defaults=ContractDefaults(min_participants=1, max_participants=1, queueable=True, access=AccessKind.FIFO),
            formation_default=_fs("cluster", AnchorKind.OBJECT),
            clips=_one("sit"),
            posture="seated",
        ),
        InteractionType.LIE_ON: InteractionKind(
            label="LIE",
            handle=_OBJECT_HANDLE,
            contract_defaults=ContractDefaults(min_participants=1, max_participants=1, queueable=True, access=AccessKind.FIFO),
            formation_default=_fs("cluster", AnchorKind.OBJECT),
            posture="prone",
        ),
        InteractionType.USE: InteractionKind(
            label="USE",
            handle=_OBJECT_HANDLE,
            contract_defaults=ContractDefaults(min_participants=1, max_participants=1, queueable=True, access=AccessKind.FIFO),
            formation_default=None,
        ),
        InteractionType.QUEUE_USE: InteractionKind(
            label="QUEUE",
            handle=_OBJECT_HANDLE,
            contract_defaults=ContractDefaults(min_participants=1, max_participants=1, queueable=True, access=AccessKind.FIFO),
            formation_default=_fs("line", AnchorKind.OBJECT, {"base_step": 1.0}),
        ),
        InteractionType.WAVE_AT: InteractionKind(
            label="WAVE",
            handle=_SYMMETRIC_HANDLE,
            contract_defaults=ContractDefaults(min_participants=2, max_participants=2, queueable=False),
            formation_default=None,
            interaction_radius=6.0,  # above typical vision_range, a wave is matched at a distance
            clips=WAVE_CLIPS,
        ),
        InteractionType.HUG: InteractionKind(
            label="HUG",
            handle=_SYMMETRIC_HANDLE,
            contract_defaults=ContractDefaults(min_participants=2, max_participants=2, queueable=False),
            formation_default=_fs("dyad", AnchorKind.CENTROID, {"separation": 0.3}),
            interaction_radius=0.3,
            clips=HUG_CLIPS,
            render_pose_override=True,
        ),
        InteractionType.SHAKE_HAND: InteractionKind(
            label="SHAKE",
            handle=_SYMMETRIC_HANDLE,
            contract_defaults=ContractDefaults(min_participants=2, max_participants=2, queueable=False),
            formation_default=_fs("dyad", AnchorKind.CENTROID, {"separation": 0.8}),
            interaction_radius=0.3,
            clips=SHAKE_CLIPS,
            render_pose_override=True,
        ),
        InteractionType.GROUP_WALK: InteractionKind(
            label="WALK",
            handle=_SYMMETRIC_HANDLE,
            contract_defaults=ContractDefaults(min_participants=2, max_participants=-1, queueable=False),
            formation_default=_fs("walk", AnchorKind.LEADER),
            interaction_radius=3.0,
        ),
        InteractionType.BLOCK: InteractionKind(
            label="BLOCK",
            handle=_AGENT_HANDLE,
            contract_defaults=ContractDefaults(min_participants=1, max_participants=2, queueable=False),
            formation_default=None,
        ),
        InteractionType.SERVICE: InteractionKind(
            label="SERVICE",
            handle=_TAG_HANDLE,
            contract_defaults=ContractDefaults(min_participants=1, max_participants=-1, queueable=True, access=AccessKind.FIFO),
            formation_default=_fs("f_formation", AnchorKind.PROVIDER),
            allows_offer=True,
            interaction_radius=3.0,
        ),
    }


def is_object_bound_name(interaction_name: str) -> bool:
    try:
        return InteractionType[interaction_name].is_object_bound
    except KeyError:
        return False
