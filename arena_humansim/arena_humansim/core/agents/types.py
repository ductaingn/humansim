__all__ = [
    "ActionDef",
    "AgentType",
    "AttentionDef",
    "AttentionRef",
    "AttentionStepDef",
    "CadenceDist",
    "ChannelDef",
    "ClipDef",
    "ConditionalTune",
    "STAIRS_KINDS",
    "GoToStepDef",
    "HEADING_SOURCES",
    "KINEMATICS",
    "LocomotionDist",
    "NeedCondition",
    "NeedDist",
    "ParamDist",
    "PerceptionDist",
    "Pose3",
    "RelativeRef",
    "RobotRef",
    "RobotStepDef",
    "RobotTrigger",
    "SampledLocomotion",
    "SampledNeed",
    "SampledParams",
    "SampledPerception",
    "SequenceDef",
    "StepDef",
    "TUNABLE",
    "TUNABLE_PERCEPTION",
    "TransitionDef",
    "TuneDef",
    "VarDef",
    "sample_agent_type",
]

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import overload

import attrs
import numpy as np

from arena_humansim.utils.types import FormationSpec, Pose2D


@attrs.frozen
class ParamDist:
    mean: float
    std: float = 0.0
    clip_low: float = 0.01
    clip_high: float = float("inf")

    def with_mean(self, mean: float) -> "ParamDist":
        """Same spread around a new mean. The clip window moves with it and never drops below its old floor or the mean."""
        delta = mean - self.mean
        return attrs.evolve(self, mean=mean, clip_low=max(self.clip_low + delta, min(self.clip_low, mean)), clip_high=self.clip_high + delta)


@overload
def _as_paramdist(val: None) -> None: ...
@overload
def _as_paramdist(val: "ParamDist | float | Mapping[str, float]") -> "ParamDist": ...
@overload
def _as_paramdist(val: object) -> "ParamDist | None": ...
def _as_paramdist(val: object) -> "ParamDist | None":
    if val is None:
        return None
    if isinstance(val, ParamDist):
        return val
    if isinstance(val, (int, float)):
        v = float(val)
        return ParamDist(mean=v, std=0.0, clip_low=v, clip_high=v)
    if isinstance(val, dict):
        return ParamDist(**val)
    return val  # type: ignore[return-value]


_DIST_FIELDS: tuple[str, ...] = ("mean", "std", "clip_low", "clip_high")


def _as_shift_dist(val: object) -> ParamDist:
    if isinstance(val, ParamDist):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v = float(val)
        return ParamDist(mean=v, std=0.0, clip_low=v, clip_high=v)
    if isinstance(val, Mapping) and "mean" in val:
        unknown = sorted(str(key) for key in val if key not in _DIST_FIELDS)
        if unknown:
            raise ValueError(f"unknown robot awareness distribution field {unknown[0]!r}; valid fields: {', '.join(_DIST_FIELDS)}")
        try:
            d = {str(key): float(value) for key, value in val.items()}
        except (TypeError, ValueError):
            raise ValueError(f"robot awareness distribution values must be numbers, got {dict(val)!r}") from None
        if d.get("std", 0.0) < 0.0:
            raise ValueError(f"robot awareness distribution std must be non-negative, got {d['std']!r}")
        d.setdefault("clip_low", -math.inf)
        d.setdefault("clip_high", math.inf)
        return ParamDist(**d)
    raise ValueError(f"robot awareness shift must be a number or a mapping with a mean, got {val!r}")


def _as_blind(val: object) -> bool:
    if not isinstance(val, bool):
        raise ValueError(f"robot awareness blind must be true or false, got {val!r}")
    return val


@attrs.frozen
class AwarenessShift:
    detection: ParamDist = attrs.field(default=ParamDist(0.0, clip_low=0.0, clip_high=0.0), converter=_as_shift_dist)
    fov: ParamDist = attrs.field(default=ParamDist(0.0, clip_low=0.0, clip_high=0.0), converter=_as_shift_dist)
    repulsion: ParamDist = attrs.field(default=ParamDist(0.0, clip_low=0.0, clip_high=0.0), converter=_as_shift_dist)
    reaction: ParamDist = attrs.field(default=ParamDist(0.0, clip_low=0.0, clip_high=0.0), converter=_as_shift_dist)
    blind: bool = attrs.field(default=False, converter=_as_blind)


@attrs.frozen
class AwarenessMixture:
    components: tuple[tuple[float, AwarenessShift], ...]


AwarenessSpec = AwarenessShift | AwarenessMixture

_SHIFT_FIELDS: tuple[str, ...] = ("detection", "fov", "repulsion", "reaction", "blind")

AWARENESS_PRESETS: dict[str, AwarenessShift] = {
    "blind": AwarenessShift(blind=True),
    "distracted": AwarenessShift(detection=-5.0, fov=-8.0, repulsion=-3.0, reaction=-3.0),
    "normal": AwarenessShift(),
    "interested": AwarenessShift(repulsion=-3.0, reaction=-1.5),
    "aggressive": AwarenessShift(fov=-1.0, repulsion=-4.0, reaction=-2.0),
    "scared": AwarenessShift(fov=2.0, repulsion=6.0, reaction=3.0),
}


def _as_mixture(items: Sequence[object]) -> AwarenessMixture:
    components: list[tuple[float, AwarenessShift]] = []
    for item in items:
        if not isinstance(item, Mapping) or "awareness" not in item:
            raise ValueError(f"robot awareness mixture entries need an awareness key, got {item!r}")
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"mixture weights must be numbers, got {item.get('weight')!r}") from None
        if weight < 0.0:
            raise ValueError(f"mixture weights must be non-negative and sum above zero, got {weight!r}")
        shift = as_awareness_spec(item["awareness"])
        if not isinstance(shift, AwarenessShift):
            raise ValueError("robot awareness mixtures cannot nest mixtures")
        components.append((weight, shift))
    if not components or sum(weight for weight, _ in components) <= 0.0:
        raise ValueError(f"mixture weights must be non-negative and sum above zero, got {[weight for weight, _ in components]}")
    return AwarenessMixture(components=tuple(components))


def as_awareness_spec(val: object) -> AwarenessSpec:
    if val is None:
        return AWARENESS_PRESETS["normal"]
    if isinstance(val, (AwarenessShift, AwarenessMixture)):
        return val
    if isinstance(val, bool):
        return AWARENESS_PRESETS["normal" if val else "blind"]
    if isinstance(val, str):
        key = val.strip().lower()
        if key not in AWARENESS_PRESETS:
            raise ValueError(f"unknown robot awareness preset {val!r}; valid presets: {', '.join(sorted(AWARENESS_PRESETS))}")
        return AWARENESS_PRESETS[key]
    if isinstance(val, Mapping):
        unknown = sorted(str(key) for key in val if key not in _SHIFT_FIELDS)
        if unknown:
            raise ValueError(f"unknown robot awareness field {unknown[0]!r}; valid fields: {', '.join(_SHIFT_FIELDS)}")
        return AwarenessShift(**dict(val))
    if isinstance(val, (list, tuple)):
        return _as_mixture(val)
    raise ValueError(f"robot awareness must be a preset name, a field mapping, a mixture list or a bool, got {val!r}")


def awareness_from_text(text: str) -> AwarenessSpec | None:
    """Text form of a robot awareness spec: empty means no override, JSON a shift mapping or a weighted list, anything else a preset name."""
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] in "{[":
        return as_awareness_spec(json.loads(stripped))
    return as_awareness_spec(stripped)


@attrs.frozen
class AwarenessDraw:
    detection: float
    fov: float
    repulsion: float
    reaction: float
    blind: bool


def sample_awareness(spec: AwarenessSpec, rng: np.random.Generator) -> AwarenessDraw:
    if isinstance(spec, AwarenessMixture):
        weights = np.array([weight for weight, _ in spec.components], dtype=np.float64)
        shift = spec.components[int(rng.choice(len(spec.components), p=weights / weights.sum()))][1]
    else:
        shift = spec
    return AwarenessDraw(
        detection=_sample_dist(shift.detection, rng),
        fov=_sample_dist(shift.fov, rng),
        repulsion=_sample_dist(shift.repulsion, rng),
        reaction=_sample_dist(shift.reaction, rng),
        blind=shift.blind,
    )


def _relative_spread(dist: ParamDist) -> float:
    """Std over mean: one sigma moves a parameter by this fraction of its value."""
    return dist.std / dist.mean if dist.mean > 0.0 else 0.0


def with_robot_awareness(perception: "SampledPerception", draw: AwarenessDraw) -> "SampledPerception":
    return attrs.evolve(
        perception,
        robot_range_offset=draw.detection * perception.vision_range_sd,
        robot_fov_offset=draw.fov * perception.vision_fov_sd,
        robot_repulsion_gain=max(0.0, 1.0 + draw.repulsion * perception.repulsion_rel_sd),
        robot_reaction_gain=max(0.05, 1.0 + draw.reaction * perception.reaction_rel_sd),
        robot_blind=draw.blind,
    )


def _as_effort_dist(val: "ParamDist | float | Mapping[str, float]") -> "ParamDist":
    if isinstance(val, dict):
        return ParamDist(**{"clip_low": 1.0, "clip_high": 3.0, **val})
    return _as_paramdist(val)


@attrs.frozen
class NeedDist:
    initial: ParamDist = attrs.field(default=ParamDist(100.0), converter=_as_paramdist)
    decay_rate: ParamDist = attrs.field(default=ParamDist(0.5, 0.1), converter=_as_paramdist)


@attrs.frozen
class NeedCondition:
    below: float | None = None
    above: float | None = None


@attrs.frozen
class VarDef:
    type: str  # "int", "float", "bool", "str"
    default: int | float | bool | str
    min: float | None = None
    max: float | None = None
    description: str = ""


@attrs.frozen
class ActionDef:
    when: dict[str, NeedCondition] = attrs.Factory(dict)
    interaction: str | None = None
    target: str | None = None
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "skip"


@attrs.frozen
class TransitionDef:
    when: dict[str, NeedCondition]
    goto: str


@attrs.frozen
class Pose3:
    x: float
    y: float
    z: float


@attrs.frozen
class RobotRef:
    name: str


@attrs.frozen
class RelativeRef:
    azimuth: float
    elevation: float
    distance: float = 3.0


AttentionRef = str | int | Pose3 | RobotRef | RelativeRef


def _as_refs(val: AttentionRef | tuple[AttentionRef, ...] | list[AttentionRef]) -> tuple[AttentionRef, ...]:
    return tuple(val) if isinstance(val, (tuple, list)) else (val,)


@attrs.frozen
class ChannelDef:
    at: tuple[AttentionRef, ...] = attrs.field(converter=_as_refs)
    dwell: float = 1.0
    advance: str = "dwell"
    hold: str = "release"
    at_z: float | None = None


@attrs.frozen
class ClipDef:
    name: str
    when: str = "always"  # always | bound (only while the agent is a participant of an active interaction)
    by_role: bool = False  # the bound interaction's clip set picks this member's clip, ``name`` is the fallback


CHANNEL_SLOTS = {"gaze": "head", "point": "arm", "point_l": "arm_l", "point_r": "arm_r", "halt": "halt", "halt_l": "halt_l", "halt_r": "halt_r"}
CLIP_SLOT = "body"
ARM_CHANNELS = ("point", "point_r", "point_l", "halt", "halt_r", "halt_l")
ATTENTION_KEYWORDS = ("partner", "partners", "target", "goal")


@attrs.frozen
class AttentionDef:
    gaze: ChannelDef | None = None
    point: ChannelDef | None = None
    point_l: ChannelDef | None = None
    point_r: ChannelDef | None = None
    halt: ChannelDef | None = None
    halt_l: ChannelDef | None = None
    halt_r: ChannelDef | None = None
    clip: ClipDef | None = None
    posture: str = ""  # standing | seated | prone held for the step, empty = leave it to the interaction
    face: bool | AttentionRef | None = None  # None = auto
    required: bool = False

    def channels(self) -> dict[str, ChannelDef]:
        pairs = (
            ("gaze", self.gaze),
            ("point", self.point),
            ("point_l", self.point_l),
            ("point_r", self.point_r),
            ("halt", self.halt),
            ("halt_l", self.halt_l),
            ("halt_r", self.halt_r),
        )
        return {name: ch for name, ch in pairs if ch is not None}

    def face_channel(self) -> str | None:
        """Winning channel for face auto/true: point > point_r > point_l > gaze."""
        present = self.channels()
        return next((name for name in (*ARM_CHANNELS, "gaze") if name in present), None)


Poly = tuple[float, ...]

TUNABLE = frozenset({"desired_velocity", "max_velocity", "max_acceleration", "max_deceleration", "min_turning_radius", "pivot_angular_velocity", "reaction_time", "personal_space_min"})
TUNABLE_PERCEPTION = frozenset({"vision_range", "vision_fov", "proximity_sense"})


@attrs.frozen
class TuneDef:
    """Polynomials in the spawn value per field, keyed `max_velocity`, `perception.vision_range` or `local_planner_params.<name>`."""

    fields: dict[str, Poly] = attrs.Factory(dict)


@attrs.frozen
class ConditionalTune:
    when: dict[str, NeedCondition]
    tune: TuneDef


@attrs.frozen
class StepDef:
    target: str | None = None
    interaction: str | None = None
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"

    autonomous: bool = False
    until: str | None = None
    until_need: dict[str, NeedCondition] | None = None
    allowed_actions: tuple[str, ...] | None = None
    blocked_actions: tuple[str, ...] | None = None

    interruptible: bool | None = None

    interaction_radius: float | None = None

    offer: bool = False
    cancel: bool = False
    queueable: bool | None = None
    min_participants: int | None = None
    max_participants: int | None = None
    formation_spec: FormationSpec | None = None
    wait_for_outcome: bool = False

    attention: AttentionDef | None = None
    tune: TuneDef | None = None


@attrs.frozen
class AttentionStepDef:
    attention: AttentionDef
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"
    interruptible: bool | None = None
    tune: TuneDef | None = None


@attrs.frozen
class GoToStepDef:
    target_pose: Pose2D | None = None
    target: str | None = None
    attention: AttentionDef | None = None
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"
    interruptible: bool | None = None
    tune: TuneDef | None = None


@attrs.frozen
class RobotTrigger:
    """Robot-relative gate where every set field must hold."""

    ahead: float | None = None
    behind: float | None = None
    within: float | None = None
    lateral: float = 5.0


ROBOT_ACTS = ("wait", "stand", "approach", "cross", "traverse", "pass")
ROBOT_GESTURES = ("", "halt", "halt_l", "halt_r")


@attrs.frozen
class RobotStepDef:
    when: RobotTrigger | None = None
    act: str = "wait"
    face: str | float | None = None
    gesture: str = ""
    lead: float = 3.0
    clearance: float = 1.0
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"
    interruptible: bool | None = None
    tune: TuneDef | None = None
    attention: None = attrs.field(default=None, init=False)


@attrs.frozen
class SequenceDef:
    steps: dict[str, StepDef | GoToStepDef | AttentionStepDef | RobotStepDef]
    then: str | None = None
    on_failure: str | None = None
    interruptible: bool = True
    transitions: tuple[TransitionDef, ...] = ()
    attention: AttentionDef | None = None
    tune: TuneDef | None = None


ROBOT_RESPONSES = ("avoid", "halt")  # a perceived robot: steer around it, or brake (ahead) / slow (beside, behind) without steering


@attrs.frozen
class PerceptionDist:
    vision_range: ParamDist = attrs.field(default=ParamDist(5.0, 0.5), converter=_as_paramdist)
    vision_fov: ParamDist = attrs.field(default=ParamDist(180.0, 10.0), converter=_as_paramdist)
    proximity_sense: ParamDist = attrs.field(default=ParamDist(1.0, 0.2, clip_low=0.5, clip_high=2.0), converter=_as_paramdist)
    vision_occlusion: bool = True
    robot_awareness: AwarenessSpec = attrs.field(default=AWARENESS_PRESETS["normal"], converter=as_awareness_spec)
    robot_response: str = attrs.field(default="avoid", validator=attrs.validators.in_(ROBOT_RESPONSES))


KINEMATICS = ("holonomic", "along_heading")
STAIRS_KINDS = ("stairs", "escalator")
HEADING_SOURCES = {"attraction": 0.0, "total": 1.0}
MAX_HARMONICS = 3

Harmonics = tuple[tuple[float, float], ...]


def _as_harmonics(val: Iterable[Iterable[float]] | None) -> Harmonics:
    if val is None:
        return ()
    out = tuple((float(amp), float(phase)) for amp, phase in val)
    if len(out) > MAX_HARMONICS:
        raise ValueError(f"at most {MAX_HARMONICS} harmonics, got {len(out)}")
    return out


def _zero_dist() -> ParamDist:
    return ParamDist(0.0, clip_low=0.0)


@attrs.frozen
class CadenceDist:
    base: ParamDist = attrs.field(default=ParamDist(0.4), converter=_as_paramdist)
    per_speed: ParamDist = attrs.field(default=ParamDist(0.55), converter=_as_paramdist)
    min: ParamDist = attrs.field(default=ParamDist(0.4), converter=_as_paramdist)
    max: ParamDist = attrs.field(default=ParamDist(2.2), converter=_as_paramdist)


@attrs.frozen
class LocomotionDist:
    kinematics: str = attrs.field(default="holonomic", validator=attrs.validators.in_(KINEMATICS))
    cadence: CadenceDist = attrs.Factory(CadenceDist)
    phase_warp_split: ParamDist = attrs.field(default=ParamDist(0.5), converter=_as_paramdist)
    speed_profile: Harmonics = attrs.field(default=(), converter=_as_harmonics)
    speed_amplitude_scale: ParamDist = attrs.field(default=ParamDist(1.0), converter=_as_paramdist)
    lateral_profile: Harmonics = attrs.field(default=(), converter=_as_harmonics)
    footprint_length: ParamDist = attrs.field(factory=_zero_dist, converter=_as_paramdist)
    recovery_stall_after_s: ParamDist = attrs.field(factory=_zero_dist, converter=_as_paramdist)
    recovery_reverse_m: ParamDist = attrs.field(factory=_zero_dist, converter=_as_paramdist)


@attrs.frozen
class AgentType:
    name: str
    mode: str = "simple"

    desired_velocity: ParamDist = attrs.field(default=ParamDist(1.1, 0.12), converter=_as_paramdist)
    agent_radius: ParamDist = attrs.field(default=ParamDist(0.35, 0.02), converter=_as_paramdist)
    max_velocity: ParamDist = attrs.field(default=ParamDist(1.5, 0.1, clip_low=0.5), converter=_as_paramdist)
    max_acceleration: ParamDist = attrs.field(default=ParamDist(1.5, 0.1, clip_low=0.3), converter=_as_paramdist)
    max_deceleration: ParamDist = attrs.field(default=ParamDist(2.5, 0.2, clip_low=0.5), converter=_as_paramdist)
    min_turning_radius: ParamDist = attrs.field(default=ParamDist(0.3, 0.03, clip_low=0.1), converter=_as_paramdist)
    pivot_angular_velocity: ParamDist = attrs.field(default=ParamDist(2.0, 0.2, clip_low=1.0), converter=_as_paramdist)

    # LogNormal: mean is interpreted as the desired median in seconds; std is the shape parameter.
    reaction_time: ParamDist = attrs.field(default=ParamDist(0.4, 0.3, clip_low=0.05, clip_high=1.5), converter=_as_paramdist)
    personal_space_min: ParamDist = attrs.field(default=ParamDist(0.6, 0.15, clip_low=0.2, clip_high=2.0), converter=_as_paramdist)

    idle_gaze_rate: ParamDist = attrs.field(default=ParamDist(0.0, 0.0, clip_low=0.0, clip_high=0.0), converter=_as_paramdist)
    handedness: dict[str, float] = attrs.Factory(lambda: {"right": 0.9, "left": 0.1})

    perception: PerceptionDist = attrs.Factory(PerceptionDist)
    local_planner_params: dict[str, ParamDist] = attrs.Factory(dict)
    locomotion: LocomotionDist = attrs.Factory(LocomotionDist)
    locomotion_active: bool = False
    pose: dict = attrs.Factory(dict)
    interaction_class: str = ""
    assets: tuple[str, ...] = ()
    climbs: tuple[str, ...] = attrs.field(default=STAIRS_KINDS, converter=tuple, validator=attrs.validators.deep_iterable(attrs.validators.in_(STAIRS_KINDS)))
    stair_effort: ParamDist = attrs.field(default=ParamDist(1.6, 0.5, clip_low=1.0, clip_high=3.0), converter=_as_effort_dist)

    perception_stack: tuple[str, ...] = ("default",)
    local_planner: str | None = None
    global_planner: str | None = None
    animation: str | None = None

    needs: dict[str, NeedDist] = attrs.Factory(dict)
    utility_weights: dict[str, float] = attrs.Factory(dict)
    actions: dict[str, ActionDef] = attrs.Factory(dict)
    sequences: dict[str, SequenceDef] = attrs.Factory(dict)
    initial_sequence: str = "default"
    tunes: tuple[ConditionalTune, ...] = ()
    vars: dict[str, VarDef] = attrs.Factory(dict)
    extends: str | None = None
    device: dict = attrs.Factory(dict)

    source_path: Path | None = attrs.field(default=None, eq=False, hash=False, repr=False)


@attrs.frozen
class SampledNeed:
    initial: float
    decay_rate: float


@attrs.frozen
class SampledPerception:
    vision_range: float = 5.0
    vision_fov: float = 180.0
    proximity_sense: float = 1.0
    vision_occlusion: bool = True
    vision_range_sd: float = 0.0
    vision_fov_sd: float = 0.0
    repulsion_rel_sd: float = 0.0
    reaction_rel_sd: float = 0.0
    robot_range_offset: float = 0.0
    robot_fov_offset: float = 0.0
    robot_repulsion_gain: float = 1.0
    robot_reaction_gain: float = 1.0
    robot_blind: bool = False
    robot_halt: bool = False


@attrs.frozen
class SampledLocomotion:
    active: bool = False
    kinematics: int = 0
    cadence_base: float = 0.4
    cadence_per_speed: float = 0.55
    cadence_min: float = 0.4
    cadence_max: float = 2.2
    phase_warp_split: float = 0.5
    speed_profile: Harmonics = attrs.field(default=(), converter=_as_harmonics)
    speed_amplitude_scale: float = 1.0
    lateral_profile: Harmonics = attrs.field(default=(), converter=_as_harmonics)
    footprint_length: float = 0.0
    recovery_stall_after_s: float = 0.0
    recovery_reverse_m: float = 0.0


@attrs.frozen
class SampledParams:
    name: str

    desired_velocity: float
    agent_radius: float
    max_velocity: float
    max_acceleration: float
    max_deceleration: float
    min_turning_radius: float
    pivot_angular_velocity: float

    reaction_time: float
    personal_space_min: float

    perception: SampledPerception = attrs.Factory(SampledPerception)
    local_planner_params: dict[str, float] = attrs.Factory(dict)
    locomotion: SampledLocomotion = attrs.Factory(SampledLocomotion)
    interaction_class: str = ""
    climbs: tuple[str, ...] = STAIRS_KINDS
    stair_effort: float = 1.0

    perception_stack: tuple[str, ...] = ("default",)
    local_planner: str | None = None
    global_planner: str | None = None
    animation: str | None = None

    needs: dict[str, SampledNeed] = attrs.Factory(dict)
    utility_weights: dict[str, float] = attrs.Factory(dict)

    idle_gaze_rate_hz: float = 0.0
    handedness: str = "r"


_HANDS = {"right": "r", "r": "r", "left": "l", "l": "l"}


def _sample_handedness(weights: dict[str, float], rng: np.random.Generator) -> str:
    hands = [_HANDS[k] for k in weights]
    w = np.asarray([max(float(v), 0.0) for v in weights.values()])
    if not hands or w.sum() <= 0.0:
        return "r"
    return hands[int(rng.choice(len(hands), p=w / w.sum()))]


def _sample_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    value = rng.normal(dist.mean, dist.std) if dist.std > 0 else dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def _sample_lognormal_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    # dist.mean is interpreted as the desired median in linear space; dist.std is sigma of the underlying normal.
    if dist.std > 0:
        value = rng.lognormal(mean=math.log(max(dist.mean, 1e-9)), sigma=dist.std)
    else:
        value = dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def _sample_locomotion(agent_type: AgentType, rng: np.random.Generator) -> SampledLocomotion:
    loc = agent_type.locomotion
    return SampledLocomotion(
        active=agent_type.locomotion_active,
        kinematics=KINEMATICS.index(loc.kinematics),
        cadence_base=_sample_dist(loc.cadence.base, rng),
        cadence_per_speed=_sample_dist(loc.cadence.per_speed, rng),
        cadence_min=_sample_dist(loc.cadence.min, rng),
        cadence_max=_sample_dist(loc.cadence.max, rng),
        phase_warp_split=_sample_dist(loc.phase_warp_split, rng),
        speed_profile=loc.speed_profile,
        speed_amplitude_scale=_sample_dist(loc.speed_amplitude_scale, rng),
        lateral_profile=loc.lateral_profile,
        footprint_length=_sample_dist(loc.footprint_length, rng),
        recovery_stall_after_s=_sample_dist(loc.recovery_stall_after_s, rng),
        recovery_reverse_m=_sample_dist(loc.recovery_reverse_m, rng),
    )


def sample_agent_type(
    agent_type: AgentType,
    rng: np.random.Generator,
    default_local_planner: str = "sfm",
    local_planner_means: Mapping[str, float] | None = None,
) -> SampledParams:
    sampled_needs: dict[str, SampledNeed] = {}
    for need_name, need_dist in agent_type.needs.items():
        sampled_needs[need_name] = SampledNeed(
            initial=_sample_dist(need_dist.initial, rng),
            decay_rate=_sample_dist(need_dist.decay_rate, rng),
        )

    desired_velocity = _sample_dist(agent_type.desired_velocity, rng)
    agent_radius = _sample_dist(agent_type.agent_radius, rng)
    max_velocity = _sample_dist(agent_type.max_velocity, rng)
    max_acceleration = _sample_dist(agent_type.max_acceleration, rng)
    max_deceleration = _sample_dist(agent_type.max_deceleration, rng)
    min_turning_radius = _sample_dist(agent_type.min_turning_radius, rng)
    pivot_angular_velocity = _sample_dist(agent_type.pivot_angular_velocity, rng)
    reaction_time = _sample_lognormal_dist(agent_type.reaction_time, rng)
    personal_space_min = _sample_dist(agent_type.personal_space_min, rng)
    vision_range = _sample_dist(agent_type.perception.vision_range, rng)
    vision_fov = _sample_dist(agent_type.perception.vision_fov, rng)
    proximity_sense = _sample_dist(agent_type.perception.proximity_sense, rng)

    from arena_humansim.local_planner import LocalPlanner

    planner_name = agent_type.local_planner or default_local_planner
    lp_dists = {**LocalPlanner.get_class(planner_name).PARAM_DEFAULTS, **agent_type.local_planner_params}
    for key, mean in (local_planner_means or {}).items():
        if key in lp_dists:
            lp_dists[key] = lp_dists[key].with_mean(mean)
    sampled_lp = {k: _sample_dist(d, rng) for k, d in lp_dists.items()}
    awareness_dists = {**LocalPlanner.get_class("sfm").PARAM_DEFAULTS, **lp_dists}
    sampled_perception = with_robot_awareness(
        SampledPerception(
            vision_range=vision_range,
            vision_fov=vision_fov,
            proximity_sense=proximity_sense,
            vision_occlusion=agent_type.perception.vision_occlusion,
            vision_range_sd=agent_type.perception.vision_range.std,
            vision_fov_sd=agent_type.perception.vision_fov.std,
            repulsion_rel_sd=_relative_spread(awareness_dists["repulsion_strength"]),
            reaction_rel_sd=_relative_spread(awareness_dists["repulsion_range"]),
            robot_halt=agent_type.perception.robot_response == "halt",
        ),
        sample_awareness(agent_type.perception.robot_awareness, rng),
    )

    idle_gaze_rate_hz = _sample_dist(agent_type.idle_gaze_rate, rng)
    handedness = _sample_handedness(agent_type.handedness, rng)
    locomotion = _sample_locomotion(agent_type, rng)
    stair_effort = _sample_dist(agent_type.stair_effort, rng)

    return SampledParams(
        name=agent_type.name,
        desired_velocity=desired_velocity,
        agent_radius=agent_radius,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_deceleration=max_deceleration,
        min_turning_radius=min_turning_radius,
        pivot_angular_velocity=pivot_angular_velocity,
        reaction_time=reaction_time,
        personal_space_min=personal_space_min,
        perception=sampled_perception,
        local_planner_params=sampled_lp,
        locomotion=locomotion,
        interaction_class=agent_type.interaction_class,
        climbs=agent_type.climbs,
        stair_effort=stair_effort,
        perception_stack=agent_type.perception_stack,
        local_planner=agent_type.local_planner,
        global_planner=agent_type.global_planner,
        animation=agent_type.animation,
        needs=sampled_needs,
        utility_weights=dict(agent_type.utility_weights),
        idle_gaze_rate_hz=idle_gaze_rate_hz,
        handedness=handedness,
    )
