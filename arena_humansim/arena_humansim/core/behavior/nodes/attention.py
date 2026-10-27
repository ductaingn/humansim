from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import py_trees

from arena_humansim.core.agents import BaseAgent, ParamDist
from arena_humansim.core.agents.types import ATTENTION_KEYWORDS, CHANNEL_SLOTS, CLIP_SLOT, AttentionDef, AttentionRef, ChannelDef, Pose3, RelativeRef, RobotRef
from arena_humansim.core.behavior.nodes.helpers import _bt_logger, _nav_command, _sample_param_dist, _set_command
from arena_humansim.core.behavior.reach import MIN_RESIDENCE_S, reachable
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, GestureIntent, InteractionState, Pose2D, WaypointMovement

FACE_ENTER_RAD = 0.25
FACE_KEEP_RAD = 0.6
FACE_TIMEOUT_S = 4.0
RESOLVE_TIMEOUT_S = 4.0
HALT_EPS = 1e-6
GESTURE_Z_OBJECT = 0.8
GESTURE_Z_AGENT = 1.2
GESTURE_Z_HEAD = 1.6

REF_PARTNER, REF_PARTNERS, REF_TARGET, REF_GOAL = ATTENTION_KEYWORDS

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE
INVALID = py_trees.common.Status.INVALID

AgentLookup = Callable[[int], BaseAgent | None]
NameLookup = Callable[[str, int | None], int | None]
Walking = Callable[[], bool]
XYZ = tuple[float, float, float]


@attrs.frozen
class _AgentTarget:
    agent_id: int


@attrs.frozen
class _ObjectTarget:
    object_id: str


_Handle = str | Pose3 | RelativeRef | _AgentTarget | _ObjectTarget


@attrs.frozen
class _Entry:
    xyz: XYZ | None
    relative: bool = False


_UNRESOLVED = _Entry(None)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class _Resolver:
    """Ref -> world points, cached per ref, agents re-resolved when they vanish."""

    def __init__(self, agent: BaseAgent, world: WorldKnowledge, agent_lookup: AgentLookup, name_lookup: NameLookup, ctx: StepContext) -> None:
        self._agent = agent
        self._world = world
        self._agent_lookup = agent_lookup
        self._name_lookup = name_lookup
        self._ctx = ctx
        self._handles: dict[AttentionRef, _Handle] = {}

    def reset(self) -> None:
        self._handles = {}

    def _resolve(self, ref: AttentionRef) -> _Handle | None:
        if isinstance(ref, (Pose3, RelativeRef)):
            return ref
        if isinstance(ref, RobotRef):
            aid = self._name_lookup(ref.name, KIND_ROBOT)
            return _AgentTarget(aid) if aid is not None else None
        if isinstance(ref, int):
            return _AgentTarget(ref) if self._agent_lookup(ref) is not None else None
        if ref in ATTENTION_KEYWORDS:
            return ref
        obj = self._world.find(ref)
        if obj is not None:
            return _ObjectTarget(obj.object_id)
        aid = self._name_lookup(ref, None)
        if aid is not None:
            return _AgentTarget(aid)
        obj = self._world.resolve(ref, self._agent.state.pose, exclude_full=False)
        return _ObjectTarget(obj.object_id) if obj is not None else None

    def _handle(self, ref: AttentionRef) -> _Handle | None:
        handle = self._handles.get(ref)
        if isinstance(handle, _AgentTarget) and self._agent_lookup(handle.agent_id) is None:
            del self._handles[ref]
            handle = None
        if handle is None:
            handle = self._resolve(ref)
            if handle is not None:
                self._handles[ref] = handle
        return handle

    def _other_participants(self) -> list[int]:
        mv = self._agent.movement
        im = self._ctx.im
        if not isinstance(mv, BehaviorTreeMovement) or im is None or mv.interaction_id is None:
            return []
        interaction = im.interactions.get(mv.interaction_id)
        if interaction is None:
            return []
        own = self._agent.state.agent_id
        return [aid for aid in interaction.participants if aid != own]

    def _goal_pose(self) -> Pose2D | None:
        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement):
            cmd = mv.command
            if cmd is None or cmd.type != CommandType.NAVIGATE or cmd.desired_velocity == 0.0:
                return None
            own = self._agent.state.pose
            if math.hypot(cmd.target_pose.x - own.x, cmd.target_pose.y - own.y) <= HALT_EPS:
                return None
            return cmd.target_pose
        if isinstance(mv, WaypointMovement) and mv.waypoints:
            return mv.waypoints[mv.index % len(mv.waypoints)]
        return None

    def _target_pose(self) -> Pose2D | None:
        if self._ctx.target_object_id is not None:
            obj = self._world.get(self._ctx.target_object_id)
            if obj is not None:
                return obj.pose
        return self._ctx.target_pose

    def _relative_xyz(self, ref: RelativeRef, z_agent: float) -> XYZ:
        own = self._agent.state.pose
        az = own.theta + math.radians(ref.azimuth)
        el = math.radians(ref.elevation)
        flat = ref.distance * math.cos(el)
        return (own.x + flat * math.cos(az), own.y + flat * math.sin(az), z_agent + ref.distance * math.sin(el))

    def _agent_xyz(self, agent_id: int, z: float) -> XYZ | None:
        other = self._agent_lookup(agent_id)
        if other is None:
            return None
        return (other.state.pose.x, other.state.pose.y, z)

    def expand(self, ref: AttentionRef, z_agent: float, at_z: float | None) -> list[_Entry]:
        """Every world point the ref stands for right now, one unresolved placeholder if none."""
        za = at_z if at_z is not None else z_agent
        zo = at_z if at_z is not None else GESTURE_Z_OBJECT
        handle = self._handle(ref)
        out: list[_Entry] = []
        if isinstance(handle, Pose3):
            out.append(_Entry((handle.x, handle.y, handle.z)))
        elif isinstance(handle, RelativeRef):
            out.append(_Entry(self._relative_xyz(handle, z_agent), relative=True))
        elif isinstance(handle, _AgentTarget):
            xyz = self._agent_xyz(handle.agent_id, za)
            if xyz is not None:
                out.append(_Entry(xyz))
        elif isinstance(handle, _ObjectTarget):
            obj = self._world.get(handle.object_id)
            if obj is not None:
                out.append(_Entry((obj.pose.x, obj.pose.y, zo)))
        elif handle == REF_TARGET:
            pose = self._target_pose()
            if pose is not None:
                out.append(_Entry((pose.x, pose.y, zo)))
        elif handle == REF_GOAL:
            pose = self._goal_pose()
            if pose is not None:
                out.append(_Entry((pose.x, pose.y, zo)))
        elif handle in (REF_PARTNER, REF_PARTNERS):
            own = self._agent.state.pose
            found = [xyz for aid in self._other_participants() if (xyz := self._agent_xyz(aid, za)) is not None]
            if handle == REF_PARTNER and found:
                found = [min(found, key=lambda p: math.hypot(p[0] - own.x, p[1] - own.y))]
            out.extend(_Entry(xyz) for xyz in found)
        return out or [_UNRESOLVED]


class _Channel:
    """Per-channel list cursor, timers, reach state and the intent last published."""

    def __init__(self, name: str, cdef: ChannelDef) -> None:
        self.name = name
        self.slot = CHANNEL_SLOTS[name]
        self.cdef = cdef
        self.z_agent = GESTURE_Z_HEAD if self.slot == "head" else GESTURE_Z_AGENT
        self.entries: list[_Entry] = []
        self.idx = 0
        self.dwell_elapsed = 0.0
        self.residence = 0.0
        self.shown = False
        self.done = False
        self.stuck_elapsed = 0.0
        self.warned = False
        self.published: GestureIntent | None = None

    def reset(self) -> None:
        self.entries = []
        self.idx = 0
        self.dwell_elapsed = 0.0
        self.residence = 0.0
        self.shown = False
        self.done = False
        self.stuck_elapsed = 0.0
        self.warned = False
        self.published = None

    def current(self) -> _Entry:
        return self.entries[self.idx]

    def lower(self) -> None:
        self.shown = False
        self.published = None

    def raise_at(self, entry: _Entry, handedness: str) -> None:
        assert entry.xyz is not None
        self.shown = True
        self.published = GestureIntent(self.slot, entry.xyz[0], entry.xyz[1], entry.xyz[2], hand=handedness if self.name in ("point", "halt") else "")


class AttentionNode(py_trees.behaviour.Behaviour):
    """Drive every channel of one attention block: resolve refs, walk lists, gate on reach, face when allowed, publish mv.gestures (plus the body clip, if any). Bare mode halts and owns the step, rider mode never finishes."""

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        attention: AttentionDef,
        world: WorldKnowledge,
        agent_lookup: AgentLookup,
        name_lookup: NameLookup,
        rng: np.random.Generator,
        dt: float,
        ctx: StepContext,
        bare: bool = False,
        duration: ParamDist | None = None,
        walking: Walking | None = None,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._att = attention
        self._rng = rng
        self._dt = dt
        self._ctx = ctx
        self._bare = bare
        self._required = bare or attention.required
        self._duration_source = duration
        self._walking = walking
        self._resolver = _Resolver(agent, world, agent_lookup, name_lookup, ctx)
        self._channels = [_Channel(cname, cdef) for cname, cdef in attention.channels().items()]
        self._clip = attention.clip
        self._clip_published: GestureIntent | None = None
        self._slots = {ch.slot for ch in self._channels} | ({CLIP_SLOT} if self._clip is not None else set())
        self._face_name = attention.face_channel()
        self._duration: float | None = None
        self._elapsed = 0.0
        self._face_elapsed = 0.0
        self._faced = False
        self._refacing = False
        self._face_gave_up = False

    def initialise(self) -> None:
        self._resolver.reset()
        for ch in self._channels:
            ch.reset()
        self._clip_published = None
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._elapsed = 0.0
        self._face_elapsed = 0.0
        self._faced = False
        self._refacing = False
        self._face_gave_up = False

    def _bt_mv(self) -> BehaviorTreeMovement | None:
        mv = self._agent.movement
        return mv if isinstance(mv, BehaviorTreeMovement) else None

    def _bound(self) -> bool:
        lookup = self._ctx.is_bound_lookup
        return bool(lookup(self._agent.state.agent_id)) if lookup is not None else False

    def _halt(self) -> None:
        if self._bare and not self._bound():
            cmd = _nav_command(self._agent, self._agent.state.pose)
            cmd.desired_velocity = 0.0
            _set_command(self._agent, cmd)

    def _expand_all(self) -> None:
        for ch in self._channels:
            entries: list[_Entry] = []
            for ref in ch.cdef.at:
                entries.extend(self._resolver.expand(ref, ch.z_agent, ch.cdef.at_z))
            ch.entries = entries
            if ch.idx >= len(entries):
                ch.idx = len(entries) - 1

    def _stuck(self, ch: _Channel) -> bool:
        if ch.cdef.advance == "dwell":
            return ch.current().xyz is None
        return all(e.xyz is None for e in ch.entries)

    def _check_resolution(self) -> py_trees.common.Status | None:
        for ch in self._channels:
            if not self._stuck(ch):
                ch.stuck_elapsed = 0.0
                continue
            if not ch.warned:
                _bt_logger.warning(f"Agent {self._agent.state.agent_id}: {self.name} {ch.name} waiting, could not resolve at={ch.cdef.at!r}")
                ch.warned = True
            ch.stuck_elapsed += self._dt
            if ch.stuck_elapsed > RESOLVE_TIMEOUT_S and self._required:
                return FAILURE
        return None

    def _face_target(self) -> XYZ | None:
        face = self._att.face
        if face is False:
            return None
        if face is None or face is True:
            if self._face_name is None:
                return None
            ch = next(c for c in self._channels if c.name == self._face_name)
            cur = ch.current()
            return None if cur.relative else cur.xyz
        entry = self._resolver.expand(face, GESTURE_Z_HEAD, None)[0]
        return None if entry.relative else entry.xyz

    def _face(self, mv: BehaviorTreeMovement | None) -> tuple[bool, py_trees.common.Status | None]:
        """Command the turn when allowed. Returns (turn in flight, early status)."""
        heading: float | None = None
        in_flight = False
        target = self._face_target()
        allowed = target is not None and not self._face_gave_up and not self._bound() and not (self._walking is not None and self._walking())
        if allowed:
            assert target is not None
            own = self._agent.state.pose
            bearing = math.atan2(target[1] - own.y, target[0] - own.x)
            err = abs(_wrap(bearing - own.theta))
            if not self._faced:
                if err <= FACE_ENTER_RAD:
                    self._faced = True
                else:
                    self._face_elapsed += self._dt
                    if self._face_elapsed <= FACE_TIMEOUT_S:
                        heading = bearing
                        in_flight = True
                    elif self._bare:
                        return True, FAILURE
                    else:
                        self._face_gave_up = True
            else:
                if err > FACE_KEEP_RAD:
                    self._refacing = True
                elif err <= FACE_ENTER_RAD:
                    self._refacing = False
                if self._refacing:
                    heading = bearing
                    in_flight = True
        if mv is not None:
            mv.heading_goal = heading
        return in_flight, None

    def _azimuth(self, xyz: XYZ, heading: float) -> float:
        own = self._agent.state.pose
        return _wrap(math.atan2(xyz[1] - own.y, xyz[0] - own.x) - heading)

    def _step_dwell(self, ch: _Channel, heading: float) -> None:
        cur = ch.current()
        if cur.xyz is not None and reachable(ch.slot, self._azimuth(cur.xyz, heading), ch.shown):
            ch.raise_at(cur, self._agent.params.handedness)
        else:
            ch.lower()
        if ch.done or cur.xyz is None:
            return
        ch.dwell_elapsed += self._dt
        if ch.dwell_elapsed >= ch.cdef.dwell:
            ch.dwell_elapsed = 0.0
            ch.residence = 0.0
            if ch.idx + 1 < len(ch.entries):
                ch.idx += 1
            else:
                ch.done = True

    def _step_unreachable(self, ch: _Channel, heading: float, in_flight: bool) -> None:
        cur = ch.current()
        if cur.xyz is not None and reachable(ch.slot, self._azimuth(cur.xyz, heading), ch.shown):
            ch.raise_at(cur, self._agent.params.handedness)
            return
        ch.lower()
        if in_flight or ch.residence < MIN_RESIDENCE_S:
            return
        n = len(ch.entries)
        for k in range(1, n):
            j = (ch.idx + k) % n
            entry = ch.entries[j]
            if entry.xyz is not None and reachable(ch.slot, self._azimuth(entry.xyz, heading), False):
                ch.idx = j
                ch.residence = 0.0
                ch.raise_at(entry, self._agent.params.handedness)
                return

    def _hold_posture(self, mv: BehaviorTreeMovement | None, posture: str) -> None:
        if mv is not None and self._att.posture:
            mv.posture = posture

    def _bound_interaction(self) -> InteractionState | None:
        mv = self._bt_mv()
        im = self._ctx.im
        if mv is None or mv.interaction_id is None or im is None:
            return None
        return im.interactions.get(mv.interaction_id)

    def _clip_render_target(self) -> tuple[float, float] | None:
        """Formation slot xy while the bound contact interaction is shown touching."""
        interaction = self._bound_interaction()
        im = self._ctx.im
        if interaction is None or im is None or not im.shows_contact(interaction):
            return None
        target = im.formation_target(self._agent.state.agent_id)
        return None if target is None else (target.x, target.y)

    def _step_clip(self) -> None:
        clip = self._clip
        if clip is None:
            return
        if clip.when == "bound" and not self._bound():
            self._clip_published = None
            return
        interaction = self._bound_interaction()
        im = self._ctx.im
        if interaction is not None and im is not None and im.is_contact(interaction) and not im.shows_contact(interaction):
            self._clip_published = None
            return
        xy = self._clip_render_target()
        x, y = xy if xy is not None else (0.0, 0.0)
        override = xy is not None
        name = clip.name
        if clip.by_role and interaction is not None and im is not None:
            name = im.role_clip(interaction, self._agent.state.agent_id) or name
        if self._clip_published is None or (self._clip_published.x, self._clip_published.y, self._clip_published.render_pose_override, self._clip_published.clip) != (x, y, override, name):
            self._clip_published = GestureIntent(CLIP_SLOT, x, y, clip=name, render_pose_override=override)

    def _published(self) -> list[GestureIntent]:
        out = [ch.published for ch in self._channels if ch.published is not None]
        if self._clip_published is not None:
            out.append(self._clip_published)
        return out

    def _publish(self, mv: BehaviorTreeMovement | None) -> None:
        if mv is None:
            return
        others = tuple(g for g in mv.gestures if g.slot not in self._slots)
        mv.gestures = others + tuple(self._published())

    def _finish(self) -> py_trees.common.Status:
        if not self._bare:
            return RUNNING
        if self._duration is None:
            return SUCCESS if all(ch.done for ch in self._channels) else RUNNING
        if self._elapsed >= self._duration:
            return SUCCESS
        self._elapsed += self._dt
        return RUNNING

    def update(self) -> py_trees.common.Status:
        self._halt()
        mv = self._bt_mv()
        self._expand_all()
        failed = self._check_resolution()
        if failed is not None:
            return failed
        in_flight, early = self._face(mv)
        if early is not None:
            return early
        heading = mv.heading_goal if mv is not None and mv.heading_goal is not None else self._agent.state.pose.theta
        for ch in self._channels:
            ch.residence += self._dt
            if ch.cdef.advance == "dwell":
                self._step_dwell(ch, heading)
            else:
                self._step_unreachable(ch, heading, in_flight)
        self._step_clip()
        self._publish(mv)
        self._hold_posture(mv, self._att.posture)
        return self._finish()

    def suspend(self) -> None:
        """Lower every channel and drop the heading without losing list position; the next tick resumes."""
        mv = self._bt_mv()
        if mv is not None:
            mv.heading_goal = None
            mine = self._published()
            mv.gestures = tuple(g for g in mv.gestures if not any(g is m for m in mine))
        for ch in self._channels:
            ch.lower()
        self._clip_published = None
        self._hold_posture(mv, "")

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        mv = self._bt_mv()
        if mv is None:
            return
        mv.heading_goal = None
        released = [ch.published for ch in self._channels if ch.published is not None and ch.cdef.hold == "release"]
        if self._clip_published is not None:
            released.append(self._clip_published)
        mv.gestures = tuple(g for g in mv.gestures if not any(g is r for r in released))
        self._hold_posture(mv, "")


@attrs.frozen
class RiderStep:
    own_attention: bool
    autonomous: bool
    walking: Walking


class SequenceRiderNode(py_trees.behaviour.Behaviour):
    """Tick a sequence's steps and its attention rider side by side, suspending the rider while the current step carries its own block or runs autonomously."""

    def __init__(self, name: str, steps: py_trees.behaviour.Behaviour, step_infos: list[RiderStep], make_rider: Callable[[Walking], AttentionNode]) -> None:
        super().__init__(name)
        self._steps = steps
        self._infos = step_infos
        self._rider = make_rider(self.walking)

    @property
    def steps(self) -> py_trees.behaviour.Behaviour:
        return self._steps

    @property
    def rider(self) -> AttentionNode:
        return self._rider

    def _current_index(self) -> int:
        steps = self._steps
        if isinstance(steps, py_trees.composites.Sequence) and steps.current_child is not None:
            return steps.children.index(steps.current_child)
        return 0

    def walking(self) -> bool:
        return self._infos[self._current_index()].walking()

    def update(self) -> py_trees.common.Status:
        self._steps.tick_once()
        status = self._steps.status
        info = self._infos[self._current_index()]
        if status == RUNNING and not info.own_attention and not info.autonomous:
            self._rider.tick_once()
            if self._rider.status == FAILURE:
                self._steps.stop(INVALID)
                return FAILURE
        elif self._rider.status == RUNNING:
            self._rider.suspend()
        return status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        if self._steps.status != INVALID:
            self._steps.stop(INVALID)
        if self._rider.status != INVALID:
            self._rider.stop(INVALID)
