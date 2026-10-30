from __future__ import annotations

import math
from typing import TYPE_CHECKING

import py_trees

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.behavior.nodes.helpers import (
    _at_target,
    _bt_logger,
    _nav_command,
    _resolve_interaction_radius,
    _set_command,
)
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D, pose_distance

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool


def _approach_pose_for(obj: WorldObject, world: WorldKnowledge) -> Pose2D:
    spec = obj.formation
    if spec is None or spec.type != "line":
        return obj.pose
    params = spec.params or {}
    front_offset = float(params.get("front_offset", 0.0))
    base_step = float(params.get("base_step", 1.0))
    slot_index = world.participants_count_for_object(obj.object_id) + world.queue_length_for_object(obj.object_id)
    offset = front_offset + slot_index * base_step
    back = obj.pose.theta + math.pi
    return Pose2D(
        x=obj.pose.x + offset * math.cos(back),
        y=obj.pose.y + offset * math.sin(back),
        theta=obj.pose.theta,
    )


def _seat_aim(obj: WorldObject, ctx: StepContext, from_pose: Pose2D) -> Pose2D | None:
    """Standing point before the free seat nearest from_pose, None when the step does not aim at seats or none is free.

    Aiming reserves nothing: the seat is taken by whoever seeks first (ClusterFormation picks the same nearest free
    seat then), and a walker whose seat is taken meanwhile aims at the next free one on its next tick.
    """
    if not ctx.aim_seats or not obj.seats or ctx.im is None:
        return None
    free = [s for s in ctx.im.free_seats(obj.object_id) if s < len(obj.seats)]
    if not free:
        return None
    best = min(free, key=lambda s: pose_distance(obj.seats[s], from_pose))
    seat = obj.seats[best]
    approach = obj.seat_approach[best] if best < len(obj.seat_approach) else 0.0
    return Pose2D(x=seat.x + approach * math.cos(seat.theta), y=seat.y + approach * math.sin(seat.theta), theta=seat.theta)


def _walk_target(obj: WorldObject, world: WorldKnowledge, ctx: StepContext, from_pose: Pose2D) -> Pose2D:
    """Where an object-bound step walks: the nearest free seat's standing point for a sitter, else the approach pose."""
    aim = _seat_aim(obj, ctx, from_pose)
    return aim if aim is not None else _approach_pose_for(obj, world)


class ResolveObjectNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        world: WorldKnowledge,
        target: str | None,
        ctx: StepContext,
        step_interaction_radius: float | None = None,
        interaction_name: str | None = None,
        exclude_full: bool = False,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._world = world
        self._target = target
        self._ctx = ctx
        self._step_interaction_radius = step_interaction_radius
        self._interaction_name = interaction_name
        self._exclude_full = exclude_full
        self._resolved: bool = False

    def initialise(self) -> None:
        self._resolved = False
        obj: WorldObject | None = None
        if self._target:
            obj = self._world.resolve(self._target, self._agent.state.pose, exclude_full=self._exclude_full)
            if obj is None:
                _bt_logger.warning(f"Agent {self._agent.state.agent_id}: step {self.name} could not resolve target={self._target!r}")

        if obj is not None:
            self._ctx.aim_seats = self._interaction_name is not None and InteractionType[self._interaction_name].kind.posture == "seated"
            self._ctx.target_pose = _walk_target(obj, self._world, self._ctx, self._agent.state.pose)
            self._ctx.target_object_id = obj.object_id
            self._ctx.interaction_radius = _resolve_interaction_radius(obj, self._step_interaction_radius, self._interaction_name)
            self._resolved = True

    def update(self) -> py_trees.common.Status:
        if self._resolved:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class GoToNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        target_pose: Pose2D | None = None,
        ctx: StepContext | None = None,
        world: WorldKnowledge | None = None,
        pool: AgentPool | None = None,
    ) -> None:
        super().__init__(name)
        if target_pose is None and ctx is None:
            raise ValueError("GoToNode requires either target_pose or ctx")
        self._agent = agent
        self._literal_pose = target_pose
        self._ctx = ctx
        self._world = world
        self._pool = pool

    def initialise(self) -> None:
        # Stale arrival latch from a previous step must not short-circuit this one.
        if self._pool is not None:
            idx = self._pool._id_to_idx.get(self._agent.state.agent_id)
            if idx is not None:
                self._pool.latched[idx] = False

    def _target(self) -> tuple[Pose2D | None, float]:
        if self._literal_pose is not None:
            return self._literal_pose, DISTANCE_TOLERANCE
        assert self._ctx is not None
        if self._world is not None and self._ctx.target_object_id is not None:
            obj = self._world.get(self._ctx.target_object_id)
            if obj is not None:
                self._ctx.target_pose = _walk_target(obj, self._world, self._ctx, self._agent.state.pose)
        return self._ctx.target_pose, self._ctx.interaction_radius

    def _arrived(self, target: Pose2D, tolerance: float) -> bool:
        # Literal go_to: couple to the physics-level arrival latch so "done" matches the stop condition.
        if self._literal_pose is not None and self._pool is not None:
            idx = self._pool._id_to_idx.get(self._agent.state.agent_id)
            if idx is not None:
                return bool(self._pool.latched[idx])
        return _at_target(self._agent, target, tolerance)

    def update(self) -> py_trees.common.Status:
        target, tolerance = self._target()
        if target is None:
            return py_trees.common.Status.FAILURE
        if self._arrived(target, tolerance):
            return py_trees.common.Status.SUCCESS
        _set_command(self._agent, _nav_command(agent=self._agent, target_pose=target))
        return py_trees.common.Status.RUNNING
