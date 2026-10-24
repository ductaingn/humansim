from __future__ import annotations

import math
from collections.abc import Callable

import attrs

from arena_humansim.utils import DISTANCE_TOLERANCE

SEAT_REACH = 0.9  # explicit seats sit inside furniture that keeps walkers out, arrival is declared from here and the agent is parked on the slot
from arena_humansim.utils.types import Pose2D, pose_distance

from . import AgentLookup, Formation
from .anchor import Anchor


@attrs.define
class _Slot:
    pose: Pose2D
    agent_id: int | None = None
    seat: int = -1


class ClusterFormation(Formation):
    """Unordered wait zone / multi-seat resource.

    Members are assigned the nearest unoccupied slot on join. Slots can be
    explicit (bench seats) or generated as a concentric ring around the anchor.
    """

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        slot_poses: list[Pose2D] | None = None,
        radius: float = 1.2,
        capacity: int = 8,
        formation_scale: float = 1.0,
        admissible: Callable[[int], bool] | None = None,
    ) -> None:
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.formation_scale = formation_scale
        self._admissible = admissible
        self._waiting: list[int] = []
        if slot_poses is not None:
            self._slots = [_Slot(pose=p, seat=i) for i, p in enumerate(slot_poses)]
            self._generated = False
        else:
            self._slots = self._generate_slots(anchor.pose(), radius * formation_scale, capacity)
            self._generated = True

    @staticmethod
    def _generate_slots(center: Pose2D, radius: float, count: int) -> list[_Slot]:
        slots: list[_Slot] = []
        for i in range(count):
            angle = 2.0 * math.pi * i / max(count, 1)
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            theta = angle + math.pi
            slots.append(_Slot(pose=Pose2D(x=x, y=y, theta=theta)))
        return slots

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        if not participant and not self._generated:
            # explicit seats belong to sitters, a queued member waits where it stopped instead of standing on the furniture
            return
        if any(s.agent_id == agent_id for s in self._slots) or agent_id in self._waiting:
            return
        if not self._assign(agent_id) and not self._generated:
            self._waiting.append(agent_id)

    def _assign(self, agent_id: int) -> bool:
        free = [s for s in self._slots if s.agent_id is None and (self._generated or self._admissible is None or self._admissible(s.seat))]
        if not free:
            return False
        agent = self.agent_lookup(agent_id)
        best = free[0] if agent is None else min(free, key=lambda s: pose_distance(s.pose, agent.state.pose))
        best.agent_id = agent_id
        return True

    def on_leave(self, agent_id: int) -> None:
        if agent_id in self._waiting:
            self._waiting.remove(agent_id)
            return
        for s in self._slots:
            if s.agent_id == agent_id:
                s.agent_id = None
                return

    def tick(self, dt: float) -> dict[int, Pose2D]:
        if self._generated:
            center = self.anchor.pose()
            if self._slots:
                anchor_x = sum(s.pose.x for s in self._slots) / len(self._slots)
                anchor_y = sum(s.pose.y for s in self._slots) / len(self._slots)
                if center.x != anchor_x or center.y != anchor_y:
                    dx = center.x - anchor_x
                    dy = center.y - anchor_y
                    for s in self._slots:
                        s.pose = Pose2D(x=s.pose.x + dx, y=s.pose.y + dy, theta=s.pose.theta)
        for agent_id in list(self._waiting):
            if self._assign(agent_id):
                self._waiting.remove(agent_id)
        return {s.agent_id: s.pose for s in self._slots if s.agent_id is not None}

    def arrived(self, agent_id: int) -> bool:
        slot = next((s for s in self._slots if s.agent_id == agent_id), None)
        if slot is None:
            return agent_id not in self._waiting
        agent = self.agent_lookup(agent_id)
        if agent is None:
            return True
        reach = DISTANCE_TOLERANCE if self._generated else SEAT_REACH
        return pose_distance(agent.state.pose, slot.pose) < reach

    def slot_of(self, agent_id: int) -> Pose2D | None:
        slot = next((s for s in self._slots if s.agent_id == agent_id), None)
        return slot.pose if slot is not None else None

    def seat_of(self, agent_id: int) -> Pose2D | None:
        if self._generated or not self.arrived(agent_id):
            return None
        return self.slot_of(agent_id)

    def seat_index_of(self, agent_id: int) -> int | None:
        slot = next((s for s in self._slots if s.agent_id == agent_id), None)
        return slot.seat if slot is not None and not self._generated else None

    def held_seats(self) -> list[int]:
        if self._generated:
            return []
        return [s.seat for s in self._slots if s.agent_id is not None]
