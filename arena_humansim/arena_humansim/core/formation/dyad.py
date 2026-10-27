from __future__ import annotations

import math

from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D, pose_distance

from . import AgentLookup, Formation
from .anchor import Anchor


class DyadFormation(Formation):
    """Two agents facing each other across a midpoint anchor.

    On second member join, the pair orients along the line connecting their
    current poses; subsequently the anchor (typically a CentroidAnchor of the
    pair) provides the midpoint and each member faces it, turned by its
    ``heading_a`` / ``heading_b`` (rad, first and second member to join): the
    stance a two-person clip was made in.
    """

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        separation: float = 1.2,
        formation_scale: float = 1.0,
        heading_a: float = 0.0,
        heading_b: float = 0.0,
    ) -> None:
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.separation = separation * formation_scale
        self.headings = (float(heading_a), float(heading_b))
        self._members: list[int] = []
        self._axis_yaw: float = 0.0

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        if agent_id in self._members:
            return
        if len(self._members) >= 2:
            return
        self._members.append(agent_id)
        if len(self._members) == 2:
            a = self.agent_lookup(self._members[0])
            b = self.agent_lookup(self._members[1])
            if a is not None and b is not None:
                self._axis_yaw = math.atan2(
                    b.state.pose.y - a.state.pose.y,
                    b.state.pose.x - a.state.pose.x,
                )

    def on_leave(self, agent_id: int) -> None:
        if agent_id in self._members:
            self._members.remove(agent_id)

    def tick(self, dt: float) -> dict[int, Pose2D]:
        if not self._members:
            return {}
        center = self.anchor.pose()
        half = self.separation / 2.0
        out: dict[int, Pose2D] = {}
        if len(self._members) == 1:
            out[self._members[0]] = Pose2D(x=center.x, y=center.y, theta=self._axis_yaw + math.pi)
            return out
        dx = half * math.cos(self._axis_yaw)
        dy = half * math.sin(self._axis_yaw)
        a_id, b_id = self._members[0], self._members[1]
        out[a_id] = Pose2D(x=center.x - dx, y=center.y - dy, theta=self._axis_yaw + self.headings[0])
        out[b_id] = Pose2D(x=center.x + dx, y=center.y + dy, theta=self._axis_yaw + math.pi + self.headings[1])
        return out

    def arrived(self, agent_id: int) -> bool:
        if agent_id not in self._members:
            return True
        targets = self.tick(0.0)
        slot = targets.get(agent_id)
        if slot is None:
            return True
        agent = self.agent_lookup(agent_id)
        if agent is None:
            return True
        pose = agent.state.pose
        return pose_distance(pose, slot) < DISTANCE_TOLERANCE
