from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Sequence

import attrs

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

WAIT_RADIUS = 1.0
HOLD_RADIUS = 0.3
SLOT_SPACING = 0.7
SLOT_MARGIN = 0.35


def _inside(pts: tuple[tuple[float, float], ...], x: float, y: float) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(pts, pts[-1:] + pts[:-1], strict=True):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def _slots(zone: Zone) -> tuple[Pose2D, ...]:
    """Spots on a grid inside the footprint, the one farthest from the entry first, the hold pose when none fits."""
    if not zone.footprint:
        return (zone.hold,)
    xs, ys = [x for x, _ in zone.footprint], [y for _, y in zone.footprint]
    axes = []
    for lo, hi in ((min(xs) + SLOT_MARGIN, max(xs) - SLOT_MARGIN), (min(ys) + SLOT_MARGIN, max(ys) - SLOT_MARGIN)):
        n = max(1, math.floor(max(hi - lo, 0.0) / SLOT_SPACING) + 1)
        mid = (lo + hi) / 2.0
        axes.append([mid + (i - (n - 1) / 2.0) * SLOT_SPACING for i in range(n)])
    spots = [Pose2D(x=x, y=y, theta=zone.hold.theta) for x in axes[0] for y in axes[1] if _inside(zone.footprint, x, y)]
    spots.sort(key=lambda p: -math.hypot(p.x - zone.entry.x, p.y - zone.entry.y))
    return tuple(spots) or (zone.hold,)


def _center(zone: Zone) -> tuple[float, float]:
    """Footprint area centroid, the vertex mean for a degenerate footprint, the hold pose without one."""
    pts = zone.footprint
    if not pts:
        return zone.hold.x, zone.hold.y
    area = cx = cy = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1], strict=True):
        cross = ax * by - bx * ay
        area += cross
        cx += (ax + bx) * cross
        cy += (ay + by) * cross
    if abs(area) < 1e-9:
        return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    return cx / (3.0 * area), cy / (3.0 * area)


@attrs.frozen
class Zone:
    name: str
    destinations: tuple[str, ...]
    entry: Pose2D
    wait: Pose2D
    hold: Pose2D
    footprint: tuple[tuple[float, float], ...] = ()
    delay: float = 0.0
    slots: tuple[Pose2D, ...] = attrs.field(init=False, default=attrs.Factory(_slots, takes_self=True))
    center: tuple[float, float] = attrs.field(init=False, default=attrs.Factory(_center, takes_self=True))

    def contains(self, x: float, y: float) -> bool:
        """Whether a pedestrian at x, y stands inside the zone."""
        if math.hypot(x - self.hold.x, y - self.hold.y) <= HOLD_RADIUS:
            return True
        return _inside(self.footprint, x, y)


class Phase(enum.IntEnum):
    WALK = 0
    WAIT = 1
    BOARD = 2
    HOLD = 3


@attrs.define
class Transfer:
    goal: tuple[float, float]
    zone: str
    destination: str
    phase: Phase = Phase.WALK
    since: int = 0
    slot: int = -1
    held: int = 0


@attrs.frozen
class _Graph:
    area: dict[int, int]
    edges: dict[int, list[tuple[str, str, int]]]


class TeleportZones:
    """Teleport zones, their access and the pedestrians transferring through them."""

    def __init__(self) -> None:
        self.zones: dict[str, Zone] = {}
        self.open: dict[str, bool] = {}
        self.transfers: dict[int, Transfer] = {}
        self._banned: dict[int, tuple[tuple[float, float], set[str]]] = {}

    def set_zones(self, zones: Sequence[Zone], is_open: Sequence[bool]) -> None:
        self.zones = {z.name: z for z in zones}
        self.open = {z.name: bool(o) for z, o in zip(zones, is_open, strict=True)}
        self.clear_transfers()

    def clear_transfers(self) -> None:
        self.transfers.clear()
        self._banned.clear()

    def forget(self, aid: int) -> None:
        self.transfers.pop(aid, None)
        self._banned.pop(aid, None)

    def steer(self, agents: Sequence[BaseAgent], cmds: dict[int, HighLevelCommand]) -> dict[int, HighLevelCommand]:
        """Commands with each transferring pedestrian's target swapped for its zone's wait pose or its spot inside."""
        if not self.zones:
            return cmds
        out = dict(cmds)
        groups: dict[GlobalPlanner, list[tuple[BaseAgent, HighLevelCommand]]] = {}
        for agent in agents:
            cmd = cmds.get(agent.state.agent_id)
            if cmd is not None and cmd.type == CommandType.NAVIGATE and agent.state.kind != KIND_ROBOT:
                groups.setdefault(agent.global_planner, []).append((agent, cmd))
        for planner, group in groups.items():
            graph = self._graph(planner)
            comps = planner.components([agent.state.pose for agent, _ in group] + [cmd.target_pose for _, cmd in group])
            for k, (agent, cmd) in enumerate(group):
                aid = agent.state.agent_id
                goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
                transfer = self.transfers.get(aid)
                if transfer is not None and transfer.goal != goal:
                    transfer = None
                    del self.transfers[aid]
                banned = self._banned.get(aid)
                if banned is not None and banned[0] != goal:
                    banned = None
                    del self._banned[aid]
                if transfer is None or transfer.phase == Phase.WALK:
                    start, end = comps[k], comps[len(group) + k]
                    edge = self._route(graph, start, end, set() if banned is None else banned[1])
                    if edge is None and banned is not None:
                        del self._banned[aid]
                        edge = self._route(graph, start, end, set())
                    if edge is None:
                        self.transfers.pop(aid, None)
                        continue
                    if transfer is None or (transfer.zone, transfer.destination) != edge:
                        transfer = self.transfers[aid] = Transfer(goal=goal, zone=edge[0], destination=edge[1])
                z = self.zones[transfer.zone]
                out[aid] = attrs.evolve(cmd, target_pose=z.wait if transfer.phase in (Phase.WALK, Phase.WAIT) else z.slots[transfer.slot])
        return out

    def hold(self, pool: AgentPool, tick: int, timeout_ticks: int, dt: float) -> list[tuple[int, float, float]]:
        """Advance every pedestrian's phase, slow the waiting ones to desired speed 0, stand the holding ones still and return the due self-teleports as id, x, y."""
        robots = self._zone_robots(pool)
        teleports: list[tuple[int, float, float]] = []
        for aid in list(self.transfers):
            transfer = self.transfers[aid]
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                self.forget(aid)
                continue
            zone = self.zones[transfer.zone]
            is_open = self.open[transfer.zone]
            x, y = float(pool.pos[idx, 0]), float(pool.pos[idx, 1])
            covered = self._covered(zone, robots.get(zone.name, ()), float(pool.agent_radius[idx]))
            if transfer.phase == Phase.WALK and math.hypot(x - zone.wait.x, y - zone.wait.y) <= WAIT_RADIUS:
                transfer.phase, transfer.since = Phase.WAIT, tick
            if transfer.phase == Phase.BOARD and transfer.slot in covered:
                slot = self._free_slot(zone, covered)
                if slot is None:
                    transfer.phase, transfer.slot = Phase.WAIT, -1
                else:
                    transfer.slot = slot
            if transfer.phase == Phase.WAIT and is_open:
                slot = self._free_slot(zone, covered)
                if slot is not None:
                    transfer.phase, transfer.slot, transfer.since = Phase.BOARD, slot, tick
            elif transfer.phase == Phase.BOARD and self._at_slot(zone, transfer, x, y):
                transfer.phase, transfer.held = Phase.HOLD, tick
            elif transfer.phase == Phase.BOARD and not is_open and not zone.contains(x, y):
                transfer.phase, transfer.slot = Phase.WAIT, -1
            elif transfer.phase == Phase.HOLD and is_open and not zone.contains(x, y):
                transfer.phase = Phase.BOARD
            if transfer.phase != Phase.WALK and tick - transfer.since >= timeout_ticks:
                del self.transfers[aid]
                self._banned.setdefault(aid, (transfer.goal, set()))[1].add(transfer.zone)
                continue
            if transfer.phase in (Phase.WAIT, Phase.HOLD):
                pool.desired_vel[idx] = 0.0
            if transfer.phase == Phase.HOLD:
                pool.vel[idx] = 0.0
                pool.prev_vel[idx] = 0.0
            if transfer.phase == Phase.HOLD and zone.delay >= 0 and tick - transfer.held >= round(zone.delay / dt):
                cx, cy = zone.center
                dx, dy = self.zones[transfer.destination].center
                teleports.append((aid, x - cx + dx, y - cy + dy))
        return teleports

    def _free_slot(self, zone: Zone, covered: set[int]) -> int | None:
        taken = {t.slot for t in self.transfers.values() if t.zone == zone.name and t.phase in (Phase.BOARD, Phase.HOLD)} | covered
        return next((i for i in range(len(zone.slots)) if i not in taken), None)

    def _zone_robots(self, pool: AgentPool) -> dict[str, list[tuple[float, float, float]]]:
        """Robots standing in each zone as x, y, radius."""
        out: dict[str, list[tuple[float, float, float]]] = {}
        for i in range(pool.n):
            if pool.kind[i] != KIND_ROBOT:
                continue
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            for zone in self.zones.values():
                if zone.contains(x, y):
                    out.setdefault(zone.name, []).append((x, y, float(pool.agent_radius[i])))
        return out

    @staticmethod
    def _covered(zone: Zone, robots: Sequence[tuple[float, float, float]], radius: float) -> set[int]:
        """Spots a pedestrian of this radius cannot stand on because a robot in the zone overlaps them."""
        return {i for i, spot in enumerate(zone.slots) for x, y, r in robots if math.hypot(spot.x - x, spot.y - y) < r + radius}

    @staticmethod
    def _at_slot(zone: Zone, transfer: Transfer, x: float, y: float) -> bool:
        spot = zone.slots[transfer.slot]
        return math.hypot(x - spot.x, y - spot.y) <= HOLD_RADIUS

    def _graph(self, planner: GlobalPlanner) -> _Graph:
        """Teleport edges between areas, an area being the components that one zone's entry and hold pose join."""
        names = sorted(self.zones)
        comps = planner.components([p for name in names for p in (self.zones[name].entry, self.zones[name].hold)])
        parent: dict[int, int] = {}

        def find(c: int) -> int:
            while parent.setdefault(c, c) != c:
                c = parent[c]
            return c

        sides = {name: comps[2 * i] | comps[2 * i + 1] for i, name in enumerate(names)}
        for side in sides.values():
            roots = sorted({find(c) for c in side})
            for r in roots[1:]:
                parent[r] = roots[0]
        area = {c: find(c) for c in parent}
        edges: dict[int, list[tuple[str, str, int]]] = {}
        for name in names:
            if not sides[name]:
                continue
            a = area[min(sides[name])]
            for dest in self.zones[name].destinations:
                if dest not in sides or not sides[dest]:
                    continue
                b = area[min(sides[dest])]
                if a != b:
                    edges.setdefault(a, []).append((name, dest, b))
        return _Graph(area=area, edges=edges)

    @staticmethod
    def _route(graph: _Graph, start: frozenset[int], goal: frozenset[int], banned: set[str]) -> tuple[str, str] | None:
        """Zone and destination of the first edge of the fewest-teleports route from start to goal, None when start reaches goal on foot or no route exists."""
        starts = {graph.area.get(c, c) for c in start}
        goals = {graph.area.get(c, c) for c in goal}
        if not starts or not goals or starts & goals:
            return None
        first: dict[int, tuple[str, str] | None] = dict.fromkeys(starts)
        queue = deque(sorted(starts))
        while queue:
            c = queue.popleft()
            for name, dest, to in graph.edges.get(c, ()):
                if name in banned or to in first:
                    continue
                first[to] = first[c] or (name, dest)
                if to in goals:
                    return first[to]
                queue.append(to)
        return None
