from __future__ import annotations

import math
from typing import TYPE_CHECKING

from arena_rclpy_mixins.Time import TimeNode
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from arena_humansim.core.agent_manager import ObstacleData
    from arena_humansim.core.world_knowledge import WorldObject
    from arena_humansim.global_planner import GlobalPlanner
    from arena_humansim.local_planner import LocalPlanner
    from arena_humansim.perception import Perception
    from arena_humansim.utils.types import Pose2D, Shape, SinkConfig, SourceConfig

_MISSING = object()

_STATIC_NS_BUCKETS: dict[str, str] = {
    "walls": "walls",
    "sources": "objects",
    "src_label": "objects",
    "sinks": "objects",
    "sink_label": "objects",
    "world_obj": "objects",
    "world_obj_label": "objects",
    "obstacles": "objects",
    "obstacle_label": "objects",
}
_STATIC_NS = frozenset(_STATIC_NS_BUCKETS)
_STATIC_BUCKETS = tuple(sorted(set(_STATIC_NS_BUCKETS.values())))


def rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def mk(ns: str, mid: int, mtype: int, stamp: Time | None) -> Marker:
    m = Marker()
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def arrow(ns: str, mid: int, stamp: Time, ox: float, oy: float, dx: float, dy: float, color: ColorRGBA, shaft: float = 0.03, head_d: float = 0.06, head_l: float = 0.06, z: float = 0.1) -> Marker:
    m = mk(ns, mid, Marker.ARROW, stamp)
    m.points = [Point(x=ox, y=oy, z=z), Point(x=ox + dx, y=oy + dy, z=z)]
    m.scale.x, m.scale.y, m.scale.z = shaft, head_d, head_l
    m.color = color
    return m


def text(ns: str, mid: int, stamp: Time, x: float, y: float, text: str, color: ColorRGBA, size: float = 0.2, z: float = 0.8) -> Marker:
    m = mk(ns, mid, Marker.TEXT_VIEW_FACING, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.scale.z = size
    m.color = color
    m.text = text
    return m


def line_strip(ns: str, mid: int, stamp: Time, pts: Iterable[tuple[float, float]], color: ColorRGBA, width: float = 0.02, z: float = 0.05) -> Marker:
    m = mk(ns, mid, Marker.LINE_STRIP, stamp)
    m.scale.x = width
    m.color = color
    m.points = [Point(x=px, y=py, z=z) for px, py in pts]
    return m


def sphere(ns: str, mid: int, stamp: Time, x: float, y: float, color: ColorRGBA, radius: float = 0.08, z: float = 0.1) -> Marker:
    m = mk(ns, mid, Marker.SPHERE, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.scale.x = m.scale.y = m.scale.z = radius * 2.0
    m.color = color
    return m


def cylinder(ns: str, mid: int, stamp: Time, x: float, y: float, color: ColorRGBA, radius: float = 0.5, height: float = 0.02, z: float = 0.0) -> Marker:
    m = mk(ns, mid, Marker.CYLINDER, stamp)
    m.pose.position.x, m.pose.position.y = x, y
    m.pose.position.z = z + height / 2.0
    m.scale.x = m.scale.y = radius * 2.0
    m.scale.z = height
    m.color = color
    return m


def cube(ns: str, mid: int, stamp: Time, x: float, y: float, z: float, sx: float, sy: float, sz: float, yaw: float, color: ColorRGBA) -> Marker:
    m = mk(ns, mid, Marker.CUBE, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
    m.pose.orientation.z = math.sin(yaw / 2.0)
    m.pose.orientation.w = math.cos(yaw / 2.0)
    m.color = color
    return m


_C_SRC = rgba(0.2, 1.0, 0.5, 0.4)
_C_SINK = rgba(1.0, 0.3, 0.3, 0.3)
_C_WALL = rgba(0.6, 0.0, 0.0, 0.6)
_C_WOBJ = rgba(0.3, 0.7, 1.0, 0.5)
_C_OBST = rgba(0.8, 0.5, 0.1, 0.4)
_C_OBST_LBL = rgba(1.0, 0.8, 0.3, 0.9)


def _shape_outline(pose: Pose2D, shape: Shape) -> list[tuple[float, float]]:
    from arena_humansim.utils.types import ShapeType

    cx, cy, th = pose.x, pose.y, pose.theta
    cos_t, sin_t = math.cos(th), math.sin(th)

    if shape.type == ShapeType.CIRCLE and shape.radius > 0:
        n = 24
        pts = []
        for i in range(n + 1):
            a = 2.0 * math.pi * i / n
            pts.append((cx + shape.radius * math.cos(a), cy + shape.radius * math.sin(a)))
        return pts
    if shape.vertices:
        pts = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in shape.vertices]
        pts.append(pts[0])
        return pts
    return []


class MarkerView:
    """Scoped handle to a marker namespace."""

    def __init__(self, pub: MarkerPublisher, ns: str, mtype: int, dirty: bool):
        self._pub = pub
        self._ns = ns
        self._mtype = mtype
        self._dirty = dirty

    def get(self, mid: int) -> tuple[Marker, bool]:
        return self._pub.get(self._ns, mid, self._mtype, self._dirty)

    def clear(self):
        """Remove all markers in this namespace from the scene.
        DELETE markers are emitted at next flush via stale detection."""
        to_remove = [k for k in self._pub._scene if k[0] == self._ns]
        for k in to_remove:
            del self._pub._scene[k]
            self._pub._dirty.discard(k)


class MarkerPublisher:
    def __init__(self, node: TimeNode, frame_id: str):
        self._node = node
        self._frame_id = frame_id
        self._pub = node.create_publisher(MarkerArray, "viz", 100)
        static_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pubs_static = {bucket: node.create_publisher(MarkerArray, f"viz_static/{bucket}", static_qos) for bucket in _STATIC_BUCKETS}
        self._scene: dict[tuple[str, int], Marker] = {}
        self._pool: dict[str, list[Marker]] = {}
        self._touched: set[tuple[str, int]] = set()
        self._touched_ns: set[str] = set()
        self._dirty: set[tuple[str, int]] = set()
        self._ns_count: dict[str, int] = {}
        self._infra_sigs: dict[str, object] = {}

    @property
    def watched(self) -> bool:
        return self._pub.get_subscription_count() > 0

    def infra_unchanged(self, bucket: str, sig: object) -> bool:
        prev = self._infra_sigs.get(bucket, _MISSING)
        if prev == sig:
            return True
        self._infra_sigs[bucket] = sig
        return False

    def _stamp(self) -> Time:
        return self._node.sim_time.to_msg()

    def view(self, ns: str, mtype: int, count: int = 0, dirty: bool = True) -> MarkerView:
        self._touched_ns.add(ns)
        have = self._ns_count.get(ns, 0)
        if count > have:
            pool = self._pool.setdefault(ns, [])
            for _ in range(count - have):
                pool.append(mk(ns, 0, mtype, None))
            self._ns_count[ns] = count
        return MarkerView(self, ns, mtype, dirty)

    def touch_ns(self, ns: str) -> None:
        self._touched_ns.add(ns)

    def get(self, ns: str, mid: int, mtype: int, dirty: bool = True) -> tuple[Marker, bool]:
        key = (ns, mid)
        self._touched.add(key)
        if key in self._scene:
            if dirty:
                self._dirty.add(key)
            return self._scene[key], False
        pool = self._pool.get(ns)
        if pool:
            m = pool.pop()
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
        else:
            m = mk(ns, mid, mtype, None)
        self._scene[key] = m
        self._dirty.add(key)
        return m, True

    def flush(self):
        stamp = self._stamp()
        stale = [(ns, mid) for ns, mid in self._scene if ns in self._touched_ns and (ns, mid) not in self._touched]

        changed_buckets: set[str] = set()
        dyn_deletes: list[Marker] = []
        static_deletes_by_bucket: dict[str, list[Marker]] = {b: [] for b in _STATIC_BUCKETS}
        for ns, mid in stale:
            del self._scene[(ns, mid)]
            self._dirty.discard((ns, mid))
            m = Marker()
            m.header.frame_id = self._frame_id
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            bucket = _STATIC_NS_BUCKETS.get(ns)
            if bucket is not None:
                static_deletes_by_bucket[bucket].append(m)
                changed_buckets.add(bucket)
            else:
                dyn_deletes.append(m)

        dyn_adds: list[Marker] = []
        for key in self._dirty:
            bucket = _STATIC_NS_BUCKETS.get(key[0])
            if bucket is not None:
                changed_buckets.add(bucket)
                continue
            m = self._scene[key]
            m.header.frame_id = self._frame_id
            m.header.stamp = stamp
            dyn_adds.append(m)

        if dyn_deletes or dyn_adds:
            # DELETEs first, a split into two messages on a depth-N queue could drop the DELETEs
            # under load, leaving stale markers in RViz indefinitely.
            ma = MarkerArray()
            ma.markers = dyn_deletes + dyn_adds
            self._pub.publish(ma)

        for bucket in changed_buckets:
            adds: list[Marker] = []
            for (ns, _mid), m in self._scene.items():
                if _STATIC_NS_BUCKETS.get(ns) == bucket:
                    m.header.frame_id = self._frame_id
                    m.header.stamp = stamp
                    adds.append(m)
            ma = MarkerArray()
            ma.markers = static_deletes_by_bucket[bucket] + adds
            self._pubs_static[bucket].publish(ma)

        self._touched.clear()
        self._touched_ns.clear()
        self._dirty.clear()

    def forget_all(self):
        m = Marker()
        m.action = Marker.DELETEALL
        ma = MarkerArray()
        ma.markers = [m]
        self._pub.publish(ma)
        for pub in self._pubs_static.values():
            pub.publish(ma)
        self._scene.clear()
        self._pool.clear()
        self._ns_count.clear()
        self._touched.clear()
        self._touched_ns.clear()
        self._dirty.clear()


def publish_infrastructure(
    pub: MarkerPublisher,
    sources: dict[str, SourceConfig],
    sinks: dict[str, SinkConfig],
    walls: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    world_objects: dict[str, WorldObject],
    obstacles: dict[str, ObstacleData] | None = None,
) -> None:
    obstacles = obstacles or {}

    walls_sig = tuple(walls.items())
    walls_changed = not pub.infra_unchanged("walls", walls_sig)
    objects_sig = (tuple(sources.keys()), tuple(sinks.keys()), tuple(world_objects.keys()), tuple(obstacles.keys()))
    objects_changed = not pub.infra_unchanged("objects", objects_sig)

    if not (walls_changed or objects_changed):
        return

    if walls_changed:
        pub.touch_ns("walls")
    if objects_changed:
        for ns in ("sources", "src_label", "sinks", "sink_label", "world_obj", "world_obj_label", "obstacles", "obstacle_label"):
            pub.touch_ns(ns)

    if not objects_changed:
        sources = {}
        sinks = {}
        world_objects = {}
        obstacles = {}
    if not walls_changed:
        walls = {}

    for i, (name, src) in enumerate(sources.items()):
        from arena_humansim.utils.types import ShapeType

        if src.shape.type == ShapeType.CIRCLE and src.shape.radius > 0:
            m, _ = pub.get("sources", i, Marker.CYLINDER)
            m.pose.position.x, m.pose.position.y = src.pose.x, src.pose.y
            m.pose.position.z = 0.01 + 0.01 / 2.0
            m.scale.x = m.scale.y = src.shape.radius * 2.0
            m.scale.z = 0.01
            m.color = _C_SRC
            m.points = []
        elif src.shape.vertices:
            m, _ = pub.get("sources", i, Marker.TRIANGLE_LIST)
            cx, cy, th = src.pose.x, src.pose.y, src.pose.theta
            cos_t, sin_t = math.cos(th), math.sin(th)
            world = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in src.shape.vertices]
            m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = _C_SRC
            m.points = []
            ox, oy = world[0]
            for vi in range(1, len(world) - 1):
                m.points.append(Point(x=ox, y=oy, z=0.01))
                m.points.append(Point(x=world[vi][0], y=world[vi][1], z=0.01))
                m.points.append(Point(x=world[vi + 1][0], y=world[vi + 1][1], z=0.01))
        else:
            m, _ = pub.get("sources", i, Marker.SPHERE)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = src.pose.x, src.pose.y, 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.15 * 2.0
            m.color = _C_SRC
            m.points = []
        m, _ = pub.get("src_label", i, Marker.TEXT_VIEW_FACING)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = src.pose.x, src.pose.y, 0.3
        m.scale.z = 0.15
        m.color = _C_SRC
        m.text = name

    for i, (name, sink) in enumerate(sinks.items()):
        from arena_humansim.utils.types import ShapeType

        if sink.shape.type == ShapeType.CIRCLE and sink.shape.radius > 0:
            m, _ = pub.get("sinks", i, Marker.CYLINDER)
            m.pose.position.x, m.pose.position.y = sink.pose.x, sink.pose.y
            m.pose.position.z = 0.01 + 0.01 / 2.0
            m.scale.x = m.scale.y = sink.shape.radius * 2.0
            m.scale.z = 0.01
            m.color = _C_SINK
            m.points = []
        elif sink.shape.vertices:
            m, _ = pub.get("sinks", i, Marker.TRIANGLE_LIST)
            cx, cy, th = sink.pose.x, sink.pose.y, sink.pose.theta
            cos_t, sin_t = math.cos(th), math.sin(th)
            world = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in sink.shape.vertices]
            m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = _C_SINK
            m.points = []
            ox, oy = world[0]
            for vi in range(1, len(world) - 1):
                m.points.append(Point(x=ox, y=oy, z=0.01))
                m.points.append(Point(x=world[vi][0], y=world[vi][1], z=0.01))
                m.points.append(Point(x=world[vi + 1][0], y=world[vi + 1][1], z=0.01))
        else:
            m, _ = pub.get("sinks", i, Marker.SPHERE)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = sink.pose.x, sink.pose.y, 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.15 * 2.0
            m.color = _C_SINK
            m.points = []
        m, _ = pub.get("sink_label", i, Marker.TEXT_VIEW_FACING)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = sink.pose.x, sink.pose.y, 0.3
        m.scale.z = 0.15
        m.color = _C_SINK
        m.text = name

    if walls:
        m, _ = pub.get("walls", 0, Marker.LINE_LIST)
        m.scale.x = 0.05
        m.color = _C_WALL
        m.points = []
        for (x1, y1), (x2, y2) in walls.values():
            m.points.append(Point(x=x1, y=y1, z=0.05))
            m.points.append(Point(x=x2, y=y2, z=0.05))

    for i, (oid, obj) in enumerate(world_objects.items()):
        m, _ = pub.get("world_obj", i, Marker.CUBE)
        m.pose.position.x, m.pose.position.y = obj.pose.x, obj.pose.y
        m.pose.position.z = 0.15
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color = _C_WOBJ
        m, _ = pub.get("world_obj_label", i, Marker.TEXT_VIEW_FACING)
        m.pose.position.x, m.pose.position.y = obj.pose.x, obj.pose.y
        m.pose.position.z = 0.4
        m.scale.z = 0.12
        m.color = _C_WOBJ
        m.text = f"{obj.type}\n{oid}"

    if obstacles:
        for i, (name, obs) in enumerate(obstacles.items()):
            x_min, x_max, y_min, y_max, z_min, z_max = obs.bb
            cx_local = (x_min + x_max) / 2.0
            cy_local = (y_min + y_max) / 2.0
            cz = (z_min + z_max) / 2.0
            sx = max(x_max - x_min, 0.01)
            sy = max(y_max - y_min, 0.01)
            sz = max(z_max - z_min, 0.01)
            cos_t = math.cos(obs.pose.theta)
            sin_t = math.sin(obs.pose.theta)
            wx = obs.pose.x + cos_t * cx_local - sin_t * cy_local
            wy = obs.pose.y + sin_t * cx_local + cos_t * cy_local
            m, _ = pub.get("obstacles", i, Marker.CUBE)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = wx, wy, cz
            m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
            m.pose.orientation.z = math.sin(obs.pose.theta / 2.0)
            m.pose.orientation.w = math.cos(obs.pose.theta / 2.0)
            m.color = _C_OBST
            label = obs.obstacle_type or name
            if obs.interaction_types:
                label += f"\n[{', '.join(obs.interaction_types)}]"
            m, _ = pub.get("obstacle_label", i, Marker.TEXT_VIEW_FACING)
            m.pose.position.x, m.pose.position.y = wx, wy
            m.pose.position.z = cz + sz / 2.0 + 0.15
            m.scale.z = 0.15
            m.color = _C_OBST_LBL
            m.text = label


def publish_module_markers(pub: MarkerPublisher, modules: Iterable[Perception | GlobalPlanner | LocalPlanner]) -> None:
    seen: set[int] = set()
    for mod in modules:
        mid = id(mod)
        if mid in seen:
            continue
        seen.add(mid)
        mod.publish_markers(pub)
