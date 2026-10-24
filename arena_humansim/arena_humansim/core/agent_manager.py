from __future__ import annotations

import array
import math
import time
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import rclpy
import rclpy.executors
import rclpy.subscription
import rclpy.timer
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentGestures as AgentGesturesMsg
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import AgentViz as AgentVizMsg
from arena_humansim_msgs.msg import Gesture as GestureMsg
from arena_humansim_msgs.msg import InteractionEvent as InteractionEventMsg
from arena_humansim_msgs.msg import InteractionFrame as InteractionFrameMsg
from arena_humansim_msgs.msg import ObstacleConfig as ObstacleConfigMsg
from arena_humansim_msgs.msg import Shape as ShapeMsg
from arena_humansim_msgs.msg import SinkConfig as SinkConfigMsg
from arena_humansim_msgs.msg import SourceConfig as SourceConfigMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from arena_humansim_msgs.msg import WorldObjectInfo as WorldObjectInfoMsg
from arena_humansim_msgs.srv import (
    AddObstacles,
    AddSink,
    AddSource,
    AddWalls,
    AddWorldObjects,
    Feedback,
    GetProfile,
    NotifyStimulus,
    RemoveAgents,
    RemoveObstacles,
    RemoveSink,
    RemoveSource,
    RemoveWalls,
    RemoveWorldObjects,
    ResetSimulation,
    SetElevatorDoors,
    SetElevators,
    SetFlow,
    SetWaypoints,
    SpawnAgents,
    TeleportAgents,
    UpdateRobot,
)
from geometry_msgs.msg import Point32, Vector3
from geometry_msgs.msg import Pose2D as Pose2DMsg
from py_trees.common import Status
from py_trees.trees import BehaviourTree
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock as RclClock
from rclpy.clock import ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Header

from arena_humansim.animation import MotionAnimation
from arena_humansim.collision import CollisionResolver
from arena_humansim.core import interaction_classes
from arena_humansim.core.agents import (
    BUILTIN_AGENTS,
    AgentType,
    BaseAgent,
    SampledParams,
    TickPhase,
    create_agent,
    rebase,
    spawn_params,
)
from arena_humansim.core.agents.loader import resolve_agent_type_name
from arena_humansim.core.agents.types import ATTENTION_KEYWORDS, ParamDist
from arena_humansim.core.animation_kinds import locomotion_states
from arena_humansim.core.behavior.compiler import BehaviorTreeFactory
from arena_humansim.core.despawn_monitor import DespawnMonitor
from arena_humansim.core.elevators import Cabin, Elevators
from arena_humansim.core.formation.clearance import Clearance
from arena_humansim.core.interaction_kinds import InteractionType, MembershipRole
from arena_humansim.core.interaction_manager import CONTACT_ENABLED, DEFAULT_STANDING_DISTANCE, GESTURE_ENABLED, InteractionEdge, InteractionManager, InteractionRow, check_gesture_mode
from arena_humansim.core.locomotion import LocomotionExtension
from arena_humansim.core.logger import SimulationLogger
from arena_humansim.core.pool import KIND_ROBOT, AgentPool, PoolAware
from arena_humansim.core.recorder import BagRecorder, default_record_dir
from arena_humansim.core.replay import ReplayManager, ReplayResult
from arena_humansim.core.robot_services import RobotServiceAdvertiser
from arena_humansim.core.spawn_scheduler import SpawnScheduler
from arena_humansim.core.viz import MarkerPublisher, publish_infrastructure, publish_module_markers
from arena_humansim.core.world_knowledge import FormationSpec, WorldKnowledge, WorldObject
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.occlusion import Occluder
from arena_humansim.perception import Perception
from arena_humansim.utils import RNG
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.scenario import EventScript, InteractionScript, ScenarioConfig
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    BehaviorTreeMovement,
    BeliefState,
    CommandType,
    GestureIntent,
    HighLevelCommand,
    InteractionOutcome,
    InteractionState,
    Pose2D,
    SeatPose,
    SeekSpec,
    Segments,
    Shape,
    SinkConfig,
    SourceConfig,
    SourceType,
    SpawnRequest,
    WallAware,
    WaypointMode,
    WaypointMovement,
    WorldAgentState,
    WorldState,
)


@attrs.define
class ObstacleData:
    name: str
    pose: Pose2D
    bb: tuple[float, float, float, float, float, float]  # x_min, x_max, y_min, y_max, z_min, z_max
    interaction_types: tuple[str, ...]
    obstacle_type: str
    wall_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


_TUNABLE_PARAMS = frozenset(
    {
        "force_local_planner",
        "waypoint_threshold",
        "min_speed_for_heading",
        "arrival_r_enter",
        "arrival_r_exit",
        "arrival_tau_brake",
        "elevator_ride_timeout",
        "profile_phases",
        "profile_interval",
    }
)
_RECONFIGURABLE_PARAMS = _TUNABLE_PARAMS | {
    "global_planner",
    "global_planner.inflation_radius",
    "global_planner.resolution",
    "global_planner.comfort_radius",
    "local_planner",
    "publish_markers",
    "rtf",
    "contact_mode",
    "locomotion_standing_distance",
    "gesture_mode",
}

_EXTERNAL_TIMEOUT_S = 2.0
_EXTERNAL_ADOPT_RADIUS = 1.0
_EXTERNAL_VELOCITY_WINDOW_S = 0.2
_EXTERNAL_MAX_SPEED = 10.0


@attrs.frozen
class _Pinned:
    """What an agent's spawn request or agent type fixed, so a reset leaves it alone."""

    policy: bool = False
    local_planner: bool = False
    global_planner: bool = False
    lp_keys: frozenset[str] = frozenset()


@attrs.define
class _ExternalEntity:
    """Pool binding for a world_state agent id, saved_policy_idx is None for spawned mirrors."""

    agent_id: int
    last_seen: int = 0
    owned: bool = True
    saved_policy_idx: int | None = None
    vel: tuple[float, float] = (0.0, 0.0)
    pose: tuple[float, float] | None = None
    stamp: float = 0.0


_NP_TYPES: dict[str, type[np.generic]] = {"B": np.uint8, "H": np.uint16, "h": np.int16, "i": np.int32, "I": np.uint32, "f": np.float32, "d": np.float64}

_CMD_LABELS = (*(c.name for c in CommandType), "INTR")
_CMD_LABEL_IDX = {c: i for i, c in enumerate(CommandType)}
_INTR_LABEL_IDX = len(_CMD_LABELS) - 1

_LATCHED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _seats(info: WorldObjectInfoMsg) -> list[SeatPose]:
    """The object's seats with their surface heights, a height missing or NaN left unmeasured."""
    heights = list(info.seat_heights) if len(info.seat_heights) == len(info.seats) else [math.nan] * len(info.seats)
    return [
        SeatPose(x=float(s.x), y=float(s.y), theta=float(s.theta), height=None if math.isnan(h) else float(h))
        for s, h in zip(info.seats, heights, strict=True)
    ]


def _flat(code: str, values: np.ndarray) -> array.array:
    out = array.array(code)
    out.frombytes(values.astype(_NP_TYPES[code]).tobytes())
    return out


def _interaction_frame(header: Header, table: tuple[InteractionRow, ...], edges: Sequence[InteractionEdge] = ()) -> InteractionFrameMsg:
    msg = InteractionFrameMsg(header=header)
    offsets = [0]
    members: list[int] = []
    roles: list[int] = []
    for row in table:
        members.extend(row.participants)
        members.extend(row.queue)
        roles.extend([MembershipRole.PARTICIPANT] * len(row.participants))
        roles.extend([MembershipRole.QUEUED] * len(row.queue))
        offsets.append(len(members))
    msg.interaction_id = array.array("i", [row.interaction_id for row in table])
    msg.interaction_type = array.array("B", [row.interaction_type for row in table])
    msg.outcome = array.array("B", [row.outcome for row in table])
    msg.object_id = [row.object_id for row in table]
    msg.arrived = [row.arrived for row in table]
    msg.holding = [row.holding for row in table]
    msg.duration = array.array("d", [row.duration for row in table])
    msg.member_offset = array.array("I", offsets)
    msg.member_id = array.array("i", members)
    msg.member_role = array.array("B", roles)
    for edge in edges:
        event = InteractionEventMsg()
        event.interaction_id = int(edge.interaction_id)
        event.interaction_type = int(edge.interaction_type)
        event.event = int(edge.edge)
        event.participants = [int(p) for p in edge.participants]
        msg.events.append(event)  # pyright: ignore[reportAttributeAccessIssue]
    return msg


def _group_by[T, K: Hashable](agents: Iterable[T], key: Callable[[T], K]) -> Iterable[tuple[K, list[T]]]:
    groups: dict[K, list[T]] = {}
    for agent in agents:
        k = key(agent)
        groups.setdefault(k, []).append(agent)
    return groups.items()


def arrival_latch_step(pool: AgentPool, r_enter: float, r_exit: float) -> None:
    n = pool.n
    if n == 0:
        return
    pos = pool.pos[:n]
    # Must run after pool.set_goals / pool.set_terminals - reads fresh goal_pos on release.
    goal = pool.goal_pos[:n]
    has_goal = pool.has_goal[:n]
    term = pool.terminal_pos[:n]
    has_term = pool.has_terminal[:n]
    latched = pool.latched[:n]

    d_term = np.hypot(term[:, 0] - pos[:, 0], term[:, 1] - pos[:, 1])

    release = latched & ((d_term > r_exit) | ~has_term)
    enter = (~latched) & has_term & (d_term < r_enter)
    new_latched = np.where(release, False, np.where(enter, True, latched))
    pool.latched[:n] = new_latched

    pool.goal_pos[:n] = np.where(new_latched[:, None], pos, goal)
    pool.has_goal[:n] = has_goal & (~new_latched)


def arrival_damp_step(pool: AgentPool, dt: float, tau_brake: float) -> None:
    n = pool.n
    if n == 0:
        return
    latched = pool.latched[:n]
    if not np.any(latched):
        return
    decay = float(np.exp(-dt / tau_brake))
    pool.vel[:n] = np.where(latched[:, None], pool.vel[:n] * decay, pool.vel[:n])


PRONE_RADIUS = 0.9  # half a body length, the disc a lying agent occupies


def _gesture_msg(intent: GestureIntent) -> GestureMsg:
    g = GestureMsg()
    g.slot = intent.slot
    g.at.x, g.at.y, g.at.z = intent.x, intent.y, intent.z
    g.clip = intent.clip
    g.hand = intent.hand
    g.render_pose_override = intent.render_pose_override
    return g


class AgentManager(Node):
    MODE_MASTER = "master"
    MODE_SUBSYSTEM = "subsystem"

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("arena_humansim", **node_kwargs)
        self._logger = self.get_logger()
        Loggable.init_logging(self)

        inherited = set(self.list_parameters([], 0).names)
        self.declare_parameter("seed", 0)
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("bt_tick_interval", 5, ParameterDescriptor(integer_range=[IntegerRange(from_value=1, to_value=100000)]))
        self.declare_parameter("perception", "default")
        self.declare_parameter("global_planner", "navmesh")
        self.declare_parameter(
            "global_planner.inflation_radius",
            0.38,
            ParameterDescriptor(description="wall clearance of planned paths [m], rounded up to whole cells", floating_point_range=[FloatingPointRange(from_value=0.0, to_value=5.0)]),
        )
        self.declare_parameter(
            "global_planner.resolution",
            0.2,
            ParameterDescriptor(description="planning grid cell size [m]", floating_point_range=[FloatingPointRange(from_value=0.01, to_value=5.0)]),
        )
        self.declare_parameter(
            "global_planner.comfort_radius",
            0.6,
            ParameterDescriptor(description="preferred wall clearance of navmesh paths [m], relaxed in narrow passages", floating_point_range=[FloatingPointRange(from_value=0.0, to_value=5.0)]),
        )
        self.declare_parameter("local_planner", "sfm")
        self.declare_parameter("force_local_planner", False)
        self.declare_parameter("robot_policy", "")
        self.declare_parameter("robot_shutdown", "")
        self.declare_parameter("force_waypoint_mode", "")
        self.declare_parameter("animation", "noop")
        self.declare_parameter("collision", "wall_projection")
        self.declare_parameter("occlusion", "bitmap")
        self.declare_parameter("mode", self.MODE_MASTER)
        self.declare_parameter("log_dir", "")
        self.declare_parameter("replay_mode", "")
        self.declare_parameter("waypoint_threshold", 0.1)
        self.declare_parameter("min_speed_for_heading", 0.1)
        self.declare_parameter("arrival_r_enter", 0.15)
        self.declare_parameter("arrival_r_exit", 0.30)
        self.declare_parameter("arrival_tau_brake", 0.15)
        self.declare_parameter("elevator_ride_timeout", 30.0)
        self.declare_parameter("publish_markers", 0, ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=2)]))
        self.declare_parameter("profile_phases", False)
        self.declare_parameter("profile_interval", 0)
        self.declare_parameter("record_bag", False)
        self.declare_parameter("strict_recording", True)
        self.declare_parameter("trial_id", "")
        self.declare_parameter("record_dir", "")
        self.declare_parameter("scenario", "")
        self.declare_parameter("ticks", 0)
        self.declare_parameter("time", 0.0)
        self.declare_parameter("rtf", 1.0, ParameterDescriptor(floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1000.0)]))
        self.declare_parameter("subsystem_overrun_policy", "lag")
        self.declare_parameter("contact_mode", CONTACT_ENABLED)
        self.declare_parameter("locomotion_standing_distance", DEFAULT_STANDING_DISTANCE)
        self.declare_parameter("gesture_mode", GESTURE_ENABLED)
        self._static_params = frozenset(self.list_parameters([], 0).names) - inherited - _RECONFIGURABLE_PARAMS

        seed = self._int_param("seed")
        self._dt = self._float_param("dt")
        self._bt_tick_interval = self._int_param("bt_tick_interval")
        self._mode = str(self.get_parameter("mode").value)
        log_dir = str(self.get_parameter("log_dir").value)
        replay_mode = str(self.get_parameter("replay_mode").value)
        self._load_tunables()
        if not (0.0 < self._arrival_r_enter < self._arrival_r_exit):
            raise ValueError(f"arrival_r_enter ({self._arrival_r_enter}) must be >0 and < arrival_r_exit ({self._arrival_r_exit})")
        self._ticks_limit = self._int_param("ticks")
        time_limit = float(self._float_param("time"))
        if self._ticks_limit == 0 and time_limit > 0.0:
            self._ticks_limit = max(1, int(round(time_limit / self._dt)))
        self._rtf = float(self._float_param("rtf"))
        self._subsystem_overrun_policy = str(self.get_parameter("subsystem_overrun_policy").value)
        if self._mode == self.MODE_SUBSYSTEM and self._subsystem_overrun_policy != "lag":
            raise ValueError(f"subsystem_overrun_policy={self._subsystem_overrun_policy!r} not implemented; only 'lag' is available")
        self._robot_policy_override = str(self.get_parameter("robot_policy").value)
        self._trial_id = str(self.get_parameter("trial_id").value)
        self._robot_shutdown_override = str(self.get_parameter("robot_shutdown").value).strip().lower()
        self._robot_shutdown = False  # resolved against scenario.simulation.robot_shutdown after load
        fwm_raw = str(self.get_parameter("force_waypoint_mode").value).strip().lower()
        if fwm_raw == "":
            self._force_waypoint_mode: WaypointMode | None = None
        else:
            try:
                self._force_waypoint_mode = WaypointMode[fwm_raw.upper()]
            except KeyError:
                valid = ", ".join(m.name.lower() for m in WaypointMode)
                raise ValueError(f"force_waypoint_mode must be empty or one of {{{valid}}}; got {fwm_raw!r}") from None
        _pm = self._int_param("publish_markers")
        if isinstance(_pm, bool):
            self._publish_markers = 2 if _pm else 0
        else:
            self._publish_markers = _pm
        self._phase_accum: dict[str, list[float]] = {}

        self._module_selections = {
            "perception": str(self.get_parameter("perception").value),
            "global_planner": str(self.get_parameter("global_planner").value),
            "local_planner": str(self.get_parameter("local_planner").value),
            "robot_policy": self._robot_policy_override or "(none)",
            "animation": str(self.get_parameter("animation").value),
            "collision": str(self.get_parameter("collision").value),
            "occlusion": str(self.get_parameter("occlusion").value),
        }

        self._rng = RNG(seed)

        self._occluder: Occluder = Occluder.get(self._module_selections["occlusion"])()
        self._perception_cache: dict[str, Perception] = {}
        default_name = self._module_selections["perception"]
        perception_cls = Perception.get(default_name)
        from arena_humansim.perception.default import DefaultPerception

        if issubclass(perception_cls, DefaultPerception):
            self._perception_cache[default_name] = perception_cls(occluder=self._occluder)
        else:
            self._perception_cache[default_name] = perception_cls()
        self._global_planner = GlobalPlanner.create(
            self._module_selections["global_planner"],
        )
        self._configure_global_planner()
        self.add_pre_set_parameters_callback(self._widen_ints)
        self.add_on_set_parameters_callback(self._validate_parameters)
        self.add_post_set_parameters_callback(self._on_parameters_set)
        self._local_planner = LocalPlanner.create(
            self._module_selections["local_planner"],
        )
        self._declare_local_planner_params(type(self._local_planner))
        self._lp_means = self._read_lp_means()
        self._pinned: dict[int, _Pinned] = {}
        self._default_policy_idx: int = 0
        self._policy_names = [self._module_selections["local_planner"]]
        self._policy_name_to_idx = {self._module_selections["local_planner"]: 0}
        self._policies = [self._local_planner]
        self._interaction_manager = InteractionManager(rng_manager=self._rng)
        self._interaction_manager.set_contact_mode(
            str(self.get_parameter("contact_mode").value),
            float(self._float_param("locomotion_standing_distance")),
        )
        self._gesture_mode = check_gesture_mode(str(self.get_parameter("gesture_mode").value))
        self.add_on_set_parameters_callback(self._on_contact_params)
        self._animation = MotionAnimation.create(
            self._module_selections["animation"],
        )
        self._collision = CollisionResolver.create(
            self._module_selections["collision"],
        )
        self._clearance = Clearance()
        self._locomotion = LocomotionExtension()
        self._wall_aware: tuple[WallAware, ...] = (
            self._local_planner,
            self._global_planner,
            self._collision,
            self._occluder,
            self._clearance,
        )
        self._pool_aware: tuple[PoolAware, ...] = (
            self._local_planner,
            self._global_planner,
            self._animation,
            self._collision,
            self._occluder,
            self._clearance,
            self._locomotion,
            *self._perception_cache.values(),
        )

        self._module_pool: dict[str, dict[str, Any]] = {
            "perception": {default_name: self._perception_cache[default_name]},
            "local_planner": {self._module_selections["local_planner"]: self._local_planner},
            "global_planner": {self._module_selections["global_planner"]: self._global_planner},
            "animation": {self._module_selections["animation"]: self._animation},
        }

        self._behavior_trees: dict[int, BehaviourTree | None] = {}
        self._bt_factories: dict[tuple, BehaviorTreeFactory] = {}
        self._world_knowledge = WorldKnowledge()
        self._interaction_manager.set_context(
            world_knowledge=self._world_knowledge,
            agent_lookup=lambda aid: self._agents.get(aid),
            visibility_lookup=lambda aid: self._pool.visible_agent_ids(aid),
            clearance=self._clearance,
        )
        self._event_bus = EventBus()
        self._event_scripts: list[EventScript] = []
        self._event_scripts_by_tick: dict[int, list[EventScript]] = {}
        self._interaction_scripts: list[InteractionScript] = []
        self._interaction_scripts_by_tick: dict[int, list[InteractionScript]] = {}

        self._spawn_scheduler = SpawnScheduler(rng=self._rng, allocate_id=self._allocate_agent_id)
        self._despawn_monitor = DespawnMonitor()
        self._last_spawned_ids: list[int] = []
        self._last_despawned_ids: list[int] = []

        self._pool = AgentPool()
        for sub in self._pool_aware:
            sub.attach(self._pool)

        self._agent_types: dict[str, AgentType] = dict(BUILTIN_AGENTS)
        self._agents: dict[int, BaseAgent] = {}
        self._pool_agent_ids: list[int] = []
        self._agent_name_to_id: dict[str, int] = {}
        self._external_entities: dict[int, _ExternalEntity] = {}
        self._external_timeout_ticks = max(1, int(round(_EXTERNAL_TIMEOUT_S / self._dt)))
        self._external_velocity_ticks = max(1, int(round(_EXTERNAL_VELOCITY_WINDOW_S / self._dt)))
        self._walls: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._obstacles: dict[str, ObstacleData] = {}
        self._elevators = Elevators()
        self._prone: set[int] = set()
        self._parked_from: dict[int, tuple[float, float, float]] = {}
        self._marker_pub = MarkerPublisher(self) if self._publish_markers > 0 else None
        self._tick_count: int = 0
        self._sim_time_ns: int = 0
        self._subsystem_epoch_ns: int = 0
        self._pending_scenario_spawns: deque[tuple[int, AgentStateMsg]] = deque()
        self._pending_stimuli: deque[tuple[int, int, str, float]] = deque()
        self._meta_dirty = True
        self._meta_published: tuple | None = None
        self._gestures_dirty = True
        self._gestures_published: list[tuple[int, GestureIntent]] | None = None
        self._interactions_published: tuple[InteractionRow, ...] | None = None
        self._interactions_had_edges = False
        self._tick_phases: dict[str, float] = {}
        self._overrun_count: int = 0
        self._last_overrun_log: float = 0.0
        self._tick_wall_start: float | None = None
        self._total_tick_compute_s: float = 0.0
        self._high_level_cmds: dict[int, HighLevelCommand] = {}
        self._robot_service_advertiser = RobotServiceAdvertiser()
        self._cached_intermediate_goals: dict[int, Pose2D] = {}
        self._next_agent_id: int = 1
        self._next_robot_id: int = -1

        self._agent_states_pub = self.create_publisher(
            AgentFrameMsg,
            "agent_states",
            10,
        )
        self._agent_meta_pub = self.create_publisher(AgentMetaMsg, "agent_meta", _LATCHED_QOS)
        self._agent_gestures_pub = self.create_publisher(AgentGesturesMsg, "agent_gestures", _LATCHED_QOS)
        self._viz_state_pub = self.create_publisher(
            AgentVizMsg,
            "viz_state",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )

        self._interactions_pub = self.create_publisher(InteractionFrameMsg, "interactions", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._world_geometry_pub = self.create_publisher(
            WorldGeometryMsg,
            "world_geometry",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self._world_state_sub = self.create_subscription(
            AgentStatesMsg,
            "world_state",
            self._world_state_callback,
            10,
        )
        self._latest_world_state: AgentStatesMsg | None = None

        self._spawn_srv = self.create_service(
            SpawnAgents,
            "spawn_agents",
            self._spawn_agents_callback,
        )
        self._remove_srv = self.create_service(
            RemoveAgents,
            "remove_agents",
            self._remove_agents_callback,
        )
        self._update_robot_srv = self.create_service(
            UpdateRobot,
            "update_robot",
            self._update_robot_callback,
        )
        self._set_waypoints_srv = self.create_service(
            SetWaypoints,
            "set_waypoints",
            self._set_waypoints_callback,
        )

        self._set_flow_srv = self.create_service(
            SetFlow,
            "set_flow",
            self._set_flow_callback,
        )
        self._notify_stimulus_srv = self.create_service(
            NotifyStimulus,
            "notify_stimulus",
            self._notify_stimulus_callback,
        )
        self._add_source_srv = self.create_service(
            AddSource,
            "add_source",
            self._add_source_callback,
        )
        self._remove_source_srv = self.create_service(
            RemoveSource,
            "remove_source",
            self._remove_source_callback,
        )
        self._add_sink_srv = self.create_service(
            AddSink,
            "add_sink",
            self._add_sink_callback,
        )
        self._remove_sink_srv = self.create_service(
            RemoveSink,
            "remove_sink",
            self._remove_sink_callback,
        )
        self._add_walls_srv = self.create_service(
            AddWalls,
            "add_walls",
            self._add_walls_callback,
        )
        self._remove_walls_srv = self.create_service(
            RemoveWalls,
            "remove_walls",
            self._remove_walls_callback,
        )
        self._add_obstacles_srv = self.create_service(
            AddObstacles,
            "add_obstacles",
            self._add_obstacles_callback,
        )
        self._remove_obstacles_srv = self.create_service(
            RemoveObstacles,
            "remove_obstacles",
            self._remove_obstacles_callback,
        )
        self._add_world_objects_srv = self.create_service(
            AddWorldObjects,
            "add_world_objects",
            self._add_world_objects_callback,
        )
        self._remove_world_objects_srv = self.create_service(
            RemoveWorldObjects,
            "remove_world_objects",
            self._remove_world_objects_callback,
        )
        self._set_elevators_srv = self.create_service(
            SetElevators,
            "set_elevators",
            self._set_elevators_callback,
        )
        self._set_elevator_doors_srv = self.create_service(
            SetElevatorDoors,
            "set_elevator_doors",
            self._set_elevator_doors_callback,
        )
        self._teleport_agents_srv = self.create_service(
            TeleportAgents,
            "teleport_agents",
            self._teleport_agents_callback,
        )
        self._reset_srv = self.create_service(
            ResetSimulation,
            "reset",
            self._reset_callback,
        )
        self._get_profile_srv = self.create_service(
            GetProfile,
            "get_profile",
            self._get_profile_callback,
        )

        self._timer: rclpy.timer.Timer | None = None
        self._clock_sub: rclpy.subscription.Subscription | None = None
        if self._mode == self.MODE_MASTER:
            self._setup_master_mode()
        elif self._mode == self.MODE_SUBSYSTEM:
            self._setup_subsystem_mode()
        else:
            raise ValueError(f"Unknown mode '{self._mode}'. Use '{self.MODE_MASTER}' or '{self.MODE_SUBSYSTEM}'.")

        self._sim_logger = None
        if log_dir:
            config_snapshot = {
                "seed": seed,
                "dt": self._dt,
                "bt_tick_interval": self._bt_tick_interval,
                "modules": self._module_selections,
                "parameters": self._collect_ros_parameters(),
            }
            self._sim_logger = SimulationLogger(log_dir, seed, config_snapshot)
            self._logger.info(f"Logging enabled: {log_dir}")

        self._replay = None
        if replay_mode:
            self._replay = ReplayManager()
            self._replay.load(replay_mode)
            self._logger.info(f"Replay mode: loaded {self._replay.tick_count} ticks from {replay_mode}")

        self._recorder: BagRecorder | None = None
        if self.get_parameter("record_bag").value:
            from pathlib import Path

            record_dir = str(self.get_parameter("record_dir").value)
            if not record_dir:
                self._logger.warning("record_bag=true but record_dir empty; using default")
                target = default_record_dir()
            else:
                target = Path(record_dir)
            self._recorder = BagRecorder(
                self,
                target,
                strict=bool(self.get_parameter("strict_recording").value),
                on_contamination=self._abort_contaminated,
                provenance={
                    "trial_id": self._trial_id,
                    "seed": seed,
                    "local_planner": self._module_selections["local_planner"],
                    "robot_policy": self._robot_policy_override,
                },
            )

        scenario_arg = str(self.get_parameter("scenario").value)
        if scenario_arg and self._mode == self.MODE_MASTER:
            self._load_scenario_file(scenario_arg)
        elif scenario_arg:
            self._logger.warning(f"scenario='{scenario_arg}' ignored in mode={self._mode} (orchestrator-driven)")

        self._logger.info(f"AgentManager initialized (seed={seed}, dt={self._dt}, mode={self._mode})")
        self._logger.info("Modules: " + ", ".join(f"{k}={v}" for k, v in self._module_selections.items()))

    def _int_param(self, name: str) -> int:
        value = self.get_parameter(name).value
        if not isinstance(value, int):
            raise TypeError(f"parameter {name!r} must be an integer, got {value!r}")
        return value

    def _float_param(self, name: str) -> float:
        value = self.get_parameter(name).value
        if not isinstance(value, (int, float)):
            raise TypeError(f"parameter {name!r} must be a number, got {value!r}")
        return value

    def _collect_ros_parameters(self) -> dict[str, Any]:
        params = {}
        for name in (
            "seed",
            "dt",
            "bt_tick_interval",
            "mode",
            "perception",
            "global_planner",
            "local_planner",
            "animation",
            "log_dir",
            "replay_mode",
        ):
            try:
                params[name] = self.get_parameter(name).value
            except Exception:
                pass
        return params

    def _allocate_agent_id(self) -> int:
        aid = self._next_agent_id
        self._next_agent_id += 1
        return aid

    def _allocate_robot_id(self) -> int:
        aid = self._next_robot_id
        self._next_robot_id -= 1
        return aid

    def _widen_ints(self, params: list[Parameter]) -> list[Parameter]:
        def widen(p: Parameter) -> Parameter:
            if p.type_ == Parameter.Type.INTEGER and isinstance(p.value, int) and self.has_parameter(p.name) and self.get_parameter(p.name).type_ == Parameter.Type.DOUBLE:
                return Parameter(p.name, Parameter.Type.DOUBLE, float(p.value))
            return p

        return [widen(p) for p in params]

    def _load_tunables(self) -> None:
        self._force_local_planner = bool(self.get_parameter("force_local_planner").value)
        self._waypoint_threshold = self._float_param("waypoint_threshold")
        self._min_speed_for_heading = self._float_param("min_speed_for_heading")
        self._arrival_r_enter = float(self._float_param("arrival_r_enter"))
        self._arrival_r_exit = float(self._float_param("arrival_r_exit"))
        self._arrival_tau_brake = float(self._float_param("arrival_tau_brake"))
        if self._arrival_tau_brake < self._dt:
            self._logger.warning(f"arrival_tau_brake ({self._arrival_tau_brake}) < dt ({self._dt}), clamping to dt")
            self._arrival_tau_brake = float(self._dt)
        self._elevator_ride_ticks = max(1, int(round(self._float_param("elevator_ride_timeout") / self._dt)))
        self._profile_phases = self.get_parameter("profile_phases").value
        self._profile_interval = self._int_param("profile_interval")

    def _validate_parameters(self, params: list[Parameter]) -> SetParametersResult:
        incoming = {p.name: p.value for p in params}
        static = sorted(self._static_params & incoming.keys())
        if static:
            return SetParametersResult(successful=False, reason=f"not reconfigurable at runtime: {', '.join(static)}")
        r_enter = incoming.get("arrival_r_enter", self._float_param("arrival_r_enter"))
        r_exit = incoming.get("arrival_r_exit", self._float_param("arrival_r_exit"))
        if not (isinstance(r_enter, (int, float)) and isinstance(r_exit, (int, float)) and 0.0 < r_enter < r_exit):
            return SetParametersResult(successful=False, reason=f"arrival_r_enter ({r_enter}) must be >0 and < arrival_r_exit ({r_exit})")
        for name, registry in (("local_planner", LocalPlanner), ("global_planner", GlobalPlanner)):
            if name in incoming and incoming[name] not in registry.list_available():
                return SetParametersResult(successful=False, reason=f"{name}: unknown module {incoming[name]!r}, available: {', '.join(registry.list_available())}")
        return SetParametersResult(successful=True)

    def set_parameters_atomically(self, parameter_list: list[Parameter]) -> SetParametersResult:
        ordered = sorted(parameter_list, key=lambda p: p.name)
        self._declare_selected_planner_params(ordered)
        return super().set_parameters_atomically(ordered)

    def _on_parameters_set(self, params: list[Parameter]) -> None:
        self._declare_selected_planner_params(params)

    def _declare_selected_planner_params(self, params: list[Parameter]) -> None:
        for p in params:
            if p.name == "local_planner" and p.value in LocalPlanner.list_available():
                self._declare_local_planner_params(LocalPlanner.get_class(p.value))

    def _apply_parameters(self) -> None:
        self._load_tunables()
        self._select_local_planner(str(self.get_parameter("local_planner").value))
        self._select_global_planner(str(self.get_parameter("global_planner").value))
        self._configure_global_planner()
        self._set_marker_level(self._int_param("publish_markers"))
        self._set_rtf(self._float_param("rtf"))
        old_means, self._lp_means = self._lp_means, self._read_lp_means()
        self._reseat_agents(old_means)
        self._prune_planners()

    def _select_local_planner(self, name: str) -> None:
        idx = self._resolve_policy_idx(name)
        self._default_policy_idx = idx
        self._local_planner = self._policies[idx]
        self._module_pool["local_planner"][name] = self._local_planner
        self._module_selections["local_planner"] = name

    def _select_global_planner(self, name: str) -> None:
        global_planners = self._module_pool["global_planner"]
        if name not in global_planners:
            global_planners[name] = GlobalPlanner.create(name)
            self._attach_late(global_planners[name])
        self._global_planner = global_planners[name]
        self._module_selections["global_planner"] = name

    def _set_marker_level(self, level: int) -> None:
        if level == self._publish_markers:
            return
        self._publish_markers = level
        if level == 0:
            self._viz_state_pub.publish(AgentVizMsg(header=self._header()))
            if self._marker_pub is not None:
                self._marker_pub.forget_all()
        if level > 0 and self._marker_pub is None:
            self._marker_pub = MarkerPublisher(self)

    def _set_rtf(self, rtf: float) -> None:
        if rtf == self._rtf:
            return
        self._rtf = rtf
        if self._mode == self.MODE_MASTER:
            assert self._timer is not None
            self._timer.timer_period_ns = int(1e9 * self._dt / rtf) if rtf > 0.0 else 0

    def _declare_local_planner_params(self, planner_cls: type[LocalPlanner]) -> None:
        for key in planner_cls.PARAM_DEFAULTS:
            if not self.has_parameter(f"local_planner.{key}"):
                self.declare_parameter(f"local_planner.{key}", 0.0, ParameterDescriptor(description=f"mean of {key}, 0 = agent type's own, a per-agent spawn value still wins"))

    def _read_lp_means(self) -> dict[str, float]:
        prefix = "local_planner."
        means = {name.removeprefix(prefix): self._float_param(name) for name in self.list_parameters([], 0).names if name.startswith(prefix)}
        return {key: mean for key, mean in means.items() if mean > 0.0}

    def _local_planner_means(self, planner_name: str) -> dict[str, float]:
        return {key: mean for key, mean in self._lp_means.items() if key in LocalPlanner.get_class(planner_name).PARAM_DEFAULTS}

    def _lp_dist(self, agent: BaseAgent, key: str) -> ParamDist | None:
        agent_type = self._agent_types.get(agent.params.name)
        if agent_type is not None and key in agent_type.local_planner_params:
            return agent_type.local_planner_params[key]
        return next((p.PARAM_DEFAULTS[key] for p in self._policies if key in p.PARAM_DEFAULTS), None)

    def _reseat_agents(self, old_means: dict[str, float]) -> None:
        """Move every agent that follows the defaults onto the current planners and planner means."""
        saved = {e.agent_id: e for e in self._external_entities.values() if e.saved_policy_idx is not None}
        for aid, agent in self._agents.items():
            pinned = self._pinned[aid]
            idx = self._pool.idx(aid)
            if not pinned.local_planner:
                agent.local_planner = self._local_planner
            if not pinned.global_planner:
                agent.global_planner = self._global_planner
            if not pinned.policy:
                if aid in saved:
                    saved[aid].saved_policy_idx = self._default_policy_idx
                elif self._pool.policy_idx[idx] >= 0:
                    self._pool.policy_idx[idx] = self._default_policy_idx

            spawn = spawn_params(agent)
            lp = dict(spawn.local_planner_params)
            for key in {*lp, *type(agent.local_planner).PARAM_DEFAULTS} - pinned.lp_keys:
                dist = self._lp_dist(agent, key)
                if dist is None:
                    continue
                old = old_means.get(key, dist.mean)
                new = dist.with_mean(self._lp_means.get(key, dist.mean))
                lp[key] = float(np.clip(lp.get(key, old) + new.mean - old, new.clip_low, new.clip_high))
            if lp != spawn.local_planner_params:
                rebase(agent, attrs.evolve(spawn, local_planner_params=lp))
                for planner in self._policies:
                    if planner.PARAM_DEFAULTS:
                        planner.on_pool_add(idx, agent)

    def _prune_planners(self) -> None:
        """Drop every planner that no agent uses and keep the defaults."""
        agents = list(self._agents.values())
        n = self._pool.n
        used = {int(i) for i in self._pool.policy_idx[:n]} | {e.saved_policy_idx for e in self._external_entities.values()} | {self._default_policy_idx}
        keep = [i for i, p in enumerate(self._policies) if i in used or any(a.local_planner is p for a in agents)]
        dropped: list[LocalPlanner | GlobalPlanner] = [p for i, p in enumerate(self._policies) if i not in keep]
        remap = np.full(len(self._policies) + 1, -1, dtype=self._pool.policy_idx.dtype)
        remap[keep] = np.arange(len(keep))
        self._pool.policy_idx[:n] = remap[self._pool.policy_idx[:n]]
        for entity in self._external_entities.values():
            if entity.saved_policy_idx is not None:
                entity.saved_policy_idx = int(remap[entity.saved_policy_idx])
        self._default_policy_idx = int(remap[self._default_policy_idx])
        self._policies = [self._policies[i] for i in keep]
        self._policy_names = [self._policy_names[i] for i in keep]
        self._policy_name_to_idx = {name: i for i, name in enumerate(self._policy_names)}
        self._meta_dirty = True

        global_planners = list(self._module_pool["global_planner"].values())
        dropped += [p for p in global_planners if p is not self._global_planner and not any(a.global_planner is p for a in agents)]

        for planner in dropped:
            self._pool.unregister_extension(planner)
        self._wall_aware = tuple(w for w in self._wall_aware if not any(w is p for p in dropped))
        self._module_pool = {kind: {name: m for name, m in pool.items() if not any(m is p for p in dropped)} for kind, pool in self._module_pool.items()}

    def _configure_global_planner(self) -> None:
        self._global_planner.configure(
            inflation_radius=self._float_param("global_planner.inflation_radius"),
            resolution=self._float_param("global_planner.resolution"),
            comfort_radius=self._float_param("global_planner.comfort_radius"),
        )

    def _attach_late(self, planner: LocalPlanner | GlobalPlanner) -> None:
        """Wire a planner created after __init__ into the pool and the walls."""
        agents = [self._agents[aid] for aid in self._pool_agent_ids if aid in self._agents]
        self._pool.attach_late(planner, agents)
        self._wall_aware = (*self._wall_aware, planner)
        segments = self._all_wall_segments()
        if segments:
            planner.set_walls(segments)

    def _resolve_policy_idx(self, name: str) -> int:
        if not name:
            return -1
        idx = self._policy_name_to_idx.get(name)
        if idx is not None:
            return idx
        planner = LocalPlanner.create(name)
        self._attach_late(planner)
        self._declare_local_planner_params(type(planner))
        idx = len(self._policies)
        self._policies.append(planner)
        self._policy_names.append(name)
        self._policy_name_to_idx[name] = idx
        self._meta_dirty = True
        return idx

    def _resolve_perception_layer(self, name: str) -> Perception:
        if name not in self._perception_cache:
            from arena_humansim.perception.default import DefaultPerception

            perception_cls = Perception.get(name)
            if issubclass(perception_cls, DefaultPerception):
                instance: Perception = perception_cls(occluder=self._occluder)
            else:
                instance = perception_cls()
            self._perception_cache[name] = instance
            self._module_pool["perception"][name] = instance
        return self._perception_cache[name]

    def _resolve_scenario_path(self, name_or_path: str) -> Path:
        from arena_humansim.utils.scenario_discovery import find_scenario_path

        return find_scenario_path(name_or_path)

    def _load_scenario_file(self, name_or_path: str) -> None:
        from arena_humansim.utils.scenario import dump_resolved_scenario, load_scenario

        path = self._resolve_scenario_path(name_or_path)
        scenario = load_scenario(str(path))
        self._logger.info(f"Loading scenario '{scenario.name}' from {path}")

        if self._recorder is not None:
            try:
                snapshot_path = self._recorder.record_dir / "scenario.yaml"
                dump_resolved_scenario(path, snapshot_path)
                self._logger.info(f"Snapshotted resolved scenario to {snapshot_path}")
            except Exception as exc:
                self._logger.warning(f"Failed to snapshot scenario: {exc}")

        self._apply_scenario(scenario)

    def _apply_scenario(self, scenario: ScenarioConfig) -> None:
        self._init_world_knowledge(scenario)

        if self._ticks_limit == 0 and scenario.simulation.max_ticks > 0:
            self._ticks_limit = int(scenario.simulation.max_ticks)

        if self._robot_shutdown_override in ("true", "1"):
            self._robot_shutdown = True
        elif self._robot_shutdown_override in ("false", "0"):
            self._robot_shutdown = False
        elif self._robot_shutdown_override == "":
            self._robot_shutdown = bool(scenario.simulation.robot_shutdown)
        else:
            raise ValueError(f"robot_shutdown must be 'true', 'false', or empty; got {self._robot_shutdown_override!r}")
        if self._robot_shutdown:
            robots = [a for a in scenario.agents if int(a.kind) == int(AgentKind.ROBOT)]
            if not robots:
                self._logger.warning("robot_shutdown=true but scenario has no robot agents; the run will not auto-terminate")
            else:
                non_once = [a for a in robots if a.waypoint_mode != WaypointMode.ONCE]
                if non_once:
                    ids = ", ".join(str(a.agent_id) for a in non_once)
                    self._logger.warning(f"robot_shutdown=true but robot agents [{ids}] use non-ONCE waypoint mode; they will never satisfy the goal-reached condition")

        self._interaction_manager.set_context(
            formation_scale=float(scenario.simulation.formation_scale),
        )

        if scenario.walls:
            walls_req = AddWalls.Request()
            for w in scenario.walls:
                walls_req.names.append(w.name)  # pyright: ignore[reportAttributeAccessIssue]
                walls_req.starts.append(Point32(x=float(w.start.x), y=float(w.start.y), z=0.0))  # pyright: ignore[reportAttributeAccessIssue]
                walls_req.ends.append(Point32(x=float(w.end.x), y=float(w.end.y), z=0.0))  # pyright: ignore[reportAttributeAccessIssue]
            self._add_walls_callback(walls_req, AddWalls.Response())

        if scenario.obstacles:
            obs_req = AddObstacles.Request()
            for o in scenario.obstacles:
                m = ObstacleConfigMsg()
                m.name = o.name
                m.pose = Pose2DMsg(x=o.pose.x, y=o.pose.y, theta=o.pose.theta)
                m.bb_x_min, m.bb_x_max = o.bb.x_min, o.bb.x_max
                m.bb_y_min, m.bb_y_max = o.bb.y_min, o.bb.y_max
                m.bb_z_min, m.bb_z_max = o.bb.z_min, o.bb.z_max
                m.obstacle_type = o.obstacle_type
                m.interaction_types = list(o.interaction_types)
                obs_req.obstacles.append(m)  # pyright: ignore[reportAttributeAccessIssue]
            self._add_obstacles_callback(obs_req, AddObstacles.Response())

        self._init_scenario_agents(scenario)

    def _init_scenario_agents(self, scenario: ScenarioConfig) -> None:
        if not scenario.agents:
            return

        from arena_humansim_msgs.msg import Waypoint as WaypointMsg
        from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

        immediate = SpawnAgents.Request()
        pending: list[tuple[int, AgentStateMsg]] = []
        for a in scenario.agents:
            if int(a.agent_id) == 0:
                a.agent_id = self._allocate_robot_id() if int(a.kind) == int(AgentKind.ROBOT) else self._allocate_agent_id()
            if int(a.kind) == int(AgentKind.ROBOT) and a.services:
                self._robot_service_advertiser.register(int(a.agent_id), a.services)
            msg = AgentStateMsg()
            msg.agent_id = int(a.agent_id)
            msg.pose = Pose2DMsg(x=a.spawn_pose.x, y=a.spawn_pose.y, theta=a.spawn_pose.theta)
            msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            msg.desired_velocity = float(a.desired_velocity)
            msg.radius = float(a.agent_radius)
            msg.agent_type = a.agent_type
            msg.kind = int(a.kind)
            if int(a.kind) == int(AgentKind.ROBOT) and self._robot_policy_override:
                msg.policy = self._robot_policy_override
            else:
                msg.policy = a.policy
            msg.policy_params = a.policy_params

            resolved_mode = self._force_waypoint_mode if (self._force_waypoint_mode is not None and int(a.kind) != int(AgentKind.ROBOT)) else a.waypoint_mode
            wps = WaypointsMsg()
            wps.mode = int(resolved_mode)
            seq = list(a.goal_sequence)
            if resolved_mode != WaypointMode.ONCE and len(seq) < 2:
                seq = [*seq, a.spawn_pose]
            for gp in seq:
                w = WaypointMsg()
                w.pose = Pose2DMsg(x=gp.x, y=gp.y, theta=gp.theta)
                w.radius = 0.0
                wps.points.append(w)  # pyright: ignore[reportAttributeAccessIssue]
            msg.waypoints = wps

            if a.spawn_tick > 0:
                pending.append((int(a.spawn_tick), msg))
            else:
                immediate.agents.append(msg)  # pyright: ignore[reportAttributeAccessIssue]

        pending.sort(key=lambda item: item[0])
        self._pending_scenario_spawns.extend(pending)

        if immediate.agents:
            resp = self._spawn_agents_callback(immediate, SpawnAgents.Response())
            if not resp.success:
                self._logger.error(f"scenario spawn failed: {resp.message}")
            else:
                self._logger.info(f"scenario spawned {len(resp.spawned_ids)} agent(s) immediately")

        if self._pending_scenario_spawns:
            self._logger.info(f"scenario has {len(self._pending_scenario_spawns)} deferred agent(s)")

    def _init_world_knowledge(self, scenario: ScenarioConfig) -> None:
        if scenario.agent_types:
            self._agent_types = {**BUILTIN_AGENTS, **scenario.agent_types}
        self._world_knowledge.clear()
        for wo_cfg in scenario.world_objects:
            obj = WorldObject(
                object_id=wo_cfg.object_id,
                type=wo_cfg.type,
                pose=Pose2D(x=wo_cfg.pose.x, y=wo_cfg.pose.y, theta=wo_cfg.pose.theta),
                capacity=wo_cfg.capacity,
                satisfies=dict(wo_cfg.satisfies),
                formation=FormationSpec.from_config(wo_cfg.formation),
                interaction_radius=wo_cfg.interaction_radius,
            )
            self._world_knowledge.add_object(obj)
        self._event_scripts = list(scenario.event_scripts)
        self._event_scripts_by_tick = {}
        for script in self._event_scripts:
            self._event_scripts_by_tick.setdefault(script.tick, []).append(script)
        self._interaction_scripts = list(scenario.interaction_scripts)
        self._interaction_scripts_by_tick = {}
        for iscript in self._interaction_scripts:
            self._interaction_scripts_by_tick.setdefault(iscript.tick, []).append(iscript)
        self._init_flow(scenario)
        self._publish_world_geometry()

    def _init_flow(self, scenario: ScenarioConfig) -> None:
        flow = scenario.flow
        if flow is None or (not flow.sources and not flow.sinks):
            return
        from arena_humansim.utils.scenario import ShapeModel
        from arena_humansim.utils.types import AgentTemplate, RateKeyframe, Shape, ShapeType, SinkAffinity, SinkConfig, SourceConfig

        def _shape_from_cfg(cfg: ShapeModel) -> Shape:
            try:
                stype = ShapeType(cfg.type) if cfg.type else ShapeType.POLYGON
            except ValueError:
                stype = ShapeType.POLYGON
            return Shape(type=stype, radius=float(cfg.radius))

        sinks: dict[str, SinkConfig] = {}
        for i, sink_cfg in enumerate(flow.sinks):
            name = f"sink_{i}"
            sink = SinkConfig(
                name=name,
                pose=Pose2D(x=sink_cfg.pose.x, y=sink_cfg.pose.y, theta=sink_cfg.pose.theta),
                shape=_shape_from_cfg(sink_cfg.shape),
                absorption_radius=float(sink_cfg.absorption_radius),
                capacity=int(sink_cfg.capacity),
            )
            sinks[name] = sink
            self._despawn_monitor.add_sink(sink)
        if sinks:
            self._spawn_scheduler.set_sinks(sinks)

        for i, src_cfg in enumerate(flow.sources):
            tmpl = src_cfg.agent_template
            src = SourceConfig(
                name=f"source_{i}",
                pose=Pose2D(x=src_cfg.pose.x, y=src_cfg.pose.y, theta=src_cfg.pose.theta),
                shape=_shape_from_cfg(src_cfg.shape),
                type=SourceType(src_cfg.type),
                rate_profile=[RateKeyframe(t=float(kf.t), rate=float(kf.rate)) for kf in src_cfg.rate_profile],
                max_concurrent=int(src_cfg.max_concurrent),
                max_total=int(src_cfg.max_total),
                agent=AgentTemplate(
                    desired_velocity_min=float(tmpl.desired_velocity_min),
                    desired_velocity_max=float(tmpl.desired_velocity_max),
                    agent_radius=float(tmpl.agent_radius),
                    agent_type=tmpl.agent_type,
                    sink_affinity=[SinkAffinity(sink_name=f"sink_{sa.sink_idx}", weight=float(sa.weight)) for sa in tmpl.sink_affinity],
                ),
            )
            self._spawn_scheduler.add_source(src)

    def _build_base_agent(
        self,
        aid: int,
        agent_msg: AgentStateMsg,
        waypoints: list[Pose2D],
    ) -> BaseAgent:
        import attrs

        type_name = agent_msg.agent_type or "adult"

        state = AgentState(
            agent_id=aid,
            pose=Pose2D(
                x=agent_msg.pose.x,
                y=agent_msg.pose.y,
                theta=agent_msg.pose.theta,
            ),
            velocity=(agent_msg.velocity.x, agent_msg.velocity.y),
            desired_velocity=agent_msg.desired_velocity if agent_msg.desired_velocity > 0 else 1.3,
        )

        agent_type = resolve_agent_type_name(type_name, self._agent_types)
        if agent_type is not None:
            self._agent_types[agent_type.name] = agent_type
            rng = self._rng.get_agent_substream(aid, "params")
            planner_name = agent_type.local_planner or self._module_selections["local_planner"]
            if agent_type.local_planner:
                self._module_pool["local_planner"][planner_name] = self._policies[self._resolve_policy_idx(planner_name)]
            agent = create_agent(agent_type, state, self._module_pool, self._module_selections, rng, local_planner_means=self._local_planner_means(planner_name))
        else:
            planner_name = self._module_selections["local_planner"]
            lp_defaults = {k: v.mean for k, v in LocalPlanner.get_class(planner_name).PARAM_DEFAULTS.items()} | self._local_planner_means(planner_name)
            params = SampledParams(
                name=type_name,
                desired_velocity=state.desired_velocity,
                agent_radius=0.35,
                max_velocity=2.0,
                max_acceleration=1.5,
                max_deceleration=2.5,
                min_turning_radius=0.3,
                pivot_angular_velocity=2.0,
                reaction_time=0.4,
                personal_space_min=0.6,
                perception_stack=("default",),
                local_planner=planner_name,
                global_planner=self._module_selections["global_planner"],
                animation=self._module_selections["animation"],
                local_planner_params=lp_defaults,
            )
            agent = create_agent(params, state, self._module_pool, self._module_selections)

        overrides = {}
        if agent_msg.radius > 0.0:
            overrides["agent_radius"] = agent_msg.radius
        if agent_msg.handedness in ("l", "r"):
            overrides["handedness"] = agent_msg.handedness
        vel_val = agent_msg.desired_velocity
        if vel_val > 0.0:
            overrides["desired_velocity"] = vel_val
            if vel_val > agent.params.max_velocity:
                overrides["max_velocity"] = vel_val

        perception_overrides = {}
        for field_name in ("vision_range", "vision_fov"):
            val = getattr(agent_msg, field_name)
            if val > 0.0:
                perception_overrides[field_name] = val
        if perception_overrides:
            overrides["perception"] = attrs.evolve(
                agent.params.perception,
                **perception_overrides,
            )

        lp_overrides = {}
        for field_name in ("relaxation_time", "repulsion_strength", "repulsion_range"):
            val = getattr(agent_msg, field_name)
            if val > 0.0:
                lp_overrides[field_name] = val
        if lp_overrides:
            overrides["local_planner_params"] = {**agent.params.local_planner_params, **lp_overrides}
        self._pinned[aid] = _Pinned(
            local_planner=agent_type is not None and bool(agent_type.local_planner),
            global_planner=agent_type is not None and bool(agent_type.global_planner),
            lp_keys=frozenset(lp_overrides),
        )

        if overrides:
            agent.params = attrs.evolve(agent.params, **overrides)
            if "desired_velocity" in overrides:
                agent.state.desired_velocity = overrides["desired_velocity"]

        agent.movement = WaypointMovement(waypoints=waypoints)
        return agent

    def _spawn_policy_idx(self, aid: int, agent: BaseAgent) -> int:
        """Policy of a human spawned without an explicit one: the planner its type pins, else the default."""
        if not self._pinned[aid].local_planner or self._force_local_planner:
            return self._default_policy_idx
        self._pinned[aid] = attrs.evolve(self._pinned[aid], policy=True)
        assert agent.params.local_planner is not None
        return self._policy_name_to_idx[agent.params.local_planner]

    def _build_base_agent_from_spawn(
        self,
        aid: int,
        spawn_req: SpawnRequest,
    ) -> BaseAgent:
        agent_msg = AgentStateMsg()
        agent_msg.agent_id = aid
        agent_msg.pose = Pose2DMsg(
            x=spawn_req.pose.x,
            y=spawn_req.pose.y,
            theta=spawn_req.pose.theta,
        )
        agent_msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        agent_msg.desired_velocity = spawn_req.desired_velocity
        agent_msg.radius = spawn_req.agent_radius
        agent_msg.agent_type = spawn_req.agent_type
        return self._build_base_agent(aid, agent_msg, list(spawn_req.waypoints))

    def _lookup_agent_name(self, name: str, kind: int | None = None) -> int | None:
        """Exact name, else the unique agent whose name ends in ``/name``, optionally restricted to one AgentState kind."""
        pool = self._pool
        candidates = self._agent_name_to_id if kind is None else {n: a for n, a in self._agent_name_to_id.items() if a in pool._id_to_idx and int(pool.kind[pool.idx(a)]) == kind}
        aid = candidates.get(name)
        if aid is not None:
            return aid
        hits = [a for n, a in candidates.items() if n.rsplit("/", 1)[-1] == name]
        return hits[0] if len(hits) == 1 else None

    def _compile_behavior_tree(self, agent: BaseAgent) -> None:
        aid = agent.state.agent_id
        type_name = agent.params.name
        agent_type = resolve_agent_type_name(type_name, self._agent_types)
        if agent_type is None or agent_type.mode == "simple" or not agent_type.sequences:
            self._behavior_trees[aid] = None
            return

        if agent_type.source_path is not None:
            key = ("path", str(agent_type.source_path))
        else:
            key = ("content", id(agent_type))
        factory = self._bt_factories.get(key)
        if factory is None:
            factory = BehaviorTreeFactory(agent_type)
            self._bt_factories[key] = factory

        bt = factory.build(
            agent=agent,
            world=self._world_knowledge,
            event_bus=self._event_bus,
            rng=self._rng.get_agent_substream(aid, "behavior"),
            dt=self._dt * self._bt_tick_interval,
            agent_lookup=lambda aid: self._agents.get(aid),
            pool=self._pool,
            is_bound_lookup=self._interaction_manager.is_bound,
            im=self._interaction_manager,
            name_lookup=self._lookup_agent_name,
        )
        self._behavior_trees[aid] = bt
        if bt is not None:
            agent.movement = BehaviorTreeMovement()

    def _remove_agent(self, aid: int) -> None:
        self._mark_roster_dirty()
        self._agents.pop(aid, None)
        self._prone.discard(aid)
        self._parked_from.pop(aid, None)
        if aid in self._pool._id_to_idx:
            idx = self._pool._id_to_idx[aid]
            swapped_id = self._pool.swap_remove(aid)
            if swapped_id is not None:
                self._pool_agent_ids[idx] = swapped_id
            self._pool_agent_ids.pop()
        self._high_level_cmds.pop(aid, None)
        self._elevators.forget(aid)
        self._behavior_trees.pop(aid, None)
        self._pinned.pop(aid, None)
        self._robot_service_advertiser.unregister(aid)
        self._interaction_manager.force_stop(aid)
        self._event_bus.clear_agent(aid)
        self._rng.remove_agent_substreams(aid)
        for name, mapped_id in list(self._agent_name_to_id.items()):
            if mapped_id == aid:
                del self._agent_name_to_id[name]
        for ext_id, entity in list(self._external_entities.items()):
            if entity.agent_id == aid:
                del self._external_entities[ext_id]

    def _phase_end(self, name: str, t0: float):
        self._tick_phases[name] = (time.perf_counter() - t0) * 1000.0

    def _get_profile_callback(self, request: GetProfile.Request, response: GetProfile.Response) -> GetProfile.Response:
        for name, times in self._phase_accum.items():
            response.phase_names.append(name)  # pyright: ignore[reportAttributeAccessIssue]
            response.phase_means_ms.append(float(np.mean(times)))
            response.phase_p95s_ms.append(float(np.percentile(times, 95)))
        response.n_ticks = sum(len(t) for t in self._phase_accum.values()) // max(len(self._phase_accum), 1)
        response.n_agents = self._pool.n
        if request.reset:
            self._phase_accum.clear()
        return response

    def _flush_profile(self):
        if not self._phase_accum:
            return
        n_agents = self._pool.n
        parts = [f"tick profile ({n_agents} agents, {self._profile_interval} ticks):"]
        means = {name: float(np.mean(times)) for name, times in self._phase_accum.items()}
        total = sum(means.values())
        for name, times in self._phase_accum.items():
            mean = means[name]
            pct = mean / total * 100 if total > 0 else 0
            p95 = np.percentile(times, 95)
            parts.append(f"  {name:<16s} {pct:5.1f}%  mean={mean:.3f}ms  p95={p95:.3f}ms")
        parts.append(f"  {'TOTAL':<16s} 100.0%  mean={total:.3f}ms  budget={self._dt * 1000:.1f}ms  rtf={self._dt * 1000 / total:.2f}" if total > 0 else "")
        self._logger.info("\n".join(parts))
        self._phase_accum.clear()

    def tick(self):
        self._last_spawned_ids = []
        self._last_despawned_ids = []
        self._tick_phases = {}

        t0 = time.perf_counter()
        despawn_requests = self._despawn_monitor.tick(
            agents=self._agents,
            interaction_check=self._interaction_manager.is_in_interaction,
            tick_count=self._tick_count,
            dt=self._dt,
        )
        for req in despawn_requests:
            if req.force:
                self._interaction_manager.force_stop(req.agent_id)
            self._remove_agent(req.agent_id)
            self._spawn_scheduler.notify_despawn(req.agent_id)
            self._despawn_monitor.unregister(req.agent_id)
            self._last_despawned_ids.append(req.agent_id)
            if self._sim_logger is not None:
                self._sim_logger.record_agent_despawn(
                    agent_id=req.agent_id,
                    reason=req.reason,
                    tick=self._tick_count,
                )
        self._phase_end("despawn", t0)

        if self._pending_scenario_spawns and self._pending_scenario_spawns[0][0] <= self._tick_count:
            due = SpawnAgents.Request()
            while self._pending_scenario_spawns and self._pending_scenario_spawns[0][0] <= self._tick_count:
                due.agents.append(self._pending_scenario_spawns.popleft()[1])  # pyright: ignore[reportAttributeAccessIssue]
            resp = self._spawn_agents_callback(due, SpawnAgents.Response())
            if not resp.success:
                self._logger.error(f"deferred scenario spawn failed at tick {self._tick_count}: {resp.message}")

        while self._pending_stimuli and self._pending_stimuli[0][0] <= self._tick_count:
            _, agent_id, stimulus, intensity = self._pending_stimuli.popleft()
            agent = self._agents.get(agent_id)
            if agent is None:
                continue
            if agent.needs is not None:
                agent.needs.set(stimulus, 100.0 * intensity)
            self._event_bus.fire(stimulus, agent_id)

        t0 = time.perf_counter()
        spawn_requests = self._spawn_scheduler.tick(self._tick_count, self._dt)
        for spawn_req in spawn_requests:
            aid = spawn_req.agent_id if spawn_req.agent_id > 0 else self._allocate_agent_id()
            self._rng.bind_agent(aid, f"{spawn_req.lifetime.source_name}/{spawn_req.lifetime.source_seq}")
            agent = self._build_base_agent_from_spawn(aid, spawn_req)
            self._agents[aid] = agent
            idx = self._pool.add_agent(agent)
            self._pool.policy_idx[idx] = self._spawn_policy_idx(aid, agent)
            self._pool_agent_ids.append(aid)
            self._mark_roster_dirty()
            self._compile_behavior_tree(agent)
            spawn_req.lifetime.agent_id = aid
            self._despawn_monitor.register(aid, spawn_req.lifetime)
            self._spawn_scheduler.register_agent(aid, spawn_req.lifetime.source_name)
            self._last_spawned_ids.append(aid)

            mv = agent.movement
            if isinstance(mv, WaypointMovement) and mv.waypoints:
                self._high_level_cmds[aid] = HighLevelCommand(
                    agent_id=aid,
                    type=CommandType.NAVIGATE,
                    target_pose=mv.waypoints[mv.index],
                    desired_velocity=agent.state.desired_velocity,
                )

            if self._sim_logger is not None:
                self._sim_logger.record_agent_spawn(
                    agent_id=aid,
                    params=agent.params,
                    agent_type_name=agent.params.name,
                )
        self._phase_end("spawn", t0)

        world_state = self._consume_world_state()

        agents = [self._agents[aid] for aid in self._pool_agent_ids]
        pool = self._pool
        is_bt_tick = self._tick_count % self._bt_tick_interval == 0

        t0 = time.perf_counter()
        default_layer = next(iter(self._perception_cache.values()), None)
        if default_layer is not None and default_layer.supports_pool:
            default_layer.compute_pool(pool)

        if is_bt_tick or any(len(a.perception) > 1 for a in agents):
            indptr = pool.neighbor_indptr
            indices = pool.neighbor_indices
            any_extra = any(len(a.perception) > 1 for a in agents)
            agent_states = {aid: agent.state for aid, agent in self._agents.items()} if any_extra else {}
            for i, agent in enumerate(agents):
                belief = BeliefState(agent_id=agent.state.agent_id)
                nbr_idxs = indices[indptr[i] : indptr[i + 1]].tolist()
                belief.observed_agents = [agents[j].state for j in nbr_idxs]
                if len(agent.perception) > 1:
                    for layer in agent.perception[1:]:
                        belief = layer.compute(agent, agent_states, world_state, belief)
                agent.belief = belief

        self._run_extra_modules(agents, TickPhase.SENSE)
        self._phase_end("sense", t0)

        if is_bt_tick:
            t0 = time.perf_counter()
            pool.sync_back(agents)

            self._process_event_scripts()

            for agent_id in self._pool_agent_ids:
                bt = self._behavior_trees.get(agent_id)
                if bt is not None:
                    bt.tick()

            for agent_id, agent in self._agents.items():
                mv = agent.movement
                if agent.tunes.dirty:
                    agent.tunes.dirty = False
                    if agent_id in pool._id_to_idx:
                        pool.update_agent(agent)
                if isinstance(mv, BehaviorTreeMovement) and mv.command is not None:
                    self._high_level_cmds[agent_id] = mv.command

            self._event_bus.clear()
            self._phase_end("decide", t0)

            t0 = time.perf_counter()
            needs_subgoal_ids: set[int] = set()
            for i in range(pool.n):
                pidx = int(pool.policy_idx[i])
                planner = self._policies[pidx] if 0 <= pidx < len(self._policies) else None
                if planner is None or planner.needs_global_subgoal:
                    needs_subgoal_ids.add(int(pool.agent_ids[i]))
            subgoal_agents = [a for a in agents if a.state.agent_id in needs_subgoal_ids]
            steered = self._elevators.steer(subgoal_agents, self._high_level_cmds)
            for planner, group in _group_by(subgoal_agents, key=lambda a: a.global_planner):
                planner.compute(group, steered)

            self._cached_intermediate_goals = {}
            for planner, _group in _group_by(subgoal_agents, key=lambda a: a.global_planner):
                self._cached_intermediate_goals.update(planner.get_cached_goals())

            for aid in self._high_level_cmds:
                if aid not in needs_subgoal_ids:
                    cmd = self._high_level_cmds[aid]
                    if cmd.type != CommandType.NAVIGATE:
                        continue
                    self._cached_intermediate_goals[aid] = cmd.target_pose

            terminals: dict[int, Pose2D] = {}
            navigating: list[tuple[BaseAgent, Pose2D]] = []
            for aid, cmd in self._high_level_cmds.items():
                if cmd.type != CommandType.NAVIGATE:
                    continue
                agent = self._agents.get(aid)
                if agent is not None:
                    navigating.append((agent, cmd.target_pose))
                else:
                    terminals[aid] = cmd.target_pose
            for planner, group in _group_by(navigating, key=lambda item: item[0].global_planner):
                snapped = planner.snap_terminals([target for _, target in group])
                terminals.update((agent.state.agent_id, pose) for (agent, _), pose in zip(group, snapped, strict=True))
            pool.set_goals(self._cached_intermediate_goals)
            pool.set_terminals(terminals)
            self._apply_arrival_latch(pool)
            self._phase_end("global_plan", t0)

        t0 = time.perf_counter()
        self._locomotion.restore(pool)
        pool.store_prev_vel()
        n = pool.n
        active_mask = pool.policy_idx[:n] != -1
        if not np.any(active_mask):
            pool.vel[:n] = 0.0
        elif np.all(active_mask) and len(self._policies) == 1:
            planner = self._policies[0]
            if planner.supports_pool:
                planner.compute_pool(pool, store_forces=self._publish_markers >= 2, dt=self._dt)
            else:
                self._local_plan_fallback(agents, self._cached_intermediate_goals, pool)
        else:
            saved_has_goal = pool.has_goal[:n].copy()
            saved_vel = pool.vel[:n].copy()
            pool.has_goal[:n] = saved_has_goal & active_mask
            accum_vel = np.zeros_like(pool.vel[:n])
            for pidx in np.unique(pool.policy_idx[:n]):
                if pidx < 0:
                    continue
                planner = self._policies[int(pidx)]
                own_mask = pool.policy_idx[:n] == int(pidx)
                if planner.supports_pool:
                    pool.has_goal[:n] = saved_has_goal & active_mask & own_mask
                    pool.vel[:n] = saved_vel
                    planner.compute_pool(pool, store_forces=self._publish_markers >= 2, dt=self._dt)
                    accum_vel[own_mask] = pool.vel[:n][own_mask]
                else:
                    group = [agents[i] for i in range(n) if int(pool.policy_idx[i]) == int(pidx)]
                    vel_dict = planner.compute(group, self._cached_intermediate_goals, dt=self._dt)
                    for i in range(n):
                        if not own_mask[i]:
                            continue
                        v = vel_dict.get(int(pool.agent_ids[i]), (0.0, 0.0))
                        accum_vel[i, 0] = v[0]
                        accum_vel[i, 1] = v[1]
            pool.vel[:n] = accum_vel
            pool.has_goal[:n] = saved_has_goal
            pool.vel[:n][~active_mask] = 0.0

        self._run_extra_modules(agents, TickPhase.PLAN)
        self._phase_end("local_plan", t0)

        t0 = time.perf_counter()
        self._process_interaction_scripts()
        robot_service_cmds = self._robot_service_advertiser.emit(self._agents)
        formation_tick = (self._tick_count + 1) % self._bt_tick_interval == 0
        interactions, formation_targets, departed_agents = self._interaction_manager.update(
            self._high_level_cmds,
            dt=self._dt,
            extra_commands=robot_service_cmds,
            formations=formation_tick,
        )

        for aid, pose in formation_targets.items() if formation_tick else ():
            agent = self._agents.get(aid)
            if agent is None:
                continue
            self._high_level_cmds[aid] = HighLevelCommand(
                agent_id=aid,
                type=CommandType.NAVIGATE,
                target_pose=pose,
                desired_velocity=agent.state.desired_velocity,
            )
        self._apply_desired_speeds(agents, pool)
        self._elevators.hold(pool, self._tick_count, self._elevator_ride_ticks)
        heading_goals = {aid: pose.theta for aid, pose in formation_targets.items()}
        for aid, agent in self._agents.items():
            if aid in heading_goals:
                continue
            mv = agent.movement
            if isinstance(mv, BehaviorTreeMovement) and mv.heading_goal is not None:
                heading_goals[aid] = mv.heading_goal
        pool.set_heading_goals(heading_goals)

        for aid in departed_agents:
            agent = self._agents.get(aid)
            if agent is None:
                continue
            mv = agent.movement
            if isinstance(mv, WaypointMovement) and mv.waypoints:
                self._high_level_cmds[aid] = HighLevelCommand(
                    agent_id=aid,
                    type=CommandType.NAVIGATE,
                    target_pose=mv.waypoints[mv.index],
                    desired_velocity=agent.state.desired_velocity,
                )
            else:
                self._high_level_cmds.pop(aid, None)

        queue_lengths: dict[str, int] = {}
        participants_counts: dict[str, int] = {}
        for interaction in interactions.values():
            if interaction.outcome != InteractionOutcome.ACTIVE:
                for pid in (*interaction.participants, *interaction.contract.queue):
                    agent = self._agents.get(pid)
                    if agent is not None and isinstance(agent.movement, BehaviorTreeMovement):
                        agent.movement.last_outcome = interaction.outcome
            if interaction.object_id:
                queue_lengths[interaction.object_id] = queue_lengths.get(interaction.object_id, 0) + interaction.contract.queue_length
                participants_counts[interaction.object_id] = participants_counts.get(interaction.object_id, 0) + len(interaction.participants)
        self._world_knowledge.set_counts(queue_lengths, participants_counts)
        self._phase_end("interactions", t0)

        t0 = time.perf_counter()
        self._apply_arrival_damp(pool)
        self._apply_kinematic_constraints_vectorized(pool)
        self._locomotion.constrain(pool, self._dt, self._provides_heading_mask(pool))
        self._apply_external_velocity_pins()
        self._phase_end("kinematics", t0)

        t0 = time.perf_counter()
        for anim, _ in _group_by(agents, key=lambda a: a.animation):
            anim.compute_batch_pool(pool, interactions, self._dt)

        self._run_extra_modules(agents, TickPhase.ACT)
        self._phase_end("animation", t0)

        t0 = time.perf_counter()
        self._integrate_state_vectorized(pool)
        self._phase_end("integrate", t0)

        t0 = time.perf_counter()
        self._collision.resolve(pool)
        self._park_seated(pool)
        self._phase_end("collision", t0)

        t0 = time.perf_counter()
        self._advance_waypoints(agents, pool)
        self._apply_postures(agents, pool)
        self._update_animation_states()
        frame = self._build_agent_frame()
        self._agent_states_pub.publish(frame)
        if self._meta_dirty:
            self._publish_agent_meta(frame.header)
        if self._gestures_dirty or is_bt_tick:
            self._publish_agent_gestures(frame.header)
        edges = self._interaction_manager.drain_edges()
        table = self._interaction_manager.membership_table()
        if edges or self._interactions_had_edges or table != self._interactions_published:
            self._interactions_published = table
            self._interactions_had_edges = bool(edges)
            self._interactions_pub.publish(_interaction_frame(frame.header, table, edges))

        if self._marker_pub is not None and self._publish_markers > 0:
            pool.sync_back(agents)
            mlvl = self._publish_markers
            if self._viz_state_pub.get_subscription_count() > 0:
                self._viz_state_pub.publish(self._build_agent_viz(agents, interactions))
            publish_infrastructure(
                self._marker_pub,
                self._spawn_scheduler._sources,
                self._despawn_monitor._sinks,
                self._walls,
                self._world_knowledge._objects,
                self._obstacles,
            )
            if mlvl >= 2 and self._marker_pub.watched:
                modules: list[Perception | GlobalPlanner | LocalPlanner] = list(self._perception_cache.values())
                modules.append(self._global_planner)
                modules.append(self._local_planner)
                publish_module_markers(self._marker_pub, modules)
            self._marker_pub.flush()
        self._phase_end("publish", t0)

        if self._publish_markers == 0:
            pool.sync_back(agents)

        if self._sim_logger is not None:
            self._sim_logger.record_tick(
                tick=self._tick_count,
                timestamp=self._tick_count * self._dt,
                agents={aid: agent.state for aid, agent in self._agents.items()},
                interactions=interactions,
                commands=self._high_level_cmds,
            )

        self._tick_count += 1

        if self._profile_phases:
            for name, ms in self._tick_phases.items():
                self._phase_accum.setdefault(name, []).append(ms)
            if self._profile_interval > 0 and self._tick_count % self._profile_interval == 0:
                self._flush_profile()

    def _apply_arrival_latch(self, pool: AgentPool) -> None:
        arrival_latch_step(
            pool,
            r_enter=self._arrival_r_enter,
            r_exit=self._arrival_r_exit,
        )

    def _apply_desired_speeds(self, agents: list[BaseAgent], pool: AgentPool) -> None:
        pool.desired_vel[: pool.n] = [a.state.desired_velocity for a in agents]
        for aid, speed in self._interaction_manager.formation_speeds().items():
            idx = pool._id_to_idx.get(aid)
            if idx is not None:
                pool.desired_vel[idx] = speed

    def _apply_arrival_damp(self, pool: AgentPool) -> None:
        arrival_damp_step(pool, dt=self._dt, tau_brake=self._arrival_tau_brake)

    def _apply_kinematic_constraints_vectorized(self, pool: AgentPool) -> None:
        n = pool.n
        if n == 0:
            return
        dt = self._dt

        bypass = np.zeros(n, dtype=np.bool_)
        for pidx, planner in enumerate(self._policies):
            if getattr(planner, "bypasses_kinematic_constraints", False):
                bypass |= pool.policy_idx[:n] == pidx

        original_vel = pool.vel[:n].copy()
        new_vel = pool.vel[:n].copy()
        cur_vel = pool.prev_vel[:n]
        r_min = pool.min_turning_radius[:n]
        w_pivot = pool.pivot_angular_velocity[:n]
        max_acc = pool.max_acceleration[:n]
        max_dec = pool.max_deceleration[:n]
        max_v = pool.max_velocity[:n]

        cur_speed = np.linalg.norm(cur_vel, axis=1)
        moving = cur_speed > self._min_speed_for_heading

        max_ang = np.maximum(w_pivot, cur_speed / np.maximum(r_min, 1e-9))

        desired_angle = np.arctan2(new_vel[:, 1], new_vel[:, 0])
        current_angle = np.arctan2(cur_vel[:, 1], cur_vel[:, 0])
        angle_delta = np.arctan2(
            np.sin(desired_angle - current_angle),
            np.cos(desired_angle - current_angle),
        )
        max_angle = max_ang * dt
        angle_delta = np.clip(angle_delta, -max_angle, max_angle)
        clamped_angle = current_angle + angle_delta
        desired_speed = np.linalg.norm(new_vel, axis=1)
        clamped_vel = np.column_stack(
            [
                desired_speed * np.cos(clamped_angle),
                desired_speed * np.sin(clamped_angle),
            ]
        )
        new_vel = np.where(moving[:, None], clamped_vel, new_vel)

        # acceleration clamping
        accel = new_vel - cur_vel
        accel_mag = np.linalg.norm(accel, axis=1)

        dot_new_cur = np.sum(new_vel * cur_vel, axis=1)
        dot_cur_cur = np.sum(cur_vel * cur_vel, axis=1)
        is_decel = dot_new_cur < dot_cur_cur
        limit = np.where(is_decel, max_dec, max_acc)
        max_dv = limit * dt

        needs_clamp = (accel_mag > 1e-9) & (accel_mag > max_dv)
        scale = np.where(needs_clamp, max_dv / np.maximum(accel_mag, 1e-9), 1.0)
        new_vel = np.where(
            needs_clamp[:, None],
            cur_vel + accel * scale[:, None],
            new_vel,
        )

        # speed clamping
        speed = np.linalg.norm(new_vel, axis=1)
        too_fast = speed > max_v
        new_vel = np.where(
            too_fast[:, None],
            new_vel * (max_v / np.maximum(speed, 1e-9))[:, None],
            new_vel,
        )

        if bypass.any():
            new_vel[bypass] = original_vel[bypass]

        pool.vel[:n] = new_vel

    def _local_plan_fallback(
        self,
        agents: Iterable[BaseAgent],
        intermediate_goals: dict[int, Pose2D],
        pool: AgentPool,
    ) -> None:
        moving = []
        for agent in agents:
            aid = agent.state.agent_id
            goal = intermediate_goals.get(aid)
            if goal is None:
                continue
            dx = agent.state.pose.x - goal.x
            dy = agent.state.pose.y - goal.y
            if dx * dx + dy * dy < self._waypoint_threshold**2:
                continue
            moving.append(agent)
        vel_dict: dict[int, tuple[float, float]] = {}
        for planner, group in _group_by(moving, key=lambda a: a.local_planner):
            vel_dict.update(planner.compute(group, intermediate_goals, dt=self._dt))
        for i in range(pool.n):
            aid = int(pool.agent_ids[i])
            v = vel_dict.get(aid, (0.0, 0.0))
            pool.vel[i, 0] = v[0]
            pool.vel[i, 1] = v[1]

    def _provides_heading_mask(self, pool: AgentPool) -> np.ndarray:
        n = pool.n
        provides_heading = np.zeros(n, dtype=np.bool_)
        for pidx, planner in enumerate(self._policies):
            if planner.provides_heading:
                provides_heading |= pool.policy_idx[:n] == pidx
        return provides_heading

    def _integrate_state_vectorized(self, pool: AgentPool) -> None:
        n = pool.n
        if n == 0:
            return
        dt = self._dt
        vel = pool.vel[:n]
        pos = pool.pos[:n]
        theta = pool.theta[:n]

        speed = np.linalg.norm(vel, axis=1)
        moving = speed > self._min_speed_for_heading
        vel_theta = np.arctan2(vel[:, 1], vel[:, 0])

        self._locomotion.overlay(pool, dt)
        pos += vel * dt

        provides_heading = self._provides_heading_mask(pool)
        heading_owned = self._locomotion.heading_owned_mask(n)

        goal_theta = pool.goal_theta[:n]
        has_goal_theta = pool.has_goal_theta[:n]
        # an explicit heading goal wins over velocity heading, formation speeds rarely settle below min_speed_for_heading
        target_theta = np.where(has_goal_theta, goal_theta, vel_theta)
        rotating = (moving & ~provides_heading & ~heading_owned & ~has_goal_theta) | has_goal_theta
        delta = np.arctan2(
            np.sin(target_theta - theta),
            np.cos(target_theta - theta),
        )
        r_min = pool.min_turning_radius[:n]
        w_pivot = pool.pivot_angular_velocity[:n]
        max_d = np.maximum(w_pivot, speed / np.maximum(r_min, 1e-9)) * dt
        delta = np.clip(delta, -max_d, max_d)
        theta += np.where(rotating, delta, 0.0)

    def _advance_waypoints(self, agents: Iterable[BaseAgent], pool: AgentPool) -> None:
        for i, agent in enumerate(agents):
            mv = agent.movement
            if not isinstance(mv, WaypointMovement):
                continue
            wps = mv.waypoints
            if len(wps) < 2:
                continue
            goal = wps[mv.index]
            r = mv.radii[mv.index] if mv.radii else 0.0
            if r <= 0.0:
                r = self._waypoint_threshold
            dx = float(pool.pos[i, 0]) - goal.x
            dy = float(pool.pos[i, 1]) - goal.y
            dist_sq = dx * dx + dy * dy
            if dist_sq > r * r and not bool(pool.latched[i]):
                continue

            idx = mv.index
            n = len(wps)

            if mv.mode == WaypointMode.REPEAT:
                idx = (idx + 1) % n
            elif mv.mode == WaypointMode.REVERSE:
                if mv.forward:
                    if idx >= n - 1:
                        mv.forward = False
                        idx = max(idx - 1, 0)
                    else:
                        idx += 1
                else:
                    if idx <= 0:
                        mv.forward = True
                        idx = min(idx + 1, n - 1)
                    else:
                        idx -= 1
            elif mv.mode == WaypointMode.ONCE:
                if idx >= n - 1:
                    continue
                idx += 1
            elif mv.mode == WaypointMode.RANDOM:
                others = [j for j in range(n) if j != idx]
                idx = int(self._rng.get_agent_substream(agent.state.agent_id, "waypoints").choice(others))

            mv.index = idx
            aid = agent.state.agent_id
            self._high_level_cmds[aid] = HighLevelCommand(
                agent_id=aid,
                type=CommandType.NAVIGATE,
                target_pose=wps[idx],
                desired_velocity=float(pool.desired_vel[i]),
            )

    def _run_extra_modules(self, agents: Iterable[BaseAgent], phase: TickPhase):
        from arena_humansim.core.agents.base import VectorizedModule

        modules_agents: dict[Any, list[BaseAgent]] = {}
        for agent in agents:
            for mod in agent.modules.values():
                if mod.phase() == phase:
                    modules_agents.setdefault(mod, []).append(agent)
        for mod, group in modules_agents.items():
            if isinstance(mod, VectorizedModule):
                mod.step_pool(self._pool, self._pool.n, self._dt)
            else:
                mod.step_batch(group, self._dt)

    def _process_event_scripts(self) -> None:
        for script in self._event_scripts_by_tick.get(self._tick_count, ()):
            if script.target_agent == -1:
                self._event_bus.fire_broadcast(script.event)
            else:
                self._event_bus.fire(script.event, script.target_agent)

    def _process_interaction_scripts(self) -> None:
        """Force-create scripted interactions directly on the InteractionManager.

        Scripted interactions are scenario-authoring primitives, not runtime intent.
        They bypass the matcher and call `_create_interaction` / `accept` directly so
        every participant is seated atomically on the script's fire tick.
        """
        for script in self._interaction_scripts_by_tick.get(self._tick_count, ()):
            if not script.participants:
                continue
            itype = InteractionType[script.interaction_type]
            duration_s = script.duration_ticks * self._dt if script.duration_ticks > 0 else None
            object_id = script.metadata.get("object_id") if script.metadata else None
            creator = script.participants[0]
            if self._agents.get(creator) is None:
                continue
            spec = SeekSpec(
                interaction_type=itype,
                target=object_id,
                duration=duration_s,
            )
            interaction = self._interaction_manager._create_interaction(creator_id=creator, spec=spec)
            if duration_s is not None and duration_s > 0:
                interaction.member_durations[creator] = duration_s
            for pid in script.participants[1:]:
                if self._agents.get(pid) is None:
                    continue
                self._interaction_manager.accept(pid, interaction.id)
                if duration_s is not None and duration_s > 0:
                    interaction.member_durations[pid] = duration_s

    def run_replay(self) -> ReplayResult | None:
        if self._replay is None:
            return None
        result = self._replay.replay(self, logger=self._logger)
        if result.success:
            self._logger.info(f"Replay verification passed: {result.total_ticks} ticks")
        else:
            div = result.first_divergence
            assert div is not None
            self._logger.warn(f"Replay DIVERGED at tick {div.tick}, agent {div.agent_id}: {div.detail}")
        return result

    def _setup_master_mode(self):
        clock_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._clock_pub = self.create_publisher(Clock, "/clock", clock_qos)
        period = self._dt / self._rtf if self._rtf > 0.0 else 0.0
        self._timer = self.create_timer(period, self._master_timer_callback, clock=RclClock(clock_type=ClockType.STEADY_TIME))

    def _setup_subsystem_mode(self):
        # Subsystem mode: external orchestrator owns /clock. If tick wall-cost
        # exceeds clock cadence our state lags the clock downstream consumers see.
        # Policies:
        #   lag          - keep ticking every period the clock has covered; stamp
        #                  header with scheduled sim-time; emit overrun warnings.
        #                  Correct physics, possibly stale realtime. (only one implemented)
        #   skip         - drop ticks to stay current with /clock. Not implemented.
        #   backpressure - signal orchestrator to throttle /clock. Not implemented.
        clock_sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._clock_sub = self.create_subscription(Clock, "/clock", self._subsystem_timer_callback, clock_sub_qos)
        # Accumulated spawn/despawn IDs returned to callers via feedback
        self._accumulated_spawned: list[int] = []
        self._accumulated_despawned: list[int] = []
        # Service receives feedback (robot poses) from orchestrator
        self._feedback_srv = self.create_service(
            Feedback,
            "feedback",
            self._feedback_callback,
        )

    def _check_overrun(self, elapsed: float):
        if elapsed <= self._dt:
            return
        self._overrun_count += 1
        now = time.perf_counter()
        if now - self._last_overrun_log < 5.0:
            return
        self._last_overrun_log = now
        phases = self._tick_phases
        breakdown = ", ".join(f"{name}={ms:.1f}ms" for name, ms in phases.items() if ms > 0.5)
        self._logger.warn(f"sim cannot keep up: tick took {elapsed * 1000:.1f}ms (budget {self._dt * 1000:.1f}ms), {self._overrun_count} overrun(s) since last report, {self._pool.n} agents" + (f" [{breakdown}]" if breakdown else ""))
        self._overrun_count = 0

    def _robots_done(self) -> bool:
        pool = self._pool
        n = pool.n
        if n == 0:
            return False
        is_robot = pool.kind[:n] == KIND_ROBOT
        if not bool(np.any(is_robot)):
            return False
        ready = is_robot & pool.has_terminal[:n] & pool.latched[:n]
        if not bool(np.array_equal(ready, is_robot)):
            return False
        for i in np.flatnonzero(is_robot):
            agent = self._agents.get(int(pool.agent_ids[i]))
            if agent is None:
                return False
            mv = agent.movement
            if isinstance(mv, WaypointMovement):
                if mv.mode != WaypointMode.ONCE or mv.index != len(mv.waypoints) - 1:
                    return False
        return True

    def _master_timer_callback(self):
        if self._tick_count == 0:
            self._publish_world_geometry()
            self._tick_wall_start = time.perf_counter()
        t0 = time.perf_counter()
        self.tick()
        self._total_tick_compute_s += time.perf_counter() - t0
        self._sim_time_ns += int(self._dt * 1e9)
        clock_msg = Clock()
        clock_msg.clock.sec = int(self._sim_time_ns // int(1e9))
        clock_msg.clock.nanosec = int(self._sim_time_ns % int(1e9))
        self._clock_pub.publish(clock_msg)
        if self._ticks_limit > 0 and self._tick_count >= self._ticks_limit:
            if self._timer is not None:
                self._timer.cancel()
            self._logger.info(f"reached ticks={self._ticks_limit}, shutting down")
            rclpy.try_shutdown()
            return
        if self._robot_shutdown and self._robots_done():
            if self._timer is not None:
                self._timer.cancel()
            self._logger.info(f"all robots reached their goal at tick={self._tick_count}, shutting down")
            rclpy.try_shutdown()

    def _subsystem_timer_callback(self, msg: Clock):
        now_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec
        if self._tick_count == 0:
            self._publish_world_geometry()
            self._subsystem_epoch_ns = now_ns
        dt_ns = int(self._dt * 1e9)
        covered = (now_ns - self._subsystem_epoch_ns) // dt_ns + 1
        owed = covered - self._tick_count
        if owed <= 0:
            return
        t0 = time.perf_counter()
        for _ in range(owed):
            self._sim_time_ns = self._tick_count * dt_ns
            self.tick()
            self._accumulated_spawned.extend(self._last_spawned_ids)
            self._accumulated_despawned.extend(self._last_despawned_ids)
        self._check_overrun(time.perf_counter() - t0)

    def _feedback_callback(
        self,
        request: Feedback.Request,
        response: Feedback.Response,
    ) -> Feedback.Response:
        for robot in request.robots:
            name = robot.name or "robot"
            radius = robot.radius if robot.radius > 0 else 0.3
            self._teleport_robot(name, robot.pose.x, robot.pose.y, robot.pose.theta, radius)

        response.success = True
        response.spawned_ids = self._accumulated_spawned
        response.despawned_ids = self._accumulated_despawned
        self._accumulated_spawned = []
        self._accumulated_despawned = []
        self._logger.debug(f"feedback: {len(request.robots)} robots, spawned={len(response.spawned_ids)}, despawned={len(response.despawned_ids)}")
        return response

    def _teleport_robot(self, name: str, x: float, y: float, theta: float, radius: float) -> None:
        aid = self._agent_name_to_id.get(name)
        if aid is None:
            self._logger.debug(f"feedback: unknown robot name '{name}', skipping")
            return
        self._teleport_agent(aid, x, y, theta, radius)

    def _teleport_agent(self, aid: int, x: float, y: float, theta: float, radius: float) -> None:
        idx = self._pool._id_to_idx.get(aid)
        if idx is None:
            self._logger.debug(f"teleport: agent id={aid} not in pool, skipping")
            return
        self._pool.pos[idx, 0] = x
        self._pool.pos[idx, 1] = y
        self._pool.theta[idx] = theta
        self._pool.agent_radius[idx] = radius
        self._pool.vel[idx] = 0.0
        self._pool.prev_vel[idx] = 0.0

    def _world_state_callback(self, msg: AgentStatesMsg):
        self._latest_world_state = msg

    def _consume_world_state(self) -> WorldState:
        world: WorldState = {}
        if self._latest_world_state is not None:
            stamp = self._latest_world_state.header.stamp
            now = stamp.sec + stamp.nanosec * 1e-9
            for agent_msg in self._latest_world_state.agents:
                world[agent_msg.agent_id] = WorldAgentState(
                    pose=Pose2D(
                        x=agent_msg.pose.x,
                        y=agent_msg.pose.y,
                        theta=agent_msg.pose.theta,
                    ),
                    velocity=(agent_msg.velocity.x, agent_msg.velocity.y),
                )
                self._ingest_external_agent(agent_msg, now)
            self._latest_world_state = None
        self._expire_external_entities()
        self._apply_external_velocity_pins()
        return world

    def _ingest_external_agent(self, agent_msg: AgentStateMsg, stamp: float) -> None:
        """Teleport the pool entity bound to this world_state id, binding or spawning it on first sight."""
        entity = self._external_entities.get(agent_msg.agent_id)
        if entity is None:
            claimed = {e.agent_id for e in self._external_entities.values()}
            if agent_msg.agent_id in self._agents and agent_msg.agent_id not in claimed:
                entity = self._bind_internal_agent(agent_msg.agent_id)
            else:
                adopted = self._match_registered_robot(agent_msg)
                if adopted is not None:
                    entity = _ExternalEntity(agent_id=adopted, owned=False)
                else:
                    entity = _ExternalEntity(agent_id=self._spawn_external_agent(agent_msg))
            self._external_entities[agent_msg.agent_id] = entity
            self._mark_roster_dirty()
        if agent_msg.name and self._agent_name_to_id.get(agent_msg.name) != entity.agent_id:
            self._agent_name_to_id[agent_msg.name] = entity.agent_id
            self._meta_dirty = True
        entity.last_seen = self._tick_count
        radius = agent_msg.radius if agent_msg.radius > 0 else 0.3
        self._teleport_agent(entity.agent_id, agent_msg.pose.x, agent_msg.pose.y, agent_msg.pose.theta, radius)
        fed = (agent_msg.velocity.x, agent_msg.velocity.y)
        entity.vel = fed if entity.saved_policy_idx is not None or any(fed) else self._velocity_between(entity, agent_msg.pose, stamp)
        entity.pose = (agent_msg.pose.x, agent_msg.pose.y)
        entity.stamp = stamp

    def _velocity_between(self, entity: _ExternalEntity, pose: Pose2DMsg, stamp: float) -> tuple[float, float]:
        """Velocity from the entity's previous fed pose to this one, zero across a stale gap or a jump."""
        elapsed = stamp - entity.stamp
        if entity.pose is None or not 0.0 < elapsed <= _EXTERNAL_VELOCITY_WINDOW_S:
            return (0.0, 0.0)
        vx = (pose.x - entity.pose[0]) / elapsed
        vy = (pose.y - entity.pose[1]) / elapsed
        return (vx, vy) if math.hypot(vx, vy) <= _EXTERNAL_MAX_SPEED else (0.0, 0.0)

    def _match_registered_robot(self, agent_msg: AgentStateMsg) -> int | None:
        """Nearest unbound robot-kind agent within _EXTERNAL_ADOPT_RADIUS, None if no match."""
        pool = self._pool
        bound = {e.agent_id for e in self._external_entities.values()}
        best_aid: int | None = None
        best_d2 = _EXTERNAL_ADOPT_RADIUS**2
        for i in range(pool.n):
            if int(pool.kind[i]) != KIND_ROBOT:
                continue
            aid = int(pool.agent_ids[i])
            if aid in bound:
                continue
            dx = float(pool.pos[i, 0]) - agent_msg.pose.x
            dy = float(pool.pos[i, 1]) - agent_msg.pose.y
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best_aid = aid
        return best_aid

    def _bind_internal_agent(self, aid: int) -> _ExternalEntity:
        """Pin an engine-spawned agent to external control, saving its autonomy for restore on lease expiry."""
        idx = self._pool.idx(aid)
        saved = int(self._pool.policy_idx[idx])
        self._pool.policy_idx[idx] = -1
        return _ExternalEntity(agent_id=aid, owned=False, saved_policy_idx=saved)

    def _spawn_external_agent(self, agent_msg: AgentStateMsg) -> int:
        """Add a non-autonomous robot-kind pool entity mirroring an externally simulated body."""
        aid = self._allocate_robot_id()
        agent = self._build_base_agent(aid, agent_msg, [])
        agent.state.kind = KIND_ROBOT
        self._agents[aid] = agent
        idx = self._pool.add_agent(agent)
        self._pool.kind[idx] = KIND_ROBOT
        self._pool.interaction_class[idx] = interaction_classes.ROBOT
        self._pool_agent_ids.append(aid)
        self._mark_roster_dirty()
        self._behavior_trees[aid] = None
        return aid

    def _expire_external_entities(self) -> None:
        """Drop bindings not refreshed within _EXTERNAL_TIMEOUT_S, removing owned mirrors and releasing bound agents."""
        for ext_id, entity in list(self._external_entities.items()):
            if self._tick_count - entity.last_seen <= self._external_timeout_ticks:
                continue
            del self._external_entities[ext_id]
            self._mark_roster_dirty()
            if entity.owned:
                if entity.agent_id in self._agents:
                    self._remove_agent(entity.agent_id)
            elif entity.saved_policy_idx is not None:
                idx = self._pool._id_to_idx.get(entity.agent_id)
                if idx is not None:
                    self._pool.policy_idx[idx] = entity.saved_policy_idx

    def _apply_external_velocity_pins(self) -> None:
        """Reassert external velocities after planning, zeroing a mirror's once its feed is older than _EXTERNAL_VELOCITY_WINDOW_S."""
        pool = self._pool
        for entity in self._external_entities.values():
            if entity.saved_policy_idx is None and self._tick_count - entity.last_seen > self._external_velocity_ticks:
                entity.vel = (0.0, 0.0)
            idx = pool._id_to_idx.get(entity.agent_id)
            if idx is None:
                continue
            pool.vel[idx, 0] = entity.vel[0]
            pool.vel[idx, 1] = entity.vel[1]
            pool.prev_vel[idx] = pool.vel[idx]

    def _park_seated(self, pool: AgentPool) -> None:
        """Seated agents stay on their seat: furniture geometry and planning forces do not move them."""
        parked = self._interaction_manager.parked()
        for aid, seat in parked.items():
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                continue
            # the pose it stopped at is outside the furniture by construction, collision put it there
            self._parked_from.setdefault(aid, (float(pool.pos[idx, 0]), float(pool.pos[idx, 1]), float(pool.theta[idx])))
            pool.pos[idx, 0] = seat.x
            pool.pos[idx, 1] = seat.y
            pool.theta[idx] = seat.theta
            pool.vel[idx] = 0.0
            pool.prev_vel[idx] = 0.0

        for aid in [aid for aid in self._parked_from if aid not in parked]:
            x, y, theta = self._parked_from.pop(aid)
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                continue
            # released: the seat sits inside the object's collision box, which the resolver cannot push out of
            pool.pos[idx, 0] = x
            pool.pos[idx, 1] = y
            pool.theta[idx] = theta
            pool.vel[idx] = 0.0
            pool.prev_vel[idx] = 0.0

    def _apply_postures(self, agents: list[BaseAgent], pool: AgentPool) -> None:
        """A prone agent (lying on something, or collapsed by an authored posture) occupies a body-length disc."""
        imposed = self._interaction_manager.postures()
        for idx, agent in enumerate(agents):
            mv = agent.movement
            aid = agent.state.agent_id
            posture = mv.posture if isinstance(mv, BehaviorTreeMovement) and mv.posture else imposed.get(aid, "standing")
            if posture == "prone":
                self._prone.add(aid)
                pool.agent_radius[idx] = PRONE_RADIUS
            elif aid in self._prone:
                self._prone.discard(aid)
                pool.agent_radius[idx] = agent.params.agent_radius

    def _update_animation_states(self) -> None:
        pool = self._pool
        n = pool.n
        pool.animation_state[:n] = locomotion_states(pool.vel[:n], pool.has_goal[:n], pool.desired_vel[:n])

    def _mark_roster_dirty(self) -> None:
        self._meta_dirty = True
        self._gestures_dirty = True

    def _force_agent_meta(self) -> None:
        """Republish AgentMeta at the next tick even if its content is unchanged."""
        self._meta_published = None
        self._meta_dirty = True

    def _own_rows(self) -> np.ndarray:
        """Pool rows of the engine's own agents, external mirrors excluded."""
        n = self._pool.n
        external = [e.agent_id for e in self._external_entities.values() if e.saved_policy_idx is None]
        if not external:
            return np.arange(n)
        return np.flatnonzero(~np.isin(self._pool.agent_ids[:n], external))

    def _header(self) -> Header:
        if self._mode == self.MODE_SUBSYSTEM:
            # covered time on the clock epoch: the stamp advances only as ticks
            # actually complete, so consumers (and lockstep gates) see true
            # coverage instead of a fresh-looking stamp masking engine lag
            ns = self._subsystem_epoch_ns + self._sim_time_ns
        else:
            ns = self._sim_time_ns
        header = Header(frame_id="map")
        header.stamp.sec = int(ns // int(1e9))
        header.stamp.nanosec = int(ns % int(1e9))
        return header

    def _build_agent_frame(self) -> AgentFrameMsg:
        pool = self._pool
        rows = self._own_rows()
        pos = pool.pos[rows]
        vel = pool.vel[rows]
        pidx = pool.policy_idx[rows]
        msg = AgentFrameMsg(header=self._header())
        msg.agent_id = _flat("i", pool.agent_ids[rows])
        msg.x = _flat("d", pos[:, 0])
        msg.y = _flat("d", pos[:, 1])
        msg.theta = _flat("d", pool.theta[rows])
        msg.vx = _flat("d", vel[:, 0])
        msg.vy = _flat("d", vel[:, 1])
        msg.desired_velocity = _flat("d", pool.desired_vel[rows])
        msg.radius = _flat("d", pool.agent_radius[rows])
        msg.kind = _flat("B", pool.kind[rows])
        msg.animation_state = _flat("B", pool.animation_state[rows])
        msg.policy_idx = _flat("h", np.where((pidx >= 0) & (pidx < len(self._policy_names)), pidx, -1))
        gait_phase, gait_cadence = self._locomotion.frame_arrays(rows)
        msg.gait_phase = _flat("d", gait_phase)
        msg.gait_cadence = _flat("d", gait_cadence)
        im = self._interaction_manager
        active = [im.active_interaction(aid) for aid in pool.agent_ids[rows].tolist()] if self._gesture_mode == GESTURE_ENABLED else [None] * len(rows)
        msg.interaction_id = array.array("i", [a[0] if a is not None else -1 for a in active])
        msg.interaction_type = array.array("B", [int(a[1]) if a is not None else 0 for a in active])
        heights = [im.support_height(aid) if a is not None else None for aid, a in zip(pool.agent_ids[rows].tolist(), active, strict=True)]
        msg.support_height = array.array("f", [-1.0 if h is None else h for h in heights])
        return msg

    def _publish_agent_meta(self, header: Header) -> None:
        self._meta_dirty = False
        ids = self._pool.agent_ids[self._own_rows()].tolist()
        names: dict[int, str] = {}
        for name, aid in self._agent_name_to_id.items():
            names.setdefault(aid, name)
        agents = [self._agents.get(aid) for aid in ids]
        lifetimes = [self._despawn_monitor.lifetime(aid) for aid in ids]
        content = (
            tuple(self._policy_names),
            tuple(ids),
            tuple(names.get(aid, "") for aid in ids),
            tuple(agent.params.handedness if agent is not None else "" for agent in agents),
            tuple(agent.params.name if agent is not None else "" for agent in agents),
            tuple(lifetime.source_name if lifetime is not None else "" for lifetime in lifetimes),
            tuple(lifetime.source_seq if lifetime is not None else 0 for lifetime in lifetimes),
        )
        if content == self._meta_published:
            return
        self._meta_published = content
        policies, _, agent_names, handedness, agent_types, source_names, source_seqs = content
        msg = AgentMetaMsg(header=header)
        msg.policies = list(policies)
        msg.agent_id = array.array("i", ids)
        msg.name = list(agent_names)
        msg.handedness = list(handedness)
        msg.agent_type = list(agent_types)
        msg.source_name = list(source_names)
        msg.source_seq = array.array("i", source_seqs)
        self._agent_meta_pub.publish(msg)

    def _publish_agent_gestures(self, header: Header) -> None:
        self._gestures_dirty = False
        owned: list[tuple[int, GestureIntent]] = []
        for aid in self._pool.agent_ids[self._own_rows()].tolist():
            agent = self._agents.get(aid)
            mv = agent.movement if agent is not None else None
            if isinstance(mv, BehaviorTreeMovement) and self._gesture_mode == GESTURE_ENABLED:
                owned.extend((aid, g) for g in mv.gestures)
        if owned == self._gestures_published:
            return
        self._gestures_published = owned
        msg = AgentGesturesMsg(header=header)
        msg.agent_id = array.array("i", [aid for aid, _ in owned])
        msg.gestures = [_gesture_msg(g) for _, g in owned]
        self._agent_gestures_pub.publish(msg)

    def _build_agent_viz(self, agents: list[BaseAgent], interactions: dict[int, InteractionState]) -> AgentVizMsg:
        """Marker inputs for every pool agent, reads agent states synced back from the pool."""
        pool = self._pool
        n = pool.n
        full = self._publish_markers >= 2
        cmds = self._high_level_cmds
        msg = AgentVizMsg(header=self._header(), level=self._publish_markers)
        msg.agent_id = _flat("i", pool.agent_ids[:n])
        msg.x = _flat("d", pool.pos[:n, 0])
        msg.y = _flat("d", pool.pos[:n, 1])
        msg.theta = _flat("d", pool.theta[:n])
        msg.vx = _flat("d", pool.vel[:n, 0])
        msg.vy = _flat("d", pool.vel[:n, 1])
        msg.radius = array.array("d", [agent.params.agent_radius for agent in agents])
        msg.kind = _flat("B", pool.kind[:n])

        in_interaction = {pid for inter in interactions.values() for pid in inter.participants}
        cmd_ids: list[int] = []
        cmd_idx: list[int] = []
        need_names: dict[str, int] = {}
        need_ids: list[int] = []
        need_name_idx: list[int] = []
        need_slot: list[int] = []
        need_value: list[float] = []
        obs_ids: list[int] = []
        obs_slot: list[int] = []
        obs_x: list[float] = []
        obs_y: list[float] = []
        path_ids: list[int] = []
        path_offset = [0]
        path_x: list[float] = []
        path_y: list[float] = []
        igoal_ids: list[int] = []
        igoal_x: list[float] = []
        igoal_y: list[float] = []
        goal_ids: list[int] = []
        goal_x: list[float] = []
        goal_y: list[float] = []
        goal_theta: list[float] = []
        wp_ids: list[int] = []
        wp_offset = [0]
        wp_x: list[float] = []
        wp_y: list[float] = []
        wp_active: list[int] = []
        wp_active_radius: list[float] = []

        cached_paths: dict[int, list[Pose2D]] = {}
        if full:
            seen: set[int] = set()
            for agent in agents:
                pid = id(agent.global_planner)
                if pid not in seen:
                    seen.add(pid)
                    cached_paths.update(agent.global_planner.get_cached_paths())

        for agent in agents:
            aid = agent.state.agent_id
            cmd = cmds.get(aid)
            if cmd is not None:
                cmd_ids.append(aid)
                cmd_idx.append(_INTR_LABEL_IDX if aid in in_interaction and cmd.type == CommandType.NAVIGATE else _CMD_LABEL_IDX[CommandType(cmd.type)])
            if agent.needs is not None:
                for i, (name, need) in enumerate(agent.needs.needs.items()):
                    need_ids.append(aid)
                    need_name_idx.append(need_names.setdefault(name, len(need_names)))
                    need_slot.append(i)
                    need_value.append(need.value)
            if not full:
                continue
            if agent.belief is not None:
                for i, obs in enumerate(agent.belief.observed_agents):
                    obs_ids.append(aid)
                    obs_slot.append(i)
                    obs_x.append(obs.pose.x)
                    obs_y.append(obs.pose.y)
            path = cached_paths.get(aid)
            if path and len(path) > 1:
                path_ids.append(aid)
                path_x.extend(wp.x for wp in path)
                path_y.extend(wp.y for wp in path)
                path_offset.append(len(path_x))
            ig = self._cached_intermediate_goals.get(aid)
            if ig is not None:
                igoal_ids.append(aid)
                igoal_x.append(ig.x)
                igoal_y.append(ig.y)
            if cmd is not None:
                goal_ids.append(aid)
                goal_x.append(cmd.target_pose.x)
                goal_y.append(cmd.target_pose.y)
                goal_theta.append(cmd.target_pose.theta)
            mv = agent.movement
            if isinstance(mv, WaypointMovement) and mv.waypoints:
                wp_ids.append(aid)
                wp_x.extend(wp.x for wp in mv.waypoints)
                wp_y.extend(wp.y for wp in mv.waypoints)
                wp_offset.append(len(wp_x))
                wp_active.append(mv.index)
                r = mv.radii[mv.index] if mv.radii and mv.index < len(mv.radii) else 0.3
                wp_active_radius.append(max(r, 0.0))

        msg.cmd_labels = list(_CMD_LABELS)
        msg.cmd_agent_id = array.array("i", cmd_ids)
        msg.cmd_label_idx = array.array("B", cmd_idx)
        msg.need_names = list(need_names)
        msg.need_agent_id = array.array("i", need_ids)
        msg.need_name_idx = array.array("B", need_name_idx)
        msg.need_slot = array.array("B", need_slot)
        msg.need_value = array.array("f", need_value)

        inter_ids: list[int] = []
        inter_labels: list[str] = []
        inter_offset = [0]
        inter_parts: list[int] = []
        for iid, inter in interactions.items():
            parts = inter.participants
            lbl = f"{InteractionType(inter.type).kind.label} [{len(parts)}p"
            if inter.contract.queue:
                lbl += f" +{len(inter.contract.queue)}q"
            lbl += "]"
            inter_ids.append(iid)
            inter_labels.append(lbl)
            inter_parts.extend(parts)
            inter_offset.append(len(inter_parts))
        msg.interaction_id = array.array("i", inter_ids)
        msg.interaction_label = inter_labels
        msg.interaction_offset = array.array("I", inter_offset)
        msg.interaction_participants = array.array("i", inter_parts)

        if not full:
            return msg
        msg.vision_range = _flat("d", pool.vision_range[:n])
        msg.vision_fov = _flat("d", pool.vision_fov[:n])
        msg.proximity_sense = _flat("d", pool.proximity_sense[:n])
        msg.observed_agent_id = array.array("i", obs_ids)
        msg.observed_slot = array.array("H", obs_slot)
        msg.observed_x = array.array("d", obs_x)
        msg.observed_y = array.array("d", obs_y)
        msg.path_agent_id = array.array("i", path_ids)
        msg.path_offset = array.array("I", path_offset)
        msg.path_x = array.array("d", path_x)
        msg.path_y = array.array("d", path_y)
        msg.igoal_agent_id = array.array("i", igoal_ids)
        msg.igoal_x = array.array("d", igoal_x)
        msg.igoal_y = array.array("d", igoal_y)
        msg.goal_agent_id = array.array("i", goal_ids)
        msg.goal_x = array.array("d", goal_x)
        msg.goal_y = array.array("d", goal_y)
        msg.goal_theta = array.array("d", goal_theta)
        msg.wp_agent_id = array.array("i", wp_ids)
        msg.wp_offset = array.array("I", wp_offset)
        msg.wp_x = array.array("d", wp_x)
        msg.wp_y = array.array("d", wp_y)
        msg.wp_active = array.array("I", wp_active)
        msg.wp_active_radius = array.array("d", wp_active_radius)
        return msg

    def _on_contact_params(self, params: list[Parameter]) -> SetParametersResult:
        """Validate and apply contact_mode / locomotion_standing_distance / gesture_mode, other params pass through."""
        mode = self._interaction_manager.contact_mode
        distance: float | None = None
        gesture_mode = self._gesture_mode
        for p in params:
            if p.name == "contact_mode":
                mode = str(p.value)
            elif p.name == "locomotion_standing_distance":
                if not isinstance(p.value, (int, float)):
                    return SetParametersResult(successful=False, reason=f"locomotion_standing_distance must be a number, got {p.value!r}")
                distance = float(p.value)
            elif p.name == "gesture_mode":
                gesture_mode = str(p.value)
        try:
            check_gesture_mode(gesture_mode)
            self._interaction_manager.set_contact_mode(mode, distance)
        except ValueError as e:
            return SetParametersResult(successful=False, reason=str(e))
        if gesture_mode != self._gesture_mode:
            self._gesture_mode = gesture_mode
            self._gestures_dirty = True
        return SetParametersResult(successful=True)

    def _spawn_agents_callback(
        self,
        request: SpawnAgents.Request,
        response: SpawnAgents.Response,
    ) -> SpawnAgents.Response:
        spawned_ids = []
        reserved = [m.name for m in request.agents if m.name in ATTENTION_KEYWORDS]
        bad_hands = [m.handedness for m in request.agents if m.handedness not in ("", "l", "r")]
        if reserved or bad_hands:
            response.success = False
            response.message = f"agent names {reserved} are reserved attention keywords {list(ATTENTION_KEYWORDS)}" if reserved else f"handedness must be 'l' or 'r', got {bad_hands}"
            self._logger.error(response.message)
            return response
        for agent_msg in request.agents:
            kind = int(agent_msg.kind)
            aid = agent_msg.agent_id
            if aid == 0:
                aid = self._allocate_robot_id() if kind == KIND_ROBOT else self._allocate_agent_id()

            waypoints = [Pose2D(x=pt.pose.x, y=pt.pose.y, theta=pt.pose.theta) for pt in agent_msg.waypoints.points]
            radii = [pt.radius for pt in agent_msg.waypoints.points]

            policy_name = agent_msg.policy
            policy_params = agent_msg.policy_params
            if self._force_local_planner:
                policy_idx = self._default_policy_idx
            elif policy_name:
                policy_idx = self._resolve_policy_idx(policy_name)
            else:
                policy_idx = self._default_policy_idx if kind == 0 else -1
            if policy_params and 0 <= policy_idx < len(self._policies):
                planner = self._policies[policy_idx]
                apply = getattr(planner, "apply_policy_params", None)
                if callable(apply):
                    apply(policy_params)

            agent = self._build_base_agent(aid, agent_msg, waypoints)
            if policy_name and not self._force_local_planner:
                self._pinned[aid] = attrs.evolve(self._pinned[aid], policy=True)
            elif kind == 0:
                policy_idx = self._spawn_policy_idx(aid, agent)
            agent.state.kind = kind
            agent.movement = WaypointMovement(
                waypoints=waypoints,
                radii=radii,
                mode=WaypointMode(agent_msg.waypoints.mode),
            )
            self._agents[aid] = agent
            idx = self._pool.add_agent(agent)
            self._pool.kind[idx] = kind
            if kind == KIND_ROBOT:
                self._pool.interaction_class[idx] = interaction_classes.ROBOT
            self._pool.policy_idx[idx] = policy_idx
            self._pool_agent_ids.append(aid)
            self._mark_roster_dirty()

            if agent_msg.name:
                self._agent_name_to_id[agent_msg.name] = aid
            if kind != 0:
                self._agent_name_to_id.setdefault(agent.params.name or f"robot_{aid}", aid)
            if kind == 0:
                self._compile_behavior_tree(agent)
            else:
                self._behavior_trees[aid] = None

            mv = agent.movement
            if isinstance(mv, WaypointMovement) and mv.waypoints:
                self._high_level_cmds[aid] = HighLevelCommand(
                    agent_id=aid,
                    type=CommandType.NAVIGATE,
                    target_pose=mv.waypoints[mv.index],
                    desired_velocity=agent.state.desired_velocity,
                )

            if self._sim_logger is not None:
                self._sim_logger.record_agent_spawn(
                    agent_id=aid,
                    params=agent.params,
                    agent_type_name=agent.params.name,
                )

            spawned_ids.append(aid)
            if aid >= self._next_agent_id:
                self._next_agent_id = aid + 1
            if aid <= self._next_robot_id:
                self._next_robot_id = aid - 1
        response.success = True
        response.message = f"Spawned {len(spawned_ids)} agent(s)"
        response.spawned_ids = spawned_ids
        self._logger.info(response.message)
        return response

    def _remove_agents_callback(
        self,
        request: RemoveAgents.Request,
        response: RemoveAgents.Response,
    ) -> RemoveAgents.Response:
        if not request.agent_ids:
            count = len(self._agents)
            self._agents.clear()
            self._pool_agent_ids.clear()
            self._pool.reset()
            self._force_agent_meta()
            self._mark_roster_dirty()
            self._high_level_cmds.clear()
            self._elevators.clear_rides()
            self._behavior_trees.clear()
            self._agent_name_to_id.clear()
            self._external_entities.clear()
            self._prone.clear()
            self._parked_from.clear()
            self._despawn_monitor.clear()
            self._spawn_scheduler.reset_counts()
            self._interaction_manager.reset()
            self._event_bus.clear()
            self._pending_stimuli.clear()
            self._next_agent_id = 1
            self._next_robot_id = -1
            self._tick_count = 0
            response.success = True
            response.message = f"Removed all {count} agent(s)"
        else:
            removed = 0
            for aid in request.agent_ids:
                if aid in self._agents:
                    self._remove_agent(aid)
                    self._despawn_monitor.unregister(aid)
                    self._spawn_scheduler.notify_despawn(aid)
                    removed += 1
            response.success = True
            response.message = f"Removed {removed} agent(s)"
        self._logger.info(response.message)
        return response

    def _reset_callback(
        self,
        request: ResetSimulation.Request,
        response: ResetSimulation.Response,
    ) -> ResetSimulation.Response:
        if not request.soft:
            self._wipe()
        self._apply_parameters()
        self._force_agent_meta()
        response.success = True
        response.message = "Parameters applied" if request.soft else "Simulation reset"
        self._logger.info(response.message)
        return response

    def _wipe(self) -> None:
        self._agents.clear()
        self._pool_agent_ids.clear()
        self._pool.reset()
        self._mark_roster_dirty()
        self._high_level_cmds.clear()
        self._behavior_trees.clear()
        self._agent_name_to_id.clear()
        self._external_entities.clear()
        self._prone.clear()
        self._parked_from.clear()
        self._despawn_monitor.clear()
        self._spawn_scheduler.reset_counts()
        self._spawn_scheduler.clear_sources()
        self._despawn_monitor.clear_sinks()
        self._interaction_manager.reset()
        self._event_bus.clear()
        self._pending_stimuli.clear()
        self._event_scripts.clear()
        self._event_scripts_by_tick.clear()
        self._interaction_scripts.clear()
        self._interaction_scripts_by_tick.clear()
        self._walls.clear()
        self._obstacles.clear()
        self._elevators.set_cabins([], [])
        for subsystem in self._wall_aware:
            subsystem.set_walls([])
        self._world_knowledge.clear()
        self._rng.reset()
        self._next_agent_id = 1
        self._next_robot_id = -1
        self._tick_count = 0
        self._sim_time_ns = 0
        self._last_spawned_ids = []
        self._last_despawned_ids = []
        self._pinned.clear()
        self._publish_world_geometry()

    def _update_robot_callback(
        self,
        request: UpdateRobot.Request,
        response: UpdateRobot.Response,
    ) -> UpdateRobot.Response:
        name = request.name or "robot"
        radius = request.radius if request.radius > 0 else 0.3
        self._teleport_robot(name, request.pose.x, request.pose.y, request.pose.theta, radius)
        response.success = True
        self._logger.debug(f"update_robot: {name} at ({request.pose.x:.2f}, {request.pose.y:.2f}), r={radius:.2f}")
        return response

    def _set_waypoints_callback(
        self,
        request: SetWaypoints.Request,
        response: SetWaypoints.Response,
    ) -> SetWaypoints.Response:
        aid = int(request.agent_id) if request.agent_id != 0 else 0
        if aid == 0:
            aid = self._agent_name_to_id.get(request.name, 0)
        agent = self._agents.get(aid) if aid else None
        if agent is None:
            response.success = False
            response.message = f"unknown agent (id={request.agent_id}, name='{request.name}')"
            return response

        waypoints = [Pose2D(x=pt.pose.x, y=pt.pose.y, theta=pt.pose.theta) for pt in request.waypoints.points]
        radii = [pt.radius for pt in request.waypoints.points]
        mode = WaypointMode(request.waypoints.mode)

        mv = agent.movement
        if isinstance(mv, WaypointMovement):
            mv.waypoints = waypoints
            mv.radii = radii
            mv.mode = mode
            mv.index = 0
            mv.forward = True
        else:
            bt = self._behavior_trees.get(aid)
            if bt is not None:
                bt.root.stop(Status.INVALID)
                self._behavior_trees[aid] = None
            mv = WaypointMovement(waypoints=waypoints, radii=radii, mode=mode)
            agent.movement = mv
            self._gestures_dirty = True

        idx = self._pool._id_to_idx.get(aid)
        if waypoints:
            self._high_level_cmds[aid] = HighLevelCommand(
                agent_id=aid,
                type=CommandType.NAVIGATE,
                target_pose=mv.waypoints[mv.index],
                desired_velocity=float(self._pool.desired_vel[idx]) if idx is not None else agent.state.desired_velocity,
            )
            if idx is not None:
                self._pool.goal_pos[idx, 0] = mv.waypoints[mv.index].x
                self._pool.goal_pos[idx, 1] = mv.waypoints[mv.index].y
                self._pool.has_goal[idx] = True
        else:
            self._high_level_cmds.pop(aid, None)
            if idx is not None:
                self._pool.has_goal[idx] = False

        response.success = True
        response.message = f"set {len(waypoints)} waypoint(s) for agent {aid}"
        return response

    @staticmethod
    def _shape_msg_to_shape(shape_msg: ShapeMsg) -> Shape:
        from arena_humansim.utils.types import ShapeType

        _RECT = 0
        _shape_type_map = {1: ShapeType.CIRCLE, 2: ShapeType.POLYGON}
        stype = _shape_type_map.get(shape_msg.type, ShapeType.POLYGON)
        vertices = [Pose2D(x=v.x, y=v.y) for v in shape_msg.vertices]
        if shape_msg.type == _RECT and shape_msg.width > 0 and shape_msg.height > 0:
            hw, hh = shape_msg.width / 2.0, shape_msg.height / 2.0
            vertices = [
                Pose2D(x=-hw, y=-hh),
                Pose2D(x=hw, y=-hh),
                Pose2D(x=hw, y=hh),
                Pose2D(x=-hw, y=hh),
            ]
        return Shape(type=stype, radius=shape_msg.radius, vertices=vertices)

    @staticmethod
    def _source_type_from_msg(src_msg: SourceConfigMsg) -> SourceType:
        types = {SourceConfigMsg.POISSON: SourceType.POISSON, SourceConfigMsg.MAX: SourceType.MAX}
        if src_msg.type not in types:
            raise ValueError(f"source {src_msg.name!r}: unknown type {src_msg.type}")
        return types[src_msg.type]

    @staticmethod
    def _source_msg_to_config(src_msg: SourceConfigMsg) -> SourceConfig:
        from arena_humansim.utils.types import AgentTemplate, RateKeyframe, SinkAffinity

        return SourceConfig(
            name=src_msg.name,
            pose=Pose2D(x=src_msg.pose.x, y=src_msg.pose.y, theta=src_msg.pose.theta),
            shape=AgentManager._shape_msg_to_shape(src_msg.shape),
            type=AgentManager._source_type_from_msg(src_msg),
            rate_profile=[RateKeyframe(t=kf.t, rate=kf.rate) for kf in src_msg.rate_profile],
            max_concurrent=src_msg.max_concurrent,
            max_total=src_msg.max_total,
            agent=AgentTemplate(
                desired_velocity_min=src_msg.agent.desired_velocity_min,
                desired_velocity_max=src_msg.agent.desired_velocity_max,
                agent_radius=src_msg.agent.agent_radius,
                agent_type=src_msg.agent.agent_type,
                sink_affinity=[SinkAffinity(sink_name=sa.sink_name, weight=sa.weight) for sa in src_msg.agent.sink_affinity],
            ),
        )

    @staticmethod
    def _sink_msg_to_config(sink_msg: SinkConfigMsg) -> SinkConfig:
        return SinkConfig(
            name=sink_msg.name,
            pose=Pose2D(x=sink_msg.pose.x, y=sink_msg.pose.y, theta=sink_msg.pose.theta),
            shape=AgentManager._shape_msg_to_shape(sink_msg.shape),
            absorption_radius=sink_msg.absorption_radius,
            capacity=sink_msg.capacity,
        )

    def _set_flow_callback(self, request: SetFlow.Request, response: SetFlow.Response) -> SetFlow.Response:
        try:
            sources = [self._source_msg_to_config(src_msg) for src_msg in request.flow.sources]
            for src in sources:
                self._spawn_scheduler.validate(src)
        except ValueError as e:
            response.success = False
            response.message = str(e)
            self._logger.error(response.message)
            return response

        self._spawn_scheduler.clear_sources()
        self._despawn_monitor.clear_sinks()

        for src in sources:
            self._spawn_scheduler.add_source(src)

        sinks = {}
        for sink_msg in request.flow.sinks:
            sink = self._sink_msg_to_config(sink_msg)
            sinks[sink.name] = sink
            self._despawn_monitor.add_sink(sink)

        self._spawn_scheduler.set_sinks(sinks)

        response.success = True
        response.message = f"Set {len(sources)} source(s), {len(sinks)} sink(s)"
        self._logger.info(response.message)
        return response

    def _notify_stimulus_callback(self, request: NotifyStimulus.Request, response: NotifyStimulus.Response) -> NotifyStimulus.Response:
        if request.agent_id == -1:
            targets = list(self._agents.keys())
        elif request.agent_id in self._agents:
            targets = [request.agent_id]
        else:
            response.success = False
            response.message = f"unknown agent id {request.agent_id}"
            return response

        for aid in targets:
            agent = self._agents[aid]
            due_tick = self._tick_count + int(round(agent.params.reaction_time / self._dt))
            self._pending_stimuli.append((due_tick, aid, request.stimulus, float(request.intensity)))
        self._pending_stimuli = deque(sorted(self._pending_stimuli, key=lambda entry: entry[0]))

        response.success = True
        response.message = f"queued {request.stimulus} for {len(targets)} agent(s)"
        self._logger.info(response.message)
        return response

    def _add_source_callback(self, request: AddSource.Request, response: AddSource.Response) -> AddSource.Response:
        try:
            src = self._source_msg_to_config(request.source)
            self._spawn_scheduler.add_source(src)
        except ValueError as e:
            response.success = False
            response.message = str(e)
            self._logger.error(response.message)
            return response
        response.success = True
        response.message = f"Added source {src.name}"
        response.name = src.name
        self._logger.debug(response.message)
        return response

    def _remove_source_callback(self, request: RemoveSource.Request, response: RemoveSource.Response) -> RemoveSource.Response:
        if not request.name:
            self._spawn_scheduler.clear_sources()
            response.message = "Removed all sources"
        else:
            self._spawn_scheduler.remove_source(request.name)
            response.message = f"Removed source {request.name}"
        response.success = True
        self._logger.debug(response.message)
        return response

    def _add_sink_callback(self, request: AddSink.Request, response: AddSink.Response) -> AddSink.Response:
        sink = self._sink_msg_to_config(request.sink)
        self._despawn_monitor.add_sink(sink)
        self._spawn_scheduler.set_sinks(self._despawn_monitor.sinks)
        response.success = True
        response.message = f"Added sink {sink.name}"
        response.name = sink.name
        self._logger.debug(response.message)
        return response

    def _remove_sink_callback(self, request: RemoveSink.Request, response: RemoveSink.Response) -> RemoveSink.Response:
        if not request.name:
            self._despawn_monitor.clear_sinks()
            self._spawn_scheduler.set_sinks({})
            response.message = "Removed all sinks"
        else:
            self._despawn_monitor.remove_sink(request.name)
            self._spawn_scheduler.set_sinks(self._despawn_monitor.sinks)
            response.message = f"Removed sink {request.name}"
        response.success = True
        self._logger.debug(response.message)
        return response

    def _all_wall_segments(self) -> Segments:
        """Collect wall segments from both explicit walls and obstacle bounding boxes."""
        segments = list(self._walls.values())
        for obs in self._obstacles.values():
            segments.extend(obs.wall_segments)
        return segments

    def _refresh_planners(self):
        """Push current wall segments to local planner, global planner, and collision."""
        segments = self._all_wall_segments()
        for subsystem in self._wall_aware:
            subsystem.set_walls(segments)

    def _publish_world_geometry(self) -> None:
        msg = WorldGeometryMsg()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for name, ((sx, sy), (ex, ey)) in self._walls.items():
            msg.wall_names.append(name)  # pyright: ignore[reportAttributeAccessIssue]
            msg.wall_starts.append(Point32(x=float(sx), y=float(sy), z=0.0))  # pyright: ignore[reportAttributeAccessIssue]
            msg.wall_ends.append(Point32(x=float(ex), y=float(ey), z=0.0))  # pyright: ignore[reportAttributeAccessIssue]

        for obs in self._obstacles.values():
            obs_msg = ObstacleConfigMsg()
            obs_msg.name = obs.name
            obs_msg.pose.x = obs.pose.x
            obs_msg.pose.y = obs.pose.y
            obs_msg.pose.theta = obs.pose.theta
            obs_msg.bb_x_min, obs_msg.bb_x_max, obs_msg.bb_y_min, obs_msg.bb_y_max, obs_msg.bb_z_min, obs_msg.bb_z_max = obs.bb
            obs_msg.interaction_types = list(obs.interaction_types)
            obs_msg.obstacle_type = obs.obstacle_type
            msg.obstacles.append(obs_msg)  # pyright: ignore[reportAttributeAccessIssue]

        for obj in self._world_knowledge._objects.values():
            info = WorldObjectInfoMsg()
            info.object_id = obj.object_id
            info.type = obj.type
            info.pose.x = obj.pose.x
            info.pose.y = obj.pose.y
            info.pose.theta = obj.pose.theta
            info.capacity = int(obj.capacity)
            info.satisfies_keys = list(obj.satisfies.keys())
            info.satisfies_values = [float(v) for v in obj.satisfies.values()]
            info.interaction_radius = float(obj.interaction_radius) if obj.interaction_radius is not None else 0.0
            if obj.formation is not None:
                info.formation_type = obj.formation.type
                info.formation_param_keys = list(obj.formation.params.keys())
                info.formation_param_values = [float(v) for v in obj.formation.params.values()]
            msg.world_objects.append(info)  # pyright: ignore[reportAttributeAccessIssue]

        self._world_geometry_pub.publish(msg)

    def _add_walls_callback(self, request: AddWalls.Request, response: AddWalls.Response) -> AddWalls.Response:
        for name, start, end in zip(request.names, request.starts, request.ends, strict=True):
            self._walls[name] = ((start.x, start.y), (end.x, end.y))
        self._refresh_planners()
        self._publish_world_geometry()
        response.success = True
        response.message = f"Added {len(request.names)} wall(s), total {len(self._walls)}"
        self._logger.debug(response.message)
        return response

    def _remove_walls_callback(self, request: RemoveWalls.Request, response: RemoveWalls.Response) -> RemoveWalls.Response:
        if not request.names:
            self._walls.clear()
            response.message = "Removed all walls"
        else:
            for name in request.names:
                self._walls.pop(name, None)
            response.message = f"Removed {len(request.names)} wall(s), remaining {len(self._walls)}"
        self._refresh_planners()
        self._publish_world_geometry()
        response.success = True
        self._logger.debug(response.message)
        return response

    def _set_elevators_callback(self, request: SetElevators.Request, response: SetElevators.Response) -> SetElevators.Response:
        names = [e.name for e in request.elevators]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        missing = sorted({e.destination for e in request.elevators if e.destination and e.destination not in names})
        if duplicates or missing:
            response.success = False
            response.message = f"Rejected elevators: duplicate name(s) {duplicates}, unknown destination(s) {missing}"
            self._logger.error(response.message)
            return response
        cabins = [
            Cabin(
                name=e.name,
                destination=e.destination,
                boarding=Pose2D(x=e.boarding.x, y=e.boarding.y, theta=e.boarding.theta),
                doorway=Pose2D(x=e.doorway.x, y=e.doorway.y, theta=e.doorway.theta),
                queue=Pose2D(x=e.queue.x, y=e.queue.y, theta=e.queue.theta),
                footprint=tuple((float(p.x), float(p.y)) for p in e.footprint),
            )
            for e in request.elevators
        ]
        self._elevators.set_cabins(cabins, [e.door_open for e in request.elevators])
        response.success = True
        response.message = f"Set {len(cabins)} elevator cabin(s)"
        self._logger.debug(response.message)
        return response

    def _set_elevator_doors_callback(self, request: SetElevatorDoors.Request, response: SetElevatorDoors.Response) -> SetElevatorDoors.Response:
        unknown = sorted({n for n in request.name if n not in self._elevators.cabins})
        if len(request.name) != len(request.open) or unknown:
            response.success = False
            response.message = f"Rejected doors: {len(request.name)} name(s) for {len(request.open)} state(s), unknown cabin(s) {unknown}"
            self._logger.error(response.message)
            return response
        self._elevators.open.update(zip(request.name, (bool(o) for o in request.open), strict=True))
        response.success = True
        response.message = f"Set {len(request.name)} elevator door(s)"
        self._logger.debug(response.message)
        return response

    def _teleport_agents_callback(self, request: TeleportAgents.Request, response: TeleportAgents.Response) -> TeleportAgents.Response:
        ids = [int(aid) for aid in request.agent_id]
        unknown = sorted({aid for aid in ids if aid not in self._pool._id_to_idx})
        if len(ids) != len(request.pose) or unknown:
            response.success = False
            response.message = f"Rejected teleport: {len(ids)} id(s) for {len(request.pose)} pose(s), unknown agent id(s) {unknown}"
            self._logger.error(response.message)
            return response
        for aid, pose in zip(ids, request.pose, strict=True):
            self._teleport_agent(aid, pose.x, pose.y, pose.theta, float(self._pool.agent_radius[self._pool.idx(aid)]))
            self._elevators.forget(aid)
        response.success = True
        response.message = f"Teleported {len(ids)} agent(s)"
        self._logger.debug(response.message)
        return response

    @staticmethod
    def _decompose_obstacle_to_walls(
        pose: Pose2D,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Convert a rotated 2D bounding box into 4 wall segments."""
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        corners_local = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min),
        ]
        corners_world = []
        for lx, ly in corners_local:
            wx = pose.x + cos_t * lx - sin_t * ly
            wy = pose.y + sin_t * lx + cos_t * ly
            corners_world.append((wx, wy))

        segments = []
        for i in range(4):
            j = (i + 1) % 4
            segments.append((corners_world[i], corners_world[j]))
        return segments

    def _add_obstacles_callback(self, request: AddObstacles.Request, response: AddObstacles.Response) -> AddObstacles.Response:
        added = 0
        for obs_msg in request.obstacles:
            if obs_msg.name in self._obstacles:
                continue  # noop if already exists
            pose = Pose2D(x=obs_msg.pose.x, y=obs_msg.pose.y, theta=obs_msg.pose.theta)
            wall_segments = self._decompose_obstacle_to_walls(
                pose,
                obs_msg.bb_x_min,
                obs_msg.bb_x_max,
                obs_msg.bb_y_min,
                obs_msg.bb_y_max,
            )
            self._obstacles[obs_msg.name] = ObstacleData(
                name=obs_msg.name,
                pose=pose,
                bb=(
                    obs_msg.bb_x_min,
                    obs_msg.bb_x_max,
                    obs_msg.bb_y_min,
                    obs_msg.bb_y_max,
                    obs_msg.bb_z_min,
                    obs_msg.bb_z_max,
                ),
                interaction_types=tuple(obs_msg.interaction_types),
                obstacle_type=obs_msg.obstacle_type,
                wall_segments=tuple(wall_segments),
            )
            added += 1
        if added:
            self._refresh_planners()
            self._publish_world_geometry()
        response.success = True
        response.message = f"Added {added} obstacle(s), total {len(self._obstacles)}"
        self._logger.debug(response.message)
        return response

    def _remove_obstacles_callback(self, request: RemoveObstacles.Request, response: RemoveObstacles.Response) -> RemoveObstacles.Response:
        if not request.names:
            self._obstacles.clear()
            response.message = "Removed all obstacles"
        else:
            for name in request.names:
                self._obstacles.pop(name, None)
            response.message = f"Removed {len(request.names)} obstacle(s), remaining {len(self._obstacles)}"
        self._refresh_planners()
        self._publish_world_geometry()
        response.success = True
        self._logger.debug(response.message)
        return response

    def _add_world_objects_callback(self, request: AddWorldObjects.Request, response: AddWorldObjects.Response) -> AddWorldObjects.Response:
        added = 0
        for info in request.objects:
            satisfies = {k: float(v) for k, v in zip(info.satisfies_keys, info.satisfies_values, strict=True)}
            interaction_radius = float(info.interaction_radius) if info.interaction_radius > 0.0 else None
            formation: FormationSpec | None = None
            if info.formation_type:
                params = {k: float(v) for k, v in zip(info.formation_param_keys, info.formation_param_values, strict=True)}
                formation = FormationSpec(type=info.formation_type, params=params)
            self._world_knowledge.add_object(
                WorldObject(
                    object_id=info.object_id,
                    type=info.type,
                    pose=Pose2D(x=float(info.pose.x), y=float(info.pose.y), theta=float(info.pose.theta)),
                    capacity=int(info.capacity),
                    satisfies=satisfies,
                    interaction_radius=interaction_radius,
                    formation=formation,
                    seats=_seats(info),
                )
            )
            added += 1
        self._publish_world_geometry()
        response.success = True
        response.message = f"Added {added} world object(s), total {len(self._world_knowledge)}"
        self._logger.debug(response.message)
        return response

    def _remove_world_objects_callback(self, request: RemoveWorldObjects.Request, response: RemoveWorldObjects.Response) -> RemoveWorldObjects.Response:
        if not request.object_ids:
            removed = len(self._world_knowledge)
            self._world_knowledge.clear()
            response.message = f"Removed all {removed} world object(s)"
        else:
            removed = 0
            for object_id in request.object_ids:
                if self._world_knowledge.remove_object(object_id) is not None:
                    removed += 1
            response.message = f"Removed {removed} world object(s), remaining {len(self._world_knowledge)}"
        self._publish_world_geometry()
        response.success = True
        self._logger.debug(response.message)
        return response

    def _abort_contaminated(self, counts: dict[str, int]) -> None:
        self._logger.error(f"strict_recording: aborting trial {self._trial_id or '<unnamed>'}, publisher counts {counts}")
        if self._timer is not None:
            self._timer.cancel()
        rclpy.try_shutdown()

    def destroy_node(self):
        if self._timer is not None:
            self._timer.cancel()
        if self._profile_phases:
            self._flush_profile()
        self._log_final_rtf()
        if self._sim_logger is not None:
            self._sim_logger.close()
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        super().destroy_node()

    def _log_final_rtf(self):
        if self._mode != self.MODE_MASTER:
            return
        if self._tick_wall_start is None or self._tick_count == 0:
            return
        wall_elapsed = time.perf_counter() - self._tick_wall_start
        sim_elapsed = self._tick_count * self._dt
        compute_elapsed = self._total_tick_compute_s
        wall_rtf = sim_elapsed / wall_elapsed if wall_elapsed > 0 else float("inf")
        compute_rtf = sim_elapsed / compute_elapsed if compute_elapsed > 0 else float("inf")
        self._logger.info(f"final rtf: wall={wall_rtf:.2f}, compute={compute_rtf:.2f} ({self._tick_count} ticks, sim={sim_elapsed:.1f}s, wall={wall_elapsed:.1f}s, compute={compute_elapsed:.1f}s)")


def main(args: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="arena_humansim node")
    parser.add_argument("--profile", action="store_true", help="Enable per-phase tick profiling")
    parser.add_argument("--profile-interval", type=int, default=0, help="Ticks between profile log dumps (0 = flush at shutdown only)")
    parsed, remaining = parser.parse_known_args(args)

    rclpy.init(args=remaining)
    node = AgentManager()
    if parsed.profile:
        node._profile_phases = True
    if parsed.profile_interval:
        node._profile_interval = parsed.profile_interval
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
