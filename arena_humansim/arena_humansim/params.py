"""Node parameters of the humansim agent manager and the mode names they take."""

from arena_rclpy_mixins.params import Choices, Param, Range

from arena_humansim.utils.types import WaypointMode

MODE_MASTER = "master"
MODE_SUBSYSTEM = "subsystem"
CONTACT_ENABLED = "enabled"
CONTACT_LOCOMOTION_ONLY = "locomotion_only"
CONTACT_MODES = (CONTACT_ENABLED, CONTACT_LOCOMOTION_ONLY)
GESTURE_ENABLED = "enabled"
GESTURE_DISABLED = "disabled"
GESTURE_MODES = (GESTURE_ENABLED, GESTURE_DISABLED)
DEFAULT_STANDING_DISTANCE = 1.2
WAYPOINT_MODES = Choices("", *(mode.name.lower() for mode in WaypointMode))

SEED = Param[int]("seed", 0, "Random seed of the simulation RNG.")
DT = Param[float]("dt", 0.05, "Simulation timestep in seconds.")
BT_TICK_INTERVAL = Param[int]("bt_tick_interval", 5, "Behavior trees tick once every N sim ticks.", constraint=Range(1, 100000))
PERCEPTION = Param[str]("perception", "default", "Perception module name.")
GLOBAL_PLANNER = Param[str]("global_planner", "navmesh", "Global planner module name.")
GLOBAL_PLANNER_INFLATION_RADIUS = Param[float]("global_planner.inflation_radius", 0.38, "Wall clearance of planned paths in meters, rounded up to whole cells.", constraint=Range(0.0, 5.0))
GLOBAL_PLANNER_RESOLUTION = Param[float]("global_planner.resolution", 0.2, "Planning grid cell size in meters.", constraint=Range(0.01, 5.0))
GLOBAL_PLANNER_COMFORT_RADIUS = Param[float]("global_planner.comfort_radius", 0.6, "Preferred wall clearance of navmesh paths in meters, relaxed in narrow passages.", constraint=Range(0.0, 5.0))
LOCAL_PLANNER = Param[str]("local_planner", "sfm", "Default local planner module name.")
FORCE_LOCAL_PLANNER = Param[bool]("force_local_planner", False, "Put every agent on local_planner, ignoring the scenario's per-agent policies.")
ROBOT_POLICY = Param[str]("robot_policy", "", "Policy that replaces the policy of every robot-kind agent at scenario load (empty = keep the scenario policy).")
ROBOT_SHUTDOWN = Param[str]("robot_shutdown", "", "End the sim once every robot latched on its final waypoint: true or false overrides the scenario, empty = use the scenario value.")
FORCE_WAYPOINT_MODE = Param[str]("force_waypoint_mode", "", "Waypoint mode that replaces the one of every human scenario agent (empty = keep the scenario value).", constraint=WAYPOINT_MODES)
ANIMATION = Param[str]("animation", "noop", "Animation module name.")
COLLISION = Param[str]("collision", "wall_projection", "Collision resolver module name.")
OCCLUSION = Param[str]("occlusion", "bitmap", "Occlusion module name.")
MODE = Param[str]("mode", MODE_MASTER, "master owns /clock and ticks on a wall timer, subsystem ticks on /clock.", constraint=Choices(MODE_MASTER, MODE_SUBSYSTEM))
LOG_DIR = Param[str]("log_dir", "", "Directory for simulation replay logs (empty = no logging).")
REPLAY_MODE = Param[str]("replay_mode", "", "Replay log to load and play back (empty = no replay).")
WAYPOINT_THRESHOLD = Param[float]("waypoint_threshold", 0.1, "Distance in meters at which a waypoint counts as reached.")
MIN_SPEED_FOR_HEADING = Param[float]("min_speed_for_heading", 0.1, "Speed in m/s below which an agent keeps its heading.")
ARRIVAL_R_ENTER = Param[float]("arrival_r_enter", 0.15, "Radius in meters at which an agent latches onto its goal, must be positive and below arrival_r_exit.")
ARRIVAL_R_EXIT = Param[float]("arrival_r_exit", 0.30, "Radius in meters at which a latched agent releases its goal.")
ARRIVAL_TAU_BRAKE = Param[float]("arrival_tau_brake", 0.15, "Braking time constant in seconds of a latched agent, at least dt.")
ZONE_WAIT_TIMEOUT = Param[float]("zone_wait_timeout", 30.0, "Sim seconds after reaching a teleport zone's wait pose or taking a spot before a rider not yet teleported leaves the zone and is routed again.")
ZONE_LET_OUT_TIMEOUT = Param[float]("zone_let_out_timeout", 10.0, "Sim seconds of an open zone that a waiting rider leaves to those getting out before it boards anyway (0 = board at once).")
PUBLISH_MARKERS = Param[int]("publish_markers", 0, "RViz marker level: 0 = off, 1 = infrastructure, labels and interactions, 2 = full.", constraint=Range(0, 2))
PROFILE_PHASES = Param[bool]("profile_phases", False, "Time the tick phases.")
PROFILE_INTERVAL = Param[int]("profile_interval", 0, "Log the tick phase profile every N ticks (0 = never).")
RECORD_BAG = Param[bool]("record_bag", False, "Record the run to a rosbag in record_dir.")
STRICT_RECORDING = Param[bool]("strict_recording", True, "Abort the trial when a second publisher appears on a recorded contract topic.")
TRIAL_ID = Param[str]("trial_id", "", "Opaque id written into the recording manifest.")
RECORD_DIR = Param[str]("record_dir", "", "Rosbag output directory (empty = default recording directory).")
SCENARIO = Param[str]("scenario", "", "Scenario name or absolute YAML path, master mode only (empty = none).")
TICKS = Param[int]("ticks", 0, "Stop after N ticks (0 = no tick limit).")
TIME = Param[float]("time", 0.0, "Stop after this many sim seconds, ignored when ticks is set (0 = no limit).")
RTF = Param[float]("rtf", 1.0, "Real-time factor of the master tick timer (0 = unthrottled).", constraint=Range(0.0, 1000.0))
SUBSYSTEM_OVERRUN_POLICY = Param[str]("subsystem_overrun_policy", "lag", "Tick overrun policy in subsystem mode.")
CONTACT_MODE = Param[str]("contact_mode", CONTACT_ENABLED, "How contact interactions play out.", constraint=Choices(*CONTACT_MODES))
LOCOMOTION_STANDING_DISTANCE = Param[float]("locomotion_standing_distance", DEFAULT_STANDING_DISTANCE, "Standing distance in meters of a locomotion-only contact pair.")
GESTURE_MODE = Param[str]("gesture_mode", GESTURE_ENABLED, "Whether agents play gestures.", constraint=Choices(*GESTURE_MODES))

AGENT_MANAGER_PARAMS = (
    SEED,
    DT,
    BT_TICK_INTERVAL,
    PERCEPTION,
    GLOBAL_PLANNER,
    GLOBAL_PLANNER_INFLATION_RADIUS,
    GLOBAL_PLANNER_RESOLUTION,
    GLOBAL_PLANNER_COMFORT_RADIUS,
    LOCAL_PLANNER,
    FORCE_LOCAL_PLANNER,
    ROBOT_POLICY,
    ROBOT_SHUTDOWN,
    FORCE_WAYPOINT_MODE,
    ANIMATION,
    COLLISION,
    OCCLUSION,
    MODE,
    LOG_DIR,
    REPLAY_MODE,
    WAYPOINT_THRESHOLD,
    MIN_SPEED_FOR_HEADING,
    ARRIVAL_R_ENTER,
    ARRIVAL_R_EXIT,
    ARRIVAL_TAU_BRAKE,
    ZONE_WAIT_TIMEOUT,
    ZONE_LET_OUT_TIMEOUT,
    PUBLISH_MARKERS,
    PROFILE_PHASES,
    PROFILE_INTERVAL,
    RECORD_BAG,
    STRICT_RECORDING,
    TRIAL_ID,
    RECORD_DIR,
    SCENARIO,
    TICKS,
    TIME,
    RTF,
    SUBSYSTEM_OVERRUN_POLICY,
    CONTACT_MODE,
    LOCOMOTION_STANDING_DISTANCE,
    GESTURE_MODE,
)
