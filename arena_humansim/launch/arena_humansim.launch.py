import os
import re
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import ExecutableInPackage

from arena_rclpy_mixins.launch_argument import LaunchArgument, forwarded

from arena_humansim import params

FORWARDED_PARAMS = (
    params.TICKS,
    params.TIME,
    params.RTF,
    params.PERCEPTION,
    params.GLOBAL_PLANNER,
    params.LOCAL_PLANNER,
    params.FORCE_LOCAL_PLANNER,
    params.ROBOT_POLICY,
    params.ROBOT_SHUTDOWN,
    params.FORCE_WAYPOINT_MODE,
    params.ANIMATION,
    params.COLLISION,
    params.OCCLUSION,
    params.SEED,
    params.FRAME_ID,
    params.STRICT_RECORDING,
    params.TRIAL_ID,
)


def _compute_record_dir(context, *args, plan=None, **kwargs):
    render = LaunchConfiguration("render").perform(context).lower() == "true"
    record = LaunchConfiguration("record").perform(context).lower() == "true"
    fmt = LaunchConfiguration("render_format").perform(context)
    if not record:
        if plan is not None:
            plan.update(render=render, record=record, rd="", fmt=fmt)
        return [SetLaunchConfiguration("_record_dir", "")]
    rd = LaunchConfiguration("record_dir").perform(context)
    if not rd:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        scenario = LaunchConfiguration("scenario").perform(context)
        mode = LaunchConfiguration("mode").perform(context)
        suffix = ""
        if scenario and mode == params.MODE_MASTER:
            stem = Path(scenario).stem
            slug = re.sub(r"[^\w.-]", "_", stem)
            if slug:
                suffix = f"_{slug}"
        rd = os.path.join(os.getcwd(), "recordings", f"{ts}{suffix}")
    if plan is not None:
        plan.update(render=render, record=record, rd=rd, fmt=fmt)
    return [SetLaunchConfiguration("_record_dir", rd)]


def _rviz_action(context, *args, **kwargs):
    rviz_val = LaunchConfiguration("rviz").perform(context)
    mode_val = LaunchConfiguration("mode").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    if rviz_val == "":
        markers_val = LaunchConfiguration("markers").perform(context)
        rviz_val = "false" if markers_val == "0" else "true"
    if mode_val != params.MODE_MASTER or rviz_val.lower() == "false":
        return []
    if rviz_val.lower() == "true":
        config = os.path.join(get_package_share_directory("arena_humansim"), "config", "arena_humansim.rviz")
    else:
        config = rviz_val
    return [ExecuteProcess(
        cmd=["rviz2", "-d", config, "--ros-args", "-p", f"use_sim_time:={use_sim_time}"],
        output="screen",
    )]


def _renderer_action(context, *args, plan=None, **kwargs):
    plan = plan or {}
    render, record, rd = plan.get("render"), plan.get("record"), plan.get("rd")
    if not (render and record and rd):
        return [Shutdown(reason="node exited; no render requested")]
    fmt = plan["fmt"]
    bag_dir = os.path.join(rd, "bag")
    output = os.path.join(rd, f"scenario.{fmt}")
    log_path = os.path.join(rd, "render.log")
    renderer_exe = ExecutableInPackage(package="arena_humansim", executable="arena_humansim_render").perform(context)
    return [ExecuteProcess(
        cmd=[renderer_exe, bag_dir, "--output", output, "--format", fmt, "--log-file", log_path],
        output="screen",
        on_exit=Shutdown(reason="renderer finished"),
    )]


def _node_actions(context, *args, declared=frozenset(), plan=None, **kwargs):
    extra = {name: ParameterValue(LaunchConfiguration(name)) for name in context.launch_configurations if name not in declared and name != "ros_remaps" and not name.startswith("_")}
    values = forwarded(context, FORWARDED_PARAMS)
    node = Node(
        package="arena_humansim",
        executable="arena_humansim_node",
        name="arena_humansim",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {"mode": LaunchConfiguration("mode"),
             "use_sim_time": LaunchConfiguration("use_sim_time"),
             "publish_markers": LaunchConfiguration("markers"),
             "record_bag": LaunchConfiguration("record"),
             "record_dir": LaunchConfiguration("_record_dir"),
             "scenario": LaunchConfiguration("scenario")},
            *([values] if values else []),
            *([extra] if extra else []),
        ],
        output="screen",
    )
    kill_rviz = ExecuteProcess(cmd=["pkill", "-INT", "-f", "rviz2.*arena_humansim.rviz"], output="log")
    return [
        node,
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[kill_rviz, OpaqueFunction(function=_renderer_action, kwargs={"plan": plan})],
        )),
    ]


def generate_launch_description():
    map_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="arena_humansim_map_tf",
        arguments=["--frame-id", "world", "--child-frame-id", "map"],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="log",
    )

    viz = Node(
        package="arena_humansim_viz",
        executable="arena_humansim_viz_node",
        name="arena_humansim_viz",
        namespace=LaunchConfiguration("namespace"),
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="screen",
    )

    renderer_plan: dict = {}

    arguments = [
        LaunchArgument("namespace", default_value="arena_humansim", description="Namespace of the humansim nodes."),
        LaunchArgument.from_param(params.MODE, keep_default=True),
        LaunchArgument("use_sim_time", default_value="true", description="Use the sim clock."),
        LaunchArgument("markers", default_value="0", description=params.PUBLISH_MARKERS.description),
        LaunchArgument("rviz", default_value="", description="true = default config, false = off, a path = that config, empty = on when markers is above 0."),
        LaunchArgument("record", default_value="false", description=params.RECORD_BAG.description),
        LaunchArgument.from_param(params.RECORD_DIR, keep_default=True),
        LaunchArgument("render", default_value="true", description="Render a video of the recording after the node exits."),
        LaunchArgument("render_format", default_value="mp4", choices=["mp4", "gif"], description="Video format of the rendered recording."),
        LaunchArgument.from_param(params.SCENARIO, keep_default=True),
        *(LaunchArgument.from_param(param) for param in FORWARDED_PARAMS),
    ]

    return LaunchDescription([
        *arguments,
        OpaqueFunction(function=_compute_record_dir, kwargs={"plan": renderer_plan}),
        map_tf,
        viz,
        OpaqueFunction(function=_node_actions, kwargs={"declared": frozenset(a.name for a in arguments), "plan": renderer_plan}),
        OpaqueFunction(function=_rviz_action),
    ])
