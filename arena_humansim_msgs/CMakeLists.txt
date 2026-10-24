cmake_minimum_required(VERSION 3.8)
project(arena_humansim_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentFrame.msg"
  "msg/AgentGestures.msg"
  "msg/AgentMeta.msg"
  "msg/AgentState.msg"
  "msg/AgentStates.msg"
  "msg/AgentViz.msg"
  "msg/Gesture.msg"
  "msg/AgentTemplate.msg"
  "msg/FlowConfig.msg"
  "msg/HighLevelCommand.msg"
  "msg/InteractionContract.msg"
  "msg/InteractionEvent.msg"
  "msg/InteractionFrame.msg"
  "msg/ObstacleConfig.msg"
  "msg/PerceptionEvent.msg"
  "msg/PerceptionEvents.msg"
  "msg/RateKeyframe.msg"
  "msg/RobotState.msg"
  "msg/Shape.msg"
  "msg/SinkAffinity.msg"
  "msg/SinkConfig.msg"
  "msg/SourceConfig.msg"
  "msg/TeleportZone.msg"
  "msg/Waypoint.msg"
  "msg/Waypoints.msg"
  "msg/WorldGeometry.msg"
  "msg/WorldObjectInfo.msg"
  "srv/AddObstacles.srv"
  "srv/AddSink.srv"
  "srv/AddSource.srv"
  "srv/AddWalls.srv"
  "srv/AddWorldObjects.srv"
  "srv/BehaviorTreeTick.srv"
  "srv/Feedback.srv"
  "srv/NotifyStimulus.srv"
  "srv/RemoveAgents.srv"
  "srv/RemoveObstacles.srv"
  "srv/ResetSimulation.srv"
  "srv/RemoveSink.srv"
  "srv/RemoveSource.srv"
  "srv/RemoveWalls.srv"
  "srv/RemoveWorldObjects.srv"
  "srv/SetFlow.srv"
  "srv/SetTeleportZones.srv"
  "srv/SetWaypoints.srv"
  "srv/SetZoneAccess.srv"
  "srv/SpawnAgents.srv"
  "srv/TeleportAgents.srv"
  "srv/GetProfile.srv"
  "srv/UpdateRobot.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
