# Per-tick state of the engine's own agents, one array entry per agent
std_msgs/Header header
int32[] agent_id
float64[] x
float64[] y
float64[] theta
float64[] vx
float64[] vy
float64[] desired_velocity   # preferred walking speed (m/s)
float64[] radius             # collision radius (m)
uint8[] kind                 # AgentState.KIND_*
uint8[] animation_state
int16[] policy_idx           # into AgentMeta.policies, -1 = teleport-only
float64[] gait_phase         # cycle phase in rad, 0 = not provided
float64[] gait_cadence       # cycle rate in Hz, 0 = unknown
int32[] interaction_id       # active interaction, -1 = none or gestures off
uint8[] interaction_type     # InteractionType, valid when interaction_id >= 0
float32[] support_height     # surface the agent sits or lies on, m above its floor, -1 = on its feet
