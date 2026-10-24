string object_id
string type
geometry_msgs/Pose2D pose
uint32 capacity
string[] satisfies_keys
float64[] satisfies_values
float64 interaction_radius     # 0.0 = use interaction-type default

# Optional formation (line | cluster | f_formation | dyad). Empty type = no formation.
string formation_type
string[] formation_param_keys
float64[] formation_param_values

# Seat poses (world frame): cluster formations on this object place participants exactly here. Empty = ring around the pose.
geometry_msgs/Pose2D[] seats
uint32[] seat_conflict_offset  # n+1 offsets into seat_conflicts, empty = none
uint32[] seat_conflicts        # seats that cannot be held together with seat i
# Height of each seat's surface (seat cushion, mattress) above the object's floor, meters, parallel to seats.
# Empty or NaN = not measured: an agent parked on that seat is reported on its feet.
float64[] seat_heights
