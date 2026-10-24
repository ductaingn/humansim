# Floor area pedestrians wait at, enter and leave by teleport, engine frame
string name                          # unique zone name
string[] destinations                # zones a teleport from here can land in
geometry_msgs/Pose2D entry           # outer edge of the opening, facing in
geometry_msgs/Pose2D wait            # where a rider waits, clear of the opening
geometry_msgs/Pose2D hold            # fallback spot inside, heading riders hold
geometry_msgs/Point32[] footprint    # zone polygon, z unused
bool open                            # passable when added
float64 delay                        # s until a holder self-teleports, <0 never
