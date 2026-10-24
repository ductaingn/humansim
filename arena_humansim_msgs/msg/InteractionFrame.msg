# Live interactions and the edges fired this tick, latched, sent on change
std_msgs/Header header
int32[] interaction_id
uint8[] interaction_type     # InteractionType
uint8[] outcome              # InteractionOutcome, FORMING or ACTIVE
string[] object_id           # anchor world object, empty = none
bool[] arrived               # every participant is on its formation slot
bool[] holding               # HOLD_ONSET fired, interaction not yet ended
float64[] duration           # hold duration in seconds, -1 = unbounded
uint32[] member_offset       # n+1 offsets into member_id, _role and _seat
int32[] member_id            # participants, then the queue in FIFO order
uint8[] member_role          # MembershipRole
int32[] member_seat          # object seat index per member, -1 = none
InteractionEvent[] events    # edges fired on this tick
