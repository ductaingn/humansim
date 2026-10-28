# Scenarios

Scenarios live here as yaml, loaded by [../../arena_humansim/utils/scenario.py](../../arena_humansim/utils/scenario.py).

## Interaction step fields

Every BT step that joins or creates an interaction uses `SeekNode` under the hood. The per-step yaml maps directly to a `SeekSpec` on the emitted `SEEK` command. Which fields are valid depends on the `InteractionType`'s handle kind (see [../../arena_humansim/core/interaction_kinds.py](../../arena_humansim/core/interaction_kinds.py)):

| Handle   | Interactions                                   | `target:` shape              | `offer:`              |
|----------|------------------------------------------------|------------------------------|-----------------------|
| `NONE`   | `TALK_TO`, `GROUP_CONVERSATION`, `GROUP_WALK`, `WAVE_AT`, `HUG`, `SHAKE_HAND` | omitted                   | not allowed           |
| `OBJECT` | `USE`, `SIT_ON`, `LIE_ON`, `QUEUE_USE`         | `str` (object id or type)    | not allowed           |
| `TAG`    | `SERVICE`                                      | `str` (service tag; required for `offer: true`, optional for seekers) | provider side only |
| `AGENT`  | `BLOCK`                                        | `int` (agent id)             | not allowed           |

`WorldKnowledge.resolve` treats an `OBJECT` target as object id first, then falls back to object type (nearest visible). The loader rejects mismatched shapes (e.g. `target: <str>` on BLOCK, `offer: true` on TALK_TO).

### Provider-side flags (SERVICE only, with `offer: true`)

- `min_participants: int` - ACTIVE threshold; counts the provider itself.
- `max_participants: int` - cap (counts the provider). `-1` = unbounded.
- `queueable: bool` - admit seekers into a FIFO queue when full.
- `formation_spec: {type, anchor_kind, params}` - formation imposed on participants. Common choice for mobile providers: `{type: line, anchor_kind: provider, params: {base_step: 0.8}}` (line trails the provider's pose, slot-0 is the provider and is skipped by the formation emitter).

### Ending an interaction

- `duration: {mean: ..., std: ...}` - contract-level timeout. IM tears down with outcome `COMPLETED` when elapsed.
- `{cancel: true}` as its own follow-up step - emits STOP on the agent's current `interaction_id` with `reason=CANCELED`.

Pick one. Mixing them on the same step is allowed (duration is a fallback if Cancel never runs).

## `interaction_radius` (meters)

Two roles, same number:

- **Request:** how close an agent must get to an object-bound target before `SeekNode` is reached in the compiled sequence.
- **Drift eviction:** how far a participant may drift from the nearest peer (or from the bound object) once `ACTIVE`, scaled by `cohesion_multiplier` (default 1.2 in `InteractionManager`). A participant past `interaction_radius * cohesion_multiplier` is evicted with `INTERRUPTED`. Tightening this field tightens both gates together.

Cascade (first non-null wins):

1. `StepDef.interaction_radius` - per-step override.
2. `WorldObject.interaction_radius` - per-object override on the target.
3. `InteractionKind.interaction_radius` - per-type default (see below).
4. `DISTANCE_TOLERANCE` = 0.5 m (from [../../arena_humansim/utils/const.py](../../arena_humansim/utils/const.py)).

### Type defaults

| Interaction          | Default (m)          |
|----------------------|----------------------|
| `USE`                | `DISTANCE_TOLERANCE` |
| `QUEUE_USE`          | `DISTANCE_TOLERANCE` |
| `SIT_ON`             | `DISTANCE_TOLERANCE` |
| `LIE_ON`             | `DISTANCE_TOLERANCE` |
| `GROUP_CONVERSATION` | 3.0                  |
| `GROUP_WALK`         | 3.0                  |
| `TALK_TO`            | 2.0                  |
| `SERVICE`            | 3.0                  |
| `WAVE_AT`            | 6.0                   |
| `BLOCK`              | `DISTANCE_TOLERANCE` |
| `HUG`                | 0.3                   |
| `SHAKE_HAND`         | 0.3                   |

Source of truth: the `InteractionKind.interaction_radius` fields in [../../arena_humansim/core/interaction_kinds.py](../../arena_humansim/core/interaction_kinds.py).

### `render_pose_override` kinds (`HUG`, `SHAKE_HAND`)

A dyad whose target formation separation is smaller than the pair's combined `agent_radius` (a hug, unlike a `TALK_TO`, wants the two bodies to actually close in) can never be reached by tuning `interaction_radius` / the formation's `separation` alone: every local planner's own collision-avoidance repulsion between the pair keeps them apart by roughly `agent_radius_a + agent_radius_b`, no matter the target - this is true of every planner (SFM/HSFM, ORCA, learned ones like SocialGAIL/NSP) since it falls out of each one's own avoidance term, not something a scenario or a single planner can be tuned around.

`InteractionKind.render_pose_override = True` (set on `HUG` and `SHAKE_HAND`) opts a kind out of that fight entirely, on the display side instead of the physics side: `AttentionNode._clip_render_target` (`core/behavior/nodes/attention.py`) publishes the interaction's live formation slot (`InteractionManager.formation_target`) as the clip's `GestureIntent.x/y`, flagged `render_pose_override=True` on the wire (`Gesture.msg`). `task_generator`'s humansim bridge (`_agent_states_to_pedestrians` in `simulators/human/arena_humansim/arena_humansim.py`) substitutes that position for the physics pose on the `Pedestrian` it publishes. Physics (collision avoidance, every other agent's avoidance, robot planning) is untouched - only what gets drawn/sensed downstream of `arena_peds` changes. This generalizes to any local planner, current or future, for free, since it never touches planner code at all.

### Directional greeting (`WAVE_AT` broadcaster/responder via `clip.when`)

`WAVE_AT` uses the symmetric `NONE` handle like `TALK_TO`, so either peer may create the
interaction and the other joins once mutually visible - there is no engine-level "who waved
first". A one-then-the-other greeting (A waves, B notices and waves back, rather than both
starting the clip in lockstep) is authored per-agent via `attention.clip.when`, not via the
interaction system:

```yaml
# broadcaster - keeps waving while it looks for a partner, regardless of binding
do_wave:
  interaction: WAVE_AT
  attention:
    clip: {name: wave, when: always}
    gaze: partner
  duration: {mean: 6.0, std: 0.0}

# responder - only raises the clip once actually bound to the broadcaster's interaction
do_wave:
  interaction: WAVE_AT
  attention:
    clip: {name: wave, when: bound}
    gaze: partner
  duration: {mean: 6.0, std: 0.0}
```

`interaction_radius: 6.0` (above typical `vision_range`, mean 5.0) keeps the pair latched for the
whole greeting even though `WAVE_AT` has no formation to draw them together - discovery still
requires mutual visibility (`_scan_symmetric`'s `visibility_lookup` gate), so nothing pairs up
outside vision range or FOV; the radius only governs post-match drift eviction. See
`worlds/*/scenarios/social_waving/waver_{1,2}.yaml` for the working example.

### Object override

```yaml
world_objects:
  - object_id: plaza
    type: gathering_area
    pose: {x: 0.0, y: 0.0}
    interaction_radius: 2.0      # overrides the 3.0 GROUP_CONVERSATION default

sequences:
  loiter:
    steps:
      join_conversation:
        target: plaza            # object id
        interaction: GROUP_CONVERSATION
        interaction_radius: 1.5  # tighter still, overrides object + type
        patience: {mean: 60.0}
```

## Examples by handle shape

### Symmetric peer (NONE)

No `target:`, no `offer:`. Both peers run the same step; whichever ticks first creates, the other joins.

```yaml
chat:
  interaction: TALK_TO
  duration: {mean: 8.0, std: 2.0}
  patience: {mean: 30.0}
```

### Object-anchored (OBJECT)

`target:` is an object id or object type. The compiler emits `Resolve -> GoTo -> Seek`.

```yaml
queue_and_drink:
  target: fountain               # object id, or "fountain" as a type
  interaction: USE
  duration: {mean: 6.0, std: 1.5}
  satisfies: {thirst: 100.0}
```

### Service provider (TAG, `offer: true`)

Creates the interaction; seekers join. Mobile provider uses line formation anchored on itself.

```yaml
offer_ride:
  interaction: SERVICE
  target: escort_ride            # service tag (free-form string)
  offer: true
  min_participants: 2
  max_participants: 2
  queueable: false
  formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}
  patience: {mean: 60.0}
```

### Service seeker (TAG, no `offer`)

Polls for a visible provider of the matching tag.

```yaml
hail:
  interaction: SERVICE
  target: escort_ride
  patience: {mean: 90.0}
```

### Agent-targeted (AGENT - BLOCK)

`target:` is an agent id; `BlockNode` pursues and emits SEEK on arrival.

```yaml
intercept:
  interaction: BLOCK
  target: 42                     # agent id to block
  duration: {mean: 3.0}
```

### Mobile provider (escort / shuttle)

Offer, drive, cancel. The interaction stays live across the drive; line formation trails the provider.

```yaml
offer_A:
  interaction: SERVICE
  target: escort_ride
  offer: true
  min_participants: 2
  max_participants: 2
  queueable: false
  formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}
ride_to_B: {kind: go_to, target_pose: {x: 7.0, y: 0.0}}
drop_B:    {cancel: true}
```

## Attention

`attention:` is one block of channels. It rides on every step kind (`go_to`, interaction, wait, cancel, BLOCK), stands as a step of its own (`kind: attention`, or just `attention:` plus duration-ish fields and no interaction/target/cancel/autonomous keys), and rides on a whole sequence (`sequences.<seq>.attention`). A kind-less step that mixes `attention:` with interaction-only fields (`offer`, `formation_spec`, `until`, ...) is rejected by the loader, add `kind:` or `interaction:`. The engine has no skeleton: it publishes the active channels on the latched `agent_gestures` topic whenever they change (`AgentGestures`: owner `agent_id` plus `Gesture{slot, at, clip, hand}`), the animation layer moves the body.

```yaml
attention:
  gaze:    partner                                   # head
  point:   {at: [bench_1, bench_2], dwell: 1.5}      # one arm, side picked by the arena layer
  point_l: exit_door                                 # left arm, explicit
  point_r: exit_door                                 # right arm, explicit
  clip:    wave_high                                 # body, a canned clip by name (or {name, when: always|bound})
  face: auto                                         # auto | true | false | <ref>
  required: false                                    # riders only, bare steps are always required
```

At least one channel (`clip` counts), and exactly one arm channel (`point` xor `point_l` xor `point_r`). Nothing is implicit: a body part without a channel stays idle. Slots on the wire: `gaze` -> `head`, `point` -> `arm` with `hand` from the ped's sampled handedness, `point_l`/`point_r` -> `arm_l`/`arm_r`, `clip` -> `body` with the clip name.

`clip` names a clip of the animation layer, the engine never reads it. `when: bound` shows it only while the agent is a participant of an active interaction, which is how interaction kinds carry a default; an authored `clip` on the step replaces the default. A bare step with only a `clip` needs `duration` or `patience`.

An interaction kind's default is a list of clip sets (`InteractionKind.clips`, `ClipSet` in [../../arena_humansim/core/interaction_kinds.py](../../arena_humansim/core/interaction_kinds.py)). Each interaction draws one set from its creator's random stream, and each member plays its role's clip, roles taken in join order and cycled for groups (a one-role set plays the same clip on everyone):

| kind | clip sets |
|---|---|
| `SIT_ON` | `sit` |
| `TALK_TO` | `talk_with_arm_gesture`, `talk_with_arm_gesture_{2,3,4}`, `talk_with_arm_gesture_v1`, `intermask_talk_{a,b}` |
| `GROUP_CONVERSATION` | `intermask_group_conversation_{a,b}` |
| `WAVE_AT` | `wave`, `intermask_wave_at_{a,b}` |
| `HUG` | `hug_v7_air_{w,m}` (0.3126 m), `hug_v7_hero_{w,m}` (0.2838 m) |
| `SHAKE_HAND` | `shake_hand_v7_doctor_pair` (0.88 m) |

A set made for a stance carries it: its `separation` and per-role `headings` (each role's turn off facing its partner, `heading_a` / `heading_b` of the `dyad` formation) replace the kind's default dyad, so contact clips touch where they were made to. `ClipSet.pair` takes them straight from role a's `pair.partner_offset` annotation (written by `arena_animation_retarget --pair`). A step's own `formation_spec` replaces the set's stance, `contact_mode: locomotion_only` the separation and headings both.

A channel is either `<targets>` (shorthand for `{at: <targets>}`) or a mapping:

```yaml
gaze: {at: [partner, bench_1], dwell: 1.0, advance: dwell, hold: release, at_z: 1.4}
```

- `at`: one ref or a list. A single ref is a list of one, so `dwell` applies to it too.
- `dwell` (default 1.0 s) with `advance: dwell` (default): entries in order, `dwell` seconds each, the list ends on the last entry and stays there until the step ends.
- `advance: unreachable`: entries cycle forever by reach. Reach is the azimuth of the entry against the commanded heading (`heading_goal` while turning, else the body yaw): an arm shows below `ARM_IN` (90 deg) and hides above `ARM_OUT` (110 deg), the head shows below `HEAD_IN` (60 deg) and hides above `HEAD_OUT` (70 deg). When the current entry leaves reach the channel jumps to the next entry in list order (wrapping) that is in reach, or holds on the current one publishing nothing until one comes back. Never advances during a face turn or before `MIN_RESIDENCE_S` (0.5 s) on the current entry. Constants live in `arena_humansim/core/behavior/reach.py`.
- `hold`: `release` (default) unpublishes the channel when the step ends, `keep` leaves it published, frozen at its last target, until a later step names the same channel or the sequence ends.
- `at_z`: height for entity refs (default head 1.6, arm 1.2 on agents, 0.8 on objects), rejected with literal or relative refs.

| ref | Resolves to |
|---|---|
| `partner` | Nearest other participant of the agent's current interaction. |
| `partners` | All other participants, as the list they expand to. |
| `target` | The step's own resolved object or `target_pose`. |
| `goal` | The agent's current navigation goal. |
| `robot:<name>` | Agent name lookup restricted to robots. |
| plain `str` | Object id, then agent name (peds and robots), then object type (nearest). Ids and names also match on their last `/` segment when unique, so `ped_3` finds `env_0/ped_3`. |
| `int` | Agent id. |
| `{x, y, z}` | Literal world point, in the frame the engine runs in (authored coordinates, the Arena adapter owns any env shift). |
| `{azimuth, elevation[, distance]}` | Degrees relative to the agent's own pose and yaw, re-evaluated each tick, default distance 3.0 m. Never drives `face`. |

`partner`, `partners`, `target` and `goal` are reserved: `SpawnAgents` rejects agents named like one. Entries are unresolved until resolved: retried every tick, never dropped, one warning. A bare step FAILS when a channel sits on an unresolved entry for `RESOLVE_TIMEOUT_S` (4 s). A rider with `required: false` (default) warns once, publishes nothing for that channel and keeps retrying, with `required: true` the host step FAILS after the same grace.

**face.** For `auto` and `true` the face target is the current entry of the winning channel (`point` > `point_r` > `point_l` > `gaze`), skipping relative entries (the heading holds). `true` forces the turn and the loader rejects it when the winning channel's first entry is relative. `false` never turns. `<ref>` turns toward that ref. Facing is skipped while the ped is walking (`GoToNode` or `BlockNode` running) or bound in a formation, so on a `go_to` it applies during the post-arrival hold, and on wait, interaction and bare steps it applies throughout. The turn must come within `FACE_ENTER_RAD` (0.25 rad), afterwards the ped re-faces only when the target leaves a `FACE_KEEP_RAD` (0.6 rad) cone. A bare step FAILS after `FACE_TIMEOUT_S` (4 s) without reaching the target, a rider gives up and keeps its heading. Channels publish against the commanded heading, so an arm rises while the body is still turning.

**Bare step end.** Without `duration`: SUCCESS once every channel finished its list, so a cycling channel needs `duration` or `patience` (loader error). With `duration`: SUCCESS at duration regardless of list progress. `patience` bounds resolve + turn + hold, on expiry FAILURE. Riders have no end of their own.

**Sequence rider.** `sequences.<seq>.attention` persists across the steps of the sequence. A step with its own `attention:` overrides it while the step runs (the rider is lowered, not reset), afterwards it resumes with its list index and held pose intact. Autonomous steps take no attention and suspend the rider for their span. Leaving the sequence releases everything, kept channels included.

```yaml
# bare: halt, turn to each bench, point 1.5 s each, keep the arm up into the next step
show_benches:
  attention: {point: {at: [bench_1, bench_2], dwell: 1.5, hold: keep}}
show_door:
  attention: {point: exit_door, gaze: partner}
  duration: {mean: 1.5}

# burst: glance around without turning
look_around:
  attention: {gaze: {at: [{azimuth: -60, elevation: 0}, {azimuth: 60, elevation: 0}], dwell: 0.7}}

# go_to while pointing at whatever landmark is in reach, cycling as the ped turns
escort: {kind: go_to, target_pose: {x: 7.0, y: 0.0}, attention: {point: {at: [ped_3, exit_door], advance: unreachable}}}

# interaction step: the formation owns the heading, the head tracks the nearest partner
chat: {interaction: TALK_TO, duration: {mean: 8.0}, attention: {gaze: partner}}

# sequence rider: keep looking at the robot across every step of the sequence
sequences:
  greet:
    attention: {gaze: {at: "robot:bot", advance: unreachable}}
    steps: {...}
```

`attention:` is not valid in the autonomous `actions` library and not on `autonomous: true` steps.

Handedness is sampled once per ped from the agent type (`handedness: {right: 0.9, left: 0.1}`, weights per hand) and can be pinned per spawn via `AgentState.handedness` (`l` | `r`); it is republished on `AgentMeta.handedness`.

## Robot services

Robot agents (`kind: 1`) can offer named services via a `services:` list. Each tick, `RobotServiceAdvertiser` emits one `SEEK` command with `offer: true` per declared service, producing parallel `SERVICE` interactions that humans bind to.

```yaml
agents:
  - agent_id: 10
    kind: 1
    policy: sfm
    services:
      - tag: water
        max_participants: 4
      - tag: trash
        max_participants: -1     # unbounded
```

- `tag: str` - free-form identifier. Matched string-equal against human seekers' `target:`.
- `max_participants: int` - cap, **counting the robot itself**. Set `N+1` for "N humans at once". Use `-1` for unbounded.
- Seeker side is a plain `{interaction: SERVICE, target: <tag>}` step - same node as any other service seeker.

Matcher semantics (visibility, provider anchor, queueing) live in [../../arena_humansim/core/README.md](../../arena_humansim/core/README.md).
