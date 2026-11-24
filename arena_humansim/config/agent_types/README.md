# Agent types

Each `.yaml` file defines one agent type. Types are *distributions*, not fixed parameter sets - the loader samples a concrete instance per agent at spawn from the `{mean, std, clip_low, clip_high}` triples.

## Fields

Values of the mobility-aid types and their sources: [SOURCES.md](SOURCES.md).

Physical:

- `desired_velocity`, `max_velocity` (m/s)
- `max_acceleration`, `max_deceleration` (m/s^2)
- `agent_radius` (m)
- `min_turning_radius` (m)
- `pivot_angular_velocity` (rad/s)
- `reaction_time` (s) - sampled log-normal: `mean` is the desired median, `std` is the underlying normal's sigma (shape), not a linear stddev.
- `personal_space_min` (m)
- `idle_gaze_rate` (Hz) - rate at which an idle agent's gaze sways around its own current heading. `0` disables the gaze sway.

Perception (`perception:`):

- `vision_range` (m)
- `vision_fov` (degrees, full angle - 360 = omnidirectional)
- `proximity_sense` (m) - near-field detection radius independent of FOV.
- `vision_occlusion` (bool, not a distribution) - whether walls block line of sight. With it, walls also block hearing and the cane.
- `hearing_range` (m, default 0 = off) - who the agent hears at any bearing: other pedestrians out to this range, robots out to this range times the robot's speed over 1 m/s times the `robot_loudness` node parameter (0 = silent robots); a robot slower than 0.02 m/s is silent.
- `cane_reach`, `cane_width` (m, default 0 = off) - the cane's sweep: the agent touches anything whose disc reaches into the rectangle `cane_reach` ahead and `cane_width` across.
- `detection_delay` (s, default 0) - a robot joins the agent's neighbours only after the agent has sensed it, by any channel, for this long without a break.
- `robot_awareness` - how this agent perceives robots. Accepts a preset name, `blind`, a boolean (`true` is `normal`, `false` is `blind`), a mapping of shifts, or a list of weighted entries for a crowd mixture. Mapping keys are `detection`, `fov`, `repulsion`, `reaction`, `blind` and `yields`; each value is a number of sigmas or a `{mean, std, clip_low, clip_high}` distribution of sigmas, and omitted keys stay at 0. A sigma is the agent's own spread of that dimension, so `detection: -2` moves the robot range by two times the `vision_range` std.
  - `detection`: robot detection range, from `vision_range`; never past the agent's own range.
  - `fov`: robot field of view, from `vision_fov`, clipped to 0-360 degrees.
  - `repulsion`: gain on the SFM repulsion strength A against robots, `1 + sigmas x (std / mean)` of `repulsion_strength` (the SFM default when the planner declares none), floored at 0. ORCA scales its robot clearance by the same gain.
  - `reaction`: gain on the SFM repulsion range B against robots, `1 + sigmas x (std / mean)` of `repulsion_range` (same fallback), floored at 0.05. A larger B starts avoiding robots earlier.
  - `blind: true` removes robots from perception entirely.
  - `yields: false` keeps the robot in perception but drops all avoidance of it: zero SFM repulsion, and ORCA leaves it out.
  - Sensing the robot (`blind`, `detection`, `fov` and the perception channels above) and yielding to it (`yields`, `repulsion`, `reaction`) are separate, so a scenario can swap one without the other.
  - The former scalar keys (a numeric `robot_awareness`, `robot_repulsion_gain`, `robot_fov_gain` and `robot_reaction_gain`) are removed; the shifts above replace them.
- `robot_response` - what the agent does with a robot it perceives (SFM planner). `avoid` (default) steers around it. `halt` never steers around it: the agent brakes to a stop while the robot is ahead of its heading and walks at `HALT_SLOW_FACTOR` (0.5) of its desired speed while the robot is beside or behind it, until the robot leaves its perception. Used by the long-cane users, who notice a robot only at close range and rely on others to give way, and by the walking-frame user, who cannot step sideways without lifting the frame.

Built-in presets for robot awareness: `blind`, `distracted`, `normal`, `interested`, `aggressive`, `scared`, `not_yielding` (senses the robot, does not give way). Their shifts are listed under the archetypes below.

### Per-planner robot interaction capability matrix

| Parameter | SFM / HSFM | ORCA | Meaning |
|---|---|---|---|
| `robot_awareness` `detection` | robot detection range, pruned in perception | distance cutoff when below `vision_range` | how far the agent sees robots |
| `robot_awareness` `repulsion` | gain on repulsion strength A against robots | gain on the robot clearance radius, floored at 0.85 | yielding urgency and clearance |
| `robot_awareness` `fov` | robot cone, pruned in perception | ignored | asymmetric field of view against robots |
| `robot_awareness` `reaction` | gain on repulsion range B against robots | gain on velocity obstacle time horizon against robots | how early the agent starts avoiding robots |
| `robot_awareness` `blind` | no robot perception | no robot perception | robots are invisible to the agent |
| `robot_awareness` `yields: false` | zero repulsion against robots | robots left out | the agent senses robots but does not give way |
| `hearing_range`, `cane_reach` | add neighbours in perception | widen the robot distance cutoff, any bearing | hearing and cane touch |
| `detection_delay` | holds robots back in perception | ignored | late noticing |

> Note: ORCA builds its private neighbor list and solves instantaneous geometric velocity obstacles, so it reads `detection`, `blind`, `repulsion` and `reaction`; `fov` acts only through the perception-gated planners.

Local planner (`local_planner_params:`):

Schema is taken from the active planner's `PARAM_DEFAULTS` (`sfm`: `relaxation_time`, `repulsion_strength`, `repulsion_range`, `anisotropy`. `hsfm` adds `lateral_gain`, `lateral_damping`, `angular_gain`, `angular_damping`, `heading_source`. the other force models declare their own keys, the rest take none). yaml entries override individual keys, unset ones keep the planner's default distribution. Full per-planner key table: [../../arena_humansim/local_planner/README.md](../../arena_humansim/local_planner/README.md).

`heading_source` (hsfm) also accepts the words `attraction` (= `0.0`, the body turns toward the goal attraction force) and `total` (= `1.0`, the body turns toward the total force including repulsion, held within 90 degrees either side of the goal bearing, so a pushed agent swings aside without turning its back on the goal). A number below `0.5` acts as `attraction`, any other as `total`.

Locomotion (`locomotion:`, optional):

Declaring the block at all (even empty) activates the locomotion profile for the type, every sub-field has a default. Without it the type moves as before: holonomic, no gait phase on the wire, no recovery.

- `kinematics` (name, not a distribution) - `holonomic` (default, velocity is free) or `along_heading` (velocity is projected onto the body heading every tick, so the agent drives like a wheelchair or a cart: forward only, turns are rate-limited by `pivot_angular_velocity` and `min_turning_radius`).
- `cadence: {base, per_speed, min, max}` (Hz, Hz per m/s, Hz, Hz) - gait cycle rate `clip(base + per_speed * speed, min, max)`. Defaults match the renderer's walk cycle (`0.4`, `0.55`, `0.4`, `2.2`).
- `phase_warp: {split}` - fraction of the cycle period taken by the first half-cycle (`0.5` = symmetric). The stored phase stays monotonic, the warp applies when the profiles read it.
- `speed_profile: {harmonics: [[amp, phase], ...], amplitude_scale}` - up to 3 harmonics (literal floats, not distributions) of a zero-mean speed modulation `s = sum_k amp_k * sin(k * phase + phase_k)`. The integrated and published velocity is the planner velocity times `1 + amplitude_scale * s` (floored at 0.05). The factor is laid over the velocity for one tick and taken out again before the planners run, so planner state never sees it and the mean speed over whole cycles stays the commanded speed. `amplitude_scale` is a distribution.
- `lateral_profile: {harmonics: [...]}` - same form, added as a lateral (body-perpendicular) sway `speed * l(phase)` to the integrated and published velocity of `holonomic` types, ignored under `along_heading`.
- `footprint_length` (m) - length of a capsule footprint along the heading, `0` = plain disk. Fills `pool.axial_offset = max(0, footprint_length / 2 - agent_radius)` for the planners and the collision resolver.
- `recovery: {stall_after_s, reverse_m}` - stall recovery. `stall_after_s = 0` (default) turns it off. When the distance to the current goal has not improved by 5 cm for `stall_after_s`, the agent stops and pivots toward the goal bearing, then drives again with a 2 s cooldown. A re-stall (no improvement for 1 s) inside the cooldown with `reverse_m > 0` backs up `reverse_m` at 0.3 m/s before pivoting again. The speed and lateral profiles pause while the agent pivots or backs up.

The zero-means-off fields (`footprint_length`, `recovery.*`) take a bare number for `0`, since the dict form's `clip_low` default of `0.01` would lift a `{mean: 0}` to `0.01`.

Device (`device:`, optional): passed through untouched to arena_runtime, which projects the held device on the ground for robot collisions; humansim never reads it. `{kind: cane | rollator | walking_frame | iv_stand, bone: <prop bone of the character>}`, or `model: <object>` for an object held in both hands that the character does not carry, centered under the `pose.hold` grips.

Pose (`pose:`, optional): passed through untouched to the renderer (arena_runtime's human animation), humansim never reads it. Per-state (`walk`, `idle`, `run`) joint profiles keyed to the published gait phase, see the renderer's profile docs. `hold: {l|r: <grip radius m> | {radius, angle, curls, wrap}}` closes each named hand on a prop of that radius (cane, pole, walker handle): the renderer solves the finger curls from the template hand skeleton; `angle` turns the grip in the palm (deg), `curls` (thumb to pinky) gives a character's own curls instead (the retarget's `--hold` reports them), `wrap: true` closes the thumb over the fingers. `while: {<interaction>: <pose section>}` gives the whole profile the agent takes while in that interaction (e.g. `caregiver` pushing a wheelchair, `wheelchair_manual` being pushed).

Interaction class (`interaction_class:`, optional name string): social-force class of the type for the local planners' per-class gain matrices. Unset = `human`, robots are always `robot`, any other word (e.g. `wheelchair`) registers a new class on first sight, see [../../arena_humansim/core/interaction_classes.py](../../arena_humansim/core/interaction_classes.py).

Assets (`assets:`, optional list of ASA tags): tag query that selects the visual model of a source-spawned agent, e.g. `human::mobility::wheelchair`. Passed through to the renderer (the env node), humansim never reads it.

Climbs (`climbs:`, optional list among `stairs` and `escalator`, default both): the teleport zone stair kinds the type is routed over, see the [teleport zones](../../../README.md#stairs) section. A pedestrian whose type does not climb a kind takes a route over cabins and other stairs, walks round a stair of that kind on its own level, or stays on its side when no way exists. `wheelchair_manual` climbs `[]`. Any other word rejects the type.

Stair effort (`stair_effort:`, a distribution like `desired_velocity`, default mean 1.6, std 0.5, clipped to 1-3, which are also the clips of a dict that sets none): how many times its walking time a climb on still stairs feels like to a pedestrian of the type, drawn per pedestrian after every other parameter. It decides between still stairs, a running escalator and walking round where a stair joins two floors the pedestrian can also walk between (see the [stairs](../../../README.md#stairs) section): with the default, about two in three of the pedestrians reaching still stairs and an escalator of the same length from the same spot ride the escalator.

Module selection (name strings, not distributions):

- `local_planner` - one of [local_planner/README.md](../../arena_humansim/local_planner/README.md)'s `Available` table (default: launch param, normally `sfm`).
- `global_planner` - one of [global_planner/README.md](../../arena_humansim/global_planner/README.md)'s `Available` table (default: launch param, normally `navmesh`).
- `animation` - one of [animation/README.md](../../arena_humansim/animation/README.md)'s `Available` table (default: `noop`).
- `perception_stack` - tuple of names from [perception/README.md](../../arena_humansim/perception/README.md)'s `Available` table, applied in order (default: `("default",)`).

Every distribution-valued field accepts either a dict `{mean, std, clip_low, clip_high}` or a bare number (shorthand for a fixed value, e.g. `desired_velocity: 1.1`). `std` defaults to `0` (no sampling noise). `clip_low`/`clip_high` default to `0.01`/`inf` and are applied after sampling. Pick them to keep unphysical samples out (e.g. `desired_velocity.clip_low: 0.5` - nobody walks at 5 cm/s).

## Inheritance

```yaml
name: elder_slow
extends: elder
desired_velocity: {mean: 0.5, std: 0.08, clip_low: 0.2, clip_high: 0.8}
```

`extends:` pulls in the parent's full field set, then this file's fields override. Nested structures (`perception`, `local_planner_params`) merge field-by-field, not as whole replacements, and `locomotion` / `pose` merge recursively (a child's `cadence: {max: 1.2}` keeps the parent's other cadence fields). Inheritance chains resolve in the loader, see [../../arena_humansim/core/agents/loader.py](../../arena_humansim/core/agents/loader.py) (`resolve_extends`).

When a scenario references a type by path (`agent_type: ./doctor.yaml`, the normal case for a `dynamic:` entry - see [arena_runtime human README](../../../../arena_runtime/arena_runtime/arena_runtime/humans/README.md)), that single file is loaded on its own and `extends:` can only reach the types shipped here (`adult`, `elder`, `child`, `hurried`, `distracted`, `robot`) - it cannot reach a sibling file in the same scenario's directory. Extending another scenario-local type only works when the whole directory is loaded together (e.g. by tooling that calls `load_agent_types(scenario_dir)`).

## Behavior trees

Set `mode: behavior_tree` to drive the agent from a compiled py_trees BT. `mode: simple` (default) ignores `sequences` / `actions` / `needs` and runs waypoint-only movement.

BT authoring is split into four fields on the agent type: `needs`, `utility_weights`, `actions`, `sequences` (plus `initial_sequence` to pick the entry sequence, defaults to `"default"`). See [../../arena_humansim/core/behavior/README.md](../../arena_humansim/core/behavior/README.md) for the cross-node invariants (patience phases, seek/cancel semantics, compiler dispatch table).

### `needs`

Scalar state in `[0, 100]` that decays every tick and is restored by actions / steps via `satisfies`.

```yaml
needs:
  thirst:
    initial: {mean: 55.0, std: 15.0, clip_low: 25.0, clip_high: 90.0}
    decay_rate: {mean: 8.0, std: 2.0, clip_low: 4.0, clip_high: 14.0}  # units/sec
```

`initial` and `decay_rate` are distribution dicts, sampled per agent. Decay is linear. `satisfies: {thirst: 100.0}` adds 100 on completion (clamped at 100).

### `utility_weights`

Per-need weights used by `AutonomousNode` when a step has `autonomous: true`. Higher weight => need drives action selection more aggressively.

### `actions`

Candidate actions the autonomous selector can pick from. Fields:

| Field | Meaning |
|---|---|
| `when` | `{need: {below\|above: X}}` - preconditions gating the action. |
| `interaction` | One of `TALK_TO`, `GROUP_CONVERSATION`, `GROUP_WALK`, `SIT_ON`, `LIE_ON`, `USE`, `QUEUE_USE`, `WAVE_AT`, `BLOCK`, `SERVICE`, `PUSH_WHEELCHAIR`. Omit for nav-only. |
| `target` | Object id / object type / service tag / agent id, per the interaction's handle kind. |
| `duration` | Distribution (seconds). |
| `patience` | Distribution (seconds) capping the whole action (nav + execute). |
| `satisfies` | `{need: amount}` applied on SUCCESS. |
| `on_failure` | Accepted (default `"skip"`) but not read by the compiler - an action FAILURE has no distinct scripted recovery yet. |

```yaml
actions:
  rest_at_bench:
    target: bench                # object type - resolves to nearest visible
    interaction: SIT_ON
    duration: {mean: 5.0, clip_low: 1.0, clip_high: 15.0}
    satisfies: {rest: 30.0}
```

### `sequences`

Named state machines. Each sequence runs its `steps` in declaration order. On completion the compiler routes via `then` (chain) or `transitions` (need-driven preemption) or `on_failure` (recovery).

```yaml
sequences:
  chat:
    steps:
      hold_conversation:
        interaction: GROUP_CONVERSATION
        duration: {mean: 45.0, std: 15.0}
        patience: {mean: 60.0}
    then: chat                       # loop
    transitions:
      - when: {thirst: {below: 30.0}}
        goto: drink                  # preempt into another sequence
  drink:
    steps:
      queue_and_drink:
        target: fountain             # object id (or type)
        interaction: USE
        duration: {mean: 6.0}
        satisfies: {thirst: 100.0}
    then: chat
```

`transitions` evaluate every tick - they can cut a step short. `then` only fires on the last step succeeding. `on_failure: <seq_name>` routes to another sequence when the current one FAILs (unset => propagate failure to the root). A sequence also accepts `interruptible` (bool, default `true`), which is parsed but not yet consulted anywhere.

`tune:` holds a transform on the agent's own parameters for as long as the sequence runs:

```yaml
sequences:
  evacuate:
    tune:
      desired_velocity: {mul: 1.5}
      max_velocity: 2.6
      perception: {vision_range: {add: -1.0}}
      local_planner_params: {repulsion_strength: {poly: [0.0, 0.0, 2.0]}}
    steps: ...
```

Each entry is a polynomial in the value sampled at spawn: a bare number sets it, `{mul: m}` scales it, `{add: b}` shifts it, both together give `add + mul * x`, and `{poly: [c0, c1, ...]}` evaluates `c0 + c1 x + c2 x^2 ...`. Tunable fields are `desired_velocity`, `max_velocity`, `max_acceleration`, `max_deceleration`, `min_turning_radius`, `pivot_angular_velocity`, `reaction_time`, `personal_space_min`, `perception.vision_range|vision_fov|proximity_sense` and any `local_planner_params` key (a key the planner does not sample can only be set, a transform on it fails when the agent spawns). Unknown fields raise at load.

A step accepts the same `tune:` and holds it for that step only. The agent type accepts a `tunes:` list whose entries hold their tune while a condition on the needs holds, independent of the running sequence, so a stimulus retunes a live agent without restarting its step. Every need named in `when` must be declared under `needs:`.

```yaml
tunes:
  - when: {alarm: {above: 50.0}}
    tune: {desired_velocity: {mul: 1.5}, perception: {vision_range: 2.0}}
```

Tunes stack by owner and fold per field in a fixed order: type condition first, then sequence, then step, then nodes such as the `BLOCK` pursuit boost, so an inner owner always acts on the outer one's result. Each owner releases its own when it ends, and releasing the last one puts the spawn values back. A sequence re-entering itself keeps its tune.

### Stimuli

The `notify_stimulus` service sets the need named by `stimulus` to `100 * intensity` after the agent's sampled `reaction_time`, and, for an intensity above 0, fires an event of the same name on the event bus (an intensity of 0 only lowers the need, so an autonomous step's `until:` does not exit when a stimulus goes away). Only agents whose type declares that need react to the value, so a `transitions` entry keyed on it (e.g. `when: {alarm: {above: 50.0}}`) preempts the current sequence. `agent_id: -1` broadcasts to every agent and is also remembered as world state: an agent spawned later starts its declared needs from the last broadcast intensity of each stimulus (no reaction delay, no event). A full `reset` forgets these levels.

World mechanisms use this channel (Arena adapter, `human:=arena`): every discrete or predicate change of a door, elevator, signal, schedule, gate, plate or sound arrives as the broadcast pair `<entity>.<field>==<old>` at 0 and `<entity>.<field>==<new>` at 1, e.g. `elev_a.cabin_door==open`. Declare such a need with `initial` and `decay_rate` pinned to 0 and key a transition on `above: 50` to wait for a mechanism state. A ped teleported by an elevator gets the per-agent stimulus `teleported` at 1. See `arena_simulation_setup/AUTHORING.md` (Pedestrians and world mechanisms).

```yaml
needs:
  "elev_a.cabin_door==open":
    initial: {mean: 0.0, clip_low: 0.0, clip_high: 0.0}
    decay_rate: {mean: 0.0, clip_low: 0.0, clip_high: 0.0}
sequences:
  wait:
    steps:
      stand: {duration: {mean: 3600.0}}
    then: wait
    transitions:
      - when: {"elev_a.cabin_door==open": {above: 50.0}}
        goto: board
```

### `steps`

Each step is either a `StepDef` (interaction, pure-wait, cancel) or a `GoToStepDef` (explicit `kind: go_to`).

`StepDef` fields:

| Field | Meaning |
|---|---|
| `interaction` | `TALK_TO` / `GROUP_CONVERSATION` / `GROUP_WALK` / `WAVE_AT` / `SIT_ON` / `LIE_ON` / `USE` / `QUEUE_USE` / `BLOCK` / `SERVICE` / `PUSH_WHEELCHAIR`. Omit for a pure-wait step. |
| `target` | Interpreted per the interaction's handle kind: object id/type for `OBJECT`, service tag (str) for `SERVICE`, agent id (int) for `BLOCK`, omit for symmetric types. |
| `offer` | SERVICE provider side. `true` makes this step create-and-wait rather than find-and-join. Required when a SERVICE interaction has no existing provider. |
| `cancel` | `true` => emit STOP with `reason=CANCELED` on the agent's current interaction. Mutually exclusive with `interaction:`. |
| `queueable` | Provider-side override (SERVICE with `offer: true`) - admit seekers into a FIFO queue when full. |
| `min_participants` / `max_participants` | Provider-side overrides on the contract. Count the provider itself. |
| `formation_spec` | Provider-side formation override: `{type, params, anchor_kind, anchor_ref, anchor_pose}`. `type` is one of `line`, `cluster`, `f_formation`, `dyad`, `walk`. `anchor_kind` is one of `object`, `agent`, `provider`, `pose`, `centroid`, `leader` (default `object`). `anchor_ref` names the object/agent to anchor on when `anchor_kind` needs one. `anchor_pose: {x, y, theta}` is used with `anchor_kind: pose`. |
| `duration` | Distribution (seconds). With `interaction:` -> contract-level timeout (outcome `COMPLETED`). On a pure-wait step -> `HoldNode` duration (NAVIGATE-to-self). |
| `patience` | Distribution (seconds). Covers nav + seek + wait-for-ACTIVE + hold. |
| `satisfies` | `{need: amount}` applied on SUCCESS. |
| `wait_for_outcome` | `true` => the interaction node blocks for the outcome even with no `duration:` set (e.g. a SERVICE seeker waiting to be picked up). |
| `autonomous` | `true` => `AutonomousNode` scores `actions` and runs the winner. |
| `allowed_actions` / `blocked_actions` | Filter the autonomous candidate pool. |
| `until` | Event-bus event name that exits the autonomous step on fire (e.g. `"agent_ready"`). |
| `until_need` | Need-condition predicate that exits the autonomous step (e.g. `{rest: {above: 80}}`). |
| `interaction_radius` | Per-step override of the radius cascade: step override, then the target object's own `interaction_radius`, then the interaction kind's default. |
| `on_failure` | Accepted (default `"abort"`) but not read by the compiler - a step FAILURE always propagates to the enclosing sequence's own `on_failure`. |
| `interruptible` | Accepted (bool, default unset) but not read anywhere. |

`offer: true` is only valid for interactions that allow it (`SERVICE`, `PUSH_WHEELCHAIR`). Provider-side fields (`queueable`, `min_participants`, `max_participants`, `formation_spec`) require `offer: true`, or loading fails. Unknown step fields fail to load with the offending key names. A few older field names are explicitly rejected with a migration hint: `accept`, `interaction: FOLLOW`, `target_object_id`, `target_object_type`, `target_agent`, `service_tag` (use `target:` and, for FOLLOW, `interaction: SERVICE` with `offer: true`).

`GoToStepDef` (`kind: go_to`) takes exactly one of:

- `target_pose: {x, y, theta}` - scripted waypoint.
- `target: <str>` - object id/type, the compiler emits `Resolve -> GoTo`.

Plus `duration:` (optional hold on arrival), `patience:`, `satisfies:`, `on_failure:` and `interruptible:` (both accepted, neither read by the compiler).

### Robot step

`kind: robot` gates and drives a step on the nearest robot (an agent of kind robot, fed from the world state). It needs the agent pool, so it only compiles through `BehaviorTreeFactory.build(pool=...)`.

```yaml
sequences:
  idle:
    steps:
      watch: {kind: robot, when: {ahead: 10.0}}   # SUCCESS once the robot is within 10 m ahead
    then: block
  block:
    steps:
      stand: {kind: robot, act: stand, face: robot, gesture: halt, duration: {mean: 30.0}}
```

The frame is the robot's: its heading is its velocity direction above 0.1 m/s, else its pose theta. `ahead: D` holds while the agent is `0 < along <= D` metres in front of the robot, `behind: D` while it is `D >= -along > 0` behind it, `within: D` while the plain distance is at most `D`, and `lateral: L` (default 5) bounds the sideways offset. Every set field must hold. Until the trigger holds the agent stands still. Leave `patience:` off, it also times the wait. A step without `when:` triggers at once.

| Field | Meaning |
|---|---|
| `when` | Trigger `{ahead, behind, within, lateral}`, at least one of the first three. |
| `act` | `wait` (default, SUCCESS at the trigger), `stand`, `approach`, `cross`, `traverse`, `pass`. |
| `face` | Not valid with `act: wait`. `robot` keeps the heading on the robot, a number is a bearing in degrees in the robot frame latched at the trigger. |
| `gesture` | `halt`, `halt_l` or `halt_r`: the stop gesture, published on the `gestures` channel at the robot's position with the agent's handedness for `halt`. Not valid with `act: wait`. |
| `lead` | `approach`: metres walked past the robot. `pass`: metres ahead of the robot aimed at. `traverse`: metres walked toward the robot's position. Default 3. |
| `clearance` | `cross`: minimum metres past the robot's path. `pass`: sideways metres from the robot's path. `traverse`: sideways shift. Default 1. |
| `duration`, `patience`, `satisfies`, `tune` | As on any step. `duration` caps the act, not the wait. |

Acts after the trigger:

- `stand` holds the pose for `duration`, faces and gestures.
- `approach` walks at the robot's current position extended `lead` metres along the line from the agent to it at the trigger. SUCCESS once the agent is a metre past the robot.
- `cross` latches the point mirrored across the robot's path line (the line through the robot along its heading at the trigger) and walks there. SUCCESS once across the line by 0.3 m or at the point.
- `traverse` latches the point `lead` metres from the agent toward the robot's position at the trigger, shifted sideways by `clearance`, and walks there.
- `pass` aims at the point `lead` metres ahead of the robot and `clearance` to the side the agent is on, re-aimed every tick. SUCCESS once the agent is a metre ahead of the robot. Pair it with a sequence `tune:` for the higher speed.

The act ends with SUCCESS when `duration` runs out or the robot disappears. Facing and gestures are released when the step ends.

### Pure-wait step

Omit `interaction:` and `cancel:`, keep `duration:`. Produces `ClearOutcome -> Hold` - `HoldNode` re-emits NAVIGATE-to-self with zero velocity each tick, so the agent parks in place without tearing down any live interaction.

### Cancel step

```yaml
drop: {cancel: true}
```

Emits STOP on the agent's current `interaction_id` with `reason=CANCELED`. Falls back to force_stop (target=-1) if `interaction_id` is `None`. IM no-ops if the agent isn't in any interaction. Mutually exclusive with `interaction:`.

### Meet at an object with a symmetric interaction

Symmetric interactions (TALK_TO, GROUP_CONVERSATION) don't take a `target:`, so "meet at the gathering area" is a two-step pattern: walk first with a `go_to` step, then run the interaction.

```yaml
sequences:
  chat:
    steps:
      walk_to_gathering: {kind: go_to, target: gathering_area}
      hold_conversation:
        interaction: GROUP_CONVERSATION
        duration: {mean: 45.0, std: 15.0}
        patience: {mean: 60.0}
```

### Where this gets compiled

Per-agent trees are built in [../../arena_humansim/core/behavior/compiler.py](../../arena_humansim/core/behavior/compiler.py) from the sampled `AgentType`. To extend the authoring surface (new step fields, new primitive kinds), see [../../arena_humansim/core/behavior/nodes/README.md](../../arena_humansim/core/behavior/nodes/README.md).

## Adding a type

1. Drop a yaml file here (or under a scenario's `agent_types/` to keep it local).
2. Either `extends` an existing type or declare every field explicitly - there's no implicit default.
3. Reference it by file name (stem, without `.yaml`) from scenarios.

## Where this gets sampled

For a `dynamic:` scenario entry, `ArenaHumanDynamicObstacle.sample_params` (in [arena_runtime's arena_humansim adapter](../../../../arena_runtime/arena_runtime/arena_runtime/humans/arena_humansim/__init__.py)) resolves `agent_type` to an `AgentType` - `load_agent_type_from_file` for a path, `BUILTIN_AGENTS[name]` for a builtin - then calls `sample_agent_type(agent_type, rng)`. The RNG is the agent's seeded substream, so the same seed + type + agent_id gives the same sampled instance across runs. See [arena_runtime human README](../../../../arena_runtime/arena_runtime/arena_runtime/humans/README.md) for the full `agent:` block schema.

## `vars` (scenario-embedded types only)

`AgentType` also accepts a `vars:` block (`{name: {type, default, min, max, description}}`) and `${expr}` templating in string fields, letting one inline agent-type definition be re-parameterized per instantiation. This is resolved by `resolve_vars` in [../../arena_humansim/utils/scenario.py](../../arena_humansim/utils/scenario.py), which only runs for agent types embedded inline under `agent_types:` in arena_humansim's own standalone scenario format (`load_scenario`). It does **not** run for the `agent_type: <name>` / `agent_type: ./file.yaml` files loaded by the Arena `dynamic:` pathway above - `vars:` and `${...}` in a file referenced that way are inert.

## Included types

- `adult` - nominal pedestrian (desired 1.1 m/s, 5 m vision, 180deg FOV).
- `elder` - slower, narrower FOV, longer SFM relaxation, and weaker, later avoidance than `adult` (older adults start avoiding about 1 m later and leave most of it to the other person). Demonstrates how heterogeneity drops out of a handful of distribution tweaks.
- `child` - small body (0.18 m), wide speed spread (0.4-1.8 m/s), 3 m vision, weaker SFM repulsion.
- `hurried` - fast (desired 1.7 m/s), narrower FOV, shorter SFM relaxation and repulsion range.
- `distracted` - phone-walker: 2 m vision, 90 degree FOV, longer SFM relaxation. Robot shifts: `detection` -5, `fov` -8, `repulsion` -3, `reaction` -3. On an adult (180 degree FOV, std 10) the `fov` shift gives 100 degrees. The shipped `distracted.yaml` type already starts at 90 degrees, so it takes no `fov` shift, which would count the narrowing twice. It uses `detection` -0.75, `repulsion` -3 and `reaction` -3 instead.
- `blind` - pedestrian oblivious to robots (`robot_awareness: blind`). Human-human social navigation remains 100% active.
- `normal` - standard adult pedestrian with no robot shift (`robot_awareness: normal`).
- `interested` - curious pedestrian watching or approaching the robot. Robot shifts: `repulsion` -3, `reaction` -1.5.
- `aggressive` - assertive pedestrian who cuts across. Robot shifts: `fov` -1, `repulsion` -4, `reaction` -2.
- `scared` - cautious pedestrian who keeps a wide clearance from robots. Robot shifts: `fov` +2, `repulsion` +6, `reaction` +3.
- `sidewalk_obstructing`, `sidewalk_conversing`, `sidewalk_queueing`, `sidewalk_frontal`, `sidewalk_lateral`, `sidewalk_overtaking`, `sidewalk_crossing`, `sidewalk_gesturing` - the eight SidewalkBench pedestrian behaviors, each `extends: adult` in `behavior_tree` mode. A type idles in sequence `idle` on a robot step until its trigger holds, runs the behavior, then parks in `done`.

| Type | Trigger | Behavior |
|---|---|---|
| `sidewalk_obstructing` | `ahead: 10` | stands, facing the robot |
| `sidewalk_conversing` | `ahead: 10` | `GROUP_CONVERSATION` with the peers that fire together |
| `sidewalk_queueing` | `ahead: 10` | stands facing 90 degrees left of the robot's heading, so a lateral line becomes a queue |
| `sidewalk_frontal` | `ahead: 20` | `approach` |
| `sidewalk_lateral` | `ahead: 4` | `cross`, 1.3 times faster |
| `sidewalk_overtaking` | `behind: 6` | `pass`, 1.5 times faster |
| `sidewalk_crossing` | `within: 8`, or the `robot_in_zone` need above 50 | `traverse` 8 m toward the robot |
| `sidewalk_gesturing` | `ahead: 10` | stands, faces the robot, `halt` gesture |

`sidewalk_crossing` declares the need `robot_in_zone`, so a runtime that knows the crosswalk zone can send the stimulus `robot_in_zone` at 1 to fire the crossing when the robot enters it.

- `robot` - fixed (non-distribution) values for a robot-driven agent: zero `min_turning_radius`, 360deg FOV, no `idle_gaze_rate`.
- `wheelchair_manual` - manually propelled wheelchair user: `extends: adult`, `along_heading` kinematics on `hsfm` with `heading_source: total`, a 1.1 m capsule footprint, 1.0 m/s comfortable speed and 1.0 m/s² braking, push-stroke speed modulation (two harmonics), stall recovery with a 0.3 m reverse, seated pose with arm push strokes and the `human::mobility::wheelchair` asset. Every number in the file is a placeholder pending literature values (push cadence, stroke speed profile, footprint), not a measured profile.
- `adult_limp_right` - adult with a right-leg limp: `extends: adult`, uneven cycle timing (`phase_warp.split`), one speed dip and one lateral lurch per cycle, and an asymmetric walk pose (stiff right knee, shortened right hip swing, trunk lean). Every number in the file is a placeholder pending a fitted profile.
