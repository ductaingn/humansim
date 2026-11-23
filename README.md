# Arena HumanSim

Modular, deterministic pedestrian simulator for the [Arena](https://github.com/Arena-Rosnav) framework. Simulates crowds of autonomous agents with realistic navigation, collision avoidance, social interactions, and behavior trees — all running as a ROS 2 node.

## Architecture

```
AgentManager (ROS 2 Node)                    [core/]
├── AgentPool          Vectorized NumPy state for all agents
├── Perception         KDTree neighbor queries + FOV filtering       [perception/]
├── Behavior Trees     py_trees decision making + needs system       [core/behavior/]
├── Global Planner     A* pathfinding on inflated occupancy grid     [global_planner/]
├── Local Planner      Force, velocity-obstacle, learned avoidance   [local_planner/]
├── Animation          Kinematic forward integration                 [animation/]
├── Collision          Wall projection overlap resolution            [collision/]
├── SpawnScheduler     Poisson-process agent spawning at sources
├── DespawnMonitor     Sink-based agent removal with TTL
├── InteractionManager Social interactions (talk, queue, service, …) [core/]
├── EventBus           Event-driven scripting
└── SimulationLogger   JSON replay logging
```

Per-module READMEs: [core](arena_humansim/arena_humansim/core/README.md) · [core/behavior](arena_humansim/arena_humansim/core/behavior/README.md) · [core/behavior/nodes](arena_humansim/arena_humansim/core/behavior/nodes/README.md) · [perception](arena_humansim/arena_humansim/perception/README.md) · [global_planner](arena_humansim/arena_humansim/global_planner/README.md) · [local_planner](arena_humansim/arena_humansim/local_planner/README.md) · [animation](arena_humansim/arena_humansim/animation/README.md) · [collision](arena_humansim/arena_humansim/collision/README.md).

### Tick Loop

Each simulation step follows a fixed pipeline:

1. **Spawn / Despawn** — schedule new agents, remove completed ones
2. **Sense** — build neighbor graph (KDTree + FOV pruning → CSR sparse matrix)
3. **Decide** — tick behavior trees (every N ticks), emit high-level commands
4. **Global Plan** — A* pathfinding with LOS simplification and wall push-back
5. **Local Plan** — local planner velocity computation (vectorized over pool)
6. **Interact** — update social interaction state machines
7. **Kinematics** — enforce acceleration, speed, and turning-radius limits
8. **Animate** — forward-integrate position and heading
9. **Collide** — resolve agent-wall overlaps
10. **Publish** - broadcast the flat `AgentFrame`, latched meta and gestures on change, and the `AgentViz` marker inputs

## Modules

All modules are swappable via a plugin registry.

| Layer | Options | Default |
|---|---|---|
| [Global Planner](arena_humansim/arena_humansim/global_planner/README.md) | `navmesh`, `astar`, `dijkstra` | `navmesh` |
| [Local Planner](arena_humansim/arena_humansim/local_planner/README.md) | `sfm`, `hsfm`, `helbing`, `johansson`, `karamouzas`, `zanlungo`, `gcf`, `orca`, `pedvo`, `straight`, `nsp`, `socialgail`, `neurosfm` | `sfm` |
| [Perception](arena_humansim/arena_humansim/perception/README.md) | `default` | `default` |
| [Animation](arena_humansim/arena_humansim/animation/README.md) | `noop`, `kinematic` | `noop` |
| [Collision](arena_humansim/arena_humansim/collision/README.md) | `wall_projection`, `noop` | `wall_projection` |
| Occlusion | `bitmap`, `noop` | `bitmap` |

Each module is selectable as a ROS parameter and as a launch argument of the same name (e.g. `local_planner:=socialgail`). Scenario YAMLs do **not** carry a `modules:` block — module choice is a runtime decision, not a property of the scenario.

## Agent Types

Defined in YAML under [`config/agent_types/`](arena_humansim/config/agent_types/README.md). Each type specifies distributions over physical and behavioral parameters:

```yaml
name: adult
desired_velocity: { mean: 1.1, std: 0.12, clip_low: 0.5, clip_high: 1.5 }
agent_radius: { mean: 0.25, std: 0.02 }
perception:
  vision_range: { mean: 5.0, std: 0.5 }
  vision_fov: { mean: 180.0, std: 10.0 }
local_planner_params:
  relaxation_time: { mean: 0.5 }
  repulsion_strength: { mean: 2.1 }
needs:
  hunger: { initial: { mean: 100 }, decay_rate: { mean: 0.5 } }
```

Types support inheritance via `extends`. Parameters are sampled per-agent from their distributions, producing a heterogeneous crowd from a single type definition.

**Included types:** `adult`, `elder`

## Behavior Trees

Agents can operate in two movement modes:

- **Waypoint** — follow an explicit waypoint list (repeat / reverse / once / random)
- **Behavior Tree** — py_trees decision tree driven by needs, perceptions, and events

BTs are compiled from the agent type's `sequences`, `actions`, and `needs`. Needs decay over time and trigger actions when thresholds are crossed (e.g. hunger < 30 → eat).

**Authoring the YAML:** [`config/agent_types/README.md`](arena_humansim/config/agent_types/README.md#behavior-trees) documents `needs`, `actions`, `sequences`, `steps`, `transitions`, `tune` (transforms on the agent's parameters held by a sequence, a step or a need condition), cancel steps, and autonomous steps with full field tables and examples.

**Internals / extending primitives:** [`core/behavior/README.md`](arena_humansim/arena_humansim/core/behavior/README.md) covers cross-node invariants (patience phases, seek/cancel semantics, compiler dispatch). [`core/behavior/nodes/README.md`](arena_humansim/arena_humansim/core/behavior/nodes/README.md) covers adding new `py_trees.Behaviour` primitives.

## Interactions

Matcher semantics (seek dispatch by handle kind, visibility gating, queueing, service binding) live in [`core/README.md`](arena_humansim/arena_humansim/core/README.md). Per-type defaults (handle kind, contract, formation, `allows_offer`, `interaction_radius`) are centralized in [`core/interaction_kinds.py`](arena_humansim/arena_humansim/core/interaction_kinds.py). The `interaction_radius` cascade and per-step field reference are documented in [`config/scenarios/README.md`](arena_humansim/config/scenarios/README.md).

| Type | Handle | Participants | Description |
|---|---|---|---|
| `TALK_TO` | NONE | 2 | Face-to-face conversation |
| `GROUP_CONVERSATION` | NONE | 2+ | Multi-agent group talk |
| `GROUP_WALK` | NONE | 2+ | Group walking with its first participant: abreast on the leader's trail, a V trailing back from the leader in moderate crowds, single file in dense crowds or where walls leave no width, circling up when the leader stops (`circle_on_stop`). Bind members at spawn with an `interaction_scripts` entry |
| `WAVE_AT` | NONE | 2 | Symmetric greeting |
| `SIT_ON` / `LIE_ON` | OBJECT | 1 | Occupy furniture (FIFO queue) |
| `USE` | OBJECT | 1 | Use a world object (FIFO queue) |
| `QUEUE_USE` | OBJECT | 1+ | Queue for a shared resource |
| `BLOCK` | AGENT | 1-2 | Pursue and block a target agent |
| `SERVICE` | TAG | 1+ | Asymmetric provider/seeker pairing by tag (also subsumes escort/follow via `formation_spec: line, anchor_kind: provider`) |
| `PUSH_WHEELCHAIR` | TAG | 2 | A wheelchair user offers it (`offer: true`), a pusher seeks it, walks to the push handles (formation `push`) and steers the pair while the chair rides at her hands ([`core/carry.py`](arena_humansim/arena_humansim/core/carry.py)); bound and published only once she holds the handles; `cancel` lets go and the chair stays where she left it. Example: [`config/scenarios/push_wheelchair.yaml`](arena_humansim/config/scenarios/push_wheelchair.yaml) |

Every interaction follows a single flow: a BT `SeekNode` emits a SEEK command, IM finds a matching open interaction and joins, or (if creation is allowed for this handle) creates one. Teardown is explicit — either a BT `CancelNode` (STOP with `reason=CANCELED`) or the contract's duration expiring (`COMPLETED`). Agents in active interactions defer despawn until the interaction ends.

## Spawning & Despawning

- **Sources** come in two types. `type: poisson` (default) emits at the `rate_profile` and loses arrivals over `max_concurrent`. `type: max` holds `max_concurrent` agents alive, refilling on despawn, and takes no `rate_profile`. Both respect `max_total`.
- **Sinks** absorb the agents released for them once inside (within `absorption_radius`, or the polygon) and heading for them: a waypoint walker whose current waypoint lies in the sink, a behavior-tree agent whose moving navigate target lies in it. One passing through or standing in it stays
- Agents are routed from source to sink via weighted `sink_affinity`
- TTL-based removal as fallback

### Pedestrian lanes

A source with a `lane` block walks a centerline instead of aiming at the sink. Each walker gets its own track: the centerline offset by a truncated-normal fraction `q` of the band that the free space allows.

- `side` is required: `right`, `left` or `center`. There is no default.
- `offset` (meters, positive to the left of travel) is optional. When given, the band centers on it, bounded by the walls. The message carries `has_offset`, so `0` is a valid offset. Without an offset, the band centers on the free space on `side`.
- Every other setting is `0` in the message for derived. `track_spacing` comes from `chord_tolerance` and is capped at 3 m. `arrival_radius` is a third of the spacing. `spawn_length` is the source extent along the lane minus the arrival radius. `sweep_length` defaults to the spacing. `merge_length` comes from the band at each junction.
- `sigma_q` (default 0.5) and `chord_tolerance` (default 0.05 m) set the spread and the track.
- A lane flow needs exactly one sink affinity. Its end band must lie in that sink: inside the polygon, or within `absorption_radius` of a circle sink.
- A wall or obstacle change always applies. A lane source whose lane no longer builds on the new walls stops spawning and resumes once it builds again.

## Teleport Zones

A teleport zone is a floor area, in the engine frame, that pedestrians wait at, enter and leave by teleport: a name, the zones a teleport from it can land in, an entry pose at the outer edge of its opening facing in, a wait pose clear of the opening, a hold pose inside whose heading holders take, the footprint polygon, whether it is open, a delay, a `kind` and a `belt` speed. An empty `kind` makes the zone an elevator cabin, `stairs` or `escalator` makes it a stair half (see Stairs below). The engine never moves a zone. `set_teleport_zones` replaces every zone, `set_zone_access` opens and closes zones by name, and `teleport_agents` moves pedestrians. Zones also come from the scenario's `teleport_zones:` list, poses and footprint points written as `{x: .., y: ..}` maps like wall ends, `open` defaulting to `true`, `delay` to `0`, `destinations` to none, `kind` to a cabin and `belt` to `0`:

```yaml
teleport_zones:
  - name: lobby
    destinations: [roof]
    entry: {x: 4.0, y: 4.6}
    wait: {x: 3.4, y: 4.6}
    hold: {x: 5.2, y: 4.6, theta: 0.0}
    footprint: [{x: 4.6, y: 4.0}, {x: 5.8, y: 4.0}, {x: 5.8, y: 5.2}, {x: 4.6, y: 5.2}]
    open: false
    delay: 2.0
```

Zones survive `reset` with `soft: true` and are dropped by a plain `reset`, scenario zones included, and by an empty `set_teleport_zones`. A `set_teleport_zones` keeps the transfers through cabins that come back unchanged and through stair halves that come back as stair halves listing the same destination, and drops the others.

A pedestrian whose navigation target lies in another free-space component of its global planner than itself is routed over the zones: a zone is entered from the components of its entry and hold pose and lands in those of each of its destinations, and the route with the fewest teleports wins. Among those, a pedestrian keeps the first zone it already walks to, and otherwise takes the one whose entry is nearest to it, ties going to the zone and then the destination first by name. Only the first zone of the route and the destination it takes are kept, and the pedestrian plans to the zone's wait pose instead of its target. Robots and pedestrians whose target lies in their own component plan to it directly. `navmesh` labels components on its edge graph, `astar` and `dijkstra` by a flood fill at `global_planner.resolution` 0.1 or finer, and coarser grids route no one over zones.

A rider walks to the wait pose and within 1 m of it waits, desired speed 0, while the zone is closed or has no free spot. The spots are a 0.7 m grid inside the footprint, 0.35 m clear of its edges, or the hold pose when none fits. Once the zone is open and a spot is free the rider takes the free spot farthest from the entry, walks to it and stands still, velocity 0, from the moment it is within 0.3 m of it. A waiting rider first lets the zone empty: while a pedestrian or robot that is not waiting for, walking into or holding in that zone stands inside it, such as a rider that just arrived, the rider keeps waiting for up to `zone_let_out_timeout` seconds of the zone being open, then boards past whoever stayed. The count restarts when the zone closes. A spot that a robot standing in the zone overlaps (closer than the sum of both radii) is not free, and a rider walking in whose spot a robot takes moves to another free spot or back to waiting. A zone that closes before the rider is inside the footprint sends it back to waiting, and a holding rider pushed out of the footprint while the zone is open walks in again.

A zone with `delay` 0 or more teleports a rider itself once it has held for `delay` seconds, at once for 0: the rider lands in the destination of its route at the same offset from that zone's center as from its own, the center being the footprint centroid or the hold pose without a footprint, keeping its heading with velocity 0. A negative `delay` leaves the teleport to the caller. Either teleport ends the transfer, and on the next BT tick the pedestrian plans from its new pose to its current target, over further zones when needed. A rider still waiting, walking in or holding `zone_wait_timeout` seconds after it reached the wait pose or took its spot leaves that zone: it is routed over zones that avoid every zone it left for the same target when a route exists, else over zones that avoid the last one it left, and otherwise waits for the same zone again. A pedestrian is never steered at a target in another component while a zone route to it exists.

### Stairs

A stair or escalator between two levels laid out side by side is two stair halves, one per level, each open at its mouth and walled on both sides and at the seam. Its entry is the mouth center facing into the stair, its hold the seam trigger point, the footprint the half's polygon (a stair without one is rejected), and its destinations the partner half or none for a half that can only be left. `belt` is the speed in m/s at which the half carries whoever stands on it along entry to hold, negative towards the entry, `0` for stairs and a stopped escalator.

A pedestrian is routed into a stair half like into a cabin, never into a closed half, a half without destinations or a half of a kind its agent type does not `climbs` (see [agent types](arena_humansim/config/agent_types/README.md)), and plans to its hold line, at the point halfway from the hold pose to the half's right side and 0.5 m off that side at least, so the two directions of one stair walk in, land and walk out on opposite sides. On a half too narrow for that it plans to the hold pose. Once inside the footprint it keeps that route even when the half closes. Every pedestrian standing on a half with a running belt stands: its desired speed drops to 0, its velocity is the belt's, it reports the `IDLE` animation state and its gait cycles as at rest. On a half with belt 0 walking is capped at 0.7 m/s.

Two stair halves can also join two floors of one level (an Arena stair whose `to` is its own level, in a scan whose floors are ramps): the halves meet in place, each ending where its partner begins, with no wall at the seam. The engine finds such a pair by its footprints (the point 0.05 m past a half's far edge lies in a half it leads to, or that leads to it) and hands the seam to the global planners as a wall, which no local planner and no collision check gets. So no plan crosses such a stair, a pedestrian crosses it only on a stair route, and the rules above hold on it: never into a closed half, a half that can only be left or a kind its type does not climb. On its route the pedestrian plans to a point 1 m short of the seam (the middle of a shorter half) at the side offset it had when it took the route, kept 0.5 m off the half's sides (the center line of a narrower half), so riders keep their lines and pass the ones coming the other way. The point stays where it is for the route: one that followed the pedestrian would be planned anew every tick. From the moment it stands on the half its subgoal is the point straight on, 1 m past the far half's mouth at its own side offset, until it has left the far half, so a pedestrian the crowd pushes back over the seam walks on over it again. Nothing teleports it. A pedestrian with no stair route whose plan would end on a half it is not routed into, closed, one that can only be left or of a kind it does not climb (the point nearest a goal beyond the stair) keeps its own pose as subgoal instead.

Fewest teleports cannot choose between such stairs, as their two floors are one area wherever a walk joins them, so time does. When the pedestrian needs no route, or its route starts on such a pair, it takes the first stair of the quickest way over walks and stair rides: walking to a boarding half's mouth, along both halves, and from the far mouth on to the goal, or, where the far mouth cannot walk on to it, as on a landing between two flights, over the stairs it can walk to. The walks are timed along `GlobalPlanner.paths` at the desired speed, a ride at the belt's speed or at the stair speed (below) times the type's `stair_effort` draw. Only its own walk is planned first: a ride that cannot lie on a quicker way even with every walk to and from it a straight line is left out, so a pedestrian far from every stair plans one path. It walks when walking alone is quicker, which it keeps for that goal until the zones are set again or a half opens or shuts, and chooses again where a ride ends. A ride it does not yet stand on is given up, and chosen anew, when its half shuts or when the crowd pushed the pedestrian to where no walk leads to the half, as from a landing back over the flight it came by.

A pedestrian that reaches the hold line along a half whose partner stands elsewhere, routed through it or not, lands on the partner half 0.1 m short of its hold pose with the same side offset relative to its travel direction and the same heading relative to the stair, velocity 0, and its transfer ends. The side offset falls back to the partner's center line when it would leave the partner's footprint. While another agent or another landing of the same tick overlaps the landing spot, the pedestrian stands still on the hold line, put back onto it when pushed past, and tries again the next tick. Two held pedestrians that cross one stair opposite ways, one of them on the other's landing spot, change places instead, each taking the exact spot the other leaves, which two-way traffic on one stair would otherwise hold for good. The pedestrians queued behind them do not prevent it, and the larger of the two changes places only where it overlaps nobody else. A pedestrian is not teleported from the half it landed on before it has left that half or been routed back over it, so one the crowd pushes back past the hold line stays on its side. Robots and externally driven agents are never routed, carried or teleported by a stair.

## Determinism & Replay

The simulator is fully deterministic given a seed. Seeded RNG substreams are maintained per agent and per component, and each substream's seed is derived by hashing `(seed, name)`, never by drawing from a parent stream, so a substream does not depend on which other substreams were requested before it. An agent's substreams (parameters, behavior, random waypoints, interaction order) are keyed on its id, or on `(source, release index)` for a source-released agent, and no stream is shared between agents. Sources sample each released agent (pose, speed and sink) from a `(seed, source, release index)` stream, so two runs with the same seed agree on every agent they both release. How many they release can differ for a `poisson` source under its cap, never for a `max` source. Humans take ids counting up from 1, robots ids counting down from -1, and 0 names no agent, so a robot shifts no pedestrian id, but once a cap drops a release in one run and not in another, the later agents of every source are numbered differently. `agent_meta` names the source (`source_name`) and release index (`source_seq`) of every source-released agent, empty and 0 for any other, so scorers pair a released pedestrian across runs and seeds by `(source_name, source_seq)`, the key of its draws, never by agent id. Exported `agent_states` tables carry both columns. Neighbor lists and contact pairs are ordered by agent id, so a robot changes only the pedestrians it reaches, directly or through other pedestrians.

`SimulationLogger` writes per-tick JSON snapshots. `ReplayManager` replays them tick-by-tick and validates state within floating-point tolerance (1e-12).

## Usage

### Launch

```bash
ros2 launch arena_humansim arena_humansim.launch.py \
  scenario:=queue \
  local_planner:=socialgail \
  markers:=2 \
  rviz:=true
```

Every node param that the launch file only forwards (`seed`, `ticks`, `time`, `rtf`, the module selectors, `robot_policy`, `robot_shutdown`, `force_waypoint_mode`, `force_local_planner`, `strict_recording`, `trial_id`, `frame_id`) is a launch argument generated from its `Param` in `arena_humansim/params.py`, with the same name, description and choices. Empty means the node default from the parameters table below. `mode`, `scenario`, `markers`, `record`, `record_dir`, `render`, `render_format`, `rviz`, `namespace` and `use_sim_time` are launch-side arguments. Any other launch arg is forwarded to the node as the param of the same name, e.g. `global_planner.resolution:=0.1` or `local_planner.relaxation_time:=0.7`. The value is typed by YAML rules, so write a double with a decimal point. A name the node does not declare is ignored.

Scenarios (world objects, agents, flow, walls) are authored under [`config/scenarios/`](arena_humansim/config/scenarios/README.md).

### Run node directly

```bash
ros2 run arena_humansim arena_humansim_node \
  --ros-args -p mode:=master -p seed:=42 -p dt:=0.05
```

### Evaluation

`ros2 run arena_humansim evaluate <verb>` (or `python3 -m arena_humansim.utils.evaluation.cli <verb>`):

| Verb | What it does |
|---|---|
| `benchmark` | Parallel sweep over scenarios x drivers (x robot policies) x seeds, with an integrity gate per trial. |
| `analyze` | Pairwise trajectory Hausdorff, the class divergence ratio K with scenario-clustered CIs, kinematics, robot metrics, failure causes. |
| `partitions` | K per bucket under every driver-class partition (fine, binary, with and without `straight`). |
| `plots` | Regenerates the paper figures from the analysis CSVs. |
| `eth` | Speed, clearance and turn-rate distributions per driver against the ETH (EWAP) pedestrian dataset, and optionally one ATC day file, on a 0.4 s grid. |
| `correspond` | Released drivers against public references on held-out scenarios: `orca` vs Python-RVO2, `sfm` vs pysocialforce, `straight` vs closed form. |
| `verify` | Retroactive integrity check of a sweep dir. |

### Benchmark

```bash
ros2 run arena_humansim benchmark
```

Config format and stage semantics: [`config/benchmark/README.md`](arena_humansim/config/benchmark/README.md).

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `mode` | `master` | `master` owns `/clock` and ticks on a wall timer, `subsystem` ticks on `/clock`, anything else is rejected |
| `seed` | `0` | RNG seed for deterministic runs |
| `dt` | `0.05` | Simulation timestep (s) |
| `bt_tick_interval` | `5` | Behavior trees tick once every N sim ticks, 1 to 100000 |
| `perception` | `default` | Perception module name |
| `global_planner` | `navmesh` | Global planner module name |
| `global_planner.inflation_radius` | `0.38` | Wall clearance of planned paths (m), rounded up to whole grid cells |
| `global_planner.resolution` | `0.2` | Planning grid cell size (m), grid planners only |
| `global_planner.comfort_radius` | `0.6` | Preferred wall clearance of navmesh paths (m), relaxed in narrow passages, navmesh only |
| `local_planner` | `sfm` | Default local planner module name |
| `local_planner.<key>` | `0` | Mean of a local planner param (`relaxation_time`, `repulsion_strength`, `repulsion_range`, `anisotropy`, and for `hsfm` `lateral_gain`, `lateral_damping`, `angular_gain`, `angular_damping`). `0` leaves the agent type's own value |
| `animation` | `noop` | Animation module name |
| `collision` | `wall_projection` | Collision resolver module name |
| `occlusion` | `bitmap` | Occlusion module name |
| `frame_id` | `map` | TF frame stamped on every published pose, wall and marker, the frame the engine runs in |
| `waypoint_threshold` | `0.1` | Distance (m) at which a waypoint counts as reached |
| `min_speed_for_heading` | `0.1` | Speed (m/s) below which an agent keeps its heading |
| `arrival_r_enter`, `arrival_r_exit` | `0.15`, `0.30` | Radii (m) at which an agent latches onto its goal and releases it |
| `arrival_tau_brake` | `0.15` | Braking time constant (s) of a latched agent, at least `dt` |
| `zone_wait_timeout` | `30.0` | Sim time (s) after reaching a teleport zone's wait pose or taking a spot before a rider not yet teleported leaves the zone and is routed again |
| `zone_let_out_timeout` | `10.0` | Sim time (s) of an open zone that a waiting rider leaves to those getting out before it boards anyway, `0` boards at once |
| `profile_phases`, `profile_interval` | `false`, `0` | Time the tick phases, log the profile every N ticks |
| `publish_markers` | `0` | RViz marker level: 0=off, 1=infrastructure+labels+interactions, 2=full |
| `log_dir` | `""` | Directory for simulation replay logs, empty = no logging |
| `replay_mode` | `""` | Replay log to load and play back, empty = no replay |
| `force_local_planner` | `false` | Force every agent onto `local_planner` and ignore the per-agent policy of the scenario |
| `robot_policy` | `""` | Policy that replaces the policy of every robot-kind agent at scenario load, empty = keep the scenario policy |
| `robot_shutdown` | `""` | `true` or `false` overrides `scenario.simulation.robot_shutdown`, empty = use the scenario value |
| `force_waypoint_mode` | `""` | `once`, `repeat`, `reverse` or `random` replaces the waypoint mode of every human scenario agent at load, empty = keep the scenario value |
| `ticks`, `time` | `0`, `0.0` | Stop after N ticks or after this many sim seconds (ignored when `ticks` is set), `0` = no limit |
| `rtf` | `1.0` | Real-time factor of the master tick timer, 0 to 1000, `0` = unthrottled |
| `subsystem_overrun_policy` | `lag` | Tick overrun policy in subsystem mode, only `lag` is implemented |
| `contact_mode` | `enabled` | `enabled` or `locomotion_only`, how contact interactions play out |
| `locomotion_standing_distance` | `1.2` | Standing distance (m) of a locomotion-only contact pair |
| `gesture_mode` | `enabled` | `enabled` or `disabled`, whether agents play gestures |
| `record_bag`, `record_dir` | `false`, `""` | Record the run to a rosbag in `record_dir`, empty = default recording directory |
| `strict_recording` | `true` | Abort the trial when a second publisher appears on a recorded contract topic |
| `trial_id` | `""` | Opaque id written into the recording manifest |
| `scenario` | `""` | Scenario name or absolute YAML path, master mode only |

An integer set on a double param is widened, so `ros2 param set <node> global_planner.resolution 1` is accepted.

#### Runtime reconfiguration

A `set_parameters` call on a running node is validated and stored at once and takes effect at the next `reset` service call. `reset` with `soft: true` is the episode boundary. It keeps agents, world, clock, and RNG, applies the stored params, re-seats the agents, and prunes unused planners. A plain `reset` wipes the simulation first and then does the same. Every param outside the table is fixed at startup, and a set on it is rejected with `not reconfigurable at runtime`. A `set_parameters_atomically` batch is sorted by name. It may name a planner and that planner's `local_planner.<key>` params in any order, and it is rejected as a whole when one entry is invalid.

| Parameter | Effect at the reset |
|---|---|
| `global_planner.inflation_radius`, `global_planner.resolution`, `global_planner.comfort_radius` | The planner rebuilds its grid or mesh from the current walls and drops cached paths |
| `global_planner`, `local_planner` | The module is created on first use and gets the current walls and pool. An unknown module name is rejected at the set |
| `local_planner.<key>` | Mean of a local planner param (`relaxation_time`, `repulsion_strength`, ...). It replaces the mean of the agent type and keeps its spread. `0` leaves the agent type's own value. Declared for the keys of the startup planner and of every planner selected through `local_planner` |
| `waypoint_threshold`, `min_speed_for_heading`, `arrival_r_enter`, `arrival_r_exit`, `arrival_tau_brake`, `zone_wait_timeout`, `zone_let_out_timeout`, `force_local_planner`, `profile_phases`, `profile_interval` | The node re-reads them. `arrival_r_enter` must stay below `arrival_r_exit` |
| `publish_markers` | The marker level changes. `0` deletes the published markers |
| `seed` | Nothing at a soft `reset`, which keeps the RNG. The next plain `reset` or `remove_agents` with no ids re-roots every stream on it, so the next crowd draws as a first crowd with that seed does |
| `rtf` | Master mode only, retimes the tick timer |

Re-seating moves every live agent that follows the defaults onto the current local and global planner and shifts its planner params by the change of the mean, so it keeps its sampled offset. Three things stay fixed: an explicit `policy` of the spawn request, a planner named by the agent type, and a per-agent planner value. Pruning drops every local and global planner that no agent uses from the policy table, the wall and pool subscribers, and the module pool. The current defaults stay.

### ROS Interface

**Publishes:**
- `agent_states` (`AgentFrame`) - per-tick positions, velocities, radii, kinds, animation states and policy indices of the engine's own agents, one array per field
- `agent_meta` (`AgentMeta`, latched) - policy table plus name, handedness, agent type, releasing source and release index per agent, and the robot awareness each agent got (range offset in m, field-of-view offset in degrees, repulsion and reaction gains, blind flag), sent when it changes and at every reset, stamped like the frame of the same tick
- `agent_gestures` (`AgentGestures`, latched) - active gestures with their owner, sent when they change
- `interactions` (`InteractionFrame`, latched) - every FORMING or ACTIVE interaction with its type, outcome, anchor object, arrival and hold state, duration and members (participants, then the queue in FIFO order, each with its `MembershipRole` and object seat index, -1 for none), plus the lifecycle edges (`InteractionEvent`) fired on that tick. Sent when any of it changes or an edge fires, stamped like the frame of the same tick. An agent can be a member of several interactions
- `viz_state` (`AgentViz`) - per-tick marker inputs while `publish_markers >= 1`, drawn into `viz` by `arena_humansim_viz_node` (started by the launch file, its `output_topic`, `offset_x` and `offset_y` parameters retarget and shift the markers at runtime, `robot_bodies: false` drops the body and heading of robot-kind agents). One message with `level: 0` clears the markers when the level drops to 0. The engine builds the per-tick message only while the topic has a subscriber, and `arena_humansim_viz_node` subscribes only while its output topic has one (checked once per second)
- `viz`, `viz_static/<bucket>` (`MarkerArray`) - infrastructure and module markers drawn by the engine itself. Module markers are drawn only while `viz` has a subscriber

**Subscribes:**
- `world_state` (`AgentStates`) - poses of externally driven robots and possessed pedestrians. A robot fed without a velocity takes the one between its last two poses, from their header stamps: zero when the stamps are more than 0.2 s apart or do not advance, or when the pose jumped faster than 10 m/s. It keeps that velocity for 0.2 s after the last message. A robot declared in the scenario moves by the feed alone while it is fed, and its own policy takes over again 2 s after the last message
- `/clock` (subsystem mode) drives the tick: every message runs the ticks its sim time has covered since the epoch, so a held clock cannot starve the engine

**Services:**
- `spawn_agents`, `remove_agents` — direct agent control
- `add_source`, `remove_source`, `add_sink`, `remove_sink` — flow control
- `add_walls`, `remove_walls` — dynamic obstacles
- `set_teleport_zones` - replace the teleport zones, an empty list removes them all. Duplicate names, a destination naming no zone, an unknown `kind` or a stair without a footprint reject the call
- `set_zone_access` - open or close teleport zones by name, an unknown name rejects the call
- `teleport_agents` - move agents to new poses with zero velocity, ending a rider's transfer. An unknown id rejects the call and moves no one
- `set_flow` — bulk configure sources, sinks, walls. A source's `agent.robot_awareness` text is a preset name, or JSON for a shift mapping or a weighted list; empty keeps the agent type's own awareness
- `notify_stimulus` - drive a need on one agent (or `-1` for all) after its reaction time, a broadcast level also seeds agents spawned later
- `reset` - clear all simulation state, or with `soft: true` keep it and only apply stored params, re-seat agents, and prune unused planners

## Development

Test layout (unit / contracts / integration / ros / perf / replay): [`arena_humansim/tests/README.md`](arena_humansim/tests/README.md).

### Linting

Linting is handled by [Ruff](https://docs.astral.sh/ruff/), driven by [pre-commit](https://pre-commit.com/). Config lives in `arena_humansim/pyproject.toml`; the hook pin is in `.pre-commit-config.yaml`. Auto-formatting is intentionally not enforced.

**One-time setup:**
```bash
pip install pre-commit
pre-commit install
```

**Everyday use:** hooks run automatically on `git commit` against staged files. To run manually:
```bash
pre-commit run            # staged files only
pre-commit run -a         # entire repo
ruff check arena_humansim # check without pre-commit
```

If the hook auto-fixes something, the commit is aborted and the fixes are left unstaged — `git add` and re-commit.

### CI

`.github/workflows/lint.yml` runs the same pre-commit hooks on every push to `master` and every pull request. The GH check uses the exact config and hook pins from `.pre-commit-config.yaml`, so local and CI never drift. Make the check required in branch protection to block merges on lint failures.

Bump the Ruff version with `pre-commit autoupdate`.