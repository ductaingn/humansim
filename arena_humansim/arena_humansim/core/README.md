# core

The simulator's central subsystems live here: the agent pool, the tick-scheduling `AgentManager`, behavior-tree compilation, interaction routing, and shared access/formation primitives. Most of the state that survives across ticks is owned by something in this directory.

This README focuses on **`InteractionManager`** - the only piece of core with non-obvious behavior that isn't already covered by a sibling readme. See also:

- [behavior/README.md](behavior/README.md) - behavior-tree node invariants (`SeekNode`, patience, cancel).

## InteractionManager

Router over `NAVIGATE` / `STOP` / `SEEK` commands. `NAVIGATE` is ignored here (consumed upstream by the locomotion layer); `STOP` drops an agent from one interaction or force-stops across all; `SEEK` is the only matcher entry point.

## `_handle_seek` flow

Each `SEEK` carries a `SeekSpec`. Dispatch consults `spec.interaction_type.kind.handle.strategy` (from the per-type registry in [interaction_kinds.py](interaction_kinds.py)):

1. **Already a member?** Iterate `_agent_membership[agent_id]`; if any existing interaction passes `strategy.matches(interaction, spec)`, return - no duplicate membership.
2. **Find existing.** Only seekers (`spec.offer=False`) enter this branch - providers always create, never join another provider's offer. `strategy.find(self, spec, agent_id)` returns an admissible interaction id or `None`. On hit, `accept(agent_id, iid)` joins (as `PARTICIPANT` or, if the access policy queues, `QUEUED`) and stamps `spec.duration` into `member_durations`.
3. **Create.** If `strategy.can_create(self, spec, agent_id)` passes, `_create_interaction(creator_id=agent_id, spec=spec)` runs. `OBJECT` handles first resolve `spec.target` through `WorldKnowledge.resolve` so the created interaction keys on a concrete `object_id`.

Within a single `update(dt)` call, commands are sorted so `spec.offer=True` dispatches before seekers. This guarantees a same-tick seeker can match a provider's freshly-created interaction instead of racing past it.

## Handle kinds

Four `HandleKind`s, one strategy each, all in `interaction_kinds.py`:

| Kind | Target shape | `find` scan | `populate_state` writes |
|---|---|---|---|
| `NONE` (symmetric: TALK_TO, GROUP_CONVERSATION, GROUP_WALK, WAVE_AT) | omitted | `_scan_symmetric` - visible peer of same type | - |
| `TAG` (SERVICE) | `str` tag | `_scan_tag` - visible provider with matching `service_tag`; requires `offer=True` to create | `state["service_tag"]` |
| `AGENT` (BLOCK) | `int` agent id | `_scan_agent` - interaction with matching `target_agent` | `state["target_agent"]` |
| `OBJECT` (SIT_ON, LIE_ON, USE, QUEUE_USE) | `str` object or type | `_find_object_bound` - `_interaction_by_object_type[(object_id, type)]`, then retry via `WorldKnowledge.resolve` | returns `object_id` (stored on `InteractionState.object_id`, not `state[]`) |

`SERVICE` offers additionally set `state["provider"] = creator_id` when `spec.offer` is true.

## Hot-path indexes

Maintained in `_add_membership` / `_drop_membership` / `_create_interaction` / `_teardown`:

- `_interactions_by_type: dict[int, set[int]]` - all live interaction ids keyed by `int(InteractionType)`; every `_scan_*` iterates one bucket.
- `_interaction_by_object_type: dict[tuple[str, int], int]` - the single active interaction for an `(object_id, type)` pair. Cleared on teardown so a new seeker can create a fresh one.
- `_agent_membership: dict[int, dict[int, MembershipRole]]` - per-agent map of interaction id -> `PARTICIPANT | QUEUED`. Replaces the older `_agent_to_interactions` / `_agent_to_queues` split; `_iter_membership(aid, role=...)` filters.

## Contract duration

`_tick_durations(dt)` advances `contract.elapsed` only while the interaction is `ACTIVE` and - when a formation is attached - every participant satisfies `formation.arrived(pid)`. On expiry:

- With a queued `access` policy and a non-empty queue, current participants are released via `_release_participant`, the policy promotes the next batch, and `contract.elapsed` resets - letting `duration` stretch across successive holders without tearing the interaction down. A per-member override in `member_durations` becomes the next `contract.duration`.
- Otherwise, `_teardown(iid, InteractionOutcome.COMPLETED)` fires.

`_teardown(outcome)` clears indexes, drops memberships, leaves formations, and propagates outcome (`COMPLETED` / `CANCELED` / `INTERRUPTED`) to every ex-member's BT movement.

## Drift eviction

`_tick_drift_eviction` runs on every formation tick, immediately before `_tick_formations`. For each `ACTIVE` interaction it computes a per-participant proximity check and `stop(aid, iid, INTERRUPTED)`s any participant past the threshold:

- **Object-bound** (`kind.is_object_bound`): distance from the participant to its `Formation.slot_of(...)` when the formation assigns explicit slots, otherwise to `WorldKnowledge.object_pose(interaction.object_id)`. A seat can sit further from the object origin than the threshold, and the participant is pinned to it.
- **Otherwise** (`NONE` / `TAG` / `AGENT`): distance from the participant to its nearest other participant. Skipped while the interaction has fewer than two locatable participants.

Threshold is `kind.interaction_radius * cohesion_multiplier` (`InteractionManager.__init__`, default 1.2). Same radius as request-time proximity, scaled up slightly for engagement-bubble slack - an evicted agent is past the seek-match boundary by construction, so a re-seek on the next BT tick can't immediately rejoin without first walking back into range.

A per-participant latch (`interaction.state["_drift_arrived"]`) gates eviction: a participant is only evictable after first being observed within the threshold. "Drift" applies to participants who arrived and then left, not to ones still inbound - without this latch, a freshly-`ACTIVE` interaction whose formation hasn't pulled members in yet would self-evict on tick 1. Object-bound participants latch only within half the threshold of their slot, so a member still closing on its slot (a queue promotion to the front) is not evicted by jostling at the threshold, and a latched member whose slot moves by more than that (the line shifting up) unlatches until it reaches the new slot.

Eviction surgically drops the offender from one interaction; if the participant count falls below `min_participants`, `stop()` cascades into `_teardown(INTERRUPTED)` as usual.

## BT state propagation

All BT side-effects funnel through one helper:

```python
_update_bt_movement(aid, *, interaction_id=..., clear_command=..., last_outcome=...)
```

Each keyword is optional (sentinel `_UNSET`). It no-ops unless a field actually changed, and it only touches agents whose movement is a `BehaviorTreeMovement`. `BehaviorTreeMovement` carries `{command, last_outcome, interaction_id}`; `SeekNode` reads `interaction_id` to know it's bound, `last_outcome` to know how the previous step ended. `_tick_formations` writes a fresh `NAVIGATE` command for every arrived-but-moving participant on each formation tick.

## Formation ticks

A formation tick is an `update(..., formations=True)` call. The agent manager makes one on the tick before each BT tick, the only ticks whose formation targets reach the global planner, and passes the time since the previous one as `dt`. Active formations advance grouped by type through `Formation.tick_all(formations, dt)`, which calls `tick` on each unless the type batches its members (`WalkFormation` senses walls and crowding for every group in one kernel call). Every formation gets the manager's `Clearance` as `formation.clearance`: free room along rays to the walls, and crowding among an observer's perceived agents. `Formation.speeds()` replaces the desired speed of its members until the next formation tick. A `LineFormation` anchored on its front member (a mobile provider such as an escort) paces that member to its slowest follower's desired speed.

## Formation resolution

`_resolve_formation(interaction)` picks the first of:

1. Step-level `spec.formation_spec` (authored on the `SeekSpec` / scenario step).
2. Object-level metadata - `WorldObject.formation` for `interaction.object_id`.
3. Registry default - `InteractionType(...).kind.formation_default`.

`WorldObject.seats` become the `slot_poses` of a `cluster` formation, so a sitter is placed on a seat rather than on a generated ring. A seat is assigned only while it fits: no live interaction on the object holds it or one of its `seat_conflicts`, so a `SIT_ON` and a `LIE_ON` on one bench never hold the same or overlapping seats. A participant with no fitting seat waits off the furniture and takes the nearest one that fits once it frees. A full seated interaction sends the next seeker to a new interaction while its object has a seat that fits and nobody waits or queues for one, so an object seats as many sitters as its seats allow, and queues it otherwise. A seated interaction admits a participant only while a seat fits for it, and when a seat fits the head of a full seated interaction's queue moves into a new interaction before newcomers seek. Only participants are given a seat: a queued member waits where it stopped and claims one on promotion. `membership_table` reports each participant's seat index, published as `InteractionFrame.member_seat`.

`_anchor_from_spec(spec, interaction)` dispatches on `AnchorKind`:

- `OBJECT` -> `ObjectAnchor` on `spec.anchor_ref or interaction.object_id`.
- `AGENT` -> `AgentAnchor` on `int(spec.anchor_ref)`.
- `PROVIDER` -> `AgentAnchor` on `interaction.provider` (used by SERVICE's `f_formation`).
- `POSE` -> `PoseAnchor` on `spec.anchor_pose`.
- `CENTROID` -> `CentroidAnchor` over live participants.
- `LEADER` -> `AgentAnchor` on the creator, `participants[0]` (used by GROUP_WALK's `walk`).

Unresolvable anchors return `None` and the interaction runs without a formation.

## What lives elsewhere

- **Proximity gating.** The compiler emits `ResolveObjectNode -> GoToNode -> SeekNode` for object-bound interaction steps; `GoToNode` only returns SUCCESS within `interaction_radius`, so `SEEK` never fires before arrival.
- **Per-type defaults.** `contract_defaults`, `formation_default`, and `interaction_radius` all live on `InteractionKind` in [interaction_kinds.py](interaction_kinds.py). See [config/scenarios/README.md](../../config/scenarios/README.md) for the authoring cascade.
