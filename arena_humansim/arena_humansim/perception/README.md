# Perception

Builds each agent's belief about its neighbors - which other agents it currently sees, and at what bearing. Runs once per tick, before decision and planning.

## Available

| Name | Class | Notes |
|---|---|---|
| `default` | `DefaultPerception` | KDTree neighbor lookup + per-agent range/FOV filtering. Populates `pool.neighbor_csr` for the whole simulation in one call. |

## Contract

```python
class Perception(Loggable, ABC):
    supports_pool: bool = False

    @abstractmethod
    def compute(self, agent, all_agents, world_state, belief) -> BeliefState: ...

    def compute_pool(self, pool) -> None: ...
```

- `compute` (per-agent) is the fallback used in tests and off the hot path.
- `supports_pool=True` + `compute_pool(pool)` is the hot path. It writes a CSR (`indptr`, `indices`) into `pool.set_neighbor_csr`. `AgentManager` consumes this directly.
- The CSR encodes directional visibility: `indptr[i]:indptr[i+1]` lists the agent_ids that *agent i sees*, in agent id order. It is not symmetric - observer-A may see B without B seeing A.

## `proximity_sense` bypasses FOV

On top of the range + FOV gate, every agent always perceives neighbors within its `proximity_sense` radius (default ~1 m, Hall's personal-zone upper bound) regardless of bearing. The gate is `(range_ok & fov_ok) | prox_ok` in both paths. Without this, a co-located peer standing behind an agent at a gathering spot would be invisible, and the interaction manager would split a single GROUP_CONVERSATION into two 1p FORMING interactions that never merge. The `query_ball_point` / `sparse_distance_matrix` call uses `max(vision_range, proximity_sense)` so short-range neighbors still reach the per-row filter.

## Hearing, cane touch and detection delay

Agents with `hearing_range` or `cane_reach` above 0 sense beyond sight. They hear pedestrians out to `hearing_range` and robots out to `hearing_range * robot_loudness * speed / (1 m/s)`, at any bearing; a robot slower than 0.02 m/s is silent. This is the law of arena_auditory's drivetrain: the motor level grows with speed, and a level falls 6 dB per doubling of distance. The `robot_loudness` node parameter scales every robot: 0 is silent, 1 is the robot the type's range assumes at 1 m/s. They also touch anything whose disc reaches into the cane rectangle ahead of them (`cane_reach` long, `cane_width` wide).

These pairs are added after the sight and robot gates. `robot_blind` still removes robots, and walls block these pairs for agents with `vision_occlusion`. Both pool paths build them with `_channel_pairs`; the KDTree path widens its pair search to the farthest hearing or cane reach and checks only those pairs. The per-agent `compute` uses `_channel_hit`, which treats targets as points and applies no delay. ORCA keeps its own neighbor search, so it reads these ranges only as a distance cut on the robot's centre, at any bearing, without the delay, and hears every robot as far as the loudest one.

With `detection_delay` above 0 and a `dt` passed to `compute_pool`, a robot is listed only after the agent has sensed it for that long without a break. The start times live on the pool, keyed by agent ids, and `AgentPool.reset` clears them. Agents that declare none of these fields perceive exactly as before.

## Dense vs KDTree paths

`DefaultPerception` switches strategies at its `small_n_threshold` (default 64):

- **Dense** (N <= 64) - pairwise distance matrix, broadcast FOV test. Faster for small N, avoids tree-build overhead.
- **KDTree** (N > 64) - `scipy.spatial.cKDTree.sparse_distance_matrix` at the max vision range in the pool, then per-row range + FOV pruning.

Both paths must produce **bitwise-identical** CSRs for the same state. The contract test `tests/contracts/test_perception_contract.py` enforces this. Note the bearing convention: target-observer (`dx = target.x - observer.x`). The dense path had this inverted and was fixed on 2026-04-15 - do not re-invert without updating both paths.

## Robot Awareness Gating

Robots (`KIND_ROBOT`) are gated per observer by three pool values:
- `robot_blind` removes robots from the neighbour list, including inside `proximity_sense`. Human-to-human edges stay 100% active.
- `robot_range_offset` moves the detection range for robots. The effective range is `clip(vision_range + offset, 0, vision_range)`, so robots are never seen past the observer's own range. The near field is `min(proximity_sense, robot range)`.
- `robot_fov_offset` moves the field of view for robots: `clip(vision_fov + offset, 0, 360)`. A positive offset widens the cone for robots only.

The dense and KD-tree paths implement the same gating. `tests/unit/test_perception_robot_awareness.py` checks each rule, and the all-human crowd is unchanged by the blind flag.

## Adding a perception module

1. Subclass `Perception` in a new file under `perception/`.
2. Implement `compute`. Optionally set `supports_pool=True` and implement `compute_pool`.
3. Register in `perception/__init__.py` via a `_load_<name>` lazy loader.
4. Contract coverage in `tests/contracts/test_perception_contract.py` - the CSR-symmetry, range-gate, and FOV-gate tests apply to every module.

## Downstream consumers

- **InteractionManager** - rule 4 of `_try_bind` resolves visibility through `pool.visible_agent_ids`, which reads the CSR.
- **SFM** - neighbor repulsion terms iterate the CSR directly.
- **Markers** - `publish_markers=2` draws FOV cones and neighbor edges.
