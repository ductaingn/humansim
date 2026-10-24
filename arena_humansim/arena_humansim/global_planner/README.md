# Global planners

Turn a `HighLevelCommand` (usually `NAVIGATE target_pose`) into the next global subgoal for each agent. Runs before the local planner each tick. Responsible for pathfinding, LOS simplification, and wall push-back.

## Available

| Name | Class | Notes |
|---|---|---|
| `navmesh` | `NavMeshPlanner` | Triangulated free space with exact passage widths. Cost follows wall count. One shortest path tree per goal, shared by all agents heading there. Default. |
| `astar` | `AStarPlanner` | Per-agent A* on an inflated occupancy grid. |
| `dijkstra` | `DijkstraPlanner` | All-pairs shortest-path on the same grid, cached until walls change. Cheaper than A* for large crowds sharing goals, pays more upfront per grid rebuild. |

## Contract

```python
class GlobalPlanner(PoolAware, WallAware, Loggable, ABC):
    @abstractmethod
    def _has_map(self) -> bool: ...

    @abstractmethod
    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]: ...

    def compute(self, agents, high_level_commands) -> dict[int, Pose2D]: ...
    def snap_terminals(self, poses) -> list[Pose2D]: ...
    def _sees(self, starts, ends) -> np.ndarray: ...
    def get_cached_goals(self) -> dict[int, Pose2D]: ...
    def get_cached_paths(self) -> dict[int, list[Pose2D]]: ...
    def configure(self, *, inflation_radius, resolution, comfort_radius) -> None: ...
    def _nearest_reachable(self, start, target) -> Pose2D | None: ...
    def components(self, poses) -> list[frozenset[int]]: ...
```

A subclass implements only `_has_map` and `_plan`. The base class owns everything else: the per-agent path cache, the replan rule, and stepping along a cached path.

- `_has_map` gates the whole planner. While false every `NAVIGATE` agent gets its raw target as its subgoal, no `_plan` call.
- `_plan` takes `(agent_id, start pose, target pose)` requests and returns final, already-smoothed waypoints per agent, first waypoint the agent's own position, last the snapped terminal, or `None` if the target is unreachable.
- `compute` is concrete: skips non-`NAVIGATE` commands, replans when the target moved past `_replan_distance` from the cached goal, advances a fresh cache entry with `advance_along_path`, and on `None` asks `_nearest_reachable` for a substitute target. A route to the substitute is cached under the original target, so it is not replanned every tick. Without a substitute it pops the cache entry and falls back to `snap_terminal(target)`.
- `compute` skips the search where a straight leg will do. An agent whose target changed, or who strayed past `_replan_distance` from its path, gets the leg from its position to the snapped target when that target is within `DIRECT_RANGE` (3 m) or the agent is already on its last leg. An agent earlier on its path whose target moved within `_replan_distance` keeps the path with only its last waypoint moved. Both need `_sees` to clear the leg, otherwise the agent is replanned.
- `_sees` is optional, default a ray cast against the walls. `navmesh` overrides it with its comfort-radius line of sight, the same test its `plan` tries before searching.
- `snap_terminals` is optional, default `snap_terminal` per pose. `navmesh` checks all poses' wall clearance in one call first.
- `_nearest_reachable` is optional, default `None`. It returns the point closest to an unreachable target that the start can still reach.
- `components` is optional, default an empty set per pose. It returns the free-space components a pose can walk into, two poses being mutually reachable when their sets meet, and the teleport zone routing reads it. `navmesh` snaps each pose off the walls and returns the edge graph components of its triangle, `astar` and `dijkstra` the flood fill label of the nearest free cell at `resolution` 0.1 or finer. Labels follow every `set_walls`.
- `get_cached_goals` is how the rest of the pipeline (local planner, markers) reads the planner state without re-triggering work.
- `get_cached_paths` is optional, only used for visualization.
- `WallAware.set_walls(segments)` invalidates any grid / graph caches through `_forget_paths`.
- `configure` is optional, default is a no-op. The node calls it with all three params at startup and at every `reset`.

## Parameters

The node exposes three ROS params under `global_planner.*` and calls `configure` at startup and at every `reset`, so a set takes effect at the next reset.

| Param | Default | Meaning |
|---|---|---|
| `global_planner.inflation_radius` | `0.38` | Hard wall clearance (m). Grid planners round it up to whole cells, navmesh applies it exactly. |
| `global_planner.resolution` | `0.2` | Grid cell size (m). Grid planners only, navmesh ignores it. |
| `global_planner.comfort_radius` | `0.6` | Preferred wall clearance (m), navmesh only. Grid planners ignore it. |

`astar` and `dijkstra` rasterize the walls onto a uniform grid and dilate them by `inflation_radius`. An opening is passable only if it is wider than twice the inflation in whole cells. With the defaults the inflation is 2 cells (0.4 m per side), which closes a 1.0 m door. At `resolution` 0.1 the same door keeps a 0.2 m corridor. A door narrower than twice `inflation_radius` stays closed at any resolution and needs a smaller radius.

An agent whose target is unreachable walks to the reachable point closest to it and stays there. `navmesh` always does this. `astar` and `dijkstra` do it only at `resolution` 0.1 or finer, because a coarser grid rasterizes metre-wide doors shut and reports reachable targets as unreachable. Where no substitute exists, an agent walled in on all sides or a coarse grid, the agent gets the target itself as its subgoal and walks straight at it.

## Navmesh

`navmesh` triangulates the free space, so passage widths are exact. A passage is open iff it is wider than twice `inflation_radius`.

Paths keep `comfort_radius` clearance from walls where the passage allows it, and relax towards `inflation_radius` in narrower passages. A `comfort_radius` at or below `inflation_radius` disables the preference and paths hug the hard clearance everywhere.

Agents and targets beyond the meshed area are joined to it by a straight leg, or routed directly when the straight line is clear.

## Shared helpers

- `simplify_path(waypoints, min_area=0.01)` - Visvalingam-Whyatt decimation, in the base module. Use this before returning paths so LOS-connected runs collapse to endpoints.
- `GlobalPlanner.advance_along_path(agent_pos, waypoints, current_idx)` - dot-product projection; returns the new cursor index.

## Adding a planner

1. Subclass `GlobalPlanner` in a new file under `global_planner/`.
2. Implement `_has_map` + `_plan`. Use `simplify_path` for LOS cleanup.
3. Register in `global_planner/__init__.py` via a `_load_<name>` lazy loader.
4. Add contract coverage in `tests/contracts/test_global_planner_contract.py` and efficacy coverage in `tests/efficacy/test_global_planner_efficacy.py`.

## Benchmarks

See [config/benchmark/astar_vs_dijkstra.yaml](../../config/benchmark/astar_vs_dijkstra.yaml) for the canonical A*-vs-Dijkstra sweep.
