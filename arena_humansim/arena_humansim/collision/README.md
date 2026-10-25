# Collision resolvers

Final step of the tick pipeline. Removes agent-agent and agent-wall overlap introduced by the integrator, walls last so they win.

## Available

| Name | Class | Notes |
|---|---|---|
| `wall_projection` | `WallProjectionResolver` | Separates overlapping agent pairs, then projects agents out of walls along the wall normal with a tangent wall-slide on velocity. Default. |
| `noop` | `NoopCollisionResolver` | Does nothing. Use when the scenario has no walls or overlap is tolerable. |

## Contract

```python
class CollisionResolver(WallAware, Loggable, ABC):
    @abstractmethod
    def resolve(self, pool: AgentPool) -> None: ...
```

- Mutates `pool.pos` and optionally `pool.vel` in place.
- `set_walls(segments)` feeds the current wall set; cache any derived arrays (e.g. segment endpoints, AB vectors) once per wall change.
- Must be idempotent in steady state: resolving a collision-free pool must be a no-op.
- Must be deterministic under a fixed wall set + pool state.

## `wall_projection` semantics

Agent contact first. Every pair closer than the sum of its radii is pushed apart along the center line by the overlap, split evenly, and the closing component of the pair's relative velocity is removed. Three Gauss-Seidel passes run over the pairs in contact at the start of the tick, in agent id order. A non-autonomous agent (`policy_idx == -1`, e.g. an externally driven robot) is immovable, and contact with it never moves its partner: the partner keeps its position and only loses its own velocity toward the immovable agent, so a robot cannot push a pedestrian out of place. Residual overlap in a jam carries over and shrinks over the next ticks.

Walls second. For each agent within `radius + margin` of a wall segment:

1. Find the closest point on the segment (clipped to endpoints).
2. Push the agent out along the outward normal by `(radius + margin) - dist`.
3. Project velocity onto the wall tangent - agents slide along walls rather than bouncing or sticking.

Three relaxation passes are run to settle corner cases where resolving wall A drives the agent into wall B. After that, residual overlap is accepted.

## Capsule footprints

A row with `pool.axial_offset > 0` is a capsule: two disks of `agent_radius` centered at `pos +- axial_offset * (cos theta, sin theta)`, `theta` being `pool.theta`. The core agent fills the offset as `max(0, footprint_length / 2 - agent_radius)`, so a disk agent keeps offset 0 and runs the unchanged disk arithmetic.

- Agent contact: each pair is tested on its closest disk-center pair (one center per disk agent, two per capsule). Penetration, push and the velocity response are computed from those two centers and applied to the agents' centers. The kd-tree pair radius grows by twice the largest offset in the pool so capsule ends are not missed.
- Walls: both disk centers run the contact test. The position corrections of both ends are summed onto the center and the velocity is projected onto every contacted wall's tangent. A wall touched by both ends therefore pushes twice its penetration in one pass, which the next pass does not undo.
- The resolver never changes `theta`. A capsule wedged across a gap narrower than its length stays wedged until the planner turns it.

## Adding a resolver

1. Subclass `CollisionResolver` under `collision/`.
2. Implement `resolve(pool)`. Override `set_walls` if you need derived caches.
3. Register in `collision/__init__.py` via a `_load_<name>` lazy loader.
4. Contract coverage: `tests/contracts/test_collision_contract.py` - no-walls no-op, deterministic, bounded-iterations.

Agent-wall `margin` defaults to 1 cm. Increase if you observe jitter at contact; decrease only if you're sure the local planner is keeping agents off walls on its own.
