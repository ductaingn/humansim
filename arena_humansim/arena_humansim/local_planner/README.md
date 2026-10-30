# Local planners

Velocity commands for the next tick, given each agent's global subgoal and neighbors. Called from the tick pipeline between `global_plan` and `kinematics`.

## Available

| Name | Class | Notes |
|---|---|---|
| `sfm` | `SFMPlanner` | Social Force Model. `supports_pool=True` - vectorized NumPy path. Gain scales per interaction-class pair and capsule footprints, see below. Default. |
| `hsfm` | `HSFMPlanner` | Headed Social Force Model (Farina/Pallottino/Bicchi 2017). Subclasses `sfm`; decomposes total force in the body frame, attenuates lateral force, and drives heading via PD toward the goal-attraction direction (or the total force within 90 degrees of the goal bearing, `heading_source`). `supports_pool=True`, `provides_heading=True`. |
| `orca` | `ORCAPlanner` | Reciprocal velocity obstacles (RVO2 agent and wall constraints). Numba kernel, `supports_pool=True`. |
| `helbing` | `HelbingPlanner` | Helbing/Farkas/Vicsek 2000 social force with body compression and sliding friction on contact, after Menge's AgtHelbing. Numba kernel on the `ForcePlanner` base (`force.py`), `supports_pool=True`. |
| `johansson` | `JohanssonPlanner` | Johansson/Helbing/Shukla 2007 elliptical repulsion from the neighbor's stride offset, after Menge's AgtJohansson. `ForcePlanner`. |
| `karamouzas` | `KaramouzasPlanner` | Karamouzas et al. 2009 predictive avoidance from the K most imminent times to collision, after Menge's AgtKaramouzas. `ForcePlanner`. |
| `zanlungo` | `ZanlungoPlanner` | Zanlungo/Ikeda/Kanda 2011 forces from the predicted positions at the time to interaction, after Menge's AgtZanlungo. Purely anticipatory: no neighbor or wall force while at rest. `ForcePlanner`. |
| `gcf` | `GCFPlanner` | Chraibi/Seyfried/Schadschneider 2010 generalized centrifugal force with speed-dependent ellipses, after Menge's AgtGCF (field of view, velocity and wall terms per the paper). No force from neighbors beside or behind, none while standing, and no wall force while walking parallel to a wall. Calibrated for one-way flow, jams in counterflow and crossing flow. Ellipses shape forces only, collisions stay circular. `ForcePlanner`. |
| `pedvo` | `PedVOPlanner` | Curtis/Manocha PedVO: ORCA with density-aware preferred speed and turning bias, after Menge's PedVO. Subclasses `orca` with its own numba kernel. |
| `straight` | `StraightToGoalPlanner` | Ignores neighbors, drives toward the subgoal at `desired_velocity`. `supports_pool=True`. For debugging and robot policies that don't want avoidance. |
| `nsp` | `NSPPlanner` | Neural Social Physics (Yue/Manocha/Wang, ECCV 2022) with SDD-pretrained checkpoints. Requires `torch`. `supports_pool=True`. |
| `socialgail` | `SocialGAILPlanner` | Learned crowd-sim policy from [William-island/SocialGAIL](https://github.com/William-island/SocialGAIL) (ICRA 2024, MIT). Pretrained HGNN actor; weights fetched on first use to `~/.cache/arena_humansim/socialgail/best.pt`. Requires `pip install torch torch-geometric`. `supports_pool=True`; re-infers every 8 sim ticks (0.4s decision interval, matching training). No wall handling - relies on `wall_projection`. |
| `neurosfm` | `NeuroSFMPlanner` | Neural social force model of [HumanoidsBonn/NeuroSFM_Pybullet](https://github.com/HumanoidsBonn/NeuroSFM_Pybullet) (MIT, `NEUROSFM_LICENSE`), the released force nets frozen and run in numpy from `neurosfm.npz`: `drive_gain` times net1 for the goal force, `social_gain` times the sum of net3 over the neighbors (the gap between body ellipses fitted to the reference male and female meshes, facing each walker's velocity, each walker's body a fixed fair draw by agent id, used within 0.89 m, else the center distance), 4 times net5 for a robot within 2.5 m, and the reference's contact-only wall push. A `ForcePlanner`, `supports_pool=True`, no torch. |

## Contract

```python
class LocalPlanner(WallAware, Loggable, ABC):
    supports_pool: bool = False       # opt into compute_pool fast-path
    needs_global_subgoal: bool = True # set False to skip global planning
    provides_heading: bool = False    # set True to own pool.theta - agent_manager skips its heading update

    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {}

    @abstractmethod
    def compute(self, agents, global_goals, dt) -> dict[int, (vx, vy)]: ...

    def compute_pool(self, pool, store_forces=False, dt=1.0) -> None: ...
```

`LocalPlanner` inherits `PoolAware` (no-op `attach` + five lifecycle hooks); planners that need pool-aligned SoA override those.

- `compute` is the per-agent fallback and is always required.
- When `supports_pool=True`, `AgentManager` calls `compute_pool` with the whole `AgentPool` and the planner writes back into `pool.vel` directly. Hot path; prefer it past ~50 agents.
- `PARAM_DEFAULTS` declares the planner's per-agent tuning schema as `name -> ParamDist`. `sample_agent_type` picks the active planner's defaults and merges them with the agent yaml's `local_planner_params:` overrides; the result is stored on the agent as `dict[str, float]`.
- `WallAware.set_walls(segments)` is called once per wall change; cache derived structures there, not in `compute*`.
- `publish_markers(pub)` is optional; SFM emits force arrows when `publish_markers=2`.

### Pool-aligned SoA via PoolAware

`PoolAware` is a universal mixin defined in `core/pool.py`; every subsystem base class (`LocalPlanner`, `GlobalPlanner`, `Perception`, `MotionAnimation`, `CollisionResolver`, `Occluder`) inherits it. `AgentManager` walks `self._pool_aware` once at startup and calls `attach(pool)` on each. If `compute_pool` needs per-agent state beyond what `AgentPool` carries (e.g. SFM's `relaxation_time`, HSFM's `angular_gain`), the planner owns its own `np.ndarray` of size `pool.capacity` and registers itself:

```python
def attach(self, pool):
    self._foo = np.zeros(pool.capacity, dtype=np.float64)
    pool.register_extension(self)

def on_pool_grow(self, new_capacity, old_capacity): ...   # resize
def on_pool_add(self, idx, agent): ...                    # populate from agent.params.local_planner_params[...]
def on_pool_update(self, idx, agent): ...                 # params changed in place (a sequence tune); re-read them, default no-op
def on_pool_swap(self, idx, last): ...                    # swap_remove copy [last] -> [idx]
def on_pool_reset(self): ...                              # usually no-op; n=0 makes slots inert
```

`AgentPool` dispatches the hooks from its lifecycle chokepoints. SFM, HSFM and the `ForcePlanner` models answer `on_pool_update` by re-running `on_pool_add`; a planner that caches per-agent params must do the same or a tune never reaches it. Registration must happen before any agents are added.

## Adding a planner

1. Subclass `LocalPlanner` in a new file under `local_planner/`.
2. Implement `compute`. Optionally set `supports_pool=True` and implement `compute_pool`. A force model can subclass `ForcePlanner` (`force.py`) instead and supply only `PARAM_DEFAULTS` and a numba `_kernel` writing goal, social and wall accelerations.
3. Declare `PARAM_DEFAULTS` for any tuning knobs you want sampled per-agent; if `compute_pool` needs them per-agent in SoA, override `attach` + the four `on_pool_*` hooks.
4. Register in `local_planner/__init__.py` via a `_load_<name>` lazy loader + `_register("<name>", _load_<name>, "<Label>", "<family>")`. The family (`force`, `geometric`, `no_avoidance`, `learned`) and the position in that list are what the evaluation tables and plots use.
5. Drop a contract test under `tests/contracts/test_local_planner_contract.py` and an efficacy test under `tests/efficacy/test_local_planner_efficacy.py`.

See the contract-test file for the invariants (velocity clipping, `set_walls` idempotency, pool/non-pool agreement) that gate every new planner.

## Parameter sampling

Per-agent local-planner params live under `local_planner_params:` in each agent type yaml (see [config/agent_types/](../../config/agent_types/)). The schema is taken from the active planner's `PARAM_DEFAULTS`; yaml entries override individual keys. Sampled once at spawn and stored on the agent as `dict[str, float]`.

| Planner | Keys |
|---|---|
| `sfm` (and family) | `relaxation_time`, `repulsion_strength`, `repulsion_range`, `anisotropy` |
| `hsfm` | the SFM keys plus `lateral_gain` (body-frame perp force gain, <=1 attenuates), `lateral_damping` (perp velocity damping), `angular_gain` (heading P-gain), `angular_damping` (angular velocity damping), `heading_source` (0 = steer toward the goal attraction, 1 = steer toward the total force, held within 90 degrees either side of the goal bearing so repulsion can swing the body aside but not turn it away from its goal. The agent type loader maps the words `attraction` and `total` to 0 and 1) |
| `helbing` | `relaxation_time`, `mass`, `agent_scale`, `obstacle_scale`, `force_distance`, `body_force`, `friction` |
| `johansson` | `relaxation_time`, `agent_scale`, `obstacle_scale`, `force_distance`, `stride_time`, `fov_weight` |
| `karamouzas` | `relaxation_time`, `wall_steepness`, `wall_distance`, `colliding_count`, `d_min`, `d_mid`, `d_max`, `agent_force`, `personal_space`, `anticipation`, `fov_angle` (degrees) |
| `zanlungo` | `relaxation_time`, `mass`, `agent_scale`, `obstacle_scale`, `force_distance` |
| `gcf` | `relaxation_time`, `nu_agent`, `max_agent_dist`, `max_agent_force`, `agent_interp_width`, `a_min`, `a_rate`, `b_max`, `b_growth`, `nu_wall`, `max_wall_dist`, `max_wall_force`, `wall_interp_width` |
| `neurosfm` | `drive_gain`, `social_gain` |
| `orca` / `pedvo` / `straight` / `socialgail` / `nsp` | none |

## Interaction classes (`sfm` family)

Agent-agent repulsion in `sfm`/`hsfm` is scaled per ordered pair (observer class, neighbor class) of `pool.interaction_class`. Class indices come from `core/interaction_classes.py`: `human` is 0, `robot` is 1, every further class name gets the next index on first sight. The gain matrices grow lazily to the highest class in the pool, new cells start at 1.0, and the pairs below get their default when both names are registered:

| Observer | Neighbor | `strength_scale` | `range_scale` |
|---|---|---|---|
| human | robot | 1.5 | 1.3 |
| human | wheelchair | 1.3 | 1.4 |
| wheelchair | robot | 1.5 | 1.3 |

A policy's `params_json` overrides any pair through `kind_gains`; an unknown class name in a key registers it:

```json
{"kind_gains": {"human_robot": {"strength_scale": 3.0, "range_scale": 2.0}, "human_scooter": {"strength_scale": 2.0}}}
```

`eff_strength = repulsion_strength * strength_scale` and `eff_range = repulsion_range * range_scale` of the observer. The scalar `compute` path reads the class from `AgentState.kind`.

## Capsule footprint (`sfm` family)

A row with `pool.axial_offset[i] > 0` is two disks of `pool.agent_radius[i]` at `pos +- axial_offset * (cos theta, sin theta)` (the core fills `axial_offset = max(0, footprint_length / 2 - agent_radius)`). The wall force is the sum of the wall force at both disk centers, each with its own wall query. The agent-agent force between two rows takes the closest pair of disk centers (a single center where the offset is 0), applies the usual force law to that distance and direction, and acts on the observer's center; the anisotropy term keeps using the observer's goal direction. Rows with offset 0 run the plain disk arithmetic unchanged. The neighbor lists come from perception and are not widened for capsules: an end disk farther than `vision_range` from the observer's center is not seen.
