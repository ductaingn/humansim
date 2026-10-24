import ast
import enum
import operator
import re
from pathlib import Path
from typing import Any, cast

import attrs
import yaml

from ..core.agents import AgentType, VarDef
from ..core.agents.loader import _load_default_agent_types_raw, _structure_raw, load_agent_types_raw_from_dir, resolve_extends
from ..core.agents.types import AttentionStepDef, GoToStepDef
from ..core.interaction_kinds import InteractionType, is_object_bound_name
from .scenario_loader import converter
from .types import Pose2D, WaypointMode


class ExecutionMode(enum.StrEnum):
    MASTER = "master"
    SUBSYSTEM = "subsystem"


@attrs.define
class SimulationParams:
    seed: int = 0
    dt: float = 0.05
    bt_tick_interval: int = 5
    max_ticks: int = 0  # 0 = run indefinitely
    execution_mode: str = "master"
    formation_scale: float = 1.0  # global multiplier on formation spacing/radii
    robot_shutdown: bool = False  # end the run once every robot has latched on its final waypoint


@attrs.define
class ModuleConfig:
    perception: str = "default"
    global_planner: str = "dijkstra"
    local_planner: str = "sfm"
    animation: str = "noop"
    occlusion: str = "bitmap"


@attrs.define
class ServiceSpec:
    tag: str = ""
    max_participants: int = -1


@attrs.define
class AgentConfig:
    agent_id: int = 0
    agent_type: str = "adult"
    spawn_pose: Pose2D = attrs.Factory(Pose2D)
    goal_sequence: list[Pose2D] = attrs.Factory(list)
    waypoint_mode: WaypointMode = WaypointMode.ONCE
    desired_velocity: float = 0.0
    agent_radius: float = 0.0
    interaction_preferences: dict[str, Any] = attrs.Factory(dict)
    kind: int = 0  # 0=human, 1=robot (see AgentKind)
    policy: str = ""  # local planner name for robots; empty = teleport-only
    policy_params: str = ""  # opaque JSON blob forwarded to the planner
    services: list[ServiceSpec] = attrs.Factory(list)
    spawn_tick: int = 0  # deferred spawn; 0 = spawn on first tick


@attrs.define
class InteractionScript:
    tick: int = 0
    interaction_type: str = "TALK_TO"
    participants: list[int] = attrs.Factory(list)
    duration_ticks: int = 0
    metadata: dict[str, Any] = attrs.Factory(dict)


@attrs.define
class ShapeModel:
    type: str = "polygon"
    radius: float = 0.0


@attrs.define
class RateKeyframeModel:
    t: float = 0.0
    rate: float = 0.0


@attrs.define
class SinkAffinityModel:
    sink_idx: int = 0
    weight: float = 1.0


@attrs.define
class AgentTemplateModel:
    desired_velocity_min: float = 0.0
    desired_velocity_max: float = 0.0
    agent_radius: float = 0.0
    agent_type: str = "adult"
    sink_affinity: list[SinkAffinityModel] = attrs.Factory(list)


@attrs.define
class SourceScenarioConfig:
    pose: Pose2D = attrs.Factory(Pose2D)
    shape: ShapeModel = attrs.Factory(ShapeModel)
    type: str = "poisson"
    rate_profile: list[RateKeyframeModel] = attrs.Factory(list)
    max_concurrent: int = -1
    max_total: int = -1
    agent_template: AgentTemplateModel = attrs.Factory(AgentTemplateModel)


@attrs.define
class SinkScenarioConfig:
    pose: Pose2D = attrs.Factory(Pose2D)
    shape: ShapeModel = attrs.Factory(ShapeModel)
    absorption_radius: float = 0.5
    capacity: int = -1


@attrs.define
class FlowScenarioConfig:
    sources: list[SourceScenarioConfig] = attrs.Factory(list)
    sinks: list[SinkScenarioConfig] = attrs.Factory(list)


@attrs.define
class AnchorConfig:
    kind: str = "object"  # "object" | "agent" | "provider" | "pose" | "centroid" | "leader"
    ref: str | None = None  # object_id or agent_id (as str) for object/agent; None for pose/centroid
    pose: Pose2D | None = None  # for kind="pose"


@attrs.define
class FormationConfig:
    type: str = ""  # "line" | "cluster" | "f_formation" | "dyad" | "walk"
    anchor: AnchorConfig | None = None  # defaults to OBJECT anchor on the owning object
    params: dict[str, float] = attrs.Factory(dict)  # strategy-specific (base_step, radius, etc)


@attrs.define
class WorldObjectConfig:
    object_id: str = ""
    type: str = ""
    pose: Pose2D = attrs.Factory(Pose2D)
    capacity: int = 1
    satisfies: dict[str, float] = attrs.Factory(dict)
    formation: FormationConfig | None = None
    interaction_radius: float | None = None


@attrs.define
class EventScript:
    tick: int = 0
    event: str = ""
    target_agent: int = -1  # -1 = broadcast


@attrs.define
class WallConfig:
    name: str = ""
    start: Pose2D = attrs.Factory(Pose2D)
    end: Pose2D = attrs.Factory(Pose2D)


@attrs.define
class TeleportZoneConfig:
    name: str = ""
    destinations: list[str] = attrs.Factory(list)
    entry: Pose2D = attrs.Factory(Pose2D)
    wait: Pose2D = attrs.Factory(Pose2D)
    hold: Pose2D = attrs.Factory(Pose2D)
    footprint: list[Pose2D] = attrs.Factory(list)
    open: bool = True
    delay: float = 0.0


@attrs.define
class ObstacleBB:
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0


@attrs.define
class ObstacleSceneConfig:
    name: str = ""
    pose: Pose2D = attrs.Factory(Pose2D)
    bb: ObstacleBB = attrs.Factory(ObstacleBB)
    obstacle_type: str = ""
    interaction_types: list[str] = attrs.Factory(list)


@attrs.define
class ScenarioConfig:
    name: str = "unnamed"
    description: str = ""
    simulation: SimulationParams = attrs.Factory(SimulationParams)
    modules: ModuleConfig = attrs.Factory(ModuleConfig)
    agents: list[AgentConfig] = attrs.Factory(list)
    interaction_scripts: list[InteractionScript] = attrs.Factory(list)
    flow: FlowScenarioConfig | None = None
    agent_types: dict[str, AgentType] = attrs.Factory(dict)
    world_objects: list[WorldObjectConfig] = attrs.Factory(list)
    walls: list[WallConfig] = attrs.Factory(list)
    teleport_zones: list[TeleportZoneConfig] = attrs.Factory(list)
    obstacles: list[ObstacleSceneConfig] = attrs.Factory(list)
    event_scripts: list[EventScript] = attrs.Factory(list)


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST, variables: dict[str, int | float | bool | str]) -> int | float | bool | str:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left, variables)
        right = _safe_eval(node.right, variables)
        return _BINARY_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand, variables)
        return _UNARY_OPS[op_type](operand)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _resolve_var_string(
    s: str,
    variables: dict[str, int | float | bool | str],
) -> Any:  # noqa: ANN401
    pattern = r"\$\{([^}]+)\}"
    matches = list(re.finditer(pattern, s))
    if not matches:
        return s

    if len(matches) == 1 and matches[0].start() == 0 and matches[0].end() == len(s):
        expr_str = matches[0].group(1).strip()
        if expr_str in variables:
            return variables[expr_str]
        tree = ast.parse(expr_str, mode="eval")
        return _safe_eval(tree, variables)

    def _replace(m: re.Match) -> str:
        expr_str = m.group(1).strip()
        if expr_str in variables:
            return str(variables[expr_str])
        tree = ast.parse(expr_str, mode="eval")
        return str(_safe_eval(tree, variables))

    return re.sub(pattern, _replace, s)


def resolve_vars(
    raw_dict: dict[str, Any],
    var_defs: dict[str, VarDef],
    overrides: dict[str, int | float | bool | str] | None = None,
) -> dict[str, Any]:
    variables: dict[str, int | float | bool | str] = {}
    for vname, vdef in var_defs.items():
        variables[vname] = vdef.default

    if overrides:
        for vname, val in overrides.items():
            if vname not in var_defs:
                raise ValueError(f"Override for unknown variable: {vname}")
            vdef = var_defs[vname]
            _type_check_var(vname, val, vdef)
            variables[vname] = val

    for vname, val in variables.items():
        vdef = var_defs[vname]
        if isinstance(val, (int, float)):
            if vdef.min is not None and val < vdef.min:
                raise ValueError(f"Variable '{vname}' value {val} below minimum {vdef.min}")
            if vdef.max is not None and val > vdef.max:
                raise ValueError(f"Variable '{vname}' value {val} above maximum {vdef.max}")

    return cast(dict[str, Any], _walk_resolve(raw_dict, variables))


def _type_check_var(
    name: str,
    value: float | bool | str,
    vdef: VarDef,
) -> None:
    expected = {
        "int": int,
        "float": (int, float),
        "bool": bool,
        "str": str,
    }.get(vdef.type)
    if expected is not None and not isinstance(value, expected):
        raise TypeError(f"Variable '{name}' expected type {vdef.type}, got {type(value).__name__}")


def _walk_resolve(
    obj: object,
    variables: dict[str, int | float | bool | str],
) -> object:
    if isinstance(obj, str):
        return _resolve_var_string(obj, variables)
    if isinstance(obj, dict):
        return {k: _walk_resolve(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(item, variables) for item in obj]
    return obj


def load_scenario(
    path: str,
    var_overrides: dict[str, int | float | bool | str] | None = None,
) -> ScenarioConfig:
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(scenario_path) as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    return _structure_manual(
        raw,
        var_overrides=var_overrides,
        scenario_dir=scenario_path.parent,
    )


def _resolve_agent_types_raw(
    raw: dict[str, Any],
    scenario_dir: Path | None,
    var_overrides: dict[str, int | float | bool | str] | None,
) -> dict[str, dict[str, Any]]:
    """Run the same agent-type resolution as _structure_manual but stop at the merged-dict stage. The result is a plain-dict map of fully-resolved agent types (extends + vars applied) suitable for serialization."""
    defaults_raw = _load_default_agent_types_raw()
    scenario_local_raw = load_agent_types_raw_from_dir(scenario_dir) if scenario_dir is not None else {}
    file_types_raw: dict[str, tuple[dict[str, Any], Path]] = {**defaults_raw, **scenario_local_raw}

    raw_agent_types = raw.get("agent_types", {}) or {}
    if raw_agent_types:
        all_var_defs: dict[str, VarDef] = {}
        for atype_fields in raw_agent_types.values():
            if atype_fields and isinstance(atype_fields, dict):
                raw_vars = atype_fields.get("vars")
                if raw_vars and isinstance(raw_vars, dict):
                    for vname, vdef_raw in raw_vars.items():
                        if vname not in all_var_defs:
                            all_var_defs[vname] = converter.structure(vdef_raw, VarDef)
        if all_var_defs:
            resolved_raw = {}
            for atype_name, atype_fields in raw_agent_types.items():
                if atype_fields and isinstance(atype_fields, dict):
                    resolved_raw[atype_name] = resolve_vars(atype_fields, all_var_defs, var_overrides)
                else:
                    resolved_raw[atype_name] = atype_fields
            raw_agent_types = resolved_raw

    inline_raw: dict[str, dict[str, Any]] = {}
    for name, fields in raw_agent_types.items():
        d = dict(fields) if fields else {}
        d["name"] = name
        inline_raw[name] = d

    file_raw: dict[str, dict[str, Any]] = {name: r for name, (r, _) in file_types_raw.items()}
    merged_raw: dict[str, dict[str, Any]] = {**file_raw, **inline_raw}

    has_extends = any(r.get("extends") is not None for r in merged_raw.values())
    if has_extends:
        defaults_dicts = {name: r for name, (r, _) in defaults_raw.items()}
        merged_raw = resolve_extends(merged_raw, defaults_dicts)
    return merged_raw


def dump_resolved_scenario(
    scenario_path: str | Path,
    out_path: str | Path,
    var_overrides: dict[str, int | float | bool | str] | None = None,
) -> None:
    """Write a self-contained, fully-resolved scenario YAML to out_path. extends and var substitutions are baked in so reloading does not depend on the global agent_types/ library or the original scenario_dir."""
    scenario_path = Path(scenario_path)
    out_path = Path(out_path)

    with open(scenario_path) as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}

    merged = _resolve_agent_types_raw(raw, scenario_path.parent, var_overrides)
    out: dict[str, Any] = dict(raw)
    out["agent_types"] = {name: merged[name] for name in sorted(merged)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))


def _structure_manual(
    data: dict[str, Any],
    var_overrides: dict[str, int | float | bool | str] | None = None,
    scenario_dir: Path | None = None,
) -> ScenarioConfig:
    defaults_raw = _load_default_agent_types_raw()
    scenario_local_raw = load_agent_types_raw_from_dir(scenario_dir) if scenario_dir is not None else {}
    file_types_raw: dict[str, tuple[dict[str, Any], Path]] = {**defaults_raw, **scenario_local_raw}

    raw_agent_types = data.get("agent_types", {})
    if raw_agent_types:
        all_var_defs: dict[str, VarDef] = {}
        for _atype_name, atype_fields in raw_agent_types.items():
            if atype_fields and isinstance(atype_fields, dict):
                raw_vars = atype_fields.get("vars")
                if raw_vars and isinstance(raw_vars, dict):
                    for vname, vdef_raw in raw_vars.items():
                        if vname not in all_var_defs:
                            all_var_defs[vname] = converter.structure(vdef_raw, VarDef)

        if all_var_defs:
            resolved_raw = {}
            for atype_name, atype_fields in raw_agent_types.items():
                if atype_fields and isinstance(atype_fields, dict):
                    resolved_raw[atype_name] = resolve_vars(atype_fields, all_var_defs, var_overrides)
                else:
                    resolved_raw[atype_name] = atype_fields
            raw_agent_types = resolved_raw

    inline_raw: dict[str, dict[str, Any]] = {}
    for name, fields in raw_agent_types.items():
        d = dict(fields) if fields else {}
        d["name"] = name
        inline_raw[name] = d

    file_raw: dict[str, dict[str, Any]] = {name: raw for name, (raw, _) in file_types_raw.items()}
    merged_raw: dict[str, dict[str, Any]] = {**file_raw, **inline_raw}

    has_extends = any(raw.get("extends") is not None for raw in merged_raw.values())
    if has_extends:
        defaults_dicts = {name: raw for name, (raw, _) in defaults_raw.items()}
        merged_raw = resolve_extends(merged_raw, defaults_dicts)

    merged_types: dict[str, AgentType] = {}
    for name, raw in merged_raw.items():
        src = file_types_raw[name][1] if name in file_types_raw else None
        merged_types[name] = _structure_raw(raw, src)

    sim = converter.structure(data.get("simulation", {}), SimulationParams)
    modules = converter.structure(data.get("modules", {}), ModuleConfig)
    agents = [converter.structure(a, AgentConfig) for a in data.get("agents", [])]
    scripts = [converter.structure(s, InteractionScript) for s in data.get("interaction_scripts", [])]
    flow_raw = data.get("flow")
    flow = converter.structure(flow_raw, FlowScenarioConfig) if flow_raw else None
    world_objects = [converter.structure(wo, WorldObjectConfig) for wo in data.get("world_objects", [])]
    walls = [converter.structure(w, WallConfig) for w in data.get("walls", [])]
    teleport_zones = [converter.structure(z, TeleportZoneConfig) for z in data.get("teleport_zones", [])]
    obstacles = [converter.structure(o, ObstacleSceneConfig) for o in data.get("obstacles", [])]
    event_scripts = [converter.structure(es, EventScript) for es in data.get("event_scripts", [])]

    config = ScenarioConfig(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        simulation=sim,
        modules=modules,
        agents=agents,
        interaction_scripts=scripts,
        flow=flow,
        agent_types=merged_types,
        world_objects=world_objects,
        walls=walls,
        teleport_zones=teleport_zones,
        obstacles=obstacles,
        event_scripts=event_scripts,
    )

    _validate_target_object_refs(config)
    _validate_interaction_scripts(config)
    return config


def _validate_target_object_refs(config: ScenarioConfig) -> None:
    world_types = {wo.type for wo in config.world_objects}
    world_ids = {wo.object_id for wo in config.world_objects}
    agent_ids = {a.agent_id for a in config.agents}
    for atype_name, atype in config.agent_types.items():
        for seq_name, seq in atype.sequences.items():
            for step_name, step in seq.steps.items():
                if isinstance(step, (GoToStepDef, AttentionStepDef)):
                    continue
                interaction = step.interaction
                target = step.target
                if interaction is not None and is_object_bound_name(interaction) and isinstance(target, str):
                    if target not in world_ids and target not in world_types:
                        raise ValueError(f"agent_type={atype_name!r} sequence={seq_name!r} step={step_name!r}: target={target!r} does not match any world_objects object_id or type")
                if interaction == InteractionType.BLOCK.name:
                    if not isinstance(target, int) or target not in agent_ids:
                        raise ValueError(f"agent_type={atype_name!r} sequence={seq_name!r} step={step_name!r}: interaction BLOCK requires target=<agent_id:int>; got target={target!r}")
                    if step.autonomous:
                        raise ValueError(f"agent_type={atype_name!r} sequence={seq_name!r} step={step_name!r}: BLOCK and autonomous are mutually exclusive")
        for action_name, action in atype.actions.items():
            if action.target is not None:
                if action.target not in world_ids and action.target not in world_types:
                    raise ValueError(f"agent_type={atype_name!r} action={action_name!r}: target={action.target!r} does not match any world_objects object_id or type")


def _validate_interaction_scripts(config: ScenarioConfig) -> None:
    known_types = {it.name for it in InteractionType}
    agent_ids = {a.agent_id for a in config.agents}
    for i, script in enumerate(config.interaction_scripts):
        if script.interaction_type not in known_types:
            raise ValueError(f"interaction_scripts[{i}]: unknown interaction_type={script.interaction_type!r}; expected one of {sorted(known_types)}")
        if script.duration_ticks < 0:
            raise ValueError(f"interaction_scripts[{i}]: duration_ticks={script.duration_ticks} must be >= 0")
        for pid in script.participants:
            if pid not in agent_ids:
                raise ValueError(f"interaction_scripts[{i}]: participant agent_id={pid} is not defined in scenario agents")
