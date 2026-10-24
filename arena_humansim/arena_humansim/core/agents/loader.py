from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import attrs
import yaml

from .types import AgentType

_log = logging.getLogger("arena_humansim.agent_types")

_DICT_MERGE_FIELDS = {
    "needs",
    "utility_weights",
    "actions",
    "sequences",
    "vars",
    "perception",
    "local_planner_params",
    "locomotion",
    "pose",
}

_DEEP_MERGE_FIELDS = {"locomotion", "pose"}

SCENARIO_ONLY_KEYS = (
    "description",
    "simulation",
    "modules",
    "agents",
    "interaction_scripts",
    "flow",
    "agent_types",
    "world_objects",
    "walls",
    "teleport_zones",
    "obstacles",
    "event_scripts",
)

_TUPLE_FIELDS = {"perception_stack"}


def resolve_extends(
    agent_types: dict[str, dict[str, Any]],
    builtins: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    all_types: dict[str, dict[str, Any]] = {**builtins, **agent_types}

    resolved: dict[str, dict[str, Any]] = {}
    in_progress: set[str] = set()

    def _resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name not in all_types:
            raise ValueError(f"Agent type '{name}' referenced by extends but not defined")
        if name in in_progress:
            raise ValueError(f"Circular extends detected involving '{name}'")

        in_progress.add(name)
        child = all_types[name]
        extends = child.get("extends")

        if extends is None:
            resolved[name] = copy.deepcopy(child)
            in_progress.discard(name)
            return resolved[name]

        parent = _resolve(extends)
        merged = _deep_merge(parent, child, name)
        resolved[name] = merged
        in_progress.discard(name)
        return merged

    for name in agent_types:
        _resolve(name)

    return {name: resolved[name] for name in agent_types}


def _merge_nested(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(parent)
    for key, val in child.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _deep_merge(
    parent: dict[str, Any],
    child: dict[str, Any],
    child_name: str,
) -> dict[str, Any]:
    merged = copy.deepcopy(parent)
    merged["name"] = child_name
    merged["extends"] = None

    for key, child_val in child.items():
        if key in ("name", "extends"):
            continue
        if key in _DEEP_MERGE_FIELDS:
            if isinstance(child_val, dict):
                parent_val = merged.get(key, {})
                merged[key] = _merge_nested(parent_val if isinstance(parent_val, dict) else {}, child_val)
        elif key in _DICT_MERGE_FIELDS:
            if isinstance(child_val, dict) and child_val:
                parent_val = merged.get(key, {})
                if not isinstance(parent_val, dict):
                    parent_val = {}
                merged_dict = copy.deepcopy(parent_val)
                merged_dict.update(copy.deepcopy(child_val))
                merged[key] = merged_dict
        elif key in _TUPLE_FIELDS:
            merged[key] = copy.deepcopy(child_val)
        else:
            merged[key] = copy.deepcopy(child_val)

    return merged


def _get_share_agent_types_dir() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("arena_humansim")) / "config" / "agent_types"
    except Exception:
        pass
    candidates = [
        Path(__file__).resolve().parents[2] / "config" / "agent_types",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _structure_raw(raw: dict[str, Any], source_path: Path | None) -> AgentType:
    from arena_humansim.utils.scenario_loader import converter

    at = converter.structure(raw, AgentType)
    if source_path is not None:
        at = attrs.evolve(at, source_path=source_path)
    return at


def load_agent_type_raw_from_file(path: str | Path) -> tuple[str, dict[str, Any], Path]:
    p = Path(path).resolve()
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if "name" not in raw:
        raw["name"] = p.stem
    return raw["name"], raw, p


def load_agent_type_from_file(path: str | Path) -> AgentType:
    _name, raw, src = load_agent_type_raw_from_file(path)
    if raw.get("extends") is not None:
        defaults_raw = _load_default_agent_types_raw()
        defaults_dicts = {n: r for n, (r, _) in defaults_raw.items()}
        resolved = resolve_extends({raw["name"]: raw}, defaults_dicts)
        raw = resolved[raw["name"]]
    return _structure_raw(raw, src)


def load_agent_types_raw_from_dir(directory: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.yaml")):
        try:
            name, raw, src = load_agent_type_raw_from_file(path)
            if any(key in raw for key in SCENARIO_ONLY_KEYS):
                continue
            _structure_raw(dict(raw), src)
            result[name] = (raw, src)
        except Exception as exc:
            _log.warning("skipping agent type %s: %s", path, exc)
    return result


def _load_default_agent_types_raw() -> dict[str, tuple[dict[str, Any], Path]]:
    share_dir = _get_share_agent_types_dir()
    if share_dir is None:
        return {}
    return load_agent_types_raw_from_dir(share_dir)


def resolve_agent_type_name(
    name: str,
    registry: dict[str, AgentType],
) -> AgentType | None:
    if "/" in name or name.endswith(".yaml"):
        p = Path(name)
        if p.is_file():
            return load_agent_type_from_file(p)
        return None
    return registry.get(name)


def load_agent_types(scenario_dir: Path | None = None) -> dict[str, AgentType]:
    defaults_raw = _load_default_agent_types_raw()
    scenario_local_raw = load_agent_types_raw_from_dir(scenario_dir) if scenario_dir is not None else {}
    merged_raw: dict[str, tuple[dict[str, Any], Path]] = {**defaults_raw, **scenario_local_raw}

    has_extends = any(raw.get("extends") is not None for raw, _ in merged_raw.values())
    if has_extends:
        raw_dicts = {name: raw for name, (raw, _) in merged_raw.items()}
        defaults_dicts = {name: raw for name, (raw, _) in defaults_raw.items()}
        resolved = resolve_extends(raw_dicts, defaults_dicts)
        return {name: _structure_raw(resolved[name], merged_raw[name][1]) for name in resolved}

    return {name: _structure_raw(raw, src) for name, (raw, src) in merged_raw.items()}
