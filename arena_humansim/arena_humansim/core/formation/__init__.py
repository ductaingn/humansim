from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Self

from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import Pose2D

from .anchor import AgentAnchor, Anchor, CentroidAnchor, ObjectAnchor, PoseAnchor

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent

    from .clearance import Clearance


AgentLookup = Callable[[int], "BaseAgent | None"]

_registry: ModuleRegistry[Formation] = ModuleRegistry()


class Formation(Loggable, ABC):
    clearance: Clearance | None = None

    @abstractmethod
    def on_join(self, agent_id: int, *, participant: bool = True) -> None: ...

    @abstractmethod
    def on_leave(self, agent_id: int) -> None: ...

    @abstractmethod
    def tick(self, dt: float) -> dict[int, Pose2D]: ...

    @classmethod
    def tick_all(cls, formations: Sequence[Self], dt: float) -> list[dict[int, Pose2D]]:
        """Ticks formations of this type together, targets in input order."""
        return [f.tick(dt) for f in formations]

    def arrived(self, agent_id: int) -> bool:
        return True

    def slot_of(self, agent_id: int) -> Pose2D | None:
        """Slot the member is headed for, arrived or not, None for free-standing formations."""
        return None

    def seat_of(self, agent_id: int) -> Pose2D | None:
        """Explicit slot the arrived agent is parked on, None for free-standing formations."""
        return None

    def seat_index_of(self, agent_id: int) -> int | None:
        """Object seat index of the member's explicit slot, None without one."""
        return None

    def held_seats(self) -> list[int]:
        """Object seat indices of the explicit slots assigned to members."""
        return []

    def speeds(self) -> dict[int, float]:
        """Desired speed per member for this tick, empty where members keep their own pace."""
        return {}

    @classmethod
    def register(cls, name: str, label: str | None = None) -> Callable[[Callable[[], type[Formation]]], Callable[[], type[Formation]]]:
        return _registry.register(name, label)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> Formation:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def labels(cls) -> dict[str, str]:
        return _registry.labels()


def _load_line() -> type[Formation]:
    from .line import LineFormation

    return LineFormation


def _load_cluster() -> type[Formation]:
    from .cluster import ClusterFormation

    return ClusterFormation


def _load_f_formation() -> type[Formation]:
    from .f_formation import FFormation

    return FFormation


def _load_dyad() -> type[Formation]:
    from .dyad import DyadFormation

    return DyadFormation


def _load_walk() -> type[Formation]:
    from .walk import WalkFormation

    return WalkFormation


_registry.register("line", "Line")(_load_line)
_registry.register("cluster", "Cluster")(_load_cluster)
_registry.register("f_formation", "F-formation")(_load_f_formation)
_registry.register("dyad", "Dyad")(_load_dyad)
_registry.register("walk", "Walk")(_load_walk)


__all__ = [
    "AgentAnchor",
    "AgentLookup",
    "Anchor",
    "CentroidAnchor",
    "Formation",
    "ObjectAnchor",
    "PoseAnchor",
]
