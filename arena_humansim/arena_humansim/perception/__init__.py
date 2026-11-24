from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from arena_humansim.core.pool import AgentPool, PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.viz import MarkerPublisher
    from arena_humansim.utils.types import AgentState, BeliefState, WorldState

_registry: ModuleRegistry[Perception] = ModuleRegistry()


class Perception(PoolAware, Loggable, ABC):
    supports_pool: bool = False

    @abstractmethod
    def compute(
        self,
        agent: BaseAgent,
        all_agents: dict[int, AgentState],
        world_state: WorldState,
        belief: BeliefState,
    ) -> BeliefState: ...

    def compute_pool(self, pool: AgentPool) -> None:
        raise NotImplementedError

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @classmethod
    def get(cls, name: str) -> type[Perception]:
        return _registry.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def labels(cls) -> dict[str, str]:
        return _registry.labels()

    @classmethod
    def register(cls, name: str, label: str | None = None) -> Callable[[Callable[[], type[Perception]]], Callable[[], type[Perception]]]:
        return _registry.register(name, label)


def _load_default() -> type[Perception]:
    from .default import DefaultPerception

    return DefaultPerception


_registry.register("default", "Default")(_load_default)
