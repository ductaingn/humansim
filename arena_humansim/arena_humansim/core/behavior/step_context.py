from __future__ import annotations

from collections.abc import Callable

import attrs

from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D


@attrs.define
class StepContext:
    target_pose: Pose2D | None = None
    target_object_id: str | None = None
    interaction_radius: float = DISTANCE_TOLERANCE
    sought: bool = False
    aim_seats: bool = False  # walk to the nearest free seat's standing point instead of the object's origin
    is_bound_lookup: Callable[[int], bool] | None = None
    im: InteractionManager | None = None
