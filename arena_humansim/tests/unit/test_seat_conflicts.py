from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.msg import WorldObjectInfo
from geometry_msgs.msg import Pose2D

from arena_humansim.core.agent_manager import _seat_conflicts


def _info(n_seats: int, offsets: list[int], conflicts: list[int]) -> WorldObjectInfo:
    return WorldObjectInfo(object_id="bench", seats=[Pose2D(x=float(i)) for i in range(n_seats)], seat_conflict_offset=offsets, seat_conflicts=conflicts)


def test_offsets_slice_the_conflicts_per_seat() -> None:
    assert _seat_conflicts(_info(3, [0, 1, 3, 4], [1, 0, 2, 1])) == [frozenset({1}), frozenset({0, 2}), frozenset({1})]


def test_no_offsets_means_no_conflicts() -> None:
    assert _seat_conflicts(_info(2, [], [])) == []


@pytest.mark.parametrize(("offsets", "conflicts"), [([0, 1], [1]), ([0, 1, 3], [1, 0])])
def test_offsets_that_do_not_cover_the_seats_are_rejected(offsets: list[int], conflicts: list[int]) -> None:
    with pytest.raises(ValueError, match="seat_conflict_offset"):
        _seat_conflicts(_info(2, offsets, conflicts))
