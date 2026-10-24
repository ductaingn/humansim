from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.types import Pose2D


def _make_wk() -> WorldKnowledge:
    return WorldKnowledge()


def _obj(object_id: str, type_: str = "bench", x: float = 0.0, y: float = 0.0, capacity: int = 1) -> WorldObject:
    return WorldObject(object_id=object_id, type=type_, pose=Pose2D(x=x, y=y), capacity=capacity)


def test_add_remove_round_trip() -> None:
    wk = _make_wk()
    assert len(wk) == 0
    assert bool(wk) is False
    obj = _obj("b1", type_="bench", x=1.0, y=2.0)
    wk.add_object(obj)
    assert len(wk) == 1
    assert bool(wk) is True
    assert wk.get("b1") is obj
    assert wk.get_by_type("bench") == [obj]
    removed = wk.remove_object("b1")
    assert removed is obj
    assert wk.get("b1") is None
    assert wk.get_by_type("bench") == []
    assert "bench" not in wk._by_type
    assert len(wk) == 0


def test_readded_object_replaces_its_type_entry() -> None:
    wk = _make_wk()
    wk.add_object(_obj("b1", type_="bench", x=9.0))
    readded = _obj("b1", type_="bench", x=1.0)
    wk.add_object(readded)
    assert wk.get_by_type("bench") == [readded]
    assert wk.nearest_object("bench", Pose2D(x=9.0)) is readded


def test_readded_object_with_a_new_type_leaves_the_old_type() -> None:
    wk = _make_wk()
    wk.add_object(_obj("b1", type_="bench"))
    wk.add_object(_obj("b1", type_="sofa"))
    assert "bench" not in wk._by_type
    assert len(wk.get_by_type("sofa")) == 1


def test_remove_unknown_returns_none() -> None:
    wk = _make_wk()
    assert wk.remove_object("ghost") is None


def test_get_by_type_returns_copy() -> None:
    wk = _make_wk()
    wk.add_object(_obj("b1", type_="bench"))
    result = wk.get_by_type("bench")
    result.clear()
    assert len(wk.get_by_type("bench")) == 1


def test_nearest_object_picks_closest_skipping_full() -> None:
    wk = _make_wk()
    filled = _obj("f", type_="bench", x=0.5, y=0.0, capacity=1)
    partial = _obj("p", type_="bench", x=2.0, y=0.0, capacity=2)
    empty = _obj("e", type_="bench", x=5.0, y=0.0, capacity=1)
    wk.add_object(filled)
    wk.add_object(partial)
    wk.add_object(empty)
    wk.set_queue_length("f", 1)
    wk.set_queue_length("p", 1)

    from_pose = Pose2D(x=0.0, y=0.0)
    assert wk.nearest_object("bench", from_pose, exclude_full=True) is partial
    assert wk.nearest_object("bench", from_pose, exclude_full=False) is filled
    assert wk.nearest_object("missing", from_pose) is None


def test_nearest_object_capacity_zero_never_excluded() -> None:
    wk = _make_wk()
    infinite = _obj("i", type_="door", x=3.0, y=0.0, capacity=0)
    wk.add_object(infinite)
    wk.set_queue_length("i", 9999)
    assert wk.nearest_object("door", Pose2D(x=0.0, y=0.0), exclude_full=True) is infinite


def test_object_pose_returns_pose_or_none() -> None:
    wk = _make_wk()
    obj = _obj("b1", type_="bench", x=4.0, y=5.0)
    wk.add_object(obj)
    pose = wk.object_pose("b1")
    assert pose is not None
    assert pose.x == 4.0
    assert pose.y == 5.0
    assert wk.object_pose("nope") is None


def test_set_queue_length_and_sum_by_type() -> None:
    wk = _make_wk()
    wk.add_object(_obj("a", type_="queue"))
    wk.add_object(_obj("b", type_="queue"))
    wk.set_queue_length("a", 2)
    wk.set_queue_length("b", 3)
    assert wk.queue_length("queue") == 5
    assert wk.queue_length("missing") == 0


def test_set_counts_zeroes_objects_left_out() -> None:
    wk = _make_wk()
    wk.add_object(_obj("a", type_="queue"))
    wk.add_object(_obj("b", type_="queue"))
    wk.set_counts({"a": 2, "b": 1}, {"a": 1, "b": 1})
    wk.set_counts({"b": 3}, {"b": 2})
    assert (wk.queue_length_for_object("a"), wk.participants_count_for_object("a")) == (0, 0)
    assert (wk.queue_length_for_object("b"), wk.participants_count_for_object("b")) == (3, 2)
    assert wk.nearest_object("queue", Pose2D(), exclude_full=True) is not None


def test_clear_empties_all() -> None:
    wk = _make_wk()
    wk.add_object(_obj("a", type_="bench"))
    wk.set_queue_length("a", 7)
    wk.clear()
    assert len(wk) == 0
    assert wk._by_type == {}
    assert wk._queue_lengths == {}
