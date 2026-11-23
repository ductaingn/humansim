"""Locomotion animation states on the wire, values match arena_people_msgs/Pedestrian."""

from __future__ import annotations

import enum

import numpy as np

WALK_SPEED = 0.05
RUN_SPEED = 1.5


class AnimationType(enum.IntEnum):
    IDLE = 0
    WALKING = 1
    RUNNING = 2
    PANIC = 3
    SURPRISED = 4
    CURIOUS = 5
    THREATENING = 6


def locomotion_states(vel: np.ndarray, has_goal: np.ndarray, desired_vel: np.ndarray, halted: np.ndarray) -> np.ndarray:
    """IDLE/WALKING/RUNNING from speed, a goal-holding agent at rest reports its desired speed (no one-tick idle at start)
    unless the planner is holding it still (halted), then it stands idle instead of stepping in place."""
    speed = np.hypot(vel[:, 0], vel[:, 1])
    effective = np.where((speed <= WALK_SPEED) & has_goal & ~halted, desired_vel, speed)
    return np.where(effective > RUN_SPEED, AnimationType.RUNNING, np.where(effective > WALK_SPEED, AnimationType.WALKING, AnimationType.IDLE)).astype(np.uint8)
