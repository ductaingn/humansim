from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import attrs
import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import AttentionDef, ChannelDef, ClipDef, ParamDist, Pose3, RelativeRef, RobotRef
from arena_humansim.core.behavior.nodes import AttentionNode
from arena_humansim.core.behavior.nodes.attention import (
    FACE_ENTER_RAD,
    FACE_KEEP_RAD,
    FACE_TIMEOUT_S,
    GESTURE_Z_AGENT,
    GESTURE_Z_HEAD,
    GESTURE_Z_OBJECT,
    RESOLVE_TIMEOUT_S,
)
from arena_humansim.core.behavior.reach import ARM_IN, ARM_OUT, HEAD_IN, HEAD_OUT, MIN_RESIDENCE_S, reachable
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import CONTACT_LOCOMOTION_ONLY, InteractionManager
from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, GestureIntent, HighLevelCommand, Pose2D, SeekSpec

DT = 0.5
RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def world() -> WorldKnowledge:
    w = WorldKnowledge()
    w.add_object(WorldObject(object_id="bench_1", type="bench", pose=Pose2D(x=5.0, y=0.0)))
    w.add_object(WorldObject(object_id="bench_2", type="bench", pose=Pose2D(x=-5.0, y=0.0)))
    w.add_object(WorldObject(object_id="lamp_1", type="lamp", pose=Pose2D(x=5.0, y=2.0)))
    return w


def _mv(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    node.tick_once()
    return node.status


def _bt_agent(agent_factory: Callable[..., BaseAgent], agent_id: int = 1, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> BaseAgent:
    agent = agent_factory(agent_id=agent_id, x=x, y=y)
    agent.state.pose.theta = theta
    agent.movement = BehaviorTreeMovement()
    return agent


def _ch(at: object, **kw: object) -> ChannelDef:
    return ChannelDef(at=at, **kw)  # type: ignore[arg-type]


def _att(**kw: object) -> AttentionDef:
    fields = {}
    for name, val in kw.items():
        if name in ("gaze", "point", "point_l", "point_r") and not isinstance(val, ChannelDef):
            val = _ch(val)
        fields[name] = val
    return AttentionDef(**fields)  # type: ignore[arg-type]


def _arm(x: float, y: float, z: float = GESTURE_Z_OBJECT, dominant: str = "r") -> GestureIntent:
    return GestureIntent("arm", x, y, z, hand=dominant)


def _head(x: float, y: float, z: float = GESTURE_Z_OBJECT) -> GestureIntent:
    return GestureIntent("head", x, y, z)


def _slots(agent: BaseAgent) -> dict[str, GestureIntent]:
    return {g.slot: g for g in _mv(agent).gestures}


def _node(
    agent: BaseAgent,
    att: AttentionDef,
    world: WorldKnowledge,
    rng: np.random.Generator,
    agents: dict[int, BaseAgent] | None = None,
    names: dict[str, int] | None = None,
    robots: dict[str, int] | None = None,
    ctx: StepContext | None = None,
    bare: bool = False,
    duration: ParamDist | None = None,
    walking: Callable[[], bool] | None = None,
    dt: float = DT,
) -> AttentionNode:
    agents = {} if agents is None else agents
    names = names or {}
    robots = robots or {}

    def name_lookup(name: str, kind: int | None) -> int | None:
        return robots.get(name) if kind == KIND_ROBOT else {**names, **robots}.get(name)

    node = AttentionNode(
        "attention",
        agent,
        att,
        world,
        agent_lookup=agents.get,
        name_lookup=name_lookup,
        rng=rng,
        dt=dt,
        ctx=ctx if ctx is not None else StepContext(),
        bare=bare,
        duration=duration,
        walking=walking,
    )
    node.setup()
    return node


def _bind(agents: dict[int, BaseAgent], itype: InteractionType = InteractionType.GROUP_CONVERSATION) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid), visibility_lookup=lambda aid: set(agents) - {aid})
    cmds = {aid: HighLevelCommand(agent_id=aid, type=CommandType.SEEK, spec=SeekSpec(interaction_type=itype)) for aid in agents}
    mgr.update(cmds)
    iid = next(iter(mgr.interactions))
    assert set(mgr.interactions[iid].participants) == set(agents)
    for a in agents.values():
        assert _mv(a).interaction_id == iid
    return mgr


# reach constants


def test_reach_hysteresis_constants() -> None:
    assert (ARM_IN, ARM_OUT, HEAD_IN, HEAD_OUT) == (math.radians(90), math.radians(110), math.radians(60), math.radians(70))
    assert MIN_RESIDENCE_S == 0.5
    assert reachable("arm", math.radians(89), False) is True
    assert reachable("arm", math.radians(95), False) is False
    assert reachable("arm", math.radians(-105), True) is True
    assert reachable("arm", math.radians(115), True) is False
    assert reachable("arm_l", math.radians(95), True) is True
    assert reachable("head", math.radians(65), False) is False
    assert reachable("head", math.radians(65), True) is True
    assert reachable("head", math.radians(75), True) is False


# resolver table (rider, face off, target ahead)


def test_resolve_object_id(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att(point="bench_2", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(-5.0, 0.0),)


def test_resolve_object_id_suffix(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    world.add_object(WorldObject(object_id="env_0/table_1", type="table", pose=Pose2D(x=0.0, y=-5.0)))
    agent = _bt_agent(agent_factory, theta=-math.pi / 2)
    node = _node(agent, _att(point="table_1", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(0.0, -5.0),)


def test_resolve_object_id_wins_over_agent_name(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    other = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    node = _node(agent, _att(point="bench_2", face=False), world, rng_np, agents={2: other}, names={"bench_2": 2})
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(-5.0, 0.0),)


def test_resolve_agent_name_gaze_head_height(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    other = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    node = _node(agent, _att(gaze="ped_2", point="ped_2", face=False), world, rng_np, agents={2: other}, names={"ped_2": 2})
    assert _tick(node) == RUNNING
    assert _slots(agent) == {"head": _head(0.0, 5.0, GESTURE_Z_HEAD), "arm": _arm(0.0, 5.0, GESTURE_Z_AGENT)}


def test_resolve_agent_name_wins_over_object_type(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    other = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    node = _node(agent, _att(point="bench", face=False), world, rng_np, agents={2: other}, names={"bench": 2})
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(0.0, 5.0, GESTURE_Z_AGENT),)


def test_resolve_object_type_nearest_with_at_z(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, x=2.0)
    node = _node(agent, _att(point=_ch("bench", at_z=0.3), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0, 0.3),)


def test_resolve_int_agent_id(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    other = _bt_agent(agent_factory, agent_id=7, x=0.0, y=3.0)
    node = _node(agent, _att(point=_ch(7, at_z=1.5), face=False), world, rng_np, agents={7: other})
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(0.0, 3.0, 1.5),)


def test_resolve_pose3_literal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=Pose3(4.0, 0.0, 2.0), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(4.0, 0.0, 2.0),)


def test_resolve_robot_ref_only_matches_robots(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=-math.pi / 2)
    human = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    bot = _bt_agent(agent_factory, agent_id=3, x=0.0, y=-5.0)
    node = _node(agent, _att(point=RobotRef("bot"), face=False), world, rng_np, agents={2: human, 3: bot}, names={"bot": 2})
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    node = _node(agent, _att(point=RobotRef("bot"), face=False), world, rng_np, agents={2: human, 3: bot}, names={"bot": 2}, robots={"bot": 3})
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(0.0, -5.0, GESTURE_Z_AGENT),)


def test_resolve_target_from_ctx_pose(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="target", face=False), world, rng_np, ctx=StepContext(target_pose=Pose2D(x=3.0, y=0.0)))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(3.0, 0.0),)


def test_resolve_target_prefers_ctx_object(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    ctx = StepContext(target_pose=Pose2D(x=3.0, y=0.0), target_object_id="bench_1")
    node = _node(agent, _att(point=_ch("target", at_z=0.5), face=False), world, rng_np, ctx=ctx)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0, 0.5),)


def test_resolve_goal_from_navigate_command(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="goal", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    _mv(agent).command = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=2.0, y=2.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(2.0, 2.0),)
    _mv(agent).command = HighLevelCommand(agent_id=1, type=CommandType.SEEK)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()


@pytest.mark.parametrize("cmd", [None, HighLevelCommand(agent_id=1, type=CommandType.STOP), HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=1.0, y=1.0), desired_velocity=0.0), HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=1.0, y=1.0))])
def test_resolve_goal_unresolved_while_halted(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator, cmd: HighLevelCommand | None) -> None:
    agent = _bt_agent(agent_factory, x=1.0, y=1.0)
    _mv(agent).command = cmd
    node = _node(agent, _att(point="goal", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    _mv(agent).command = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=3.0, y=1.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(3.0, 1.0),)


def test_resolve_partner_nearest_participant(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=1)
    near = _bt_agent(agent_factory, agent_id=2, x=1.0, y=0.5)
    far = _bt_agent(agent_factory, agent_id=3, x=2.0, y=-1.0)
    agents = {1: agent, 2: near, 3: far}
    mgr = _bind(agents)
    node = _node(agent, _att(point="partner", face=False), world, rng_np, agents=agents, ctx=StepContext(im=mgr))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(1.0, 0.5, GESTURE_Z_AGENT),)
    near.state.pose.x = 3.0
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(2.0, -1.0, GESTURE_Z_AGENT),)


def test_resolve_partner_waits_without_interaction(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    mgr = InteractionManager(RNG(0))
    node = _node(agent, _att(point="partner", face=False), world, rng_np, agents={1: agent}, ctx=StepContext(im=mgr))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()


def test_resolve_partners_is_the_expanded_list(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=1)
    b = _bt_agent(agent_factory, agent_id=2, x=1.0, y=0.5)
    c = _bt_agent(agent_factory, agent_id=3, x=2.0, y=-1.0)
    agents = {1: agent, 2: b, 3: c}
    mgr = _bind(agents)
    node = _node(agent, _att(point=_ch("partners", dwell=0.5), face=False), world, rng_np, agents=agents, ctx=StepContext(im=mgr))
    seen = []
    for _ in range(3):
        assert _tick(node) == RUNNING
        seen.append([g.x for g in _mv(agent).gestures])
    assert all(entry in ([1.0], [2.0]) for entry in seen), seen
    assert seen[1] == seen[2], seen


def test_resolve_relative_ref_follows_yaw(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, x=1.0, y=1.0, theta=0.0)
    node = _node(agent, _att(point=RelativeRef(azimuth=45.0, elevation=0.0, distance=2.0)), world, rng_np, bare=True, duration=ParamDist(5.0))
    c = 2.0 * math.cos(math.radians(45.0))
    assert _tick(node) == RUNNING
    g = _slots(agent)["arm"]
    assert (g.x, g.y, g.z) == pytest.approx((1.0 + c, 1.0 + c, GESTURE_Z_AGENT))
    assert _mv(agent).heading_goal is None
    agent.state.pose.theta = math.pi / 2
    assert _tick(node) == RUNNING
    g = _slots(agent)["arm"]
    assert (g.x, g.y, g.z) == pytest.approx((1.0 - c, 1.0 + c, GESTURE_Z_AGENT))
    assert _mv(agent).heading_goal is None


def test_resolve_relative_ref_elevation_head_height(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(gaze=RelativeRef(azimuth=0.0, elevation=30.0, distance=2.0)), world, rng_np)
    assert _tick(node) == RUNNING
    g = _slots(agent)["head"]
    assert (g.x, g.y, g.z) == pytest.approx((2.0 * math.cos(math.radians(30.0)), 0.0, GESTURE_Z_HEAD + 1.0))


# unresolved


@pytest.mark.parametrize("at", ["nothing_here", 99, "target", "goal"])
def test_unresolved_rider_waits(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator, at: object) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=at), world, rng_np)
    for _ in range(3):
        assert _tick(node) == RUNNING
        assert _mv(agent).gestures == ()
        assert _mv(agent).heading_goal is None
    assert node._channels[0].warned is True


def test_unresolved_then_resolved(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    agents: dict[int, BaseAgent] = {}
    node = _node(agent, _att(point=4, face=False), world, rng_np, agents=agents, bare=True, duration=ParamDist(5.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    agents[4] = _bt_agent(agent_factory, agent_id=4, x=0.0, y=2.0)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(0.0, 2.0, GESTURE_Z_AGENT),)


def test_bare_unresolved_fails_after_timeout(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="nobody"), world, rng_np, bare=True)
    for _ in range(int(RESOLVE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
    assert _tick(node) == FAILURE
    assert _mv(agent).gestures == ()


def test_bare_partial_unresolved_fails_on_stuck_entry(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=("bench_1", "nobody"), face=False), world, rng_np, bare=True)
    assert _tick(node) == RUNNING
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    for _ in range(int(RESOLVE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
        assert _mv(agent).gestures == ()
    assert _tick(node) == FAILURE


def test_required_rider_unresolved_fails_after_timeout(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="nobody", required=True), world, rng_np)
    for _ in range(int(RESOLVE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
    assert _tick(node) == FAILURE


def test_cosmetic_rider_unresolved_waits_past_timeout(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="nobody"), world, rng_np)
    for _ in range(int(RESOLVE_TIMEOUT_S / DT) + 3):
        assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()


def test_bare_unresolved_halts(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=3, x=1.0, y=1.0)
    node = _node(agent, _att(point="nobody"), world, rng_np, bare=True)
    assert _tick(node) == RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.agent_id == 3
    assert (cmd.target_pose.x, cmd.target_pose.y) == (1.0, 1.0)
    assert cmd.desired_velocity == 0.0


# face


def test_bare_face_auto_turns_and_shows_against_commanded_heading(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=3, x=1.0, y=1.0, theta=math.pi)
    node = _node(agent, _att(point="bench_1"), world, rng_np, bare=True, duration=ParamDist(1.0))
    bearing = math.atan2(0.0 - 1.0, 5.0 - 1.0)
    for _ in range(2):
        _mv(agent).command = None
        assert _tick(node) == RUNNING
        assert _mv(agent).heading_goal == pytest.approx(bearing)
        assert _mv(agent).gestures == (_arm(5.0, 0.0),)
        cmd = _mv(agent).command
        assert cmd is not None
        assert cmd.desired_velocity == 0.0
    agent.state.pose.theta = bearing + 0.2
    assert _tick(node) == SUCCESS
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gestures == ()


def test_rider_walking_skips_facing_and_hides_behind(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    moving = [True]
    node = _node(agent, _att(point="bench_1"), world, rng_np, walking=lambda: moving[0])
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gestures == ()
    moving[0] = False
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)


def test_face_true_never_writes_heading_while_bound(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 4)
    ctx = StepContext(is_bound_lookup=lambda _aid: True)
    node = _node(agent, _att(point="bench_1", face=True), world, rng_np, ctx=ctx)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)


def test_bare_bound_keeps_formation_command(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    sentinel = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=5.0, y=0.0))
    _mv(agent).command = sentinel
    ctx = StepContext(is_bound_lookup=lambda _aid: True)
    node = _node(agent, _att(point="bench_1"), world, rng_np, ctx=ctx, bare=True, duration=ParamDist(1.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).command is sentinel


def test_face_false_never_turns(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att(point="bench_1", face=False), world, rng_np, bare=True, duration=ParamDist(5.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gestures == ()


def test_face_ref_turns_toward_ref(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    node = _node(agent, _att(point="lamp_1", face="bench_2"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(math.pi)


def test_face_precedence_point_over_gaze(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    node = _node(agent, _att(gaze="bench_2", point="bench_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    node = _node(agent, _att(gaze="bench_2", point_l="bench_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    node = _node(agent, _att(gaze="bench_2"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(math.pi)


def test_face_skips_relative_entries(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    node = _node(agent, _att(point=_ch((RelativeRef(azimuth=0.0, elevation=0.0), "bench_1"), dwell=0.5)), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)


def test_face_hysteresis_enter_keep(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=FACE_ENTER_RAD + 0.05)
    other = _bt_agent(agent_factory, agent_id=2, x=5.0, y=0.0)
    node = _node(agent, _att(point=2), world, rng_np, agents={2: other}, bare=True, duration=ParamDist(50.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    agent.state.pose.theta = FACE_ENTER_RAD - 0.05
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    agent.state.pose.theta = 0.0
    other.state.pose.y = 5.0 * math.tan(FACE_KEEP_RAD - 0.05)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    other.state.pose.y = 5.0 * math.tan(FACE_KEEP_RAD + 0.1)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(math.atan2(other.state.pose.y, 5.0))
    other.state.pose.y = 5.0 * math.tan(FACE_KEEP_RAD - 0.05)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(math.atan2(other.state.pose.y, 5.0))
    other.state.pose.y = 0.0
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None


def test_bare_face_timeout_fails(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att(point="bench_1"), world, rng_np, bare=True, duration=ParamDist(50.0))
    for _ in range(int(FACE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
    assert _tick(node) == FAILURE
    assert _mv(agent).gestures == ()
    assert _mv(agent).heading_goal is None


def test_rider_face_timeout_gives_up_and_hides_out_of_reach(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att(point="bench_1", face=True), world, rng_np)
    for _ in range(int(FACE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
        assert _mv(agent).heading_goal == pytest.approx(0.0)
        assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gestures == ()


# hold


def test_release_clears_on_terminate(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=0.1)
    node = _node(agent, _att(point="bench_1", gaze="lamp_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert set(_slots(agent)) == {"arm", "head"}
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).gestures == ()
    assert _mv(agent).heading_goal is None


def test_keep_leaves_intent_until_slot_taken_over(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch("bench_1", hold="keep"), gaze="lamp_1"), world, rng_np)
    assert _tick(node) == RUNNING
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    node = _node(agent, _att(gaze="lamp_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _slots(agent) == {"arm": _arm(5.0, 0.0), "head": _head(5.0, 2.0)}
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    node = _node(agent, _att(point="lamp_1", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 2.0),)


def test_dwell_list_end_keeps_tracking_and_keep_freezes_on_terminate(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    other = _bt_agent(agent_factory, agent_id=2, x=4.0, y=0.0)
    node = _node(agent, _att(point=_ch(2, hold="keep", dwell=0.5), face=False), world, rng_np, agents={2: other})
    assert _tick(node) == RUNNING
    assert node._channels[0].done is True
    other.state.pose.x = 4.5
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(4.5, 0.0, GESTURE_Z_AGENT),)
    node.stop(py_trees.common.Status.INVALID)
    other.state.pose.x = 6.0
    assert _mv(agent).gestures == (_arm(4.5, 0.0, GESTURE_Z_AGENT),)


def test_terminate_clears_heading_mid_face(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att(point="bench_1", face=True), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is not None
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).heading_goal is None


# lists and dwell


def test_single_ref_dwell_then_success(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="bench_1"), world, rng_np, bare=True)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gestures == ()


def test_dwell_list_ends(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=("bench_1", "lamp_1"), face=False), world, rng_np, bare=True)
    xs = []
    while _tick(node) == RUNNING:
        xs.append(_slots(agent)["arm"].y)
    assert node.status == SUCCESS
    assert xs == [0.0, 0.0, 2.0]
    assert _mv(agent).gestures == ()


def test_duration_ends_regardless_of_list(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "lamp_1"), dwell=0.5), face=False), world, rng_np, bare=True, duration=ParamDist(2.5))
    ys = []
    while _tick(node) == RUNNING:
        ys.append([g.y for g in _mv(agent).gestures])
    assert node.status == SUCCESS
    assert ys == [[0.0], [2.0], [2.0], [2.0], [2.0]]


def test_rider_dwell_list_ends_on_last_entry(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "lamp_1"), dwell=0.5), face=False), world, rng_np)
    ys = []
    for _ in range(4):
        assert _tick(node) == RUNNING
        ys.append([g.y for g in _mv(agent).gestures])
    assert ys == [[0.0], [2.0], [2.0], [2.0]]
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).gestures == ()


def test_dwell_waits_on_unresolved_entry(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    agents: dict[int, BaseAgent] = {}
    node = _node(agent, _att(point=_ch((9, "bench_1"), dwell=0.5), face=False), world, rng_np, agents=agents)
    for _ in range(3):
        assert _tick(node) == RUNNING
        assert _mv(agent).gestures == ()
    agents[9] = _bt_agent(agent_factory, agent_id=9, x=3.0, y=0.0)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(3.0, 0.0, GESTURE_Z_AGENT),)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)


# reach and unreachable cycling


def test_hysteresis_in_out(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=-math.radians(100))
    node = _node(agent, _att(point="bench_1", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    agent.state.pose.theta = -math.radians(80)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = -math.radians(100)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = -math.radians(115)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    agent.state.pose.theta = -math.radians(100)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()


def test_head_envelope_is_tighter(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=-math.radians(65))
    node = _node(agent, _att(gaze="bench_1", point="bench_1", face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert set(_slots(agent)) == {"arm"}
    agent.state.pose.theta = -math.radians(55)
    assert _tick(node) == RUNNING
    assert set(_slots(agent)) == {"arm", "head"}


def test_unreachable_cycles_on_leaving_reach(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "bench_2"), advance="unreachable"), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = math.pi
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(-5.0, 0.0),)
    for _ in range(3):
        assert _tick(node) == RUNNING
        assert _mv(agent).gestures == (_arm(-5.0, 0.0),)
    agent.state.pose.theta = 0.0
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)


def test_unreachable_jumps_past_unreachable_entries(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    world.add_object(WorldObject(object_id="a", type="a", pose=Pose2D(x=5.0, y=0.0)))
    world.add_object(WorldObject(object_id="b", type="b", pose=Pose2D(x=5.0, y=-5.0)))
    world.add_object(WorldObject(object_id="c", type="c", pose=Pose2D(x=0.0, y=5.0)))
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("a", "b", "c"), advance="unreachable"), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = math.radians(135)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(0.0, 5.0),)


def test_unreachable_holds_when_none_reachable(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "lamp_1"), advance="unreachable"), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = math.radians(143)
    for _ in range(3):
        assert _tick(node) == RUNNING
        assert _mv(agent).gestures == ()
    assert node._channels[0].idx == 0
    agent.state.pose.theta = math.radians(100)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 2.0),)


def test_unreachable_respects_min_residence(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "bench_2"), advance="unreachable"), face=False), world, rng_np, dt=0.2)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = math.pi
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(-5.0, 0.0),)


def test_unreachable_never_advances_while_face_turn_in_flight(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "bench_2"), advance="unreachable")), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    agent.state.pose.theta = math.pi
    for _ in range(3):
        assert _tick(node) == RUNNING
        assert _mv(agent).heading_goal == pytest.approx(0.0)
        assert _mv(agent).gestures == (_arm(5.0, 0.0),)


def test_unreachable_needs_in_after_showing_nothing(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=-math.radians(100))
    node = _node(agent, _att(point=_ch("bench_1", advance="unreachable"), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    agent.state.pose.theta = -math.radians(85)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)


# published intent shape


def test_point_carries_dominant_hand(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    agent.params = attrs.evolve(agent.params, handedness="l")
    node = _node(agent, _att(point="bench_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0, dominant="l"),)


def test_explicit_arm_slots(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point_l="bench_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (GestureIntent("arm_l", 5.0, 0.0, GESTURE_Z_OBJECT),)
    node.stop(py_trees.common.Status.INVALID)
    node = _node(agent, _att(point_r="bench_1"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (GestureIntent("arm_r", 5.0, 0.0, GESTURE_Z_OBJECT),)


def test_all_shown_channels_published(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    other = _bt_agent(agent_factory, agent_id=2, x=4.0, y=1.0)
    node = _node(agent, _att(gaze=2, point="bench_1", face=False), world, rng_np, agents={2: other})
    assert _tick(node) == RUNNING
    assert _slots(agent) == {"head": _head(4.0, 1.0, GESTURE_Z_HEAD), "arm": _arm(5.0, 0.0)}


def test_tracked_agent_reresolves_after_respawn(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    agents = {2: _bt_agent(agent_factory, agent_id=2, x=4.0, y=0.0)}
    names = {"ped_2": 2}
    node = _node(agent, _att(point="ped_2", face=False), world, rng_np, agents=agents, names=names, bare=True, duration=ParamDist(50.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(4.0, 0.0, GESTURE_Z_AGENT),)
    del agents[2]
    del names["ped_2"]
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    agents[9] = _bt_agent(agent_factory, agent_id=9, x=3.0, y=1.0)
    names["ped_2"] = 9
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(3.0, 1.0, GESTURE_Z_AGENT),)


# suspend / resume


def test_suspend_lowers_and_resume_keeps_index(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point=_ch(("bench_1", "lamp_1"), dwell=1.0), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    node.suspend()
    assert _mv(agent).gestures == ()
    assert node.status == RUNNING
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 0.0),)
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (_arm(5.0, 2.0),)


# clip


def _clip(name: str, when: str = "always") -> ClipDef:
    return ClipDef(name=name, when=when)


def test_clip_publishes_body_slot_and_releases_on_terminate(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(point="bench_1", clip=_clip("wave"), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _slots(agent)["body"] == GestureIntent("body", clip="wave")
    assert _slots(agent)["arm"] == _arm(5.0, 0.0)
    node.suspend()
    assert _mv(agent).gestures == ()
    assert _tick(node) == RUNNING
    assert "body" in _slots(agent)
    node.stop(SUCCESS)
    assert "body" not in _slots(agent)


def test_bare_clip_only_halts_and_runs_for_duration(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(clip=_clip("sit")), world, rng_np, bare=True, duration=ParamDist(1.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (GestureIntent("body", clip="sit"),)
    assert _mv(agent).command is not None and _mv(agent).command.desired_velocity == 0.0
    assert _tick(node) == RUNNING
    assert _tick(node) == SUCCESS


def test_clip_when_bound_waits_for_the_interaction(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    bound = {"v": False}
    node = _node(agent, _att(clip=_clip("wave", when="bound")), world, rng_np, ctx=StepContext(is_bound_lookup=lambda _aid: bound["v"]))
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()
    bound["v"] = True
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == (GestureIntent("body", clip="wave"),)
    bound["v"] = False
    assert _tick(node) == RUNNING
    assert _mv(agent).gestures == ()


def _hold(mgr: InteractionManager, agents: dict[int, BaseAgent]) -> None:
    """Walk every participant onto its formation slot and tick the manager so the hold latches."""
    for aid, agent in agents.items():
        target = mgr.formation_target(aid)
        assert target is not None
        agent.state.pose = Pose2D(x=target.x, y=target.y, theta=target.theta)
    mgr.update({})


@pytest.mark.parametrize("itype, clip", [(InteractionType.HUG, "hug"), (InteractionType.SHAKE_HAND, "shake_hand")])
def test_contact_clip_publishes_render_pose_override_once_holding(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator, itype: InteractionType, clip: str) -> None:
    a = _bt_agent(agent_factory, agent_id=1, x=0.0, y=0.0)
    b = _bt_agent(agent_factory, agent_id=2, x=2.0, y=0.0)
    agents = {1: a, 2: b}
    mgr = _bind(agents, itype)
    node = _node(a, _att(clip=_clip(clip, when="bound")), world, rng_np, agents=agents, ctx=StepContext(im=mgr, is_bound_lookup=mgr.is_bound))
    assert _tick(node) == RUNNING
    assert "body" not in _slots(a)
    _hold(mgr, agents)
    assert _tick(node) == RUNNING
    target = mgr.formation_target(1)
    assert target is not None
    body = _slots(a)["body"]
    assert body.clip == clip
    assert body.render_pose_override is True
    assert (body.x, body.y) == pytest.approx((target.x, target.y))


def test_locomotion_only_contact_never_publishes_the_clip(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    a = _bt_agent(agent_factory, agent_id=1, x=0.0, y=0.0)
    b = _bt_agent(agent_factory, agent_id=2, x=2.0, y=0.0)
    agents = {1: a, 2: b}
    mgr = InteractionManager(RNG(0))
    mgr.set_contact_mode(CONTACT_LOCOMOTION_ONLY)
    mgr.set_context(world_knowledge=WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid), visibility_lookup=lambda aid: set(agents) - {aid})
    mgr.update({aid: HighLevelCommand(agent_id=aid, type=CommandType.SEEK, spec=SeekSpec(interaction_type=InteractionType.HUG)) for aid in agents})
    node = _node(a, _att(clip=_clip("hug", when="bound")), world, rng_np, agents=agents, ctx=StepContext(im=mgr, is_bound_lookup=mgr.is_bound))
    _hold(mgr, agents)
    assert mgr.is_holding(next(iter(mgr.interactions.values())))
    assert _tick(node) == RUNNING
    assert "body" not in _slots(a)


def test_non_contact_clip_does_not_publish_render_pose_override(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    a = _bt_agent(agent_factory, agent_id=1, x=0.0, y=0.0)
    b = _bt_agent(agent_factory, agent_id=2, x=2.0, y=0.0)
    agents = {1: a, 2: b}
    mgr = _bind(agents, InteractionType.GROUP_CONVERSATION)
    node = _node(a, _att(clip=_clip("talk_with_arm_gesture", when="bound")), world, rng_np, agents=agents, ctx=StepContext(im=mgr, is_bound_lookup=mgr.is_bound))
    assert _tick(node) == RUNNING
    body = _slots(a)["body"]
    assert body.render_pose_override is False
    assert (body.x, body.y) == (0.0, 0.0)


def test_posture_held_for_the_step_and_dropped_after(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(clip=_clip("collapse_to_ground"), posture="prone"), world, rng_np, bare=True, duration=ParamDist(1.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).posture == "prone"
    node.suspend()
    assert _mv(agent).posture == ""
    assert _tick(node) == RUNNING
    assert _mv(agent).posture == "prone"
    node.stop(SUCCESS)
    assert _mv(agent).posture == ""


def test_role_clip_publishes_each_members_clip_of_the_drawn_set(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    a = _bt_agent(agent_factory, agent_id=1, x=0.0, y=0.0)
    b = _bt_agent(agent_factory, agent_id=2, x=2.0, y=0.0)
    agents = {1: a, 2: b}
    mgr = _bind(agents, InteractionType.HUG)
    interaction = next(iter(mgr.interactions.values()))
    fallback = InteractionType.HUG.kind.clip
    nodes = {aid: _node(agent, _att(clip=ClipDef(name=fallback, when="bound", by_role=True)), world, rng_np, agents=agents, ctx=StepContext(im=mgr, is_bound_lookup=mgr.is_bound)) for aid, agent in agents.items()}
    _hold(mgr, agents)
    for aid, node in nodes.items():
        assert _tick(node) == RUNNING
        body = _slots(agents[aid])["body"]
        assert body.clip == mgr.role_clip(interaction, aid)
        assert body.render_pose_override is True
    assert _slots(a)["body"].clip != _slots(b)["body"].clip or len(mgr.clip_set(interaction).roles) == 1
