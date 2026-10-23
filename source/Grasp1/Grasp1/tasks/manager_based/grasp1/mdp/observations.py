"""构造Teacher策略观测：UR5-Allegro为153维（下列布局），FR3-Inspire为132维。

FR3使用19个受控关节、13个真实手部link点及7个臂部link点，字段顺序保持一致。

前 102 维严格保持 ``Environment.hpp::updateObservation`` 的连续字段顺序；
末尾 51 维对应 ``RaisimGymVecEnvOther.observe_vision_new`` 的
17 个 nearest-affordance vectors。

布局::

    0:22      gc_r_
    22:44     actual clamped target - current joint position
    44:57     contacts_r_af
    57:70     impulses_r_af
    70:87     17 hand keypoint heights
    87:93     6 UR5 keypoint heights
    93:96     hand_center in the environment-local world frame
    96:99     wrist Euler relative to reset pose
    99:102    current wrist Euler
    102:153   17 * 3 affordance vectors in world frame

所有逐步计算均保持为 Torch tensor。物体 mesh 在 observation term 初始化时
按不同物体各采样一次，并把同一份点云发布到 ``env._teacher_affordance_points_o``，
供 reset / reward / observation 共用，避免三个模块各自随机采样一套 affordance。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from Grasp1.data.object_set import POINTS_PER_MESH, load_unique_object_metadata
from Grasp1.robots.robot_profile import (
    ARM_CONTACT_LINK_NAMES,
    CONTROLLED_JOINT_NAMES,
    HAND_CENTER,
    HAND_CONTACT_LINK_NAMES,
    HAND_KEYPOINT_COUNT,
    NUM_CONTROLLED_JOINTS,
    UR5_LINK_NAMES,
    FR3_JOINT_NAMES, FR3_LINK_NAMES, INSPIRE_JOINT_NAMES,
    INSPIRE_HAND_KEYPOINT_NAMES, INSPIRE_DISTAL_LINK_NAMES,
)

from .keypoints import (
    arm_keypoints_w,
    hand_keypoints_w,
    hand_keypoints_o,
    initialize_teacher_body_indices,
    _object_top_pose_w,
)



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


# -----------------------------------------------------------------------------
# Teacher observation constants
# -----------------------------------------------------------------------------

TEACHER_BASE_OBSERVATION_DIM: Final[int] = 102
TEACHER_AFFORDANCE_OBSERVATION_DIM: Final[int] = HAND_KEYPOINT_COUNT * 3
TEACHER_OBSERVATION_DIM: Final[int] = (
    TEACHER_BASE_OBSERVATION_DIM + TEACHER_AFFORDANCE_OBSERVATION_DIM
)

AFFORDANCE_POINT_COUNT: Final[int] = POINTS_PER_MESH
CONTACT_IMPULSE_THRESHOLD: Final[float] = 0.01

AFFORDANCE_CONTACT_SENSOR_NAMES: Final[tuple[str, ...]] = tuple(
    f"teacher_af_contact_{index}"
    for index in range(len(HAND_CONTACT_LINK_NAMES))
)

_RUNTIME_CONTROLLED_JOINT_NAMES: Final[tuple[str, ...]] = tuple(
    name.replace(".", "_")
    for name in CONTROLLED_JOINT_NAMES
)

_RUNTIME_CONTACT_BODY_NAMES: Final[tuple[str, ...]] = tuple(
    name.replace(".", "_")
    for name in HAND_CONTACT_LINK_NAMES
)


# -----------------------------------------------------------------------------
# Affordance points needed by the observation manager
# -----------------------------------------------------------------------------

def _load_affordance_points(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return the shared affordance point cache in environment order.

    IsaacLab constructs ``ObservationManager`` before applying startup events,
    so the policy observation has to make the 200-point affordance sample
    available during manager construction.  The cache is stored on ``env`` so
    task31 reset geometry and task30 rewards use this exact same sample.
    """
    object_names = tuple(env.cfg.object_names)

    cached = getattr(env, "_teacher_affordance_points_o", None)
    if isinstance(cached, torch.Tensor):
        if cached.shape != (env.num_envs, AFFORDANCE_POINT_COUNT, 3):
            raise RuntimeError(
                "_teacher_affordance_points_o shape mismatch: "
                f"expected=({env.num_envs},{AFFORDANCE_POINT_COUNT},3), "
                f"got={tuple(cached.shape)}."
            )
        return cached

    metadata = load_unique_object_metadata(
        env.cfg.object_dataset,
        object_names,
    )
    env._teacher_object_metadata = metadata
    points = torch.as_tensor(
        np.stack([metadata[name].affordance_points for name in object_names]),
        dtype=torch.float32,
        device=env.device,
    )

    if points.shape != (env.num_envs, AFFORDANCE_POINT_COUNT, 3):
        raise RuntimeError(
            "Teacher affordance points shape mismatch: "
            f"expected=({env.num_envs},{AFFORDANCE_POINT_COUNT},3), "
            f"got={tuple(points.shape)}."
        )

    env._teacher_affordance_points_o = points
    return points


# -----------------------------------------------------------------------------
# RaiSim-compatible Euler conversion
# -----------------------------------------------------------------------------

def _raisim_euler(
    quat_w: torch.Tensor,
) -> torch.Tensor:
    """Convert ``wxyz`` quaternion to the Euler convention used by Teacher."""
    rotation = matrix_from_quat(quat_w)

    cy = torch.sqrt(
        rotation[..., 2, 2].square()
        + rotation[..., 1, 2].square()
    )

    # Keep the old double-precision singularity convention.  Away from the
    # exact gimbal-lock branch this is equivalent to the source formula.
    regular = cy > 8.881784197001252e-16

    roll = torch.where(
        regular,
        -torch.atan2(
            rotation[..., 1, 2],
            rotation[..., 2, 2],
        ),
        torch.zeros_like(cy),
    )
    pitch = -torch.atan2(
        -rotation[..., 0, 2],
        cy,
    )
    yaw = torch.where(
        regular,
        -torch.atan2(
            rotation[..., 0, 1],
            rotation[..., 0, 0],
        ),
        -torch.atan2(
            -rotation[..., 1, 0],
            rotation[..., 1, 1],
        ),
    )

    return torch.stack(
        (roll, pitch, yaw),
        dim=-1,
    )


def _resolve_exact_body_id(
    asset: Articulation,
    body_name: str,
) -> int:
    """Resolve one literal body name and validate uniqueness."""
    try:
        body_ids, resolved_names = asset.find_bodies(
            re.escape(body_name),
            preserve_order=True,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Unable to resolve body {body_name!r}. "
            f"Available body names: {tuple(asset.body_names)}."
        ) from exc

    if (
        len(body_ids) != 1
        or len(resolved_names) != 1
        or resolved_names[0] != body_name
    ):
        raise RuntimeError(
            f"Body {body_name!r} must resolve exactly once, got "
            f"ids={body_ids}, names={resolved_names}."
        )

    return int(body_ids[0])


def nearest_affordance_vectors_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """共享当前步世界系最近 affordance 向量 [N,H,3]，不计算完整观测。"""
    def compute(ids: slice | torch.Tensor) -> torch.Tensor:
        """只为本次失效行计算最近点；成对距离仅作为临时张量。"""
        _, object_quat_w = _object_top_pose_w(env)
        hand_positions_o = hand_keypoints_o(env)[ids]
        points_o = env._teacher_affordance_points_o[ids]
        nearest_index = torch.cdist(hand_positions_o, points_o).argmin(dim=-1)
        nearest_points_o = torch.gather(
            points_o, dim=1, index=nearest_index.unsqueeze(-1).expand(-1, -1, 3)
        )
        vectors_o = nearest_points_o - hand_positions_o
        quaternions = object_quat_w[ids, None, :].expand(-1, hand_positions_o.shape[1], -1)
        return quat_apply(quaternions.reshape(-1,4), vectors_o.reshape(-1,3)).reshape_as(vectors_o)

    return compute(slice(None))


# -----------------------------------------------------------------------------
# Stateful Manager observation
# -----------------------------------------------------------------------------

class TeacherObservation(ManagerTermBase):
    """保持原153维字段顺序，与奖励共享接触采样。"""

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
    ) -> None:
        super().__init__(cfg, env)
        self._base_dim, self._observation_dim = 102, 153
        if getattr(env.cfg, "robot_type", "ur5_allegro") == "fr3_inspire":
            self._initialize_fr3(env)
            return

        self._robot = env.scene["robot"]
        self._object = env.scene["object"]
        initialize_teacher_body_indices(env)


        if not isinstance(self._robot, Articulation):
            raise TypeError(
                "Teacher scene entry 'robot' must be an Articulation, "
                f"got {type(self._robot)!r}."
            )
        if not isinstance(self._object, Articulation):
            raise TypeError(
                "Teacher scene entry 'object' must be an Articulation, "
                f"got {type(self._object)!r}."
            )

        # Resolve the 22 controlled joints exactly once.  Per-step regex/name
        # resolution is unnecessary CPU work and can silently change ordering
        # if an expression becomes ambiguous.
        joint_expressions = [
            re.escape(name)
            for name in _RUNTIME_CONTROLLED_JOINT_NAMES
        ]
        joint_ids, joint_names = self._robot.find_joints(
            joint_expressions,
            preserve_order=True,
        )

        if (
            len(joint_ids) != NUM_CONTROLLED_JOINTS
            or tuple(joint_names)
            != _RUNTIME_CONTROLLED_JOINT_NAMES
        ):
            raise RuntimeError(
                "Teacher controlled-joint resolution mismatch. "
                f"Expected={_RUNTIME_CONTROLLED_JOINT_NAMES}, "
                f"resolved={tuple(joint_names)}."
            )

        self._joint_ids = list(joint_ids)

        # Original hand/wrist frame is reconstructed from wrist_3_link plus
        # the URDF fixed Rx(-pi/2) chain.
        self._wrist_body_id = _resolve_exact_body_id(
            self._robot,
            UR5_LINK_NAMES[-1],
        )
        self._object_top_body_id = _resolve_exact_body_id(
            self._object,
            "top",
        )

        zeros = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=self._robot.data.joint_pos.dtype,
        )
        self._wrist_offset_quat = quat_from_euler_xyz(
            torch.full_like(zeros, -math.pi / 2.0),
            zeros,
            zeros,
        )
        self._hand_center = torch.tensor(
            HAND_CENTER,
            device=env.device,
            dtype=self._robot.data.joint_pos.dtype,
        )

        # Cache and validate the 13 one-body filtered contact sensors.
        for (
            sensor_name,
            expected_body_name,
        ) in zip(
            AFFORDANCE_CONTACT_SENSOR_NAMES,
            _RUNTIME_CONTACT_BODY_NAMES,
            strict=True,
        ):
            try:
                sensor = env.scene[sensor_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing Teacher contact sensor {sensor_name!r}. "
                    "teacher_env_cfg.py must create one filtered "
                    "ContactSensor for every HAND_CONTACT_LINK_NAMES entry."
                ) from exc

            if not isinstance(sensor, ContactSensor):
                raise TypeError(
                    f"{sensor_name!r} must be ContactSensor, "
                    f"got {type(sensor)!r}."
                )

            if sensor.num_bodies != 1:
                raise RuntimeError(
                    f"{sensor_name!r} must match exactly one robot body, "
                    f"got {sensor.num_bodies}: {sensor.body_names}."
                )

            if sensor.body_names[0] != expected_body_name:
                raise RuntimeError(
                    f"{sensor_name!r} resolved the wrong body: "
                    f"expected={expected_body_name!r}, "
                    f"resolved={sensor.body_names[0]!r}."
                )

        # Geometry has to be available for ObservationManager's initial
        # dimension probe, which occurs before startup events.
        _load_affordance_points(env)
        env._teacher_reward_state = TeacherRewardState(env)
        env._teacher_grasp = TeacherGraspState(env)

        # Source reset_state():
        #   wrist_mat_r_init = current wrist matrix
        #   wrist_euler_previous = 0
        self._initial_wrist_quat_w = torch.zeros(
            (env.num_envs, 4),
            device=env.device,
            dtype=self._robot.data.joint_pos.dtype,
        )
        self._previous_wrist_euler = torch.zeros(
            (env.num_envs, 3),
            device=env.device,
            dtype=self._robot.data.joint_pos.dtype,
        )
        self._needs_wrist_init = torch.ones(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    def _initialize_fr3(self, env):
        """沿用字段顺序；真实19轴、13手部link点、7臂部link点。"""
        self._robot, self._object = env.scene["robot"], env.scene["object"]
        initialize_teacher_body_indices(env)
        names = FR3_JOINT_NAMES + INSPIRE_JOINT_NAMES
        self._joint_ids, resolved = self._robot.find_joints(list(names), preserve_order=True)
        if tuple(resolved) != names:
            raise RuntimeError(f"FR3-Inspire runtime joints differ: {resolved}")
        self._wrist_body_id = env._teacher_body_indices.wrist
        self._object_top_body_id = env._teacher_body_indices.object_top
        self._wrist_offset_quat = torch.zeros(env.num_envs, 4, device=env.device)
        self._wrist_offset_quat[:, 0] = 1.
        self._hand_center = torch.tensor(env.cfg.palm_center, device=env.device)
        self._base_dim, self._observation_dim = 93, 132
        _load_affordance_points(env)
        env._teacher_reward_state = TeacherRewardState(env)
        env._teacher_grasp = TeacherGraspState(env)
        self._initial_wrist_quat_w = torch.zeros(env.num_envs, 4, device=env.device)
        self._previous_wrist_euler = torch.zeros(env.num_envs, 3, device=env.device)
        self._needs_wrist_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Reset the wrist-reference state for selected environments."""
        if env_ids is None:
            env_ids = slice(None)

        self._needs_wrist_init[env_ids] = True
        self._previous_wrist_euler[env_ids] = 0.0

    def _joint_observation(
        self,
        env: ManagerBasedRLEnv,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回关节位置和实际发送的限位目标相对当前测量位置的误差。"""
        joint_pos = self._robot.data.joint_pos[
            :,
            self._joint_ids,
        ]

        action_term = env.action_manager.get_term(
            "teacher"
        )
        target = action_term.target - joint_pos

        if target.shape != joint_pos.shape:
            raise RuntimeError(
                "Teacher action/observation shape mismatch: "
                f"target={tuple(target.shape)}, "
                f"joint_pos={tuple(joint_pos.shape)}."
            )

        return joint_pos, target

    def _contact_observation(
        self,
        env: ManagerBasedRLEnv,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """13个槽的完整接触冲量：同一物理步的（法向力 + 摩擦力）× dt。"""
        env._teacher_grasp.sample(env._sim_step_counter)
        state = env._teacher_reward_state
        state.update()
        impulses = state.affordance_impulses.norm(dim=-1)
        contacts = (
            impulses > CONTACT_IMPULSE_THRESHOLD
        ).to(impulses.dtype)

        return contacts, impulses

    def _wrist_observation(
        self,
        hand_positions_w: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return hand center, relative wrist Euler and current wrist Euler."""
        wrist_parent_quat_w = (
            self._robot.data.body_link_pose_w[
                :,
                self._wrist_body_id,
                3:7,
            ]
        )

        wrist_quat_w = quat_mul(
            wrist_parent_quat_w,
            self._wrist_offset_quat,
        )

        # hand_positions_w[:,0] is the recovered source
        # Flange2hand_fixed_joint position from keypoints.py.
        hand_center_w = (
            hand_positions_w[:, 0]
            + quat_apply(
                wrist_quat_w,
                self._hand_center.unsqueeze(0).expand(
                    hand_positions_w.shape[0],
                    -1,
                ),
            )
        )

        init_mask = self._needs_wrist_init
        if torch.any(init_mask):
            self._initial_wrist_quat_w[
                init_mask
            ] = wrist_quat_w[init_mask]
            self._previous_wrist_euler[
                init_mask
            ] = 0.0
            self._needs_wrist_init[
                init_mask
            ] = False

        relative_quat = quat_mul(
            quat_inv(self._initial_wrist_quat_w),
            wrist_quat_w,
        )
        euler_diff = _raisim_euler(
            relative_quat
        )
        wrist_euler = _raisim_euler(
            wrist_quat_w
        )

        # Exact source continuity rule: only unwrap when the previous Euler
        # norm is already non-trivial.
        unwrap = (
            torch.linalg.vector_norm(
                self._previous_wrist_euler,
                dim=-1,
                keepdim=True,
            )
            > 0.01
        )
        difference = (
            wrist_euler
            - self._previous_wrist_euler
        )

        wrist_euler = torch.where(
            unwrap & (difference > math.pi),
            wrist_euler - 2.0 * math.pi,
            wrist_euler,
        )
        wrist_euler = torch.where(
            unwrap & (difference < -math.pi),
            wrist_euler + 2.0 * math.pi,
            wrist_euler,
        )

        self._previous_wrist_euler.copy_(
            wrist_euler
        )

        return (
            hand_center_w,
            euler_diff,
            wrist_euler,
        )

    def _affordance_vectors(
        self,
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
        """Return flattened Hx3 nearest-affordance vectors in world frame."""
        return nearest_affordance_vectors_w(env).reshape(
            env.num_envs, -1
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        support_height: float,
    ) -> torch.Tensor:
        """返回当前机器人训练和播放共享的观测。"""
        joint_pos, target_error = (
            self._joint_observation(env)
        )
        contacts, impulses = (
            self._contact_observation(env)
        )

        hand_positions_w = hand_keypoints_w(
            env
        )
        arm_positions_w = arm_keypoints_w(
            env
        )

        (
            hand_center_w,
            euler_diff,
            wrist_euler,
        ) = self._wrist_observation(
            hand_positions_w
        )

        base_observation = torch.cat(
            (
                joint_pos,  # 0:22
                target_error,  # 22:44
                contacts,  # 44:57
                impulses,  # 57:70
                hand_positions_w[..., 2]
                - env.scene.env_origins[:, None, 2] - support_height,  # 70:87
                arm_positions_w[..., 2]
                - env.scene.env_origins[:, None, 2] - support_height,  # 87:93
                # XY 相对真实底座；Z 保留源 Teacher 相对地面的高度语义。
                hand_center_w - self._robot.data.root_pos_w
                + hand_center_w.new_tensor((0.0, 0.0, support_height)),  # 93:96
                euler_diff,  # 96:99
                wrist_euler,  # 99:102
            ),
            dim=-1,
        )

        if base_observation.shape != (
            env.num_envs,
            self._base_dim,
        ):
            raise RuntimeError(
                "Teacher base observation shape mismatch: "
                f"expected=({env.num_envs},"
                f"{self._base_dim}), "
                f"got={tuple(base_observation.shape)}."
            )

        affordance_vectors = (
            self._affordance_vectors(
                env,
            )
        )

        observation = torch.cat(
            (
                base_observation,
                affordance_vectors,
            ),
            dim=-1,
        )

        if observation.shape != (
            env.num_envs,
            self._observation_dim,
        ):
            raise RuntimeError(
                "Teacher observation shape mismatch: "
                f"expected=({env.num_envs},"
                f"{self._observation_dim}), "
                f"got={tuple(observation.shape)}."
            )

        return observation


def contact_force(env, name):
    """单刚体、单过滤对的法向接触力；空接触不是虚构接触。"""
    return env.scene[name].data.force_matrix_w[:, 0].sum(dim=1)


def dexsuite_good_finger_contact(env, threshold=1.0):
    """共享DexSuite双指接触：拇指及至少另一指的合成法向力严格超过阈值。"""
    if getattr(env.cfg, "robot_type", "ur5_allegro") == "fr3_inspire":
        forces = torch.stack([contact_force(env, f"teacher_af_contact_{slot}")
                              for slot in (2, 4, 6, 8, 12)], dim=1)
        contact = forces.norm(dim=-1) > threshold
        return contact[:, -1] & contact[:, :-1].any(1)
    # distal link与独立fixed tip的法向力先作向量和，再取模；不混入摩擦力或冲量。
    forces = torch.stack([
        contact_force(env, f"teacher_af_contact_{slot}") + contact_force(env, f"teacher_tip_object_{finger}")
        for finger, slot in enumerate((3, 6, 9, 12))
    ], dim=1)
    contact = forces.norm(dim=-1) > threshold
    return contact[:, 3] & contact[:, :3].any(1)


def contact_friction_force(env, name):
    """PhysX 实际切向摩擦力（N）；世界 XY 分量不等于接触切向分量。"""
    return env.scene[name].data.friction_forces_w[:, 0].sum(dim=1)


class TeacherRewardState:
    """末物理步的源接触槽、四项机械臂接触标志及Reset参考位置。

    PhysX按normal contact和friction patch分别报告数据，无法逐点对应RaiSim
    的完整冲量；源逐接触点|J|<0.001过滤尚不能等效复现，不能改成聚合后过滤。
    这里保留真实合冲量，不另造阈值；其余接触阈值和奖励裁剪仍使用源公式。
    """

    def __init__(self, env):
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdPhysics
        import omni.usd

        self.env = env
        robot = env.scene["robot"]
        self.body_names = robot.body_names
        fr3 = getattr(env.cfg, "robot_type", "ur5_allegro") == "fr3_inspire"
        hand_names = INSPIRE_HAND_KEYPOINT_NAMES if fr3 else _RUNTIME_CONTACT_BODY_NAMES
        arm_names = FR3_LINK_NAMES if fr3 else ARM_CONTACT_LINK_NAMES
        self.arm_count = len(arm_names)
        self.inner_arm_count = self.arm_count - 2
        n, device = env.num_envs, env.device
        self.initial_object_pos = env.scene["object"].data.root_pos_w.clone()
        self.last_frame = torch.full((n,), env._sim_step_counter, dtype=torch.long, device=device)
        self.affordance_impulses = torch.zeros(n, 13, 3, device=device)
        self.table_impulses = torch.zeros_like(self.affordance_impulses)
        self.arm_impulses = torch.zeros(n, self.arm_count, 3, device=device)
        self.arm_contacts_all = torch.zeros(n, self.inner_arm_count, dtype=torch.bool, device=device)

        # 源RaiSim将固定tip碰撞体并入distal刚体；USD中的独立tip回归相同接触槽。
        hand_slots = dict(zip(hand_names, range(13)))
        if not fr3:
            hand_slots.update({f"link_{joint}_0_tip": slot for joint, slot in zip((3, 7, 11, 15), (3, 6, 9, 12))})
        self.hand_slots = torch.tensor([hand_slots.get(name, 0) for name in self.body_names], device=device)
        arm_slots = dict(zip(arm_names, range(self.arm_count)))
        # 源std::map::operator[]把未映射的机器人接触刚体归到槽0，不能擅自丢弃。
        self.arm_slots = torch.tensor([arm_slots.get(name, 0) for name in self.body_names], device=device)

        patterns, filters, all_patterns, all_filters = [], [], [], []
        stage = omni.usd.get_context().get_stage()
        ground = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(env.cfg.scene.ground.prim_path))
                  if prim.HasAPI(UsdPhysics.CollisionAPI)]
        for index in range(n):
            root = f"/World/envs/env_{index}"
            bodies = [f"{root}/Robot/{name}" for name in self.body_names]
            pairs = [f"{root}/Object/top", f"{root}/Object/bottom", f"{root}/Table"]
            patterns.extend(bodies)
            filters.extend([pairs]*len(bodies))
            all_patterns.extend(f"{root}/Robot/{name}" for name in arm_names[1:-1])
            all_filters.extend([bodies+pairs+ground]*self.inner_arm_count)
        physics = SimulationManager.get_physics_sim_view()
        # 显式逐刚体/逐环境过滤，避免ContactSensor多刚体many-to-many的限制。
        self.contact_view = physics.create_rigid_contact_view(
            patterns, filters, max_contact_data_count=32*len(patterns))
        self.arm_all_view = physics.create_rigid_contact_view(
            all_patterns, all_filters, max_contact_data_count=32*len(all_patterns))
        print("ADAPTED: RaiSim per-contact impulse < 0.001 filtering has no exact PhysX friction-patch equivalent; real body resultant impulses retained without a substitute threshold.", flush=True)

    def reset(self, ids):
        self.initial_object_pos[ids] = self.env.scene["object"].data.root_pos_w[ids]
        for value in (self.affordance_impulses, self.table_impulses, self.arm_impulses, self.arm_contacts_all):
            value[ids] = 0
        # Reset未推进物理时间，不能读取前一候选姿态遗留的接触。
        self.last_frame[ids] = self.env._sim_step_counter

    def update(self):
        env = self.env
        fresh = self.last_frame < env._sim_step_counter
        if not bool(fresh.any()):
            return
        normal = self.contact_view.get_contact_force_matrix(dt=env.physics_dt)
        friction, _, counts, starts = self.contact_view.get_friction_data(dt=env.physics_dt)
        counts, starts = counts.flatten().long(), starts.flatten().long()
        rows = torch.repeat_interleave(torch.arange(counts.numel(), device=env.device), counts)
        offsets = torch.arange(rows.numel(), device=env.device) - (counts.cumsum(0)-counts)[rows]
        summed = torch.zeros(counts.numel(), 3, device=env.device)
        summed.index_add_(0, rows, friction[starts[rows]+offsets])
        # RaiSim只读取最后一个simulation step的接触；不乘control_dt或累积decimation。
        impulses = (normal + summed.reshape_as(normal))*env.physics_dt
        impulses = impulses.view(env.num_envs, len(self.body_names), 3, 3)
        affordance = torch.zeros_like(self.affordance_impulses)
        table = torch.zeros_like(self.table_impulses)
        arm = torch.zeros_like(self.arm_impulses)
        affordance.index_add_(1, self.hand_slots, impulses[:, :, 0])
        table.index_add_(1, self.hand_slots, impulses[:, :, 2])
        # 先对物体top/bottom/桌面冲量求矢量和，再计算每个源arm槽的模。
        arm.index_add_(1, self.arm_slots, impulses.sum(dim=2))
        self.affordance_impulses[fresh] = affordance[fresh]
        self.table_impulses[fresh] = table[fresh]
        self.arm_impulses[fresh] = arm[fresh]
        # 接触存在与否读取真实contact count，不能以合力非零代替（合力可能抵消）。
        count = self.arm_all_view.get_contact_data(dt=env.physics_dt)[4]
        contact = count.view(env.num_envs, self.inner_arm_count, -1).sum(dim=2) > 0
        self.arm_contacts_all[fresh] = contact[fresh]
        self.last_frame[fresh] = env._sim_step_counter


class TeacherGraspState:
    """逐物理步采样非有限状态和接触；接触、速度和抬升均不触发终止。"""

    def __init__(self, env):
        self.env = env
        self.robot, self.obj = env.scene["robot"], env.scene["object"]
        self.joints = env.action_manager.get_term("teacher")._joint_ids
        self.top = self.obj.find_bodies("top")[0][0]
        fr3 = getattr(env.cfg, "robot_type", "ur5_allegro") == "fr3_inspire"
        tip_names = list(INSPIRE_DISTAL_LINK_NAMES) if fr3 else [f"link_{i}_0_tip" for i in (3, 7, 11, 15)]
        self.tips = self.robot.find_bodies(tip_names, preserve_order=True)[0]
        self.arm_count = 7 if fr3 else 6
        self.tip_count = len(tip_names)
        self.table_names = ([f"teacher_table_contact_{i}" for i in range(13)]
                            + [f"teacher_tip_table_{i}" for i in range(self.tip_count)])
        if not fr3:
            self.table_names += [f"teacher_hand_table_{i}" for i in range(5)]
        n, device = env.num_envs, env.device
        self.lift = torch.zeros(n, device=device)
        self.max_lift = torch.zeros(n, device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.invalid = self.active.clone()
        self.grip = self.active.clone()
        self.good_contact = self.active.clone()
        self.last_frame = torch.full((n,), -1, dtype=torch.long, device=device)
        self.previous_q = self.robot.data.joint_pos.clone()
        self.tip_forces = torch.zeros(n, self.tip_count, 3, device=device)
        self.hand_normal = torch.zeros(n, 13, 3, device=device)
        self.hand_friction = torch.zeros_like(self.hand_normal)
        self.safety_failure = None

    def reset(self, ids):
        self.env._teacher_reward_state.reset(ids)
        for value in (self.lift, self.max_lift, self.invalid, self.grip, self.good_contact):
            value[ids] = 0
        self.tip_forces[ids] = 0
        self.hand_normal[ids] = 0
        self.hand_friction[ids] = 0
        self.previous_q[ids] = self.robot.data.joint_pos[ids]
        self.last_frame[ids] = self.env._sim_step_counter
        self.active[ids] = True

    def sample(self, frame):
        env, robot, obj = self.env, self.robot, self.obj
        fresh = self.active & (self.last_frame < frame)
        if not bool(fresh.any()):
            return
        q, qd = robot.data.joint_pos, robot.data.joint_vel
        limits = robot.data.joint_pos_limits
        violation = torch.maximum(limits[..., 0]-q, q-limits[..., 1]).clamp_min(0)
        diff_speed = (q-self.previous_q).abs()/env.physics_dt
        finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for value in (robot.data.body_state_w, q, qd, obj.data.body_state_w,
                      obj.data.joint_pos, obj.data.joint_vel):
            finite &= torch.isfinite(value).flatten(1).all(1)
        arm_vectors = torch.stack([env.scene[f"teacher_arm_contact_{i}"].data.net_forces_w[:, 0]
                                   for i in range(self.arm_count)], dim=1)
        arm_pairs = torch.stack([env.scene[f"teacher_arm_contact_{i}"].data.force_matrix_w[:, 0]
                                 for i in range(self.arm_count)], dim=1)
        arm_friction = torch.stack([env.scene[f"teacher_arm_contact_{i}"].data.friction_forces_w[:, 0]
                                    for i in range(self.arm_count)], dim=1)
        table_names = self.table_names
        table_normal = torch.stack([contact_force(env, name) for name in table_names], dim=1)
        table_friction = torch.stack([contact_friction_force(env, name) for name in table_names], dim=1)
        table_vectors = table_normal + table_friction
        self.hand_normal[fresh] = torch.stack([contact_force(env, name)
                                              for name in AFFORDANCE_CONTACT_SENSOR_NAMES], dim=1)[fresh]
        self.hand_friction[fresh] = torch.stack([contact_friction_force(env, name)
                                                for name in AFFORDANCE_CONTACT_SENSOR_NAMES], dim=1)[fresh]
        self.tip_forces[:] = torch.stack([contact_force(env, f"teacher_tip_object_{i}") for i in range(self.tip_count)], dim=1)
        finite &= torch.isfinite(arm_vectors).flatten(1).all(1)
        finite &= torch.isfinite(arm_pairs).flatten(1).all(1)
        finite &= torch.isfinite(arm_friction).flatten(1).all(1)
        finite &= torch.isfinite(table_vectors).flatten(1).all(1)
        finite &= torch.isfinite(self.tip_forces).flatten(1).all(1)
        finite &= torch.isfinite(self.hand_normal + self.hand_friction).flatten(1).all(1)
        # Teacher物体参考为真实top碰撞体的link原点。某些bottom仅为远置的
        # 无碰撞占位体，不能把其COM绕转速度误报成抓取物体的平移速度。
        obj_lin = obj.data.body_link_lin_vel_w[:, self.top].norm(dim=-1)
        obj_ang = obj.data.body_ang_vel_w[:, self.top].norm(dim=-1)
        obj_limits = obj.data.joint_pos_limits
        obj_violation = torch.maximum(obj_limits[..., 0]-obj.data.joint_pos,
                                      obj.data.joint_pos-obj_limits[..., 1]).clamp_min(0).amax(1)
        # 仅非有限数值可触发异常终止；不再使用DexSuite的两倍速度门槛。
        bad = ~finite & fresh
        self.invalid |= ~finite & fresh
        # 首次异常现场保存在内存并输出终端；无日志训练不写 checkpoint。
        if self.safety_failure is None and bool(bad.any()):
            ids = bad.nonzero().flatten()
            self.safety_failure = dict(
                physics_frame=int(frame), env_ids=ids.cpu().tolist(),
                object_names=[env.cfg.object_names[i] for i in ids.cpu().tolist()],
                robot_joint_names=list(robot.joint_names), target_joint_names=list(env.action_manager.get_term("teacher")._joint_names),
                robot_bodies=list(robot.body_names), arm_bodies=[env.scene[f"teacher_arm_contact_{i}"].body_names[0] for i in range(self.arm_count)],
                pair_order=["Object/top", "Object/bottom", "Table"],
                q=q[ids].cpu().clone(), qd=qd[ids].cpu().clone(), difference_velocity=diff_speed[ids].cpu().clone(),
                target=env.action_manager.get_term("teacher").target[ids].cpu().clone(),
                arm_vectors=arm_vectors[ids].cpu().clone(), arm_pairs=arm_pairs[ids].cpu().clone(),
                arm_friction=arm_friction[ids].cpu().clone(),
                hand_object_forces=self.hand_normal[ids].cpu().clone(),
                hand_object_friction_forces=self.hand_friction[ids].cpu().clone(),
                tip_object_forces=self.tip_forces[ids].cpu().clone(),
                table_forces=table_vectors[ids].cpu().clone(),
                table_normal=table_normal[ids].cpu().clone(), table_friction=table_friction[ids].cpu().clone(),
                joint_violation=violation[ids].cpu().clone(), object_joint_violation=obj_violation[ids].cpu().clone(),
                object_state=obj.data.body_state_w[ids].cpu().clone(), finite=finite[ids].cpu().clone(),
                object_linear_speed=obj_lin[ids].cpu().clone(), object_angular_speed=obj_ang[ids].cpu().clone(),
                table_bodies=[env.scene[name].body_names[0] for name in self.table_names],
            )
            print("[UNSAFE]", self.safety_failure, flush=True)
        top = obj.data.body_link_pose_w[:, self.top]
        force = self.tip_forces.norm(dim=-1)
        contact = force > env.cfg.tip_contact_threshold_N
        self.good_contact[fresh] = dexsuite_good_finger_contact(env, 1.0)[fresh]
        tip_pos = robot.data.body_link_pos_w[:, self.tips]-top[:, None, :3]
        radial = torch.nn.functional.normalize(tip_pos, dim=-1)
        normals = torch.nn.functional.normalize(self.tip_forces, dim=-1)
        opposite = ((radial[:, :-1]*radial[:, -1:]).sum(-1) < -0.2)
        opposite &= ((normals[:, :-1]*normals[:, -1:]).sum(-1) < -0.2)
        grip = contact[:, -1] & (contact[:, :-1] & opposite).any(1)
        # 对向接触仅供诊断使用，不参与成功判定。
        self.grip[fresh] = grip[fresh]
        # 源global_state[107]是top link世界Z；初始值来自Reset写入的root Z。
        self.lift[fresh] = (top[:, 2]-env._teacher_reward_state.initial_object_pos[:, 2])[fresh]
        self.max_lift[fresh] = torch.maximum(self.max_lift, self.lift)[fresh]
        self.previous_q[fresh] = q[fresh]
        self.last_frame[fresh] = frame

# Keep the entry-point name already referenced by teacher_env_cfg.py.
teacher_observation = TeacherObservation


__all__ = [
    "TEACHER_BASE_OBSERVATION_DIM",
    "TEACHER_AFFORDANCE_OBSERVATION_DIM",
    "TEACHER_OBSERVATION_DIM",
    "AFFORDANCE_POINT_COUNT",
    "CONTACT_IMPULSE_THRESHOLD",
    "AFFORDANCE_CONTACT_SENSOR_NAMES",
    "dexsuite_good_finger_contact",
    "TeacherObservation",
    "teacher_observation",
]
