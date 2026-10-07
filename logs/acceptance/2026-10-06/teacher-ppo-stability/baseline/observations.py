"""构造 RobustDexGrasp Teacher 的 153 维策略观测。

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
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from Grasp1.data.object_set import POINTS_PER_MESH, load_unique_object_metadata
from Grasp1.robots.robot_profile import (
    CONTROLLED_JOINT_NAMES,
    HAND_CENTER,
    HAND_CONTACT_LINK_NAMES,
    HAND_KEYPOINT_COUNT,
    NUM_CONTROLLED_JOINTS,
    UR5_LINK_NAMES,
)

from .keypoints import (
    arm_keypoints_w,
    hand_keypoints_w,
    hand_keypoints_o,
    initialize_teacher_body_indices,
    _object_top_pose_w,
)

from .runtime import TeacherRuntimeFeatures, filtered_contact_impulses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


# -----------------------------------------------------------------------------
# Teacher observation constants
# -----------------------------------------------------------------------------

TEACHER_BASE_OBSERVATION_DIM: Final[int] = 102
TEACHER_AFFORDANCE_OBSERVATION_DIM: Final[int] = HAND_KEYPOINT_COUNT * 3
TEACHER_OBSERVATION_DIM: Final[int] = (
    TEACHER_BASE_OBSERVATION_DIM
    + TEACHER_AFFORDANCE_OBSERVATION_DIM
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
    """共享当前步世界系最近 affordance 向量 [N,17,3]，不计算完整观测。"""
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
        quaternions = object_quat_w[ids, None, :].expand(-1, HAND_KEYPOINT_COUNT, -1)
        return quat_apply(quaternions.reshape(-1,4), vectors_o.reshape(-1,3)).reshape_as(vectors_o)

    return env._teacher_runtime_features.get("affordance_vectors_w", compute)


# -----------------------------------------------------------------------------
# Stateful Manager observation
# -----------------------------------------------------------------------------

class TeacherObservation(ManagerTermBase):
    """按旧 Teacher 字段顺序生成 ``[N,153]`` policy observation。"""

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
    ) -> None:
        super().__init__(cfg, env)

        self._robot = env.scene["robot"]
        self._object = env.scene["object"]
        initialize_teacher_body_indices(env)
        env._teacher_runtime_features = TeacherRuntimeFeatures(env)

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

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Reset the wrist-reference state for selected environments."""
        if env_ids is None:
            env_ids = slice(None)

        self._env._teacher_runtime_features.invalidate(env_ids)
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
        """Return 13 top-contact flags and approximate contact impulses.

        RaiSim exposed per-step contact impulses directly. IsaacLab's
        ContactSensor exposes normal/friction forces, so the migrated quantity
        is ``||F_normal + F_friction|| * physics_dt``.

        Filtered friction/contact buffers may contain non-finite values for
        inactive pairs depending on the PhysX contact backend.  Those represent
        "no contact", not an invalid robot state, so both force components are
        sanitized before they enter the policy observation.
        """
        impulses = filtered_contact_impulses(env, "teacher_af_contact_")[3]
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
        """Return flattened 17x3 nearest-affordance vectors in world frame."""
        return nearest_affordance_vectors_w(env).reshape(
            env.num_envs, TEACHER_AFFORDANCE_OBSERVATION_DIM
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        support_height: float,
    ) -> torch.Tensor:
        """Return the exact-layout ``[num_envs,153]`` Teacher observation."""
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
                - support_height,  # 70:87
                arm_positions_w[..., 2]
                - support_height,  # 87:93
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
            TEACHER_BASE_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                "Teacher base observation shape mismatch: "
                f"expected=({env.num_envs},"
                f"{TEACHER_BASE_OBSERVATION_DIM}), "
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
            TEACHER_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                "Teacher observation shape mismatch: "
                f"expected=({env.num_envs},"
                f"{TEACHER_OBSERVATION_DIM}), "
                f"got={tuple(observation.shape)}."
            )

        return observation


# Keep the entry-point name already referenced by teacher_env_cfg.py.
teacher_observation = TeacherObservation


__all__ = [
    "TEACHER_BASE_OBSERVATION_DIM",
    "TEACHER_AFFORDANCE_OBSERVATION_DIM",
    "TEACHER_OBSERVATION_DIM",
    "AFFORDANCE_POINT_COUNT",
    "CONTACT_IMPULSE_THRESHOLD",
    "AFFORDANCE_CONTACT_SENSOR_NAMES",
    "TeacherObservation",
    "teacher_observation",
]
