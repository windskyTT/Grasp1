"""UR5 + Allegro Teacher reward terms.

The original RobustDexGrasp Teacher reward was split between
``Environment.hpp`` and ``allegro_teacher/train.py``.  This module merges those
raw reward metrics into IsaacLab Manager-Based terms.

Only reward formulas live here.  Source YAML coefficients remain in
``teacher_env_cfg.py``.  Because IsaacLab's ``RewardManager`` multiplies every
term by ``env.step_dt``, the environment config converts each old per-control-
step coefficient to ``source_coeff / SOURCE_CONTROL_DT`` to preserve reward
strength per simulated second.

Important migration notes
-------------------------
- RaiSim exposes contact impulses directly; IsaacLab ContactSensor exposes
  contact forces.  The migration approximates impulse as ``force * physics_dt``.
- ``affordance_impulse_reward`` uses tangential/friction impulse magnitude,
  corresponding to the old contact-frame XY impulse magnitude.
- The source line ``reward_r.clip(min=-2.0)`` does not mutate the NumPy array
  because its return value is ignored.  Therefore no final reward floor is
  applied here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from Grasp1.robots.robot_profile import (
    ARM_CONTACT_LINK_NAMES,
    HAND_CONTACT_LINK_NAMES,
)

from .keypoints import (
    arm_keypoints_w,
    hand_keypoints_w,
)

from .observations import nearest_affordance_vectors_w
from .runtime import filtered_contact_impulses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


# -----------------------------------------------------------------------------
# Source constants
# -----------------------------------------------------------------------------

NUM_HAND_CONTACTS: Final[int] = len(HAND_CONTACT_LINK_NAMES)
NUM_ARM_CONTACTS: Final[int] = len(ARM_CONTACT_LINK_NAMES)

CONTACT_THRESHOLD: Final[float] = 0.01

TABLE_HEIGHT_CLIP_MIN: Final[float] = 0.002
TABLE_HEIGHT_CLIP_MAX: Final[float] = 0.02
TABLE_HEIGHT_LOG_SCALE: Final[float] = 50.0

WRIST_SPEED_THRESHOLD: Final[float] = 0.25
WRIST_HIGH_SPEED_MULTIPLIER: Final[float] = 10.0

ARM_JOINT_SPEED_THRESHOLD: Final[float] = 0.5
ARM_JOINT_HIGH_SPEED_MULTIPLIER: Final[float] = 4.0

# Environment.hpp:
# first 10 contact slots -> max impulse 0.1
# last 3 contact slots  -> max impulse 0.2
_IMPULSE_HIGH: Final[tuple[float, ...]] = (
    *((0.1,) * 10),
    *((0.2,) * 3),
)

# Source frame queried by getFrameVelocity(body_parts_r_[0]) is reached from
# wrist_3_link by wrist_3_link -> tool0 translation.  The following fixed
# flange/hand-base joints have zero translation.
_WRIST_FRAME_OFFSET_LOCAL: Final[tuple[float, float, float]] = (
    0.0,
    0.0823,
    0.0,
)

AFFORDANCE_CONTACT_SENSOR_PREFIX: Final[str] = "teacher_af_contact_"
TABLE_CONTACT_SENSOR_PREFIX: Final[str] = "teacher_table_contact_"
ARM_CONTACT_SENSOR_PREFIX: Final[str] = "teacher_arm_contact_"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _require_articulation(
    env: ManagerBasedRLEnv,
    asset_name: str,
) -> Articulation:
    """Return a scene articulation with a useful error on mismatch."""
    asset = env.scene[asset_name]
    if not isinstance(asset, Articulation):
        raise TypeError(
            f"Scene asset {asset_name!r} must be Articulation, "
            f"got {type(asset)!r}."
        )
    return asset


def _resolve_body_id(
    asset: Articulation,
    body_name: str,
) -> int:
    """Resolve one literal body name from the loaded articulation."""
    try:
        body_ids, resolved_names = asset.find_bodies(
            re.escape(body_name),
            preserve_order=True,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to resolve body {body_name!r}; "
            f"available={tuple(asset.body_names)}."
        ) from exc

    if (
        len(body_ids) != 1
        or len(resolved_names) != 1
        or resolved_names[0] != body_name
    ):
        raise RuntimeError(
            f"Body {body_name!r} must resolve exactly once; "
            f"ids={body_ids}, names={resolved_names}."
        )

    return int(body_ids[0])


def _weighted(
    values: torch.Tensor,
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted sum over the last dimension."""
    weight_tensor = values.new_tensor(tuple(weights))
    if weight_tensor.shape != values.shape[-1:]:
        raise ValueError(
            f"Weight length {weight_tensor.numel()} does not match "
            f"value dimension {values.shape[-1]}."
        )
    return (values * weight_tensor).sum(dim=-1)


def _clip_hand_impulses(
    values: torch.Tensor,
) -> torch.Tensor:
    """Apply the source per-contact impulse upper bounds."""
    if values.shape[-1] != NUM_HAND_CONTACTS:
        raise ValueError(
            f"Expected {NUM_HAND_CONTACTS} hand-contact values, "
            f"got shape {tuple(values.shape)}."
        )
    high = values.new_tensor(_IMPULSE_HIGH)
    return torch.minimum(values.clamp_min(0.0), high)


def _filtered_impulses(
    env: ManagerBasedRLEnv,
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """读取当前控制步共享的 total/tangential/normal impulse [N,K]。"""
    return filtered_contact_impulses(env, prefix)[:3]


# -----------------------------------------------------------------------------
# Python-side rewards from allegro_teacher/train.py
# -----------------------------------------------------------------------------

def affordance_reward(
    env: ManagerBasedRLEnv,
    finger_weights: tuple[float, ...],
) -> torch.Tensor:
    """Negative weighted nearest distance from hand keypoints to affordance.

    Source ``train.py`` uses ``dis_info[:, 1:17]``.  Keypoint zero is therefore
    excluded; the configured weight for it is also zero.
    """
    # Reward 与 observation 共用当前控制步的最近点向量，不重算 policy 观测。
    vectors_w = nearest_affordance_vectors_w(env)
    nearest_distance = torch.linalg.vector_norm(vectors_w, dim=-1)
    return -_weighted(nearest_distance, finger_weights)


def table_reward(
    env: ManagerBasedRLEnv,
    finger_weights: tuple[float, ...],
    support_height: float,
) -> torch.Tensor:
    """Hand/table proximity metric from ``train.py``."""
    heights = hand_keypoints_w(env)[..., 2] - support_height
    heights = heights.clamp(
        TABLE_HEIGHT_CLIP_MIN,
        TABLE_HEIGHT_CLIP_MAX,
    )

    return -_weighted(
        torch.log(TABLE_HEIGHT_LOG_SCALE * heights),
        finger_weights,
    )


def arm_height_reward(
    env: ManagerBasedRLEnv,
    support_height: float,
) -> torch.Tensor:
    """Arm/table proximity metric using the source observation slice 89:93.

    The six arm heights occupy observation indices 87:93, so source
    ``obs_new_r[:, 89:93]`` intentionally selects only keypoints 2:6.
    """
    heights = arm_keypoints_w(env)[:, 2:6, 2] - support_height
    heights = heights.clamp(
        TABLE_HEIGHT_CLIP_MIN,
        TABLE_HEIGHT_CLIP_MAX,
    )
    return -torch.log(
        TABLE_HEIGHT_LOG_SCALE * heights
    ).sum(dim=-1)


# -----------------------------------------------------------------------------
# C++ hand contact / impulse rewards
# -----------------------------------------------------------------------------

def affordance_contact_reward(
    env: ManagerBasedRLEnv,
    contact_weights: tuple[float, ...],
) -> torch.Tensor:
    """Weighted top-body contact count divided by 13."""
    total_impulse, _, _ = _filtered_impulses(
        env,
        AFFORDANCE_CONTACT_SENSOR_PREFIX,
    )
    contacts = (total_impulse > CONTACT_THRESHOLD).to(total_impulse.dtype)
    return _weighted(contacts, contact_weights) / NUM_HAND_CONTACTS


def affordance_impulse_reward(
    env: ManagerBasedRLEnv,
    contact_weights: tuple[float, ...],
) -> torch.Tensor:
    """Weighted clipped tangential impulse on object ``top``."""
    _, tangential_impulse, _ = _filtered_impulses(
        env,
        AFFORDANCE_CONTACT_SENSOR_PREFIX,
    )
    return _weighted(
        _clip_hand_impulses(tangential_impulse),
        contact_weights,
    )


def push_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Normal-impulse push metric from ``Environment.hpp``.

    The current source coefficient is ``-0.0``; the formula is still retained
    so re-enabling the term does not require another code change.
    """
    _, _, normal_impulse = _filtered_impulses(
        env,
        AFFORDANCE_CONTACT_SENSOR_PREFIX,
    )

    palm_push = torch.clamp_min(
        normal_impulse[:, 0] - 1.0,
        0.0,
    )
    finger_push = torch.clamp_min(
        normal_impulse[:, 1:] - 2.0,
        0.0,
    ).sum(dim=-1)

    return torch.clamp(
        palm_push + finger_push,
        max=10.0,
    )


# -----------------------------------------------------------------------------
# C++ hand/table rewards
# -----------------------------------------------------------------------------

def table_contact_reward(
    env: ManagerBasedRLEnv,
    contact_weights: tuple[float, ...],
) -> torch.Tensor:
    """Weighted table-contact count divided by 13."""
    total_impulse, _, _ = _filtered_impulses(
        env,
        TABLE_CONTACT_SENSOR_PREFIX,
    )
    contacts = (total_impulse > CONTACT_THRESHOLD).to(total_impulse.dtype)
    return _weighted(contacts, contact_weights) / NUM_HAND_CONTACTS


def table_impulse_reward(
    env: ManagerBasedRLEnv,
    contact_weights: tuple[float, ...],
) -> torch.Tensor:
    """Weighted clipped total hand/table impulse."""
    total_impulse, _, _ = _filtered_impulses(
        env,
        TABLE_CONTACT_SENSOR_PREFIX,
    )
    return _weighted(
        _clip_hand_impulses(total_impulse),
        contact_weights,
    )


# -----------------------------------------------------------------------------
# C++ arm contact / collision rewards
# -----------------------------------------------------------------------------

def arm_contact_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """L2 norm of the six filtered object/table arm-contact flags."""
    total_impulse, _, _ = _filtered_impulses(
        env,
        ARM_CONTACT_SENSOR_PREFIX,
    )
    contacts = (total_impulse > CONTACT_THRESHOLD).to(total_impulse.dtype)
    return torch.linalg.vector_norm(contacts, dim=-1)


def arm_impulse_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """L2 norm of the six filtered object/table arm impulses."""
    total_impulse, _, _ = _filtered_impulses(
        env,
        ARM_CONTACT_SENSOR_PREFIX,
    )
    return torch.linalg.vector_norm(total_impulse, dim=-1)


def arm_collision_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Count source ``contacts_arm_all[1:5]``.

    ``contacts_arm_all`` is based on any contact involving each arm body, not
    only the object/table filtered pair. The arm sensors use unfiltered
    ``current_contact_time`` to reproduce this distinction.
    """
    sensors = env._teacher_runtime_features.contact_sensors[ARM_CONTACT_SENSOR_PREFIX]
    contacts_arm_all = torch.stack(
        [sensor.data.current_contact_time[:, 0] > 0.0 for sensor in sensors], dim=-1
    )

    # Source global_state[124:128] == contacts_arm_all[1:5].
    return contacts_arm_all[:, 1:5].to(
        dtype=_require_articulation(env, "robot").data.joint_pos.dtype
    ).sum(dim=-1)


# -----------------------------------------------------------------------------
# Velocity / displacement rewards
# -----------------------------------------------------------------------------

def _wrist_frame_velocity_w(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source hand-base-frame linear and angular velocity in world."""
    robot = _require_articulation(env, "robot")
    wrist_body_id = env._teacher_body_indices.wrist

    wrist_pose_w = robot.data.body_link_pose_w[:, wrist_body_id, :]
    linear_velocity_w = robot.data.body_link_lin_vel_w[:, wrist_body_id, :]
    angular_velocity_w = robot.data.body_link_ang_vel_w[:, wrist_body_id, :]

    offset_local = wrist_pose_w.new_tensor(
        _WRIST_FRAME_OFFSET_LOCAL
    ).unsqueeze(0).expand(
        env.num_envs,
        -1,
    )
    offset_w = quat_apply(
        wrist_pose_w[:, 3:7],
        offset_local,
    )

    # Velocity of a fixed point offset from the wrist_3_link origin.
    frame_linear_velocity_w = (
        linear_velocity_w
        + torch.linalg.cross(
            angular_velocity_w,
            offset_w,
            dim=-1,
        )
    )

    return frame_linear_velocity_w, angular_velocity_w


def wrist_vel_reward_(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Squared wrist-frame linear speed, times 10 above 0.25 m/s."""
    linear_velocity_w, _ = _wrist_frame_velocity_w(env)
    squared_speed = linear_velocity_w.square().sum(dim=-1)

    return torch.where(
        squared_speed > WRIST_SPEED_THRESHOLD**2,
        squared_speed * WRIST_HIGH_SPEED_MULTIPLIER,
        squared_speed,
    )


def wrist_qvel_reward_(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Squared wrist angular speed."""
    _, angular_velocity_w = _wrist_frame_velocity_w(env)
    return angular_velocity_w.square().sum(dim=-1)


def _object_top_velocity(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object ``top`` link linear/angular world velocity."""
    obj = _require_articulation(env, "object")
    top_body_id = env._teacher_body_indices.object_top

    return (
        obj.data.body_link_lin_vel_w[:, top_body_id, :],
        obj.data.body_link_ang_vel_w[:, top_body_id, :],
    )


def obj_vel_reward_(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Squared world linear speed of object ``top``."""
    linear_velocity_w, _ = _object_top_velocity(env)
    return linear_velocity_w.square().sum(dim=-1)


def obj_qvel_reward_(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Squared world angular speed of object ``top``."""
    _, angular_velocity_w = _object_top_velocity(env)
    return angular_velocity_w.square().sum(dim=-1)


class ObjectDisplacementReward(ManagerTermBase):
    """Distance from current object-top position to reset object-base position.

    Source ``Environment.hpp`` computes::

        || Obj_Position(top) - obj_pos_init_.head(3) ||

    where ``obj_pos_init_.head(3)`` is the reset base/root translation.
    The reset event has already written the root pose when RewardManager
    calls this term's ``reset`` method.
    """

    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRLEnv,
    ) -> None:
        super().__init__(cfg, env)

        self._object = _require_articulation(env, "object")
        self._top_body_id = _resolve_body_id(self._object, "top")
        self._initial_base_position_w = (
            self._object.data.root_pos_w.clone()
        )

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._initial_base_position_w[env_ids] = self._object.data.root_pos_w[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
        top_position_w = self._object.data.body_link_pose_w[
            :,
            self._top_body_id,
            :3,
        ]
        return torch.linalg.vector_norm(
            top_position_w - self._initial_base_position_w,
            dim=-1,
        )


# String entry point used by teacher_env_cfg.py.
obj_displacement_reward = ObjectDisplacementReward


def arm_joint_vel_reward_(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """UR5 joint-velocity metric from ``Environment.hpp``."""
    robot = _require_articulation(env, "robot")
    joint_ids = env._teacher_body_indices.arm_joints

    velocity = robot.data.joint_vel[
        :,
        list(joint_ids),
    ]
    scaled_velocity = torch.where(
        velocity.abs() > ARM_JOINT_SPEED_THRESHOLD,
        velocity * ARM_JOINT_HIGH_SPEED_MULTIPLIER,
        velocity,
    )
    return scaled_velocity.square().sum(dim=-1)


__all__ = [
    "NUM_HAND_CONTACTS",
    "NUM_ARM_CONTACTS",
    "CONTACT_THRESHOLD",
    "AFFORDANCE_CONTACT_SENSOR_PREFIX",
    "TABLE_CONTACT_SENSOR_PREFIX",
    "ARM_CONTACT_SENSOR_PREFIX",
    "affordance_reward",
    "affordance_contact_reward",
    "affordance_impulse_reward",
    "table_reward",
    "table_contact_reward",
    "table_impulse_reward",
    "arm_height_reward",
    "arm_contact_reward",
    "arm_impulse_reward",
    "arm_collision_reward",
    "push_reward",
    "wrist_vel_reward_",
    "wrist_qvel_reward_",
    "obj_vel_reward_",
    "obj_qvel_reward_",
    "ObjectDisplacementReward",
    "obj_displacement_reward",
    "arm_joint_vel_reward_",
]
