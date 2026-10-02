"""Teacher 控制步内的共享小张量；只供物理步结束后的 MDP 读取。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import ContactSensor
from Grasp1.robots.robot_profile import ARM_CONTACT_LINK_NAMES, HAND_CONTACT_LINK_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

EnvIds = slice | torch.Tensor
Feature = torch.Tensor | tuple[torch.Tensor, ...]


class TeacherRuntimeFeatures:
    """环境私有的按步缓存，不保留 cdist 的完整成对距离张量。

    Isaac Lab 在物理积分后、termination/reward 前递增 common_step_counter。
    新控制步首次读取时清空动态结果，因此 reward 不会读取上一控制步的数据。
    reset、碰撞回退及 bias 写状态时显式 invalidate；partial reset 的下一次
    读取只重算对应行，其它环境的缓存行保持原值。静态名称索引另行保存。
    """

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        """绑定环境与控制步计数器，建立空的动态结果和待更新行集合。"""
        self._env = env
        self._step = env.common_step_counter
        self._values: dict[str, Feature] = {}
        self._pending: dict[str, list[torch.Tensor]] = {}
        # 只解析/校验一次 sensor 对象；每步读取其最新的 GPU data。
        self.contact_sensors: dict[str, tuple[ContactSensor, ...]] = {}
        for prefix, count in (
            ("teacher_af_contact_", len(HAND_CONTACT_LINK_NAMES)),
            ("teacher_table_contact_", len(HAND_CONTACT_LINK_NAMES)),
            ("teacher_arm_contact_", len(ARM_CONTACT_LINK_NAMES)),
        ):
            sensors = tuple(env.scene[f"{prefix}{index}"] for index in range(count))
            for sensor in sensors:
                if not isinstance(sensor, ContactSensor) or sensor.num_bodies != 1:
                    raise RuntimeError("Teacher contact sensors must monitor exactly one body.")
                normal, friction = sensor.data.force_matrix_w, sensor.data.friction_forces_w
                if normal is None or friction is None:
                    raise RuntimeError("Teacher contact sensors require filtered normal/friction forces.")
                if normal.ndim != 4 or friction.ndim != 4 or normal.shape[1] != 1 or friction.shape[1] != 1:
                    raise RuntimeError("Teacher filtered contact tensors must have shape [N,1,M,3].")
                if prefix == "teacher_af_contact_" and (normal.shape[1:3] != (1, 1) or friction.shape[1:3] != (1, 1)):
                    raise RuntimeError("Teacher affordance sensors must be one-body-to-one-filter sensors.")
            self.contact_sensors[prefix] = sensors

    def _update_step(self) -> None:
        """每个新控制步丢弃上一控制步结果；比较 Python 计数器，不同步 GPU。"""
        step = self._env.common_step_counter
        if step != self._step:
            self._values.clear()
            self._pending.clear()
            self._step = step

    def invalidate(self, env_ids: EnvIds | Sequence[int] | None = None) -> None:
        """状态写入后标记失效；None/slice 为全量 reset，Tensor 为指定环境。"""
        self._update_step()
        if env_ids is None or isinstance(env_ids, slice):
            self._values.clear()
            self._pending.clear()
        else:
            env_ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
            for name in self._values:
                self._pending.setdefault(name, []).append(env_ids)

    def get(self, name: str, compute: Callable[[EnvIds], Feature]) -> Feature:
        """首次读取计算结果；同一步复用，失效的 partial reset 行按需更新。"""
        self._update_step()
        if name not in self._values:
            self._values[name] = compute(slice(None))
        elif name in self._pending:
            pending = self._pending.pop(name)
            ids = pending[0] if len(pending) == 1 else torch.cat(pending).unique()
            updated = compute(ids)
            cached = self._values[name]
            if isinstance(cached, tuple):
                for target, source in zip(cached, updated, strict=True):
                    target[ids] = source
            else:
                cached[ids] = updated
        return self._values[name]


def filtered_contact_impulses(env: ManagerBasedRLEnv, prefix: str) -> tuple[torch.Tensor, ...]:
    """各接触组每步只整理一次，返回 total/tangential/normal [N,K]。

    affordance 组另返回原 observation 所需的 sanitized total；奖励仍使用
    未清理的力，保持两条原有路径对非有限值的不同处理。
    """
    cache = env._teacher_runtime_features

    def compute(ids: EnvIds) -> tuple[torch.Tensor, ...]:
        """按传感器原顺序 stack，在 filter body 维求和后取模并乘 physics_dt。"""
        sensors = cache.contact_sensors[prefix]
        normal = torch.stack([sensor.data.force_matrix_w[ids, 0] for sensor in sensors], dim=1)
        friction = torch.stack([sensor.data.friction_forces_w[ids, 0] for sensor in sensors], dim=1)
        normal_vector = normal.sum(dim=2)
        tangential_vector = friction.sum(dim=2)
        result = (
            torch.linalg.vector_norm(normal_vector + tangential_vector, dim=-1) * env.physics_dt,
            torch.linalg.vector_norm(tangential_vector, dim=-1) * env.physics_dt,
            torch.linalg.vector_norm(normal_vector, dim=-1) * env.physics_dt,
        )
        if prefix == "teacher_af_contact_":
            safe_total = (
                torch.nan_to_num(normal[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
                + torch.nan_to_num(friction[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
            )
            result += (torch.linalg.vector_norm(safe_total, dim=-1) * env.physics_dt,)
        return result

    return cache.get(f"contact/{prefix}", compute)
