"""管理 UR5 + Allegro Teacher 使用的物体列表与物体几何元数据。

本模块从物体数据集目录读取网格和最低点信息，并为 affordance 区域及
非 affordance 区域各采样固定数量的表面点和法线，供环境初始化使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from Grasp1.utils.paths import object_asset_dir, object_dataset_dir


TRAINING_DATASET = "new_training_set"  # Teacher 训练时使用的物体数据集目录名。
EVALUATION_DATASET = "shapenet-30obj"  # 评估时使用的 ShapeNet 物体数据集目录名。
DUMMY_DATASET = "dummy"  # 用于占位或单物体场景的测试数据集目录名。
TRAINING_REPEAT_PER_OBJECT = 2  # 将完整训练物体列表重复两次，以匹配原始训练采样方式。
POINTS_PER_MESH = 200  # 每个网格表面采样的点数。

# 对这些较难物体额外增加训练样本；之后再整体重复训练列表。
# 字典值表示在基础列表之外追加的次数。
DIFFICULT_OBJECT_EXTRA_REPEATS = {
    "037_scissors": 2,
    "off_water_body": 2,
    "019_pitcher_base": 1,
    "011_banana": 1,
    "mouse": 1,
    "hammer": 1,
    "small_block": 1,
}


@dataclass(frozen=True)
class ObjectMetadata:
    """单个物体的文件路径、网格和表面采样数据。

    网格和 NumPy 数组在 CPU 上加载；环境可在初始化时将采样数据搬到仿真设备。
    """

    dataset_name: str  # 该物体所属的数据集目录名。
    name: str  # 物体目录名，也是其资产文件的名称前缀。
    directory: Path  # 物体资产目录。
    urdf_path: Path  # 物体的 URDF 文件路径。
    usd_path: Path  # 物体的 USD 文件路径。
    lowest_point: float  # lowest_point_new.txt 中记录的物体最低点高度。
    affordance_mesh: trimesh.Trimesh  # 可抓取 affordance 区域的网格。
    affordance_points: np.ndarray  # affordance 网格表面采样点，形状为 (N, 3)。
    affordance_normals: np.ndarray  # 与 affordance_points 对应的表面法线，形状为 (N, 3)。
    affordance_center: np.ndarray  # affordance 网格质心，形状为 (3,)。
    non_affordance_mesh: trimesh.Trimesh  # 非 affordance 区域的网格。
    non_affordance_points: np.ndarray  # 非 affordance 网格表面采样点，形状为 (N, 3)。
    non_affordance_normals: np.ndarray  # 与 non_affordance_points 对应的表面法线，形状为 (N, 3)。
    non_affordance_center: np.ndarray  # 非 affordance 网格质心；不存在独立部分时使用占位值。
    has_two_parts: bool  # 非 affordance 网格是否被判断为独立的第二部分。


def object_names(dataset_name: str) -> list[str]:
    """按目录遍历顺序列出物体，保持原 Teacher 的列表构造方式。"""
    return [
        entry.name
        for entry in object_dataset_dir(dataset_name).iterdir()
        if entry.is_dir()
    ]


def training_object_names() -> list[str]:
    """生成 Teacher 训练物体列表。

    先读取训练集中的物体，再按困难物体配置追加重复项，最后将完整列表重复
    ``TRAINING_REPEAT_PER_OBJECT`` 次；列表中的重复项用于提高相应物体的采样权重。
    """
    names = object_names(TRAINING_DATASET)
    for name, extra_repeats in DIFFICULT_OBJECT_EXTRA_REPEATS.items():
        names.extend([name] * extra_repeats)
    return names * TRAINING_REPEAT_PER_OBJECT


def evaluation_object_names(repeat_per_object: int = 1) -> list[str]:
    """生成评估物体列表，并按指定次数重复每个 ShapeNet 评估物体。"""
    return object_names(EVALUATION_DATASET) * repeat_per_object


def _sample_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """从网格表面均匀采样固定数量的点，并取出对应面的法线。

    返回的点和法线均转换为 ``float32`` NumPy 数组，形状为 ``(N, 3)``。
    """
    points, face_ids = trimesh.sample.sample_surface(mesh, POINTS_PER_MESH)
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(mesh.face_normals[face_ids], dtype=np.float32),
    )


def load_object_metadata(
    dataset_name: str,
    name: str,
    *,
    two_hand: bool = False,
) -> ObjectMetadata:
    """读取单个物体的最低点、网格及两类网格的表面采样数据。

    默认将 ``top`` 网格作为 affordance 区域、``bottom`` 网格作为非 affordance
    区域；``two_hand=True`` 时交换两者。网格读取、表面采样和质心计算均在 CPU
    上完成，返回的数据可由环境在初始化期间传到仿真设备。
    """
    directory = object_asset_dir(dataset_name, name)
    top_path = directory / "top_watertight_tiny.obj"
    bottom_path = directory / "bottom_watertight_tiny.obj"
    affordance_path, non_affordance_path = (
        (bottom_path, top_path) if two_hand else (top_path, bottom_path)
    )

    # 保持 RobustDexGrasp 的 trimesh 默认处理方式。加载后的非 affordance 网格
    # 顶点数用于复现原始的“两部分物体至少有 25 个顶点”判定规则。
    affordance_mesh = trimesh.load_mesh(affordance_path)
    non_affordance_mesh = trimesh.load_mesh(non_affordance_path)
    affordance_points, affordance_normals = _sample_mesh(affordance_mesh)
    non_affordance_points, non_affordance_normals = _sample_mesh(non_affordance_mesh)

    has_two_parts = non_affordance_mesh.vertices.shape[0] >= 25
    # 只有确认为独立第二部分时才使用其网格质心；否则沿用原实现的远距离占位中心。
    non_affordance_center = (
        np.asarray(non_affordance_mesh.centroid, dtype=np.float32)
        if has_two_parts
        else np.full(3, 100.0, dtype=np.float32)
    )

    # 汇总资产路径、最低点高度，以及两类网格的几何数据和采样结果。
    return ObjectMetadata(
        dataset_name=dataset_name,
        name=name,
        directory=directory,
        urdf_path=directory / f"{name}.urdf",
        usd_path=directory / f"{name}.usd",
        # 场景放置物体时使用的最低点高度。
        lowest_point=float((directory / "lowest_point_new.txt").read_text()),
        affordance_mesh=affordance_mesh,
        affordance_points=affordance_points,
        affordance_normals=affordance_normals,
        # affordance 网格的质心。
        affordance_center=np.asarray(affordance_mesh.centroid, dtype=np.float32),
        non_affordance_mesh=non_affordance_mesh,
        non_affordance_points=non_affordance_points,
        non_affordance_normals=non_affordance_normals,
        non_affordance_center=non_affordance_center,
        has_two_parts=has_two_parts,
    )


def load_unique_object_metadata(
    dataset_name: str,
    names: Sequence[str],
    *,
    two_hand: bool = False,
) -> dict[str, ObjectMetadata]:
    """为名称列表中的每个不同物体加载一次元数据。

    即使 ``names`` 因训练权重而含有重复名称，返回字典仍只为每个物体加载并采样
    一次。按名称排序后加载，匹配原 VecEnv 的 ``np.unique(obj_list)`` 顺序。
    """
    return {
        name: load_object_metadata(dataset_name, name, two_hand=two_hand)
        for name in sorted(set(names))
    }


# 对外公开的数据集配置、元数据类型及列表/加载函数。
__all__ = [
    "TRAINING_DATASET",
    "EVALUATION_DATASET",
    "DUMMY_DATASET",
    "TRAINING_REPEAT_PER_OBJECT",
    "POINTS_PER_MESH",
    "DIFFICULT_OBJECT_EXTRA_REPEATS",
    "ObjectMetadata",
    "object_names",
    "training_object_names",
    "evaluation_object_names",
    "load_object_metadata",
    "load_unique_object_metadata",
]
