"""只合并拇指末端固定刚体，保留碰撞几何、总质量、COM及惯量。诊断资产另存。"""
import json
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


def merge_thumb_tip(source, output):
    stage = Usd.Stage.Open(source)
    destination = Path(output).resolve()
    stage.Flatten().Export(str(destination))
    stage = Usd.Stage.Open(str(destination))
    root = stage.GetDefaultPrim().GetPath()
    parent_path = root.AppendChild('link_15_0')
    tip_path = root.AppendChild('link_15_0_tip')
    parent, tip = stage.GetPrimAtPath(parent_path), stage.GetPrimAtPath(tip_path)
    transforms = UsdGeom.XformCache()
    relative = transforms.GetLocalToWorldTransform(tip) * transforms.GetLocalToWorldTransform(parent).GetInverse()
    rotation = np.asarray(relative.ExtractRotationMatrix()).T
    translation = np.asarray(relative.ExtractTranslation())

    def mass_properties(prim):
        api = UsdPhysics.MassAPI(prim)
        mass = float(api.GetMassAttr().Get())
        com = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=float)
        axes = Gf.Matrix3d(api.GetPrincipalAxesAttr().Get())
        axes = np.asarray(axes).T
        inertia = axes @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ axes.T
        return mass, com, inertia

    pm, pc, pi = mass_properties(parent)
    tm, tc, ti = mass_properties(tip)
    tc = rotation @ tc + translation
    ti = rotation @ ti @ rotation.T
    total = pm + tm
    center = (pm * pc + tm * tc) / total

    def parallel_axis(mass, offset):
        return mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))

    inertia = pi + ti + parallel_axis(pm, pc - center) + parallel_axis(tm, tc - center)
    diagonal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    principal = Gf.Matrix4d(Gf.Matrix3d(*axes.T.flatten().tolist()), Gf.Vec3d(0)).ExtractRotationQuat()
    api = UsdPhysics.MassAPI(parent)
    api.CreateMassAttr(total)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
    api.CreatePrincipalAxesAttr(Gf.Quatf(principal.GetReal(), Gf.Vec3f(*principal.GetImaginary())))

    # 将整个tip子树移动至parent；原网格/碰撞API不变，仅移除其独立刚体和固定关节。
    tip.RemoveAPI(UsdPhysics.RigidBodyAPI)
    tip.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    tip.RemoveAPI(UsdPhysics.MassAPI)
    tip.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
    UsdGeom.Xformable(tip).ClearXformOpOrder()
    UsdGeom.Xformable(tip).AddTransformOp().Set(relative)
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(tip_path, parent_path.AppendChild('merged_tip'))
    assert stage.GetRootLayer().Apply(edit)
    stage.RemovePrim(root.AppendPath('joints/joint_15_0_tip'))
    stage.GetRootLayer().Save()
    evidence = dict(source=source, diagnostic_asset=str(destination), merged_fixed_joint='joint_15_0_tip',
                    original_masses=[pm, tm], combined_mass=total, combined_com=center.tolist(),
                    combined_inertia=inertia.tolist(), eigenvalues=diagonal.tolist(),
                    tip_transform_in_parent=np.asarray(relative).tolist(), collision_geometry='unchanged')
    destination.with_suffix('.json').write_text(json.dumps(evidence, indent=2))
    return str(destination)
