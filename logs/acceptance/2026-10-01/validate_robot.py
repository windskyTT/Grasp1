from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
app=launcher.app
from pxr import Usd,UsdPhysics,UsdGeom,Gf,Sdf
from pathlib import Path
import numpy as np
asset=Path('/home/windsky/project/Grasp1/assets/robots/ur5_allegro')
stage=Usd.Stage.Open(str(asset/'ur5_allegro.usd'))
prims=list(Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
assert not any(p.GetName() in ('world','platform_base_link','Link_1') for p in prims)
joints=[p for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
assert len(joints)==22
collisions=[p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
visuals=[p for p in prims if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath())]
print('ROBOT_ONLY_USD','revolute',len(joints),'collision',len(collisions),'visual_meshes',len(visuals),'bodies',sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),flush=True)
xf=UsdGeom.XformCache()
for p in prims:
    if p.IsA(UsdGeom.Mesh) and '/base_link/' in str(p.GetPath()):
        pts=np.asarray([xf.GetLocalToWorldTransform(p).Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()])
        print('BASE_BOUND',p.GetPath(),pts.min(0).tolist(),pts.max(0).tolist(),'radius',float(np.linalg.norm(pts[:,:2],axis=1).max()),flush=True)
# URDF importer 给空 visual 创建了指向不存在目标的引用，清除这些空 visual specs。
layer=Sdf.Layer.FindOrOpen(str(asset/'configuration/ur5_allegro_base.usd'))
removed=[]
for name in ('base','ee_link','tool0','Flange_base_link'):
    path=f'/ur5/{name}/visuals'
    spec=layer.GetPrimAtPath(path)
    refs=spec.referenceList.GetAppliedItems()
    for ref in refs:
        target_layer=Sdf.Layer.FindOrOpen(str(asset/'configuration'/'ur5_allegro_physics.usd'))
        assert not target_layer.GetPrimAtPath(ref.primPath),(path,ref)
    spec.referenceList.ClearEdits()
    removed.append(path)
layer.Save()
print('EMPTY_VISUAL_REFERENCES_REMOVED',removed,flush=True)
app.close()
