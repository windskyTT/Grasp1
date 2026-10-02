from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, Gf
import numpy as np
from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import Grasp1SceneCfg
cfg = Grasp1SceneCfg()
for label, path, primpath in [('BASE_VISUAL',cfg.robot.spawn.usd_path,'/ur5/base_link/visuals/base/Scene/mesh'),('BASE_COLLISION',cfg.robot.spawn.usd_path,'/ur5/base_link/collisions/base_stl/World/mesh'),('TABLE_COLLISION',cfg.table.spawn.usd_path,'/Table/Collisions/Cube')]:
    stage=Usd.Stage.Open(path)
    xf=UsdGeom.XformCache()
    prim=stage.GetPrimAtPath(primpath)
    matrix=xf.GetLocalToWorldTransform(prim)
    if label=='TABLE_COLLISION':
        matrix=matrix * xf.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
        matrix=matrix * Gf.Matrix4d().SetScale(Gf.Vec3d(*cfg.table.spawn.scale)) * Gf.Matrix4d().SetRotate(Gf.Quatd(cfg.table.init_state.rot[0],Gf.Vec3d(*cfg.table.init_state.rot[1:]))) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*cfg.table.init_state.pos))
    pts=np.asarray([matrix.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    print(label, 'min', pts.min(0).tolist(), 'max',pts.max(0).tolist(),'radius',np.linalg.norm(pts[:,:2],axis=1).max(),flush=True)
app.close()
