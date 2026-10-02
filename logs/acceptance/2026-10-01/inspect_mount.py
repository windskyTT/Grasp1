from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, UsdPhysics
from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import Grasp1SceneCfg
cfg = Grasp1SceneCfg()
for label, path in [('ROBOT', cfg.robot.spawn.usd_path), ('TABLE', cfg.table.spawn.usd_path)]:
    stage = Usd.Stage.Open(path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=False)
    xf = UsdGeom.XformCache()
    print('ASSET', label, path, flush=True)
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if label == 'ROBOT' and 'base' not in str(p.GetPath()).lower():
            continue
        if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube) or p.HasAPI(UsdPhysics.RigidBodyAPI):
            box = cache.ComputeWorldBound(p).ComputeAlignedRange()
            print('BOUND', p.GetPath(), p.GetTypeName(), p.GetAppliedSchemas(), 'min', box.GetMin(), 'max', box.GetMax(), 'transform', xf.GetLocalToWorldTransform(p), flush=True)
app.close()
