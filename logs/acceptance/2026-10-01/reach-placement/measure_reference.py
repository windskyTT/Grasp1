"""测量本机 Franka Reach 的 Table 与 Robot 相对位姿，并比较 Grasp1。"""
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
app=launcher.app
from pxr import Usd,UsdGeom,Gf
import numpy as np
import json
from pathlib import Path
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg
from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import Grasp1SceneCfg
stage=Usd.Stage.Open('/home/windsky/project/Grasp1/assets/table/table.usda')
xf=UsdGeom.XformCache()
mesh=stage.GetPrimAtPath('/Table/Collisions/Cube')
local=xf.GetLocalToWorldTransform(mesh)*xf.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
result={}
for name,cfg in [('franka_reach',FrankaReachEnvCfg().scene),('grasp1',Grasp1SceneCfg())]:
    scale=cfg.table.spawn.scale or (1,1,1)
    q=cfg.table.init_state.rot
    matrix=local*Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))*Gf.Matrix4d().SetRotate(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))*Gf.Matrix4d().SetTranslate(Gf.Vec3d(*cfg.table.init_state.pos))
    points=np.asarray([matrix.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
    lo,hi=points.min(0),points.max(0)
    robot=np.asarray(cfg.robot.init_state.pos)
    center=(lo+hi)/2
    result[name]={'table_min':lo.tolist(),'table_max':hi.tolist(),'table_center':center.tolist(),'robot_pos':robot.tolist(),'robot_quat':list(cfg.robot.init_state.rot),'robot_minus_table_center_xy':(robot[:2]-center[:2]).tolist(),'near_edge_distance':float(robot[0]-lo[0])}
assert np.allclose(result['franka_reach']['robot_minus_table_center_xy'],result['grasp1']['robot_minus_table_center_xy'],atol=2e-5)
assert result['franka_reach']['robot_quat']==result['grasp1']['robot_quat']
Path('logs/acceptance/2026-10-01/reach-placement/reference.json').write_text(json.dumps(result,indent=2))
print('REFERENCE_RELATION_PASS',json.dumps(result),flush=True)
app.close()
