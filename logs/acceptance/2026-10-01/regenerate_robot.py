"""按原关节与碰撞配置重新导入仅含 UR5 和 Allegro 的 URDF。"""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
cfg=UrdfConverterCfg(asset_path='/tmp/grasp1-ur5-arm-hand-import/ur5_allegro.urdf',usd_dir='/home/windsky/project/Grasp1/assets/robots/ur5_allegro',usd_file_name='ur5_allegro.usd',fix_base=True,merge_fixed_joints=False,force_usd_conversion=True,joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0,damping=0.0)))
converter=UrdfConverter(cfg)
print('CONVERSION_PASS',converter.usd_path,flush=True)
app.close()
