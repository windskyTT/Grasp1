from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import Grasp1.tasks
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from pxr import Usd, UsdPhysics
try:
    task = 'Grasp1-UR5-Allegro-Teacher-v0'
    cfg = parse_env_cfg(task, device='cuda:0', num_envs=1)
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    print('CONFIG_PASS', gym.spec(task), type(cfg), type(agent), flush=True)
    paths = [('robot', cfg.scene.robot.spawn.usd_path), ('table', cfg.scene.table.spawn.usd_path)]
    paths += [('object', p) for p in cfg.scene.object.spawn.usd_path]
    for kind, path in paths:
        stage = Usd.Stage.Open(path)
        assert stage is not None, path
        prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
        bodies = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints = [p.GetName() for p in prims if p.IsA(UsdPhysics.Joint)]
        collisions = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert collisions, (kind, path, 'no collision')
        print('USD_OPEN', kind, path, 'bodies', bodies, 'joints', joints, 'collisions',len(collisions), flush=True)
    print('ASSET_OPEN_PASS', len(paths), flush=True)
finally:
    app.close()
