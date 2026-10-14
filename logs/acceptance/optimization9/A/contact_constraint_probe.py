"""复用优化8原样回放，补测全部机器人刚体接触及关节反力；不改训练配置。"""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--diagnostic_filter_body')
parser.add_argument('--diagnostic_position_iterations', type=int)
parser.add_argument('--diagnostic_merge_tip', action='store_true')
parser.add_argument('--diagnostic_external_forces_every_iteration', action='store_true')
extra, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
historical = Path('logs/acceptance/optimization8/A')
sys.path.insert(0, str(historical.resolve()))


def prepare(cfg, variant):
    original_prepare(cfg, variant)
    if extra.diagnostic_external_forces_every_iteration:
        cfg.sim.physx.enable_external_forces_every_iteration = True
    if extra.diagnostic_position_iterations:
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = extra.diagnostic_position_iterations
    if extra.diagnostic_merge_tip:
        from merge_tip_asset import merge_thumb_tip
        output = Path(sys.argv[sys.argv.index('--output_dir') + 1])
        output.mkdir(parents=True, exist_ok=True)
        cfg.scene.robot.spawn.usd_path = merge_thumb_tip(cfg.scene.robot.spawn.usd_path, output/'robot_thumb_merged.usda')
    if extra.diagnostic_filter_body:
        from isaaclab.managers import EventTermCfg
        from pxr import Sdf, UsdPhysics

        def filter_pair(env, env_ids):
            body = env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/' + extra.diagnostic_filter_body)
            UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().AddTarget(Sdf.Path('/World/envs/env_0/Table'))

        cfg.events.optimization9_pair_isolation = EventTermCfg(func=filter_pair, mode='prestartup')


def inspect(env):
    import torch
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.simulation_manager import SimulationManager

    result = original_audit(env)
    scene_api = PhysxSchema.PhysxSceneAPI(env.sim.stage.GetPrimAtPath(env.cfg.sim.physics_prim_path))
    result['external_forces_every_iteration'] = dict(configured=env.cfg.sim.physx.enable_external_forces_every_iteration,
                                                    authored=scene_api.GetEnableExternalForcesEveryIterationAttr().Get())
    output = Path(sys.argv[sys.argv.index('--output_dir') + 1])
    robot = env.scene['robot']
    term = env.action_manager.get_term('teacher')
    body_paths = ['/World/envs/env_0/Robot/' + name for name in robot.body_names]
    physics = SimulationManager.get_physics_sim_view()
    views = [physics.create_rigid_contact_view(path, ['/World/envs/env_0/Table'], 128) for path in body_paths]
    # 原审计只记录带CollisionAPI的Xform；同时检查其实际后代Mesh和点数。
    transform = UsdGeom.XformCache(Usd.TimeCode.Default())
    geometry = []
    for prim in env.sim.stage.Traverse(Usd.TraverseInstanceProxies()):
        if not str(prim.GetPath()).startswith('/World/envs/env_0/'):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            matrix = transform.GetLocalToWorldTransform(prim)
            world = [matrix.Transform(Gf.Vec3d(*p)) for p in points]
            geometry.append(dict(path=str(prim.GetPath()), points=len(points), collision_api=prim.HasAPI(UsdPhysics.CollisionAPI),
                                 min=[min(p[i] for p in world) for i in range(3)], max=[max(p[i] for p in world) for i in range(3)]))
    (output/'actual_mesh_geometry.json').write_text(json.dumps(geometry, indent=2))
    frames = []
    update = env.scene.update
    close = env.close

    def collect(dt):
        update(dt)
        if dt != env.physics_dt:
            return
        table = torch.stack([view.get_contact_force_matrix(dt).reshape(3).clone() for view in views])
        net = torch.stack([view.get_net_contact_forces(dt).reshape(3).clone() for view in views])
        tip_name = 'link_15_0' if extra.diagnostic_merge_tip else 'link_15_0_tip'
        tip = views[robot.body_names.index(tip_name)]
        force, point, normal, distance, count, start = tip.get_contact_data(dt)
        slots = torch.arange(128, device=env.device)
        indices = (start[0, 0] + slots).long().clamp(0, len(force) - 1)
        valid = slots < count[0, 0]
        tip_contact = torch.cat((force[indices].reshape(-1, 1), point[indices], normal[indices], distance[indices].reshape(-1, 1)), 1)
        tip_contact = torch.where(valid[:, None], tip_contact, torch.zeros_like(tip_contact))
        frames.append(dict(table_force=table, net_force=net,
                           tip_contact=tip_contact, tip_contact_count=count.clone(),
                           joint_wrench=robot.root_physx_view.get_link_incoming_joint_force().clone(),
                           projected_force=robot.root_physx_view.get_dof_projected_joint_forces().clone(),
                           q=robot.data.joint_pos[:, term._joint_ids].clone(),
                           qd=robot.data.joint_vel[:, term._joint_ids].clone()))

    def save():
        if frames:
            data = {key: torch.stack([frame[key] for frame in frames]).cpu() for key in frames[0]}
            data.update(body_paths=body_paths, joints=term._joint_names, physics_dt=env.physics_dt,
                        joint_wrench_frame='PhysX Tensor API: child joint frame', diagnostic_filter_body=extra.diagnostic_filter_body)
            torch.save(data, output/'contact_constraint_trace.pt')
            table_peak = data['table_force'].norm(dim=-1).amax(0)
            net_peak = data['net_force'].norm(dim=-1).amax(0)
            summary = dict(physics_steps=len(frames), body_table_peak_N=dict(zip(robot.body_names, table_peak.tolist())),
                           body_net_peak_N=dict(zip(robot.body_names, net_peak.tolist())),
                           joint_wrench_peak=data['joint_wrench'].abs().amax(dim=(0,1)).tolist(),
                           projected_force_peak=data['projected_force'].abs().amax(dim=(0,1)).tolist())
            (output/'contact_constraint_summary.json').write_text(json.dumps(summary, indent=2))
            print('OPT9_CONTACT_CONSTRAINT', json.dumps(summary), flush=True)
        close()

    env.scene.update = collect
    env.close = save
    return result


# 在历史脚本启动AppLauncher之后安装采样钩子；原文件和回放逻辑保持不变。
source = (historical/'physics_probe.py').read_text()
source = source.replace(
    'from physics_runtime_audit import prepare_collision_variant, runtime_audit, subscribe_contacts',
    'import physics_runtime_audit as audit\n'
    'original_prepare = audit.prepare_collision_variant\n'
    'original_audit = audit.runtime_audit\n'
    'audit.prepare_collision_variant = prepare\n'
    'audit.runtime_audit = inspect\n'
    'from physics_runtime_audit import prepare_collision_variant, runtime_audit, subscribe_contacts',
)
exec(compile(source, str(historical/'physics_probe.py'), 'exec'), globals())
