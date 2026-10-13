"""优化8诊断专用：实际碰撞几何、固定安装过滤和PhysX碰撞对报告。"""
import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from omni.physx import get_physx_simulation_interface


def prepare_collision_variant(cfg, variant):
    if variant == 'box':
        from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import TABLE_SIZE, TABLE_CENTER_XY, SUPPORT_HEIGHT
        cfg.scene.table.spawn = sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )
        cfg.scene.table.init_state.pos = (*TABLE_CENTER_XY, SUPPORT_HEIGHT / 2)
        cfg.scene.table.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    def configure(env, env_ids):
        stage = env.sim.stage
        env.sim.carb_settings.set_bool('/physics/disableContactProcessing', False)
        for index in range(env.num_envs):
            root = f'/World/envs/env_{index}'
            base = stage.GetPrimAtPath(root + '/Robot/base_link')
            if variant == 'base_filter':
                UsdPhysics.FilteredPairsAPI.Apply(base).CreateFilteredPairsRel().AddTarget(Sdf.Path(root + '/Table'))
            # 额外报告固定底座与物体、桌子的接触；不改变碰撞过滤或动力学。
            for path in (root + '/Robot/base_link', root + '/Table', root + '/Object/top'):
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.0)
    cfg.events.optimization8_contact_reporting = EventTermCfg(func=configure, mode='prestartup')


def runtime_audit(env):
    stage = env.sim.stage
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    colliders, rigid_bodies, groups = [], [], []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith('/World/envs/env_0/'):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(dict(path=path, kinematic=UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(),
                filtered_pairs=[str(p) for p in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]))
        if prim.IsA(UsdPhysics.CollisionGroup):
            groups.append(dict(path=path, relationships={r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships()}))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # CollisionAPI可位于instance proxy的Xform上，直接BBoxCache会返回空范围；
        # 从真实Mesh/Cube顶点和世界变换计算外包围盒，包含guide碰撞几何。
        box = Gf.Range3d()
        for shape in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if shape.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(shape).GetPointsAttr().Get()
            elif shape.IsA(UsdGeom.Cube):
                half = UsdGeom.Cube(shape).GetSizeAttr().Get() / 2
                points = [(x*half, y*half, z*half) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            else:
                continue
            transform = transforms.GetLocalToWorldTransform(shape)
            for point in points:
                box.UnionWith(transform.Transform(Gf.Vec3d(*point)))
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        material_values = {}
        if material:
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            material_values = dict(path=str(material.GetPath()), static_friction=api.GetStaticFrictionAttr().Get(),
                dynamic_friction=api.GetDynamicFrictionAttr().Get(), restitution=api.GetRestitutionAttr().Get())
        colliders.append(dict(path=path, type=prim.GetTypeName(), instance_proxy=prim.IsInstanceProxy(),
            enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
            approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
            aabb_min=list(box.GetMin()), aabb_max=list(box.GetMax()), material=material_values))
    table = next(c for c in colliders if '/Table/' in c['path'] or c['path'].endswith('/Table'))
    base = next(c for c in colliders if '/Robot/base_link/' in c['path'])
    overlap = [min(base['aabb_max'][i], table['aabb_max'][i]) - max(base['aabb_min'][i], table['aabb_min'][i]) for i in range(3)]
    return dict(physics_dt=env.physics_dt, control_dt=env.step_dt, colliders=colliders, rigid_bodies=rigid_bodies,
        collision_groups=groups, base_table_aabb_overlap_m=overlap,
        aabb_note='Broad-phase bounds are evidence of proximity; overlap alone does not prove a contact constraint.')


def subscribe_contacts(step):
    pairs = {}
    def report(headers, data):
        for header in headers:
            actor0, actor1, collider0, collider1 = [str(PhysicsSchemaTools.intToSdfPath(p)) for p in
                (header.actor0, header.actor1, header.collider0, header.collider1)]
            if not (actor0.startswith('/World/envs/env_0/') or actor1.startswith('/World/envs/env_0/')):
                continue
            key = collider0 + ' | ' + collider1
            record = pairs.setdefault(key, dict(actor0=actor0, actor1=actor1, collider0=collider0, collider1=collider1,
                events=0, contacts=0, impulse_peak_Ns=0.0, separation_min_m=None, first_physics_step=step[0]+1, samples=[]))
            record['events'] += 1
            for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                contact = data[i]
                impulse = list(contact.impulse)
                magnitude = math.sqrt(sum(v*v for v in impulse))
                record['contacts'] += 1
                peak = magnitude > record['impulse_peak_Ns']
                record['impulse_peak_Ns'] = max(record['impulse_peak_Ns'], magnitude)
                record['separation_min_m'] = contact.separation if record['separation_min_m'] is None else min(record['separation_min_m'], contact.separation)
                if len(record['samples']) < 8 or peak:
                    record['samples'].append(dict(physics_step=step[0]+1, position=list(contact.position), normal=list(contact.normal),
                        impulse_Ns=impulse, separation_m=contact.separation))
                    record['samples'] = record['samples'][-32:]
    # ManagerBased环境直接simulate/fetch，不依赖GUI timeline派发；立即API按物理步读取。
    def collect():
        headers, data = get_physx_simulation_interface().get_contact_report()
        report(headers, data)
    return pairs, collect
