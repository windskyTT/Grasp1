"""完整抓取真实状态：独立tip、真实碰撞顶点、阶段和连续保持。"""
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

TIP_NAMES = ('link_3_0_tip', 'link_7_0_tip', 'link_11_0_tip', 'link_15_0_tip')


def collision_vertices(stage, body_path):
    """读取真实碰撞Mesh/Cube顶点到刚体局部系；不以视觉AABB替代碰撞。"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    body = stage.GetPrimAtPath(body_path)
    inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    result = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        ancestor = prim
        collision = body.HasAPI(UsdPhysics.CollisionAPI)
        while ancestor.GetPath() != body.GetPath():
            collision |= ancestor.HasAPI(UsdPhysics.CollisionAPI)
            ancestor = ancestor.GetParent()
        if not collision:
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            points = [(x*half, y*half, z*half) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        else:
            continue
        transform = cache.GetLocalToWorldTransform(prim) * inverse
        result.extend([list(transform.Transform(Gf.Vec3d(*point))) for point in points])
    return result


def world_vertices(pose, points):
    return quat_apply(pose[:, None, 3:7].expand(-1, len(points), -1), points[None].expand(len(pose), -1, -1)) + pose[:, None, :3]


def contact_forces(env):
    """各tip的top/bottom/Table力；正常冲量只乘一个physics_dt。"""
    normal = torch.stack([env.scene[f'full_tip_{i}'].data.force_matrix_w[:, 0] for i in range(4)], 1)
    friction = torch.stack([env.scene[f'full_tip_{i}'].data.friction_forces_w[:, 0] for i in range(4)], 1)
    total = normal + friction
    obj = total[:, :, :2].norm(dim=-1).sum(-1)
    table = total[:, :, 2].norm(dim=-1)
    support = torch.stack([env.scene[f'full_support_{name}'].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                           for name in ('top', 'bottom')], -1)
    return obj, table, support, normal[:, :, :2].sum(2)


def hold_transition(contact, opposing, lift, unsupported, drift, angle, speed, safe, previous, dt, height, duration):
    """纯张量持续成功判定，供真实MDP和合成轨迹方向测试共用。"""
    holding = contact & opposing & (lift >= height) & unsupported & (drift <= 0.04) & (angle <= 0.4) & (speed <= 0.35) & safe
    elapsed = torch.where(holding, previous + dt, torch.zeros_like(previous))
    return holding, elapsed, elapsed >= duration


def phase_from_state(tip_distance, grasp, holding):
    return torch.where(holding, 3, torch.where(grasp, 2, torch.where(tip_distance < .06, 1, 0)))


def relative_velocity(top_velocity, palm_velocity, relative_position_w, palm_quaternion):
    """物体在随手掌旋转的坐标系中的真实相对速度。"""
    linear = top_velocity[:, :3]-palm_velocity[:, :3]-torch.cross(palm_velocity[:, 3:], relative_position_w, dim=-1)
    angular = top_velocity[:, 3:]-palm_velocity[:, 3:]
    return torch.cat((quat_apply_inverse(palm_quaternion, linear), quat_apply_inverse(palm_quaternion, angular)), -1)


class FullGraspState:
    def __init__(self, env):
        self.env = env
        robot, obj = env.scene['robot'], env.scene['object']
        self.tip_ids = [robot.body_names.index(name) for name in TIP_NAMES]
        self.palm_id = robot.body_names.index('Allegro_base_link')
        self.top_id = obj.body_names.index('top')
        self.hand_geometry = []
        for name in robot.body_names:
            if name.startswith('link_') or name == robot.body_names[self.palm_id]:
                points = collision_vertices(env.sim.stage, '/World/envs/env_0/Robot/'+name)
                if points:
                    self.hand_geometry.append((robot.body_names.index(name), torch.tensor(points, device=env.device)))
        self.object_geometry = []
        for name in ('top', 'bottom'):
            points = collision_vertices(env.sim.stage, '/World/envs/env_0/Object/'+name)
            # 源bottom是无碰撞的占位刚体；离桌间距取实际存在的碰撞形状。
            if points:
                self.object_geometry.append((obj.body_names.index(name), torch.tensor(points, device=env.device)))
        for i, name in enumerate(TIP_NAMES):
            sensor = env.scene[f'full_tip_{i}']
            assert sensor.body_names == [name] and sensor.num_bodies == 1
            assert sensor.data.force_matrix_w.shape == (env.num_envs, 1, 3, 3)
            assert sensor.data.friction_forces_w is not None
        for name in ('top', 'bottom'):
            sensor = env.scene[f'full_support_{name}']
            assert sensor.body_names == [name] and sensor.data.force_matrix_w.shape == (env.num_envs, 1, 1, 3)
        n, device = env.num_envs, env.device
        self.initial_z = obj.data.body_link_pos_w[:, self.top_id, 2].clone()
        self.hold = torch.zeros(n, device=device)
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.valid_reset = torch.ones(n, dtype=torch.bool, device=device)
        self.reset_attempts = torch.zeros(n, dtype=torch.long, device=device)
        self.grip_reference = torch.zeros((n, 7), device=device)
        self.grip_reference[:, 3] = 1
        self.grip_active = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_step = torch.full((n,), -1, device=device, dtype=torch.long)
        self.last_success = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_failure = torch.zeros(n, dtype=torch.long, device=device)
        self.last_lift = torch.zeros(n, device=device)
        self.last_hold = torch.zeros(n, device=device)
        self.last_phase = torch.zeros(n, dtype=torch.long, device=device)
        self.last_reset_valid = torch.ones(n, dtype=torch.bool, device=device)
        self.finished = torch.zeros(n, dtype=torch.bool, device=device)
        self.snapshot = None
        self.snapshot_step = -1

    def geometry(self):
        env = self.env
        from ..grasp1_env_cfg import SUPPORT_HEIGHT, TABLE_XY_BOUNDS
        robot, obj = env.scene['robot'], env.scene['object']
        min_z = torch.full((env.num_envs,), torch.inf, device=env.device)
        tip_z = []
        for body, points in self.hand_geometry:
            vertices = world_vertices(robot.data.body_link_pose_w[:, body], points) - env.scene.env_origins[:, None]
            inside = (vertices[..., 0] >= TABLE_XY_BOUNDS[0]) & (vertices[..., 0] <= TABLE_XY_BOUNDS[1]) & (vertices[..., 1] >= TABLE_XY_BOUNDS[2]) & (vertices[..., 1] <= TABLE_XY_BOUNDS[3])
            min_z = torch.minimum(min_z, torch.where(inside, vertices[..., 2], torch.inf).amin(1))
            if body in self.tip_ids:
                tip_z.append((self.tip_ids.index(body), vertices[..., 2].amin(1), vertices[..., 2].amax(1)))
        object_min = torch.stack([world_vertices(obj.data.body_link_pose_w[:, body], points)[..., 2].amin(1)
                                  for body, points in self.object_geometry], -1).amin(1) - env.scene.env_origins[:, 2]
        tip_bounds = torch.stack([torch.stack((lo, hi), -1) for _, lo, hi in sorted(tip_z)], 1)
        return min_z - SUPPORT_HEIGHT, object_min - SUPPORT_HEIGHT, tip_bounds

    def read(self):
        env = self.env
        if self.snapshot is not None and self.snapshot_step == env.common_step_counter:
            return self.snapshot
        robot, obj = env.scene['robot'], env.scene['object']
        term = env.action_manager.get_term('teacher')
        q = robot.data.joint_pos[:, term._joint_ids]
        qd = robot.data.joint_vel[:, term._joint_ids]
        limits = robot.data.joint_pos_limits[:, term._joint_ids]
        violation = torch.maximum(limits[..., 0]-q, q-limits[..., 1]).clamp_min(0)
        top = obj.data.body_link_pose_w[:, self.top_id]
        palm = robot.data.body_link_pose_w[:, self.palm_id]
        relpos = quat_apply_inverse(palm[:, 3:7], top[:, :3]-palm[:, :3])
        relquat = quat_mul(quat_conjugate(palm[:, 3:7]), top[:, 3:7])
        relvel = relative_velocity(obj.data.body_link_vel_w[:, self.top_id], robot.data.body_link_vel_w[:, self.palm_id],
            top[:, :3]-palm[:, :3], palm[:, 3:7])
        force, table, support, normal = contact_forces(env)
        contacts = force > env.cfg.tip_contact_threshold_N
        grasp = contacts[:, 3] & contacts[:, :3].any(1)
        direction = torch.nn.functional.normalize(normal, dim=-1)
        opposing = ((direction[:, :3] * direction[:, 3:4]).sum(-1) < -0.1).logical_and(contacts[:, :3]).any(1)
        new_step = (self.last_step != env.common_step_counter) & (env.episode_length_buf > 0)
        clearance, object_clearance, tip_bounds = self.geometry()
        arm_force = torch.stack([s.data.net_forces_w[:, 0].norm(dim=-1) for s in env._teacher_runtime_features.contact_sensors['teacher_arm_contact_']], -1).amax(1)
        finite = torch.isfinite(torch.cat((q, qd, term.target, top, relpos, relquat, relvel, force, table, support), -1)).all(1)
        unsafe = (~finite) | (violation.amax(1) > 0.005) | (clearance < -0.002) | (table.amax(1) > 20) | (arm_force > 5) | (qd[:, 6:].abs().amax(1) > 14) | (obj.data.body_link_vel_w[:, self.top_id, :3].norm(dim=-1) > 2)
        lift = top[:, 2] - self.initial_z
        unsupported = (support.sum(1) < 0.1) & (object_clearance > 0.005)
        # Hold参考从真实离桌并稳定夹持时建立；闭合形成夹持的正常调整不算保持漂移。
        eligible = grasp & opposing & (lift >= env.cfg.lift_height) & unsupported & ~unsafe & self.valid_reset & (relvel[:, :3].norm(dim=-1) <= .35)
        acquire = new_step & eligible & ~self.grip_active
        self.grip_reference[acquire] = torch.cat((relpos, relquat), -1)[acquire]
        drift_vector = relpos - self.grip_reference[:, :3]
        drift = drift_vector.norm(dim=-1)
        angle = 2*torch.acos((relquat*self.grip_reference[:, 3:]).sum(-1).abs().clamp(0, 1))
        self.grip_active[new_step] = (eligible & (drift <= .04) & (angle <= .4))[new_step]
        holding, elapsed, success = hold_transition(grasp, opposing, lift, unsupported, drift, angle,
            relvel[:, :3].norm(dim=-1), ~unsafe & self.valid_reset, self.hold, env.step_dt, env.cfg.lift_height, env.cfg.hold_duration_s)
        self.hold[new_step] = elapsed[new_step]
        from .observations import nearest_affordance_vectors_w
        tip_distance = nearest_affordance_vectors_w(env)[:, (4,8,12,16)].norm(dim=-1).mean(1)
        phase = phase_from_state(tip_distance, grasp, holding)
        self.phase[new_step] = phase[new_step]
        self.last_step[new_step] = env.common_step_counter
        self.snapshot = dict(q=q, qd=qd, violation=violation, top=top, palm=palm, relpos=relpos, relquat=relquat, relvel=relvel,
                    force=force, table=table, support=support, contacts=contacts, grasp=grasp, opposing=opposing,
                    clearance=clearance, object_clearance=object_clearance, tip_bounds=tip_bounds, arm_force=arm_force,
                    unsafe=unsafe, lift=lift, holding=holding, success=(self.hold >= env.cfg.hold_duration_s) & holding,
                    drift_vector=drift_vector, angle=angle)
        self.snapshot_step = env.common_step_counter
        return self.snapshot


def state(env):
    return env._full_grasp_state.read()
