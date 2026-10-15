"""窄训练课程的真实IK/碰撞几何reset；回合动作始终由actor控制。"""
import torch
from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat
from . import events, geometry


def configure_hand_friction(env, env_ids):
    """新任务专用候选：显式设置旧摩擦，保留表面摩擦和所有碰撞。"""
    from pxr import PhysxSchema, UsdPhysics
    value = env.cfg.legacy_joint_friction
    if value is None:
        return
    for prim in env.sim.stage.Traverse():
        if '/Robot/joints/joint_' in str(prim.GetPath()) and prim.IsA(UsdPhysics.RevoluteJoint):
            PhysxSchema.PhysxJointAPI(prim).GetJointFrictionAttr().Set(value)


def configure_hand_model(env, env_ids):
    """独立候选：使用有力矩单位的新摩擦模型，区别于旧反力比例系数。"""
    if env.cfg.hand_static_friction is not None:
        robot = env.scene['robot']
        hand = list(env.action_manager.get_term('teacher')._joint_ids[6:])
        robot.write_joint_friction_coefficient_to_sim(env.cfg.hand_static_friction, joint_ids=hand)


def reset_full_grasp(env, env_ids, dataset_name, object_names, palm_quaternion_w=None, tcp_offset_h=None, tcp_height=None):
    """最多8次静态IK/真实碰撞顶点筛选；首物理步仍检查真实危险接触。"""
    from ..grasp1_env_cfg import SUPPORT_HEIGHT
    ids = events._env_ids(env, env_ids)
    data, memory = env._teacher_reset_data, env._full_grasp_state
    robot, obj = env.scene['robot'], env.scene['object']
    completed = env.episode_length_buf[ids] > 0
    memory.finished[ids] = completed
    if bool(completed.any()):
        done_ids = ids[completed]
        previous = memory.read()
        memory.last_lift[done_ids] = previous['lift'][done_ids]
        memory.last_hold[done_ids] = memory.hold[done_ids]
        memory.last_phase[done_ids] = memory.phase[done_ids]
        memory.last_reset_valid[done_ids] = memory.valid_reset[done_ids]
        memory.last_success[done_ids] = env.termination_manager.get_term('success')[done_ids]
        memory.last_failure[done_ids] = (env.termination_manager.get_term('unsafe')[done_ids].long()
            + 2*env.termination_manager.get_term('reset_failed')[done_ids].long()
            + 4*env.termination_manager.get_term('dropped')[done_ids].long()
            + 8*env.termination_manager.get_term('time_out')[done_ids].long())
    memory.valid_reset[ids] = False
    memory.reset_attempts[ids] = 0
    pending = ids
    for attempt in range(env.cfg.reset_max_attempts):
        if pending.numel() == 0:
            break
        n = len(pending)
        pose = torch.zeros((n, 7), device=env.device)
        center = pose.new_tensor(env.cfg.reset_xy_center)
        pose[:, :2] = env.scene.env_origins[pending, :2] + center + (torch.rand((n, 2), device=env.device)*2-1)*env.cfg.reset_xy_range
        pose[:, 2] = SUPPORT_HEIGHT-data['lowest'][pending]+env.scene.env_origins[pending, 2]+0.001
        zeros = torch.zeros(n, device=env.device)
        yaw = (torch.rand(n, device=env.device)*2-1)*env.cfg.reset_yaw_range
        pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        cloud = geometry.object_points_to_world(data['points'][pending], pose[:, :3], pose[:, 3:7])
        direction = torch.zeros((n, 3), device=env.device)
        direction[:, 2] = 1
        base_pos, base_rotation = events._solver_base_pose(robot, pending)
        if palm_quaternion_w is None:
            rotations, widths = geometry.sample_rot_mats(direction, 8, cloud)
            targets = geometry.wrist_poses_for_ur5_ik(cloud, direction, rotations, base_pos, base_rotation,
                pregrasp_distance=env.cfg.pregrasp_distance)
        else:
            # 新任务固定掌轴，避免Reset在8个朝向间隐式切换；绝不旋转机器人root。
            rotation = matrix_from_quat(pose.new_tensor(palm_quaternion_w).expand(n, -1))
            tcp = cloud.mean(1)+direction*env.cfg.pregrasp_distance
            if tcp_height is not None:
                tcp[:, :2] = env.scene.env_origins[pending, :2]+pose.new_tensor(env.cfg.reset_xy_center)
                tcp[:, 2] = SUPPORT_HEIGHT+tcp_height+env.scene.env_origins[pending, 2]
            targets = geometry.tcp_pose_for_ur5_ik(tcp, rotation, pose.new_tensor(tcp_offset_h).expand(n, -1),
                base_pos, base_rotation)[:, None]
            widths = torch.zeros((n, 1), device=env.device)
        solutions, feasible = geometry.ur5_inverse_kinematics(targets)
        # 按可抓宽度优先，依次尝试不同姿态，不以不合法的零IK作为fallback。
        order = torch.argsort(torch.where(feasible, widths, torch.inf), dim=-1)
        choice = order[:, attempt % targets.shape[1]]
        rows = torch.arange(n, device=env.device)
        q = robot.data.default_joint_pos[pending].clone()
        q[:, list(data['joint_ids'][:6])] = solutions[rows, choice]
        events._write_reset_state(env, pending, q, pose)
        env.sim.forward()
        env.scene.update(0.0)
        env._teacher_runtime_features.invalidate(pending)
        clearance, _, _ = memory.geometry()
        limits = robot.data.joint_pos_limits[pending]
        in_limits = ((q >= limits[..., 0]) & (q <= limits[..., 1])).all(1)
        valid = feasible[rows, choice] & in_limits & (clearance[pending] >= env.cfg.reset_clearance)
        if not hasattr(env, '_full_grasp_reset_diagnostics'):
            env._full_grasp_reset_diagnostics = torch.zeros((env.num_envs, 3), device=env.device)
        env._full_grasp_reset_diagnostics[pending] = torch.stack((feasible[rows, choice].float(), in_limits.float(), clearance[pending]), -1)
        memory.valid_reset[pending] = valid
        memory.reset_attempts[pending] += 1
        data['joint_pos'][pending] = q
        data['object_pose'][pending] = pose
        env._teacher_last_ik_feasible[pending] = feasible[rows, choice]
        pending = pending[~valid]
    # 筛选失败必须在下一步终止并计入分母；不无限重采样、不移动物体到手里。
    data['bias_pending'][ids] = False
    data['collision_pending'][ids] = False
    memory.initial_z[ids] = obj.data.body_link_pos_w[ids, memory.top_id, 2]
    memory.hold[ids] = 0
    memory.phase[ids] = 0
    memory.grip_active[ids] = False
    memory.grip_reference[ids] = 0
    memory.grip_reference[ids, 3] = 1
    memory.last_step[ids] = env.common_step_counter
    memory.snapshot = None
