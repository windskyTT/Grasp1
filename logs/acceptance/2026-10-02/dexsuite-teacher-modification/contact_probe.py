"""用真实重叠检查两次 physics substeps 后的初始 arm collision 数据。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app
import json
import torch
import gymnasium as gym
import Grasp1.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply
from Grasp1.tasks.manager_based.grasp1.mdp.events import _current_arm_collision

cfg = parse_env_cfg('Grasp1-UR5-Allegro-Teacher-v0', num_envs=16)
cfg.set_num_envs(16)
env = gym.make('Grasp1-UR5-Allegro-Teacher-v0', cfg=cfg)
e = env.unwrapped
with torch.inference_mode():
    env.reset(seed=42)
    robot, obj = e.scene['robot'], e.scene['object']
    ids = torch.tensor([0], device=e.device)
    body_id = robot.body_names.index('upper_arm_link')
    pose = obj.data.root_pose_w[ids].clone()
    # 将 affordance 云中心置于 upper_arm 的质量中心，制造实际几何重叠。
    center = e._teacher_affordance_points_o[ids].mean(1)
    pose[:, :3] = robot.data.body_com_pos_w[ids, body_id] - quat_apply(pose[:, 3:7], center)
    obj.write_root_pose_to_sim(pose, env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros((1,6), device=e.device), env_ids=ids)
    e.scene.write_data_to_sim()
    for _ in range(2):
        e.sim.step(render=False)
        e.scene.update(e.physics_dt)
    forces = [float(e.scene[f'teacher_arm_contact_{i}'].data.net_forces_w[0].norm()) for i in range(6)]
    detected = bool(_current_arm_collision(e, ids)[0])
    assert detected and all(torch.isfinite(e.scene[f'teacher_arm_contact_{i}'].data.net_forces_w).all() for i in range(6))
    print('CONTACT_PROBE', json.dumps(dict(physics_steps=2, simulated_seconds=2*e.physics_dt, forces=forces, collision_detected=detected)), flush=True)
env.close()
app.close()
