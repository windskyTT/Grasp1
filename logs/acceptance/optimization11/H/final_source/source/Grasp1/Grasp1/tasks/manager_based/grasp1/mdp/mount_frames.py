"""只读法兰/掌/指尖/TCP坐标轴；标记不含碰撞和刚体。"""
import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate
from Grasp1.robots.robot_profile import GRASP_TCP_OFFSET


class MountFrames:
    def __init__(self, env, tcp_offset=GRASP_TCP_OFFSET):
        self.env = env
        robot = env.scene['robot']
        self.names = ('tool0', 'Flange_base_link', 'Allegro_base_link',
                      'link_3_0_tip', 'link_7_0_tip', 'link_11_0_tip', 'link_15_0_tip')
        self.ids = [robot.body_names.index(name) for name in self.names]
        self.tcp_offset = torch.tensor(tcp_offset, device=env.device)[None]
        cfg = FRAME_MARKER_CFG.copy()
        cfg.prim_path = '/Visuals/MountFrames'
        del cfg.markers['connecting_line']
        cfg.markers['frame'].scale = (.06, .06, .06)
        self.markers = VisualizationMarkers(cfg)

    def update(self):
        robot = self.env.scene['robot']
        poses = robot.data.body_link_pose_w[0, self.ids]
        palm = poses[2:3]
        tcp = palm[:, :3]+quat_apply(palm[:, 3:7], self.tcp_offset)
        self.markers.visualize(torch.cat((poses[:, :3], tcp)), torch.cat((poses[:, 3:7], palm[:, 3:7])))
        from ..grasp1_env_cfg import SUPPORT_HEIGHT
        return dict(frame_names=(*self.names, 'geometric_TCP'),
                    positions_w=torch.cat((poses[:, :3], tcp)).cpu().tolist(),
                    quaternions_wxyz=torch.cat((poses[:, 3:7], palm[:, 3:7])).cpu().tolist(),
                    tcp_height_above_table_m=float(tcp[0, 2]-self.env.scene.env_origins[0, 2]-SUPPORT_HEIGHT),
                    flange_to_hand_xyz=quat_apply_inverse(poses[1, 3:7], poses[2, :3]-poses[1, :3]).cpu().tolist(),
                    flange_to_hand_quaternion_wxyz=quat_mul(quat_conjugate(poses[1, 3:7]), poses[2, 3:7]).cpu().tolist())
