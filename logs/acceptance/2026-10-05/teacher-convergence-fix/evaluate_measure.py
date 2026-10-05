"""执行现有evaluate.py，采集阶段7所需抓取失败诊断；不改变动作或评估判据。"""
import json
from pathlib import Path
import runpy
import sys
from isaaclab.app import AppLauncher

output_dir=Path(sys.argv[sys.argv.index('--output_dir')+1])
data={}
steps=0
action_sum=action_peak=None
contact_steps=contact_peak=None
launcher_init=AppLauncher.__init__
def launch_and_measure(self,*args,**kwargs):
    launcher_init(self,*args,**kwargs)
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from Grasp1.tasks.manager_based.grasp1.mdp.keypoints import hand_keypoints_o
    original_reset=RslRlVecEnvWrapper.reset
    def measured_reset(self,*args,**kwargs):
        result=original_reset(self,*args,**kwargs)
        e=self.unwrapped
        distances=torch.cdist(hand_keypoints_o(e),e._teacher_affordance_points_o).amin(-1)
        data.update(objects=list(e.cfg.object_names),initial_height=e.scene['object'].data.root_pos_w[:,2].tolist(),
            initial_joint_pos=e.scene['robot'].data.joint_pos.tolist(),
            pregrasp_keypoint_distance=distances.tolist(),
            action_joint_names=[e.scene['robot'].joint_names[i] for i in e.action_manager.get_term('teacher')._joint_ids],
            arm_scale=e.cfg.actions.teacher.scale['shoulder_pan_joint'])
        return result
    RslRlVecEnvWrapper.reset=measured_reset
    original_step=RslRlVecEnvWrapper.step
    def measured_step(self,actions):
        global steps,action_sum,action_peak,contact_steps,contact_peak
        e=self.unwrapped
        if steps<240:
            if action_sum is None: action_sum=torch.zeros_like(actions[0]); action_peak=torch.zeros_like(actions[0])
            action_sum+=actions.abs().mean(0)
            action_peak=torch.maximum(action_peak,actions.abs().max(0).values)
        result=original_step(self,actions)
        steps+=1
        if steps<=240:
            forces=torch.stack([e.scene[f'teacher_af_contact_{i}'].data.force_matrix_w[:,0,0].norm(dim=-1) for i in range(13)],dim=1).max(1).values
            if contact_steps is None:
                contact_steps=torch.zeros(e.num_envs,device=e.device,dtype=torch.long)
                contact_peak=torch.zeros_like(forces)
            contact_steps+=(forces>0.1).long()
            contact_peak=torch.maximum(contact_peak,forces)
        if steps==1:
            data.update(first_step_joint_pos=e.scene['robot'].data.joint_pos.tolist(),
                first_step_keypoint_distance=torch.cdist(hand_keypoints_o(e),e._teacher_affordance_points_o).amin(-1).tolist(),
                first_step_done=result[2].tolist())
        if steps==240:
            data.update(grasp_end_joint_pos=e.scene['robot'].data.joint_pos.tolist(),
                grasp_raw_action_mean_abs_by_joint=(action_sum/240).tolist(),
                grasp_raw_action_peak_abs_by_joint=action_peak.tolist(),
                grasp_end_keypoint_distance=torch.cdist(hand_keypoints_o(e),e._teacher_affordance_points_o).amin(-1).tolist(),
                grasp_contact_steps_above_01N=contact_steps.tolist(),
                grasp_max_filtered_contact_force=contact_peak.tolist())
        return result
    RslRlVecEnvWrapper.step=measured_step
    original_close=RslRlVecEnvWrapper.close
    def measured_close(self):
        e=self.unwrapped
        data.update(final_height=e.scene['object'].data.root_pos_w[:,2].tolist(),
            final_joint_pos=e.scene['robot'].data.joint_pos.tolist(),
            height_gain=(e.scene['object'].data.root_pos_w[:,2]-torch.tensor(data['initial_height'],device=e.device)).tolist(),
            object_joint_pos=e.scene['object'].data.joint_pos.tolist(),
            robot_joint_names=e.scene['robot'].joint_names)
        output_dir.mkdir(parents=True,exist_ok=True)
        (output_dir/'diagnostics.json').write_text(json.dumps(data,indent=2))
        return original_close(self)
    RslRlVecEnvWrapper.close=measured_close
AppLauncher.__init__=launch_and_measure
sys.path.insert(0,str(Path('scripts/rsl_rl').resolve()))
sys.argv[0]='scripts/rsl_rl/evaluate.py'
runpy.run_path('scripts/rsl_rl/evaluate.py',run_name='__main__')
