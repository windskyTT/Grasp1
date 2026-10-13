"""合成方向测试：调用真实Reward函数，不作为物理抓取成功证据。"""
from types import SimpleNamespace as NS
import torch
from Grasp1.tasks.manager_based.grasp1.mdp import rewards as r
from Grasp1.tasks.manager_based.grasp1.mdp import rewards_grasp_v1 as aligned
from Grasp1.tasks.manager_based.grasp1.config.ur5_allegro.teacher_env_cfg import FINGER_REWARD_WEIGHTS, CONTACT_REWARD_WEIGHTS


def run():
    batch=2
    def sensors(count):
        return [NS(data=NS(force_matrix_w=torch.zeros(batch,1,1,3),friction_forces_w=torch.zeros(batch,1,1,3),current_contact_time=torch.zeros(batch,1))) for _ in range(count)]
    groups={r.AFFORDANCE_CONTACT_SENSOR_PREFIX:sensors(13),r.TABLE_CONTACT_SENSOR_PREFIX:sensors(13),r.ARM_CONTACT_SENSOR_PREFIX:sensors(6)}
    cache=NS(contact_sensors=groups,get=lambda name,compute:compute(slice(None)))
    pose=torch.zeros(batch,2,7);pose[:,:,3]=1
    robot=NS(data=NS(body_link_pose_w=pose.clone(),body_link_lin_vel_w=torch.zeros(batch,2,3),body_link_ang_vel_w=torch.zeros(batch,2,3),joint_pos=torch.zeros(batch,22),joint_vel=torch.zeros(batch,22)))
    obj=NS(body_names=['bottom','top'],data=NS(body_link_pose_w=pose.clone(),root_pos_w=torch.zeros(batch,3),body_link_lin_vel_w=torch.zeros(batch,2,3),body_link_ang_vel_w=torch.zeros(batch,2,3)))
    env=NS(scene={'robot':robot,'object':obj},num_envs=batch,physics_dt=1/120,step_dt=4/120,
           _teacher_runtime_features=cache,_teacher_body_indices=NS(wrist=0,object_top=1,arm_joints=tuple(range(6))))
    original={n:getattr(r,n) for n in ['_require_articulation','_resolve_body_id','nearest_affordance_vectors_w','hand_keypoints_w','arm_keypoints_w']}
    cases={}
    try:
        r._require_articulation=lambda e,name:e.scene[name]
        r._resolve_body_id=lambda asset,name:asset.body_names.index(name)
        vector=torch.zeros(batch,17,3);vector[0,:,0]=.04;vector[1,:,0]=.01
        r.nearest_affordance_vectors_w=lambda e:vector
        cases['approach_distance_improves']=bool(r.affordance_reward(env,FINGER_REWARD_WEIGHTS)[1]>r.affordance_reward(env,FINGER_REWARD_WEIGHTS)[0])
        points=torch.zeros(batch,17,3);points[0,:,2]=.791;points[1,:,2]=.772
        r.hand_keypoints_w=lambda e:points
        penalty=r.table_reward(env,FINGER_REWARD_WEIGHTS,.771)*(-.03)
        cases['table_proximity_penalty_direction']=bool(penalty[1]<penalty[0])
        arm=torch.zeros(batch,6,3);arm[0,:,2]=.791;arm[1,:,2]=.772
        r.arm_keypoints_w=lambda e:arm
        cases['arm_height_penalty_direction']=bool(r.arm_height_reward(env,.771)[1]>r.arm_height_reward(env,.771)[0])
        af=groups[r.AFFORDANCE_CONTACT_SENSOR_PREFIX]
        for i in [3]:af[i].data.force_matrix_w[0,0,0,2]=12
        for i in [3,12]:af[i].data.force_matrix_w[1,0,0,2]=12
        cases['thumb_other_contact_exceeds_single']=bool(r.affordance_contact_reward(env,CONTACT_REWARD_WEIGHTS)[1]>r.affordance_contact_reward(env,CONTACT_REWARD_WEIGHTS)[0])
        for sensor in af:sensor.data.force_matrix_w.zero_()
        af[3].data.force_matrix_w[:,:,0,:]=torch.tensor([12.,0.,12.])
        af[3].data.friction_forces_w[:,:,0,:]=torch.tensor([0.,6.,0.])
        projection=aligned.world_contact_impulses(env,r.AFFORDANCE_CONTACT_SENSOR_PREFIX)
        cases['one_physics_dt_world_vector']=bool(torch.allclose(projection[:,3],torch.tensor([.1,.05,.1]).expand(batch,-1)))
        cases['oblique_world_xy_differs_from_friction']=bool((aligned.affordance_impulse_reward(env,CONTACT_REWARD_WEIGHTS)>r.affordance_impulse_reward(env,CONTACT_REWARD_WEIGHTS)).all())
        old=projection.clone();env.step_dt=2/120
        cases['no_decimation_impulse_multiplier']=bool(torch.equal(old,aligned.world_contact_impulses(env,r.AFFORDANCE_CONTACT_SENSOR_PREFIX)))
        for sensor in af:sensor.data.force_matrix_w.zero_();sensor.data.friction_forces_w.zero_()
        af[0].data.force_matrix_w[0,0,0,2]=240;af[0].data.force_matrix_w[1,0,0,2]=-240
        cases['push_signed_z']=bool(aligned.push_reward(env)[0]==1 and aligned.push_reward(env)[1]==0)
        table=groups[r.TABLE_CONTACT_SENSOR_PREFIX]
        table[3].data.force_matrix_w[1,0,0,2]=12
        cases['hand_table_contact_penalty']=bool(r.table_contact_reward(env,CONTACT_REWARD_WEIGHTS)[1]>r.table_contact_reward(env,CONTACT_REWARD_WEIGHTS)[0])
        cases['hand_table_impulse_penalty']=bool(r.table_impulse_reward(env,CONTACT_REWARD_WEIGHTS)[1]>r.table_impulse_reward(env,CONTACT_REWARD_WEIGHTS)[0])
        arm_s=groups[r.ARM_CONTACT_SENSOR_PREFIX];arm_s[1].data.force_matrix_w[1,0,0,2]=12;arm_s[1].data.current_contact_time[1]=.01
        cases['arm_contact_penalty']=bool(r.arm_contact_reward(env)[1]>r.arm_contact_reward(env)[0])
        cases['arm_impulse_penalty']=bool(r.arm_impulse_reward(env)[1]>r.arm_impulse_reward(env)[0])
        cases['arm_collision_unfiltered']=bool(r.arm_collision_reward(env)[1]==1 and r.arm_collision_reward(env)[0]==0)
        robot.data.body_link_lin_vel_w[1,0,0]=.3;robot.data.body_link_ang_vel_w[1,0,2]=2
        obj.data.body_link_lin_vel_w[1,1,2]=.2;obj.data.body_link_ang_vel_w[1,1,2]=2
        for name in ['wrist_vel_reward_','wrist_qvel_reward_','obj_vel_reward_','obj_qvel_reward_']:
            val=getattr(r,name)(env);cases[name+'_direction']=bool(val[1]>val[0] and torch.isfinite(val).all())
        robot.data.joint_vel[1,0]=.6
        cases['arm_qvel_source_threshold']=bool(torch.allclose(r.arm_joint_vel_reward_(env),torch.tensor([0.,5.76])))
        term=r.ObjectDisplacementReward(NS(),env);obj.data.body_link_pose_w[1,1,2]=.1
        cases['top_displacement_reference_preserved']=bool(torch.allclose(term(env),torch.tensor([0.,.1])))
        # grasp-only的17项不能把lift、弹飞或桌面支撑分类为成功。
        cases['grasp_only_lift_velocity_penalty_preserved']=bool((r.obj_vel_reward_(env)*-15)[1]<0)
        for dt in [2/120,4/120]:
            env.step_dt=dt
            env.termination_manager=NS(get_term=lambda name:torch.tensor([True,False]))
            cases[f'terminal_exact_minus10_D{round(dt*120)}']=bool(torch.equal((r.invalid_hand_height_terminal(env)*dt*-10).float(),torch.tensor([-10.,0.])))
            env.termination_manager=NS(get_term=lambda name:torch.tensor([False,False]))
            cases[f'timeout_invalidobs_no_penalty_D{round(dt*120)}']=bool((r.invalid_hand_height_terminal(env)==0).all())
    finally:
        for name,value in original.items():setattr(r,name,value)
    assert all(cases.values()),cases
    return dict(test_kind='synthetic_cpu_calls_to_actual_reward_functions',cases=cases,pass_=True,
                physical_grasp_evidence=False,limitations=['No A-qualified physical trajectory','Contact/opposition count is not force closure','Source velocity penalties are intentionally unbounded; no hidden clipping','Action-change/opposition/lift terms not enabled before physical validation'])
