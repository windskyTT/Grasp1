"""优化12训练采样观测：不修改动作、奖励、reset或成功判定。"""
import json
from pathlib import Path
import subprocess
import time

import torch
from isaaclab.utils.math import quat_apply
from Grasp1.tasks.manager_based.grasp1.grasp1_env_cfg import SUPPORT_HEIGHT


def install_physics_monitor(real, log_dir):
    """训练/播放共用每120Hz子步真实状态检查；越限现场保留且立即停止。"""
    original_update = real.scene.update
    term = real.action_manager.get_term('teacher')
    previous = real.scene['robot'].data.joint_pos[:,term._joint_ids].clone()
    def checked_update(dt):
        original_update(dt)
        if dt != real.physics_dt:
            return
        robot = real.scene['robot']
        q, qd = robot.data.joint_pos[:,term._joint_ids], robot.data.joint_vel[:,term._joint_ids]
        limits = robot.data.joint_pos_limits[:,term._joint_ids]
        violation = torch.maximum(limits[...,0]-q,q-limits[...,1]).clamp_min(0)
        bad = (violation>.005).any(1) | (qd[:,6:].abs()>14).any(1) | ~torch.isfinite(q).all(1) | ~torch.isfinite(qd).all(1)
        if bool(bad.any()):
            from Grasp1.tasks.manager_based.grasp1.mdp.full_grasp_metrics import contact_forces
            force, table, support, normal = contact_forces(real)
            out = Path(log_dir); out.mkdir(parents=True,exist_ok=True)
            torch.save(dict(q=q.cpu(),qd=qd.cpu(),delta_qd=((q-previous)/dt).cpu(),target=term.target.cpu(),
                violation=violation.cpu(),limits=limits.cpu(),bad_envs=bad.nonzero().flatten().cpu(),
                physics_step=real._sim_step_counter,seed=real.cfg.seed,joint_names=term._joint_names,
                raw_actions=term.raw_actions.cpu(),projection_fraction=getattr(term,'projection_fraction',None),
                object_pose=real.scene['object'].data.body_link_pose_w.cpu(),robot_pose=robot.data.body_link_pose_w.cpu(),
                tip_object_N=force.cpu(),tip_table_N=table.cpu(),object_support_N=support.cpu(),normal=normal.cpu()),
                out/'physics_failure.pt')
            raise RuntimeError('FullGrasp真实物理越限/速度异常；现场已保存，停止运行。')
        previous.copy_(q)
    real.scene.update = checked_update


class TrainingDiagnostics:
    def __init__(self, env, runner, log_dir):
        self.env, self.runner = env, runner
        self.out = Path(log_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows, self.samples, self.fractions, self.phases = [], [], [], []
        self.grad_norms = []
        self.started = time.perf_counter()
        self.completed = runner.current_learning_iteration
        self.episode_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.counts = dict(attempts=0, successes=0, unsafe=0, reset_failed=0, dropped=0, timeouts=0)
        self.max_lift = self.max_hold = 0.
        self.original_step = env.step
        self.original_update = runner.alg.update
        self.original_optimizer_step = runner.alg.optimizer.step
        env.step = self.step
        runner.alg.update = self.update
        runner.alg.optimizer.step = self.optimizer_step
        real = env.unwrapped
        reset = real._reset_idx
        def stop_before_unsafe_reset(env_ids):
            unsafe_ids = env_ids[real.termination_manager.get_term('unsafe')[env_ids]]
            if unsafe_ids.numel():
                # 在自动reset擦掉现场前保留真实控制步危险原因；训练在首个unsafe停止。
                s = real._full_grasp_state.read()
                snapshot = {key:value.detach().cpu().clone() for key,value in s.items()}
                snapshot.update(env_ids=unsafe_ids.cpu(),episode_steps=real.episode_length_buf.cpu(),
                    hold=real._full_grasp_state.hold.cpu(),target=real.action_manager.get_term('teacher').target.cpu(),
                    env_origins=real.scene.env_origins.cpu(),completed_updates=self.completed,
                    object_velocity=real.scene['object'].data.body_link_vel_w.cpu())
                torch.save(snapshot,self.out/'policy_safety_failure.pt')
                raise RuntimeError('Policy unsafe termination; captured before automatic reset, PPO stopped.')
            reset(env_ids)
        real._reset_idx = stop_before_unsafe_reset
        term = env.unwrapped.action_manager.get_term('teacher')
        process = term.process_actions
        def capture_targets(actions):
            previous = term.target.clone()
            process(actions)
            limits = env.unwrapped.scene['robot'].data.joint_pos_limits[:,term._joint_ids]
            self.executed = dict(scaled=term.processed_actions.clone(),actual_target=term.target.clone(),
                preprojection=(previous+term.processed_actions).clamp(limits[...,0],limits[...,1]))
        term.process_actions = capture_targets

    def step(self, actions):
        real = self.env.unwrapped
        term, memory = real.action_manager.get_term('teacher'), real._full_grasp_state
        previous = term.target.clone()
        output = self.original_step(actions)
        self.episode_steps += 1
        dones = output[2].bool()
        s = memory.read()
        phase = torch.where(dones, memory.last_phase, memory.phase)
        self.fractions.append(term.projection_fraction.detach().clone())
        self.phases.append(phase.clone())
        # 每个控制步保留前4个环境的全部22D，分位数使用所有环境。
        sample = dict(raw_policy=actions[:4], clipped=actions[:4].clamp(-1, 1),
            scaled=term.processed_actions[:4], target_before=previous[:4], actual_target=term.target[:4],
            q=s['q'][:4], qd=s['qd'][:4], phase=phase[:4], projection=term.projection_fraction[:4],
            done=dones[:4], tcp=s['palm'][:4,:3]+quat_apply(s['palm'][:4,3:7],
                s['palm'].new_tensor(real.cfg.tcp_offset_h).expand(min(4, real.num_envs), -1)),
            top=s['top'][:4], tip_clearance=s['tip_bounds'][:4,:,0]-real.scene.env_origins[:4,None,2]-SUPPORT_HEIGHT,
            contacts=s['force'][:4], support=s['support'][:4], tracking_error=term.target[:4]-s['q'][:4])
        sample.update({key:value[:4] for key,value in self.executed.items()})
        limits = real.scene['robot'].data.joint_pos_limits[:4, term._joint_ids]
        sample['target_at_limit'] = ((term.target[:4]-limits[...,0]).abs()<1e-5) | ((term.target[:4]-limits[...,1]).abs()<1e-5)
        self.samples.append({key:value.detach().cpu().clone() for key,value in sample.items()})
        lift = torch.where(dones, memory.last_lift, s['lift'])
        hold = torch.where(dones, memory.last_hold, memory.hold)
        self.max_lift = max(self.max_lift, float(lift.max()))
        self.max_hold = max(self.max_hold, float(hold.max()))
        for index in dones.nonzero(as_tuple=True)[0].tolist():
            bits = int(memory.last_failure[index])
            row = dict(controller='policy_sampled', env=index, update=self.completed,
                success=bool(memory.last_success[index]), failure_bits=bits,
                elapsed_s=float(self.episode_steps[index])*real.step_dt,
                lift_m=float(memory.last_lift[index]), hold_s=float(memory.last_hold[index]))
            with (self.out/'training_episodes.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            self.counts['attempts'] += 1
            self.counts['successes'] += row['success']
            for key, mask in (('unsafe',1),('reset_failed',2),('dropped',4),('timeouts',8)):
                self.counts[key] += bool(bits & mask)
        self.episode_steps[dones] = 0
        return output

    def optimizer_step(self, *args, **kwargs):
        self.grad_norms.append(float(torch.stack([p.grad.detach().square().sum() for p in
            self.runner.alg.policy.parameters() if p.grad is not None]).sum().sqrt()))
        return self.original_optimizer_step(*args, **kwargs)

    def update(self):
        policy, storage = self.runner.alg.policy, self.runner.alg.storage
        before = torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
        result = self.original_update()
        self.completed += 1
        with torch.no_grad():
            mean = policy.act_inference(storage.observations.flatten(0,1))
            old_mean, old_std = storage.mu.flatten(0,1), storage.sigma.flatten(0,1)
            std = policy.std.expand_as(mean)
            kl = (torch.log(std/old_std)+(old_std.square()+(old_mean-mean).square())/(2*std.square())-.5).sum(-1).mean()
            after = torch.cat([p.detach().flatten() for p in policy.parameters()])
            if not bool(torch.isfinite(after).all()):
                raise RuntimeError('PPO parameters became nonfinite')
            delta = float((after-before).norm())
        fractions, phases = torch.stack(self.fractions), torch.stack(self.phases)
        by_phase = {}
        for i, name in enumerate(('Approach','Close','Lift','Hold')):
            values = fractions[phases == i]
            if values.numel():
                by_phase[name] = dict(count=values.numel(), mean=float(values.mean()), min=float(values.min()),
                    p1=float(torch.quantile(values,.01)),p10=float(torch.quantile(values,.1)),
                    p50=float(values.median()),below_01=float((values<.1).float().mean()),
                    below_05=float((values<.5).float().mean()))
        sample = {key:torch.stack([s[key] for s in self.samples]) for key in self.samples[0]}
        torch.save(sample,self.out/f'actions_update_{self.completed:05d}.pt')
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'], text=True)
        used, total = map(int,gpu.strip().split(','))
        row = dict(completed_updates=self.completed, losses=result, learning_rate=self.runner.alg.learning_rate,
            actor_std=policy.std.detach().cpu().tolist(), rollout_KL_old_new=float(kl),
            parameter_delta_norm=delta, gradient_norm_after_clip_max=max(self.grad_norms),
            projection_by_phase=by_phase, sampled_episodes=dict(self.counts),
            max_lift_m=self.max_lift, max_hold_s=self.max_hold,
            action_clip_fraction=float((storage.actions.abs()>1).float().mean()),
            target_tracking_abs_mean=float(sample['tracking_error'].abs().mean()),
            target_limit_fraction=float(sample['target_at_limit'].float().mean()),
            reset_valid_fraction=float(self.env.unwrapped._full_grasp_state.valid_reset.float().mean()),
            driver_memory_MiB=used, device_memory_MiB=total,
            torch_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
            elapsed_s=time.perf_counter()-self.started,
            transitions_this_run=(len(self.rows)+1)*storage.actions.shape[0]*self.env.num_envs)
        self.rows.append(row)
        with (self.out/'learning_diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        self.samples.clear(); self.fractions.clear(); self.phases.clear(); self.grad_norms.clear()
        return result
