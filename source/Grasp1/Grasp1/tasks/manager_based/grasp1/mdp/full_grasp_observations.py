"""Teacher153之外显式添加58D自主Lift/Hold状态；旧观测不变。"""
import torch
from .observations import TeacherObservation
from .full_grasp_metrics import FullGraspState, state


class FullGraspObservation(TeacherObservation):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._full_grasp_state = FullGraspState(env)

    def __call__(self, env, support_height):
        base = super().__call__(env, support_height)
        s, memory = state(env), env._full_grasp_state
        extra = torch.cat((s['qd'], s['relpos'], s['relvel'], s['relquat'],
            s['force']*env.physics_dt, s['table']*env.physics_dt, s['support']*env.physics_dt,
            torch.nn.functional.one_hot(memory.phase, 4).float(),
            torch.stack((s['lift'], torch.full_like(s['lift'], env.cfg.lift_height), memory.hold,
                (env.max_episode_length-env.episode_length_buf)*env.step_dt, memory.valid_reset.float()), -1),
            s['drift_vector'], s['angle'][:, None]), -1)
        return torch.cat((base, extra), -1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        memory = self._env._full_grasp_state
        ids = torch.arange(self._env.num_envs, device=self._env.device) if env_ids is None else env_ids
        finished = ids[memory.finished[ids]]
        if len(finished):
            log = self._env.extras['log']
            log['FullGrasp/success'] = memory.last_success[finished].float().mean()
            log['FullGrasp/reset_valid'] = memory.last_reset_valid[finished].float().mean()
            log['FullGrasp/hold_s'] = memory.last_hold[finished].mean()
            log['FullGrasp/lift_m'] = memory.last_lift[finished].mean()
            for phase in range(4):
                log[f'FullGrasp/final_phase_{phase}'] = (memory.last_phase[finished] == phase).float().mean()
