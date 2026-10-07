# Source action range semantics

源目录：/home/windsky/project/RobustDexGrasp；本轮直接读取本地源代码。

1. `algo/ppo/module.py:18-21,115-117`：Actor.sample调用Gaussian fast_sampler，MLP输出层为Linear，无tanh。
2. `env/VectorizedEnvironment.hpp:510-517,539-555`：std::normal_distribution，sample = mean + noise * std，无[-1,1]裁剪。
3. `algo/ppo/ppo.py:80-84`：Gaussian sample直接返回。
4. `env/envs/allegro_teacher/train.py:579-591`：ppo_r.act结果转换float32后直接env.step。
5. `env/RaisimGymVecEnvOther.py:115-116`及`env/VectorizedEnvironment.hpp:243,414`：动作直接转交Environment.step。
6. `env/envs/allegro_teacher/Environment.hpp:602-613,640`：target = action * actionStd_r_ + actionMean_r_，然后joint-limit clamp；actionMean_r_更新为实际q。
7. `algo/ppo/module.py:129-132`及Teacher `train.py:697-699`：完整PPO更新之后std=max(std,0.2)。

结论：源实现不存在[-1,1] policy action clipping。clip_actions=None保留此范围语义；clip_actions=1.0是工程A/B候选，不是source parity。源延迟与当前逐子步控制差异不在本轮修改范围内。

评估保持原固定UR5抬升轨迹：策略抓取阶段应用candidate clipping；抬升阶段的手部repeat动作先应用同样clipping，随后禁用wrapper clipping供脚本UR5轨迹使用。A/B hand模式保持。
