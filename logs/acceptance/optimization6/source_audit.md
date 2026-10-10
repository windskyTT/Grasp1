# 原始依据与时间语义核对（2026-10-09）

原始本地仓库 `/home/windsky/project/RobustDexGrasp`，HEAD `b00a6cdb162d35a9e6d4feccf6526b5e6b4b8742`。读取 `Environment.hpp`、`train.py`、`quantitative_eval.py`、`cfgs/cfg_reg.yaml`、`env/VectorizedEnvironment.hpp`，及 Grasp1 优化1–5文档/执行报告。

- [论文v3](https://arxiv.org/html/2504.05287v3)，4.1评价定义：抬升0.1m，并稳定至少5秒。公开源码末帧高度指标与此定义分开报告。
- [官方仓库](https://github.com/zdchan/RobustDexGrasp)，原始 Teacher 位于 `raisimGymTorch/raisimGymTorch/env/envs/allegro_teacher/`。
- `cfg_reg.yaml:82–88`：simulation_dt=0.01、control_dt=0.2；grasp_steps=70。Grasp1本轮固定120Hz物理，仅比较60/30Hz策略，不复原源物理或5Hz控制。
- `Environment.hpp:602–613`：一个策略周期内先生成 `action * actionStd + actionMean`，裁剪一次。`623–636`在物理循环复用target；源随机延迟可能第一子步使用prev target。本轮禁用此延迟，明确这一差异。
- `Environment.hpp:427`：速度PD目标为零。Isaac articulation初始化的joint velocity target也是零，动作应用显式写0，与源语义一致。
- `quantitative_eval.py:150–152`：70+30=100抓取步、100抬升步；`cfg_reg.yaml`原0.2秒周期，故20+20秒。
- `quantitative_eval.py:247,493–506`：theta0=(0,-1.57,1.57,0,1.57,-1.57)，抬升时复用最后抓取动作的手部；`Environment.hpp:606–608`手臂对lift-start插值80步，即16秒。Grasp1用实际step_dt将16秒换成960/480步，以 `(arm_target-q)/arm_scale`产生同一控制周期的手臂目标。
- 默认源时长协议=`source_timing_20_20_ramp16`。`paper_stability_20_20_ramp14`仍20+20秒，14秒Ramp，最终5秒完整高度/接触/相对位姿窗口；14秒不是源插值时长。
- 源PD、延迟、物理步长和动作scale等未完整复原，不能宣称算法/物理完全等价。
- `Environment.hpp:1065–1084`：关键点低于桌面时terminalReward=-10；非法观测没有此罚。`VectorizedEnvironment.hpp:423–429`终止后只加一次terminalReward。本轮RewardManager在termination后reward前的顺序使专用奖励项可读精确条件；额外扣分不按秒弱化。其余17项仍source_coeff/0.2，经RewardManager乘真实step_dt。
- `train.py`的 `reward_r.clip(min=-2.0)`未接收返回值，未原地裁剪；不新增总reward clipping。

## 接触冲量审计

`Environment.hpp:744–811`每次updateObservation先清零，再读取完成最后物理积分后 `getContacts()` 的每个接触点冲量。按body在世界系向量求和、取模；并非把整个0.2秒所有物理子步冲量再求和。源 `impulses_r_af_xy`是世界XY分量的模，源 `impulses_r_af_z`是世界Z分量；它们不严格等同接触面切向/法向分解。

IsaacLab2.3.2 `contact_sensor.py:373,385,409–410`在查询法向/摩擦力时统一传入 `_sim_physics_dt`，所以返回N。Grasp1 `mdp/runtime.py`在过滤shape维先求向量和，再取模乘 `env.physics_dt`，得到N·s，读控制周期末帧；不再乘decimation，无D4二次累积。Sensor update_period=0，每物理步timestamp推进；懒读取刷新最新帧。诊断每物理步取峰值，reward继续读末帧。

现有迁移近似的边界：affordance_impulse_reward采用friction vector模，而源采用世界XY总冲量；push采用法向模，源采用有符号世界Z；源还按每接触点冲量0.001预过滤。引擎接触、shape聚合和阈值不严格等价。本轮只核对单位/刷新/聚合、记录差异，不改reward。

传感器在同一机器人body上分别过滤top/table/arm对象；诊断bottom另有独立过滤。不同过滤sensor的未过滤net_force可能重复同一body，因此诊断取max而不是求和，避免重复计数。2+/3+是不同接触body数；四指分组和拇指参与另算，不把它们直接称为真实力闭合。

## 保留优化与独立课题

保留优化1缓存与partial invalidation、优化3目标限位/物体零PD、优化4实际target−q、优化5Gaussian无统一[-1,1]clip、std下界0.2、fixed 5e-4、128×128、rollout32、4 epoch/4 mini-batch。gamma/lambda随控制周期按预定义公式换算。armature=0.1、self-collision=False、原PD和摩擦不变；没有D4-S2。

优化5实测2000节点hand qvel≈406.544rad/s、末控制帧接触力≈87.395kN、target clip≈26.578%，critic持续数百级；不能沿用旧checkpoint当本轮从头训练结果。旧checkpoint仅可作新控制语义兼容性/行为回归。
