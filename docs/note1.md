# Grasp1 最新代码审查：距离完整自主抓取还有哪些问题？

## 一、当前结论

我已确认仓库最新版本为 **V2.7—优化13**，对应版本 `0582bae30fb0`，更新于 **2026年10月16日**。

仓库：[windskyTT/Grasp1](https://github.com/windskyTT/Grasp1)

**目前项目尚未达到你的最终目标。**

最新代码已经恢复了 Teacher-v0 的基本训练链路，并按照你的要求将 Allegro 调整为：

- `stiffness = 3.0`
- `damping = 0.1`
- `effort_limit_sim = 0.7`
- `velocity_limit_sim = 7.0`
- `friction = 0.01`
- 22维动作、153维观测。
- 物理频率120 Hz，控制频率60 Hz。

但最新的 [优化13执行报告](https://github.com/windskyTT/Grasp1/blob/main/docs/优化13_执行报告.md) 明确给出了以下结果：

| 验收项目 | 最新结果 |
|---|---|
| 22关节 URDF、USD、Runtime 配置核对 | PASS |
| 1、16、128环境初始化 | PASS |
| PPO 真实优化器更新 | 完成2次 |
| 128环境零动作物理测试 | **FAIL** |
| 128环境随机动作物理测试 | **FAIL** |
| 关节越限、异常接触 | **仍存在** |
| 长时间 PPO 训练 | 未执行 |
| 与当前153维观测匹配的 checkpoint 验证 | 未完成 |
| GUI 纯策略完整抓取 | 未执行 |
| 已证明的自主抓取成功率 | **未知，不能宣称成功** |

其中，128环境测试发现 `joint_13_0` 实际位置越限约 **0.003003 rad**；随机动作测试还记录了约 **533.81 N** 的机械臂过滤接触法向力峰值。

这些数据不等于最终训练必然失败，但足以说明：**当前最大障碍已经不只是 PPO 参数，而是物理稳定性、训练目标与策略验证尚未形成完整闭环。**

下面按实际代码进一步区分哪些问题已经确认、哪些属于高风险设计问题。

## 二、目前存在的主要问题

### 问题1：128环境下物理稳定性没有通过（P0，已确认）

这是目前最需要解决的问题。

根据最新报告：

| 模式 | 最大实际关节速度 | 最大位置越限 | 最大法向接触力 |
|---|---:|---:|---:|
| 16环境，零动作 | 0.914 rad/s | 0.000029 rad | 14.21 N |
| 16环境，随机动作 | 7.082 rad/s | 0.000879 rad | 156.76 N |
| 128环境，固定动作 | **15.845 rad/s** | **0.004845 rad** | 393.85 N |
| 128环境，随机动作 | 7.676 rad/s | 0.004067 rad | **533.81 N** |

需要强调，表中的接触力峰值包含不同碰撞对象，不能全部解释为指尖抓取力。

最新报告定位到三个值得重点检查的物体：

| 物体 | 发现的问题 |
|---|---|
| `solder_iron_head` | 零动作也出现 Allegro `joint_13_0` 越限 |
| `off_water_body` | 随机动作触发指尖与桌面异常接触 |
| `cracker_box_oriented` | 随机动作触发 UR5 前臂接触 |

其中，`solder_iron_head` 最值得优先排查，因为它**不需要策略产生动作，就已经能够触发当前物理验收条件**。

需要检查接触几何、初始穿透、关节驱动、被动物体关节以及 PhysX 求解响应。

但不能仅凭目前的峰值就认定 `3.0/0.1` 错误，也不应直接恢复 `600/20`。

**建议：先单独复现 `solder_iron_head` 零动作越限，再定位具体物理原因。**

---

### 问题2：Reset 只验证 IK 可行性，没有完整验证初始碰撞（P0，源码确认）

文件：

[mdp/events.py](https://github.com/windskyTT/Grasp1/blob/main/source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/events.py)

当前 `reset_teacher()` 的核心有效性判断为：

```python
valid = (
    feasible
    & torch.isfinite(q).all(1)
    & ((q >= limits[..., 0]) & (q <= limits[..., 1])).all(1)
)
```

这段代码能够检查：

- IK 是否有解；
- 关节位置是否有限；
- 关节位置是否满足硬限位。

**但是，没有将机器人与物体、桌面之间的实际碰撞检查纳入预抓取状态的验收条件。**

这意味着可能发生以下情况：

```text
IK 计算成功
    ↓
22关节位置合法
    ↓
机器人被直接放到预抓取姿态
    ↓
手指可能已经接触或穿入物体/桌面
    ↓
PhysX 开始处理接触
    ↓
出现异常力、速度或关节越限
```

这是一种有源码依据的潜在根因，但尚不能证明全部异常都由它引起。

**建议修改：**

在现有 `reset_teacher()` 中完善预抓取候选状态验证和碰撞处理。

重点检查 UR5 前臂、Allegro 拇指、掌部和四个指尖。

需要保留当前 URDF 关节硬限位，不新增策略动作裁剪器。

---

### 问题3：训练过程中首次异常会中断整个 PPO（P0，已确认）

文件：

[scripts/rsl_rl/train.py](https://github.com/windskyTT/Grasp1/blob/main/scripts/rsl_rl/train.py)

当前代码：

```python
if raw._teacher_grasp.safety_failure is not None:
    raise RuntimeError(
        "Teacher unsafe physics; stopped PPO before the next optimizer update."
    )
```

而 `safety_failure` 一旦记录首次异常，就会持续保留，直到训练进程结束。

因此：

```text
128环境并行训练
    ↓
其中一个环境发生异常
    ↓
记录 safety_failure
    ↓
整个 PPO 停止
```

这个机制适合在物理调试阶段阻止明显的数值失稳。

但作为正式训练机制，存在两个问题。

首先，当前异常条件混合了关节越限、速度异常、物体速度异常和接触力阈值。

其次，接触力阈值目前仍然使用：

```python
arm_contact_limit_N = 5.0
hand_table_contact_limit_N = 5.0
```

**5 N 是现有工程终止阈值，不是已经标定的物理安全上限。**

如果所有超过5 N的接触都被视为需要全局停止的异常，探索中的部分正常或可恢复碰撞，也可能阻止整个训练。

建议将异常分为两个层级：

| 类型 | 建议处理 |
|---|---|
| NaN、Inf、严重物理数值失稳 | 停止训练并保存现场 |
| 已定义的不允许接触或可恢复碰撞 | 终止对应环境，记录惩罚和原因 |
| 尚未标定的接触力峰值 | 保留诊断，不直接认定为物理硬上限 |

**不是取消安全检测，而是避免把所有接触事件都变成全局停训条件。**

---

### 问题4：Allegro 的低 PD 与当前动作尺度是否匹配，尚未验证（P0，高风险）

文件：

[ur5_allegro_cfg.py](https://github.com/windskyTT/Grasp1/blob/main/source/Grasp1/Grasp1/robots/ur5_allegro_cfg.py)

[teacher_env_cfg.py](https://github.com/windskyTT/Grasp1/blob/main/source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/teacher_env_cfg.py)

目前：

```python
_HAND_STIFFNESS = 3.0
_HAND_DAMPING = 0.1
```

对应动作尺度：

```python
ARM_ACTION_SPEED = 0.075
HAND_ACTION_SPEED = 0.225
```

由于控制周期为 `1/60 s`，当策略输出为1时，Allegro 的目标位置增量为：

\[
\Delta q=\frac{0.225}{60}=0.00375\ \mathrm{rad}
\]

在忽略阻尼影响的简化 PD 模型中：

\[
\tau=K_p\Delta q=3\times0.00375
\]

即约：

\[
\tau=0.01125\ \mathrm{N\cdot m}
\]

而原始 Allegro 力矩上限为：

\[
0.7\ \mathrm{N\cdot m}
\]

这并不意味着实际关节只能产生0.01125 N·m，因为目标跟踪误差可能持续累积，阻尼和接触也会影响驱动力矩。

但它反映出一个重要问题：

**当前控制增量与较低 PD 增益的组合，是否足以让手指迅速闭合并保持物体，还没有得到验证。**

而且 `ARM_ACTION_SPEED` 和 `HAND_ACTION_SPEED` 在这里主要用于生成动作缩放量，并不是实际关节速度的硬上限。

建议检查实际手指闭合时间、目标跟踪误差、接触时的关节力矩响应，以及物体是否会因夹持力不足而滑落。

按照你之前的要求，**不建议增加额外的 PPO 动作幅值裁剪、目标速度裁剪或其他控制层限制**。

应该先判断现有相对关节控制和 PD 参数是否能够完成任务。

---

### 问题5：奖励函数虽然包含完整抓取，但学习信号可能过于稀疏（P0，高风险）

文件：

[mdp/rewards.py](https://github.com/windskyTT/Grasp1/blob/main/source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/rewards.py)

当前主要奖励：

| 奖励 | 权重 | 用途 |
|---|---:|---|
| `approach` | 2.0 | 接近物体 |
| `direction` | 0.5 | 调整手掌方向 |
| `contact` | 1.0 | 指尖接触 |
| `opposing` | 2.0 | 形成相对接触 |
| `stable_grasp` | 2.0 | 稳定夹持 |
| `lift` | 4.0 | 抬升物体 |
| `hold` | 3.0 | 保持物体 |
| `success_bonus` | 10.0 | 完整成功 |
| `unsafe_penalty` | -10.0 | 异常惩罚 |

整体方向是正确的，比优化前单纯模仿原始 Teacher 接触奖励，更接近完整抓取任务。

但是当前 `lift` 奖励存在明显的学习依赖：

```python
return (
    s.grip.float()
    * (s.lift / env.cfg.success_lift_m).clamp(0, 1)
)
```

这意味着：

**只有满足 `grip` 判据，抬升奖励才会出现。**

同时，`hold` 必须满足稳定抓取、离桌、未受桌面支撑、抬升高度等条件。

如果策略始终无法达到 `grip=True`，后续奖励就难以提供有效学习信号。

这一风险已经得到初步验证。

优化13的两次 PPO 更新中：

```text
contact  = 0
opposing = 0
lift     = 0
hold     = 0
success  = 0
```

两次更新样本量太少，不能据此判断策略无法收敛。但它说明当前训练尚未证明能够形成任何有效抓取行为。

**建议先验证奖励的可达性，而不是立即提高 PPO 迭代次数。**

例如，使用不参与正式训练的受控物理诊断，检查真实形成的指尖接触、夹持、抬升和保持，是否分别产生预期奖励。

如果物理上已经成功抓取，但 `grip` 或 `stable_grasp` 仍为零，就必须修正奖励判据。

---

### 问题6：稳定抓取判据可能过于严格（P1，需要验证）

文件：

[mdp/observations.py](https://github.com/windskyTT/Grasp1/blob/main/source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/observations.py)

当前 `grip` 不只要求接触，还要求拇指与其他手指的空间方向、接触力方向符合一定条件。

`stable` 又增加了：

| 条件 | 当前数值 |
|---|---:|
| 手掌—物体相对位置漂移 | < 0.02 m |
| 相对姿态变化 | < 0.35 rad |
| 相对线速度 | < 0.10 m/s |
| 相对角速度 | < 1.0 rad/s |
| 物体线速度 | < 0.15 m/s |
| 物体角速度 | < 1.0 rad/s |

完整成功还要求：

```python
success_lift_m = 0.10
success_hold_s = 1.0
```

即抬升至少10 cm，并持续满足保持条件1秒。

成功标准本身是合理的工程目标。

但其中的接触方向、位置漂移和速度阈值，是否能够正确识别 Allegro 的真实抓取，尚未验证。

尤其应该区分：

- 物体被手指稳定夹持，但接触几何判定为失败。
- 物体只是被推离桌面，却被错误判定为成功。

需要把现有 `grip`、`stable`、`lift`、`hold` 与 GUI 中真实的物理抓取行为进行对照。

---

### 问题7：PPO 训练配置不足以证明可以完成完整抓取（P1）

当前配置：

[rsl_rl_ppo_cfg.py](https://github.com/windskyTT/Grasp1/blob/main/source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/agents/rsl_rl_ppo_cfg.py)

```python
num_steps_per_env = 32
max_iterations = 100

learning_rate = 3e-4
num_learning_epochs = 5
num_mini_batches = 4

init_noise_std = 0.5
```

在128个并行环境下，100次更新对应409,600个采样 transition。

这对完整22自由度、多个物体、多指接触和抬升保持任务而言，只适合作为初步训练阶段，不能当作充分的收敛预算。

此外，优化13的两次更新记录了较大的更新后 KL：

| PPO 更新 | 记录的 KL |
|---|---:|
| 第1次 | 0.1182 |
| 第2次 | 0.0613 |

配置的 `desired_kl=0.01`。

这些是自定义更新后诊断值，不能直接等同于 PPO 内部用于控制学习率的 KL 指标，但值得核对计算方法和训练稳定性。

**建议先解决物理异常与奖励可达性，再逐步增加训练更新次数，并以真实成功率判断是否继续训练。**

---

### 问题8：当前 checkpoint 与 play.py 不兼容（P0，已确认）

文件：

[scripts/rsl_rl/play.py](https://github.com/windskyTT/Grasp1/blob/main/scripts/rsl_rl/play.py)

优化13报告明确记录，现有模型：

```text
model_10.pt
```

使用的是203维观测。

而目前 Teacher-v0 使用：

```text
observation_dim = 153
action_dim = 22
```

因此，旧模型不能直接用于新版 `play.py`。

当前 `play.py` 正确拒绝了这个不兼容模型，这不是需要绕过的错误。

真正缺少的是：

**从当前153维 Teacher-v0 训练得到的、具有完整 checkpoint 契约的有效 `.pt` 模型。**

另外，优化13完成的两次 PPO 更新使用：

```text
--no_log
```

所以没有生成 checkpoint。

因此，目前还没有验证：

```text
train.py
   ↓
训练 PPO
   ↓
保存 model_x.pt
   ↓
play.py 加载
   ↓
GUI 运行 Policy
   ↓
自主夹持、抬升、保持
```

这条完整链路。

当前 `play.py` 已采用纯策略推理：

```python
actions = policy(obs)
obs, _, dones, extras = env.step(actions)
```

这个实现方向是正确的，不需要加入脚本夹持或预编排抬升动作。

但还必须用**真实新 checkpoint**验证加载、推理及 GUI 完整抓取。

---

## 三、我认为应该优先修改哪些文件？

| 优先级 | 文件 | 主要目标 |
|---|---|---|
| **P0** | `mdp/events.py` | 完善 Reset 候选姿态的碰撞验证 |
| **P0** | `mdp/observations.py` | 定位零动作关节越限、接触异常；验证抓取状态判据 |
| **P0** | `mdp/terminations.py` | 区分环境级异常终止与真正的数值失稳 |
| **P0** | `scripts/rsl_rl/train.py` | 修正过于宽泛的全局停训条件 |
| **P0** | `robots/ur5_allegro_cfg.py` | 核实低 PD、摩擦及执行器实际响应 |
| **P1** | `mdp/rewards.py` | 验证接触、夹持、抬升奖励的可达性 |
| **P1** | `mdp/actions.py` | 检查目标跟踪与手指运动能力，保持原始硬限位语义 |
| **P1** | `teacher_env_cfg.py` | 核对阈值、传感器配置及抓取成功条件 |
| **P1** | `rsl_rl_ppo_cfg.py` | 在物理稳定后确定合适的训练预算 |
| **P1** | `scripts/rsl_rl/play.py` | 验证新 checkpoint 的 GUI 纯策略回放 |

注意：这并不意味着上述文件都必须修改。应先通过诊断确定原因，再只改真正需要调整的部分。

不建议现阶段重新大幅调整机器人模型、增加新的 Task，或改变现有22维动作与153维观测结构。

---

## 四、从当前状态到最终成功，建议分四个阶段

### 第一阶段：解决物理异常

**目标：使当前机器人能够在安全、可重复的物理条件下运动和接触物体。**

优先处理：

1. `solder_iron_head` 零动作关节越限。
2. `off_water_body` 手指—桌面异常接触。
3. `cracker_box_oriented` UR5 前臂异常接触。
4. Reset 后初始碰撞的检查和处理。
5. 真正导致停训的物理异常与一般环境终止之间的区别。

验证1、16、128环境下固定动作和随机动作的物理结果。

不能通过提高关节限位、关闭必要碰撞或随意放宽阈值来掩盖问题。

### 第二阶段：验证完整抓取任务确实可实现

**目标：证明当前机器人、接触条件和成功判据允许发生完整抓取。**

可以在独立物理验收中使用受控动作作为诊断，但不得把这些动作加入正式策略训练或 `play.py`。

应验证：

| 抓取阶段 | 必须满足 |
|---|---|
| Approach | 手掌和指尖能够接近目标 |
| Contact | 指尖产生有效接触 |
| Grip | 多指能够形成稳定夹持 |
| Lift | 物体离开桌面至少10 cm |
| Hold | 物体持续保持至少1秒 |
| Success | 以上条件同时成立，并且没有不允许的物理异常 |

如果物理动作已经完成某一步，但对应奖励或状态始终不触发，说明抓取任务定义还需要修正。

### 第三阶段：正式训练 PPO

在物理与任务判据验收后，使用当前 `train.py` 正常训练并保存 checkpoint。

建议分阶段检查训练指标：

| 训练阶段 | 重点观察 |
|---|---|
| 初期 | Approach 是否改善 |
| 接触学习 | Contact 与 Good Contact 是否增加 |
| 夹持学习 | Grip 和 Stable Grasp 是否增加 |
| 抬升学习 | Lift 高度是否提高 |
| 保持学习 | Hold 时间是否增加 |
| 最终 | 真实成功率是否稳定提高 |

不能把 Reward 上升等同于完整抓取成功。

对于 RTX 3070 Ti 8GB，建议先实测128环境的显存与吞吐，再决定是否提高环境数量。

当前接触传感器数量较多，还应检查实际 PhysX 接触缓冲区占用和性能。

### 第四阶段：GUI 验收

使用 `train.py` 生成的真实 checkpoint，通过 `play.py` 启动 Isaac Sim GUI。

验收时必须：

- 直接加载对应 `.pt`；
- 由 Policy 输出全部22维动作；
- 不调用脚本夹持、IK 抬升或预编排动作；
- 使用 `new_training_set` 中的物体；
- 显示完整夹持、抬升和保持过程；
- 输出成功、抬升高度、保持时间以及异常状态。

单次成功还不代表策略已经稳定。最终应对多个物体、多次回合统计完整成功率。

---

## 五、最终判断

结合最新源码和优化13执行报告，我认为当前真正阻碍目标实现的因素，按优先级排序如下：

| 排名 | 问题 | 当前判断 |
|---|---|---|
| **1** | 128环境物理异常，包括零动作越限 | **已确认，首要阻碍** |
| **2** | Reset 缺少完整碰撞可行性验证 | **源码确认存在检查缺口** |
| **3** | 单环境异常可能直接停止全部 PPO 训练 | **已确认** |
| **4** | Allegro 低 PD 与当前动作尺度的实际抓取能力 | **未验证，高风险** |
| **5** | Grip、Stable、Lift、Hold 奖励链条的可达性 | **未验证，高风险** |
| **6** | 当前 PPO 训练量和收敛证据不足 | **已确认** |
| **7** | 缺少当前153维配置的可用 checkpoint | **已确认** |
| **8** | 新模型通过 GUI 实现完整自主抓取 | **尚未验证** |

### 最重要的结论

**我不建议现在直接开始数千次 PPO 长时间训练。**

因为当前最基础的128环境零动作物理验收还没有通过，而且训练代码会在首次全局安全异常后停止训练。

我的建议是：

**先修复 Reset 和 PhysX 接触异常，确认 Allegro 的 `3.0/0.1` 确实能够产生稳定夹持，再验证 Grip → Lift → Hold 的奖励链条，最后进行正式 PPO 训练和 GUI 回放。**

这条路线不需要新建任务、不需要更换机器人，也不需要添加额外的动作控制限制。

**审查范围说明：** 本次核对的是 GitHub 最新 `0582bae30fb0` 版本的源码和已提交的执行报告。我没有在你的本地 Isaac Sim 5.1.0 环境重新运行仿真，因此物理根因、完整训练收敛和 GUI 成功率仍需要本机实测确认。