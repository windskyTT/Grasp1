# 物理诊断结论：Gate A失败，根因与修复尚未确立

本轮保留正式场景、PD、solver、armature、碰撞、动作尺度和Reward默认值。诊断分支中的移动物体/桌面和参数覆写只在各自进程生效。不存在已验证并采用的物理修复；`physics_gate.json`为`fail`，四候选均`global_block`。

## 1. 当前最强失败证据

修正B计量后，四候选各以seed1、同一16物体排序、zero/random/fixed三种模式运行12秒，即每模式1440个120Hz物理帧。D2/T10随机动作在第776帧、6.46667秒、env7的`wood_block_oriented`上出现：

| 字段 | 实测 |
|---|---:|
| 关节 | joint_15_0 |
| PhysX reported qvel | 120.8145599 rad/s |
| 位置差分速度 | -0.2516055 rad/s |
| 实际q | -0.17325361 rad |
| cached target | -0.16199999 rad |
| 此帧越下限幅度/target−q | 0.01125363 rad |
| raw/processed action | -0.42320821 / -0.00158703 |
| implicit PD近似力矩 | -0.7 N·m |
| 此env传感器最大法向/摩擦模 | 53.23922 / 36.26887 N |

第一次timeout/reset在第1200帧，故第776帧异常发生在原trial内部，不能归为自动reset后的污染。120.81是配置7rad/s参考的17.26倍。7rad/s和历史100rad/s均不单独定义安全；actual q越限和速度/差分不一致须一起核对。差分速度与PhysX端点速度可能因约束积分不同，当前证据不支持直接修改或替换qvel。

证据：`peak_case.json`、`trace_peak_E2_random.pt`、`peak_window_E2_random.csv`（726–826帧，共101帧），以及`../C/E2_screen/trace_random.pt`。目标在同一控制周期内不变，目标越限为0；这只能排除“子步重新计算目标/目标本身越限”，不能排除接触、驱动和articulation约束耦合。

## 2. 原优化6事件的复现边界

原D4/T10 zero记录在第892帧达到112.9021759rad/s，env7同一物体与关节，q=-0.1764402、target=-0.162、差分=-1.1850517rad/s、contact约53.91N。

原`optimization6/check.py`重跑：zero峰11.1426096rad/s，未精确重现112.9；random峰41.6926613rad/s重现了原脚本本次对应随机结果。旧/新zero的q在前159帧一致，第160帧首次分叉；峰差不应称为修复。新的详细collector D4 baseline zero为23.9431229rad/s；collector与旧脚本的采样量不同，也不作性能或逐帧完全等价结论。原D2随机76.9241为历史参考，本轮D2/T10 random出现120.8146，当前并未解除同类失控。

证据：`legacy_comparison.json`、`legacy_repro/`及原`optimization6/diagnosis/zero_physics_trace.pt`。原目录未被覆盖。当前脚本`minimal_repro.py`自动记录目标子步一致性与适用模式的旧峰偏差；历史早期结果没有后来增加的字段，以原文件为准。

## 3. 单环境缩减及接触隔离

从16env初始快照复制env7的实际q/qvel、对象状态和reset数据到1env；世界位置按env origin差换算。回放旧zero的cached target，通过正式相对动作接口生成action。每组D4/T10运行10秒、seed1，只有表列变量变化：

| 条件（同一回放） | hand qvel峰rad/s | actual q越限峰rad |
|---|---:|---:|
| 物体/桌面保留 | 8.619441 | 0.00956531 |
| 移开物体，桌面保留 | 30.957081 | 0.08780555 |
| 物体保留，移开桌面 | 0.282938 | 0.000001863 |
| 二者都移开 | 0.282938 | 0.000001863 |

复核`table_filtered_unfiltered/`与`collider_audit_replay/`：移开物体、桌面保留的回放仍精确达到30.957081rad/s，所有已监测传感器的filtered及net法向峰均0，目标子步一致、目标越限0，第1197帧尖峰早于第1200帧timeout。**桌面存在与症状相关，但尚未量化确定造成该尖峰的接触对。** 未覆盖的robot base/shape接触、求解器约束冲量等仍未观测；不能把“移开桌面有效”直接写成“手指撞桌就是已证实根因”。

独立zero/fixed及部分random实验也已保存；仅上述回放在移开桌面/物体分支确认监测力为0。`single_no_contacts_D4`的random记录存在3838N传感器峰，不能因目录名宣称其严格无接触。不同env数的Gaussian抽样不同，1env random不能当作16env env7的同动作复现。

## 4. 限位与单变量A/B

无桌面/物体接触、joint15靠近下限的固定目标、向内/外0.01rad请求（最终clamp）分别得到0.029993/0.069074/0.030017rad/s；向外实际微小越限约2.056e-6rad。外力测试在2.0–2.1秒对link15施加2.5N、0.0423m力臂，峰0.109251rad/s、越限2.116e-6rad。此fixture是数值参考，不是新安全容差；正式候选的0.004–0.278rad越限远超这个量级。

| 单变量 | 关键结果 | 决定 |
|---|---|---|
| robot velocity iterations 1→4 | D4 zero28.899/random51.976，仍越限 | 不采用 |
| object velocity iterations 0→4 | 与robot迭代4结果相同 | 不采用；island耦合仅推测 |
| hand armature=0.0006 | D2 random8.616；D4 random5.922，仍约0.0015rad越限 | 不采用 |
| hand armature=0.0012，35物体×12秒 | D2 zero/random4.667/5.366；D4 5.470/5.851，越限最高0.001445/0.001150rad | 改善速度，未证实模型参数与全局修复 |
| 清除旧USD jointFriction | D4 random75.066，回放95.784，位置越限恶化 | 不采用 |
| TGS→PGS | D2 random约7.000，actual q越限1.079014rad | 不采用 |
| UR5六joint armature=1.0（源rotor候选） | D2 random29.754、越限0.407200；D4 24.681、越限0.323630 | 源模型差异确认，未解决尖峰，不采用 |

各组加载后的参数、质量/惯量、USD joint属性、原始完整物理帧、reset时刻及mode结果在对应目录，统一索引为`experiments.json`、`physics_ab.csv`、`peak_cases.csv`。没有联合调参，没有提高门槛、硬裁qvel或更改正式碰撞场景。

## 5. 候选Gate与剩余阻断

| 候选 | zero/random/fixed最高速度rad/s | 任一mode最大实际越限rad | Gate |
|---|---|---:|---|
| E1 D2/4s | 8.152 / 19.378 / 0.036 | 0.114575 | fail |
| E2 D2/10s | 37.899 / 120.815 / 0.036 | 0.277936 | fail |
| E3 D4/4s | 23.943 / 7.412 / 0.036 | 0.004486 | fail |
| E4 D4/10s | 23.943 / 41.693 / 0.036 | 0.173090 | fail |

逐mode总接触峰、越限env×joint×物理帧计数/累计秒和速度差在各`C/E*_screen/result.json`。累计秒为环境与关节求和，不能解释为一个trial连续越限时长；整段trace可以复算连续区间。四组未以统一新安全阈值获得pass；当前严重异常机制未解除，全部正式训练阻断。多seed安全证据尚未完成，现有seed1失败已足以冻结。

唯一关键下一步是从现有固定快照与回放出发，量化定位未观测的碰撞对/约束、驱动与关节限位耦合，找到有因果证据的最小模型修复，再按相同初始状态同时复验D4 zero与D2 random及四候选。当前不输出虚构的修复后曲线或安全checkpoint。
