# 优化10｜Grasp1 V2.2 训练优先：修复指尖—桌面物理异常，完成训练集自主抓取与 GUI 播放

> **执行对象：Codex；任务类型：直接修改项目代码、运行 Isaac Sim、保存证据与训练产物，不是仅撰写分析报告。**  
> **当前代码仓库：** https://github.com/windskyTT/Grasp1  
> **本次审查基线（2026-10-14）：** `main` → `04603ca58023a47175d2b1dc61dd2b5a9b75e9d2`，提交名称 `V2.2—优化9`。  
> **工作目录：** `/home/windsky/project/Grasp1`，运行前激活 Conda 环境 `grasp`。  
> **关键工具链：** Isaac Lab 2.3.2、Isaac Sim 5.1.0 预编译安装、RSL-RL 3.1.2、NVIDIA driver 580、RTX 3070 Ti Laptop 8 GB。  
> **任务主目标：** 仅使用 `new_training_set`，先在一个容易抓的物体上训练出**策略自主完成接近→闭合→抬升→保持**，再扩大训练集；能够通过现有 `scripts/rsl_rl/play.py` 加载真实训练生成的 `.pt`，在 Isaac Sim GUI 直接观察**策略自身**完成抓取。`evaluate.py`、ShapeNet 验证集和论文复现**不属于本轮主线**。

---

## 0. 本轮审查的已确认事实：先消除误判

审查必须结合**本地最新工作区**重新确认；下面证据来自远程已提交的 V2.2，不代表未提交的本地更改。

1. 从 V2.1 `eea2e97d0c97797119b802760e4dcc9fccad57f8` 到 V2.2 `04603ca...`，**没有任何 `source/` 正式Python代码、`scripts/rsl_rl/train.py` 或 `scripts/rsl_rl/play.py` 的已提交变更**；主要增加 `logs/acceptance/optimization9` 中的物理诊断脚本、轨迹、GUI基线及报告。
2. 优化9已实际进行 **13次回放 / 15,600 个 120Hz 物理步**；**Gate A=fail**。正式物理修复、脚本抓取、完整RL任务、PPO短训和8–10小时训练都没有完成。
3. `logs/acceptance/optimization9/summary.json`：`ppo_updates=0`、`scripted_attempts=0`、`policy_attempts=0`，新训练checkpoint为空，成功率为 `null`（**不是 0%**）。
4. 无物体+桌面场景：Allegro峰值 `30.957081 rad/s`，最大实际关节越限 `0.08780555 rad`；有物体+桌面：`9.590271 rad/s` 和 `0.01650780 rad`。D2 历史随机目标亦未通过门槛。
5. 优化9的**新因果发现**：`link_15_0_tip`（独立的拇指tip刚体）与 `/Table` 的接触力峰值 `69.092651 N`，第470个物理步首次观测到接触；旧Teacher接触槽没有报告此接触。只隔离该 tip/Table 碰撞对，峰值下降至`0.282938 rad/s`、越限降至`1.86e−6 rad`，但该隔离**仅是诊断负对照，不能作为正式抓取配置**。
6. 增加到 **255 position / 32 velocity iterations** 虽将最大越限降至 `0.00066465 rad`，仍报告 `18.434395 rad/s` 与该帧差分速度 `0.559625 rad/s`；实际机制和性能代价没有解释清楚，不能直接判定该候选“已安全”。
7. V2.2现有 `GraspTrain-v1` 仍只继承 Teacher 的grasp-only逻辑：4秒episode、D4（30Hz）、22维相对关节动作、153维观测、`source_physics_aligned`奖励，以及 `C1_pending_physics` 课程占位；**没有自主Lift/Hold任务设计**。
8. `play.py` 能在非headless GUI真实播放**历史Teacher**的 `model_999.pt`；该回放使用了`--allow_mismatched_checkpoint`和2秒episode覆盖，仅为接口证明，**不是优化9的抓取成果**。

**审查证据文件：**

- `docs/优化9_执行报告.md`
- `logs/acceptance/optimization9/summary.json`、`gates.json`、`A/diagnosis.md`、`A/physics_gate.json`、`A/before_after_physics.csv`
- `logs/acceptance/optimization9/A/all_body_contact/contact_constraint_summary.json`
- `logs/acceptance/optimization9/A/tip_table_pair_isolation/result.json`
- `logs/acceptance/optimization9/0/gui_baseline_result.json`
- `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/teacher_env_cfg.py`
- `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/grasp_train_env_cfg.py`
- `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/{events,observations,rewards,rewards_grasp_v1,terminations,runtime,keypoints}.py`
- `scripts/rsl_rl/{train,play,checkpoint_contract}.py`

> 证据边界：上面是真实日志支持的事实；具体是碰撞几何、物理材料、关节摩擦、限位、隐式PD还是TGS耦合中哪一项造成症状，**尚未得到唯一根因**。尤其不能把GPU报告`qd`与`Δq/Δt`不一致本身等同于“速度造假”或“已证明哪一项错误”；它们是需要并列分析的不同观测通道。新修复应优先消除**实际关节越限、非物理弹飞和不合理冲量**。

---

## 1. 本次需要解决的问题清单（严格排序）

| 编号 | 优先级 | 已知问题 / 风险 | 代码或证据定位 | 应完成的修复及验证 |
|---|---|---|---|---|
| P0-01 | 最高 | thumb tip/Table碰撞触发严重关节越限；没有可用物理配置 | `optimization9/A/diagnosis.md` | 真实修复碰撞—约束—驱动耦合；保留需要的手物/物桌/手桌碰撞；重放与多环境验收 |
| P0-02 | 最高 | 旧Teacher接触槽**漏掉4个独立tip刚体**的真实碰撞 | `teacher_env_cfg.py`、`mdp/runtime.py`、`optimization9/A/all_body_contact` | 在独立新训练Task中建立真实tip传感器/接触特征；区分接触手物、手桌、手部其他；逐一校验名称与力数据 |
| P0-03 | 最高 | `reset_teacher`后的碰撞回退主要检查arm接触，没有证据证明tip/table异常在初始可抓姿态中被排除 | `mdp/events.py::_current_arm_collision/check_initial_collision` | 新Task加入tip和真实碰撞安全检查，重置合理的IK/桌面安全距离；统计失败率、不要“第一步就撞桌” |
| P0-04 | 最高 | 仍无**可用的脚本抓取—抬升正例** | `optimization9/B/scripted_gate.json` | 实现 `scripted_grasp_probe.py`，在训练物体上做真实接触闭合/上抬/保持，至少多次真实试验 |
| P0-05 | 最高 | 新Task不是“完整自主抓取”任务 | `grasp_train_env_cfg.py` | 新语义Task，设计Approach/Close/Lift/Hold、相位与目标观测、真实成功判定 |
| P0-06 | 最高 | 4s episode、当前arm/hand动作尺度未证明可完成完整抓取 | `grasp_train_env_cfg.py`、`mdp/actions.py` | 先测到达/闭合/上抬时间，再确定6–10s等候选，不把名义动作尺度当作真实执行力 |
| P0-07 | 最高 | Teacher位移/速度负项可能压制自主抬升；没有持续成功正向奖励 | `TeacherRewardsCfg`，`rewards_grasp_v1.py` | 新Task阶段条件奖励；消除抬升阶段负向冲突；惩罚弹飞/撞桌/刷接触力 |
| P0-08 | 最高 | 无PPO训练结果、无策略自身抬升结果 | `optimization9/D/G` | 通过物理+脚本Gate后真实短训、再训练；独立训练集真实试次及曲线 |
| P0-09 | 最高 | 没有新策略在GUI里完成完整抓取 | `play.py`、`optimization9/0/gui_baseline_result.json` | 原生`play.py`加载真正训练的`.pt`，**纯policy动作**完成全流程；支持显示/记录真实成功 |
| P1-01 | 高 | `invalid_hand_height`：任一hand关键点低于桌面就终止，未证明能区分合法接触与穿透 | `mdp/terminations.py` | 新Task单独检验/修正基于真实碰撞、深度、状态的安全终止，不能放开所有撞桌行为 |
| P1-02 | 高 | 缺少Top物体相对速度、阶段/剩余时间、抬升目标等完整任务状态 | `mdp/observations.py` | 显式、可重复、无未来信息泄漏的新观察契约，网络维度同步 |
| P1-03 | 高 | 训练Reset使用源IK、fallback、随机位置及一次性Object Bias；单木块上可抓初态比例未知 | `mdp/events.py` | 窄分布课程，统计IK可行率/实际碰撞/指尖接近/成功重置；先关闭或版本化bias |
| P1-04 | 高 | 物体Articulation的top/bottom位置、接触、动态载荷是否满足抓取未验证 | `teacher_env_cfg.py`, USD, `mdp/runtime.py` | 保留原Articulation，用top高度和真实接触判定，不擅自改RigidObject |
| P1-05 | 高 | Sensor每个filtered body需唯一匹配；某个正则可能同时匹配tip及父link或根本漏报 | `teacher_env_cfg.py` sensor regex | 新Task每个传感器 `body_names/num_bodies/filter` 真正运行检查；对应IsaacLab 2.3.2 1-to-many规范 |
| P1-06 | 高 | `play.py`的旧Teacher物体选择/episode覆盖/契约路径未针对新自主任务真实证明 | `scripts/rsl_rl/play.py`, `checkpoint_contract.py` | 新Task train→checkpoint→play端到端同配置测试，显示训练集物体；绝不为了可播放默认强开mismatch |
| P1-07 | 高 | PPO的normalizer、噪声、STD下界、critic是否适用于全任务尚无证据 | `rsl_rl_grasp_train_ppo_cfg.py`、`train.py` | 只在合格物理轨迹上对比PPO稳定性；记录收益/critic/动作饱和/GPU资源 |
| P2-01 | 次要 | 优化9提交包含大量原始轨迹、图/重复训练权重文件 | `logs/acceptance/optimization9` | 后续实验应分层存储大文件，必要时Git LFS，不删除旧证据或旧checkpoint |
| P2-02 | 次要 | 历史评估协议很完整，但本轮目标仅训练集GUI | `evaluate.py` | 本轮不优先改`evaluate.py`，不用验证集；可用轻量训练集统计工具 |

**拒绝无效“修复”：**永久移走桌子；禁用全部tip或手/物真实碰撞；把实际q强制clip后假称限位正常；把物体直接绑定到手上；训练或播放时偷偷写物体上抬pose；用评估脚本替策略控制UR5再声称“自主抓取”；未训练就把历史`.pt`当作优化10成果；单纯让日志零报错/Reward变大就判定抓取学会了。

---

## 2. 不能突破的运行和实现约束

1. **PhysX物理步长固定：`sim.dt == 1/120`秒**，所有Task及诊断脚本断言；可在实验中调整decimation、PD、资产物理属性、Reward、Episode、PPO，但必须版本化。
2. 默认保持UR5+Allegro本体、训练物体bottom/top **Articulation**、真实桌面、重力、桌面支撑、手物摩擦和真正物理碰撞。保留原Teacher-v0，禁止静默把源物体改为`RigidObject`；若最后确有必要做简化，只允许**单独实验Task、明确新MDP标记**，不作为默认成功路径。
3. 只使用`assets/objects/new_training_set`；先`wood_block_oriented`，再训练集其他已有物体。**不使用**`shapenet-30obj`、heldout验证集或论文复现作为本轮Gate。
4. 现有IsaacLab `ManagerBasedRLEnv` + RSL-RL3.1.2标准训练接口；不要复制自定义PPO框架；全部Isaac Sim运行在3070Ti**串行**，先1/16/128env再根据VRAM/FPS考虑256/512/1024/2048。
5. 最终`play.py`中的UR5、Allegro动作全部来自策略输出；不得使用与`scripted_grasp_probe.py`相同的上抬控制器代替策略。课程中的示范/辅助如确实使用，必须明示，最终取消辅助后重新验收。
6. 训练目标8–10小时是**通过短训学习Gate之后**的GPU训练墙钟预算，不保证必然成功。若发生真实物理异常须停止**不安全长训**，但允许并要求继续完成不依赖物理Gate的独立代码实现、单元测试及诊断工具。
7. 工作区不是天然干净：先保存`git status`与diff，保护未提交物理修复/实验，不擅自`reset --hard`或删除日志；基线以本地实际HEAD为准记录；必要时创建新分支`optimization10-full-grasp`。
8. 所有失败写`fail`或`blocked`，未发生的attempt必须为0、对应成功率为`null`；不得用“准备好代码”“脚本能启动”“GUI能看到机器人”代替实际抓取成功。

---

## 3. Phase 0：立即盘点最新真实工作区，并保留已有证据（轻量）

在项目根目录执行并记录实际输出：

```bash
cd /home/windsky/project/Grasp1
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1
git rev-parse HEAD
git status --short
git diff --stat
nvidia-smi
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --help
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --help
```

- 若当前HEAD不是`04603ca...`，对照新增/变动内容并以**实际新代码**为准，不擅自硬退回旧版本。
- 创建 `logs/acceptance/optimization10/{0,A,B,C,D,E,F,G,H}/`，首个阶段保存`0/baseline.json`和diff备份；不覆盖optimization7/8/9。
- 创建可重复的`commands.md`、`gates.json`、`summary.json`。精确区分静态/合成/真实回放/脚本抓取/PPO/GUI。
- 把优化9已有效的`A/contact_constraint_probe.py`、旧trace和碰撞pair隔离对照**直接复用**，避免重新发明13个重复实验。

**Gate 0**：环境、源码、训练资产、日志与实际CLI全部可重现。

---

## 4. Phase A：优先解决thumb tip/Table冲突，同时修复“接触看不见”

> 已知触发碰撞对，不要再次泛泛查“桌面是不是有问题”；以`link_15_0_tip`为中心开展可验证的局部根因和最小物理修复，**不要因为没有底层完整constraint impulse API就无限停留在诊断**。

### A1. 先分离“异常初始姿态”与“真实物理缺陷”

1. 复用优化9原始`initial_replay.pt/trace_replay.pt`，记录第470步前后至少0.5s实际 thumb tip 的**世界坐标最小/最大顶点Z、与桌面碰撞形状的间距、接触点、法线、separation、相对速度、关节状态/限位/目标和接触力**，明确是在预抓取、运动中的擦桌还是穿透修正中发生。
2. 找出可见的`link_15_0_tip`、`link_15_0`、其他三tip的碰撞prim与fixed joint关系；核对真实shape，而非仅AABB；审查PhysX material、mesh近似形状/凸包、contact offset、rest offset、惯量、关节限位、legacy friction与drive参数。
3. 做**两个都保留所有必要碰撞**的对照：A. 原始保存初态目标；B. 合法的高桌面安全余量预抓取/手指静止→闭合→上抬目标。比较什么时候tip碰撞出现，区分历史异常轨迹是否由危险初始化放大。不能用B通过掩盖A原样失败，应分别报告“历史replay兼容性”和“新任务安全域”。
4. 审查是否存在`q`与`qd`在TGS+摩擦模型下产生显著不同步；必要时做**物理合理的单变量**摩擦模型/solver与驱动对照。对`qd`、位置差分、世界刚体速度和实际关节越限分别统计。资料可以参考PhysX项目关于TGS/关节摩擦的研究 https://github.com/NVIDIA-Omniverse/PhysX/issues/498 ，但**不能把外部issue当作本机原因已确诊**。

### A2. 优先测试的“保留抓取能力”的候选

仅在独立诊断资产和配置上进行小范围A/B，每次保持固定历史初态/目标与新合法抓取轨迹两套测试：

- **预抓取/reset候选**：调整`events.py`中新任务IK选择/手掌偏移/物体XY和yaw范围，使初始tip与桌面存在经验证的安全间隙；如果不是物理模型本身缺陷，不必强求在不合理的老动作轨迹上零接触。
- **真实几何和碰撞候选**：检查thumb tip convex collision相对视觉/URDF尺寸、固定连接、接触偏移与桌面几何，必要时修复明显错误的几何穿透/重叠/凸包，而不是去除正常tip碰撞；每次保存USD差异/质量/COM/惯量守恒并在接触对象上验证夹持作用。
- **PhysX驱动与摩擦候选**：优先在有物理依据时做joint friction model、手指PD/限矩、contact/rest offset及求解器单变量对照；记录真实性能成本。旧`friction=0.05`、`255P/32V`和thumb融合均已单独失败/证据不足，**不得原样复用并宣称成功**。
- **必要接触仍保留**：tip↔Table、tip↔Object、Object↔Table实际继续工作。可以屏蔽明显不应存在的**装配内部自碰撞**，但必须独立证明不影响真实抓握。

### A3. 把真实tip接触加入新任务传感器

- 在**新任务**中为4个tip刚体（如`link_3_0_tip`, `link_7_0_tip`, `link_11_0_tip`, `link_15_0_tip`）创建分别针对object top/bottom和table的filtered contact sensor；或者先选运行环境验证兼容的最简传感器集合。不要把regex写成可能同时匹配tip及其父刚体。
- IsaacLab 2.3.2要求**单环境每个filtered sensor匹配一个sensor body**。在运行时assert `sensor.body_names`、`sensor.num_bodies`、filtered shape维度、数据为非`None`、正接触分支出现非零值。原13-slot/source布局可保留旧Teacher；新任务明确定义新增tip接触语义、Obs/Reward版本，不要把新17-slot数组误当作源13-slot直接拼入原153D。
- 必须在优化9的第470步tip/table回放中读取到非零contact；在合法脚本抓握tip/object试验中能读到手物接触。分别检测接触与非接触的误报，失败立即记录传感器正则匹配表。
- 修改新任务的reset安全检查：**不能仅用`_current_arm_collision`检查机械臂碰撞**。实际检查tip桌面、手指危险碰撞、IK可行性、有效桌面高度与关节限位；仅保留合法初态。设重采样次数上限并记录失败/回退比例，不能无限重采样或悄悄把物体放到手里。

### A4. 双重物理验收：不做假安全

1. **旧轨迹复验**：固定历史target的真实`q/qd/limit/contact/force`与初态一致性，并对比原始峰值和候选。对于涉及初态课程合法域变化的候选，额外报告是否仍能重放旧异常，不能删掉失败数据。
2. **新训练可达轨迹**：合法预抓取、渐进闭合、上抬和随机动作在1→16→128env、多个seed、完整episode均稳定；真实关节越限工程初验`p99.99 ≤0.001 rad、max ≤0.005 rad`，监测速度、异常冲量/物体弹飞、tip/Table穿透/法向力、真实动力学一致性与FPS/VRAM。阈值若不适用只能附实验证据修订，禁止为了通过Gate随意放宽。
3. 物理候选应证明**手能抓住物体**；只在空场景或关闭接触环境稳定不合格。安全的关节轨迹可以作为训练初始课程，但不得在策略运行后覆盖策略动作假造抓取。
4. 把约束“无法获取细粒度contact/limit/drive impulse”的API限制记入报告；如果可利用真实状态、必要接触与强对照充分支持安全性，**不以获取不可用API作为无期限前置条件**。

**Gate A**：在保留必要碰撞的真实新任务安全域内，可复现稳定运行、tip接触可观测，并通过脚本预抓取方向的安全筛查。若仍fail，不执行不安全PPO长训，但**立即进入Phase C的纯代码实现及静态/合成测试**，不要继续只写物理报告。

---

## 5. Phase B：真实脚本抓取基准（必须产生物理正例）

- 新建或复用 `scripts/rsl_rl/scripted_grasp_probe.py`，支持`--task`、`--dataset new_training_set`、`--object`、`--num_envs`、`--seed`、`--headless`、`--output_dir`等经真实`--help`验证的选项。
- **仅在诊断入口中**：IK可达预抓取→机械臂接近→独立4tip参与闭合/形成夹持→真实UR5命令上抬≥可达合理高度→持续保持。物体必须由接触力搬起，绝不写object root pose、关闭重力或添加运动学跟随。
- 记录per-step：初始top Z/手掌位姿、目标q与真实q、tip↔object/tip↔table/object↔table接触、相对手物位姿、lift高度和持续时长、关节越限/速度、失败原因；输出逐次CSV + 精选trace + GUI截图/视频（如支持）。
- 初期使用`new_training_set/wood_block_oriented`及几个可行XY/yaw，做≥10次合法初态（失败包含在分母）。工程初验参考至少3次可重复真实抬升且短保持；如不通过，具体定位`IK/reset/接触/力量/姿态/物体质量/表面摩擦`而不是一味加PPO迭代。
- 实测闭合、上抬、保持所需秒数，反推episode和动作尺度，公开哪些由脚本控制、哪些未来应由策略控制。

**Gate B**：在保留桌面与全必要碰撞的环境下，有真实多次脚本抓物离桌、维持接触与稳定保持的正例。所有接触、q、lift均不是外部假造数据。

---

## 6. Phase C：必须落地独立完整抓取MDP（无论Gate A当前是否通过，都先完成可独立实现的代码）

### C1. Task和文件架构

优先注册新完整任务 `Grasp1-UR5-Allegro-GraspTrain-v2`（如果本地已有等价的完整自主抓取Task，检查后复用并标记真实Task ID）；原`GraspTrain-v1`和旧`Teacher-v0`保留，不把旧模型误当兼容。新Task继续使用Isaac Lab Manager系统，避免复制train loop。

建议新建：

```text
source/Grasp1/Grasp1/tasks/manager_based/grasp1/
├── config/ur5_allegro/
│   ├── full_grasp_env_cfg.py                 # 任务、阶段、episode、物体池
│   └── agents/rsl_rl_full_grasp_ppo_cfg.py  # 训练runner/网络独立版本
└── mdp/
    ├── full_grasp_events.py                  # 有效IK/reset与课程安全域
    ├── full_grasp_observations.py            # 对象top/手相对运动、目标/phase
    ├── full_grasp_rewards.py                 # approach/grasp/lift/hold奖励
    ├── full_grasp_terminations.py            # success/drop/unsafe/timeout
    └── full_grasp_metrics.py                 # 训练集真实成功率与tip接触
scripts/rsl_rl/
├── scripted_grasp_probe.py                   # 只验证物理可抓
├── train.py                                  # 复用现有标准入口，必要的最小补丁
└── play.py                                   # 纯policy GUI播放，最小功能增强
```

不要为完成目录清单而机械创建空文件；实际按代码复用情况整合，**真实实现并注册**。如版本化阶段课程独立需要新模块，可添加但控制复杂度。

### C2. 新MDP的正确目标

每个episode：

1. `Approach`：策略调整UR5/Allegro使手靠近物体可抓区域，远离桌面非法碰撞；
2. `Grasp`：策略自主闭合手指，真实tip与物体接触且手物关系趋稳；
3. `Lift`：**策略自己输出UR5关节动作**使物体离开桌面，且不是碰撞弹飞；
4. `Hold`：在物体离桌后继续保持真实夹持和高度、相对姿态稳定，例如≥1秒（可作为初始门槛并由脚本测试校准）。

阶段转移使用真实状态，不允许独立的环境脚本替策略发送机械臂上抬目标；如果Phase有限状态影响期望行为，必须给actor相位/目标/剩余时间等可观测信息。不要让`Hold`在成功一次高度阈值时立刻结束而错报成功。

- `episode_length_s`根据脚本闭合和抬升时间实测选择，例如6/8/10秒候选；4秒不强行沿用。
- 默认保留22D物理动作（6 UR5+16 Allegro）；先用脚本量化D4当前arm`0.0008333333`/hand`0.0025`是否足以在episode内抬升，再通过明确的新动作配置调整。可以考虑更合理的动作速度/位姿表示，但绝不可静默把某些actor输出屏蔽然后仍假设策略能自主控制。
- 观察至少包含物体top相对手位置及速度、手物接触/可达距离、实际机器人状态与动作偏差、物体相对桌面实际高度、抬升目标、phase和保持计时；如改Obs维度，重建网络输入并更新checkpoint schema，不强行加载153D旧checkpoint。
- `reset`初期只用训练集木块，较窄的XY/yaw/IK分布，减少失败初态；保留训练场景真实随机性与可重复seed；统计有效初始化比例。可禁用`apply_object_position_bias`或将其纳入可观测/明示的课程版本，避免在抓取临界点突然移动物体。
- `terminations`区分危险碰撞/真实穿透、物体掉落、非有限/明显物理异常、timeout、**持续满足的成功**。对旧`invalid_hand_height`独立审查：不能把合法擦桌与穿透等价，也不得取消安全保护。

### C3. 奖励和成功定义（不能只靠Reward刷分）

构建新任务独立Reward，初期基于已审计`source_physics_aligned`，但**不照抄grasp-only所有系数**。至少实现：

- 趋近可抓区域、姿态对齐；
- 真实tip/object稳定多指接触，优先多指相向夹持，不能把“接触数量≥2”视为绝对力闭合；
- 物体真实离桌高度、随手相对移动、连续维持上抬；
- 达标持续保持的终止或一次性成功奖励；
- 关节越限、危险tip/table接触、手臂撞桌、极端冲量、过大速度/动作抖动、弹飞和回合重置失败惩罚。

重点检查旧Teacher：`obj_vel_reward_`源系数`−15`、`obj_displacement_reward`源系数`−5`，当物体升高时可能与Lift奖励直接冲突。按阶段禁用、条件化或重权；每项在真实/合成轨迹上证明方向和量级合理。

新RewardManager依控制步`env.step_dt`加权；连续奖励、事件/终止奖励分别正确缩放；contact force→impulse只乘真实物理`1/120 s`一次，不能额外乘decimation；不把接触传感器的正常接触峰奖得越大越好。

**真实成功判定（必须同时满足）：**物体top离桌高度增量达到配置阈值（如起始候选0.10m，物理可达性先验证）、物体已真实脱离桌面支撑、真实tip/手指与物体保持稳定接触、物体相对手位姿不发生异常漂移、连续保持指定时间、没有危险碰撞/穿透/弹飞/提前终止。成功率一定给出`successes/attempts`以及seed和物体名。

### C4. Gate C：先验证可学性接口再允许真训练

- 运行Task实际注册、真实obs/action/reward维度、partial reset、4tip正/负接触、阶段转换、脚本物理上抬中Reward变化、1/16/128env、D4与选定D2对照（仅测试、正式实验不可跨MDP续训）。
- 检查step/substep动作目标不抖动、相位随env reset独立复位、物体top参考一致、失败原因不互相覆盖。
- 奖励合成方向测试+脚本真实轨迹分项分布；不把“测试通过”当成“学到抓取”。
- Checkpoint的新Task ID、动作尺度、观测语义、Reward/phase/episode参数、训练物体和课程必须写入可恢复元数据。改变MDP禁止静默resuming旧任务策略。

**Gate C**：新自主完整抓取Task已真实创建，模型配置、Reset、Reward、Termination和训练接口一致，物理仍须过Gate A/B后才能执行大规模PPO。

---

## 7. Phase D：短PPO学习Gate——从训练集一个物体学会自主上抬

满足Gate A/B/C后开始真PPO：

```bash
cd /home/windsky/project/Grasp1
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1
# TASK_ID必须替换为代码实际注册成功的新自主任务，例如：
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Grasp1-UR5-Allegro-GraspTrain-v2 \
  --num_envs 128 --max_iterations 100 --headless
```

此命令是**在Codex实现并成功注册新Task之后的计划命令**，不是当前V2.2已能执行的命令。以本机真实CLI/显存实测调整。

- `100 update`检查物理/张量/Reward/critic/std、真实策略是否开始触摸物体；`500 update`与`1000 update`逐级继续或对照，在物理安全前提下运行，不能用纯空白动作积分充数。
- 起步128/256环境，记录真实FPS、3070Ti 8GB显存峰、GPU内存、重置成功率、每个阶段进入比例、奖励分项占比、动作实际目标饱和、手指接触与物体离桌持续时间。
- 训练过程中定期保存checkpoint；用**训练集**单物体及未在当前测试中重复使用的初始随机seed进行简单短试，要求**只输出策略动作**。可以有自动统计脚本，但`quick_lift_eval.py`不是必须；不得替策略额外抬升。
- 与随机策略和确定性脚本基线区分：脚本只能证明环境能抓，policy必须以自己的控制在真实物体上产生持续离桌成功。若成功率没有改善，先复盘Reset可行率/奖励冲突/动作幅度/critic，而非只提高`max_iterations`。
- 如果训练全无接触，应优先降低初态难度和改Reward/Action；如果有碰但无抬升，应查Lift奖励/动作方向；如果物体弹飞算成功，应修success逻辑和接触安全；如果PPO critic崩溃，应诊断奖励尖峰和物理，并只做少变量调整。

**Gate D**：有真实`.pt`、真实PPO训练日志，并且至少一个训练物体在不同reset/seed下出现**可重复的策略自主“抓握—抬升—保持”非零成功**；比随机无策略基线有可靠行为改善。若短训不达Gate，禁止宣称长期训练已具备效果。

---

## 8. Phase E：先用play.py在GUI证明新策略抓物

- 使用当前`script/rsl_rl/play.py`的真实CLI（路径是`**scripts**/rsl_rl/play.py`），优先复用现有非headless显示能力，只做必要的新Task兼容和可视化/成功提示。
- `--dataset new_training_set --object wood_block_oriented`，1环境，选择真实新任务checkpoint（不是历史Teacher `model_999.pt`、不是合成Fixture），使用与训练兼容的配置/课程；**不要默认`--allow_mismatched_checkpoint`**。
- GUI运行时`actions = policy(obs) -> env.step(actions)`；若用户配置了phase标签/接触/物体top height的显示，则该信息只读，不改变控制输入。
- 保存至少一次完整动作的GUI截图/录屏、实际日志、checkpoint SHA、训练run和真实分母；每次回合能反复reset，物体再次从桌面被策略自主抓起。

在新任务已注册、存在正式训练权重后，预期命令形式：

```bash
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Grasp1-UR5-Allegro-GraspTrain-v2 \
  --checkpoint logs/rsl_rl/<actual_new_experiment>/<run>/model_<n>.pt \
  --dataset new_training_set --object wood_block_oriented \
  --num_envs 1 --episodes 5 --real-time
```

**Gate E**：新模型可用`play.py`非headless GUI反复看到纯policy完成接近、闭合、离桌、保持。若可视化看出“仅闭合未上抬”，Gate E失败、回到Phase C/D，不允许用脚本抬升掩饰。

---

## 9. Phase F/G：训练集扩容和8–10小时正式训练

仅在D/E真实成功后：

1. 从单个训练物体到2–3个易抓物体，再5–10个，再完整`new_training_set`（难物体按实测课程调整）。任何扩容都使用训练集内部物体，不需要验证集。
2. 课程阶段与物体池、Reset随机化范围、Reward profile、新Task版本、动作/观测维度都记录于checkpoint和manifest；改变语义时新开训练实验或经过明确兼容测试。
3. 依据显存与吞吐选择环境数，不盲目追求2048；正式训练总GPU墙钟目标8–10小时，但**不包括之前物理诊断时间**；自动记录中断、续训、best/last。
4. 每50–100 update统计loss、KL、entropy/std、奖励分项、步数/FPS/显存，最重要是**训练集真实自主抓取成功数/次数、连续hold**；PPO reward上升但自主成功不提升不得当作成功。
5. 出现真实关节限位越限反复恶化、异常碰撞、弹飞刷分、非有限/显存不足等问题时保存现场并停训，不吞掉错误继续8小时。
6. 最终展示：一个木块的稳定成功和至少几个训练集物体的自主抓取GUI（受训练结果限制如实报告），同时保存量化表；暂时不触碰`shapenet-30obj`与正式论文评估。

**Gate F/G**：以实际训练run、best/last权重、独立随机初态真实成功率和`play.py` GUI记录为准；不得许诺“8小时一定训练出成功”。

---

## 10. 必须实际修改与交付的内容

### 10.1 必须优先修改的训练核心

- `source/Grasp1/Grasp1/robots/ur5_allegro_cfg.py` / UR5+Allegro相关USD与PhysX配置：与已定位thumb tip/Table碰撞相关的**被验证**最小修复（必要时新任务专用版本，旧Teacher保持）。
- `source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/`：新FullGrasp任务 + PPO配置。
- `source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/`：真实tip接触/重置安全、完整抓取Obs/Action所需修改、阶段Reward、真实成功与Termination。
- `scripts/rsl_rl/scripted_grasp_probe.py`：物理脚本可抓基线工具。
- `scripts/rsl_rl/train.py`：仅为新完整Task必要的诊断/正确checkpoint/课程记录做最小修改。
- `scripts/rsl_rl/play.py`：能够播放新任务checkpoint的非headless GUI；必要时加训练集成功/高度/相位只读显示。**不插入上抬脚本**。

### 10.2 不要偏离主线

- 不要优先重写`evaluate.py`、ShapeNet评估协议、复杂checkpoint兼容矩阵。
- 不要在没有物理/脚本抓取正例时直接进行长PPO训练。
- 不要在需要继续实现Phase C的情况下，以“物理Gate失败”为由只输出日志然后提前停止**全部代码实现**。
- 不要大量重复优化9已失败的纯摩擦/求解迭代实验而不解释原因。

### 10.3 交付路径

```text
docs/优化10_执行报告.md
logs/acceptance/optimization10/
  0/baseline.json
  A/physics_gate.json              # 包含旧回放、合法预抓取新场景的对照与真实tip记录
  A/physics_candidates.csv         # 每个候选的唯一物理参数与拒绝理由
  B/scripted_grasp_trials.csv      # ≥10次/真实分母，非成功的失败原因
  B/scripted_gate.json
  C/full_grasp_task_contract.json  # 新Task、Obs/Act/Reward/phase/Assets/物体集
  C/task_gate.json
  D/ppo_short_runs.csv             # 真update/FPS/Reward/critic等
  D/policy_train_trials.csv        # 纯policy的训练集抓取分子/分母
  D/learning_gate.json
  E/gui_play_trials.csv
  E/play_gate.json
  F/training_set_curriculum.csv
  G/formal_training.json
  G/training_set_success.csv
  commands.md
  gates.json
  summary.json
```

如某阶段没有运行，文件写`blocked/not_run`及准确原因，**不要填假的0%成功率**。只有成功运行的`.pt`才给真实checkpoint路径，不能把导入Teacher旧checkpoint或fixture报作优化10训练成果。

---

## 11. 成功验收优先级：不要把软件工程检查误当作抓取学习

按以下顺序给出各Gate的实际证据：

| Gate | 真正要证明的事情 | 不能替代它的伪证据 |
|---|---|---|
| A | 真实tip/桌/物接触可观测、动力学与限位稳定、可抓路径可达 | 仅把table关掉、只降峰、target合法或finite |
| B | 真实脚本能多次抓起木块且维持 | 直接搬动物体pose、关闭重力 |
| C | 新Task能够学习完整动作且Obs/Action/Reward/阶段正确 | 维数相同、配置能实例化 |
| D | 真实PPO在训练物体上学会自主Lift+Hold | Reward上升、PPO跑完更新、零物理异常 |
| E | `.pt`可通过`play.py`在GUI反复纯policy成功 | GUI看见机器人/物体、历史Teacher接口播放 |
| F/G | 扩大训练集后仍有真实成功，8–10小时run可追溯 | 训练用了8小时但失败、没有successes/attempts |

**工程初期最低有价值成功**：单物体`wood_block_oriented`、至少多次不同初态的真实策略自主抬升且连续保持；每次不写物体位姿、不借外部抬升脚本。若目前没有这种结果，如实把优化10状态写为未完成，而不是宣称交付了成功抓取项目。

---

## 12. Codex开始执行的最终指令

**现在在`/home/windsky/project/Grasp1`中实际执行优化10。** 首先审查本地最新代码和备份修改；直接复用优化9已定位的`link_15_0_tip`/Table因果证据，用**保留所有必要碰撞**的安全初态/碰撞几何/材料/驱动候选修复明显实际关节越限，并让真正tip接触进入新Task传感器；不要无期限反复做同类诊断。随后实现并真实运行`scripted_grasp_probe.py`，让训练集木块在物理上可重复抓取、抬升和保持。**无论物理Gate是否暂时失败，都先写完可独立实现的完整自主抓取新Task、Reward、Observation、Termination、课程与play入口兼容，并进行静态/合成测试**；只有物理+脚本正例合格时才投入短PPO与8–10小时正式训练。使用真正新训练的`.pt`通过`play.py`非headless GUI展示**策略自己**完成全流程，不使用验证集，不优先修改`evaluate.py`。逐阶段执行真实命令、保存原始物理/训练证据、失败原因和可复现执行报告。**实际运行优先于再写一份空泛建议。**
