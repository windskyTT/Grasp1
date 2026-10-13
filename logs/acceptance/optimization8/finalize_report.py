"""仅汇总本次真实证据；Gate失败后的训练/评估保留null。"""
import csv
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path('logs/acceptance/optimization8')

def write(path,data):
    (root/path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

states={'0':'pass','A':'fail','B':'blocked','C':'blocked','D':'blocked','E':'partial_cpu_pass','F':'blocked','G':'blocked','H':'partial_checks_pass'}
write('gates.json',states)
reason='Gate A复现同初态/同target异常；安装碰撞过滤、同摩擦Box、手部刚度150均未消除越限，没有通过物理Gate的正式配置。'
write('B/reward_gate.json',dict(status='blocked',independent_static_synthetic_tests='pass',source_terms_audited=17,diagnostic_contact_samples=1165,qualified_physical_distribution_test='not_run',reason='缺少Gate A通过的真实轨迹；方向/单位单测通过不能替代真实Gate B。',profile_isaac_grasp_v1='not_enabled',weights_tuned=False))
write('C/task_gate.json',dict(status='blocked',native_task_registered=True,independent_interfaces='pass',strict_contract_tests='pass',formal_candidate_qualified=False,reason='物理修复、窄范围可抓初始化、脚本抓握正对照及128env真实物理Gate尚未完成。'))
qa=[]
for path in [root/'C/new_D4_1_retry',root/'C/new_D4_16',root/'C/new_D2_16',root/'H/new_task_final',root/'H/new_task_final_object_contract',root/'H/old_teacher_final']:
    p=path/'interface_result.json'
    if p.exists():qa.append(json.loads(p.read_text()))
write('C/initialization_qa.json',dict(status='interface_only',cases=qa,not_run=['C1 narrow XY/yaw/pregrasp optimization','scripted grasp positive control','128env Gate C physical acceptance'],reset_ik_qa='Existing reset/IK retained; feasibility and geometric penetration gate not qualified'))
final=root/'H/new_task_final_object_contract'
if (final/'checkpoint_contract_tests.json').exists():shutil.copyfile(final/'checkpoint_contract_tests.json',root/'C/checkpoint_contract_tests.json')
if (final/'candidate_contract.json').exists():shutil.copyfile(final/'candidate_contract.json',root/'C/candidate_contract.json')
all_rows=[]
for path in [root/'C/new_D4_16',root/'C/new_D2_16']:
    p=path/'reward_component_distribution.csv'
    if p.exists():
        with p.open() as f:
            for row in csv.DictReader(f):all_rows.append(dict(experiment_id=path.name,**row))
if all_rows:
    with (root/'B/reward_component_distribution.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
for phase,name in [('D','learning_gate'),('F','training_gate'),('G','evaluation_gate')]:
    write(f'{phase}/{name}.json',dict(status='blocked',reason=reason,formal_updates=0,formal_wall_s=0,attempts=0,success_rate=None,checkpoint=None))
write('D/ppo_smoke.json',dict(status='not_run',reason='Gate A failed; no PPO update performed even in interface diagnostics',updates=0))
(root/'D/candidate_matrix.csv').write_text('experiment_id,status,updates,short_lift_success_rate\nD4-source-rate,blocked,0,\nD2-source-rate,blocked,0,\n')
(root/'D/quick_lift_trials.csv').write_text('trial,object,seed,contact_success,short_lift_success\n')
(root/'G/mvp_trials.csv').write_text('trial,object,seed,short_lift_success\n')
write('G/quick_lift_eval.json',dict(status='not_run',attempts=0,success_rate=None))
for name in ['shapenet_20_20','engineering_hold']:
    folder=root/'G'/name;folder.mkdir(exist_ok=True)
    write(f'G/{name}/summary.json',dict(status='blocked',attempts=0,success_rate=None,reason=reason,grasp_s=20,lift_s=20,ramp_s=16 if name=='shapenet_20_20' else 14,hold_s=None if name=='shapenet_20_20' else 5))
    (folder/'trials.csv').write_text('trial,object,seed,source_height_success,active_height_success\n')
manifest=json.loads((root/'A/physics_manifest.json').read_text())
peaks=list(csv.DictReader((root/'A/peak_cases.csv').open()))
summary=dict(task_complete=False,executed_at=datetime.now(ZoneInfo('Europe/Berlin')).isoformat(),baseline_git_sha='5eae098f0ed8c11335c1b2c06a31b8a4dea953b0',task_ids=['Grasp1-UR5-Allegro-Teacher-v0','Grasp1-UR5-Allegro-GraspTrain-v1'],phases=states,physics_stable=False,physical_fix_adopted=False,RL_training_completed=False,MVP_grasp_success=False,source_40s_evaluated=False,engineering_hold_evaluated=False,paper_reproduced=False,formal_training_updates=0,formal_training_transitions=0,formal_training_wall_s=0,diagnostic_ppo_updates=0,formal_evaluation_attempts=0,formal_success_rate=None,quick_lift_attempts=0,best_checkpoint=None,last_checkpoint=None,physics_experiments=len(manifest['experiments']),cpu_forward_actor_max_error=0.,cpu_forward_critic_max_error=0.,blocked_reason=reason,not_run=['Gate A accepted-configuration 16/128env multiple seeds','A/D scripted grasp-and-lift positive controls','A-qualified B real reward distribution acceptance','C1 narrow initial state and curriculum','D PPO short learning and throughput comparison','E source weights PhysX transfer','F 8-10h formal training','G MVP, ShapeNet90 and engineering hold evaluation'],source_changes=[p for p in subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()]+['source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/grasp_train_env_cfg.py','source/Grasp1/Grasp1/tasks/manager_based/grasp1/config/ur5_allegro/agents/rsl_rl_grasp_train_ppo_cfg.py','source/Grasp1/Grasp1/tasks/manager_based/grasp1/mdp/rewards_grasp_v1.py'])
write('summary.json',summary)
(root/'A/diagnosis.md').write_text('''# 真实物理诊断：Gate A失败，未采用修复

从优化7 `baseline_D4/initial_zero.pt` env7复制实际机器人/物体/reset状态，按env origin换算；重放优化6保存的120Hz目标，通过正式相对位置Action接口请求目标。1env与原16envGPU求解布局不同，不能声称历史112.90/120.81峰逐帧精确复现。目标最大回放误差3.73e-9rad；无物体原样与重复的q/qvel/target整条轨迹严格相等。

| 同fixture D4/10秒条件 | 手部qvel峰rad/s | 所有关节最大越限rad | 决定 |
|---|---:|---:|---|
| 桌面保留、物体移开（两次） | 30.957081 | 0.08780555 | 失败，重复复现 |
| 仅固定base_link/Table FilteredPairs安装过滤 | 30.957081 | 0.08780555 | 完全无改善，拒绝 |
| 同尺寸单Box，同摩擦1/1、恢复0 | 12.232162 | 0.07874259 | 局部降峰但严重越限，拒绝 |
| 仅hand stiffness600→150，其他参数相同 | 38.485245 | 0.07981350 | 仍失败，拒绝 |
| 桌面移开、物体保留 | 0.282938 | 1.86265e-6 | 诊断负对照，不能采用训练场景 |
| 桌面/物体均保留 | 9.590271 | 0.01650780 | 失败，有真实手物接触 |

D2单环境回放优化7 E2 random的已保存目标而非新Gaussian，得到15.715107rad/s、最大越限0.00314794rad、p99.99=0.00176542rad，仍超过p99.99≤0.001门槛。它不是原16env120.81456峰的精确复现，也不是D2通过证明。

最重要无物体样本：joint_15_0的最大越限在第1024帧，q=-0.24980554、target=-0.162；首次timeout第1200帧。最大连续>0.001rad持续3.3333秒，不能归为单帧噪声。速度峰第1197帧30.95708，位置差分-0.05482rad/s。真实link位置差分同时在各101帧peak_window.csv保存，未替换/clip PhysX速度。全体target合法且子步固定，不能据此认定真实关节安全。

加载后遍历instance proxy得到27个实际CollisionAPI prim。修正后的碰撞顶点世界AABB显示安装底座和桌面Z间隔约2.67e-8m，几何上近似贴合；AABB不证明solver接触身份。初两次BBoxCache(Xform proxy)产生空范围，原始文件保留且排除几何结论；最终root loaded_colliders.json使用Mesh/Cube顶点+实际世界变换。桌面原材质实际静/动摩擦1/1，Box保留这一值。

尝试本机omni.physx回调和get_contact_report立即API，并设置ContactReportAPI及启用contact processing。两种报告接口均返回空，包括Tensor Sensor明确有55.51/150.43N接触的正接触诊断；不能由空报告或未覆盖sensor零值推断全场景无约束。该报告路径尚未提供可用shape pair身份，不宣称base/table接触已确诊。

C短接口实测机器人/对象质量、COM、惯量矩阵和PD已保存，惯量正定；旧legacy摩擦接口与新摩擦接口差异保留在A加载参数。没有新的轴/质量/惯量根因证据。旧PGS/velocity iterations/armature失败只作为优化7历史排除项，不算本次实验。保留bottom/top、rotation近固定关节、物理dt1/120及原USD；未启用RigidObject。

未找到可接受修复，停止更大环境、多seed安全矩阵、脚本抓握正对照和正式PPO。下一步应先取得GPU articulation真实shape pair/约束冲量证据，以这个同输入fixture量化剩余table/solver耦合；不得采用关掉桌面碰撞或仅降低峰值的配置。
''')
# 实际命令与同一时刻代码对应；所有Isaac进程串行。
commands='''# 优化8实际执行命令

工作目录：`/home/windsky/project/Grasp1`。所有仿真串行；GPU为3070Ti Laptop8GB，driver580.178.04。

```bash
source /home/windsky/miniconda3/etc/profile.d/conda.sh
conda activate grasp
export TERM=xterm PYTHONUNBUFFERED=1
```

## Phase 0

```bash
git status --short
git rev-parse HEAD
nvidia-smi
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --help
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization7/A/minimal_repro.py --headless --num_envs 1 --decimation 2 --episode 4 --duration 0.2 --modes zero fixed --output_dir logs/acceptance/optimization8/baseline/teacher_smoke
```

初次importlib.metadata查询isaacsim失败：预构建安装没有该分发元数据，改读实际VERSION。首次help因TERM=dumb导致tabs退出，按文档export TERM后重试通过。没有改依赖/官方安装。基线diff为空；初始仅用户优化8.md未跟踪。

## Phase A

每次保存1200个物理步，原始stdout在对应同名.log。前两个audit曾使用空BBoxCache范围，后改顶点AABB；前期使用callback，后期使用立即API，均未得到可用报告。这些是诊断采样变更，没有作为修复。

'''
for row in manifest['experiments']:
    name=row['experiment_id']
    base='/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/A/physics_probe.py --headless --num_envs 1 '
    if name=='D2_historical_random_target_replay':
        cmd=base+'--decimation 2 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/C/E2_screen/initial_random.pt --replay logs/acceptance/optimization7/C/E2_screen/trace_random.pt --source_env 7'
    else:
        cmd=base+'--decimation 4 --episode 10 --duration 10 --modes replay --initial_state logs/acceptance/optimization7/A/baseline_D4/initial_zero.pt --source_env 7'
        if 'object_off' in name:cmd+=' --isolation no_object'
        if 'table_off' in name:cmd+=' --isolation no_table'
        if name.startswith('base_filter'):cmd+=' --collision_variant base_filter'
        if name.startswith('box'):cmd+=' --collision_variant box'
        if name.startswith('hand150'):cmd+=' --hand_stiffness 150'
    commands+='```bash\n'+cmd+' --output_dir logs/acceptance/optimization8/A/'+name+'\n```\n\n'
commands+='''```bash
python logs/acceptance/optimization8/A/summarize_physics.py
```

## B/C/E/H独立检查

```bash
python logs/acceptance/optimization8/E/source_forward_audit.py
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task GraspTrain-v1 --num_envs 1 --decimation 4 --static_checks --output_dir logs/acceptance/optimization8/H/new_task_final_object_contract
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task GraspTrain-v1 --num_envs 16 --decimation 4 --output_dir logs/acceptance/optimization8/C/new_D4_16
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task GraspTrain-v1 --num_envs 16 --decimation 2 --output_dir logs/acceptance/optimization8/C/new_D2_16
/home/windsky/IsaacLab/isaaclab.sh -p logs/acceptance/optimization8/C/interface_check.py --headless --task Teacher-v0 --num_envs 1 --decimation 2 --output_dir logs/acceptance/optimization8/H/old_teacher_final
python logs/acceptance/optimization8/H/legacy_protocol_tests.py
/home/windsky/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --headless --task Grasp1-UR5-Allegro-GraspTrain-v1 --num_envs 16 --decimation 2 --cfg job
python -m compileall -q source scripts logs/acceptance/optimization8
python logs/acceptance/optimization8/finalize_report.py
git diff --check
```

第一次C static检查在new_D4_1.log失败，通用USD依赖枚举无法解析Kit内置OmniPBR.mdl。契约补入真实`${kit}/mdl/core/Base/OmniPBR.mdl`字节后new_D4_1_retry通过，最终H/new_task_final_object_contract对新旧契约、heldout评估允许更换对象及normalizer/网络差异重新验证。每次失败/重试目录保留。

`--cfg job`只检查真实Hydra注册/配置解析，未运行train主循环或PPO。旧protocol脚本复制至优化8/H并改产出路径，未写优化7。

## 未执行

D正式短PPO、128/512/1024/2048吞吐比较、E物理迁移、F8–10小时训练、G抓取/MVP/90次ShapeNet和工程保持均因A失败未运行。不存在合格best/last；不提供指向虚构checkpoint的评估命令。
'''
(root/'commands.md').write_text(commands)
report='''# 优化8执行报告

执行日期：2026-10-12；项目`/home/windsky/project/Grasp1`。**task_complete=false：Gate A真实复现失败，没有采用物理修复；正式训练与抓取/抬升评估均未运行。** 依据优化8的硬前置停止D/F/G，完成可独立验证的B源审计/合成方向测试、C新Task/接口/契约及E源网络CPU映射。

## 阶段与结果

| Phase | Gate | 本次实际完成 | 被阻断/尚未完成 |
|---|---|---|---|
| 0 | pass | HEAD/diff/版本/GPU/历史证据、旧Teacher smoke、真实CLI | 无 |
| A | fail | 8组真实1env×10秒同初态/target诊断，120Hz状态、AABB、参数和峰窗 | 根因确证、有效修复、修复后1→16→128多seed Gate、脚本抓握 |
| B | blocked | 原17项对应表、方向/时间/终止合成测试、1165真实但未合格接触样本、两profile | A通过后的真实分布与符号/逐contact近似验证；不调权重 |
| C | blocked | 独立原生新Task，新D4×1/16及D2×16短接口、部分reset、契约正反例 | 合格物理、窄XY/yaw/预抓取、128env及脚本抓取正例 |
| D | blocked | 状态文件，PPO updates=0 | 脚本闭合/抬升、500–1000更新、quick-lift、吞吐 |
| E | partial_cpu_pass | 源checkpoint每层/strict加载，64固定输入actor/critic误差0 | 真实PhysX迁移，fine-tune |
| F | blocked | 明确停止记录 | 8–10小时正式训练，课程、best/last |
| G | blocked | attempts=0、success_rate=null，MVP/源协议/工程协议分开 | 独立实际抓取/90attempt/5秒保持 |
| H | partial_checks_pass | compile、diff、旧/新Task smoke、方向/契约/CLI回归 | 修复后完整物理/训练最终验收 |

正式训练更新=0、transitions=0、墙钟=0秒；本轮诊断PPO更新也为0。best/last=null。formal/MVP/quick-lift attempts=0；成功率null，不能写0%。physics_stable=false、RL_training_completed=false、MVP_grasp_success=false、source_40s_evaluated=false、engineering_hold_evaluated=false、paper_reproduced=false。

## 基线与边界

HEAD=`5eae098f0ed8c11335c1b2c06a31b8a4dea953b0`，初始源码diff为空，只有用户优化8.md未跟踪。Conda grasp，Python3.11.15，Torch2.7.0+cu128，RSL-RL3.1.2，IsaacLab VERSION2.3.2/包0.54.2，IsaacSim预构建5.1.0-rc.19，driver580.178.04。实际launcher/Conda路径与文档一致。

物理步长始终1/120秒；旧Teacher仍D2/4秒。旧Teacher、优化1–7、旧checkpoint、USD没有覆盖，没有Git commit/push，没有升级安装。历史优化5约17–18k FPS/2000更新critic恶化/A1/90与B2/90、优化7 120.81456rad/s/0.277936rad和无物体30.95708rad/s/0.08780555rad均作为旧证据标记，未冒充本次成绩。

## A：物理前后证据

| 当前1env同fixture D4/10秒条件 | 手部qvel峰rad/s | 最大实际越限rad | 接受判断 |
|---|---:|---:|---|
| 原样桌面、物体移开，重复两次 | 30.957081 | 0.08780555 | 失败，两次q/qvel/target逐帧完全相等 |
| 仅过滤base_link/Table安装碰撞 | 30.957081 | 0.08780555 | 拒绝，无改善 |
| 同尺寸单Box、摩擦1/1 | 12.232162 | 0.07874259 | 拒绝，降峰不等于修复 |
| 仅hand stiffness600→150 | 38.485245 | 0.07981350 | 拒绝，仍严重异常 |
| 移开桌面、保留物体 | 0.282938 | 1.86265e-6 | 仅诊断负对照，禁止作为正式场景 |
| 桌面和物体都保留 | 9.590271 | 0.01650780 | 失败，有真实接触 |

D2回放旧random目标得到15.715107rad/s、越限max0.00314794/p99.99=0.00176542rad，后者超过0.001门槛。无物体主fixture joint15最大越限第1024帧、连续>0.001rad持续3.3333秒；速度峰第1197帧早于timeout1200，位置差分−0.05482rad/s，而PhysX报30.95708rad/s。实际link位移和全部q/qvel/target等保存；未clip物理统计。

1env复用了16env实际初态与目标，GPU求解布局不同，**不宣称112.90/120.81历史峰精确重现**。有物体分支9.590不是旧8.619逐帧精确复现；无物体关键fixture严格重现。

运行时27个collider包含instance proxy；底座/桌面AABB近似贴合，不能据此认定安装约束致因。PhysX回调与立即API均为空，即使Tensor Sensor有55.51/150.43N，shape pair诊断路径还未可用。安装对过滤、Box、PD均不能解掉主异常；**根因仍未确立**。mass/COM/惯量/PD在C加载参数保存，所测惯量正定，不能据此排除约束耦合。

详见[A诊断](../logs/acceptance/optimization8/A/diagnosis.md)、[物理Gate](../logs/acceptance/optimization8/A/physics_gate.json)、[前后CSV](../logs/acceptance/optimization8/A/before_after_physics.csv)、[峰值CSV](../logs/acceptance/optimization8/A/peak_cases.csv)。所有实验覆写仅本次诊断进程，默认配置未采用这些候选。

## B：Reward定义、单位与验证

保留旧17项及其权重，新增source_legacy/source_physics_aligned可选profile。source_legacy保留旧实际Reward行为；aligned仅把affordance冲量从摩擦模改为世界XY法向+摩擦总向量模，把push从法向模改为有符号世界Z，源push系数仍0。冲量只乘一个physics_dt；没有多乘decimation。source_coeff/0.2仅为名义每模拟秒强度对齐，不能称跨引擎接触逐帧等价。

源单contact先过滤<0.001Ns、再body聚合>0.01Ns，Sensor聚合无法精确重建；保留近似声明。终止−10按一次事件，timeout/invalidobs不额外扣；源reward.clip返回未保存，不引入总Reward下限。当前是grasp-only，保留obj_vel−15/位移−5，不宣称策略学主动lift；没有启用isaac_grasp_v1附加项或调权重。

合成输入直接调用真实Reward函数，覆盖接近、手/桌、arm接触、thumb+other接触计数、斜向worldXY、signedZ、速度、top/reset-root位移、D2/D4时间及terminal。未把多body接触计数当作对向夹持/force closure。真实但未通过A的1165接触样本里，weighted raw affordance冲量legacy/aligned平均绝对差约0.00905/0.00926，最大0.11640/0.16546；只记录定义差异，不作训练效果结论。两频率短接口还导出全部active分项raw、每步/每秒值及p10/p90/p99；没有critic训练，因此value target/returns=N/A。

[17项源对应表](../logs/acceptance/optimization8/B/reward_source_mapping.md)、[方向测试](../logs/acceptance/optimization8/B/reward_validation.json)、[实际方向投影](../logs/acceptance/optimization8/B/contact_projection.csv)、[诊断分布](../logs/acceptance/optimization8/B/reward_component_distribution.csv)。B整体blocked，缺少A-qualified真实轨迹。

## C：必要源码改动

| 文件 | 最终实现 |
|---|---|
| `grasp_train_env_cfg.py`（新增） | Teacher兼容子类，独立GraspTrain-v1，wood_block候选池可复制，Articulation/top/153D/22D；D4首选、D2对照；当前课程C1_pending_physics |
| `rsl_rl_grasp_train_ppo_cfg.py`（新增） | 独立日志实验名、LR3e−4候选、100保存间隔，原RSL-RL网络/runner复用 |
| `ur5_allegro/__init__.py` | 注册新旧Task并存、标准ManagerBasedRLEnv/RSL入口 |
| `mdp/rewards_grasp_v1.py`（新增） | 两profile及世界XY/signedZ修定义，旧Reward公式保持，纠正原注释对source worldXY的误述 |
| `mdp/rewards.py` | 仅更正legacy摩擦模与源worldXY总冲量差别的注释 |
| `grasp1_env_cfg.py` | control_dt消费时断言physics_dt=1/120，禁止覆盖后混用时间 |
| `scripts/rsl_rl/evaluate.py` | arm脚本目标按实际Action term scale反算action，支持D2/D4新尺度；旧默认等价 |
| `scripts/rsl_rl/checkpoint_contract.py` | 旧schema1继续原行为；新schema2追加Reward/终止/normalizer/网络/场景/课程/资产字节与权重旁路SHA，严格续训与评估对象扩展区分 |

新Task重载synchronize_control_timing，所有现有train/play/evaluate/zero/random入口通过继承复用现有Teacher分支，自动同步D2/D4动作尺度和profile；未复制第二套训练循环，也未扩散多套入口分支。D4 Arm/Hand scale=0.0008333333/0.0025；D2=0.0004166667/0.00125，旧Teacher0.00125/0.00375保留。gamma/lam按真实control_dt同步。

新契约实测读取22个USD/mesh层/材质/纹理依赖（mesh数据内嵌USD），含远程桌面与Kit内置OmniPBR字节，能检出未提交资产变化。机器人/桌面共享依赖用于推理/续训严格匹配；训练对象依赖/课程/reset用于严格续训；推理对相同对象路径的资产变化仍拒绝，允许合法heldout评估对象替换。新权重最终hash写sidecar，未嵌回权重。合成正反例覆盖相同配置、D2↔D4、Action/Reward/normalizer/网络/课程/共享资产/对象资产、USD字节和权重字节变化。旧schema1七例与CLI八例通过。

这些只完成独立C接口/契约工作，**C Gate没有通过**：物理修复、窄初始化、脚本抓取正例和128env验证未完成。新配置仅诊断候选，不是安全正式训练配置。checkpoint_fixture是合成文件，不是训练策略；累计wall/真实loaded physics/checkpoint optimizer恢复须在物理通过后的原生训练验证，本轮未声称完成。

## E与最终检查

公开本地full_12500_r.pt，153→128→128→22 actor、153→128→128→1 critic，LeakyReLU负斜率0.01，逐key/shape严格映射到本机RSL-RL，64固定CPU输入误差均0。source observe_vision_new直接拼接102D状态+51D旋转向量，不使用normalizer；CSV存在不代表这个路径消费。Gaussian std同步，optimizer未迁移。未运行源PhysX策略试次、BC或示范生成。[CPU审计](../logs/acceptance/optimization8/E/source_checkpoint_audit.json)、[前向对照](../logs/acceptance/optimization8/E/forward_equivalence.json)。

旧Teacher及新Task真实短smoke、部分reset/action/obs/reward/top检查通过；合成Reward和契约、真实Hydra注册/配置、CLI负例、compileall与git diff通过。H只对已实现代码通过，完整物理/训练验收被A阻断。运行产出都在[optimization8](../logs/acceptance/optimization8/summary.json)，源码最终diff和静态检查记录在H。

## 未完成项与下一步

先在保存的无物体/table-on同输入fixture取得可解释的GPU碰撞shape/约束冲量，修复且重复通过物理门槛，再进入1/16/128×D4/D2×4/10秒×多个seed安全矩阵和脚本抓取。随后才能B合格分布、C1窄初态、D短PPO/quick-lift、F8–10小时及G独立MVP/90试次。当前不能用移开桌面、放宽限位、一次局部降峰、finite或CPU权重可映射替代上述门槛。

[全部复现命令和失败重试](../logs/acceptance/optimization8/commands.md)、[阶段Gate](../logs/acceptance/optimization8/gates.json)、[汇总](../logs/acceptance/optimization8/summary.json)。
'''
Path('docs/优化8_执行报告.md').write_text(report)
print('OPT8_REPORT_SAVED',summary['task_complete'],states)
