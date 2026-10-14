# 优化9物理诊断：触发碰撞对已定位，安全修复未通过

PhysX始终120 Hz，全部使用保存初态和保存目标。原样回放与优化8的q/qd/target逐帧完全相等；加入被动采样后仍完全相等。其他D4候选的实际目标最大误差≤7.46e-9 rad，D2≤9.32e-10 rad。没有用新随机动作替代历史输入。

## 已得到的因果证据

原13个手部槽只覆盖finger非固定刚体，不含4个独立tip刚体。补测全部32个机器人刚体后，无物体/有桌面回放在第470物理步（3.916667秒）发生thumb tip/table接触，峰值69.092651 N；原32个Teacher传感器对此场景均为0。唯一非零机器人桌面接触是link_15_0_tip。

仅过滤该tip/table碰撞对，保留桌子、重力和相同初态/目标后，手部速度峰30.957081→0.282938 rad/s，实际越限0.08780555→0.00000186 rad。这个对照定位了触发接触；**必要手桌碰撞不得作为正式修复删除**。原回调和get_contact_report仍为空，未据此推断无接触；刚体身份来自GPU Tensor API，tip和Table分别只有一个已审计碰撞体，shape路径映射来自USD审计，未声称获得原生GPU shape/constraint impulse回调。

get_link_incoming_joint_force记录完整6D反力，get_dof_projected_joint_forces记录沿运动轴分量。它们包含接触/约束反力，不能把它们当成精确隐式电机输出力矩。原torque_approx是裁限PD估计，精确分离的implicit-drive力矩当前API未给出。力→冲量只使用一个1/120秒物理dt。

## 候选结果

| 同fixture条件 | 手qvel峰rad/s | 越限p99.99 rad | 越限max rad | 决定 |
|---|---:|---:|---:|---|
| original_replay | 30.957081 | 0.08779494 | 0.08780555 | 不采用 |
| all_body_contact | 30.957081 | 0.08779494 | 0.08780555 | 不采用 |
| tip_table_pair_isolation | 0.282938 | 0.00000184 | 0.00000186 | 仅诊断负对照 |
| friction005_replay | 14.490803 | 0.07364126 | 0.07364997 | 不采用 |
| position128_replay | 86.756943 | 0.00168572 | 0.00196877 | 不采用 |
| merged_thumb_retry | 6.102415 | 0.08631409 | 0.08632708 | 不采用 |
| position128_velocity8_replay | 11.240615 | 0.00166223 | 0.00180331 | 不采用 |
| position128_velocity32_replay | 11.057040 | 0.00171761 | 0.00191623 | 不采用 |
| external_each_iteration_replay | 51.723156 | 0.06137899 | 0.06235142 | 不采用 |
| position255_velocity32_replay | 18.434395 | 0.00057208 | 0.00066465 | 不采用 |
| position255_velocity32_external_replay | 87.455124 | 0.00090683 | 0.00092153 | 不采用 |
| full_scene_replay | 9.590271 | 0.01631157 | 0.01650780 | 不采用 |
| D2_historical_random_replay | 15.715107 | 0.00176542 | 0.00314794 | 不采用 |

迭代试验按32P/1V→128P/1V→128P/8V→128P/32V→255P/32V推进，每条边只改变一个参数。外力逐迭代开关分别对原32P/1V和255P/32V单独对照。摩擦、指尖合并都是原默认32P/1V上的独立试验，没有将全部候选一起调参。

thumb合并不改旧资产：诊断USD另存，末节+tip质量0.0857924 kg；加载后机器人总质量前后完全相等，合并惯量最大误差7.24e-12 kg m²，22关节名/限位不变。它使遗漏接触被原末节传感器读到，但越限0.08632708 rad仍严重，不能采用。

255P/32V越限达门槛，但峰值帧reported qd=18.434395、差分qd=0.559625 rad/s；开启外力逐迭代后reported qd=87.455124、差分0.750350 rad/s。没有简单把7rad/s当独立绝对安全标准；当前大的速度/积分差异及接触反力增长尚未得到可接受解释，未认定稳定。增加迭代和外力开关不构成已验证修复。

官方资料指出TGS外力应用时序可能产生稳态速度与位置的差异：[PhysX场景开关](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/struct_px_scene_flag.html)。本机IsaacLab simulation_cfg.py:123及simulation_context.py:834同样建议该开关。实际原样对照并未解决本项目异常，不能把文档建议当成成功证据。legacy jointFriction是无量纲反力相关系数，[PhysX关节API](https://nvidia-omniverse.github.io/PhysX/physx/latest/_api_build/classPxArticulationJointReducedCoordinate.html)；源URDF的5/10/12与源引擎摩擦含义不可简单等同。0.05仅诊断，不能作为已标定的物理参数。

## Gate与下一步

Gate A=fail，无候选采用。1env已有硬失败，16/128env、多seed、4秒及完整Episode安全矩阵blocked；脚本夹持/PPO/完整策略GUI保持blocked。完整桌面+物体的原始场景仍有0.0165078 rad越限；D2历史random目标回放p99.99=0.00176542 rad仍超门槛。

下一步范围已经缩小到thumb tip/Table接触下的Allegro关节限位—驱动—TGS耦合：使用本轮接触点、法线、separation和child-joint-frame反力做定点约束诊断，取得可分离的contact/limit/drive impulse证据；另需在新任务中显式覆盖真实tip接触，保留旧Teacher语义。不能继续用删碰撞、只降峰或扩大门槛替代修复。

原始轨迹、加载参数、网格审计、tip接触短窗口、合并守恒检查和对照图均在本目录。没有裁剪q/qd，也没有使用差分qd替换GPU报告。
