# 真实物理诊断：Gate A失败，未采用修复

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
