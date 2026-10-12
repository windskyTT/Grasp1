# 物理源码、加载参数及数据可见性审查

依据本地RobustDexGrasp源Teacher、Grasp1配置、IsaacLab 2.3.2安装源码和真实PhysX tensor读取。以`loaded_parameters.json`实际值为主；USD drive导入初值不能替代IsaacLab运行后写入的增益。

## 参数事实

- 120Hz物理，D2/D4只改变策略间隔；正式TGS，robot solver position/velocity=32/1，object=16/0。机器人self-collision沿用配置，无资产变更。
- Allegro运行时16joint均stiffness600、damping20、effort0.7Nm、velocity7rad/s、armature0。UR5保持源逐joint增益、effort150/28Nm与velocity3.15/3.2rad/s。`applied_torque`是本机ImplicitActuator的PD近似诊断，不能当作真实约束/驱动力矩。
- `joint_15_0`父/子为`link_14_0`→`link_15_0`，USD axis=Y、localPos0=(0,0,0.0514)。USD下/上限-9.281916°/98.491440°，runtime约-0.162/1.719rad；目标最终限位按runtime rad进行，无度/弧度转换错误证据。
- 木块body=`bottom`,`top`，mass=0.00098/0.24875kg。bottom惯量对角约1.6333e-8kg·m²，top约0.00102271/0.00031813/0.00102777。rotation限位0–0.001rad，zero PD，armature0.1，effort0.1、velocity0.1。保留两个body；轻bottom/近固定joint是条件数疑点，现有实验不足以断言它单独是根因。
- 现代`get_dof_friction_properties()`三分量为0，旧`get_dof_friction_coefficients()`仍返回手部10/5/10/12等导入值；joint15旧USD scalar为12。两个接口的零值不可混为“导入摩擦不存在”。清除旧摩擦单变量反而恶化，本轮没有把旧摩擦写成已证实错误。
- 源URDF六UR5 joint的`rotor_inertia=1.0`，加载的Isaac armature=0。RaiSim rotor inertia加到质量矩阵对角是已确认的源/迁移差异，证据见[官方RaiSim说明](https://raisim.com/sections/articulated_system/Creation.html)。真实D2/D4单变量armature1.0候选仍有大幅越限，不能当作本轮尖峰修复。Allegro没有同样明确的源rotor数值，0.0006/0.0012是诊断候选，未纳入默认。

## 碰撞资产与传感器

初次普通stage.Traverse()只得到joint而遗漏实例化碰撞，早期`usd_colliders=[]`不代表没有碰撞体。最终collector使用`Usd.TraverseInstanceProxies()`，`collider_audit_replay/loaded_parameters.json`记录env0 **27个CollisionAPI prim**：UR5 convexHull、Allegro box、fingertip convexHull、对象top convexHull及桌面convexHull等均存在。joint15附近box与tip未关闭。桌面contact/rest offset的USD -inf是未显式设置的schema默认值，不能据此直接认定无穷接触距离；没有读取到运行时求解器采用的默认offset值。

本机ContactSensor可给sensor body名、filter body路径、世界法向/摩擦聚合、接触点/法线/距离/force buffer及计数。`contact_detail`在选定env保存8个相关finger/table过滤传感器。无法从该Tensor API可靠返回每个contact的两侧具体collision-shape身份，未伪造shape pair或约束flags。robot base不在现有手/臂传感器覆盖范围；filtered/net=0只覆盖这些传感器，不能证明全场景无约束冲量。

`collider_audit_replay`再次得到table保留/no_object的30.957081rad/s、actual q越限0.08780555、已监测net/filtered均0。这证明此前接触解释仍不充分；资产存在性与关节轴/限位核对也没有发现足以直接修复的确定错误。

## 物理API解释边界

[NVIDIA articulation稳定指南](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html)说明PD、质量/惯量及armature会影响稳定性；[PhysX articulation文档](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html)用于核对约束与驱动含义。两者不是本模型安全证明。

[PhysX issue 498](https://github.com/NVIDIA-Omniverse/PhysX/issues/498)涉及另一版本/平台的TGS与摩擦差异，只作为排查线索。不能将其当成本机IsaacSim5.1 GPU异常已确诊，不能据此升级二进制或改安装目录。reported qvel与位置差分显著不一致已经实测；是否是TGS约束积分、未观测接触或模型问题仍`unresolved`。

结论：已得到可重复的小场景状态越限及现有模型的严重速度案例，已排除目标越限/子步目标更新和“只看现代摩擦为0”等误判；尚未完成具体物理根因定位、资产修复或修复后Gate。
