可以开始 Teacher 训练，现有训练入口、任务注册、PPO 配置、依赖和资产路径都已接通，不需要再添加代码。

推荐使用完整的 88 个环境：

```bash
cd /home/windsky/project/Grasp1
conda activate grasp

python scripts/rsl_rl/train.py \
  --task Grasp1-UR5-Allegro-Teacher-v0 \
  --num_envs 88 \
  --max_iterations 20001 \
  --device cuda:0 \
  --headless
```
```bash
cd /home/windsky/project/Grasp1
conda activate grasp

python scripts/rsl_rl/train.py \
  --task Grasp1-FR3-Inspire-Teacher-v0 \
  --num_envs 88 \
  --max_iterations 20001 \
  --device cuda:0 \
  --headless
  ```

默认配置：

- 50,001 次迭代
- 每次 rollout 为 70 步
- 默认关闭 biased 扰动
- 默认关闭训练中定期评估
- 日志和 checkpoint 位于：

```text
/home/windsky/project/Grasp/runs_teacher/<任务名_时间>/
```

3070Ti 8GB 如果出现 CUDA 显存不足，可以降低为：

```bash
cd /home/windsky/project/Grasp1
conda activate grasp

python scripts/rsl_rl/train.py \
  --task Grasp1-UR5-Allegro-Teacher-v0 \
  --num_envs 44 \
  --max_iterations 20001 \
  --device cuda:0 \
  --headless
```

启用 biased 扰动时增加：

```text
--biased
```

续训命令：

```bash
python scripts/train.py \
  --task Grasp1-UR5-Allegro-Teacher-v0 \
  --num_envs 88 \
  --device cuda:0 \
  --headless \
  --checkpoint /home/windsky/project/Grasp/runs_teacher/<运行目录>/full_500_r.pt
```

play指令
```bash
cd /home/windsky/project/Grasp1
conda activate grasp

python scripts/rsl_rl/visual_eval.py \
  --task Grasp1-UR5-Allegro-Teacher-v0 \
  --checkpoint /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_ur5_allegro_teacher/2026-10-22_22-17-39_751671/model_11480.pt \
  --num_envs 1 \
  --episodes 150 \
  --device cuda:0 \
  --real-time
```
```bash
cd /home/windsky/project/Grasp1
conda activate grasp

python scripts/rsl_rl/visual_eval.py \
  --task Grasp1-FR3-Inspire-Teacher-v0 \
  --checkpoint /home/windsky/project/Grasp1/logs/rsl_rl/grasp1_fr3_inspire_teacher/2026-10-25_21-51-12_133017/model_20000.pt \
  --num_envs 1 \
  --episodes 150 \
  --device cuda:0 \
  --real-time
```



当前确认结果包括 Isaac Lab `v2.3.2`、Isaac Sim 5.1.0、Grasp editable 安装、训练依赖和训练入口静态检查通过。本轮遵守项目限制，没有实际启动训练；如果启动后产生报错，将内容写入 `错误.md` 后再让我读取处理。


TensorBoard 可视化 Reward 曲线
```bash
cd /home/windsky/project/Grasp1
conda activate grasp 
tensorboard --logdir logs/rsl_rl --port 6006
```