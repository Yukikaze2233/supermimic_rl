# 运行说明

## 本地测试

测试不需要 Isaac，但需要上游 `transformer_rl` 源码与 PyTorch（`src` 与 `tests` 由 pyproject 加入路径）：

```bash
pip install -e '.[test]'
export PYTHONPATH=<transformer_rl>/src
python -m pytest -q
```

## 上游补丁

当前课程路径（`supermimic_rl.course`）使用未修改的固定上游源码。`patches/` 中的两个补丁只用于旧路径的源码副本：
`upstream-jump-height-ownership.patch` 修复旧平地环境 `JumpFSM` 的站高张量共享，
`full-vehicle-bootstrap.patch` 为旧整车 bootstrap（`adapters.factory`）开放启动动力学构造。

## Isaac 镜像上的作业

镜像自带 `/isaac-sim` 与 `/opt/isaaclab-venv`。Isaac 相关脚本必须经启动器运行（它设置 libpython、Kit 与 CUDA 环境；直接调用 venv python 会初始化失败）：

```bash
SUPERMIMIC_SCRIPT=scripts/<script>.py bash scripts/run_gpufree.sh <script arguments>
```

`SUPERMIMIC_ROOT` 指向要运行的源码目录。长作业约定：每次运行使用一份不可变的源码快照（打包后校验 sha256 再解包到新目录），在 tmux 中启动，输出目录不复用；factory JSON 用 sha256 钉住 runtime、course 与资产。

只读工具（拟合、数据报告）不需要 Isaac 环境，但需要 `PYTHONPATH` 中包含上游源码；GPU 设备写成 `cuda:0`。

## PPO 教师

```bash
# 训练（新运行或 --resume 完整 checkpoint）
SUPERMIMIC_SCRIPT=scripts/train_ppo.py bash scripts/run_gpufree.sh train \
  --factory-config configs/gpufree/ppo-gated4-fullturn-factory4096.json \
  --config configs/gpufree/ppo-gated4-fullturn-resume-r5.json \
  --resume <checkpoint.pt> --output <output>

# 27 场景分技能评价（factory 需含 evaluation 段；评价永不安装跳跃助力）
SUPERMIMIC_SCRIPT=scripts/evaluate_ppo_skills.py bash scripts/run_gpufree.sh \
  --factory-config configs/gpufree/ppo-gated3-fullturn-skills64.json \
  --checkpoint <checkpoint.pt> --steps 1000 --output <output>

# ONNX
python scripts/train_ppo.py export --checkpoint <checkpoint.pt> --output teacher.onnx
```

输出：`training/metrics.jsonl`（每更新一行，含门槛状态与原生技能计数增量）、`training/checkpoint.pt`（完整可恢复）、`training/checkpoints/update-*.pt`（轻量）。门槛超时会保存后以 `gate_timeout` 结束，必须先诊断再续训。

## SuperMimic（世界模型 V2）

```bash
# 1. 教师导出（含原生评价证据）
python scripts/export_ppo_teacher.py --checkpoint <ppo.pt> --evaluation <evaluation.json> \
  --qualified-for <scope> --output <teacher.pt>
# 2. 种子数据（固定课程坐标，记录任务状态）
SUPERMIMIC_SCRIPT=scripts/collect_teacher_v2.py bash scripts/run_gpufree.sh \
  --factory-config <wm-factory.json> --checkpoint <teacher.pt> --checkpoint-sha256 <sha> --course-coordinate <c> \
  --record-tasks --record-stair-source --record-jumps --record-recovery --steps 300 --output <seed>
# 3. 离线拟合
python scripts/fit_world_v2.py <seed>/replay.pt --linkage configs/linkage/leg_linkage_08bf7e00.json \
  --terrain-atlas <atlas> --device cuda:0 --output <fit>
# 4. 在线训练（学生从 PPO 精确初始化，见在线配置 initial_actor）
SUPERMIMIC_SCRIPT=scripts/train_world_v2.py bash scripts/run_gpufree.sh train \
  --factory-config <wm-factory.json> --config <online.json> --teacher <teacher.pt> --teacher-sha256 <sha> \
  --seed-dataset <seed>/replay.pt --world-checkpoint <fit>.checkpoint.pt \
  --linkage configs/linkage/leg_linkage_08bf7e00.json --output <online>
# 5. 数据覆盖与预测报告（无 Isaac）
python scripts/report_world_v2_data.py --online-checkpoint <online>/training/resume-a.pt \
  --linkage configs/linkage/leg_linkage_08bf7e00.json --output <report.json>
```

在线运行每 `replay_checkpoint_interval` 轮写入一对轮换的完整 checkpoint（回放约 15 GB/份，需预留两份空间）。推扰只放在 PPO runtime 中；世界模型采集使用无推扰 runtime，因为模型输入看不到推扰。
