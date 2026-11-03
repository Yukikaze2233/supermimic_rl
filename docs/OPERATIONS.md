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

v5 的 `jump_height_gates` 把请求高度上限固定在 0.10 / 0.20 / 0.30 / 0.40 m；只有对应阶段内门槛通过，坐标跨过整数边界后才提升上限。各高度档只读取实际未施加助力冲量的成功/结束计数，不能用全部请求的成功率替代。力幅衰减与请求抽样分开：`assist_request_probability=0.8` 是候选训练比例，保留 20% 请求不施加助力，避免课程等待时助力也被冻结、永远收集不到无助力证据；它不是电机或机械参数。剩余 80% 请求仍按课程坐标衰减力幅，最终同样归零。门槛窗口、成功率和该比例需依据分高度档样本量与原生表现调整。

R7 启动前分别运行 `short`、`long` 探针，使用相同完整 checkpoint 和各自的新输出目录。400 更新对应每环境 192 s 仿真曝光；这是诊断预算，不是技能完成判据。未到达的高度档报告为未知。

```bash
SUPERMIMIC_SCRIPT=scripts/probe_jump_reference.py bash scripts/run_gpufree.sh \
  --factory-config <r7-training-factory.json> --config <r7-ppo.json> \
  --checkpoint <r6-full.pt> --checkpoint-sha256 <sha256> \
  --profile short --updates 400 --stage-budget-updates 400 --output <new-probe-output>

# 选定参考参数后，训练与评价 factory 的 jump_takeoff 必须一致。
python scripts/run_ppo_gated_course.py \
  --factory-config <r7-training-factory.json> --evaluation-factory-config <r7-skills64-factory.json> \
  --config <r7-ppo.json> --evaluation-contract configs/gpufree/ppo-gated5-evaluation.json \
  --resume <full-checkpoint.pt> --output <new-course-output>
```

正式编排要求 `stop_after_gate=true`：通过门槛立即保存不可变 checkpoint，退出该 Isaac 进程，完成 64×1000 无助力的 27 场景评价后才续训。已要求技能缺少物理成功时返回 `diagnostic_required`，进程失败时保留待评价 checkpoint；`--continue-run` 只恢复中断的编排，不跳过技能失败。固定评价没有 200/300 mm 跳跃案例，这两档仍看训练中的无助力计数。`budget_complete` 仅表示预算结束，完整能力仍须审阅各场景结果。

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
  --linkage configs/linkage/leg_linkage_08bf7e00.json --runtime-json <runtime.json> --output <report.json>
```

在线运行每 `replay_checkpoint_interval` 轮写入一对轮换的完整 checkpoint（回放约 15 GB/份，需预留两份空间）。推扰只放在 PPO runtime 中；世界模型采集使用无推扰 runtime，因为模型输入看不到推扰。

数据报告读取 checkpoint 中已有的接触滤波阈值，在同一批重新采样的留出窗口上同时报告原始预测和滤波后的接触事件 precision/recall；没有正样本时 recall 为 null。各预测时距的瞬时事件统计覆盖末尾区间四个物理子步；提供 SHA256 匹配的 `--runtime-json` 后，另按原生确认时间统计整个预测窗口中的接触终止误报/漏报。报告不会重拟合滤波阈值，也不会把滤波后的载荷误差冒充原始模型误差；报告样本与标定共享留出环境总体，不能据此声称使用了独立测试集。

启用跳跃触发门控时，采集器额外记录上次请求结束时间与门控状态，想象按原生顺序检查上一时刻的指令高度、回合时间和冷却。Actor 与运动网络维度不变，Critic 追加三个任务特征。旧的无门控配对 checkpoint 可通过 `scripts/upgrade_online_jump_trigger.py --checkpoint <resume.pt> --reference-dataset <gated-native-replay.pt> --output <new-directory>` 显式扩展：保留运动监督，旧任务起点全部标为未知，重新采集后才用于想象；新增 Critic 列及 Adam 历史补零，再用新数据标定归一化。

这项迁移只允许新增门控及其声明的参考轨迹，不允许同时更换奖励、课程、物理或传感器合同。跨奖励/课程版本的旧运行还需要单独验证这些合同的迁移，不能直接替换来源标识后续训。
