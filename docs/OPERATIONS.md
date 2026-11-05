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

当前连续训练使用 `course_gate_mode="report"`、`stop_after_gate=false`：记录课程表现，普通失败和未达标继续学习。跳跃高度阶梯仍按各高度档的实际表现控制难度。不要把下文可选的逐门槛停训编排用于这条连续训练线。

```bash
# 训练（新运行或 --resume 完整 checkpoint）
SUPERMIMIC_SCRIPT=scripts/train_ppo.py bash scripts/run_gpufree.sh train \
  --factory-config configs/gpufree/ppo-gated4-fullturn-factory4096.json \
  --config configs/gpufree/ppo-gated4-fullturn-resume-r5.json \
  --resume <checkpoint.pt> --output <output>

# 分技能评价（factory 含 evaluation 段、desynchronize_episodes=false；无跳跃助力）
SUPERMIMIC_SCRIPT=scripts/evaluate_ppo_skills.py bash scripts/run_gpufree.sh \
  --factory-config configs/gpufree/ppo-gated3-fullturn-skills64.json \
  --checkpoint <checkpoint.pt> --steps 1000 --output <output>

# ONNX
python scripts/train_ppo.py export --checkpoint <checkpoint.pt> --output teacher.onnx
```

输出：`training/metrics.jsonl`（每更新一行，含门槛状态与原生技能计数增量）、`training/checkpoint.pt`（完整可恢复）、`training/checkpoints/update-*.pt`（轻量）。门槛超时会保存后以 `gate_timeout` 结束，必须先诊断再续训。

固定场景评价默认使用 Actor 均值；`--action-sampling ppo-gaussian --sampling-seed <seed>` 可单独检查 checkpoint 的训练探索噪声，动作 RNG 与环境 RNG 分开。`recovery_diagnostics` 按回合记录交接/超时、各物理和估计判据及最长稳定时间；`recovery_phases` 将实际恢复与交接后的跟踪分开。`jump_diagnostics` 按请求记录峰高、上升/离地证据、实际助力冲量和最长落地静稳，并分开协议结束与物理认证。不完整或无效区间保留诊断标记，不计作成功。

需要排除原生提前交接造成的恢复认证截断时，在评价 factory 的 `evaluation` 段显式设置 `"recovery_qualification": true`（默认关闭）。固定恢复场景从每次出生起，在原生超时预算内保持事先给定的零平移/零偏航率与该场景恢复高度；预算按策略拥有控制权的物理子步累计，当前配置为 5 s，零力矩释放不消耗预算。原生交接后仍继续独立检查相同物理判据与连续稳定时间，成功也不会提前释放固定指令。

该选项只在原生命令准入之前覆盖请求，保留状态机、随机数、限幅和 slew；出生首帧仍采用原生初始化高度，非默认目标高度从后续命令更新进入原生准入链。报告的 `recovery_diagnostics.qualification` 与原生交接/物理成功计数分开，逐回合列出成功、失败、超时及不完整/无效证据；资格事件须等待完整控制区间有效后才确认。`scheduled_segment` 此时也包含固定资格请求覆盖。此独立评估不会修改课程门槛或训练奖励，排除无效/不完整回合后的成功率不能单独作为晋级依据。

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

上述可选的历史编排要求 `stop_after_gate=true`：通过门槛立即保存不可变 checkpoint，退出该 Isaac 进程，完成 64×1000 无助力的 27 场景评价后才续训。已要求技能缺少物理成功时返回 `diagnostic_required`，进程失败时保留待评价 checkpoint；`--continue-run` 只恢复中断的编排，不跳过技能失败。它不适用于当前 `report` 连续训练。固定评价没有 200/300 mm 跳跃案例，这两档仍看训练中的无助力计数。`budget_complete` 仅表示预算结束，完整能力仍须审阅各场景结果。

## SuperMimic（世界模型 V2）

当前在线学生使用 `guard_mode="report"`，回报监控只供评审，不回滚策略或降低学习率；恢复后在原有策略预热结束的前提下立即恢复统计。单独设置 `course_coordinate` 会固定课程采样位置，增加 cycles 不会自动获得后续台阶/地形数据。可从完整断点启用 `course_progression`，沿原生阶段顺序和比例推进采样；跳跃高度独立按实际结果提升，其他阶段继续。配置与恢复语义见 [世界模型课程接续](WORLD_V2_COURSE_PROGRESS.md)。阶段走完不代表能力通过，也不会自动补齐尚未建模的外部推扰。

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

在线恢复会重新初始化整批物理出生状态。启用原生回报 guard 时，每次 cold resume 复用 `policy_start_cycle` 预算，前相应数量的新周期暂缓回退判定及最佳快照替换，训练更新照常进行。已有 best 和回退次数保留，重置前的连续坏周期计数被记录并清零，避免跨出生等待期拼成连续退化；`guard.deferred_reason`、`remaining_cycles`（该周期结束后的剩余数）和 `cold_resume_previous_bad_cycles` 标明过程。预算为 0 时保持原行为，新运行仍按原有绝对 `policy_start_cycle` 启动。

数据报告读取 checkpoint 中已有的接触滤波阈值，在同一批重新采样的留出窗口上同时报告原始预测和滤波后的接触事件 precision/recall；没有正样本时 recall 为 null。各预测时距的瞬时事件统计覆盖末尾区间四个物理子步；提供 SHA256 匹配的 `--runtime-json` 后，另按原生确认时间统计整个预测窗口中的接触终止误报/漏报。报告不会重拟合滤波阈值，也不会把滤波后的载荷误差冒充原始模型误差；报告样本与标定共享留出环境总体，不能据此声称使用了独立测试集。

完整在线配对报告通过只读 mmap 视图读取 replay，避免再分配一份完整训练回放；采样窗口仍拥有独立存储。训练恢复默认复制数据，不受此选项影响。读取活跃运行时，应依据最后写入的 `resume.json` 固定同一代 checkpoint/replay，避免两槽轮换期间混用不同循环；报告记录对应循环和 replay 文件名。

离线续拟合可用 `fit_world_v2.py --online-checkpoint <resume.pt> --contact-event-fraction <fraction>` 增加接触事件窗口的采样比例，默认值为 0。事件定义沿用原生两轮支撑载荷 ≤2 N、连续两个 5 ms 子步，包含跨控制步的连续事件；合法窗口和训练/留出环境划分保持不变。其余窗口均匀采样，只有均匀部分更新归一化统计，整个 batch 使用原损失；采样比例需要根据留出集运动与接触指标验证。

`--wheel-contact-weight <weight>` 可额外单独归约双轮连续载荷的归一化 MSE，默认 0 保持原损失。该项沿用每个子步的输出尺度与误差下限，按两轮、四个子步及拟合时域取均值；总损失为 `state + ticks + weight * wheel_contact`。候选值 1 表示额外组的系数为 1，叠加原有 56 维 ticks 损失后，双轮通道相对原项的权重为 29 倍；它是待验证的训练系数，不是物理参数。使用相同留出集同时检查运动与接触事件，离线配置和日志会保存权重及分项损失。续训继承已存权重，旧 checkpoint 缺省为 0；改变权重需从明确的在线源另建离线实验。在线训练默认不启用该项。

检查离线续拟合的模型时，在配对报告命令中增加 `--world-checkpoint <fit>.checkpoint.pt`。报告校验原始配对身份和训练/留出环境列表，记录新权重 SHA256，并报告原始接触预测；旧在线模型标定的滤波阈值不会沿用到新权重。

启用跳跃触发门控时，采集器额外记录上次请求结束时间与门控状态，想象按原生顺序检查上一时刻的指令高度、回合时间和冷却。Actor 与运动网络维度不变，Critic 追加三个任务特征。旧的无门控配对 checkpoint 可通过 `scripts/upgrade_online_jump_trigger.py --checkpoint <resume.pt> --reference-dataset <gated-native-replay.pt> --output <new-directory>` 显式扩展：保留运动监督，旧任务起点全部标为未知，重新采集后才用于想象；新增 Critic 列及 Adam 历史补零，再用新数据标定归一化。

这项迁移只允许新增门控及其声明的参考轨迹，不允许同时更换奖励、课程、物理或传感器合同。跨奖励/课程版本的旧运行还需要单独验证这些合同的迁移，不能直接替换来源标识后续训。

`scripts/upgrade_online_task_semantics.py` 专门处理 a05 到 R7 的任务语义变更。它要求真实目标原生数据、两份固定 runtime/course，以及在目标环境重新评价导出的同权重教师。runtime 只允许开启 ToF 模式标志；资产、物理、执行链、传感器必须相同，源和目标都不能包含助力或未建模外扰。参考轨迹以目标原生记录为准，短/长参考的选择仍由探针结果决定。

迁移保留世界模型、Actor、估计器及其优化器，历史 replay 继续提供物理监督；所有旧任务有效标志清零，不伪造新任务历史。奖励改变后重建 Critic 和 Adam、重设回报回退基线，并在新原生数据上校准完整 Critic 输入、执行显式预热。课程坐标仅指定采样位置，不代表能力晋级；历史采样计数和 RNG 保留，任务混合配额重新开始。

```sh
python scripts/upgrade_online_task_semantics.py \
  --checkpoint <a05>/resume-a.pt --reference-dataset <target-native>/replay.pt \
  --source-runtime <a05-runtime.json> --target-runtime <tof-runtime-without-disturbances.json> \
  --source-course <old-course.json> --target-course <v5-course.json> \
  --source-teacher <old-teacher.pt> --target-teacher <reevaluated-same-teacher.pt> \
  --course-coordinate <sampling-coordinate> --critic-warmup-cycles <warmup-cycles> \
  --sampler-device cuda:0 \
  --output <new-handoff-directory>
```

`--sampler-device` 必须与源课程采样器的设备类型一致；远端 a05 使用 `cuda:0`，需在可用 CUDA 的机器上迁移。CPU 来源使用默认 `cpu`。迁移保留原 RNG 字节，不将 CUDA Philox 状态转换成 CPU 状态或重新播种。

输出新的 checkpoint/replay 配对和配置，不覆盖源运行。启动前先用真实记录核验请求、冷却、ToF 和奖励链路，再检查按技能划分的预测与接触事件报告；软件迁移通过不等于跳跃、台阶或翻正已学会。完整推扰课程还需要为世界模型记录并推进外扰，不能直接复用 PPO 的有推扰 runtime。

带跳跃任务记录的数据可直接核验想象奖励组装：

```sh
python scripts/check_reward_assembly.py <target-native>/replay.pt \
  --terrain-atlas <atlas> --table configs/linkage/leg_linkage_08bf7e00.json \
  --asset <asset-directory> --bundle configs/identified/20261031 \
  --recorded-jumps --device cpu --output <reward-parity.json>
```

此检查用真实四子步物理状态替代网络输出，分别报告普通、跳跃、活动跳跃及 HOLD 未起跳区间与原生总奖励的误差。参考曲线仍从记录的任务记忆重建，不是独立记录四条原生参考曲线的对照。
