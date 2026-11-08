# V2 解析翻正任务模块

任务模块为 `world_model_v2/recovery_task.py`；可选 `record_recovery` 接入采集、replay、online 与想象链路。
它复用 pinned upstream 的 `RecoveryLifecycle.observe_physics / finish_interval` 和
`LabCourseTasks.begin_tracking`，不新增翻正动作脚本、动力学网络、Actor 输入或奖励项。
同一个 Actor 负责策略接管后的四向翻正及随后的行驶；目标四向出生为左侧倒、右侧倒、
`prone_belly_legs_above` 与 `prone_back_legs_above`。

## 状态和配置

`RecoveryTaskConfig.capture(native)` 保存实际 `RecoveryLifecycleConfig`，以及命令池的
`handoff_s / handoff_forward_m_s / handoff_yaw_rad_s`。不使用另设的翻正判据或随机指令幅值。
无命令池的旧路径明确记录 `handoff=None`。当前模块要求 200 Hz 物理、50 Hz 策略。

`RecoveryTaskState.capture(native)` 必须在 `env.step()` 前捕获：

| replay 字段 | 类型／形状 | 内容 |
| --- | --- | --- |
| `recovery_flags` | bool `[N,2]` | `active`、`_verified` |
| `recovery_clock` | float64 `[N,5]` | 边界 `now_s`、策略接管后 elapsed、估计稳定累计、物理稳定累计、`_last_estimate_s` |

保留未使用估计的 `-inf` 时间戳语义；Critic 不接收该非有限占位值。采集同时保存
`recovery_record_valid` 标记，旧数据缺失的生命周期不能被默认补零当作真实记录。
以上状态每行 42 bytes，加一个 float32 valid 后，1000×4096 条记录增加约 0.18 GiB。
统计总计数不影响状态转移，无需复制到每条 replay。

`state.critic_features(config)` 为 7 维：active、verified、归一化策略 elapsed、估计／物理
稳定累计、上次估计是否存在、相对估计年龄。绝对时间平移不改变特征；无姿态类别 one-hot、
绝对位置或额外动作。Actor/WM 维度保持；完整 jump/source 分支的 Critic 从 401 扩到 408 维。

## 调用次序

1. 在当前 RECOVERY 模式下预测四个物理子步；`observe_interval()` 复用原生物理证据计算。
2. 共享解析 reward 仍按当前命令和 `RewardPhase.RECOVERY` 累计四步；物理失败先完成锁存。
3. `finish_interval(state, now_s=..., estimate=..., policy_owned=..., failed=...)` 使用**前一帧**
   完成的带噪声估计，产生 handoff、timeout 和物理 verified 的上升沿。
4. `apply_handoff()` 在 `MotionTask.advance_admission()` **之前**执行；handoff 行切换 NORMAL，
   原生 `begin_tracking` 更新轨迹和段期限。旧区间的奖励不改成 NORMAL 奖励。
5. 再推进下一边界的命令 admission、jump/stair 仲裁、云台任务和观测。

物理证据包含地形相对 root 高度、指令高度、倾角、角速度模、**base COM 世界 z 速度**、
轮接触**净力模**和 base **法向载荷**。COM 速度计算 `v_origin + omega × (R @ com_offset)`。
支持整个区间 `[N]` 或逐物理 tick `[N,4]` 的实际策略控制权；脚本拥有的准备段不积累稳定时长。

原生判断有两条独立链：估计稳定可以触发模式 handoff，物理稳定只能认证诊断成功。
因此“估计已经交接”与“物理确实翻正”必须分别统计。未通过物理认证的估计交接不能计为技能成功。

## 随机 handoff 和截断

原生 `begin_tracking` 会从命令池抽 vx/wz，设置 HOLD 轨迹、清零段时钟、将段期限置为
**旧** episode elapsed 加 handoff 持续时间，并关闭 heading。它不会立刻改写 `stream.current`；
下一次 admission 仍负责限速和流式命令更新。

`apply_handoff(boundary, motion, task_family=..., segment_end=..., handoff_uniform=...)`
接收整批预生成的 float32 `[N,2]`、取值 `[0,1)` 的 uniform。完整 batch、microbatch、
梯度重试复用同一组 draw。只替换原生命令池的随机输入，其余 mutation 语义仍调用原生方法。
命令高度连续梯度保留，随机 vx/wz 不携带梯度。无命令池时复用原生将 handoff 行 family 改为
HEIGHT 的行为；有池时 family 保持 RECOVERY，不能把这类正常行驶后继数据过滤掉。

这与原生的随机**分布**一致；若需要逐值核对实际发生的 handoff，还需记录并复放课程 RNG
或独立记录实际 handoff 指令，不能从后继指令反推后声称完成独立随机源验证。

原生 recovery timeout 属于 `truncated/time_limit`，**不属于物理 terminated**。模块只返回
`timeout`，不会把它加入失败 mask。接线应在相同边界截断想象并保留原生 bootstrap 语义。
`verified_success` 仅为诊断事件，原生没有翻正成功额外支付；不要把它接到 jump 的
`success_credited` 或新建成功奖励。现有共享 reward 已包含 upright、恢复高度与姿态等连续项。

## 主循环接线合同

- `collection/records/online`：可选记录开关、上述字段、来源配置及 valid；只对完整记录开放
  TaskFamily.RECOVERY=5 / CommandMode.RECOVERY=3 的想象起点。
- `imagined_reward`：允许配置过的恢复任务，四 tick 提供物理证据；每 tick 的 phase 与原生一致。
- `imagination/factory`：恢复状态和 Critic 特征、原生 finish/handoff 次序、timeout 截断。
- `StairSourceState.boundary_known`：RECOVERY 模式和原生一样阻塞普通命令随机重采样；现有
  jump/stair active 判断不足。handoff 后同步更新 `segment_end`，再决定后续边界是否已知。
- 显式迁移：旧数据仍可拟合 WM，但不伪造恢复记忆；新增 Critic 输入独立迁移。

采集 CLI `--record-recovery` 自动启用 jump、motion 和 stair/source 记录依赖；dataset 保存
`recovery_task_config`，checkpoint 保存 `recovery_scenario`。普通旧配置默认关闭此选项。
上述软件接线后的实际剩余工作是原生翻正数据采集与逐步一致性审计、按四向工况验证 WM 预测、
在线翻正训练，以及原生四向成功率和翻正后跟踪评价。

现有 `_RecoveryPreparation` 是策略接管前的关节动作域准备／被动释放，其记录仍按原生控制权
排除。四向评价必须记录**策略接管时的实际姿态**和准备段，不能把准备段完成的翻身算作完整 RL 翻正。

## 可选恢复接稳奖励

`reward_extensions.recovery_quiet` 默认是 0。启用时，原生环境与想象共用以下连续奖励密度：

```text
recovery_quiet × 1[mode == RECOVERY] × positive.height × exp(-||omega_body||² / 0.25)
```

`positive.height` 沿用已有站高与直立条件，角速度包含三个轴；翻倒或塌低时几乎不给分，
handoff 后不再支付。候选系数 1/s 与核尺度分别取自已有 yaw 正奖励幅度和高斯核，
属于待训练评价的选择，不是已验证的最优值，也不意味着侧倒翻正已经学会。
它不改变出生姿态、成功判据、探索 std、观测维度或控制器。

完整在线断点可通过 `upgrade_online_reward_reference.py --change recovery-quiet` 迁移。
该路径只接受已知旧 kernel 到 `recovery_quiet=1` 的变化，并要求同权重教师的新原生评价。
模型、优化器、课程、归一化、RNG 与历史 replay 全部保留；旧 reward 标签仍注明旧语义，
想象中的 Actor/Critic 目标重新计算，不额外 warmup。

独立辅助数据只用于动力学拟合，可以保留不同解析奖励 kernel/扩展权重下的物理记录。
文件 SHA、原 schema 和留出环境不变；物理、执行、任务及记录合同仍须一致，
该兼容规则不用于 Actor 想象起点。验证报告记录原 schema，并明确未使用历史奖励标签。

## 显式迁移已有 jump 分支

`world_model_v2/migrations/recovery.py` 仅接受已校准的 chassis35、motion/stair/source/jump 完整
checkpoint/replay 配对。目标原生 reference 必须同 schema、同物理 source、同 `jump_first_v1`
课程，带真实 `record_recovery` 字段、有效策略拥有的记录，以及采集得到的
`recovery_task_config`。已有 jump/stair 配置不可改变。

```sh
python scripts/upgrade_online_recovery_tasks.py \
  --checkpoint /absolute/path/jump-training/resume-a.pt \
  --reference-dataset /absolute/path/native-recovery/replay.pt \
  --course /absolute/path/pinned-ppo-course-jump-first.json \
  --course-coordinate RECOVERY_COURSE_COORDINATE \
  --world-only-cycles ADAPTATION_CYCLES \
  --output /absolute/path/new-recovery-branch
```

两个大写值须根据同一 pinned 课程的实际阶段和数据准备情况明确指定；脚本不推断能力晋级。
`--course-coordinate` 只改变课程查询位置，保留实际 collection steps、训练计数、教师混采进度、
所有采样 RNG 和原课程样本账。课程及阶段配额不重置，物理来源和网络的运动部分不改变。

迁移操作：

- 给旧 replay 增加恢复字段占位值，但全部 `recovery_record_valid=0`；原物理窗口仍可拟合 WM。
  这些旧行即使是普通模式也不能冒充已记录恢复记忆的策略起点。
- Critic 新增 7 列权重和 Adam 动量为零，初始输出与旧 Critic 相同；旧统计、参数和优化器历史保留。
  新均值为零／方差为一，`critic_norm_from=401` 等待真实有效记录校准，不能直接用于能力声明。
- `record_recovery=true`，`recovery_scenario` 保存原生配置；`policy_start_cycle` 设置为当前 cycle
  加指定的 world-only 周期数，Actor/Estimator、WM、其他 Adam 和 RNG 保留。
- 输出新的完整 `resume-a.pt` / `replay-a.pt`、轻量 actor、在线配置和带输入 SHA256 的迁移说明。
  源 checkpoint/replay 不覆盖；新分支从新物理回合恢复，不恢复旧求解器瞬态。

`tests/test_world_v2_recovery_upgrade.py` 覆盖函数等价、Adam 可继续更新、所有旧数据与
训练账保留、有效新起点筛选、只校准新增 7 列、来源不变，以及完整新配对写盘与旧文件不变。

## 测试

`tests/test_world_v2_recovery_task.py` 用物理状态轨迹逐步对照上游生命周期（出生朝向、低姿态接触、直立、handoff 与物理认证、超时与控制权分离、命令池 handoff、critic 时钟平移不变性）；不启动 Isaac，不代表机器人已学会翻正。
