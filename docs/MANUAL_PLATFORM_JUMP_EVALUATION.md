# 主动行进跳台评价接口

`ManualPlatformJumpEvaluation` 是固定原生评价的附加编排。它只调用已有跳跃、台阶请求接口，不改 Actor 的 5×35 输入、六个动作、训练课程、公共协议或成功标准。目标仅接受已审核 catalog 的 `101:profile:1` 和 `102:profile:1`。它尚未实现独立落台判定，因此不能把原生平地跳跃成功直接当作落台成功。

调用方必须明确给出每行的目标、catalog 字节 SHA256、最早触发时间、跳跃幅度和支撑高度，没有隐含机械参数。固定 suite 必须全部是 `TERRAIN`，每行一份意图；实际 router tile、atlas manifest 和 geometry SHA 都需与 catalog 对应。速度仍由所选固定场景产生并经过原生命令接纳，模块仅把显式支撑高度送入原有 stream。

```python
from supermimic_rl.course.platform_jump_evaluation import (
    ManualPlatformJumpEvaluation,
    ManualPlatformJumpIntent,
)
from supermimic_rl.world_model_v2.evaluation import evaluate_native

# All values below come from the caller's explicit evaluation specification.
intents = [
    ManualPlatformJumpIntent(
        env_id=row,
        target_id=targets_by_row[row],
        jump_at_episode_s=earliest_jump_at_by_row[row],
        rise_m=rise_by_row[row],
        support_height_m=support_height_by_row[row],
    )
    for row in range(adapter.num_envs)
]
with ManualPlatformJumpEvaluation(
    adapter,
    intents=intents,
    catalog_path=catalog_path,
    catalog_sha256=catalog_sha256,
    sidecar_path=output_dir / "manual-platform-jump.jsonl",
) as manual:
    report = evaluate_native(
        environment, actor, steps=steps,
        record_dataset=output_dir / "replay.pt",
        recording_source=manual.recording_source(actor_recording_source),
    )
```

上例的 `adapter` 是 `environment.adapter`；`environment` 和 `actor` 使用现有 PPO 或学生评价入口提供的实际对象。先固定场景再使用此 context，且每个 context 只能 begin 一次。完成及异常退出都会恢复 hooks，不修改运行中的训练 adapter。

## 发起时序

1. 从该 episode 开始，经原有 stream 维持 caller 支撑高度；最早触发时间前，ToF 行为不变。
2. 到 `jump_at_episode_s` 后，只保留 ToF 的探测和时钟，暂不允许它提交新 STAIR。已有 STAIR 先提交 cancel，在下一边界确认协议 idle 和 FSM NORMAL 后才考虑 JUMP。
3. 等原有 stream 已接纳 caller 高度，使用相同 dtype 的张量精确比较；再等待已有 `JumpTriggerGate.ready`。后者使用真实双轮支撑、高度、既有 episode 时间和 cooldown，因此这是**使用原生真值的评价触发**，不是部署控制方案。
4. 提交高于原生 ID watermark 的新 JUMP。若被 NOT_READY 等拒绝，已消费的 ID 不复用。等待、拒绝、实际发包时刻与相对最早时间的延迟均记入 JSONL。
5. 未发起的 pending 窗口到最早时间加原生 timeout 为止；发起后窗口使用真实 acceptance clock 加同一 timeout。协议估计完成不提前释放 ToF reservation，因为物理尾窗仍可能存在。取消、失败协议结束、reset 或时间窗结束都会释放。

没有获得 gate 的可发起条件时，记录 `initiation_outcome=not_initiated`，不能算作策略跳跃失败。实际是否腾空、是否峰高达标、是否两轮落在目标三角面和后续接稳，应从同请求物理回放独立判断；不新增静止停台时长或清障阈值。

## 数据用途

`evaluation_summary`、JSONL 首行和 `manual.recording_source(...)` 均带有 `joint_imagination_forbidden=True`。手动未来意图、pending 与 reservation 状态没有进入当前世界模型逐步记录；因此该数据不能作为联合跳台任务的想象训练或可恢复任务状态。它可作为物理动力学数据候选，仍需已有 source/留出划分检查。此次不改数据 ingest，也不修改旧 flat-jump 成功或奖励语义。

JSONL 以独占创建方式打开并逐事件写入，原生评价异常时仍保留已发生的请求证据。结束 JSON、完整 replay 与 `native/failure.json` 仍由调用方的现有评价进程检查；不能只用进程 exit code 判定评价完成。
