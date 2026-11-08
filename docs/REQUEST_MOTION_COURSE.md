# REQUEST 原地与行进目标

`jump_first_v1` 的出生采样按已有 REQUEST ordinal 交替分配原地与行进。早期 motion scale 为零时全部原地；开启行进后，行进目标按已有 `moving_jump_min_fraction` 设置下限。它不保证当前机器人速度或请求接受率为 50/50。

此前原生 command pool 会在初始段和后续段重新采样，覆盖上述出生命令。现在 `JumpFirstCommandPool.bind_episode_targets` 在原生 `_begin_episode_segments` 前保存 `task.command[:, 0]`，仅在 REQUEST 的 `_draw_segments` 返回时恢复该前向目标。原地使用 STAND，行进使用 NORMAL，避免原生 `_apply_segment` 根据旧随机 STAND 清零目标。

- 同一 REQUEST episode 的多次请求保持同一类别、方向和速度；不同出生继续使用原课程采样。高度、段时长仍由原池采样。
- 普通运动、地形、恢复的命令和随机数消费保持不变；REQUEST 不增加随机源。
- 原生协议继续决定活跃跳跃期间是否冻结重采。人工轨迹和外部请求不在每个 tick 被出生缓存重写。人工控制若独占 `_draw_segments`，拥有其段命令。
- 固定评价直接替换 `_draw_segments`，因此其每段速度由固定场景决定，不受训练缓存覆盖。评价结束恢复原 wrapper。

现有云台平移课程在 REQUEST 尚处 NORMAL 等待状态时仍可能对平地 NORMAL 段覆盖底盘命令；进入 JUMP 后该覆盖被原生 eligibility 清除。本补丁不修改该既有机制。因此 50/50 是 REQUEST 出生的前向目标分配，不是所有等待 tick 的最终命令比例，更不代表接受请求比例。若独立调整等待阶段的云台任务，必须同步原生和 `MotionTask.finish_command` 的 eligibility。

## 保存恢复

前向缓存只属于当前物理 episode；冷恢复后由实际新出生填充，不恢复旧求解器或旧 episode 缓存，没有新增 checkpoint 字段。

恢复采样的保证取决于调用方，不能混用：世界模型的 `CourseAdapter.load_training_state_dict` 恢复课程 `sample_counts`、原生 RNG 和 ID 后调用 `reset()`，可延续已有交替 ordinal 和段随机序列；当前 PPO cold resume 只恢复其声明的 Torch RNG、课程坐标/门等学习状态，不恢复原生 sampler RNG 或 episode ordinal。本补丁沿用各自既有语义，不宣称 PPO 获得新增的精确出生恢复能力。

## 世界模型边界

`MotionTask.capture` 已记录真实 `trajectory.start/end/current`。修复后的前向目标由同一记录路径进入 replay 和想象，Actor 输入仍为原始 5×35。

想象只推进已记录的 segment。`StairSourceState.boundary_known` 在下一次未被活跃 jump/stair/recovery 阻止的随机重采前结束该想象段；Critic 在边界提供 bootstrap。活跃 jump 仍按原生协议保持当前段。随机未来高度和段时长没有被假定已知，因此本补丁不新增想象侧随机池或隐藏缓存字段。恢复 handoff 继续复用原生 `begin_tracking`，不走 REQUEST 出生覆写。

## 来源契约

安装回执增加：

```json
{
  "request_motion": "birth_forward_held_v1",
  "request_motion_resume": "cold_physics_reset_repopulates_cache"
}
```

这是命令采样语义变更。物理、reward kernel、Actor/估计器、replay 字段形状不变，但 `course_schedule` 来源身份必须变化。已有 replay 仍包含其实际历史目标，不能通过改 SHA 把旧采样宣称为新采样；在线世界模型接续需要显式的 source 变更记录。PPO 可在完整训练断点的原有冷恢复路径接续。

生产改动仅 `src/supermimic_rl/course/jump_first.py`，该文件可独立带入正式 `fec7b02`。没有修改原生冻结 source、硬件参数、奖励权重、请求周期、成功门槛或优化器设置。

测试覆盖原生出生→step→段重采、部分 reset、50/50与行进最小目标、非 REQUEST RNG/命令、恢复 handoff、冷恢复、在途人工轨迹、固定评价替换与恢复、32步原生/想象命令一致性，以及随机段边界 bootstrap。
