# 世界模型在线课程接续

`OnlineConfig.course_progression` 把实际采集 vector-step 时钟映射到原生课程的采样坐标。
阶段顺序、预算比例、任务混合和域范围沿用工厂指定的课程。采样坐标决定后续出生和命令，
不是实际转移数，也不代表技能已经通过。`collection.transitions` 始终保留真实采集量。

配置字段：

- `start_collection_step` / `end_collection_step`：实际采集步区间。
- `start_coordinate` / `end_coordinate`：对应的原生采样坐标区间。
- `jump_height_gates`：完整的原生跳跃高度阶梯，复用当前 PPO 的门槛、窗口和最少事件数。
- `jump_height_progression`：默认 `performance`，通过门槛才提高采样上限；显式设为
  `schedule` 时，上限跟随既有课程坐标和高度阶梯，成功门槛只报告真实能力。
  PPO 的同名字段位于训练配置顶层，配合 `course_gate_mode=report` 使用。

首次接续必须从完整 checkpoint 的采集步数及课程坐标开始。此后完整 checkpoint 同时保存
高度阶梯和事件窗口；普通续训不得悄悄更换映射。想延长训练预算时可以继续在末阶段采样。
可显式切换 `jump_height_progression` 并记录续训迁移；保留原坐标、失败窗口、模型、优化器
及 replay，不把采样到更高请求记为能力通过。旧 checkpoint 缺少该字段时仍按 `performance` 解释。
配置终点应放在预算内最后一次采集的步序号，即
`saved_collection_steps + (cycles - saved_cycle) * collection_steps - 1`。

模型、优化器、归一化、replay、RNG 和累计更新数保留；不增加 warmup。
`policy_start_cycle` 仍是绝对轮次，已经超过它的学生立即继续更新。
物理仿真仍使用既有 cold-reset 恢复方式，新 episode namespace 隔离断点前后的物理状态。

跳跃高度使用 `CourseGates(mode='report')`，不阻塞台阶、恢复等其他阶段，也不停止 Actor。
世界模型采集通过只读的逐物理 tick 观察器记录各高度区间的无辅助跳跃结束/成功事件，
不安装辅助力、不改动作、不消费随机数。计数沿用 PPO 的 metric 名称以复用高度条件；
名称中的 `jump_assist` 不表示世界模型施加了辅助力。

每轮日志增加 `course`：采样阶段、真实事件增量、当前高度、未达标阶梯与其等待轮数。
课程推进只改变未来样本分布；历史 replay 和想象任务继续使用各自记录的任务状态，
不把旧样本当作新阶段的数据。末阶段到达后仍需独立验证平地、跳跃、翻正和台阶能力。

上线前检查：现有源配置/物理合同一致；后续阶段的原生记录可以被解析任务与奖励复现；
恢复后实际模型和 Actor 更新继续执行。短期低成功率或回报下降留给评审，不触发自动回滚。
