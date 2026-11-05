# V2 解析跳跃任务接口

`world_model_v2/jump_task.py` 复用 pinned upstream 的 `RequestProtocol.step`、
`JumpReferenceProgram.sample` 和 `LabCourseTasks.observe_physics`。它不新增动力学网络或
Actor 输入。模块记录字段可选；普通平地旧 replay 不需要迁移。原生课程配置由
`JumpTaskConfig.capture(native)` 读取，随采集 provenance 保存一次。

## 数据接口

`JumpTaskState.capture(native)` 在 `env.step()` 前捕获；`record()` / `decode(record)`
与 compact replay 对接。新增三个字段：

| 字段 | 类型与形状 | 内容 |
| --- | --- | --- |
| `jump_ids` | int64 `[N,9]` | 协议水位、活动 ID、估计腾空计数；下一个 ID、物理 ID、物理腾空计数、已支付 ID、成功 ID、奖励 phase |
| `jump_clock` | float64 `[N,5]` | 当前边界、协议接受、上次协议调用、估计落地开始、物理接受时刻 |
| `jump_values` | float32 `[N,17]` | 请求高度及标志、自动请求期限、rise、锁定高度和地面锚点、可见时钟、物理证据与峰高／落地累计时间 |

具名顺序以模块的 `JUMP_ID_NAMES / JUMP_CLOCK_NAMES / JUMP_VALUE_NAMES` 为准。
调用方单独保存 `jump_record_valid`，不可把缺少状态的旧记录当作有效跳跃起点。
三个字段共 180 bytes/row；加一个 float32 valid，1000×4096 条记录约增加 0.70 GiB。

地面锚点和上次机身高度在捕获时减去 `router.origins[:,2]`，与 `PhysicalState.position`
使用同一 tile 坐标。请求 ID 和物理时钟不压成 float32。尚未消费的 external jump packet
不能静默丢弃：`capture()` 会明确拒绝；显式已知 packet 可传给 `advance()`。

## 原生顺序

1. 当前动作预测四个 5 ms 物理子步。
2. `observe_interval()` 逐子步推进物理证据，再给出该子步的 phase 和高度／速度参考。
   请求时钟在四个子步内保持不变。返回的新 state 仍属于旧策略边界。
3. 现有解析奖励累计四个子步，`finish()` 使用本区间 `success_credited`；终止和控制权筛选仍由调用方负责。
4. MotionTask 推进命令与 episode elapsed，然后调用 `advance()`：传已 admitted command、
   新边界时刻、**前一帧**噪声观测的 CompletedEstimate、当前地面和台阶阻塞标志。
5. `advance()` 返回新 state、锁定基准高度后的 command、mode、request_height 和 request_clock。
   调用方再进行台阶仲裁及下一帧观测组装。协议估计完成不等于物理成功；待认证 physical ID 可继续存活。

自动请求复用原生 `3 + 5 * uniform` 秒间隔。调用方必须在整批 imagination 准备时提供
`interval_uniform[N]`，不要在 microbatch 或重试内部重抽。传显式 `RequestCommand` 时自动源休眠。

物理输入 `wheel_support_load[N,4,2]` 是两轮接触面法向力提供的向上支撑载荷，与原生
`state.wheel_ground_force_n` 一致；不能用包含切向摩擦力的净力范数替代。跳跃证据使用
base COM 的世界 z 速度；`com_velocity()` 根据已有 COM 偏移计算
`v_origin + omega × (R @ com_offset)`。共享奖励跟踪仍使用 base_link 原点速度。
倾角采用原生 `atan2(norm(gravity_xy), -gravity_z)`。

协议、物理事件和参考曲线沿用原生离散／停梯度语义。预测位置、速度及解析跟踪代价仍可反传；
落地参考中的当前地面与命令高度也保留连续梯度。物理成功必须满足原生腾空、上升、峰高、
支持起点与稳定落地条件，不能由请求时钟或预测的奖励网络制造。

开启共享跳跃扩展时，高度／速度参考只有在首次真实腾空后接地才切到落地目标。
压缩时两轮短暂卸载并重新接触仍可能使原生接触 phase 变成 `LANDING`，但不能因此
把尚未执行的蹬伸目标清零。原生采样入口和 `JumpTask.observe_interval()` 共用
`jump_tracking_targets()`，使用已有 first-flight latch；请求协议、接触 phase、
Actor 输入、力矩限制和成功判据不变。未开启 latch 的原生对齐测试保留旧行为。
采集来源中的 `reward_kernel.jump_tracking_reference=first_real_flight_landing_v1`
标识这一奖励变化，旧来源的 checkpoint/replay 不得直接冒充新来源恢复。

## 可选 Critic 特征

`state.critic_features(physical, motion, config)` 返回 `JUMP_FEATURE_DIM=32`：
26 个状态／相对量及 6 个 phase one-hot。它包括估计与物理腾空 latch、峰高、落地累计时间、
请求和物理接受的相对年龄、距下一请求的时间、支付关系，以及相对机身高度的锚点。
时间和长度按原生期限／高度范围归一化；不输入绝对 ID、时间或世界坐标。
普通分支不增维；跳跃分支只扩展 privileged Critic 输入，Actor 的 35D 历史输入和 WorldModel 不变。

H16 为 0.32 秒，不能从请求开始覆盖原生要求的 2 秒稳定落地；阶段中间起点和带完整任务记忆的
Critic bootstrap 仍然必要。解析链路打通与跳跃动力学学好是两件事。

## 测试

`tests/test_world_v2_jump_task.py` 覆盖请求、四子步腾空与落地、唯一成功事件、过期估计、重复 ID、取消与超时、终止后的物理认证及 critic 不变性；`tests/test_world_v2_jump_imagination.py` 检验 H16 想象中的请求、仲裁、成功事件一次性支付与有限差分梯度。测试使用小型可微夹具，不代表机器人已学会跳跃。

## 训练入口

采集脚本的 `--record-jumps` 会同时开启 terrain、motion、stair 和 stair-source 记录，
在现有数据集里额外保存 `jump_task_config`。例如在已确认可用的教师上：

```sh
SUPERMIMIC_SCRIPT=scripts/collect_teacher_v2.py bash scripts/run_gpufree.sh \
  --factory-config /absolute/path/factory.json \
  --checkpoint /absolute/path/teacher.pt --checkpoint-sha256 TEACHER_SHA256 \
  --steps 1000 --course-coordinate REQUEST_COURSE_COORDINATE \
  --record-jumps --output outputs/jump-teacher-data
```

`REQUEST_COURSE_COORDINATE` 要取当前 pinned course 的实际请求阶段，不能把示例符号当数值。
在线配置同时设置 `record_tasks=true`、`record_stair_source=true`、`record_jumps=true`，
并给出明确的 `course_coordinate`。其 checkpoint 保存 `jump_scenario`，恢复时检查与原生协议及参考配置一致。

新分支仅接纳 `jump_record_valid > 0.5` 的完整记录；`REQUEST` 家族和 `JUMP` 模式因此可成为
想象起点，`RECOVERY` 仍未开放。正在执行的 jump 与 stair 都会阻塞命令段重采样；
请求释放后若段已到期，就在该已知边界 bootstrap，不编造下一段随机指令。
前一帧估计器决定 jump 是否可接收，然后执行 stair 协议及 gimbal 任务，次序与原生一致。

可见 goal 仍是五个 mode one-hot、缩放请求高度和请求时钟：`goal[4:9]`、`goal[9]`、
`goal[10]`。请求高度大于 0.2 m 时缩放值可能超过 1，不能把它一起参与 mode 的 argmax。
Actor 仍只消费原生历史前 35 维，动力学网络和物理记录不增加输出头。

旧平地配置默认关闭此选项，字段和网络维度不变。直接用旧断点开启该选项会被恢复契约拒绝；
显式新分支可以保留旧物理数据用于世界模型拟合，但缺失 jump 隐状态的旧行不能伪装成新记录。

## 原生数据一致性审计

新采集的数据可用现有字段审计，无需加载 Actor、世界模型或启动 Isaac：

```sh
python scripts/check_native_jump_task.py /absolute/path/replay.pt \
  --atlas /absolute/path/pinned-atlas.json --device cuda:0 \
  --output outputs/jump-native-consistency.json
```

设备类型要与采集的 stair RNG 一致。入口直接解码四个已记录物理子步、轮法向支撑载荷和
COM 偏移，推进 JumpTask；用真实下一记录的传感器队列与当前已部署估计推进命令、请求和台阶仲裁。
对照相邻记录的 phase、腾空／上升标志、峰高、落地时间、paid ID、协议、可见 goal 和来源状态。
最后一个真实子步还与单独记录的 `end_physical` 对照。

比较仅取连续 vector step、两端 valid、同 episode/control epoch、policy-owned 且命令段边界已知的行。
重置、数值无效、终止／截断、翻正和未知命令重采样边界单独计数，不当作一致性成功。

自动 jump 间隔的实际随机数未记录；审计从下一条 `next_request_s` 与已推进 elapsed 恢复本次间隔，
核对请求是否到期、ID 与原生 3–8 秒范围，但把这个恢复出来的 deadline 字段排除出独立相等检查。
台阶来源已有完整 vector-step RNG 记录，因此仍按原始环境数和设备直接复放。

compact 数据没有独立保存四个子步的原生 phase／reward reference。
因此报告只声称最后物理 phase、持续任务记忆及下一边界协议的一致性；下一参考曲线来自
实际记录任务状态的重建，不能称为四子步独立 reference 真值验证。
审计通过也不证明数据源教师有跳跃能力：foundation 教师在 REQUEST 课程采到的失败样本仍是有效动力学数据。
