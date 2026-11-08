# 侧倾恢复出生桥接

`base.recovery_roll_bridge: true` 是课程 JSON 的可选教学采样开关，默认关闭。
它增加左/右 45°、60°、75° 出生，不增加网络、动作限制、奖励项或成功门槛。
角度是离散教学网格，不是机械参数或已验证的恢复能力。

## 份额与标签

仅在原 16 个恢复姿态全部开放后，按现有 RECOVERY 累计出生 ordinal 的两轮周期分配：

| 每 32 次恢复出生 | 数量 |
|---|---:|
| 左/右 45°、60°、75° | 各 1，共 6 |
| 原六种简单姿态 | 各 1，共 6 |
| 原十种困难姿态，包括真正左右 90° | 各 2，共 20 |

桥接替换第二轮的 `upright`、`crouched`、`upright_release`、`crouched_release`、
`pitch_forward_45`、`pitch_backward_45` 槽位。其他任务族及原困难姿态的份额不变。
恢复族占全部出生 16% 时，桥接目标份额是全部出生的 3%；这不是有效策略时间。
部分姿态池尚未开放齐全时保持原始轮换。

原生 pose compiler 仍用侧倒基姿态 ID 4/5；出生 ledger 在开关启用时扩展至 22 个命名标签，
追加 `left_side_45`、`right_side_45`、`left_side_60`、`right_side_60`、
`left_side_75`、`right_side_75`。原 ID 4/5 的 ledger 计数只包含真正 90° 出生。
这些都是出生计数，不是成功计数。固定评价关闭桥接应用，仍严格评价原 16 种姿态。

## 几何与记录

六个新姿态通过原 `RecoveryPoseCompiler.compile(roll_pitch_offset_rad=...)` 和
`AssetCollisionGeometry.place` 编译。零 yaw、零地面高度时只平移已编译缓存；其他 yaw/地面
高度复用原编译路径。禁止只旋转旧 90° quaternion 后沿用旧 z。原 16 个姿态仍调用原 pose 路径。

沿用原闭链解、native joint 名称映射、安装净空和 root-origin/COM 速度转换；
静态 geometry_ready 不代表物理恢复成功。现有 native 放置几何覆盖核心 19 bodies，
云台两 bodies 的中立出生净空另有静态审计；这不证明后续云台运动、接触动态或自碰撞。

`native_log_directory/recovery-births.jsonl` 在首次非 terminal snapshot 记录：

- variant 与复用的 native base-pose ID；
- requested roll、placed roll、实际 snapshot roll；
- origin z、placed root z、实际 root z、实际 base COM z；
- adapter episode ID 与首次 snapshot 时间。

实际角度与高度来自已捕获的物理读回，不用 requested/compiled 值冒充反馈。
日志是出生身份的 sidecar；Actor 5×35、世界模型输入输出和 replay 物理布局不增加字段。

## 可复现课程配置

两份 v8 候选配置保留各自 v7 课程的阶段预算、任务份额、指令范围和恢复姿态解锁，
只增加 `base.recovery_roll_bridge: true`：

| 配置 | 台阶出生安排 | 文件 SHA-256 |
|---|---|---|
| [ppo-course-gated-v8-recovery-bridge.json](../configs/gpufree/ppo-course-gated-v8-recovery-bridge.json) | 原 v7 课程 | `027525a91edfdcb18cae98ab48ff2e5aa2c383e7d9a0d6f6fdd5006fc4b6f859` |
| [ppo-course-gated-v8-early-authored.json](../configs/gpufree/ppo-course-gated-v8-early-authored.json) | 保留提前抽样真实原包台阶的 `foundation_authored_rehearsal` | `361f7f31ad36d3dc13b3a29ef0d67eed9ac2faf504ef12d5119fa1339e5d833e` |

这些文件可供 PPO 或世界模型的原生环境使用，文件名不限定学习算法。
SHA 对应完整原始文件字节；重新排版会改变课程来源身份。
配置入库不表示正式训练已经采用，也不表示桥接姿态已经学会。

ToF 同立面超时重试是 factory 的独立开关 `stair_timeout_retry`，不属于 course JSON。
选择 v8 课程不会自动启用重试；两者同时启用时分别记录并认证各自的来源变化。

## 断点与启用迁移

课程使用已有 RECOVERY family 总出生数决定两轮相位，不增加独立游标或随机源。
课程 state_dict 保存该 ordinal、22 项 ledger、原 sampler RNG 和 quota 债务。
WM 原生训练状态恢复后重新 reset，下一批变体可续接；当前 PPO 冷恢复不恢复原生 sampler，
不能承诺 PPO 重启后保持此相位。

关闭时 config.to_dict 省略字段，factory 不安装 wrapper、不增加 source receipt，保持旧身份。
启用时 source 增加 `recovery_roll_bridge` 字段，其 receipt 合同为
`recovery_roll_bridge_v1`，且 course SHA 改变。旧 WM pair 不能未经认证直接续训。

[recovery_bridge.py](../src/supermimic_rl/world_model_v2/migrations/recovery_bridge.py)
已提供纯迁移 `upgrade_recovery_roll_bridge`，返回升级后的内存 pair 和证明，不启动环境、
不保存文件或部署。`recovery-roll-bridge-sampling-v1` 证明认证原/新 course 原始字节 SHA，
只允许开关 false→true 及精确 receipt；其余 physics、action、observer、reward、任务生命周期
和采样参数保持相同。迁移用新课程配置重建环境 curriculum，保留原 16 项出生计数并追加六个 0，
保留 family ordinal、sampler RNG/债务、collection/course 时钟、四优化器、网络和 normalizer。
旧记录不补造桥接标签，不改物理、task 或 reward 真值。

[source_lineage.py](../src/supermimic_rl/world_model_v2/migrations/source_lineage.py)
已接入训练恢复、教师和辅助数据加载。它按迁移 ledger 的实际顺序逆向验证，
每个新证明必须匹配完整 `after` source 才能撤回完整 `before` source，直到原 artifact 身份。
原有课程采样和提示生命周期证明保留各自的历史语义；未知记录不能授权来源差异。
这使桥接与独立 ToF 重试迁移可以组合，而不依赖固定的兼容检查顺序。

教师 checkpoint 的原 source、权重、qualification 和评价范围均保留；辅助数据的原 source、
训练/留出行划分及采样 RNG 也保留。旧教师和旧数据不被重标成已验证新桥接或重试能力。
旧 `verify_course_sampling` 仍不允许桥接字段，不能用它代替新增的明确证明。
单独桥接迁移不改 replay；若同时启用 ToF 重试，其独立迁移仅清除旧 `stair_source_valid`，
保留已记录的物理、任务和奖励真值。

实际启用还需针对目标 factory 做短原生出生读回，核对真实 45/60/75°、原 90°/倒扣端点、
root/COM 高度、episode 标签与 schema，并通过完整 pair、教师和辅助数据的恢复检查。
出生读回证明来源与放置接线，不证明动态恢复能力。此采样变化不要求重置学习器、重写旧奖励
或额外世界模型 warmup；固定场景评价仍强制禁用桥接，保留原 16 个端点用于能力比较。

## 验证

CPU 专项覆盖默认兼容、32 次精确配额、其他族/部分姿态池不变、非整批断点恢复、source 严格区分、
真实选中资产的几何净空、原 90°端点逐值一致、缓存无别名、非零 yaw/升高地面重新放置、真实 snapshot
与请求标签区分。独立云台中立出生检查在六角度均有正净空；能力仍需原生物理评价。
