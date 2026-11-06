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

## 断点和后续启用迁移

课程使用已有 RECOVERY family 总出生数决定两轮相位，不增加独立游标或随机源。
课程 state_dict 保存该 ordinal、22 项 ledger、原 sampler RNG 和 quota 债务。
WM 原生训练状态恢复后重新 reset，下一批变体可续接；当前 PPO 冷恢复不恢复原生 sampler，
不能承诺 PPO 重启后保持此相位。

关闭时 config.to_dict 省略字段，factory 不安装 wrapper、不增加 source receipt，保持旧身份。
启用时 source 增加 `recovery_roll_bridge_v1` receipt 且 course SHA 改变，旧 WM pair 直接续训会拒绝。
此候选没有实现兼容 alias 或迁移，也没有修改正式课程。

后续正式启用需要一个窄的出生采样迁移：

1. 认证原/新 course 原始字节 SHA，只允许该开关 false→true 及其精确 receipt；其余 physics、
   action、observer、reward、任务生命周期与其他采样参数逐项相同。
2. 完整 pair 的环境 curriculum 换用新 config；原 16 项 ledger 原值保留、尾部补六个 0。
   保留 family ordinal、sampler RNG/债务、collection/course 时钟、四优化器、网络、normalizer、
   auxiliary split/RNG。旧记录不补造桥接标签、不改物理/task/reward 真值。
3. 现有 `verify_course_sampling` 尚不允许这个字段/receipt，不能直接套旧兼容函数。
   新的明确 lineage 必须同时用于 retained teacher 和 auxiliary 数据兼容；旧教师的评价范围
   保持不变，不能宣称已评价或掌握桥接。
4. 新 release 做一次短原生出生读回验证，记录真实 45/60/75/90°、root/COM 高度、episode 标签与
   schema。出生采样改变本身不要求重置学习器、重写旧奖励或额外世界模型 warmup。

## 验证

CPU 专项覆盖默认兼容、32 次精确配额、其他族/部分姿态池不变、非整批断点恢复、source 严格区分、
真实选中资产的几何净空、原 90°端点逐值一致、缓存无别名、非零 yaw/升高地面重新放置、真实 snapshot
与请求标签区分。独立云台中立出生检查在六角度均有正净空；能力仍需原生物理评价。
