# 恢复姿态成本的几何倾角

Factory 的 `reward_extensions.recovery_tilt_geodesic: true` 只替换 RECOVERY 的
`cost.attitude`。默认关闭，继续使用原式；不修改课程、出生姿态、Actor 5×35 输入、
动作与电机参数、恢复成功条件或超时。

原式为 `2 * sum(h((roll_pitch - target_roll_pitch) / 0.8))`，
其中 `h(x) = hypot(1, x) - 1`。原生四元数转 Euler 的 pitch 范围为 ±90°，
真实 pitch 跨过 90° 时，roll 会跳到 ±180°。同一条连续俯仰路径上，
89.9° 的成本为 2.40307/s，90.1° 为 8.50770/s；这不是物理状态的突变。

开启后，使用同一物理 tick 的 `RewardPhysics.gravity_body` 和既有
`RewardReference.target_roll_pitch_rad = (r, p)`：

```text
current_up = -gravity_body
target_up  = (-sin(p), sin(r) cos(p), cos(r) cos(p))
theta      = atan2(norm(cross(current_up, target_up)), dot(current_up, target_up))
cost.attitude[RECOVERY] = 2 * h(theta / 0.8)
```

参考姿态按原生 roll/pitch 约定构造，非零参考也生效。绝对世界 yaw 不参与。
不需要新的传感器或状态字段。直立附近的二次尺度与原式一致；零参考下单轴
roll 0–180°、pitch 0–90° 保持原尺度。超过 Euler 分支后，成本仍随实际倾角
连续变化；89.9°→90.1° 对应 2.40307→2.41084/s。倒置成本有限，为 6.10463/s。
组合姿态使用两条上方向间的最短球面角，因此不会等于旧式两个 Euler 成本之和。

正向 `recovery_upright`、高度和 `recovery_quiet` 奖励保持原式，仍以世界竖直为目标。
非零参考仅影响这个原本就接受该参考的姿态成本；不借此重定义其他奖励。
普通运动和 JUMP 的所有项逐值不变。改变恢复姿态成本不代表已学会翻正，
纯侧倒也不能仅用俯仰分支断点解释。

## 来源与世界模型

关闭时，扩展 receipt 省略新字段；开启时记录
`extensions.recovery_tilt_geodesic: true`，与既有 `kernel_sha256` 共同认证公式。
内核文件更新本身会改变 SHA，即使开关关闭，也不能把新源码冒充旧冻结源码。
无需增加 jump progress、几何 producer、replay 字段或独立随机流。

本实现不迁移、部署或重标任何现有 WM checkpoint/replay。原生和想象路径均调用
共享 kernel，但开启后的完整训练源仍需要各自明确绑定。组件测试只证明新姿态成本
对有效重力向量和非零目标的梯度有限，并通过非奇异点有限差分检查；不等于证明
整个 WM 恢复链可微或有效。原生 `_roll_pitch(quaternion)` 在 exact ±90° 的
`asin` 导数问题仍可能影响其他计算分支，不在此次修改范围；PPO 不反传奖励梯度。

在恰好相反的两条上方向处，最短翻转方向不唯一。`vector_norm` 的零次梯度选择
保持数值有限，不人为指定左翻/右翻方向；这不是倒置时提供非零定向梯度的保证。

## 验证

`tests/test_recovery_tilt_geodesic.py` 使用原生姿态编译器和观测转换，覆盖：

- 正负俯仰跨 90°、组合 roll/pitch、绝对 yaw 不变及纯 roll 对照；
- 非零目标、直立附近尺度、直立/倒置分量梯度与有限差分；
- 默认关闭的旧成本逐值相同、其他模式全部分项不变、正向恢复项不变；
- 原生 begin→四个 5 ms tick→finish 积分、扩展 receipt 和事件身份保持。

测试不启动仿真，不代替后续同口径原生能力评价。
