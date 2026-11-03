# 2026-10-31 辨识执行链冻结与接入边界

这里保留 Kaiser 当前翻正训练使用的小模型及原执行代码。源仓库
`/home/kaiser/workspace/isaac_sim-6-task-200hz`，commit
`444749daa37bba264bcfa014a641216018702d55`。模型、来源文件及其远端路径逐项列在
`source_manifest.json`；这些文件是逐字复制，未修改已运行的源仓库。

| 模型 | SHA-256 | 用途 |
|---|---|---|
| `wheel_model.json` | `01d52214771bb9dea2cb592eff4dc6fbd27bb096faa90ff1e17b5d898a1e877c` | 15.62、24 V、轮输出坐标；左右独立辨识 |
| `leg_model.json` | `90d299995f24ba38e6c1ed67e32dd40b9b2fa26e041f2f814db1e177ac040b52` | 左腿局部运动摩擦，右腿按已授权对称近似 |
| `gas_spring_model.json` | `698f67ccc7d6364b73f1b426ed3e6386ffc9e11912ef95a89d716eb7ee0e06b9` | 实测等效平均负载及原训练的端点外推 |

## 原生接线

`IdentifiedExecution(count, device, physics_dt, asset_directory=..., bundle_directory=...,
friction_mode=...)` 返回独立纯 Torch 组件。六轴固定为
`L_joint1, LL_joint1, R_joint1, RR_joint1, L_joint3, R_joint3`；单位为原生输出轴的
rad、rad/s、N·m。这与原 prone 的左右三元组次序不同，代码和对照测试已显式重排；
没有再乘减速比、RMCS 控制符号或编码器比例。

对 e726 的 `model_spec.json` 与冻结 canonical `identified_model_spec.json`，在各自
`source_closed_joint_pos` 用树 FK 变换六根主动轴到共同底盘坐标：同名轴点积均为 1
（最大方向向量差 `1.10e-15`）。因此这里需要重排轴次序，不需要再翻转力矩符号。
编码器角度零位仍不同，这一检查不授权直接复用 canonical holding-Jacobian 表。

1. 关闭旧 438 的六轴额外 armature、摩擦、黏滞、first-order tau。以
   `native_parameters()` 的声明替换，并读回核对。腿未辨识的惯量、静摩擦、黏滞和
   Stribeck 不能从旧配置填入。
2. 控制器双轮速度 P 改为源模型的 `0.5 N·m/(rad/s)`；电机限幅四腿 40 N·m、
   双轮 4.5 N·m。关闭旧轮 torque-speed envelope，改用此组件的方向相关 envelope。
   未补入任何新的腿电机滞后或速度限幅。
3. 每个 physics tick 先计算 PD 和电机限幅，然后调用
   `step(capped_motor_effort, actual_native_velocity, motor_enabled=...)`。
   输出 `motor_effort_nm` 供电机/饱和诊断，`native_effort_nm` 是加上被动软件摩擦后应提交
   原生的六轴力矩。原生 actuator 的最终 effort limit 必须给被动项留空间，见
   `native_effort_limits()`；电机电流 cap 本身不增大。
4. `reset(ids)` 清选中环境的轮 torque queue。`state_memory()` 的
   `wheel_torque_history_nm[N,H,2]` 为最新在前的全部历史，必须纳入世界模型的物理状态。
   不依赖绝对时钟或额外的浮点 ring head。固定 dt 与 delay 是 context/provenance。
   若启用源 DR，wheel body inertia 与 additional armature 使用同一个
   `mass_scale * independent_inertia_scale`，不能分别抽样。后端负责原生写入/读回，然后
   调用 `set_wheel_inertia_scale([N,2])`；软件残差的防反向界随同一 draw 更新。
   `context_parameters()` 返回该 draw、请求总 J 与 additional armature，须写入物理 context。
5. 气簧采用 `mount_force_element(core_body_names)` 替换旧 `env._spring_force`，接口
   `evaluate(link_pose_xyz_xyzw, world_com_xyz, spring_strength)` 与原遥测字段相容。
   它读取所选资产真实安装点并施加大小相等、方向相反的世界力及绕 COM 力矩，原生闭链
   负责实际 `JᵀF`。或者使用 `slider_effort(native_slider_q[N,2])`，正值表示伸长；
   **安装点和滑块压力只能启用一个**。原 10 MPa cubic pressure 必须关闭。
6. `receipt()` 是声明和 source identity，不是原生读回结果；后端必须另留实际 armature、
   摩擦、限幅读回。`native_coefficients_are_requested_not_readback=true` 明确区分这一点。

## 单一所有权与不能等价的部分

- 轮总延迟左 5.5 ms、右 3.5 ms，直接复用原 torque history 的整步＋小数步插值。
  它是 command→reported RPM 的辨识总延迟，不是单向 USB 实测。不能另叠旧 wheel tau
  或把这两值再用作 USB transport 延迟。CPU 对照覆盖 1/2.5/5/10 ms physics dt。
- `native_floor` 与源实现分工相同：原生承担 static breakaway 和 dynamic floor，软件
  承担速度表的剩余摩擦。只有后端能表达并读回这些系数时，才可用这个选项。
- `software_dynamic` 要求原生 wheel friction 全零，软件承担完整 moving-friction 表。
  它没有静态 breakaway 模型，静止行为与原 PhysX 分工不等价；不是暗中模拟已经支持。
  双轮有效总惯量仍由真实 CAD 的轴惯量加 additional armature 组成。重绑使用
  `axisᵀ I_COM axis + m|axis×COM|²`，额外量等于实测总 J 减去所选 CAD 已有量。
- 腿使用源 `0.05 rad/s` 数值正则带；带外常 Coulomb。该带不是辨识的黏滞或 Stribeck，
  原测量姿态/速度范围之外是原训练声明的 prior，不能当全域实测。
- 气簧保留 16 个实测分段和原 698f67 的正指数/线性端点趋势；超出原几何表压缩范围时
  保持端点力。这是旧训练的显式 extrapolation prior。原 source curve 的 `gas_pressure_pa`
  为 null，不把资产中历史 `pressure_mpa=10` 当成本次辨识压力。
- 698f67 的 encoder holding Jacobian 表绑定 canonical `8a9d7595...` model spec，不能
  直接当作 e726 资产编码器表。当前只迁移物理弹簧力，不迁移 encoder holding compensation；
  receipt 明确 `encoder_holding_compensation=false`。若需相同硬件 feedforward，需为当前
  编码器几何独立重绑并验证，不能重复加上物理力矩冒充控制补偿。

## 验证范围

`tests/test_identified_execution.py` 对冻结原源码进行 CPU 数值对照，覆盖轮 delay/envelope/
residual、腿摩擦、气簧实测节点和外推、左右轴顺序、队列清除、动摩擦耗散性、轮 J 重绑，
以及安装点力/总力矩守恒和虚功方向。未在此任务启动原生仿真；后端读回与整机试跑由
factory 集成负责。

`course/execution.py`（`supermimic_rl.course.execution`）提供 `configure_identified_actuators(cfg, execution, backend=...)`
和 `make_environment(base_type, execution)`。前者同步更新 `cfg.robot_cfg` 与声明式
`cfg.scene.robot` 副本；后者复用父 `_apply_action`，仅拦截一次最终整车 effort write，
保留云台和被动列，并将父 `_last_effort` 改为实际延迟后的电机力矩。
`_last_native_effort` 与 `_last_passive_effort` 分别记录实际主动轴原生总力矩和被动项。
气簧滑块遥测的势能解析积分与新物理曲线一致，虚功导数已经 CPU 检查。

**41e 特有接入点**：源 `OutputFeedbackController.effort()` 仍硬编码 wheel P=0.6，
并将 strength 约束为 `(0,1]`。仅修改 controller 的 damping 元数据不会改变实际 PD。
factory 还必须安装本项目 feedback 扩展，使实际 wheel P=0.5，并按最新源训练顺序执行
`nominal PD -> 固定 cap -> strength -> 固定 cap`；strength 大于 1 提高未饱和响应，
不会提高电机最大 40/4.5 N·m。不得把未安装扩展的组合称为完整新执行链。
