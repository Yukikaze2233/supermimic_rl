# 架构

本工程在固定版本的上游原生环境（`transformer_rl` 41e8280，Isaac Lab `WheeledLabEnv`）上训练闭链轮足整车（带独立云台）。两条训练线共用同一套原生课程、执行链和奖励：

- **PPO 教师**：在原生环境上按能力门槛晋级的完整课程训练，产出可部署 actor，同时作为世界模型数据的教师。
- **SuperMimic（世界模型 V2）**：在原生数据上拟合确定性多步世界模型，在想象中做短视界 BPTT 更新 5×35 学生 actor；学生从 PPO 精确初始化。

当前课程路径不修改上游源码（旧路径补丁见运行说明）。所有改动以包装、注入或替换类型的方式安装，并写入 provenance/readback 回执。

## 包结构与依赖方向

```text
supermimic_rl/
  contracts.py       张量合同：StateSchema、Transition、动作界（从原生 source 读取）
  reward_kernel.py   原生与想象共用的可微奖励密度；可选扩展项默认关闭
  course/            原生整车训练课程（依赖 adapters，反向不依赖）
  adapters/          通用模拟器边界：采集、打包、子步、力学读回、核心投影、旧平地环境
  ppo/               PPO 教师：策略、训练器、原生环境包装、课程门槛、评价
  world_model_v2/    SuperMimic V2：记录、回放、模型、想象、学习、在线训练、任务协议
    migrations/      旧在线 checkpoint 到新任务合同的显式迁移
  (顶层其余模块)     V1 平地流水线与 CLI；runtime.atomic_save 等工具被各线共用
scripts/             训练、采集、拟合、评价、导出与审计入口
tests/               无 Isaac 的单元/合同测试；共享构造器在 tests/support/
```

依赖只朝一个方向：`ppo`、`world_model_v2` → `course` → `adapters` → `contracts`。`reward_kernel` 被 `course`（原生）和 `world_model_v2`（想象）同时使用，保证两边奖励公式一致。

## 原生课程 `supermimic_rl.course`

`course.factory.create_course(configuration)` 由 factory JSON 描述（runtime、course、asset 都用 sha256 钉住），按顺序构造：

| 步骤 | 模块 | 作用 |
|---|---|---|
| 物理后端 | `backend`、`kamino_solver`、`health`、`twist`、`coordinates` | PhysX TGS 或 Kamino；后端健康诊断与 Kamino 专用修正 |
| 执行链 | `identified_execution`、`execution` | 辨识轮模型（左右聚合延迟 5.5/3.5 ms、高速包络、摩擦节点）、腿摩擦、气簧 16 节点；作用在原生力矩边界 |
| 域随机化 | `randomization`、`materials` | requested-v2：质量/惯量/质心、电机强度、编码器与 IMU 偏置、传感器噪声；可选持续机身外力与速度冲击（原生 EpisodeRandomization 语义） |
| 动作合同 | `action_contract` | 以 runtime `task.action_bounds` 对齐原生控制器与 ownership 校验（腿 ±4π 即公共转角 ±π，支持整圈回转） |
| 奖励 | `reward` | 把 `reward_kernel` 装入原生 composer；扩展项（`jump_flight`、`jump_lift`、`jump_push`、`jump_missed`、`success_event`）写入回执 |
| 课程 | `curriculum`、`jump_first` | 按真实转移坐标推进的八阶段课程；jump-first 顺序：平地 → 小跳 → 动态指令 → 地形 → 台阶 → 翻正 → 混合 DR |
| 路由与评价 | `routing`、`evaluation`、`adapter` | 台阶终点路由到实际地图块；固定场景评价套件；`CourseAdapter` 采集原生传感器与任务所有权 |
| 技能任务 | `jump_takeoff`、`stair_label` | 跳跃：到期的自动请求等待原生支撑站姿、最短回合时间与冷却后才发出（进行中的请求与外部请求不受影响），参考改为 10 cm/0.3 s 下蹲 + 0.2 s 伸展的长蹬伸；台阶：STAIR 模式只由 ToF 台阶状态机置位。属于任务定义，训练、评价、采集都安装 |
| 训练专用 | `jump_assist` | 跳跃请求推送窗口内的竖直助力（概率衰减或力幅衰减），持续到真实离地（真实上升后腾空）；按实际是否施加冲量区分有/无助力，并按请求高度分档统计结束、成功与真实离地次数；门槛为报告模式时，助力强度随已打开的高度门槛逐级减弱；评价与世界模型 factory 不安装 |
| 回合错开 | `adapter` | 可选 `desynchronize_episodes`：冷启动后把各行首回合的时限均匀错开，避免全体同时超时重生；只影响调度，不进入世界模型 schema 身份 |

段内指令（站立调高、普通行驶、两档自旋、冲刺）由原生指令池产生；出生族（HEIGHT/MOTION/ROTATION/TERRAIN/REQUEST/RECOVERY）决定出生方式和请求，不直接决定段内指令。普通出生中有一半是地面坐姿起身（原生 birth randomization）。

跳跃扩展项的判据：
- 原生 FLIGHT 只表示双轮载荷 ≤2 N 持续 10 ms，快速收腿不起跳也能触发。腾空奖励（`jump_flight`、`jump_lift`）只发给请求的第一段真实腾空：真实上升（原生 `ascent_seen`：竖直速度 > 0.15 m/s 且升高 > 1 cm）之后才开始，第一次落地即结束，所以每个请求有上限，反复弹跳不得分。窗口由物理起跳事件打开，不依赖参考时钟，晚起跳的真实跳跃同样计分。该锁存由原生 `FirstFlightLatch` 维护。
- `jump_push`：参考要求向上速度（蹬伸段）且机体仍受支撑时，按实际竖直速度占参考速度的比例付分。
- `jump_missed`：参考进入 HOLD 时仍无真实上升，即判为漏跳。此时关闭前进、偏航、高度和竖直速度的正跟踪项，保留跳跃姿态稳定正项及全部代价，并按秒计代价。

参考阶段在 50 Hz 边界采样并保持四个子步；物理相位和锁存逐个 5 ms 子步更新。原生 composer 与想象奖励直接调用同一个核。想象回放目前还没有记录首段腾空锁存：遇到需要腾空项的跳跃行时会直接报错，不做近似。权重是候选值，以原生无助力物理评价为准。

## PPO 教师 `supermimic_rl.ppo`

| 模块 | 内容 |
|---|---|
| `policy` | `TeacherActor`：独立监督的估计器（速度 xyz、轮接触），输出 detach 后进入 actor；均值 `tanh(net)×action_bounds`；高斯策略 |
| `environment` | `CourseEnvironment`：直接读原生 `obs_buf`（actor 5×35 历史、critic 特权特征），汇总原生技能计数（跳跃结局、台阶整场证据、翻正生命周期、出生族、助力计数） |
| `course_gate` | `StageGate`/`CourseGates`，两种模式：<br>• `hold`：在阶段边界前一步停住坐标，窗口内指标或计数比值达标才晋级，超时返回 `gate_timeout`。<br>• `report`：坐标始终按真实转移前进，每个已到达的门槛每轮评判并记录等待轮数，不截停训练；已打开的门槛只决定与其绑定的难度（跳跃高度阶梯）。<br>`rewind` 在修正后重做某阶段 |
| `trainer` | `PPOTrainer`：4096×24 每更新，KL 自适应学习率；`initial_actor`（SHA 绑定）、模仿退火、`course_rewind_stage`、`course_gate_resume` |
| `evaluation` | 原生评价 |

能力判定以 27 场景固定评价（`scripts/evaluate_ppo_skills.py`）为准，不以训练奖励为准。

## SuperMimic 世界模型 V2 `supermimic_rl.world_model_v2`

数据流：原生采集（教师/学生混合，`collection`、`mixing`）→ 紧凑回放（`records`、`layout`、`replay`）→ 多步拟合（`model`、`fitting`、`validation`）→ 想象 rollout（`imagination`，奖励 `imagined_reward`/`reward_assembly`，终止 `termination`）→ actor/critic 更新（`learning`）→ 新的原生采集（`online`）。

- **模型**：一次前向预测四个 5 ms 物理 tick 的增量（位姿、速度、关节、支撑力、各刚体法向载荷、被动膝），以每回合仿真上下文为条件；闭链用带有效域的查找表（`leg_linkage`），地形用固定图集（`terrain`）。
- **任务协议**：`motion_task`、`stair_task`/`stair_source`、`jump_task`、`recovery_task` 在想象中逐 tick 复现原生请求、相位和成功判据。
- **学习稳定性**（`LearningConfig`/`OnlineConfig`，默认关闭以保持旧行为）：
  - `critic_input_clip`：critic 归一化输入截断，防止想象状态特征远离起点统计时自举外推发散。
  - `calibrate_contact_floor`：每轮在留出行上为各非轮刚体标定接触载荷下限，只在想象中把低于下限的预测置零。
  - `bootstrap_gradient=false`：末端 critic 值只作为目标，不向 actor 传梯度。
  - 原生回退护栏（`guard`）：用最近 N 轮采集的学生回报中位数与历史最佳中位数比较，短暂的全体重生不会被误判。
    - `restore` 模式：恢复最佳 actor，并把 actor 学习率减半。
    - `report` 模式：只记录判断，不回滚、不改学习率，由人工评审。
    - 续训时只允许改护栏设置，改动记录在迁移回执中。
- **学生与教师**：`student_transfer` 在同一 5×35 接口和动作界下逐值复制 PPO actor；`teacher` 封装原生 PPO 教师用于混合采集。
- **数据报告**：`data_report` 按技能统计覆盖（接触相位、宽腿动作）与多步预测、接触事件误报。

模块组成：

| 当前模块 | 实际实现 |
|---|---|
| 运动网络 | 209 → 2048 → 2048 → 2048 → 224；隐藏层 LayerNorm + SiLU |
| 输入 | 物理特征30 + 闭链几何12 + 执行记忆19 + 底盘动作6 + 物理上下文65 + 地形扫描77；不输入绝对xy/yaw |
| 输出与时间 | 50 Hz 一次前向输出4×56，分别对应5/10/15/20 ms；每子步26维运动增量、26维接触标签、4维实际被动膝位置/速度。末子步就是区间端点 |
| 解析模块 | 闭链一维相对角查表；轮力矩延迟队列、传感器、地形atlas、已接入任务状态、200 Hz原生奖励和终止时钟由代码推进 |
| 底盘 Actor | 5×35 历史，不显式输入云台，输出 6 个底盘动作；45 维祖先 checkpoint 经 migrations 显式选列迁移 |
| 云台 | 独立控制；真实物理与WM保留云台运动、质量/惯量及反作用。云台不属于RL动作 |
| 训练 | 原生 4096×24 采集；世界模型 H8 拟合、策略 H16 想象；回放可放在 CPU |

任务协议细节见 [WORLD_MODEL_V2_JUMP_TASK.md](WORLD_MODEL_V2_JUMP_TASK.md)、[WORLD_MODEL_V2_RECOVERY_TASK.md](WORLD_MODEL_V2_RECOVERY_TASK.md)。

## 部署接口

- 动作 6 维：四腿位置目标（标称角 + 0.25·a rad，最近绕圈）、两轮速度目标；腿界由 runtime 决定（当前 ±4π）。
- Actor 输入：5 帧 × 35 维历史（旧 → 新），估计器在图内；ONNX 输入 `[1, 5, 35]`，输出 `[1, 6]`。
- 部署端动作界必须与训练 runtime 一致。

技能触发在部署端必须复现训练中的任务语义（观测 35 维中的模式 one-hot、请求高度、指令时钟由下列状态驱动）：

| 技能 | 触发与准入 | 部署端职责 |
|---|---|---|
| 跳跃 | 操作员请求（模式 JUMP + 高度 ≤ 0.40 m）；请求期间偏航角速度指令为 0，可保留前进速度（训练中的跳跃请求不带偏航） | 原生协议对新 ID 一律消费（被拒的 ID 不会再执行），所以部署端须保存待执行请求，在准入成立时才发出：<br>• 双轮着地且安静：竖直速度 ≤ 0.15 m/s、倾角 ≤ 0.25 rad、角速度 ≤ 0.5 rad/s<br>• 高度与指令差 ≤ 1.5 cm<br>• 启动 ≥ 1 s，上次跳跃结束 ≥ 1 s<br>请求结束（成功需落地后连续 2 s 安静，或 6 s 超时）前保持偏航指令为 0 |
| 翻正 | 倒地检测 → 模式 RECOVERY | 先执行准备轨迹（腿以 1 rad/s 插值到动作框内的指定姿态）再交给 Actor，与原生准备段一致 |
| 台阶 | 前向 ToF 检测台阶 | 运行同一 ToF 台阶状态机：STAIR 模式 + 高度指令 +0.16 m（上限 0.40 m）保持 2 s；速度指令仍由操作员给出 |

同一边界的新跳跃优先于新台阶请求；已进行中的台阶任务会阻止新跳跃，部署端须保留待执行请求，退出后使用新 ID 发出。仿真台阶完成条件还使用后轮越过立面的几何信息，不能声称单个前向 ToF 已能提供实车完成判据；RMCS 实施时需补齐这一估计与退出合同。本工程当前不实施 RMCS 控制。

## 测试

`tests/` 不依赖 Isaac。原生行为用 `tests/support/` 中的桩和夹具模拟（原生任务协议、学习器夹具、核心投影假对象、PPO 目标环境）；测试模块之间不互相导入。原生物理验证通过 `scripts/` 中的审计脚本在 Isaac 镜像上执行。
