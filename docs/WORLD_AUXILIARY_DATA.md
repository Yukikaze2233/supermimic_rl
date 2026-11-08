# 世界模型的辅助原生数据

`OnlineConfig.world_datasets` 可以让已采集的技能轨迹持续参与动力学训练。
每个来源指定 `name`、`path`、`sha256`、`fraction`、`train_env_ids` 和
`validation_env_ids`。默认列表为空，原训练行为和随机数路径保持不变。

数据使用 `supermimic-world-v2-dataset-v1` 紧凑格式，模型合同、状态 schema、
训练字段和当前原生环境一致。每条序列仍使用 `CompactReplay` 的边界规则，
不跨重置、无效区间或控制权交接；训练行和验证行明确分开。
多个来源的 `fraction` 之和小于 1，其余窗口来自当前在线 replay。

辅助文件保存在 CPU，通过 mmap 只读加载，每批仅传输抽到的窗口。
在线 sampler 先按原路径取完整批次，再由独立 RNG 替换选中的动力学窗口。
Actor 的想象起点与估计器训练仍按在线 replay 采样；辅助数据不会变成
行为克隆目标，也不冒充 4096 环境的物理恢复状态。

完整 checkpoint 保存来源配置、训练计数和辅助训练/验证 RNG，不复制数据文件。
搬移运行时需要同时保留所引用的文件；恢复会核对 SHA256。修改来源配置会写入
`resume_migrations`，模型、优化器、在线 replay 和已有采样 RNG 都保留。

每轮 `auxiliary_world_windows` 记录各来源实际参加训练的窗口数。
常规 checkpoint 周期另写 `auxiliary-world-validation-*.json`，在各来源独立验证行上
检查多步运动和接触预测，同时保留原在线数据验证。辅助诊断的 `ValueError` 单独记录，
不会截停有效的训练循环。新增数据是否改善技能，仍需用原生仿真评价学生。

采样比例是训练设置，应根据数据曝光与留出结果选择。增加同一批数据的曝光不等于
增加新接触工况；跳跃、台阶和恢复仍需持续采集新的真实轨迹。
