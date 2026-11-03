# SuperMimic RL for Wheeled-Legged Robots

闭链轮足整车（带独立活动云台）的强化学习工程，包含两条共用原生环境的训练线：

- **PPO 教师**：在原生 Isaac 课程上按能力门槛晋级训练（平地 → 小跳 → 动态指令 → 地形 → 200 mm 台阶与二级平台 → 四向翻正 → 混合域随机化），输出 5×35 历史输入的可部署 actor。
- **SuperMimic 世界模型 V2**：参考 [SuperMimic](https://arxiv.org/abs/2610.28480) 的在线模型学习与短视界策略梯度，在原生数据上拟合多步世界模型，在想象中改进从 PPO 精确初始化的学生 actor。

这是面向轮足平台的适配研究实现，**不是论文作者的官方代码**。

上游环境（`transformer_rl` 固定版本的 `WheeledLabEnv`）、机械资产与辨识执行链（电机曲线、轮延迟与高速包络、气簧）被复用且不修改；本工程以包装与注入的方式安装课程、动作合同、奖励扩展、域随机化与训练专用助力，并把这些改动写入每次运行的回执。

## 文档

- [架构](docs/ARCHITECTURE.md)：包结构、依赖方向、原生课程、PPO、世界模型 V2、部署接口与测试组织。
- [运行说明](docs/OPERATIONS.md)：本地测试、Isaac 镜像上的作业、PPO 与世界模型流水线命令。
- 世界模型 V2 任务合同：[跳跃任务](docs/WORLD_MODEL_V2_JUMP_TASK.md)、[翻正任务](docs/WORLD_MODEL_V2_RECOVERY_TASK.md)。
- 辨识执行链绑定：[configs/identified/20261031](configs/identified/20261031/README.md)。

## 快速开始

```bash
pip install -e '.[test]'
export PYTHONPATH=<transformer_rl>/src
python -m pytest -q          # 不需要 Isaac
```

Isaac 相关入口（训练、采集、评价）经 `scripts/run_gpufree.sh` 启动，详见运行说明。

## 目录

| 路径 | 内容 |
|---|---|
| `src/supermimic_rl/course/` | 原生整车训练课程 |
| `src/supermimic_rl/ppo/` | PPO 教师 |
| `src/supermimic_rl/world_model_v2/` | SuperMimic 世界模型 V2 |
| `src/supermimic_rl/adapters/` | 通用模拟器边界 |
| `configs/gpufree/` | runtime、课程、factory 与训练配置（sha256 绑定） |
| `scripts/` | 训练、采集、拟合、评价、导出与审计入口 |
| `tests/` | 无 Isaac 的测试；共享构造器在 `tests/support/` |
