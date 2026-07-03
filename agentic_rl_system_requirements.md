# Agentic RL 系统需求分析

> 本文档面向产品、开发和测试人员描述 Agentic RL 在线训练系统的系统需求。需求依据 `agentic_rl_design.md` 的架构与接口设计整理，并结合 `agentic_rl_design_opt.md` 中“核心逻辑拆分、统一入口、可由 CLI/HTTP/Actions/Lambda 调用”的思路，强调模块边界清晰、调用路径可复用、验收结果可观察。

## 0. 范围说明

### 0.1 系统目标

Agentic RL 系统需要把 Agent 真实执行轨迹转化为可训练样本，按调度策略触发 RL 训练，产出并发布 LoRA 版本，再让后续推理请求可以使用新的策略版本。系统需要同时支持内置 Agent 和第三方 Agent 接入，并提供可量化的训练效果评估能力。

### 0.2 主要角色

| 角色 | 关注点 |
| --- | --- |
| Agent 用户 | 希望 Agent 在持续使用中变得更稳定、更准确、更符合任务目标 |
| 一体机管理员 | 希望查看轨迹、触发训练、管理 LoRA 版本、回滚异常版本 |
| 第三方 Agent 开发者 | 希望用标准协议上传轨迹或接入 rollout 环境，不绑定 JiuwenSwarm 内部实现 |
| 训练/算法工程师 | 希望样本、奖励、训练配置和指标可追踪，便于调参与复现实验 |
| 测试人员 | 希望从接口、状态流转、异常处理、指标产出和回归效果等角度完成验收 |

### 0.3 核心模块

| 模块 | 主要职责 |
| --- | --- |
| `Agent Gateway` | 对外统一接入轨迹、查询轨迹、管理 LoRA；承担鉴权、参数校验、协议适配 |
| `Trajectory API` | 接收 `RolloutTrajectory`，校验并归一化为 `TrainingSample` |
| `RL Scheduler` | 监控样本状态、判断训练触发条件、触发多采样或训练任务、更新样本状态 |
| `Agent Env` | 执行 Agent rollout，支持多采样、任务隔离、超时控制和结果回调 |
| `Agent Infer` | 提供模型推理和 LoRA 热加载/卸载能力 |
| `Model Trainer` | 执行 PPO/GRPO 等训练任务，产出 LoRA、checkpoint 和训练指标 |
| `LoRA Repo` | 管理 LoRA 版本、active 指针、指标元数据和回滚能力 |
| `Redis/Storage` | 保存轨迹、样本状态、调度状态和训练元数据 |

## 1. 系统需求一：RL 流程

### 1.1 需求描述

系统需要支持从轨迹采集、样本入库、奖励计算、训练调度、模型训练、LoRA 发布、推理热加载到状态回写的完整闭环。测试验收时应关注每个环节是否可独立验证、状态是否可追踪、失败是否可恢复、产物是否可回滚。

### 1.2 用户场景

| 场景 ID | 用户场景 | 期望结果 |
| --- | --- | --- |
| RL-FLOW-001 | JiuwenSwarm 在用户会话中采集 LLM/Agent 执行轨迹，并批量上传到 Gateway | Gateway 返回 accepted 结果，样本进入 `pending` 状态 |
| RL-FLOW-002 | Scheduler 发现某个用户或任务的 pending 样本达到阈值 | Scheduler 将样本标记为 `training`，并触发训练或多采样 |
| RL-FLOW-003 | 使用 GRPO 时，Scheduler 对同一个 prompt 拉起多路 rollout | Agent Env 返回同一 `group_id` 下的多条轨迹，Scheduler 可聚合并计算组内奖励 |
| RL-FLOW-004 | Model Trainer 完成训练并输出 LoRA | Scheduler 发布 LoRA 到 LoRA Repo，并调用 Agent Infer 热加载 |
| RL-FLOW-005 | 管理员查询训练状态和 LoRA 版本 | 能看到样本数、训练任务状态、指标、active LoRA 版本和历史版本 |
| RL-FLOW-006 | 训练失败或 LoRA 热加载失败 | 系统按错误类型回滚样本状态或标记失败，不产生错误 active 版本 |

### 1.3 功能需求

| 编号 | 需求 | 测试关注点 |
| --- | --- | --- |
| RL-FLOW-F01 | Gateway 必须支持批量上传 rollout trajectory | 请求字段校验、批量部分成功、重复 ID 幂等、非法状态拒绝 |
| RL-FLOW-F02 | Gateway 必须把有效轨迹归一化为可训练样本 | `trajectory_id`、`rollout_id`、`group_id`、`user_id`、`task_id`、`llm_id`、`policy_version` 保留完整 |
| RL-FLOW-F03 | Scheduler 必须按阈值、用户、任务、算法配置触发调度 | 未达阈值不触发；达阈值只触发一次；并发调度不重复消费 |
| RL-FLOW-F04 | Scheduler 必须支持单样本训练和 GRPO 多采样训练路径 | 分支选择正确；多采样结果按 `group_id` 聚合；无效轨迹被剔除 |
| RL-FLOW-F05 | Model Trainer 必须返回结构化训练结果 | `job_id`、`status`、`lora_path`、`checkpoint_path`、`metrics`、`error` 字段可验证 |
| RL-FLOW-F06 | LoRA Repo 必须支持发布、查询、激活、回滚、删除 | active 指针唯一；历史版本可查；回滚后推理路由使用旧版本 |
| RL-FLOW-F07 | Agent Infer 必须支持 LoRA 热加载和卸载 | 加载成功可推理；重复加载幂等；加载失败不会切换 active 指针 |
| RL-FLOW-F08 | 系统必须提供状态查询和审计信息 | 可查询 pending/training/trained/failed 数量、训练任务、错误原因和版本元数据 |

### 1.4 主要服务模块

| 流程阶段 | 主模块 | 依赖模块 |
| --- | --- | --- |
| 轨迹接入 | `Agent Gateway` / `Trajectory API` | `Redis/Storage` |
| 样本调度 | `RL Scheduler` | `Redis/Storage`、`LoRA Repo` |
| 多采样 rollout | `Agent Env` | `Agent Infer`、`Sandbox` |
| 奖励计算 | `RL Scheduler` 或 Reward 子模块 | `Redis/Storage` |
| 训练执行 | `Model Trainer` | 训练集群、模型存储 |
| LoRA 发布 | `LoRA Repo` | 文件存储、元数据存储 |
| LoRA 生效 | `Agent Infer` | `LoRA Repo`、`RL Scheduler` |

### 1.5 验收标准

| 验收项 | 前置条件 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| RL-FLOW-A01 轨迹上传成功 | Gateway、Redis 可用 | 上传 1 批合法 `RolloutTrajectory` | 返回 `ok=true`，`accepted_samples` 等于合法样本数，Redis 中样本状态为 `pending` |
| RL-FLOW-A02 非法轨迹被拒绝 | Gateway 可用 | 上传缺少 `trajectory_id`、`user_id` 或 `llm_id` 的轨迹 | 返回明确错误；非法样本不入库；合法样本不受影响 |
| RL-FLOW-A03 调度去重 | Redis 中同一用户有足量 pending 样本 | 并发触发两次 `TriggerSchedule` | 只有一个训练任务消费该批样本；样本不会重复进入多个 job |
| RL-FLOW-A04 GRPO 多采样聚合 | 算法配置为 `grpo`，`num_rollouts=4` | Scheduler 提交 rollout 任务 | Agent Env 返回 4 条同 `group_id` 轨迹；Scheduler 生成 1 组训练样本和对应 rewards |
| RL-FLOW-A05 训练成功闭环 | Trainer 可用，LoRA Repo 可用，Infer 可用 | 完成一次训练 | 样本状态变为 `trained`；LoRA Repo 新增版本；Agent Infer 加载对应 `lora_name` |
| RL-FLOW-A06 训练失败恢复 | Trainer 模拟 `retryable_failed` | 触发训练 | 样本回到 `pending` 或按重试策略等待；错误信息可查询 |
| RL-FLOW-A07 不可恢复失败 | Trainer 模拟数据格式错误 | 触发训练 | 样本标记 `failed`；失败原因记录；不会发布 LoRA |
| RL-FLOW-A08 LoRA 热加载失败 | LoRA Repo 发布成功但 Infer 加载失败 | Scheduler 调用加载 | active 指针不切到不可用版本，或版本标记为 `load_failed`；管理员可查询错误 |
| RL-FLOW-A09 回滚可用 | 已有至少两个 LoRA 版本 | 管理员回滚到旧版本 | active 版本变更；后续推理请求使用旧 `lora_name/path` |

## 2. 系统需求二：RL 支持第三方 Agent

### 2.1 需求描述

系统需要允许第三方 Agent 在不依赖 JiuwenSwarm 内部实现的情况下接入 RL 闭环。第三方 Agent 可以通过标准协议上传已完成轨迹，也可以由 Agent Env 通过适配器拉起执行 rollout。测试验收时应关注协议兼容、身份隔离、错误隔离、回调可靠性和多租户数据边界。

### 2.2 用户场景

| 场景 ID | 用户场景 | 期望结果 |
| --- | --- | --- |
| RL-3P-001 | 第三方 Agent 已有自己的执行框架，只希望上传轨迹参与训练 | 通过 Gateway 上传 `agent-rollout-v1` 轨迹即可入库 |
| RL-3P-002 | 第三方 Agent 希望由 RL 系统拉起多采样 rollout | Agent Env 通过 adapter 启动第三方 Agent，并按标准结构返回轨迹 |
| RL-3P-003 | 第三方 Agent 使用自己的工具调用和环境元数据 | 轨迹中的 `steps`、`tools`、`metadata` 可以保留并用于奖励计算 |
| RL-3P-004 | 第三方 Agent 上传错误或超时 | 错误只影响该 rollout/job，不影响其他用户或其他 Agent |
| RL-3P-005 | 管理员按第三方 Agent 查询训练产物 | 可按 `source`、`agent_id`、`llm_id`、`user_id` 查询轨迹、样本、LoRA 和指标 |

### 2.3 功能需求

| 编号 | 需求 | 测试关注点 |
| --- | --- | --- |
| RL-3P-F01 | Gateway 必须支持标准 `RolloutTrajectory` 协议 | 第三方只需满足协议字段，不依赖内部 Rail 实现 |
| RL-3P-F02 | Gateway 必须记录第三方来源信息 | `source`、`agent_id`、`agent_version`、`workspace_id` 可入库可查询 |
| RL-3P-F03 | Agent Env 必须支持 Agent adapter 扩展 | 可配置不同 adapter；同一 rollout 接口可启动不同 Agent |
| RL-3P-F04 | 第三方 Agent rollout 必须受 Sandbox 隔离 | 文件、网络、环境变量、超时和资源限制可控 |
| RL-3P-F05 | 第三方 Agent 回调必须幂等 | 重复 callback 不重复生成样本；乱序 callback 状态可收敛 |
| RL-3P-F06 | 第三方轨迹必须支持协议版本演进 | 不支持的 `protocol_version` 返回明确错误；兼容版本可正常处理 |
| RL-3P-F07 | 多租户数据必须隔离 | `user_id`、`workspace_id`、`llm_id` 查询和训练不会串数据 |
| RL-3P-F08 | 第三方 Agent 异常必须可诊断 | 超时、启动失败、回调失败、格式错误都有结构化错误码和日志关联 ID |

### 2.4 主要服务模块

| 能力 | 主模块 | 依赖模块 |
| --- | --- | --- |
| 标准协议接入 | `Agent Gateway` / `Trajectory API` | `Redis/Storage` |
| 第三方 Agent 启动 | `Agent Env` | Agent adapter、`Sandbox` |
| 推理调用 | 第三方 Agent 或 `Agent Env` | `Agent Infer` |
| 回调与聚合 | `RL Scheduler` | `Agent Env`、`Redis/Storage` |
| 来源查询 | `Agent Gateway` | `Redis/Storage`、`LoRA Repo` |
| 错误诊断 | `Agent Gateway`、`Agent Env`、`RL Scheduler` | 日志、Trace、审计表 |

### 2.5 验收标准

| 验收项 | 前置条件 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| RL-3P-A01 标准轨迹上传 | 第三方 Agent 准备合法 `agent-rollout-v1` JSON | 调用 `POST /v1/gateway/rollouts` | 样本入库；`source=third_party`、`agent_id` 可查询 |
| RL-3P-A02 协议缺字段校验 | 第三方轨迹缺少必要字段 | 上传轨迹 | Gateway 返回字段级错误；不产生 pending 样本 |
| RL-3P-A03 协议版本拒绝 | `protocol_version=unknown-v9` | 上传轨迹 | 返回不支持协议版本；错误可在日志中定位 |
| RL-3P-A04 Adapter rollout | 已配置第三方 Agent adapter | Scheduler 提交 `SubmitRolloutTask` | Agent Env 成功启动 adapter，并返回标准 `RolloutTrajectory[]` |
| RL-3P-A05 Sandbox 隔离 | 第三方 Agent 尝试访问越权路径或超出资源 | 执行 rollout | 任务失败并记录安全错误；其他 rollout 不受影响 |
| RL-3P-A06 回调幂等 | 同一 `rollout_id` 完成后重复发送 callback | 连续发送两次 callback | 只生成一份训练样本；第二次返回已处理或幂等成功 |
| RL-3P-A07 多租户隔离 | 两个用户上传同一 `task_id` 轨迹 | 分别查询和触发训练 | 用户 A 无法查询或训练用户 B 数据；LoRA 版本按用户隔离 |
| RL-3P-A08 异常可观测 | 第三方 Agent 启动失败 | 执行 rollout | `RolloutTaskStatus` 返回 `failed`；包含错误码、错误信息和 trace id |

## 3. 系统需求三：RL 训练效果

### 3.1 需求描述

系统需要能验证 RL 训练是否真正带来效果提升，而不是只验证训练任务“跑通”。训练效果需要从离线指标、在线灰度指标、版本对比、回滚能力和负向指标控制几个维度验收。测试人员需要能拿到固定评测集、基线版本、新 LoRA 版本和指标产物，并能复现一次评估。

### 3.2 用户场景

| 场景 ID | 用户场景 | 期望结果 |
| --- | --- | --- |
| RL-EVAL-001 | 算法工程师完成一次 RL 训练后评估新 LoRA | 系统产出 reward、成功率、失败率、KL、loss 等指标 |
| RL-EVAL-002 | 管理员决定是否激活新 LoRA | 可以查看新旧版本在固定评测集上的对比 |
| RL-EVAL-003 | 新版本在线表现变差 | 管理员可以快速回滚到上一 active 版本 |
| RL-EVAL-004 | 测试人员做回归测试 | 使用相同评测集和随机种子可复现核心指标 |
| RL-EVAL-005 | 训练数据质量下降 | 系统能暴露无效轨迹比例、奖励分布异常和样本过滤数量 |

### 3.3 功能需求

| 编号 | 需求 | 测试关注点 |
| --- | --- | --- |
| RL-EVAL-F01 | 每次训练必须记录训练配置和数据范围 | `algorithm`、样本数量、时间范围、用户/任务维度、基础模型、LoRA rank 可追踪 |
| RL-EVAL-F02 | 每次训练必须产出结构化训练指标 | 至少包含 `loss`、`kl`、`reward_mean`、`reward_std`、`sample_count` |
| RL-EVAL-F03 | 系统必须支持固定评测集对比 | base/old LoRA/new LoRA 使用相同 prompts、采样配置和评估器 |
| RL-EVAL-F04 | 系统必须定义版本激活门禁 | 未达到阈值的新 LoRA 不自动 active，或需管理员手动确认 |
| RL-EVAL-F05 | 系统必须支持在线灰度观察 | 可按用户、任务或流量比例使用新 LoRA，并记录效果指标 |
| RL-EVAL-F06 | 系统必须支持负向指标监控 | 延迟、失败率、工具调用错误率、格式错误率不能显著恶化 |
| RL-EVAL-F07 | 系统必须支持效果回滚 | 指标不达标或线上异常时可回滚 active LoRA |
| RL-EVAL-F08 | 评估结果必须可导出 | 支持 JSON/Markdown 或接口查询，便于测试报告归档 |

### 3.4 主要服务模块

| 能力 | 主模块 | 依赖模块 |
| --- | --- | --- |
| 训练指标产出 | `Model Trainer` | 训练框架、样本存储 |
| 指标汇总与门禁 | `RL Scheduler` | `Model Trainer`、评估器 |
| 版本指标登记 | `LoRA Repo` | 元数据存储 |
| 固定评测执行 | `Agent Env` 或 Evaluation Runner | `Agent Infer`、评测集 |
| 线上灰度 | `Agent Gateway` / `Agent Infer` | LoRA 路由、监控系统 |
| 回滚 | `LoRA Repo` / `Agent Infer` | 管理接口、热加载接口 |

### 3.5 指标口径

| 指标 | 说明 | 验收建议 |
| --- | --- | --- |
| `reward_mean` | 训练或评估样本平均奖励 | 新版本应高于基线阈值，阈值按任务配置 |
| `success_rate` | 固定评测集任务成功比例 | 新版本不低于旧版本，核心任务应提升 |
| `invalid_rate` | 无效输出、格式错误、无法解析的比例 | 不得高于配置阈值 |
| `tool_error_rate` | 工具调用失败或参数错误比例 | 不得高于旧版本显著比例 |
| `latency_p95` | 推理或 Agent 任务 P95 延迟 | 不得超过性能预算 |
| `kl` | 策略偏离参考模型程度 | 不得超过算法配置上限 |
| `sample_count` | 参与训练样本数 | 必须达到训练触发阈值 |
| `rollback_time` | 从发现异常到旧版本生效时间 | 应满足运维 SLA |

### 3.6 验收标准

| 验收项 | 前置条件 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| RL-EVAL-A01 训练指标完整 | 完成一次训练 | 查询 `TrainJobResult` 和 LoRA metadata | 指标字段完整；训练配置和样本范围可追踪 |
| RL-EVAL-A02 固定评测可复现 | 准备固定 prompts、固定评估器、固定采样参数 | 对 base、旧 LoRA、新 LoRA 各跑一次评测 | 评测结果包含版本号、评测集版本、随机种子和指标 |
| RL-EVAL-A03 新版本达到门禁 | 新 LoRA 指标高于阈值 | Scheduler 执行发布流程 | LoRA 可被激活；metadata 记录通过门禁的指标 |
| RL-EVAL-A04 新版本未达门禁 | 模拟 reward 或成功率低于阈值 | Scheduler 执行发布流程 | LoRA 不自动 active；状态为 `rejected` 或等待人工确认 |
| RL-EVAL-A05 负向指标保护 | 新版本输出格式错误率升高 | 执行评测 | 即使 reward 提升，也因负向指标超阈值阻止自动激活 |
| RL-EVAL-A06 线上灰度验证 | 新版本已发布但未全量 | 灰度 5% 用户或指定测试用户 | 灰度请求携带新 `lora_version`；指标按版本分组可查询 |
| RL-EVAL-A07 回滚验证 | 新版本 active 后模拟线上异常 | 调用回滚接口 | active 指针切回旧版本；Infer 加载旧版本；后续推理使用旧版本 |
| RL-EVAL-A08 评估报告导出 | 完成评测 | 导出 JSON/Markdown 报告 | 报告包含版本、样本范围、指标、结论和失败样例摘要 |

## 4. 测试验收建议

### 4.1 分层测试

| 测试层级 | 建议覆盖 |
| --- | --- |
| 单元测试 | 字段校验、状态机转换、奖励归一化、LoRA 版本选择、指标门禁判断 |
| 接口测试 | Gateway、Scheduler、Agent Env、Trainer、LoRA Repo、Agent Infer 的成功和失败响应 |
| 集成测试 | 从轨迹上传到 LoRA 热加载的端到端链路 |
| 兼容性测试 | `agent-rollout-v1`、`rail-v1`、第三方 Agent adapter 协议兼容 |
| 幂等测试 | 重复上传、重复 callback、重复训练触发、重复 LoRA 加载 |
| 异常测试 | Redis 异常、Trainer OOM、Agent 超时、LoRA 文件缺失、Infer 加载失败 |
| 效果回归测试 | 固定评测集、新旧版本对比、负向指标门禁、回滚 |

### 4.2 最小端到端验收路径

1. 上传一批合法轨迹，确认样本入 `pending`。
2. 手动触发 Scheduler，确认样本进入 `training`。
3. 使用 mock Agent Env 返回多路 rollout，确认 Scheduler 聚合和奖励计算。
4. 使用 mock Trainer 返回成功训练结果和 LoRA 路径。
5. 发布 LoRA 到 LoRA Repo，确认 active 版本生成。
6. 调用 Agent Infer 热加载，确认推理请求能按 `lora_name` 路由。
7. 查询状态和指标，确认样本、训练任务、LoRA 版本和评估报告可追踪。
8. 执行回滚，确认旧版本恢复生效。

### 4.3 测试数据要求

| 数据类型 | 要求 |
| --- | --- |
| 合法轨迹 | 覆盖单轮、多轮、工具调用、成功/失败状态 |
| 非法轨迹 | 缺字段、字段类型错误、重复 ID、未知协议版本、跨用户数据 |
| 多采样数据 | 同一 `group_id` 下至少 4 条不同 rollout |
| 训练样本 | 覆盖正奖励、负奖励、零奖励、无效样本过滤 |
| LoRA 版本 | 至少准备 base、旧 active、新候选三个版本 |
| 评测集 | 固定版本、固定 prompts、固定评估器、固定采样配置 |

## 5. 非目标

以下内容不作为本文档第一阶段验收范围：

- 具体 PPO/GRPO 算法数学实现的正确性证明。
- 具体训练框架、GPU 拓扑和算子级性能优化。
- 具体奖励模型训练方案。
- 长期 A/B 实验平台建设。
- 多地域灾备和跨集群 LoRA 分发。
