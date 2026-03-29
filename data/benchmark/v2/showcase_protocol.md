# Showcase Protocol

这个协议定义 `v2` 的 `showcase` 轨道。  
它不是完整 benchmark 的替代，而是当前深度研究控制器的公开演示轨道。

## 为什么要单独有 Showcase

diagnostic 题集的职责是暴露问题。  
showcase 题集的职责是证明当前产品边界内，系统可以做到：

- 证据绑定
- 子问题完成
- 决策可回放
- 在证据不足时保守输出

这两者不是一个工作。

## 当前目标包络

`trust_showcase_v2` 的目标不是“写得最好”，而是先过这些门槛：

- `wrong_entity_rate <= 0.0`
- `wrong_period_rate <= 0.10`
- `unsupported_claim_rate <= 0.15`
- `required_subquestion_coverage >= 0.75`
- `required_slot_coverage >= 0.75`
- `decision_replay_consistency >= 1.0`

如果没过，就直接报没过，不包装。

## 题目规则

showcase 题目必须：

- 单公司
- 锁定时期
- `must_cover` 不超过 `3`
- 至少需要 `2` 个不同来源
- 更适合 evidence-bound 输出，而不是长篇 thesis

showcase 题目不应该：

- 让系统自由比较两家公司孰优孰劣
- 要求开放式投资建议
- 需要很多隐含推理链条
- 鼓励没有证据的因果解释

## 当前题目池

`trust_showcase_v2` 包含：

- `BO-002`
- `NUM-001`
- `TS-002`
- `TS-003`
- `SRC-001`

共同特征：

- 中等难度
- 单公司
- 时期明确
- 槽位清楚
- 适合验证研究控制器而不是拼文风

## 反 P-Hacking 规则

- 不能在看到最新分数后再换题
- 不能把 showcase 当完整 benchmark 报告
- 如果 item list 变化，必须改 profile 名
- 如果题目 wording、evidence 或 target 变化，必须 bump benchmark version
