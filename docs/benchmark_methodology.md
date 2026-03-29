# Benchmark Methodology

这份文档描述当前 benchmark 如何评价“可信财务深度研究 RAG Agent”，而不是历史版本的 section-style writer。

## 1. 评测目标

benchmark 不是为了证明系统比大模型“更聪明”。  
它只验证更窄的事情：

1. 能不能在固定资料包内找回对的证据。
2. 能不能把研究问题拆成合理的子问题。
3. 能不能把子问题放进正确的研究通道。
4. 能不能在证据不足时补查或拒答。
5. 能不能把决策轨迹完整回放出来。

## 2. 评测层次

### 传统结果层

- `retrieval_hit`
- `required_fact_recall`
- `verified_claim_coverage`
- `unsupported_claim_rate`
- `answer_quality`

这层回答“最后写出来的东西怎么样”。

### 研究控制层

- `subquestion_completion_rate`
- `required_subquestion_coverage`
- `required_slot_coverage`
- `decision_trace_coverage`
- `decision_replay_consistency`
- `refused_subquestion_rate`
- `hard_fact_completion_rate`
- `semantic_completion_rate`

这层回答“研究控制器是否在按预期工作”。

### 安全层

- `wrong_entity_rate`
- `wrong_period_rate`

这层回答“是否越过了可信边界”。

## 3. 指标优先级

接受顺序必须是：

1. `wrong_entity_rate`
2. `wrong_period_rate`
3. `unsupported_claim_rate`
4. `required_slot_coverage`
5. `decision_replay_consistency`
6. `verified_claim_coverage`
7. `answer_quality`

也就是说：

- 先不犯低级错误
- 再看有没有答到关键点
- 最后才看写得好不好

## 4. 必答槽位

每道 benchmark 题都带 `must_cover` 或 `required_slots`。  
系统不能靠“写了很多句子”拿高分，必须看：

- 这些槽位有没有被完成的子问题实际覆盖
- 覆盖是否来自合法证据和正确时期

## 5. 决策回放

当前 benchmark 会读取控制器的：

- `subquestion_results`
- `research_trace.subquestions`
- `research_trace.decisions`

并检查：

- 每个子问题是否有结果状态
- 每个子问题是否有对应决策记录
- 回放状态是否与最终状态一致

如果不一致，记为 `decision_replay_consistency = 0`。

## 6. 失败标签

当前失败标签分三类：

### 安全失败

- `R1_missing_primary_source`
- `R2_entity_leakage`
- `S4_temporal_confusion`

### 研究失败

- `S1_question_tree_incomplete`
- `S1_missing_required_slot`
- `A2_replay_inconsistent`
- `V4_conflict_not_detected`

### 结果失败

- `S5_overclaim`
- `S3_wrong_comparison_dimension`

## 7. Showcase 与 Diagnostic

### Diagnostic

目标是揭露系统问题，不是好看。

### Showcase

目标是演示：

- 保守输出
- 证据绑定
- 子问题完成
- 决策可回放

它不能被当成完整 benchmark 的替代。

## 8. 当前边界

benchmark 题集保留了历史 comparison 题。  
但当前主系统的承诺能力仍然是单公司深度研究优先。

所以要诚实地区分两件事：

- comparison 题对系统是压力测试
- single-company 题才是当前产品边界内的核心验证

如果未来把多公司研究做成正式能力，应该新增版本，而不是假装当前版本已经完全支持。
