---
name: auditable-rag-oss-workbench
description: 将 FinTrust RAG 定位为可信金融 RAG 控制层，并补齐 benchmark 取证、高 star OSS 启发的本地审计能力。
status: backlog
created: 2026-04-30T06:06:13Z
updated: 2026-05-01T10:11:43Z
---

# PRD：FinTrust RAG 可信金融审计工作台

## 1. 执行摘要

我们要把 FinTrust RAG 从“多 Agent 财务研究 demo”升级为“可信金融 RAG 审计工作台”。目标用户是作品集/面试评审者、金融 RAG 构建者和项目维护者；他们需要判断系统是否真的能在财务场景中做到证据可查、数字可信、失败可诊断，而不是只看到一段流畅 memo。

本次升级的解决方案不是堆更多 Agent，也不是接入一整套开源平台，而是把现有的检索、验证、trace 和 benchmark 能力产品化：让评审者能在本地看到 benchmark forensics、retrieval lab、trace viewer、document fidelity harness 和 run ledger。成功标准是：3 分钟内能看懂项目可信在哪里，30 分钟内能复盘一次失败，后续每次 benchmark 改动都能用本地 artifact 解释。

## 2. 问题陈述

### 2.1 用户视角问题框架

**我是谁：**
我是一个需要评估 FinTrust RAG 含金量的人，可能是面试官、作品集评审者、金融 RAG 开发者，或者项目维护者。

**我想完成什么：**

- 快速判断这个项目是否超过普通 RAG / Agent demo。
- 确认财务数字、期间、来源和结论是否有证据支撑。
- 定位 benchmark 失败到底发生在检索、文档解析、事实召回、claim verification 还是最终发布。
- 判断引入 GraphRAG 或其他 backend 是否真的带来收益，而不是只增加复杂度。

**但我被什么阻碍：**

- README 第一印象仍偏“多 Agent 编排”，容易让人以为核心卖点是概念包装。
- benchmark 结果有分数，但 failure diagnostics 没有成为主展示面。
- retrieval trace、verification rows 和 artifact 已存在，但需要读 raw JSON 才能理解。
- 文档解析质量对 benchmark 影响很大，却没有独立的 fidelity report。
- 高 star OSS 借鉴点尚未转化为明确、可复现、可验收的本地能力。

**根因：**
项目已经有 trust-first 的工程基础，但产品表达仍围绕“系统如何跑起来”，而不是“用户如何验证系统可信”。

**最终问题陈述：**
评审者和维护者需要一种低成本方式来验证 FinTrust RAG 的财务回答是否可信，因为当前最有价值的证据分散在 trace、verification 和 benchmark 文件中，导致项目工程深度不够直观、失败原因不够可见。

### 2.2 已知证据

本 PRD 基于当前仓库已有能力和前序调研形成：

- 项目已有 `memo.md`、`trace.json`、`summary.json`、`verification.csv` 等 artifact。
- benchmark runner 已输出 trust metrics、retrieval candidates、claim results 和 failure tags。
- README 和部分文档仍包含多 Agent 主叙事。
- 高 star OSS 调研显示，最值得复现的是局部能力：RAGFlow 的检索可解释性、Langfuse 的 trace/span、DuckDB 的 run analytics、MinerU/PaddleOCR/Docling 的文档 fidelity 思路。

限制说明：GitHub star 数和 license 状态会变化，对外发布前必须重新联网确认；PRD 不固化实时 star 数。

## 3. 目标用户与 Proto-Persona

### 3.1 Primary Persona：评审型技术负责人

- **角色：** 面试官、作品集评审者、技术负责人或投资研究系统 reviewer。
- **目标：** 在有限时间内判断项目是否有真实工程难度和可信边界。
- **痛点：** 不想只看漂亮 demo；需要看到 benchmark、失败案例、trace 和证据绑定。
- **行为：** 会先扫 README，再看 UI 或 artifacts，最后才可能读源码。
- **代表性表达：** “不要告诉我用了几个 Agent，告诉我错的时候怎么发现、怎么拦住。”

### 3.2 Secondary Persona：金融 RAG 构建者

- **角色：** 构建检索、验证、benchmark 或文档解析系统的工程师。
- **目标：** 复用项目里的诊断方法，比较 retrieval backend 或 verification gate。
- **痛点：** 很多 RAG demo 只展示最终答案，不展示检索排序、证据缺口和 claim-level 失败。
- **行为：** 会查看 benchmark JSON、trace、retrieval candidates 和 evaluator 代码。

### 3.3 Secondary Persona：项目维护者

- **角色：** FinTrust RAG 的开发者或维护者。
- **目标：** 快速判断一次改动是改善了系统，还是只让单个样例更好看。
- **痛点：** benchmark failure 分散，document fidelity 和 publication gate 问题容易混在一起。
- **行为：** 会反复跑 benchmark、比较结果、检查 artifact、调整检索和验证规则。

## 4. Strategic Context

### 4.1 为什么现在做

现在最重要的问题不是继续增加新 backend 或更复杂的 Agent 编排，而是让已有可信能力可见、可解释、可复核。原因有三点：

1. **项目叙事需要升级。** “多 Agent”已经不是强差异化，可信控制层、benchmark 取证和审计 artifact 更能证明工程质量。
2. **benchmark 已经成为核心资产。** 如果 benchmark 只是结果文件，项目只能展示分数；如果 benchmark 变成 forensics dashboard，就能展示工程判断力。
3. **后续扩展需要基线。** GraphRAG、Docling、OCR、向量库或 workflow 工具是否值得接入，必须先有本地诊断面来衡量收益。

### 4.2 竞争与参考格局

高 star OSS 项目给出的启发是：成熟系统通常不只回答问题，还提供调试、观测、评估和数据管理能力。

- RAGFlow 强在文档处理与检索可解释性。
- Langfuse 强在 trace/span 和 eval dashboard。
- DuckDB 强在本地分析。
- MinerU、PaddleOCR、Docling 强在文档解析 fidelity。
- Dify、Open WebUI、AnythingLLM、Flowise、n8n 强在产品壳或 workflow，但它们不是 FinTrust RAG 的核心方向。

FinTrust RAG 的机会不是复刻完整产品，而是在“金融 RAG 的可信审计”这一窄场景里做得更深。

### 4.3 行业评测范式与金融特化

本项目的幻觉控制和审计规则应明确建立在已有 RAG evaluation / LLM observability 范式之上，再做金融场景硬化。

| 外部范式 | 对 FinTrust RAG 的启发 | FinTrust RAG 金融特化 |
|---|---|---|
| RAGAS / DeepEval 的 faithfulness、context precision、context recall、answer relevancy | 区分“答案是否忠于证据”和“检索上下文是否足够” | 将 `required_fact_recall`、`unsupported_claim_rate`、`retrieval_hit` 拆开诊断 |
| TruLens 的 RAG Triad：answer relevance、context relevance、groundedness | 把回答质量拆成答案相关性、上下文相关性、证据 groundedness | 金融答案即使相关，也必须通过数字、期间、来源门禁 |
| Phoenix / Langfuse 的 trace、eval、score、span 观测 | 用执行链定位错误发生在哪个步骤 | 将 planning、retrieval、assessment、generation、verification、publication gate 做成本地 span rows |
| OpenAI Evals / Graders / Trace Grading | 用结构化 graders 和 trace-level grading 做回归评估 | 将 LLM judge 仅用于语义判断，数字、期间、币种、单位使用 deterministic checks |
| Anthropic Citations 的 source-grounded response | 回答必须能回到 source documents、page/block/character 等证据位置 | 固化 `chunk_id/source_id/page/period/primary_source` 为不可绕过的信任边界 |

因此，FinTrust RAG 的规则不是“自创一套幻觉检测”，而是：

1. 用 RAGAS / DeepEval / TruLens / Phoenix 的范式定义 RAG 质量维度。
2. 用 OpenAI / Anthropic 的 eval、trace、citation 思路组织可审计 artifact。
3. 用金融场景的 deterministic hard gates 约束数字、期间、币种、单位、方向和来源类型。

## 5. 产品目标

### 5.1 Outcome Goals

1. **建立可信金融 RAG 的主叙事。**
   将项目从“多 Agent 财务研究 demo”重定位为“可审计的金融 RAG 控制层”，让评审者第一眼看到 benchmark、verification 和 trace，而不是 Agent 名词。

2. **降低可信度验证成本。**
   让评审者无需读源码，也能通过本地 artifact 判断一次回答的证据来源、检索路径、claim verification 和 publication gate。

3. **将 benchmark 从分数文件升级为诊断系统。**
   不只展示 pass rate，而是解释失败发生在 retrieval、document fidelity、required fact recall、unsupported claim，还是 final answer publication。

4. **复现高 star OSS 的局部工程能力。**
   借鉴 RAGFlow 的检索可解释性、Langfuse 的 trace/span、DuckDB 的 run analytics、MinerU/PaddleOCR/Docling 的文档保真思路，但只复现对 FinTrust RAG 核心可信度有帮助的局部能力。

5. **保持系统轻量、本地、可审计。**
   所有新增能力必须输出本地 artifact，不依赖云服务，不新增默认重依赖，不绕过 `chunk_id/source_id/page/period/primary_source` 信任边界。

6. **形成可解释的金融硬门禁体系。**
   将行业 RAG 评测范式转化为产品内可执行的 gates：语义 groundedness 可用 judge 辅助判断，财务数字、期间、单位、币种、方向和 primary-source 要用确定性规则拦截。

### 5.2 明确降级的目标

- GraphRAG 不是本阶段产品目标，而是实验后端。
- 高 star OSS 名称不是产品目标，而是能力来源说明。
- 单个漂亮 demo 不是产品目标，完整 cohort diagnostics 才是核心证明。
- LLM judge 不是最终仲裁者；金融硬事实必须被 deterministic gates 复核。

## 6. 解决方案概览

本阶段建设一个围绕本地 artifact 的“审计工作台”，覆盖五条用户路径：

1. **看项目是否可信。**
   README 第一屏和 Streamlit 首页展示 benchmark cohort、trust gates、失败分布和 artifact 能力。

2. **看检索是否可信。**
   Retrieval Lab 展示 query、lane、filters、candidate ranks、BM25/dense/rerank/fusion score、source type、period、primary-source flag、fallback reason。

3. **看执行链是否可信。**
   Local Trace Viewer 将现有 trace payload 转成 planning、retrieval、assessment、follow-up、generation、verification、publication gate 的 span rows。

4. **看 benchmark 与文档是否可信。**
   Benchmark Forensics Dashboard 解释 cohort metrics、failure tags、逐题 diff；Document Fidelity Harness 检查 source metadata、page mapping、table-like text、unit、period 和 chunk boundary。

5. **看规则依据是否可信。**
   Standards Mapping 将每条 hallucination-control 规则映射到 RAG evaluation、trace grading、citation 或 deterministic finance gate，让评审者知道哪些判断来自通用行业范式，哪些是 FinTrust RAG 的金融特化。

系统仍保持本地优先：所有新能力消费或生成本地 Markdown、JSON、CSV/JSONL artifacts，不新增默认云服务依赖。

### 6.1 三层信任门禁

本项目的幻觉控制应拆成三层，而不是只依赖 prompt 或单一 judge：

1. **Retrieval Gate：证据是否找到了。**
   对齐 context recall、context precision、retrieval relevance。关注是否检索到正确公司、正确期间、正确 source type、正确 fact slot。

2. **Grounding Gate：答案是否忠于证据。**
   对齐 faithfulness、groundedness、answer relevance。关注 claim 是否能被 retrieved chunks 支持，citation 是否有效。

3. **Finance Hard Gate：财务事实是否可发布。**
   FinTrust RAG 特化。对数字、单位、币种、期间、方向、实体、source type、primary-source 做确定性校验；不通过则删除、降级或拒答。

## 7. 成功指标

### 7.1 Primary Metric

**评审可信度验证时间。**

- 当前：评审者需要读 README、raw trace、benchmark JSON 和源码才能理解可信机制。
- 目标：评审者在 3 分钟内能理解系统可信边界，在 30 分钟内能复盘一个成功案例和一个失败案例。
- 衡量方式：README 首屏检查、demo walkthrough checklist、artifact 可见性检查。

### 7.2 Secondary Metrics

- **Benchmark forensics coverage：** benchmark report 至少覆盖 cohort summary、status distribution、failure tags、代表性成功/失败案例、关键 trust metrics。
- **Standards mapping coverage：** 每个核心 trust metric 都能映射到外部评测范式或 FinTrust finance hard gate。
- **Trace inspectability：** traced run 能生成 span rows，覆盖 planning、retrieval、assessment、generation、verification、publication gate。
- **Retrieval inspectability：** retrieval candidate rows 保留 `chunk_id`、`source_id`、`page`、`period`、`primary_source`，并展示可用 scores。
- **Document fidelity coverage：** processed source pack 能输出 metadata coverage 与 issue counts。
- **Run ledger usability：** benchmark rows 可导出为 DuckDB-compatible CSV/JSONL。

### 7.3 Guardrail Metrics

- 不新增默认 runtime 重依赖。
- 不降低现有 trust gates：`unsupported_claim_rate`、`unsupported_numeric_claim_rate`、`fabricated_citation_rate`、wrong entity、wrong period 不得因展示层改动恶化。
- 不绕过原始 evidence metadata。
- 不将 experimental GraphRAG 包装成主能力。
- 不用 LLM judge 覆盖 deterministic finance gate 的失败结果。

## 8. 用户故事与需求

### Story 1：README 与首页重新定位

作为评审者，我希望第一眼看到 FinTrust RAG 是可信金融 RAG 控制层，而不是多 Agent demo。

验收标准：

- README 第一屏出现 benchmark、verification、trace、forensics。
- 多 Agent 不再作为主标题、主 badge 或主卖点。
- OSS 借鉴被明确写成局部能力启发。
- GraphRAG 被标注为 experimental。

### Story 2：Benchmark Forensics

作为维护者，我希望加载一个或两个 benchmark JSON，看到 cohort diagnostics 和 regression diff。

验收标准：

- 支持 candidate-only report。
- 支持 baseline vs candidate comparison。
- 展示 `unsupported_claim_rate`、`unsupported_numeric_claim_rate`、`required_fact_recall`、`hard_fact_complete_but_not_published_rate`。
- 失败原因按 retrieval、document fidelity、claim support、publication gate 等类别归纳。
- 每个核心指标标注它对应的是 retrieval quality、grounding quality、finance hard gate 还是 publication gate。
- 输出 Markdown report 与 ledger export。

### Story 2.5：Standards Mapping

作为评审者，我希望看到每条幻觉控制规则背后的评测范式和金融特化理由，从而判断规则不是拍脑袋设计的。

验收标准：

- 展示 RAGAS/DeepEval/TruLens/Phoenix 对应的 RAG 质量维度。
- 展示 OpenAI eval/trace grading 与 Anthropic citation 思路如何映射到本地 artifacts。
- 区分 LLM judge 适用场景与 deterministic hard gate 适用场景。
- 每条 hard gate 都明确失败后的行为：删除、降级、拒答或标记 insufficient evidence。

### Story 3：Retrieval Lab

作为评审者，我希望看到每个问题检索到了什么、为什么排序靠前、证据是否来自主来源。

验收标准：

- 表格展示 query、subquestion、lane、backend、filters、rank、scores、source metadata、snippet。
- 支持按 company、source type、period、primary-source flag、lane 过滤。
- 字段缺失时优雅降级，不崩溃。
- 保留 raw identifiers，不做 UI 层重写。

### Story 4：Local Trace Viewer

作为评审者，我希望用 span 方式查看一次 run 的执行链。

验收标准：

- trace payload 能被转成 span rows。
- span 至少覆盖 planning、retrieval、assessment、follow-up、generation、verification、publication gate。
- 每个 span 展示 status、duration、metrics、failure reason、相关 evidence/source IDs。
- 保留 raw trace 下载。

### Story 5：Document Fidelity Harness

作为维护者，我希望在 benchmark 前检查 processed documents 的基础质量。

验收标准：

- CLI 可按 company 或 folder 运行。
- 输出 page、period、source type、primary_source、source_id、chunk_id coverage。
- 标记 table-like chunks、numeric units、period mentions、chunk boundary warnings。
- 支持 Markdown 与 JSON 输出。

### Story 6：DuckDB-Compatible Run Ledger

作为维护者，我希望把 benchmark run 当成可查询数据资产。

验收标准：

- 导出 CSV 或 JSONL。
- 每行包含 run metadata、item metadata、answer status、metrics、failure tags、backend metadata、artifact references。
- README/docs 给出 DuckDB 示例查询。
- 项目默认依赖不包含 DuckDB。

## 9. 功能需求

### FR1：Narrative Reset

- 更新 README 第一屏和 Streamlit 主文案。
- 将“多 Agent”从主叙事降级为实现细节或历史表述。
- 前置 benchmark cohort、failure diagnostics、artifact bundle。
- 新增 OSS inspiration section。

### FR2：Benchmark Forensics Module

- 读取现有 benchmark JSON。
- 容忍历史格式差异。
- 归一化 row-level fields。
- 计算 cohort summary、metric averages、failure tags、status distribution。
- 将 metrics 分类为 retrieval quality、grounding quality、finance hard gate、publication gate、presentation quality。
- 生成 Markdown report。
- 支持 baseline/candidate comparison。
- 导出 ledger-friendly CSV/JSONL。

### FR2.5：Standards Mapping Layer

- 建立 trust metric 到外部评测范式的映射表。
- 标记每个指标的判断方式：deterministic、LLM judge、hybrid、metadata check。
- 标记每个失败原因的处理动作：delete、downgrade、refuse、warn、needs_followup。
- 在 README、benchmark forensics report 和 UI 中展示简化版映射。
- 不把外部项目作为默认 runtime dependency；仅复现评测思想和本地 artifact 结构。

### FR3：Benchmark Forensics Dashboard

- Streamlit 支持上传或选择 benchmark artifacts。
- 展示 summary、safety gates、completeness metrics、failure distribution、representative cases。
- baseline 存在时展示逐题 diff。
- 提供 report 和 ledger 下载。

### FR4：Retrieval Lab

- 从 trace 和 benchmark artifacts 归一化 retrieval candidate rows。
- 展示 rank、score、source metadata、fallback reason、snippet。
- 支持过滤与导出。
- 保留 `chunk_id/source_id/page/period/primary_source`。

### FR5：Local Trace Viewer

- 将 trace JSON 转成 span rows。
- 展示 planning、retrieval、assessment、follow-up、generation、verification、publication gate。
- 展示 status、timing、metrics、failure reason。
- 支持 trace/span 级 error localization：错误发生在 planning、retrieval、grounding、finance gate 还是 publication gate。
- 保留 raw trace。

### FR6：Document Fidelity Harness

- CLI 检查 processed chunks 与 source metadata。
- 输出 metadata coverage、table-like text、numeric units、period mentions、boundary warnings。
- 支持 JSON/Markdown。
- 不默认接入 MinerU、PaddleOCR、Docling 或 Marker。

### FR7：Run Ledger Export

- 将 benchmark rows 导出为 stable schema。
- 兼容 DuckDB `read_csv_auto` / `read_json_auto`。
- 提供示例查询。
- 不新增 DuckDB 默认依赖。

### FR8：Experimental Backend Reporting

- GraphRAG 继续作为 optional experimental backend。
- comparison report 单独展示 safety delta 与 completeness delta。
- 后端实验不得丢失 citation/source metadata。

### FR9：Finance Hard Gate Registry

- 维护一份可审计的金融硬门禁 registry。
- 至少包含 entity、period、number、unit、currency、directionality、source type、primary-source、citation validity。
- 每个 gate 记录输入字段、判断方式、失败原因、失败动作和对应 artifact 字段。
- `unsupported_numeric_claim_rate`、wrong entity、wrong period、fabricated citation 必须优先由 deterministic 或 metadata checks 驱动。

## 10. 数据契约

### 10.1 Benchmark Row

- `run_id`
- `item_id`
- `company_id`
- `question`
- `question_type`
- `answer_status`
- `retrieval_backend`
- `retrieval_path`
- `metrics`
- `failure_tags`
- `claim_results`
- `retrieval_candidates`
- `artifact_paths`

### 10.2 Retrieval Candidate Row

- `item_id`
- `subquestion_id`
- `query`
- `lane`
- `retrieval_backend`
- `retrieval_path`
- `rank`
- `score`
- `bm25_score`
- `dense_score`
- `fusion_score`
- `rerank_score`
- `chunk_id`
- `source_id`
- `page`
- `company_id`
- `source_type`
- `period`
- `primary_source`
- `fallback_reason`
- `snippet`

### 10.3 Trace Span Row

- `trace_id`
- `span_id`
- `parent_span_id`
- `stage`
- `name`
- `status`
- `started_at`
- `ended_at`
- `duration_ms`
- `input_summary`
- `output_summary`
- `metrics`
- `failure_reason`
- `related_item_ids`
- `related_chunk_ids`
- `related_source_ids`

### 10.4 Document Fidelity Row

- `company_id`
- `source_id`
- `chunk_id`
- `page`
- `period`
- `source_type`
- `primary_source`
- `text_length`
- `has_numeric_content`
- `has_unit`
- `has_period_mention`
- `table_like`
- `boundary_warning`
- `metadata_issues`

### 10.5 Trust Metric Mapping Row

- `metric_name`
- `metric_family`
- `external_pattern`
- `fintrust_gate`
- `judge_type`
- `required_fields`
- `failure_reason`
- `failure_action`
- `artifact_fields`
- `display_priority`

### 10.6 Finance Hard Gate Row

- `gate_name`
- `claim_type`
- `required_evidence_fields`
- `check_type`
- `pass_condition`
- `failure_reason`
- `failure_action`
- `severity`
- `example_artifact`

## 11. 范围外

- hosted SaaS observability。
- 完整 Langfuse integration。
- 完整 Dify、Flowise、n8n、Open WebUI、AnythingLLM 产品壳。
- 用 LangChain、LlamaIndex、Haystack、Elastic、Vespa、FAISS 替换主检索链路。
- 默认接入 MinerU、PaddleOCR、Marker、Docling。
- 复制 OSS code、prompt、UI content 或 assets。
- 未刷新数据就公开宣称当前 star 数。
- 在 benchmark 证据不足前把 GraphRAG 作为核心卖点。

## 12. 依赖与约束

### 12.1 现有依赖

- 现有 benchmark runner 输出。
- 现有 artifact bundle：`memo.md`、`trace.json`、`summary.json`、`verification.csv`。
- 现有 retrieval trace、claim results、verification rows。
- 现有 Streamlit app。
- 现有 processed source pack layout。

### 12.2 约束

- 系统运行在固定本地 source packs 上，不做 live web search。
- 默认 runtime 不新增重依赖。
- parser 必须兼容历史 benchmark 结果。
- UI 层不能改写原始 evidence identifiers。
- license 不清晰或 copyleft 项目不得复制实现。
- 仓库已有未提交改动，实施时必须避免无关重写。

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| PRD 变成功能堆叠 | 执行失焦 | 用 success metrics 和 guardrails 约束每个功能 |
| OSS 名称变成包装噱头 | 叙事虚浮 | 每个 OSS 借鉴点必须对应本地 artifact |
| Streamlit UI 变臃肿 | 审计体验下降 | 使用 tabs、filters 和 reusable modules |
| benchmark parser 不兼容旧格式 | 历史 run 不可分析 | tolerant normalization 与 fixture tests |
| completeness gain 掩盖 safety regression | 可信承诺削弱 | 报告中强制分离 safety gates 与 completeness metrics |
| GraphRAG 抢主线 | 产品焦点发散 | 保持 experimental 标签，只有指标证明后再升级 |

## 14. 发布计划

### Phase -1：标准映射与门禁注册表

- 建立 trust metric mapping。
- 建立 finance hard gate registry。
- 明确哪些规则使用 deterministic checks，哪些允许 LLM judge，哪些需要 hybrid。
- 将外部范式映射到本地 artifact：benchmark report、verification rows、trace spans、ledger。

退出标准：每个核心 hallucination-control 规则都有来源范式、判断方式、失败动作和 artifact 字段。

### Phase 0：叙事重置

- 更新 README 第一屏。
- 更新 Streamlit 主文案。
- 新增 OSS inspiration section。
- 新增 standards-backed trust gates section。
- 将 benchmark diagnostics 前置到单案例 demo 之前。

退出标准：评审者第一屏能理解“可信金融 RAG 控制层”。

### Phase 1：Benchmark Forensics

- 实现 forensics module。
- 实现 CLI report generation。
- 实现 ledger export。
- 增加 tests。

退出标准：一个 benchmark JSON 可生成 Markdown report；两个 JSON 可生成 regression comparison。

### Phase 2：Retrieval Lab

- 归一化 retrieval candidate rows。
- 增加 Streamlit table、filters、CSV export。

退出标准：无需 raw JSON 即可检查检索排序、分数、来源和 fallback reason。

### Phase 3：Local Trace Viewer

- trace JSON 转 span rows。
- Streamlit 展示 span table/timeline。
- 保留 raw trace download。

退出标准：无需 raw JSON 即可理解执行链。

### Phase 4：Document Fidelity Harness

- 实现 CLI harness。
- 输出 Markdown/JSON。
- 增加 metadata 与 boundary warning tests。

退出标准：benchmark 前可发现 source metadata 与 chunking 问题。

### Phase 5：Experimental Backend Reporting

- 保持 GraphRAG experimental。
- comparison report 展示 backend-specific safety/completeness deltas。

退出标准：GraphRAG 以可度量实验后端出现，而不是主叙事卖点。

## 15. 开放问题

- Benchmark Forensics 应做成独立 Streamlit 页面，还是当前 workbench 的 tab？
- 是否将 selected release 的 forensics report 固化到 `docs/showcases`？
- 第一批 golden documents 应选择哪些公司和文档类型？
- run ledger schema 先支持 CSV、JSONL，还是两者都支持？
- GraphRAG 需要达到什么 metric improvement 才能从 experimental 升级为 supported？

## 16. 实施就绪检查清单

- 问题不是“缺少更多 Agent”，而是“可信证据不可见”。
- 目标用户、JTBD 和评审路径明确。
- 成功指标包含 primary、secondary 和 guardrail。
- 每个 OSS 借鉴点都有本地可交付 artifact。
- 功能范围不新增默认重依赖。
- 信任边界明确：`chunk_id/source_id/page/period/primary_source` 不可绕过。
- 发布顺序先诊断、再 UI、再实验后端。
