# 离线样本公司与测试系统

当前离线系统不再假设“随便换一家公司都能马上跑”。  
它有一套明确的样本公司和离线语料边界。

## 样本公司

- `stripe`
  - 角色：demo / 单公司轻量烟雾测试
  - 语料形态：legacy `company_packs`

- `cloudflare`
  - 角色：benchmark 主样本
  - 覆盖：商业模式、AI 增长叙事、管理层表述变化
  - 语料形态：`raw` + `normalized` + `processed` + `benchmark`

- `fastly`
  - 角色：benchmark 主样本
  - 覆盖：收入结构、盈利能力、风险与冲突处理
  - 语料形态：`raw` + `normalized` + `processed` + `benchmark`

## 离线准备入口

```bash
make prepare-offline-suite
```

它会：

- 校验样本公司规范
- 校验 `raw / normalized / processed` 语料是否齐全
- 确保 benchmark `local_facts.jsonl` 已准备好
- 生成 `data/offline_suite/report.json`

## 离线测试入口

```bash
make test-offline-suite
```

它会：

- 先校验离线样本套件
- 用 `LLM_MODE=stub` 跑一轮 `trust_showcase_v2` benchmark 冒烟
- 跑离线套件相关 pytest

注意：

- 这里的 stub benchmark 是测试“离线路径是否完整”，不是测试“研究质量是否达标”。
- 分数低通常代表当前 agent 能力或策略还弱，不代表离线语料或测试系统坏了。
- 真正的质量门槛要看 live benchmark、人工抽检和失败标签，而不是 stub 冒烟分数。

## 设计原则

- 样本公司必须少而硬，不追求面广。
- benchmark 题集必须围绕这些样本公司构建。
- 测试系统优先验证：
  - 离线语料是否完整
  - benchmark 约束是否可复现
  - 研究控制器在 stub 模式下是否能稳定回放
