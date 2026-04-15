# Raw Benchmark Source Packs

`data/raw/` 存的是 benchmark 用到的原始资料包，还没有经过解析、切块或索引。

这层数据的目标只有两个：

- 固定 benchmark 的原始输入
- 给后续 `normalized/` 和 `processed/` 提供可追溯的源文档

如果你想看整体数据流，先读 [data/benchmark/README.md](/Users/jiang/Documents/cv%20project/bizintel-agent/data/benchmark/README.md)。

## 目录结构

每个公司一个目录：

```text
data/raw/
  <company>/
    manifest.json
    periods.json
    download_receipts.jsonl
    docs/
      <raw html/pdf/txt files>
```

各文件职责：

- `manifest.json`：资料包清单。定义每份文档的 `doc_id`、本地路径、标题、来源类型、时期、URL、是否 primary source。
- `periods.json`：这个公司可用的时期边界，例如 `2024FY`、`2025Q4`、`latest_period`、`comparison_period`。
- `download_receipts.jsonl`：每次下载的收据，记录来源 URL、HTTP 状态、文件大小、SHA256、下载时间。
- `docs/`：原始文档本体，通常是 HTML、PDF，偶尔也可能是纯文本。

## 导入前提

导入 raw 资料之前，目标公司目录里至少要有：

- `manifest.json`
- `periods.json`

抓取脚本不会替你生成这两个文件；它只会按照 `manifest.json` 把文档下载到 `docs/`，并把结果追加到 `download_receipts.jsonl`。

`manifest.json` 里每个 `documents` 条目至少应包含：

```json
{
  "doc_id": "cloudflare_2024_10k",
  "path": "data/raw/cloudflare/docs/cloudflare_2024_10k.html",
  "title": "Cloudflare Form 10-K for fiscal year 2024",
  "source_type": "annual_report",
  "period": "2024FY",
  "url": "https://...",
  "is_primary": true
}
```

要求：

- `path` 必须落在 `data/raw/<company>/docs/` 下
- `doc_id` 在同一个公司目录内必须唯一
- `period` 要和 `periods.json` 中的时期体系一致
- 财务强事实优先使用 filing、results release、transcript 等一级材料

## 导入命令

### 1. 先看计划，不真正下载

```bash
make fetch-benchmark-sources COMPANY=cloudflare ARGS="--dry-run"
```

这一步会打印每个 `doc_id` 准备下载到哪里，以及对应的源 URL。

### 2. 下载整个公司的 raw 资料包

```bash
make fetch-benchmark-sources COMPANY=cloudflare
```

脚本会：

- 读取 `data/raw/cloudflare/manifest.json`
- 按清单把文件下载到 `data/raw/cloudflare/docs/`
- 追加写入 `data/raw/cloudflare/download_receipts.jsonl`

### 3. 只补某一份文档

```bash
make fetch-benchmark-sources COMPANY=cloudflare DOC_ID=cloudflare_2024_10k
```

底层等价于：

```bash
.venv/bin/python -m tools.fetch_benchmark_sources --company cloudflare --doc-id cloudflare_2024_10k
```

### 4. 强制重下已有文件

```bash
.venv/bin/python -m tools.fetch_benchmark_sources --company cloudflare --force
```

`--force` 会覆盖本地已存在文件，并生成新的 receipt 记录。

## 导入后的检查

至少检查这几项：

1. `docs/` 下是否真的出现了目标文件。
2. `download_receipts.jsonl` 是否新增了对应 `doc_id` 的下载记录。
3. `http_code` 是否为 `200`。
4. `bytes_local` 是否大于 `0`。
5. `sha256` 是否已记录。

例如：

```bash
ls data/raw/cloudflare/docs
tail -n 5 data/raw/cloudflare/download_receipts.jsonl
```

注意：

- 脚本对 primary source 下载失败会返回非零退出码。
- 已存在文件默认会 `skip`，不会重复下载。
- 某些 `raw.githubusercontent.com` 链接会自动走 GitHub Contents API fallback。

## 从 Raw 到可用语料

raw 导入完成后，下一步通常不是直接跑 benchmark，而是先规范化：

```bash
make normalize-benchmark-corpus COMPANY=cloudflare
```

这一步会把 `data/raw/<company>/docs/` 解析成：

- `data/normalized/<company>/documents.jsonl`
- `data/normalized/<company>/chunks.jsonl`
- `data/processed/<company>/sources.json`
- `data/processed/<company>/chunks.json`

如果你在维护 benchmark 版本，还会继续走这些步骤：

```bash
make freeze-benchmark-snapshot VERSION=v2
make resolve-benchmark-evidence VERSION=v2
make prepare-benchmark-local-facts VERSION=v2
```

## 新增公司的最小流程

如果是给 benchmark 新增一个公司 source pack，最小流程是：

1. 新建 `data/raw/<company>/manifest.json`
2. 新建 `data/raw/<company>/periods.json`
3. 运行 `make fetch-benchmark-sources COMPANY=<company> ARGS="--dry-run"` 检查路径和 URL
4. 运行 `make fetch-benchmark-sources COMPANY=<company>` 真正导入 raw 文档
5. 运行 `make normalize-benchmark-corpus COMPANY=<company>`
6. 再进入 benchmark snapshot / evidence / local facts 的后续冻结流程

## 边界

- `data/raw/` 只存原始文档和下载元数据，不放 benchmark 标签、答案或 evidence 绑定结果。
- 不要在 raw 文本里人工混入题目答案、局部事实卡或评测标签。
- `target_periods` 和 `periods.json` 是硬边界，不能把跨期材料混成一个时期。
