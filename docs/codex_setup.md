# Codex + gstack Setup Notes

这份文件只做两件事：

1. 记录这个仓库已经落地的项目级 Codex 配置。
2. 给出可直接合并到 `~/.codex` 的全局片段，避免覆盖你现有的个人设置。

## 已落地的项目级文件

- `AGENTS.md`
- `.codex/config.toml`

这些文件的目标很直接：

- 把仓库真实命令写清楚，而不是继续假设有 `Makefile`
- 把 gstack 在 Codex 下的推荐使用方式写清楚
- 让 Codex 默认处于 `workspace-write + on-request`，同时允许网络能力按需使用

## 建议合并到 `~/.codex/config.toml` 的片段

不要整文件覆盖。把下面内容合并到你已有配置里即可。

```toml
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
```

你当前全局配置里已经有：

- `model = "gpt-5.4"`
- `model_reasoning_effort = "high"`
- `features.multi_agent = true`
- Notion MCP server

这些可以保留，不需要因为 gstack 再改一遍。

## 建议追加到 `~/.codex/AGENTS.md` 的片段

同样不要整文件覆盖，直接在现有规则后面追加即可。

```md
## gstack

- If gstack is installed, prefer explicit skill invocation over assuming Claude-style slash command behavior.
- Default workflow: office-hours -> plan-eng-review -> implement -> review -> qa -> ship -> document-release.
- Inside Codex, do not call the gstack codex second-opinion skill unless the user explicitly asks for cross-model review.
- Ask for approval before destructive git commands, schema changes, or production-affecting operations.
```

## 为什么这样配

- OpenAI 官方文档确认 `workspace-write` 在版本控制目录里是推荐起点，`network_access` 需要在配置里显式打开。
- OpenAI 官方文档确认 Codex 会先读取全局 `AGENTS.md`，再叠加项目 `AGENTS.md`。
- OpenAI 官方文档确认技能会从仓库 `.agents/skills` 和用户目录自动发现。
- gstack README 明确支持 `./setup --host codex`，但它原生工作流是围绕 skill 组织的，不是围绕仓库里的固定命令组织的，所以项目层必须补一层现实约束。

## 参考

- OpenAI Codex CLI: <https://developers.openai.com/codex/cli>
- OpenAI AGENTS.md guide: <https://developers.openai.com/codex/guides/agents-md>
- OpenAI approvals and security: <https://developers.openai.com/codex/agent-approvals-security>
- OpenAI skills guide: <https://developers.openai.com/codex/skills>
- gstack repo: <https://github.com/garrytan/gstack/tree/main>
