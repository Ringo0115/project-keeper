# project-keeper

> CLI 命令：`keeper`

给一个本地项目配一个「长期在岗」的 AI 管家：跨会话记住项目的状态、决策和踩过的坑；
任务开始前把记忆装进上下文，任务结束后在人工审批下更新记忆。

当前进度：**M0 / M1 / M2 已完成；下一步 M3（任务队列）**。

## 为什么做

通用 coding agent 每次会话都从零开始，但项目状态、决策和教训属于代码库，不属于某次对话。
keeper 把这些内容写成可读、可 diff、可手改的 Markdown，并且所有写入都必须经过人审。

## 能力

- **项目记忆**：`.agent/memory/` 下的 Markdown；跨会话注入；替换不丢历史。
- **工具循环**：读文件、列目录、搜索、写文件、受限 shell。
- **人审边界**：写入 / 执行展示 diff 或命令，终端确认；拒绝后不执行。
- **会话续跑**：JSONL 只追加，`--resume` 继续上次会话。
- **CLI + OpenAI 兼容模型**：默认 DeepSeek，可通过配置切换。

## 快速开始

```bash
cd keeper
python -m pip install -e .      # 可选，装完可以直接用 keeper 命令
cp .env.example .env            # 填入 API key
python -m keeper "看看项目里有哪些文件"
```

不安装也能跑：`python -m keeper ...`。

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `KEEPER_API_KEY` / `DEEPSEEK_API_KEY` | 无 | 必填 |
| `KEEPER_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容地址，需带版本号 |
| `KEEPER_MODEL` | `deepseek-chat` | 模型名 |
| `KEEPER_MAX_STEPS` | `30` | 单任务最大步数 |
| `KEEPER_APPROVAL_TIMEOUT` | `60` | 等待 y/N 的秒数，`0` 表示不超时；超时按拒绝处理 |

## 工具与审批

| 工具 | 权限 | 说明 |
|---|---|---|
| `read_file` | 自动放行 | 带行号读取，支持分段 |
| `list_dir` | 自动放行 | 非递归列目录，跳过 `.git` / `.agent` / `__pycache__` |
| `search` | 自动放行 | 纯文本子串搜索 |
| `write_file` | 需确认 | 新建或覆盖，覆盖时展示 diff |
| `update_memory` | 需确认 | 更新 `.agent/memory/` 的决策 / 教训 / 状态；展示 diff，INDEX 自动生成 |
| `shell` | 需确认 | argv 数组、命令白名单、固定 cwd、超时和输出截断 |

审批规则：只读放行；写入 / 执行询问 `[y/N]`；删除 / 高危直接拒绝；
无审批器、异常、EOF、超时都拒绝（fail-closed）。

所有路径限制在项目根目录内；`write_file` 拒绝写入 `.git/` 和 `.agent/`；
`update_memory` 是 `.agent/memory/` 的写入入口，拒绝后文件字节不变。

## 项目记忆

`.agent/memory/` 下：

| 文件 | 内容 |
|---|---|
| `PROJECT_STATE.md` | 现在为真：焦点、进行中、阻塞、下一步 |
| `DECISIONS.md` | 选了什么 + 为什么 |
| `LESSONS.md` | 可复用规则、踩过的坑 |
| `INDEX.md` | 导航，自动生成，不手改 |
| `*_ARCHIVE.md` | 被取代的条目，保留不删除 |

一条 entry 是一个 `- ` bullet；细节缩进；`## ` 是主题分组。
替换旧条目时，旧条目原样移入 archive，新条目写 `supersedes: <旧 bullet 原文>`。
启动注入顺序：`DECISIONS → LESSONS → INDEX → PROJECT_STATE`。

## 常用命令

```bash
keeper "看看项目里有哪些文件"       # 单任务
keeper                               # 交互模式
keeper --resume                      # 续跑最近一次会话
keeper --session .agent/sessions/xxx.jsonl
keeper --root ../other-project "..." # 指定项目根目录
```

## 数据落在哪

目标项目的 `.agent/`：会话 JSONL 只追加；记忆在 `.agent/memory/`；
任务队列（M3）将写入 `.agent/tasks.json`。

## 路线

- **M0 骨架**：已完成。
- **M1 工具与审批**：已完成。
- **M2 记忆**：已完成。
- **M3 任务队列**：下一步；持久任务、中断恢复、显式完成 / 阻塞。
- **M4 最小压缩**：token 阈值 + 六段摘要。
- **M5 可观测 / eval**：本地 trace + 固定样本。

不做：Web UI、多 agent、触发器、Skills、OS 沙箱、RAG / 向量库。

## 开发

```bash
pytest -q
```

要求 Python 3.11+；项目按 Windows 11 + CLI 开发。