# project-keeper

> CLI 命令：`keeper`

给一个本地项目配一个「长期在岗」的 AI 管家：跨会话记住项目的状态、决策和踩过的坑；
任务开始前把记忆装进上下文，任务结束后自动整理记忆：人工写入需审批，自动整理可回滚地合并 / 归档。

当前进度：**M0–M5.1b 已完成（四项治理增强）；后台整理不做，下一步 M6 项目上下文层与任务路由**。

## 为什么做

通用 coding agent 每次会话都从零开始，但项目状态、决策和教训属于代码库，不属于某次对话。
keeper 把这些内容写成可读、可 diff、可手改的 Markdown：人工写入必须经过人审，自动整理靠备份 + id 守恒可回滚。

## 能力

- **项目记忆**：`.agent/memory/` 下的 Markdown；跨会话注入；替换不丢历史。
- **记忆治理**：条目 id、soft / hard 预算（随模型窗口缩放）、注入折叠；会话结束自动整理，备份 + id 守恒可回滚；`pinned` 代码级保护、agent 主动 checkpoint、自适应冷却。
- **工具循环**：读文件、列目录、搜索、写文件、受限 shell。
- **人审边界**：写入 / 执行展示 diff 或命令，终端确认；拒绝后不执行。自动整理不逐条确认，走备份 + 守恒。
- **会话续跑**：JSONL 只追加，`--resume` 继续上次会话。
- **上下文压缩**：接近窗口上限时摘要旧前缀；JSONL 追加 `compaction`，`--resume` 可重建。
- **持久任务队列**：`.agent/tasks.json`；跨进程中断恢复；完成 / 阻塞必须显式标记。
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
| `KEEPER_CONTEXT_WINDOW` | `65536` | 模型上下文窗口（tokens） |
| `KEEPER_RESERVE_TOKENS` | `16384` | 为响应预留的 tokens；超过窗口减预留值就压缩 |
| `KEEPER_KEEP_TOKENS` | `20000` | 压缩时尽量保留的最近 tokens |
| `KEEPER_MEMORY_AUTO` | 开 | 会话 / task 结束后自动整理记忆；`0` / `false` / `off` / `no` 关闭 |

## 工具与审批

| 工具 | 权限 | 说明 |
|---|---|---|
| `read_file` | 自动放行 | 带行号读取，支持分段 |
| `list_dir` | 自动放行 | 非递归列目录，跳过 `.git` / `.agent` / `__pycache__` |
| `search` | 自动放行 | 纯文本子串搜索 |
| `write_file` | 需确认 | 新建或覆盖，覆盖时展示 diff |
| `update_memory` | 需确认 | 更新 `.agent/memory/` 的决策 / 教训 / 状态；展示 diff，INDEX 自动生成 |
| `request_memory_checkpoint` | 自动放行 | 只写 `.agent/memory/.tidy-request.json`，请求会话结束后整理；不立即整理，也不绕过 due 闸门 |
| `shell` | 需确认 | argv 数组、命令白名单、固定 cwd、超时和输出截断 |
| `complete_task` / `block_task` | 自动放行（仅任务模式） | 显式完成 / 阻塞当前任务；block 原因必填 |

审批规则：只读放行；写入 / 执行询问 `[y/N]`；删除 / 高危直接拒绝；
无审批器、异常、EOF、超时都拒绝（fail-closed）。

所有路径限制在项目根目录内；`write_file` 拒绝写入 `.git/` 和 `.agent/`；
`update_memory` 是 `.agent/memory/` 的写入入口，拒绝后文件字节不变。
`update_memory` 与自动整理共用同一把单飞锁；整理进行中的人工记忆写入会返回明确错误，不会静默覆盖。

## 项目记忆

`.agent/memory/` 下：

| 文件 | 内容 |
|---|---|
| `PROJECT_STATE.md` | 现在为真：焦点、进行中、阻塞、下一步 |
| `DECISIONS.md` | 选了什么 + 为什么 |
| `LESSONS.md` | 可复用规则、踩过的坑 |
| `INDEX.md` | 导航，自动生成，不手改 |
| `*_ARCHIVE.md` | 被取代的条目，保留不删除 |

`*_ARCHIVE.md` 只允许追加；整理不会删除归档条目，备份里已有的归档 id 被删会从备份恢复。
整理规则：只归档「不再为真 / 不再有效」的条目，不按年龄、不因超预算归档。

一条 entry 是一个 `- ` bullet；细节缩进；`## ` 是主题分组。
每条 entry 的 bullet 下附一行 `<!--mem id:… created:… touched:…-->`；id 由内容派生，已有条目不重写 id，内容变化只更新 `touched`。注释在注入前剥离，不占上下文；id 用于整理前后的守恒校验，重复 id 的第二条会换新 id。
替换旧条目时，旧条目原样移入 archive，新条目写 `supersedes: <旧 bullet 原文>`。
启动注入顺序：`DECISIONS → LESSONS → INDEX → PROJECT_STATE`。

预算按「UTF-8 字节数 / 3」计，计入 bullet 与缩进细节，不含 mem 注释和 archive；下表是 `KEEPER_CONTEXT_WINDOW=65536` 的基准值，实际按窗口等比缩放 0.5x–2x（窗口未知时用原表）：

| 文件 | soft | hard |
|---|---|---|
| `DECISIONS.md` | 3000 | 4500 |
| `LESSONS.md` | 2500 | 4000 |
| `PROJECT_STATE.md` | 1200 | 1800 |
| `INDEX.md` | 600 | 900 |

超过 hard 时，注入视图先丢掉缩进细节；仍超限则按 `## ` 分组折叠（每组保留 `touched` 最新一条并标出省略数），最后头部截断。文件本身不会因为超预算被改写。预算数值尚未用产品级记忆校准，可按 `tidy-run.json` 的前后计数调整。

自动整理在一次性 chat、REPL 的 `/exit` / `/quit` / EOF、task attempt 正常结束后触发；Ctrl+C 不触发。启动时只做廉价的 due 检查并提示，只读 state、文件指纹和记忆文本，不读会话、不调用模型。闸门是「超 soft + 内容有变化 + 冷却（基础 30 分钟，连续 no-change 会延长）+ 单飞锁」，不满足就不调用模型。默认开启、同步执行，退出可能因此多等一会儿；`KEEPER_MEMORY_AUTO=0` 或 `--no-tidy` 可关闭。整理只在 keeper 自身命令边界触发，没有定时器或后台进程；后台整理不做，只有同步整理。

安全网：整理前把 live + archive + INDEX 备份到 `backups/tidy-<ts>/`（保留 5 份），备份失败不编辑；规则要求只 live→archive、不删除，代码对 `DECISIONS` / `LESSONS` 做 id 严格守恒（`PROJECT_STATE` 不套），丢失条目从备份恢复；INDEX 重建，结果写入 `tidy-run.json`。整理员还可读取最近 3 个会话作为证据（只在备份成功后读取，最多约 3000 字符），但证据也是提示词级约束。
整理 pass 只使用 `memory_read` / `memory_write`，只改 `.agent/memory/` 下的 Markdown，不改任务队列，也不改会话文件；写入由只对 `.agent/memory/` 生效的独立审批器放行（不弹 `[y/N]`，不能写 INDEX）；单轮最多 8 次工具调用、600 秒，超时或模型失败仍写 run record。
报告包含 tokens / 条目数前后、归档 / 合并 / 恢复计数和 evidence / cited 会话数；`evidence_sessions` / `cited_sessions` 只存会话 id，不存会话原文。自动整理失败只打 warning，不改变原命令退出码。

会话证据只读 `.agent/sessions/` 下平铺的 `<id>.jsonl`，刚结束的会话优先，其余按最近活动倒序；坏行、半行和非法字段跳过，超过 1 MiB 只读尾部。读取不写盘、不建目录、不碰锁。

`pinned:1` 写在条目的 mem 注释里；整理后若被模型改写、移动、删除、转档或跨文件复制，keeper 会按备份把原块（含注释、位置、CRLF）回滚；未编辑的 pinned 文件不会被重写。人工 `update_memory` 经 diff 审批仍可 supersede pinned 条目。

agent 可用 `request_memory_checkpoint` 写一个请求标记；下一次 CLI 边界整理时 reason 进入 prompt 与 `tidy-run.json`，但仍要过 due 闸门。pass 真正执行后才尝试清除标记，跳过 / 失败 / 期间的新请求会保留。

冷却自适应：连续「跑了但没有改动」的整理把 30 分钟冷却按 30→60→120→240 分钟延长，上限 4 小时；有实际编辑或恢复就回到 30 分钟。

`keeper memory tidy` 是强制 / 调试入口（跳过 due 闸门，仍备份 / 守恒）；`keeper memory tidy --check` 只读查看 due、预算、锁和上次报告，不需要 API key。「合并只缩不加」仍是提示词规则；整理只改磁盘，当前会话的 system prompt 是启动快照，结果下次启动才生效。

## 上下文压缩（M4）

每个 step 调模型前检查最近一次有效 usage（优先 `total_tokens`，否则 `prompt + completion`）加上其后消息的估算；没有 usage 时用全量估算（含工具 schema）。超过 `KEEPER_CONTEXT_WINDOW - KEEPER_RESERVE_TOKENS` 时，摘要切点之前的前缀，保留最近约 `KEEPER_KEEP_TOKENS` tokens。

- 切点不拆开 assistant 的 `tool_calls` 和 `tool` 结果；切在轮次中间时，历史摘要和轮次前缀摘要分开生成再合并。保留区仍超过窗口预算时整段摘要；前缀只剩旧摘要时跳过，避免重复压缩。
- 摘要输入把单条工具结果和工具调用参数头截断到 2000 字符；整体仍超限时保留首条 + 最近消息。
- 摘要固定六段：Goal / Constraints & Preferences / Progress / Key Decisions / Next Steps / Critical Context。
- 校验六段非空；拒绝 tool_calls / JSON / 空文本，内容失败重试一次。路径和 `ERROR:` 原文缺失时确定性补齐；仍不可用时用确定性摘要兜底，记录 `degraded`。
- 会话 JSONL 只追加 `compaction` 记录（含 `tokens_before` / `tokens_after`，降级时含 `degraded`），旧消息不删；`--resume` 按顺序回放。
- 摘要不可用不中断任务，压缩照常完成；`read_file` 单次结果上限 50KB。

## 常用命令

```bash
keeper "看看项目里有哪些文件"       # 单任务
keeper                               # 交互模式
keeper --resume                      # 有 queued 任务先跑任务；否则续跑最近会话
keeper --session .agent/sessions/xxx.jsonl
keeper --root ../other-project "..." # 指定项目根目录

keeper task add "修复 login 测试"     # 入队，stdout 输出 task id
keeper task list                     # 查看队列；会先做一次启动恢复
keeper task run                      # 跑队首 queued 任务（一次一个 attempt）
keeper task run t-20261122-153012-a1b2c3

keeper memory tidy                   # 强制整理一次（仍备份 / 守恒）
keeper memory tidy --check           # 只读查看 due / 预算 / 锁 / 上次报告
keeper memory tidy --root ../other-project  # 对另一个项目强制整理
```

`keeper --no-tidy` / `keeper task run --no-tidy` 只关闭本次结束后的自动整理。

## 数据落在哪

目标项目的 `.agent/`：会话 JSONL 只追加；记忆在 `.agent/memory/`；
任务队列在 `.agent/tasks.json`（原子写）；M4 的 `compaction` 记录也追加在会话 JSONL 里，旧消息不删。

整理另在 `.agent/memory/` 下产生 `backups/tidy-<ts>/`（保留 5 份）、`tidy-run.json`（最近一次结果）、
`.tidy-state.json`（文件指纹与冷却水位）、`.tidy.lock`（单飞锁）和可选的 `.tidy-request.json`（agent checkpoint 请求）。记忆文件用 tmp + `os.replace` 原子写。
`.tidy-state.json` 还记录 `last_attempt_at` / `last_success_at`；`.tidy.lock` 含 pid + host，死进程或锁龄超过 2 小时可回收。

## 路线

- **M0 骨架**：已完成。
- **M1 工具与审批**：已完成。
- **M2 记忆**：已完成。
- **M3 任务队列**：已完成；持久任务、中断恢复、显式完成 / 阻塞。
- **M4 最小压缩**：已完成；阈值 + 安全切点 + 六段摘要 + JSONL `compaction`。
- **M5 记忆治理**：M5.0 自动整理（id、预算、折叠、备份、守恒）、M5.1a 会话证据与 M5.1b 四项增强（pinned、checkpoint、自适应冷却、预算缩放）已完成；后台整理不做，保持同步整理。
- **M6 项目上下文层与任务路由**：项目注册表、worker 适配、收件箱、映射优先 + 模型兜底、飞书来源（前台 `keeper watch`，不是服务端）。
- **M7 可观测 / eval**：本地 trace + 固定样本 + 量化指标。

不做：Web UI、后台常驻服务 / installer、内部子 agent 编排、外部事件触发器、Skills、OS 沙箱、RAG / 向量库、分支摘要 / 插件钩子 / TUI 流式、多 provider 适配层。

## 开发

```bash
pytest -q
```

要求 Python 3.11+；项目按 Windows 11 + CLI 开发。