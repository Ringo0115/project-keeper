# project-keeper

> CLI 命令：`keeper`

给本地项目配一个「长期在岗」的 AI 管家。

通用 coding agent 每次会话都从零开始，但项目状态、决策和教训属于代码库，不属于某次对话。keeper 把这些内容写成可读、可 diff、可手改的 Markdown，放进项目自己的 `.agent/` 里：任务开始前装进上下文，任务结束后自动整理。人工写入必须经过人审，自动整理靠备份和条目 id 守恒可以回滚。

项目是主对象，会话不是。记忆存在项目里，所以一个 agent 做不下去时，换另一个 agent 读同一份记忆就能接着做。管理者是前台 CLI 会话，可以把任务派给可替换的 worker（内置 loop、Claude Code、Codex、外部 pi），也可以并行派发 2–5 个互不依赖的子任务。

## 能力

- **项目记忆**：`.agent/memory/` 下的 Markdown，跨会话注入，替换旧条目时保留历史。
- **记忆治理**：条目 id、预算与注入折叠、会话结束自动整理；备份 + id 守恒可回滚；`pinned` 保护的条目整理后会被还原；agent 可以主动请求整理；连续没东西可整理时自动拉长冷却。
- **多项目**：一个全局注册表记住每个项目的根目录、可用 worker 和部署指令；也可以继续按目录跑。
- **管理者与派发**：`keeper manager` 起一个前台会话，自己处理读文件和短改动，编码和长任务派给 worker；单发或并行 fanout，全部结束后一次性汇总。
- **worker 可替换**：内置 loop、Claude Code、Codex、外部 pi。交接靠共享记忆 + 对话 recap + 上一轮完成记录，换人不用重讲背景。
- **持久任务队列**：`.agent/tasks.json`；跨进程中断恢复；完成 / 阻塞必须显式标记；可指派 worker、可换 worker 重试。
- **工具与审批**：读文件、列目录、搜索、写文件、受限 shell；写入和执行先展示再确认。
- **会话续跑与压缩**：JSONL 只追加，`--resume` 继续上次会话；接近窗口上限时摘要旧前缀。

## 快速开始

```bash
cd keeper
python -m pip install -e .      # 可选，装完可以直接用 keeper 命令
cp .env.example .env            # 填入 API key
python -m keeper "看看项目里有哪些文件"
```

不安装也能跑：`python -m keeper ...`。要求 Python 3.11+。

第一次在某个项目里用，先在仓库根目录跑一条命令即可，keeper 会在该目录建 `.agent/` 并写入初始记忆：

```bash
cd ../your-project
keeper "读一下这个项目的结构，把项目现状写进记忆"
```

## 常用命令

```bash
keeper "看看项目里有哪些文件"       # 单任务
keeper                               # 交互模式
keeper --resume                      # 有 queued 任务先跑任务；否则续跑最近会话
keeper --session .agent/sessions/xxx.jsonl
keeper --root ../other-project "..." # 指定项目根目录

keeper task add "修复 login 测试"     # 入队，stdout 输出 task id
keeper task add "重构支付模块" --worker claude-code   # 指派 worker
keeper task list                     # 查看队列
keeper task run                      # 跑队首 queued 任务
keeper task retry t-20261122-153012-a1b2c3 --worker codex   # 阻塞 / 失败后换 worker 重跑

keeper manager "把 login 测试修好"    # 管理者会话：自己处理小事，长任务派给 worker
keeper project register ../api --id api
keeper project list
keeper project use api               # 之后不加 --root 就跑这个项目

keeper memory tidy                   # 立刻整理一次（仍备份 / 守恒）
keeper memory tidy --check           # 只看状态：是否该整理、预算、锁、上次结果
```

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
| `KEEPER_MEMORY_AUTO` | 开 | 会话 / 任务结束后自动整理记忆；`0` / `false` / `off` / `no` 关闭 |
| `KEEPER_HOME` | `~/.keeper` | 全局项目注册表所在目录 |
| `KEEPER_WORKER_<SLUG>_ARGV` | 见下 | 覆盖外部 worker 的启动命令，JSON 数组，`{input}` 为任务文本占位 |

## 多项目与管理 agent

注册表放在 `KEEPER_HOME/projects.json`（默认 `~/.keeper/projects.json`），每个项目记一条：id、显示名、工作区根目录、启用的 worker、默认 worker、项目级部署指令。项目自己的数据仍然只待在它工作区的 `.agent/` 里。

```bash
keeper project register ../api --id api --name "订单服务" --worker built-in --worker claude-code
keeper project list
keeper project show api
keeper project use api
```

不注册也能用：`keeper --root ../api "..."`，或者在项目目录里直接跑。root 的优先级是显式 `--root` > 已选项目 > 当前目录；显式 `--root` 和已选项目不一致时会提示，数据只写进 `--root`，不会悄悄写错项目。

**管理者**：`keeper manager "把 login 测试修好"` 会起一个前台会话。它自己处理读文件和短改动，编码、重构这类长任务派给 worker。派发时确认一次，之后 worker 内部按 hands-off 跑，不再逐条回到终端。

**worker 名册**（可用 `--worker` 按项目裁剪）：

| slug | 类型 | 适合 |
|---|---|---|
| `built-in` | 内置 loop | 小改动、读文件、短问答；用 keeper 当前的模型和工具，在本进程内跑 |
| `claude-code` | 外部 CLI | 项目内编码、重构、较长任务 |
| `codex` | 外部 CLI | 独立实现与验证代码、修测试 |
| `pi` | 外部 CLI | 已有 pi 工作流时 |

外部 worker 的启动方式分两种。`claude-code` 走 `claude-agent-sdk`：`permission_mode="default"` 加 `can_use_tool` 回调，工具请求回到 Python 里按 hands-off 放行，CLI 路径由 keeper 换成原生 `claude.exe`（SDK 不接受 npm 的 `.cmd` 垫片）。`codex` 与 `pi` 是命令行模板 `codex exec - -s workspace-write` 与 `pi --print`，任务文本从 stdin 传入。要换命令用 `KEEPER_WORKER_CODEX_ARGV='["codex","exec","-"]'` 这类环境变量，值是 JSON 数组，`{input}` 会被替换成任务文本。

**worker 能写什么**：内置 worker 只拿到只读文件工具和一个受限写工具，只能写任务指定的范围和自己那份 `MEMORY.md`，四个项目记忆文件及其备份、归档一律拦住；也没有 shell 和派发工具。外部 CLI 的真实权限由它自己的配置决定，keeper 的保护是派发前给四个记忆文件拍快照，结束后比对，被改了就还原并把这次记录标成失败，同时保留 transcript。

**交接**：worker 的输入包括四个记忆文件的路径（自己读，不内联正文）、它自己的 `MEMORY.md`、同项目其他 worker 已有的 `MEMORY.md`、对话 recap、会话日志路径、上一轮完成记录和任务说明。换 worker 时这些都交过去，所以不用重讲背景。每个 worker 在 `.agent/sub_agents/<slug>/MEMORY.md` 有自己的工作笔记，跨派发保留；共享记忆是事实源，私有记忆只是笔记。

**怎么算完成**：所有 worker 的输出统一成完成记录（结果、摘要、完整回复、transcript 路径、耗时）。退出码非 0 → 失败并留原因；超时或中断 → 中断；退出码 0 但回复为空不算成功。纯文本不算完成，最终由管理者判定。

## 项目记忆

`.agent/memory/` 下：

| 文件 | 内容 |
|---|---|
| `PROJECT_STATE.md` | 现在为真：焦点、进行中、阻塞、下一步 |
| `DECISIONS.md` | 选了什么 + 为什么 |
| `LESSONS.md` | 可复用规则、踩过的坑 |
| `INDEX.md` | 导航，自动生成，不手改 |
| `*_ARCHIVE.md` | 被取代的条目，保留不删除 |

一条 entry 是一个 `- ` bullet，细节缩进，`## ` 是主题分组。每条 bullet 下附一行 `<!--mem id:… created:… touched:…-->`：注释在注入前剥掉，不占上下文，id 用于整理前后的守恒校验。替换旧条目时，旧条目原样移入 archive，新条目写 `supersedes: <旧 bullet 原文>`。

启动注入顺序是 `DECISIONS → LESSONS → INDEX → PROJECT_STATE`。

预算按「UTF-8 字节数 / 3」计，计入 bullet 与缩进细节，不含 mem 注释和 archive。下表是 `KEEPER_CONTEXT_WINDOW=65536` 的基准值，实际按窗口等比缩放 0.5x–2x：

| 文件 | soft | hard |
|---|---|---|
| `DECISIONS.md` | 3000 | 4500 |
| `LESSONS.md` | 2500 | 4000 |
| `PROJECT_STATE.md` | 1200 | 1800 |
| `INDEX.md` | 600 | 900 |

超过 hard 后，注入视图先丢掉缩进细节，仍超限再按 `## ` 分组折叠（每组保留最新一条并标出省略数）。文件本身不会因为超预算被改写。

### 自动整理

一次性 chat、交互模式 `/exit` / `/quit` / EOF、任务正常结束后触发，Ctrl+C 不触发。整理是同步的，退出时可能多等一会儿；`KEEPER_MEMORY_AUTO=0` 或 `--no-tidy` 可关闭。

不是每次都跑：要同时满足「超过 soft 预算 + 内容有变化 + 过了冷却」才调用模型。冷却基础 30 分钟，连续几次跑完没有改动会按 30→60→120→240 分钟拉长，上限 4 小时；有实际改动就回到 30 分钟。

整理只归档「不再为真 / 不再有效」的条目，不按年龄、不因超预算归档；归档文件只追加，不删除。

安全网：整理前把 live + archive + INDEX 备份到 `backups/tidy-<ts>/`（保留 5 份），备份失败就不编辑；`DECISIONS` / `LESSONS` 严格校验 id 守恒，丢了条目会从备份恢复；INDEX 重建；结果写进 `tidy-run.json`。整理失败只打印警告，不改变原命令的退出码。

**每次运行结束后还会先跑一次记忆记录回合**：把这次运行值得长期留下的事实写进四个记忆文件。它只能读写这四个文件，越界调用直接拒绝；只记录本次事实，不做归档和合并。整理和人工 `update_memory` 与它共用同一把锁，锁忙时跳过。同样有备份、id 守恒和 run record 兜底，失败时四个文件一个字节都不变。

**想保留某条不被整理改写**：在它的 mem 注释里写 `pinned:1`。整理后被改写、移动、删除或转档，keeper 会按备份把原块连注释和位置一起还原；没被编辑的文件不会被重写。人工 `update_memory` 经 diff 确认仍然可以替换 pinned 条目。

**想让它马上整理**：让 agent 调用 `request_memory_checkpoint` 写一个请求标记，也可以自己跑 `keeper memory tidy`（强制整理，仍备份 / 守恒）或 `keeper memory tidy --check`（只看状态，不需要 API key）。请求标记只是「建议整理」，仍然要过上面的冷却判断。

## 上下文压缩

每个 step 调模型前检查上一次的 usage 加上其后消息的估算；超过 `KEEPER_CONTEXT_WINDOW - KEEPER_RESERVE_TOKENS` 时，把切点之前的前缀摘要掉，保留最近约 `KEEPER_KEEP_TOKENS` tokens。

摘要固定六段：Goal / Constraints & Preferences / Progress / Key Decisions / Next Steps / Critical Context。切点不会拆开 assistant 的工具调用和它的结果；摘要不可用不会中断任务，压缩照常完成。会话 JSONL 只追加 `compaction` 记录，旧消息不删，`--resume` 按顺序回放。

## 工具与审批

| 工具 | 权限 | 说明 |
|---|---|---|
| `read_file` | 自动放行 | 带行号读取，支持分段 |
| `list_dir` | 自动放行 | 非递归列目录，跳过 `.git` / `.agent` / `__pycache__` |
| `search` | 自动放行 | 纯文本子串搜索 |
| `write_file` | 需确认 | 新建或覆盖，覆盖时展示 diff |
| `update_memory` | 需确认 | 更新 `.agent/memory/` 的决策 / 教训 / 状态；展示 diff，INDEX 自动生成 |
| `request_memory_checkpoint` | 自动放行 | 只写一个请求标记，请会话结束后整理一次 |
| `shell` | 需确认 | argv 数组、命令白名单、固定 cwd、超时和输出截断 |
| `complete_task` / `block_task` | 自动放行（仅任务模式） | 显式完成 / 阻塞当前任务；阻塞原因必填 |
| `dispatch_task` | 需确认（仅管理者） | 把任务派给一个 worker；确认时展示项目、worker、任务和可写范围 |
| `dispatch_fanout` | 需确认（仅管理者） | 并行派 2–5 个互不依赖的子任务，全部结束后一次性返回 |

审批规则：只读放行；写入和执行问 `[y/N]`；删除和高危命令直接拒绝；没有审批器、异常、EOF、超时都按拒绝处理。

所有路径限制在项目根目录内；`write_file` 拒绝写 `.git/` 和 `.agent/`；`update_memory` 是 `.agent/memory/` 的唯一人工写入入口，被拒绝时文件字节不变。

## 数据落在哪

都在项目的 `.agent/` 里：

- `sessions/*.jsonl`：会话记录，只追加，压缩记录也在这里。
- `memory/`：四个记忆文件、归档、整理前的备份 `backups/tidy-<ts>/`、整理结果 `tidy-run.json`。
- `tasks.json`：任务队列。
- `sub_agents/<slug>/MEMORY.md`：worker 私有记忆。
- `sub_agents/<slug>/runs/<id>.log`：外部 worker 的 transcript。

项目注册表不在项目里，在 `KEEPER_HOME/projects.json`（默认 `~/.keeper/projects.json`）。记忆和队列都用临时文件 + 原子替换写入。

## 不做

Web UI、后台常驻服务、OS 级沙箱、向量库 / RAG、外部事件触发器（GitHub / 飞书 / 日历）、递归的内部子 agent（管理者可以派发，worker 不能再派）。

## 开发

```bash
pytest -q
```

## 文档

[需求](docs/REQUIREMENTS.md) · [决策](docs/DECISIONS.md) · [维护](docs/MAINTENANCE.md) · [参考](docs/REFERENCES.md)
