# project-keeper

> CLI 命令：`keeper`

给一个本地项目配一个「长期在岗」的 AI 管家：记住项目的状态、决策和踩过的坑，
每次任务开始前把记忆装进上下文，任务结束后自己更新。

当前进度：**M1（工具与审批）**。

## 定位

「项目管家」：给一个本地项目配一个长期在岗的 AI 管家。
和普通「聊天 + 记忆」的区别不在会不会记住对话，而在工作单位是**项目**：
记忆跟着文件夹走、任务必须有始有终、不可逆操作要人批准。

## M1 能做什么

- CLI 单任务 / 交互模式
- 一轮完整 agent loop：模型 → 工具 → 回灌 → 停机
- 5 个工具：`read_file`、`list_dir`、`search`、`write_file`、`shell`
- 三级审批：只读放行、写入/执行确认、删除与高危命令直接拒绝
- `shell` 使用 argv 数组 + 命令白名单，固定项目根目录，带超时和输出截断
- 会话 JSONL append-only，`--resume` 续跑
- 工具永不抛异常，错误以 `ERROR:` 返回给模型

M1 **没有**：记忆、任务队列、上下文压缩、Langfuse、eval。按 M2–M5 逐步加。

## 快速开始

```bash
cd keeper
python -m pip install -e .      # 可选，装完可以直接用 keeper 命令
cp .env.example .env            # 填入 DeepSeek API key
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

## 工具与限制

| 工具 | 权限 | 说明 |
|---|---|---|
| `read_file` | 自动放行 | 带行号读取，支持 `start_line` / `max_lines` 分段 |
| `list_dir` | 自动放行 | 非递归列目录，跳过 `.git` / `.agent` / `__pycache__`，默认 200 条 |
| `search` | 自动放行 | 纯文本子串搜索，输出 `file:line: 内容`，跳过二进制/超大文件，默认 100 条 |
| `write_file` | 需确认 | 新建或覆盖，父目录自动创建；覆盖时审批里展示 diff |
| `shell` | 需确认 | argv 数组、命令白名单、固定 cwd，默认 30s 超时 |

所有路径都限制在项目根目录内；`write_file` 额外拒绝写入 `.git/` 和 `.agent/`。

`shell` 白名单：`git`、`python`、`python3`、`py`、`pytest`、`pip`、`pip3`。
不使用 `shell=True`，因此管道、重定向和 `;` 都只是普通参数。
超时范围 0.1–120 秒，stdout/stderr 合计按 20k 字符截断（保留头尾）。

## 审批规则

| 级别 | 行为 |
|---|---|
| 只读（`read_file` / `list_dir` / `search`） | 直接执行 |
| 写入 / 执行（`write_file` / `shell`） | 终端询问 `[y/N]`，只有 `y` / `yes` 放行 |
| 删除 / 高危 | 直接拒绝，不询问 |

高危示例：`rm -rf`、`del /s`、`rd /s`、`format`、`shutdown`、
`git reset --hard`、`git push --force` / `-f`、`git clean -f`。
策略是 fail-closed：未知权限级别、审批器异常、EOF、无输入、超时都按拒绝处理。

拒绝不是异常：工具返回 `ToolResult(ok=False, "ERROR: denied ...")`，
模型看到后可以换方案继续。审批器通过参数注入，测试不依赖 stdin。

## 常用命令

```bash
keeper "看看项目里有哪些文件"       # 单任务
keeper                               # 交互模式
keeper --resume                      # 续跑最近一次会话
keeper --session .agent/sessions/xxx.jsonl
keeper --root ../other-project "..." # 指定项目根目录
```

## 数据落在哪

`<项目根>/.agent/sessions/<时间戳>-<id>.jsonl`，一行一条记录：

```json
{"type":"meta","session_id":"...","model":"deepseek-chat","cwd":"..."}
{"type":"message","message":{"role":"user","content":"..."}}
{"type":"message","message":{"role":"assistant","content":"","tool_calls":[...]}}
{"type":"message","message":{"role":"tool","tool_call_id":"...","content":"..."}}
```

会话只追加不改写；中断后 resume 会给没跑完的工具调用补一条 `[interrupted]` 结果。
M1 没有改动这个格式，M0 的会话仍可 `--resume`。审批事件只用于终端渲染，不写入 JSONL；
被拒绝的操作会以 `ERROR: denied ...` 的形式作为普通 tool 消息落盘。

## 设计约定

- 工具 `execute` 永不抛异常，失败也走 `ToolResult(ok=False, content="ERROR: ...")`
- `Registry.dispatch` 再兜一层，工具自身的 bug 不能弄死 agent
- 路径限制在项目根目录内（fail-closed）
- 审批在执行前完成；没有审批器或审批器出错时默认拒绝
- loop 只产出事件，CLI 负责渲染（以后接 Web 不用改 loop）

## 路线

M0 骨架 → **M1 工具与审批（已完成）** → M2 记忆 → M3 任务队列 → M4 上下文工程 → M5 可观测 + eval。