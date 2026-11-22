# project-keeper

> CLI 命令：`keeper`

给一个本地项目配一个「长期在岗」的 AI 管家：记住项目的状态、决策和踩过的坑，
每次任务开始前把记忆装进上下文，任务结束后自己更新。

当前进度：**M0（骨架）**。

## 定位

「项目管家」：给一个本地项目配一个长期在岗的 AI 管家。
和普通「聊天 + 记忆」的区别不在会不会记住对话，而在工作单位是**项目**：
记忆跟着文件夹走、任务必须有始有终、不可逆操作要人批准。

## M0 能做什么

- CLI 单任务 / 交互模式
- 一轮完整 agent loop：模型 → 工具 → 回灌 → 停机
- 唯一工具：`read_file`（限制在项目根目录内）
- 会话 JSONL append-only，`--resume` 续跑
- 工具永不抛异常，错误以 `ERROR:` 返回给模型

M0 **没有**：记忆、任务队列、审批、压缩、Langfuse、eval。按 M1–M5 逐步加。

## 快速开始

```bash
cd keeper
python -m pip install -e .      # 可选，装完可以直接用 keeper 命令
cp .env.example .env            # 填入 DeepSeek API key
python -m keeper "读一下 README 并总结"
```

不安装也能跑：`python -m keeper ...`。

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `KEEPER_API_KEY` / `DEEPSEEK_API_KEY` | 无 | 必填 |
| `KEEPER_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容地址，需带版本号 |
| `KEEPER_MODEL` | `deepseek-chat` | 模型名 |
| `KEEPER_MAX_STEPS` | `30` | 单任务最大步数 |

## 常用命令

```bash
keeper "读一下 README 并总结"        # 单任务
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

## 设计约定

- 工具 `execute` 永不抛异常，失败也走 `ToolResult(ok=False, content="ERROR: ...")`
- `Registry.dispatch` 再兜一层，工具自身的 bug 不能弄死 agent
- 路径限制在项目根目录内（fail-closed）
- loop 只产出事件，CLI 负责渲染（以后接 Web 不用改 loop）

## 路线

M0 骨架 → M1 工具与审批 → M2 记忆 → M3 任务队列 → M4 上下文工程 → M5 可观测 + eval。