# My Agent

一个从只读诊断 loop 逐步生长出来的最小 coding agent。它现在具备流式模型输出、
tool 进度、可恢复失败、命令行历史和自动 context compaction。当前主循环是：

```text
user task -> model -> tool call -> tool result -> model -> final answer
```

当前暴露这些工具：

- `list_files`
- `read_file`
- `search`
- `apply_patch`
- `git_diff`
- `exec_command`

`exec_command` 接受结构化 `argv`、workspace-relative `cwd` 和受 harness 上限约束的
`timeout_seconds`，不接受 shell command string。例如：

```json
{
  "argv": ["python3", "-m", "unittest", "discover", "-s", "tests"],
  "cwd": ".",
  "timeout_seconds": 60
}
```

当前 policy 只允许常见 test/check/build 形态，例如 `pytest`、`python -m unittest`、
`ruff check`、`black --check`、`mypy`、`npm run lint`、`cargo check` 和 `go build`。
它拒绝 shell operator、可执行文件路径、任意 Python 脚本、`python -c`、shell、安装命令
和文件操作命令。子进程不接收 stdin/TTY，只继承筛选后的基础环境变量；provider API key
不会传入。超时会终止整个 process group，stdout/stderr observation 各自最多保留
12,000 bytes。
`apply_patch` 每次只允许修改一个 workspace-relative 文件，支持 `Add File`、`Update File`
和 `Delete File`。例如：

```text
*** Begin Patch
*** Update File: app.py
@@
-return 1
+return 2
*** End Patch
```

更新使用精确上下文匹配；匹配失败或上下文不唯一时不会写入。绝对路径、`..`、symlink、
`.git` 和 `.my-agent` 都会被拒绝。Add/Update 使用同目录临时文件和 `os.replace()`
完成单文件原子替换。单次 patch 最多 100,000 字符，Update 目标最多 1 MB。

`git_diff` 使用固定参数直接调用 Git，不启动 shell。它返回 `HEAD` 到当前工作树的 tracked
diff，因此同时覆盖 staged 和 unstaged 修改，并附带 untracked 文件名。可以用可选的
workspace-relative `path` 限定范围；untracked 内容需要继续调用 `read_file`。输出最多
12,000 字符、untracked 最多 200 个，并且要求 workspace 位于已有 `HEAD` commit 的
Git 仓库中。

## Run

先安装唯一的终端 UI 依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

CLI 只提供 `ask`、`chat` 和 `resume` 三个 session-aware 入口。
执行一个可恢复的单轮任务：

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python cli.py ask \
  --model deepseek-v4-flash \
  --workspace . \
  "Inspect this repo and suggest what to test."
```

`session: <session_id>`、tool 状态和重试信息写到 stderr；模型正文以 delta 形式流式写到
stdout。之后可以用同一个 ID 继续对话：

```bash
.venv/bin/python cli.py resume <session_id> --workspace .
```

要直接进入多轮 REPL：

```bash
.venv/bin/python cli.py chat --workspace .
```

REPL 支持这些本地命令：

- `/exit`：保存并退出
- `/new`：新建 session
- `/sessions`：列出 session
- `/summary`：显示当前 session 摘要
- `/trace`：显示当前 raw trace 路径
- `/tools`：列出可用工具
- `/context`：查看估算的 context 使用量和 compaction 状态
- `/retry`：从失败的 model step 恢复，不重复用户消息或已完成 tool
- `/abort`：丢弃 pending turn 的消息；已经产生的文件或进程副作用不会回滚
- `/help`：显示命令帮助

REPL 使用 `prompt_toolkit`，支持左右移动光标、行内插入和上下方向键历史。输入 `/` 会显示
全部 slash commands 及说明；`Tab`/`Shift-Tab` 选择候选，`Enter` 执行。底部状态栏持续显示
context 百分比、估算用量和 pending 状态。历史保存在 `.my-agent/history`，权限设为 `0600`。
输入过程中按 `Ctrl-C` 只清空当前行；模型调用失败后可以执行 `/retry` 或 `/abort`。

`/context` 会展开当前 request 的分类账单：`System prompt`、`Tool definitions`、
`Conversation summary`、`Conversation` 和 `Protocol overhead`。它还会显示完整 context limit、
输入预算、输出预留、最近一次 provider input usage，以及 active/summarized message 数量。没有
注入 request 的 Rules、Skills、MCP 或 subagent 不会为了界面效果显示成虚构分类。

默认启用流式输出；`--no-stream` 可以只在本轮结束时显示完整答案。thinking 默认仍会发给
DeepSeek，但 CLI 只显示 `thinking...` 状态；使用 `--show-thinking` 才会流式打印
`reasoning_content`。context 上限和输出预留可以显式覆盖：

```bash
.venv/bin/python cli.py chat \
  --context-window-tokens 1000000 \
  --max-output-tokens 64000 \
  --show-thinking
```

默认 session 目录是：

```text
<workspace>/.my-agent/sessions/<session_id>/
  session.json
  messages.jsonl
  trace.jsonl
  trajectory.json
<workspace>/.my-agent/history
```

可以用 `--sessions-dir` 把 session 放到 workspace 之外。`list_files` 默认会忽略
`.my-agent`，避免 agent 把自己的运行记录当成待诊断源码。

如果没有设置 `DEEPSEEK_API_KEY`，CLI 会直接报错，不会假装完成诊断。
`trace.jsonl` 还会记录 provider retry、context compaction、pending turn 恢复和错误，后续
可以用来做失败分析、eval 或训练数据整理。

默认 provider 是 `deepseek`，使用 DeepSeek 的 OpenAI-format Chat Completions API。
默认模型是 `deepseek-v4-flash`；也可以通过 `DEEPSEEK_MODEL` 或 `--model` 覆盖。
默认开启 DeepSeek thinking mode；可以通过 `DEEPSEEK_THINKING=disabled` 或
`--thinking disabled` 关闭。开启 thinking 且发生 tool calls 时，agent 会把
`reasoning_content` 保存在内部 `AssistantMessage` 中，并在后续 API 请求里回传。

## Runtime Layers

代码按这几个边界拆分：

- `cli.py`：解析 `ask/chat/resume`，控制 REPL，并在每轮后保存 session
- `terminal.py`：封装 command completion、context UI、history 和流式渲染，不参与 agent 决策
- `session.py`：序列化统一 `Message`、context 状态和可重试的 `PendingTurn`
- `context.py`：估算 token、保留 active turn，并摘要压缩较老的 completed turns
- `agent_loop.py`：维护 model -> tool -> observation -> model 循环，并发出统一 `AgentEvent`
- `provider.py`：定义 provider streaming event 和统一模型返回值
- `providers/deepseek.py`：转换协议、解析 SSE delta、聚合 tool calls 并执行有限重试
- `tools.py`：定义 tool schema、registry、执行入口和统一 `ToolResult`
- `execution.py`：在清理后的环境中管理子进程、输出捕获、timeout 和 process group
- `patches.py`：解析并提交单文件 patch，不依赖 shell 命令
- `workspace.py` / `permissions.py`：限制路径范围、受保护目录和命令形态
- `trace.py` / `trajectory.py`：分别保存 raw event log 和规范化 trajectory

一次 turn 的调用顺序是：

```text
CLI loads SessionState
  -> AgentLoop appends UserMessage
  -> PendingTurn records the stable turn_id, step, and completed tool calls
  -> ContextManager estimates the next request
     -> if needed, summarize old completed turns and keep recent raw messages
  -> Provider converts messages and tool schemas to a DeepSeek streaming payload
  -> SSE deltas become ProviderEvent, then AgentEvent, then terminal output
  -> completed response contains content/reasoning_content/tool_calls
  -> AgentLoop either accepts a final answer
     or asks ToolRegistry to execute every tool call
  -> ToolResult becomes ToolResultMessage
  -> next model request sees the new observation
  -> after edits, git_diff exposes the actual working-tree delta
  -> exec_command runs the narrowest allowlisted validation
  -> CLI saves messages, trace, and trajectory
```

Context token 数量是保守估算，不是 provider tokenizer 的精确计数。达到输入预算的 85% 时，
`ContextManager` 会按完整 user turn 边界压缩较老历史，目标降到 70%；当前 active turn、最近
原始消息和完整本地 `messages.jsonl` 会保留。摘要是确定性的 extractive summary，不会递归
调用模型，因此它便宜且可预测，但不等同于语义无损记忆。分类 breakdown 与发送前的同一份
`ContextSnapshot` 一起生成，类别之和必须等于 `estimated_tokens`，终端不会另算一套数字。

DeepSeek adapter 只会在尚未向终端输出任何 `content`/`reasoning_content` delta 时自动重试
可重试错误。流中途失败时自动重放可能造成重复文本，所以 harness 保留 `PendingTurn`，由用户
执行 `/retry` 从同一个 model step 恢复。已经成功执行的 tool call ID 会被记住，不会在恢复时
再次执行。

`exec_command` 的 allowlist 是 harness policy，不是安全 sandbox。被允许的 test/build
命令仍会执行仓库中的代码，因此仍可能写文件、访问用户可读路径或尝试网络访问。

`final_answer` 仍然是 harness 的判断：当规范化后的 `ProviderResponse.tool_calls`
为空时，这次 turn 完成。它不代表整个 session 结束；是否继续下一轮由 CLI REPL 决定。

## Test

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

## Development Log

项目从最小只读 loop、DeepSeek adapter、thinking/tool use、canonical trajectory，
逐步演进到 session-aware runtime 和受控文件编辑。完整里程碑和设计决策见
[docs/development-log.md](docs/development-log.md)。

## Export Trajectory

```bash
WORKSPACE=/path/to/repo
SESSION_ID=20260803-120000-abcdef
PYTHONPATH=. python3 scripts/export_trajectory.py \
  --trace "$WORKSPACE/.my-agent/sessions/$SESSION_ID/trace.jsonl" \
  --output "$WORKSPACE/.my-agent/sessions/$SESSION_ID/trajectory.json"
```

`trace.jsonl` 是原始事件流；导出的 `trajectory.json` 使用
`my-agent.trajectory.v3`，先按 turn 分组，再按 step 分组，避免不同
turn 中相同 step 编号发生碰撞。每轮结束后会自动刷新 `trajectory.json`，手动导出主要
用于重新生成或校验现有 session trajectory。v3 还包含 context snapshot、provider retry、
turn error/abort 和对应 metrics。
