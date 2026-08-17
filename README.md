# My Agent

一个从只读诊断 loop 逐步生长出来的最小 coding agent。它现在具备流式模型输出、
tool 进度、交互审批、可中断/恢复 turn、命令行历史和自动 context compaction。当前主循环是：

```text
user task -> model -> tool proposal -> approval -> tool result -> model -> final answer
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

`chat` 和 `resume` 中，`apply_patch` 与 `exec_command` 在执行前会暂停当前 turn 并要求审批：

- `y` / `once`：只允许本次调用
- `a` / `session`：本进程内允许相同的 action/resource
- `n` / `deny`：不执行工具，将拒绝作为 tool observation 返回给模型

`apply_patch` 的 session resource 是同一 workspace-relative path；`exec_command` 是同一组
normalized `argv` 与 `cwd`。`/new` 会创建新 runtime，因此也会清空 session approvals。
非交互 `ask` 不会等待 stdin 审批：它会自动批准已经通过静态 file/command policy 的操作，
适合明确受信任的任务，不应作为无人值守安全边界。

REPL 支持这些本地命令：

- `/exit`：保存并退出
- `/new`：新建 session
- `/sessions`：列出 session
- `/summary`：显示当前 session 摘要
- `/trace`：显示当前 raw trace 路径
- `/tools`：列出可用工具
- `/context`：查看估算的 context 使用量和 compaction 状态
- `/model`：打开模型选择器，用上下方向键选择、`Enter` 切换、`Esc` 取消
- `/retry`：从失败的 model step 恢复，不重复用户消息或已完成 tool
- `/abort`：丢弃 pending turn 的消息；已经产生的文件或进程副作用不会回滚
- `/help`：显示命令帮助

REPL 使用 `prompt_toolkit`，支持左右移动光标、行内插入和上下方向键历史。输入 `/` 会显示
全部 slash commands 及说明；菜单打开时用上下方向键移动，按 `Enter` 直接执行选中的命令。
`Tab`/`Shift-Tab` 仍可用于正向或反向补全。执行 `/model` 后会直接打开模型选择器，当前模型
默认高亮；用上下方向键移动并按 `Enter` 确认，不需要手动输入 model ID。底部状态栏持续显示当前
`provider/model`、context 百分比、估算用量和 pending 状态。历史保存在 `.my-agent/history`，
权限设为 `0600`。
输入过程中按 `Ctrl-C` 只清空当前行；turn 运行或等待审批时按 `Ctrl-C` 会发送
`InterruptTurn`。中断后的 pending turn 可以执行 `/retry` 继续，或用 `/abort` 丢弃。
模型调用失败后同样可以执行 `/retry` 或 `/abort`。
交互 stdin/stdout 是 TTY 时，CLI 会显式启用完整 terminal layout；即使宿主错误地报告
`TERM=dumb`，slash menu 也不会退化为纯文本 `input()`。

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
交互选择器当前包含 `deepseek-v4-flash` 和 `deepseek-v4-pro`；自定义 model ID 仍可通过
`DEEPSEEK_MODEL` 或启动参数 `--model` 设置。交互切换只允许发生在没有 running/pending turn
时；选择会保存到 `session.json`，后续 `resume` 在没有显式传入 `--model` 时继续使用该模型。
默认开启 DeepSeek thinking mode；可以通过 `DEEPSEEK_THINKING=disabled` 或
`--thinking disabled` 关闭。开启 thinking 且发生 tool calls 时，agent 会把
`reasoning_content` 保存在内部 `AssistantMessage` 中，并在后续 API 请求里回传。

## Runtime Layers

代码按这几个边界拆分：

- `cli.py`：解析 `ask/chat/resume`，控制 REPL，并在每轮后保存 session
- `terminal.py`：封装 command completion、context UI、history 和流式渲染，不参与 agent 决策
- `runtime.py`：接收 `RuntimeCommand`、在 worker 中驱动 turn，并发布 `RuntimeEvent`
- `control.py`：定义 cancellation、permission request 和 approval decision 契约
- `session.py`：序列化统一 `Message`、context 状态和可重试的 `PendingTurn`
- `context.py`：估算 token、保留 active turn，并摘要压缩较老的 completed turns
- `agent_loop.py`：维护 model -> tool -> observation -> model 循环，并发出统一 `AgentEvent`
- `provider.py`：定义厂商无关的 request、typed event、ordered output、usage、metadata、
  capability 和 classified error
- `model_invoker.py`：位于 harness 内，负责 retry/backoff、cancellation 和 provider stream
  contract 检查
- `providers/deepseek.py`：只执行一次 DeepSeek API attempt，转换 Chat Completions payload、
  解析 SSE，并把原生响应或错误提升为统一协议
- `providers/factory.py`：把 CLI provider 配置绑定为一个具体 adapter；它不是运行时全局插件注册表
- `tools.py`：定义 tool schema、registry、执行入口和统一 `ToolResult`
- `execution.py`：在清理后的环境中管理子进程、输出捕获、timeout 和 process group
- `patches.py`：解析并提交单文件 patch，不依赖 shell 命令
- `workspace.py` / `permissions.py`：限制路径范围、受保护目录和命令形态
- `trace.py` / `trajectory.py`：分别保存 raw event log 和规范化 trajectory

一次 turn 的调用顺序是：

```text
CLI loads SessionState
  -> CLI sends StartTurn to AgentRuntime
  -> AgentRuntime starts one worker and publishes lifecycle events
  -> AgentLoop appends UserMessage
  -> PendingTurn records the stable turn_id, step, and completed tool calls
  -> ContextManager estimates the next request
     -> if needed, summarize old completed turns and keep recent raw messages
  -> AgentLoop creates one provider-neutral ModelRequest
  -> ModelInvoker applies harness retry/cancellation policy
  -> DeepSeekAdapter performs one API attempt and maps the wire protocol
  -> typed ModelEvent deltas become AgentEvent and terminal output
  -> ModelCompleted contains ordered text/reasoning/tool-call output plus usage/metadata
  -> AgentLoop either accepts a final answer
     or asks ToolRegistry to validate and describe every tool call
  -> write/exec action becomes PermissionRequest
     -> CLI receives approval_requested
     -> user sends ResolveApproval(allow_once/allow_session/deny)
     -> the same suspended tool call resumes; the model is not called again first
  -> approved call executes; denied call becomes a failed ToolResult
  -> ToolResult becomes ToolResultMessage
  -> next model request sees the new observation
  -> after edits, git_diff exposes the actual working-tree delta
  -> exec_command runs the narrowest allowlisted validation
  -> CLI saves messages, trace, and trajectory
```

`AgentRuntime` 是 CLI 和 harness 的双向边界。输入命令目前是 `StartTurn`、
`InterruptTurn`、`ResolveApproval`；输出事件覆盖 model delta、tool lifecycle、approval lifecycle
和 turn terminal state。状态机使用 `idle`、`running_model`、`waiting_approval`、
`running_tool`、`completed`、`failed`、`interrupted`。`AgentLoop` 不读取 stdin，也不知道
prompt_toolkit，因此未来替换为 TUI、IDE 或远程客户端时不需要把交互逻辑塞回 loop。

Context token 数量是保守估算，不是 provider tokenizer 的精确计数。达到输入预算的 85% 时，
`ContextManager` 会按完整 user turn 边界压缩较老历史，目标降到 70%；当前 active turn、最近
原始消息和完整本地 `messages.jsonl` 会保留。摘要是确定性的 extractive summary，不会递归
调用模型，因此它便宜且可预测，但不等同于语义无损记忆。分类 breakdown 与发送前的同一份
`ContextSnapshot` 一起生成，类别之和必须等于 `estimated_tokens`，终端不会另算一套数字。

`ModelInvoker` 只会在尚未向终端输出任何 text、reasoning 或 tool-call delta 时自动重试
adapter 标记为 retryable 的错误。DeepSeek adapter 本身从不循环重试，只分类并返回错误。
流中途失败时自动重放可能造成重复输出，所以 harness 保留 `PendingTurn`，由用户
执行 `/retry` 从同一个 model step 恢复。已经成功执行的 tool call ID 会被记住，不会在恢复时
再次执行。中断检查发生在 request、SSE chunk、重试等待和工具执行边界；Python `urllib`
正在等待网络读取时无法立即强制关闭底层 socket，因此模型中断最坏仍可能等到当前 I/O 返回
或请求 timeout。子进程中断会直接终止整个 process group。

`exec_command` 的 allowlist 是 harness policy，不是安全 sandbox。被允许的 test/build
命令仍会执行仓库中的代码，因此仍可能写文件、访问用户可读路径或尝试网络访问。
交互 approval 表达用户意图，也不改变这一安全边界。

`final_answer` 仍然是 harness 的判断：当规范化后的 `ModelResponse.tool_calls`
为空时，这次 turn 完成。它不代表整个 session 结束；是否继续下一轮由 CLI REPL 决定。

## Adding a Provider

当前真正实现并测试的 adapter 只有 DeepSeek。统一协议已经把接入新厂商所需的改动限制在
provider 边界内，但不能把这一点表述为“已经兼容所有模型 API”。新增 OpenAI、Anthropic、
Gemini 或 OpenAI-compatible endpoint 时，按下面顺序扩展：

1. 实现一个 `ProviderAdapter`，提供不可变的 `ModelProfile` 和 `stream_once()`；每次调用只发起
   一次厂商请求，不在 adapter 内做 retry、tool execution、context compaction 或权限判断。
2. 把统一 `ModelRequest` 降低为厂商 payload；把厂商 stream 提升为 `TextDelta`、
   `ReasoningDelta`、`ToolCallDelta` 和唯一的 `ModelCompleted`。
3. 把厂商输出保存在有序 `ModelOutputItem` 中，将 stop reason、usage 和 error 映射到统一类型；
   response ID、实际模型 ID、request ID 和少量诊断字段放入 `ModelMetadata`。
4. 在 `providers/factory.py` 添加显式构造分支和该 adapter 的配置校验，然后增加相同的 contract
   tests：message/tool 转换、fragmented tool arguments、usage、错误分类、stream 终止和 retry 边界。

`AgentLoop` 只依赖 `ModelInvoker`，因此一个合格的新 adapter 不应要求修改 loop、session、
tool registry、approval 或 trajectory 的控制逻辑。若新厂商需要图像输入、prompt caching、
structured output 等当前统一 request 尚未表达的能力，应先扩展 canonical protocol 和 capability，
再在各 adapter 中显式支持或拒绝，不能把厂商私有字段直接传进 harness。

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
`my-agent.trajectory.v4`，先按 turn 分组，再按 step 分组，避免不同
turn 中相同 step 编号发生碰撞。每轮结束后会自动刷新 `trajectory.json`，手动导出主要
用于重新生成或校验现有 session trajectory。v4 在 context snapshot、provider retry 和
turn error/abort 基础上，增加 approvals、interrupted outcome 及对应 metrics。
