# My Agent

一个最小的只读诊断型 coding agent 原型。目标是先跑通 agent loop：

```text
user task -> model -> tool call -> tool result -> model -> final answer
```

首版只暴露这些工具：

- `list_files`
- `read_file`
- `search`
- `run_tests`

`run_tests` 只允许常见测试命令，例如 `python3 -m unittest`、`pytest`、`npm test`、`go test`。

## Run

CLI 只提供 `ask`、`chat` 和 `resume` 三个 session-aware 入口。
执行一个可恢复的单轮任务：

```bash
export DEEPSEEK_API_KEY=...
python3 cli.py ask \
  --model deepseek-v4-flash \
  --workspace . \
  "Inspect this repo and suggest what to test."
```

`session: <session_id>` 会写到 stderr，最终答案写到 stdout。之后可以用同一个 ID
继续对话：

```bash
python3 cli.py resume <session_id> --workspace .
```

要直接进入多轮 REPL：

```bash
python3 cli.py chat --workspace .
```

REPL 支持这些本地命令：

- `/exit`：保存并退出
- `/new`：新建 session
- `/sessions`：列出 session
- `/summary`：显示当前 session 摘要
- `/trace`：显示当前 raw trace 路径
- `/tools`：列出可用工具

默认 session 目录是：

```text
<workspace>/.readonly-agent/sessions/<session_id>/
  session.json
  messages.jsonl
  trace.jsonl
  trajectory.json
```

可以用 `--sessions-dir` 把 session 放到 workspace 之外。`list_files` 默认会忽略
`.readonly-agent`，避免 agent 把自己的运行记录当成待诊断源码。

如果没有设置 `DEEPSEEK_API_KEY`，CLI 会直接报错，不会假装完成诊断。
`trace.jsonl` 会记录 `model_request`、`model_response`、`tool_call`、`tool_result`
和 `final_answer`，后续可以用来做失败分析、eval 或训练数据整理。

默认 provider 是 `deepseek`，使用 DeepSeek 的 OpenAI-format Chat Completions API。
默认模型是 `deepseek-v4-flash`；也可以通过 `DEEPSEEK_MODEL` 或 `--model` 覆盖。
默认开启 DeepSeek thinking mode；可以通过 `DEEPSEEK_THINKING=disabled` 或
`--thinking disabled` 关闭。开启 thinking 且发生 tool calls 时，agent 会把
`reasoning_content` 保存在内部 `AssistantMessage` 中，并在后续 API 请求里回传。

## Runtime Layers

代码按这几个边界拆分：

- `cli.py`：解析 `ask/chat/resume`，控制 REPL，并在每轮后保存 session
- `session.py`：序列化统一 `Message`，不包含任何 DeepSeek 专属格式
- `agent_loop.py`：维护 model -> tool -> observation -> model 循环，并判断本轮何时结束
- `provider.py`：定义统一 provider 协议和统一模型返回值
- `providers/deepseek.py`：在统一消息与 DeepSeek Chat Completions 格式之间转换
- `tools.py`：定义 tool schema、registry、执行入口和统一 `ToolResult`
- `workspace.py` / `permissions.py`：限制路径范围和测试命令
- `trace.py` / `trajectory.py`：分别保存 raw event log 和规范化 trajectory

一次 turn 的调用顺序是：

```text
CLI loads SessionState
  -> AgentLoop appends UserMessage
  -> Provider converts messages and tool schemas to DeepSeek payload
  -> model returns content/reasoning_content/tool_calls
  -> AgentLoop either accepts a final answer
     or asks ToolRegistry to execute every tool call
  -> ToolResult becomes ToolResultMessage
  -> next model request sees the new observation
  -> CLI saves messages, trace, and trajectory
```

`final_answer` 仍然是 harness 的判断：当规范化后的 `ProviderResponse.tool_calls`
为空时，这次 turn 完成。它不代表整个 session 结束；是否继续下一轮由 CLI REPL 决定。

## Test

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

## Development Log

项目从最小只读 loop、DeepSeek adapter、thinking/tool use、canonical trajectory，
逐步演进到 session-aware runtime。完整里程碑和设计决策见
[docs/development-log.md](docs/development-log.md)。

## Export Trajectory

```bash
WORKSPACE=/path/to/repo
SESSION_ID=20260803-120000-abcdef
PYTHONPATH=. python3 scripts/export_trajectory.py \
  --trace "$WORKSPACE/.readonly-agent/sessions/$SESSION_ID/trace.jsonl" \
  --output "$WORKSPACE/.readonly-agent/sessions/$SESSION_ID/trajectory.json"
```

`trace.jsonl` 是原始事件流；导出的 `trajectory.json` 使用
`readonly-cli-agent.session-trajectory.v1`，先按 turn 分组，再按 step 分组，避免不同
turn 中相同 step 编号发生碰撞。每轮结束后会自动刷新 `trajectory.json`，手动导出主要
用于重新生成或校验现有 session trajectory。
