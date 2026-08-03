# Development Log

这份文档记录 `my-agent` 的真实演进过程。Git commit history 从 2026-08-03
首次发布到 GitHub 开始；此前阶段根据当前源码、测试、trace 和开发对话整理，
不会伪造成过去已经存在的 commits。

## Milestone 1: Read-only Diagnostic Loop

第一版目标是理解最小 coding-agent loop，而不是立即构建完整 IDE 或 TUI。

- 定义统一的 `UserMessage`、`AssistantMessage`、`ToolCall` 和 `ToolResultMessage`。
- 实现 `list_files`、`read_file`、`search` 和 `run_tests` 四个只读诊断工具。
- 使用 `Workspace` 限制文件访问范围。
- 使用 `ReadOnlyPermissionPolicy` 限制可执行的测试命令。
- 跑通 `user -> model -> tool call -> tool result -> model -> final answer`。

## Milestone 2: Provider Adapter and DeepSeek Thinking

模型厂商协议被隔离在 provider adapter 中，agent loop 只依赖统一接口。

- `Provider` protocol 定义 harness 需要的最小能力。
- `DeepSeekProvider` 负责 Chat Completions 请求和响应转换。
- 默认启用 thinking mode。
- `reasoning_content` 保存在统一 assistant message 中，并在后续 tool-use 请求中回传。

这使未来新增 provider 时不需要改动 session、tools 和 agent loop。

## Milestone 3: Trace and Canonical Trajectory

为了分析 agent 行为，运行过程被拆成 raw trace 和 canonical trajectory 两层。

- `TraceRecorder` 逐条写入 `model_request`、`model_response`、`tool_call`、
  `tool_result` 和 `final_answer`。
- `trajectory.py` 将事件按 turn 和 step 重新组织。
- trajectory 显式区分 model response、action、observation 和 final answer。
- `final_answer` 由 harness 在“没有 tool calls”时判断，不是 provider 的特殊返回类型。

## Milestone 4: Session-aware Runtime

单次进程调用被升级为可恢复的 session runtime。

- CLI 提供 `ask`、`chat` 和 `resume`。
- `SessionStore` 持久化 `session.json`、`messages.jsonl`、`trace.jsonl` 和
  `trajectory.json`。
- `AgentLoop.run_turn()` 在已有消息历史上执行新的一轮。
- 每轮使用独立 `turn_id`，轮内 model/tool 循环使用 `step`。
- 最终 assistant answer 会写回消息历史，保证下一轮可以看到上一轮结论。

## Milestone 5: Direct-upgrade Policy

项目不再保留旧 one-shot wrapper 或旧 trajectory fallback。

- `AgentLoop.run_turn()` 是唯一 loop 入口。
- CLI 只接受 `ask`、`chat` 和 `resume`。
- trajectory 只接受 session trace。
- 后续修改直接升级当前接口和 schema，不为未发布旧版维护兼容分支。

## Current Boundaries

- Agent tools 不提供文件写入能力。
- `run_tests` 使用命令 allowlist，但不是操作系统级 sandbox；测试代码仍可能产生副作用。
- 本地 `.readonly-agent/`、`trace.jsonl` 和 `trajectory.json` 默认不提交到 Git。
- 当前只有 DeepSeek adapter，provider abstraction 已为其他模型保留边界。

## Next Candidates

- 增加第二个 provider adapter，验证统一协议是否足够稳定。
- 增加 context budget 和 conversation compaction。
- 将测试命令放入更强的 sandbox。
- 为真实任务建立 eval cases，并使用 trajectory 分析失败类型。
