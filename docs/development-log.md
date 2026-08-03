# Development Log

这份文档记录 `my-agent` 的真实演进过程。Git commit history 从 2026-08-03
首次发布到 GitHub 开始；此前阶段根据当前源码、测试、trace 和开发对话整理，
不会伪造成过去已经存在的 commits。

## Milestone 1: Read-only Diagnostic Loop

第一版目标是理解最小 coding-agent loop，而不是立即构建完整 IDE 或 TUI。

- 定义统一的 `UserMessage`、`AssistantMessage`、`ToolCall` 和 `ToolResultMessage`。
- 实现 `list_files`、`read_file`、`search` 和 `run_tests` 四个只读诊断工具。
- 使用 `Workspace` 限制文件访问范围。
- 使用 `PermissionPolicy` 限制可执行的测试命令。
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

## Milestone 6: Capability-neutral Runtime Contracts

在增加写工具之前，先解除核心运行时与“只读模式”的命名耦合。

- Python package 现在使用能力中立的 `my_agent` 名称。
- 权限类现在使用 `PermissionPolicy`，为后续统一表达
  read、write 和 execute 决策保留稳定边界。
- session state 目录统一为 `.my-agent/`。
- trajectory schema 升级为 `my-agent.trajectory.v2`。
- 不提供旧 import、旧类名、旧目录或旧 schema 的兼容层。

这一步只改变身份和契约，不增加能力。当前 agent 仍然只暴露诊断工具，避免在权限模型、
审计字段和失败语义尚未设计完成时提前获得写权限。

## Milestone 7: Single-file Apply Patch

第一个写能力采用单文件 patch，而不是先开放通用 shell。

- `ApplyPatchTool` 暴露一个严格 JSON Schema 参数：`patch`。
- `patches.py` 解析 `Add File`、`Update File` 和 `Delete File`。
- 每次调用只允许一个文件操作，使 action、observation 和失败边界保持清晰。
- Update 使用精确上下文；不匹配或存在多个匹配位置时拒绝修改。
- `Workspace` 拒绝绝对路径、`..` 和 symlink traversal。
- `PermissionPolicy` 拒绝 `.git` 和 `.my-agent` 中的写操作。
- parser 在执行层限制 100,000 字符 patch 和 1 MB Update 目标，不只依赖 tool schema。
- Add/Update 先写同目录临时文件，再通过 `os.replace()` 原子替换目标文件。
- `ToolResult.metadata` 记录 `applied`、`operation` 和 `changed_files`，现有 trajectory
  可以直接把它保存为 observation，因此不需要升级 schema。

当前没有多文件事务、模糊上下文匹配、二进制/CRLF 编辑、交互 approval 或 OS sandbox。
这些限制是显式契约，不由模型提示词代替执行层检查。

## Current Boundaries

- `apply_patch` 可以自动修改 workspace 内的单个普通 UTF-8/LF 文件。
- 文件写入没有交互 approval，也没有操作系统级 sandbox。
- `run_tests` 使用命令 allowlist，但不是操作系统级 sandbox；测试代码仍可能产生副作用。
- 本地 `.my-agent/`、`trace.jsonl` 和 `trajectory.json` 默认不提交到 Git。
- 当前只有 DeepSeek adapter，provider abstraction 已为其他模型保留边界。

## Next Candidates

- 增加只读 `git_diff`，让模型验证实际修改结果。
- 将 `run_tests` 扩展为受控 shell policy，并明确命令、环境变量和工作目录边界。
- 将测试命令放入更强的 sandbox。
- 增加 context budget 和 conversation compaction。
- 为真实任务建立 eval cases，并使用 trajectory 分析失败类型。
