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

## Milestone 8: Read-only Git Diff

文件编辑之后增加独立的只读验证工具，而不是要求模型通过通用 shell 拼接 Git 命令。

- `GitDiffTool` 只接受可选的 workspace-relative `path`，不接受任意 Git 参数。
- tracked 变化使用固定的 `git diff HEAD -- <path>`，同时覆盖 staged 和 unstaged 修改。
- untracked 文件使用 `git ls-files --others --exclude-standard` 单独发现，只返回文件名。
- 禁用 pager、颜色、external diff 和 textconv，并设置 `GIT_OPTIONAL_LOCKS=0`。
- workspace 是大仓库子目录时，pathspec 仍限制在 workspace 内，不暴露兄弟目录变化。
- `.my-agent` session state 无论目标仓库是否忽略它，都不会进入 untracked observation。
- stdout 最多 12,000 字符，untracked metadata 最多 200 项。
- 非 Git workspace 或没有 `HEAD` commit 时返回明确失败 observation。

`git_diff` 不写文件、不修改 index，也不运行 shell。它的结果继续使用通用
`ToolResult -> ToolResultMessage -> observation` 路径，所以 trajectory schema 保持 v2。

## Milestone 9: Structured Controlled Execution

专用 `run_tests` 被直接替换为能力更通用、边界更明确的 `exec_command`，不保留旧接口。

- tool input 从 command string 改为结构化 `argv`，因此不会启动 shell，也不解析管道、
  重定向、command substitution 或 chaining。
- `PermissionPolicy.decide_command()` 分别返回 `invalid`、`deny` 或 `allow`，并为允许的
  命令标记 `test`、`check` 或 `build` category。
- policy 按 executable、subcommand 和关键 flag 判断命令形态；拒绝 executable path、
  Python `-c`/脚本、shell、package install 和直接文件操作。
- `cwd` 必须是存在的 workspace-relative directory；`..`、绝对路径和解析到 workspace
  外部的 symlink 都会被拒绝。
- `ProcessRunner` 使用筛选后的环境变量，不把 DeepSeek 或其他 provider secret 传给子进程。
- 子进程没有 stdin/TTY；输出通过临时文件捕获并截断；timeout 会终止整个 process group，
  避免测试启动的子进程继续存活。
- observation metadata 区分 `success`、`nonzero_exit`、`timed_out`、`spawn_error`、
  `denied` 和 `invalid_request`，同时记录 category、normalized argv、cwd 和 duration。
- agent loop、provider adapter、统一 message 和 trajectory schema 都不需要改变；它们只看到
  新的 tool schema 和通用 `ToolResult`。

## Milestone 10: Streaming Usability Runtime

这一阶段不增加新的代码修改能力，而是把已有 loop 变成可以持续使用和恢复的交互程序。

- `Provider.send()` 被直接升级为 `Provider.stream()`；统一 `ProviderEvent` 表达 reasoning、
  content、retry 和 completed，不保留旧 blocking provider 接口。
- `DeepSeekProvider` 使用 Chat Completions SSE，逐块聚合 `reasoning_content`、`content`、
  fragmented tool calls、finish reason 和 usage。
- 429、500、503、连接失败和未完成 stream 只有在第一个可见 delta 之前才自动指数退避重试。
- `AgentLoop` 把 provider delta、tool 生命周期、context compaction 和 turn error 转换为
  provider-neutral `AgentEvent`；terminal renderer 只处理展示。
- REPL 使用 Python `readline` 或 macOS `libedit`，获得左右移动、行内插入、历史导航、
  `Ctrl-C` 清行和权限为 `0600` 的持久历史。
- `PendingTurn` 持久化稳定的 `turn_id`、当前 step 和已执行 call ID。`/retry` 不重复用户消息
  或已完成 tool，`/abort` 显式放弃 pending messages。
- `ContextManager` 在发送前估算 context；达到阈值时按完整 user turn 边界压缩旧历史，并保留
  active turn、最近原始消息和磁盘上的完整消息日志。
- trajectory schema 升级为 `my-agent.trajectory.v3`，记录 context snapshot、provider retry、
  turn error/abort 和新 metrics。

## Milestone 11: Discoverable Terminal and Context Accounting

交互层从基础 line editing 升级为可发现的 command palette，同时让 context UI 使用真实的
request breakdown，而不是只展示一个总数。

- 直接用 `prompt_toolkit` 替换 `readline/libedit` 输入后端，不保留两套终端实现。
- 输入 `/` 自动展示 slash commands 和说明；`Shift-Tab` 反向选择，`Tab` 接受候选；完整
  slash command 再按 `Tab` 会直接提交，与 Codex CLI 的两阶段行为一致。
- 底部 toolbar 显示 context 百分比、估算用量、context limit 和 pending turn 状态。
- `enable_history_search` 会关闭 `complete_while_typing`，因此保持普通上下键历史但禁用该选项；
  对 stdin/stdout 都是 TTY 的进程显式提供 output，避免 `TERM=dumb` 触发无菜单的 dumb prompt。
- prompt 出现前只计算一次 context snapshot，输入每个字符时复用缓存，避免长会话反复扫描。
- `ContextBreakdown` 把实际 request 拆为 system prompt、tool definitions、conversation summary、
  active conversation 和 protocol overhead；所有类别之和必须等于 `estimated_tokens`。
- `/context` 展示分类进度条、input budget、reserved output、最近 provider input usage 和消息数。
- UI 不显示当前 request 中不存在的 Rules、Skills、MCP 或 subagent 分类。
- history 继续持久化到 `.my-agent/history`，每次读取后收紧为 `0600`。

## Milestone 12: Bidirectional Runtime and Approval Flow

CLI 与 agent loop 之间从同步函数调用升级为双向 runtime protocol，为交互审批和主动中断提供
稳定边界。

- `AgentRuntime` 在单独 worker 中执行一个 turn；CLI 发送 `StartTurn`、`InterruptTurn` 和
  `ResolveApproval`，只消费 provider-neutral `RuntimeEvent`。
- runtime 状态显式区分 `idle`、`running_model`、`waiting_approval`、`running_tool`、
  `completed`、`failed` 和 `interrupted`。
- `AgentLoop` 仍负责 model/tool loop，但不再负责 stdin、审批 UI 或线程调度；终端可以在
  `approval_requested` 后暂停，并用同一 `request_id` 恢复原 tool call。
- 静态 `PermissionPolicy` 先判断操作是否合法；`PermissionRequest` 再询问用户是否授权这个
  合法副作用。两层职责不同，拒绝不会绕过 policy，也不会执行工具。
- `allow_once` 只批准当前请求；`allow_session` 按精确 `(action, resource)` 缓存在当前
  runtime；`deny` 转换为失败 `ToolResult`，让模型能看到并调整计划。
- cancellation token 贯穿 provider、approval wait、agent loop 和 `ProcessRunner`。中断会记录
  `turn_interrupted` 并保留 `PendingTurn`；正在运行的子进程会终止整个 process group。
- 如果中断发生在一组 tool calls 尚未全部形成 observation 时，harness 会为未处理 call 写入
  `interrupted` ToolResult，保证 `/retry` 发给 provider 的 assistant/tool 消息序列完整。
- trajectory schema 升级为 `my-agent.trajectory.v4`。每个 step 增加 approvals，记录
  `call_id`、decision、source、prompted 和时间；metrics 增加 approval 与 interruption 计数。
- `chat/resume` 使用交互审批；非交互 `ask` 保持自动批准静态 policy 已允许的操作，避免脚本
  因等待 stdin 卡住。该模式适用于受信任调用，不是 unattended sandbox。

## Current Boundaries

- `apply_patch` 可以自动修改 workspace 内的单个普通 UTF-8/LF 文件。
- `git_diff` 只比较 `HEAD` 和当前工作树；untracked 文件只展示名称，不展示内容。
- `chat/resume` 会为 `apply_patch` 和 `exec_command` 请求交互 approval；`ask` 自动批准静态
  policy 已允许的操作。两种模式都没有操作系统级 sandbox。
- `exec_command` 只执行 allowlisted test/check/build argv，但不是操作系统级 sandbox；
  被允许的项目代码仍可能写文件、读取用户文件或访问网络。
- `exec_command` 不支持 shell command string、pipe/redirection、stdin、TTY、后台 session、
  package install。
- context token 是字符启发式估算，不是 DeepSeek tokenizer 的精确结果；compaction 使用
  确定性 extractive summary，不是模型生成的语义摘要。
- context breakdown 同样是估算值；provider usage 只能给出精确总 input，不能反推每类的精确
  tokenizer 计数。
- stream 在已经显示部分 delta 后不会自动重放；用户必须使用 `/retry` 或 `/abort`。
- model cancellation 是协作式检查；底层网络 read 期间可能要等待 I/O 返回或 timeout。
- `/abort` 只清理 conversation state，不回滚已经执行的文件修改或进程副作用。
- 本地 `.my-agent/`、`trace.jsonl` 和 `trajectory.json` 默认不提交到 Git。
- 当前只有 DeepSeek adapter，provider abstraction 已为其他模型保留边界。

## Next Candidates

- 使用 macOS Seatbelt、Linux bubblewrap 或 container 将允许的进程放入 OS sandbox。
- 在确定性 compactor 之上增加可选的 provider semantic summary 和 summary 质量验证。
- 增加更多 provider adapter，并对 streaming/tool-call edge cases 建立 contract tests。
- 为真实任务建立 eval cases，并使用 trajectory 分析失败类型。
