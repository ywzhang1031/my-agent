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

## Milestone 13: Provider Protocol and Harness-side Invocation

早期 `Provider` 接口能运行 DeepSeek，但把厂商协议、字符串 stream event 和 retry policy 放在
同一个类中。这个里程碑直接替换旧接口，不保留兼容别名。

- `provider.py` 定义 `ModelRequest`、typed `ModelEvent`、ordered `ModelOutputItem`、
  `ModelUsage`、`ModelMetadata`、`ProviderCapabilities` 和 classified `ModelError`。
- `ProviderAdapter.stream_once()` 的契约是“一次厂商 API attempt”；adapter 只负责把 canonical
  message/tool 降低为 wire payload，再把 response/error 提升回统一类型。
- `ModelInvoker` 属于 harness，集中负责 retry/backoff/cancellation。只有 retryable error 且尚未
  出现 text、reasoning 或 tool-call delta 时才会自动重试，避免流式输出被重复播放。
- invoker 强制每个成功 stream 恰好产生一个 terminal `ModelCompleted`，并拒绝 completed 后继续
  发事件的 adapter，避免 string event 约定产生非法状态。
- `DeepSeekAdapter` 删除内部 retry loop；HTTP、connection、stream、protocol 和 context overflow
  被分类为统一错误，并保留 status、provider code、request ID 和 Retry-After 等可用信息。
- DeepSeek `reasoning_content` 继续保存在 canonical assistant message 中，tool-use 后续请求仍会
  回传；这属于 DeepSeek wire mapping，不进入 agent loop 的厂商分支。
- CLI 通过显式 `ProviderConfig -> create_provider_adapter()` 绑定 adapter；当前只有 DeepSeek 实现，
  factory 不虚构 OpenAI、Anthropic 或 Gemini 已可用，也不引入全局动态插件系统。
- provider contract tests 覆盖 canonical payload、typed deltas、fragmented tool call、usage、
  自动重试边界、无 completed stream 和非法 tool arguments。

## Milestone 14: Runtime Model Selection

模型身份从启动参数升级为可发现、可切换并随 session 恢复的交互状态。

- 底部 toolbar 同时显示当前 `provider/model` 和 context 使用量，不需要执行额外命令确认模型。
- `/model` 打开独立模型选择器，当前模型优先高亮；上下方向键移动、`Enter` 确认、`Esc` 取消，
  不要求用户输入 model ID。catalog 外的自定义 ID 只通过启动配置传入。
- slash completer 负责一级命令发现；菜单激活时，上下方向键切换高亮项，`Enter` 直接执行，
  `Tab`/`Shift-Tab` 继续作为快捷补全。模型候选由 terminal 选择器渲染，不把二级交互编码进
  slash command 字符串；普通输入状态下，上下方向键仍由 history 导航处理。
- `AgentRuntime.switch_model()` 只允许在 worker idle 且没有 pending turn 时调用，保留当前
  conversation 和 session approval。
- `AgentLoop.switch_model()` 同步替换 `ModelInvoker`，并根据新 `ModelProfile` 重建 context
  limit，同时保留 compaction threshold、target 和 summary budget。
- `SessionState` 保存 `provider_id` 和 `model_id`；`resume` 在没有显式 `--model` override 时
  恢复上次选择。
- 当前 catalog 只包含已接入 provider 的 DeepSeek V4 Flash/Pro；该功能切换 model，不切换
  provider，也不代表其他厂商 adapter 已实现。

## Milestone 15: OpenAI Responses and User Provider Config

provider 选择从编译期常量升级为用户级配置，同时增加第二个真实 wire adapter。

- `config.py` 内置可运行的 DeepSeek/OpenAI provider metadata，并从
  `~/.config/my-agent/config.json` 合并用户模型；每个模型显式记录 context window 和最大输出。
- API key 单独从 `credentials.json` 读取，结构为 provider-keyed `type=api` credential，文件权限
  必须是 `0600`。provider 对应环境变量优先，CLI 不打印 key，也不写入 session/trace。
- DeepSeek 和 OpenAI 都只接收 config/factory 已解析的 key、模型元数据、base URL 和 options；
  adapter 不再自行读取环境变量，避免第二套优先级。
- `OpenAIAdapter` 使用 Responses API，映射 `response.output_text.delta`、reasoning summary、
  function call arguments、usage、terminal status 和统一错误；adapter 仍只执行一次 HTTP attempt。
- OpenAI reasoning/tool loop 需要回传原生 response output。canonical `AssistantMessage` 新增 opaque
  `provider_data`，harness 只持久化；OpenAI adapter 保存并恢复 continuation items，DeepSeek
  adapter 不读取该字段。
- `/model` 现在展示配置中的完整 `provider/model`，可以在 idle session 中跨 provider 切换；
  `/config` 只展示路径、默认模型和 provider IDs。
- factory 仍是显式 adapter registry。向 config 添加未知 provider 不等于该 provider 已可调用；
  只有实现并注册 adapter 后才会进入 CLI 可选目录。

## Milestone 16: Plan/Build Collaboration Modes

agent 从单一 Build/ReAct 执行路径升级为显式的 Plan -> Approve -> Build 状态转换。

- `modes.py` 集中定义 mode instructions、tool allowlist 和 completion protocol。Build 保持全部工具；
  Plan 只注入 `git_diff`、`list_files`、`read_file`、`search`，安全边界由 tool registry 保证，
  不依赖 prompt 要求模型自律。
- Plan mode 可继续使用只读 tool loop 探索仓库；响应中第一对完整的 `<proposed_plan>` block 由
  harness 提升为 `ProposedPlan`。标签前引导语或标签后确认话术不会破坏 gate；provider adapter
  不理解 mode、计划或任务完成语义。
- `SessionState` 持久化 `mode_id`、plan lifecycle 和 pending turn 的 mode/approved plan ID。
  pending turn 只能在原 mode 下 retry，避免恢复时悄悄更换 system prompt 和工具集合。
- `/mode` 使用现有上下方向键选择器；toolbar、`/summary` 和 `/tools` 立即反映当前 mode。
- proposed plan 展示后立即打开 `Execute plan / Revise plan` selector。Execute 将计划标记为
  approved、切到 Build 并启动关联的新执行 turn；Revise 标记为 revising，旧计划必须由新 proposal
  替换后才能执行。`/approve` 保留给 Esc、resume 和非 TTY 场景。
- Build final answer 将计划标为 completed；abort、length 或 max-steps 会恢复为 proposed。
  model/tool 错误仍保留 pending turn，沿用 `/retry` 和 `/abort` 语义。
- terminal 使用增量 protocol filter 隐藏跨 chunk 的 `<proposed_plan>` 标签、丢弃结束标签后的
  冗余确认话术并实时输出正文；普通 Plan 回复、reasoning 和 tool progress 也保持流式。
- trajectory schema 直接升级为 `my-agent.trajectory.v5`，记录 turn mode、approved plan、停止原因、
  plan lifecycle events 和 metrics，包括 `plan_revision_requested`，不保留 v4 输出开关。
- collaboration modes 与 permission modes 保持正交。本里程碑没有加入 `acceptEdits`、auto、
  bypass 或 subagent；这些需要独立 policy/state 设计，不能通过增加 system prompt 冒充。

## Milestone 17: Agent Service and Typed Client Protocol

CLI 与 agent harness 之间新增 transport-neutral application boundary，避免未来每个 UI 重写业务流程。

- `protocol.py` 定义 versioned command/event dataclass 和 JSON 编解码。command 带 command ID 与
  session ID；event 带 event ID、session ID、单 session 递增 sequence 和显式 runtime state。
- 客户端事件按 model、tool、approval、turn、mode/model change 和 plan lifecycle 建立具体类型，
  不再要求 renderer 解析任意 `kind: str + data: dict`。
- `AgentService` 注册单 session `AgentRuntime`，路由 turn、interrupt、approval、model/mode、plan 和
  abort command，并把内部 runtime event 转成外部协议事件。
- Execute Plan 的 approve -> switch Build tools -> persist -> start approved turn 由 service 编排；
  CLI 不再复制这个状态转换。model/mode switch、revision 和 abort 同样由 service 持有。
- `AgentRuntime` 保持单 session worker、cancellation 和 approval suspension；`AgentLoop` 保持
  model -> tool -> observation 算法，没有加入 transport、UI 或 session catalog 逻辑。
- CLI 仍在同一 Python 进程内调用 service。当前没有 HTTP/WebSocket/stdio server，也没有远程
  session create/list protocol；这些是稳定 command/event schema 之上的后续 transport 工作。

## Milestone 18: Durable Events and JSONL Transport

typed protocol 从进程内 queue 扩展到可恢复的客户端时间线和真实进程边界。

- `EventStore` 将每个客户端事件以 compact JSONL 追加到 session 的 `events.jsonl`；写入发生在
  publish 之前，sequence 必须严格递增，损坏、session mismatch 或乱序日志会显式失败。
- `AgentService.register()` 从 event log 恢复 sequence，因此 `serve --session-id` 后的新事件不会
  从 1 重新开始。`ReplayEventsCommand(after_sequence)` 重放原 event ID/sequence，并以新的
  `ReplayCompletedEvent` 标记 replay 边界。
- `JsonlAgentServer` 使用 stdin 接收 `command_to_dict()`，用 stdout 输出 `event_to_dict()`；独立 writer
  thread 允许模型生成期间继续接收 interrupt 和 approval command。无效 JSON、版本、类型或 session
  mismatch 返回 `server.error`，不污染 stdout 协议。
- `serve` 启动时创建 session，或用 `--session-id` 恢复；第一条消息为 `server.ready`。当前每个 server
  进程绑定一个 session，先验证 transport 和 recovery，不提前实现多租户 app daemon。
- `ask/chat/resume` 也启用同一 EventStore。event log 服务 UI replay；raw trace 继续服务 trajectory，
  两者不混用。

## Milestone 19: Multi-session AppServer and RPC Control Plane

单 session stdio server 被直接升级为 workspace-bound AppServer，为桌面 UI 或其他客户端提供稳定的
session 生命周期和请求响应边界。

- `rpc.py` 定义独立 `RequestEnvelope`、`ResponseEnvelope`、`RpcError` 和 `rpc_version`。RPC response
  用 `request_id` 关联同步操作是否被接受；agent event 继续使用 `event_id`、`session_id` 和单 session
  `sequence`，两套身份不混用。
- `AppServer` 持有一个 `AgentService` 和多个 `AgentRuntime`，实现 `session.create/list/read/resume/close`、
  `model.list`、`mode.list`，并把 turn、approval、model/mode、plan 和 replay RPC 翻译为原有 typed
  command。session close 只卸载 runtime，不删除 durable state。
- 同一 session 的第二个 active turn 返回结构化 `conflict`；不同 session 的 runtime worker 和 event
  sequence 相互独立，可以并发运行。
- `JsonlAppServer` stdout 明确复用三个 channel：`control` 输出 ready，`response` 输出 RPC 结果，
  `event` 输出异步 agent event。response/event 不依赖全局顺序，客户端分别按 request ID 和 session
  sequence 消费。
- `serve --session-id` 和直接 command JSONL 被删除。新客户端必须先发送 `session.create` 或
  `session.resume` RPC，再发送 control RPC；不保留旧 wire protocol 的兼容分支。
- 当前 AppServer 只绑定一个 workspace，stdio 只承载一个本地 client connection。鉴权、多 client、
  HTTP/WebSocket 和运行中 turn 的进程崩溃恢复仍未实现。

## Milestone 20: Local Web Client and Loopback HTTP Bridge

在原有 AppServer 上增加第一版可用浏览器客户端，没有复制 agent loop 或 provider 逻辑。

- `LocalWebServer` 只绑定 loopback，通过 `POST /api/rpc` 转发原有 RPC，通过 `GET /api/events` long-poll
  转发 typed events，并直接提供无外部依赖的 HTML/CSS/JavaScript。
- server 启动时生成随机页面 token；RPC/event 要求 `X-My-Agent-Token`，Host 必须是 loopback，响应
  设置 no-store、CSP、nosniff、no-referrer 和禁止 framing。该边界只面向本机，不等价于远程鉴权。
- Web UI 从 `session.read` 构建权威历史，从 typed events 构建运行中投影；支持 session、流式
  answer/thinking、tool activity、approval、Plan execute/revise、model/mode、interrupt/retry/abort 和
  context breakdown。
- 新增 `context.inspect` RPC；浏览器不读取 `AgentRuntime`、`AgentLoop` 或 session 文件内部对象。
- terminal event 发布前先保存 session，保证客户端收到 `turn.completed/failed/interrupted` 后立即读取
  时不会看到旧投影。
- 当前 long-poll 仍消费一个全局 queue，只支持一个 UI consumer。对照 Codex 的 per-connection
  notification 与 turn/item lifecycle、OpenCode 的 session projection + durable log + live delta 分层，
  下一步应先增加 subscriber fan-out 和结构化 item projection，再考虑多窗口或远程客户端。

## Milestone 21: Provider Credential Recovery

Web UI 将 provider authentication failure 从通用 pending-turn 恢复提升为可操作的凭据流程。

- `AgentLoop` 将 `ModelError` 的 kind、phase、provider/model 和 action requirement 写入结构化
  `turn.error`，并持久化到 `PendingTurn`；页面刷新后仍能恢复失败原因。
- `provider.credential.set` RPC 从目标 session 推导 provider，不接受浏览器指定 provider；配置层原子
  更新既有 provider-keyed `credentials.json`，新文件权限固定为 `0600`，响应不包含 key。
- 保存后 AppServer 重建当前 session runtime，使 adapter 立即读取新凭据；Web 客户端自动重试原
  pending turn，不重复追加用户消息。
- 缺失 key 和 HTTP authentication error 都进入凭据面板；其他 retryable/provider 错误仍使用原有
  retry/abort 恢复流程。
- API key 不进入 trace、session、event、activity 或浏览器存储。loopback token 继续保护 RPC，远程
  credential management 仍不在当前安全边界内。

## Milestone 22: Protocol V2, Item Projection, and Event Fan-out

客户端协议从按渲染动作划分的零散 event，升级为可重建的 `session -> turn -> item` 生命周期。

- `protocol_version=2` 为 event 增加稳定 `turn_id` 和 `durable`，并统一使用
  `item.started/delta/completed/failed`。assistant text、reasoning、tool call、user message 和 plan 都是
  带稳定 `item_id` 的 item。
- `AgentService` 把 model 完整响应和 tool 完整结果收敛为 durable `item.completed`；started/delta 只用于
  低延迟实时显示。`events-v2.jsonl` 因此保存完整终态而不保存逐 token delta。
- `SessionProjection` 在服务端消费同一 typed event stream，形成可序列化的 turn/item 物化视图；
  `session.read` 返回该 projection，浏览器刷新时不需要猜测尚未结束的 UI 状态。
- `runtime_state` 单独表示当前进程是否真的在执行；服务重启后即使 durable projection 最后停在
  `in_progress`，pending turn 也会显示为可重试，而不会被误判成仍在运行。
- `EventBroker` 为每个客户端创建独立 subscription，可按 session 过滤并 fan-out 同一 event。订阅时
  使用 `after_sequences` 从 durable log 补齐断线窗口，再无缝接收新事件。
- Web UI 按 `item_id` 更新节点，而不是依赖“当前 assistant 气泡”这类隐式全局状态；subscription
  过期时会重新订阅并读取服务端 projection。
- V1 event log 保持原文件不动，但运行时直接使用 `events-v2.jsonl`，没有双协议兼容分支。

概念边界同步固定：session/thread 是长生命周期容器，turn 是一次执行，item 是 turn 内原子内容或动作；
mode 是横切的协作策略。plan artifact、subagent execution 和 cross-turn goal state 都不等于 mode。

## Milestone 23: User-Controlled Step Budget

固定 `max_steps` 不再结束 turn，而是成为一批连续模型决策的软预算。

- 每批达到预算后，`AgentLoop` 保留 `PendingTurn`、消息、工具 observation 和下一 step，不再追加
  `Stopped after...` 伪 final answer，也不结束关联的 plan execution。
- Runtime 进入 `waiting_user`，Service 发布 durable `turn.continuation_required`；Web 与 CLI 都让用户选择
  继续下一批或停止。每次继续获得同样大小的新预算，因此不存在固定的 turn 总步数上限。
- `session.json` 保存 pause reason/details，页面刷新或进程重启后仍能恢复决策面板。
- trajectory schema 升级为 `my-agent.trajectory.v6`，记录 continuation events 和
  `turn_continuation_requests`，区分“等待用户”与 completed/failed。

## Milestone 24: Single AppServer Ownership

Web UI 支持多页面连接，但 session 存储目录不再允许多个 AppServer 进程并发写入。

- `ServerInstanceLease` 使用操作系统 `flock` 独占 `<sessions-dir>/.app-server.lock`；锁随进程退出自动释放，
  文件中的 PID、instance ID、启动时间和路径只用于诊断。
- 第二个 `ui` / `serve` 实例会立即失败并提示复用现有服务或选择不同的 `--sessions-dir`。
- HTTP health/bootstrap/response header 和 JSONL `server.ready` 暴露同一个 instance ID。Web 页面检测到
  服务重启后自动 reload，重新获取启动 token；缺少 instance ID 的旧后端会提示先重启服务。
- 这个约束不限制客户端数量：多个浏览器或 Codex 页面仍使用独立 subscription 接收同一事件 fan-out。

## Current Boundaries

- `apply_patch` 可以自动修改 workspace 内的单个普通 UTF-8/LF 文件。
- `git_diff` 只比较 `HEAD` 和当前工作树；untracked 文件只展示名称，不展示内容。
- `chat/resume` 会为 `apply_patch` 和 `exec_command` 请求交互 approval；`ask` 自动批准静态
  policy 已允许的操作。审批策略与 sandbox policy 保持独立。
- `exec_command` 默认在 macOS Seatbelt `workspace-write` sandbox 中执行：写入限于 workspace/
  temp，`.git`、`.my-agent` 与网络被拒绝；它仍可读取宿主可读文件。
- Linux/Windows backend 尚未实现；`workspace-write` 会失败关闭。`danger-full-access` 必须由 CLI
  显式选择。
- `exec_command` 不支持 shell command string、pipe/redirection、stdin、TTY、后台 session、
  package install。
- context token 是字符启发式估算，不是 provider tokenizer 的精确结果；compaction 使用
  确定性 extractive summary，不是模型生成的语义摘要。
- context breakdown 同样是估算值；provider usage 只能给出精确总 input，不能反推每类的精确
  tokenizer 计数。
- stream 在已经显示部分 delta 后不会自动重放；用户必须使用 `/retry` 或 `/abort`。
- model cancellation 是协作式检查；底层网络 read 期间可能要等待 I/O 返回或 timeout。
- `/abort` 只清理 conversation state，不回滚已经执行的文件修改或进程副作用。
- Plan mode 不暴露 `exec_command`，因此也不运行可能写 cache/build artifact 的测试命令；审批后由
  Build mode 执行验证。
- 每个 session 当前只支持一个结构化 Goal；没有 Goal 依赖图、deadline、blocked 状态或自动
  subagent 调度。
- 本地 `.my-agent/`、`trace.jsonl` 和 `trajectory.json` 默认不提交到 Git。
- 当前实现 DeepSeek Chat Completions 和 OpenAI Responses adapter；其他厂商仍需实现并测试自己的
  wire adapter。
- `serve` 是单 workspace、多 session 的 stdio AppServer；`ui` 是 loopback-only HTTP client，已支持
  同一 AppServer 上多页面独立 subscription/fan-out，并禁止多个进程共用 session 存储目录，但尚未提供
  远程鉴权、WebSocket/SSE transport、event retention/compaction 或运行中 turn 的跨进程恢复。

## Next Candidates

- 为 subscription 增加连接 capability negotiation，并评估 WebSocket/SSE transport。
- 为 durable event log 增加 retention/compaction；live-only delta 已不落盘，但 terminal facts 仍会持续增长。
- 增加 Linux/Windows sandbox backend，并收紧 macOS 的只读范围与资源限制。
- 在确定性 compactor 之上增加可选的 provider semantic summary 和 summary 质量验证。
- 增加更多 provider adapter，并对 streaming/tool-call edge cases 建立 contract tests。
- 把 permission mode 独立建模为 manual/accept-edits/auto 等策略，不与 collaboration mode 枚举耦合。
- 在有真实并发需求后增加 Explore/Review subagent 和父子 trajectory，而不是把它们当主 mode。
- 为真实任务建立 eval cases，并使用 trajectory 分析失败类型。

## Milestone 25: Sandbox Execution Boundary

允许的测试和构建命令不再只依赖 argv allowlist。新 `sandbox.py` 将持久化 policy 与平台 backend
分开，`AgentLoop` 把 policy 传给 `ToolContext`，`ProcessRunner` 在 spawn 前包装命令，并在 tool
observation 中记录实际 `sandbox_mode` 和 `sandbox_backend`。

- 默认 `workspace-write` 在 macOS 使用 Seatbelt：全盘只读，workspace/temp 可写，`.git`、
  `.my-agent` 和网络显式拒绝。
- `danger-full-access` 是显式逃生口；`trusted` permission profile 只跳过审批，不绕过 sandbox。
- session、CLI toolbar、AppServer view 和 Web topbar 都显示同一持久化 sandbox mode。
- 不支持的平台返回 `sandbox_error`，不静默降级为无隔离执行。

## Milestone 26: Sandbox Control Plane

Sandbox policy 从启动参数和只读状态提升为可查询、可切换的 session 控制面。

- `SANDBOX_PROFILES` 明确描述每种模式的宿主读取、写入范围、网络、危险级别和 backend 可用性；
  `sandbox.list` 将这些事实提供给客户端，不让 UI 猜测安全语义。
- `SwitchSandboxCommand` 与 `SandboxChangedEvent` 进入 typed protocol；`AgentService` 只在 turn 边界切换，
  更新 runtime policy、持久化 `session.json`，再发布事件。
- `session.create` 可指定 `sandbox_mode`；CLI `/sandbox` 和 Web 顶栏选择器调用同一控制面。
  不可用 backend 在 Web 中禁选，`danger-full-access` 显式标为高风险。
- `exec_command` 的 approval details 包含实际 sandbox mode、写入边界和网络策略，用户批准的是完整
  执行条件，而不只是 argv。
- 本里程碑仍未增加 Linux/Windows backend，也没有把 permission approval 与 OS sandbox 合并；
  前者决定“是否允许执行”，后者决定“获准后进程实际上能访问什么”。

## Milestone 27: Runtime Permission Grant Control Plane

`allow_session` 不再是 broker 内部不可见的集合，而是可查询、可撤销的运行时授权。

- `ApprovalGrant` 保留精确的 `(action, resource)` 和用户审批时看到的 description；
  `RuntimeApprovalBroker` 在同一把锁下完成命中、添加、撤销和清空。
- `RevokePermissionGrantCommand`、`ClearPermissionGrantsCommand` 和
  `PermissionGrantsChangedEvent` 进入 typed protocol；grant snapshot event 是 live-only，避免重启后
  从 durable event log 错误恢复已经失效的授权。
- AppServer 增加 `permission.grants.list`、`permission.grant.revoke` 和
  `permission.grants.clear`。查询结果显式返回 `lifetime=runtime`、`persistent=false`。
- CLI `/grants` 与 Web inspector 使用同一 Service 控制面，可逐项撤销或全部清空。
- 本里程碑不增加 project/global 永久授权。撤销只约束后续工具调用，已开始执行的工具不会被追溯中断。

## Milestone 28: Structured Goal State

跨 turn 目标不再只存在于自然语言历史中，而是成为 session 的结构化、可持久化状态。

- `GoalState` 保存稳定 ID、objective、有序 steps、派生状态和更新时间；v1 每个 session 只有一个当前
  Goal，最多 12 个 step，且最多一个 step 处于 `in_progress`。
- Build mode 新增 `update_goal` harness tool。模型提交完整快照，harness 统一校验并替换旧状态；Plan
  mode 不暴露该工具，仍只产出可审批的 `ProposedPlan` artifact。
- 当前 Goal 会追加到每次模型请求的 system instructions；tool observation 仍反馈本次更新结果，下一
  step 因而同时看到即时 observation 和持久状态。
- session 保存 Goal，trace 记录 `goal_updated` / `goal_cleared`，trajectory schema 升级到 v7，并提供
  Goal event 与 metrics。
- typed protocol 增加 `goal.updated`、`goal.cleared` 和 `goal.clear`；AppServer 提供 `goal.read` /
  `goal.clear`。CLI `/goal` 和 Web inspector 使用同一 service 边界。
- v1 不实现多个并行 Goal、依赖图、deadline、blocked 状态或 subagent 调度；Goal 也不改变原有
  model -> tool -> observation loop。

## Milestone 29: Goal Validation and Lifecycle Boundaries

本轮只补 Goal 契约，不新增 mode、调度机制或验证证据模块。

- 修复 `status: []` / `status: {}` 导致 `TypeError` 的问题；先检查类型再判断枚举，非法输入返回
  `invalid_goal` observation，允许模型在同一 turn 内纠正。
- 校验根参数对象、拒绝 schema 之外的字段；objective 的校验由工具更新和 session 恢复共用。
  tool schema 同步公开长度上限。校验完成前不替换当前 Goal。
- 磁盘 Goal 必须具有有效的 ID、更新时间和显式派生 status；不再隐式补齐缺失的 status。
- 明确 Goal completed 是模型报告的步骤状态，而非测试通过证明；讨论型 final answer 不会强制完成
  Goal，旧 Goal 也不构成自动执行授权。
- 新增 11 项测试，覆盖参数边界、同 turn 修正、Goal ID、恢复/重试/中止、running/pending 清除限制，
  以及 Plan mode 即使收到模型误发的 `update_goal` 也不能修改状态。
