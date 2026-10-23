# My Agent

一个从只读诊断 loop 逐步生长出来的最小 coding agent。它现在具备 Plan/Build collaboration
mode、可持久化 permission profile、流式模型输出、tool 进度、交互审批、可中断/恢复 turn、命令行历史和自动 context
compaction。Build 主循环是：

```text
user task -> model -> tool proposal -> approval -> tool result -> model -> final answer
```

Build mode 暴露这些工具：

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

CLI 创建的新 session 默认使用 `--sandbox workspace-write`。在 macOS 上，`exec_command` 会由
`/usr/bin/sandbox-exec` 以 Seatbelt profile 启动：允许读取宿主文件，写入只开放给当前 workspace
和临时目录，并额外禁止写 `.git`、`.my-agent` 与网络访问。`--sandbox danger-full-access` 会显式
关闭这层 OS 隔离；permission profile 只控制审批，`trusted` 不会自动关闭 sandbox。当前尚未实现
Linux/Windows backend，请求 `workspace-write` 时会失败关闭而不是静默裸跑。
运行中的 session 可通过 CLI `/sandbox` 或 Web 顶栏切换；切换会经过 typed
`sandbox.switch -> sandbox.changed` 控制面并写回 `session.json`。running 或 pending turn 存在时
拒绝切换，避免同一个 turn 的不同工具调用处于不同隔离边界。
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

## Provider Configuration

程序默认从同一个用户级目录读取 provider 配置和凭据：

```text
~/.config/my-agent/
  config.json
  credentials.json
```

`config.json` 保存默认 `provider/model`、base URL、模型目录、context/output 上限和 provider
options；`credentials.json` 只保存 API key。完整格式见
[`docs/config.example.json`](docs/config.example.json) 和
[`docs/credentials.example.json`](docs/credentials.example.json)。credentials 使用与 OpenCode
`auth.json` 类似的 provider-keyed 结构：

```json
{
  "deepseek": {"type": "api", "key": "..."},
  "openai": {"type": "api", "key": "..."}
}
```

`credentials.json` 必须是 `0600`；权限更宽时程序会拒绝读取。`DEEPSEEK_API_KEY` 和
`OPENAI_API_KEY` 优先于文件中的对应 key。`MY_AGENT_CONFIG`、`MY_AGENT_CREDENTIALS` 或 CLI
的 `--config`、`--credentials` 可以覆盖默认路径。配置文件不存在时使用内置的 DeepSeek 和
OpenAI 模型目录；凭据文件不存在时仍可使用环境变量。

要增加同一 provider 的模型，只需在该 provider 的 `models` 对象中加入模型 ID、说明和 token
上限，模型便会出现在 `/model` 选择器中。只有已经实现 adapter 的 provider 才会显示为可选项；
当前实现为 `deepseek` 和 `openai`。

CLI 提供 `ask`、`chat`、`resume`、本地 Web `ui` 和 transport-facing `serve` 入口。
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

## Local Web UI

`ui` 在 loopback 地址上启动与 CLI 共用的 multi-session AppServer，并提供浏览器客户端：

```bash
DEEPSEEK_API_KEY=... .venv/bin/python cli.py ui --workspace . --open-browser
```

默认地址是 `http://127.0.0.1:8765/`；端口被占用时可以传 `--port 0` 让操作系统选择可用端口，
实际 URL 会打印到 stderr。页面支持：

- 创建、恢复、重命名和切换持久化 session；侧边栏支持 `…`、右键和 `Shift+F10`
  打开会话菜单，名称保存在 `session.json`
- 流式 answer/thinking、provider retry 和 tool activity
- `apply_patch` / `exec_command` 的 once/session/deny 审批
- Plan mode 计划正文、执行或修改操作
- provider/model 与 Build/Plan mode 切换；模型、权限和 sandbox 使用紧凑、可键盘导航的弹出菜单
- `supervised` / `accept_edits` / `trusted` 权限策略切换
- 分类 context 使用量、pending turn 的 retry/abort 和运行中 turn 的 interrupt
- 每批模型步骤达到预算后暂停并询问“继续或停止”，继续时从下一步恢复，不伪造 final answer
- provider 报告 API key 缺失或认证失败时，显示凭据面板并将 key 原子写入
  `credentials.json`（`0600`）；保存后热重载当前 session 并自动重试 pending turn

浏览器不直接调用 provider 或 tool。所有操作都通过 `POST /api/rpc` 进入 `AppServer`，异步状态通过
`GET /api/events` long-poll 消费 typed events；历史消息和未完成 turn 的物化视图都由
`session.read` 返回。
HTTP server 只允许 loopback bind/Host，启动时生成随机 token，RPC/event 请求必须携带
`X-My-Agent-Token`。它不返回 CORS header，并设置 CSP、`frame-ancestors 'none'`、`no-store` 和
`nosniff`。这个 token 是本地页面的 CSRF/跨站访问防线，不是远程部署的用户鉴权机制。
API key 只通过 loopback RPC 传给配置层，不进入 trace、session、event 或 RPC response；浏览器不会把
key 写入 `localStorage`。环境变量仍是启动时的最高优先级，但通过页面保存的 key 会立即成为当前
进程的有效凭据，并在后续启动未覆盖环境变量时从 `credentials.json` 加载。

每个页面先通过 `events.subscribe` 创建独立 subscription，再使用自己的 `subscription_id` long-poll。
同一事件会 fan-out 到所有匹配的页面，多个 tab 不再争抢一个全局队列。subscription 只存在于当前
进程并有空闲过期时间；页面收到 `410` 后会携带各 session 的最后 sequence 自动重建订阅。

同一 session 存储目录只允许一个 AppServer 进程持有。`ui` 和 `serve` 启动时会对
`<sessions-dir>/.app-server.lock` 获取操作系统级 lease；重复启动会打印现有实例的 PID、instance ID
和启动时间，而不是让两个进程并发改写 `session.json` / `events-v2.jsonl`。一个 AppServer 仍可同时
服务外部浏览器、Codex 内置浏览器和多个 tab。HTTP response、`/api/health`、`bootstrap.js` 以及
JSONL `server.ready` 都携带当前 instance ID；页面发现后端重启后会自动刷新并获取新 token。

## Multi-session JSONL AppServer

`serve` 启动一个绑定当前 workspace 的长生命周期 AppServer。它不会在启动时隐式创建 session；
客户端通过同步 RPC 管理多个 session，再从同一个 stdout 接收异步 agent events：

```bash
DEEPSEEK_API_KEY=... .venv/bin/python cli.py serve --workspace .
```

```json
{"channel":"request","rpc_version":1,"request_id":"req-1","method":"session.create","params":{"mode_id":"plan"}}
```

对应 response 使用同一个 `request_id`：

```json
{"channel":"response","rpc_version":1,"request_id":"req-1","ok":true,"result":{"session":{"session_id":"<id>","active":true}}}
```

创建后可以启动 turn：

```json
{"channel":"request","rpc_version":1,"request_id":"req-2","method":"turn.start","params":{"session_id":"<id>","user_input":"Inspect this repo"}}
```

`turn.start` response 只说明 command 已被接收。模型、tool、approval 和 terminal event 通过
`channel=event` 异步发送，仍使用 `protocol_version`、`event_id`、`session_id` 和单 session
递增 `sequence`。response 与 event 之间不承诺全局先后顺序；客户端按 `request_id` 关联 RPC，
按 `(session_id, sequence)` 处理事件。

control plane 提供：

- `session.create/list/read/resume/close/rename`
- `model.list`、`mode.list`、`permission.list`、`permission.grants.list`、`sandbox.list`、`context.inspect`
- `turn.start/retry/interrupt/abort`
- `approval.resolve`、`model.switch`、`mode.switch`、`permission.switch`、
  `permission.grant.revoke`、`permission.grants.clear`、`sandbox.switch`
- `plan.execute/revise`、`events.subscribe/unsubscribe`

`session.close` 只停止并卸载内存 runtime，不删除磁盘记录；`session.resume` 将它重新加载。AppServer
保证同一 session 同时只有一个 active turn，不同 session 可以并发运行。客户端断线后调用
`events.subscribe`，通过 `after_sequences` 提交各 session 最后确认的 sequence；service 先把遗漏的
durable events 放入该 subscription，再开始 fan-out 新事件。`session.read` 同时返回服务端维护的
`turn -> item` projection，用于刷新后恢复未完成视图。

stdio transport 适合本地 UI 子进程并持有一个连接级 subscription；local Web transport 为每个页面
提供独立 long-poll subscription。它们仍只服务本地进程，不提供远程用户鉴权；未来换成
WebSocket/SSE 时可以继续复用同一 AppServer、subscription 和 projection 边界。

`events-v2.jsonl` 与 `trace.jsonl` 用途不同：前者只保存客户端 UI 恢复所需的 durable terminal facts，
例如完整的 `item.completed` 和 turn 终态；`item.started` / `item.delta` 只实时 fan-out，不逐 token
落盘。后者记录更详细的 model request/response、tool call/result 和 plan lifecycle，供 trajectory
和调试使用。

## Runtime Concepts

这些概念不是同一层的枚举：

- `thread/session`：长生命周期对话容器，持有历史、配置、计划和多个 turn。本项目代码使用 `session`；
  产品界面也可以把它称为 thread。
- `turn`：一次用户输入触发的一次执行，从 `turn.started` 到 completed/failed/interrupted/aborted。
- `item`：turn 内具有稳定 identity 的内容或动作，例如 user message、reasoning、assistant message、
  tool call 和 plan；经历 started/delta/completed/failed 生命周期。
- `mode`：应用于运行时的协作策略预设，决定 system instructions、可见工具和完成协议。它不拥有独立
  对话，也不是执行容器。
- `plan`：`plan mode` 是一种 mode；模型产出的具体计划则是跨 turn 保存、可修改和审批的 artifact。
- `subagent`：拥有独立 context、turn 和执行状态的子 runtime，通常通过父子关系和主 agent 协作；
  它是一种执行拓扑，不是 mode。
- `goal`：跨多个 turn 持续存在的目标与进度状态，可由不同 mode 或 subagent 推进；它是持久状态，
  也不是 mode。

## Collaboration Modes

当前 collaboration mode 是两个显式的 harness 策略，不是让模型在一个无限 ReAct loop 中自行决定
是否写代码：

- `build`：默认模式，暴露全部读写和受控命令工具，继续执行 model -> tool -> observation loop。
- `plan`：只暴露 `list_files`、`read_file`、`search`、`git_diff`，不向模型提供 `apply_patch` 或
  `exec_command`。模型可以多轮探索和澄清，但不能执行实现。

可以启动时指定，也可以在 idle session 中用 `/mode` 的上下方向键选择：

```bash
.venv/bin/python cli.py chat --workspace . --mode plan
```

当 Plan mode 给出完整方案时，它必须输出一个完整的 `<proposed_plan>...</proposed_plan>` block。
这个 block 是 model 和 harness 之间的 completion protocol：provider adapter 只返回普通文本；
terminal 使用增量过滤器隐藏可能跨多个 chunk 的协议标签，同时实时显示计划正文；`AgentLoop`
在完整响应结束后提取第一对完整标签中的 block，并把 `ProposedPlan(status=proposed)` 保存到
`session.json`。模型偶尔添加的标签前引导语或标签后确认话术不会阻止 gate；结束标签后的冗余
话术不会渲染。普通回答不会生成可审批计划，但仍会正常流式显示，session 会继续停留在 Plan mode。

交互终端会在计划正文之后立即打开 action selector：`Execute plan` 默认高亮，确认后 harness
将计划标记为 `approved`、切换到 Build mode，并启动一个带 `approved_plan_id` 的新 turn；选择
`Revise plan` 会把旧计划标记为 `revising`，保持 Plan mode，等待用户输入修改意见。模型必须重新
输出完整计划后才会再次进入 `proposed`，旧计划不能直接执行。按 `Esc` 会保留 proposed plan，
稍后仍可执行 `/approve`。非交互终端不会阻塞等待 selector，也会提示使用 `/approve` 或继续修改。

只有关联的 Build turn 返回真正的 final answer，计划才变为 `completed`；model error 保留 pending
turn 供 `/retry`。`--max-steps` 是单批执行预算：达到预算会发出
`turn.continuation_required`，保留 pending turn 和已完成 observation，由用户继续下一批或 `/abort`。
只有 `/abort` 或输出截断才会把执行中的计划恢复为 `proposed`，允许再次审批。

collaboration mode 与 permission profile 是两个不同维度。`/mode` 选择任务协作方式；
`/permissions` 选择敏感工具的审批策略：

- `supervised`：`apply_patch` 和通过硬限制的 `exec_command` 都询问，默认值
- `accept_edits`：自动允许 workspace 内 patch，命令仍询问
- `trusted`：自动允许通过硬限制的 patch 和命令

Profile 使用有序的 `action + resource pattern -> allow/ask/deny` 规则，未匹配的新 action 默认拒绝。
Profile 选择保存在 `session.json`，可以按 session 独立切换；running 或 pending turn 存在时拒绝
切换。它只决定是否进入人工审批，不会覆盖 workspace/path 检查、`.git` / `.my-agent` 保护、
结构化 argv 要求或命令白名单。

`chat` 和 `resume` 中，`apply_patch` 与 `exec_command` 在执行前会暂停当前 turn 并要求审批：

- `y` / `once`：只允许本次调用
- `a` / `session`：本进程内允许相同的 action/resource
- `n` / `deny`：不执行工具，将拒绝作为 tool observation 返回给模型

`apply_patch` 的 session resource 是同一 workspace-relative path；`exec_command` 是同一组
normalized `argv`。`/grants` 可以查看这些精确授权，并逐项撤销或全部清空。grant 只存在于当前
runtime 内存中，不写入 `session.json`，关闭/恢复 session、`/new` 或重启 AppServer 都会清空。
撤销只影响后续权限判断，不会中断已经开始执行的工具。审批事实仍会进入 trace 和 durable
`approval.resolved` event；grant 快照使用 live-only event，不会被 event replay 恢复。
非交互 `ask` 无法展示审批 UI，因此 `supervised` / `accept_edits` 中需要询问的操作会被拒绝；
只有显式选择 `--permission-profile trusted` 才会自动执行通过硬限制的敏感工具。

REPL 支持这些本地命令：

- `/exit`：保存并退出
- `/new`：新建 session
- `/sessions`：列出 session
- `/summary`：显示当前 session 摘要
- `/trace`：显示当前 raw trace 路径
- `/tools`：列出可用工具
- `/context`：查看估算的 context 使用量和 compaction 状态
- `/model`：打开模型选择器，用上下方向键选择、`Enter` 切换、`Esc` 取消
- `/mode`：用同样的选择器切换 Build/Plan；running 或 pending turn 存在时拒绝切换
- `/permissions`：切换 `supervised`、`accept_edits` 或 `trusted` 权限策略
- `/grants`：查看并撤销当前 runtime 的精确 `allow_session` 授权
- `/sandbox`：切换命令的 OS 隔离边界；running 或 pending turn 存在时拒绝切换
- `/approve`：执行保留的 proposed plan，主要用于取消 selector、恢复 session 或非 TTY 场景
- `/config`：显示实际配置/凭据路径、默认模型和已解析 provider，不显示 API key
- `/retry`：从失败的 model step 恢复，不重复用户消息或已完成 tool
- `/abort`：丢弃 pending turn 的消息；已经产生的文件或进程副作用不会回滚
- `/help`：显示命令帮助

REPL 使用 `prompt_toolkit`，支持左右移动光标、行内插入和上下方向键历史。输入 `/` 会显示
全部 slash commands 及说明；菜单打开时用上下方向键移动，按 `Enter` 直接执行选中的命令。
`Tab`/`Shift-Tab` 仍可用于正向或反向补全。执行 `/model` 后会直接打开模型选择器，当前模型
默认高亮；用上下方向键移动并按 `Enter` 确认，不需要手动输入 model ID。`/mode` 和
`/permissions`、`/grants` 和 `/sandbox` 使用相同交互。底部状态栏持续显示当前 `provider/model`、collaboration mode、
permission profile、context 百分比、估算用量和 pending
状态，同时显示当前 sandbox mode。历史保存在 `.my-agent/history`，权限设为 `0600`。
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

如果所选 provider 没有可用 API key，CLI 会直接报错，不会假装完成诊断。
`trace.jsonl` 还会记录 provider retry、context compaction、pending turn 恢复和错误，后续
可以用来做失败分析、eval 或训练数据整理。

内置默认模型是 `deepseek/deepseek-v4-flash`。DeepSeek adapter 使用 Chat Completions API；
OpenAI adapter 使用 Responses API。`/model` 会显示配置中所有已有 adapter 的 `provider/model`
并支持跨 provider 切换。交互切换只允许发生在没有 running/pending turn 时；选择会保存到
`session.json`，后续 `resume` 在没有显式 provider/model override 时继续使用该模型。
默认通过 DeepSeek provider config 开启 thinking mode；可以在 `config.json` 的
`providers.deepseek.options.thinking` 中配置，或通过 `--thinking disabled` 关闭。
开启 thinking 且发生 tool calls 时，agent 会把
`reasoning_content` 保存在内部 `AssistantMessage` 中，并在后续 API 请求里回传。
OpenAI reasoning 默认使用配置中的 `medium` effort 和 `auto` summary，也可以用
`--reasoning-effort` 覆盖。Responses API 返回的原生 output items 会作为 opaque
`provider_data` 保存，以便 reasoning model 在 tool result 后继续调用；harness 不解析这些字段。

## Runtime Layers

代码按这几个边界拆分：

- `cli.py`：解析 `ask/chat/resume/serve/ui`，控制 REPL 或组装 AppServer transport，不参与 agent 决策
- `terminal.py`：封装 command completion、context UI、history 和 typed event 流式渲染，不参与 agent 决策
- `protocol.py`：定义 transport-neutral command/event 类型、V2 turn/item 生命周期、序号和 JSON 编解码
- `rpc.py`：定义同步 request/response envelope、`request_id` 和结构化 RPC error
- `event_store.py`：按 session 追加 durable typed events，校验单调 sequence，并按 cursor 重放
- `event_broker.py`：管理 per-client subscription、session filter、fan-out 和空闲过期
- `server_instance.py`：对 session 存储根目录持有跨进程 AppServer lease，并记录实例诊断元数据
- `projection.py`：把 typed events 物化为可供 UI 刷新恢复的 `session -> turn -> item` 视图
- `service.py`：按 session 路由命令，编排 model/mode/plan/abort 状态转换，保存 session，并把内部
  runtime event 提升为客户端协议事件
- `app_server.py`：管理多 session runtime 生命周期、catalog，并把 RPC 翻译为 typed command
- `transport.py`：实现 multiplexed JSONL stdio transport，区分 control/response/event channel
- `web_server.py`：实现 loopback HTTP RPC、event long-poll、随机 token 和静态 UI 资源服务
- `web/`：浏览器端 session/context 投影与 typed event 渲染，不包含 agent 决策
- `runtime.py`：单 session 执行器；在 worker 中驱动一个 turn，并发布内部 `RuntimeEvent`
- `control.py`：定义 cancellation、permission request 和 approval decision 契约
- `session.py`：序列化统一 `Message`、context 状态和可重试的 `PendingTurn`
- `config.py`：合并内置模型目录、用户 provider 配置和独立凭据，并解析 `provider/model`
- `context.py`：估算 token、保留 active turn，并摘要压缩较老的 completed turns
- `modes.py`：定义 collaboration mode 的 system instructions、tool allowlist 和 plan completion protocol
- `agent_loop.py`：维护 model -> tool -> observation -> model 循环，并发出统一 `AgentEvent`
- `provider.py`：定义厂商无关的 request、typed event、ordered output、usage、metadata、
  capability 和 classified error
- `model_invoker.py`：位于 harness 内，负责 retry/backoff、cancellation 和 provider stream
  contract 检查
- `providers/deepseek.py`：只执行一次 DeepSeek API attempt，转换 Chat Completions payload、
  解析 SSE，并把原生响应或错误提升为统一协议
- `providers/openai.py`：只执行一次 OpenAI Responses API attempt，解析语义化 SSE、function call、
  reasoning summary、usage 和 continuation items
- `providers/factory.py`：把 CLI provider 配置绑定为一个具体 adapter；它不是运行时全局插件注册表
- `tools.py`：定义 tool schema、registry、执行入口和统一 `ToolResult`
- `execution.py`：在清理后的环境中管理子进程、输出捕获、timeout 和 process group
- `sandbox.py`：把 sandbox policy 映射为平台 execution backend；当前实现 macOS Seatbelt
- `patches.py`：解析并提交单文件 patch，不依赖 shell 命令
- `workspace.py` / `permissions.py`：限制路径范围、受保护目录和命令形态
- `trace.py` / `trajectory.py`：分别保存 raw event log 和规范化 trajectory

一次 turn 的调用顺序是：

```text
CLI or AppServer loads SessionState
  -> resolves the persisted collaboration mode
     -> builds a mode-specific ToolRegistry and system prompt
  -> local CLI sends StartTurnCommand directly
     or remote client sends turn.start RPC to AppServer
        -> AppServer validates params and creates StartTurnCommand
  -> AgentService routes the command to the registered session
     -> translates it into the internal StartTurn command
  -> AgentRuntime starts one worker and publishes lifecycle events
  -> AgentLoop appends UserMessage
  -> PendingTurn records the stable turn_id, step, and completed tool calls
  -> ContextManager estimates the next request
     -> if needed, summarize old completed turns and keep recent raw messages
  -> AgentLoop creates one provider-neutral ModelRequest
  -> ModelInvoker applies harness retry/cancellation policy
  -> selected ProviderAdapter performs one API attempt and maps its wire protocol
  -> typed ModelEvent deltas become internal AgentEvent / RuntimeEvent
  -> AgentService converts them to versioned, sequenced turn/item protocol events
     -> SessionProjection applies every live event by stable turn_id/item_id
     -> EventStore appends durable terminal facts such as ItemCompletedEvent
     -> EventBroker fans every live event out to matching client subscriptions
  -> CLI or remote client renders ItemStarted/Delta/Completed/Failed lifecycle events
  -> ModelCompleted contains ordered text/reasoning/tool-call output plus usage/metadata
  -> AgentLoop either accepts a final answer
     -> in Plan mode, a complete proposed_plan block becomes persisted ProposedPlan state
     or asks ToolRegistry to validate and describe every tool call
  -> write/exec action becomes PermissionRequest
     -> client receives ApprovalRequestedEvent
     -> local CLI sends ResolveApprovalCommand
        or remote client sends approval.resolve RPC
     -> the same suspended tool call resumes; the model is not called again first
  -> approved call executes; denied call becomes a failed ToolResult
     -> exec_command applies the persisted sandbox policy before spawning the process
  -> ToolResult becomes ToolResultMessage
  -> next model request sees the new observation
  -> after edits, git_diff exposes the actual working-tree delta
  -> exec_command runs the narrowest allowlisted validation
  -> AgentService saves messages, trace, and trajectory after the terminal event

Plan approval is a separate transition:

proposed plan -> action selector -> ExecutePlanCommand -> AgentService
                              -> approve -> Build -> StartTurn(approved_plan_id)
                              \-> Revise -> revising -> new proposed plan
                              \-> Esc -> keep proposed for /approve
Build execution -> final answer: completed
                -> step budget: waiting_user -> continue next batch / abort
                -> abort/length: proposed again
```

`AppServer` 是 UI/transport 的 control-plane 边界；`AgentService` 是 session runtime 与 harness 的
应用边界。`StartTurnCommand`、
`InterruptTurnCommand`、`ResolveApprovalCommand`、`SwitchModelCommand`、`SwitchModeCommand`、
`SwitchPermissionProfileCommand`、`SwitchSandboxCommand`、
`ExecutePlanCommand`、`RevisePlanCommand` 和 `AbortTurnCommand` 都只包含可序列化字段。输出不再是
`kind: str + data: dict`，而是 `ItemStartedEvent`、`ItemDeltaEvent`、`ItemCompletedEvent`、
`TurnContinuationRequiredEvent`、
`ApprovalRequestedEvent`、`TurnCompletedEvent` 等具体 dataclass；每个事件包含协议版本、event ID、
session ID、可选 turn ID、单 session 递增序号、durable 标记和显式状态，并可转换为 JSON object。

`AgentRuntime` 降为 service 内部的单 session 执行器，状态机仍使用 `idle`、`running_model`、
`waiting_approval`、`running_tool`、`completed`、`failed`、`interrupted`。`ask/chat/resume` 仍在进程
内调用 service；`serve` 用一个 AppServer 持有多个 runtime。后续 WebSocket adapter 只需解码 RPC、
调用 `AppServer.handle()` 并编码 response/event，不需要复制 session lifecycle、plan approval 或
mode switching。`AgentLoop` 不读取 stdin，也不知道 prompt_toolkit、AppServer、event store 或
transport。

Context token 数量是保守估算，不是 provider tokenizer 的精确计数。达到输入预算的 85% 时，
`ContextManager` 会按完整 user turn 边界压缩较老历史，目标降到 70%；当前 active turn、最近
原始消息和完整本地 `messages.jsonl` 会保留。摘要是确定性的 extractive summary，不会递归
调用模型，因此它便宜且可预测，但不等同于语义无损记忆。分类 breakdown 与发送前的同一份
`ContextSnapshot` 一起生成，类别之和必须等于 `estimated_tokens`，终端不会另算一套数字。

`ModelInvoker` 只会在尚未向终端输出任何 text、reasoning 或 tool-call delta 时自动重试
adapter 标记为 retryable 的错误。provider adapter 本身从不循环重试，只分类并返回错误。
流中途失败时自动重放可能造成重复输出，所以 harness 保留 `PendingTurn`，由用户
执行 `/retry` 从同一个 model step 恢复。已经成功执行的 tool call ID 会被记住，不会在恢复时
再次执行。中断检查发生在 request、SSE chunk、重试等待和工具执行边界；Python `urllib`
正在等待网络读取时无法立即强制关闭底层 socket，因此模型中断最坏仍可能等到当前 I/O 返回
或请求 timeout。子进程中断会直接终止整个 process group。

`exec_command` 的 allowlist 是 harness policy，macOS Seatbelt 是独立的 execution boundary。
前者决定“哪类命令可被提出”，permission profile 决定“是否询问用户”，后者限制获准进程的实际
写入和网络能力。当前 `workspace-write` 仍允许读取宿主可读文件，因此不是秘密隔离；也尚未限制
CPU、内存、进程数或总运行时。开放任意 shell、包安装、外部 MCP 或无人值守 subagent 前仍需
更完整的跨平台 sandbox 与资源治理。

`final_answer` 仍然是 harness 的判断：当规范化后的 `ModelResponse.tool_calls` 为空时，这次 turn
完成。Plan mode 在这个分支上进一步检查 completion protocol，区分普通回答与 `proposed_plan`；
provider API 层不判断任务或计划是否完成。turn 完成不代表整个 session 结束，是否继续下一轮由
CLI REPL 决定。

## Adding a Provider

当前真正实现并测试的 adapter 是 DeepSeek Chat Completions 和 OpenAI Responses。统一协议已经
把接入新厂商所需的改动限制在 provider 边界内，但不能把这一点表述为“已经兼容所有模型 API”。
新增 Anthropic、Gemini 或 OpenAI-compatible endpoint 时，按下面顺序扩展：

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

`trace.jsonl` 是原始事件流；导出的 `trajectory.json` 使用 `my-agent.trajectory.v6`，先按 turn
分组，再按 step 分组，避免不同 turn 中相同 step 编号发生碰撞。每轮结束后会自动刷新
`trajectory.json`，手动导出主要用于重新生成或校验现有 session trajectory。v5 为每个 turn
增加 `mode_id`、`approved_plan_id`、`stopped_by`、`plan_events` 和 plan lifecycle metrics；
approval、interruption 和 context 字段继续保留。
