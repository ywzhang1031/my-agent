# 工具输出保存与按需回读

这是工具证据的持久化和上下文优化，不是跨会话自动记忆、语义摘要或 RAG。
首版只处理 `exec_command` 的 stdout/stderr，不改变 read_file、search、git_diff 的行为。
没有引入新的模型 API、模型摘要或 UI 面板；CLI 与 Web 共用现有 AgentLoop 和事件通道。

## 为什么要拆开

过去命令结束后只读取 stdout/stderr 各自前 12,000 bytes，随后临时文件关闭。
丢掉的中间和尾部不能再读；为找错误重跑测试既浪费时间，也可能再次产生副作用。

现在将“保存证据”和“发给模型的消息”分开：小输出原样展示，大输出每路取 3 KiB 开头和 1 KiB
结尾，中间标记省略；原始字节另存，每路最多 1 MiB。预览由代码截取，不是模型生成的摘要。
工具结果携带 `artifact_id`、来源 turn/call、各流的大小、截断状态和保留部分的 SHA-256。
后续请求只重复携带预览与引用，不会展开数据库里的整份日志。

## 调用流程

```text
AgentLoop: 绑定当前 session 的 ToolOutputStore 和 turn/call ID
  -> 原有 permission / approval / sandbox 检查
  -> exec_command -> ProcessRunner 执行一次
  -> 子进程结束、超时或取消：仍在临时文件关闭前
       -> 截取首尾预览
       -> output_sink -> ToolOutputStore.save -> SQLite 事务保存原始字节
  -> ToolResult: 真实退出状态 + 小预览 + output_artifact 元数据
  -> 原有 Service 检查点 / typed event / trace
  -> 下一次模型请求
       -> 需要细节时调用 read_tool_output
       -> 当前 session + artifact_id 查库，返回有限片段和 next_offset
       -> 片段作为普通 tool result 进入上下文
```

Provider 不读取文件或数据库，只序列化原有 ToolResultMessage。源数据不成为 system instruction，
不能授权操作。模型没有额外文件访问能力，也不能把别的会话 ID 作为工具参数传入。

工具调用例子（ID 使用本次 exec_command 实际返回的值）：

```json
{
  "artifact_id": "0123456789abcdef0123456789abcdef",
  "stream": "stdout",
  "offset": 50000,
  "limit": 4096
}
```

返回片段在 `stdout`，分页信息在 `metadata`：

- `offset`、`next_offset` 是原始 byte offset，不是字符数或行号；继续读取时用返回的 next_offset。
- `eof` 表示保留部分已读完。`storage_truncated=true` 时，即使 eof 为 true，也不是全部原文。
- UTF-8 页尾的未完整字符留到下一页；任意跳到字符中部或原数据非 UTF-8 时使用替换字符。
- `total_bytes` 是观察到的原流大小；`stored_bytes` 是实际保存的前缀大小，sha256 只覆盖保留字节。
- 字段不合法、ID 不存在、跨 session 和存储故障分别返回明确错误，不自动重跑原命令。

## 存放与上限

使用已有 `--sessions-dir/agent.sqlite3`，默认是 workspace 下的 `.my-agent/sessions/agent.sqlite3`。
schema 6 新增 `tool_outputs` 表及 session 索引；原 session/event/attempt 数据不改写。
数据库沿用 owner-only `0600` 文件权限，不是加密保险箱：命令主动打印的敏感内容也可能被保留。
环境变量清理依旧有效，但不能声称输出已经自动脱敏。

| 范围 | 限制 |
|---|---|
| 单个 stdout 或 stderr 保存 | 前 1 MiB 原始字节 |
| 单次回读 | 默认 4 KiB，最大 8 KiB，最小 4 bytes |
| 单会话 | 32 MiB 保存内容或 256 个 artifact |
| 同一个会话数据库 | 256 MiB 保存内容或 4096 个 artifact |

数量和内容配额在同一事务中检查并插入，避免并发绕过；它们不是 SQLite 文件/索引/journal 的精确磁盘上限。
满额时拒绝保存新 artifact，不静默删除旧引用。暂不提供自动淘汰、跨会话共享、全文搜索或管理 RPC。
同一会话重启后仍可读取已保存引用。已结束历史的模型摘要会被要求保留有用的证据引用，
但不保证每个 ID 都被写进摘要；原始工具结果和它携带的 ID 仍保存在 SQLite。
回读片段也会占上下文，重复读完整日志仍然会变大；新增工具 schema/元数据对小日志有固定开销，
不能保证每一次请求或账单都变小。

## 失败与恢复边界

- 命令失败、超时、取消也尝试保存已经捕获的两路输出，保留真实 exit_code 和状态。
- 未获准、未启动或 sandbox 准备失败的命令不生成虚假输出证据。
- 保存失败或配额满：`output_artifact.status=unavailable` 并标记原因，保留命令实际执行结果。
  命令已执行不能被错误标成“没运行”，也不能因日志没存好自动重新执行。
- artifact 先提交，随后原有工具结果检查点提交。若进程在两者之间退出，可能留下无引用产物；
  它仍计入配额，但不能当作工具结果已确认。恢复仍遵循现有 unknown-attempt 审批路径。
- 现有运行中临时捕获文件仍可能增长，宿主崩溃前的输出也不保证归档；本改动不实现流式日志持久化、
  进程磁盘硬配额或更强的 OS sandbox。存储范围隔离不等于抵御宿主进程直接读取数据库。

## 同一 turn 内的压缩检查点

例：第 1 步跑测试，第 2 步回读日志，第 3 步请求模型时，测试预览和刚读的片段原本都要重发。
当估算超过**输入预算**的 85%（窗口先扣除预留输出）时，harness 先请当前模型摘要可压缩的已结束 turn；
若仍高于 70% 的目标，从本轮最早的完整工具批次开始缩短结果。最近一次模型回复里的**整批**
调用及结果保持原样，所以此例只改第 1 步的请求视图。达不到 70% 但未超过硬预算时仍可继续。

- 只选有已提交的相同工具结果、同会话 artifact、输出完整保留且每条至少省 128 个估算 tokens 的
  `exec_command`（成功或非零退出）和 `read_tool_output`。过长的每路预览保留前 64 和后 128 个字符；
  页面的 stream、字节区间、命令退出码、来源不变。短诊断不裁剪。未决、未知、超时、没归档的结果不选。
- 新请求保留原来的消息顺序、角色、call ID、模型原生字段、本轮指令与 Skill。数据库里的原始消息和日志不改写。
  若缩完仍超硬预算，返回 `ContextWindowError`，不推进历史摘要或本轮检查点，也不发送模型请求。
- 检查点在 `PendingTurn.compacted_outputs`：每条记录版本、原消息位置、指纹和 artifact ID。
  Service 将它与 `context_compacted` 事件同事务提交，成功后才放行下一次模型请求；
  事件含压缩前后本地估算和本次新压缩条数。正常完成或放弃本轮后，检查点随 pending turn 清除。
- 重启仍须人工确认恢复。提交前中断：从上一个已提交的工具结果重新计算视图；提交后中断：
  核对原消息、artifact 和完成记录，按相同版本重建视图。两者都不为压缩而重跑工具或推进 step。
  引用缺失时恢复报错，原始消息还在。未知执行仍按既有恢复流程处理，不拿日志存在当作成功凭证。

压缩的目的是给下一次推理腾出空间，不是承诺任意长的任务一定能继续。用户交代的约束、模型自己的
回复、最新工具批次，或没有 artifact 的大文件，如果单独就撑满预算，仍需要处理这个限制。

## 与记忆的关系

可以把它看作后续记忆/检索的基础，但当前保存的是一次执行的原始证据。
长期记忆还需要决定哪些事实值得跨会话保留、来源与时效、冲突、删除及何时召回。
不能把测试输出或模型说过的话自动升级为长期正确事实。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -p test_tool_output.py -v
.venv/bin/python -m unittest discover -s tests -p test_output_runtime.py -v
.venv/bin/python -m unittest discover -s tests -p test_output_checkpoint.py -v
.venv/bin/python -m unittest discover -s tests
```

测试使用临时会话、模型替身和真实本地测试进程，不访问模型 API。验证会话隔离、UTF-8 分页、
配额与事务回滚、版本升级、退出状态、取消/超时、按需回读、恢复和事件引用。
真实 loop 用例生成 100,015 bytes stdout：中间证据不在首尾预览中，第二次工具调用回读后才进入
请求。运行计数保持一次。这是工程行为证明，不是模型选用率、真实 tokenizer 用量或任务质量成绩。

同一离线样例的 stdout 预览为 4,112 bytes（含标记），含元数据的整条工具结果 JSON 为 5,285 bytes。
按旧版前 12,000 bytes 的截断规则重建、移除新 artifact 元数据后，同一结果为 12,650 bytes，
本例减少约 58%。这里只比较工具结果消息，不包括新增工具定义的固定开销，也不是实际 API 计费对照。
