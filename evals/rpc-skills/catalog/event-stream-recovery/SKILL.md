---
name: event-stream-recovery
description: Implement reconnect and replay for agent event subscriptions using cursors, event identity, and snapshot reconciliation. Use for missing or duplicated displayed events after SSE disconnection, not resubmitting mutating RPC commands.
---

# 事件订阅恢复

分开检查事件产生、持久化、传输和前端投影。订阅重连是恢复观察，不是重新启动任务；
先确认重复的是事件展示还是业务执行，避免用重新发起控制请求修复流中断。

定义游标的包含/排除语义、排序范围和保留期；重复事件按稳定身份合并，不按文本去重。
游标过期时使用快照和后续事件衔接，检查快照边界处是否有丢失或双重应用。

用固定事件序列测试断线、重复投递、乱序、过期游标和跨会话切换。
比较最终投影与无断线结果，并确认重连没有新增业务命令。
