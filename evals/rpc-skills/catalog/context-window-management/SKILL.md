---
name: context-window-management
description: Design agent context accounting, compaction, and message retention within a model token window. Use for overflowing history, summary duplication, and tool-call/result pairing during truncation, not database recovery of RPC commands.
---

# 上下文窗口管理

区分模型实际计量、发送前估算和为输出预留的空间；展示各部分来源，不能把估算显示成精确账单。
先检查重复注入的历史、指导和摘要，再考虑截断。

保留消息依赖关系：tool call 与对应结果不能被任意拆开，当前任务约束和未完成操作不能只留下结果。
压缩后要标记摘要覆盖范围，防止摘要与原文同时重复进入下一次请求。

用固定消息序列验证窗口边界、超大工具结果、恢复后的摘要与原消息去重。
对摘要丢失关键事实与模型拒绝请求分别记录，不通过伪造 usage 或无界扩大窗口掩盖问题。
