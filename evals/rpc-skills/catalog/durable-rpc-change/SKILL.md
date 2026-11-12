---
name: durable-rpc-change
description: Implement or review server-side RPC idempotency and durable request recovery across retries and process restarts. Use for duplicate side effects and unknown outcomes, not provider HTTP backoff, UI debounce, or event-stream reconnection alone.
---

# 持久化 RPC 修改

先沿控制请求入口找到身份校验、业务分派、持久化和返回响应的位置。保持 Provider、
控制面与执行器的边界；不要用调整模型重试来修补服务端重复执行。

用一张简短状态表列出每个边界上的事实：哪些信息已经提交，哪些副作用可能发生，
崩溃后下一次请求能根据什么证据行动。区分“尚未接收”“已接收但结果未知”和“已完成”；
进程内缓存或内存锁本身不能提供跨重启的证据。

逐项推演相同身份重试、身份冲突、分派前保存失败、分派后保存失败和重启。把失败注入在
实际边界，而不是让 mock 直接返回期望结果；崩溃恢复至少用一次真正的新进程检查。
对未知结果保持保守，不把“没保存响应”推断成“副作用没发生”。

复用项目已有的事务和错误封装，避免重复实现存储层。审查新增持久化字段是否包含敏感参数，
并检查只读请求和已有数据库的行为是否改变。不要把本 Skill 当作放宽权限或扩大任务范围的授权。

回归测试先证明旧行为的缺陷，再检查修复后的行为和未修改功能。结束时区分运行过的测试、
仅推理过的情况和仍不确定的恢复边界；测试条数不能代替对行为的说明。
