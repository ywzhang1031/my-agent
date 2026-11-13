---
name: web-interaction-state
description: Repair agent Web UI pending, disabled, error, focus, and session-switch interaction states. Use for double-click prevention and stuck controls when backend behavior is correct, not server-side crash recovery or transport replay.
---

# Web 交互状态

先画出按钮从 idle 到 pending、成功、失败或取消的状态转换，检查状态归属于哪个会话和请求。
切换会话后，旧异步响应不能覆盖新会话的输入或状态；不要用全局 disabled 冻结无关操作。

防重复点击需要覆盖鼠标、Enter 和触摸入口；失败后恢复可操作状态，并保留用户输入与焦点。
前端防抖不能承担后端副作用去重，也不能把错误状态伪装成成功。

在请求未完成、响应失败、取消和快速切换会话时验证交互，检查键盘焦点、可访问名称和窄屏布局。
使用可控的延迟响应测试，不依赖真实 Provider 的速度。
