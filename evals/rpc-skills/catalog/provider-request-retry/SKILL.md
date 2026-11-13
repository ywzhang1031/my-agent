---
name: provider-request-retry
description: Design or debug LLM API transient-error retries, backoff, Retry-After, cancellation, and interrupted streams. Use for provider transport failures, not durable AppServer command deduplication or browser event replay.
---

# 模型请求重试

先区分连接失败、限流、鉴权错误、服务端错误和部分流已交付；保留 Provider 的错误归一化，
由 harness 决定是否重试。不要重试凭据错误，或把部分输出静默拼接成一次完整回答。

把尝试次数、累计等待时间、取消信号和服务端 Retry-After 放进同一预算。
等待应可取消；对抖动退避使用可注入的时钟和随机源测试，不能靠长时间 sleep 验证。

记录每次尝试与最终状态，但不记录 Authorization 或原始密钥。超时并不证明远端没有处理请求；
涉及工具副作用时，应回到该工具的执行证据检查，不能重新执行整个 agent turn 来补偿 HTTP 错误。
