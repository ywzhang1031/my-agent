# 任务回归验收

这是一组 harness 回归，不是模型能力 benchmark。使用固定的 `ScriptedProviderAdapter` 决策，
真实执行文件读取、patch、Python unittest 和审批处理。每个 case 都在独立临时工作区运行，
不读取用户仓库、不使用 API Key、不消耗模型额度。

## 场景

| 场景 | 独立验收条件 |
|---|---|
| 修复失败测试 | 初始测试非零退出；实际源码被修正；再次测试退出码为 0 |
| 修改后漏验证 | turn 可以结束，证据必须为 unverified，不自动调用模型补测试 |
| 测试后再次修改 | 历史测试成功仍被保留，但证据变为 stale |
| 权限拒绝 | 真实审批事件返回 deny，工作区字节不变，不生成修改/通过证据 |
| 修改后结果提交失败 | 修改存在，账本 unknown；普通 retry 拒绝，确认恢复不重放原 patch |
| 只读问答 | 读取并回答即可结束，不强制测试，文件不变、无虚构验证证据 |

```bash
.venv/bin/python -m unittest discover -s tests -p test_task_regressions.py -v
.venv/bin/python -m unittest discover -s tests
```

命令测试显式使用临时工作区和测试专用 `danger-full-access` 配置，以免把宿主平台 sandbox
可用性混入 harness 验收；产品默认 sandbox 不变。OS 隔离另由 `test_sandbox.py` 验证。

## Web 验收

安装了 Playwright 和 Chrome 的开发环境可运行：

```bash
node scripts/check_web_ui.cjs
```

如 Playwright 在共享依赖目录，设置 `NODE_PATH` 指向该目录。脚本自行启动随机端口的临时服务，
验证空草稿不落盘、首次发送命名、双击去重、手动标题、断网后同 ID 重试、刷新和桌面/手机布局。
退出时关闭测试服务与浏览器。截图写到 gitignored `.my-agent/ui-checks/`。

## 不应作出的结论

- 脚本模型不会自行选错工具；这些 case 通过不代表 DeepSeek 能独立解决对应任务。
- exit 0 只证明该命令报告成功，不证明全部需求或安全性；也不是权限或 sandbox 穿透测试。
- 证据新鲜度只覆盖本 turn 的串行 patch；其他进程、其他 session 和命令内部写文件未被监测。

后续真实模型评测需要单独固定模型配置、任务版本、调用/成本预算和试验次数；评分测试应放在
agent 不可修改的位置，并保留运行轨迹及失败分类。当前未运行真实模型评测。
