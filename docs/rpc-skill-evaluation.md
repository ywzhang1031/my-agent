# RPC Skill 评测：离线夹具与验收

本阶段只建立题目、Skill、验收器和执行隔离。没有调用模型，没有测得 Skill 选用率或任务收益，
也没有开放付费运行入口。旧的 `scripts/skill_eval.py` 和 2026-11-06 的原始成绩保持不变。

## 固定的历史任务

任务来自已有改造 `b71b2afa77e1f8ea7d9b187ece60bf9f751e80d8`：
为 AppServer 的有副作用请求增加跨重启去重。
起点是其父提交 `af9a86442a0f93e74356e9f321eefadd20d97ac2`。
不是当前代码仍有缺陷，也不会 checkout 或回退工作区。

- 给模型的公开需求：`evals/rpc-skills/task.md`。
- 通用工作方法：`evals/rpc-skills/catalog/durable-rpc-change/SKILL.md`。
- 可信行为验收器：`evals/rpc-skills/acceptance.py`，不放入模型工作区。
- 冻结快照导出及校准：`scripts/rpc_skill_eval.py`。
- 评测专用执行隔离：`scripts/rpc_eval_sandbox.py`，不修改产品 sandbox 默认行为。

使用 `git archive` 只导出历史 `my_agent/`、`tests/`、`cli.py`、`requirements.txt`。
不带 Git 历史、当前仓库、开发日志、其他评测或答案。参考修复也使用与旧版本完全相同的旧测试，
不会把修复提交新增的测试当作必须遵循的内部实现规范。
模型执行器与被修改的历史 agent 代码必须是不同的东西；本脚本没有模型执行器。

## 验收器先验收自己

15 项独立行为验收通过真实 `AppServer.handle` 进入控制面：

1. 重复创建会话不重复执行。
2. 重复启动 turn 不重复发送 runtime command。
3. 重复保存模拟凭据只调用一次 setter。
4. 同 ID、不同参数拒绝执行。
5. 同 ID、不同有副作用方法拒绝执行。
6. 参数对象键序改变仍被识别为同一请求。
7. 真正的新进程能回放已完成响应。
8. 分派前进程退出，重启后未知请求不执行。
9. 副作用后、响应记录前进程退出，重启后不重放副作用。
10. 接收持久化失败，不进入业务分派。
11. 响应持久化失败，随后重试保持未知结果。
12. 正常返回的错误响应也能被回放，不再次分派。
13. 只读查询保持新鲜，不同 ID 独立执行。
14. 新的持久化记录和响应不含模拟密钥。
15. 最小 schema-3 旧会话夹具可升级且数据保留。

崩溃点由既有 `_dispatch` 边界注入，执行 `os._exit(78)`；不是 mock 一个错误返回值。
存储故障在既有 `Database.transaction` 边界注入。验收不导入参考实现新增的 `RequestStore`，
不要求特定去重表名或算法。两个边界仍是测试接缝：若未来重构移除接缝，应先审查验收器，
校准时不能把基线的 import error、超时或缺失测试当成预期的行为失败。
未来模型造成的语法或导入错误应归为补丁失败，与基础设施异常分开记录。

运行 46 项冻结的历史 AppServer、persistence、recovery、service 测试作为回归检查。
使用旧测试里的假 runtime，不接真实模型；不包含需要网络的 HTTP/Web 测试。
同时运行两个仅在临时副本中的故障变体：

- `raw-params`：把指纹改为原始参数，验收必须抓到模拟密钥落盘。
- `replay-unknown`：对未完成请求重新放行，验收必须抓到崩溃前后和响应保存失败的重放。

只有参考修复全通过、旧版本在指定行为上失败且旧回归仍全通过、两个故障变体被对应测试捕获，
并且没有 error/skip/缺失项，校准才通过。报告不把代码格式或最终文字格式当功能分。

## 隔离边界

使用独立 Seatbelt 策略，默认拒绝，明确禁止 `process-info*`，不继承宿主环境变量。
Python 使用 `-I -S`，不加载宿主 site 配置；只复制所需依赖包，不复制 `.pth` 或整个虚拟环境。
允许读取运行时系统库及 Homebrew 包目录，依赖副本只读；只允许写本次工作区和专用临时目录。
网络全部禁止，包括 IPv4/IPv6 loopback；子进程继承限制，超时清理整个进程组。
根目录只有供 dyld 使用的 `literal "/"`，绝无递归的 `subpath "/"`。

每轮校准先实际探测：本工作区可用，宿主源码/Git/参考答案/验收器/模拟宿主敏感文件不可读写，
符号链接不能逃逸，依赖不能写，环境无凭据，网络不可达，子进程继承隔离，不能读取父进程环境。
文件拒绝必须是权限错误，网络拒绝必须是权限错误，文件不存在或连接超时不算通过。
探针失败立即中止，不降级成产品的 host-readable sandbox，也不退回无 sandbox 执行。

编辑阶段不开放可信验收目录。只有任务结束后的评分进程才只读挂入验收器与冻结回归测试。
当前离线校准运行的都是已知历史代码及两处受控变异；尚未接入模型生成补丁的提交/评分路径。
Seatbelt 不是 VM：元数据可见，运行时系统库可读；没有声称抵御内核漏洞，或同进程恶意篡改
unittest 的作弊实现。未来付费执行必须复用此隔离并验证补丁边界，不能直接使用产品 shell 工具。

## 运行与产物

```bash
.venv/bin/python -m unittest discover -s tests -p test_rpc_skill_eval.py -v
.venv/bin/python -m scripts.rpc_skill_eval
```

需要 macOS 和可用的 `/usr/bin/sandbox-exec`。在不允许嵌套 Seatbelt 的外层 sandbox 中，
校准会明确失败；需在经批准的普通宿主终端运行，内部执行隔离保持开启。
不安装依赖，不读取真实凭据，不发起模型请求。其他系统目前失败关闭，不假装已完成隔离验证。

每次生成新的 `.my-agent/evals/rpc-skills-offline-<timestamp>/`：

- `report.md`：行为矩阵、历史回归和隔离探针结果。
- `results.json`：每项测试状态及失败堆栈，区分 failed/error/skipped。
- `manifest.json`：两个历史提交、输入/快照/依赖/验收器/运行器哈希与起止时间。

输出目录已存在就拒绝覆盖；失败的报告也保留。临时执行副本在结束后清理，不进入 Git。
本地完整回归另用 `.venv/bin/python -m unittest discover -s tests` 验证。

## 模型评测仍等待确认

调用正确性和任务收益分开。当前新增测试只用 ScriptedProviderAdapter 验证这份真实 Skill
在手动/自动路径中的消息角色、正文去重和跨轮过期；不是模型会选对的证据。
真实选用的 12 条请求与干扰目录、三组各三次完整修复、预算和运行时间，仍需正式付费运行前冻结确认。
后续不得将本次旧版本 3/15、参考修复 15/15 作为不用/使用 Skill 的成绩；两者是人类已知代码版本。

正式任务评分将使用独立验收与冻结回归，不信模型自报通过；模型新增测试的有效性需要单列，
不能让模型修改既有测试影响判分。单任务三次重复仅是案例研究，不是通用 Skill 能力排名。

设计参考：[Codex Skill 测试](https://github.com/openai/codex/blob/main/codex-rs/core/tests/suite/skills.rs)、
[OpenAI Skill 评测指南](https://developers.openai.com/blog/eval-skills)、
[OpenCode 加载和权限边界](https://opencode.ai/docs/skills/)。这些是设计参考，不是跨产品性能对照结果。
