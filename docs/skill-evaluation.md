# Skill 选用与任务质量：首轮小样本评测

问题不是“能否加载文件”，而是“模型能否选对，以及选对后有没有更好地解决任务”。
使用真实 AgentLoop、上下文与工具实现，不更改产品配置；每个 turn 新建临时工作区。
唯一任务文件是合成的 evidence.txt。评分答案、报告、其他仓库和个人 Skill 均不在可读工作区。

## 冻结设计

- 使用中文任务、英文代码式证据和英文 Skill 描述。4 个正例（2 个边界测试审查、2 个授权逻辑审查），比较 off、manual、auto。
- 2 个负例（普通字段查找），比较 off、auto；不存在“正确的手动 Skill”，不重复做伪对照。
- off 不提供 Skill 目录或加载工具；manual 预先指定正确 Skill，仍提供与 auto 相同的目录和工具；
  auto 不预选。三组共享系统提示、任务文本、证据、read_file、模型及输出限制。
- 每个组合重复两次，共 32 个 turn。固定种子 42，第二次反转同题组顺序以减少顺序偏差。
  这不是关闭缓存的实验，延迟包含等待与缓存影响。不用私有仓库、个人 Skill 或真实对话。

## 指标

- 任务通过：正常完成、确实读取完整证据、最终 JSON 数组和独立答案完全匹配（忽略顺序，不容许重复）。
  选对 Skill 不给任务加分。读文件失败、格式失败和步数触顶算失败，不手动续跑。
- 正例自动精确选用率：尝试的均为预期 Skill，且成功激活它。负例误触发率：曾请求加载任何 Skill。
- 报告正负例分开的通过分子/分母；同题同次配对比较胜/平/负，不以负例数量不同的整体通过率排名。
- 报告请求数、工具次数、输入/输出/缓存命中 tokens、turn 耗时中位数、按官方时段单价估算的人民币费用。
  reasoning 是输出的组成部分，不重复计费。缺缓存拆分按全部未命中估价；整次 usage 缺失即停止，
  标记未知费用，不记作零。真实账单以供应商为准。

## 运行与保护

```bash
.venv/bin/python -m scripts.skill_eval
.venv/bin/python -m unittest discover -s tests -p test_skill_eval.py
```

默认运行仅验证离线评分与调用链。已授权的付费试验仅在 2026-11-06 北京时间 18:05（含）至
23:55（不含）发起请求：

```bash
.venv/bin/python -m scripts.skill_eval --live-date 2026-11-06
```

只向官方 DeepSeek 地址发送请求，固定 deepseek-flash、thinking enabled。每 turn 最多 4 次，
全局最多 128 次 API 请求、每次最多 1536 输出 tokens，不重试、不切模型。每次调用前保守预留费用
（请求 JSON 的 UTF-8 字节数加协议余量作为输入估计），以 ¥1 为运行器预算；这是本地保护，不是供应商账单硬限。
晚间价格依据 [DeepSeek 官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。

本地输出 `.my-agent/evals/skills-2026-11-06-live/`：manifest.json 冻结版本、样本哈希与顺序，
calls.jsonl 逐请求记录预留和实际 usage，trials.jsonl 保留逐题成绩和答案，每个 turn 的 trace 可回放。
report.md 在每个完成的 turn 后更新。输出目录排他创建，即使上次中断也拒绝自动重复消费。
认证、预算、时段或 usage 异常时报告覆盖缺口，不补跑、不外推模型能力。
