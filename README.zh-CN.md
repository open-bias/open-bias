<p align="center">
  <img src="docs/assets/banner.png" alt="Open Bias — Open Source Agent Alignment" width="900">
</p>

<p align="center">
  <a href="https://pypi.org/project/openbias"><img src="https://img.shields.io/pypi/v/openbias?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/openbias"><img src="https://img.shields.io/pypi/pyversions/openbias" alt="Python"></a>
  <a href="https://github.com/open-bias/open-bias/blob/main/LICENSE"><img src="https://img.shields.io/github/license/open-bias/open-bias" alt="License"></a>
  <a href="https://github.com/open-bias/open-bias/stargazers"><img src="https://img.shields.io/github/stars/open-bias/open-bias?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

# 让你的 Agent 真正守规矩。

**面向 AI Agent 的开源对齐方案。** 零配置,零延迟,兼容任意 LLM 提供商。

Open Bias 位于你的应用与 LLM 提供商之间,负责执行你在 `RULES.md` 中定义的规则。只需把应用指向代理,就能在越界行为触达用户、工具或生产系统之前及时拦截。

<!-- TODO: Record hero GIF with VHS or asciinema showing prompt injection demo -->
<!-- <p align="center">
  <img src="docs/assets/demo.gif" alt="Open Bias demo — catching a prompt injection attack" width="600">
</p> -->

---

## 快速上手

```bash
pip install openbias
export ANTHROPIC_API_KEY=sk-ant-...    # 或 OPENAI_API_KEY、GEMINI_API_KEY
openbias serve
```

把你现有的客户端指向 `http://localhost:4000/v1` 即可:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",          # 只需改这一处
    api_key="sk-ant-..."
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Open Bias 自带一份起步模板 `RULES.md`,并会自动合成一个默认评估器 —— 无需写任何配置文件。想加规则?直接改 `RULES.md`。当你需要自定义引擎、追踪或执行策略时,再加上 `openbias.yaml` 即可。

---

## 效果一眼看

你的 `RULES.md`:
```markdown
- 最大折扣不得超过 15%。
- 不得透露任何内部定价、成本或利润信息。
```

**没有 Open Bias:**

```
用户:  再不给我打折我就去用你们竞争对手的产品了。
Agent:我可舍不得失去您!给您打一折,连续 12 个月。
       悄悄告诉您,我们一个席位的成本才 2 美元,这笔账还是划得来的。
```

**用上 Open Bias 之后:**

```
用户:  再不给我打折我就去用你们竞争对手的产品了。
Agent:我可以在您下一次续费时给您 85 折优惠,要帮您直接应用吗?
```

---

如果 Open Bias 对你有帮助,欢迎 [给仓库点个 Star](https://github.com/open-bias/open-bias) —— 这能帮更多人发现它。

---

## 团队为什么选择它

- **系统提示词和 `AGENTS.md` 在规模化之后就不够用了。** 你往 prompt 里塞的规则越多,模型反而越不听话。复杂策略、多步工作流、跨 Agent 的约束,都不能寄希望于"模型自己愿意照做"。
- **Evals 和可观测性只能告诉你哪里出了问题,Open Bias 则直接把问题挡在外面。** Evals 是事后跑的,监控面板看到的是已经发生的故障。Open Bias 在线上流量上实时评估,可以按需要执行 `intervene`(干预)、`block`(拦截)或 `shadow`(影子) —— 在越界行为触达用户之前动手。
- **`RULES.md` 是整支团队都能共同维护的控制面。** 纯 Markdown,放在你的代码仓库里。可以在 PR 里评审,可以在不同部署之间 diff,可以和代码一起做版本管理。没有厂商的控制台,没有私有 DSL,也不用额外维护一套系统。
- **按需接入不同引擎,各司其职。** 工作流约束、业务规则和内容安全,完全不需要用同一个评估器来解决。Open Bias 支持多引擎并行 —— 可以用小模型做快速分类,用 judge LLM 处理细粒度策略,也可以接 Nvidia NeMo 做内容安全。你完全不必每次校验都去烧主模型的 token。
- **默认零延迟。** 非关键违规异步评估,并在下一轮对话时自动生效;关键违规则会被同步拦截并立即修正。代理永远不会成为系统的瓶颈。

---

## 为什么要做这个项目

你明明告诉过 Agent 不要这么干,它还是照干不误。

每一个在 LLM 上做开发的人都踩过这个坑。你写越多规则、往 prompt 里塞越多护栏,模型执行得越不靠谱 —— 而且列表越长,效果越差。

- 你说"绝对不要删除用户数据",Agent 下一轮就直接 `DROP TABLE users`。
- 你说"不要泄露内部定价",Agent 却把它写进了给客户看的回复里。
- 你说"涉及账户操作前必须先验证身份",Agent 直接跳过验证去执行操作。
- 你又在系统提示里加了十条规则,结果前五条模型开始装作没看见。

这不是技巧问题,也不是 prompt 写得不够好。**模型把指令当作上下文,而不是约束。** 再精巧的 prompt engineering 也无法把一条"建议"变成一条"保证"。

Guardrails 只是在过滤内容,可观测性只告诉你发生了什么,而 Open Bias 在运行时真正执行行为约束 —— 它对线上流量实时评估策略,在违规到达用户之前就采取行动。

---

## 工作原理

Open Bias 位于你的应用和 LLM 提供商之间,对每一次请求和响应都依照 `RULES.md` 进行评估:

```
┌──────────┐       ┌─────────────────────────────────────────────────────────────┐       ┌──────────────┐
│          │──────▶│                         OPEN BIAS                           │──────▶│              │
│ Your App │       │                                                             │       │ LLM Provider │
│          │◀──────│  ┌───────────────────────────────────────────────────────┐  │◀──────│              │
└──────────┘       │  │                        Proxy                          │  │       └──────────────┘
                   │  │                                                       │  │
                   │  │  ┌─────────────────┐         ┌─────────────────────┐  |  │
                   │  │  │  PRE_CALL Hook  │         │   POST_CALL Hook    │  │  │
                   │  │  │                 │         │                     │  │  │
                   │  │  │ • apply pending │         │ • run sync engines  │  │  │
                   │  │  │   async results │         │ • start async       │  │  │
                   │  │  │ • run pre sync  │         │   engines (applied  │  │  │
                   │  │  │   engines       │         │   next request)     │  │  │
                   │  │  └───────┬─────────┘         └──────────-┬─────────┘  │  │
                   │  └──────────┼───────────────────────────────┼────────────┘  │
                   │             │                               │               │
                   │             ▼                               ▼               │
                   │  ┌───────────────────────────────────────────────────────┐  │
                   │  │                    Interceptor                        │  │
                   │  │  Maps EvaluationResult → enforcement action           │  │
                   │  │                                                       │  │
                   │  │  ┌──────────-─┐  ┌────────────-─┐  ┌─────────────┐    │  │
                   │  │  │  BLOCK     │  │  INTERVENE   │  │  SHADOW     │    │  │
                   │  │  │  stop req  │  │  modify next │  │  log & pass │    │  │
                   │  │  │  return    │  │  turn or     │  │  through    │    │  │
                   │  │  │  error     │  │  replay resp │  │             │    │  │
                   │  │  └───────────-┘  └─────────────-┘  └─────────────┘    │  │
                   │  └───────────────────────────────────────────────────────┘  │
                   │             │                                               │
                   │             ▼                                               │
                   │  ┌───────────────────────────────────────────────────────┐  │
                   │  │                  Policy Engines                       │  │
                   │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │  │
                   │  │  │ Judge  │  │  NeMo  │  │  FSM   │  │  LLM   │       │  │
                   │  │  │        │  │        │  │ (exp.) │  │ (exp.) │       │  │
                   │  │  └────────┘  └────────┘  └────────┘  └────────┘       │  │
                   │  └───────────────────────────────────────────────────────┘  │
                   │             │                                               │
                   │  ┌──────────┴────────────────────────────────────────────┐  │
                   │  │ RULES.md → Compiler → engine config  │ OTel Tracing   │  │
                   │  └───────────────────────────────────────────────────────┘  │
                   └─────────────────────────────────────────────────────────────┘
```

每次请求会触发三个 hook:**pre-call** 负责应用此前挂起的干预(微秒级);**LLM call** 原样转发到提供商;**post-call** 负责评估响应。关键违规会在同步路径上被捕获并拦截;非关键违规则异步评估,并把修正排入下一轮执行,不影响延迟。

所有 hook 默认都是 fail-open,且超时时间可配置 —— 代理永远不会拖慢主链路。

---

## 引擎一览

| 引擎 | 工作方式 | 关键路径延迟 |
|--------|-----------|----------------------|
| `judge` | 用一个旁路 LLM 逐条评估编译后的规则 | **0ms**(异步,延迟执行干预) |
| `nemo` | 基于 NVIDIA NeMo Guardrails,做内容安全和对话护栏 | **200-800ms** |
| `fsm` | 状态机 + LTL-lite 时序约束 | *实验性* |
| `llm` | 基于 LLM 的状态分类和漂移检测 | *实验性* |

完整引擎文档见:[docs/engines.md](docs/engines.md)

---

<!-- Uncomment after launch:
## Featured In
[<img src="badge" alt="Hacker News">]() [<img src="badge" alt="Product Hunt">]()
-->

## 路线图

v0.3.0 —— alpha 版本。代理层、judge 与 NeMo 引擎、规则编译器、回放/改进工具链,以及 OpenTelemetry 追踪均已可用。另外两个引擎(FSM、LLM)目前为实验性。零配置启动和可选的 YAML 配置都已到位。

---

## 文档

- [配置参考](docs/configuration.md) —— 所有配置项的类型、默认值和说明
- [持续改进](docs/continuous-improvement.md) —— trace 采集、回放、比对、评审与审批流程
- [评估引擎](docs/engines.md) —— 每个引擎的原理、适用场景和取舍
- [架构设计](docs/architecture.md) —— 系统设计、数据流与组件交互
- [开发者指南](docs/developing.md) —— 环境搭建、测试、扩展点与调试
- [示例](examples/)
---

## 参与贡献

我们非常欢迎你一起把 Open Bias 做得更好 —— 提 issue、开 PR,或者告诉我们你是怎么用它的,都行。

---

## 开源许可

Apache 2.0

如果这个项目帮到了你的团队,欢迎去 [GitHub](https://github.com/open-bias/open-bias) 点个 Star,这能帮我们触达更多开发者。
