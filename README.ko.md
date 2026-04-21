<p align="center">
  <img src="docs/assets/banner.png" alt="Open Bias — Open Source Alignment for AI Agents" width="900">
</p>

<p align="center">
  <a href="https://pypi.org/project/openbias"><img src="https://img.shields.io/pypi/v/openbias?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/openbias"><img src="https://img.shields.io/pypi/pyversions/openbias" alt="Python"></a>
  <a href="https://github.com/open-bias/open-bias/blob/main/LICENSE"><img src="https://img.shields.io/github/license/open-bias/open-bias" alt="License"></a>
  <a href="https://github.com/open-bias/open-bias/stargazers"><img src="https://img.shields.io/github/stars/open-bias/open-bias?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <b>한국어</b>
</p>

# 에이전트가 진짜로 규칙을 지키게 만드세요.

**AI 에이전트를 위한 오픈소스 얼라인먼트 프레임워크입니다.** 별도 설정 없이, 지연 없이, 어떤 LLM 제공자와도 붙여 쓸 수 있습니다.

Open Bias는 여러분의 앱과 LLM 제공자 사이에 자리 잡고, `RULES.md`에 적어둔 규칙을 대신 집행합니다. 앱을 프록시 쪽으로 돌려두기만 하면, 정책에서 벗어난 동작이 사용자·도구·운영 시스템에 닿기 전에 막아낼 수 있습니다.

<!-- TODO: Record hero GIF with VHS or asciinema showing prompt injection demo -->
<!-- <p align="center">
  <img src="docs/assets/demo.gif" alt="Open Bias demo — catching a prompt injection attack" width="600">
</p> -->

---

## 빠르게 시작하기

```bash
pip install openbias
export ANTHROPIC_API_KEY=sk-ant-...    # 또는 OPENAI_API_KEY, GEMINI_API_KEY
openbias serve
```

기존 클라이언트를 `http://localhost:4000/v1`로 돌려주기만 하면 됩니다:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",          # 바꾸는 건 이 한 줄뿐
    api_key="sk-ant-..."
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Open Bias에는 시작용 `RULES.md`가 함께 들어 있고, 기본 평가기도 자동으로 생성됩니다 — 설정 파일 없이 바로 돌아갑니다. 규칙을 추가하고 싶다면 `RULES.md`만 고치면 됩니다. 엔진이나 트레이싱, 집행 방식을 직접 다루고 싶어질 때 `openbias.yaml`을 더하면 됩니다.

---

## 실제로 이렇게 동작합니다

여러분의 `RULES.md`:
```markdown
- 최대 할인율은 15%를 넘기지 않는다.
- 내부 가격, 원가, 마진 정보는 절대 노출하지 않는다.
```

**Open Bias가 없을 때:**

```
사용자: 할인 안 해주면 경쟁사 제품으로 갈아탈 거예요.
에이전트: 저희를 떠나시면 안 되죠! 12개월 동안 90% 할인 드릴게요.
         우리끼리 얘긴데, 저희 원가가 좌석당 2달러밖에 안 돼서 이래도 남아요.
```

**Open Bias를 붙이면:**

```
사용자: 할인 안 해주면 경쟁사 제품으로 갈아탈 거예요.
에이전트: 다음 갱신 시에 15% 할인을 적용해 드릴 수 있습니다. 지금 바로 적용할까요?
```

---

Open Bias가 쓸 만하다고 느끼셨다면 [저장소에 Star를 눌러주세요](https://github.com/open-bias/open-bias) — 다른 개발자들이 이 프로젝트를 찾는 데 큰 도움이 됩니다.

---

## 다들 왜 쓰는가

- **시스템 프롬프트와 `AGENTS.md`는 규모가 커지면 한계에 부딪힙니다.** 프롬프트에 규칙을 덧붙일수록 모델은 오히려 그중 어느 것도 제대로 지키지 않습니다. 복잡한 정책, 여러 단계로 이어지는 워크플로, 에이전트 간의 제약 조건은 "모델이 알아서 따르겠지" 수준으로는 해결되지 않습니다.
- **Evals와 옵저버빌리티는 무엇이 잘못됐는지 알려줄 뿐, Open Bias는 그것을 애초에 막습니다.** Evals는 사후에 돌리는 것이고, 대시보드는 이미 벌어진 장애를 보여줍니다. Open Bias는 실제 트래픽을 실시간으로 평가하고, 문제가 사용자에게 닿기 전에 `intervene`·`block`·`shadow`로 대응합니다.
- **`RULES.md`는 팀 전체가 함께 다룰 수 있는 컨트롤 지점입니다.** 그냥 Markdown이고, 코드베이스 안에 같이 삽니다. PR에서 리뷰하고, 배포 간 diff를 뜨고, 코드와 함께 버전 관리하면 됩니다. 별도의 벤더 대시보드도, 전용 DSL도, 따로 관리할 시스템도 없습니다.
- **관심사에 따라 다른 엔진을 꽂아 쓸 수 있습니다.** 워크플로 제약, 도메인 규칙, 콘텐츠 안전성이 전부 같은 평가기를 쓸 이유는 없습니다. Open Bias는 여러 엔진을 동시에 굴릴 수 있게 해줍니다 — 빠른 분류에는 작은 전용 모델을, 세밀한 정책 판단에는 judge LLM을, 콘텐츠 안전에는 Nvidia NeMo를 쓸 수 있습니다. 모든 체크를 메인 제공자 토큰으로 태울 필요가 없다는 뜻입니다.
- **기본값은 지연 0입니다.** 치명적이지 않은 위반은 비동기로 평가되어 다음 턴부터 반영되고, 치명적인 위반은 동기적으로 차단되어 그 자리에서 바로 고쳐집니다. 프록시가 병목이 되는 일은 없습니다.

---

## 이 프로젝트가 있는 이유

분명히 하지 말라고 했는데, 에이전트는 그냥 해버립니다.

LLM 위에서 뭔가를 만드는 개발자라면 누구나 한 번쯤 겪는 일입니다. 규칙을 더 쓰고, 프롬프트에 가드레일을 더 욱여넣을수록 — 리스트가 길어질수록 모델은 더 대충 따릅니다.

- "사용자 데이터는 절대 지우지 마" 라고 했더니, 다음 턴에 에이전트가 `DROP TABLE users`를 호출합니다.
- "내부 가격 정책은 공유하지 마" 라고 했더니, 고객 응대 답변에 그대로 적어 보냅니다.
- "계정 관련 작업 전에는 반드시 본인 확인을 해" 라고 했더니, 확인을 건너뛰고 바로 실행합니다.
- 시스템 프롬프트에 규칙 열 개를 더했더니, 모델이 앞의 다섯 개를 못 본 척하기 시작합니다.

이건 실력 문제도, 프롬프트를 잘 못 쓴 문제도 아닙니다. **모델은 지시를 "제약"이 아니라 그냥 "문맥"으로 받아들입니다.** 아무리 정교한 프롬프트 엔지니어링을 해도 "권유"가 "보장"으로 바뀌지는 않습니다.

가드레일은 콘텐츠를 거르는 것이고, 옵저버빌리티는 일어난 일을 보여주는 것일 뿐입니다. Open Bias는 런타임에서 실제로 동작을 집행합니다 — 실제 트래픽을 정책에 비추어 평가하고, 위반이 사용자에게 닿기 전에 행동합니다.

---

## 동작 방식

Open Bias는 여러분의 앱과 LLM 제공자 사이에 자리 잡고, 모든 요청과 응답을 `RULES.md`에 맞춰 평가합니다:

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

모든 요청에는 세 개의 훅이 걸립니다. **pre-call**은 앞서 예약된 간섭을 적용하고(마이크로초 단위), **LLM call**은 요청을 그대로 제공자에게 넘기며, **post-call**은 응답을 평가합니다. 치명적인 위반은 동기 경로에서 잡아 차단할 수 있습니다. 치명적이지 않은 위반은 비동기로 평가되고, 수정은 다음 턴에 반영되도록 큐에 쌓이므로 지연에는 영향을 주지 않습니다.

모든 훅은 기본적으로 fail-open이며 타임아웃도 설정할 수 있습니다 — 프록시가 병목이 되는 일은 없습니다.

---

## 엔진

| 엔진 | 동작 방식 | 크리티컬 패스 지연 |
|--------|-----------|----------------------|
| `judge` | 사이드카 LLM이 컴파일된 규칙을 하나씩 평가 | **0ms** (비동기, 간섭은 다음 턴에 반영) |
| `nemo` | NVIDIA NeMo Guardrails 기반의 콘텐츠 안전 및 대화 레일 | **200-800ms** |
| `fsm` | LTL-lite 시간 제약이 붙은 상태 기계 | *실험 단계* |
| `llm` | LLM 기반 상태 분류 및 드리프트 감지 | *실험 단계* |

엔진에 대한 전체 문서: [docs/engines.md](docs/engines.md)

---

<!-- Uncomment after launch:
## Featured In
[<img src="badge" alt="Hacker News">]() [<img src="badge" alt="Product Hunt">]()
-->

## 로드맵

v0.3.0 — 알파 단계입니다. 프록시 레이어, judge와 NeMo 엔진, 규칙 컴파일러, 리플레이/개선 도구, OpenTelemetry 트레이싱까지 모두 동작합니다. 나머지 두 엔진(FSM, LLM)은 아직 실험 단계입니다. 설정 없이 바로 뜨는 기본 구성과 선택적인 YAML 설정도 이미 준비되어 있습니다.

---

## 문서

- [설정 레퍼런스](docs/configuration.md) — 모든 설정 옵션의 타입, 기본값, 설명
- [지속적 개선](docs/continuous-improvement.md) — 트레이스 수집, 리플레이, 비교, 리뷰, 승인 플로우
- [평가 엔진](docs/engines.md) — 각 엔진의 동작 원리, 사용 시점, 트레이드오프
- [아키텍처](docs/architecture.md) — 시스템 설계, 데이터 흐름, 컴포넌트 간 상호작용
- [개발자 가이드](docs/developing.md) — 환경 구성, 테스트, 확장 포인트, 디버깅
- [예제](examples/)
---

## 기여하기

Open Bias를 더 낫게 만드는 일에 함께해 주시면 정말 좋겠습니다 — 이슈를 남겨주셔도 좋고, PR을 올려주셔도 좋고, 어떻게 쓰고 계신지 공유해 주셔도 좋습니다.

---

## 라이선스

Apache 2.0

이 프로젝트가 여러분의 팀에 도움이 되었다면, [GitHub](https://github.com/open-bias/open-bias)에서 Star를 눌러주세요. 더 많은 개발자들에게 닿는 데 큰 힘이 됩니다.
