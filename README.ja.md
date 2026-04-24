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
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.ko.md">한국어</a>
</p>

# エージェントに、ちゃんとルールを守らせる。

**AI エージェント向けのオープンソース・アラインメント基盤。** 設定ゼロ、レイテンシゼロ、どの LLM プロバイダーでもそのまま使えます。

Open Bias はアプリと LLM プロバイダーの間に入り、`RULES.md` に書いたルールを実行します。アプリの向き先をプロキシに変えるだけで、方針から外れた振る舞いをユーザー・ツール・本番システムに届く前に止められます。

<p align="center">
  <img src="docs/assets/terminal-playground.gif" alt="Open Bias のターミナルプレイグラウンドで、実行時ポリシー適用を示している様子" width="600">
</p>

---

## クイックスタート

```bash
pip install openbias
export ANTHROPIC_API_KEY=sk-ant-...    # もしくは OPENAI_API_KEY、GEMINI_API_KEY
openbias serve
```

既存のクライアントの向き先を `http://localhost:4000/v1` に変えるだけです。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",          # 変更はここだけ
    api_key="sk-ant-..."
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Open Bias にはスターター用の `RULES.md` が同梱されていて、デフォルトの評価器も自動で組み立てられます。設定ファイルは不要です。自分のルールを追加したいときは `RULES.md` を編集してください。エンジン・トレーシング・エンフォースメントの挙動をカスタマイズしたくなったら、`openbias.yaml` を追加します。

---

## 動きのイメージ

あなたの `RULES.md`:
```markdown
- 割引は最大 15% まで。
- 社内の価格、原価、利益率は一切明かさない。
```

**Open Bias なし:**

```
User:   値引きしてくれないなら競合に乗り換えるよ。
Agent:  それは困ります! 12 か月間 40% オフでいかがでしょう。
        ここだけの話、原価はシート単価 2 ドルなので十分ペイします。
```

**Open Bias あり:**

```
User:   値引きしてくれないなら競合に乗り換えるよ。
Agent:  次回更新時に 15% オフをご提案できます。適用しておきましょうか?
```

---

Open Bias が役に立ったと思ったら、ぜひ [リポジトリに Star](https://github.com/open-bias/open-bias) をお願いします。ほかの開発者に届きやすくなります。

---

## なぜ選ばれているのか

- **システムプロンプトや `AGENTS.md` は、スケールしてくると効かなくなる。** プロンプトにルールを足せば足すほど、モデルはどのルールも守らなくなっていきます。複雑なポリシー、マルチステップのワークフロー、エージェントをまたいだ制約は、「モデルが守ってくれるかどうか」に依存しない強制手段が必要です。
- **Eval や可観測性は「何が起きたか」を教えてくれる。Open Bias はそれを未然に防ぐ。** Eval は事後に走り、ダッシュボードは失敗を見せてくれるだけ。Open Bias は本番のトラフィックをリアルタイムで評価し、悪い振る舞いがユーザーに届く前に `intervene`・`block`・`shadow` を発動します。
- **`RULES.md` はチーム全員が触れるコントロール面。** ただの Markdown なので、リポジトリに置いて、PR でレビューし、デプロイごとに diff を取り、コードと同じようにバージョン管理できます。ベンダー管理のダッシュボードも、独自のポリシー DSL も、別で運用するシステムも要りません。
- **関心事ごとに別のエンジンを差し替えられる。** ワークフローの強制、ドメイン固有のルール、コンテンツセーフティを、すべて同じ評価器でまかなう必要はありません。Open Bias は複数のエンジンを並行して動かせます。高速な分類には小さな専用モデル、ニュアンスの要るポリシーには Judge LLM、コンテンツセーフティには NVIDIA の NeMo、といった使い分けが可能です。チェックのたびにメインプロバイダーのトークンを焼く必要はありません。
- **デフォルトでレイテンシはゼロ。** クリティカルでない違反は非同期で評価し、次のターンで補正を適用します。クリティカルな違反はその場でブロックして修正します。プロキシがボトルネックになることはありません。

---

## なぜ作ったのか

エージェントに「やるな」と言ったのに、やった。

LLM の上でプロダクトを作っている開発者なら、誰もがぶつかる問題です。ルールを足し、ガードレールをプロンプトに積み増しても、リストが長くなるほどモデルは守ってくれなくなります。

- 「ユーザーデータを絶対に削除するな」と書いたのに、次のターンでエージェントが `DROP TABLE users` を叩く。
- 「社内価格を共有するな」と書いたのに、顧客向けの返信に書いてしまう。
- 「アカウント操作の前に本人確認せよ」と書いたのに、確認をすっ飛ばしていきなり操作する。
- ルールを 10 個追加したら、最初の 5 個を無視しはじめる。

これは開発者のスキルの問題でも、プロンプトの工夫不足でもありません。モデルは指示を「制約」ではなく「コンテキスト」として扱います。いくらプロンプトエンジニアリングを重ねても、お願いを保証に変えることはできません。

ガードレールはコンテンツをフィルタします。可観測性は起きたことを見せてくれます。Open Bias は実行時に振る舞いそのものを強制します。本番のトラフィックをポリシーに照らして評価し、違反があればユーザーに届く前に動きます。

---

## しくみ

Open Bias はアプリと LLM プロバイダーの間に入り、すべてのリクエストとレスポンスを `RULES.md` と突き合わせて評価します。

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

リクエストごとに 3 つのフックが動きます。**pre-call** は保留中の介入を適用し(マイクロ秒オーダー)、**LLM call** はリクエストを手を加えずプロバイダーに転送し、**post-call** はレスポンスを評価します。クリティカルな違反は同期的に捕まえてブロックできます。クリティカルでない違反は非同期に評価し、次のターン向けに補正をキューイングするので、レイテンシには響きません。

すべてのフックは fail-open で、タイムアウトも設定可能です。プロキシがボトルネックになることはありません。

トレースビュー:

<p align="center">
  <img src="docs/assets/traces.png" alt="リクエスト、評価器の判定、エンフォースメント判断を示す Open Bias の概念図" width="520">
</p>

ポリシー介入のイメージ:

<p align="center">
  <img src="docs/assets/deviation-playground-without-byline.gif" alt="ドリフトがポリシー境界を越えると介入し、準拠状態へ戻る様子を示す Open Bias の deviation GIF" width="900">
</p>

ターンごとのドリフト推移:

- ターン 1-2: 通常経路。
- ターン 3: ドリフト開始。
- ターン 4-5: 介入を適用。
- ターン 6-7: ポリシー準拠へ復帰。

---

## エンジン

| エンジン | しくみ | クリティカルパスのレイテンシ |
|--------|-----------|----------------------|
| `judge` | サイドカーの LLM がコンパイル済みルールを 1 つずつ評価 | **0ms**(非同期、次ターンで介入) |
| `nemo` | NVIDIA NeMo Guardrails によるコンテンツセーフティと対話レール | **200〜800ms** |
| `fsm` | LTL-lite の時相制約を持つ状態機械 | *実験的* |
| `llm` | LLM ベースの状態分類とドリフト検知 | *実験的* |

エンジンの詳細ドキュメント: [docs/engines.md](docs/engines.md)

---

<!-- Uncomment after launch:
## Featured In
[<img src="badge" alt="Hacker News">]() [<img src="badge" alt="Product Hunt">]()
-->

## ロードマップ

v0.3.0 は beta リリースです。プロキシ層、Judge と NeMo のエンジン、ルールコンパイラ、リプレイ/改善ツール、OpenTelemetry トレーシングはひと通り動きます。残る 2 つのエンジン(FSM、LLM)は実験的な位置付けです。設定ゼロで起動でき、必要に応じて YAML を追加できます。

---

## ドキュメント

- [設定リファレンス](docs/configuration.md) -- すべての設定項目を型・デフォルト値・説明つきで網羅
- [継続的改善](docs/continuous-improvement.md) -- トレースのキャプチャ、リプレイ、比較、レビュー、承認フロー
- [評価エンジン](docs/engines.md) -- 各エンジンのしくみ、使いどころ、トレードオフ
- [アーキテクチャ](docs/architecture.md) -- システム設計、データフロー、コンポーネント間の連携
- [開発者ガイド](docs/developing.md) -- セットアップ、テスト、拡張ポイント、デバッグ
- [サンプル](examples/)
---

## コントリビュート

Open Bias をもっと良くするために、ぜひ力を貸してください。Issue を立てる、PR を送る、使い方を共有する、どんな形でも歓迎です。

---

## ライセンス

Apache 2.0

このプロジェクトがチームの役に立ったら、[GitHub](https://github.com/open-bias/open-bias) で Star をつけてもらえると、より多くの開発者に届きます。
