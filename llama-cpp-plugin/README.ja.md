# llama-cpp-plugin

[English version](README.md)

`llama-cpp-plugin` は、`llama.cpp` をバックエンドとして利用する JobworkerP 向けマルチメソッドプラグインランナーです。

## 概要

このcrateは、GGUF モデルを `llama.cpp` 経由で読み込み、JobworkerP のプラグインインターフェースへ 3 つの実行メソッドを公開します。

- `run`: このプラグイン独自の `llama_cpp.LlamaArg` を使う prompt ベース生成
- `chat`: `jobworkerp.runner.llm.LLMChatArgs` を使う chat completion 互換の生成
- `completion`: `jobworkerp.runner.llm.LLMCompletionArgs` を使う単一プロンプトの text completion

このcrateが使う protobuf 定義は `llama-protobuf/` にあり、共通処理は `mtmd-support/` や `modules/jobworkerp-client/` などのcrateにあります。

## 主な機能

- ローカル GGUF ファイル、または Hugging Face から取得したモデルの読み込み
- サンプリング設定付きのテキスト生成
- 複数ターンの chat 形式リクエスト
- jobworkerp Ollama / GenAI completion API 互換の単一プロンプト completion
- chat メソッドでの OpenAI 互換 client-side tool calling (詳細: [docs/client-tool-calling.ja.md](docs/client-tool-calling.ja.md))
- JSON Schema 制約による structured output
- `MtmdSettings` と media input を使ったマルチモーダル入力
- JobworkerP 連携のためのメタデータおよび JSON Schema 公開

## メソッド

### `run`

`run` メソッドは、このプラグイン独自の protobuf 引数形式である `llama_cpp.LlamaArg` を受け取り、生成結果のテキストを `prompt` に書き戻した同じメッセージ型を返します。単純な prompt 入力 / text 出力のワークフローに向いています。

関連 protobuf:

- settings: `llama-protobuf/protobuf/llama_cpp/llama_cpp_runner.proto`
- args/result: `llama-protobuf/protobuf/llama_cpp/llama_cpp_arg.proto`

### `chat`

`chat` メソッドは `jobworkerp.runner.llm.LLMChatArgs` を受け取り、`jobworkerp.runner.llm.LLMChatResult` を返します。このメソッドでは以下を扱えます。

- 複数ターンの会話メッセージ
- system / user / assistant / tool ロール
- tool call の生成と実行委譲
- `json_schema` による structured output
- モデル対応時のマルチモーダル message content

関連 protobuf:

- args: `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_args.proto`
- result: `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_result.proto`

### `completion`

`completion` メソッドは `jobworkerp.runner.llm.LLMCompletionArgs` を受け取り、`jobworkerp.runner.llm.LLMCompletionResult` を返します。単一ターンの「プロンプトの続きを生成する」用途を想定しており、jobworkerp の Ollama / GenAI completion API とワイヤ互換です。

multimodal 入力は本メソッドのスコープ外で、必要な場合は `chat` メソッドを使ってください。

jobworkerp 標準 completion runner (Ollama / GenAI) との互換性差分:

| 項目 | jobworkerp 標準 | 本プラグイン |
|---|---|---|
| `output_type` | `Both`(streaming + non-streaming) | `NonStreaming` |
| `context.ollama_context` | KV キャッシュとして保持 | warn を出して捨てる(multi-turn は `chat` を利用) |
| `model` フィールド | リクエストごとに切替可 | load 時固定。warn を出して無視 |
| `function_options.use_function_calling = true` | 対応 | エラーで拒否 |
| マルチモーダル入力 | completion 自体は両者とも text-only(chat 側で対応) | text-only |

関連 protobuf:

- args: `llama-protobuf/protobuf/jobworkerp/runner/llm/completion_args.proto`
- result: `llama-protobuf/protobuf/jobworkerp/runner/llm/completion_result.proto`

## キャンセル

`chat` / `completion` の streaming 実行では、JobworkerP の cancel token を共有 cancel flag に変換し、sink と `llama.cpp` の abort callback の両方で監視します。sink はトークン境界で停止し、abort callback は長い prompt prefill や `llama_decode` 中のキャンセルを backend が対応する粒度で中断します。cancel が観測されると、プラグインは部分 KV キャッシュを書き戻さずにキャンセル扱いで終了します。

## Runner Settings

プラグインの初期化には `llama_cpp.LlamaRunnerSettings` を使います。

主要フィールド:

- `model`: ローカルのモデルパス、またはカンマ区切りのモデルファイル名
- `hf_repo`: `model` を解決する Hugging Face リポジトリ
- `disable_gpu`: GPU offload を無効化
- `threads`: 生成時のスレッド数
- `threads_batch`: prompt / batch 処理のスレッド数
- `ctx_size`: コンテキストサイズ
- `n_batch`: prompt 処理の論理バッチサイズ。プロンプト評価 (TTFT) に影響し、トークン毎の生成速度そのものには影響しません。**未指定時は実効コンテキスト長 (`ctx_size`、未指定ならモデルの学習時コンテキスト長) に揃えます**。これによりプロンプト全体を 1 回の decode で処理できます。
- `n_ubatch`: 物理マイクロバッチサイズ。`n_batch` より小さくすると prompt 評価時のピークメモリを抑えられます。デフォルト挙動はバックエンド依存です。**Metal / ROCm ビルド時**は速度重視のため実効 `n_batch` に追従させますが、計算バッファ肥大を防ぐため上限 2048 で頭打ちにします (例: 実効 n_batch が 262144 でも n_ubatch は 2048)。それ以外のバックエンドは llama.cpp 既定値 (512) のままです。いずれも明示指定すれば常にその値が優先されます。
- `type_k`: KV キャッシュの K 側データ型。未指定時は llama.cpp 既定値 (F16)。`KV_CACHE_TYPE_Q8_0` 等に量子化すると長コンテキストの KV キャッシュ使用量を削減できます。
- `type_v`: KV キャッシュの V 側データ型。未指定時は llama.cpp 既定値 (F16)。**V 側の量子化は通常 flash attention が必須**です(`use_flash_attention` が無効だと warn を出します)。
- `reuse_kv_prefix`: リクエスト間で共通する先頭(システムプロンプト+文書、または同一画像など)の KV を残し、差分だけ prefill します。テキストはトークン単位、画像は内容ハッシュで同一性を判定し、共通する chunk を再利用します(同一画像+異なる質問の連続では画像の encode/decode もスキップ)。共通文脈を更新しながら連続リクエストするワークロードで TTFT を大幅短縮できます。デフォルト false では毎回 KV を全消去し、リクエストは完全独立・決定的です。
- `use_flash_attention`: 対応環境で flash attention を有効化
- `system_prompt`: ランナー既定の system prompt
- `mtmd`: マルチモーダル projector 設定

`hf_repo` を省略した場合、`model` はローカルパスとして扱います。`hf_repo` を指定した場合は、Hugging Face からモデルを取得するか、既存キャッシュを再利用します。

## 推論速度の計測と切り分け

macOS (Metal) などで生成速度 (token/sec) が想定より遅い場合、まず原因を計測で切り分けてください。

### 計測方法

プラグイン経由 (`run` / `chat` / `completion`) で実行すると、`RUST_LOG=info` 時に以下のログが出力されます。

- `context created in N s (n_batch=..., n_ubatch=...)`: リクエストごとの `LlamaContext` 生成 (KV キャッシュ確保) にかかった時間。Metal では毎リクエストでこのコストが発生します。
- `decoded N tokens in N s, speed N t/s`: 実際のトークン生成速度。
- `ctx.timings()`: llama.cpp が計測した prompt eval (プロンプト評価) と eval (生成) の内訳。

### 切り分けの指針

- **context 生成時間が支配的**: リクエストごとに新しい `LlamaContext` を生成しているため。短いプロンプト / 短い生成を繰り返す用途では特に顕著です。
- **eval (生成) の token/sec 自体が低い**: Metal が実際に使われているか確認してください。`--features metal` 付きでビルドし、`disable_gpu` が `false` であることが前提です。
- **prompt eval (TTFT) が遅い**: `n_batch` / `n_ubatch` の調整が効く余地があります。ただし既定値 (n_batch=2048) で十分大きいため、効果は限定的です。

### ollama との比較

同一モデル・同一プロンプトで `ollama run --verbose` を実行し、`eval rate` (token/sec) を本プラグインの `speed N t/s` と並べて比較すると、差がどのフェーズで生じているか判断しやすくなります。

> 注意: `cargo run --bin sample` はプラグインとは別の独自コンテキスト構築・生成ループを使っており、上記のプラグイン用ログは出力されません。プラグインの挙動を計測したい場合は JobworkerP 経由で実行してください。

### 超ロングコンテキスト (数万〜数十万 token) 向けチューニング

巨大なコンテキスト (例: ctx_size=20万、入力数万 token) を Apple Silicon の unified memory で扱う場合、ボトルネックは prompt eval (大量の入力トークンの処理) と KV キャッシュのメモリ帯域に移ります。以下が効きます。

- **`n_ubatch` を大きくする (例: 2048)**: 数万 token の入力は `n_ubatch` 単位のチャンクに分割して処理されます。`n_ubatch` を上げるとチャンク分割が減り、prefill が速くなります。Apple も大プロンプト処理で `-ub 2048` を推奨しています。**Metal / ROCm ビルドでは `n_ubatch` 未指定時に実効 `n_batch` へ自動追従 (上限 2048)** するため、明示設定しなくても 2048 になります。さらに上げたい場合は `n_ubatch` を明示指定してください (上限を超えられます)。代償は計算バッファのメモリ増ですが、unified memory に余裕があれば許容範囲です。
- **KV キャッシュを量子化する (`type_k` / `type_v` = `KV_CACHE_TYPE_Q8_0`)**: 長コンテキストでは KV キャッシュが巨大になります。F16 → Q8_0 でフットプリントが半減し、メモリ帯域も削減できます。`type_v` の量子化には flash attention が必要です (本プラグインは既定で有効)。
- **`reuse_kv_prefix=true` で共通 prefix を再利用する**: システムプロンプトや文書が共通で末尾だけ変わる連続リクエストでは、共通部分の KV を残して差分だけ prefill します。prompt eval(数万トークンの再計算)をほぼスキップでき、TTFT が激減します。最大の効果はこのワークロードで得られます。リクエストを完全独立にしたい場合は false(デフォルト)のままにしてください。
- **注意: `n_ubatch` は「大きいほど速い」ではありません**。GPU のキャッシュ特性に依存し、ある値で急に速くなったり遅くなったりします。M 系チップ・対象モデルでの最適値は `llama-bench` で 512 / 1024 / 2048 / 4096 をスイープして決めてください。`n_ubatch` ≤ `n_batch` の制約があります。

## ビルド

```bash
cargo build -p jobworkerp-llama-cpp-plugin
```

リリースビルド:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin
```

CUDA ビルド:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin --features cuda
```

Metal ビルド:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin --features metal
```

ROCm (AMD GPU) ビルド:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin --features rocm
```

## テストと静的解析

```bash
cargo test -p jobworkerp-llama-cpp-plugin -- --test-threads=1
cargo fmt --all --check
cargo clippy -p jobworkerp-llama-cpp-plugin --all-targets
```

共有 protobuf や共有サポート crateへ影響する変更では、ワークスペース全体の検証も実行してください。

## CLI サンプル

このcrateにはローカル検証用の sample バイナリも含まれます。

```bash
cargo run -p jobworkerp-llama-cpp-plugin --bin sample -- \
  hf-model TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf \
  --prompt "Hello"
```

この sample バイナリは `llama-cpp-2` のローカル実験用です。JobworkerP が読み込むプラグインエントリポイントとは別物です。

## 補足

- エクスポートされるプラグイン名は `LLMPromptRunner` です。この名前を変えると、すでにこの名前を参照しているシステムが壊れる可能性があります。
- 2 つのメソッドはいずれも現在は non-streaming API として公開されています。
- media input は `llama_cpp.MediaInput` で渡され、prompt 中の marker と添付 media の並びが一致している必要があります。
- manual mode の tool calling では、最終的な assistant 応答の代わりに pending tool calls が chat result に入る場合があります。
- OpenAI 互換 `tools: []` を渡して client-side でツール実行する詳細フロー (proto スキーマ、tool_choice / parallel_tool_calls / reasoning_format / chat_template_kwargs、streaming partial 規約、エラーケース) は [docs/client-tool-calling.ja.md](docs/client-tool-calling.ja.md) を参照してください。

## 関連ファイル

- プラグインエントリポイント: `llama-cpp-plugin/src/lib.rs`
- モデル実行本体: `llama-cpp-plugin/src/model.rs`
- 共有 protobuf: `llama-protobuf/`
- ドキュメント: `llama-cpp-plugin/docs/`

## ライセンス

MIT License
