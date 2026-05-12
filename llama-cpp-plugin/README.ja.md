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
- chat メソッドでの tool calling
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

## Runner Settings

プラグインの初期化には `llama_cpp.LlamaRunnerSettings` を使います。

主要フィールド:

- `model`: ローカルのモデルパス、またはカンマ区切りのモデルファイル名
- `hf_repo`: `model` を解決する Hugging Face リポジトリ
- `disable_gpu`: GPU offload を無効化
- `threads`: 生成時のスレッド数
- `threads_batch`: prompt / batch 処理のスレッド数
- `ctx_size`: コンテキストサイズ
- `use_flash_attention`: 対応環境で flash attention を有効化
- `system_prompt`: ランナー既定の system prompt
- `mtmd`: マルチモーダル projector 設定

`hf_repo` を省略した場合、`model` はローカルパスとして扱います。`hf_repo` を指定した場合は、Hugging Face からモデルを取得するか、既存キャッシュを再利用します。

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

## 関連ファイル

- プラグインエントリポイント: `llama-cpp-plugin/src/lib.rs`
- モデル実行本体: `llama-cpp-plugin/src/model.rs`
- 共有 protobuf: `llama-protobuf/`

## ライセンス

MIT License
