# クライアント側ツール呼び出し (client-side tool calling)

llama-cpp-plugin の `chat` メソッドは、OpenAI 互換の `tools: []` 定義をクライアントから受け取り、モデルが生成したツール呼び出しをパースして返す経路をサポートしています。サーバ側でのツール自動実行 (`use_function_calling=true`) は本プラグインでは引き続き拒否されます。

本ドキュメントの対象読者は、jobworkerp 経由で本プラグインを呼び出すサービス / ツール開発者です。

## 全体フロー

```
クライアント                            jobworkerp + llama-cpp-plugin
-----------                            -----------------------------
1. LlmChatArgs を組み立てて送信
   - function_options.client_tools_json   ────>  chat メソッド
   - messages: 通常通り                        対応済みテンプレ + 文法を Rust で適用
                                               モデル生成 → Rust parser で抽出
                                          <──── LlmChatResult
                                                - pending_tool_calls
                                                - requires_tool_execution=true
                                                - content: MessageContent::ToolCalls

2. 受信した pending_tool_calls を
   クライアント側で実行

3. 結果を ROLE=TOOL の
   ToolResults として継続送信            ────>  chat メソッド (次ターン)
                                          <──── LlmChatResult
                                                - content: MessageContent::Text (最終回答)
```

streaming (`begin_stream` / `receive_stream`) の場合も同じ意味論で動作します。詳細は後述「ストリーミング」セクションを参照。

## 入力 (LlmChatArgs)

proto 定義: `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_args.proto`

### `function_options.client_tools_json` (tag 7, optional string)

OpenAI Chat Completions API がそのまま受理する `tools` 配列の JSON 文字列。

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather in a given city.",
      "parameters": {
        "type": "object",
        "properties": {
          "city": { "type": "string", "description": "City name, e.g. Tokyo" }
        },
        "required": ["city"]
      }
    }
  }
]
```

このフィールドが `Some` の場合、本プラグインはクライアント側ツール実行モードに切り替わります。

### `function_options.tool_choice` (tag 8, optional string)

OpenAI 互換の `tool_choice`。受理する値:

| 値 | 振る舞い |
|---|---|
| 省略 (`None`) | `"auto"` と同じ |
| `"auto"` | モデルが必要と判断したときだけツールを呼ぶ (lazy grammar) |
| `"none"` | ツール呼び出しを抑制 |
| `"required"` | ツール呼び出しを必須化 (eager grammar) |
| `{"type":"function","function":{"name":"<n>"}}` | 指定された関数を必ず呼ぶ (filter + `"required"` に変換される) |

JSON object 形式で指定された場合、本プラグインは `client_tools_json` を該当関数 1 件にフィルタしたうえで `"required"` を渡します。存在しない関数名はエラーになります。

### `function_options.parallel_tool_calls` (tag 9, optional bool, default false)

OpenAI 互換。`true` のとき、モデルは 1 回の応答内に複数のツール呼び出しを発行できます。streaming で interleave される場合の処理については後述。

### `function_options.reasoning_format` (tag 10, optional string)

`"deepseek"` などのヒント。クライアント側ツール呼び出しの Rust 経路では、現時点で Qwen tagged / Gemma4 の専用 parser が template format に基づいて reasoning を分離します。この値は未対応テンプレートを legacy parser へ送るためには使われません。

### `function_options.chat_template_kwargs` (tag 11, optional string)

任意の JSON object 文字列。chat template (jinja) に渡される追加の `kwargs`。モデル固有のスイッチ (例: Qwen の `{"enable_thinking":false}`) を渡すための入口です。`enable_thinking` キーは特別扱いで、Rust renderer と Rust parser が同じ値を見るように反映されます。これは jinja テンプレートと grammar/parser が同じ前提で動くようにするためで、二者が食い違うと Qwen 系のような think モードを持つモデルで tool call の生成・解析が不整合になります。値が bool でないか JSON object でない場合は無視され、デフォルト (`false`) が使われます。

### 既存フィールドとの排他

| 組み合わせ | 結果 |
|---|---|
| `client_tools_json` + `use_function_calling=true` | エラー: "client_tools_json and use_function_calling are mutually exclusive" |
| `client_tools_json` + `LLMChatArgs.json_schema` | エラー: "json_schema and client_tools_json are mutually exclusive" (文法スロットの競合) |
| `client_tools_json` + multimodal (`Image` content) | エラー: "multimodal input combined with client_tools_json is not supported yet" |

非ストリーミング・ストリーミングともに同じバリデーションが `run_chat_with_sink_tools` の入口で適用されます。

## 出力 (LlmChatResult)

proto 定義: `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_result.proto`

ツール呼び出しが発行されたとき、本プラグインは以下を埋めて返します。

```text
LlmChatResult {
  content: MessageContent {
    content: ToolCalls(ToolCalls {
      calls: [
        ToolCall {
          call_id:      "call_xxx...",    // モデル / パーサ生成 (空のときはサーバ側で補完)
          fn_name:      "get_weather",
          fn_arguments: "{\"city\":\"Tokyo\"}",  // OAI 仕様で JSON string
          delta_index:  Some(0),          // streaming partial 時のみ
        },
      ],
    }),
  },
  pending_tool_calls: Some(PendingToolCalls {
    calls: [ToolCallRequest { call_id, fn_name, fn_arguments }, ...],
  }),
  requires_tool_execution: Some(true),
  reasoning_content: Some("..."),         // reasoning_format を指定した場合
  usage:             Some(Usage { ... }),
  done:              true,
}
```

クライアントは原則として `pending_tool_calls` (= canonical な集約結果) を信頼すれば十分です。`content.ToolCalls` はストリーミングでの逐次表示用です。

> 上記は **非ストリーミング** (`run` 経由) のレスポンス形です。ストリーミング (`run_stream` 経由) では `done=true` のチャンクと `pending_tool_calls` を運ぶチャンクが**別チャンクに分離**されます。詳細は「ストリーミング」セクションを参照してください。

## 多ターン継続 (tool 結果の返却)

ツール実行後、クライアントは結果を `ROLE=TOOL` メッセージの `ToolResults` として次のリクエストに含めます。

```text
LlmChatArgs {
  function_options: Some(FunctionOptions { client_tools_json: Some(...), ... }),
  messages: [
    ChatMessage { role: User,      content: Text("...") },
    ChatMessage { role: Assistant, content: ToolCalls(...) },   // 直前の pending_tool_calls をそのまま入れる
    ChatMessage { role: Tool,      content: ToolResults(ToolResults {
      results: [
        ToolResult {
          call_id:  "call_xxx...",       // assistant.ToolCalls と一致させること
          fn_name:  "get_weather",       // 任意。OAI tool メッセージでは使わないため本プラグインでは無視される
          content:  "{\"temperature_c\":22,\"sky\":\"clear\"}",  // tool 実行結果テキスト
          is_error: false,
        },
      ],
    }) },
  ],
}
```

`ToolResults` 1 メッセージに複数の `ToolResult` を入れると、OAI 経路では配列順に `{"role":"tool","tool_call_id":..,"content":..}` メッセージが fan-out されます (順序は保持されます)。

`ToolResult.call_id` は必須 (空文字はエラー)。`ToolResult.fn_name` は OAI tool メッセージでは利用しないため本プラグインでは無視します (genai 等の他 provider で必要な場合のためにクライアントが埋めて構いません)。`is_error=true` を指定した場合、本プラグインは `content` の先頭に `[ERROR] ` を付けて jinja テンプレートに渡します (OAI / jinja に専用フィールドがないため文字列として伝達します)。

旧 `ToolExecutionRequests` を TOOL ロールで使う非公式拡張は本リリースで廃止されており、リクエストすると移行を促すエラーが返ります。

レスポンスは通常の `MessageContent::Text` (最終回答) になります。モデルが再度ツール呼び出しを必要と判断した場合は再び `pending_tool_calls` が返るため、loop で処理してください。

## ストリーミング (`begin_stream` / `receive_stream`)

`function_options.client_tools_json` が設定されている場合、streaming chunk は OpenAI streaming 仕様の partial accumulation に従います。**ただし tool_call 確定時の wire shape は分離型** (Anthropic / Gemini 流) であり、`done=true` の終端チャンクと `pending_tool_calls` のシグナルチャンクが別チャンクに分かれます (詳細は後述「tool_call 確定時のチャンク分離」)。

### chunk が運ぶフィールド

`LlmChatResult` の `content` は oneof のため、**1 つの chunk は text か tool_calls のどちらかしか運べません**。本プラグインは:

- preface のテキスト (例: "Let me check the weather…") は `MessageContent::Text` の chunk として送信
- 同じパーサ batch に tool_calls が含まれていた場合、続けて `MessageContent::ToolCalls` の chunk として **別 chunk** で送信

の順で送ります。クライアント側でテキストとツール呼び出しの両方が同時に欠落しないよう、両方の chunk を保持してください。

### tool_calls delta の累積規約

各 `ToolCall` チャンクは OpenAI streaming 形式の partial です。

| フィールド | 出現タイミング |
|---|---|
| `delta_index` | 常に設定される (parallel 呼び出しの demux 用) |
| `call_id` | その index の **最初の delta** だけ非空。以降は空文字列 |
| `fn_name` | 同上 |
| `fn_arguments` | 各 delta で chunk 分を連結する |

`parallel_tool_calls=true` のときは複数の `delta_index` が interleave されて届きます。クライアント accumulator は `delta_index` で振り分けてください (`call_id`/`fn_name` の空文字列は「同 index の継続」を意味します)。

なお `MessageContent::ToolCalls` の partial chunk は **UI 用の途中表示** を想定したオプションのストリームです。クライアントが accumulator を持たない場合は、後述の「中間 finalize chunk」が運ぶ canonical な `pending_tool_calls` だけを消費すれば十分です。

### tool_call 確定時のチャンク分離

モデルが tool_call を確定したとき、本プラグインは以下の **2 つの chunk を順に** 送出します。

```text
chunk N  : LlmChatResult {
  done: false,
  content: None,                       // 中間 finalize chunk はペイロードなし
  pending_tool_calls: Some(PendingToolCalls { calls: [...] }),
  requires_tool_execution: Some(true),
  usage: None,
  ..
}                                     // ← (1) 中間 finalize chunk: tool 呼び出し決定のシグナル

chunk N+1: LlmChatResult {
  done: true,
  content: Some(MessageContent { content: Some(Text("")) }),
  pending_tool_calls: None,
  requires_tool_execution: None,
  usage: Some(Usage { prompt_tokens, completion_tokens, ... }),
  ..
}                                     // ← (2) 終端 chunk: ストリーム終端マーカー (Usage のみ)
```

クライアントは:

- **`pending_tool_calls=Some(non_empty)` または `requires_tool_execution=Some(true)`** が見えた時点で「ツール呼び出しが確定した」と判断し、ツール実行の準備に入って構いません (`done` の値で待つ必要はありません)。
- **`done=true`** チャンクは「LLM 側のストリーム生成が完了した」だけを意味します。直前に受信した `pending_tool_calls` の中身は (2) の終端チャンクには載りません。

> ⚠️ **アンチパターン**: `done=true` を待ってから `pending_tool_calls` を読むコードは tool_call を取り逃がします。`pending_tool_calls` / `requires_tool_execution` を検出した時点で確定として扱ってください — `done=true` チャンクは終端マーカーだけで、`pending_tool_calls` を運びません。

### tool_call **無し** のテキスト応答

ツールを呼ばずに通常のテキストで完結する応答は単一の `done=true` チャンクで終端します。

```text
chunk N-1: LlmChatResult { done: false, content: Some(Text(<delta>)), .. }
chunk N  : LlmChatResult { done: true,  content: Some(Text(<最終 text または "">)),
                            pending_tool_calls: None,
                            usage: Some(...), .. }
```

`done=true` チャンクに最終テキストが乗ります。

### collect_stream (内部経路)

`MultiMethodPluginRunner::collect_stream` を使うと、ストリームを内部で完全集約した単一 `LlmChatResult` を取得できます (jobworkerp の `STREAMING_TYPE_INTERNAL` 経路)。`ToolCallAccumulator` が delta を再構成し、**中間 finalize chunk** が運ぶ `pending_tool_calls` を canonical として採用します (= 二重 finalize 防止、`done=true` チャンク側には pending を持たせない)。

## エラーケース

| シナリオ | 返るエラー |
|---|---|
| `client_tools_json` が JSON 配列でない | "client_tools_json is not valid JSON array" |
| `tool_choice` が不明な bare 文字列 (例: `"nope"`) | "unsupported tool_choice ..." |
| `tool_choice` JSON object に `function.name` がない | "tool_choice object must carry function.name" |
| `tool_choice` で指定した関数が tools に存在しない | "tool_choice requested function ... but it is not present in client_tools_json" |
| tool calling 非対応の chat template | "unsupported chat template for Rust tool calling" |
| 排他制約違反 | "...are mutually exclusive" |
| multimodal + tools | "multimodal input combined with client_tools_json is not supported yet" |

streaming 経路では、これらのエラーは `begin_stream` から同期的に返るか、最初の `receive_stream` 呼び出しで `StreamItem::Error` として届きます。

## モデル別の注意

- `enable_thinking` などモデル固有のフラグは **本プラグインからはハードコードしません**。必要なら `chat_template_kwargs` で渡してください。
- tool calling 対応済み template は Qwen tagged format と Gemma4 format です。これらは template rendering、grammar、非ストリーミング parse、streaming delta parse を Rust 実装で処理します。
- 未対応 template で `client_tools_json` を指定した場合は明示エラーになります。旧 fork 専用 OAI parser への fallback は行いません。
- Qwen tagged では think mode の有無により生成テキストの prefix (`<|im_start|>assistant\n<think>...</think>`) が変わるため、Rust parser は `generation_prompt` 相当の prefix を strip してから Qwen tagged body を解析します。

## サンプル (Rust, jobworkerp プラグイン直叩き)

```rust
use jobworkerp_llama_protobuf::protobuf::llm::{
    llm_chat_args::{self, message_content::Content, ChatMessage, ChatRole, FunctionOptions,
                   MessageContent},
    LlmChatArgs, LlmChatResult,
};
use prost::Message;

let tools = r#"[
  {"type":"function","function":{
    "name":"get_weather",
    "description":"Get the current weather in a city.",
    "parameters":{
      "type":"object",
      "properties":{"city":{"type":"string"}},
      "required":["city"]
    }
  }}
]"#;

let request = LlmChatArgs {
    function_options: Some(FunctionOptions {
        client_tools_json: Some(tools.to_string()),
        tool_choice: Some("auto".to_string()),
        ..Default::default()
    }),
    messages: vec![ChatMessage {
        role: ChatRole::User as i32,
        content: Some(MessageContent {
            content: Some(Content::Text(
                "What's the weather in Tokyo?".to_string(),
            )),
        }),
    }],
    ..Default::default()
};

let mut buf = Vec::with_capacity(request.encoded_len());
request.encode(&mut buf).unwrap();
let (res, _meta) = plugin.run(buf, std::collections::HashMap::new(), Some("chat"));
let chat: LlmChatResult =
    LlmChatResult::decode(&mut std::io::Cursor::new(res.unwrap())).unwrap();

assert_eq!(chat.requires_tool_execution, Some(true));
let pending = chat.pending_tool_calls.unwrap();
let call = &pending.calls[0];
// call.fn_name == "get_weather"
// call.fn_arguments  == "{\"city\":\"Tokyo\"}"
// → クライアント側でツール実行、結果を ROLE=TOOL の ToolResults で次ターン
```

ストリーミング版のサンプルは `llama-cpp-plugin/src/lib.rs` の `test_streaming_chat_tool_calls_aggregated` および `test_collect_stream_aggregates_tool_calls_into_final_result` を参照してください (`begin_stream` → `receive_stream` ループ、または `collect_stream` の 2 通り)。

## 関連ファイル

| 用途 | パス |
|---|---|
| 入力 proto | `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_args.proto` |
| 出力 proto | `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_result.proto` |
| エントリ (非 streaming) | `llama-cpp-plugin/src/model.rs::run_chat_with_sink_tools` |
| ストリーミング worker | `llama-cpp-plugin/src/lib.rs::spawn_worker_with_tools` |
| OpenAI 形式 messages / Rust renderer/parser ヘルパ | `llama-cpp-plugin/src/oai_chat.rs` |
| 入力経路の振り分け | `llama-cpp-plugin/src/model.rs::run_chat_with_sink` 冒頭 |
