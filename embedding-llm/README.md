# embedding-llm

LLM ベースの embedding 生成プラグイン（llama.cpp 統合版）

## 概要

`embedding-llm`は、llama.cpp を使用して LLM（Large Language Model）から embedding ベクトルを生成する JobworkerP プラグインです。従来の BERT ベースの実装とは異なり、LLM の最終層の出力を正規化して embedding として返します。
文章が長い場合には、スライディングウィンドウを使用して分割し、各セグメントの embedding を生成して返します。

### 主な特徴

- **hidden states を用いた mebedding vecotor 取得**: llama.cpp の`llama_get_embeddings()`を使用した直接アクセス
- **GGUF モデル対応**: embedding 専用 GGUF モデルの完全サポート
- **動的 instruction**: 実行時に instruction prefix を変更可能
- **オプション正規化**: 実行時に L2 正規化の有無を選択可能
- **スライディングウィンドウ**: 長文処理のためのウィンドウ分割機能
- **メモリ効率**: llama.cpp の最適化されたメモリ管理を活用

## アーキテクチャ

### コンポーネント構成

```
embedding-llm/
├── src/
│   ├── lib.rs              # PluginRunnerインタフェース実装
│   ├── embedding.rs        # メインのembedding生成器
│   ├── llamacpp_bridge.rs  # llama.cppとの統合レイヤー
│   ├── tokenization.rs     # 独立したトークナイザー処理
│   ├── sliding_window.rs   # 長文処理用スライディングウィンドウ
│   └── error.rs            # エラー定義
├── protobuf/
│   ├── llm_runner_settings.proto  # 初期化設定
│   ├── embedding_args.proto       # 実行時引数
│   └── llm_result.proto           # 出力形式
├── build.rs                       # protobufビルドスクリプト
└── Cargo.toml                     # 依存関係定義
```

### 処理フロー

1. **モデル初期化**: Settings 使用（GGUF 形式必須）
2. **実行時引数受け取り**: Args 使用（text, instruction, normalize）
3. **テキスト前処理**: instruction prefix 追加
4. **LLM 推論実行**: llama.cpp による embedding 生成
5. **最終層出力取得・正規化**: hidden states, L2 正規化（オプション）
6. **embedding vector 返却**: protobuf 形式で結果返却

## 使用例

### Settings（初期化時） - GGUF 形式

```rust
use embedding_llm::protobuf::embedding_llm::*;

let settings = EmbeddingLlmRunnerSettings {
    model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
    use_cpu: true,
    dtype: "f32".to_string(),
    max_seq_length: 512,
    model_type: ModelType::Gguf as i32,
    model_files: vec!["Qwen3-Embedding-4B-Q8_0.gguf".to_string()],
    tokenizer_model_id: Some("Qwen/Qwen3-Embedding-0.6B".to_string()),
    sliding_window_config: Some(SlidingWindowConfig {
        window_stride: 128,
        min_window_size: 64,
    }),
};
```

### Args（実行時）

```rust
let args = EmbeddingArgs {
    text: "What is the capital of France?".to_string(),
    instruction: Some("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:".to_string()),
    normalize_embeddings: true,
};
```

## ビルドとテスト

### ビルド

```bash
# 開発版ビルド
cargo build -p embedding-llm

# リリース版ビルド(cuda)
cargo build --release -p embedding-llm --features "cuda"
# リリース版ビルド(metal)
cargo build --release -p embedding-llm --features "metal"
```

### テスト

```bash
# 単体テスト実行
cargo test -p embedding-llm

# テスト詳細出力
cargo test -p embedding-llm -- --nocapture
```

### 依存関係確認

```bash
# コンパイル確認
cargo check -p embedding-llm

# フォーマット確認
cargo fmt -p embedding-llm --check
```

## API 仕様

### PluginRunner Interface

```rust
impl PluginRunner for EmbeddingLlmRunnerPlugin {
    fn name(&self) -> String; // "EmbeddingLlmRunner-LlamaCpp"
    fn description(&self) -> String;
    fn load(&mut self, settings: Vec<u8>) -> Result<()>;
    fn run(&mut self, arg: Vec<u8>, metadata: HashMap<String, String>) -> (Result<Vec<u8>>, HashMap<String, String>);
    // ... その他のメソッド
}
```

## パフォーマンス

### メモリ効率

- llama.cpp の最適化されたメモリ管理を活用
- スライディングウィンドウによる長文の効率的処理

### CPU/GPU 対応

- CPU 専用モードのサポート
- GPU 加速の準備（実装時に有効化）
- 適応的な負荷分散

## トラブルシューティング

### 一般的な問題

1. **モデルファイルが見つからない**

   - GGUF ファイルのパスを確認
   - `model_files`配列にファイル名が正しく指定されているか確認

2. **トークナイザーエラー**

   - HuggingFace Hub へのアクセス確認
   - `tokenizer_model_id`の指定確認

3. **メモリ不足**
   - `use_cpu`フラグの確認
   - `max_seq_length`の調整

### デバッグ

```bash
# デバッグログを有効化
RUST_LOG=debug cargo test -p embedding-llm

# トレースログ
RUST_LOG=embedding_llm=trace cargo test -p embedding-llm
```

## ライセンス

MIT ライセンスまたは Apache-2.0 ライセンス
