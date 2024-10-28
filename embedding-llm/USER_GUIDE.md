# embedding-llm プラグイン利用者ガイド

## 概要

embedding-llmは、JobWorkerP用のLLMベース埋め込み生成プラグインです。llama.cppライブラリを使用してGGUFフォーマットのモデルから高品質な埋め込みベクトルを生成し、位置情報付きの詳細な結果を提供します。

### 主な特徴

- **GGUF モデル対応**: llama.cpp互換のGGUFフォーマットモデルを使用
- **位置情報付き埋め込み**: テキスト内の各埋め込みの文字位置を正確に追跡
- **階層的チャンキング**: 長いテキストを適切なサイズのチャンクに自動分割
- **GPU/CPU アクセラレーション**: CUDA、Metal、OpenMPによる高速化対応
- **L2正規化**: オプションで埋め込みベクトルの正規化が可能
- **バッチ処理**: メモリ効率的な複数テキストの処理

## JobWorkerPについて

JobWorkerP-rsは、CPUおよびI/O集約的なタスクを非同期で処理するためのスケーラブルなジョブワーカーシステムです。gRPCフロントエンド、柔軟なワーカーコンポーネント、Redis/RDBストレージ層を備え、プラグインシステムによる拡張性を提供します。

## 設定（Settings）

プラグインの初期化時に指定する設定項目です。

### EmbeddingLlmRunnerSettings

| フィールド | 型 | 必須 | 説明 |
|------------|----|----|------|
| `model_id` | string | ✓ | HuggingFace モデルIDまたはローカルモデルパス |
| `use_cpu` | bool | - | CPU使用を強制（デフォルト: false、GPU使用） |
| `dtype` | DType | - | モデル精度（デフォルト: GPU時F16、CPU時F32） |
| `max_seq_length` | uint32 | ✓ | トークン化の最大シーケンス長 |
| `model_type` | ModelType | ✓ | モデルフォーマット（現在はGGUFのみ） |
| `model_files` | string[] | ✓ | 使用するGGUFファイルのリスト |
| `tokenizer_model_id` | string | - | 別のトークナイザーモデルID |
| `chunking_config` | HierarchicalChunkingConfig | - | 階層的チャンキング設定 |
| `max_batch_size` | uint32 | - | バッチ処理の最大サイズ |

### DType（データ型）

```protobuf
enum DType {
  F32 = 0;   // 32-bit float (CPU用デフォルト)
  F16 = 1;   // 16-bit float (GPU用デフォルト)
  BF16 = 2;  // 16-bit brain float (CUDA環境で問題が発生する場合あり)
}
```

### HierarchicalChunkingConfig（階層的チャンキング設定）

| フィールド | 型 | 説明 |
|------------|----|----|
| `max_chunk_tokens` | uint32 | 最大チャンクトークン数 |
| `min_chunk_tokens` | uint32 | 最小チャンクトークン数 |
| `enable_paragraph_merging` | bool | 小さな段落の統合を有効化 |
| `enable_sentence_splitting` | bool | 大きな段落の文分割を有効化 |
| `enable_forced_splitting` | bool | 最終手段としての強制分割を有効化 |

## 引数（Arguments）

プラグイン実行時に指定する引数です。

### EmbeddingArgs

| フィールド | 型 | 必須 | 説明 |
|------------|----|----|------|
| `text` | string | ✓ | 埋め込み生成対象のテキスト |
| `instruction` | string | - | 埋め込みタスク用の指示プレフィックス |
| `normalize_embeddings` | bool | ✓ | L2正規化の適用有無 |

### 使用例

```json
{
  "text": "人工知能と機械学習は現代のテクノロジーを変革しています。",
  "instruction": "検索用ドキュメント埋め込みを生成:",
  "normalize_embeddings": true
}
```

## 出力（Output）

### EmbeddingLlmResult

| フィールド | 型 | 説明 |
|------------|----|----|
| `embeddings` | Embedding[] | 生成された埋め込みリスト |
| `model_info` | ModelInfo | モデルメタデータ |

### Embedding

| フィールド | 型 | 説明 |
|------------|----|----|
| `values` | float[] | 埋め込みベクトルの値 |
| `begin_position` | uint32 | 元テキスト内の開始文字位置 |
| `end_position` | uint32 | 元テキスト内の終了文字位置 |
| `content` | string | この埋め込みに対応するテキスト内容 |

### ModelInfo

| フィールド | 型 | 説明 |
|------------|----|----|
| `model_name` | string | モデル名 |
| `embedding_dimension` | uint32 | 埋め込み次元数 |
| `dtype_used` | string | 使用されたデータ型 |

## 使用例

### 1. ワークフローでの使用

日記ファイルをベクトル化してデータベースに保存する例：

```yaml
# diary-to-vector.yml
document:
  id: 123
  specVersion: "0.8"
  name: diary-summary-workflow

do:
  - DirectoryDiggerWorker:
      run:
        function:
          runnerName: DirectoryDigger
          arguments:
            include_regex: .*\.md
            target_directory: /path/to/documents

  - FileLoop:
      for:
        in: ${.matchedFiles}
        each: file
      do:
        - ReadFileWorker:
            run:
              function:
                runnerName: filesystem
                arguments:
                  tool_name: read_text_file
                  arg_json: |
                    $${{
                      "path": "/path/to/documents/{{file}}"
                    }}
            output:
              as:
                content: ${.content[0].text.text}
                filename: |
                  $${/path/to/documents/{{file}}}

        - generate_embedding:
            run:
              runner:
                name: "EmbeddingLlmRunner"
                settings:
                  model_id: "Qwen/Qwen3-Embedding-4B-GGUF"
                  max_seq_length: 4096
                  model_type: GGUF
                  model_files: ["Qwen3-Embedding-4B-Q6_K.gguf"]
                  tokenizer_model_id: "Qwen/Qwen3-Embedding-4B"
                arguments:
                  text: ${.content}
                  normalize_embeddings: true
                options:
                  channel: llm
                  useStatic: true
            output:
              as:
                embedding_result: ${.}

        - store_embeddings:
            for:
              each: "embedding"
              in: ${.embedding_result.embeddings}
            do:
              - store_vector:
                  run:
                    runner:
                      name: "ArticleVectorDbStoreRunner"
                      settings:
                        lancedb_uri: "data/lancedb/memories.lancedb"
                        table_name: "document_fragments"
                        vector_size: 2560
                      arguments:
                        operation_type: 1
                        document_id:
                          location: |
                            $${/path/to/documents/{{file}}}
                          location_type: 0
                        metadata:
                          project_uri: file://Documents/
                          sub_path: ${$file}
                        fragments:
                          - content: ${$embedding.content}
                            embedding:
                              values: ${$embedding.values}
                            range_start: ${$embedding.begin_position}
                            range_end: ${$embedding.end_position}
```

### 2. jobworkerp-clientでの使用

Rustクライアントライブラリを使用してプラグインを呼び出す例：

```rust
use jobworkerp_client::{
    client::JobworkerpClient,
    command::helper::UseJobworkerpClientHelper,
    jobworkerp::data::{JobRequest, Priority, QueueType, ResponseType},
};
use embedding_llm::protobuf::embedding_llm::{
    EmbeddingLlmRunnerSettings, EmbeddingArgs, DType, ModelType
};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // JobWorkerPクライアントの初期化
    let client = JobworkerpClient::new(
        "http://localhost:8081".to_string(),
        Some(std::time::Duration::from_secs(30))
    ).await?;

    // プラグイン設定の準備
    let settings = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: true,
        dtype: Some(DType::F32 as i32),
        max_seq_length: 1024,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: Some("Qwen/Qwen3-Embedding-4B".to_string()),
        chunking_config: None,
        max_batch_size: Some(4),
    };

    // 引数の準備
    let args = EmbeddingArgs {
        text: "これは埋め込み生成のテストです。".to_string(),
        instruction: Some("検索用埋め込み生成:".to_string()),
        normalize_embeddings: true,
    };

    // Protobufエンコーディング
    let mut settings_buf = Vec::new();
    settings.encode(&mut settings_buf)?;

    let mut args_buf = Vec::new();
    args.encode(&mut args_buf)?;

    // ジョブリクエストの作成
    let job_request = JobRequest {
        runner_name: "EmbeddingLlmRunner".to_string(),
        settings: settings_buf,
        arg: args_buf,
        queue_type: QueueType::StepByStep as i32,
        response_type: ResponseType::SmallBinary as i32,
        priority: Priority::Medium as i32,
        timeout: 60,
        metadata: HashMap::new(),
        ..Default::default()
    };

    // ジョブの実行
    let result = client.execute_job_direct(
        None,
        Arc::new(HashMap::new()),
        job_request
    ).await?;

    // 結果のデコード
    if let Some(result_data) = result.result {
        let embedding_result = embedding_llm::protobuf::embedding_llm::EmbeddingLlmResult::decode(
            &result_data.arg[..]
        )?;

        println!("生成された埋め込み数: {}", embedding_result.embeddings.len());

        for (i, embedding) in embedding_result.embeddings.iter().enumerate() {
            println!(
                "埋め込み {}: 位置 {}-{}, 次元数: {}, 内容: \"{}\"",
                i,
                embedding.begin_position,
                embedding.end_position,
                embedding.values.len(),
                embedding.content
            );
        }

        if let Some(model_info) = &embedding_result.model_info {
            println!("モデル: {}", model_info.model_name);
            println!("埋め込み次元: {}", model_info.embedding_dimension);
        }
    }

    Ok(())
}
```

### 3. 設定例集

#### 基本設定（CPU使用）

```json
{
  "model_id": "Qwen/Qwen3-Embedding-4B-GGUF",
  "use_cpu": true,
  "dtype": "F32",
  "max_seq_length": 1024,
  "model_type": "GGUF",
  "model_files": ["Qwen3-Embedding-4B-Q4_K_M.gguf"],
  "tokenizer_model_id": "Qwen/Qwen3-Embedding-4B"
}
```

#### GPU使用（高速処理）

```json
{
  "model_id": "Qwen/Qwen3-Embedding-4B-GGUF",
  "use_cpu": false,
  "dtype": "F16",
  "max_seq_length": 4096,
  "model_type": "GGUF",
  "model_files": ["Qwen3-Embedding-4B-Q6_K.gguf"],
  "tokenizer_model_id": "Qwen/Qwen3-Embedding-4B",
  "max_batch_size": 8
}
```

#### 階層的チャンキングパラメータ指定

```json
{
  "model_id": "Qwen/Qwen3-Embedding-4B-GGUF",
  "use_cpu": true,
  "max_seq_length": 2048,
  "model_type": "GGUF",
  "model_files": ["Qwen3-Embedding-4B-Q4_K_M.gguf"],
  "chunking_config": {
    "max_chunk_tokens": 512,
    "min_chunk_tokens": 50,
    "enable_paragraph_merging": true,
    "enable_sentence_splitting": true,
    "enable_forced_splitting": true
  }
}
```

## トラブルシューティング

### よくある問題

1. **モデルファイルが見つからない**
   - HuggingFace Hubからのダウンロードに時間がかかる場合があります
   - ネットワーク接続を確認してください
   - ローカルにモデルファイルを配置する場合は絶対パスを使用してください

2. **メモリ不足エラー**
   - `use_cpu: true`に設定してGPUメモリ使用を回避
   - `max_seq_length`を小さくして処理するテキスト長を制限
   - `max_batch_size`を小さくしてバッチサイズを調整

3. **CUDA関連エラー**
   - `dtype`を`F32`または`F16`に変更（`BF16`を避ける）
   - CUDA対応のllama.cppビルドを使用していることを確認

4. **空のトークンIDエラー**
   - 階層的チャンキング設定の`min_chunk_tokens`を0より大きく設定
   - 処理するテキストが十分な長さであることを確認

### パフォーマンス最適化

1. **GPU使用時**
   - `dtype: F16`を使用
   - 適切な`max_batch_size`を設定（メモリ容量に応じて調整）

2. **CPU使用時**
   - `dtype: F32`を使用
   - OpenMP対応ビルドを使用してマルチスレッド処理を有効化

3. **長いテキスト処理**
   - 階層的チャンキングを有効化
   - `max_chunk_tokens`を適切に設定

## サポートされるモデル

現在のバージョンでは以下のモデルフォーマットをサポートしています：

- **GGUF**: llama.cpp互換のGGUFフォーマット
  - 推奨モデル: Qwen3-Embedding-4B-GGUF
  - 各種量子化レベル対応（Q4_K_M、Q6_K等）

## 次のステップ

1. [JobWorkerP-rsのドキュメント](https://github.com/jobworkerp-rs/jobworkerp-rs)でシステム全体の使用方法を確認
2. [llama.cppのドキュメント](https://github.com/ggerganov/llama.cpp)でモデル詳細を確認
3. サンプルワークフローを参考に独自の処理パイプラインを構築

## ライセンス

このプラグインはJobWorkerP-rsプロジェクトの一部として提供されています。詳細はプロジェクトのライセンスファイルを参照してください。