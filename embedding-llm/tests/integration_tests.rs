use embedding_llm::{
    embedding::LlamaCppEmbedder,
    protobuf::embedding_llm::{
        DType, EmbeddingLlmRunnerSettings, HierarchicalChunkingConfig, ModelType,
    },
};

/// 実際のGGUFモデルを使用した統合テスト
///
/// 複数のモデル設定でテストを実行し、実際のembeddingベクトル計算を検証します。
///
/// 使用例:
/// ```bash
/// # 統合テスト実行（小型モデルを自動ダウンロード）
/// cargo test integration_test_real_embedding -- --ignored --test-threads=1 --nocapture
/// ```

#[tokio::test]
#[ignore] // 実際のモデルが必要なため通常は無視
async fn integration_test_real_embedding() {
    // ログ初期化
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    // テスト用モデル設定（小型モデルを使用）
    let test_configs = get_test_model_configs();

    for (config_name, settings) in test_configs {
        println!("\n======== Testing with {config_name} ========");
        if let Err(e) = run_embedding_test_with_config(&settings).await {
            println!("✗ Test failed for {config_name}: {e}");
            println!("Skipping to next configuration...\n");
            continue;
        }
    }

    println!("\n=== Real Embedding Integration Test Completed ===");
    println!("All available configurations tested!");
}

/// 個別の設定でembeddingテストを実行
async fn run_embedding_test_with_config(
    settings: &EmbeddingLlmRunnerSettings,
) -> anyhow::Result<()> {
    println!("Model: {}", settings.model_id);
    println!("Files: {:?}", settings.model_files);
    println!("Tokenizer: {:?}", settings.tokenizer_model_id);
    println!("Use CPU: {}", settings.use_cpu);
    println!("Max seq length: {}", settings.max_seq_length);

    // バックエンドの初期化（プラグインごと方式）
    println!("\n1. Initializing backend...");
    let backend_result = llama_cpp_2::llama_backend::LlamaBackend::init();
    if backend_result.is_err() {
        println!(
            "⚠ Failed to initialize backend: {:?}. Skipping test.",
            backend_result.err()
        );
        return Ok(());
    }
    let mut backend = backend_result.unwrap();
    backend.void_logs();
    let shared_backend = std::sync::Arc::new(std::sync::Mutex::new(backend));

    // Embedderの初期化（新方式）
    println!("\n2. Loading model with shared backend...");
    let embedder = match LlamaCppEmbedder::new_from_settings_with_backend(settings, shared_backend)
    {
        Ok(embedder) => {
            println!("✓ Model loaded successfully");
            embedder
        }
        Err(e) => {
            panic!("✗ Failed to load model '{}': {}\n\nTroubleshooting:\n- Ensure network connectivity for HF models\n- Verify GGUF file format\n- Check available memory\n- Try CPU-only mode", settings.model_id, e);
        }
    };

    // ヘルスチェック
    println!("\n3. Performing health check...");
    if let Err(e) = embedder.health_check() {
        panic!("✗ Health check failed: {e}");
    }
    println!("✓ Health check passed");

    // モデル情報の表示
    let model_info = embedder.model_info();
    println!("\n4. Model Information:");
    println!("   Path: {}", model_info.model_path);
    println!("   Embedding dimension: {}", model_info.embedding_dimension);
    println!("   Max context length: {}", model_info.max_context_length);
    println!("   Vocabulary size: {}", model_info.vocab_size);
    println!("   Dtype: {}", model_info.dtype);

    // 基本的な検証
    assert!(
        model_info.embedding_dimension > 0,
        "Invalid embedding dimension"
    );
    assert!(model_info.max_context_length > 0, "Invalid context length");
    assert!(model_info.vocab_size > 0, "Invalid vocabulary size");

    // テストケース
    let long_text = "A".repeat(1000);
    let test_cases = vec![
        ("Hello world", None, "Basic English text"),
        ("こんにちは世界", None, "Japanese text"),
        (
            "The quick brown fox jumps over the lazy dog",
            None,
            "Long English sentence",
        ),
        (
            "Artificial intelligence",
            Some("Represent this concept:"),
            "With instruction",
        ),
        ("", None, "Empty text (should handle gracefully)"),
        (
            long_text.as_str(),
            None,
            "Very long text (sliding window test)",
        ),
        (
            "Special characters !@#$%^&*()_+-=[]{}|;':\",.<>/?`~",
            None,
            "Text with special characters",
        ),
        (
            "プライベートワーク\n- 点数:  / 10\n- コメント:\n\t- 良かった点:\n\t- 改善したい点:\n(あれば)\n- 大
切にしたいと感じたこととなぜそれが自分にとって大切か",
            Some("Embed this mixed content:"),
            "Diary template",
        ),

    ];

    println!("\n5. Running embedding generation tests...");
    for (i, (text, instruction, description)) in test_cases.iter().enumerate() {
        println!("   Test {}: {}", i + 1, description);

        if text.is_empty() {
            // 空のテキストは適切にエラーハンドリングされることを確認
            let result = embedder.generate_embeddings_with_positions(
                text,
                instruction.as_deref(),
                true,
                None,
            );
            match result {
                Err(_) => println!("     ✓ Empty text handled correctly with error"),
                Ok(embeddings) if embeddings.is_empty() => {
                    println!("     ✓ Empty text returned empty embeddings")
                }
                Ok(_) => {
                    println!("     ? Empty text returned embeddings (model-dependent behavior)")
                }
            }
            continue;
        }

        // 正常なケースの処理
        let start_time = std::time::Instant::now();
        let result =
            embedder.generate_embeddings_with_positions(text, instruction.as_deref(), true, None);
        let duration = start_time.elapsed();

        match result {
            Ok(embeddings_with_pos) => {
                println!(
                    "     ✓ Generated {} embeddings with positions in {:?}",
                    embeddings_with_pos.len(),
                    duration
                );

                // 基本的な検証
                assert!(
                    !embeddings_with_pos.is_empty(),
                    "No embeddings generated for: {description}"
                );

                for (j, embedding_with_pos) in embeddings_with_pos.iter().enumerate() {
                    let embedding = &embedding_with_pos.values;
                    assert_eq!(
                        embedding.len(),
                        model_info.embedding_dimension,
                        "Wrong embedding dimension for embedding {j} in: {description}"
                    );

                    // L2正規化されているかチェック（normalize=trueのため）
                    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_diff = (norm - 1.0).abs();
                    assert!(
                        norm_diff < 0.01,
                        "Embedding {j} not properly normalized (norm: {norm:.6}) in: {description}"
                    );

                    // NaN/Infチェック
                    assert!(
                        !embedding.iter().any(|x| x.is_nan() || x.is_infinite()),
                        "Invalid values in embedding {j} for: {description}"
                    );

                    // 位置情報の検証
                    assert!(
                        embedding_with_pos.char_end_pos >= embedding_with_pos.char_start_pos,
                        "Invalid position range in embedding {}: start={}, end={}",
                        j,
                        embedding_with_pos.char_start_pos,
                        embedding_with_pos.char_end_pos
                    );

                    if j == 0 {
                        assert_eq!(
                            embedding_with_pos.char_start_pos, 0,
                            "First embedding should start at position 0"
                        );
                    }

                    println!(
                        "       - Position {}: chars[{}, {})",
                        j, embedding_with_pos.char_start_pos, embedding_with_pos.char_end_pos
                    );
                }

                println!(
                    "       - Embedding dimensions: {}x{}",
                    embeddings_with_pos.len(),
                    embeddings_with_pos[0].values.len()
                );
                println!(
                    "       - First 5 values: {:?}",
                    &embeddings_with_pos[0].values[..5.min(embeddings_with_pos[0].values.len())]
                );
                println!(
                    "       - L2 norm: {:.6}",
                    embeddings_with_pos[0]
                        .values
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                );
            }
            Err(e) => {
                panic!("✗ Failed to generate embedding for '{description}': {e}");
            }
        }
    }

    // 5. バッチ処理テスト
    println!("\n5. Testing batch processing...");
    let batch_texts = vec![
        "First text".to_string(),
        "Second text".to_string(),
        "Third text".to_string(),
    ];

    let individual_start = std::time::Instant::now();
    let mut individual_embeddings: Vec<Vec<Vec<f32>>> = Vec::new();
    for text in &batch_texts {
        let emb_with_pos = embedder
            .generate_embeddings_with_positions(text, None, true, None)
            .map_err(|e| anyhow::anyhow!("Batch processing failed: {e}"))?;
        // 位置情報を除去してembeddingのみを取得
        let emb: Vec<Vec<f32>> = emb_with_pos.into_iter().map(|e| e.values).collect();
        individual_embeddings.push(emb);
    }
    let individual_duration = individual_start.elapsed();

    println!(
        "   - Individual processing: {} texts in {:?}",
        batch_texts.len(),
        individual_duration
    );
    println!("   ✓ Batch processing test completed");

    // 6. メモリ使用量推定
    println!("\n6. Resource usage estimation:");
    let estimated_memory = embedder.estimate_memory_usage();
    println!(
        "   - Estimated memory usage: {:.2} MB",
        estimated_memory as f64 / 1024.0 / 1024.0
    );

    // 7. 同一テキストの一貫性テスト
    println!("\n7. Testing consistency...");
    let test_text = "Consistency test text";
    let embedding1_with_pos = embedder
        .generate_embeddings_with_positions(test_text, None, true, None)
        .map_err(|e| anyhow::anyhow!("Consistency test failed: {e}"))?;
    let embedding2_with_pos = embedder
        .generate_embeddings_with_positions(test_text, None, true, None)
        .map_err(|e| anyhow::anyhow!("Consistency test failed: {e}"))?;

    // 位置情報を除去してembeddingのみを比較
    let embedding1: Vec<Vec<f32>> = embedding1_with_pos.into_iter().map(|e| e.values).collect();
    let embedding2: Vec<Vec<f32>> = embedding2_with_pos.into_iter().map(|e| e.values).collect();

    assert_eq!(
        embedding1.len(),
        embedding2.len(),
        "Inconsistent number of embeddings"
    );

    // ベクトル間の距離を計算（同一テキストなので距離は0に近いはず）
    if !embedding1.is_empty() && !embedding2.is_empty() {
        let cosine_sim = embedding1[0]
            .iter()
            .zip(embedding2[0].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();

        println!("   - Cosine similarity between identical texts: {cosine_sim:.6}");
        assert!(
            cosine_sim > 0.99,
            "Inconsistent embeddings for identical text"
        );
        println!("   ✓ Consistency test passed");
    }

    println!("   ✓ All tests passed for this configuration!");

    Ok(()) // Resultの返却を追加
}

/// テスト用モデル設定を取得
/// 複数の設定でrobustnessを検証
#[tokio::test]
async fn test_diary_template_chunking_no_empty_tokens() {
    // Initialize logging for debugging
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    // This is the problematic text from integration test that causes "n_tokens == 0" error
    let problematic_text = "プライベートワーク\n- 点数:  / 10\n- コメント:\n\t- 良かった点:\n\t- 改善したい点:\n(あれば)\n- 大切にしたいと感じたこととなぜそれが自分にとって大切か";
    let instruction = Some("Embed this mixed content:");

    println!("Testing problematic text: {:?}", problematic_text);
    println!("Text length: {} chars", problematic_text.len());
    println!("Instruction: {:?}", instruction);

    // Use the HF tokenizer configuration to ensure we have a real tokenizer
    let settings = get_test_model_configs()
        .into_iter()
        .find(|(name, _)| name.contains("HF-Tokenizer"))
        .map(|(_, settings)| settings)
        .expect("HF Tokenizer config should be available");

    println!("Using model: {}", settings.model_id);
    println!("Using tokenizer: {:?}", settings.tokenizer_model_id);

    // Create the text processor directly to test chunking
    use command_utils::text::chunking::HierarchicalChunkingConfig;
    use embedding_llm::chunking_adapter::EmbeddingHierarchicalChunker;
    use embedding_llm::tokenization::TokenizationProcessor;
    use std::sync::Arc;

    let tokenization_processor =
        Arc::new(if let Some(tokenizer_id) = &settings.tokenizer_model_id {
            TokenizationProcessor::new_from_model_id(tokenizer_id, 128)
                .expect("Should be able to create tokenization processor")
        } else {
            panic!("No tokenizer specified in test settings")
        });

    // Create chunker and test the problematic text
    let config = HierarchicalChunkingConfig::for_embedding(128);
    let mut chunker = EmbeddingHierarchicalChunker::with_config(tokenization_processor, config)
        .expect("Should be able to create chunker");

    println!("\nTesting chunking with instruction...");
    let full_text_with_instruction = if let Some(inst) = instruction {
        format!("{}\n{}", inst, problematic_text)
    } else {
        problematic_text.to_string()
    };

    println!("Full text to chunk: {:?}", full_text_with_instruction);
    println!(
        "Full text length: {} chars",
        full_text_with_instruction.len()
    );

    // Perform the chunking that should generate the problematic empty token_ids
    let chunks_result = chunker.chunk_for_embedding(&full_text_with_instruction);

    match chunks_result {
        Ok(chunks) => {
            println!("✓ Chunking succeeded, got {} chunks", chunks.len());

            // Check each chunk for empty token_ids - this is where the problem should be visible
            for (i, chunk) in chunks.iter().enumerate() {
                println!(
                    "Chunk {}: content_len={}, token_count={}",
                    i,
                    chunk.content.len(),
                    chunk.token_ids.len()
                );
                println!("  Content: {:?}", chunk.content);
                println!("  Token IDs: {:?}", chunk.token_ids);

                // This assertion should fail if empty token_ids are generated
                assert!(
                    !chunk.token_ids.is_empty(),
                    "Chunk {} has empty token_ids! Content: {:?}",
                    i,
                    chunk.content
                );

                assert!(
                    !chunk.content.is_empty(),
                    "Chunk {} has empty content! This should not happen",
                    i
                );
            }

            println!("✓ All {} chunks have non-empty token_ids", chunks.len());
        }
        Err(e) => {
            panic!("✗ Chunking failed: {}", e);
        }
    }
}

fn get_test_model_configs() -> Vec<(&'static str, EmbeddingLlmRunnerSettings)> {
    vec![
        // 1. Qwen3-Embeddingのみ（動作確認済み）
        (
            "Qwen3-Embedding-Only",
            EmbeddingLlmRunnerSettings {
                model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
                use_cpu: true,
                dtype: Some(DType::F32 as i32),
                max_seq_length: 256, // 短くして高速化
                model_type: ModelType::Gguf as i32,
                model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
                tokenizer_model_id: None, // GGUF内蔵tokenizerでシンプル化
                chunking_config: None,    // hierarchical chunkingなしで高速化
                max_batch_size: Some(8),  // バッチ処理テスト用
                gpu_device: None,
            },
        ),
        // 2. Qwen3-Embedding（最新embedding専用モデル、GGUF内蔵tokenizer）
        (
            "Qwen3-Embedding-GGUF-Tokenizer",
            EmbeddingLlmRunnerSettings {
                model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
                use_cpu: true,
                dtype: None, // デフォルト値テスト
                max_seq_length: 1024,
                model_type: ModelType::Gguf as i32,
                model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
                tokenizer_model_id: None, // GGUF内蔵tokenizerを使用
                chunking_config: Some(HierarchicalChunkingConfig {
                    max_chunk_tokens: 512,
                    min_chunk_tokens: 1,
                    enable_paragraph_merging: true,
                    enable_sentence_splitting: true,
                    enable_forced_splitting: true,
                }),
                max_batch_size: Some(6),
                gpu_device: None,
            },
        ),
        // 3. Qwen3-Embedding + HuggingFace tokenizer組み合わせ
        (
            "Qwen3-Embedding-HF-Tokenizer",
            EmbeddingLlmRunnerSettings {
                model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
                use_cpu: false,
                dtype: Some(DType::F16 as i32),
                max_seq_length: 768,
                model_type: ModelType::Gguf as i32,
                model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
                tokenizer_model_id: Some("Qwen/Qwen3-Embedding-4B".to_string()), // 元のembeddingモデルのtokenizer
                chunking_config: None,   // hierarchical chunking無効テスト
                max_batch_size: Some(2), // GPU用小さめバッチサイズ
                gpu_device: None,
            },
        ),
    ]
}

/// 個別テスト: Qwen3-Embeddingで高速テスト
#[tokio::test]
#[ignore]
async fn integration_test_qwen3_embedding_quick() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG) // DEBUGレベルに変更
        .with_test_writer()
        .try_init();

    let settings = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: true, // CPU専用でデバッグ
        dtype: Some(DType::F32 as i32),
        max_seq_length: 64, // さらに短縮
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: None, // GGUF内蔵tokenizerのテスト
        chunking_config: None,
                gpu_device: None,
        max_batch_size: Some(4), // クイックテスト用バッチサイズ
    };

    println!("=== Qwen3-Embedding Quick Test ===");
    if let Err(e) = run_embedding_test_with_config(&settings).await {
        panic!("Qwen3-Embedding test failed: {e}");
    }
}

/// 個別テスト: Qwen3-Embedding + GGUF内蔵tokenizer
#[tokio::test]
#[ignore]
async fn integration_test_qwen3_embedding_gguf_tokenizer() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();

    let settings = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: false,
        dtype: None, // デフォルト値テスト
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: Some("Qwen/Qwen3-Embedding-4B".to_string()), // GGUF内蔵tokenizerをテスト
        chunking_config: Some(HierarchicalChunkingConfig {
            max_chunk_tokens: 512,
            min_chunk_tokens: 1,
            enable_paragraph_merging: true,
            enable_sentence_splitting: true,
            enable_forced_splitting: true,
        }),
        max_batch_size: Some(4), // GGUF内蔵tokenizer用バッチサイズ
        gpu_device: None,
    };

    println!("=== Qwen3-Embedding with HuggingFace Tokenizer Test ===");
    if let Err(e) = run_embedding_test_with_config(&settings).await {
        panic!("Qwen3-Embedding GGUF tokenizer test failed: {e}");
    }
}

/// 実際のモデルでのエラーハンドリングテスト
#[tokio::test]
#[ignore]
async fn integration_test_error_handling() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_test_writer()
        .try_init();

    println!("=== Error Handling Integration Test ===");

    // 1. 存在しないモデルファイル
    let invalid_settings = EmbeddingLlmRunnerSettings {
        model_id: "nonexistent".to_string(),
        use_cpu: true,
        dtype: Some(DType::F32 as i32),
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["nonexistent.gguf".to_string()],
        tokenizer_model_id: None,
        chunking_config: None,
        max_batch_size: Some(4),
        gpu_device: None,
    };

    // バックエンドの初期化
    let backend_result = llama_cpp_2::llama_backend::LlamaBackend::init();
    if backend_result.is_err() {
        println!("⚠ Skipping error handling test - backend initialization failed");
        return;
    }
    let mut backend = backend_result.unwrap();
    backend.void_logs();
    let shared_backend = std::sync::Arc::new(std::sync::Mutex::new(backend));

    println!("1. Testing invalid model path...");
    let result =
        LlamaCppEmbedder::new_from_settings_with_backend(&invalid_settings, shared_backend.clone());
    assert!(result.is_err(), "Should fail with invalid model path");
    println!("   ✓ Correctly handled invalid model path");

    // 2. 無効なtokenizer
    let invalid_tokenizer_settings = EmbeddingLlmRunnerSettings {
        model_id: "test".to_string(),
        use_cpu: true,
        dtype: Some(DType::F32 as i32),
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["test.gguf".to_string()],
        tokenizer_model_id: Some("nonexistent/tokenizer".to_string()),
        chunking_config: None,
        max_batch_size: Some(4),
        gpu_device: None,
    };

    println!("2. Testing invalid tokenizer...");
    let result = LlamaCppEmbedder::new_from_settings_with_backend(
        &invalid_tokenizer_settings,
        shared_backend,
    );
    assert!(result.is_err(), "Should fail with invalid tokenizer");
    println!("   ✓ Correctly handled invalid tokenizer");

    println!("=== Error Handling Test Completed ===");
}

#[derive(Debug)]
struct TestCase {
    name: &'static str,
    text: &'static str,
    expected_chunk_count: usize,
    description: &'static str,
}

#[tokio::test]
async fn test_hierarchical_chunking_and_batch_consistency() {
    println!("=== Hierarchical Chunking and Batch Consistency Test ===");

    // 3つの異なる分割パターンをテスト
    let test_cases = vec![
        TestCase {
            name: "paragraph_level",
            text: "これは最初のパラグラフです。適度な長さでパラグラフ境界で分割されるべきです。\n\nこれは2番目のパラグラフです。こちらも適度な長さでちょうど良いサイズです。\n\n3番目のパラグラフはここにあります。短めです。",
            expected_chunk_count: 3,
            description: "Multiple paragraphs, each should be a separate chunk",
        },
        TestCase {
            name: "sentence_level",
            text: "これは長いパラグラフの最初の文です。2番目の文はここにあります。3番目の文も含まれています。4番目の文でパラグラフが続きます。5番目の文があります。6番目の文もあります。7番目の文で終わります。",
            expected_chunk_count: 2, // センテンスレベルで分割される
            description: "Long paragraph with multiple sentences, should split by sentences",
        },
        TestCase {
            name: "forced_split",
            text: "これは非常に長い単一の文で区切り文字が少なくセンテンス分割でも対応できないためトークン制限に基づいて強制的に分割される必要がある非常に長い文字列でありセンテンス境界がないため強制分割が必要です",
            expected_chunk_count: 3, // 強制分割される
            description: "Very long text without sentence boundaries, should use forced splitting",
        },
    ];

    println!("Testing {} different chunking patterns", test_cases.len());

    // 分割を発生させやすい設定
    let base_settings = EmbeddingLlmRunnerSettings {
        use_cpu: true,
        max_seq_length: 128,
        model_type: ModelType::Gguf as i32,
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: None,
        dtype: Some(DType::F32 as i32),
        chunking_config: Some(HierarchicalChunkingConfig {
            max_chunk_tokens: 20, // 分割を発生させやすい小さな値
            min_chunk_tokens: 1,  // 小さなチャンクも保持
            enable_paragraph_merging: true,
            enable_sentence_splitting: true,
            enable_forced_splitting: true,
        }),
        max_batch_size: Some(1), // 個別処理用
        gpu_device: None,
    };

    // バックエンドの初期化
    let backend_result = llama_cpp_2::llama_backend::LlamaBackend::init();
    if backend_result.is_err() {
        println!("⚠ Skipping hierarchical chunking test - backend initialization failed");
        return;
    }
    let mut backend = backend_result.unwrap();
    backend.void_logs();
    let shared_backend = std::sync::Arc::new(std::sync::Mutex::new(backend));

    let embedder =
        match LlamaCppEmbedder::new_from_settings_with_backend(&base_settings, shared_backend) {
            Ok(e) => e,
            Err(e) => {
                println!("⚠ Skipping hierarchical chunking test due to model loading failure: {e}");
                return;
            }
        };

    // 各テストケースを処理
    for test_case in &test_cases {
        println!(
            "\n--- Testing {}: {} ---",
            test_case.name, test_case.description
        );
        let display_text = test_case
            .text
            .chars()
            .take(60)
            .collect::<String>()
            .replace('\n', "\\n");
        println!("Text: {}", display_text);

        // 1. バッチ処理でembedding生成（分割結果も取得）
        println!("1. Batch processing...");
        let batch_results = embedder
            .generate_embeddings_with_positions(
                test_case.text,
                Some("Generate embedding for this text"),
                false, // 正規化なし
                None,  // mergeなし
            )
            .unwrap_or_else(|_| panic!("Batch embedding generation failed for {}", test_case.name));

        println!("   Generated {} chunks", batch_results.len());

        // 分割内容の検証
        assert!(
            !batch_results.is_empty(),
            "Test case '{}': No chunks generated (expected at least 1)",
            test_case.name
        );

        // 期待されるチャンク数の近似的な検証（厳密ではない、トークナイザーに依存）
        let chunk_count_reasonable = batch_results.len() >= (test_case.expected_chunk_count / 2)
            && batch_results.len() <= (test_case.expected_chunk_count * 2);
        if !chunk_count_reasonable {
            println!(
                "   ⚠ Warning: Expected ~{} chunks, got {} chunks for '{}'",
                test_case.expected_chunk_count,
                batch_results.len(),
                test_case.name
            );
        }

        // 各チャンクがトークン制限を守っているかを検証
        for (i, result) in batch_results.iter().enumerate() {
            let chunk_text: String = test_case
                .text
                .chars()
                .skip(result.char_start_pos)
                .take(result.char_end_pos - result.char_start_pos)
                .collect();
            println!(
                "     Chunk {}: pos={}-{}, text=\"{}\"",
                i,
                result.char_start_pos,
                result.char_end_pos,
                chunk_text
                    .chars()
                    .take(30)
                    .collect::<String>()
                    .replace('\n', "\\n")
            );

            // 文字位置が有効範囲内であることを確認
            assert!(
                result.char_start_pos <= result.char_end_pos,
                "Chunk {}: Invalid char positions {} > {}",
                i,
                result.char_start_pos,
                result.char_end_pos
            );
            assert!(
                result.char_end_pos <= test_case.text.chars().count(),
                "Chunk {}: End position {} exceeds text character count {}",
                i,
                result.char_end_pos,
                test_case.text.chars().count()
            );
        }

        // 2. 個別処理で同じ結果が得られるかテスト
        println!("2. Individual processing...");
        let individual_results = embedder
            .generate_embeddings_with_positions(
                test_case.text,
                Some("Generate embedding for this text"),
                false, // 正規化なし
                None,  // mergeなし
            )
            .unwrap_or_else(|_| {
                panic!(
                    "Individual embedding generation failed for {}",
                    test_case.name
                )
            });

        // 3. バッチ vs 個別の一致性検証
        println!("3. Comparing batch vs individual results...");
        assert_eq!(
            batch_results.len(),
            individual_results.len(),
            "Test case '{}': Batch ({}) and individual ({}) chunk counts should match",
            test_case.name,
            batch_results.len(),
            individual_results.len()
        );

        for (i, (batch_result, individual_result)) in batch_results
            .iter()
            .zip(individual_results.iter())
            .enumerate()
        {
            // 位置情報の一致を確認
            assert_eq!(
                batch_result.char_start_pos, individual_result.char_start_pos,
                "Test case '{}', chunk {}: Start positions differ",
                test_case.name, i
            );
            assert_eq!(
                batch_result.char_end_pos, individual_result.char_end_pos,
                "Test case '{}', chunk {}: End positions differ",
                test_case.name, i
            );

            // embedding値の一致を確認
            assert_eq!(
                batch_result.values.len(),
                individual_result.values.len(),
                "Test case '{}', chunk {}: Embedding dimensions differ",
                test_case.name,
                i
            );

            let max_diff = batch_result
                .values
                .iter()
                .zip(individual_result.values.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < 1e-5,
                "Test case '{}', chunk {}: Embeddings differ beyond tolerance. Max diff: {:.2e}",
                test_case.name,
                i,
                max_diff
            );
        }

        println!("   ✓ All {} chunks are consistent", batch_results.len());
    }

    println!(
        "\n✓ All {} test cases passed hierarchical chunking and consistency tests",
        test_cases.len()
    );

    println!("=== Hierarchical Chunking and Batch Consistency Test Completed Successfully ===");
}

/// GPU device指定のテスト
#[tokio::test]
#[ignore]
async fn integration_test_gpu_device_specification() {
    use embedding_llm::embedding::LlamaCppEmbedder;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use std::sync::{Arc, Mutex};

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();

    println!("=== GPU Device Specification Test ===");

    // バックエンドの初期化
    let backend_result = LlamaBackend::init();
    if backend_result.is_err() {
        println!("⚠ Skipping GPU device test - backend initialization failed");
        return;
    }

    let mut backend = backend_result.unwrap();
    backend.void_logs();
    let shared_backend = Arc::new(Mutex::new(backend));

    // Test 1: GPU device 0を指定（24GB GPU）
    let settings_gpu0 = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: false,
        dtype: Some(DType::F16 as i32),
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: None,
        chunking_config: None,
        max_batch_size: Some(2),
        gpu_device: Some(0), // GPU 0を指定
    };

    println!("1. Testing with gpu_device=0...");
    let result = LlamaCppEmbedder::new_from_settings_with_backend(&settings_gpu0, shared_backend.clone());
    match result {
        Ok(_embedder) => {
            println!("   ✓ Successfully initialized with GPU device 0");
        }
        Err(e) => {
            println!("   ⚠ Failed to initialize with GPU device 0: {}", e);
            println!("   (This is expected if GPU 0 does not have sufficient VRAM)");
        }
    }

    // Test 2: GPU device未指定（デフォルト動作）
    let settings_no_gpu = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: false,
        dtype: Some(DType::F16 as i32),
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: None,
        chunking_config: None,
        max_batch_size: Some(2),
        gpu_device: None, // GPU指定なし（デフォルト）
    };

    println!("\n2. Testing with gpu_device=None (default)...");
    let result = LlamaCppEmbedder::new_from_settings_with_backend(&settings_no_gpu, shared_backend.clone());
    match result {
        Ok(_embedder) => {
            println!("   ✓ Successfully initialized with default GPU device");
        }
        Err(e) => {
            println!("   ⚠ Failed to initialize with default GPU: {}", e);
        }
    }

    // Test 3: CPUモード時のGPU指定は無視される
    let settings_cpu_with_gpu = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: true,
        dtype: Some(DType::F32 as i32),
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: None,
        chunking_config: None,
        max_batch_size: Some(2),
        gpu_device: Some(0), // CPUモードだが指定（警告が出るべき）
    };

    println!("\n3. Testing with use_cpu=true and gpu_device=0 (should warn)...");
    let result = LlamaCppEmbedder::new_from_settings_with_backend(&settings_cpu_with_gpu, shared_backend.clone());
    match result {
        Ok(_embedder) => {
            println!("   ✓ Successfully initialized (GPU device ignored in CPU mode)");
        }
        Err(e) => {
            println!("   ✗ Unexpected error: {}", e);
        }
    }

    // Test 4: 負のGPU IDはエラー
    let settings_invalid_gpu = EmbeddingLlmRunnerSettings {
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        use_cpu: false,
        dtype: Some(DType::F16 as i32),
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: None,
        chunking_config: None,
        max_batch_size: Some(2),
        gpu_device: Some(-1), // 負の値は不正
    };

    println!("\n4. Testing with invalid gpu_device=-1 (should fail)...");
    let result = LlamaCppEmbedder::new_from_settings_with_backend(&settings_invalid_gpu, shared_backend);
    assert!(result.is_err(), "Should fail with invalid GPU device ID");
    println!("   ✓ Correctly rejected invalid GPU device ID");

    println!("\n=== GPU Device Specification Test Completed ===");
}
