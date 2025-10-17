use embedding_llm::{
    protobuf::embedding_llm::{
        DType, EmbeddingArgs, EmbeddingLlmResult, EmbeddingLlmRunnerSettings,
        HierarchicalChunkingConfig, ModelType,
    },
    EmbeddingLlmRunnerPlugin,
};
use jobworkerp_client::plugins::PluginRunner;
use prost::Message;
use std::collections::HashMap;

/// EmbeddingLlmRunnerPluginの結合テスト
///
/// プラグインの完全なライフサイクル（load → run）をテストし、
/// 実際のembedding生成処理が正しく動作することを確認します。

#[tokio::test]
async fn test_plugin_full_lifecycle_with_embedding_generation() {
    println!("=== Plugin Full Lifecycle Integration Test ===");

    // ロギング初期化
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    // 1. プラグインの初期化
    let mut plugin = EmbeddingLlmRunnerPlugin::new().expect("Failed to create plugin");

    println!("Plugin created: {}", plugin.name());
    assert_eq!(plugin.name(), "EmbeddingLlmRunner");
    assert!(!plugin.description().is_empty());

    // 2. 設定データの準備とload
    let settings = EmbeddingLlmRunnerSettings {
        use_cpu: false,
        max_seq_length: 128, // Sufficient sequence length
        model_type: ModelType::Gguf as i32,
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: Some("Qwen/Qwen3-Embedding-4B".to_string()),
        dtype: Some(DType::Bf16 as i32),
        chunking_config: Some(HierarchicalChunkingConfig {
            max_chunk_tokens: 64, // Small chunks to force hierarchical chunking
            min_chunk_tokens: 0,  // no minimum size
            enable_paragraph_merging: true,
            enable_sentence_splitting: true,
            enable_forced_splitting: true,
        }),
        max_batch_size: Some(1), // Sequential processing to avoid batch space issues
        gpu_device: None,
    };

    let mut settings_buf = Vec::new();
    settings
        .encode(&mut settings_buf)
        .expect("Failed to encode settings");

    println!("Loading plugin with model: {}", settings.model_id);
    let load_result = plugin.load(settings_buf);

    match load_result {
        Ok(()) => {
            println!("✓ Plugin loaded successfully");
        }
        Err(e) => {
            println!("⚠ Plugin load failed: {e}. Skipping embedding tests.");
            println!("This may be due to model unavailability or network issues.");
            return;
        }
    }

    // 3. 実際のembedding生成テスト
    println!("3. Testing embedding generation...");

    let test_cases = vec![
        (
            "Simple text embedding test",
            EmbeddingArgs {
                text: "The quick brown fox jumps over the lazy dog.".to_string(),
                instruction: Some("Generate embedding for this text".to_string()),
                normalize_embeddings: false,
            }
        ),
        (
            "Normalized embedding test", 
            EmbeddingArgs {
                text: "Machine learning and artificial intelligence are transforming technology.".to_string(),
                instruction: Some("Create semantic embedding".to_string()),
                normalize_embeddings: true,
            }
        ),
        (
            "Long text sliding window test",
            EmbeddingArgs {
                text: "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. This is a very long text that should exceed the token limits and create multiple sliding windows for comprehensive testing of position calculation across different windows and segments.".to_string(),
                instruction: Some("Generate comprehensive embedding".to_string()),
                normalize_embeddings: false,
            }
        ),
        (
            "Multi-window position test",
            EmbeddingArgs {
                text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.".to_string(),
                instruction: None,
                normalize_embeddings: false,
            }
        ),
        (
            "Short text test",
            EmbeddingArgs {
                text: "Hello world".to_string(),
                instruction: None, // No instruction test
                normalize_embeddings: true,
            }
        ),
        (
            "Extremely long text for guaranteed multiple windows",
            EmbeddingArgs {
                text: "This is an extremely long text designed to exceed token limits and create multiple sliding windows. The purpose of this test is to validate that the character position calculation works correctly when text is split into multiple segments. Natural language processing involves many complex tasks including tokenization, semantic analysis, syntactic parsing, named entity recognition, sentiment analysis, machine translation, question answering, text summarization, and many other applications. Machine learning models have revolutionized how we approach these problems by learning patterns from large datasets rather than relying solely on hand-crafted rules. Deep learning architectures such as transformers have been particularly successful in capturing long-range dependencies in sequential data. The attention mechanism allows models to focus on relevant parts of the input sequence when making predictions. This has led to significant improvements in tasks like machine translation where the model needs to align words and phrases between source and target languages. Pre-trained language models like BERT, GPT, and T5 have further advanced the field by providing strong baselines that can be fine-tuned for specific tasks. These models are trained on massive amounts of text data and learn rich representations of language that capture both syntactic and semantic information. The emergence of large language models has opened up new possibilities for few-shot and zero-shot learning where models can perform tasks with minimal or no task-specific training data.".to_string(),
                instruction: Some("Generate comprehensive embeddings for this extensive text".to_string()),
                normalize_embeddings: false,
            }
        ),
    ];

    for (test_name, args) in test_cases {
        println!("\n--- {test_name} ---");

        // Encode arguments
        let mut args_buf = Vec::new();
        args.encode(&mut args_buf).expect("Failed to encode args");

        // Create metadata
        let metadata = HashMap::from([
            ("test_case".to_string(), test_name.to_string()),
            ("timestamp".to_string(), chrono::Utc::now().to_rfc3339()),
        ]);

        // Execute plugin run
        let (result, result_metadata) = plugin.run(args_buf, metadata);

        match result {
            Ok(result_buf) => {
                // Decode result
                let embedding_result = EmbeddingLlmResult::decode(&result_buf[..])
                    .expect("Failed to decode embedding result");

                println!("  ✓ Embedding generation successful");
                println!(
                    "  - Number of embeddings: {}",
                    embedding_result.embeddings.len()
                );

                assert!(
                    !embedding_result.embeddings.is_empty(),
                    "Should generate at least one embedding"
                );

                // Check first embedding
                let first_embedding = &embedding_result.embeddings[0];
                println!(
                    "  - First embedding dimension: {}",
                    first_embedding.values.len()
                );
                assert!(
                    !first_embedding.values.is_empty(),
                    "Embedding values should not be empty"
                );

                // Check position information for all embeddings
                println!("  - Position information:");
                // Calculate actual text length including instruction if present
                // Note: instruction is concatenated with single newline
                let actual_text = if let Some(instruction) = &args.instruction {
                    format!("{}\n{}", instruction, args.text)
                } else {
                    args.text.clone()
                };
                let text_char_len = actual_text.chars().count() as u32;
                for (i, embedding) in embedding_result.embeddings.iter().enumerate() {
                    println!(
                        "    Embedding {}: begin_pos={}, end_pos={} (span: {} chars)",
                        i,
                        embedding.begin_position,
                        embedding.end_position,
                        embedding
                            .end_position
                            .saturating_sub(embedding.begin_position)
                    );

                    // First embedding should start at position 0
                    if i == 0 {
                        assert_eq!(
                            embedding.begin_position, 0,
                            "First embedding should start at position 0"
                        );
                    } else {
                        // Subsequent embeddings should have non-zero begin position
                        assert!(
                            embedding.begin_position > 0,
                            "Embedding {} should have non-zero begin position, got {}",
                            i,
                            embedding.begin_position
                        );
                    }

                    // End position should be greater than begin position
                    assert!(
                        embedding.end_position > embedding.begin_position,
                        "Embedding {} end position {} should be greater than begin position {}",
                        i,
                        embedding.end_position,
                        embedding.begin_position
                    );

                    // Position should be within reasonable text bounds
                    assert!(
                        embedding.end_position <= text_char_len,
                        "Embedding {} end position {} should not exceed text length {}",
                        i,
                        embedding.end_position,
                        text_char_len
                    );

                    // Check for reasonable position ordering in multi-embedding cases
                    if i > 0 {
                        let prev_embedding = &embedding_result.embeddings[i - 1];
                        // Current embedding should not start before previous one ends
                        // (allowing for overlap in sliding window)
                        assert!(embedding.begin_position >= prev_embedding.begin_position,
                               "Embedding {} begin position {} should not be before previous embedding's begin position {}",
                               i, embedding.begin_position, prev_embedding.begin_position);
                    }
                }

                // Special validation for multi-embedding results
                if embedding_result.embeddings.len() > 1 {
                    println!("  - Multi-embedding validation:");
                    println!(
                        "    Total embeddings: {}",
                        embedding_result.embeddings.len()
                    );
                    println!("    Text length: {text_char_len} characters");

                    // Verify that embeddings after the first have non-zero begin positions
                    for (i, embedding) in embedding_result.embeddings.iter().enumerate().skip(1) {
                        assert!(
                            embedding.begin_position > 0,
                            "Embedding {} should have non-zero begin position, got {}",
                            i,
                            embedding.begin_position
                        );
                        println!(
                            "    ✓ Embedding {}: begin_position = {} (non-zero)",
                            i, embedding.begin_position
                        );
                    }

                    let last_embedding = embedding_result.embeddings.last().unwrap();
                    assert!(
                        last_embedding.end_position >= text_char_len / 2,
                        "Last embedding should cover a reasonable portion of text"
                    );

                    println!("    ✓ Multi-embedding position validation passed");
                } else if test_name.contains("long text") || test_name.contains("Extremely long") {
                    // For tests designed to produce multiple embeddings, warn if only one was generated
                    println!(
                        "  - WARNING: Expected multiple embeddings for '{test_name}', but only got 1"
                    );
                    println!(
                        "    This may indicate that sliding window configuration needs adjustment"
                    );
                    println!(
                        "    Text length: {text_char_len} characters, Tokens may have been under the limit"
                    );
                }

                // Check model info
                if let Some(model_info) = &embedding_result.model_info {
                    println!("  - Model: {}", model_info.model_name);
                    println!("  - Dimension: {}", model_info.embedding_dimension);
                    println!("  - Data type: {}", model_info.dtype_used);

                    assert_eq!(
                        first_embedding.values.len(),
                        model_info.embedding_dimension as usize,
                        "Embedding dimension should match model info"
                    );
                } else {
                    panic!("Model info should be provided");
                }

                // Validate normalization if requested
                if args.normalize_embeddings {
                    let norm_squared: f32 = first_embedding.values.iter().map(|x| x * x).sum();
                    let norm = norm_squared.sqrt();
                    println!("  - L2 norm: {norm:.6}");

                    // Allow small floating point tolerance for normalized vectors
                    assert!(
                        (norm - 1.0).abs() < 1e-5,
                        "Normalized embedding should have L2 norm ≈ 1.0, got: {norm}"
                    );
                }

                // Check metadata (now used for tracing, not statistics)
                println!("  - Metadata entries: {}", result_metadata.len());
                // Metadata should be preserved for OpenTelemetry tracing context
                // Statistics are now recorded as span attributes instead of metadata

                // Validate embedding values (no NaN, no infinite values)
                for (i, embedding) in embedding_result.embeddings.iter().enumerate() {
                    for (j, &value) in embedding.values.iter().enumerate() {
                        assert!(
                            value.is_finite(),
                            "Embedding[{i}][{j}] should be finite, got: {value}"
                        );
                        assert!(!value.is_nan(), "Embedding[{i}][{j}] should not be NaN");
                    }
                }

                println!("  ✓ All validations passed");
            }
            Err(e) => {
                panic!("Embedding generation failed for {test_name}: {e}");
            }
        }
    }

    println!("\n=== Plugin Integration Test Completed Successfully ===");
}

#[tokio::test]
async fn test_plugin_error_handling() {
    println!("=== Plugin Error Handling Test ===");

    let mut plugin = EmbeddingLlmRunnerPlugin::new().expect("Failed to create plugin");

    // Test 1: Invalid model type
    println!("1. Testing invalid model type rejection...");
    let invalid_settings = EmbeddingLlmRunnerSettings {
        use_cpu: true,
        max_seq_length: 512,
        model_type: 999, // Invalid model type
        model_id: "test".to_string(),
        model_files: vec!["test.gguf".to_string()],
        tokenizer_model_id: None,
        dtype: Some(DType::F32 as i32),
        chunking_config: None,
        max_batch_size: Some(4),
        gpu_device: None,
    };

    let mut settings_buf = Vec::new();
    invalid_settings
        .encode(&mut settings_buf)
        .expect("Failed to encode settings");

    let load_result = plugin.load(settings_buf);
    assert!(load_result.is_err(), "Should reject invalid model type");
    println!("  ✓ Correctly rejected invalid model type");

    // Test 2: Empty model files
    println!("2. Testing empty model files rejection...");
    let empty_files_settings = EmbeddingLlmRunnerSettings {
        use_cpu: true,
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_id: "test".to_string(),
        model_files: vec![], // Empty model files
        tokenizer_model_id: None,
        dtype: Some(DType::F32 as i32),
        chunking_config: None,
        max_batch_size: Some(4),
        gpu_device: None,
    };

    let mut settings_buf = Vec::new();
    empty_files_settings
        .encode(&mut settings_buf)
        .expect("Failed to encode settings");

    let load_result = plugin.load(settings_buf);
    assert!(load_result.is_err(), "Should reject empty model files");
    println!("  ✓ Correctly rejected empty model files");

    // Test 3: Run without initialization
    println!("3. Testing run without initialization...");
    let args = EmbeddingArgs {
        text: "Test text".to_string(),
        instruction: None,
        normalize_embeddings: false,
    };

    let mut args_buf = Vec::new();
    args.encode(&mut args_buf).expect("Failed to encode args");

    let (result, _) = plugin.run(args_buf, HashMap::new());
    assert!(result.is_err(), "Should fail when embedder not initialized");
    println!("  ✓ Correctly failed when not initialized");

    println!("=== Error Handling Test Completed ===");
}

#[tokio::test]
async fn test_plugin_protobuf_schemas() {
    println!("=== Plugin Protobuf Schema Test ===");

    let plugin = EmbeddingLlmRunnerPlugin::new().expect("Failed to create plugin");

    // Test protobuf schema access
    let settings_proto = plugin.runner_settings_proto();
    let args_proto = plugin.job_args_proto();

    println!("Settings proto length: {} characters", settings_proto.len());
    println!("Args proto length: {} characters", args_proto.len());

    assert!(
        !settings_proto.is_empty(),
        "Settings proto should not be empty"
    );
    assert!(!args_proto.is_empty(), "Args proto should not be empty");

    // Verify key protobuf content
    assert!(
        settings_proto.contains("EmbeddingLlmRunnerSettings"),
        "Settings proto should contain main message"
    );
    assert!(
        settings_proto.contains("model_id"),
        "Settings proto should contain model_id field"
    );
    assert!(
        settings_proto.contains("max_batch_size"),
        "Settings proto should contain max_batch_size field"
    );

    assert!(
        args_proto.contains("EmbeddingArgs"),
        "Args proto should contain main message"
    );
    assert!(
        args_proto.contains("text"),
        "Args proto should contain text field"
    );
    assert!(
        args_proto.contains("instruction"),
        "Args proto should contain instruction field"
    );
    assert!(
        args_proto.contains("normalize_embeddings"),
        "Args proto should contain normalize_embeddings field"
    );

    println!("✓ Protobuf schemas are correctly accessible");
    println!("=== Protobuf Schema Test Completed ===");
}

#[tokio::test]
async fn test_plugin_cancellation_interface() {
    println!("=== Plugin Cancellation Interface Test ===");

    let plugin = EmbeddingLlmRunnerPlugin::new().expect("Failed to create plugin");

    // Test cancellation interface (currently not implemented)
    assert!(
        !plugin.cancel(),
        "Cancel should return false (not implemented)"
    );
    assert!(
        !plugin.is_canceled(),
        "is_canceled should return false (not implemented)"
    );

    println!("✓ Cancellation interface behaves as expected");
    println!("=== Cancellation Interface Test Completed ===");
}
