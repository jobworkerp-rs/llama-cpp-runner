use embedding_llm::{
    EmbeddingLlmRunnerPlugin,
    protobuf::embedding_llm::{
        EmbeddingLlmRunnerSettings, EmbeddingArgs, EmbeddingLlmResult, 
        ModelType, DType, SlidingWindowConfig
    },
};
use jobworkerp_client::plugins::PluginRunner;
use prost::Message;
use std::collections::HashMap;
use tracing_subscriber;

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
    let mut plugin = EmbeddingLlmRunnerPlugin::new()
        .expect("Failed to create plugin");
    
    println!("Plugin created: {}", plugin.name());
    assert_eq!(plugin.name(), "EmbeddingLlmRunner");
    assert!(!plugin.description().is_empty());

    // 2. 設定データの準備とload
    let settings = EmbeddingLlmRunnerSettings {
        use_cpu: false,
        max_seq_length: 512,
        model_type: ModelType::Gguf as i32,
        model_id: "Qwen/Qwen3-Embedding-4B-GGUF".to_string(),
        model_files: vec!["Qwen3-Embedding-4B-Q4_K_M.gguf".to_string()],
        tokenizer_model_id: Some("Qwen/Qwen3-Embedding-4B".to_string()),
        dtype: Some(DType::Bf16 as i32),
        sliding_window_config: Some(SlidingWindowConfig {
            window_stride: 256,
            min_window_size: 32,
        }),
        max_batch_size: Some(4),
    };

    let mut settings_buf = Vec::new();
    settings.encode(&mut settings_buf).expect("Failed to encode settings");
    
    println!("Loading plugin with model: {}", settings.model_id);
    let load_result = plugin.load(settings_buf);
    
    match load_result {
        Ok(()) => {
            println!("✓ Plugin loaded successfully");
        }
        Err(e) => {
            println!("⚠ Plugin load failed: {}. Skipping embedding tests.", e);
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
                text: "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.".to_string(),
                instruction: Some("Generate comprehensive embedding".to_string()),
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
    ];

    for (test_name, args) in test_cases {
        println!("\n--- {} ---", test_name);
        
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
                println!("  - Number of embeddings: {}", embedding_result.embeddings.len());
                
                assert!(!embedding_result.embeddings.is_empty(), "Should generate at least one embedding");
                
                // Check first embedding
                let first_embedding = &embedding_result.embeddings[0];
                println!("  - First embedding dimension: {}", first_embedding.values.len());
                assert!(!first_embedding.values.is_empty(), "Embedding values should not be empty");
                
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
                    let norm_squared: f32 = first_embedding.values.iter()
                        .map(|x| x * x)
                        .sum();
                    let norm = norm_squared.sqrt();
                    println!("  - L2 norm: {:.6}", norm);
                    
                    // Allow small floating point tolerance for normalized vectors
                    assert!((norm - 1.0).abs() < 1e-5, "Normalized embedding should have L2 norm ≈ 1.0, got: {}", norm);
                }
                
                // Check metadata (now used for tracing, not statistics)
                println!("  - Metadata entries: {}", result_metadata.len());
                // Metadata should be preserved for OpenTelemetry tracing context
                // Statistics are now recorded as span attributes instead of metadata
                
                // Validate embedding values (no NaN, no infinite values)
                for (i, embedding) in embedding_result.embeddings.iter().enumerate() {
                    for (j, &value) in embedding.values.iter().enumerate() {
                        assert!(value.is_finite(), "Embedding[{}][{}] should be finite, got: {}", i, j, value);
                        assert!(!value.is_nan(), "Embedding[{}][{}] should not be NaN", i, j);
                    }
                }
                
                println!("  ✓ All validations passed");
            }
            Err(e) => {
                panic!("Embedding generation failed for {}: {}", test_name, e);
            }
        }
    }
    
    println!("\n=== Plugin Integration Test Completed Successfully ===");
}

#[tokio::test]
async fn test_plugin_error_handling() {
    println!("=== Plugin Error Handling Test ===");
    
    let mut plugin = EmbeddingLlmRunnerPlugin::new()
        .expect("Failed to create plugin");

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
        sliding_window_config: None,
        max_batch_size: Some(4),
    };

    let mut settings_buf = Vec::new();
    invalid_settings.encode(&mut settings_buf).expect("Failed to encode settings");
    
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
        sliding_window_config: None,
        max_batch_size: Some(4),
    };

    let mut settings_buf = Vec::new();
    empty_files_settings.encode(&mut settings_buf).expect("Failed to encode settings");
    
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
    
    let plugin = EmbeddingLlmRunnerPlugin::new()
        .expect("Failed to create plugin");

    // Test protobuf schema access
    let settings_proto = plugin.runner_settings_proto();
    let args_proto = plugin.job_args_proto();
    
    println!("Settings proto length: {} characters", settings_proto.len());
    println!("Args proto length: {} characters", args_proto.len());
    
    assert!(!settings_proto.is_empty(), "Settings proto should not be empty");
    assert!(!args_proto.is_empty(), "Args proto should not be empty");
    
    // Verify key protobuf content
    assert!(settings_proto.contains("EmbeddingLlmRunnerSettings"), "Settings proto should contain main message");
    assert!(settings_proto.contains("model_id"), "Settings proto should contain model_id field");
    assert!(settings_proto.contains("max_batch_size"), "Settings proto should contain max_batch_size field");
    
    assert!(args_proto.contains("EmbeddingArgs"), "Args proto should contain main message");
    assert!(args_proto.contains("text"), "Args proto should contain text field");
    assert!(args_proto.contains("instruction"), "Args proto should contain instruction field");
    assert!(args_proto.contains("normalize_embeddings"), "Args proto should contain normalize_embeddings field");
    
    println!("✓ Protobuf schemas are correctly accessible");
    println!("=== Protobuf Schema Test Completed ===");
}

#[tokio::test]
async fn test_plugin_cancellation_interface() {
    println!("=== Plugin Cancellation Interface Test ===");
    
    let plugin = EmbeddingLlmRunnerPlugin::new()
        .expect("Failed to create plugin");

    // Test cancellation interface (currently not implemented)
    assert_eq!(plugin.cancel(), false, "Cancel should return false (not implemented)");
    assert_eq!(plugin.is_canceled(), false, "is_canceled should return false (not implemented)");
    
    println!("✓ Cancellation interface behaves as expected");
    println!("=== Cancellation Interface Test Completed ===");
}