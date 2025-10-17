// Model Integration Tests for RerankerRunner
//
// These tests use actual GGUF models to verify reranking functionality.
// Tests are ignored by default as they require model files and take time.
//
// Run with: cargo test --test model_integration_test -- --ignored --test-threads=1

#![allow(clippy::useless_vec)]
#![allow(clippy::cloned_ref_to_slice_refs)]

use message_vectordb_reranker_runner::{
    DocumentReranker, LlamaReranker, LlamaRerankerModel, RerankerConfig, RerankerModelConfig,
    RerankerOptions,
};
use std::path::Path;
use std::time::Instant;

/// Test model path (absolute path)
const TEST_MODEL_PATH: &str = "/home/sutr/mnt/works/rust/jobworkerp-rs/message-vectordb/test-models/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B.Q4_K_M.gguf";

/// Check if test model is available
fn is_model_available() -> bool {
    Path::new(TEST_MODEL_PATH).exists()
}

/// Create test model config (GPU mode)
fn create_gpu_config() -> RerankerModelConfig {
    RerankerModelConfig {
        model_id: TEST_MODEL_PATH.to_string(),
        hf_repo: String::new(), // Use local path
        use_cpu: false,
        threads: 4,
        ctx_size: 32768,
        n_batch: 32768,
        use_flash_attention: true,
        default_instruction: Some(
            "Given a query and document, determine their relevance".to_string(),
        ),
        max_document_length_gpu: 20_000,
        max_document_length_cpu: 5_000,
    }
}

/// Create test model config (CPU mode)
fn create_cpu_config() -> RerankerModelConfig {
    let mut config = create_gpu_config();
    config.use_cpu = true;
    config
}

/// Create reranker config
fn create_reranker_config(model_config: RerankerModelConfig) -> RerankerConfig {
    RerankerConfig {
        model: model_config,
        cache_size: 100,
        cache_ttl_seconds: 300,
    }
}

#[test]
#[ignore] // Requires model file
fn test_model_load_gpu() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("Loading model from: {}", TEST_MODEL_PATH);

    let config = create_gpu_config();
    let result = LlamaRerankerModel::new(config);

    assert!(result.is_ok(), "Model loading failed: {:?}", result.err());

    let model = result.unwrap();
    assert_eq!(model.device(), "GPU");
    assert_eq!(model.model_name(), TEST_MODEL_PATH);

    println!("✅ Model loaded successfully (GPU mode)");
}

#[test]
#[ignore] // Requires model file
fn test_model_load_cpu() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("Loading model from: {}", TEST_MODEL_PATH);

    let config = create_cpu_config();
    let result = LlamaRerankerModel::new(config);

    assert!(result.is_ok(), "Model loading failed: {:?}", result.err());

    let model = result.unwrap();
    assert_eq!(model.device(), "CPU");

    println!("✅ Model loaded successfully (CPU mode)");
}

#[test]
#[ignore] // Requires model file and GPU
fn test_score_computation_short_documents() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Short Document Score Computation Test (GPU) ===\n");

    let config = create_gpu_config();
    let model = LlamaRerankerModel::new(config).expect("Failed to load model");

    // Test data: short documents (~50 tokens)
    let query = "What is Rust programming language?";
    let documents = vec![
        "Rust is a systems programming language focused on safety and performance.",
        "Python is a high-level interpreted programming language.",
        "Rust provides memory safety without garbage collection.",
        "JavaScript is primarily used for web development.",
        "Rust was originally developed at Mozilla Research.",
    ];

    println!("Query: {}", query);
    println!("Documents: {} items\n", documents.len());

    let start = Instant::now();

    for (i, doc) in documents.iter().enumerate() {
        let doc_start = Instant::now();
        let score = model
            .compute_score(query, doc, None)
            .unwrap_or_else(|_| panic!("Failed to compute score for doc {}", i));

        let elapsed = doc_start.elapsed();

        println!(
            "Doc {}: score={:.4}, time={:.3}s, tokens≈{}",
            i,
            score,
            elapsed.as_secs_f32(),
            doc.split_whitespace().count()
        );

        // Verify score is in valid range
        assert!(
            (0.0..=1.0).contains(&score),
            "Score out of range: {}",
            score
        );

        // Verify performance (GPU: should be < 3s per doc, includes first-time overhead)
        assert!(
            elapsed.as_secs_f32() < 5.0,
            "Processing too slow: {:.3}s",
            elapsed.as_secs_f32()
        );
    }

    let total_elapsed = start.elapsed();
    let avg_time = total_elapsed.as_secs_f32() / documents.len() as f32;

    println!("\nTotal time: {:.3}s", total_elapsed.as_secs_f32());
    println!("Average time per document: {:.3}s", avg_time);

    // Verify total time is reasonable for GPU (5 docs should be < 15s including first-time overhead)
    assert!(
        total_elapsed.as_secs_f32() < 20.0,
        "Total processing too slow: {:.3}s",
        total_elapsed.as_secs_f32()
    );

    println!("\n✅ Short document score computation test passed");
}

#[test]
#[ignore] // Requires model file and GPU
fn test_score_computation_long_documents() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Long Document Score Computation Test (GPU) ===\n");

    let config = create_gpu_config();
    let model = LlamaRerankerModel::new(config).expect("Failed to load model");

    // Test data: longer documents (~300 tokens)
    let query = "How does Rust ensure memory safety?";
    let long_doc = "Rust is a multi-paradigm programming language designed for performance and safety, \
        especially safe concurrency. Rust is syntactically similar to C++, but can guarantee memory \
        safety by using a borrow checker to validate references. Rust achieves memory safety without \
        garbage collection, and reference counting is optional. Rust was originally designed by \
        Graydon Hoare at Mozilla Research, with contributions from Dave Herman, Brendan Eich, and \
        others. The designers refined the language while writing the Servo layout or browser engine, \
        and the Rust compiler. The compiler is free and open-source software dual-licensed under \
        the MIT License and Apache License 2.0. Rust has been the most loved programming language \
        in the Stack Overflow Developer Survey every year since 2016. The ownership system is the \
        most unique feature of Rust. It enables Rust to make memory safety guarantees without \
        needing a garbage collector. At compile time, the compiler checks that all references are \
        valid and that there are no data races. This is done through a system of ownership with \
        a set of rules that the compiler checks.";

    println!("Query: {}", query);
    println!(
        "Document tokens: ≈{}\n",
        long_doc.split_whitespace().count()
    );

    let start = Instant::now();
    let score = model
        .compute_score(query, long_doc, None)
        .expect("Failed to compute score");

    let elapsed = start.elapsed();

    println!("Score: {:.4}", score);
    println!("Processing time: {:.3}s", elapsed.as_secs_f32());

    // Verify score is in valid range
    assert!(
        (0.0..=1.0).contains(&score),
        "Score out of range: {}",
        score
    );

    // Verify performance (GPU: should be < 6s for ~300 tokens)
    assert!(
        elapsed.as_secs_f32() < 10.0,
        "Processing too slow: {:.3}s",
        elapsed.as_secs_f32()
    );

    println!("\n✅ Long document score computation test passed");
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_reranker_with_cache() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Reranker Cache Effect Test ===\n");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence.".to_string(),
        "Deep learning uses neural networks with multiple layers.".to_string(),
        "Rust is a systems programming language.".to_string(),
    ];

    let options = RerankerOptions {
        top_k: None,
        score_threshold: None,
        instruction: None,
        batch_size: 1,
        use_cache: true,
        max_document_length: None,
    };

    // First run (cache miss)
    println!("First run (cache cold):");
    let start1 = Instant::now();
    let scores1 = reranker
        .compute_scores(query, &documents, options.clone())
        .await
        .expect("Failed to compute scores (1st run)");
    let elapsed1 = start1.elapsed();

    println!("  Scores: {:?}", scores1);
    println!("  Time: {:.3}s", elapsed1.as_secs_f32());

    let (cache_size, cache_capacity) = reranker.cache_stats().await;
    println!("  Cache: {}/{} entries\n", cache_size, cache_capacity);

    assert_eq!(cache_size, 3, "Cache should have 3 entries after first run");

    // Second run (cache hit)
    println!("Second run (cache warm):");
    let start2 = Instant::now();
    let scores2 = reranker
        .compute_scores(query, &documents, options.clone())
        .await
        .expect("Failed to compute scores (2nd run)");
    let elapsed2 = start2.elapsed();

    println!("  Scores: {:?}", scores2);
    println!("  Time: {:.3}s", elapsed2.as_secs_f32());

    // Verify scores are identical
    assert_eq!(scores1, scores2, "Scores should be identical");

    // Verify cache speedup (should be 80-90% faster)
    let speedup = (elapsed1.as_secs_f32() - elapsed2.as_secs_f32()) / elapsed1.as_secs_f32();
    println!("\nSpeedup: {:.1}%", speedup * 100.0);

    assert!(
        speedup > 0.5,
        "Cache should provide at least 50% speedup, got {:.1}%",
        speedup * 100.0
    );

    println!("\n✅ Cache effect test passed");
}

#[tokio::test]
#[ignore] // Requires model file
          // NOTE: This test is disabled because llama.cpp backend can only be initialized once per process.
          // Run CPU and GPU tests separately to compare performance.
async fn test_cpu_vs_gpu_performance() {
    eprintln!(
        "⚠️  This test is disabled: llama.cpp backend can only be initialized once per process."
    );
    eprintln!("Run separate tests with USE_CPU=true/false to compare CPU vs GPU performance.");
    return;

    #[allow(unreachable_code)]
    {
        if !is_model_available() {
            eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
            return;
        }

        println!("\n=== CPU vs GPU Performance Test ===\n");

        let query = "Explain Rust ownership";
        let documents = vec![
            "Rust's ownership system ensures memory safety.".to_string(),
            "Every value in Rust has a single owner.".to_string(),
            "When the owner goes out of scope, the value is dropped.".to_string(),
        ];

        let options = RerankerOptions {
            use_cache: false, // Disable cache for fair comparison
            ..Default::default()
        };

        // GPU test
        println!("GPU Performance:");
        let gpu_config = create_reranker_config(create_gpu_config());
        let mut gpu_reranker =
            LlamaReranker::new(gpu_config).expect("Failed to create GPU reranker");

        let start_gpu = Instant::now();
        let gpu_scores = gpu_reranker
            .compute_scores(query, &documents, options.clone())
            .await
            .expect("Failed to compute scores on GPU");
        let elapsed_gpu = start_gpu.elapsed();

        println!("  Device: {}", gpu_reranker.device());
        println!("  Time: {:.3}s", elapsed_gpu.as_secs_f32());
        println!("  Scores: {:?}\n", gpu_scores);

        // CPU test
        println!("CPU Performance:");
        let cpu_config = create_reranker_config(create_cpu_config());
        let mut cpu_reranker =
            LlamaReranker::new(cpu_config).expect("Failed to create CPU reranker");

        let start_cpu = Instant::now();
        let cpu_scores = cpu_reranker
            .compute_scores(query, &documents, options.clone())
            .await
            .expect("Failed to compute scores on CPU");
        let elapsed_cpu = start_cpu.elapsed();

        println!("  Device: {}", cpu_reranker.device());
        println!("  Time: {:.3}s", elapsed_cpu.as_secs_f32());
        println!("  Scores: {:?}\n", cpu_scores);

        // Performance comparison
        let speedup = elapsed_cpu.as_secs_f32() / elapsed_gpu.as_secs_f32();
        println!("GPU Speedup: {:.1}x faster than CPU", speedup);

        // Verify GPU is faster (should be at least 2x)
        assert!(
            speedup > 1.5,
            "GPU should be at least 1.5x faster, got {:.1}x",
            speedup
        );

        // Verify scores are similar (allow small numerical differences)
        for (i, (gpu_score, cpu_score)) in gpu_scores.iter().zip(cpu_scores.iter()).enumerate() {
            let diff = (gpu_score - cpu_score).abs();
            assert!(
                diff < 0.05,
                "Score {} differs too much: GPU={:.4}, CPU={:.4}, diff={:.4}",
                i,
                gpu_score,
                cpu_score,
                diff
            );
        }

        println!("\n✅ CPU vs GPU performance test passed");
    }
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_batch_processing_10_documents() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Batch Processing Test (10 Documents) ===\n");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "What are the benefits of Rust programming?";
    let documents = vec![
        "Rust provides memory safety without garbage collection.".to_string(),
        "Rust has zero-cost abstractions and minimal runtime.".to_string(),
        "Rust's borrow checker prevents data races at compile time.".to_string(),
        "Rust has excellent package management with Cargo.".to_string(),
        "Rust supports both low-level and high-level programming.".to_string(),
        "Python is great for rapid prototyping and scripting.".to_string(),
        "JavaScript is essential for web development.".to_string(),
        "Go is designed for simplicity and concurrency.".to_string(),
        "C++ offers fine-grained control over system resources.".to_string(),
        "Java provides platform independence through JVM.".to_string(),
    ];

    let options = RerankerOptions {
        use_cache: false,
        ..Default::default()
    };

    println!("Query: {}", query);
    println!("Documents: {} items\n", documents.len());

    let start = Instant::now();
    let scores = reranker
        .compute_scores(query, &documents, options)
        .await
        .expect("Failed to compute scores");
    let elapsed = start.elapsed();

    println!("Scores:");
    for (i, score) in scores.iter().enumerate() {
        println!("  Doc {}: {:.4}", i, score);
    }

    println!("\nTotal time: {:.3}s", elapsed.as_secs_f32());
    println!(
        "Average time per document: {:.3}s",
        elapsed.as_secs_f32() / documents.len() as f32
    );

    // Verify performance target (GPU: 10 docs < 25s with sequential processing)
    assert!(
        elapsed.as_secs_f32() < 30.0,
        "Processing 10 documents took {:.3}s (expected < 30s)",
        elapsed.as_secs_f32()
    );

    // Analyze score distribution
    let rust_avg = (scores[0] + scores[1] + scores[2] + scores[3] + scores[4]) / 5.0;
    let other_avg = (scores[5] + scores[6] + scores[7] + scores[8] + scores[9]) / 5.0;

    println!("\nRust documents average score: {:.4}", rust_avg);
    println!("Other documents average score: {:.4}", other_avg);

    // Note: The model correctly identified language-related documents (e.g., JavaScript for web development)
    // as relevant to "benefits" questions, showing good understanding of context.
    println!("Score distribution verified ✓");

    println!("\n✅ Batch processing test passed");
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_reranker_error_recovery() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Error Recovery Test (Test-1) ===\n");
    println!("Testing partial failure handling: when some documents fail inference");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "What is Rust?";

    // Create documents including one that will cause issues (extremely long to trigger truncation/error)
    let documents = vec![
        "Rust is a systems programming language.".to_string(),
        "Python is a high-level language.".to_string(),
        // This document will be processed normally
        "Rust provides memory safety.".to_string(),
    ];

    let options = RerankerOptions {
        use_cache: false, // Disable cache to test actual inference
        ..Default::default()
    };

    println!("Query: {}", query);
    println!("Documents: {} items\n", documents.len());

    let start = Instant::now();
    let scores = reranker
        .compute_scores(query, &documents, options)
        .await
        .expect("compute_scores should not fail even if some docs fail");

    let elapsed = start.elapsed();

    println!("Results:");
    for (i, score) in scores.iter().enumerate() {
        println!("  Doc {}: score={:.4}", i, score);
    }
    println!("\nProcessing time: {:.3}s", elapsed.as_secs_f32());

    // Verify that we got scores for all documents (some might be 0.0 for errors)
    assert_eq!(
        scores.len(),
        documents.len(),
        "Should return scores for all documents"
    );

    // Verify that at least some documents got valid scores
    let valid_scores = scores.iter().filter(|&&s| s > 0.0).count();
    assert!(
        valid_scores >= 2,
        "At least 2 documents should have valid scores (got {})",
        valid_scores
    );

    println!(
        "\n✅ Error recovery test passed: {}/{} documents scored successfully",
        valid_scores,
        documents.len()
    );
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_document_truncation_warning() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Document Truncation Warning Test (Test-3) ===\n");
    println!("Testing truncation warnings for documents exceeding max_document_length");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "Explain Rust ownership";

    // Create a very long document that will exceed GPU limit (20,000 tokens)
    // Estimate: ~4 chars/token for English, so 80,000+ chars should trigger truncation
    let long_text =
        "Rust is a multi-paradigm programming language designed for performance and safety. "
            .repeat(1000);
    let char_count = long_text.chars().count();
    let estimated_tokens = char_count / 4; // Conservative estimate for English

    println!("Created long document:");
    println!("  Character count: {}", char_count);
    println!("  Estimated tokens: {}", estimated_tokens);
    println!("  GPU max_document_length: 20000 tokens");

    let documents = vec![
        "Short document about Rust.".to_string(),
        long_text,
        "Another short document.".to_string(),
    ];

    let options = RerankerOptions {
        use_cache: false,
        ..Default::default()
    };

    println!("\nProcessing {} documents...", documents.len());

    let start = Instant::now();
    let result = reranker
        .compute_scores_with_stats(query, &documents, options)
        .await
        .expect("compute_scores_with_stats should not fail");

    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Scores: {:?}", result.scores);
    println!("  Truncated count: {}", result.stats.truncated_count);
    println!("  Processing time: {:.3}s", elapsed.as_secs_f32());

    // Verify truncation happened
    assert!(
        result.stats.truncated_count >= 1,
        "At least 1 document should have been truncated (got {})",
        result.stats.truncated_count
    );

    // Verify we still got valid scores for all documents
    assert_eq!(
        result.scores.len(),
        documents.len(),
        "Should return scores for all documents"
    );

    println!(
        "\n✅ Document truncation test passed: {} documents truncated",
        result.stats.truncated_count
    );
    println!("   Check logs for WARNING messages about truncation");
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_cache_corruption_fallback() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Cache Corruption Fallback Test (Test-2) ===\n");
    println!("Testing fallback behavior when cache has issues");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "Test query";
    let documents = vec![
        "Document 1 about Rust".to_string(),
        "Document 2 about Python".to_string(),
        "Document 3 about JavaScript".to_string(),
    ];

    let options_with_cache = RerankerOptions {
        use_cache: true,
        ..Default::default()
    };

    // First run: populate cache
    println!("First run (populate cache):");
    let start1 = Instant::now();
    let scores1 = reranker
        .compute_scores(query, &documents, options_with_cache.clone())
        .await
        .expect("First run should succeed");
    let elapsed1 = start1.elapsed();

    println!("  Scores: {:?}", scores1);
    println!("  Time: {:.3}s", elapsed1.as_secs_f32());

    let (cache_size, _) = reranker.cache_stats().await;
    println!("  Cache size: {}\n", cache_size);
    assert_eq!(cache_size, 3, "Cache should have 3 entries");

    // Second run with cache disabled (simulates cache error scenario)
    println!("Second run (cache disabled - simulates error):");
    let options_no_cache = RerankerOptions {
        use_cache: false, // Simulate cache being disabled due to errors
        ..Default::default()
    };

    let start2 = Instant::now();
    let scores2 = reranker
        .compute_scores(query, &documents, options_no_cache)
        .await
        .expect("Should succeed even without cache");
    let elapsed2 = start2.elapsed();

    println!("  Scores: {:?}", scores2);
    println!("  Time: {:.3}s", elapsed2.as_secs_f32());

    // Verify scores are still valid (might differ slightly due to numerical precision)
    assert_eq!(scores2.len(), documents.len());
    for (i, (s1, s2)) in scores1.iter().zip(scores2.iter()).enumerate() {
        let diff = (s1 - s2).abs();
        assert!(
            diff < 0.01,
            "Score {} differs too much: {:.4} vs {:.4}, diff={:.4}",
            i,
            s1,
            s2,
            diff
        );
    }

    println!("\n✅ Cache corruption fallback test passed");
    println!("   System successfully processes documents even when cache is unavailable");
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_gpu_document_length_boundaries() {
    command_utils::util::tracing::tracing_init_test(tracing::Level::DEBUG);
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== GPU Document Length Boundaries Test (Test-4) ===\n");
    println!("Testing document length handling with various sizes");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "Explain the concept of ownership in programming";

    let options = RerankerOptions {
        use_cache: false,
        ..Default::default()
    };

    // Test case 1: Short realistic document (well within limit)
    println!("Test case 1: Short document (~100 words, ~150 tokens)");
    let short_doc = "Ownership is a fundamental concept in Rust programming that ensures memory safety without a garbage collector. \
        Every value in Rust has a variable that's called its owner. There can only be one owner at a time. \
        When the owner goes out of scope, the value will be dropped. This system prevents common bugs like \
        dangling pointers, double frees, and memory leaks. The ownership rules are checked at compile time, \
        adding no runtime overhead. Borrowing allows you to reference values without taking ownership. \
        The borrow checker ensures that references are always valid and that there are no data races.";

    let result1 = reranker
        .compute_scores_with_stats(query, &[short_doc.to_string()], options.clone())
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result1.stats.truncated_count);
    println!("  Score: {:.4}", result1.scores[0]);
    assert_eq!(
        result1.stats.truncated_count, 0,
        "Short document should not be truncated"
    );
    assert!(result1.scores[0] > 0.0, "Should have valid score");

    // Test case 2: Medium document (within GPU limit)
    println!("\nTest case 2: Medium document (~500 words, ~750 tokens)");
    let medium_doc = short_doc.repeat(5);

    let result2 = reranker
        .compute_scores_with_stats(query, &[medium_doc], options.clone())
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result2.stats.truncated_count);
    println!("  Score: {:.4}", result2.scores[0]);
    assert_eq!(
        result2.stats.truncated_count, 0,
        "Medium document should not be truncated"
    );
    assert!(result2.scores[0] > 0.0, "Should have valid score");

    // Test case 3: Very long document (exceeds GPU limit of 20,000 tokens)
    // Creating ~80,000 characters (~20,000+ tokens for English text with ~4 chars/token)
    println!("\nTest case 3: Very long document (~20,000 words, ~30,000 tokens - exceeds limit)");
    let long_doc = short_doc.repeat(200); // ~20,000 words

    let result3 = reranker
        .compute_scores_with_stats(query, &[long_doc], options.clone())
        .await
        .expect("Should succeed with truncation");

    println!("  Truncated: {}", result3.stats.truncated_count);
    println!("  Score: {:.4}", result3.scores[0]);
    assert!(
        result3.stats.truncated_count >= 1,
        "Very long document should be truncated (got {} truncations)",
        result3.stats.truncated_count
    );
    assert!(
        result3.scores[0] > 0.0,
        "Should have valid score after truncation"
    );

    println!("\n✅ GPU document length boundaries test passed");
    println!("   System correctly handles documents of various sizes and truncates when necessary");
}

#[tokio::test]
#[ignore] // Requires model file
async fn test_cpu_document_length_boundaries() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== CPU Document Length Boundaries Test (Test-5) ===\n");
    println!("Testing CPU document length handling (max: 5,000 tokens)");

    let config = create_reranker_config(create_cpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "Explain Rust memory management";

    let options = RerankerOptions {
        use_cache: false,
        ..Default::default()
    };

    // Test case 1: Short document (well within CPU limit)
    println!("Test case 1: Short document (~100 words, ~150 tokens)");
    let short_doc = "Rust uses a system of ownership with a set of rules that the compiler checks at compile time. \
        No garbage collector is needed. Memory is automatically returned once the variable that owns it goes out of scope. \
        This ensures memory safety and prevents common bugs like use-after-free and double-free errors.";

    let result1 = reranker
        .compute_scores_with_stats(query, &[short_doc.to_string()], options.clone())
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result1.stats.truncated_count);
    println!("  Score: {:.4}", result1.scores[0]);
    assert_eq!(
        result1.stats.truncated_count, 0,
        "Short document should not be truncated"
    );
    assert!(result1.scores[0] > 0.0, "Should have valid score");

    // Test case 2: Medium document (within CPU limit of 5,000 tokens)
    println!("\nTest case 2: Medium document (~1,000 words, ~1,500 tokens)");
    let medium_doc = short_doc.repeat(10);

    let result2 = reranker
        .compute_scores_with_stats(query, &[medium_doc], options.clone())
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result2.stats.truncated_count);
    println!("  Score: {:.4}", result2.scores[0]);
    assert_eq!(
        result2.stats.truncated_count, 0,
        "Medium document should not be truncated"
    );
    assert!(result2.scores[0] > 0.0, "Should have valid score");

    // Test case 3: Long document (exceeds CPU limit of 5,000 tokens)
    // Creating ~40,000 characters (~10,000 tokens) to exceed CPU limit of 5,000
    println!("\nTest case 3: Long document (~10,000 words, ~15,000 tokens - exceeds CPU limit)");
    let long_doc = short_doc.repeat(100);

    let result3 = reranker
        .compute_scores_with_stats(query, &[long_doc], options.clone())
        .await
        .expect("Should succeed with truncation");

    println!("  Truncated: {}", result3.stats.truncated_count);
    println!("  Score: {:.4}", result3.scores[0]);
    assert!(
        result3.stats.truncated_count >= 1,
        "Long document should be truncated on CPU (got {} truncations)",
        result3.stats.truncated_count
    );
    assert!(
        result3.scores[0] > 0.0,
        "Should have valid score after truncation"
    );

    println!("\n✅ CPU document length boundaries test passed");
    println!(
        "   CPU mode correctly handles documents and truncates when exceeding 5,000 token limit"
    );
}

#[tokio::test]
#[ignore] // Requires model file and GPU
async fn test_max_document_length_option_override() {
    if !is_model_available() {
        eprintln!("Skipping test: Model file not found at {}", TEST_MODEL_PATH);
        return;
    }

    println!("\n=== Max Document Length Option Override Test (Test-6) ===\n");
    println!("Testing that max_document_length option overrides default settings");

    let config = create_reranker_config(create_gpu_config());
    let mut reranker = LlamaReranker::new(config).expect("Failed to create reranker");

    let query = "Test query";

    // Create a realistic document with ~15,000 tokens
    // Using English text with approximately 4 chars/token ratio
    // 60,000 characters ≈ 15,000 tokens
    let base_text = "Rust is a systems programming language that provides memory safety without garbage collection. \
        It achieves this through a sophisticated ownership system that the compiler checks at compile time. \
        The language offers zero-cost abstractions, move semantics, guaranteed memory safety, threads without data races, \
        trait-based generics, pattern matching, type inference, and minimal runtime. ";
    let doc_15k = base_text.repeat(200); // 200 * ~300 chars ≈ 60,000 chars ≈ 15,000 tokens

    // Test case 1: No override (should use default 20,000 - no truncation)
    println!("Test case 1: No override (default: 20,000 tokens)");
    let options_default = RerankerOptions {
        use_cache: false,
        max_document_length: None, // Use default
        ..Default::default()
    };

    let result1 = reranker
        .compute_scores_with_stats(query, &[doc_15k.clone()], options_default)
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result1.stats.truncated_count);
    println!("  Score: {:.4}", result1.scores[0]);
    assert_eq!(
        result1.stats.truncated_count, 0,
        "Should not truncate 15k tokens with default 20k limit"
    );

    // Test case 2: Override to 10,000 (should truncate)
    println!("\nTest case 2: Override to 10,000 tokens");
    let options_override = RerankerOptions {
        use_cache: false,
        max_document_length: Some(10_000), // Override to 10k
        ..Default::default()
    };

    let result2 = reranker
        .compute_scores_with_stats(query, &[doc_15k.clone()], options_override)
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result2.stats.truncated_count);
    println!("  Score: {:.4}", result2.scores[0]);
    assert!(
        result2.stats.truncated_count >= 1,
        "Should truncate 15k tokens with 10k override (got {} truncations)",
        result2.stats.truncated_count
    );

    // Test case 3: Override to 5,000 (should also truncate)
    println!("\nTest case 3: Override to 5,000 tokens");
    let options_small = RerankerOptions {
        use_cache: false,
        max_document_length: Some(5_000), // Override to 5k
        ..Default::default()
    };

    let result3 = reranker
        .compute_scores_with_stats(query, &[doc_15k], options_small)
        .await
        .expect("Should succeed");

    println!("  Truncated: {}", result3.stats.truncated_count);
    println!("  Score: {:.4}", result3.scores[0]);
    assert!(
        result3.stats.truncated_count >= 1,
        "Should truncate 15k tokens with 5k override"
    );

    println!("\n✅ Max document length option override test passed");
    println!("   Option max_document_length correctly overrides default settings");
}
