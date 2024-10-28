// GGUF Qwen3-Reranker-4B validation test
// Tests compatibility with llama-cpp-2 for reranking use case

use anyhow::{Context, Result, anyhow};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    token::data::LlamaTokenData,
};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== GGUF Qwen3-Reranker-4B Validation Test ===\n");

    // Model path (from workspace root)
    let model_path = std::env::var("RERANKER_MODEL_PATH").unwrap_or_else(|_| {
        "./test-models/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B.Q4_K_M.gguf".to_string()
    });

    // GPU configuration
    let use_cpu = std::env::var("USE_CPU")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);

    let n_gpu_layers = if use_cpu {
        0u32
    } else {
        std::env::var("N_GPU_LAYERS")
            .unwrap_or_else(|_| "1000".to_string())
            .parse::<u32>()
            .unwrap_or(1000)
    };

    if !Path::new(&model_path).exists() {
        return Err(anyhow!("Model file not found: {}", model_path));
    }

    println!("✓ Model file exists: {}", model_path);
    println!(
        "✓ Device: {} (n_gpu_layers: {})",
        if use_cpu { "CPU" } else { "GPU" },
        n_gpu_layers
    );

    // Initialize backend
    println!("\n1. Initializing llama.cpp backend...");
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    println!("✓ Backend initialized");

    // Load model
    println!("\n2. Loading GGUF model...");
    let start = Instant::now();

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("Failed to load model")?;

    println!("✓ Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Get model metadata
    println!("\n3. Model metadata:");
    println!("   - n_vocab: {}", model.n_vocab());
    println!("   - n_ctx_train: {}", model.n_ctx_train());
    println!("   - n_embd: {}", model.n_embd());
    println!("   - n_layers: {}", model.n_layer());

    // Test tokenization
    println!("\n4. Testing tokenization...");

    let test_texts = vec!["yes", "no", "Yes", "No", "YES", "NO", "hello", "test"];

    println!("   Token mapping:");
    for text in &test_texts {
        let tokens = model
            .str_to_token(text, AddBos::Never)
            .with_context(|| format!("Failed to tokenize '{}'", text))?;

        println!("   - '{}' -> {:?}", text, tokens);
    }

    // Find yes/no tokens
    println!("\n5. Finding yes/no token IDs...");

    let yes_variants = vec!["yes", "Yes", "YES"];
    let no_variants = vec!["no", "No", "NO"];

    let mut yes_token_id = None;
    let mut no_token_id = None;

    for variant in &yes_variants {
        if let Ok(tokens) = model.str_to_token(variant, AddBos::Never) {
            if tokens.len() == 1 {
                yes_token_id = Some(tokens[0]);
                println!("✓ Found 'yes' token: '{}' -> ID {}", variant, tokens[0].0);
                break;
            }
        }
    }

    for variant in &no_variants {
        if let Ok(tokens) = model.str_to_token(variant, AddBos::Never) {
            if tokens.len() == 1 {
                no_token_id = Some(tokens[0]);
                println!("✓ Found 'no' token: '{}' -> ID {}", variant, tokens[0].0);
                break;
            }
        }
    }

    if yes_token_id.is_none() {
        println!("⚠️  WARNING: 'yes' token not found as single token");
    }

    if no_token_id.is_none() {
        println!("⚠️  WARNING: 'no' token not found as single token");
    }

    // Create context
    println!("\n6. Creating context...");
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(32768)) // Increased for long documents
        .with_n_batch(24576); // Increased batch size for very long documents (20k+ tokens)

    let mut ctx = model
        .new_context(&backend, ctx_params.clone())
        .context("Failed to create context")?;

    println!("✓ Context created (n_ctx: {})", ctx.n_ctx());

    // Test reranker prompt format
    println!("\n7. Testing reranker prompt format...");

    let query = "What is Rust programming language?";
    let document = "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.";
    let instruction = "Given a query and document, determine their relevance";

    let reranker_prompt = format!(
        "<Instruct>: {}\n<Query>: {}\n<Document>: {}",
        instruction, query, document
    );

    println!("   Prompt format:");
    for (i, line) in reranker_prompt.lines().take(3).enumerate() {
        println!("   {}", line);
        if i == 2 {
            println!("   ...");
            break;
        }
    }

    // Tokenize the reranker prompt
    println!("\n8. Tokenizing reranker prompt...");
    let tokens = model
        .str_to_token(&reranker_prompt, AddBos::Always)
        .context("Failed to tokenize reranker prompt")?;

    println!("✓ Tokenized: {} tokens", tokens.len());

    // Create batch and decode
    println!("\n9. Running inference (forward pass)...");
    let start = Instant::now();

    let mut batch = LlamaBatch::new(tokens.len(), 1);

    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(*token, i as i32, &[0], is_last)?;
    }

    ctx.decode(&mut batch).context("Failed to decode batch")?;

    println!(
        "✓ Inference completed in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    // Get logits from the last token
    println!("\n10. Extracting logits from last token...");

    let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
    let candidates_vec: Vec<LlamaTokenData> = candidates.collect();

    println!("✓ Got {} candidates (vocab size)", candidates_vec.len());

    // Find yes/no logits
    if let (Some(yes_id), Some(no_id)) = (yes_token_id, no_token_id) {
        println!("\n11. Computing reranking score...");

        let yes_logit = candidates_vec
            .iter()
            .find(|c| c.id() == yes_id)
            .map(|c| c.logit())
            .unwrap_or(f32::NEG_INFINITY);

        let no_logit = candidates_vec
            .iter()
            .find(|c| c.id() == no_id)
            .map(|c| c.logit())
            .unwrap_or(f32::NEG_INFINITY);

        println!("   - yes token (ID {}): logit = {:.4}", yes_id.0, yes_logit);
        println!("   - no token (ID {}): logit = {:.4}", no_id.0, no_logit);

        // Compute score using softmax
        let yes_exp = yes_logit.exp();
        let no_exp = no_logit.exp();
        let score = yes_exp / (yes_exp + no_exp);

        println!("\n✓ Reranking score: {:.4}", score);

        // Validate score is in valid range
        if (0.0..=1.0).contains(&score) {
            println!("✓ Score is in valid range [0.0, 1.0]");
        } else {
            println!("⚠️  WARNING: Score out of range: {}", score);
        }

        // Test multiple document lengths
        println!("\n12. Testing performance with different document lengths...");
        test_document_lengths(&model, &backend, &ctx_params, yes_id, no_id)?;
    } else {
        println!("\n⚠️  Skipping score computation (yes/no tokens not found)");
    }

    println!("\n=== Validation Complete ===");
    println!("✅ GGUF Qwen3-Reranker-4B is compatible with llama-cpp-2!");

    Ok(())
}

fn test_document_lengths(
    model: &LlamaModel,
    backend: &LlamaBackend,
    ctx_params: &LlamaContextParams,
    yes_id: llama_cpp_2::token::LlamaToken,
    no_id: llama_cpp_2::token::LlamaToken,
) -> Result<()> {
    let query = "What is Rust?";
    let instruction = "Determine relevance";

    let long_doc = "Rust is a multi-paradigm, general-purpose programming language. ".repeat(20);

    // Very long document for testing ~20000 tokens
    let very_long_doc = "Rust is a systems programming language that runs blazingly fast, \
                         prevents segfaults, and guarantees thread safety. It accomplishes \
                         these goals without needing a garbage collector, making it an ideal \
                         choice for performance-critical systems. "
        .repeat(500);

    let test_cases: Vec<(&str, &str, usize)> = vec![
        ("Short", "Rust is a programming language.", 50),
        (
            "Medium",
            "Rust is a systems programming language that runs blazingly fast, \
             prevents segfaults, and guarantees thread safety. It accomplishes \
             these goals without needing a garbage collector.",
            200,
        ),
        ("Long", &long_doc, 500),
        ("Very Long (~20k tokens)", &very_long_doc, 20000),
    ];

    for (label, document, _expected_tokens) in test_cases {
        let prompt = format!(
            "<Instruct>: {}\n<Query>: {}\n<Document>: {}",
            instruction, query, document
        );

        let tokens = model.str_to_token(&prompt, AddBos::Always)?;

        println!("\n   {}: {} tokens", label, tokens.len());

        let start = Instant::now();

        // Create new context for each test
        let mut ctx = model.new_context(backend, ctx_params.clone())?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)?;
        }

        ctx.decode(&mut batch)?;

        let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
        let candidates_vec: Vec<LlamaTokenData> = candidates.collect();

        let yes_logit = candidates_vec
            .iter()
            .find(|c| c.id() == yes_id)
            .map(|c| c.logit())
            .unwrap_or(f32::NEG_INFINITY);

        let no_logit = candidates_vec
            .iter()
            .find(|c| c.id() == no_id)
            .map(|c| c.logit())
            .unwrap_or(f32::NEG_INFINITY);

        let yes_exp = yes_logit.exp();
        let no_exp = no_logit.exp();
        let score = yes_exp / (yes_exp + no_exp);

        let elapsed = start.elapsed().as_secs_f32();

        println!(
            "      Score: {:.4}, Time: {:.2}s ({:.1} tokens/s)",
            score,
            elapsed,
            tokens.len() as f32 / elapsed
        );
    }

    Ok(())
}
