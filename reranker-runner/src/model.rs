use crate::config::RerankerModelConfig;
use crate::error::RerankerError;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    token::{LlamaToken, data::LlamaTokenData},
};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

/// llama.cpp-based reranker model wrapper
///
/// This struct wraps the llama.cpp model and provides high-level
/// reranking functionality using the Qwen3-Reranker-4B-GGUF model.
pub struct LlamaRerankerModel {
    /// llama.cpp backend (shared)
    #[allow(dead_code)]
    backend: Arc<LlamaBackend>,

    /// Loaded GGUF model
    model: LlamaModel,

    /// Model configuration
    config: RerankerModelConfig,

    /// Token ID for "yes" (reranker output)
    yes_token_id: LlamaToken,

    /// Token ID for "no" (reranker output)
    no_token_id: LlamaToken,

    /// Device string ("CPU" or "GPU")
    device: String,
}

impl LlamaRerankerModel {
    /// Create a new reranker model
    ///
    /// # Arguments
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Initialized model ready for inference
    ///
    /// # Errors
    /// - Model file not found
    /// - Model loading failed
    /// - Yes/No tokens not found in vocabulary
    pub fn new(config: RerankerModelConfig) -> Result<Self, RerankerError> {
        // Initialize backend
        let mut backend = LlamaBackend::init().map_err(|e| {
            RerankerError::model_load_failed(format!("Backend init failed: {:?}", e))
        })?;

        // Disable llama.cpp logs
        backend.void_logs();

        let backend = Arc::new(backend);

        // Determine device (based on configuration)
        // Note: Currently trusts use_cpu flag. Future improvement could check
        // actual GPU availability from llama.cpp backend capabilities.
        let device = if config.use_cpu {
            "CPU".to_string()
        } else {
            "GPU".to_string()
        };

        // Load model (from HF or local path)
        let model_path = if !config.hf_repo.is_empty() && config.hf_repo != config.model_id {
            // Try to load from HuggingFace Hub
            Self::download_from_hf(&config)?
        } else {
            // Use local path
            config.model_id.clone()
        };

        if !Path::new(&model_path).exists() {
            return Err(RerankerError::ModelNotFound { path: model_path });
        }

        // Set GPU layers (0 for CPU mode, high value for GPU mode)
        let n_gpu_layers = if config.use_cpu {
            0
        } else {
            1000 // Use all available layers on GPU
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| RerankerError::model_load_failed(format!("Model load failed: {:?}", e)))?;

        // Find yes/no token IDs
        let yes_token_id = Self::find_token_id(&model, &["yes", "Yes", "YES"])?;
        let no_token_id = Self::find_token_id(&model, &["no", "No", "NO"])?;

        Ok(Self {
            backend,
            model,
            config,
            yes_token_id,
            no_token_id,
            device,
        })
    }

    /// Download model from HuggingFace Hub
    fn download_from_hf(config: &RerankerModelConfig) -> Result<String, RerankerError> {
        if config.hf_repo.is_empty() {
            return Err(RerankerError::config_error("HF repo not specified"));
        }

        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .map_err(|e| RerankerError::HfHubError(format!("API init failed: {:?}", e)))?;

        let repo_api = api.model(config.hf_repo.clone());

        // Download the model file (assumes model_id is the filename)
        let local_path = repo_api
            .get(&config.model_id)
            .map_err(|e| RerankerError::HfHubError(format!("Download failed: {:?}", e)))?;

        Ok(local_path.to_string_lossy().to_string())
    }

    /// Find token ID for a list of variants
    ///
    /// Tries each variant and returns the first one found as a single token
    fn find_token_id(model: &LlamaModel, variants: &[&str]) -> Result<LlamaToken, RerankerError> {
        for variant in variants {
            if let Ok(tokens) = model.str_to_token(variant, AddBos::Never) {
                if tokens.len() == 1 {
                    return Ok(tokens[0]);
                }
            }
        }

        Err(RerankerError::TokenNotFound {
            token: format!("{:?}", variants),
        })
    }

    /// Format reranker prompt
    ///
    /// Uses the format expected by Qwen3-Reranker-4B:
    /// ```text
    /// <Instruct>: {instruction}
    /// <Query>: {query}
    /// <Document>: {document}
    /// ```
    fn format_prompt(&self, query: &str, document: &str, instruction: Option<&str>) -> String {
        // Use official Qwen3-Reranker-4B default instruction if not specified
        let instruction = instruction
            .or(self.config.default_instruction.as_deref())
            .unwrap_or(
                "Given a web search query, retrieve relevant passages that answer the query",
            );

        format!(
            "<Instruct>: {}\n<Query>: {}\n<Document>: {}",
            instruction, query, document
        )
    }

    /// Compute relevance score for a single query-document pair
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `document` - Document to rank
    /// * `instruction` - Optional custom instruction
    ///
    /// # Returns
    /// Relevance score (0.0-1.0)
    ///
    /// # Errors
    /// - Tokenization failed
    /// - Inference failed
    /// - Logits extraction failed
    pub fn compute_score(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> Result<f32, RerankerError> {
        // Format prompt
        let prompt = self.format_prompt(query, document, instruction);

        // Tokenize
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| RerankerError::tokenization_failed(format!("{:?}", e)))?;

        // Check token count against limit based on device
        let max_tokens = if self.config.use_cpu {
            self.config.max_document_length_cpu
        } else {
            self.config.max_document_length_gpu
        };

        if tokens.len() > max_tokens {
            return Err(RerankerError::DocumentTooLong {
                actual: tokens.len(),
                max: max_tokens,
            });
        }

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.config.ctx_size as u32))
            .with_n_batch(self.config.n_batch as u32);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| {
                RerankerError::inference_failed(format!("Context creation failed: {:?}", e))
            })?;

        // Create batch
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last).map_err(|e| {
                RerankerError::inference_failed(format!("Batch add failed: {:?}", e))
            })?;
        }

        // Run inference
        ctx.decode(&mut batch)
            .map_err(|e| RerankerError::inference_failed(format!("Decode failed: {:?}", e)))?;

        // Get logits from last token
        let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
        let candidates_vec: Vec<LlamaTokenData> = candidates.collect();

        // Extract yes/no logits
        let yes_logit = candidates_vec
            .iter()
            .find(|c| c.id() == self.yes_token_id)
            .map(|c| c.logit())
            .ok_or_else(|| {
                RerankerError::logits_extraction_failed("Yes token not found in logits")
            })?;

        let no_logit = candidates_vec
            .iter()
            .find(|c| c.id() == self.no_token_id)
            .map(|c| c.logit())
            .ok_or_else(|| {
                RerankerError::logits_extraction_failed("No token not found in logits")
            })?;

        // Compute score using softmax
        let score = Self::softmax_score(yes_logit, no_logit);

        Ok(score)
    }

    /// Compute softmax-based score from yes/no logits
    ///
    /// # Arguments
    /// * `yes_logit` - Logit value for "yes" token
    /// * `no_logit` - Logit value for "no" token
    ///
    /// # Returns
    /// Normalized score (0.0-1.0)
    pub fn softmax_score(yes_logit: f32, no_logit: f32) -> f32 {
        let yes_exp = yes_logit.exp();
        let no_exp = no_logit.exp();
        yes_exp / (yes_exp + no_exp)
    }

    /// Get device information
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.config.model_id
    }

    /// Tokenize text
    ///
    /// # Arguments
    /// * `text` - Text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Errors
    /// - Tokenization failed
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, RerankerError> {
        self.model
            .str_to_token(text, AddBos::Never)
            .map_err(|e| RerankerError::tokenization_failed(format!("{:?}", e)))
    }

    /// Detokenize tokens back to string
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to convert
    ///
    /// # Returns
    /// Decoded string
    ///
    /// # Errors
    /// - Detokenization failed
    pub fn detokenize(&self, tokens: &[LlamaToken]) -> Result<String, RerankerError> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut result = String::new();
        for token in tokens {
            let piece = self
                .model
                .token_to_piece(*token, &mut decoder, /* special= */ true, None)
                .map_err(|e| RerankerError::tokenization_failed(format!("{:?}", e)))?;
            result.push_str(&piece);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_score() {
        // Equal logits should give 0.5
        let score = LlamaRerankerModel::softmax_score(0.0, 0.0);
        assert!((score - 0.5).abs() < 0.001);

        // Yes >> No should give close to 1.0
        let score = LlamaRerankerModel::softmax_score(10.0, -10.0);
        assert!(score > 0.99);

        // No >> Yes should give close to 0.0
        let score = LlamaRerankerModel::softmax_score(-10.0, 10.0);
        assert!(score < 0.01);
    }

    #[test]
    fn test_format_prompt() {
        // Need actual model for this, so we can't fully test without integration test
        // This test is more of a placeholder - real testing happens in integration tests
    }
}
