use std::time::Duration;

/// Configuration for reranker model
#[derive(Debug, Clone)]
pub struct RerankerModelConfig {
    /// Model identifier (e.g., "Qwen/Qwen3-Reranker-4B-GGUF")
    pub model_id: String,

    /// HuggingFace repository (same as model_id for most cases)
    pub hf_repo: String,

    /// Use CPU instead of GPU (default: false)
    pub use_cpu: bool,

    /// Number of CPU threads (default: 4)
    pub threads: usize,

    /// Context size (default: 32768)
    pub ctx_size: usize,

    /// Batch size for parallel processing (default: 32768)
    pub n_batch: usize,

    /// Use flash attention (default: true)
    pub use_flash_attention: bool,

    /// Default instruction for reranking
    pub default_instruction: Option<String>,

    /// Maximum document length for GPU environment (tokens)
    /// Documents longer than this will be truncated
    /// Default: 20000 tokens (validated: ~4.1s per document on GPU)
    pub max_document_length_gpu: usize,

    /// Maximum document length for CPU environment (tokens)
    /// Shorter than GPU to avoid long processing times
    /// Default: 5000 tokens (estimated: ~4s per document on CPU)
    pub max_document_length_cpu: usize,
}

impl Default for RerankerModelConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen3-Reranker-4B-Q4_K_M.gguf".to_string(),
            hf_repo: "QuantFactory/Qwen3-Reranker-4B-GGUF".to_string(),
            use_cpu: false,
            threads: 4,
            ctx_size: 32768, // Updated from 8192 to match spec and validation
            n_batch: 32768,  // Updated from 512 to match spec and validation
            use_flash_attention: true,
            default_instruction: None,
            max_document_length_gpu: 20_000, // Updated from 5000 to match validation results
            max_document_length_cpu: 5_000,  // Updated from 3000 to match spec
        }
    }
}

/// Complete reranker configuration
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Model configuration
    pub model: RerankerModelConfig,

    /// Cache size (number of query-document pairs to cache)
    /// Default: 10,000 entries
    pub cache_size: usize,

    /// Cache TTL in seconds
    /// Default: 3600 seconds (1 hour)
    pub cache_ttl_seconds: u64,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            model: RerankerModelConfig::default(),
            cache_size: 10_000,
            cache_ttl_seconds: 3600,
        }
    }
}

impl RerankerConfig {
    /// Create config from protobuf RerankerSettings
    pub fn from_proto(settings: &crate::proto::RerankerSettings) -> Self {
        let defaults = RerankerModelConfig::default();
        let model = RerankerModelConfig {
            model_id: settings.model_id.clone(),
            hf_repo: settings
                .hf_repo
                .clone()
                .unwrap_or(settings.model_id.clone()),
            use_cpu: settings.use_cpu,
            threads: settings
                .threads
                .map(|t| t as usize)
                .unwrap_or(defaults.threads),
            ctx_size: settings
                .ctx_size
                .map(|c| c as usize)
                .unwrap_or(defaults.ctx_size),
            n_batch: settings
                .n_batch
                .map(|n| n as usize)
                .unwrap_or(defaults.n_batch),
            use_flash_attention: settings
                .use_flash_attention
                .unwrap_or(defaults.use_flash_attention),
            default_instruction: settings.default_instruction.clone(),
            max_document_length_gpu: settings
                .max_document_length_gpu
                .map(|m| m as usize)
                .unwrap_or(defaults.max_document_length_gpu),
            max_document_length_cpu: settings
                .max_document_length_cpu
                .map(|m| m as usize)
                .unwrap_or(defaults.max_document_length_cpu),
        };

        Self {
            model,
            cache_size: settings.cache_size.map(|c| c as usize).unwrap_or(10_000),
            cache_ttl_seconds: settings.cache_ttl_seconds.unwrap_or(3600),
        }
    }

    /// Get cache TTL as Duration
    pub fn cache_ttl(&self) -> Duration {
        Duration::from_secs(self.cache_ttl_seconds)
    }

    /// Get maximum document length based on CPU/GPU mode
    pub fn max_document_length(&self) -> usize {
        if self.model.use_cpu {
            self.model.max_document_length_cpu
        } else {
            self.model.max_document_length_gpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_config() {
        let config = RerankerModelConfig::default();
        assert_eq!(config.model_id, "Qwen3-Reranker-4B-Q4_K_M.gguf");
        assert_eq!(config.hf_repo, "QuantFactory/Qwen3-Reranker-4B-GGUF");
        assert!(!config.use_cpu);
        assert_eq!(config.threads, 4);
        assert_eq!(config.ctx_size, 32768);
        assert_eq!(config.n_batch, 32768);
        assert!(config.use_flash_attention);
        assert_eq!(config.max_document_length_gpu, 20_000);
        assert_eq!(config.max_document_length_cpu, 5_000);
    }

    #[test]
    fn test_default_reranker_config() {
        let config = RerankerConfig::default();
        assert_eq!(config.cache_size, 10_000);
        assert_eq!(config.cache_ttl_seconds, 3600);
        assert_eq!(config.cache_ttl(), Duration::from_secs(3600));
    }

    #[test]
    fn test_max_document_length() {
        // GPU mode
        let mut config = RerankerConfig::default();
        assert_eq!(config.max_document_length(), 20_000);

        // CPU mode
        config.model.use_cpu = true;
        assert_eq!(config.max_document_length(), 5_000);
    }
}
