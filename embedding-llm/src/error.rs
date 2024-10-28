use thiserror::Error;
use tracing::{debug, error, warn};

/// Error types for embedding-llm plugin with production-ready categorization
#[derive(Error, Debug)]
pub enum EmbeddingLlmError {
    #[error("llama.cpp error: {0}")]
    LlamaCpp(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Sliding window processing error: {0}")]
    SlidingWindow(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] prost::EncodeError),

    #[error("Deserialization error: {0}")]
    Deserialization(#[from] prost::DecodeError),

    #[error("HuggingFace Hub error: {0}")]
    HfHub(String),

    #[error("Tokenizers error: {0}")]
    Tokenizers(String),

    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Thread synchronization error: {0}")]
    ThreadSync(String),
}

pub type Result<T> = std::result::Result<T, EmbeddingLlmError>;

impl EmbeddingLlmError {
    pub fn llamacpp<S: Into<String>>(msg: S) -> Self {
        Self::LlamaCpp(msg.into())
    }

    pub fn tokenization<S: Into<String>>(msg: S) -> Self {
        Self::Tokenization(msg.into())
    }

    pub fn model_loading<S: Into<String>>(msg: S) -> Self {
        Self::ModelLoading(msg.into())
    }

    pub fn inference<S: Into<String>>(msg: S) -> Self {
        Self::Inference(msg.into())
    }

    pub fn sliding_window<S: Into<String>>(msg: S) -> Self {
        Self::SlidingWindow(msg.into())
    }

    pub fn text_processing<S: Into<String>>(msg: S) -> Self {
        Self::SlidingWindow(msg.into()) // Reuse SlidingWindow variant for text processing
    }

    pub fn configuration<S: Into<String>>(msg: S) -> Self {
        Self::Configuration(msg.into())
    }

    pub fn hf_hub<S: Into<String>>(msg: S) -> Self {
        Self::HfHub(msg.into())
    }

    pub fn tokenizers<S: Into<String>>(msg: S) -> Self {
        Self::Tokenizers(msg.into())
    }

    pub fn resource_exhaustion<S: Into<String>>(msg: S) -> Self {
        Self::ResourceExhaustion(msg.into())
    }

    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        Self::Timeout(msg.into())
    }

    pub fn thread_sync<S: Into<String>>(msg: S) -> Self {
        Self::ThreadSync(msg.into())
    }

    /// Check if error is recoverable (can retry)
    /// Uses message content analysis for more accurate recovery assessment
    pub fn is_recoverable(&self) -> bool {
        match self {
            EmbeddingLlmError::LlamaCpp(msg) => {
                // llama.cpp errors - analyze message content
                let msg_lower = msg.to_lowercase();

                // Non-recoverable: model loading, backend initialization, file corruption
                if msg_lower.contains("failed to load model")
                    || msg_lower.contains("backend")
                    || msg_lower.contains("file not found")
                    || msg_lower.contains("invalid model")
                    || msg_lower.contains("corrupted")
                    || msg_lower.contains("unsupported")
                {
                    return false;
                }

                // Recoverable: memory issues, temporary resource problems, CUDA/GPU issues
                if msg_lower.contains("out of memory")
                    || msg_lower.contains("memory")
                    || msg_lower.contains("cuda")
                    || msg_lower.contains("gpu")
                    || msg_lower.contains("device")
                    || msg_lower.contains("timeout")
                {
                    return true;
                }

                // Conservative default for unknown llama.cpp errors
                false
            }

            EmbeddingLlmError::Tokenization(msg) => {
                // Tokenization errors - analyze message content
                let msg_lower = msg.to_lowercase();

                // Non-recoverable: model mismatch, encoding issues, configuration problems
                if msg_lower.contains("failed to tokenize")
                    || msg_lower.contains("unknown token")
                    || msg_lower.contains("encoding")
                    || msg_lower.contains("unsupported")
                    || msg_lower.contains("model mismatch")
                {
                    return false;
                }

                // Recoverable: temporary processing errors
                if msg_lower.contains("timeout")
                    || msg_lower.contains("busy")
                    || msg_lower.contains("retry")
                {
                    return true;
                }

                // Conservative default
                false
            }

            EmbeddingLlmError::ModelLoading(_) => false, // Always configuration issue

            EmbeddingLlmError::Inference(msg) => {
                // Inference errors - distinguish between configuration and temporary issues
                let msg_lower = msg.to_lowercase();

                // Non-recoverable: configuration issues, data problems
                if msg_lower.contains("zero embedding")
                    || msg_lower.contains("sequence too long")
                    || msg_lower.contains("context length")
                    || msg_lower.contains("invalid input")
                    || msg_lower.contains("empty token sequence")
                    || msg_lower.contains("dimension mismatch")
                    || msg_lower.contains("model configuration")
                {
                    return false;
                }

                // Recoverable: temporary processing issues, resource constraints
                if msg_lower.contains("failed to create context")
                    || msg_lower.contains("decode failed")
                    || msg_lower.contains("memory")
                    || msg_lower.contains("timeout")
                    || msg_lower.contains("busy")
                    || msg_lower.contains("lock")
                {
                    return true;
                }

                // Default to recoverable for unknown inference errors (may be temporary)
                true
            }

            EmbeddingLlmError::SlidingWindow(msg) => {
                // Sliding window errors - distinguish between config and processing issues
                let msg_lower = msg.to_lowercase();

                // Non-recoverable: configuration problems
                if msg_lower.contains("window size")
                    || msg_lower.contains("stride")
                    || msg_lower.contains("configuration")
                    || msg_lower.contains("invalid parameter")
                    || msg_lower.contains("overlap")
                {
                    return false;
                }

                // Recoverable: processing issues, temporary failures
                if msg_lower.contains("processing")
                    || msg_lower.contains("tokenization")
                    || msg_lower.contains("merge")
                    || msg_lower.contains("failed to process")
                {
                    return true;
                }

                // Default to recoverable for processing-related issues
                true
            }

            EmbeddingLlmError::Configuration(_) => false, // Always config issue
            EmbeddingLlmError::Io(_) => true,             // I/O may be temporary
            EmbeddingLlmError::Serialization(_) => false, // Data corruption
            EmbeddingLlmError::Deserialization(_) => false, // Data corruption
            EmbeddingLlmError::HfHub(_) => true,          // Network issue

            EmbeddingLlmError::Tokenizers(msg) => {
                // Tokenizers library errors - analyze message content
                let msg_lower = msg.to_lowercase();

                // Non-recoverable: model file issues, configuration problems
                if msg_lower.contains("file not found")
                    || msg_lower.contains("invalid model")
                    || msg_lower.contains("corrupted")
                    || msg_lower.contains("unsupported")
                    || msg_lower.contains("configuration")
                {
                    return false;
                }

                // Recoverable: temporary access issues
                if msg_lower.contains("network")
                    || msg_lower.contains("timeout")
                    || msg_lower.contains("download")
                {
                    return true;
                }

                // Conservative default
                false
            }

            EmbeddingLlmError::ResourceExhaustion(_) => true, // May free up
            EmbeddingLlmError::Timeout(_) => true,            // Can retry
            EmbeddingLlmError::ThreadSync(_) => true,         // May be temporary
        }
    }

    /// Log error with appropriate level
    pub fn log_error(&self) {
        match self {
            // Critical errors - immediate attention required
            EmbeddingLlmError::LlamaCpp(msg) => {
                error!("Critical llama.cpp error: {}", msg);
            }
            EmbeddingLlmError::ModelLoading(msg) => {
                error!("Model loading failed: {}", msg);
            }
            EmbeddingLlmError::Configuration(msg) => {
                error!("Configuration error: {}", msg);
            }

            // Warning level - recoverable but concerning
            EmbeddingLlmError::ResourceExhaustion(msg) => {
                warn!("Resource exhaustion (may recover): {}", msg);
            }
            EmbeddingLlmError::Timeout(msg) => {
                warn!("Timeout occurred (retryable): {}", msg);
            }
            EmbeddingLlmError::ThreadSync(msg) => {
                warn!("Thread synchronization issue: {}", msg);
            }
            EmbeddingLlmError::HfHub(msg) => {
                warn!("HuggingFace Hub access failed: {}", msg);
            }

            // Debug level - expected in some scenarios
            EmbeddingLlmError::Inference(msg) => {
                debug!("Inference error: {}", msg);
            }
            EmbeddingLlmError::SlidingWindow(msg) => {
                debug!("Sliding window processing issue: {}", msg);
            }

            // Error level - unexpected but handled
            _ => {
                error!("Embedding LLM error: {}", self);
            }
        }
    }

    /// Get retry delay suggestion in milliseconds
    /// Takes into account error type and severity for optimal retry timing
    pub fn retry_delay_ms(&self) -> Option<u64> {
        if !self.is_recoverable() {
            return None;
        }

        match self {
            EmbeddingLlmError::LlamaCpp(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("memory") || msg_lower.contains("gpu") {
                    Some(5000) // GPU/memory issues need longer recovery
                } else if msg_lower.contains("cuda") || msg_lower.contains("device") {
                    Some(3000) // Device issues
                } else {
                    Some(1000) // General llama.cpp issues
                }
            }

            EmbeddingLlmError::Tokenization(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("timeout") {
                    Some(2000) // Timeout issues
                } else {
                    Some(1000) // Other recoverable tokenization issues
                }
            }

            EmbeddingLlmError::Inference(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("memory") {
                    Some(3000) // Memory issues need time to recover
                } else if msg_lower.contains("lock") || msg_lower.contains("busy") {
                    Some(500) // Concurrency issues resolve quickly
                } else if msg_lower.contains("context") {
                    Some(2000) // Context creation issues
                } else {
                    Some(1000) // General inference issues
                }
            }

            EmbeddingLlmError::SlidingWindow(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("tokenization") {
                    Some(1000) // Tokenization-related issues
                } else if msg_lower.contains("processing") {
                    Some(500) // Processing issues resolve quickly
                } else {
                    Some(1000) // Other sliding window issues
                }
            }

            EmbeddingLlmError::Io(_) => Some(2000), // I/O issues
            EmbeddingLlmError::HfHub(_) => Some(5000), // Network issues need longer delay

            EmbeddingLlmError::Tokenizers(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("network") || msg_lower.contains("download") {
                    Some(5000) // Network/download issues
                } else {
                    Some(2000) // Other recoverable tokenizer issues
                }
            }

            EmbeddingLlmError::ResourceExhaustion(_) => Some(10000), // Resource exhaustion needs long recovery
            EmbeddingLlmError::Timeout(_) => Some(3000),             // Timeout issues
            EmbeddingLlmError::ThreadSync(_) => Some(100),           // Sync issues resolve quickly

            _ => Some(1000), // Default 1 second for any other recoverable error
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_recoverable_llamacpp_errors() {
        // Non-recoverable llama.cpp errors
        let non_recoverable_errors = vec![
            EmbeddingLlmError::llamacpp("Failed to load model: file not found"),
            EmbeddingLlmError::llamacpp("Backend initialization failed"),
            EmbeddingLlmError::llamacpp("Invalid model format"),
            EmbeddingLlmError::llamacpp("Corrupted model file"),
            EmbeddingLlmError::llamacpp("Unsupported model architecture"),
        ];

        for error in non_recoverable_errors {
            assert!(
                !error.is_recoverable(),
                "Error should be non-recoverable: {error}"
            );
            assert!(
                error.retry_delay_ms().is_none(),
                "Non-recoverable error should not have retry delay"
            );
        }

        // Recoverable llama.cpp errors
        let recoverable_errors = vec![
            EmbeddingLlmError::llamacpp("Out of memory error"),
            EmbeddingLlmError::llamacpp("CUDA device error"),
            EmbeddingLlmError::llamacpp("GPU memory allocation failed"),
            EmbeddingLlmError::llamacpp("Device timeout occurred"),
        ];

        for error in recoverable_errors {
            assert!(
                error.is_recoverable(),
                "Error should be recoverable: {error}"
            );
            assert!(
                error.retry_delay_ms().is_some(),
                "Recoverable error should have retry delay"
            );
        }
    }

    #[test]
    fn test_is_recoverable_inference_errors() {
        // Non-recoverable inference errors
        let non_recoverable_errors = vec![
            EmbeddingLlmError::inference("Generated zero embedding vector"),
            EmbeddingLlmError::inference("Token sequence too long: 1024 > 512"),
            EmbeddingLlmError::inference("Context length exceeded"),
            EmbeddingLlmError::inference("Empty token sequence"),
            EmbeddingLlmError::inference("Dimension mismatch in embeddings"),
        ];

        for error in non_recoverable_errors {
            assert!(
                !error.is_recoverable(),
                "Error should be non-recoverable: {error}"
            );
        }

        // Recoverable inference errors
        let recoverable_errors = vec![
            EmbeddingLlmError::inference("Failed to create context for inference"),
            EmbeddingLlmError::inference("Context decode failed due to memory"),
            EmbeddingLlmError::inference("Model lock acquisition failed"),
            EmbeddingLlmError::inference("Inference timeout occurred"),
            EmbeddingLlmError::inference("Unknown processing error"),
        ];

        for error in recoverable_errors {
            assert!(
                error.is_recoverable(),
                "Error should be recoverable: {error}"
            );
        }
    }

    #[test]
    fn test_is_recoverable_sliding_window_errors() {
        // Non-recoverable sliding window errors
        let non_recoverable_errors = vec![
            EmbeddingLlmError::sliding_window("Invalid window size configuration"),
            EmbeddingLlmError::sliding_window("Window stride must be positive"),
            EmbeddingLlmError::sliding_window("Configuration parameter invalid"),
            EmbeddingLlmError::sliding_window("Overlap size exceeds window size"),
        ];

        for error in non_recoverable_errors {
            assert!(
                !error.is_recoverable(),
                "Error should be non-recoverable: {error}"
            );
        }

        // Recoverable sliding window errors
        let recoverable_errors = vec![
            EmbeddingLlmError::sliding_window("Failed to process sliding window"),
            EmbeddingLlmError::sliding_window("Tokenization failed during processing"),
            EmbeddingLlmError::sliding_window("Merge operation encountered error"),
        ];

        for error in recoverable_errors {
            assert!(
                error.is_recoverable(),
                "Error should be recoverable: {error}"
            );
        }
    }

    #[test]
    fn test_retry_delay_ms_scaling() {
        // Test that different error types have appropriate retry delays
        let memory_error = EmbeddingLlmError::llamacpp("GPU memory allocation failed");
        let lock_error = EmbeddingLlmError::inference("Model lock acquisition failed");
        let network_error = EmbeddingLlmError::hf_hub("Network timeout during model download");
        let resource_error = EmbeddingLlmError::resource_exhaustion("System memory exhausted");

        assert_eq!(memory_error.retry_delay_ms(), Some(5000)); // Long delay for GPU issues
        assert_eq!(lock_error.retry_delay_ms(), Some(500)); // Short delay for concurrency
        assert_eq!(network_error.retry_delay_ms(), Some(5000)); // Long delay for network
        assert_eq!(resource_error.retry_delay_ms(), Some(10000)); // Longest for resource exhaustion
    }

    #[test]
    fn test_always_non_recoverable_errors() {
        let non_recoverable_types = vec![
            EmbeddingLlmError::model_loading("Model file not found"),
            EmbeddingLlmError::configuration("Invalid max_seq_length"),
        ];

        for error in non_recoverable_types {
            assert!(
                !error.is_recoverable(),
                "Error should always be non-recoverable: {error}"
            );
            assert!(
                error.retry_delay_ms().is_none(),
                "Non-recoverable error should not have retry delay"
            );
        }

        // Test I/O error conversion (always recoverable)
        let io_error = EmbeddingLlmError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Permission denied",
        ));
        assert!(
            io_error.is_recoverable(),
            "I/O errors should be recoverable"
        );
    }

    #[test]
    fn test_always_recoverable_errors() {
        let recoverable_types = vec![
            EmbeddingLlmError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Connection timeout",
            )),
            EmbeddingLlmError::hf_hub("Network connection failed"),
            EmbeddingLlmError::resource_exhaustion("Memory limit exceeded"),
            EmbeddingLlmError::timeout("Request timeout"),
            EmbeddingLlmError::thread_sync("Mutex lock failed"),
        ];

        for error in recoverable_types {
            assert!(
                error.is_recoverable(),
                "Error should always be recoverable: {error}"
            );
            assert!(
                error.retry_delay_ms().is_some(),
                "Recoverable error should have retry delay"
            );
        }
    }
}
