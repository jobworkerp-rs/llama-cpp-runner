use thiserror::Error;

/// Detailed error types for reranker-runner internal use
///
/// # Error Handling Strategy
///
/// This crate uses a two-tier error handling approach:
///
/// 1. **Internal code**: Uses `RerankerError` for detailed error information
/// 2. **Public API** (`DocumentReranker` trait): Uses `anyhow::Result` for consistency
///    with existing search-runner
///
/// ## Conversion Pattern
///
/// Convert `RerankerError` to `anyhow::Error` at trait boundaries:
///
/// ```rust,ignore
/// // Internal method returns Result<T, RerankerError>
/// fn internal_method(&self) -> Result<Vec<f32>, RerankerError> {
///     // ...
/// }
///
/// // Public trait method converts to anyhow::Result
/// async fn compute_scores(&mut self, ...) -> anyhow::Result<Vec<f32>> {
///     self.internal_method()
///         .map_err(|e| anyhow::Error::from(e))?;
///     Ok(result)
/// }
/// ```
#[derive(Debug, Error)]
pub enum RerankerError {
    /// Model file not found or inaccessible
    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },

    /// Model loading failed
    #[error("Failed to load model: {reason}")]
    ModelLoadFailed { reason: String },

    /// Inference execution failed
    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },

    /// Document exceeds maximum allowed length
    #[error("Document too long: {actual} tokens (max: {max})")]
    DocumentTooLong { actual: usize, max: usize },

    /// Invalid input data
    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },

    /// Configuration error
    #[error("Configuration error: {reason}")]
    ConfigError { reason: String },

    /// Cache operation failed
    #[error("Cache error: {reason}")]
    CacheError { reason: String },

    /// Tokenization failed
    #[error("Tokenization failed: {reason}")]
    TokenizationFailed { reason: String },

    /// Token ID not found in vocabulary
    #[error("Token ID not found: {token}")]
    TokenNotFound { token: String },

    /// Logits extraction failed
    #[error("Failed to extract logits: {reason}")]
    LogitsExtractionFailed { reason: String },

    /// llama.cpp library error
    #[error("llama.cpp error: {0}")]
    LlamaCppError(String),

    /// HuggingFace Hub error
    #[error("HuggingFace Hub error: {0}")]
    HfHubError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

impl RerankerError {
    /// Create ModelLoadFailed error
    pub fn model_load_failed(reason: impl Into<String>) -> Self {
        Self::ModelLoadFailed {
            reason: reason.into(),
        }
    }

    /// Create InferenceFailed error
    pub fn inference_failed(reason: impl Into<String>) -> Self {
        Self::InferenceFailed {
            reason: reason.into(),
        }
    }

    /// Create InvalidInput error
    pub fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }

    /// Create ConfigError
    pub fn config_error(reason: impl Into<String>) -> Self {
        Self::ConfigError {
            reason: reason.into(),
        }
    }

    /// Create TokenizationFailed error
    pub fn tokenization_failed(reason: impl Into<String>) -> Self {
        Self::TokenizationFailed {
            reason: reason.into(),
        }
    }

    /// Create LogitsExtractionFailed error
    pub fn logits_extraction_failed(reason: impl Into<String>) -> Self {
        Self::LogitsExtractionFailed {
            reason: reason.into(),
        }
    }
}

// Convert llama-cpp-2 errors to RerankerError
impl From<llama_cpp_2::LlamaContextLoadError> for RerankerError {
    fn from(err: llama_cpp_2::LlamaContextLoadError) -> Self {
        RerankerError::LlamaCppError(format!("Context load error: {:?}", err))
    }
}

impl From<llama_cpp_2::StringToTokenError> for RerankerError {
    fn from(err: llama_cpp_2::StringToTokenError) -> Self {
        RerankerError::TokenizationFailed {
            reason: format!("{:?}", err),
        }
    }
}

// Convert hf_hub errors to RerankerError
impl From<hf_hub::api::sync::ApiError> for RerankerError {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        RerankerError::HfHubError(format!("{:?}", err))
    }
}

// Note: Conversion to anyhow::Error is automatic via thiserror::Error trait
// Public API methods convert using `.map_err(|e| anyhow::Error::from(e))?`

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RerankerError::ModelNotFound {
            path: "/path/to/model.gguf".to_string(),
        };
        assert_eq!(err.to_string(), "Model file not found: /path/to/model.gguf");

        let err = RerankerError::DocumentTooLong {
            actual: 6000,
            max: 5000,
        };
        assert_eq!(
            err.to_string(),
            "Document too long: 6000 tokens (max: 5000)"
        );
    }

    #[test]
    fn test_error_constructors() {
        let err = RerankerError::model_load_failed("GPU not available");
        assert!(matches!(err, RerankerError::ModelLoadFailed { .. }));

        let err = RerankerError::inference_failed("OOM");
        assert!(matches!(err, RerankerError::InferenceFailed { .. }));
    }

    #[test]
    fn test_anyhow_conversion() {
        let err = RerankerError::invalid_input("Empty document");
        let anyhow_err: anyhow::Error = err.into();
        assert!(anyhow_err.to_string().contains("Invalid input"));
    }
}
