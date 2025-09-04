use thiserror::Error;
use tracing::{error, warn, debug};

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
    pub fn is_recoverable(&self) -> bool {
        match self {
            EmbeddingLlmError::LlamaCpp(_) => false,  // Usually fatal
            EmbeddingLlmError::Tokenization(_) => false,  // Usually data issue
            EmbeddingLlmError::ModelLoading(_) => false,  // Configuration issue
            EmbeddingLlmError::Inference(_) => true,   // May be temporary
            EmbeddingLlmError::SlidingWindow(_) => true,  // May be temporary
            EmbeddingLlmError::Configuration(_) => false, // Config issue
            EmbeddingLlmError::Io(_) => true,         // I/O may be temporary
            EmbeddingLlmError::Serialization(_) => false, // Data corruption
            EmbeddingLlmError::Deserialization(_) => false, // Data corruption
            EmbeddingLlmError::HfHub(_) => true,      // Network issue
            EmbeddingLlmError::Tokenizers(_) => false, // Usually config issue
            EmbeddingLlmError::ResourceExhaustion(_) => true, // May free up
            EmbeddingLlmError::Timeout(_) => true,    // Can retry
            EmbeddingLlmError::ThreadSync(_) => true, // May be temporary
        }
    }
    
    /// Log error with appropriate level
    pub fn log_error(&self) {
        match self {
            // Critical errors - immediate attention required
            EmbeddingLlmError::LlamaCpp(msg) => {
                error!("Critical llama.cpp error: {}", msg);
            },
            EmbeddingLlmError::ModelLoading(msg) => {
                error!("Model loading failed: {}", msg);
            },
            EmbeddingLlmError::Configuration(msg) => {
                error!("Configuration error: {}", msg);
            },
            
            // Warning level - recoverable but concerning
            EmbeddingLlmError::ResourceExhaustion(msg) => {
                warn!("Resource exhaustion (may recover): {}", msg);
            },
            EmbeddingLlmError::Timeout(msg) => {
                warn!("Timeout occurred (retryable): {}", msg);
            },
            EmbeddingLlmError::ThreadSync(msg) => {
                warn!("Thread synchronization issue: {}", msg);
            },
            EmbeddingLlmError::HfHub(msg) => {
                warn!("HuggingFace Hub access failed: {}", msg);
            },
            
            // Debug level - expected in some scenarios
            EmbeddingLlmError::Inference(msg) => {
                debug!("Inference error: {}", msg);
            },
            EmbeddingLlmError::SlidingWindow(msg) => {
                debug!("Sliding window processing issue: {}", msg);
            },
            
            // Error level - unexpected but handled
            _ => {
                error!("Embedding LLM error: {}", self);
            }
        }
    }
    
    /// Get retry delay suggestion in milliseconds
    pub fn retry_delay_ms(&self) -> Option<u64> {
        if !self.is_recoverable() {
            return None;
        }
        
        match self {
            EmbeddingLlmError::Inference(_) => Some(1000),      // 1 second
            EmbeddingLlmError::SlidingWindow(_) => Some(500),   // 500ms
            EmbeddingLlmError::Io(_) => Some(2000),            // 2 seconds
            EmbeddingLlmError::HfHub(_) => Some(5000),         // 5 seconds
            EmbeddingLlmError::ResourceExhaustion(_) => Some(10000), // 10 seconds
            EmbeddingLlmError::Timeout(_) => Some(3000),       // 3 seconds
            EmbeddingLlmError::ThreadSync(_) => Some(100),     // 100ms
            _ => Some(1000),                                    // Default 1 second
        }
    }
}