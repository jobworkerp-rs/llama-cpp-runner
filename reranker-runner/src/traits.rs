use anyhow::Result;
use async_trait::async_trait;

/// Options for reranking operations
#[derive(Debug, Clone)]
pub struct RerankerOptions {
    /// Number of top results to return (defaults to SearchOptions::limit)
    pub top_k: Option<usize>,

    /// Score threshold (exclude results below this score)
    pub score_threshold: Option<f32>,

    /// Custom instruction for the reranker model
    pub instruction: Option<String>,

    /// Batch size for processing (default: 1)
    pub batch_size: usize,

    /// Whether to use caching
    pub use_cache: bool,

    /// Maximum document length in tokens
    /// Documents exceeding this length will be truncated
    /// None uses config default (GPU: 20000, CPU: 5000)
    pub max_document_length: Option<usize>,
}

impl Default for RerankerOptions {
    fn default() -> Self {
        Self {
            top_k: None,
            score_threshold: None,
            instruction: None,
            batch_size: 1, // Phase 1: Sequential only
            use_cache: true,
            max_document_length: None,
        }
    }
}

/// Trait for document reranking functionality
///
/// # Error Handling Policy
///
/// For consistency with existing search-runner,
/// the public API (this trait) uses `anyhow::Result`.
///
/// Internal implementations use `RerankerError` for detailed error information,
/// and convert to `anyhow::Result` at trait boundaries.
///
/// ## Conversion Pattern
/// ```rust,ignore
/// // Internal method → Public API
/// internal_method()
///     .map_err(|e| anyhow::Error::from(e))?;
/// ```
///
/// This trait only provides compute_scores() for string-based reranking.
/// UnifiedMessage integration is handled by unified_message_runner.rs.
#[async_trait]
pub trait DocumentReranker: Send + Sync {
    /// Get the name of this reranker
    fn name(&self) -> &str;

    /// Get device information ("CPU" or "GPU")
    ///
    /// # Note
    /// This method is a reranker-runner specific extension.
    /// It does not exist in search-runner's existing implementation.
    fn device(&self) -> &str {
        "unknown" // Default implementation
    }

    /// Compute relevance scores for documents in batch
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `documents` - List of documents to rerank
    /// * `options` - Reranking options
    ///
    /// # Returns
    /// List of scores (0.0-1.0) for each document
    ///
    /// # Errors
    /// Model load failure, inference errors, etc.
    async fn compute_scores(
        &mut self,
        query: &str,
        documents: &[String],
        options: RerankerOptions,
    ) -> Result<Vec<f32>>;

    /// Clear internal caches
    async fn clear_cache(&mut self) -> Result<()>;

    /// Get cache statistics
    ///
    /// # Returns
    /// (current_size, capacity) - Current cache size and capacity
    ///
    /// # Note
    /// This method is a reranker-runner specific extension.
    /// It does not exist in search-runner's existing implementation.
    async fn cache_stats(&self) -> (usize, usize) {
        (0, 0) // Default implementation
    }
}
