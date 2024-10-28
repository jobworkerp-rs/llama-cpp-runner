use crate::{
    cache::RerankerCache,
    config::RerankerConfig,
    error::RerankerError,
    model::LlamaRerankerModel,
    traits::{DocumentReranker, RerankerOptions},
    utils,
};
use async_trait::async_trait;
use std::time::Instant;

/// Prompt template overhead in tokens
///
/// Estimated token count for the prompt template:
/// - `<Instruct>: {instruction}` (~15-20 tokens)
/// - `<Query>: {query}` (~5-10 tokens depending on query length)
/// - `<Document>: ` (~2 tokens)
/// - Total overhead: ~20-30 tokens
///
/// We use a conservative estimate of 100 tokens to account for:
/// - Variable instruction length
/// - Variable query length (can be long)
/// - Safety margin for tokenization differences
const PROMPT_OVERHEAD_TOKENS: usize = 100;

/// Statistics for a single scoring operation
#[derive(Debug, Clone, Default)]
pub struct ScoringStats {
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
    /// Number of truncated documents
    pub truncated_count: usize,
}

/// Result of a scoring operation with statistics
#[derive(Debug, Clone)]
pub struct ScoringResult {
    /// Computed scores for each document
    pub scores: Vec<f32>,
    /// Statistics about the scoring operation
    pub stats: ScoringStats,
}

/// llama.cpp-based Reranker implementation
///
/// This struct implements the DocumentReranker trait using LlamaRerankerModel
/// with cache integration and document preprocessing.
pub struct LlamaReranker {
    /// Underlying llama.cpp model wrapper
    model: LlamaRerankerModel,

    /// LRU cache for reranking scores
    cache: Option<RerankerCache>,

    /// Configuration
    config: RerankerConfig,
}

impl LlamaReranker {
    /// Create a new LlamaReranker instance
    ///
    /// # Arguments
    /// * `config` - Reranker configuration
    ///
    /// # Returns
    /// Initialized reranker ready for scoring
    ///
    /// # Errors
    /// - Model loading failed
    /// - Invalid configuration
    pub fn new(config: RerankerConfig) -> Result<Self, RerankerError> {
        tracing::info!(
            "Initializing LlamaReranker: model={}, device={}",
            config.model.model_id,
            if config.model.use_cpu { "CPU" } else { "GPU" }
        );

        // Initialize model
        let model = LlamaRerankerModel::new(config.model.clone())?;

        // Initialize cache if enabled
        let cache = if config.cache_size > 0 {
            Some(RerankerCache::new(config.cache_size, config.cache_ttl()))
        } else {
            None
        };

        tracing::info!(
            "LlamaReranker initialized: device={}, cache_enabled={}",
            model.device(),
            cache.is_some()
        );

        Ok(Self {
            model,
            cache,
            config,
        })
    }

    /// Truncate document if it exceeds maximum token length
    ///
    /// # Important: Prompt Overhead
    /// The `max_length` parameter represents the maximum **document** token count.
    /// However, the final prompt includes additional tokens:
    /// - `<Instruct>: {instruction}` (~15-20 tokens)
    /// - `<Query>: {query}` (variable, ~5-50 tokens)
    /// - `<Document>: ` (~2 tokens)
    ///
    /// To ensure the full prompt stays within model limits, we reserve
    /// `PROMPT_OVERHEAD_TOKENS` (100 tokens) for the prompt template.
    /// Therefore, documents are truncated to `max_length - PROMPT_OVERHEAD_TOKENS`.
    ///
    /// # Strategy (Two-phase optimization)
    /// 1. **Phase 1**: Conservative character-based estimation (1 char = 1 token)
    ///    - If char_count <= effective_max_length, return original (no tokenization cost)
    ///    - Conservative to avoid false negatives (especially for Japanese/CJK)
    /// 2. **Phase 2**: Actual tokenization (only if char_count > effective_max_length)
    ///    - Tokenize to get actual token count
    ///    - If exceeds effective_max_length, truncate and detokenize
    ///
    /// # Performance Impact
    /// - Short documents (≤effective_max_length chars): ~0 overhead (no tokenization)
    /// - Long documents: Same as before (one tokenization)
    ///
    /// # Arguments
    /// * `document` - Document text
    /// * `max_length` - Maximum allowed token count for the document (excluding prompt overhead)
    ///
    /// # Returns
    /// Truncated document (or original if within limit)
    ///
    /// # Errors
    /// - Tokenization failed (only in Phase 2)
    fn truncate_document(
        &self,
        document: &str,
        max_length: usize,
    ) -> Result<String, RerankerError> {
        // Reserve tokens for prompt template overhead
        let effective_max_length = if max_length > PROMPT_OVERHEAD_TOKENS {
            max_length - PROMPT_OVERHEAD_TOKENS
        } else {
            // If max_length is too small, use at least some tokens for document
            max_length / 2 // Use 50% for document, 50% for prompt
        };

        // Phase 1: Conservative estimation (1 char = 1 token)
        let char_count = document.chars().count();

        if char_count <= effective_max_length {
            // Definitely within limit, return original without tokenization
            tracing::trace!(
                "Document within safe limit: {} chars (effective max: {} tokens, overhead: {} tokens)",
                char_count,
                effective_max_length,
                PROMPT_OVERHEAD_TOKENS
            );
            return Ok(document.to_string());
        }

        // Phase 2: Actual tokenization (char_count > effective_max_length)
        tracing::debug!(
            "Document may exceed limit: {} chars (effective max: {} tokens, overhead: {} tokens), tokenizing...",
            char_count,
            effective_max_length,
            PROMPT_OVERHEAD_TOKENS
        );

        let tokens = self
            .model
            .tokenize(document)
            .map_err(|e| RerankerError::tokenization_failed(format!("{:?}", e)))?;

        if tokens.len() <= effective_max_length {
            // False positive from estimation (e.g., English text with high compression)
            tracing::debug!(
                "Document within actual limit: {} tokens ≤ {} (estimation: {} chars)",
                tokens.len(),
                effective_max_length,
                char_count
            );
            return Ok(document.to_string());
        }

        // Truncate tokens to effective_max_length
        let truncated_tokens = &tokens[..effective_max_length];

        // Detokenize back to string
        let truncated_text = self
            .model
            .detokenize(truncated_tokens)
            .map_err(|e| RerankerError::tokenization_failed(format!("{:?}", e)))?;

        tracing::warn!(
            "Document truncated: {} tokens → {} tokens (max: {}, overhead: {} tokens), {} chars → {} chars",
            tokens.len(),
            effective_max_length,
            max_length,
            PROMPT_OVERHEAD_TOKENS,
            char_count,
            truncated_text.chars().count()
        );

        Ok(truncated_text)
    }

    /// Get maximum document length based on CPU/GPU mode
    ///
    /// # Arguments
    /// * `options` - Reranker options (may override default)
    ///
    /// # Returns
    /// Maximum document length in tokens
    fn get_max_document_length(&self, options: &RerankerOptions) -> usize {
        options
            .max_document_length
            .unwrap_or_else(|| self.config.max_document_length())
    }

    /// Compute scores sequentially (one document at a time)
    ///
    /// # Error Handling
    /// If a single document fails inference, log a warning and assign score 0.0.
    /// Continue processing other documents (do not fail the entire batch).
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `documents` - List of documents
    /// * `options` - Reranker options
    ///
    /// # Returns
    /// Vector of scores (0.0-1.0) for each document
    async fn compute_scores_sequential(
        &self,
        query: &str,
        documents: &[String],
        options: &RerankerOptions,
    ) -> Result<Vec<f32>, RerankerError> {
        let instruction = options.instruction.as_deref();
        let mut scores = Vec::with_capacity(documents.len());

        for (idx, doc) in documents.iter().enumerate() {
            // Compute score for single document
            match self.model.compute_score(query, doc, instruction) {
                Ok(score) => {
                    scores.push(score);
                    tracing::debug!("Document {}: score = {:.4}", idx, score);
                }
                Err(e) => {
                    tracing::warn!("Failed to score document {}: {}. Using score 0.0", idx, e);
                    scores.push(0.0);
                }
            }
        }

        Ok(scores)
    }

    /// Compute scores with cache integration (4-phase process)
    ///
    /// # Process
    ///
    /// ## Phase 1: Cache Check
    /// Check cache for each document and identify cache hits/misses.
    ///
    /// ## Phase 2: Sequential Processing
    /// Compute scores for cache misses using compute_scores_sequential().
    ///
    /// ## Phase 3: Cache Save
    /// Store newly computed scores in cache.
    ///
    /// ## Phase 4: Merge
    /// Merge cached scores and new scores in original document order.
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `documents` - List of documents
    /// * `options` - Reranker options
    ///
    /// # Returns
    /// Vector of scores (0.0-1.0) for each document, with cache statistics
    async fn compute_scores_with_cache(
        &self,
        query: &str,
        documents: &[String],
        options: &RerankerOptions,
    ) -> Result<(Vec<f32>, ScoringStats), RerankerError> {
        let cache = match &self.cache {
            Some(c) => c,
            None => {
                // No cache available, fall back to sequential
                let scores = self
                    .compute_scores_sequential(query, documents, options)
                    .await?;
                let stats = ScoringStats {
                    cache_hits: 0,
                    cache_misses: documents.len(),
                    cache_hit_rate: 0.0,
                    truncated_count: 0, // Will be set by caller
                };
                return Ok((scores, stats));
            }
        };

        // Phase 1: Cache check
        let mut cache_results = Vec::with_capacity(documents.len());
        let mut uncached_indices = Vec::new();

        for (idx, doc) in documents.iter().enumerate() {
            let cache_key = utils::generate_cache_key(query, doc);

            if let Some(score) = cache.get(&cache_key).await {
                cache_results.push(Some(score));
                tracing::trace!("Cache hit for document {}", idx);
            } else {
                cache_results.push(None);
                uncached_indices.push(idx);
                tracing::trace!("Cache miss for document {}", idx);
            }
        }

        let cache_hit_count = cache_results.iter().filter(|r| r.is_some()).count();
        let cache_miss_count = uncached_indices.len();
        let cache_hit_rate = if !documents.is_empty() {
            cache_hit_count as f32 / documents.len() as f32
        } else {
            0.0
        };

        tracing::debug!(
            "Cache: {} hits, {} misses ({:.1}% hit rate)",
            cache_hit_count,
            cache_miss_count,
            cache_hit_rate * 100.0
        );

        // Phase 2: Sequential processing for uncached documents
        let new_scores = if !uncached_indices.is_empty() {
            let uncached_docs: Vec<String> = uncached_indices
                .iter()
                .map(|&idx| documents[idx].clone())
                .collect();

            self.compute_scores_sequential(query, &uncached_docs, options)
                .await?
        } else {
            Vec::new()
        };

        // Phase 3: Cache save
        for (i, &uncached_idx) in uncached_indices.iter().enumerate() {
            let doc = &documents[uncached_idx];
            let score = new_scores[i];
            let cache_key = utils::generate_cache_key(query, doc);
            cache.put(cache_key, score).await;
        }

        // Phase 4: Merge cached and new scores
        let mut final_scores = Vec::with_capacity(documents.len());
        let mut new_score_idx = 0;

        for cached_result in cache_results {
            if let Some(cached_score) = cached_result {
                final_scores.push(cached_score);
            } else {
                final_scores.push(new_scores[new_score_idx]);
                new_score_idx += 1;
            }
        }

        let stats = ScoringStats {
            cache_hits: cache_hit_count,
            cache_misses: cache_miss_count,
            cache_hit_rate,
            truncated_count: 0, // Will be set by caller
        };

        Ok((final_scores, stats))
    }

    /// Compute scores with full statistics tracking
    ///
    /// This is an internal method that wraps the public compute_scores()
    /// and returns detailed statistics.
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `documents` - List of documents
    /// * `options` - Reranker options
    ///
    /// # Returns
    /// ScoringResult with scores and statistics
    pub async fn compute_scores_with_stats(
        &mut self,
        query: &str,
        documents: &[String],
        options: RerankerOptions,
    ) -> Result<ScoringResult, RerankerError> {
        // 1. Document preprocessing (truncation)
        let max_len = self.get_max_document_length(&options);
        let mut preprocessed = Vec::with_capacity(documents.len());
        let mut truncated_count = 0;

        for doc in documents {
            match self.truncate_document(doc, max_len) {
                Ok(truncated) => {
                    if truncated.len() < doc.len() {
                        truncated_count += 1;
                    }
                    preprocessed.push(truncated);
                }
                Err(e) => {
                    tracing::warn!("Document truncation failed: {}. Using empty string", e);
                    preprocessed.push(String::new());
                    truncated_count += 1;
                }
            }
        }

        // 2. Compute scores (with or without cache)
        let (scores, mut stats) = if options.use_cache && self.cache.is_some() {
            self.compute_scores_with_cache(query, &preprocessed, &options)
                .await?
        } else {
            let scores = self
                .compute_scores_sequential(query, &preprocessed, &options)
                .await?;
            let stats = ScoringStats {
                cache_hits: 0,
                cache_misses: 0,
                cache_hit_rate: 0.0,
                truncated_count,
            };
            (scores, stats)
        };

        // 3. Set truncated_count in stats
        stats.truncated_count = truncated_count;

        Ok(ScoringResult { scores, stats })
    }
}

#[async_trait]
impl DocumentReranker for LlamaReranker {
    fn name(&self) -> &str {
        "LlamaReranker"
    }

    fn device(&self) -> &str {
        self.model.device()
    }

    async fn compute_scores(
        &mut self,
        query: &str,
        documents: &[String],
        options: RerankerOptions,
    ) -> anyhow::Result<Vec<f32>> {
        let start = Instant::now();

        tracing::info!(
            "Computing reranking scores: query_len={}, doc_count={}, cache_enabled={}",
            query.len(),
            documents.len(),
            options.use_cache && self.cache.is_some()
        );

        // Use compute_scores_with_stats internally
        let scoring_result = self
            .compute_scores_with_stats(query, documents, options)
            .await
            .map_err(|e| anyhow::anyhow!("Scoring failed: {}", e))?;

        let elapsed = start.elapsed();
        let avg_time = elapsed.as_secs_f32() / documents.len() as f32;

        tracing::info!(
            "Reranking completed: {} docs in {:.2}s ({:.3}s/doc), cache: {}/{} hits ({:.1}%)",
            documents.len(),
            elapsed.as_secs_f32(),
            avg_time,
            scoring_result.stats.cache_hits,
            documents.len(),
            scoring_result.stats.cache_hit_rate * 100.0
        );

        Ok(scoring_result.scores)
    }

    async fn clear_cache(&mut self) -> anyhow::Result<()> {
        if let Some(cache) = &mut self.cache {
            cache.clear().await;
            tracing::info!("Cache cleared");
        }
        Ok(())
    }

    async fn cache_stats(&self) -> (usize, usize) {
        if let Some(cache) = &self.cache {
            cache.stats().await
        } else {
            (0, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests moved to integration tests (tests/plugin_integration_test.rs)
    // as they require actual model initialization
}
