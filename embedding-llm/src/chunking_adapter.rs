//! Adapter for integrating command-utils hierarchical chunking with embedding-llm

use crate::error::{EmbeddingLlmError, Result};
use crate::tokenization::TokenizationProcessor;
use command_utils::text::chunking::{
    config::TokenProvider, HierarchicalChunk, HierarchicalChunker, HierarchicalChunkingConfig,
};
use std::sync::Arc;
use tracing::{debug, info};

/// Adapter that makes TokenizationProcessor compatible with command-utils TokenProvider trait
pub struct TokenizationAdapter {
    tokenization_processor: Arc<TokenizationProcessor>,
}

impl TokenizationAdapter {
    pub fn new(tokenization_processor: Arc<TokenizationProcessor>) -> Self {
        Self {
            tokenization_processor,
        }
    }
}

impl TokenProvider for TokenizationAdapter {
    type Error = EmbeddingLlmError;

    fn tokenize(&self, text: &str) -> std::result::Result<Vec<u32>, Self::Error> {
        let tokenized = self
            .tokenization_processor
            .tokenize_with_instruction(text, None)?;
        Ok(tokenized.token_ids)
    }

    fn tokenize_batch(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<u32>>, Self::Error> {
        // For now, use sequential tokenization
        // Could be optimized later with true batch processing
        texts.iter().map(|text| self.tokenize(text)).collect()
    }

    fn estimate_token_count(&self, text: &str) -> std::result::Result<usize, Self::Error> {
        // Fast estimation without full tokenization
        // Use rough heuristic: 4 characters per token for most languages
        Ok(text.len().div_ceil(4))
    }

    fn get_token_spans(
        &self,
        text: &str,
    ) -> std::result::Result<Option<Vec<(usize, usize)>>, Self::Error> {
        let tokenized = self
            .tokenization_processor
            .tokenize_with_instruction(text, None)?;
        Ok(tokenized.char_positions)
    }

    fn token_to_char(
        &self,
        text: &str,
        token_pos: usize,
    ) -> std::result::Result<Option<usize>, Self::Error> {
        let tokenized = self
            .tokenization_processor
            .tokenize_with_instruction(text, None)?;

        if let Some(char_positions) = tokenized.char_positions {
            Ok(char_positions.get(token_pos).map(|(start, _end)| *start))
        } else {
            Ok(None)
        }
    }

    fn char_to_token(
        &self,
        text: &str,
        char_pos: usize,
    ) -> std::result::Result<Option<usize>, Self::Error> {
        let tokenized = self
            .tokenization_processor
            .tokenize_with_instruction(text, None)?;

        if let Some(char_positions) = tokenized.char_positions {
            // Find the token that contains this character position
            for (token_idx, &(start, end)) in char_positions.iter().enumerate() {
                if char_pos >= start && char_pos < end {
                    return Ok(Some(token_idx));
                }
            }
            // If char_pos is beyond all tokens, return the last token
            Ok(Some(char_positions.len().saturating_sub(1)))
        } else {
            Ok(None)
        }
    }
}

/// Hierarchical chunker specialized for embedding generation
pub struct EmbeddingHierarchicalChunker {
    chunker: HierarchicalChunker<TokenizationAdapter>,
    #[allow(dead_code)]
    tokenization_processor: Arc<TokenizationProcessor>,
}

impl EmbeddingHierarchicalChunker {
    /// Create new hierarchical chunker for embedding generation
    pub fn new(
        tokenization_processor: Arc<TokenizationProcessor>,
        max_seq_length: usize,
    ) -> Result<Self> {
        let config = HierarchicalChunkingConfig::for_embedding(max_seq_length);
        let adapter = TokenizationAdapter::new(tokenization_processor.clone());

        let chunker = HierarchicalChunker::new(config, adapter, None).map_err(|e| {
            EmbeddingLlmError::sliding_window(format!("Failed to create hierarchical chunker: {e}"))
        })?;

        Ok(Self {
            chunker,
            tokenization_processor,
        })
    }

    /// Create hierarchical chunker with custom configuration
    pub fn with_config(
        tokenization_processor: Arc<TokenizationProcessor>,
        config: HierarchicalChunkingConfig,
    ) -> Result<Self> {
        let adapter = TokenizationAdapter::new(tokenization_processor.clone());

        let chunker = HierarchicalChunker::new(config, adapter, None).map_err(|e| {
            EmbeddingLlmError::sliding_window(format!("Failed to create hierarchical chunker: {e}"))
        })?;

        Ok(Self {
            chunker,
            tokenization_processor,
        })
    }

    /// Perform hierarchical chunking on text for embedding generation
    pub fn chunk_for_embedding(&mut self, text: &str) -> Result<Vec<EmbeddingChunk>> {
        debug!(
            "Starting hierarchical chunking for embedding: {} chars",
            text.len()
        );

        let chunks = self.chunker.chunk_efficiently(text).map_err(|e| {
            EmbeddingLlmError::sliding_window(format!("Hierarchical chunking failed: {e}"))
        })?;

        info!("Hierarchical chunking produced {} chunks", chunks.len());

        // Convert to EmbeddingChunk format
        let embedding_chunks = chunks
            .into_iter()
            .map(EmbeddingChunk::from_hierarchical_chunk)
            .collect();

        Ok(embedding_chunks)
    }

    /// Get the underlying chunking configuration
    pub fn config(&self) -> &HierarchicalChunkingConfig {
        self.chunker.config()
    }
}

/// Chunk specifically designed for embedding generation
#[derive(Debug, Clone)]
pub struct EmbeddingChunk {
    /// Text content of the chunk
    pub content: String,
    /// Token IDs for this chunk
    pub token_ids: Vec<u32>,
    /// Character start position in original text
    pub char_start: usize,
    /// Character end position in original text
    pub char_end: usize,
    /// Type of chunking strategy used
    pub chunk_type: ChunkType,
    /// Sequential index of this chunk
    pub chunk_index: usize,
    /// Quality metrics for this chunk
    pub quality_metrics: ChunkQualityMetrics,
}

impl EmbeddingChunk {
    /// Convert from command-utils HierarchicalChunk
    pub fn from_hierarchical_chunk(chunk: HierarchicalChunk) -> Self {
        let quality_metrics = ChunkQualityMetrics::calculate(&chunk);

        Self {
            content: chunk.content,
            token_ids: chunk.tokens,
            char_start: chunk.char_start,
            char_end: chunk.char_end,
            chunk_type: ChunkType::from_hierarchical(chunk.chunk_type),
            chunk_index: chunk.chunk_index,
            quality_metrics,
        }
    }

    /// Get the character length of this chunk
    pub fn char_length(&self) -> usize {
        self.char_end - self.char_start
    }

    /// Get the token count for this chunk
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    /// Check if this chunk preserves semantic boundaries
    pub fn preserves_boundaries(&self) -> bool {
        self.chunk_type.preserves_boundaries()
    }
}

/// Chunk type for embedding generation (mapped from command-utils)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    CompleteParagraph,
    MergedParagraphs,
    SplitParagraph,
    SentenceBasedSplit,
    ForcedSplit,
    Custom,
}

impl ChunkType {
    fn from_hierarchical(chunk_type: command_utils::text::chunking::ChunkType) -> Self {
        use command_utils::text::chunking::ChunkType as HChunkType;
        match chunk_type {
            HChunkType::CompleteParagraph => Self::CompleteParagraph,
            HChunkType::MergedParagraphs => Self::MergedParagraphs,
            HChunkType::SplitParagraph => Self::SplitParagraph,
            HChunkType::SentenceBasedSplit => Self::SentenceBasedSplit,
            HChunkType::ForcedSplit => Self::ForcedSplit,
            HChunkType::Custom(_) => Self::Custom,
        }
    }

    pub fn preserves_boundaries(&self) -> bool {
        matches!(self, Self::CompleteParagraph | Self::MergedParagraphs)
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::CompleteParagraph => "Complete paragraph",
            Self::MergedParagraphs => "Merged small paragraphs",
            Self::SplitParagraph => "Split large paragraph",
            Self::SentenceBasedSplit => "Sentence-based split",
            Self::ForcedSplit => "Forced character split",
            Self::Custom => "Custom splitting strategy",
        }
    }
}

/// Quality metrics for chunk evaluation
#[derive(Debug, Clone)]
pub struct ChunkQualityMetrics {
    /// Token density (tokens per character)
    pub token_density: f32,
    /// Whether the chunk ends with complete sentence
    pub ends_with_sentence: bool,
    /// Boundary quality score (0.0 to 1.0)
    pub boundary_quality: f32,
    /// Estimated semantic coherence (basic heuristic)
    pub semantic_coherence: f32,
}

impl ChunkQualityMetrics {
    fn calculate(chunk: &HierarchicalChunk) -> Self {
        let token_density = if chunk.content.is_empty() {
            0.0
        } else {
            chunk.tokens.len() as f32 / chunk.content.len() as f32
        };

        let ends_with_sentence = chunk
            .content
            .trim_end()
            .chars()
            .last()
            .map(|c| "。！？.!?".contains(c))
            .unwrap_or(false);

        let boundary_quality = if chunk.chunk_type.preserves_boundaries() {
            0.9
        } else if ends_with_sentence {
            0.7
        } else {
            0.4
        };

        // Basic semantic coherence heuristic
        let semantic_coherence = match chunk.chunk_type {
            command_utils::text::chunking::ChunkType::CompleteParagraph => 0.9,
            command_utils::text::chunking::ChunkType::MergedParagraphs => 0.8,
            command_utils::text::chunking::ChunkType::SentenceBasedSplit => 0.7,
            command_utils::text::chunking::ChunkType::SplitParagraph => 0.6,
            command_utils::text::chunking::ChunkType::ForcedSplit => 0.3,
            command_utils::text::chunking::ChunkType::Custom(_) => 0.5,
        };

        Self {
            token_density,
            ends_with_sentence,
            boundary_quality,
            semantic_coherence,
        }
    }

    /// Get overall quality score (0.0 to 1.0)
    pub fn overall_quality(&self) -> f32 {
        (self.boundary_quality * 0.4 + self.semantic_coherence * 0.6).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenization::TokenizationProcessor;
    use command_utils::text::chunking::HierarchicalChunkingConfig;
    use std::sync::Arc;

    #[test]
    fn test_chunk_type_mapping() {
        use command_utils::text::chunking::ChunkType as HChunkType;

        assert_eq!(
            ChunkType::from_hierarchical(HChunkType::CompleteParagraph),
            ChunkType::CompleteParagraph
        );

        assert_eq!(
            ChunkType::from_hierarchical(HChunkType::Custom("test".to_string())),
            ChunkType::Custom
        );
    }

    #[test]
    fn test_chunk_type_properties() {
        assert!(ChunkType::CompleteParagraph.preserves_boundaries());
        assert!(ChunkType::MergedParagraphs.preserves_boundaries());
        assert!(!ChunkType::ForcedSplit.preserves_boundaries());

        assert_eq!(
            ChunkType::CompleteParagraph.description(),
            "Complete paragraph"
        );
        assert_eq!(
            ChunkType::ForcedSplit.description(),
            "Forced character split"
        );
    }

    #[test]
    fn test_quality_metrics() {
        let mut chunk = HierarchicalChunk::new(
            "これはテストです。".to_string(),
            vec![1, 2, 3, 4, 5],
            0,
            10,
            command_utils::text::chunking::ChunkType::CompleteParagraph,
            0,
        );

        let metrics = ChunkQualityMetrics::calculate(&chunk);

        assert!(metrics.ends_with_sentence);
        assert!(metrics.boundary_quality > 0.8);
        assert!(metrics.semantic_coherence > 0.8);
        assert!(metrics.overall_quality() > 0.8);

        // Test forced split quality
        chunk.chunk_type = command_utils::text::chunking::ChunkType::ForcedSplit;
        chunk.content = "これは途中で切れた".to_string();

        let forced_metrics = ChunkQualityMetrics::calculate(&chunk);
        assert!(!forced_metrics.ends_with_sentence);
        assert!(forced_metrics.semantic_coherence < 0.5);
        assert!(forced_metrics.overall_quality() < metrics.overall_quality());
    }

}
