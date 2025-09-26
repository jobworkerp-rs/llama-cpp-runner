//! Text processing functionality for embedding generation
//!
//! Provides text chunking and processing capabilities optimized for embedding generation.
//! Uses hierarchical chunking from command-utils for intelligent text segmentation.

use crate::chunking_adapter::EmbeddingHierarchicalChunker;
use crate::error::{EmbeddingLlmError, Result};
use crate::tokenization::TokenizationProcessor;
use command_utils::text::chunking::{EmbeddingMerger, HierarchicalChunkingConfig, MergeStrategy};
use std::sync::Arc;
use tracing::{debug, info};

/// Text processing configuration for embedding generation
#[derive(Debug, Clone)]
pub struct TextChunkingConfig {
    /// Maximum sequence length for chunking
    pub max_seq_length: usize,
    /// Configuration for hierarchical chunking
    pub hierarchical_config: HierarchicalChunkingConfig,
}

impl TextChunkingConfig {
    /// Create configuration for embedding generation
    pub fn for_embedding(max_seq_length: usize) -> Self {
        Self {
            max_seq_length,
            hierarchical_config: HierarchicalChunkingConfig::for_embedding(max_seq_length),
        }
    }

    /// Create configuration for high-quality chunking
    pub fn for_quality_chunking(max_seq_length: usize) -> Self {
        Self {
            max_seq_length,
            hierarchical_config: HierarchicalChunkingConfig::for_quality(),
        }
    }
}

/// Text processor for embedding generation
pub struct TextProcessor {
    tokenization_processor: Arc<TokenizationProcessor>,
    config: TextChunkingConfig,
}

impl TextProcessor {
    /// Create new text processor
    pub fn new(
        tokenization_processor: Arc<TokenizationProcessor>,
        config: TextChunkingConfig,
    ) -> Self {
        Self {
            tokenization_processor,
            config,
        }
    }

    /// Process text with paragraph-aware chunking for embedding generation
    pub fn process_text_for_embedding(
        &self,
        text: &str,
        instruction: Option<&str>,
    ) -> Result<Vec<TextWindow>> {
        debug!("Processing text for embedding: {} chars", text.len());

        // Create hierarchical chunker
        let mut chunker = EmbeddingHierarchicalChunker::with_config(
            self.tokenization_processor.clone(),
            self.config.hierarchical_config.clone(),
        )?;

        // Perform hierarchical chunking
        let chunks = chunker.chunk_for_embedding(text)?;

        info!("Created {} paragraph-aware windows", chunks.len());

        // Convert to TextWindow format
        let windows: Vec<TextWindow> = chunks
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let full_text = if let Some(inst) = instruction {
                    format!("{}\n{}", inst, chunk.content)
                } else {
                    chunk.content.clone()
                };

                TextWindow {
                    text: full_text,
                    original_text: chunk.content,
                    token_ids: chunk.token_ids,
                    char_start_pos: chunk.char_start,
                    char_end_pos: chunk.char_end,
                    window_index: idx,
                    chunk_type: chunk.chunk_type,
                    quality_score: chunk.quality_metrics.overall_quality(),
                }
            })
            .collect();

        debug!("Created {} text windows for embedding", windows.len());
        Ok(windows)
    }

    /// Merge multiple embeddings using specified strategy
    pub fn merge_embeddings_static(
        embeddings: &[Vec<f32>],
        merge_strategy: MergeStrategy,
    ) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Err(EmbeddingLlmError::text_processing(
                "Cannot merge empty embeddings".to_string(),
            ));
        }

        EmbeddingMerger::merge_embeddings(embeddings, merge_strategy)
            .map_err(|e| EmbeddingLlmError::text_processing(format!("Embedding merge failed: {e}")))
    }
}

/// Processed text window for embedding generation
#[derive(Debug, Clone)]
pub struct TextWindow {
    /// Final text (with instruction if provided)
    pub text: String,
    /// Original text content without instruction
    pub original_text: String,
    /// Token IDs for the original text (without instruction)
    pub token_ids: Vec<u32>,
    /// Character start position in original input text
    pub char_start_pos: usize,
    /// Character end position in original input text
    pub char_end_pos: usize,
    /// Window index for ordering
    pub window_index: usize,
    /// Type of chunking applied
    pub chunk_type: crate::chunking_adapter::ChunkType,
    /// Quality score for this chunk
    pub quality_score: f32,
}

impl TextWindow {
    /// Get the character length of the original text
    pub fn char_length(&self) -> usize {
        self.char_end_pos - self.char_start_pos
    }

    /// Check if this is a high-quality chunk
    pub fn is_high_quality(&self) -> bool {
        self.quality_score > 0.7
    }

    /// Get a description of the chunking strategy used
    pub fn chunk_description(&self) -> &'static str {
        self.chunk_type.description()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_chunking_config() {
        let config = TextChunkingConfig::for_embedding(512);
        assert_eq!(config.max_seq_length, 512);
        assert!(config.hierarchical_config.max_chunk_tokens <= 512);

        let quality_config = TextChunkingConfig::for_quality_chunking(256);
        assert_eq!(quality_config.max_seq_length, 256);
    }

    #[test]
    fn test_text_window_properties() {
        let window = TextWindow {
            text: "Instruction\nContent".to_string(),
            original_text: "Content".to_string(),
            token_ids: vec![1, 2, 3, 4],
            char_start_pos: 10,
            char_end_pos: 17,
            window_index: 0,
            chunk_type: crate::chunking_adapter::ChunkType::CompleteParagraph,
            quality_score: 0.85,
        };

        assert_eq!(window.char_length(), 7);
        assert!(window.is_high_quality());
        assert_eq!(window.chunk_description(), "Complete paragraph");
        assert_eq!(window.token_ids.len(), 4);
    }

    #[test]
    fn test_merge_embeddings_static() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let result =
            TextProcessor::merge_embeddings_static(&embeddings, MergeStrategy::Average).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);

        // Test empty embeddings
        let empty_result = TextProcessor::merge_embeddings_static(&[], MergeStrategy::Average);
        assert!(empty_result.is_err());
    }
}
