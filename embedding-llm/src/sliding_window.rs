use crate::error::{EmbeddingLlmError, Result};
use crate::token_position::TokenPositionMapper;
use crate::tokenization::TokenizationProcessor;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Sliding Window processing configuration
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Window stride for sliding window processing
    pub window_stride: usize,
    /// Minimum window size
    pub min_window_size: usize,
}

impl Default for SlidingWindowConfig {
    #[inline]
    fn default() -> Self {
        Self {
            window_stride: 256,  // Default stride
            min_window_size: 64, // Minimum useful window size
        }
    }
}

impl SlidingWindowConfig {
    /// Create SlidingWindowConfig with max_seq_length-based defaults
    /// - window_stride: 10% of max_seq_length
    /// - min_window_size: 10 (fixed)
    pub fn new(max_seq_length: usize) -> Self {
        Self {
            window_stride: max_seq_length / 10,
            min_window_size: 10,
        }
    }
}

/// Sliding Window processor for long text handling
pub struct SlidingWindowProcessor {
    max_seq_length: usize,
    config: SlidingWindowConfig,
    tokenizer_processor: Arc<TokenizationProcessor>,
    position_mapper: TokenPositionMapper,
}

impl SlidingWindowProcessor {
    /// Core logic: Calculate sliding window positions with instruction consideration
    /// This is the main algorithm that determines how to split text into windows
    pub fn calculate_sliding_windows_static(
        text_length: usize,
        instruction_length: usize,
        max_seq_length: usize,
        window_stride: usize,
        min_window_size: usize,
    ) -> Result<Vec<(usize, usize)>> {
        // Validate instruction length
        if instruction_length >= max_seq_length {
            return Err(EmbeddingLlmError::sliding_window(format!(
                "Instruction too long: {} tokens > max_seq_length {}",
                instruction_length, max_seq_length
            )));
        }

        let effective_window_size = max_seq_length - instruction_length;

        // Single window case
        if text_length <= effective_window_size {
            return Ok(vec![(0, text_length)]);
        }

        // Multiple windows case - sliding window algorithm
        let mut positions = Vec::new();
        let mut start_pos = 0;
        let mut window_index = 0;

        while start_pos < text_length {
            let end_pos = std::cmp::min(start_pos + effective_window_size, text_length);

            // Window size validation
            let window_size = end_pos - start_pos;
            if window_size < min_window_size && window_index > 0 {
                break;
            }

            positions.push((start_pos, end_pos));

            // Move to next window
            if end_pos >= text_length {
                break;
            }

            start_pos += window_stride;
            window_index += 1;
        }

        Ok(positions)
    }
    pub fn new(
        max_seq_length: usize,
        tokenizer_processor: Arc<TokenizationProcessor>,
        config: Option<SlidingWindowConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();

        // Validate configuration
        if config.window_stride > max_seq_length {
            warn!(
                "Window stride {} is larger than max_seq_length {}, using max_seq_length",
                config.window_stride, max_seq_length
            );
        }

        let position_mapper = TokenPositionMapper::new(tokenizer_processor.clone());

        Self {
            max_seq_length,
            config,
            tokenizer_processor,
            position_mapper,
        }
    }

    /// llama.cpp用に最適化された長文処理 (高レベルAPI)
    pub fn process_long_text(
        &self,
        text: &str,
        instruction: Option<&str>,
    ) -> Result<Vec<TokenizedWindow>> {
        debug!("Processing text of {} characters", text.len());

        // Step 1: Tokenize input text and instruction separately
        let (text_tokens, instruction_tokens, text_tokenized) =
            self.tokenize_text_and_instruction(text, instruction)?;

        // Step 2: Split tokenized text into sliding windows with instruction consideration
        let mut windows = self.split_tokens_into_windows(&text_tokens, &instruction_tokens)?;

        // Step 3: Calculate character positions for each window (reuse tokenization)
        self.calculate_character_positions_with_tokenized(text, &text_tokenized, &mut windows)?;

        info!("Created {} windows for long text processing", windows.len());
        Ok(windows)
    }

    /// Step 1: Tokenize text and instruction separately
    pub fn tokenize_text_and_instruction(
        &self,
        text: &str,
        instruction: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<u32>, crate::tokenization::TokenizedText)> {
        debug!("Tokenizing text and instruction separately");

        // Tokenize text only - this will be reused for character position calculation
        let text_tokenized = self
            .tokenizer_processor
            .tokenize_with_instruction(text, None)?;
        let text_tokens = text_tokenized.token_ids.clone();

        // Tokenize instruction if present
        let instruction_tokens = if let Some(inst) = instruction {
            let instruction_tokenized = self
                .tokenizer_processor
                .tokenize_with_instruction(inst, None)?;
            instruction_tokenized.token_ids
        } else {
            Vec::new()
        };

        debug!(
            "Text tokens: {}, Instruction tokens: {}",
            text_tokens.len(),
            instruction_tokens.len()
        );

        // Return the TokenizedText for reuse in character position calculation
        Ok((text_tokens, instruction_tokens, text_tokenized))
    }

    /// Step 2: Split tokenized text into sliding windows considering instruction length
    pub fn split_tokens_into_windows(
        &self,
        text_tokens: &[u32],
        instruction_tokens: &[u32],
    ) -> Result<Vec<TokenizedWindow>> {
        // Calculate window positions using the core sliding window algorithm
        let positions = Self::calculate_sliding_windows_static(
            text_tokens.len(),
            instruction_tokens.len(),
            self.max_seq_length,
            self.config.window_stride,
            self.config.min_window_size,
        )?;

        // Create TokenizedWindow objects from positions
        let mut windows = Vec::new();
        for (window_index, (start_pos, end_pos)) in positions.iter().enumerate() {
            let text_window_tokens = &text_tokens[*start_pos..*end_pos];

            // Combine instruction and text tokens
            let final_tokens = if instruction_tokens.is_empty() {
                text_window_tokens.to_vec()
            } else {
                let mut combined = instruction_tokens.to_vec();
                combined.extend_from_slice(text_window_tokens);
                combined
            };

            windows.push(TokenizedWindow {
                token_ids: final_tokens,
                start_pos: *start_pos,
                end_pos: *end_pos,
                window_index,
                char_start_pos: 0, // Will be calculated later
                char_end_pos: 0,   // Will be calculated later
            });
        }

        Ok(windows)
    }

    /// Step 3: Calculate character positions for each window using pre-tokenized text
    fn calculate_character_positions_with_tokenized(
        &self,
        text: &str,
        text_tokenized: &crate::tokenization::TokenizedText,
        windows: &mut [TokenizedWindow],
    ) -> Result<()> {
        debug!(
            "Calculating character positions for {} windows",
            windows.len()
        );

        // Collect token positions for batch calculation
        let window_positions: Vec<(usize, usize)> =
            windows.iter().map(|w| (w.start_pos, w.end_pos)).collect();

        // Calculate all character positions at once using pre-tokenized text
        let char_positions = self
            .position_mapper
            .calculate_cumulative_positions_with_tokenized(
                text,
                text_tokenized,
                &window_positions,
            )?;

        // Update each window with its calculated character positions
        for (window, (char_start, char_end)) in windows.iter_mut().zip(char_positions.iter()) {
            window.char_start_pos = *char_start;
            window.char_end_pos = *char_end;

            debug!(
                "Window {}: tokens[{}, {}) -> chars[{}, {})",
                window.window_index,
                window.start_pos,
                window.end_pos,
                window.char_start_pos,
                window.char_end_pos
            );
        }

        Ok(())
    }

    /// Static utility: Merge embeddings from multiple windows
    pub fn merge_embeddings_static(
        embeddings: &[Vec<f32>],
        merge_strategy: MergeStrategy,
    ) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Err(EmbeddingLlmError::sliding_window(
                "No embeddings to merge".to_string(),
            ));
        }

        if embeddings.len() == 1 {
            return Ok(embeddings[0].clone());
        }

        let embedding_dim = embeddings[0].len();

        // Validate all embeddings have same dimension
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != embedding_dim {
                return Err(EmbeddingLlmError::sliding_window(format!(
                    "Embedding {} has dimension {} but expected {}",
                    i,
                    emb.len(),
                    embedding_dim
                )));
            }
        }

        match merge_strategy {
            MergeStrategy::Average => Self::merge_by_average_static(embeddings),
            MergeStrategy::WeightedAverage => Self::merge_by_weighted_average_static(embeddings),
            MergeStrategy::FirstWindow => Ok(embeddings[0].clone()),
            MergeStrategy::LastWindow => Ok(embeddings[embeddings.len() - 1].clone()),
        }
    }

    /// Merge embeddings from multiple windows (instance method)
    pub fn merge_embeddings(
        &self,
        embeddings: &[Vec<f32>],
        merge_strategy: MergeStrategy,
    ) -> Result<Vec<f32>> {
        Self::merge_embeddings_static(embeddings, merge_strategy)
    }

    /// Static utility: Merge embeddings by simple averaging
    pub fn merge_by_average_static(embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        let embedding_dim = embeddings[0].len();
        let mut merged = vec![0.0f32; embedding_dim];

        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                merged[i] += value;
            }
        }

        // Average
        let num_embeddings = embeddings.len() as f32;
        for value in merged.iter_mut() {
            *value /= num_embeddings;
        }

        debug!("Merged {} embeddings by averaging", embeddings.len());
        Ok(merged)
    }

    /// Static utility: Merge embeddings by weighted averaging (giving more weight to middle windows)
    pub fn merge_by_weighted_average_static(embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        let embedding_dim = embeddings[0].len();
        let mut merged = vec![0.0f32; embedding_dim];

        // Simple weight scheme: give more weight to middle windows
        let weights = Self::calculate_window_weights_static(embeddings.len());
        let total_weight: f32 = weights.iter().sum();

        for (embedding, weight) in embeddings.iter().zip(weights.iter()) {
            for (i, &value) in embedding.iter().enumerate() {
                merged[i] += value * weight;
            }
        }

        // Normalize by total weight
        for value in merged.iter_mut() {
            *value /= total_weight;
        }

        debug!(
            "Merged {} embeddings by weighted averaging",
            embeddings.len()
        );
        Ok(merged)
    }

    /// Static utility: Calculate weights for windows (giving more weight to middle windows)
    pub fn calculate_window_weights_static(num_windows: usize) -> Vec<f32> {
        let mut weights = vec![1.0f32; num_windows];

        // Give slightly more weight to middle windows for better representation
        if num_windows > 2 {
            for (i, weight) in weights.iter_mut().enumerate() {
                let position = i as f32 / (num_windows - 1) as f32; // 0.0 to 1.0
                let distance_from_center = (position - 0.5).abs() * 2.0; // 0.0 to 1.0
                *weight = 1.0 + (1.0 - distance_from_center) * 0.2; // 1.0 to 1.2
            }
        }

        weights
    }

    #[inline]
    pub fn config(&self) -> &SlidingWindowConfig {
        &self.config
    }
}

/// Merge strategies for combining embeddings from multiple windows
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Simple average of all windows
    Average,
    /// Weighted average (more weight to middle windows)
    WeightedAverage,
    /// Use only the first window
    FirstWindow,
    /// Use only the last window
    LastWindow,
}

impl Default for MergeStrategy {
    #[inline]
    fn default() -> Self {
        Self::WeightedAverage
    }
}

/// Represents a tokenized window for sliding window processing
#[derive(Debug, Clone)]
pub struct TokenizedWindow {
    /// Token IDs for this window
    pub token_ids: Vec<u32>,
    /// Start position in the original token sequence
    pub start_pos: usize,
    /// End position in the original token sequence
    pub end_pos: usize,
    /// Window index (0-based)
    pub window_index: usize,
    /// Start character position in the original text
    pub char_start_pos: usize,
    /// End character position in the original text
    pub char_end_pos: usize,
}

impl TokenizedWindow {
    /// Get window size
    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Check if this is a complete window (not truncated)
    #[inline]
    pub fn is_complete(&self, max_seq_length: usize) -> bool {
        self.len() == max_seq_length || self.end_pos == self.start_pos + self.len()
    }

    /// Get overlap with next window
    #[inline]
    pub fn overlap_with_next(&self, stride: usize) -> usize {
        if self.len() > stride {
            self.len() - stride
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test processor - we'll test the functions directly without a processor
    // since we need to avoid the tokenizer dependency

    #[test]
    fn test_sliding_window_algorithm() {
        // Test the core sliding window algorithm with various scenarios

        // Test case 1: Single window (text fits within effective window size)
        let result_single = SlidingWindowProcessor::calculate_sliding_windows_static(
            400, // text_length
            10,  // instruction_length
            512, // max_seq_length
            256, // window_stride
            64,  // min_window_size
        );
        assert!(result_single.is_ok());
        let windows_single = result_single.unwrap();
        assert_eq!(windows_single.len(), 1);
        assert_eq!(windows_single[0], (0, 400));

        // Test case 2: Multiple windows (1000 tokens, instruction 2 tokens, max 512)
        let result_multiple = SlidingWindowProcessor::calculate_sliding_windows_static(
            1000, // text_length
            2,    // instruction_length
            512,  // max_seq_length
            256,  // window_stride
            64,   // min_window_size
        );
        assert!(result_multiple.is_ok());
        let windows_multiple = result_multiple.unwrap();
        assert_eq!(windows_multiple.len(), 3);
        assert_eq!(windows_multiple[0], (0, 510)); // Window 0: [0..510)
        assert_eq!(windows_multiple[1], (256, 766)); // Window 1: [256..766)
        assert_eq!(windows_multiple[2], (512, 1000)); // Window 2: [512..1000)

        // Test case 3: Instruction too long (should return error)
        let result_error = SlidingWindowProcessor::calculate_sliding_windows_static(
            100, // text_length
            600, // instruction_length (too long)
            512, // max_seq_length
            256, // window_stride
            64,  // min_window_size
        );
        assert!(result_error.is_err());

        // Test case 4: Small final window skipping
        let result_skip = SlidingWindowProcessor::calculate_sliding_windows_static(
            800, // text_length
            2,   // instruction_length
            512, // max_seq_length
            256, // window_stride
            100, // min_window_size (larger to trigger skipping)
        );
        assert!(result_skip.is_ok());
        let windows_skip = result_skip.unwrap();
        // Window 2 would be [512..800) = 288 tokens, which is > 100, so it should be included
        assert_eq!(windows_skip.len(), 3);
        assert_eq!(windows_skip[2], (512, 800));
    }

    #[test]
    fn test_merge_embeddings_static() {
        // Test embedding merge functionality
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        // Test average merge
        let result_avg =
            SlidingWindowProcessor::merge_embeddings_static(&embeddings, MergeStrategy::Average);
        assert!(result_avg.is_ok());
        let merged_avg = result_avg.unwrap();
        assert_eq!(merged_avg, vec![4.0, 5.0, 6.0]); // (1+4+7)/3, (2+5+8)/3, (3+6+9)/3

        // Test weighted average merge
        let result_weighted = SlidingWindowProcessor::merge_embeddings_static(
            &embeddings,
            MergeStrategy::WeightedAverage,
        );
        assert!(result_weighted.is_ok());
        let merged_weighted = result_weighted.unwrap();
        // Should be weighted differently (more weight to middle window)
        assert_eq!(merged_weighted.len(), 3);

        // Test first window strategy
        let result_first = SlidingWindowProcessor::merge_embeddings_static(
            &embeddings,
            MergeStrategy::FirstWindow,
        );
        assert!(result_first.is_ok());
        let merged_first = result_first.unwrap();
        assert_eq!(merged_first, vec![1.0, 2.0, 3.0]);

        // Test last window strategy
        let result_last =
            SlidingWindowProcessor::merge_embeddings_static(&embeddings, MergeStrategy::LastWindow);
        assert!(result_last.is_ok());
        let merged_last = result_last.unwrap();
        assert_eq!(merged_last, vec![7.0, 8.0, 9.0]);

        // Test single embedding
        let single_embedding = vec![vec![1.0, 2.0, 3.0]];
        let result_single = SlidingWindowProcessor::merge_embeddings_static(
            &single_embedding,
            MergeStrategy::Average,
        );
        assert!(result_single.is_ok());
        let merged_single = result_single.unwrap();
        assert_eq!(merged_single, vec![1.0, 2.0, 3.0]);

        // Test empty embeddings (should error)
        let empty_embeddings: Vec<Vec<f32>> = vec![];
        let result_empty = SlidingWindowProcessor::merge_embeddings_static(
            &empty_embeddings,
            MergeStrategy::Average,
        );
        assert!(result_empty.is_err());

        // Test mismatched dimensions (should error)
        let mismatched_embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0], // Different dimension
        ];
        let result_mismatch = SlidingWindowProcessor::merge_embeddings_static(
            &mismatched_embeddings,
            MergeStrategy::Average,
        );
        assert!(result_mismatch.is_err());
    }

    #[test]
    fn test_window_weights_calculation() {
        // Test window weight calculation

        // Single window
        let weights_1 = SlidingWindowProcessor::calculate_window_weights_static(1);
        assert_eq!(weights_1, vec![1.0]);

        // Two windows
        let weights_2 = SlidingWindowProcessor::calculate_window_weights_static(2);
        assert_eq!(weights_2, vec![1.0, 1.0]);

        // Three windows (middle should have higher weight)
        let weights_3 = SlidingWindowProcessor::calculate_window_weights_static(3);
        assert_eq!(weights_3.len(), 3);
        assert!(weights_3[1] > weights_3[0]); // Middle > first
        assert!(weights_3[1] > weights_3[2]); // Middle > last

        // Five windows (middle should have highest weight)
        let weights_5 = SlidingWindowProcessor::calculate_window_weights_static(5);
        assert_eq!(weights_5.len(), 5);
        assert!(weights_5[2] > weights_5[0]); // Center > edge
        assert!(weights_5[2] > weights_5[4]); // Center > edge
        assert!(weights_5[1] > weights_5[0]); // Closer to center > edge
        assert!(weights_5[3] > weights_5[4]); // Closer to center > edge
    }
}
