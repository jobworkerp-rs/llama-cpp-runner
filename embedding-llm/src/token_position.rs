use crate::error::{EmbeddingLlmError, Result};
use crate::tokenization::TokenizationProcessor;
use std::sync::Arc;
use tracing::debug;

/// Token positions to character positions mapping
pub struct TokenPositionMapper {
    tokenizer: Arc<TokenizationProcessor>,
}

impl TokenPositionMapper {
    pub fn new(tokenizer: Arc<TokenizationProcessor>) -> Self {
        Self { tokenizer }
    }

    /// Calculate character positions for token ranges within the original text
    /// This method calculates positions based on text-only tokenization
    pub fn calculate_char_positions(
        &self,
        original_text: &str,
        token_start_pos: usize,
        token_end_pos: usize,
    ) -> Result<(usize, usize)> {
        debug!(
            "Calculating character positions for token range [{}, {})",
            token_start_pos, token_end_pos
        );

        // Tokenize text only (no instruction) to match sliding window behavior
        let tokenized = self
            .tokenizer
            .tokenize_with_instruction(original_text, None)?;

        // Validate token positions
        if token_end_pos > tokenized.token_ids.len() {
            return Err(EmbeddingLlmError::tokenization(format!(
                "Token end position {} exceeds tokenized length {}",
                token_end_pos,
                tokenized.token_ids.len()
            )));
        }

        if token_start_pos >= token_end_pos {
            return Err(EmbeddingLlmError::tokenization(format!(
                "Invalid token range: start {} >= end {}",
                token_start_pos, token_end_pos
            )));
        }

        // Calculate character positions by decoding token ranges
        let char_start = if token_start_pos == 0 {
            0
        } else {
            self.calculate_char_position_for_token_index(&tokenized.token_ids, token_start_pos)?
        };

        let char_end = self.calculate_char_position_for_token_index(&tokenized.token_ids, token_end_pos)?;

        // Ensure positions don't exceed original text length
        let text_char_len = original_text.chars().count();
        let clamped_start = char_start.min(text_char_len);
        let clamped_end = char_end.min(text_char_len);

        debug!(
            "Character positions calculated: [{}, {}) (clamped: [{}, {}))",
            char_start, char_end, clamped_start, clamped_end
        );

        Ok((clamped_start, clamped_end))
    }

    /// Calculate character position for a specific token index by decoding incrementally
    fn calculate_char_position_for_token_index(
        &self,
        all_tokens: &[u32],
        token_index: usize,
    ) -> Result<usize> {
        if token_index == 0 {
            return Ok(0);
        }

        if token_index > all_tokens.len() {
            return Ok(self.tokenizer.decode_tokens(all_tokens)?.chars().count());
        }

        // Decode tokens up to the target index to get character count
        let tokens_slice = &all_tokens[..token_index];
        let decoded_text = self.tokenizer.decode_tokens(tokens_slice)?;
        
        Ok(decoded_text.chars().count())
    }

    /// Calculate cumulative character positions using pre-tokenized text
    /// This avoids redundant tokenization when TokenizedText is already available
    pub fn calculate_cumulative_positions_with_tokenized(
        &self,
        original_text: &str,
        text_tokenized: &crate::tokenization::TokenizedText,
        window_positions: &[(usize, usize)], // (token_start, token_end) pairs
    ) -> Result<Vec<(usize, usize)>> {
        debug!(
            "Calculating cumulative positions for {} windows using pre-tokenized text",
            window_positions.len()
        );

        let mut char_positions = Vec::new();

        // Pre-calculate character positions for common token indices to optimize
        let mut token_to_char_cache = std::collections::HashMap::new();
        token_to_char_cache.insert(0, 0usize);

        for (token_start, token_end) in window_positions {
            // Validate positions
            if *token_end > text_tokenized.token_ids.len() {
                return Err(EmbeddingLlmError::tokenization(format!(
                    "Token end position {} exceeds tokenized length {}",
                    token_end,
                    text_tokenized.token_ids.len()
                )));
            }

            // Calculate or retrieve from cache
            let char_start = if let Some(&cached) = token_to_char_cache.get(token_start) {
                cached
            } else {
                let char_pos = self
                    .calculate_char_position_for_token_index(&text_tokenized.token_ids, *token_start)?;
                token_to_char_cache.insert(*token_start, char_pos);
                char_pos
            };

            let char_end = if let Some(&cached) = token_to_char_cache.get(token_end) {
                cached
            } else {
                let char_pos = self
                    .calculate_char_position_for_token_index(&text_tokenized.token_ids, *token_end)?;
                token_to_char_cache.insert(*token_end, char_pos);
                char_pos
            };

            // Clamp to text bounds
            let text_char_len = original_text.chars().count();
            let clamped_start = char_start.min(text_char_len);
            let clamped_end = char_end.min(text_char_len);

            char_positions.push((clamped_start, clamped_end));
        }

        debug!(
            "Calculated {} character position pairs using pre-tokenized text",
            char_positions.len()
        );

        Ok(char_positions)
    }

    /// Calculate cumulative character positions for a sliding window
    /// This is more efficient for batch processing of multiple windows
    pub fn calculate_cumulative_positions(
        &self,
        original_text: &str,
        window_positions: &[(usize, usize)], // (token_start, token_end) pairs
    ) -> Result<Vec<(usize, usize)>> {
        debug!(
            "Calculating cumulative positions for {} windows",
            window_positions.len()
        );

        // Tokenize text only (no instruction) to match sliding window behavior
        let tokenized = self
            .tokenizer
            .tokenize_with_instruction(original_text, None)?;

        let mut char_positions = Vec::new();

        // Pre-calculate character positions for common token indices to optimize
        let mut token_to_char_cache = std::collections::HashMap::new();
        token_to_char_cache.insert(0, 0usize);

        for (token_start, token_end) in window_positions {
            // Validate positions
            if *token_end > tokenized.token_ids.len() {
                return Err(EmbeddingLlmError::tokenization(format!(
                    "Token end position {} exceeds tokenized length {}",
                    token_end,
                    tokenized.token_ids.len()
                )));
            }

            // Calculate or retrieve from cache
            let char_start = if let Some(&cached) = token_to_char_cache.get(token_start) {
                cached
            } else {
                let char_pos = self
                    .calculate_char_position_for_token_index(&tokenized.token_ids, *token_start)?;
                token_to_char_cache.insert(*token_start, char_pos);
                char_pos
            };

            let char_end = if let Some(&cached) = token_to_char_cache.get(token_end) {
                cached
            } else {
                let char_pos = self
                    .calculate_char_position_for_token_index(&tokenized.token_ids, *token_end)?;
                token_to_char_cache.insert(*token_end, char_pos);
                char_pos
            };

            // Clamp to text bounds
            let text_char_len = original_text.chars().count();
            let clamped_start = char_start.min(text_char_len);
            let clamped_end = char_end.min(text_char_len);

            char_positions.push((clamped_start, clamped_end));
        }

        debug!(
            "Calculated {} character position pairs",
            char_positions.len()
        );

        Ok(char_positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[cfg(test)]
    fn create_test_tokenizer() -> Option<Arc<crate::tokenization::TokenizationProcessor>> {
        match crate::tokenization::TokenizationProcessor::new_from_model_id(
            "Qwen/Qwen3-Embedding-4B", 512
        ) {
            Ok(processor) => Some(Arc::new(processor)),
            Err(_) => None,
        }
    }

    #[test]
    fn test_calculate_char_position_for_token_index() {
        let tokenizer = match create_test_tokenizer() {
            Some(t) => t,
            None => {
                println!("Skipping test - tokenizer unavailable");
                return;
            }
        };

        let mapper = TokenPositionMapper::new(tokenizer.clone());

        // Test with simple English text
        let text = "Hello world";
        let tokenized = tokenizer.tokenize_with_instruction(text, None).unwrap();
        
        // Test calculate_char_position_for_token_index function directly
        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, 1);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0);

        // Test with Japanese text
        let text = "こんにちは";
        let tokenized = tokenizer.tokenize_with_instruction(text, None).unwrap();
        
        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        if tokenized.token_ids.len() > 1 {
            let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, 1);
            assert!(result.is_ok());
            assert!(result.unwrap() > 0);
        }

        // Test with emoji
        let text = "😀😊";
        let tokenized = tokenizer.tokenize_with_instruction(text, None).unwrap();
        
        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_calculate_char_positions() {
        let tokenizer = match create_test_tokenizer() {
            Some(t) => t,
            None => {
                println!("Skipping test - tokenizer unavailable");
                return;
            }
        };

        let mapper = TokenPositionMapper::new(tokenizer.clone());
        let text = "Hello world";

        // Test calculate_char_positions function directly
        let result = mapper.calculate_char_positions(text, 0, 2);
        assert!(result.is_ok());
        let (start, end) = result.unwrap();
        assert_eq!(start, 0);
        assert!(end <= text.chars().count());

        // Test with Japanese
        let text = "こんにちは世界";
        let result = mapper.calculate_char_positions(text, 0, 1);
        assert!(result.is_ok());
        let (start, end) = result.unwrap();
        assert_eq!(start, 0);
        assert!(end <= text.chars().count());
    }

    #[test]
    fn test_calculate_cumulative_positions() {
        let tokenizer = match create_test_tokenizer() {
            Some(t) => t,
            None => {
                println!("Skipping test - tokenizer unavailable");
                return;
            }
        };

        let mapper = TokenPositionMapper::new(tokenizer.clone());
        let text = "Hello world test";
        let window_positions = vec![(0, 1), (1, 2)];

        // Test calculate_cumulative_positions function directly
        let result = mapper.calculate_cumulative_positions(text, &window_positions);
        assert!(result.is_ok());
        let positions = result.unwrap();
        assert_eq!(positions.len(), 2);
        
        // Verify first position starts at 0
        assert_eq!(positions[0].0, 0);
        // Verify positions are monotonic
        assert!(positions[0].1 <= positions[1].1);
    }

    #[test]
    fn test_calculate_cumulative_positions_with_tokenized() {
        let tokenizer = match create_test_tokenizer() {
            Some(t) => t,
            None => {
                println!("Skipping test - tokenizer unavailable");
                return;
            }
        };

        let mapper = TokenPositionMapper::new(tokenizer.clone());
        let text = "Hello world test";
        let tokenized = tokenizer.tokenize_with_instruction(text, None).unwrap();
        let window_positions = vec![(0, 1), (1, 2)];

        // Test calculate_cumulative_positions_with_tokenized function directly
        let result = mapper.calculate_cumulative_positions_with_tokenized(text, &tokenized, &window_positions);
        println!("====== Result: {:?}", result);
        assert!(result.is_ok());
        let positions = result.unwrap();
        assert_eq!(positions.len(), 2);
        
        // Verify first position starts at 0
        assert_eq!(positions[0].0, 0);
        // Verify positions are within text bounds
        let text_len = text.chars().count();
        assert!(positions[0].1 <= text_len);
        assert!(positions[1].1 <= text_len);
    }

    #[test] 
    fn test_tokenization_error_handling() {
        let tokenizer = match create_test_tokenizer() {
            Some(t) => t,
            None => {
                println!("Skipping test - tokenizer unavailable");
                return;
            }
        };

        let mapper = TokenPositionMapper::new(tokenizer.clone());
        let text = "test";
        let tokenized = tokenizer.tokenize_with_instruction(text, None).unwrap();

        // Test with invalid token index (beyond bounds)
        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, tokenized.token_ids.len() + 10);
        assert!(result.is_ok()); // Should return text length, not error

        // Test calculate_char_positions with invalid range
        let result = mapper.calculate_char_positions(text, 5, 3); // start > end
        assert!(result.is_err());

        // Test calculate_cumulative_positions with position beyond tokenized length
        let window_positions = vec![(0, tokenized.token_ids.len() + 1)];
        let result = mapper.calculate_cumulative_positions(text, &window_positions);
        assert!(result.is_err());
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = match create_test_tokenizer() {
            Some(t) => t,
            None => {
                println!("Skipping test - tokenizer unavailable");
                return;
            }
        };

        let mapper = TokenPositionMapper::new(tokenizer.clone());

        // Test complex Unicode: flag emoji
        let text = "🇯🇵";
        println!("=== Flag Emoji Test ===");
        println!("Original text: '{}' (char count: {})", text, text.chars().count());
        println!("Bytes: {:?}", text.as_bytes());
        println!("Chars: {:?}", text.chars().collect::<Vec<_>>());
        
        let tokenized = tokenizer.tokenize_with_instruction(text, None).unwrap();
        println!("Token count: {}", tokenized.token_ids.len());
        println!("Token IDs: {:?}", tokenized.token_ids);
        
        // Test incremental decoding
        for i in 0..=tokenized.token_ids.len() {
            let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, i);
            assert!(result.is_ok());
            let char_pos = result.unwrap();
            
            if i < tokenized.token_ids.len() {
                let decoded_slice = tokenizer.decode_tokens(&tokenized.token_ids[..i]).unwrap();
                println!("Token[0..{}]: decoded='{}', char_count={}, calc_pos={}", 
                         i, decoded_slice, decoded_slice.chars().count(), char_pos);
            } else {
                println!("Token[0..{}]: calc_pos={}", i, char_pos);
            }
        }
        
        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        let result = mapper.calculate_char_position_for_token_index(&tokenized.token_ids, tokenized.token_ids.len());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), text.chars().count()); // Should be 2 for flag emoji

        // Test mixed Unicode
        let text = "Hello😀こんにちは";
        println!("\n=== Mixed Unicode Test ===");
        println!("Original text: '{}' (char count: {})", text, text.chars().count());
        
        let result = mapper.calculate_char_positions(text, 0, 1);
        assert!(result.is_ok());
        let (start, end) = result.unwrap();
        assert_eq!(start, 0);
        assert!(end <= text.chars().count());
        println!("First token span: [{}, {}) chars", start, end);
    }
}