use crate::error::{EmbeddingLlmError, Result};
use crate::llamacpp_bridge::LlamaCppModel;
use std::sync::Arc;
use tracing::{debug, info, warn};

pub struct TokenizationProcessor {
    // llama.cppモデルから直接tokenizationを行うtokenizer
    model_ref: Option<Arc<std::sync::Mutex<LlamaCppModel>>>,
    
    // tokenizer情報（HuggingFaceから取得）
    fallback_tokenizer: Option<tokenizers::Tokenizer>,
    eos_token_id: u32,
    bos_token_id: u32,
    pad_token_id: u32,
    max_seq_length: usize,
    _model_id: String,  // Store for potential future use
}

impl TokenizationProcessor {
    /// llama.cppモデルから直接作成（推奨方法）
    pub fn new_from_llama_model(
        model: Arc<std::sync::Mutex<LlamaCppModel>>, 
        max_seq_length: usize
    ) -> Result<Self> {
        let (eos_token_id, bos_token_id, model_path) = {
            let model_lock = model.lock()
                .map_err(|_| EmbeddingLlmError::tokenization("Failed to acquire model lock".to_string()))?;
            (
                model_lock.eos_token_id(),
                model_lock.bos_token_id(),
                model_lock.model_path().to_string()
            )
        };
        
        info!("TokenizationProcessor initialized with llama.cpp model");
        info!("  model_path: {}", model_path);
        info!("  eos_token_id: {}", eos_token_id);
        info!("  bos_token_id: {}", bos_token_id);
        info!("  max_seq_length: {}", max_seq_length);
        
        Ok(Self {
            model_ref: Some(model),
            fallback_tokenizer: None,
            eos_token_id,
            bos_token_id,
            pad_token_id: eos_token_id, // Use EOS as padding
            max_seq_length,
            _model_id: model_path,
        })
    }
    
    /// HuggingFace model IDからFallback Tokenizerを初期化（互換性のため）
    pub fn new_from_model_id(model_id: &str, max_seq_length: usize) -> Result<Self> {
        info!("Creating fallback tokenizer from: {}", model_id);
        
        // HuggingFace Hubからtokenizer.jsonファイルをダウンロード
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| EmbeddingLlmError::hf_hub(format!("Failed to create HF API: {}", e)))?;
        let repo = api.model(model_id.to_string());
        let tokenizer_file = repo.get("tokenizer.json")
            .map_err(|e| EmbeddingLlmError::hf_hub(format!("Failed to download tokenizer.json: {}", e)))?;
        
        // Tokenizerファイルから初期化
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(|e| EmbeddingLlmError::tokenizers(format!(
                "Failed to load tokenizer from {}: {}", model_id, e
            )))?;
        
        // 特殊トークンIDの取得
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))  // for Qwen models
            .unwrap_or(151645); // Qwen3のデフォルトEOS
        
        let bos_token_id = tokenizer
            .token_to_id("<|startoftext|>")
            .or_else(|| tokenizer.token_to_id("<s>"))
            .or_else(|| tokenizer.token_to_id("<bos>"))
            .or_else(|| tokenizer.token_to_id("<|im_start|>"))
            .unwrap_or(151644); // Qwen3のデフォルトBOS
        
        let pad_token_id = tokenizer
            .token_to_id("<|pad|>")
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(eos_token_id); // EOS をpaddingに使用
        
        info!("Fallback tokenizer loaded:");
        info!("  eos_token_id: {}", eos_token_id);
        info!("  bos_token_id: {}", bos_token_id);
        info!("  pad_token_id: {}", pad_token_id);
        info!("  max_seq_length: {}", max_seq_length);
        
        Ok(Self {
            model_ref: None,
            fallback_tokenizer: Some(tokenizer),
            eos_token_id,
            bos_token_id,
            pad_token_id,
            max_seq_length,
            _model_id: model_id.to_string(),
        })
    }
    
    /// テキストをトークン化（instruction付き）
    pub fn tokenize_with_instruction(
        &self,
        text: &str,
        instruction: Option<&str>,
    ) -> Result<TokenizedText> {
        let full_text = if let Some(inst) = instruction {
            format!("{} {}", inst, text)
        } else {
            text.to_string()
        };
        
        debug!("Tokenizing text of {} chars", full_text.len());
        
        // Priority 1: HuggingFace tokenizer (if explicitly specified)
        if let Some(tokenizer) = &self.fallback_tokenizer {
            let encoding = tokenizer.encode(full_text, false)
                .map_err(|e| EmbeddingLlmError::tokenization(format!("Tokenization failed: {}", e)))?;
            
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();
            let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
            let original_length = token_ids.len();
            
            // 長さ制限の適用
            let (final_tokens, final_mask, truncated) = if token_ids.len() > self.max_seq_length {
                warn!("Token sequence truncated from {} to {}", token_ids.len(), self.max_seq_length);
                (
                    token_ids[..self.max_seq_length].to_vec(),
                    attention_mask[..self.max_seq_length].to_vec(),
                    true
                )
            } else {
                (token_ids, attention_mask, false)
            };
            
            debug!("Tokenized to {} tokens using HuggingFace tokenizer", final_tokens.len());
            
            Ok(TokenizedText {
                token_ids: final_tokens,
                attention_mask: final_mask,
                original_length,
                truncated,
            })
        }
        // Priority 2: llama.cpp built-in tokenizer (fallback)
        else if let Some(model_ref) = &self.model_ref {
            let token_ids = {
                let model = model_ref.lock()
                    .map_err(|_| EmbeddingLlmError::tokenization("Failed to acquire model lock".to_string()))?;
                
                model.tokenize(&full_text, true)? // add_bos = true for embedding
            };
            
            let original_length = token_ids.len();
            
            // 長さ制限の適用
            let (final_tokens, truncated) = if token_ids.len() > self.max_seq_length {
                warn!("Token sequence truncated from {} to {}", token_ids.len(), self.max_seq_length);
                (token_ids[..self.max_seq_length].to_vec(), true)
            } else {
                (token_ids, false)
            };
            
            debug!("Tokenized to {} tokens using llama.cpp built-in tokenizer", final_tokens.len());
            
            Ok(TokenizedText {
                token_ids: final_tokens,
                attention_mask: vec![1u32; original_length.min(self.max_seq_length)],
                original_length,
                truncated,
            })
        } else {
            Err(EmbeddingLlmError::tokenization("No tokenizer available".to_string()))
        }
    }
    
    // TODO huggingface LLM でのembedding生成時に使用するため、cancleが直ったら統合する
    /// バッチテキストの左パディング処理（llama.cpp / embedding用）
    pub fn tokenize_batch_with_padding(
        &self,
        texts: &[String],
        instruction: Option<&str>,
    ) -> Result<BatchTokenized> {
        let mut tokenized_texts = Vec::new();
        
        // 各テキストをトークン化
        for text in texts {
            let tokenized = self.tokenize_with_instruction(text, instruction)?;
            tokenized_texts.push(tokenized);
        }
        
        // 最大長の決定
        let max_len = tokenized_texts
            .iter()
            .map(|t| t.token_ids.len())
            .max()
            .unwrap_or(0);
        
        let mut batch_tokens = Vec::new();
        let mut batch_masks = Vec::new();
        
        for tokenized in tokenized_texts {
            let ids = tokenized.token_ids;
            let attention_mask = tokenized.attention_mask;
            
            if ids.len() < max_len {
                // 左側パディング（embedding用）
                let pad_len = max_len - ids.len();
                
                let mut padded_ids = vec![self.pad_token_id; pad_len];
                padded_ids.extend_from_slice(&ids);
                
                let mut padded_mask = vec![0u32; pad_len];
                padded_mask.extend_from_slice(&attention_mask);
                
                batch_tokens.push(padded_ids);
                batch_masks.push(padded_mask);
            } else {
                batch_tokens.push(ids);
                batch_masks.push(attention_mask);
            }
        }
        
        debug!("Tokenized batch: {} sequences, max_length: {}", batch_tokens.len(), max_len);
        
        Ok(BatchTokenized {
            token_ids: batch_tokens,
            attention_masks: batch_masks,
            sequence_length: max_len,
        })
    }
    
    /// 特殊トークンID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
    
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }
    
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
    
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }
    
    /// トークンをテキストにデコード
    pub fn decode_tokens(&self, token_ids: &[u32]) -> Result<String> {
        // Priority 1: HuggingFace tokenizer (if explicitly specified)
        if let Some(tokenizer) = &self.fallback_tokenizer {
            tokenizer.decode(token_ids, true)
                .map_err(|e| EmbeddingLlmError::tokenization(format!("Decode failed: {}", e)))
        }
        // Priority 2: llama.cpp built-in tokenizer (fallback)
        else if let Some(model_ref) = &self.model_ref {
            let model = model_ref.lock()
                .map_err(|_| EmbeddingLlmError::tokenization("Failed to acquire model lock".to_string()))?;
            
            if !token_ids.is_empty() {
                model.detokenize(token_ids)
            } else {
                Ok(String::new())
            }
        } else {
            Err(EmbeddingLlmError::tokenization("No tokenizer available for decoding".to_string()))
        }
    }
    
    /// 使用可能なtokenizerタイプを取得（デバッグ用）
    pub fn tokenizer_type(&self) -> &'static str {
        if self.model_ref.is_some() {
            "llama.cpp"
        } else if self.fallback_tokenizer.is_some() {
            "huggingface-fallback"
        } else {
            "none"
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizedText {
    pub token_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub original_length: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct BatchTokenized {
    pub token_ids: Vec<Vec<u32>>,
    pub attention_masks: Vec<Vec<u32>>,
    pub sequence_length: usize,
}

impl TokenizedText {
    /// Check if sequence is within max length
    pub fn is_valid_length(&self, max_length: usize) -> bool {
        self.token_ids.len() <= max_length
    }
    
    /// Get actual sequence length
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Note: These tests require actual tokenizer models or llama.cpp models to be available
    
    #[test]
    fn test_tokenization_processor_creation_fallback() {
        // Test fallback tokenizer creation (will likely fail without network/model)
        let result = TokenizationProcessor::new_from_model_id("nonexistent/model", 512);
        assert!(result.is_err()); // Expected to fail without real model
    }
    
    #[test]
    fn test_tokenized_text_methods() {
        let tokenized = TokenizedText {
            token_ids: vec![1, 2, 3, 4, 5],
            attention_mask: vec![1, 1, 1, 1, 1],
            original_length: 5,
            truncated: false,
        };
        
        assert_eq!(tokenized.len(), 5);
        assert!(!tokenized.is_empty());
        assert!(tokenized.is_valid_length(10));
        assert!(!tokenized.is_valid_length(3));
        assert!(!tokenized.truncated);
    }
    
    #[test]
    fn test_tokenized_text_empty() {
        let tokenized = TokenizedText {
            token_ids: vec![],
            attention_mask: vec![],
            original_length: 0,
            truncated: false,
        };
        
        assert_eq!(tokenized.len(), 0);
        assert!(tokenized.is_empty());
        assert!(tokenized.is_valid_length(10));
    }
}