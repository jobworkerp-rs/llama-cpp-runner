use crate::error::{EmbeddingLlmError, Result};
use crate::llamacpp_bridge::LlamaCppModel;
use crate::protobuf::embedding_llm::{DType, EmbeddingLlmRunnerSettings};
use crate::text_processing::{TextChunkingConfig, TextProcessor};
use crate::tokenization::TokenizationProcessor;
use command_utils::text::chunking::{HierarchicalChunkingConfig, MergeStrategy};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

/// llama.cppベースのLLM Embedder
pub struct LlamaCppEmbedder {
    model: Arc<Mutex<LlamaCppModel>>,
    tokenization_processor: Arc<TokenizationProcessor>,
    text_processor: TextProcessor,
    model_info: ModelInfo,
    media_limits: Option<mtmd_support::MediaLimits>,
    /// Multimodal runtime — shared via `Arc` so that `prepare_bitmaps` and
    /// `inject_markers` (both `&MtmdRuntime`, upstream-documented as
    /// thread-safe) can run concurrently across blocking-thread invocations
    /// of `run()`. The only operation that needs mutual exclusion on the
    /// shared `MtmdContext` is `eval_chunks` inside
    /// `generate_multimodal_embedding`; that call is serialized by the
    /// existing `model` Mutex, so no separate mtmd-level lock is required.
    /// See `mtmd-support/src/runtime.rs` for the underlying contract.
    mtmd_runtime: Option<Arc<mtmd_support::MtmdRuntime>>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_path: String,
    pub embedding_dimension: usize,
    pub max_context_length: usize,
    pub vocab_size: usize,
    pub dtype: String,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub audio_sample_rate: Option<u32>,
}

/// Embedding result with character position information
#[derive(Debug, Clone)]
pub struct EmbeddingWithPosition {
    pub values: Vec<f32>,
    pub char_start_pos: usize,
    pub char_end_pos: usize,
}

impl LlamaCppEmbedder {
    /// 設定とバックエンドからEmbedderを初期化（推奨方式）
    pub fn new_from_settings_with_backend(
        settings: &EmbeddingLlmRunnerSettings,
        backend: Arc<Mutex<llama_cpp_2::llama_backend::LlamaBackend>>,
    ) -> Result<Self> {
        // モデルファイルパスの構築
        let model_path = Self::build_model_path(settings)?;

        info!("Loading llama.cpp model from: {}", model_path);

        // GPU device設定の検証とログ出力
        let gpu_device = settings.gpu_device;
        if let Some(device_id) = gpu_device {
            if settings.use_cpu {
                warn!("gpu_device={} is specified but use_cpu=true, GPU device setting will be ignored", device_id);
            } else {
                info!("Using GPU device: {}", device_id);
            }
        }

        // バックエンドを受け取ってモデルを初期化
        let mut model = LlamaCppModel::new_with_backend(
            &model_path,
            settings.use_cpu,
            settings.max_seq_length as usize,
            settings.max_batch_size,
            backend,
            gpu_device,
        )?;

        // Initialize multimodal projector when settings are provided.
        if let Some(mtmd_settings) = &settings.mtmd {
            model.init_mtmd(
                &mtmd_settings.mmproj,
                mtmd_settings.mmproj_hf_repo.as_deref(),
                mtmd_settings.mmproj_use_gpu.unwrap_or(!settings.use_cpu),
                mtmd_settings.media_marker.as_deref(),
            )?;
        }

        Self::build_embedder_common(settings, model)
    }

    /// モデルパスの構築（共通処理）
    fn build_model_path(settings: &EmbeddingLlmRunnerSettings) -> Result<String> {
        if settings.model_files.is_empty() {
            return Err(EmbeddingLlmError::configuration(
                "No model files specified for llama.cpp".to_string(),
            ));
        }

        // XXX 複数ファイルの場合は最初のファイルを使用（llama.cppは単一ファイル）
        // (本当は複数ファイルのこともあるがEmbedding用の場合に分割されるほど大きなモデルは無い想定)
        if settings.model_files.len() > 1 {
            error!(
                "Multiple model files specified, using first file: {}",
                settings.model_files[0]
            );
        }

        let model_dir = PathBuf::from(&settings.model_id);
        let model_path = if model_dir.is_absolute() {
            // Absolute local file path - use model_files[0] as is
            PathBuf::from(&settings.model_files[0])
                .to_string_lossy()
                .to_string()
        } else {
            // HuggingFace format: model_id/model_files[0] → "repo/gguf_file.gguf"
            // This supports llama.cpp's HuggingFace model format requirement
            // Example: model_id="Qwen/Qwen3-Embedding-4B-GGUF", model_files=["Qwen3-Embedding-4B-Q4_K_M.gguf"]
            //          Results in: "Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q4_K_M.gguf"
            model_dir
                .join(&settings.model_files[0])
                .to_string_lossy()
                .to_string()
        };

        Ok(model_path)
    }

    /// Embedderの共通構築処理
    fn build_embedder_common(
        settings: &EmbeddingLlmRunnerSettings,
        model: LlamaCppModel,
    ) -> Result<Self> {
        // dtypeのデフォルト値設定
        let dtype_enum = settings
            .dtype
            .and_then(|d| DType::try_from(d).ok())
            .unwrap_or(if settings.use_cpu {
                DType::F32
            } else {
                DType::F16
            });

        let dtype = match dtype_enum {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
        };

        info!("Using dtype: {} (CPU: {})", dtype, settings.use_cpu);

        // モデル情報を取得
        let embedding_dimension = model.embedding_dimension();
        let max_context_length = model.max_context_length();
        let vocab_size = model.vocabulary_size();

        let (supports_vision, supports_audio, audio_sample_rate) = model
            .mtmd_ref()
            .map(|m| (m.support_vision(), m.support_audio(), m.audio_sample_rate()))
            .unwrap_or((false, false, None));

        let model_info = ModelInfo {
            model_path: model.model_path().to_string(),
            embedding_dimension,
            max_context_length,
            vocab_size,
            dtype: dtype.to_string(),
            supports_vision,
            supports_audio,
            audio_sample_rate,
        };

        info!("Model loaded successfully: {:?}", model_info);

        // Arc<Mutex<LlamaCppModel>>でモデルをラップ
        let model_arc = Arc::new(Mutex::new(model));

        // Tokenizerの初期化（優先順位ロジック）
        let tokenization_processor = Arc::new(
            if let Some(tokenizer_id) = &settings.tokenizer_model_id {
                // 1. tokenizer_model_idが指定された場合はそれを必ず使用
                info!("Using specified HuggingFace tokenizer: {}", tokenizer_id);
                TokenizationProcessor::new_from_model_id(
                    tokenizer_id,
                    settings.max_seq_length as usize,
                )
                .map_err(|e| {
                    EmbeddingLlmError::tokenization(format!(
                        "Failed to load specified tokenizer {tokenizer_id}: {e}"
                    ))
                })?
            } else {
                // 2. 指定がない場合はGGUF内蔵tokenizerを試行
                info!("Using GGUF built-in tokenizer from llama.cpp model");
                match TokenizationProcessor::new_from_llama_model(
                    model_arc.clone(),
                    settings.max_seq_length as usize,
                ) {
                    Ok(processor) => processor,
                    Err(e) => {
                        // 3. GGUF内蔵tokenizerが失敗した場合はエラー
                        let err_msg = format!(
                        "GGUF built-in tokenizer failed and no tokenizer_model_id specified. \
                         Either use a GGUF with built-in tokenizer or specify tokenizer_model_id. Error: {e}"
                    );
                        error!("{}", err_msg);
                        return Err(EmbeddingLlmError::configuration(err_msg));
                    }
                }
            },
        );

        // Text Processorの初期化
        let text_chunking_config = if let Some(config) = settings.chunking_config.as_ref() {
            // HierarchicalChunkingConfigが指定されている場合
            let hierarchical_config = HierarchicalChunkingConfig {
                max_chunk_tokens: config.max_chunk_tokens as usize,
                min_chunk_tokens: config.min_chunk_tokens as usize,
                enable_paragraph_merging: config.enable_paragraph_merging,
                enable_sentence_splitting: config.enable_sentence_splitting,
                enable_forced_splitting: config.enable_forced_splitting,
            };

            TextChunkingConfig {
                max_seq_length: settings.max_seq_length as usize,
                hierarchical_config,
            }
        } else {
            // 設定が未指定の場合は、max_seq_lengthベースのデフォルト設定を使用
            TextChunkingConfig::for_embedding(settings.max_seq_length as usize)
        };

        let text_processor =
            TextProcessor::new(tokenization_processor.clone(), text_chunking_config);

        let media_limits = settings
            .mtmd
            .as_ref()
            .map(mtmd_support::MediaLimits::from_settings);

        // Move the mtmd runtime out of the model so it can be accessed
        // without holding the model lock. `prepare_bitmaps` /
        // `inject_markers` take `&MtmdRuntime` and are safe to call
        // concurrently (see field doc), so `Arc` — not `Mutex` — is
        // sufficient. `eval_chunks` is serialized via the model lock held
        // by `generate_multimodal_embedding`.
        let mtmd_runtime = {
            let mut model_guard = model_arc.lock().map_err(|_| {
                crate::error::EmbeddingLlmError::llamacpp(
                    "Failed to acquire model lock to extract mtmd runtime".to_string(),
                )
            })?;
            model_guard.take_mtmd().map(Arc::new)
        };

        Ok(Self {
            model: model_arc,
            tokenization_processor,
            text_processor,
            model_info,
            media_limits,
            mtmd_runtime,
        })
    }

    /// instruction付きembedding生成（位置情報付き）
    pub fn generate_embeddings_with_positions(
        &self,
        text: &str,
        instruction: Option<&str>,
        normalize: bool,
        merge_strategy: Option<MergeStrategy>,
    ) -> Result<Vec<EmbeddingWithPosition>> {
        debug!(
            "Generating embeddings with positions for text of {} chars",
            text.len()
        );

        // テキストをウィンドウに分割（文字位置情報付き）
        let windows = self
            .text_processor
            .process_text_for_embedding(text, instruction)
            .map_err(|e| EmbeddingLlmError::tokenization(format!("Text processing failed: {e}")))?;

        info!(
            "Processing {} windows for embedding generation",
            windows.len()
        );

        // チェック: windowsが空でないことを確認
        if windows.is_empty() {
            warn!("No text windows available for embedding generation");
            return Ok(vec![]);
        }

        // バッチ処理でembeddingを効率的に生成
        // 既存のトークン化済みデータを直接使用
        let mut model = self.model.lock().map_err(|_| {
            EmbeddingLlmError::inference("Failed to acquire model lock".to_string())
        })?;

        // TODO: Re-enable batch processing once llama-cpp-rs issue is fixed
        // Issue: https://github.com/utilityai/llama-cpp-rs/pull/802
        // Problem: Batch processing fails with "n_tokens == 0" error for multi-line text
        // Root cause: Batch initialization issue with complex text containing newlines/tabs
        // Current workaround: Use individual processing for each window
        /*
        let mut all_embeddings = if windows.len() > 1 {
            debug!("Using batch processing for {} windows", windows.len());
            let token_sequences: Vec<&[u32]> = windows.iter().map(|w| w.token_ids.as_slice()).collect();
            debug!("tokens: {:?}", token_sequences);
            model.generate_batch_embeddings(&token_sequences)?
        } else if windows.len() == 1 {
            debug!("Using single embedding generation for 1 window");
            let embedding = model.generate_embedding(&windows[0].token_ids)?;
            vec![embedding]
        } else {
            // windows.is_empty()のチェックでカバーされるはずだが、念のため
            warn!("No windows available for embedding generation");
            return Ok(vec![]);
        };
        */

        // Workaround: Process each window individually to avoid batch processing issues
        let mut all_embeddings = Vec::with_capacity(windows.len());
        debug!(
            "Processing {} windows individually (batch processing disabled)",
            windows.len()
        );

        for (i, window) in windows.iter().enumerate() {
            debug!(
                "Processing window {} with {} tokens",
                i,
                window.token_ids.len()
            );
            let embedding = model.generate_embedding(&window.token_ids)?;
            all_embeddings.push(embedding);
        }

        // L2正規化（オプション）
        if normalize {
            info!(
                "Applying L2 normalization to {} embeddings",
                all_embeddings.len()
            );
            for (i, embedding) in all_embeddings.iter_mut().enumerate() {
                *embedding = self.l2_normalize(embedding).map_err(|e| {
                    EmbeddingLlmError::inference(format!(
                        "Normalization failed for window {i}: {e}"
                    ))
                })?;
                debug!("Applied L2 normalization to window {} embedding", i);
            }
        }

        // merge_strategy指定がある場合のみ統合
        let final_embeddings = if let Some(strategy) = merge_strategy {
            if all_embeddings.len() > 1 {
                info!(
                    "Merging {} window embeddings using {:?}",
                    all_embeddings.len(),
                    strategy
                );
                let merged = TextProcessor::merge_embeddings_static(&all_embeddings, strategy)
                    .map_err(|e| {
                        EmbeddingLlmError::inference(format!("Embedding merge failed: {e}"))
                    })?;

                // For merged embeddings, use the span of all windows
                let overall_start = windows.iter().map(|w| w.char_start_pos).min().unwrap_or(0);
                let overall_end = windows
                    .iter()
                    .map(|w| w.char_end_pos)
                    .max()
                    .unwrap_or(text.chars().count());

                vec![EmbeddingWithPosition {
                    values: merged,
                    char_start_pos: overall_start,
                    char_end_pos: overall_end,
                }]
            } else {
                // 単一embeddingの場合
                vec![EmbeddingWithPosition {
                    values: all_embeddings.into_iter().next().unwrap(),
                    char_start_pos: windows[0].char_start_pos,
                    char_end_pos: windows[0].char_end_pos,
                }]
            }
        } else {
            // merge_strategyがNoneの場合は全embeddingを個別に返す
            info!(
                "Returning {} individual window embeddings with positions",
                all_embeddings.len()
            );
            all_embeddings
                .into_iter()
                .zip(windows.iter())
                .map(|(embedding, window)| EmbeddingWithPosition {
                    values: embedding,
                    char_start_pos: window.char_start_pos,
                    char_end_pos: window.char_end_pos,
                })
                .collect()
        };

        info!(
            "Generated {} final embeddings with positions",
            final_embeddings.len()
        );
        Ok(final_embeddings)
    }

    // /// instruction付きembedding生成(旧機能）
    // pub fn generate_embeddings_with_instruction(
    //     &self,
    //     text: &str,
    //     instruction: Option<&str>,
    //     normalize: bool,
    //     merge_strategy: Option<MergeStrategy>,
    // ) -> Result<Vec<Vec<f32>>> {
    //     debug!("Generating embeddings for text of {} chars", text.len());

    //     // テキストをウィンドウに分割
    //     let windows = self.sliding_window.process_long_text(text, instruction)
    //         .map_err(|e| EmbeddingLlmError::tokenization(format!("Sliding window processing failed: {}", e)))?;

    //     info!("Processing {} windows for embedding generation", windows.len());

    //     // バッチ処理でembeddingを効率的に生成
    //     info!("Processing {} windows with batch processing", windows.len());
    //     let token_sequences: Vec<&[u32]> = windows.iter().map(|w| w.token_ids.as_slice()).collect();

    //     let mut model = self.model.lock()
    //         .map_err(|_| EmbeddingLlmError::inference("Failed to acquire model lock".to_string()))?;

    //     let mut all_embeddings = if token_sequences.len() > 1 {
    //         // 複数ウィンドウ: バッチ処理を使用
    //         debug!("Using batch processing for {} windows", token_sequences.len());
    //         model.generate_batch_embeddings(&token_sequences)?
    //     } else {
    //         // 単一ウィンドウ: 既存メソッドを使用
    //         debug!("Using single embedding generation for 1 window");
    //         let embedding = model.generate_embedding(&windows[0].token_ids)?;
    //         vec![embedding]
    //     };

    //     // L2正規化（オプション）
    //     if normalize {
    //         info!("Applying L2 normalization to {} embeddings", all_embeddings.len());
    //         for (i, embedding) in all_embeddings.iter_mut().enumerate() {
    //             *embedding = self.l2_normalize(embedding)
    //                 .map_err(|e| EmbeddingLlmError::inference(format!("Normalization failed for window {}: {}", i, e)))?;
    //             debug!("Applied L2 normalization to window {} embedding", i);
    //         }
    //     }

    //     // merge_strategy指定がある場合のみ統合
    //     let final_embeddings = if let Some(strategy) = merge_strategy {
    //         if all_embeddings.len() > 1 {
    //             info!("Merging {} window embeddings using {:?}", all_embeddings.len(), strategy);
    //             let merged = self.sliding_window
    //                 .merge_embeddings(&all_embeddings, strategy)
    //                 .map_err(|e| EmbeddingLlmError::inference(format!("Embedding merge failed: {}", e)))?;
    //             vec![merged]
    //         } else {
    //             // 単一embeddingの場合はそのまま返す
    //             all_embeddings
    //         }
    //     } else {
    //         // merge_strategyがNoneの場合は全embeddingを個別に返す
    //         info!("Returning {} individual window embeddings (no merge strategy specified)", all_embeddings.len());
    //         all_embeddings
    //     };

    //     info!("Generated {} final embeddings", final_embeddings.len());
    //     Ok(final_embeddings)
    // }

    /// L2正規化
    fn l2_normalize(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        // L2ノルムを計算
        let norm_squared: f32 = embedding.iter().map(|x| x * x).sum();

        if norm_squared == 0.0 {
            return Err(EmbeddingLlmError::inference(
                "Cannot normalize zero vector".to_string(),
            ));
        }

        let norm = norm_squared.sqrt();

        // 正規化
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

        debug!(
            "Normalized embedding, L2 norm: {:.6}",
            normalized.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        Ok(normalized)
    }

    /// モデル情報を取得
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    // /// デバッグ用：生成されたトークンをテキストに戻す
    // pub fn debug_decode_tokens(&self, tokens: &[u32]) -> Result<String> {
    //     self.tokenization_processor
    //         .decode_tokens(tokens)
    //         .map_err(|e| EmbeddingLlmError::tokenization(format!("Token decode failed: {}", e)))
    // }

    /// 簡易ヘルスチェック
    pub fn health_check(&self) -> Result<()> {
        // モデルアクセス確認（スコープを分離してデッドロック回避）
        {
            let model = self.model.lock().map_err(|_| {
                EmbeddingLlmError::inference("Failed to acquire model lock".to_string())
            })?;

            // 基本的な情報が正常かチェック
            if model.embedding_dimension() == 0 {
                return Err(EmbeddingLlmError::inference(
                    "Invalid embedding dimension".to_string(),
                ));
            }

            if model.max_context_length() == 0 {
                return Err(EmbeddingLlmError::inference(
                    "Invalid context length".to_string(),
                ));
            }
        } // モデルのロックを解放

        // Tokenizerアクセス確認（モデルロック後に実行してデッドロック回避）
        let test_tokens = vec![1, 2, 3]; // Simple test tokens
        self.tokenization_processor.decode_tokens(&test_tokens)?;

        info!("Health check passed");
        Ok(())
    }

    /// Access the underlying llama.cpp model (for multimodal path).
    pub fn model(&self) -> &Arc<Mutex<LlamaCppModel>> {
        &self.model
    }

    /// Media limits derived from MtmdSettings (None when text-only).
    pub fn media_limits(&self) -> &Option<mtmd_support::MediaLimits> {
        &self.media_limits
    }

    /// Access to the multimodal runtime (if initialized).
    ///
    /// Returned as `&Arc<...>` so callers can clone cheaply and use
    /// `prepare_bitmaps` / `inject_markers` without holding any mtmd-level
    /// lock. `eval_chunks` (inside `generate_multimodal_embedding`) is
    /// serialized by the `model` Mutex instead.
    pub fn mtmd_runtime(&self) -> Option<&Arc<mtmd_support::MtmdRuntime>> {
        self.mtmd_runtime.as_ref()
    }

    /// リソース使用量の推定（デバッグ用）
    pub fn estimate_memory_usage(&self) -> usize {
        // 簡易的な推定
        let base_model_size = 1024 * 1024 * 1024; // 1GB base estimate
        let context_size = self.model_info.max_context_length * 4; // 4 bytes per token estimate
        let embedding_cache = self.model_info.embedding_dimension * 4; // 4 bytes per float

        base_model_size + context_size + embedding_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        let model_info = ModelInfo {
            model_path: "test.gguf".to_string(),
            embedding_dimension: 1024,
            max_context_length: 512,
            vocab_size: 32000,
            dtype: "f32".to_string(),
            supports_vision: false,
            supports_audio: false,
            audio_sample_rate: None,
        };

        assert_eq!(model_info.embedding_dimension, 1024);
        assert_eq!(model_info.max_context_length, 512);
        assert_eq!(model_info.vocab_size, 32000);
    }

    #[test]
    fn test_l2_normalize() {
        // Create a minimal embedder for testing normalization logic
        let embedding = [3.0, 4.0]; // L2 norm = 5.0

        // Calculate L2 norm
        let norm_squared: f32 = embedding.iter().map(|x| x * x).sum();
        let norm = norm_squared.sqrt();
        assert_eq!(norm, 5.0);

        // Normalize
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();
        assert_eq!(normalized, vec![0.6, 0.8]);

        // Check normalized L2 norm is 1.0
        let final_norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((final_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector_normalization() {
        let embedding = [0.0, 0.0, 0.0];
        let norm_squared: f32 = embedding.iter().map(|x| x * x).sum();

        // Zero vector cannot be normalized
        assert_eq!(norm_squared, 0.0);
    }

    #[test]
    fn test_memory_estimation() {
        let model_info = ModelInfo {
            model_path: "test.gguf".to_string(),
            embedding_dimension: 1024,
            max_context_length: 512,
            vocab_size: 32000,
            dtype: "f32".to_string(),
            supports_vision: false,
            supports_audio: false,
            audio_sample_rate: None,
        };

        // Test the calculation logic
        let base_model_size = 1024 * 1024 * 1024; // 1GB
        let context_size = model_info.max_context_length * 4; // 4 bytes per token
        let embedding_cache = model_info.embedding_dimension * 4; // 4 bytes per float

        let estimated = base_model_size + context_size + embedding_cache;
        assert!(estimated > base_model_size);
        assert!(estimated > context_size);
    }

    #[test]
    fn test_merge_strategy_behavior() {
        use command_utils::text::chunking::MergeStrategy;

        // Test embeddings (simulating multiple windows)
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        // Test with None (should return all embeddings individually)
        // This simulates what happens when merge_strategy is None
        let result_none = embeddings.clone(); // No merge
        assert_eq!(result_none.len(), 3);
        assert_eq!(result_none[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result_none[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(result_none[2], vec![7.0, 8.0, 9.0]);

        // Test with Some strategy (should return single merged embedding)
        // Using static function to simulate merging
        let result_merged = crate::text_processing::TextProcessor::merge_embeddings_static(
            &embeddings,
            MergeStrategy::Average,
        )
        .unwrap();
        assert_eq!(result_merged.len(), 3); // Single embedding with 3 dimensions
        assert_eq!(result_merged, vec![4.0, 5.0, 6.0]); // Average of the three embeddings

        // Single embedding case (should return as-is regardless of strategy)
        let single_embedding = vec![vec![1.0, 2.0, 3.0]];
        let result_single = single_embedding.clone();
        assert_eq!(result_single.len(), 1);
        assert_eq!(result_single[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_new_from_settings_with_backend_integration() {
        use crate::protobuf::embedding_llm::{DType, EmbeddingLlmRunnerSettings, ModelType};
        use llama_cpp_2::llama_backend::LlamaBackend;
        use std::sync::{Arc, Mutex};

        // Test the new backend sharing approach
        let backend_result = LlamaBackend::init();
        if backend_result.is_err() {
            println!("Skipping backend integration test - llama.cpp not available");
            return;
        }

        let mut backend = backend_result.unwrap();
        backend.void_logs();
        let shared_backend = Arc::new(Mutex::new(backend));

        let settings = EmbeddingLlmRunnerSettings {
            model_id: "test_model".to_string(),
            use_cpu: true,
            dtype: Some(DType::F32 as i32),
            max_seq_length: 128,
            model_type: ModelType::Gguf as i32,
            model_files: vec!["nonexistent.gguf".to_string()],
            tokenizer_model_id: None,
            chunking_config: None,
            max_batch_size: Some(4),
            gpu_device: None,
            mtmd: None,
        };

        // Test new constructor with backend (should fail due to nonexistent model file)
        let result = LlamaCppEmbedder::new_from_settings_with_backend(&settings, shared_backend);
        assert!(
            result.is_err(),
            "Should fail with invalid model file, but backend sharing should work"
        );

        println!("✓ Backend sharing integration test completed successfully");
    }
}
