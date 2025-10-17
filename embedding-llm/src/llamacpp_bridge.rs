use crate::error::{EmbeddingLlmError, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::LlamaToken,
};
use llama_cpp_sys_2::{LLAMA_FLASH_ATTN_TYPE_AUTO, LLAMA_FLASH_ATTN_TYPE_DISABLED};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// システムメモリ容量をGB単位で取得
fn get_system_memory_gb() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("MemTotal:"))
                    .and_then(|line| {
                        line.split_whitespace()
                            .nth(1)
                            .and_then(|kb_str| kb_str.parse::<usize>().ok())
                            .map(|kb| kb / 1024 / 1024) // KB -> GB
                    })
            })
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|output| {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .map(|bytes| bytes / 1024 / 1024 / 1024) // Bytes -> GB
            })
    }
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        Command::new("wmic")
            .args(&["computersystem", "get", "TotalPhysicalMemory", "/value"])
            .output()
            .ok()
            .and_then(|output| {
                String::from_utf8(output.stdout).ok().and_then(|s| {
                    s.lines()
                        .find(|line| line.starts_with("TotalPhysicalMemory="))
                        .and_then(|line| {
                            line.split('=')
                                .nth(1)
                                .and_then(|bytes_str| bytes_str.parse::<usize>().ok())
                                .map(|bytes| bytes / 1024 / 1024 / 1024) // Bytes -> GB
                        })
                })
            })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// llama.cppモデルのラッパー（プロダクション実装）
pub struct LlamaCppModel {
    backend: Arc<Mutex<LlamaBackend>>,
    model: LlamaModel,
    ctx_params: LlamaContextParams,
    model_path: String,
    n_embd: usize,       // embedding dimension
    n_ctx: usize,        // context size
    vocab_size: usize,   // vocabulary size
    max_batch_size: u32, // maximum batch size for batch processing
    _use_cpu: bool,      // Store for potential future use
}

impl LlamaCppModel {
    /// 既存バックエンドを使用したGGUFモデルの初期化（推奨方式）
    pub fn new_with_backend(
        model_path: &str,
        use_cpu: bool,
        max_seq_length: usize,
        max_batch_size: Option<u32>,
        backend: Arc<Mutex<LlamaBackend>>,
        gpu_device: Option<i32>,
    ) -> Result<Self> {
        // Model file validation
        self::helpers::validate_model_file(model_path)?;

        info!("Using provided llama.cpp backend");

        // Resolve model path (may download from HF if needed)
        let resolved_path = self::helpers::resolve_model_path(model_path)?;

        // モデルパラメータの設定
        let model_params = if use_cpu {
            LlamaModelParams::default()
        } else {
            let mut params = LlamaModelParams::default().with_n_gpu_layers(1000); // Offload all layers to GPU
            
            // GPU device specification (if provided)
            if let Some(device_id) = gpu_device {
                if device_id < 0 {
                    return Err(EmbeddingLlmError::configuration(format!(
                        "Invalid gpu_device: {}. Must be >= 0",
                        device_id
                    )));
                }
                info!("Setting main GPU device to: {}", device_id);
                params = params.with_main_gpu(device_id);
            }
            params
        };

        info!("Loading model from: {}", resolved_path.display());
        let model = {
            let backend_guard = backend.lock().map_err(|_| {
                EmbeddingLlmError::llamacpp("Failed to acquire backend lock".to_string())
            })?;
            LlamaModel::load_from_file(&backend_guard, &resolved_path, &model_params).map_err(
                |e| EmbeddingLlmError::model_loading(format!("Failed to load model: {e}")),
            )?
        };

        Self::build_model_common(
            model_path,
            use_cpu,
            max_seq_length,
            max_batch_size,
            backend,
            model,
        )
    }

    /// LlamaCppModelの共通構築処理
    fn build_model_common(
        model_path: &str,
        use_cpu: bool,
        max_seq_length: usize,
        max_batch_size: Option<u32>,
        backend: Arc<Mutex<LlamaBackend>>,
        model: LlamaModel,
    ) -> Result<Self> {
        // コンテキストパラメータの設定（パフォーマンス最適化）
        let ctx_size = NonZeroU32::new(max_seq_length as u32).ok_or_else(|| {
            EmbeddingLlmError::configuration("max_seq_length must be > 0".to_string())
        })?;

        // バッチサイズ設定（設定値優先、フォールバック有り）
        let optimal_batch_size = if use_cpu {
            // CPU: 設定されたmax_batch_size、または適応的設定
            max_batch_size.unwrap_or_else(|| {
                let system_memory_gb = get_system_memory_gb().unwrap_or(16);
                if system_memory_gb < 16 {
                    16
                } else if system_memory_gb < 32 {
                    32
                } else {
                    64
                }
            })
        } else {
            // GPU: 設定されたmax_batch_size、またはVRAM制約を考慮した小さなデフォルト値
            // より大きな値が必要な場合は外部から設定で指定
            max_batch_size.unwrap_or(4) // GPU向けデフォルト（VRAMに配慮）
        };

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_batch(ctx_size.into()) // Set batch size to match context size for embedding processing
            .with_n_ubatch(ctx_size.into()) // Set physical batch size to match for single-pass embedding
            .with_embeddings(true) // Enable embedding mode
            .with_flash_attention_policy(if use_cpu {
                LLAMA_FLASH_ATTN_TYPE_DISABLED
            } else {
                LLAMA_FLASH_ATTN_TYPE_AUTO
            });

        // モデル情報の取得
        let n_embd = model.n_embd() as usize;
        let vocab_size = model.n_vocab() as usize;

        info!("llama.cpp model loaded:");
        info!("  model_path: {}", model_path);
        info!("  embedding_dim: {}", n_embd);
        info!("  context_size: {}", max_seq_length);
        info!("  vocab_size: {}", vocab_size);
        info!("  use_cpu: {}", use_cpu);
        let memory_info = if let Some(configured) = max_batch_size {
            format!("configured: {configured}")
        } else {
            format!(
                "adaptive based on {}GB system memory",
                get_system_memory_gb().unwrap_or(0)
            )
        };
        info!("  batch_size: {} ({})", optimal_batch_size, memory_info);

        Ok(Self {
            backend,
            model,
            ctx_params,
            model_path: model_path.to_string(),
            n_embd,
            n_ctx: max_seq_length,
            vocab_size,
            max_batch_size: optimal_batch_size,
            _use_cpu: use_cpu,
        })
    }

    /// トークン列からembeddingを生成（プロダクション実装）
    pub fn generate_embedding(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        debug!("Generating embedding for {} tokens", token_ids.len());

        // トークン数の検証
        if token_ids.len() > self.n_ctx {
            return Err(EmbeddingLlmError::inference(format!(
                "Token sequence too long: {} > {}",
                token_ids.len(),
                self.n_ctx
            )));
        }

        if token_ids.is_empty() {
            return Err(EmbeddingLlmError::inference(
                "Empty token sequence".to_string(),
            ));
        }

        // コンテキストを作成
        let mut ctx = {
            let backend_guard = self.backend.lock().map_err(|_| {
                EmbeddingLlmError::llamacpp("Failed to acquire backend lock".to_string())
            })?;
            self.model
                .new_context(&backend_guard, self.ctx_params.clone())
                .map_err(|e| {
                    EmbeddingLlmError::llamacpp(format!("Failed to create context: {e}"))
                })?
        };

        // Convert u32 to LlamaToken（公式実装と同じ方式）
        let llama_tokens: Vec<LlamaToken> = token_ids
            .iter()
            .map(|&t| LlamaToken::new(t as i32))
            .collect();

        // バッチの準備（公式実装に合わせる）
        let n_ctx = self.n_ctx;
        let mut batch = LlamaBatch::new(n_ctx, 1); // n_embd = 1 sequence
        batch.add_sequence(&llama_tokens, 0, false).map_err(|e| {
            EmbeddingLlmError::inference(format!("Failed to add sequence to batch: {e}"))
        })?;

        debug!("Added {} tokens to batch as sequence", llama_tokens.len());

        // KVキャッシュをクリア（公式実装と同様）
        ctx.clear_kv_cache();

        // 推論実行
        ctx.decode(&mut batch)
            .map_err(|e| EmbeddingLlmError::inference(format!("Context decode failed: {e}")))?;

        // embeddings の取得（公式実装と同じ方式: embeddings_seq_ithを使用）
        debug!("Getting sequence embeddings...");
        let embeddings = match ctx.embeddings_seq_ith(0) {
            // sequence 0の embeddings
            Ok(emb) => emb.to_vec(),
            Err(e) => {
                return Err(EmbeddingLlmError::inference(format!(
                    "Failed to get sequence embeddings: {e:?}"
                )))
            }
        };

        debug!("Generated embedding with dimension: {}", embeddings.len());
        debug!(
            "First 10 values: {:?}",
            &embeddings[..10.min(embeddings.len())]
        );

        // 詳細なゼロベクトル診断
        let zero_count = embeddings.iter().filter(|&&x| x == 0.0).count();
        let non_zero_count = embeddings.len() - zero_count;
        let sum: f32 = embeddings.iter().sum();
        let norm_squared: f32 = embeddings.iter().map(|x| x * x).sum();

        debug!("Embedding analysis:");
        debug!("  - Zero values: {}/{}", zero_count, embeddings.len());
        debug!("  - Non-zero values: {}", non_zero_count);
        debug!("  - Sum: {}", sum);
        debug!("  - L2 norm squared: {}", norm_squared);
        debug!("  - L2 norm: {}", norm_squared.sqrt());

        if norm_squared == 0.0 {
            return Err(EmbeddingLlmError::inference(
                format!("Generated zero embedding vector! All {} values are zero. This indicates an issue with llama.cpp embedding computation.", embeddings.len())
            ));
        }

        Ok(embeddings.to_vec())
    }

    /// 複数のトークン列からバッチでembeddingを生成（sliding window最適化）
    pub fn generate_batch_embeddings(
        &mut self,
        token_sequences: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        if token_sequences.is_empty() {
            return Ok(vec![]);
        }

        // バッチサイズによる分割処理
        let mut all_embeddings = Vec::with_capacity(token_sequences.len());
        let batch_size = self.max_batch_size as usize;

        for chunk in token_sequences.chunks(batch_size) {
            debug!(
                "Processing batch of {} sequences (max batch size: {})",
                chunk.len(),
                batch_size
            );

            if chunk.len() == 1 {
                // 単一シーケンスは既存メソッドを使用
                let embedding = self.generate_embedding(chunk[0])?;
                all_embeddings.push(embedding);
            } else {
                // 複数シーケンスはバッチ処理
                let batch_embeddings = self.generate_batch_embeddings_internal(chunk)?;
                all_embeddings.extend(batch_embeddings);
            }
        }

        Ok(all_embeddings)
    }

    /// 内部バッチ処理実装（max_batch_size以下での処理）
    fn generate_batch_embeddings_internal(
        &mut self,
        token_sequences: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        debug!(
            "Generating batch embeddings for {} sequences",
            token_sequences.len()
        );

        // 全シーケンスの長さをチェック
        for (i, tokens) in token_sequences.iter().enumerate() {
            if tokens.len() > self.n_ctx {
                return Err(EmbeddingLlmError::inference(format!(
                    "Sequence {} too long: {} > {}",
                    i,
                    tokens.len(),
                    self.n_ctx
                )));
            }
            if tokens.is_empty() {
                return Err(EmbeddingLlmError::inference(format!(
                    "Empty token sequence at index {i}"
                )));
            }
        }

        // コンテキストを作成
        let mut ctx = {
            let backend_guard = self.backend.lock().map_err(|_| {
                EmbeddingLlmError::llamacpp("Failed to acquire backend lock".to_string())
            })?;
            self.model
                .new_context(&backend_guard, self.ctx_params.clone())
                .map_err(|e| {
                    EmbeddingLlmError::llamacpp(format!("Failed to create context: {e}"))
                })?
        };

        // バッチの準備（複数シーケンス対応）
        let n_ctx = self.n_ctx;
        let n_sequences = token_sequences.len();

        // シーケンス数がmax_batch_sizeを超えないかチェック
        if n_sequences > self.max_batch_size as usize {
            return Err(EmbeddingLlmError::inference(format!(
                "Number of sequences ({}) exceeds max batch size ({}). Consider reducing batch size.",
                n_sequences, self.max_batch_size
            )));
        }

        // LlamaBatch作成（公式実装に合わせる: n_ctx, n_sequences）
        let mut batch = LlamaBatch::new(n_ctx, n_sequences as i32);

        // 各シーケンスをバッチに追加
        for (seq_id, tokens) in token_sequences.iter().enumerate() {
            let llama_tokens: Vec<LlamaToken> =
                tokens.iter().map(|&t| LlamaToken::new(t as i32)).collect();
            batch
                .add_sequence(&llama_tokens, seq_id as i32, false)
                .map_err(|e| {
                    EmbeddingLlmError::inference(format!(
                        "Failed to add sequence {seq_id} to batch: {e}"
                    ))
                })?;
        }

        debug!("Added {} sequences to batch", n_sequences);

        // KVキャッシュをクリア（シーケンス間の独立性を保つ）
        ctx.clear_kv_cache();

        // 推論実行
        ctx.decode(&mut batch)
            .map_err(|e| EmbeddingLlmError::inference(format!("Batch decode failed: {e}")))?;

        // 各シーケンスのembeddingsを取得
        let mut embeddings = Vec::with_capacity(n_sequences);
        for seq_id in 0..n_sequences {
            debug!("Getting embeddings for sequence {}", seq_id);
            let sequence_embeddings = match ctx.embeddings_seq_ith(seq_id as i32) {
                Ok(emb) => emb.to_vec(),
                Err(e) => {
                    return Err(EmbeddingLlmError::inference(format!(
                        "Failed to get embeddings for sequence {seq_id}: {e:?}"
                    )))
                }
            };

            // ゼロベクトルチェック
            let norm_squared: f32 = sequence_embeddings.iter().map(|x| x * x).sum();
            if norm_squared == 0.0 {
                return Err(EmbeddingLlmError::inference(format!(
                    "Generated zero embedding for sequence {}! All {} values are zero.",
                    seq_id,
                    sequence_embeddings.len()
                )));
            }

            debug!(
                "Generated embedding for sequence {} with dimension: {} (L2 norm: {})",
                seq_id,
                sequence_embeddings.len(),
                norm_squared.sqrt()
            );
            embeddings.push(sequence_embeddings);
        }

        debug!(
            "Successfully generated {} batch embeddings",
            embeddings.len()
        );
        Ok(embeddings)
    }

    /// 文字列からトークン化
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let tokens = self
            .model
            .str_to_token(
                text,
                if add_bos {
                    AddBos::Always
                } else {
                    AddBos::Never
                },
            )
            .map_err(|e| EmbeddingLlmError::tokenization(format!("Failed to tokenize: {e}")))?;

        Ok(tokens.into_iter().map(|t| t.0 as u32).collect())
    }

    /// トークンから文字列に変換
    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let llama_tokens: Vec<LlamaToken> =
            tokens.iter().map(|&t| LlamaToken::new(t as i32)).collect();
        self.model
            .token_to_str(llama_tokens[0], llama_cpp_2::model::Special::Tokenize)
            .map_err(|e| EmbeddingLlmError::tokenization(format!("Failed to detokenize: {e}")))
    }

    /// embedding次元数を取得
    pub fn embedding_dimension(&self) -> usize {
        self.n_embd
    }

    /// 最大コンテキスト長を取得
    pub fn max_context_length(&self) -> usize {
        self.n_ctx
    }

    /// 語彙サイズを取得
    pub fn vocabulary_size(&self) -> usize {
        self.vocab_size
    }

    /// モデルパス取得（デバッグ用）
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// EOS token ID を取得
    pub fn eos_token_id(&self) -> u32 {
        self.model.token_eos().0 as u32
    }

    /// BOS token ID を取得  
    pub fn bos_token_id(&self) -> u32 {
        self.model.token_bos().0 as u32
    }
}

/// llama.cpp統合用のヘルパー関数
pub mod helpers {
    use super::*;

    /// モデルファイルの存在確認とパス解決
    pub fn validate_model_file(model_path: &str) -> Result<()> {
        let path = Path::new(model_path);

        // If it's already an existing file, validate it
        if path.exists() {
            if path.is_file() && path.extension().map(|e| e == "gguf").unwrap_or(false) {
                return Ok(());
            } else {
                return Err(EmbeddingLlmError::configuration(format!(
                    "File exists but is not a GGUF file: {model_path}"
                )));
            }
        }

        // If it's not an existing file, it might be a HuggingFace model path
        // We'll let resolve_model_path handle the download
        Ok(())
    }

    /// モデルパスの解決（HuggingFaceからの自動ダウンロード対応）
    pub fn resolve_model_path(model_path: &str) -> Result<PathBuf> {
        let path = Path::new(model_path);

        // If it's already an existing file, return it
        if path.exists() && path.is_file() {
            return Ok(path.to_path_buf());
        }

        // Try to parse as HuggingFace model (format: "repo/file.gguf" or just "file.gguf")
        if model_path.contains('/') && model_path.ends_with(".gguf") {
            // Split into repo and filename from the last '/'
            if let Some(last_slash_pos) = model_path.rfind('/') {
                let repo = &model_path[..last_slash_pos];
                let filename = &model_path[last_slash_pos + 1..];

                info!("Downloading model from HuggingFace: {}/{}", repo, filename);

                let api = ApiBuilder::new()
                    .with_progress(true)
                    .build()
                    .map_err(|e| {
                        EmbeddingLlmError::hf_hub(format!("Failed to create HF API: {e}"))
                    })?
                    .model(repo.to_string());

                let downloaded_path = api.get(filename).map_err(|e| {
                    EmbeddingLlmError::hf_hub(format!("Failed to download model: {e}"))
                })?;

                info!("Downloaded model to: {}", downloaded_path.display());
                return Ok(downloaded_path);
            }
        }

        Err(EmbeddingLlmError::configuration(format!(
            "Cannot resolve model path: {model_path}. Expected existing file or HuggingFace format 'repo/model.gguf'"
        )))
    }

    /// GPUメモリ使用量の推定
    pub fn estimate_gpu_memory(model_path: &str) -> Result<usize> {
        let resolved_path = resolve_model_path(model_path)?;
        let metadata = std::fs::metadata(resolved_path)?;
        let file_size = metadata.len() as usize;

        // 概算：ファイルサイズ + コンテキストバッファ + その他
        Ok(file_size + 1024 * 1024 * 512) // +512MB buffer
    }

    /// モデルのメタデータ情報を取得
    pub fn get_model_metadata(model_path: &str) -> Result<ModelMetadata> {
        let resolved_path = resolve_model_path(model_path)?;
        let metadata = std::fs::metadata(&resolved_path)?;

        Ok(ModelMetadata {
            path: resolved_path,
            size_bytes: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

/// モデルメタデータ情報
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub modified: Option<std::time::SystemTime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_model_file() {
        // Test non-existent file that could be HF model
        let result = helpers::validate_model_file("microsoft/phi-4-gguf/phi-4-Q4_0.gguf");
        assert!(result.is_ok()); // Should pass validation, will be resolved later

        // Test invalid format
        let result = helpers::validate_model_file("not-a-model.txt");
        assert!(result.is_ok()); // We allow non-existing files to pass initial validation
    }

    #[test]
    fn test_hf_model_path_parsing() {
        // Test HuggingFace model path parsing logic
        let model_path = "microsoft/phi-4-gguf/phi-4-Q4_0.gguf";
        if let Some(last_slash_pos) = model_path.rfind('/') {
            let repo = &model_path[..last_slash_pos];
            let filename = &model_path[last_slash_pos + 1..];

            assert_eq!(filename, "phi-4-Q4_0.gguf");
            assert_eq!(repo, "microsoft/phi-4-gguf");
        } else {
            panic!("Failed to find last slash");
        }
    }

    #[test]
    fn test_estimate_gpu_memory() {
        // This test requires a valid GGUF file to exist
        let result = helpers::estimate_gpu_memory("/nonexistent/path.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_system_memory_detection() {
        let memory_gb = get_system_memory_gb();
        println!("Detected system memory: {memory_gb:?} GB");
        // System should have at least 1GB of memory
        assert!(memory_gb.is_none() || memory_gb.unwrap() >= 1);
    }
}
