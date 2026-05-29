#![allow(improper_ctypes_definitions)]

pub mod chunking_adapter;
pub mod embedding;
pub mod error;
pub mod llamacpp_bridge;
pub mod text_processing;
pub mod token_position;
pub mod tokenization;

use anyhow::{Result, anyhow};
use command_utils::trace::Tracing;
use jobworkerp_plugin_abi::v2::{CancelToken, HighLevelSink, PluginV2};
use jobworkerp_plugin_abi_macros::register_plugin_v2;
use prost::Message;
use proto::jobworkerp::data::{MethodJsonSchema, MethodSchema, StreamingOutputType};
use std::collections::HashMap;

use crate::embedding::LlamaCppEmbedder;
use llama_cpp_2::llama_backend::LlamaBackend;
use std::sync::{Arc, Mutex, OnceLock};

/// Generated protobuf modules
pub mod protobuf {
    pub mod embedding_llm {
        include!(concat!(env!("OUT_DIR"), "/embedding_llm.rs"));
    }
}

/// グローバルバックエンドのシングルトン
static GLOBAL_BACKEND: OnceLock<Arc<Mutex<LlamaBackend>>> = OnceLock::new();

pub const METHOD_RUN: &str = "run";

/// Error-string surface for cooperative cancellation, shared with the host's
/// cancellation contract. Callers may match against this exact string.
pub const CANCELLED: &str = "cancelled";

/// Serialize a `schemars::JsonSchema` type into a JSON string for the
/// `MethodJsonSchema` / settings_schema slots.
fn json_schema_string<T: schemars::JsonSchema>(label: &str) -> String {
    let schema = schemars::schema_for!(T);
    match serde_json::to_string(&schema) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("error in {label}: {e:?}");
            String::new()
        }
    }
}

/// Main plugin structure for embedding-llm
pub struct EmbeddingLlmRunnerPlugin {
    backend: Option<Arc<Mutex<LlamaBackend>>>,
    // Shared so it can be cloned into the `spawn_blocking` closure backing
    // V2's `run`. `LlamaCppEmbedder` exposes `&self` methods for the embedding
    // path; the inner `model` Mutex serializes the actual llama.cpp call.
    embedder: Option<Arc<LlamaCppEmbedder>>,
    /// Plugin-owned tokio runtime. The dylib's `tokio` has its own
    /// `thread_local!` reactor that the host cannot share — long-running work
    /// is spawned here. Held in an `Option` so `Drop` can `shutdown_background`
    /// the runtime instead of blocking the current thread (which would panic
    /// when dropped from inside an async context like `tokio::test`).
    rt: Option<tokio::runtime::Runtime>,
    /// Refreshed before each job via `set_cancellation_token`.
    token: Option<CancelToken>,
}

impl EmbeddingLlmRunnerPlugin {
    pub const RUNNER_NAME: &'static str = "EmbeddingLlmRunner";

    pub fn new() -> Result<Self> {
        tracing::info!("Creating EmbeddingLlmRunner plugin (backend will be initialized on load)");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("embedding-llm-plugin")
            .build()
            .map_err(|e| anyhow!("failed to build plugin tokio runtime: {e}"))?;
        Ok(Self {
            backend: None,
            embedder: None,
            rt: Some(rt),
            token: None,
        })
    }

    fn rt_handle(&self) -> tokio::runtime::Handle {
        self.rt
            .as_ref()
            .expect("plugin runtime accessed after Drop")
            .handle()
            .clone()
    }
}

impl Drop for EmbeddingLlmRunnerPlugin {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
}

impl EmbeddingLlmRunnerPlugin {
    /// Ensure backend is initialized (lazy initialization)
    fn ensure_backend(&mut self) -> Result<Arc<Mutex<LlamaBackend>>> {
        if let Some(backend) = &self.backend {
            return Ok(backend.clone());
        }

        // Initialize global backend on first use (in load() method)
        let backend = GLOBAL_BACKEND
            .get_or_init(|| {
                tracing::info!("Initializing global llama.cpp backend for embedding plugin");
                let mut backend = LlamaBackend::init().expect("Failed to init llama.cpp backend");
                backend.void_logs(); // Disable llama.cpp logs
                Arc::new(Mutex::new(backend))
            })
            .clone();

        tracing::info!("Using shared llama.cpp backend for embedding plugin");
        self.backend = Some(backend.clone());
        Ok(backend)
    }
}

impl Default for EmbeddingLlmRunnerPlugin {
    fn default() -> Self {
        Self::new().expect("Failed to create plugin")
    }
}

impl Tracing for EmbeddingLlmRunnerPlugin {}

impl EmbeddingLlmRunnerPlugin {
    fn load_sync(&mut self, settings: Vec<u8>) -> Result<()> {
        // command_utils::util::tracing::tracing_init_test(tracing::Level::DEBUG);
        let settings = protobuf::embedding_llm::EmbeddingLlmRunnerSettings::decode(&settings[..])?;

        // llama.cpp版の検証
        if settings.model_type != (protobuf::embedding_llm::ModelType::Gguf as i32) {
            return Err(anyhow!("llama-cpp version only supports GGUF models"));
        }

        if settings.model_files.is_empty() {
            return Err(anyhow!("GGUF model files must be specified"));
        }

        // Initialize backend here (lazy initialization - only when actually loading model)
        let backend = self.ensure_backend()?;

        // llama.cpp embedderの初期化（バックエンドを渡す）
        let embedder = LlamaCppEmbedder::new_from_settings_with_backend(&settings, backend)?;

        self.embedder = Some(Arc::new(embedder));

        tracing::info!(
            "{} loaded: model_id={}, files={:?}",
            Self::RUNNER_NAME,
            settings.model_id,
            settings.model_files
        );

        Ok(())
    }

    fn run_sync(
        embedder: Option<Arc<LlamaCppEmbedder>>,
        arg: Vec<u8>,
        metadata: &HashMap<String, String>,
    ) -> Result<Vec<u8>> {
        // OpenTelemetryスパンの作成（metadataから親コンテキストを抽出）
        let mut span = EmbeddingLlmRunnerPlugin::otel_span_from_metadata(
            metadata,
            "embedding-llm",
            "embedding.run",
        );

        let result: Result<Vec<u8>> = (|| -> Result<Vec<u8>> {
            let args = protobuf::embedding_llm::EmbeddingArgs::decode(&arg[..])?;

            // Reject when both text and medias are empty (contract in embedding_args.proto)
            if args.text.is_empty() && args.medias.is_empty() {
                return Err(anyhow!("InvalidArgument: both text and medias are empty"));
            }

            // Validate pooling_type early, before any bitmap decoding or
            // model work. proto3 open-enum lets callers stuff arbitrary i32
            // into the wire format, so we must explicitly reject anything
            // that is not a defined `PoolingType` variant — otherwise the
            // multimodal path would silently fold unknown values into the
            // model default and hide caller mistakes.
            use protobuf::embedding_llm::PoolingType;
            let pooling = PoolingType::try_from(args.pooling_type).map_err(|_| {
                anyhow!(
                    "InvalidArgument: unknown pooling_type {}; must be one of \
                     UNSPECIFIED(0), NONE(1), MEAN(2), CLS(3), LAST(4)",
                    args.pooling_type
                )
            })?;

            // Args validation runs before the "not initialized" check so
            // syntactically-bad requests surface their contract error
            // ("InvalidArgument: ...") instead of a load-time message.
            let embedder_arc = embedder
                .as_ref()
                .ok_or_else(|| anyhow!("Embedder not initialized"))?;
            let embedder: &LlamaCppEmbedder = embedder_arc.as_ref();

            let model_info = embedder.model_info();

            let build_proto_model_info = || protobuf::embedding_llm::ModelInfo {
                model_name: format!("llama.cpp-{}", model_info.model_path),
                embedding_dimension: model_info.embedding_dimension as u32,
                dtype_used: model_info.dtype.clone(),
                supports_vision: model_info.supports_vision,
                supports_audio: model_info.supports_audio,
                audio_sample_rate: model_info.audio_sample_rate,
            };

            let result = if args.medias.is_empty() {
                // === Text-only path (unchanged) ===
                // pooling_type is only wired through the multimodal path today;
                // text-only uses the model's built-in pooling_type from GGUF.
                // Reject anything other than UNSPECIFIED here instead of silently
                // ignoring the request, so the API contract matches the
                // implementation.
                if pooling != PoolingType::Unspecified {
                    anyhow::bail!(
                        "pooling_type is only supported for multimodal embedding \
                         (text + media) at the moment. Text-only input uses the \
                         model's built-in pooling type from the GGUF. Pass \
                         pooling_type=UNSPECIFIED(0) for text-only input."
                    );
                }
                let embeddings_with_positions = embedder.generate_embeddings_with_positions(
                    &args.text,
                    args.instruction.as_deref(),
                    args.normalize_embeddings,
                    None,
                )?;

                protobuf::embedding_llm::EmbeddingLlmResult {
                    embeddings: embeddings_with_positions
                        .into_iter()
                        .map(|embedding_with_pos| {
                            let content = args
                                .text
                                .chars()
                                .skip(embedding_with_pos.char_start_pos)
                                .take(
                                    embedding_with_pos
                                        .char_end_pos
                                        .saturating_sub(embedding_with_pos.char_start_pos),
                                )
                                .collect::<String>();

                            protobuf::embedding_llm::embedding_llm_result::Embedding {
                                values: embedding_with_pos.values,
                                begin_position: embedding_with_pos.char_start_pos as u32,
                                end_position: embedding_with_pos.char_end_pos as u32,
                                content,
                            }
                        })
                        .collect(),
                    model_info: Some(build_proto_model_info()),
                }
            } else {
                // === Multimodal path ===
                //
                // Lock strategy: `MtmdRuntime` exposes only `&self` methods.
                // `prepare_bitmaps` and `inject_markers` are upstream-documented
                // as thread-safe on a shared `MtmdContext`, so we run them
                // *without* any mtmd-level lock — concurrent requests can
                // decode images / validate markers in parallel on blocking
                // threads. Only `eval_chunks` (inside
                // `generate_multimodal_embedding`) must be serialized on the
                // shared `MtmdContext`; that serialization is provided by the
                // existing `model` Mutex we acquire below.
                let mtmd = embedder.mtmd_runtime().ok_or_else(|| {
                    anyhow!("multimodal input given but mmproj is not configured")
                })?;

                let media_limits = match embedder.media_limits() {
                    Some(l) => l.clone(),
                    None => mtmd_support::MediaLimits::default(),
                };

                let bitmaps = mtmd
                    .prepare_bitmaps(&args.medias, &media_limits)
                    .map_err(|e| anyhow::anyhow!(e).context("mtmd: preparing bitmaps"))?;

                // instruction is treated as metadata only (same as text-only
                // path, where chunking_adapter tokenizes without instruction).
                // We warn if a caller passes one so the silent no-op is
                // observable, but do not reject or prepend it.
                if args.instruction.is_some() {
                    tracing::warn!(
                        "instruction is ignored for multimodal embedding \
                         (consistent with text-only behavior: treated as metadata)"
                    );
                }

                let prompt = mtmd
                    .inject_markers(&args.text, bitmaps.len())
                    .map_err(|e| anyhow::anyhow!(e).context("mtmd: injecting markers"))?;

                let char_count = args.text.chars().count();

                let model_guard = embedder.model().lock().map_err(|_| {
                    anyhow!("Failed to acquire model lock for multimodal embedding")
                })?;
                let values = model_guard
                    .generate_multimodal_embedding(
                        mtmd,
                        &prompt,
                        &bitmaps,
                        args.normalize_embeddings,
                        pooling,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}").context("mtmd: generate embedding"))?;
                drop(model_guard);

                tracing::info!(
                    "multimodal embedding: medias={}, text_chars={}, dim={}",
                    args.medias.len(),
                    char_count,
                    values.len()
                );

                // Position metadata refers to the text portion only.
                // For media-only input (text=""), this yields begin=0, end=0, content="".
                // The embedding values themselves cover the full multimodal input.
                protobuf::embedding_llm::EmbeddingLlmResult {
                    embeddings: vec![protobuf::embedding_llm::embedding_llm_result::Embedding {
                        values,
                        begin_position: 0,
                        end_position: char_count as u32,
                        content: args.text.clone(),
                    }],
                    model_info: Some(build_proto_model_info()),
                }
            };

            let mut buf = Vec::with_capacity(result.encoded_len());
            result.encode(&mut buf)?;
            Ok(buf)
        })();

        // トレーシング情報の記録
        match &result {
            Ok(result_buf) => {
                EmbeddingLlmRunnerPlugin::trace_response(&mut span, result_buf);
            }
            Err(e) => {
                EmbeddingLlmRunnerPlugin::trace_error(&mut span, e.as_ref());
            }
        }

        result
    }
}

/// Resolved EmbeddingArgs proto with media-input imports inlined.
fn args_proto_resolved() -> String {
    static RESOLVED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            jobworkerp_llama_protobuf::proto_resolve::resolve_proto_imports(
                include_str!("../protobuf/embedding_args.proto"),
                &[jobworkerp_llama_protobuf::proto_resolve::MEDIA_INPUT_IMPORT],
            )
            .expect("EmbeddingLlmRunner: args proto resolution failed")
        })
        .clone()
}

fn settings_proto_resolved() -> String {
    static RESOLVED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            jobworkerp_llama_protobuf::proto_resolve::resolve_proto_imports(
                include_str!("../protobuf/llm_runner_settings.proto"),
                &[jobworkerp_llama_protobuf::proto_resolve::MEDIA_INPUT_IMPORT],
            )
            .expect("EmbeddingLlmRunner: settings proto resolution failed")
        })
        .clone()
}

#[async_trait::async_trait]
impl PluginV2 for EmbeddingLlmRunnerPlugin {
    fn name(&self) -> String {
        String::from(Self::RUNNER_NAME)
    }

    fn description(&self) -> String {
        String::from(
            "Generate text embeddings with positional information and optional instruction prefixes",
        )
    }

    fn runner_settings_proto(&self) -> String {
        settings_proto_resolved()
    }

    fn method_proto_map(&self) -> HashMap<String, Vec<u8>> {
        static CACHED: std::sync::OnceLock<HashMap<String, Vec<u8>>> = std::sync::OnceLock::new();
        CACHED
            .get_or_init(|| {
                let mut schemas = HashMap::new();
                schemas.insert(
                    METHOD_RUN.to_string(),
                    MethodSchema {
                        args_proto: args_proto_resolved(),
                        result_proto: include_str!("../protobuf/llm_result.proto").to_string(),
                        description: Some(
                            "Generate text/multimodal embeddings with positional information"
                                .to_string(),
                        ),
                        output_type: StreamingOutputType::NonStreaming as i32,
                        ..Default::default()
                    }
                    .encode_to_vec(),
                );
                schemas
            })
            .clone()
    }

    fn method_json_schema_map(&self) -> Option<HashMap<String, Vec<u8>>> {
        static CACHED: std::sync::OnceLock<HashMap<String, Vec<u8>>> = std::sync::OnceLock::new();
        Some(
            CACHED
                .get_or_init(|| {
                    let mut schemas = HashMap::new();
                    schemas.insert(
                        METHOD_RUN.to_string(),
                        MethodJsonSchema {
                            args_schema: json_schema_string::<
                                protobuf::embedding_llm::EmbeddingArgs,
                            >("run_args_schema"),
                            result_schema: Some(json_schema_string::<
                                protobuf::embedding_llm::EmbeddingLlmResult,
                            >(
                                "run_result_schema"
                            )),
                            ..Default::default()
                        }
                        .encode_to_vec(),
                    );
                    schemas
                })
                .clone(),
        )
    }

    fn settings_schema(&self) -> String {
        json_schema_string::<protobuf::embedding_llm::EmbeddingLlmRunnerSettings>("settings_schema")
    }

    fn set_cancellation_token(&mut self, token: CancelToken) {
        self.token = Some(token);
    }

    async fn load(&mut self, settings: Vec<u8>) -> std::result::Result<(), String> {
        ensure_tracing_initialized().await;
        self.load_sync(settings).map_err(|e| e.to_string())
    }

    async fn run(
        &mut self,
        args: Vec<u8>,
        metadata: HashMap<String, String>,
        _using: Option<String>,
    ) -> (
        std::result::Result<Vec<u8>, String>,
        HashMap<String, String>,
    ) {
        let token = self.token.clone();
        let handle = self.rt_handle();
        let embedder = self.embedder.clone();
        // Clone metadata into the blocking task for the tracing helper, which
        // reads OTEL parent context from the metadata map. The outer future
        // keeps its own copy to echo back to the host.
        let meta_for_work = metadata.clone();

        // Short-circuit on a pre-cancelled token so callers see a consistent
        // "cancelled" error regardless of whether the embedder has been
        // initialised yet.
        if token.as_ref().is_some_and(|t| t.is_cancelled()) {
            return (Err(CANCELLED.to_string()), metadata);
        }
        let work = handle.spawn_blocking(move || -> std::result::Result<Vec<u8>, String> {
            Self::run_sync(embedder, args, &meta_for_work).map_err(|e| e.to_string())
        });

        // Two-stage wait: select on cancel vs completion. If cancel wins
        // we still await the blocking task. `LlamaCppEmbedder` has no
        // sink/cancel hook today, so embedding inference runs to completion
        // regardless. Joining here ensures the inner model lock is released
        // BEFORE the host moves on to the next job.
        tokio::pin!(work);
        let result = match token.as_ref() {
            Some(t) => {
                let cancel_branch = async {
                    t.cancelled().await;
                };
                tokio::pin!(cancel_branch);
                let joined: std::result::Result<
                    std::result::Result<Vec<u8>, String>,
                    tokio::task::JoinError,
                > = tokio::select! {
                    biased;
                    _ = &mut cancel_branch => (&mut work).await,
                    j = &mut work => j,
                };
                let cancel_observed = t.is_cancelled();
                match joined {
                    Ok(Ok(_)) if cancel_observed => Err(CANCELLED.to_string()),
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(_)) if cancel_observed => Err(CANCELLED.to_string()),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(format!("join error: {e}")),
                }
            }
            None => match work.await {
                Ok(inner) => inner,
                Err(e) => Err(format!("join error: {e}")),
            },
        };
        (result, metadata)
    }

    async fn run_stream(
        &mut self,
        _args: Vec<u8>,
        _metadata: HashMap<String, String>,
        _using: Option<String>,
        _output: HighLevelSink,
    ) -> std::result::Result<HashMap<String, String>, String> {
        // Embedding generation is a single-shot, non-streaming operation.
        Err("streaming is not supported by embedding-llm".to_string())
    }
}

/// One-time process bootstrap shared by every plugin instance. The macro's
/// init expression runs once per `load_multi_method_plugin_v2` invocation,
/// so only synchronous setup belongs here — async tracing initialisation
/// runs lazily inside `PluginV2::load`. Spinning a throwaway tokio runtime
/// here would panic if the host loads the plugin from a Tokio worker.
fn build_plugin_instance() -> EmbeddingLlmRunnerPlugin {
    dotenvy::dotenv().ok();
    EmbeddingLlmRunnerPlugin::new().expect("Failed to construct EmbeddingLlmRunnerPlugin")
}

/// Idempotent guard for tracing initialisation: the first `load()` call wins,
/// every subsequent call short-circuits. We use `tokio::sync::OnceCell` so the
/// guard cooperates with the surrounding async runtime.
async fn ensure_tracing_initialized() {
    static TRACING_INIT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    TRACING_INIT
        .get_or_init(|| async {
            // Surface init failures on stderr — tracing itself is the channel
            // we'd normally log through, so we can't rely on it to report its
            // own setup error.
            if let Err(e) = command_utils::util::tracing::tracing_init_from_env().await {
                eprintln!("embedding-llm: tracing init failed: {e:?}");
            }
        })
        .await;
}

register_plugin_v2!(EmbeddingLlmRunnerPlugin, build_plugin_instance());

#[cfg(test)]
mod tests {
    use super::*;
    use jobworkerp_plugin_abi::cancel::FfiCancellationToken;

    #[test]
    fn test_plugin_creation() {
        let plugin = EmbeddingLlmRunnerPlugin::new();
        assert!(plugin.is_ok());

        let plugin = plugin.unwrap();
        assert_eq!(plugin.name(), "EmbeddingLlmRunner");
        assert!(plugin.description().contains("embeddings"));
    }

    #[test]
    fn test_protobuf_schema_availability() {
        let plugin = EmbeddingLlmRunnerPlugin::new().unwrap();

        let settings_proto = plugin.runner_settings_proto();
        assert!(!settings_proto.is_empty());
        assert!(settings_proto.contains("EmbeddingLlmRunnerSettings"));
        assert!(
            !settings_proto
                .lines()
                .any(|l| l.trim().starts_with("import ")),
            "runner_settings_proto must not contain import statements"
        );

        let methods = plugin.method_proto_map();
        let run_bytes = methods
            .get(METHOD_RUN)
            .expect("`run` method must be registered");
        let run = MethodSchema::decode(run_bytes.as_slice()).expect("MethodSchema must decode");
        assert!(run.args_proto.contains("EmbeddingArgs"));
        assert!(
            !run.args_proto
                .lines()
                .any(|l| l.trim().starts_with("import ")),
            "method args_proto must not contain import statements"
        );
        assert!(run.result_proto.contains("EmbeddingLlmResult"));
    }

    #[test]
    fn test_multiple_plugin_creation() {
        // Test that multiple plugin instances can be created without BackendAlreadyInitialized error
        let plugin1 = EmbeddingLlmRunnerPlugin::new().unwrap();
        let plugin2 = EmbeddingLlmRunnerPlugin::new().unwrap();
        let plugin3 = EmbeddingLlmRunnerPlugin::new().unwrap();

        assert_eq!(plugin1.name(), "EmbeddingLlmRunner");
        assert_eq!(plugin2.name(), "EmbeddingLlmRunner");
        assert_eq!(plugin3.name(), "EmbeddingLlmRunner");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_with_precancelled_token_returns_err() {
        // Cancellation must be observed even when no embedder is loaded: the
        // future short-circuits on the cancelled token before the inner
        // spawn_blocking can report "Embedder not initialized".
        //
        // Note: flavor = "multi_thread" is required so the plugin's owned
        // Runtime can be dropped inside the test's async context — under the
        // default current-thread flavor, dropping a Runtime panics with
        // "Cannot drop a runtime in a context where blocking is not allowed".
        let mut plugin = EmbeddingLlmRunnerPlugin::new().unwrap();
        let (ffi, handle) = FfiCancellationToken::new_owned();
        handle.cancel();
        plugin.set_cancellation_token(CancelToken::from_ffi(ffi));

        let (result, _) = plugin.run(vec![], HashMap::new(), None).await;
        assert!(
            matches!(result, Err(ref e) if e == CANCELLED),
            "precancelled token must surface as Err(CANCELLED), got {result:?}"
        );

        // Move the plugin off-runtime so its inner Runtime drops on a blocking
        // thread rather than inside the test future.
        tokio::task::spawn_blocking(move || drop(plugin))
            .await
            .expect("plugin drop must not panic");
    }
}
