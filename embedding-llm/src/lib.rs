#![allow(improper_ctypes_definitions)]

pub mod chunking_adapter;
pub mod embedding;
pub mod error;
pub mod llamacpp_bridge;
pub mod text_processing;
pub mod token_position;
pub mod tokenization;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use command_utils::trace::Tracing;
use jobworkerp_client::{
    plugins::PluginRunner, schema_to_json_string, schema_to_json_string_option,
};
use prost::Message;
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

/// Main plugin structure for embedding-llm
pub struct EmbeddingLlmRunnerPlugin {
    backend: Option<Arc<Mutex<LlamaBackend>>>,
    embedder: Option<LlamaCppEmbedder>,
}

impl EmbeddingLlmRunnerPlugin {
    pub const RUNNER_NAME: &'static str = "EmbeddingLlmRunner";

    pub fn new() -> Result<Self> {
        tracing::info!("Creating EmbeddingLlmRunner plugin (backend will be initialized on load)");
        Ok(Self {
            backend: None,
            embedder: None,
        })
    }

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

#[async_trait]
impl PluginRunner for EmbeddingLlmRunnerPlugin {
    fn name(&self) -> String {
        String::from(Self::RUNNER_NAME)
    }

    fn description(&self) -> String {
        String::from("Generate text embeddings with positional information and optional instruction prefixes")
    }

    fn load(&mut self, settings: Vec<u8>) -> Result<()> {
        command_utils::util::tracing::tracing_init_test(tracing::Level::DEBUG);
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
        let embedder =
            LlamaCppEmbedder::new_from_settings_with_backend(&settings, backend)?;

        self.embedder = Some(embedder);

        tracing::info!(
            "{} loaded: model_id={}, files={:?}",
            Self::RUNNER_NAME,
            settings.model_id,
            settings.model_files
        );

        Ok(())
    }

    fn run(
        &mut self,
        arg: Vec<u8>,
        metadata: HashMap<String, String>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        // OpenTelemetryスパンの作成（metadataから親コンテキストを抽出）
        let mut span = EmbeddingLlmRunnerPlugin::otel_span_from_metadata(
            &metadata,
            "embedding-llm",
            "embedding.run",
        );

        // metadataをそのまま通すため、変更しない
        let result_metadata = metadata;

        let result: Result<Vec<u8>> = (|| -> Result<Vec<u8>> {
            let args = protobuf::embedding_llm::EmbeddingArgs::decode(&arg[..])?;

            let embedder = self
                .embedder
                .as_ref()
                .ok_or(anyhow!("Embedder not initialized"))?;

            // llama.cppベースのembedding生成（位置情報付き）
            let embeddings_with_positions = embedder.generate_embeddings_with_positions(
                &args.text,
                args.instruction.as_deref(),
                args.normalize_embeddings,
                None, // merge_strategy: None (全embeddings個別に返す)
            )?;

            let model_info = embedder.model_info();

            let result = protobuf::embedding_llm::EmbeddingLlmResult {
                embeddings: embeddings_with_positions
                    .into_iter()
                    .map(|embedding_with_pos| {
                        // Extract content from original text using character positions
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
                model_info: Some(protobuf::embedding_llm::ModelInfo {
                    model_name: format!("llama.cpp-{}", model_info.model_path),
                    embedding_dimension: model_info.embedding_dimension as u32,
                    dtype_used: model_info.dtype.clone(),
                }),
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

        (result, result_metadata)
    }

    fn cancel(&self) -> bool {
        false
    }
    fn is_canceled(&self) -> bool {
        false
    }

    fn runner_settings_proto(&self) -> String {
        include_str!("../protobuf/llm_runner_settings.proto").to_string()
    }

    fn job_args_proto(&self) -> String {
        include_str!("../protobuf/embedding_args.proto").to_string()
    }

    fn result_output_proto(&self) -> Option<String> {
        Some(include_str!("../protobuf/llm_result.proto").to_string())
    }
    fn settings_schema(&self) -> String {
        schema_to_json_string!(
            protobuf::embedding_llm::EmbeddingLlmRunnerSettings,
            "settings_schema"
        )
    }
    fn arguments_schema(&self) -> String {
        schema_to_json_string!(protobuf::embedding_llm::EmbeddingArgs, "arguments_schema")
    }
    fn output_json_schema(&self) -> Option<String> {
        schema_to_json_string_option!(protobuf::embedding_llm::EmbeddingLlmResult, "output_schema")
    }
}

// Plugin entry points
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
    dotenvy::dotenv().ok();
    let plugin = EmbeddingLlmRunnerPlugin::new().expect("Failed to load plugin");
    Box::new(plugin)
}

#[no_mangle]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let args_proto = plugin.job_args_proto();
        assert!(!args_proto.is_empty());
        assert!(args_proto.contains("EmbeddingArgs"));

        let result_proto = plugin.result_output_proto();
        assert!(result_proto.is_some());
        let result_proto = result_proto.unwrap();
        assert!(result_proto.contains("EmbeddingLlmResult"));
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

        println!("✓ Successfully created multiple plugin instances sharing the same backend");
    }

    #[test]
    fn test_load_plugin_multiple_times() {
        // Test that load_plugin() can be called multiple times without errors
        let plugin1 = load_plugin();
        let plugin2 = load_plugin();
        let plugin3 = load_plugin();

        assert_eq!(plugin1.name(), "EmbeddingLlmRunner");
        assert_eq!(plugin2.name(), "EmbeddingLlmRunner");
        assert_eq!(plugin3.name(), "EmbeddingLlmRunner");

        // Clean up
        free_plugin(plugin1);
        free_plugin(plugin2);
        free_plugin(plugin3);

        println!("✓ Successfully loaded and freed multiple plugin instances via C API");
    }
}
