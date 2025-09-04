pub mod embedding;
pub mod error;
pub mod llamacpp_bridge;
pub mod sliding_window;
pub mod tokenization;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use jobworkerp_client::plugins::PluginRunner;
use prost::Message;
use std::collections::HashMap;

use crate::embedding::LlamaCppEmbedder;

/// Generated protobuf modules
pub mod protobuf {
    pub mod embedding_llm {
        include!(concat!(env!("OUT_DIR"), "/embedding_llm.rs"));
    }
}

/// Main plugin structure for embedding-llm
pub struct EmbeddingLlmRunnerPlugin {
    embedder: Option<LlamaCppEmbedder>,
}

impl EmbeddingLlmRunnerPlugin {
    pub const RUNNER_NAME: &'static str = "EmbeddingLlmRunner";

    pub fn new() -> Result<Self> {
        Ok(Self { embedder: None })
    }
}

impl Default for EmbeddingLlmRunnerPlugin {
    fn default() -> Self {
        Self::new().expect("Failed to create plugin")
    }
}

#[async_trait]
impl PluginRunner for EmbeddingLlmRunnerPlugin {
    fn name(&self) -> String {
        String::from(Self::RUNNER_NAME)
    }

    fn description(&self) -> String {
        String::from("Generate embeddings using LLM via llama.cpp with LLM hidden states")
    }

    fn load(&mut self, settings: Vec<u8>) -> Result<()> {
        let settings = protobuf::embedding_llm::EmbeddingLlmRunnerSettings::decode(&settings[..])?;

        // llama.cpp版の検証
        if settings.model_type != (protobuf::embedding_llm::ModelType::Gguf as i32) {
            return Err(anyhow!("llama-cpp version only supports GGUF models"));
        }

        if settings.model_files.is_empty() {
            return Err(anyhow!("GGUF model files must be specified"));
        }

        // llama.cpp embedderの初期化
        let embedder = LlamaCppEmbedder::new_from_settings(&settings)?;

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
        let mut result_metadata = metadata;

        let result: Result<Vec<u8>> = (|| -> Result<Vec<u8>> {
            let args = protobuf::embedding_llm::EmbeddingArgs::decode(&arg[..])?;

            let embedder = self
                .embedder
                .as_ref()
                .ok_or(anyhow!("Embedder not initialized"))?;

            // llama.cppベースのembedding生成
            let embeddings = embedder.generate_embeddings_with_instruction(
                &args.text,
                args.instruction.as_deref(),
                args.normalize_embeddings,
                None, // merge_strategy: None (全embeddings個別に返す)
            )?;

            let model_info = embedder.model_info();

            // メタデータに統計情報を追加
            result_metadata.insert("embedding_count".to_string(), embeddings.len().to_string());
            result_metadata.insert(
                "embedding_dimension".to_string(),
                model_info.embedding_dimension.to_string(),
            );
            result_metadata.insert("model_path".to_string(), model_info.model_path.clone());

            let result = protobuf::embedding_llm::EmbeddingLlmResult {
                embeddings: embeddings
                    .into_iter()
                    .map(
                        |values| protobuf::embedding_llm::embedding_llm_result::Embedding {
                            values,
                        },
                    )
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
        assert!(plugin.description().contains("embedding"));
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
}
