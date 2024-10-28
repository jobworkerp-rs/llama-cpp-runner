pub mod model;

use anyhow::{Context, Result, anyhow};
use jobworkerp_client::{plugins::MultiMethodPluginRunner, schema_to_json_string};
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings};
use jobworkerp_llama_protobuf::protobuf::llm::LlmChatArgs;
use model::{LlamaModelConfig, LlamaModelWrapper};
use prost::Message;
use std::{collections::HashMap, io::Cursor};

const METHOD_RUN: &str = "run";
const METHOD_CHAT: &str = "chat";

// suppress warn improper_ctypes_definitions
#[allow(improper_ctypes_definitions)]
#[unsafe(no_mangle)]
pub extern "C" fn load_multi_method_plugin() -> Box<dyn MultiMethodPluginRunner + Send + Sync> {
    std::panic::catch_unwind(|| {
        dotenvy::dotenv().ok();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                command_utils::util::tracing::tracing_init_from_env()
                    .await
                    .unwrap_or_default();
            });
        let p = LlamaCppPlugin::new();
        Box::new(p)
    })
    .unwrap_or_else(|e| {
        tracing::error!(
            "load_multi_method_plugin panic: {:?}, try to load by default config",
            e
        );
        Box::new(LlamaCppPlugin { llama_model: None })
    })
}

#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_multi_method_plugin(ptr: Box<dyn MultiMethodPluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct LlamaCppPlugin {
    pub llama_model: Option<LlamaModelWrapper>,
}

impl LlamaCppPlugin {
    pub fn new() -> Self {
        Self { llama_model: None }
    }
    fn load_config_from_env() -> Result<LlamaModelConfig> {
        envy::prefixed("LLAMA_")
            .from_env::<LlamaModelConfig>()
            .context("cannot read model config from env:")
    }
    pub fn load_model(&mut self, config: LlamaModelConfig) -> Result<()> {
        self.llama_model = Some(LlamaModelWrapper::new(config)?);
        Ok(())
    }
    pub fn load_model_from_env(&mut self) -> Result<()> {
        self.llama_model = Some(LlamaModelWrapper::new(Self::load_config_from_env()?)?);
        Ok(())
    }
    pub fn set_system_prompt(&mut self, system_prompt: &str) {
        if let Some(llama_model) = self.llama_model.as_mut() {
            llama_model.set_system_prompt(system_prompt);
        }
    }

    fn run_legacy(
        &mut self,
        arg: Vec<u8>,
        metadata: HashMap<String, String>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        let res = || -> Result<Vec<u8>> {
            if let Some(llama_model) = self.llama_model.as_mut() {
                let args = LlamaArg::decode(&mut Cursor::new(arg))
                    .map_err(|e| anyhow!("decode error: {e}"))?;
                tracing::debug!("LLMRunner run: {args:?}",);
                let text = llama_model
                    .run(args.clone().into())
                    .context("failed to decode")?;
                tracing::debug!("END OF LLMRunner: {text:?}",);
                let buf = LlamaArg {
                    prompt: text,
                    // Drop media inputs from the response so chained runners
                    // don't re-feed them on the next turn.
                    medias: vec![],
                    ..args
                };
                Ok(buf.encode_to_vec())
            } else {
                Err(anyhow!("llama_model is not loaded"))
            }
        };
        (res(), metadata)
    }

    fn run_chat(
        &mut self,
        arg: Vec<u8>,
        metadata: HashMap<String, String>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        let res = || -> Result<Vec<u8>> {
            if let Some(llama_model) = self.llama_model.as_mut() {
                let args = LlmChatArgs::decode(&mut Cursor::new(arg))
                    .map_err(|e| anyhow!("decode error: {e}"))?;
                tracing::debug!("LLMRunner chat: {args:?}",);
                let result = llama_model.run_chat(args)?;
                tracing::debug!("END OF LLMRunner chat: {result:?}",);
                Ok(result.encode_to_vec())
            } else {
                Err(anyhow!("llama_model is not loaded"))
            }
        };
        (res(), metadata)
    }
}

impl Default for LlamaCppPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiMethodPluginRunner for LlamaCppPlugin {
    fn name(&self) -> String {
        // Plugin loader matches this name against existing worker.operation
        // records, so renaming it would break deployed job definitions.
        String::from("LLMPromptRunner")
    }
    fn description(&self) -> String {
        String::from(
            "LLMPromptRunner is a plugin that lets you run LLM models with your own prompts and custom settings. Supports both legacy prompt mode and LLM chat completion API.",
        )
    }
    fn load(&mut self, settings: Vec<u8>) -> Result<()> {
        let settings = LlamaRunnerSettings::decode(&mut Cursor::new(settings))
            .map_err(|e| anyhow!("decode error: {e}"))?;
        tracing::debug!("LLMRunner load: {settings:?}",);
        self.load_model(settings.into())?;
        Ok(())
    }
    fn run(
        &mut self,
        arg: Vec<u8>,
        metadata: HashMap<String, String>,
        using: Option<&str>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        match using {
            Some(METHOD_CHAT) => self.run_chat(arg, metadata),
            _ => self.run_legacy(arg, metadata),
        }
    }
    fn cancel(&mut self) -> bool {
        tracing::warn!("LLMRunner cancel: not implemented!");
        false
    }
    fn is_canceled(&self) -> bool {
        tracing::warn!("LLMRunner is_canceled: not implemented!");
        false
    }
    fn runner_settings_proto(&self) -> String {
        static RESOLVED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        RESOLVED
            .get_or_init(|| {
                jobworkerp_llama_protobuf::proto_resolve::resolve_proto_imports(
                    include_str!("../../llama-protobuf/protobuf/llama_cpp/llama_cpp_runner.proto"),
                    &[jobworkerp_llama_protobuf::proto_resolve::MEDIA_INPUT_IMPORT],
                )
                .expect("LlamaCppPlugin: runner_settings_proto resolution failed")
            })
            .clone()
    }

    fn method_proto_map(
        &self,
    ) -> HashMap<String, jobworkerp_client::jobworkerp::data::MethodSchema> {
        static CACHED: std::sync::OnceLock<
            HashMap<String, jobworkerp_client::jobworkerp::data::MethodSchema>,
        > = std::sync::OnceLock::new();
        CACHED
            .get_or_init(|| {
                static RESOLVED_ARGS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
                let args_proto = RESOLVED_ARGS
                    .get_or_init(|| {
                        jobworkerp_llama_protobuf::proto_resolve::resolve_proto_imports(
                            include_str!(
                                "../../llama-protobuf/protobuf/llama_cpp/llama_cpp_arg.proto"
                            ),
                            &[jobworkerp_llama_protobuf::proto_resolve::MEDIA_INPUT_IMPORT],
                        )
                        .expect("LlamaCppPlugin: args_proto resolution failed")
                    })
                    .clone();

                let mut schemas = HashMap::new();
                schemas.insert(
                    METHOD_RUN.to_string(),
                    jobworkerp_client::jobworkerp::data::MethodSchema {
                        args_proto: args_proto.clone(),
                        result_proto: args_proto,
                        description: Some(
                            "Legacy LLM prompt execution with LlamaArg protobuf".to_string(),
                        ),
                        output_type:
                            jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming
                                as i32,
                        ..Default::default()
                    },
                );
                schemas.insert(
                    METHOD_CHAT.to_string(),
                    jobworkerp_client::jobworkerp::data::MethodSchema {
                        args_proto: include_str!(
                            "../../llama-protobuf/protobuf/jobworkerp/runner/llm/chat_args.proto"
                        )
                        .to_string(),
                        result_proto: include_str!(
                            "../../llama-protobuf/protobuf/jobworkerp/runner/llm/chat_result.proto"
                        )
                        .to_string(),
                        description: Some(
                            "LLM chat completion API compatible method with multi-turn conversation support"
                                .to_string(),
                        ),
                        output_type:
                            jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming
                                as i32,
                        ..Default::default()
                    },
                );
                schemas
            })
            .clone()
    }

    fn method_json_schema_map(
        &self,
    ) -> Option<HashMap<String, jobworkerp_client::jobworkerp::data::MethodJsonSchema>> {
        static CACHED: std::sync::OnceLock<
            HashMap<String, jobworkerp_client::jobworkerp::data::MethodJsonSchema>,
        > = std::sync::OnceLock::new();
        Some(
            CACHED
                .get_or_init(|| {
                    let mut schemas = HashMap::new();
                    schemas.insert(
                        METHOD_RUN.to_string(),
                        jobworkerp_client::jobworkerp::data::MethodJsonSchema {
                            args_schema: schema_to_json_string!(LlamaArg, "run_args_schema"),
                            result_schema: Some(schema_to_json_string!(
                                LlamaArg,
                                "run_result_schema"
                            )),
                            ..Default::default()
                        },
                    );
                    schemas.insert(
                        METHOD_CHAT.to_string(),
                        jobworkerp_client::jobworkerp::data::MethodJsonSchema {
                            args_schema: schema_to_json_string!(LlmChatArgs, "chat_args_schema"),
                            result_schema: Some(schema_to_json_string!(
                                jobworkerp_llama_protobuf::protobuf::llm::LlmChatResult,
                                "chat_result_schema"
                            )),
                            ..Default::default()
                        },
                    );
                    schemas
                })
                .clone(),
        )
    }

    fn settings_schema(&self) -> String {
        schema_to_json_string!(LlamaRunnerSettings, "settings_schema")
    }
}

#[cfg(test)]
mod test {
    use jobworkerp_llama_protobuf::protobuf::llama_cpp::LlamaArg;

    // create a test that loads the plugin model from environment variables and runs it internal model (llama_model)
    use super::*;
    #[tokio::test]
    async fn test_plugin_runner() {
        tracing_subscriber::fmt::init();
        let env = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf #Llama-3-ELYZA-JP-8B-q4_k_m.gguf # Phi-3-medium-128k-instruct.Q4_K.gguf # Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf #llama-2-7b-chat.Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF #elyza/Llama-3-ELYZA-JP-8B-GGUF # legraphista/Phi-3-medium-128k-instruct-IMat-GGUF # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF #TheBloke/Llama-2-7B-Chat-GGUF
#LLAMA_MODEL=tokyotech-llm-Llama-3.1-Swallow-70B-Instruct-v0.1-Q4_K_M.gguf # Phi-3-medium-128k-instruct.Q4_K.gguf # Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf #llama-2-7b-chat.Q4_K_M.gguf
#LLAMA_HF_REPO=mmnga/tokyotech-llm-Llama-3.1-Swallow-70B-Instruct-v0.1-gguf # legraphista/Phi-3-medium-128k-instruct-IMat-GGUF # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF #TheBloke/Llama-2-7B-Chat-GGUF
#LLAMA_MODEL=c4ai-command-r-plus-08-2024-Q4_K_M-00001-of-00002.gguf,c4ai-command-r-plus-08-2024-Q4_K_M-00002-of-00002.gguf # Phi-3-medium-128k-instruct.Q4_K.gguf # Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf #llama-2-7b-chat.Q4_K_M.gguf
#LLAMA_HF_REPO=grapevine-AI/c4ai-command-r-plus-08-2024-gguf # legraphista/Phi-3-medium-128k-instruct-IMat-GGUF # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF #TheBloke/Llama-2-7B-Chat-GGUF

LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=次の文章を日本語に翻訳してください。翻訳結果のみを出力してください
";
        dotenvy::from_read(env.as_bytes()).ok();

        let user_prompt = r#"
Daily Submission Limit Change
Hey ARC Prize contestants!

Greg from the ARC Prize team here. We are reducing the daily submission limit from 5 to 3 submissions per day.

Why we're making this change:

Discourage test probing: We want to ensure that the competition remains focused on developing robust, generalizable solutions rather than overfitting to the private evaluation data through repeated submissions.
Maintain competition integrity: This change helps mitigate the risk of model selection bias, where participants might inadvertently learn enough about the private test set through frequent submissions to gain an unfair advantage.
Encourage thoughtful iterations: By limiting submissions, we hope to promote deliberate and well-considered improvements to your models.
What this means for you:

You will now have 3 submission opportunities per day, not 5.
This change reduces the total potential submissions over the remaining competition period by approximately 40%.
We encourage you to use the public evaluation set for more frequent testing and iteration.
We believe this change strikes a reasonable balance between allowing for necessary iterations and maintaining the integrity of the challenge. It also aligns our competition more closely with best practices in machine learning competitions.

If you want to test more frequently we've made a secondary leaderboard, ARC-AGI-Pub, just for this check out our launch post for more information.

We appreciate your understanding and continued participation in ARC Prize. If you have any questions, you can reach us at: team@arcprize.org

Good luck in the competition and in advancing AI research!
        "#;
        let prompt = user_prompt.to_string();

        let mut plugin = LlamaCppPlugin::new();
        plugin
            .load_model_from_env()
            .expect("failed to load model from env");
        let request = LlamaArg {
            prompt,
            sample_len: 2048,
            temperature: Some(0.3),
            top_p: Some(0.9),
            repeat_penalty: Some(0.9),
            repeat_last_n: Some(8),
            seed: Some(30),
            need_print: true,
            medias: vec![],
        };
        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let res = plugin
            .run(buf, HashMap::new(), None)
            .0
            .expect("failed to run plugin");
        let res = LlamaArg::decode(&mut Cursor::new(res.clone()))
            .map_err(|e| anyhow!("decode error: {e}"))
            .unwrap();
        println!("response: {:?}", res.prompt);
        assert!(res.prompt.len() > 10 && res.prompt.len() < 4096);
    }

    #[test]
    fn test_protobuf_schema_resolution() {
        let plugin = LlamaCppPlugin::new();

        let settings = plugin.runner_settings_proto();
        assert!(
            !settings.lines().any(|l| l.trim().starts_with("import ")),
            "runner_settings_proto must not contain import statements"
        );
        assert!(settings.contains("message LlamaRunnerSettings"));
        assert!(settings.contains("message MtmdSettings"));

        let schemas = plugin.method_proto_map();

        let run_schema = schemas.get("run").expect("run method schema");
        assert!(
            !run_schema
                .args_proto
                .lines()
                .any(|l| l.trim().starts_with("import ")),
            "run args_proto must not contain import statements"
        );
        assert!(run_schema.args_proto.contains("message LlamaArg"));
        assert!(run_schema.args_proto.contains("message MediaInput"));
        assert!(run_schema.result_proto.contains("message LlamaArg"));

        let chat_schema = schemas.get("chat").expect("chat method schema");
        assert!(
            !chat_schema
                .args_proto
                .lines()
                .any(|l| l.trim().starts_with("import ")),
            "chat args_proto must not contain import statements"
        );
        assert!(chat_schema.args_proto.contains("message LLMChatArgs"));
        assert!(chat_schema.result_proto.contains("message LLMChatResult"));
    }

    #[test]
    fn test_method_json_schema_map() {
        let plugin = LlamaCppPlugin::new();
        let schemas = plugin.method_json_schema_map().expect("json schemas");

        assert!(schemas.contains_key("run"), "run schema must exist");
        assert!(schemas.contains_key("chat"), "chat schema must exist");

        let run_schema = &schemas["run"];
        serde_json::from_str::<serde_json::Value>(&run_schema.args_schema)
            .expect("run args_schema must be valid JSON");
        serde_json::from_str::<serde_json::Value>(
            run_schema
                .result_schema
                .as_ref()
                .expect("run result_schema"),
        )
        .expect("run result_schema must be valid JSON");

        let chat_schema = &schemas["chat"];
        serde_json::from_str::<serde_json::Value>(&chat_schema.args_schema)
            .expect("chat args_schema must be valid JSON");
        serde_json::from_str::<serde_json::Value>(
            chat_schema
                .result_schema
                .as_ref()
                .expect("chat result_schema"),
        )
        .expect("chat result_schema must be valid JSON");
    }

    #[test]
    fn test_extract_reasoning() {
        let (text, reasoning) = LlamaModelWrapper::extract_reasoning("Hello world");
        assert_eq!(text, "Hello world");
        assert!(reasoning.is_none());

        let (text, reasoning) = LlamaModelWrapper::extract_reasoning(
            "<think>Let me think about this</think>The answer is 42",
        );
        assert_eq!(text, "The answer is 42");
        assert_eq!(reasoning.unwrap(), "Let me think about this");

        let (text, reasoning) = LlamaModelWrapper::extract_reasoning("<think>Incomplete reasoning");
        assert_eq!(text, "<think>Incomplete reasoning");
        assert!(reasoning.is_none());

        // Reversed tag order must not panic
        let (text, reasoning) =
            LlamaModelWrapper::extract_reasoning("</think>some text<think>reasoning");
        assert_eq!(text, "</think>some text<think>reasoning");
        assert!(reasoning.is_none());
    }

    #[ignore = "depends on model"]
    #[tokio::test]
    async fn test_plugin_chat_runner() {
        use jobworkerp_llama_protobuf::protobuf::llm::{
            LlmChatResult, llm_chat_args, llm_chat_result,
        };

        let env = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=You are a helpful assistant.
";
        dotenvy::from_read(env.as_bytes()).ok();

        let mut plugin = LlamaCppPlugin::new();
        plugin
            .load_model_from_env()
            .expect("failed to load model from env");

        let request = LlmChatArgs {
            model: None,
            options: Some(llm_chat_args::LlmOptions {
                max_tokens: Some(256),
                temperature: Some(0.3),
                ..Default::default()
            }),
            function_options: None,
            messages: vec![
                llm_chat_args::ChatMessage {
                    role: llm_chat_args::ChatRole::System as i32,
                    content: Some(llm_chat_args::MessageContent {
                        content: Some(llm_chat_args::message_content::Content::Text(
                            "You are a helpful assistant. Answer briefly.".to_string(),
                        )),
                    }),
                },
                llm_chat_args::ChatMessage {
                    role: llm_chat_args::ChatRole::User as i32,
                    content: Some(llm_chat_args::MessageContent {
                        content: Some(llm_chat_args::message_content::Content::Text(
                            "What is 2+2?".to_string(),
                        )),
                    }),
                },
                llm_chat_args::ChatMessage {
                    role: llm_chat_args::ChatRole::Assistant as i32,
                    content: Some(llm_chat_args::MessageContent {
                        content: Some(llm_chat_args::message_content::Content::Text(
                            "4".to_string(),
                        )),
                    }),
                },
                llm_chat_args::ChatMessage {
                    role: llm_chat_args::ChatRole::User as i32,
                    content: Some(llm_chat_args::MessageContent {
                        content: Some(llm_chat_args::message_content::Content::Text(
                            "And 3+3?".to_string(),
                        )),
                    }),
                },
            ],
            json_schema: None,
        };

        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let res = plugin
            .run(buf, HashMap::new(), Some(METHOD_CHAT))
            .0
            .expect("failed to run chat plugin");
        let res = LlmChatResult::decode(&mut Cursor::new(res))
            .map_err(|e| anyhow!("decode error: {e}"))
            .unwrap();
        println!("chat response: {:?}", res);
        assert!(res.done);
        let content = res.content.expect("should have content");
        match content.content {
            Some(llm_chat_result::message_content::Content::Text(text)) => {
                println!("chat text: {text}");
                assert!(!text.is_empty());
            }
            other => panic!("expected text content, got: {other:?}"),
        }
    }

    #[ignore = "depends on model"]
    #[tokio::test]
    async fn test_plugin_chat_json_schema() {
        use jobworkerp_llama_protobuf::protobuf::llm::{
            LlmChatResult, llm_chat_args, llm_chat_result,
        };

        let env = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=You are a helpful assistant.
";
        dotenvy::from_read(env.as_bytes()).ok();

        let mut plugin = LlamaCppPlugin::new();
        plugin
            .load_model_from_env()
            .expect("failed to load model from env");

        let schema = r#"{
            "type": "object",
            "properties": {
                "answer": { "type": "integer" },
                "explanation": { "type": "string" }
            },
            "required": ["answer", "explanation"]
        }"#;

        let request = LlmChatArgs {
            model: None,
            options: Some(llm_chat_args::LlmOptions {
                max_tokens: Some(256),
                temperature: Some(0.3),
                ..Default::default()
            }),
            function_options: None,
            messages: vec![llm_chat_args::ChatMessage {
                role: llm_chat_args::ChatRole::User as i32,
                content: Some(llm_chat_args::MessageContent {
                    content: Some(llm_chat_args::message_content::Content::Text(
                        "What is 2+2? Respond in JSON.".to_string(),
                    )),
                }),
            }],
            json_schema: Some(schema.to_string()),
        };

        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let res = plugin
            .run(buf, HashMap::new(), Some(METHOD_CHAT))
            .0
            .expect("failed to run chat with json_schema");
        let res = LlmChatResult::decode(&mut Cursor::new(res))
            .map_err(|e| anyhow!("decode error: {e}"))
            .unwrap();
        println!("json_schema response: {:?}", res);
        assert!(res.done);
        let content = res.content.expect("should have content");
        match content.content {
            Some(llm_chat_result::message_content::Content::Text(text)) => {
                println!("json_schema text: {text}");
                assert!(!text.is_empty());
                let parsed: serde_json::Value =
                    serde_json::from_str(&text).expect("output must be valid JSON");
                assert!(parsed.get("answer").is_some(), "must have 'answer' field");
                assert!(
                    parsed.get("explanation").is_some(),
                    "must have 'explanation' field"
                );
            }
            other => panic!("expected text content, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_rejects_function_calling() {
        use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args;

        let env = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
";
        dotenvy::from_read(env.as_bytes()).ok();

        let mut plugin = LlamaCppPlugin::new();
        plugin
            .load_model_from_env()
            .expect("failed to load model from env");

        let request = LlmChatArgs {
            model: None,
            options: None,
            function_options: Some(llm_chat_args::FunctionOptions {
                use_function_calling: true,
                ..Default::default()
            }),
            messages: vec![llm_chat_args::ChatMessage {
                role: llm_chat_args::ChatRole::User as i32,
                content: Some(llm_chat_args::MessageContent {
                    content: Some(llm_chat_args::message_content::Content::Text(
                        "hello".to_string(),
                    )),
                }),
            }],
            json_schema: None,
        };

        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let (res, _) = plugin.run(buf, HashMap::new(), Some(METHOD_CHAT));
        let err = res.expect_err("function_calling should be rejected");
        assert!(
            err.to_string().contains("function calling"),
            "error should mention function calling: {err}"
        );
    }

    #[tokio::test]
    async fn test_chat_rejects_unknown_role() {
        use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args;

        let env = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
";
        dotenvy::from_read(env.as_bytes()).ok();

        let mut plugin = LlamaCppPlugin::new();
        plugin
            .load_model_from_env()
            .expect("failed to load model from env");

        let request = LlmChatArgs {
            model: None,
            options: None,
            function_options: None,
            messages: vec![llm_chat_args::ChatMessage {
                role: 0, // UNSPECIFIED
                content: Some(llm_chat_args::MessageContent {
                    content: Some(llm_chat_args::message_content::Content::Text(
                        "hello".to_string(),
                    )),
                }),
            }],
            json_schema: None,
        };

        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let (res, _) = plugin.run(buf, HashMap::new(), Some(METHOD_CHAT));
        let err = res.expect_err("UNSPECIFIED role should be rejected");
        assert!(
            err.to_string().contains("unsupported or unknown chat role"),
            "error should mention role: {err}"
        );
    }
}
