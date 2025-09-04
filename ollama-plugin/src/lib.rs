use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use jobworkerp_client::plugins::PluginRunner;
use jobworkerp_client::{schema_to_json_string, schema_to_json_string_option};
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings};
use jobworkerp_llama_protobuf::protobuf::ollama::ollama_args::{self, MessageRole};
use jobworkerp_llama_protobuf::protobuf::ollama::{OllamaArgs, OllamaRunnerSettings};
use jobworkerp_util::runner::OLLAMA_PROMPT;
use ollama_rs::generation::chat;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::{
    generation::{
        chat::ChatMessage,
        completion::{request::GenerationRequest, GenerationResponse},
    },
    models::ModelOptions,
    Ollama,
};
use prost::Message;
use std::collections::HashMap;
use std::io::Cursor;
use std::vec;

// suppress warn improper_ctypes_definitions
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
    std::panic::catch_unwind(|| {
        dotenvy::dotenv().ok();
        // tokio::runtime::Runtime::new()
        //     .unwrap()
        //     .block_on(async move {
        //         command_utils::util::tracing::tracing_init_from_env()
        //             .await
        //             .unwrap_or_default();
        //     });
        let p = OllamaPlugin::new();
        Box::new(p)
    })
    .inspect_err(|e| tracing::error!("load_plugin panic: {:?}, try to load by default config", e))
    .unwrap()
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct OllamaPlugin {
    pub ollama: Option<Ollama>,
    pub model: String,
    pub system_prompt: Option<String>,
}
// static DATA: OnceCell<Bytes> = OnceCell::new();

static INIT: std::sync::Once = std::sync::Once::new();
impl OllamaPlugin {
    const URL_BASE: &'static str = "http://localhost:11434";
    pub fn new() -> Self {
        use tracing::Level;
        INIT.call_once(|| {
            let _ = std::panic::catch_unwind(|| {
                tracing_subscriber::fmt()
                    .with_max_level(Level::WARN) // TODO configurable
                    .compact()
                    .init();
            });
        });
        Self {
            ollama: None,
            model: "".to_string(),
            system_prompt: None,
        }
    }

    pub fn load_model_from_env(&mut self) -> Result<()> {
        let url_base = std::env::var("LLAMA_URL_BASE").unwrap_or(Self::URL_BASE.to_string());
        let model = std::env::var("LLAMA_MODEL").context("LLAMA_MODEL is not set")?;
        let system_prompt = std::env::var("LLAMA_SYSTEM_PROMPT").ok();
        let pull_model = std::env::var("LLAMA_PULL_MODEL")
            .ok()
            .map(|b| b.as_str() == "true");

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            self.load_model(OllamaRunnerSettings {
                base_url: Some(url_base),
                model,
                system_prompt,
                pull_model,
            })
            .await
        })?;
        Ok(())
    }
    pub async fn load_model(&mut self, settings: OllamaRunnerSettings) -> Result<()> {
        let llama = Ollama::try_new(settings.base_url.unwrap_or(Self::URL_BASE.to_string()))?;
        if settings.pull_model.unwrap_or(true) {
            let pull = llama
                .pull_model(settings.model.clone(), false)
                .await
                .map_err(|e| anyhow!("{}", e))?;
            tracing::debug!("model loaded: result = {:?}", pull);
        };
        self.ollama = Some(llama);
        self.model = settings.model;
        self.system_prompt = settings.system_prompt;
        Ok(())
    }
    pub fn to_ollama_chat(mes: &ollama_args::ChatMessage) -> chat::ChatMessage {
        chat::ChatMessage {
            content: mes.content.clone(),
            role: match mes.role() {
                MessageRole::User => chat::MessageRole::User,
                MessageRole::Assistant => chat::MessageRole::Assistant,
                MessageRole::System => chat::MessageRole::System,
                MessageRole::Tool => chat::MessageRole::Tool,
            },
            tool_calls: vec![],
            images: None,
            // TODO
            //     mes
            //     .images
            //     .map(|i| {
            //         i.into_iter()
            //             .map(|i| i.to_base64().as_bytes().to_vec())
            //             .collect()
            //     })
            //     .unwrap_or_default(),
            thinking: None,
        }
    }
    pub fn to_jobworker_chat(mes: &chat::ChatMessage, gen_id: bool) -> ollama_args::ChatMessage {
        ollama_args::ChatMessage {
            chat_id: if gen_id {
                Some(ollama_args::ChatId {
                    id: command_utils::util::id_generator::new_generator_by_ip()
                        .generate()
                        .inspect_err(|e| tracing::error!("id_generator error: {:?}", e))
                        .unwrap_or(rand::random::<i64>()),
                })
            } else {
                None
            },
            content: mes.content.clone(),
            role: match mes.role {
                chat::MessageRole::User => MessageRole::User as i32,
                chat::MessageRole::Assistant => MessageRole::Assistant as i32,
                chat::MessageRole::System => MessageRole::System as i32,
                chat::MessageRole::Tool => MessageRole::Tool as i32,
            },
            tool_calls: vec![],
            images: mes
                .images
                .as_ref()
                .map(|imgs| {
                    imgs.iter()
                        .map(|i| i.to_base64().as_bytes().to_vec())
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
    pub fn create_options(args: &OllamaArgs) -> ModelOptions {
        let mut options = ModelOptions::default();

        if let Some(opts) = args.options {
            if let Some(sample_len) = opts.sample_len {
                options = options.num_predict(sample_len);
            } else {
                // fill context
                options = options.num_predict(-2);
            }
            if let Some(temperature) = opts.temperature {
                options = options.temperature(temperature);
            }
            if let Some(top_p) = opts.top_p {
                options = options.top_p(top_p);
            }
            if let Some(repeat_penalty) = opts.repeat_penalty {
                options = options.repeat_penalty(repeat_penalty);
            }
            if let Some(repeat_last_n) = opts.repeat_last_n {
                options = options.repeat_last_n(repeat_last_n);
            }
            if let Some(seed) = opts.seed {
                options = options.seed(seed);
            }
        }
        options
    }
    pub fn request_chat_with_history(&mut self, args: OllamaArgs) -> Result<OllamaArgs> {
        if let Some(ollama) = self.ollama.as_mut() {
            let options = Self::create_options(&args);
            let mut histories: Vec<chat::ChatMessage> = args
                .histories
                .iter()
                .map(|c| Self::to_ollama_chat(c))
                .collect();
            if !histories
                .first()
                .is_some_and(|m| m.role == chat::MessageRole::System)
            {
                let system = if let Some(system_prompt) = args.override_system_prompt.clone() {
                    system_prompt
                } else if let Some(system_prompt) = self.system_prompt.clone() {
                    system_prompt
                } else {
                    "".to_string()
                };
                histories.insert(0, ChatMessage::system(system));
            }
            let latest = ChatMessage::user(args.prompt);
            let mut request = ChatMessageRequest::new(self.model.clone(), vec![latest]);
            request = request.options(options);

            let res = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(ollama.send_chat_messages_with_history(&mut histories, request))
                .map_err(|e| anyhow!("Generation error(chat_with_history): {}", e))?;
            tracing::debug!(
                "END OF chat {}: created_at: {}",
                OLLAMA_PROMPT,
                res.created_at
            );
            let histories = histories
                .into_iter()
                .zip_longest(args.histories.into_iter())
                .flat_map(|pair| match pair {
                    itertools::EitherOrBoth::Both(m, h) => {
                        assert_eq!(m.content, h.content);
                        Some(ollama_args::ChatMessage {
                            chat_id: h.chat_id,
                            content: m.content,
                            role: match m.role {
                                chat::MessageRole::User => MessageRole::User as i32,
                                chat::MessageRole::Assistant => MessageRole::Assistant as i32,
                                chat::MessageRole::System => MessageRole::System as i32,
                                chat::MessageRole::Tool => MessageRole::Tool as i32,
                            },
                            tool_calls: h.tool_calls,
                            images: h.images,
                        })
                    }
                    itertools::EitherOrBoth::Left(m) => {
                        // assert!(
                        //     m.role == chat::MessageRole::Assistant
                        //         || m.role == chat::MessageRole::System
                        // );
                        Some(ollama_args::ChatMessage {
                            chat_id: Some(ollama_args::ChatId {
                                id: command_utils::util::id_generator::new_generator_by_ip()
                                    .generate()
                                    .inspect_err(|e| tracing::error!("id_generator error: {:?}", e))
                                    .unwrap_or(rand::random::<i64>()),
                            }),
                            content: m.content,
                            role: match m.role {
                                chat::MessageRole::User => MessageRole::User as i32,
                                chat::MessageRole::Assistant => MessageRole::Assistant as i32,
                                chat::MessageRole::System => MessageRole::System as i32,
                                chat::MessageRole::Tool => MessageRole::Tool as i32,
                            },
                            tool_calls: vec![], // TODO
                            images: m
                                .images
                                .map(|i| {
                                    i.into_iter()
                                        .map(|i| i.to_base64().as_bytes().to_vec())
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                    }
                    itertools::EitherOrBoth::Right(h) => {
                        tracing::error!("histories is shorter than response: {:?}", h);
                        None
                    }
                })
                .collect();
            if args.divide_think_tag {
                let (prompt, think) = Self::divide_think_tag(res.message.content);
                Ok(OllamaArgs {
                    prompt,
                    think,
                    histories,
                    ..args
                })
            } else {
                Ok(OllamaArgs {
                    prompt: res.message.content,
                    histories,
                    ..args
                })
            }
        } else {
            Err(anyhow!("llama_model is not loaded"))
        }
    }
    pub fn request_with_history_refreshed(&mut self, mut args: OllamaArgs) -> Result<OllamaArgs> {
        if let Some(ollama) = self.ollama.as_mut() {
            let options = Self::create_options(&args);
            // history to messages (refreshed)
            let mut messages: Vec<chat::ChatMessage> = args
                .histories
                .iter()
                .map(|c| Self::to_ollama_chat(c))
                .collect();
            if messages
                .first()
                .is_some_and(|m| m.role == chat::MessageRole::System)
            {
                // replace system prompt if override_system_prompt is specified
                if let Some(system_prompt) = args.override_system_prompt.clone() {
                    let r = messages.remove(0);
                    tracing::debug!(
                        "replace system_prompt: {}\n >>>>>>>\n {}",
                        r.content,
                        system_prompt
                    );
                    let new_prompt = ChatMessage::system(system_prompt);
                    args.histories
                        .splice(0..1, vec![Self::to_jobworker_chat(&new_prompt, true)]);
                    messages.insert(0, new_prompt);
                }
            } else {
                let system = if let Some(system_prompt) = args.override_system_prompt.clone() {
                    system_prompt
                } else if let Some(system_prompt) = self.system_prompt.clone() {
                    system_prompt
                } else {
                    "".to_string()
                };
                messages.insert(0, ChatMessage::system(system));
            }
            let latest = ChatMessage::user(args.prompt);
            messages.append(&mut vec![latest.clone()]);

            let mut request = ChatMessageRequest::new(self.model.clone(), messages);
            request = request.options(options);
            let res = {
                // request contains all chat messages
                tokio::runtime::Runtime::new()
                    .unwrap()
                    .block_on(ollama.send_chat_messages(request))
            }
            .map_err(|e| anyhow!("Generation error(chat_with_history_refleshed): {}", e))?;
            tracing::debug!(
                "END OF chat {}: created_at: {}",
                OLLAMA_PROMPT,
                res.created_at
            );
            // next histories are updated (append latest and response with generated id)
            let mut histories = args.histories;
            histories.append(&mut vec![
                Self::to_jobworker_chat(&latest, true),
                Self::to_jobworker_chat(&res.message, true),
            ]);
            if args.divide_think_tag {
                let (prompt, think) = Self::divide_think_tag(res.message.content);
                Ok(OllamaArgs {
                    prompt,
                    think,
                    histories,
                    ..args
                })
            } else {
                Ok(OllamaArgs {
                    prompt: res.message.content,
                    histories,
                    ..args
                })
            }
        } else {
            Err(anyhow!("llama_model is not loaded"))
        }
    }
    pub fn request_generation(&mut self, args: OllamaArgs) -> Result<OllamaArgs> {
        if let Some(ollama) = self.ollama.as_mut() {
            let options = Self::create_options(&args);
            let mut request = GenerationRequest::new(self.model.clone(), args.prompt);
            if let Some(system_prompt) = args.override_system_prompt.clone() {
                request = request.system(system_prompt);
            } else if let Some(system_prompt) = self.system_prompt.clone() {
                request = request.system(system_prompt);
            }
            request = request.options(options);
            //XXX only support json format (TODO schema)
            if let Some(_schema) = &args.schema_json {
                // cannot use StructuredJson (schema is private field)
                // let schema_root = serde_json::from_str(schema).map_err(|e| anyhow!("{}", e))?;
                // let json_schema: RootSchema = schema_root;
                // let serialized: ollama_rs::generation::parameters::JsonStructure =
                //     json_schema.into();
                // ollama_rs::generation::parameters::FormatType::StructuredJson(
                //     ollama_rs::generation::parameters::JsonStructure {
                //         schema: json_schema,
                //     },
                // ),
                request = request.format(ollama_rs::generation::parameters::FormatType::Json);
            }
            let res: GenerationResponse = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(ollama.generate(request))
                .map_err(|e| anyhow!("Generation error(generation): {}", e))?;
            tracing::debug!(
                "END OF generation {}: duration: {}",
                OLLAMA_PROMPT,
                res.total_duration.unwrap_or_default()
            );
            if args.divide_think_tag {
                let (prompt, think) = Self::divide_think_tag(res.response);
                Ok(OllamaArgs {
                    prompt,
                    think,
                    ..args
                })
            } else {
                Ok(OllamaArgs {
                    prompt: res.response,
                    ..args
                })
            }
        } else {
            Err(anyhow!("llama_model is not loaded"))
        }
    }
    // divide <think></think> tag from prompt
    fn divide_think_tag(prompt: String) -> (String, Option<String>) {
        if let Some(think_start) = prompt.find("<think>") {
            if let Some(think_end) = prompt.find("</think>") {
                let think = Some(prompt[think_start + 7..think_end].trim().to_string());
                let new_prompt = prompt[..think_start].to_string() + &prompt[think_end + 8..];
                return (new_prompt.trim().to_string(), think);
            }
        }
        (prompt.trim().to_string(), None)
    }
}

impl PluginRunner for OllamaPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.settings
        OLLAMA_PROMPT.to_string()
    }
    fn description(&self) -> String {
        "OllamaPromptRunner connects to Ollama server and lets you generate text with your own prompts and settings"
            .to_string()
    }
    fn load(&mut self, settings: Vec<u8>) -> Result<()> {
        let settings = OllamaRunnerSettings::decode(&mut Cursor::new(settings))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async { self.load_model(settings).await })?;
        tracing::info!("{} loaded", OLLAMA_PROMPT);
        Ok(())
    }
    fn run(
        &mut self,
        arg: Vec<u8>,
        metadata: HashMap<String, String>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        let res = || -> Result<Vec<u8>> {
            let args = OllamaArgs::decode(&mut Cursor::new(arg))
                .map_err(|e| anyhow!("decode error: {}", e))?;
            let res = if args.use_chat {
                if args.refresh_history {
                    self.request_with_history_refreshed(args)
                } else {
                    self.request_chat_with_history(args)
                }
            } else {
                self.request_generation(args)
            }?;
            let mut buf = Vec::with_capacity(res.encoded_len());
            res.encode(&mut buf).unwrap();
            Ok(buf)
        };
        (res(), metadata)
    }
    fn cancel(&self) -> bool {
        tracing::warn!("OllamaPromptRunner cancel: not implemented!");
        false
    }
    fn is_canceled(&self) -> bool {
        tracing::warn!("OllamaPromptRunner is_canceled: not implemented!");
        false
    }
    fn runner_settings_proto(&self) -> String {
        include_str!("../../llama-protobuf/protobuf/ollama/ollama_runner.proto").to_string()
    }
    fn job_args_proto(&self) -> String {
        include_str!("../../llama-protobuf/protobuf/ollama/ollama_args.proto").to_string()
    }
    fn result_output_proto(&self) -> Option<String> {
        Some(include_str!("../../llama-protobuf/protobuf/ollama/ollama_args.proto").to_string())
    }
    fn settings_schema(&self) -> String {
        schema_to_json_string!(LlamaRunnerSettings, "settings_schema")
    }
    fn arguments_schema(&self) -> String {
        schema_to_json_string!(LlamaArg, "arguments_schema")
    }
    fn output_json_schema(&self) -> Option<String> {
        schema_to_json_string_option!(LlamaArg, "arguments_schema")
    }
    fn output_type(&self) -> jobworkerp_client::jobworkerp::data::StreamingOutputType {
        jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming
    }

    fn begin_stream(
        &mut self,
        arg: Vec<u8>,
        metadata: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        // default implementation (return empty)
        let (_, _) = (arg, metadata);
        Err(anyhow::anyhow!("not implemented"))
    }

    fn receive_stream(&mut self) -> Result<Option<Vec<u8>>> {
        // default implementation (return empty)
        Err(anyhow::anyhow!("not implemented"))
    }
}

#[cfg(test)]
mod test {
    use jobworkerp_llama_protobuf::protobuf::ollama::{
        ollama_args::OllamaOptions, OllamaRunnerSettings,
    };
    use tracing::Level;

    // create a test that loads the plugin model from environment variables and runs it internal model (llama_model)
    use super::*;
    #[ignore = "need to run with local server"]
    #[test]
    fn test_plugin_runner() {
        command_utils::util::tracing::tracing_init_test(Level::DEBUG);

        let settings = OllamaRunnerSettings {
            base_url: Some("http://localhost:11434".to_string()),
            model: "phi4".to_string(),
            system_prompt: Some(
                "次の文章を日本語に翻訳してください。翻訳結果のみを出力してください".to_string(),
            ),
            pull_model: Some(false),
        };
        let mut buf = Vec::with_capacity(settings.encoded_len());
        OllamaRunnerSettings::encode(&settings, &mut buf).unwrap();
        let settings = buf;
        let mut plugin = OllamaPlugin::new();
        plugin
            .load(settings)
            .expect("failed to load model from env");

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

If you want to test more frequently we’ve made a secondary leaderboard, ARC-AGI-Pub, just for this check out our launch post for more information.

We appreciate your understanding and continued participation in ARC Prize. If you have any questions, you can reach us at: team@arcprize.org

Good luck in the competition and in advancing AI research!
        "#;
        let prompt = user_prompt.to_string();

        let request = OllamaArgs {
            prompt,
            options: Some(OllamaOptions {
                sample_len: Some(2048),
                temperature: Some(0.4),
                top_p: Some(0.9),
                repeat_penalty: Some(0.9),
                repeat_last_n: Some(8),
                seed: Some(32),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let res = plugin
            .run(buf, HashMap::new())
            .0
            .expect("failed to run plugin");
        let res = OllamaArgs::decode(&mut Cursor::new(res.clone()))
            .map_err(|e| anyhow!("decode error: {}", e))
            .unwrap();
        println!("response: {:?}", res.prompt);
        assert!(res.prompt.len() > 10 && res.prompt.len() < 4096);
    }

    #[cfg(test)]
    #[test]
    fn test_divide_think_tag() {
        let (prompt, think) = OllamaPlugin::divide_think_tag(
            "aaa\n<think>\nbbb\nccc\n</think>\nddd\neee".to_string(),
        );
        assert_eq!(prompt, "aaa\n\nddd\neee");
        assert_eq!(think, Some("bbb\nccc".to_string()));

        let (prompt, think) =
            OllamaPlugin::divide_think_tag("<think>\naaa\nbbb\nccc\nddd\neee".to_string());
        assert_eq!(prompt, "<think>\naaa\nbbb\nccc\nddd\neee");
        assert_eq!(think, None);
    }
}
