pub mod model;

use anyhow::{anyhow, Context, Result};
use jobworkerp_client::{
    plugins::PluginRunner, schema_to_json_string, schema_to_json_string_option,
};
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings};
use model::{LlamaModelConfig, LlamaModelWrapper};
use prost::Message;
use std::io::Cursor;

// suppress warn improper_ctypes_definitions
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn load_plugin() -> Box<dyn PluginRunner + Send + Sync> {
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
        tracing::error!("load_plugin panic: {:?}, try to load by default config", e);
        Box::new(LlamaCppPlugin { llama_model: None })
    })
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn free_plugin(ptr: Box<dyn PluginRunner + Send + Sync>) {
    drop(ptr);
}

pub struct LlamaCppPlugin {
    pub llama_model: Option<LlamaModelWrapper>,
}
// static DATA: OnceCell<Bytes> = OnceCell::new();

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
}

impl Default for LlamaCppPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRunner for LlamaCppPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.operation
        String::from("LLMPromptRunner")
    }
    fn description(&self) -> String {
        String::from(
            "LLMPromptRunner is a plugin that lets you run LLM models with your own prompts and custom settings",        )
    }
    fn load(&mut self, settings: Vec<u8>) -> Result<()> {
        let settings = LlamaRunnerSettings::decode(&mut Cursor::new(settings))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        tracing::debug!("LLMRunner load: {settings:?}",);
        self.load_model(settings.into())?;
        Ok(())
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        if let Some(llama_model) = self.llama_model.as_mut() {
            let args = LlamaArg::decode(&mut Cursor::new(arg))
                .map_err(|e| anyhow!("decode error: {}", e))?;
            tracing::debug!("LLMRunner run: {args:?}",);
            let text = llama_model
                .run(args.clone().into())
                .context("failed to decode")?;
            tracing::debug!("END OF LLMRunner: {text:?}",);
            let buf = LlamaArg {
                prompt: text,
                ..args
            };
            // serialize and return
            let bytes = buf.encode_to_vec();
            Ok(vec![bytes])
        } else {
            Err(anyhow!("llama_model is not loaded"))
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
        include_str!("../../llama-protobuf/protobuf/llama_cpp/llama_cpp_runner.proto").to_string()
    }
    fn job_args_proto(&self) -> String {
        include_str!("../../llama-protobuf/protobuf/llama_cpp/llama_cpp_arg.proto").to_string()
    }
    fn result_output_proto(&self) -> Option<String> {
        // for prompt chain
        Some(
            include_str!("../../llama-protobuf/protobuf/llama_cpp/llama_cpp_arg.proto").to_string(),
        )
    }
    fn settings_schema(&self) -> String {
        schema_to_json_string!(LlamaRunnerSettings, "settings_schema")
    }
    fn arguments_schema(&self) -> String {
        schema_to_json_string!(LlamaArg, "arguments_schema")
    }
    fn output_json_schema(&self) -> Option<String> {
        schema_to_json_string_option!(LlamaRunnerSettings, "settings_schema")
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
LLAMA_MODEL=phi-4-q4.gguf #Llama-3-ELYZA-JP-8B-q4_k_m.gguf # Phi-3-medium-128k-instruct.Q4_K.gguf # Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf #llama-2-7b-chat.Q4_K_M.gguf
LLAMA_HF_REPO=microsoft/phi-4-gguf #elyza/Llama-3-ELYZA-JP-8B-GGUF # legraphista/Phi-3-medium-128k-instruct-IMat-GGUF # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF #TheBloke/Llama-2-7B-Chat-GGUF
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

If you want to test more frequently we’ve made a secondary leaderboard, ARC-AGI-Pub, just for this check out our launch post for more information.

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
            temperature: Some(0.8),
            top_p: Some(0.9),
            repeat_penalty: Some(0.9),
            repeat_last_n: Some(8),
            seed: Some(32),
            need_print: true,
        };
        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let res = plugin.run(buf).expect("failed to run plugin");
        let res = LlamaArg::decode(&mut Cursor::new(res[0].clone()))
            .map_err(|e| anyhow!("decode error: {}", e))
            .unwrap();
        println!("response: {:?}", res.prompt);
        assert!(res.prompt.len() > 10 && res.prompt.len() < 4096);
    }
}
