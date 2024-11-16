pub mod model;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use model::{LlamaModelConfig, LlamaModelWrapper};
use prost::Message;
use protobuf::llama_cpp::{LlamaArg, LlamaOperation};
use std::io::Cursor;

pub mod protobuf {
    pub mod llama_cpp {
        include!(concat!(env!("OUT_DIR"), "/llama_cpp.rs"));
    }
}

#[async_trait]
pub trait PluginRunner: Send + Sync {
    fn name(&self) -> String;
    fn load(&mut self, operation: Vec<u8>) -> Result<()>;
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>>;
    fn cancel(&self) -> bool;
    fn operation_proto(&self) -> String;
    fn job_args_proto(&self) -> String;
    fn result_output_proto(&self) -> Option<String>;
    // if true, use job result of before job, else use job args from request
    fn use_job_result(&self) -> bool;
}

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
    fn load(&mut self, operation: Vec<u8>) -> Result<()> {
        let operation = LlamaOperation::decode(&mut Cursor::new(operation))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        tracing::debug!("LLMRunner load: {operation:?}",);
        self.load_model(operation.into())?;
        Ok(())
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        if let Some(llama_model) = self.llama_model.as_mut() {
            let args = LlamaArg::decode(&mut Cursor::new(arg))
                .map_err(|e| anyhow!("decode error: {}", e))?;
            tracing::debug!("LLMRunner run: {args:?}",);
            let text = llama_model.run(args.into()).context("failed to decode")?;
            tracing::debug!("END OF LLMRunner: {text:?}",);
            // serialize and return
            Ok(vec![text.bytes().collect()])
        } else {
            Err(anyhow!("llama_model is not loaded"))
        }
    }
    fn cancel(&self) -> bool {
        tracing::warn!("LLMRunner cancel: not implemented!");
        false
    }
    fn operation_proto(&self) -> String {
        include_str!("../protobuf/llama_cpp_operation.proto").to_string()
    }
    fn job_args_proto(&self) -> String {
        include_str!("../protobuf/llama_cpp_arg.proto").to_string()
    }
    fn result_output_proto(&self) -> Option<String> {
        Some("".to_string()) // string
    }
    // if true, use job result of before job, else use job args from request
    fn use_job_result(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use protobuf::llama_cpp::LlamaArg;

    // create a test that loads the plugin model from environment variables and runs it internal model (llama_model)
    use super::*;
    #[tokio::test]
    async fn test_plugin_runner() {
        tracing_subscriber::fmt::init();
        let env = "
#LLAMA_MODEL=Llama-3-ELYZA-JP-8B-q4_k_m.gguf # Phi-3-medium-128k-instruct.Q4_K.gguf # Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf #llama-2-7b-chat.Q4_K_M.gguf
#LLAMA_HF_REPO=elyza/Llama-3-ELYZA-JP-8B-GGUF # legraphista/Phi-3-medium-128k-instruct-IMat-GGUF # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF #TheBloke/Llama-2-7B-Chat-GGUF
LLAMA_MODEL=tokyotech-llm-Llama-3.1-Swallow-70B-Instruct-v0.1-Q4_K_M.gguf # Phi-3-medium-128k-instruct.Q4_K.gguf # Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf #llama-2-7b-chat.Q4_K_M.gguf
LLAMA_HF_REPO=mmnga/tokyotech-llm-Llama-3.1-Swallow-70B-Instruct-v0.1-gguf # legraphista/Phi-3-medium-128k-instruct-IMat-GGUF # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF #TheBloke/Llama-2-7B-Chat-GGUF

LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=次の英語の文章を日本語に翻訳してください。翻訳結果のみを出力してください
";
        dotenvy::from_read(env.as_bytes()).ok();

        let user_prompt = r#"
There have been many moments of extreme danger over the past year. This is the worst.
In the past seven days, Hezbollah leader Hassan Nasrallah has been assassinated, Israel has launched a ground invasion of Lebanon, and Iran has fired nearly 200 ballistic missiles at targets across Israel.
Western and regional powers - led by the US - have pushed for de-escalation. The UN Security Council called for an "immediate end" to hostilities and the G7, which includes the US, UK and Germany, has called for “restraint”.
But so far those efforts have failed - and the Middle East stands closer than ever to all-out war.
Here’s how the last week played out.
"#;
        let prompt = user_prompt.to_string();

        //         let prompt = format!(
        //             "<|user|>
        // {user_prompt}<|end|>
        // <|assistant|>"
        //         );
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
        let text = String::from_utf8_lossy(&res[0]);
        println!("response: {:?}", text);
        assert!(text.len() > user_prompt.len() && text.len() < 2048);
    }
}
