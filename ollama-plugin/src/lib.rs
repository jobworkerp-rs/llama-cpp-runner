use anyhow::{anyhow, Context, Result};
use jobworkerp_llama_protobuf::{
    protobuf::ollama::{OllamaArg, OllamaOperation},
    PluginRunner,
};
use ollama_rs::{
    generation::{
        completion::{request::GenerationRequest, GenerationResponse},
        options::GenerationOptions,
    },
    Ollama,
};
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

impl OllamaPlugin {
    const URL_BASE: &'static str = "http://localhost:11434";
    pub fn new() -> Self {
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
            self.load_model(OllamaOperation {
                base_url: Some(url_base),
                model,
                system_prompt,
                pull_model,
            })
            .await
        })?;
        Ok(())
    }
    pub async fn load_model(&mut self, operation: OllamaOperation) -> Result<()> {
        let llama = Ollama::try_new(operation.base_url.unwrap_or(Self::URL_BASE.to_string()))?;
        if operation.pull_model.unwrap_or(true) {
            let pull = llama.pull_model(operation.model.clone(), false).await?;
            tracing::debug!("model loaded: result = {:?}", pull);
        };
        self.ollama = Some(llama);
        self.model = operation.model;
        self.system_prompt = operation.system_prompt;
        Ok(())
    }
}

impl PluginRunner for OllamaPlugin {
    fn name(&self) -> String {
        // specify as same string as worker.operation
        String::from("OllamaPromptRunner")
    }
    fn load(&mut self, operation: Vec<u8>) -> Result<()> {
        let operation = OllamaOperation::decode(&mut Cursor::new(operation))
            .map_err(|e| anyhow!("decode error: {}", e))?;
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async { self.load_model(operation).await })?;
        tracing::info!("OllamaPromptRunner loaded",);
        Ok(())
    }
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>> {
        if let Some(ollama) = self.ollama.as_mut() {
            let args = OllamaArg::decode(&mut Cursor::new(arg))
                .map_err(|e| anyhow!("decode error: {}", e))?;
            tracing::debug!("OllamaPromptRunner run: {args:?}",);

            let mut request = GenerationRequest::new(self.model.clone(), args.prompt);
            if let Some(system_prompt) = self.system_prompt.clone() {
                request = request.system(system_prompt);
            }
            let mut options = GenerationOptions::default();
            if let Some(sample_len) = args.sample_len {
                options = options.num_predict(sample_len);
            } else {
                // fill context
                options = options.num_predict(-2);
            }
            if let Some(temperature) = args.temperature {
                options = options.temperature(temperature);
            }
            if let Some(top_p) = args.top_p {
                options = options.top_p(top_p);
            }
            if let Some(repeat_penalty) = args.repeat_penalty {
                options = options.repeat_penalty(repeat_penalty);
            }
            if let Some(repeat_last_n) = args.repeat_last_n {
                options = options.repeat_last_n(repeat_last_n);
            }
            if let Some(seed) = args.seed {
                options = options.seed(seed);
            }
            request = request.options(options);

            // if let Some(context) = context.clone() {
            //     request = request.context(context);
            // }
            let res: GenerationResponse = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(ollama.generate(request))?;

            let text = res.response;
            tracing::debug!(
                "END OF OllamaPromptRunner: duration: {}",
                res.total_duration.unwrap_or_default()
            );
            let buf = OllamaArg {
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
    fn cancel(&self) -> bool {
        tracing::warn!("OllamaPromptRunner cancel: not implemented!");
        false
    }
    fn operation_proto(&self) -> String {
        include_str!("../../llama-protobuf/protobuf/ollama/ollama_operation.proto").to_string()
    }
    fn job_args_proto(&self) -> String {
        include_str!("../../llama-protobuf/protobuf/ollama/ollama_arg.proto").to_string()
    }
    fn result_output_proto(&self) -> Option<String> {
        Some(include_str!("../../llama-protobuf/protobuf/ollama/ollama_arg.proto").to_string())
    }
    // if true, use job result of before job, else use job args from request
    fn use_job_result(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {
    use tracing::Level;

    // create a test that loads the plugin model from environment variables and runs it internal model (llama_model)
    use super::*;
    #[ignore = "need to run with local server"]
    #[test]
    fn test_plugin_runner() {
        command_utils::util::tracing::tracing_init_test(Level::DEBUG);

        let operation = OllamaOperation {
            base_url: Some("http://localhost:11434".to_string()),
            model: "qwq".to_string(),
            system_prompt: Some(
                "次の文章を日本語に翻訳してください。翻訳結果のみを出力してください".to_string(),
            ),
        };
        let mut buf = Vec::with_capacity(operation.encoded_len());
        OllamaOperation::encode(&operation, &mut buf).unwrap();
        let operation = buf;
        let mut plugin = OllamaPlugin::new();
        plugin
            .load(operation)
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

        let request = OllamaArg {
            prompt,
            sample_len: None, //2048,
            temperature: Some(0.4),
            top_p: Some(0.9),
            repeat_penalty: Some(0.9),
            repeat_last_n: Some(8),
            seed: Some(32),
        };
        let mut buf = Vec::with_capacity(request.encoded_len());
        request.encode(&mut buf).unwrap();
        let res = plugin.run(buf).expect("failed to run plugin");
        let res = OllamaArg::decode(&mut Cursor::new(res[0].clone()))
            .map_err(|e| anyhow!("decode error: {}", e))
            .unwrap();
        println!("response: {:?}", res.prompt);
        assert!(res.prompt.len() > 10 && res.prompt.len() < 4096);
    }
}
