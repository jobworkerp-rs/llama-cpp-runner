use anyhow::{bail, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        params::{kv_overrides::ParamOverrideValue, LlamaModelParams},
        AddBos, LlamaChatMessage, LlamaModel, Special,
    },
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use std::{ffi::CString, num::NonZeroU32, path::PathBuf, time::Duration};

/// for deserialization.
#[derive(Debug, Clone, PartialEq, Deserialize)]
enum ParamOverrideValueWrapper {
    /// A string value
    Bool(bool),
    /// A float value
    Float(f64),
    /// A integer value
    Int(i64),
    /// A string value
    Str(String),
}
impl From<ParamOverrideValueWrapper> for ParamOverrideValue {
    fn from(v: ParamOverrideValueWrapper) -> Self {
        match v {
            ParamOverrideValueWrapper::Bool(v) => ParamOverrideValue::Bool(v),
            ParamOverrideValueWrapper::Float(v) => ParamOverrideValue::Float(v),
            ParamOverrideValueWrapper::Int(v) => ParamOverrideValue::Int(v),
            ParamOverrideValueWrapper::Str(v) => {
                let cstr = CString::new(v).expect("CString::new failed");
                let mut array = [0i8; 128];
                let bytes = cstr.as_bytes_with_nul();
                for (i, &byte) in bytes.iter().enumerate() {
                    array[i] = byte as i8;
                    if i >= 127 {
                        break;
                    }
                }
                ParamOverrideValue::Str(array)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Model {
    /// Use an already downloaded model
    /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
    local_path: Option<PathBuf>,
    /// Download a model from huggingface (or use a cached version)
    /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
    hf_repo: Option<String>,
    /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
    hf_model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaModelConfig {
    /// The path or name to the model
    /// if hf_repo is Some, the model is filename on huggingface
    /// if hf_remo is None, the model is a local path
    model: String,
    /// Download a model from huggingface (or use a cached version)
    /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
    hf_repo: Option<String>,
    /// override some parameters of the model
    key_value_overrides: Option<Vec<(String, ParamOverrideValueWrapper)>>,
    /// Disable offloading layers to the gpu
    disable_gpu: bool,
    // RNG seed (default: 1234) TODO
    #[allow(unused)]
    seed: Option<u32>,
    // number of threads to use during generation (default: use all available threads)
    threads: Option<u32>,
    // number of threads to use during batch and prompt processing (default: use all available threads)
    threads_batch: Option<u32>,
    // size of the prompt context (default: model size)
    ctx_size: Option<NonZeroU32>,
    // use flash attention (default true)
    use_flash_attention: Option<bool>,
    // system prompt before the user prompt
    // e.g. `The system will respond to your prompt`
    // This is useful for instructing the user on how to use the model
    // or to provide some context to the user
    system_prompt: Option<String>,
}
impl From<LlamaRunnerSettings> for LlamaModelConfig {
    fn from(op: LlamaRunnerSettings) -> Self {
        Self {
            model: op.model,
            hf_repo: op.hf_repo,
            key_value_overrides: None, // TODO
            // op.key_value_overrides.map(|v| {
            //     v.into_iter()
            //         .map(|(k, v)| (k, v.into()))
            //         .collect::<Vec<_>>()
            // }),
            disable_gpu: op.disable_gpu,
            seed: op.seed,
            threads: op.threads,
            threads_batch: op.threads_batch,
            ctx_size: op.ctx_size.and_then(NonZeroU32::new),
            use_flash_attention: op.use_flash_attention,
            system_prompt: op.system_prompt,
        }
    }
}

impl LlamaModelConfig {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load_model(&self) -> Result<Vec<PathBuf>> {
        let modelfiles = self.model.split(',').map(PathBuf::from).collect::<Vec<_>>();
        match self.hf_repo.clone() {
            None => Ok(modelfiles),
            Some(repo) => {
                let api = ApiBuilder::new()
                    .with_progress(true)
                    .build()
                    .with_context(|| "unable to create huggingface api")?
                    .model(repo);
                modelfiles
                    .iter()
                    .map(|model| {
                        api.get(model.to_string_lossy().as_ref())
                            .with_context(|| "unable to download model")
                    })
                    .collect()
            }
        }
    }
}
impl Default for LlamaModelConfig {
    fn default() -> Self {
        Self {
            model: "llama-2-7b-chat.Q4_K_M.gguf".to_string(),
            hf_repo: Some("TheBloke/Llama-2-7B-Chat-GGUF".to_string()),
            key_value_overrides: None,
            disable_gpu: false,
            seed: None,
            threads: None,
            threads_batch: None,
            ctx_size: None,
            use_flash_attention: None,
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceArgs {
    /// The prompt
    prompt: String,
    /// set the length of the prompt + output in tokens
    sample_len: i32,
    /// The temperature used to generate samples.
    temperature: Option<f64>,
    /// Nucleus sampling probability cutoff.
    // TODO
    top_p: Option<f64>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    /// TODO
    repeat_penalty: Option<f32>,
    /// The context size to consider for the repeat penalty.
    /// TODO
    repeat_last_n: Option<u32>,
}
impl From<LlamaArg> for InferenceArgs {
    fn from(req: LlamaArg) -> Self {
        Self {
            prompt: req.prompt,
            sample_len: req.sample_len as i32,
            temperature: req.temperature,
            top_p: req.top_p,
            repeat_penalty: req.repeat_penalty,
            repeat_last_n: req.repeat_last_n,
        }
    }
}

pub struct LlamaModelWrapper {
    model: LlamaModel,
    backend: LlamaBackend,
    ctx_params: LlamaContextParams,
    system_prompt: String,
}

impl LlamaModelWrapper {
    pub fn new(config: LlamaModelConfig) -> Result<Self> {
        let mut backend = LlamaBackend::init()?;
        backend.void_logs();

        let model_paths = config
            .get_or_load_model()
            .with_context(|| "failed to get model from args")?;

        // offload all layers to the gpu
        let model_params = {
            // #[cfg(any(feature = "cuda", feature = "vulkan"))]
            if !config.disable_gpu {
                LlamaModelParams::default().with_n_gpu_layers(1000)
            } else {
                LlamaModelParams::default()
            }
            // #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
            // LlamaModelParams::default()
        };

        let mut model_params = std::pin::pin!(model_params);
        if let Some(key_value_overrides) = &config.key_value_overrides {
            for (k, v) in key_value_overrides {
                let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
                model_params
                    .as_mut()
                    .append_kv_override(k.as_c_str(), v.clone().into());
            }
        }
        let model = LlamaModel::load_from_file(
            &backend,
            model_paths[0].as_ref() as &std::path::Path,
            &model_params,
        )
        .with_context(|| "unable to load model")?;

        // initialize the context
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(config.ctx_size)
            // .with_seed(config.seed.unwrap_or(1234))
            .with_flash_attention(config.use_flash_attention.unwrap_or(true));
        if let Some(threads) = config.threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }
        if let Some(threads_batch) = config.threads_batch.or(config.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch as i32);
        }

        Ok(Self {
            model,
            backend,
            ctx_params,
            system_prompt: config.system_prompt.unwrap_or_else(|| "".to_string()),
        })
    }

    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
    }

    pub fn run(&mut self, args: InferenceArgs) -> Result<String> {
        if args.prompt.is_empty() {
            bail!("prompt is empty")
        };

        // tokenize the prompt
        self.decode(args)
    }

    fn create_batch(&self, prompt: &str, ctx: &LlamaContext, n_len: i32) -> Result<LlamaBatch> {
        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {prompt}"))?;
        let total_len = tokens_list.len() as i32 + n_len;

        self.check_token_length(&tokens_list, ctx, total_len)?;

        // print the prompt token-by-token

        // create a llama_batch with size 512
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(7542, 1);
        // let mut batch = LlamaBatch::new(total_len as usize, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }
        Ok(batch)
    }

    fn check_token_length(
        &self,
        tokens_list: &[LlamaToken],
        ctx: &LlamaContext,
        n_len: i32,
    ) -> Result<()> {
        let n_cxt = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

        tracing::info!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if n_kv_req > n_cxt {
            bail!(
                "n_kv_req > n_ctx ({n_kv_req} > {n_cxt}), the required kv cache size is not big enough either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= usize::try_from(n_len)? {
            bail!("the prompt is too long, it has more tokens than n_len")
        }

        Ok(())
    }

    // #[inline]
    // fn setup_candidates(
    //     ctx: &mut LlamaContext,
    //     candidates: &mut LlamaTokenDataArray,
    //     history: &[LlamaToken],
    //     args: &InferenceArgs,
    // ) -> Result<()> {
    //     // // setup sampler
    //     // if let Some(penalty) = args.repeat_penalty.as_ref() {
    //     //     let repeat_last_n = args.repeat_last_n.unwrap_or(8) as usize;
    //     //     candidates.sample_repetition_penalty(
    //     //         Some(ctx),
    //     //         history,
    //     //         repeat_last_n,
    //     //         *penalty,
    //     //         0.0, // no frequency penalty
    //     //         0.0, // no present penalty
    //     //     );
    //     // }
    //     // // sampler.push_step(&|c, _| c.sample_top_k(Some(ctx), 40, 1));
    //     // // sampler.push_step(&|c, _| c.sample_tail_free(Some(ctx), 1.0, 1));
    //     // // sampler.push_step(&|c, _| c.sample_typical(Some(ctx), 1.0, 1));
    //     // if let Some(top_p) = args.top_p.as_ref() {
    //     //     let top_p = *top_p as f32;
    //     //     candidates.sample_top_p(Some(ctx), top_p, 1);
    //     // }
    //     // // sampler.push_step(&|c, _| c.sample_min_p(Some(ctx), 0.05, 1));
    //     // if let Some(temperature) = args.temperature {
    //     //     candidates.sample_temp(Some(ctx), temperature as f32);
    //     // }
    //     Ok(())
    // }

    fn decode(&mut self, args: InferenceArgs) -> Result<String> {
        let chats = match (
            LlamaChatMessage::new("system".to_string(), self.system_prompt.clone()),
            LlamaChatMessage::new("user".to_string(), args.prompt.clone()),
        ) {
            (Ok(system), Ok(user)) => Some(vec![system, user]),
            (e1, e2) => {
                tracing::warn!("cannot create chat messages: {:?}, {:?}", e1, e2);
                None
            }
        };
        let prompt = if let Some(vec) = chats {
            match self.model.apply_chat_template(None, vec, true) {
                Ok(v) => {
                    tracing::debug!("applied chat template: {}", &v);
                    v
                }
                Err(e) => {
                    tracing::warn!("cannot apply chat template (use simple prompt): {:?}", e);
                    format!("{}\n\n{}", self.system_prompt, &args.prompt)
                }
            }
        } else {
            format!("{}\n\n{}", self.system_prompt, &args.prompt)
        };

        let n_len: i32 = args.sample_len;
        // let mut history = Vec::<LlamaToken>::with_capacity(args.sample_len as usize);

        let mut ctx: LlamaContext = self
            .model
            .new_context(&self.backend, self.ctx_params.clone())
            .with_context(|| "unable to create the llama_context")?;

        let mut batch = self.create_batch(prompt.as_str(), &ctx, n_len)?;

        // first decode the prompt
        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        // main loop

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;

        let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // XXX assume string byte size 4
        let mut output_buffer = String::with_capacity((n_len * 4) as usize);
        // not output the prompt
        // output_buffer.push_str(prompt.as_str());
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

        while n_cur <= n_len {
            // sample the next token
            {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);

                sampler.accept(token);

                // is it an end of stream?
                if self.model.is_eog_token(token) {
                    eprintln!();
                    break;
                }

                let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    decoder.decode_to_string(&output_bytes, &mut output_string, false);
                // output_buffer.push_str(&output_string);
                // print!("{output_string}");
                // std::io::stdout().flush()?;

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;

                // let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

                // let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                // Self::setup_candidates(&mut ctx, &mut candidates_p, &history, &args)?;
                // // sample the most likely token
                // // candidates_p.sample_softmax(Some(&mut ctx));
                // candidates_p.sample_token_greedy();
                // let new_token = candidates_p.data[0];
                // let new_token_id = new_token.id();
                // // is it an end of stream?
                // if self.model.is_eog_token(new_token_id) {
                //     break;
                // }
                // history.push(new_token_id);

                // let output_bytes = self.model.token_to_bytes(new_token_id, Special::Tokenize)?;
                // // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                // let mut output_string = String::with_capacity(32);
                // let _decode_result =
                //     decoder.decode_to_string(&output_bytes, &mut output_string, false);
                // // print!("{output_string}");
                output_buffer.push_str(&output_string);
                // // std::io::stdout().flush()?;

                // batch.clear();
                // batch.add(new_token_id, n_cur, &[0], true)?;
            }

            n_cur += 1;

            ctx.decode(&mut batch).with_context(|| "failed to eval")?;

            n_decode += 1;
        }

        let t_main_end = ggml_time_us();

        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

        tracing::info!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        tracing::info!("{}", ctx.timings());

        Ok(output_buffer)
    }
}
