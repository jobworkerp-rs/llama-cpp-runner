use anyhow::{bail, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings, MediaInput};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        params::{kv_overrides::ParamOverrideValue, LlamaModelParams},
        AddBos, LlamaChatMessage, LlamaModel,
    },
    sampling::LlamaSampler,
    token::LlamaToken,
};
use llama_cpp_sys_2::{LLAMA_FLASH_ATTN_TYPE_DISABLED, LLAMA_FLASH_ATTN_TYPE_ENABLED};
use mtmd_support::{MediaLimits, MtmdRuntime};
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

// #[derive(Debug, Clone, Serialize, Deserialize)]
// struct Model {
//     /// Use an already downloaded model
//     /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
//     local_path: Option<PathBuf>,
//     /// Download a model from huggingface (or use a cached version)
//     /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
//     hf_repo: Option<String>,
//     /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
//     hf_model: Option<String>,
// }

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
    /// Multimodal projector settings. When None the runner is text-only.
    #[serde(default)]
    mtmd: Option<jobworkerp_llama_protobuf::MtmdSettings>,
}
impl From<LlamaRunnerSettings> for LlamaModelConfig {
    fn from(op: LlamaRunnerSettings) -> Self {
        Self {
            model: op.model,
            hf_repo: op.hf_repo,
            key_value_overrides: None, // TODO
            disable_gpu: op.disable_gpu,
            seed: op.seed,
            threads: op.threads,
            threads_batch: op.threads_batch,
            ctx_size: op.ctx_size.and_then(NonZeroU32::new),
            use_flash_attention: op.use_flash_attention,
            system_prompt: op.system_prompt,
            mtmd: op.mtmd,
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
                let api = ApiBuilder::from_env()
                    .with_progress(false)
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
            mtmd: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceArgs {
    /// The prompt
    prompt: String,
    /// set the length of the prompt + output in tokens
    sample_len: i32,
    /// The temperature used to generate samples.
    temperature: Option<f64>,
    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: Option<f32>,
    /// The context size to consider for the repeat penalty.
    repeat_last_n: Option<u32>,
    /// Media items attached to the prompt.
    #[serde(default, skip_serializing)]
    medias: Vec<MediaInput>,
}

impl std::fmt::Debug for InferenceArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceArgs")
            .field("prompt", &self.prompt)
            .field("sample_len", &self.sample_len)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("repeat_last_n", &self.repeat_last_n)
            .field("medias", &format!("[{} items]", self.medias.len()))
            .finish()
    }
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
            medias: req.medias,
        }
    }
}

pub struct LlamaModelWrapper {
    model: LlamaModel,
    backend: LlamaBackend,
    ctx_params: LlamaContextParams,
    system_prompt: String,
    /// No Mutex needed: all access is through `&mut self` on `run()`,
    /// which provides exclusive access at the Rust borrow-checker level.
    mtmd: Option<MtmdRuntime>,
    media_limits: MediaLimits,
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
            .with_flash_attention_policy(if config.use_flash_attention.unwrap_or(true) {
                LLAMA_FLASH_ATTN_TYPE_ENABLED
            } else {
                LLAMA_FLASH_ATTN_TYPE_DISABLED
            });
        if let Some(threads) = config.threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }
        if let Some(threads_batch) = config.threads_batch.or(config.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch as i32);
        }

        // Initialize multimodal projector when settings are provided.
        let (mtmd, media_limits) = match &config.mtmd {
            Some(mtmd_settings) => {
                let mmproj_path = Self::resolve_mmproj_path(
                    &mtmd_settings.mmproj,
                    mtmd_settings.mmproj_hf_repo.as_deref(),
                )?;
                let use_gpu = mtmd_settings.mmproj_use_gpu.unwrap_or(!config.disable_gpu);
                let runtime = MtmdRuntime::from_settings(
                    mmproj_path.to_string_lossy().as_ref(),
                    &model,
                    use_gpu,
                    mtmd_settings.media_marker.as_deref(),
                )
                .map_err(|e| anyhow::anyhow!(e).context("mtmd init failed"))?;
                let limits = MediaLimits::from_settings(mtmd_settings);
                (Some(runtime), limits)
            }
            None => (None, MediaLimits::default()),
        };

        Ok(Self {
            model,
            backend,
            ctx_params,
            system_prompt: config.system_prompt.unwrap_or_default(),
            mtmd,
            media_limits,
        })
    }

    /// Resolve mmproj file path — download from HF if hf_repo is provided.
    fn resolve_mmproj_path(mmproj: &str, hf_repo: Option<&str>) -> Result<PathBuf> {
        match hf_repo {
            Some(repo) => {
                let api = ApiBuilder::from_env()
                    .with_progress(false)
                    .build()
                    .with_context(|| "unable to create huggingface api for mmproj")?
                    .model(repo.to_string());
                api.get(mmproj)
                    .with_context(|| format!("unable to download mmproj: {mmproj}"))
            }
            None => Ok(PathBuf::from(mmproj)),
        }
    }

    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
    }

    pub fn run(&mut self, args: InferenceArgs) -> Result<String> {
        if args.prompt.is_empty() && args.medias.is_empty() {
            bail!("prompt is empty and no media provided")
        };

        // tokenize the prompt
        self.decode(args)
    }

    fn create_batch(&self, prompt: &str, ctx: &LlamaContext, n_len: i32) -> Result<LlamaBatch<'_>> {
        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {prompt}"))?;

        // n_len is an absolute position cap (total tokens including prompt),
        // consistent with the proto contract and the multimodal path.
        self.check_token_length(&tokens_list, ctx, n_len)?;

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
        // n_len is an absolute position cap (same semantics as multimodal).
        let n_ctx = ctx.n_ctx() as i32;
        let prompt_tokens = tokens_list.len() as i32;

        tracing::info!("sample_len = {n_len}, n_ctx = {n_ctx}, prompt_tokens = {prompt_tokens}");

        if n_len > n_ctx {
            bail!(
                "sample_len > n_ctx ({n_len} > {n_ctx}). \
                 Increase ctx_size or reduce sample_len."
            );
        }
        if prompt_tokens >= n_len {
            bail!(
                "prompt ({prompt_tokens} tokens) already meets or exceeds \
                 sample_len ({n_len}). Increase sample_len to allow generation."
            );
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
        match (args.medias.is_empty(), self.mtmd.is_some()) {
            (true, _) => self.decode_text_only(args),
            (false, false) => bail!("multimodal input given but mmproj is not configured"),
            (false, true) => self.decode_multimodal(args),
        }
    }

    fn decode_text_only(&mut self, args: InferenceArgs) -> Result<String> {
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
            let tmpl = self.model.chat_template(None)?;
            match self.model.apply_chat_template(&tmpl, vec.as_slice(), true) {
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

                let output_bytes = token_to_piece_bytes_retry(&self.model, token)?;
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

    fn decode_multimodal(&mut self, args: InferenceArgs) -> Result<String> {
        // Fail fast on obviously invalid sample_len before touching any
        // expensive resource (bitmap decode, projector prefill, context
        // allocation). n_ctx is only known after new_context() below, but
        // `n_len <= 0` can be checked here.
        let n_len: i32 = args.sample_len;
        if n_len <= 0 {
            bail!("sample_len must be > 0 (got {n_len})");
        }

        let mtmd = self
            .mtmd
            .as_ref()
            .expect("mtmd should be Some in multimodal path");

        // Build the context up front so we can validate sample_len against
        // n_ctx before doing any expensive I/O (bitmap decode) or projector
        // prefill. Context allocation is relatively cheap; prefill and
        // bitmap decoding are not.
        let mut ctx: LlamaContext = self
            .model
            .new_context(&self.backend, self.ctx_params.clone())
            .with_context(|| "unable to create context for multimodal")?;
        let n_ctx = ctx.n_ctx() as i32;
        if n_len > n_ctx {
            bail!(
                "sample_len > n_ctx ({n_len} > {n_ctx}) for mtmd path. \
                 Increase ctx_size or reduce sample_len."
            );
        }

        let bitmaps = mtmd
            .prepare_bitmaps(&args.medias, &self.media_limits)
            .map_err(|e| anyhow::anyhow!(e).context("mtmd: preparing bitmaps"))?;

        let prompt_with_markers = mtmd
            .inject_markers(&args.prompt, bitmaps.len())
            .map_err(|e| anyhow::anyhow!(e).context("mtmd: injecting markers"))?;

        // Reject system_prompt containing media markers — consistent with
        // embedding-llm's instruction validation.
        if self.system_prompt.contains(mtmd.media_marker()) {
            bail!(
                "system_prompt must not contain media marker '{}'",
                mtmd.media_marker()
            );
        }

        // Build chat messages with markers inside the user turn.
        // Use the same fallback strategy as the text-only path: if chat
        // template application fails, fall back to simple concatenation.
        let chats = match (
            LlamaChatMessage::new("system".to_string(), self.system_prompt.clone()),
            LlamaChatMessage::new("user".to_string(), prompt_with_markers.clone()),
        ) {
            (Ok(system), Ok(user)) => Some(vec![system, user]),
            (e1, e2) => {
                tracing::warn!(
                    "cannot create chat messages in multimodal path: {:?}, {:?}",
                    e1,
                    e2
                );
                None
            }
        };
        let formatted = if let Some(vec) = chats {
            let tmpl = self.model.chat_template(None);
            match tmpl {
                Ok(tmpl) => match self.model.apply_chat_template(&tmpl, vec.as_slice(), true) {
                    Ok(v) => {
                        tracing::debug!("multimodal: applied chat template: {}", &v);
                        v
                    }
                    Err(e) => {
                        tracing::warn!(
                            "multimodal: cannot apply chat template (use simple prompt): {:?}",
                            e
                        );
                        format!("{}\n\n{}", self.system_prompt, &prompt_with_markers)
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "multimodal: cannot get chat template (use simple prompt): {:?}",
                        e
                    );
                    format!("{}\n\n{}", self.system_prompt, &prompt_with_markers)
                }
            }
        } else {
            format!("{}\n\n{}", self.system_prompt, &prompt_with_markers)
        };

        tracing::debug!("multimodal formatted prompt: {}", &formatted);

        // Marker count is validated by inject_markers() on the user prompt.
        // system_prompt markers are stripped above. The final formatted
        // prompt should contain exactly bitmaps.len() markers; if not,
        // tokenize_and_prefill() will return a Tokenize error.

        // Prefill via mtmd (tokenize + eval_chunks)
        let n_past = mtmd
            .tokenize_and_prefill(&mut ctx, &formatted, &bitmaps, /* n_batch= */ 512)
            .map_err(|e| anyhow::anyhow!(e).context("mtmd: tokenize_and_prefill"))?;

        // sample_len is an absolute position cap (total tokens including
        // prefill), consistent with the text-only path. The `n_len > n_ctx`
        // branch is already rejected above; here we only need to ensure
        // there is room to generate at least one token after prefill.
        if n_past >= n_len {
            bail!(
                "prefill ({n_past} tokens) already meets or exceeds \
                 sample_len ({n_len}). Increase sample_len to allow generation."
            );
        }

        // Generation loop — start from the position returned by eval_chunks.
        let mut batch = LlamaBatch::new(1, 1);
        let mut n_cur = n_past;
        let n_stop = n_len; // absolute position cap, same semantics as text path
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output_buffer = String::with_capacity(((n_len - n_past) * 4) as usize);
        let mut first = true;
        let mut n_decode = 0;

        let t_main_start = ggml_time_us();

        while n_cur <= n_stop {
            // First sample uses the logit from eval_chunks (logits_last=true),
            // subsequent samples use the single token in the batch.
            let sample_idx = if first { -1 } else { batch.n_tokens() - 1 };
            let token = sampler.sample(&ctx, sample_idx);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let output_bytes = token_to_piece_bytes_retry(&self.model, token)?;
            let mut output_string = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            output_buffer.push_str(&output_string);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .with_context(|| "failed to eval in multimodal loop")?;
            first = false;
            n_decode += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        tracing::info!(
            "multimodal: decoded {} tokens in {:.2} s, speed {:.2} t/s",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );
        tracing::info!("{}", ctx.timings());

        Ok(output_buffer)
    }
}

/// Convert a token to bytes, retrying with a larger buffer if the first
/// attempt reports `InsufficientBufferSpace`. Mirrors the internal retry
/// logic of `LlamaModel::token_to_piece`, but returns raw bytes so the
/// caller can feed them to an incremental UTF-8 decoder.
fn token_to_piece_bytes_retry(
    model: &LlamaModel,
    token: LlamaToken,
) -> Result<Vec<u8>, llama_cpp_2::TokenToStringError> {
    match model.token_to_piece_bytes(token, 8, /* special= */ true, None) {
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(i)) => {
            let size = (-i).try_into().expect("Error buffer size is positive");
            model.token_to_piece_bytes(token, size, true, None)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_args(prompt: &str, medias: Vec<MediaInput>) -> InferenceArgs {
        InferenceArgs {
            prompt: prompt.to_string(),
            sample_len: 128,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            medias,
        }
    }

    fn dummy_media() -> MediaInput {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        MediaInput {
            kind: MediaKind::Image as i32,
            source: Some(Source::Encoded(vec![0xFF, 0xD8, 0xFF])),
            id: None,
        }
    }

    #[test]
    fn test_run_rejects_empty_prompt_and_no_media() {
        // Cannot test run() without a loaded model, but we can verify that
        // InferenceArgs conversion works correctly and the guard condition
        // in run() would allow media-only input.
        let args = make_args("", vec![]);
        assert!(args.prompt.is_empty());
        assert!(args.medias.is_empty());
        // Both empty → should be rejected by run()
    }

    #[test]
    fn test_media_only_args_are_valid() {
        // Media-only input (empty prompt + medias) should NOT be rejected.
        let args = make_args("", vec![dummy_media()]);
        assert!(args.prompt.is_empty());
        assert!(!args.medias.is_empty());
        // Only prompt empty → should be accepted by run()
    }

    #[test]
    fn test_text_only_args_are_valid() {
        let args = make_args("Hello", vec![]);
        assert!(!args.prompt.is_empty());
        assert!(args.medias.is_empty());
    }

    #[test]
    fn test_text_plus_media_args_are_valid() {
        let args = make_args("Describe this image", vec![dummy_media()]);
        assert!(!args.prompt.is_empty());
        assert!(!args.medias.is_empty());
    }

    #[test]
    fn test_inference_args_from_llama_arg() {
        let arg = LlamaArg {
            prompt: String::new(),
            sample_len: 100,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            need_print: false,
            medias: vec![dummy_media(), dummy_media()],
        };
        let args: InferenceArgs = arg.into();
        assert!(args.prompt.is_empty());
        assert_eq!(args.medias.len(), 2);
    }

    /// Multimodal generation happy-path with a real model.
    ///
    /// Requires gemma-4-E4B-it (downloaded from HuggingFace on first run).
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_multimodal_generation_with_real_model \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_multimodal_generation_with_real_model() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;

        // Use the test image shipped in the repo root.
        let image_bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../output.png"))
                .expect("output.png should exist in repo root");

        let config = LlamaModelConfig {
            model: "gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            hf_repo: Some("unsloth/gemma-4-E4B-it-GGUF".to_string()),
            key_value_overrides: None,
            disable_gpu: false,
            seed: None,
            threads: None,
            threads_batch: None,
            ctx_size: std::num::NonZeroU32::new(2048),
            use_flash_attention: Some(true),
            system_prompt: Some("You are a helpful assistant.".to_string()),
            mtmd: Some(jobworkerp_llama_protobuf::MtmdSettings {
                mmproj: "mmproj-F16.gguf".to_string(),
                mmproj_hf_repo: Some("unsloth/gemma-4-E4B-it-GGUF".to_string()),
                mmproj_use_gpu: None,
                media_marker: None,
                allow_url_fetch: false,
                max_media_bytes: 10_000_000,
                max_decoded_media_bytes: 100_000_000,
                allowed_media_dirs: vec![],
            }),
        };

        let mut wrapper = LlamaModelWrapper::new(config).expect("Failed to load model");
        assert!(wrapper.mtmd.is_some(), "mmproj should be loaded");

        let args = InferenceArgs {
            prompt: "What do you see in this image?".to_string(),
            sample_len: 512,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            medias: vec![MediaInput {
                kind: MediaKind::Image as i32,
                source: Some(Source::Encoded(image_bytes)),
                id: None,
            }],
        };

        let output = wrapper.run(args).expect("Multimodal generation failed");
        assert!(!output.is_empty(), "Output should not be empty");
        println!("Multimodal output: {output}");
    }

    /// Multimodal audio generation happy-path with a real model.
    ///
    /// Requires Qwen2.5-Omni-7B (audio-capable mmproj).
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_multimodal_audio_generation \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_multimodal_audio_generation() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;

        let audio_bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../hello_test.wav"),
        )
        .expect("hello_test.wav should exist in repo root");

        let config = LlamaModelConfig {
            model: "Qwen2.5-Omni-7B-Q4_K_M.gguf".to_string(),
            hf_repo: Some("ggml-org/Qwen2.5-Omni-7B-GGUF".to_string()),
            key_value_overrides: None,
            disable_gpu: false,
            seed: None,
            threads: None,
            threads_batch: None,
            ctx_size: std::num::NonZeroU32::new(4096),
            use_flash_attention: Some(true),
            system_prompt: Some("You are a helpful assistant.".to_string()),
            mtmd: Some(jobworkerp_llama_protobuf::MtmdSettings {
                mmproj: "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf".to_string(),
                mmproj_hf_repo: Some("ggml-org/Qwen2.5-Omni-7B-GGUF".to_string()),
                mmproj_use_gpu: None,
                media_marker: None,
                allow_url_fetch: false,
                max_media_bytes: 10_000_000,
                max_decoded_media_bytes: 100_000_000,
                allowed_media_dirs: vec![],
            }),
        };

        let mut wrapper = LlamaModelWrapper::new(config).expect("Failed to load model");
        assert!(wrapper.mtmd.is_some(), "mmproj should be loaded");

        let args = InferenceArgs {
            prompt: "Transcribe the following audio.".to_string(),
            sample_len: 1024,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            medias: vec![MediaInput {
                kind: MediaKind::Audio as i32,
                source: Some(Source::Encoded(audio_bytes)),
                id: None,
            }],
        };

        let output = wrapper.run(args).expect("Audio generation failed");
        assert!(!output.is_empty(), "Output should not be empty");
        println!("Audio output: {output}");
    }
}
