use anyhow::{Context, Result, anyhow, bail};
use hf_hub::api::sync::ApiBuilder;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings, MediaInput};
use jobworkerp_llama_protobuf::protobuf::llm::{
    LlmChatArgs, LlmChatResult, LlmCompletionArgs, LlmCompletionResult, llm_chat_args,
    llm_chat_result, llm_completion_result,
};
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        AddBos, LlamaChatMessage, LlamaModel,
        params::{LlamaModelParams, kv_overrides::ParamOverrideValue},
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

/// `LlamaChatMessage` has private fields, so we keep raw (role, content) pairs
/// alongside for fallback template formatting.
type ChatBuildResult = (
    Vec<LlamaChatMessage>,
    Vec<(String, String)>,
    Vec<MediaInput>,
);

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceArgs {
    /// The prompt
    prompt: String,
    /// Absolute position cap (prompt + output tokens). Used by legacy "run" path.
    /// None when max_new_tokens is used instead.
    sample_len: Option<i32>,
    /// Max new tokens to generate (relative to prompt length). When set,
    /// overrides sample_len by computing prompt_tokens + max_new_tokens.
    #[serde(default)]
    max_new_tokens: Option<i32>,
    /// The temperature used to generate samples.
    temperature: Option<f64>,
    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: Option<f32>,
    /// The context size to consider for the repeat penalty.
    repeat_last_n: Option<u32>,
    /// RNG seed for sampling. None uses the default (1234).
    seed: Option<u32>,
    /// JSON Schema for structured output (llguidance constraint).
    #[serde(default, skip_serializing)]
    json_schema: Option<String>,
    /// Media items attached to the prompt.
    #[serde(default, skip_serializing)]
    medias: Vec<MediaInput>,
}

impl std::fmt::Debug for InferenceArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceArgs")
            .field("prompt", &self.prompt)
            .field("sample_len", &self.sample_len)
            .field("max_new_tokens", &self.max_new_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("repeat_last_n", &self.repeat_last_n)
            .field("seed", &self.seed)
            .field(
                "json_schema",
                &self.json_schema.as_deref().map(|s| &s[..s.len().min(80)]),
            )
            .field("medias", &format!("[{} items]", self.medias.len()))
            .finish()
    }
}

const DEFAULT_SAMPLER_SEED: u32 = 1234;

/// Result of a single decode pass, carrying both the generated text and the
/// token counts needed to populate `Usage` in chat/completion responses.
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl From<LlamaArg> for InferenceArgs {
    fn from(req: LlamaArg) -> Self {
        Self {
            prompt: req.prompt,
            sample_len: Some(req.sample_len as i32),
            max_new_tokens: None,
            temperature: req.temperature,
            top_p: req.top_p,
            repeat_penalty: req.repeat_penalty,
            repeat_last_n: req.repeat_last_n,
            seed: req.seed.map(|s| s as u32),
            json_schema: None,
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
        // n_batch must cover the whole prompt (decoded in one call): the C++
        // core aborts the process (GGML_ASSERT n_tokens_all <= n_batch) when a
        // prompt exceeds n_batch, so keep it in step with n_ctx.
        if let Some(n_ctx) = config.ctx_size {
            ctx_params = ctx_params.with_n_batch(n_ctx.get());
        }
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

    fn build_sampler(&self, args: &InferenceArgs) -> Result<LlamaSampler> {
        let seed = args.seed.unwrap_or(DEFAULT_SAMPLER_SEED);
        let mut samplers: Vec<LlamaSampler> = vec![LlamaSampler::dist(seed)];

        if let Some(penalty) = args.repeat_penalty {
            let last_n = args.repeat_last_n.map_or(64, |n| n as i32);
            samplers.push(LlamaSampler::penalties(last_n, penalty, 0.0, 0.0));
        }
        if let Some(temp) = args.temperature {
            #[allow(clippy::cast_possible_truncation)]
            samplers.push(LlamaSampler::temp(temp as f32));
        }
        if let Some(p) = args.top_p {
            #[allow(clippy::cast_possible_truncation)]
            samplers.push(LlamaSampler::top_p(p as f32, 1));
        }
        if let Some(schema) = &args.json_schema {
            let llg = LlamaSampler::llguidance(&self.model, "json", schema)
                .map_err(|e| anyhow!("llguidance init failed: {e:?}"))?;
            samplers.push(llg);
        }

        samplers.push(LlamaSampler::greedy());
        Ok(LlamaSampler::chain_simple(samplers))
    }

    pub fn run(&mut self, args: InferenceArgs) -> Result<String> {
        if args.prompt.is_empty() && args.medias.is_empty() {
            bail!("prompt is empty and no media provided")
        };

        // tokenize the prompt
        self.decode(args).map(|o| o.text)
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

    fn decode(&mut self, args: InferenceArgs) -> Result<DecodeOutput> {
        match (args.medias.is_empty(), self.mtmd.is_some()) {
            (true, _) => self.decode_text_only(args),
            (false, false) => bail!("multimodal input given but mmproj is not configured"),
            (false, true) => self.decode_multimodal(args),
        }
    }

    /// Legacy entry point: wraps the user prompt in a system+user chat template
    /// (or falls back to plain concatenation) before delegating to the core
    /// generation loop. Used by the `run` (LlamaArg) path.
    fn decode_text_only(&mut self, args: InferenceArgs) -> Result<DecodeOutput> {
        let formatted = self.format_single_turn(&args.prompt)?;
        self.decode_text_only_core(&formatted, &args)
    }

    /// Core text-only generation loop. Receives a `formatted_prompt` that the
    /// caller is responsible for templating; no chat template or system/user
    /// wrapping happens here. This keeps `run_chat` / `run_completion` from
    /// applying the template twice.
    fn decode_text_only_core(
        &mut self,
        formatted_prompt: &str,
        args: &InferenceArgs,
    ) -> Result<DecodeOutput> {
        let mut ctx: LlamaContext = self
            .model
            .new_context(&self.backend, self.ctx_params.clone())
            .with_context(|| "unable to create the llama_context")?;

        let tokens_list = self
            .model
            .str_to_token(formatted_prompt, AddBos::Always)
            .with_context(|| "failed to tokenize prompt")?;
        let prompt_tokens = tokens_list.len() as i32;

        // Cap the position budget at n_ctx so a default `max_new_tokens` of
        // 4096 doesn't fail outright on small-context models. This mirrors
        // the multimodal core path which already applies `.min(n_ctx)`.
        let n_ctx = ctx.n_ctx() as i32;
        let n_len: i32 = if let Some(max_new) = args.max_new_tokens {
            (prompt_tokens + max_new).min(n_ctx)
        } else {
            args.sample_len.unwrap_or(4096).min(n_ctx)
        };
        self.check_token_length(&tokens_list, &ctx, n_len)?;

        // The prompt is decoded in one shot, so the batch must hold every
        // token. check_token_length already bounds prompt_tokens by n_ctx.
        let mut batch = LlamaBatch::new(prompt_tokens as usize, 1);
        let last_index: i32 = prompt_tokens - 1;
        for (i, token) in (0_i32..).zip(tokens_list) {
            batch.add(token, i, &[0], i == last_index)?;
        }

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
        let mut sampler = self.build_sampler(args)?;

        while n_cur < n_len {
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

        Ok(DecodeOutput {
            text: output_buffer,
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: n_decode as u32,
        })
    }

    /// Legacy entry point: builds the user prompt with media markers, applies
    /// the chat template (wrapping in system+user), and forwards to the
    /// multimodal core. The system_prompt marker check is performed here so
    /// that core callers (chat/completion) can skip it.
    fn decode_multimodal(&mut self, args: InferenceArgs) -> Result<DecodeOutput> {
        let mtmd = self
            .mtmd
            .as_ref()
            .expect("mtmd should be Some in multimodal path");

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

        // Markers are embedded inside the user prompt; chat template
        // application preserves them so tokenize_and_prefill can match the
        // marker count with bitmaps.len().
        let formatted = self.format_single_turn(&prompt_with_markers)?;
        self.decode_multimodal_core(&formatted, &bitmaps, &args)
    }

    /// Core multimodal generation loop. The caller must supply a
    /// `formatted_prompt` that already contains the correct number of media
    /// markers (matching `bitmaps.len()`); marker injection and chat template
    /// application happen in the caller. Marker/bitmap mismatch surfaces as a
    /// `Tokenize` error from `tokenize_and_prefill`.
    fn decode_multimodal_core(
        &mut self,
        formatted_prompt: &str,
        bitmaps: &[llama_cpp_2::mtmd::MtmdBitmap],
        args: &InferenceArgs,
    ) -> Result<DecodeOutput> {
        let mtmd = self
            .mtmd
            .as_ref()
            .expect("mtmd should be Some in multimodal path");

        let mut ctx: LlamaContext = self
            .model
            .new_context(&self.backend, self.ctx_params.clone())
            .with_context(|| "unable to create context for multimodal")?;
        let n_ctx = ctx.n_ctx() as i32;

        tracing::debug!("multimodal formatted prompt: {}", formatted_prompt);

        // Prefill via mtmd (tokenize + eval_chunks)
        let n_past = mtmd
            .tokenize_and_prefill(&mut ctx, formatted_prompt, bitmaps, /* n_batch= */ 512)
            .map_err(|e| anyhow::anyhow!(e).context("mtmd: tokenize_and_prefill"))?;

        let n_len = if let Some(max_new) = args.max_new_tokens {
            (n_past + max_new).min(n_ctx)
        } else {
            args.sample_len.unwrap_or(4096).min(n_ctx)
        };
        if n_past >= n_len {
            bail!(
                "prefill ({n_past} tokens) already meets or exceeds \
                 position cap ({n_len}). Increase max_tokens to allow generation."
            );
        }

        let mut batch = LlamaBatch::new(1, 1);
        let mut n_cur = n_past;
        let n_stop = n_len;
        let mut sampler = self.build_sampler(args)?;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output_buffer = String::with_capacity(((n_len - n_past) * 4) as usize);
        let mut first = true;
        let mut n_decode = 0;

        let t_main_start = ggml_time_us();

        while n_cur < n_stop {
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

        Ok(DecodeOutput {
            text: output_buffer,
            prompt_tokens: n_past as u32,
            completion_tokens: n_decode as u32,
        })
    }

    pub fn run_chat(&mut self, args: LlmChatArgs) -> Result<LlmChatResult> {
        if let Some(ref fo) = args.function_options
            && fo.use_function_calling
        {
            bail!("function calling is not supported by this plugin");
        }
        if args.model.is_some() {
            tracing::warn!(
                "LLMChatArgs.model is ignored: model is fixed at load time in this plugin"
            );
        }

        let options = args.options.unwrap_or_default();
        let extract_reasoning = options.extract_reasoning_content.unwrap_or(false);

        let (chat_messages, raw_messages, medias) = self.build_chat_messages(&args.messages)?;
        let formatted_prompt = self.apply_chat_template_multi(&chat_messages, &raw_messages)?;

        // `prompt` is left empty and `medias` is empty because the core path
        // consumes `formatted_prompt` and `bitmaps` directly; only the
        // sampler/limits fields of InferenceArgs are read downstream.
        let inference_args = InferenceArgs {
            prompt: String::new(),
            sample_len: None,
            max_new_tokens: Some(options.max_tokens.unwrap_or(4096)),
            temperature: options.temperature.map(f64::from),
            top_p: options.top_p.map(f64::from),
            repeat_penalty: options.repeat_penalty,
            repeat_last_n: options.repeat_last_n.map(|v| v as u32),
            seed: options.seed.map(|s| s as u32),
            json_schema: args.json_schema,
            medias: Vec::new(),
        };

        let t_start = ggml_time_us();
        let output: DecodeOutput = if medias.is_empty() {
            self.decode_text_only_core(&formatted_prompt, &inference_args)?
        } else {
            let mtmd = self
                .mtmd
                .as_ref()
                .context("multimodal input given but mmproj is not configured")?;
            let bitmaps = mtmd
                .prepare_bitmaps(&medias, &self.media_limits)
                .map_err(|e| anyhow::anyhow!(e).context("mtmd: preparing bitmaps"))?;
            self.decode_multimodal_core(&formatted_prompt, &bitmaps, &inference_args)?
        };
        let t_end = ggml_time_us();
        let total_time = Duration::from_micros((t_end - t_start) as u64);

        // Split reasoning content from the output if requested
        let (text_content, reasoning_content) = if extract_reasoning {
            Self::extract_reasoning(&output.text)
        } else {
            (output.text, None)
        };

        Ok(LlmChatResult {
            content: Some(llm_chat_result::MessageContent {
                content: Some(llm_chat_result::message_content::Content::Text(
                    text_content,
                )),
            }),
            reasoning_content,
            done: true,
            usage: Some(llm_chat_result::Usage {
                model: String::new(),
                prompt_tokens: Some(output.prompt_tokens),
                completion_tokens: Some(output.completion_tokens),
                total_prompt_time_sec: None,
                total_completion_time_sec: Some(total_time.as_secs_f32()),
            }),
            pending_tool_calls: None,
            requires_tool_execution: None,
            tool_execution_results: vec![],
            tool_execution_started: None,
        })
    }

    /// Execute a text-only completion request. Unlike `run_chat`, this method
    /// does NOT accept media — multimodal callers must use the chat method.
    /// `args.context.ollama_context` and `args.model` are accepted with a warn
    /// (see `validate_completion_args`) so jobworkerp completion workers can
    /// reuse their existing payload shape.
    pub fn run_completion(&mut self, args: LlmCompletionArgs) -> Result<LlmCompletionResult> {
        validate_completion_args(&args)?;

        let options = args.options.unwrap_or_default();
        let extract_reasoning = options.extract_reasoning_content.unwrap_or(false);

        // Per-request system_prompt overrides the load-time one for this call
        // only. `Some("")` is preserved as an explicit "no system message"
        // signal so callers can disable the runner's default without changing
        // runner state; `format_single_turn_inner` drops empty system content.
        let sys = args
            .system_prompt
            .as_deref()
            .unwrap_or(self.system_prompt.as_str());
        let formatted_prompt = self.format_single_turn_inner(sys, &args.prompt)?;

        let inference_args = InferenceArgs {
            prompt: String::new(),
            sample_len: None,
            max_new_tokens: Some(options.max_tokens.unwrap_or(4096)),
            temperature: options.temperature.map(f64::from),
            top_p: options.top_p.map(f64::from),
            repeat_penalty: options.repeat_penalty,
            repeat_last_n: options.repeat_last_n.map(|v| v as u32),
            seed: options.seed.map(|s| s as u32),
            json_schema: args.json_schema,
            medias: Vec::new(),
        };

        let t_start = ggml_time_us();
        let output = self.decode_text_only_core(&formatted_prompt, &inference_args)?;
        let t_end = ggml_time_us();
        let total_time = Duration::from_micros((t_end - t_start) as u64);

        let (text_content, reasoning_content) = if extract_reasoning {
            Self::extract_reasoning(&output.text)
        } else {
            (output.text, None)
        };

        Ok(LlmCompletionResult {
            content: Some(llm_completion_result::MessageContent {
                content: Some(llm_completion_result::message_content::Content::Text(
                    text_content,
                )),
            }),
            reasoning_content,
            done: true,
            context: None,
            usage: Some(llm_completion_result::Usage {
                model: String::new(),
                prompt_tokens: Some(output.prompt_tokens),
                completion_tokens: Some(output.completion_tokens),
                total_prompt_time_sec: None,
                total_completion_time_sec: Some(total_time.as_secs_f32()),
            }),
        })
    }

    /// Media is only committed after the corresponding `LlamaChatMessage` is
    /// successfully created, to avoid marker/media count mismatch in multimodal paths.
    fn build_chat_messages(
        &self,
        messages: &[llm_chat_args::ChatMessage],
    ) -> Result<ChatBuildResult> {
        let mut chat_msgs = Vec::with_capacity(messages.len() + 1);
        let mut raw_msgs: Vec<(String, String)> = Vec::with_capacity(messages.len() + 1);
        let mut medias = Vec::new();

        // Prepend runner's system_prompt if no system message is in the request,
        // matching the legacy decode_text_only behavior.
        let has_system = messages.iter().any(|m| {
            llm_chat_args::ChatRole::try_from(m.role) == Ok(llm_chat_args::ChatRole::System)
        });
        if !has_system && !self.system_prompt.is_empty() {
            let content = self.system_prompt.clone();
            let sys_msg = LlamaChatMessage::new("system".to_string(), content.clone())
                .map_err(|e| anyhow!("invalid system_prompt: {e}"))?;
            chat_msgs.push(sys_msg);
            raw_msgs.push(("system".to_string(), content));
        }

        for msg in messages {
            let role = match llm_chat_args::ChatRole::try_from(msg.role) {
                Ok(llm_chat_args::ChatRole::System) => "system",
                Ok(llm_chat_args::ChatRole::User) => "user",
                Ok(llm_chat_args::ChatRole::Assistant) => "assistant",
                Ok(llm_chat_args::ChatRole::Tool) => "tool",
                Ok(llm_chat_args::ChatRole::Unspecified) | Err(_) => {
                    bail!("unsupported or unknown chat role value: {}", msg.role);
                }
            };

            let mut pending_media = None;

            let text = match &msg.content {
                Some(content) => match &content.content {
                    Some(llm_chat_args::message_content::Content::Text(t)) => t.clone(),
                    Some(llm_chat_args::message_content::Content::Image(img)) => {
                        let source = img.source.as_ref().context("image message has no source")?;
                        let encoded = if !source.base64.is_empty() {
                            use base64::Engine;
                            base64::engine::general_purpose::STANDARD
                                .decode(&source.base64)
                                .context("invalid base64 in image")?
                        } else {
                            bail!("image URL fetch is not supported in this plugin");
                        };
                        pending_media = Some(MediaInput {
                            kind: jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind::Image
                                as i32,
                            source: Some(
                                jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source::Encoded(encoded),
                            ),
                            id: None,
                        });
                        // Embed media marker so decode_multimodal places the
                        // image at the correct position in the prompt.
                        self.mtmd
                            .as_ref()
                            .map(|m| m.media_marker().to_string())
                            .unwrap_or_default()
                    }
                    Some(llm_chat_args::message_content::Content::ToolCalls(tc)) => {
                        serde_json::to_string(&tc.calls).unwrap_or_default()
                    }
                    Some(llm_chat_args::message_content::Content::ToolExecutionRequests(ter)) => {
                        serde_json::to_string(&ter.requests).unwrap_or_default()
                    }
                    None => String::new(),
                },
                None => String::new(),
            };

            let role_owned = role.to_string();
            let chat_msg = LlamaChatMessage::new(role_owned.clone(), text.clone())
                .map_err(|e| anyhow!("invalid chat message (role={role}): {e}"))?;
            chat_msgs.push(chat_msg);
            raw_msgs.push((role_owned, text));
            if let Some(media) = pending_media {
                medias.push(media);
            }
        }

        if chat_msgs.is_empty() {
            bail!("no valid chat messages provided");
        }

        Ok((chat_msgs, raw_msgs, medias))
    }

    /// Format a single user turn using the model's chat template, preserving
    /// the runner's load-time `system_prompt`. Used by the legacy `run` path
    /// and multimodal generation, both of which lack an explicit per-request
    /// system message.
    fn format_single_turn(&self, user_prompt: &str) -> Result<String> {
        self.format_single_turn_inner(self.system_prompt.as_str(), user_prompt)
    }

    /// Format a single user turn with an explicit system prompt. An empty
    /// `system` is treated as "no system message" — common in completion
    /// requests where the caller wants raw user continuation without the
    /// runner's default system message.
    fn format_single_turn_inner(&self, system: &str, user_prompt: &str) -> Result<String> {
        let mut chat_msgs: Vec<LlamaChatMessage> = Vec::with_capacity(2);
        let mut raw_msgs: Vec<(String, String)> = Vec::with_capacity(2);
        if !system.is_empty() {
            let sys_owned = system.to_string();
            chat_msgs.push(
                LlamaChatMessage::new("system".to_string(), sys_owned.clone())
                    .map_err(|e| anyhow!("invalid system message: {e}"))?,
            );
            raw_msgs.push(("system".to_string(), sys_owned));
        }
        let user_owned = user_prompt.to_string();
        chat_msgs.push(
            LlamaChatMessage::new("user".to_string(), user_owned.clone())
                .map_err(|e| anyhow!("invalid user message: {e}"))?,
        );
        raw_msgs.push(("user".to_string(), user_owned));
        self.apply_chat_template_multi(&chat_msgs, &raw_msgs)
    }

    fn apply_chat_template_multi(
        &self,
        messages: &[LlamaChatMessage],
        raw_messages: &[(String, String)],
    ) -> Result<String> {
        match self.model.chat_template(None) {
            Ok(tmpl) => match self.model.apply_chat_template(&tmpl, messages, true) {
                Ok(v) => {
                    tracing::debug!("applied chat template for multi-turn: {}", &v);
                    Ok(v)
                }
                Err(e) => {
                    tracing::warn!(
                        "cannot apply chat template for multi-turn (using fallback): {:?}",
                        e
                    );
                    Ok(Self::fallback_format_messages(raw_messages))
                }
            },
            Err(e) => {
                tracing::warn!("no chat template available (using fallback): {:?}", e);
                Ok(Self::fallback_format_messages(raw_messages))
            }
        }
    }

    fn fallback_format_messages(raw_messages: &[(String, String)]) -> String {
        let mut result: String = raw_messages
            .iter()
            .map(|(role, content)| format!("{role}: {content}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        // Mimic apply_chat_template(add_ass=true) by appending an
        // assistant generation prefix so the model responds in the
        // correct role instead of continuing user text.
        result.push_str("\n\nassistant:");
        result
    }

    pub(crate) fn extract_reasoning(output: &str) -> (String, Option<String>) {
        let Some(start) = output.find("<think>") else {
            return (output.to_string(), None);
        };
        let after_open = start + 7;
        if let Some(rel_end) = output[after_open..].find("</think>") {
            let end = after_open + rel_end;
            let reasoning = output[after_open..end].trim().to_string();
            let mut text = String::with_capacity(output.len());
            text.push_str(&output[..start]);
            text.push_str(&output[end + 8..]);
            return (text.trim().to_string(), Some(reasoning));
        }
        // Open <think> with no matching </think>: max_tokens cut off the
        // reasoning block before the answer started. Treat the tail as
        // reasoning so callers don't receive a half-open <think>...</think>
        // structure in `content.text` and so the answer field stays empty
        // (no answer was actually produced).
        let reasoning = output[after_open..].trim().to_string();
        let text = output[..start].trim().to_string();
        let reasoning_opt = if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        };
        (text, reasoning_opt)
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

/// Validate `LlmCompletionArgs` independently of model loading state so that
/// rejections (empty prompt, function calling) can be unit tested without a
/// model. Warn-only fields (`model`, `context`) are observed here.
pub fn validate_completion_args(args: &LlmCompletionArgs) -> Result<()> {
    if args.prompt.is_empty() {
        bail!("prompt is empty");
    }
    if let Some(fo) = &args.function_options
        && fo.use_function_calling
    {
        bail!("function calling is not supported by this plugin");
    }
    if args.model.is_some() {
        tracing::warn!(
            "LLMCompletionArgs.model is ignored: model is fixed at load time in this plugin"
        );
    }
    if args.context.is_some() {
        tracing::warn!(
            "ollama_context is ignored: llama-cpp completion does not preserve KV cache; \
             use the chat method for multi-turn conversations"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_args(prompt: &str, medias: Vec<MediaInput>) -> InferenceArgs {
        InferenceArgs {
            prompt: prompt.to_string(),
            sample_len: Some(128),
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            json_schema: None,
            medias,
        }
    }

    fn dummy_media() -> MediaInput {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;
        MediaInput {
            kind: MediaKind::Image as i32,
            source: Some(Source::Encoded(vec![0xFF, 0xD8, 0xFF])),
            id: None,
        }
    }

    #[test]
    fn with_n_batch_raises_batch_to_n_ctx() {
        // Regression for the `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`
        // process abort when a prompt exceeds n_batch.
        let p = LlamaContextParams::default().with_n_batch(262_144);
        assert_eq!(p.n_batch(), 262_144);
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
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;

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
            sample_len: Some(512),
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            json_schema: None,
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
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;

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
            sample_len: Some(1024),
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            json_schema: None,
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

    fn make_chat_msg(role: llm_chat_args::ChatRole, text: &str) -> llm_chat_args::ChatMessage {
        llm_chat_args::ChatMessage {
            role: role as i32,
            content: Some(llm_chat_args::MessageContent {
                content: Some(llm_chat_args::message_content::Content::Text(
                    text.to_string(),
                )),
            }),
        }
    }

    fn make_image_msg(base64_data: &str) -> llm_chat_args::ChatMessage {
        llm_chat_args::ChatMessage {
            role: llm_chat_args::ChatRole::User as i32,
            content: Some(llm_chat_args::MessageContent {
                content: Some(llm_chat_args::message_content::Content::Image(
                    llm_chat_args::message_content::Image {
                        content_type: "image/png".to_string(),
                        source: Some(llm_chat_args::message_content::ImageSource {
                            url: String::new(),
                            base64: base64_data.to_string(),
                        }),
                    },
                )),
            }),
        }
    }

    #[test]
    fn test_build_chat_messages_basic() {
        let msgs = vec![
            make_chat_msg(llm_chat_args::ChatRole::System, "You are helpful"),
            make_chat_msg(llm_chat_args::ChatRole::User, "Hello"),
            make_chat_msg(llm_chat_args::ChatRole::Assistant, "Hi there"),
        ];
        assert_eq!(msgs[0].role, llm_chat_args::ChatRole::System as i32);
        assert_eq!(msgs[1].role, llm_chat_args::ChatRole::User as i32);
        assert_eq!(msgs[2].role, llm_chat_args::ChatRole::Assistant as i32);
    }

    #[test]
    fn test_build_chat_messages_empty_rejected() {
        let msgs: Vec<llm_chat_args::ChatMessage> = vec![];
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_image_message_construction() {
        use base64::Engine;
        let raw_bytes = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw_bytes);
        let msg = make_image_msg(&b64);

        if let Some(content) = &msg.content {
            if let Some(llm_chat_args::message_content::Content::Image(img)) = &content.content {
                let source = img.source.as_ref().unwrap();
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(&source.base64)
                    .unwrap();
                assert_eq!(decoded, raw_bytes);
            } else {
                panic!("expected image content");
            }
        } else {
            panic!("expected content");
        }
    }

    #[test]
    fn test_tool_calls_message_construction() {
        let tc = llm_chat_args::message_content::ToolCalls {
            calls: vec![llm_chat_args::message_content::ToolCall {
                call_id: "call_1".to_string(),
                fn_name: "get_weather".to_string(),
                fn_arguments: r#"{"city":"Tokyo"}"#.to_string(),
            }],
        };
        let msg = llm_chat_args::ChatMessage {
            role: llm_chat_args::ChatRole::Assistant as i32,
            content: Some(llm_chat_args::MessageContent {
                content: Some(llm_chat_args::message_content::Content::ToolCalls(tc)),
            }),
        };
        if let Some(content) = &msg.content {
            if let Some(llm_chat_args::message_content::Content::ToolCalls(tc)) = &content.content {
                let json = serde_json::to_string(&tc.calls).unwrap();
                assert!(json.contains("get_weather"));
                assert!(json.contains("Tokyo"));
            }
        }
    }

    #[test]
    fn test_fallback_format_messages() {
        let raw = vec![
            ("system".to_string(), "Be helpful".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let result = LlamaModelWrapper::fallback_format_messages(&raw);
        assert!(result.contains("system: Be helpful"));
        assert!(result.contains("user: Hello"));
        assert!(result.contains("\n\n"));
        assert!(
            result.ends_with("assistant:"),
            "fallback must end with assistant generation prefix"
        );
    }

    fn make_completion_args(prompt: &str) -> LlmCompletionArgs {
        LlmCompletionArgs {
            model: None,
            system_prompt: None,
            prompt: prompt.to_string(),
            options: None,
            context: None,
            function_options: None,
            json_schema: None,
        }
    }

    #[test]
    fn test_validate_completion_rejects_empty_prompt() {
        let args = make_completion_args("");
        let err = validate_completion_args(&args).expect_err("empty prompt must be rejected");
        assert!(
            err.to_string().contains("prompt is empty"),
            "error message must mention 'prompt is empty': {err}"
        );
    }

    #[test]
    fn test_validate_completion_rejects_function_calling() {
        use jobworkerp_llama_protobuf::protobuf::llm::llm_completion_args;
        let mut args = make_completion_args("hi");
        args.function_options = Some(llm_completion_args::FunctionOptions {
            use_function_calling: true,
            ..Default::default()
        });
        let err =
            validate_completion_args(&args).expect_err("function_calling=true must be rejected");
        assert!(
            err.to_string().contains("function calling"),
            "error message must mention 'function calling': {err}"
        );
    }

    #[test]
    fn test_validate_completion_accepts_ollama_context() {
        use jobworkerp_llama_protobuf::protobuf::llm::llm_completion_args;
        let mut args = make_completion_args("hi");
        args.context = Some(llm_completion_args::GenerationContext {
            context: Some(
                llm_completion_args::generation_context::Context::OllamaContext(
                    llm_completion_args::OllamaContext {
                        data: vec![1, 2, 3],
                    },
                ),
            ),
        });
        assert!(validate_completion_args(&args).is_ok());
    }

    #[test]
    fn test_validate_completion_accepts_model_field() {
        let mut args = make_completion_args("hi");
        args.model = Some("override-model".to_string());
        assert!(validate_completion_args(&args).is_ok());
    }

    #[test]
    fn test_validate_completion_minimal_args_ok() {
        let args = make_completion_args("hello");
        assert!(validate_completion_args(&args).is_ok());
    }
}
