//! Configuration, argument, and process-shared-resource types for the `model`
//! module.
//!
//! This holds everything about *how a model is configured, loaded, and kept
//! resident* — split out of `model.rs` so the large `LlamaModelWrapper` impl
//! (initialization/templating in `wrapper`, the decode loop in `decode`) reads
//! against a clear data layer. Behavior is unchanged from the pre-split
//! single-file layout — this is a pure move; visibility is widened to
//! `pub(in crate::model)` only where `wrapper`/`decode` reference these items.

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::ApiBuilder;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings, MediaInput};
use jobworkerp_llama_protobuf::protobuf::llm::{
    LlmCompletionResult, llm_chat_args, llm_completion_args, llm_completion_result,
};
use llama_cpp_2::context::{LlamaContext, params::KvCacheType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel, params::kv_overrides::ParamOverrideValue};
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

// Error-message constants that callers (tests) match against. Centralising
// them here keeps the bail!() string in sync with assertions; otherwise a
// trivial wording change in this file silently breaks the test module.
pub(crate) const ERR_USE_FUNCTION_CALLING_UNSUPPORTED: &str =
    "function calling is not supported by this plugin";
pub(crate) const ERR_CLIENT_TOOLS_WITH_FUNCTION_CALLING: &str =
    "client_tools_json and use_function_calling are mutually exclusive";
pub(crate) const ERR_CLIENT_TOOLS_WITH_JSON_SCHEMA: &str =
    "json_schema and client_tools_json are mutually exclusive";
pub(crate) const ERR_CLIENT_TOOLS_WITH_MULTIMODAL: &str =
    "multimodal input combined with client_tools_json is not supported yet";
pub(crate) const ERR_TOOL_EXECUTION_REQUESTS_REJECTED: &str =
    "ToolExecutionRequests is no longer accepted; use ToolResults on a TOOL message";

/// for deserialization.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(in crate::model) enum ParamOverrideValueWrapper {
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

/// Map a proto `KvCacheType` enum value to a `llama_cpp_2::KvCacheType`.
/// Returns `None` for `UNSPECIFIED` (keep llama.cpp default F16) and for any
/// unrecognized value (so an out-of-range int silently keeps the default
/// rather than failing model load).
pub(in crate::model) fn proto_kv_cache_type(value: i32) -> Option<KvCacheType> {
    use jobworkerp_llama_protobuf::protobuf::llama_cpp::KvCacheType as Proto;
    let proto = Proto::try_from(value).ok()?;
    let mapped = match proto {
        Proto::Unspecified => return None,
        Proto::F32 => KvCacheType::F32,
        Proto::F16 => KvCacheType::F16,
        Proto::Q40 => KvCacheType::Q4_0,
        Proto::Q41 => KvCacheType::Q4_1,
        Proto::Q50 => KvCacheType::Q5_0,
        Proto::Q51 => KvCacheType::Q5_1,
        Proto::Q80 => KvCacheType::Q8_0,
        Proto::Q81 => KvCacheType::Q8_1,
        Proto::Q2K => KvCacheType::Q2_K,
        Proto::Q3K => KvCacheType::Q3_K,
        Proto::Q4K => KvCacheType::Q4_K,
        Proto::Q5K => KvCacheType::Q5_K,
        Proto::Q6K => KvCacheType::Q6_K,
        Proto::Q8K => KvCacheType::Q8_K,
        Proto::Iq2Xxs => KvCacheType::IQ2_XXS,
        Proto::Iq2Xs => KvCacheType::IQ2_XS,
        Proto::Iq3Xxs => KvCacheType::IQ3_XXS,
        Proto::Iq1S => KvCacheType::IQ1_S,
        Proto::Iq4Nl => KvCacheType::IQ4_NL,
        Proto::Iq3S => KvCacheType::IQ3_S,
        Proto::Iq2S => KvCacheType::IQ2_S,
        Proto::Iq4Xs => KvCacheType::IQ4_XS,
        Proto::I8 => KvCacheType::I8,
        Proto::I16 => KvCacheType::I16,
        Proto::I32 => KvCacheType::I32,
        Proto::I64 => KvCacheType::I64,
        Proto::F64 => KvCacheType::F64,
        Proto::Iq1M => KvCacheType::IQ1_M,
        Proto::Bf16 => KvCacheType::BF16,
        Proto::Tq10 => KvCacheType::TQ1_0,
        Proto::Tq20 => KvCacheType::TQ2_0,
        Proto::Mxfp4 => KvCacheType::MXFP4,
    };
    Some(mapped)
}

/// Normalize a batch-size setting, treating an explicit 0 as "unset".
/// These fields arrive as proto `optional uint32`, so a caller can send 0.
/// 0 is not a valid batch size: llama.cpp clamps n_batch to 0, after which
/// chunked prefill (`tokens_list.chunks(n_batch)`) panics and the multimodal
/// `eval_chunks` hits `GGML_ASSERT(n_batch > 0)`. Mirrors how ctx_size drops 0.
pub(in crate::model) fn normalize_batch_size(value: Option<u32>) -> Option<u32> {
    value.filter(|&v| v != 0)
}

/// Resolve the n_ubatch to apply. An explicit value always wins. Otherwise, on
/// GPU backends with large memory budgets (`gpu_default = true`), follow the
/// effective n_batch (capped at [`MAX_AUTO_N_UBATCH`]) so prompt eval runs in a
/// single large micro-batch; on other backends return `None` to keep llama.cpp's
/// memory-conservative default (512).
pub(in crate::model) fn resolve_n_ubatch(
    explicit_n_ubatch: Option<u32>,
    effective_n_batch: u32,
    gpu_default: bool,
) -> Option<u32> {
    explicit_n_ubatch.or_else(|| gpu_default.then(|| effective_n_batch.min(MAX_AUTO_N_UBATCH)))
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaModelConfig {
    /// The path or name to the model
    /// if hf_repo is Some, the model is filename on huggingface
    /// if hf_remo is None, the model is a local path
    pub(in crate::model) model: String,
    /// Download a model from huggingface (or use a cached version)
    /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
    pub(in crate::model) hf_repo: Option<String>,
    /// override some parameters of the model
    pub(in crate::model) key_value_overrides: Option<Vec<(String, ParamOverrideValueWrapper)>>,
    /// Disable offloading layers to the gpu
    pub(in crate::model) disable_gpu: bool,
    // RNG seed (default: 1234) TODO
    #[allow(unused)]
    pub(in crate::model) seed: Option<u32>,
    // number of threads to use during generation (default: use all available threads)
    pub(in crate::model) threads: Option<u32>,
    // number of threads to use during batch and prompt processing (default: use all available threads)
    pub(in crate::model) threads_batch: Option<u32>,
    // size of the prompt context (default: model size)
    pub(in crate::model) ctx_size: Option<NonZeroU32>,
    pub(in crate::model) n_batch: Option<u32>,
    pub(in crate::model) n_ubatch: Option<u32>,
    // Raw proto `KvCacheType` enum values; converted to `llama_cpp_2::KvCacheType`
    // via `proto_kv_cache_type` at context build time.
    pub(in crate::model) type_k: Option<i32>,
    pub(in crate::model) type_v: Option<i32>,
    // Reuse the KV cache across requests by keeping the longest common prompt
    // prefix. None lets each request decide; explicit values win over requests.
    #[serde(default)]
    pub(in crate::model) reuse_kv_prefix: Option<bool>,
    // use flash attention (default true)
    pub(in crate::model) use_flash_attention: Option<bool>,
    // system prompt before the user prompt
    // e.g. `The system will respond to your prompt`
    // This is useful for instructing the user on how to use the model
    // or to provide some context to the user
    pub(in crate::model) system_prompt: Option<String>,
    /// Multimodal projector settings. When None the runner is text-only.
    #[serde(default)]
    pub(in crate::model) mtmd: Option<jobworkerp_llama_protobuf::MtmdSettings>,
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
            n_batch: op.n_batch,
            n_ubatch: op.n_ubatch,
            type_k: op.type_k,
            type_v: op.type_v,
            reuse_kv_prefix: op.reuse_kv_prefix,
            use_flash_attention: op.use_flash_attention,
            system_prompt: op.system_prompt,
            mtmd: op.mtmd,
        }
    }
}

impl LlamaModelConfig {
    /// Convert the model to a path - may download from huggingface
    pub(in crate::model) fn get_or_load_model(&self) -> Result<Vec<PathBuf>> {
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
            n_batch: None,
            n_ubatch: None,
            type_k: None,
            type_v: None,
            reuse_kv_prefix: None,
            use_flash_attention: None,
            system_prompt: None,
            mtmd: None,
        }
    }
}

/// `LlamaChatMessage` has private fields, so we keep raw (role, content) pairs
/// alongside for fallback template formatting.
pub(in crate::model) type ChatBuildResult = (
    Vec<LlamaChatMessage>,
    Vec<(String, String)>,
    Vec<MediaInput>,
);

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct InferenceArgs {
    /// The prompt
    pub(in crate::model) prompt: String,
    /// Absolute position cap (prompt + output tokens). Used by legacy "run" path.
    /// None when max_new_tokens is used instead.
    pub(in crate::model) sample_len: Option<i32>,
    /// Max new tokens to generate (relative to prompt length). When set,
    /// overrides sample_len by computing prompt_tokens + max_new_tokens.
    #[serde(default)]
    pub(in crate::model) max_new_tokens: Option<i32>,
    /// The temperature used to generate samples.
    pub(in crate::model) temperature: Option<f64>,
    /// Nucleus sampling probability cutoff.
    pub(in crate::model) top_p: Option<f64>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub(in crate::model) repeat_penalty: Option<f32>,
    /// The context size to consider for the repeat penalty.
    pub(in crate::model) repeat_last_n: Option<u32>,
    /// RNG seed for sampling. None uses the default (1234).
    pub(in crate::model) seed: Option<u32>,
    /// Per-request KV prefix reuse override. Ignored when runner settings are explicit.
    #[serde(default)]
    pub(in crate::model) reuse_kv_prefix: Option<bool>,
    /// JSON Schema for structured output (llguidance constraint).
    #[serde(default, skip_serializing)]
    pub(in crate::model) json_schema: Option<String>,
    /// Media items attached to the prompt.
    #[serde(default, skip_serializing)]
    pub(in crate::model) medias: Vec<MediaInput>,
    /// Grammar specification emitted by the OAI chat template (tools path
    /// only). When `Some`, `build_sampler` prepends a grammar/grammar_lazy
    /// sampler ahead of the terminal selector. Mutually exclusive with
    /// `json_schema` — the run_chat entry point rejects the conflict before
    /// reaching the sampler.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub(in crate::model) grammar_spec: Option<crate::oai_chat::GrammarSpec>,
    /// Cooperative cancel flag installed on the underlying `LlamaContext`
    /// as a ggml abort callback. See `LlamaContext::set_abort_flag`.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub(in crate::model) cancel_flag: Option<Arc<AtomicBool>>,
}

impl InferenceArgs {
    /// `true` iff a host-side cancel flag was attached and is currently set.
    pub(in crate::model) fn is_cancel_requested(&self) -> bool {
        cancel_flag_requested(&self.cancel_flag)
    }
}

pub(in crate::model) fn cancel_flag_requested(cancel_flag: &Option<Arc<AtomicBool>>) -> bool {
    cancel_flag
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model) struct ResolvedReuse {
    pub(in crate::model) value: bool,
    pub(in crate::model) conflict: bool,
}

pub(in crate::model) fn resolve_reuse_kv_prefix(
    runner: Option<bool>,
    request: Option<bool>,
) -> ResolvedReuse {
    // Precedence: an explicit runner value wins, else the request decides,
    // else default off. Conflict only when both are set and disagree.
    let value = runner.or(request).unwrap_or(false);
    let conflict = matches!((runner, request), (Some(r), Some(q)) if r != q);
    ResolvedReuse { value, conflict }
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
            .field("reuse_kv_prefix", &self.reuse_kv_prefix)
            .field(
                "json_schema",
                &self.json_schema.as_deref().map(|s| &s[..s.len().min(80)]),
            )
            .field("medias", &format!("[{} items]", self.medias.len()))
            .finish()
    }
}

pub(in crate::model) fn chat_inference_args(
    options: llm_chat_args::LlmOptions,
    json_schema: Option<String>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> InferenceArgs {
    InferenceArgs {
        prompt: String::new(),
        sample_len: None,
        max_new_tokens: Some(options.max_tokens.unwrap_or(4096)),
        temperature: options.temperature.map(f64::from),
        top_p: options.top_p.map(f64::from),
        repeat_penalty: options.repeat_penalty,
        repeat_last_n: options.repeat_last_n.map(|v| v as u32),
        seed: options.seed.map(|s| s as u32),
        reuse_kv_prefix: options.reuse_kv_prefix,
        json_schema,
        medias: Vec::new(),
        grammar_spec: None,
        cancel_flag,
    }
}

pub(in crate::model) fn completion_inference_args(
    options: llm_completion_args::LlmOptions,
    json_schema: Option<String>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> InferenceArgs {
    InferenceArgs {
        prompt: String::new(),
        sample_len: None,
        max_new_tokens: Some(options.max_tokens.unwrap_or(4096)),
        temperature: options.temperature.map(f64::from),
        top_p: options.top_p.map(f64::from),
        repeat_penalty: options.repeat_penalty,
        repeat_last_n: options.repeat_last_n.map(|v| v as u32),
        seed: options.seed.map(|s| s as u32),
        reuse_kv_prefix: options.reuse_kv_prefix,
        json_schema,
        medias: Vec::new(),
        grammar_spec: None,
        cancel_flag,
    }
}

pub(in crate::model) fn cancelled_completion_result() -> LlmCompletionResult {
    LlmCompletionResult {
        content: Some(llm_completion_result::MessageContent {
            content: Some(llm_completion_result::message_content::Content::Text(
                String::new(),
            )),
        }),
        reasoning_content: None,
        done: true,
        context: None,
        usage: Some(llm_completion_result::Usage {
            model: String::new(),
            prompt_tokens: Some(0),
            completion_tokens: Some(0),
            total_prompt_time_sec: None,
            total_completion_time_sec: Some(0.0),
        }),
    }
}

pub(in crate::model) const DEFAULT_SAMPLER_SEED: u32 = 1234;

/// llguidance grammar tag selecting JSON-Schema-constrained decoding.
pub(in crate::model) const GRAMMAR_TYPE_JSON: &str = "json";

/// Upper bound for the auto-derived n_ubatch on Metal/ROCm. n_ubatch follows
/// the effective n_batch but is capped here: a very large n_ctx (e.g. 262k)
/// would otherwise blow up the per-ubatch compute buffer, and llama.cpp's own
/// guidance is that prompt-eval throughput collapses past a few thousand. 2048
/// matches Apple's recommended `-ub 2048` for large-prompt processing.
pub(in crate::model) const MAX_AUTO_N_UBATCH: u32 = 2048;

/// A reusable `LlamaContext` held across requests so the (expensive) KV-cache
/// allocation happens once per process instead of per request.
///
/// `LlamaContext` is `!Send + !Sync` because it wraps a raw pointer. We assert
/// both here because jobworkerp guarantees a single execution at a time per
/// runner instance (use-and-discard, or runner-pool serialization when
/// `use_static=true`), and every plugin entry point takes `&mut self`. The
/// context is therefore never accessed from two threads at once. This mirrors
/// the rationale behind `llama-cpp-2`'s own `unsafe impl Send + Sync for
/// LlamaModel`.
/// A chunk recorded in the KV cache, for multimodal prefix reuse. Text chunks
/// compare by their token list; media (image/audio) chunks compare by an
/// identity key (hash of the decoded bitmap) plus their KV position count.
///
/// `key` is `None` when the chunk couldn't be paired with a bitmap (tiling
/// models split one bitmap into several chunks, so chunks can outnumber
/// bitmaps). Such chunks must never be treated as a reusable prefix match — see
/// `plan_chunk_keep`, which stops at the first `None`-keyed media chunk.
#[derive(Clone, PartialEq)]
pub(in crate::model) enum CachedChunk {
    Text(Vec<LlamaToken>),
    Media { key: Option<u64>, n_pos: i32 },
}

pub(in crate::model) struct SyncContext {
    ctx: LlamaContext<'static>,
    /// Tokens whose KV is currently present in `ctx` (the previous text
    /// request's prompt). Used for text prefix reuse. Empty when the KV cache is
    /// empty or its contents are unknown (e.g. after an error). Bound to the
    /// context so a rebuilt context always starts with an empty record.
    pub(in crate::model) cached_tokens: Vec<LlamaToken>,
    /// Chunk sequence whose KV is present in `ctx` (the previous multimodal
    /// request's prompt). Used for multimodal prefix reuse. Empty when unknown.
    /// Kept disjoint from `cached_tokens`: a text request clears this and vice
    /// versa, so the record always describes whichever path last filled the KV.
    pub(in crate::model) cached_chunks: Vec<CachedChunk>,
}

// SAFETY: see the type-level comment — exclusive `&mut self` access plus
// jobworkerp's one-execution-at-a-time guarantee preclude concurrent use.
unsafe impl Send for SyncContext {}
unsafe impl Sync for SyncContext {}

impl SyncContext {
    pub(in crate::model) fn new(ctx: LlamaContext<'static>) -> Self {
        Self {
            ctx,
            cached_tokens: Vec::new(),
            cached_chunks: Vec::new(),
        }
    }

    pub(in crate::model) fn ctx_mut(&mut self) -> &mut LlamaContext<'static> {
        &mut self.ctx
    }

    /// Take the text prefix-reuse record (leaving it empty) and forget the
    /// multimodal record. Only one path's record can describe the current KV, so
    /// entering the text path always invalidates the multimodal one — otherwise a
    /// later multimodal request could reuse a prefix this request's KV no longer
    /// holds. The returned value is written back only on success.
    pub(in crate::model) fn take_text_cache(&mut self) -> Vec<LlamaToken> {
        self.cached_chunks.clear();
        std::mem::take(&mut self.cached_tokens)
    }

    /// Multimodal counterpart of [`Self::take_text_cache`].
    pub(in crate::model) fn take_chunk_cache(&mut self) -> Vec<CachedChunk> {
        self.cached_tokens.clear();
        std::mem::take(&mut self.cached_chunks)
    }
}

/// Moves a value out of an `&mut Option<T>` and guarantees it is put back when
/// the guard drops — including on `?`/early-return and panics. Used so the
/// reusable context is never lost if a decode fails partway through.
pub(in crate::model) struct RestoreOnDrop<'a, T> {
    slot: &'a mut Option<T>,
    value: Option<T>,
}

impl<'a, T> RestoreOnDrop<'a, T> {
    pub(in crate::model) fn new(slot: &'a mut Option<T>) -> Self {
        let value = slot.take();
        Self { slot, value }
    }
}

impl<T> std::ops::Deref for RestoreOnDrop<'_, T> {
    type Target = Option<T>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> std::ops::DerefMut for RestoreOnDrop<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> Drop for RestoreOnDrop<'_, T> {
    fn drop(&mut self) {
        *self.slot = self.value.take();
    }
}

/// Result of a single decode pass, carrying both the generated text and the
/// token counts needed to populate `Usage` in chat/completion responses.
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// RAII guard: installs an abort flag on the wrapped `LlamaContext` on
/// construction, clears it on drop. Holds a raw pointer rather than `&mut`
/// so the surrounding code can keep its own `&mut LlamaContext` borrow
/// active — the guard does not touch `ctx` between construction and drop.
/// A `None` flag is a no-op.
///
/// SAFETY invariants for callers:
/// - The `LlamaContext` referenced by `ctx_ptr` must outlive this guard.
/// - No other code may call `set_abort_flag` / `clear_abort_callback` on
///   that context while the guard is alive.
pub(in crate::model) struct AbortGuard {
    ctx_ptr: *mut llama_cpp_2::context::LlamaContext<'static>,
    armed: bool,
}

impl AbortGuard {
    pub(in crate::model) fn new(
        ctx: &mut llama_cpp_2::context::LlamaContext<'static>,
        flag: Option<Arc<AtomicBool>>,
    ) -> Self {
        let armed = flag.is_some();
        if armed && let Some(flag) = flag {
            ctx.set_abort_flag(flag);
        }
        Self {
            ctx_ptr: ctx as *mut _,
            armed,
        }
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        if self.armed {
            // SAFETY: see struct-level invariants. The pointer is valid
            // because the caller's `&mut LlamaContext` outlives this guard,
            // and no other concurrent access is permitted while we hold it.
            unsafe { (*self.ctx_ptr).clear_abort_callback() };
        }
    }
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
            reuse_kv_prefix: req.reuse_kv_prefix,
            json_schema: None,
            medias: req.medias,
            grammar_spec: None,
            cancel_flag: None,
        }
    }
}

/// `LlamaBackend::init()` may only succeed once per process (it guards a global
/// atomic and errors on the second call). The plugin can build a new
/// `LlamaModelWrapper` on every `load`, so the backend is shared process-wide
/// rather than owned per wrapper. `LlamaBackend` is `Send + Sync` and only ever
/// borrowed (`new_context`, `load_from_file`), so a `&'static` is sufficient.
///
/// `get_or_init` runs the closure exactly once even under races, so `init()` is
/// never called twice and the stored backend is never dropped (which would reset
/// the global atomic). A genuine init failure is fatal to the plugin, so it
/// panics here and is caught by `catch_unwind` in the plugin loader.
pub(in crate::model) fn shared_backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init().expect("failed to init llama backend");
        backend.void_logs();
        backend
    })
}

/// Build a string capturing every config field that affects how the model is
/// loaded (resolved paths, GPU offload, KV overrides) — i.e. the inputs to
/// `LlamaModel::load_from_file`. Two configs that load an identical model
/// produce the same string. Context-side settings (ctx_size/n_batch/mtmd) are
/// excluded: they don't affect the model, and mmproj is rebuilt per wrapper.
pub(in crate::model) fn model_load_identity(
    model_paths: &[PathBuf],
    config: &LlamaModelConfig,
) -> String {
    format!(
        "paths={:?};disable_gpu={};kv_overrides={:?}",
        model_paths, config.disable_gpu, config.key_value_overrides
    )
}

/// The model is stored process-wide so a reused `LlamaContext` can borrow it for
/// `'static` (a `LlamaContext<'a>` borrows `&'a LlamaModel`). Like the backend,
/// it is loaded once and never dropped, which matches the deployment model
/// (one model loaded at startup) and avoids leaking a model on every reload.
///
/// The first load wins; a later load with a **different** `identity` (see
/// `model_load_identity`) is rejected rather than silently serving the original
/// model under settings it wasn't loaded with. A reload with the same identity
/// reuses the stored model.
pub(in crate::model) fn shared_model(
    identity: &str,
    build: impl FnOnce() -> Result<LlamaModel>,
) -> Result<&'static LlamaModel> {
    static MODEL: OnceLock<(LlamaModel, String)> = OnceLock::new();

    if MODEL.get().is_none() {
        // Build is fallible, so it can't run inside `get_or_init`.
        let model = build()?;
        // A racing thread may have stored first; ignore the loser's `set` — the
        // identity check below validates every caller against whatever won.
        let _ = MODEL.set((model, identity.to_string()));
    }

    let (stored_model, stored_identity) = MODEL.get().expect("model stored above");
    if stored_identity != identity {
        bail!(
            "a different model is already loaded in this process \
             (loaded: {stored_identity}, requested: {identity}). \
             This plugin supports a single model per process."
        );
    }
    Ok(stored_model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_support::*;
    use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings};
    use jobworkerp_llama_protobuf::protobuf::llm::{llm_chat_args, llm_completion_args};
    use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
    use std::sync::atomic::Ordering;

    #[test]
    fn with_n_batch_raises_batch_to_n_ctx() {
        // Regression for the `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`
        // process abort when a prompt exceeds n_batch.
        let p = LlamaContextParams::default().with_n_batch(262_144);
        assert_eq!(p.n_batch(), 262_144);
    }

    #[test]
    fn effective_n_ctx_falls_back_to_trained_context() {
        // The n_batch sizing must cover the prompt even when ctx_size is unset:
        // an explicit ctx_size wins, otherwise the model's trained context is
        // used (the same value llama.cpp adopts for with_n_ctx(None)). Without
        // the fallback, n_batch stays at the 2048 default and a longer prompt
        // aborts the process. n_ctx_train() is stubbed here since loading a
        // model is not possible in a unit test.
        let pick = |ctx_size: Option<NonZeroU32>, n_ctx_train: u32| {
            ctx_size.map_or(n_ctx_train, NonZeroU32::get)
        };
        assert_eq!(pick(NonZeroU32::new(8192), 4096), 8192);
        assert_eq!(pick(None, 32_768), 32_768);
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
            reuse_kv_prefix: Some(true),
        };
        let args: InferenceArgs = arg.into();
        assert!(args.prompt.is_empty());
        assert_eq!(args.medias.len(), 2);
        assert_eq!(args.reuse_kv_prefix, Some(true));
    }

    #[test]
    fn test_resolve_reuse_kv_prefix_precedence() {
        let cases = [
            (Some(true), Some(false), true, true),
            (Some(false), Some(true), false, true),
            (Some(true), Some(true), true, false),
            (Some(false), Some(false), false, false),
            (Some(true), None, true, false),
            (None, Some(true), true, false),
            (None, Some(false), false, false),
            (None, None, false, false),
        ];

        for (settings, request, value, conflict) in cases {
            let resolved = resolve_reuse_kv_prefix(settings, request);
            assert_eq!(
                (resolved.value, resolved.conflict),
                (value, conflict),
                "settings={settings:?}, request={request:?}"
            );
        }
    }

    #[test]
    fn test_chat_completion_and_legacy_args_propagate_reuse_kv_prefix() {
        for value in [Some(true), Some(false), None] {
            let chat = chat_inference_args(
                llm_chat_args::LlmOptions {
                    reuse_kv_prefix: value,
                    ..Default::default()
                },
                None,
                None,
            );
            assert_eq!(chat.reuse_kv_prefix, value);

            let completion = completion_inference_args(
                llm_completion_args::LlmOptions {
                    reuse_kv_prefix: value,
                    ..Default::default()
                },
                None,
                None,
            );
            assert_eq!(completion.reuse_kv_prefix, value);

            let legacy: InferenceArgs = LlamaArg {
                prompt: String::new(),
                sample_len: 100,
                temperature: None,
                top_p: None,
                repeat_penalty: None,
                repeat_last_n: None,
                seed: None,
                need_print: false,
                medias: vec![],
                reuse_kv_prefix: value,
            }
            .into();
            assert_eq!(legacy.reuse_kv_prefix, value);
        }
    }

    #[test]
    fn test_chat_inference_args_preserve_cancel_flag() {
        let cancel = Arc::new(AtomicBool::new(false));
        let args = chat_inference_args(
            llm_chat_args::LlmOptions {
                max_tokens: Some(32),
                ..Default::default()
            },
            None,
            Some(cancel.clone()),
        );

        cancel.store(true, Ordering::Relaxed);
        assert!(
            args.is_cancel_requested(),
            "chat cancellation must remain visible to the decode layer"
        );
    }

    #[test]
    fn test_completion_inference_args_preserve_cancel_flag() {
        let cancel = Arc::new(AtomicBool::new(false));
        let args = completion_inference_args(
            llm_completion_args::LlmOptions {
                max_tokens: Some(32),
                ..Default::default()
            },
            None,
            Some(cancel.clone()),
        );

        cancel.store(true, Ordering::Relaxed);
        assert!(
            args.is_cancel_requested(),
            "completion cancellation must remain visible to the decode layer"
        );
    }

    #[test]
    fn test_cancel_flag_requested_reflects_token_bridge_flag() {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Some(cancel.clone());

        assert!(
            !cancel_flag_requested(&flag),
            "fresh token bridge flag must start as not cancelled"
        );
        cancel.store(true, Ordering::Relaxed);
        assert!(
            cancel_flag_requested(&flag),
            "set_cancellation_token bridge flag must become visible to model input/decode"
        );
    }

    #[test]
    fn test_cancelled_completion_result_is_terminal_empty_output() {
        let result = cancelled_completion_result();

        assert!(result.done, "cancelled completion result must be terminal");
        assert_eq!(
            result
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens),
            Some(0),
            "cancelled completion must not report generated tokens"
        );
        match result.content.and_then(|content| content.content) {
            Some(llm_completion_result::message_content::Content::Text(text)) => {
                assert!(text.is_empty(), "cancelled completion text must be empty");
            }
            other => panic!("cancelled completion must be empty text, got {other:?}"),
        }
    }

    #[test]
    fn test_runner_settings_maps_batch_fields_to_config() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::KvCacheType as ProtoKv;
        let settings = LlamaRunnerSettings {
            model: "m.gguf".to_string(),
            hf_repo: None,
            disable_gpu: false,
            seed: None,
            threads: None,
            threads_batch: None,
            ctx_size: None,
            n_batch: Some(2048),
            n_ubatch: Some(512),
            type_k: Some(ProtoKv::Q80 as i32),
            type_v: Some(ProtoKv::Q80 as i32),
            reuse_kv_prefix: Some(true),
            use_flash_attention: None,
            system_prompt: None,
            mtmd: None,
        };
        let config: LlamaModelConfig = settings.clone().into();
        assert_eq!(config.n_batch, Some(2048));
        assert_eq!(config.n_ubatch, Some(512));
        assert_eq!(config.type_k, Some(ProtoKv::Q80 as i32));
        assert_eq!(config.type_v, Some(ProtoKv::Q80 as i32));
        assert_eq!(config.reuse_kv_prefix, Some(true));
        // Default leaves reuse unspecified so requests can opt in.
        assert_eq!(LlamaModelConfig::default().reuse_kv_prefix, None);

        let omitted: LlamaModelConfig = LlamaRunnerSettings {
            reuse_kv_prefix: None,
            ..settings.clone()
        }
        .into();
        assert_eq!(omitted.reuse_kv_prefix, None);

        let explicit_false: LlamaModelConfig = LlamaRunnerSettings {
            reuse_kv_prefix: Some(false),
            ..settings
        }
        .into();
        assert_eq!(explicit_false.reuse_kv_prefix, Some(false));
    }

    #[test]
    fn test_proto_kv_cache_type_conversion() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::KvCacheType as ProtoKv;
        // UNSPECIFIED and out-of-range values map to None so the llama.cpp
        // default (F16) is kept rather than failing model load.
        assert_eq!(proto_kv_cache_type(ProtoKv::Unspecified as i32), None);
        assert_eq!(proto_kv_cache_type(9999), None);
        assert_eq!(
            proto_kv_cache_type(ProtoKv::Q80 as i32),
            Some(KvCacheType::Q8_0)
        );
        assert_eq!(
            proto_kv_cache_type(ProtoKv::F16 as i32),
            Some(KvCacheType::F16)
        );
        assert_eq!(
            proto_kv_cache_type(ProtoKv::Q40 as i32),
            Some(KvCacheType::Q4_0)
        );
    }

    #[test]
    fn test_context_params_reflect_kv_cache_type() {
        // Mirror the builder chain in `LlamaModelWrapper::new`.
        let params = LlamaContextParams::default()
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
        assert_eq!(params.type_k(), KvCacheType::Q8_0);
        assert_eq!(params.type_v(), KvCacheType::Q8_0);

        let default = LlamaContextParams::default();
        assert_eq!(default.type_k(), KvCacheType::F16);
        assert_eq!(default.type_v(), KvCacheType::F16);
    }

    #[test]
    fn test_context_params_reflect_batch_settings() {
        // Mirror the builder chain in `LlamaModelWrapper::new`: only apply
        // n_batch/n_ubatch when present, otherwise keep the crate default.
        let with_batch = LlamaContextParams::default()
            .with_n_batch(2048)
            .with_n_ubatch(512);
        assert_eq!(with_batch.n_batch(), 2048);
        assert_eq!(with_batch.n_ubatch(), 512);

        // Crate defaults: n_batch 2048, n_ubatch 512 — the plugin already
        // benefits from these without any explicit tuning.
        let default = LlamaContextParams::default();
        assert_eq!(default.n_batch(), 2048);
        assert_eq!(default.n_ubatch(), 512);
    }

    #[test]
    fn test_normalize_batch_size_treats_zero_as_unset() {
        // An explicit 0 must become None so it never reaches with_n_batch /
        // chunked prefill, which would clamp to 0 and then panic.
        assert_eq!(normalize_batch_size(Some(0)), None);
        assert_eq!(normalize_batch_size(None), None);
        assert_eq!(normalize_batch_size(Some(2048)), Some(2048));
    }

    #[test]
    fn test_resolve_n_ubatch() {
        // Explicit n_ubatch always wins, regardless of backend.
        assert_eq!(resolve_n_ubatch(Some(1024), 8192, true), Some(1024));
        assert_eq!(resolve_n_ubatch(Some(1024), 8192, false), Some(1024));

        // Non-GPU backends keep the llama.cpp default (None → 512 downstream).
        assert_eq!(resolve_n_ubatch(None, 8192, false), None);

        // GPU backends follow the effective n_batch when it is small...
        assert_eq!(resolve_n_ubatch(None, 1024, true), Some(1024));
        // ...and cap at MAX_AUTO_N_UBATCH for a large n_ctx (e.g. 262k), so the
        // compute buffer stays bounded.
        assert_eq!(
            resolve_n_ubatch(None, 262_144, true),
            Some(MAX_AUTO_N_UBATCH)
        );
    }

    #[test]
    fn test_restore_on_drop_returns_value_on_normal_scope_exit() {
        let mut slot = Some(42);
        {
            let mut guard = RestoreOnDrop::new(&mut slot);
            assert_eq!(*guard, Some(42), "value is moved into the guard");
            *guard = Some(7); // mutate while held
        }
        assert_eq!(slot, Some(7), "guard writes its value back on drop");
    }

    #[test]
    fn test_restore_on_drop_returns_value_on_early_return() {
        // Simulate an early `?`-style return while the guard is held: the value
        // must still be restored to the slot (this is why a failed decode does
        // not lose the reusable context).
        let mut slot = Some(String::from("ctx"));
        fn body(slot: &mut Option<String>) -> Result<()> {
            let _guard = RestoreOnDrop::new(slot);
            bail!("simulated mid-request failure");
        }
        let err = body(&mut slot);
        assert!(err.is_err());
        assert_eq!(slot.as_deref(), Some("ctx"), "value survives early return");
    }

    #[test]
    fn test_model_load_identity_distinguishes_load_settings() {
        let paths = vec![PathBuf::from("model.gguf")];
        let base = LlamaModelConfig::default();
        let id_base = model_load_identity(&paths, &base);

        // Same load-affecting settings → same identity (reload is allowed).
        assert_eq!(id_base, model_load_identity(&paths, &base));

        // A different model path → different identity (load is rejected).
        let other_paths = vec![PathBuf::from("other.gguf")];
        assert_ne!(id_base, model_load_identity(&other_paths, &base));

        // disable_gpu changes how the model loads → different identity.
        let gpu_off = LlamaModelConfig {
            disable_gpu: true,
            ..LlamaModelConfig::default()
        };
        assert_ne!(id_base, model_load_identity(&paths, &gpu_off));

        // ctx_size does NOT affect model loading → identity unchanged.
        let bigger_ctx = LlamaModelConfig {
            ctx_size: std::num::NonZeroU32::new(8192),
            ..LlamaModelConfig::default()
        };
        assert_eq!(id_base, model_load_identity(&paths, &bigger_ctx));
    }
}
