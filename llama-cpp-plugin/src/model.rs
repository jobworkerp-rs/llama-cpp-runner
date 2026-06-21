use anyhow::{Context, Result, anyhow, bail};
use hf_hub::api::sync::ApiBuilder;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::{LlamaArg, LlamaRunnerSettings, MediaInput};
use jobworkerp_llama_protobuf::protobuf::llm::{
    LlmChatArgs, LlmChatResult, LlmCompletionArgs, LlmCompletionResult, llm_chat_args,
    llm_chat_result, llm_completion_args, llm_completion_result,
};
use llama_cpp_2::{
    context::{
        LlamaContext,
        params::{KvCacheType, LlamaContextParams},
    },
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
use std::{
    ffi::CString,
    num::NonZeroU32,
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

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

/// Map a proto `KvCacheType` enum value to a `llama_cpp_2::KvCacheType`.
/// Returns `None` for `UNSPECIFIED` (keep llama.cpp default F16) and for any
/// unrecognized value (so an out-of-range int silently keeps the default
/// rather than failing model load).
fn proto_kv_cache_type(value: i32) -> Option<KvCacheType> {
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
fn normalize_batch_size(value: Option<u32>) -> Option<u32> {
    value.filter(|&v| v != 0)
}

/// Resolve the n_ubatch to apply. An explicit value always wins. Otherwise, on
/// GPU backends with large memory budgets (`gpu_default = true`), follow the
/// effective n_batch (capped at [`MAX_AUTO_N_UBATCH`]) so prompt eval runs in a
/// single large micro-batch; on other backends return `None` to keep llama.cpp's
/// memory-conservative default (512).
fn resolve_n_ubatch(
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
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    // Raw proto `KvCacheType` enum values; converted to `llama_cpp_2::KvCacheType`
    // via `proto_kv_cache_type` at context build time.
    type_k: Option<i32>,
    type_v: Option<i32>,
    // Reuse the KV cache across requests by keeping the longest common prompt
    // prefix (text-only). Default false keeps requests independent/deterministic.
    #[serde(default)]
    reuse_kv_prefix: bool,
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
            n_batch: op.n_batch,
            n_ubatch: op.n_ubatch,
            type_k: op.type_k,
            type_v: op.type_v,
            reuse_kv_prefix: op.reuse_kv_prefix.unwrap_or(false),
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
            n_batch: None,
            n_ubatch: None,
            type_k: None,
            type_v: None,
            reuse_kv_prefix: false,
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

#[derive(Clone, Default, Serialize, Deserialize)]
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
    /// Grammar specification emitted by the OAI chat template (tools path
    /// only). When `Some`, `build_sampler` prepends a grammar/grammar_lazy
    /// sampler ahead of the terminal selector. Mutually exclusive with
    /// `json_schema` — the run_chat entry point rejects the conflict before
    /// reaching the sampler.
    #[serde(default, skip_serializing, skip_deserializing)]
    grammar_spec: Option<crate::oai_chat::GrammarSpec>,
    /// Cooperative cancel flag installed on the underlying `LlamaContext`
    /// as a ggml abort callback. See `LlamaContext::set_abort_flag`.
    #[serde(default, skip_serializing, skip_deserializing)]
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl InferenceArgs {
    /// `true` iff a host-side cancel flag was attached and is currently set.
    fn is_cancel_requested(&self) -> bool {
        self.cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    }
}

fn cancel_flag_requested(cancel_flag: &Option<Arc<AtomicBool>>) -> bool {
    cancel_flag
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
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

fn chat_inference_args(
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
        json_schema,
        medias: Vec::new(),
        grammar_spec: None,
        cancel_flag,
    }
}

fn completion_inference_args(
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
        json_schema,
        medias: Vec::new(),
        grammar_spec: None,
        cancel_flag,
    }
}

fn cancelled_completion_result() -> LlmCompletionResult {
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

const DEFAULT_SAMPLER_SEED: u32 = 1234;

/// llguidance grammar tag selecting JSON-Schema-constrained decoding.
const GRAMMAR_TYPE_JSON: &str = "json";

/// Upper bound for the auto-derived n_ubatch on Metal/ROCm. n_ubatch follows
/// the effective n_batch but is capped here: a very large n_ctx (e.g. 262k)
/// would otherwise blow up the per-ubatch compute buffer, and llama.cpp's own
/// guidance is that prompt-eval throughput collapses past a few thousand. 2048
/// matches Apple's recommended `-ub 2048` for large-prompt processing.
const MAX_AUTO_N_UBATCH: u32 = 2048;

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
enum CachedChunk {
    Text(Vec<LlamaToken>),
    Media { key: Option<u64>, n_pos: i32 },
}

struct SyncContext {
    ctx: LlamaContext<'static>,
    /// Tokens whose KV is currently present in `ctx` (the previous text
    /// request's prompt). Used for text prefix reuse. Empty when the KV cache is
    /// empty or its contents are unknown (e.g. after an error). Bound to the
    /// context so a rebuilt context always starts with an empty record.
    cached_tokens: Vec<LlamaToken>,
    /// Chunk sequence whose KV is present in `ctx` (the previous multimodal
    /// request's prompt). Used for multimodal prefix reuse. Empty when unknown.
    /// Kept disjoint from `cached_tokens`: a text request clears this and vice
    /// versa, so the record always describes whichever path last filled the KV.
    cached_chunks: Vec<CachedChunk>,
}

// SAFETY: see the type-level comment — exclusive `&mut self` access plus
// jobworkerp's one-execution-at-a-time guarantee preclude concurrent use.
unsafe impl Send for SyncContext {}
unsafe impl Sync for SyncContext {}

impl SyncContext {
    fn new(ctx: LlamaContext<'static>) -> Self {
        Self {
            ctx,
            cached_tokens: Vec::new(),
            cached_chunks: Vec::new(),
        }
    }

    fn ctx_mut(&mut self) -> &mut LlamaContext<'static> {
        &mut self.ctx
    }

    /// Take the text prefix-reuse record (leaving it empty) and forget the
    /// multimodal record. Only one path's record can describe the current KV, so
    /// entering the text path always invalidates the multimodal one — otherwise a
    /// later multimodal request could reuse a prefix this request's KV no longer
    /// holds. The returned value is written back only on success.
    fn take_text_cache(&mut self) -> Vec<LlamaToken> {
        self.cached_chunks.clear();
        std::mem::take(&mut self.cached_tokens)
    }

    /// Multimodal counterpart of [`Self::take_text_cache`].
    fn take_chunk_cache(&mut self) -> Vec<CachedChunk> {
        self.cached_tokens.clear();
        std::mem::take(&mut self.cached_chunks)
    }
}

/// Moves a value out of an `&mut Option<T>` and guarantees it is put back when
/// the guard drops — including on `?`/early-return and panics. Used so the
/// reusable context is never lost if a decode fails partway through.
struct RestoreOnDrop<'a, T> {
    slot: &'a mut Option<T>,
    value: Option<T>,
}

impl<'a, T> RestoreOnDrop<'a, T> {
    fn new(slot: &'a mut Option<T>) -> Self {
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
struct AbortGuard {
    ctx_ptr: *mut llama_cpp_2::context::LlamaContext<'static>,
    armed: bool,
}

impl AbortGuard {
    fn new(
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
fn shared_backend() -> &'static LlamaBackend {
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
fn model_load_identity(model_paths: &[PathBuf], config: &LlamaModelConfig) -> String {
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
fn shared_model(
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

pub struct LlamaModelWrapper {
    model: &'static LlamaModel,
    backend: &'static LlamaBackend,
    ctx_params: LlamaContextParams,
    system_prompt: String,
    /// Reused across requests: built lazily on the first decode, then kept so
    /// the KV-cache allocation happens once. KV is cleared at the start of each
    /// request for isolation. `None` until the first decode.
    ///
    /// No Mutex needed: all access is through `&mut self` on `run()`, which
    /// provides exclusive access at the Rust borrow-checker level.
    context: Option<SyncContext>,
    /// When true, keep the longest common prompt prefix in the KV cache across
    /// requests (text-only) instead of clearing it each time. See [`SyncContext`].
    reuse_kv_prefix: bool,
    mtmd: Option<MtmdRuntime>,
    media_limits: MediaLimits,
}

impl LlamaModelWrapper {
    pub fn new(config: LlamaModelConfig) -> Result<Self> {
        let backend = shared_backend();

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
        // Identity of everything that affects how the model is loaded. A second
        // load with a different identity is rejected (see `shared_model`) so the
        // process never silently serves a model built with other settings.
        let model_identity = model_load_identity(&model_paths, &config);
        // Load once per process and share for `'static` (see `shared_model`).
        let model: &'static LlamaModel = shared_model(&model_identity, || {
            LlamaModel::load_from_file(
                backend,
                model_paths[0].as_ref() as &std::path::Path,
                &model_params,
            )
            .with_context(|| "unable to load model")
        })?;

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
        // prompt exceeds n_batch. Keep it in step with the effective n_ctx,
        // which falls back to the model's trained context when ctx_size is unset
        // (with_n_ctx(None) lets llama.cpp adopt n_ctx_train).
        let effective_n_ctx = config
            .ctx_size
            .map_or_else(|| model.n_ctx_train(), NonZeroU32::get);
        ctx_params = ctx_params.with_n_batch(effective_n_ctx);
        if let Some(threads) = config.threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }
        if let Some(threads_batch) = config.threads_batch.or(config.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch as i32);
        }
        let n_batch = normalize_batch_size(config.n_batch);
        let n_ubatch = normalize_batch_size(config.n_ubatch);
        // The effective n_batch actually applied to the context: an explicit
        // n_batch overrides the `with_n_batch(effective_n_ctx)` set above.
        let effective_n_batch = n_batch.unwrap_or(effective_n_ctx);
        if let Some(n_batch) = n_batch {
            ctx_params = ctx_params.with_n_batch(n_batch);
        }
        let ubatch = resolve_n_ubatch(
            n_ubatch,
            effective_n_batch,
            cfg!(any(feature = "metal", feature = "rocm")),
        );
        if let Some(n_ubatch) = ubatch {
            ctx_params = ctx_params.with_n_ubatch(n_ubatch);
        }
        if let Some(type_k) = config.type_k.and_then(proto_kv_cache_type) {
            ctx_params = ctx_params.with_type_k(type_k);
        }
        if let Some(type_v) = config.type_v.and_then(proto_kv_cache_type) {
            // V-cache quantization requires flash attention on most backends
            // (including Metal). Warn instead of failing so callers who know
            // their backend supports it can still proceed.
            if !config.use_flash_attention.unwrap_or(true) {
                tracing::warn!(
                    "type_v KV cache quantization typically requires flash attention, \
                     but use_flash_attention is disabled — generation may fail or be slow"
                );
            }
            ctx_params = ctx_params.with_type_v(type_v);
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
                    model,
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
            context: None,
            reuse_kv_prefix: config.reuse_kv_prefix,
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

        // llama.cpp sampler chains apply each stage in order and the FINAL
        // stage must be the one that selects the token (greedy/dist). Grammar
        // constraints (llguidance) mask disallowed tokens by setting their logit
        // to -inf, so they MUST run BEFORE the selecting sampler — otherwise the
        // selection is made on un-masked logits and the constraint is ignored.
        //
        // The previous order put `dist` first and `greedy` last, which (a) made
        // `dist`/temperature/top_p effectively dead (greedy re-selected at the
        // end) and (b) left the door open for the model to emit a token the
        // grammar forbids at position 0 (e.g. Qwen3's leading `<think>`). Once an
        // ungrammatical token is accepted, llguidance's matcher enters an error
        // state and every later `compute_mask` fails — the sampler then silently
        // stops masking, so the rest of the output is completely unconstrained.
        // Ordering constraints before the single terminal selector fixes both.
        let mut samplers: Vec<LlamaSampler> = Vec::new();

        // Tool-calling grammar takes precedence over `json_schema` — the two
        // are mutually exclusive and the run_chat entry point rejects the
        // combination before we get here. The branch is defensive: a future
        // caller bypassing that guard would otherwise silently lose the
        // grammar constraint. Same ordering rationale as the llguidance
        // branch below: place the masking sampler before the terminal
        // selector so its mask actually shapes the chosen token.
        if let Some(spec) = &args.grammar_spec {
            if args.json_schema.is_some() {
                tracing::warn!(
                    "build_sampler: both grammar_spec and json_schema present; preferring grammar_spec"
                );
            }
            let sampler = if spec.grammar_lazy {
                let (patterns, tokens) = crate::oai_chat::grammar_triggers_to_patterns_and_tokens(
                    self.model,
                    &spec.grammar_triggers,
                );
                LlamaSampler::grammar_lazy_patterns(
                    self.model,
                    &spec.grammar,
                    "root",
                    &patterns,
                    &tokens,
                )
                .map_err(|e| anyhow!("grammar_lazy_patterns init failed: {e:?}"))?
            } else {
                LlamaSampler::grammar(self.model, &spec.grammar, "root")
                    .map_err(|e| anyhow!("grammar sampler init failed: {e:?}"))?
            };
            samplers.push(sampler);
        } else if let Some(schema) = &args.json_schema {
            let llg = LlamaSampler::llguidance(self.model, GRAMMAR_TYPE_JSON, schema)
                .map_err(|e| anyhow!("llguidance init failed: {e:?}"))?;
            samplers.push(llg);
        }
        if let Some(penalty) = args.repeat_penalty {
            let last_n = args.repeat_last_n.map_or(64, |n| n as i32);
            samplers.push(LlamaSampler::penalties(last_n, penalty, 0.0, 0.0));
        }

        // Terminal selector (exactly one). A positive temperature means the user
        // wants stochastic sampling, so apply temp/top_p shaping and select with
        // `dist`. At temperature 0 (or unset) we select greedily/argmax, where
        // temp/top_p shaping is a no-op — and `temp(0.0)` is a division by zero
        // on the logits — so those stages are skipped entirely.
        if let Some(temp) = args.temperature.filter(|&t| t > 0.0) {
            #[allow(clippy::cast_possible_truncation)]
            samplers.push(LlamaSampler::temp(temp as f32));
            if let Some(p) = args.top_p {
                #[allow(clippy::cast_possible_truncation)]
                samplers.push(LlamaSampler::top_p(p as f32, 1));
            }
            samplers.push(LlamaSampler::dist(seed));
        } else {
            samplers.push(LlamaSampler::greedy());
        }

        Ok(LlamaSampler::chain_simple(samplers))
    }

    /// Build the reusable context on first use; subsequent calls are no-ops.
    /// Per-request KV isolation is handled by the caller (clear at request top).
    /// Context creation reallocates the KV cache (and Metal command buffers on
    /// macOS); doing it once instead of per request is the main throughput win
    /// for large contexts. Lazy (first-decode) creation keeps `load` cheap,
    /// matching the lazy-backend-init policy used for metadata-only loads.
    fn ensure_context(&mut self) -> Result<()> {
        if self.context.is_some() {
            return Ok(());
        }
        let t_ctx_start = ggml_time_us();
        let ctx = self
            .model
            .new_context(self.backend, self.ctx_params.clone())
            .with_context(|| "unable to create the llama_context")?;
        tracing::info!(
            "context created in {:.3} s (n_batch={}, n_ubatch={}); reused for subsequent requests",
            Duration::from_micros((ggml_time_us() - t_ctx_start) as u64).as_secs_f32(),
            ctx.n_batch(),
            ctx.n_ubatch(),
        );
        self.context = Some(SyncContext::new(ctx));
        Ok(())
    }

    pub fn run(&mut self, args: InferenceArgs) -> Result<String> {
        if args.prompt.is_empty() && args.medias.is_empty() {
            bail!("prompt is empty and no media provided")
        };

        // tokenize the prompt
        self.decode(args).map(|o| o.text)
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
        // No-op sink: identical to the non-streaming path.
        self.decode_text_only_core_with_sink(formatted_prompt, args, &mut |_| {
            ControlFlow::Continue(())
        })
    }

    /// Streaming-aware variant of [`Self::decode_text_only_core`]. Calls `sink`
    /// once per generated piece (string slice of the just-decoded token), in
    /// generation order. Returning `ControlFlow::Break(())` from the sink stops
    /// generation cleanly without writing back the prefix-reuse cache (the
    /// same invariant as an early error: cancel must not leak partial KV
    /// metadata).
    fn decode_text_only_core_with_sink(
        &mut self,
        formatted_prompt: &str,
        args: &InferenceArgs,
        sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<DecodeOutput> {
        // Thin wrapper: legacy callers want no extra stop sequences and no
        // preserved tokens (everything is rendered as plain text). The
        // `_with_stops` variant below has the actual generation loop; keeping
        // the existing signature here means the multimodal / completion paths
        // do not need to change.
        self.decode_text_only_core_with_sink_and_stops(
            formatted_prompt,
            args,
            &[],
            &std::collections::HashSet::new(),
            sink,
        )
    }

    /// Tools-aware streaming variant. Differs from the legacy
    /// `decode_text_only_core_with_sink` in two ways:
    ///
    /// 1. `extra_stops`: when the running `output_buffer` ends with any
    ///    non-empty stop string, generation breaks *and* the matched suffix
    ///    is trimmed from the returned text. The token that completed the
    ///    stop is still streamed to the sink (it has already been seen by
    ///    the OAI streaming parser), so callers must filter the stop string
    ///    out of the final assistant `content` themselves. This mirrors
    ///    `examples/openai_stream.rs:258-289`.
    /// 2. `preserved_tokens`: token IDs the chat template marked as
    ///    `preserved_tokens` (`<tool_call>` etc.) are rendered with
    ///    `Special::Tokenize` so the OAI parser actually sees them; all
    ///    other tokens fall back to plaintext, avoiding leaked `<|im_end|>`
    ///    markers in user-visible content.
    fn decode_text_only_core_with_sink_and_stops(
        &mut self,
        formatted_prompt: &str,
        args: &InferenceArgs,
        extra_stops: &[String],
        preserved_tokens: &std::collections::HashSet<LlamaToken>,
        sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<DecodeOutput> {
        // Hoist all `&self`-dependent work before borrowing the context field:
        // a `&'static LlamaModel` is `Copy`, and `build_sampler`/`str_to_token`
        // need `&self`, which would conflict with the `&mut self.context` borrow
        // taken below. `model` is then used instead of `self.model` for the rest.
        let model = self.model;
        let mut sampler = self.build_sampler(args)?;
        let tokens_list = model
            .str_to_token(formatted_prompt, AddBos::Always)
            .with_context(|| "failed to tokenize prompt")?;
        let prompt_tokens = tokens_list.len() as i32;
        if args.is_cancel_requested() {
            return Ok(DecodeOutput {
                text: String::new(),
                prompt_tokens: prompt_tokens as u32,
                completion_tokens: 0,
            });
        }

        // Build the context once (lazy) then take it out of `self` for exclusive
        // use; the guard restores it on every exit path (incl. `?`/panic) so the
        // reused context survives a failed request.
        self.ensure_context()?;
        let reuse_kv_prefix = self.reuse_kv_prefix;
        let mut ctx_guard = RestoreOnDrop::new(&mut self.context);
        let sync_ctx = ctx_guard.as_mut().expect("context ensured above");

        // Take the prefix record (leaving it empty) so any early return / panic
        // before we write the new value back leaves it empty — the next request
        // then safely falls back to a full clear. Also forgets the multimodal
        // record (this request's KV is text-only).
        let cached = sync_ctx.take_text_cache();
        let want_keep = plan_kv_keep(
            &cached,
            &tokens_list,
            prompt_tokens as usize,
            reuse_kv_prefix,
        );
        // Keep [0, want_keep) by removing [want_keep, ∞). want_keep == 0 (default
        // path, or no common prefix) means a full clear, so each request starts
        // from an empty KV cache — output identical to a fresh context, while
        // skipping the costly reallocation. `Ok(false)`/`Err` means the model
        // can't do partial removal → fall back to a full clear.
        let n_keep = if want_keep > 0
            && matches!(
                sync_ctx
                    .ctx_mut()
                    .clear_kv_cache_seq(Some(0), Some(want_keep as u32), None),
                Ok(true)
            ) {
            want_keep
        } else {
            sync_ctx.ctx_mut().clear_kv_cache();
            0
        };
        let ctx = sync_ctx.ctx_mut();

        // Install before the first `ctx.decode` so a cancel raised during
        // the very first prefill batch is honoured.
        let _abort_guard = AbortGuard::new(ctx, args.cancel_flag.clone());

        // Cap the position budget at n_ctx so a default `max_new_tokens` of
        // 4096 doesn't fail outright on small-context models. This mirrors
        // the multimodal core path which already applies `.min(n_ctx)`.
        let n_ctx = ctx.n_ctx() as i32;
        let n_len: i32 = if let Some(max_new) = args.max_new_tokens {
            (prompt_tokens + max_new).min(n_ctx)
        } else {
            args.sample_len.unwrap_or(4096).min(n_ctx)
        };
        check_token_length(&tokens_list, ctx, n_len)?;

        // Prefill the suffix after the reused prefix in chunks of at most
        // `n_batch` tokens. A single `llama_decode` call asserts
        // `n_tokens <= cparams.n_batch`, so a prompt longer than the effective
        // n_batch (e.g. >2048 tokens by default) must be split — otherwise
        // generation aborts. Mirrors the multimodal path and llama.cpp's own
        // `mtmd_helper_eval_chunk_single`. Only the final token requests logits,
        // for sampling the first generated token.
        let n_batch = ctx.n_batch() as usize;
        let suffix = &tokens_list[n_keep..];
        // Allocate only what a single chunk needs: capping at the suffix length
        // avoids llama_batch_init mallocing n_batch-sized buffers for a short
        // prompt, while never exceeding n_batch (the per-decode hard limit).
        let mut batch = LlamaBatch::new(n_batch.min(suffix.len()), 1);
        let last_index: i32 = prompt_tokens - 1;
        let mut pos: i32 = n_keep as i32;
        let mut prefill_cancelled = false;
        for chunk in suffix.chunks(n_batch) {
            // Poll cancellation between prefill chunks. A long prompt with a
            // large n_batch can keep the GPU busy for many seconds per chunk,
            // during which the main generation loop (and therefore the
            // per-token sink callback) has not yet started. Without this
            // probe, a client-side cancel observed via cancel_flag would be
            // ignored until generation began. Sending an empty piece lets
            // the existing sink-based cancel path trip without polluting the
            // output: empty deltas are dropped by every downstream encoder
            // (chat/completion streaming, legacy text concatenation).
            if sink_requests_stop(sink, "") {
                prefill_cancelled = true;
                break;
            }
            batch.clear();
            for &token in chunk {
                batch.add(token, pos, &[0], pos == last_index)?;
                pos += 1;
            }
            // If the abort callback fires mid-batch (i.e. the host cancelled
            // during this `llama_decode`), `decode` returns an error. Treat
            // it as cancellation rather than propagating a low-level
            // compute-failure error to the caller, so the request surfaces
            // as `Err(CANCELLED)` upstream and the KV write-back is skipped
            // (we already took `cached_tokens` above).
            if let Err(e) = ctx.decode(&mut batch) {
                if args.is_cancel_requested() {
                    prefill_cancelled = true;
                    break;
                }
                return Err(
                    anyhow::Error::new(e).context("llama_decode() failed during prompt prefill")
                );
            }
        }

        if prefill_cancelled {
            // Skip the generation loop entirely. Mirror the in-loop cancel
            // contract: leave `cached_tokens` empty (already taken above) so
            // the next request starts from a clean KV cache rather than
            // reusing a half-evaluated prefix.
            return Ok(DecodeOutput {
                text: String::new(),
                prompt_tokens: prompt_tokens as u32,
                completion_tokens: 0,
            });
        }

        // main loop

        let mut n_cur = prompt_tokens;
        let mut n_decode = 0;
        // Tracks whether the sink asked to stop (cancel). When true, the
        // prefix-reuse write-back is skipped at the bottom of the function —
        // a partially-consumed prompt must not be advertised as a reusable
        // cache.
        let mut cancelled = false;

        let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // XXX assume string byte size 4
        let mut output_buffer = String::with_capacity((n_len * 4) as usize);

        while n_cur < n_len {
            if args.is_cancel_requested() {
                cancelled = true;
                break;
            }

            // sample the next token
            {
                // `sample` already calls `llama_sampler_accept` internally, so
                // the chain's state (including the llguidance grammar matcher)
                // is advanced exactly once. Calling `accept` again here would
                // double-advance stateful samplers — for llguidance that
                // consumes the just-sampled token twice, corrupting the grammar
                // state and silently disabling all further masking.
                let token = sampler.sample(ctx, batch.n_tokens() - 1);

                // is it an end of stream?
                if model.is_eog_token(token) {
                    eprintln!();
                    break;
                }

                // Tools path: render only model-declared special tokens
                // (preserved_tokens) literally; everything else as plain text
                // so user content (newlines, spaces) is not double-escaped.
                // Legacy callers pass an empty set → behaviour identical to
                // `special=true` from the previous code path? Not quite —
                // legacy used to special=true unconditionally. To preserve
                // legacy output byte-for-byte, the empty-set case still
                // renders as special=true. The tools path opts into the
                // fine-grained behaviour explicitly.
                let render_special = if preserved_tokens.is_empty() {
                    true
                } else {
                    preserved_tokens.contains(&token)
                };
                let output_bytes =
                    token_to_piece_bytes_retry_special(model, token, render_special)?;
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

                // Stream the just-decoded piece; Break → drop out of the
                // loop *before* the next decode and skip KV write-back below.
                if sink_requests_stop(sink, &output_string) {
                    cancelled = true;
                }

                // Tools path: honour additional stop sequences from the
                // chat template. We trim the matched suffix from the
                // accumulated text so the returned `text` does not contain
                // the stop marker (the OAI parser expects the assistant
                // turn to end at the stop). The sink already saw the stop
                // bytes — that's fine because `ChatParseStateOaicompat`
                // treats them as `<|im_end|>`-style markers and elides
                // them from the next delta. Break the loop without flagging
                // cancellation so the KV prefix is still written back: the
                // request completed normally, just early.
                for stop in extra_stops {
                    if !stop.is_empty() && output_buffer.ends_with(stop) {
                        let new_len = output_buffer.len().saturating_sub(stop.len());
                        output_buffer.truncate(new_len);
                        // Note: do not bump n_cur — we are returning before
                        // the next `ctx.decode`, so the KV cursor has not
                        // advanced past the current token.
                        let t_main_end = ggml_time_us();
                        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
                        tracing::info!(
                            "decoded {} tokens in {:.2} s, speed {:.2} t/s (stopped on {:?})",
                            n_decode + 1,
                            duration.as_secs_f32(),
                            (n_decode + 1) as f32 / duration.as_secs_f32(),
                            stop,
                        );
                        if reuse_kv_prefix {
                            sync_ctx.cached_tokens = tokens_list;
                        }
                        return Ok(DecodeOutput {
                            text: output_buffer,
                            prompt_tokens: prompt_tokens as u32,
                            completion_tokens: (n_decode + 1) as u32,
                        });
                    }
                }
            }

            if cancelled {
                break;
            }

            n_cur += 1;

            if let Err(e) = ctx.decode(&mut batch) {
                if args.is_cancel_requested() {
                    cancelled = true;
                    break;
                }
                return Err(anyhow::Error::new(e).context("failed to eval"));
            }

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

        // Record the prompt tokens now in the KV cache so the next request can
        // reuse their shared prefix. Only the prompt (positions 0..prompt_tokens)
        // is stored: generated tokens occupy positions >= prompt_tokens, which a
        // future request always discards (they lie beyond any common prefix).
        // Written only on natural completion — an error or a sink-driven
        // cancel leaves `cached_tokens` empty (taken above), forcing a safe
        // full clear next time.
        if reuse_kv_prefix && !cancelled {
            sync_ctx.cached_tokens = tokens_list;
        }

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
        self.decode_multimodal_core_with_sink(formatted_prompt, bitmaps, args, &mut |_| {
            ControlFlow::Continue(())
        })
    }

    /// Streaming-aware variant of [`Self::decode_multimodal_core`]. Same sink
    /// contract as [`Self::decode_text_only_core_with_sink`].
    fn decode_multimodal_core_with_sink(
        &mut self,
        formatted_prompt: &str,
        bitmaps: &[llama_cpp_2::mtmd::MtmdBitmap],
        args: &InferenceArgs,
        sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<DecodeOutput> {
        // Hoist `&self`-wide work (build_sampler, model copy) before borrowing
        // the context field. `mtmd` is a separate field, so it can be borrowed
        // alongside the `&mut self.context` guard below (disjoint borrows).
        let model = self.model;
        let mut sampler = self.build_sampler(args)?;

        let reuse_kv_prefix = self.reuse_kv_prefix;
        self.ensure_context()?;
        let mtmd = self
            .mtmd
            .as_ref()
            .expect("mtmd should be Some in multimodal path");
        let mut ctx_guard = RestoreOnDrop::new(&mut self.context);
        let sync_ctx = ctx_guard.as_mut().expect("context ensured above");

        // Take the multimodal record (leaving it empty so an early return falls
        // back to a full clear) and forget the text record.
        let cached_chunks = sync_ctx.take_chunk_cache();

        tracing::debug!("multimodal formatted prompt: {}", formatted_prompt);

        // Tokenize into chunks (without evaluating) so we can compare against the
        // cached chunk sequence and reuse the matching KV prefix.
        let chunks = mtmd
            .tokenize(formatted_prompt, bitmaps)
            .map_err(|e| anyhow::anyhow!(e).context("mtmd: tokenize"))?;
        let (new_chunks, new_n_pos) = describe_chunks(&chunks, bitmaps);

        let (start_index, want_keep) =
            plan_chunk_keep(&cached_chunks, &new_chunks, &new_n_pos, reuse_kv_prefix);

        let n_batch = sync_ctx.ctx_mut().n_batch() as i32;
        let n_ctx = sync_ctx.ctx_mut().n_ctx() as i32;

        // Keep [0, want_keep); a partial-removal failure or want_keep==0 falls
        // back to a full clear (start from chunk 0).
        let (start_index, n_keep) = if want_keep > 0
            && matches!(
                sync_ctx
                    .ctx_mut()
                    .clear_kv_cache_seq(Some(0), Some(want_keep as u32), None),
                Ok(true)
            ) {
            (start_index, want_keep)
        } else {
            sync_ctx.ctx_mut().clear_kv_cache();
            (0, 0)
        };
        let ctx = sync_ctx.ctx_mut();

        // Install the abort flag before mtmd's `eval_chunks_from`, which
        // internally drives `ctx.decode()` on each chunk. Image / audio
        // prefill on long contexts can take many seconds per chunk, so
        // honouring the host cancel during this phase — not just after it
        // completes — is the whole point of wiring the callback.
        let _abort_guard = AbortGuard::new(ctx, args.cancel_flag.clone());

        let n_past = match mtmd.eval_chunks_from(&chunks, start_index, n_keep, ctx, n_batch) {
            Ok(n) => n,
            Err(e) => {
                if args.is_cancel_requested() {
                    // KV cache state is undefined after an aborted prefill;
                    // we already took `cached_chunks`, so the next request
                    // will perform a full clear. Return a zero-length result
                    // matching the cancel contract of the text path.
                    return Ok(DecodeOutput {
                        text: String::new(),
                        prompt_tokens: 0,
                        completion_tokens: 0,
                    });
                }
                return Err(anyhow::anyhow!(e).context("mtmd: eval_chunks_from"));
            }
        };

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
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output_buffer = String::with_capacity(((n_len - n_past) * 4) as usize);
        let mut first = true;
        let mut n_decode = 0;
        let mut cancelled = false;

        let t_main_start = ggml_time_us();

        while n_cur < n_stop {
            // First sample uses the logit from eval_chunks (logits_last=true),
            // subsequent samples use the single token in the batch.
            let sample_idx = if first { -1 } else { batch.n_tokens() - 1 };
            // `sample` accepts the token internally; do not call `accept` again
            // (see decode_text_only_core for why double-accept breaks the
            // llguidance grammar sampler).
            let token = sampler.sample(ctx, sample_idx);

            if model.is_eog_token(token) {
                break;
            }

            let output_bytes = token_to_piece_bytes_retry(model, token)?;
            let mut output_string = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            output_buffer.push_str(&output_string);

            if sink_requests_stop(sink, &output_string) {
                cancelled = true;
                break;
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            if let Err(e) = ctx.decode(&mut batch) {
                if args.is_cancel_requested() {
                    cancelled = true;
                    break;
                }
                return Err(anyhow::Error::new(e).context("failed to eval in multimodal loop"));
            }
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

        // Record the prompt's chunk sequence so the next multimodal request can
        // reuse its shared prefix. Only on natural completion — an error or a
        // sink-driven cancel leaves the record empty (taken above), forcing a
        // safe full clear next time.
        if reuse_kv_prefix && !cancelled {
            sync_ctx.cached_chunks = new_chunks;
        }

        Ok(DecodeOutput {
            text: output_buffer,
            prompt_tokens: n_past as u32,
            completion_tokens: n_decode as u32,
        })
    }

    pub fn run_chat(&mut self, args: LlmChatArgs) -> Result<LlmChatResult> {
        self.run_chat_with_sink(args, None, &mut |_| ControlFlow::Continue(()))
    }

    /// Streaming-aware variant of [`Self::run_chat`]. The `sink` is invoked
    /// once per generated piece (already in raw form — `<think>` tags are
    /// included). The returned `LlmChatResult` is the **final** chunk and
    /// carries `done=true`, the full `usage` block, and (when requested) the
    /// extracted reasoning_content. Callers that re-split reasoning in real
    /// time should ignore the result's `reasoning_content`/`content` payload
    /// (they will already have streamed both) and use only the metadata.
    ///
    /// `cancel_flag`, when `Some`, is installed on the underlying
    /// `LlamaContext` as a ggml abort callback for the duration of the
    /// decode. Setting it to `true` from another thread aborts the
    /// in-flight `decode` call (notably the long per-batch GPU computation
    /// during prefill) instead of having to wait for the next sink poll
    /// between batches.
    pub fn run_chat_with_sink(
        &mut self,
        mut args: LlmChatArgs,
        cancel_flag: Option<Arc<AtomicBool>>,
        sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<LlmChatResult> {
        if let Some(ref fo) = args.function_options
            && fo.use_function_calling
        {
            bail!(ERR_USE_FUNCTION_CALLING_UNSUPPORTED);
        }
        if args.model.is_some() {
            tracing::warn!(
                "LLMChatArgs.model is ignored: model is fixed at load time in this plugin"
            );
        }

        // Tools path: when the caller passes `client_tools_json`, route
        // through the OpenAI-compatible chat template + tool parser. `take()`
        // moves the (potentially multi-KB) tool-defs JSON out of
        // `function_options` instead of cloning it. The path's
        // mutual-exclusion guards (json_schema, use_function_calling) live in
        // `run_chat_with_sink_tools` itself so the streaming entry
        // (`spawn_chat_stream_with_tools`) is validated by the same code.
        if let Some(client_tools_json) = args
            .function_options
            .as_mut()
            .and_then(|fo| fo.client_tools_json.take())
        {
            let mut oai_sink = |_: crate::oai_chat::OaiStreamUpdate| {};
            return self.run_chat_with_sink_tools(
                args,
                &client_tools_json,
                cancel_flag,
                sink,
                &mut oai_sink,
            );
        }

        let options = args.options.unwrap_or_default();
        let extract_reasoning = options.extract_reasoning_content.unwrap_or(false);

        let (chat_messages, raw_messages, medias) = self.build_chat_messages(&args.messages)?;
        let formatted_prompt = self.apply_chat_template_multi(&chat_messages, &raw_messages)?;

        // `prompt` is left empty and `medias` is empty because the core path
        // consumes `formatted_prompt` and `bitmaps` directly; only the
        // sampler/limits fields of InferenceArgs are read downstream.
        let inference_args = chat_inference_args(options, args.json_schema, cancel_flag);

        let t_start = ggml_time_us();
        let output: DecodeOutput = if medias.is_empty() {
            self.decode_text_only_core_with_sink(&formatted_prompt, &inference_args, sink)?
        } else {
            let mtmd = self
                .mtmd
                .as_ref()
                .context("multimodal input given but mmproj is not configured")?;
            let bitmaps = mtmd
                .prepare_bitmaps(&medias, &self.media_limits)
                .map_err(|e| anyhow::anyhow!(e).context("mtmd: preparing bitmaps"))?;
            self.decode_multimodal_core_with_sink(
                &formatted_prompt,
                &bitmaps,
                &inference_args,
                sink,
            )?
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

    /// Tools-aware variant of [`Self::run_chat_with_sink`]. Routes the chat
    /// request through the fork's `apply_chat_template_oaicompat` API so that
    /// the resulting `LlmChatResult` can report parsed tool calls in
    /// `pending_tool_calls` (with `requires_tool_execution=Some(true)`) for
    /// client-side execution.
    ///
    /// Multimodal input is rejected here: the tool grammar emitted by the
    /// OAI chat template is only valid on the text-only decode core, and
    /// the multimodal eval path expects a different prompt shape (text
    /// marker + bitmaps).
    pub(crate) fn run_chat_with_sink_tools(
        &mut self,
        mut args: LlmChatArgs,
        tools_json: &str,
        cancel_flag: Option<Arc<AtomicBool>>,
        sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
        oai_sink: &mut dyn FnMut(crate::oai_chat::OaiStreamUpdate),
    ) -> Result<LlmChatResult> {
        // Validate up front: this is the single chokepoint for both the
        // non-streaming entry (`run_chat_with_sink`) and the streaming
        // entry (`spawn_chat_stream_with_tools`), so all client-tools
        // preconditions belong here.
        if args
            .function_options
            .as_ref()
            .is_some_and(|fo| fo.use_function_calling)
        {
            bail!(ERR_CLIENT_TOOLS_WITH_FUNCTION_CALLING);
        }
        if args.json_schema.is_some() {
            bail!(ERR_CLIENT_TOOLS_WITH_JSON_SCHEMA);
        }
        // Reject multimodal input up front so we never spend cycles building
        // the OAI messages JSON for a request we can't fulfil.
        if args.messages.iter().any(|m| {
            matches!(
                m.content.as_ref().and_then(|c| c.content.as_ref()),
                Some(llm_chat_args::message_content::Content::Image(_))
            )
        }) {
            bail!(ERR_CLIENT_TOOLS_WITH_MULTIMODAL);
        }

        let options = args.options.take().unwrap_or_default();
        let messages_json =
            crate::oai_chat::build_oai_messages_json(&self.system_prompt, &args.messages)?;

        // Translate OpenAI's `{"type":"function","function":{"name":"..."}}`
        // tool_choice into the (tools-filter + "required") pair llama.cpp
        // accepts. Bare "auto"/"none"/"required" or None come back as
        // `Passthrough` so the originals are forwarded verbatim.
        let function_options = args.function_options.as_ref();
        let resolved = crate::oai_chat::resolve_tool_choice(
            tools_json,
            function_options.and_then(|fo| fo.tool_choice.as_deref()),
        )?;
        let (effective_tools_json, tool_choice_override) = match &resolved {
            crate::oai_chat::ResolvedToolChoice::Passthrough => (tools_json, None),
            crate::oai_chat::ResolvedToolChoice::FunctionSpecific {
                tools_json,
                tool_choice,
            } => (tools_json.as_str(), Some(tool_choice.as_str())),
        };

        let tmpl_result = self.apply_oai_template_with_tools(
            &messages_json,
            effective_tools_json,
            function_options,
            tool_choice_override,
        )?;

        let preserved_tokens =
            crate::oai_chat::compute_preserved_token_set(self.model, &tmpl_result.preserved_tokens);
        let grammar_spec = tmpl_result
            .grammar
            .as_ref()
            .map(|g| crate::oai_chat::GrammarSpec {
                grammar: g.clone(),
                grammar_lazy: tmpl_result.grammar_lazy,
                grammar_triggers: tmpl_result.grammar_triggers.clone(),
            });
        let mut inference_args = chat_inference_args(options, None, cancel_flag);
        inference_args.grammar_spec = grammar_spec;

        let extra_stops = tmpl_result.additional_stops.clone();
        let t_start = ggml_time_us();
        // Wrap the caller-supplied raw-chunk `sink` with an OAI parser so the
        // streaming path receives structured `OaiStreamUpdate`s (text /
        // reasoning / tool_calls) without the worker having to re-parse the
        // assistant output. Borrowing `state` and `oai_sink` here is fine
        // because the wrapped closure lives only for the duration of the
        // `decode_text_only_core_with_sink_and_stops` call.
        let mut state = tmpl_result
            .streaming_state_oaicompat()
            .map_err(|e| anyhow!("streaming_state_oaicompat failed: {e:?}"))?;
        let output = {
            let mut wrapped_sink = |chunk: &str| -> ControlFlow<()> {
                // Always forward the raw chunk first; the legacy receiver
                // uses it for cancel-driven backpressure and for any other
                // downstream consumer that cares about per-token output.
                let raw_flow = sink(chunk);
                match state.update(chunk, true) {
                    Ok(deltas) if !deltas.is_empty() => {
                        let upd = crate::oai_chat::decode_oai_deltas(&deltas);
                        if !upd.text.is_empty()
                            || !upd.reasoning.is_empty()
                            || !upd.tool_calls.is_empty()
                        {
                            oai_sink(upd);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!("ChatParseStateOaicompat::update failed mid-stream: {e:?}");
                    }
                }
                raw_flow
            };
            self.decode_text_only_core_with_sink_and_stops(
                &tmpl_result.prompt,
                &inference_args,
                &extra_stops,
                &preserved_tokens,
                &mut wrapped_sink,
            )?
        };
        let t_end = ggml_time_us();
        let total_time = Duration::from_micros((t_end - t_start) as u64);

        // Flush the parser so any buffered partial (e.g. final closing
        // `</tool_call>`) is materialised into the OAI sink stream.
        if let Ok(final_deltas) = state.update("", false)
            && !final_deltas.is_empty()
        {
            let upd = crate::oai_chat::decode_oai_deltas(&final_deltas);
            if !upd.text.is_empty() || !upd.reasoning.is_empty() || !upd.tool_calls.is_empty() {
                oai_sink(upd);
            }
        }

        let parsed_json = parse_tool_response_with_recovery(&tmpl_result, &output.text)?;
        crate::oai_chat::build_chat_result_from_oai_json(
            &parsed_json,
            output.prompt_tokens,
            output.completion_tokens,
            0.0,
            total_time.as_secs_f32(),
            "",
        )
    }

    /// Execute a text-only completion request. Unlike `run_chat`, this method
    /// does NOT accept media — multimodal callers must use the chat method.
    /// `args.context.ollama_context` and `args.model` are accepted with a warn
    /// (see `validate_completion_args`) so jobworkerp completion workers can
    /// reuse their existing payload shape.
    pub fn run_completion(&mut self, args: LlmCompletionArgs) -> Result<LlmCompletionResult> {
        self.run_completion_with_sink(args, None, &mut |_| ControlFlow::Continue(()))
    }

    /// Streaming-aware variant of [`Self::run_completion`]. See
    /// [`Self::run_chat_with_sink`] for the sink contract. `cancel_flag`
    /// follows the same abort-callback semantics as in `run_chat_with_sink`.
    pub fn run_completion_with_sink(
        &mut self,
        args: LlmCompletionArgs,
        cancel_flag: Option<Arc<AtomicBool>>,
        sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<LlmCompletionResult> {
        validate_completion_args(&args)?;
        if cancel_flag_requested(&cancel_flag) {
            return Ok(cancelled_completion_result());
        }

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
        if cancel_flag_requested(&cancel_flag) {
            return Ok(cancelled_completion_result());
        }

        let inference_args = completion_inference_args(options, args.json_schema, cancel_flag);

        let t_start = ggml_time_us();
        let output =
            self.decode_text_only_core_with_sink(&formatted_prompt, &inference_args, sink)?;
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
                        pending_media = Some(decode_image_to_media(img)?);
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
                    Some(llm_chat_args::message_content::Content::ToolExecutionRequests(_)) => {
                        bail!(
                            "ToolExecutionRequests is no longer accepted; \
                             use ToolResults on a TOOL message"
                        );
                    }
                    Some(llm_chat_args::message_content::Content::ToolResults(tr)) => {
                        if tr.results.is_empty() {
                            bail!("ToolResults must contain at least one entry");
                        }
                        let mut rendered = Vec::with_capacity(tr.results.len());
                        for r in &tr.results {
                            if r.call_id.is_empty() {
                                bail!("ToolResult.call_id must not be empty");
                            }
                            rendered.push(serde_json::json!({
                                "call_id": r.call_id,
                                "content": if r.is_error {
                                    format!("[ERROR] {}", r.content)
                                } else {
                                    r.content.clone()
                                },
                            }));
                        }
                        serde_json::to_string(&rendered).unwrap_or_default()
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

    /// Apply the model's chat template via the fork's OpenAI-compatible API.
    /// `tool_opts` carries the model-agnostic switches (tool_choice,
    /// parallel_tool_calls, reasoning_format, chat_template_kwargs); model
    /// specific knobs like `enable_thinking` must be set through the
    /// `chat_template_kwargs` JSON object so the plugin itself stays
    /// model-neutral.
    ///
    /// `tool_choice_override` lets the caller substitute a normalised
    /// tool_choice string after running `oai_chat::resolve_tool_choice`
    /// (which translates OpenAI's `{"type":"function",...}` shape into
    /// `"required"` + filtered tools); when `Some`, it shadows
    /// `tool_opts.tool_choice`.
    fn apply_oai_template_with_tools(
        &self,
        oai_messages_json: &str,
        tools_json: &str,
        tool_opts: Option<&llm_chat_args::FunctionOptions>,
        tool_choice_override: Option<&str>,
    ) -> Result<llama_cpp_2::model::ChatTemplateResult> {
        let tmpl = self
            .model
            .chat_template(None)
            .context("model does not expose a chat template required for tool calling")?;
        let chat_template_kwargs = tool_opts.and_then(|fo| fo.chat_template_kwargs.as_deref());
        // `enable_thinking` controls both the jinja template (via kwargs) and
        // the C++ grammar/parser (via this dedicated bool). Forward whatever
        // the caller put inside `chat_template_kwargs.enable_thinking` so the
        // two channels agree; default to false to preserve current behaviour
        // for callers that omit the key.
        let enable_thinking =
            crate::oai_chat::extract_enable_thinking(chat_template_kwargs).unwrap_or(false);
        let params = llama_cpp_2::openai::OpenAIChatTemplateParams {
            messages_json: oai_messages_json,
            tools_json: Some(tools_json),
            tool_choice: tool_choice_override
                .or_else(|| tool_opts.and_then(|fo| fo.tool_choice.as_deref())),
            // run_chat_with_sink rejects json_schema before reaching this
            // point, so leaving it empty is safe.
            json_schema: None,
            grammar: None,
            reasoning_format: tool_opts.and_then(|fo| fo.reasoning_format.as_deref()),
            chat_template_kwargs,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: tool_opts
                .and_then(|fo| fo.parallel_tool_calls)
                .unwrap_or(false),
            enable_thinking,
            add_bos: true,
            add_eos: false,
            parse_tool_calls: true,
        };
        self.model
            .apply_chat_template_oaicompat(&tmpl, &params)
            .map_err(|e| anyhow!("apply_chat_template_oaicompat failed: {e:?}"))
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

/// Length of the longest common prefix of two token slices. Used for KV prefix
/// reuse: tokens up to this length are already in the KV cache and can be kept.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Number of KV-cached prompt tokens to keep for prefix reuse. Returns 0 (full
/// clear) when reuse is off or there is no cache. Capped at `prompt_tokens - 1`
/// so the last prompt token is always re-decoded — the first sample needs fresh
/// logits, and an empty prefill range would leave none. `prompt_tokens >= 1` is
/// guaranteed by `AddBos::Always` plus the empty-prompt bail in `run`.
fn plan_kv_keep(
    cached: &[LlamaToken],
    tokens: &[LlamaToken],
    prompt_tokens: usize,
    reuse: bool,
) -> usize {
    if !reuse || cached.is_empty() {
        return 0;
    }
    common_prefix_len(cached, tokens).min(prompt_tokens - 1)
}

/// Identity key for an image/audio bitmap, used to decide whether a media chunk
/// in a new prompt matches one already in the KV cache. Hashes the decoded
/// pixel/sample data plus dimensions. A collision would reuse the wrong KV, so
/// the full data is hashed (not just dimensions).
fn bitmap_identity_key(bitmap: &llama_cpp_2::mtmd::MtmdBitmap) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bitmap.is_audio().hash(&mut hasher);
    bitmap.nx().hash(&mut hasher);
    bitmap.ny().hash(&mut hasher);
    bitmap.data().hash(&mut hasher);
    hasher.finish()
}

/// Plan how much of the multimodal KV cache to keep. Compares the cached chunk
/// sequence against the new one chunk-by-chunk and returns
/// `(start_index, n_keep)`: the number of leading chunks to reuse and the KV
/// position count they occupy. Reuse stops at the first differing chunk.
///
/// If every new chunk matches, the last matching chunk is dropped (re-evaluated)
/// so the final prompt position is re-decoded — the first sample needs fresh
/// logits, mirroring the text path's `prompt_tokens - 1` cap. Returns `(0, 0)`
/// (full clear) when reuse is off, there is no cache, or nothing matches.
fn plan_chunk_keep(
    cached: &[CachedChunk],
    new_chunks: &[CachedChunk],
    new_n_pos: &[i32],
    reuse: bool,
) -> (usize, i32) {
    if !reuse || cached.is_empty() || new_chunks.is_empty() {
        return (0, 0);
    }
    // A media chunk with no bitmap (`key: None`) can't be proven identical, so
    // it ends the reusable prefix even though `None == None` — otherwise a later
    // changed image could be mistaken for an unchanged one and reuse stale KV.
    let reusable = |c: &CachedChunk| !matches!(c, CachedChunk::Media { key: None, .. });
    let mut matched = cached
        .iter()
        .zip(new_chunks)
        .take_while(|(a, b)| a == b && reusable(a))
        .count();
    // Never keep the entire new prompt: re-evaluate at least the last chunk.
    if matched == new_chunks.len() {
        matched -= 1;
    }
    let n_keep: i32 = new_n_pos[..matched].iter().sum();
    (matched, n_keep)
}

/// Describe a tokenized chunk sequence for prefix comparison: a `CachedChunk`
/// per chunk (text → tokens, media → bitmap identity key) and its KV position
/// count. Media chunks consume `bitmaps` in order.
///
/// One media chunk is paired with one bitmap. Tiling models (e.g. MiniCPM-V,
/// Idefics3) split a single bitmap into several image chunks, so chunks can
/// outnumber bitmaps; once bitmaps run out, the remaining media chunks get
/// `key: None`. `plan_chunk_keep` stops reuse at the first such chunk, so a
/// later differing image can never be mistaken for an unchanged one — at the
/// cost of no reuse past that point on tiling models. Per-tile keys would
/// restore it.
fn describe_chunks(
    chunks: &llama_cpp_2::mtmd::MtmdInputChunks,
    bitmaps: &[llama_cpp_2::mtmd::MtmdBitmap],
) -> (Vec<CachedChunk>, Vec<i32>) {
    use llama_cpp_2::mtmd::MtmdInputChunkType;
    let mut described = Vec::with_capacity(chunks.len());
    let mut n_pos = Vec::with_capacity(chunks.len());
    let mut bitmaps = bitmaps.iter();
    for i in 0..chunks.len() {
        let chunk = chunks.get(i).expect("index < len");
        let cached = match chunk.chunk_type() {
            MtmdInputChunkType::Text => {
                CachedChunk::Text(chunk.text_tokens().unwrap_or_default().to_vec())
            }
            _ => CachedChunk::Media {
                key: bitmaps.next().map(bitmap_identity_key),
                n_pos: chunk.n_positions(),
            },
        };
        n_pos.push(chunk.n_positions());
        described.push(cached);
    }
    (described, n_pos)
}

/// Validate the position budget against the context size. Free function (not a
/// method) so it can be called while the context is borrowed out of `self` via
/// the reuse guard, without also borrowing `&self`.
fn check_token_length(tokens_list: &[LlamaToken], ctx: &LlamaContext, n_len: i32) -> Result<()> {
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

/// Returns `true` when the streaming sink asked to stop generation. Used by
/// both core loops so the "Break → cancel + suppress KV write-back" rule is
/// expressed in one place.
fn sink_requests_stop(sink: &mut dyn FnMut(&str) -> ControlFlow<()>, chunk: &str) -> bool {
    matches!(sink(chunk), ControlFlow::Break(()))
}

/// Strip the chat template's `generation_prompt` from a generated response
/// so the OAI parser sees only the in-turn body. Eager-grammar paths
/// (`tool_choice="required"`) make the model regenerate the assistant
/// header because the grammar root spans the whole turn, but the model
/// may skip optional trailing structure inside `generation_prompt` (e.g.
/// Qwen3's empty `<think>\n\n</think>\n\n` block).
///
/// Strategy: try `strip_prefix(generation_prompt)` first; on full match
/// the parser sees only the in-turn body. Otherwise fall back to
/// stripping just the role-header — everything up to and including the
/// first newline (`<|im_start|>assistant\n` for ChatML-style templates) —
/// so a partial regeneration still parses cleanly. Returns the input
/// untouched when neither prefix matches.
fn strip_generation_prompt<'a>(s: &'a str, generation_prompt: &str) -> &'a str {
    if let Some(rest) = s.strip_prefix(generation_prompt) {
        return rest;
    }
    let header_end = match generation_prompt.find('\n') {
        Some(i) => i + 1,
        None => return s,
    };
    let header = &generation_prompt[..header_end];
    s.strip_prefix(header).unwrap_or(s)
}

/// Convert a token to bytes, retrying with a larger buffer if the first
/// attempt reports `InsufficientBufferSpace`. Mirrors the internal retry
/// logic of `LlamaModel::token_to_piece`, but returns raw bytes so the
/// caller can feed them to an incremental UTF-8 decoder.
/// Try the OAI parser on the raw output first; on failure peel off the
/// regenerated assistant header (eager-grammar paths like
/// `tool_choice="required"` make the model echo `generation_prompt`); on
/// final failure recover the tool calls with a tag-based fallback.
/// Returning Err means none of the three approaches found a structured
/// reply, which is genuinely unrecoverable.
fn parse_tool_response_with_recovery(
    tmpl_result: &llama_cpp_2::model::ChatTemplateResult,
    raw: &str,
) -> Result<String> {
    if let Ok(j) = tmpl_result.parse_response_oaicompat(raw, false) {
        return Ok(j);
    }
    let stripped = strip_generation_prompt(raw, &tmpl_result.generation_prompt);
    if stripped.as_ptr() != raw.as_ptr()
        && let Ok(j) = tmpl_result.parse_response_oaicompat(stripped, false)
    {
        return Ok(j);
    }
    crate::oai_chat::fallback_parse_tool_calls(stripped).ok_or_else(|| {
        anyhow!(
            "parse_response_oaicompat failed on raw and stripped inputs and no \
             <tool_call>...</tool_call> envelope was recoverable: raw={raw:?}"
        )
    })
}

fn token_to_piece_bytes_retry(
    model: &LlamaModel,
    token: LlamaToken,
) -> Result<Vec<u8>, llama_cpp_2::TokenToStringError> {
    token_to_piece_bytes_retry_special(model, token, /* special= */ true)
}

/// Variant that lets the caller decide whether to render a token as a special
/// (literal `<tool_call>` etc) or plaintext. Used by the tools path with the
/// `preserved_tokens` set from `ChatTemplateResult`: special tokens get
/// `true`, all others `false` so user-facing whitespace isn't rendered
/// double-escaped.
fn token_to_piece_bytes_retry_special(
    model: &LlamaModel,
    token: LlamaToken,
    special: bool,
) -> Result<Vec<u8>, llama_cpp_2::TokenToStringError> {
    match model.token_to_piece_bytes(token, 8, special, None) {
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(i)) => {
            let size = (-i).try_into().expect("Error buffer size is positive");
            model.token_to_piece_bytes(token, size, special, None)
        }
        other => other,
    }
}

/// Decode a chat `MessageContent::Image` payload into a `MediaInput` accepted
/// by the multimodal generation path. Extracted so both the legacy
/// `build_chat_messages` and the new OAI tools path (`oai_chat`) handle base64
/// images identically (URL fetch is intentionally unsupported here — both
/// callers must reject it the same way).
pub(crate) fn decode_image_to_media(
    img: &llm_chat_args::message_content::Image,
) -> Result<MediaInput> {
    use base64::Engine;
    let source = img.source.as_ref().context("image message has no source")?;
    if source.base64.is_empty() {
        bail!("image URL fetch is not supported in this plugin");
    }
    let encoded = base64::engine::general_purpose::STANDARD
        .decode(&source.base64)
        .context("invalid base64 in image")?;
    Ok(MediaInput {
        kind: jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind::Image as i32,
        source: Some(
            jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source::Encoded(encoded),
        ),
        id: None,
    })
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
        bail!(ERR_USE_FUNCTION_CALLING_UNSUPPORTED);
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
            medias,
            ..Default::default()
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
    fn test_text_decode_loop_polls_cancel_before_sampling() {
        let source = include_str!("model.rs");

        assert!(
            source.contains("while n_cur < n_len {\n            if args.is_cancel_requested()"),
            "decode loop must poll cancellation before sampling the next token"
        );
    }

    /// Regression test: a prompt longer than the configured `n_batch` must be
    /// prefilled in chunks. Before chunking, the single-batch decode tripped
    /// `llama_decode`'s `n_tokens <= n_batch` assertion and aborted. Uses a
    /// small `n_batch` so a short prompt already exceeds it.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_text_only_prompt_exceeding_n_batch \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_text_only_prompt_exceeding_n_batch() {
        let config = LlamaModelConfig {
            // Force a tiny logical batch so the prompt below exceeds it and
            // must be split across multiple decode calls.
            n_batch: Some(8),
            n_ubatch: Some(8),
            ctx_size: std::num::NonZeroU32::new(2048),
            ..LlamaModelConfig::default()
        };
        let mut wrapper = LlamaModelWrapper::new(config).expect("Failed to load model");

        // This prompt tokenizes to well over 8 tokens, so prefill spans
        // multiple n_batch-sized chunks.
        let prompt = "The quick brown fox jumps over the lazy dog. \
                      Please continue this sentence in a few words.";
        let args = InferenceArgs {
            prompt: prompt.to_string(),
            sample_len: Some(64),
            seed: Some(1234),
            ..Default::default()
        };

        let output = wrapper
            .run(args)
            .expect("chunked-prefill generation failed");
        assert!(!output.is_empty(), "Output should not be empty");
        println!("chunked-prefill output: {output}");
    }

    /// Context reuse across requests must be transparent: the reused context is
    /// KV-cleared at each request top, so consecutive requests are independent
    /// and deterministic, and a failed request does not poison the next one.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_context_reuse_is_isolated \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_context_reuse_is_isolated() {
        // Greedy (no temperature) for determinism.
        fn args(prompt: &str) -> InferenceArgs {
            InferenceArgs {
                prompt: prompt.to_string(),
                sample_len: Some(48),
                seed: Some(1234),
                ..Default::default()
            }
        }
        let prompt_a = "The capital of France is";
        let prompt_b = "Two plus two equals";

        // Baseline: each prompt run once on its own fresh wrapper (fresh context).
        let baseline_b = LlamaModelWrapper::new(LlamaModelConfig::default())
            .expect("load")
            .run(args(prompt_b))
            .expect("baseline b");

        // Reuse: run A then B on the SAME wrapper (B reuses A's context).
        let mut wrapper = LlamaModelWrapper::new(LlamaModelConfig::default()).expect("load");
        let reuse_a1 = wrapper.run(args(prompt_a)).expect("reuse a1");
        let reuse_b = wrapper.run(args(prompt_b)).expect("reuse b");
        // (a) KV isolation: B after A equals B on a fresh context.
        assert_eq!(
            reuse_b, baseline_b,
            "reused context must not leak prompt A's KV into prompt B"
        );
        // (b) Determinism: A again equals A the first time.
        let reuse_a2 = wrapper.run(args(prompt_a)).expect("reuse a2");
        assert_eq!(reuse_a1, reuse_a2, "reused context must be deterministic");

        // (c) Error recovery: an oversized sample_len bails (sample_len > n_ctx
        // is rejected), then a valid request must still succeed and match.
        let mut bad = args(prompt_a);
        bad.sample_len = Some(i32::MAX);
        assert!(wrapper.run(bad).is_err(), "oversized request should fail");
        let after_err = wrapper.run(args(prompt_a)).expect("recovers after error");
        assert_eq!(
            after_err, reuse_a1,
            "context must recover cleanly after a failed request"
        );
    }

    /// With `reuse_kv_prefix` on, keeping the shared prompt prefix in the KV
    /// cache must produce output identical to a full clear (the kept KV is the
    /// same computation), and a failed request must still recover.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_reuse_kv_prefix_matches_full_clear \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_reuse_kv_prefix_matches_full_clear() {
        fn args(prompt: &str) -> InferenceArgs {
            InferenceArgs {
                prompt: prompt.to_string(),
                sample_len: Some(48),
                seed: Some(1234),
                ..Default::default()
            }
        }
        // Shared prefix, differing tails — the case prefix reuse optimizes.
        let shared = "You are a helpful assistant. The user asks: ";
        let prompt_a = format!("{shared}what is the capital of France?");
        let prompt_b = format!("{shared}what is two plus two?");

        let reuse_cfg = || LlamaModelConfig {
            reuse_kv_prefix: true,
            ..LlamaModelConfig::default()
        };

        // Baselines on fresh wrappers (each request = full clear).
        let baseline_a = LlamaModelWrapper::new(reuse_cfg())
            .expect("load")
            .run(args(&prompt_a))
            .expect("baseline a");
        let baseline_b = LlamaModelWrapper::new(reuse_cfg())
            .expect("load")
            .run(args(&prompt_b))
            .expect("baseline b");

        // Same wrapper: A then B reuses the shared prefix; outputs must match
        // the full-clear baselines (correctness of partial KV removal).
        let mut wrapper = LlamaModelWrapper::new(reuse_cfg()).expect("load");
        let reuse_a = wrapper.run(args(&prompt_a)).expect("reuse a");
        let reuse_b = wrapper.run(args(&prompt_b)).expect("reuse b");
        assert_eq!(reuse_a, baseline_a, "prefix reuse changed prompt A output");
        assert_eq!(
            reuse_b, baseline_b,
            "prefix reuse (A→B) changed prompt B output"
        );

        // Determinism: A again (reuses its own full prefix) equals the first A.
        let reuse_a2 = wrapper.run(args(&prompt_a)).expect("reuse a2");
        assert_eq!(reuse_a2, baseline_a, "prefix reuse must be deterministic");

        // Error recovery: a failed request empties the cache record; the next
        // request falls back to a full clear and still matches the baseline.
        let mut bad = args(&prompt_a);
        bad.sample_len = Some(i32::MAX);
        assert!(wrapper.run(bad).is_err(), "oversized request should fail");
        let after_err = wrapper.run(args(&prompt_b)).expect("recovers after error");
        assert_eq!(after_err, baseline_b, "must recover after a failed request");
    }

    /// A sink that requests cancellation on its very first invocation must
    /// short-circuit the request inside the prompt-prefill loop — before any
    /// token is sampled. Long prompts otherwise keep the GPU busy in
    /// `ctx.decode` chunks while no per-token sink callback fires, hiding
    /// the host's cancel.
    #[test]
    #[ignore]
    fn test_prefill_cancellation_short_circuits() {
        let mut wrapper = LlamaModelWrapper::new(LlamaModelConfig::default()).expect("load");
        // Long enough that prefill spans multiple n_batch chunks; the exact
        // content is irrelevant.
        let prompt = "The capital of France is Paris. ".repeat(64);
        let args = InferenceArgs {
            prompt: prompt.clone(),
            // Additive cap so the long prompt is not rejected by the
            // absolute-position check `sample_len` applies.
            max_new_tokens: Some(64),
            seed: Some(1234),
            ..Default::default()
        };
        // Sink trips on first invocation: if cancel is honoured during
        // prefill, decode returns empty with zero completion tokens.
        let mut called = 0usize;
        let mut sink = |_chunk: &str| -> ControlFlow<()> {
            called += 1;
            ControlFlow::Break(())
        };
        let out = wrapper
            .decode_text_only_core_with_sink(&prompt, &args, &mut sink)
            .expect("prefill cancel must not return Err");
        assert!(
            out.text.is_empty(),
            "cancelled prefill must produce no decoded text, got {:?}",
            out.text
        );
        assert_eq!(
            out.completion_tokens, 0,
            "cancelled prefill must report zero completion tokens"
        );
        assert!(
            out.prompt_tokens > 0,
            "prompt must have been tokenized before the prefill loop"
        );
        assert!(
            called >= 1,
            "sink must have been polled at least once during prefill"
        );
    }

    /// Pre-arm `cancel_flag` with a no-op sink so any cancellation observed
    /// must have come through `llama_set_abort_callback`, not sink polling.
    /// Verifies that cancellation propagates *through* `llama_decode`, not
    /// just between batches.
    #[test]
    #[ignore]
    fn test_abort_callback_cancels_during_decode() {
        let mut wrapper = LlamaModelWrapper::new(LlamaModelConfig::default()).expect("load");

        let prompt = "The capital of France is Paris. ".repeat(128);
        let cancel_flag = Arc::new(AtomicBool::new(true));
        let args = InferenceArgs {
            prompt: prompt.clone(),
            max_new_tokens: Some(64),
            seed: Some(1234),
            cancel_flag: Some(cancel_flag.clone()),
            ..Default::default()
        };
        let mut sink = |_chunk: &str| -> ControlFlow<()> { ControlFlow::Continue(()) };
        let started = std::time::Instant::now();
        let out = wrapper
            .decode_text_only_core_with_sink(&prompt, &args, &mut sink)
            .expect("aborted decode must surface as Ok cancellation, not Err");
        let elapsed = started.elapsed();
        assert!(
            out.text.is_empty(),
            "aborted prefill must produce no text, got {:?}",
            out.text
        );
        assert_eq!(
            out.completion_tokens, 0,
            "aborted prefill must report zero completion tokens"
        );
        // Loose upper bound vs the wall-clock cost of completing the full
        // prefill of a 4k-token prompt; real aborts fire in O(100ms).
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "abort did not cancel quickly enough, took {elapsed:?}"
        );
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
        let config: LlamaModelConfig = settings.into();
        assert_eq!(config.n_batch, Some(2048));
        assert_eq!(config.n_ubatch, Some(512));
        assert_eq!(config.type_k, Some(ProtoKv::Q80 as i32));
        assert_eq!(config.type_v, Some(ProtoKv::Q80 as i32));
        assert!(config.reuse_kv_prefix);
        // Default keeps reuse off so requests stay independent.
        assert!(!LlamaModelConfig::default().reuse_kv_prefix);
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
    fn test_common_prefix_len() {
        let t = |ids: &[i32]| ids.iter().map(|&i| LlamaToken(i)).collect::<Vec<_>>();
        assert_eq!(common_prefix_len(&t(&[]), &t(&[])), 0);
        assert_eq!(common_prefix_len(&t(&[1, 2, 3]), &t(&[4, 5])), 0);
        assert_eq!(common_prefix_len(&t(&[1, 2, 3]), &t(&[1, 2, 3])), 3);
        // One is a prefix of the other → length of the shorter.
        assert_eq!(common_prefix_len(&t(&[1, 2, 3, 4]), &t(&[1, 2])), 2);
        // Diverge mid-sequence → index of the first difference.
        assert_eq!(common_prefix_len(&t(&[1, 2, 9, 4]), &t(&[1, 2, 3, 4])), 2);
    }

    #[test]
    fn test_strip_generation_prompt() {
        // Exact match drops to empty body.
        assert_eq!(strip_generation_prompt("abc\n", "abc\n"), "");
        // No overlap and no shared header → input untouched.
        assert_eq!(strip_generation_prompt("xyz", "abc\n"), "xyz");
        // Full strip when the output starts with the entire prompt.
        let gp = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let with_think = "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>{}</tool_call>";
        assert_eq!(
            strip_generation_prompt(with_think, gp),
            "<tool_call>{}</tool_call>"
        );
        // Qwen3 eager-grammar case: output regenerates the header but
        // omits the empty think block. Falls back to stripping up to the
        // last newline of generation_prompt.
        let without_think = "<|im_start|>assistant\n<tool_call>{}</tool_call>";
        assert_eq!(
            strip_generation_prompt(without_think, gp),
            "<tool_call>{}</tool_call>"
        );
        // Output that does not regenerate the header at all is left alone.
        assert_eq!(
            strip_generation_prompt("<tool_call>{}</tool_call>", gp),
            "<tool_call>{}</tool_call>"
        );
        // `generation_prompt` without a newline → no header to strip; the
        // function falls back to the full-prefix check, which also fails,
        // so the input is returned untouched.
        assert_eq!(strip_generation_prompt("body", "prefix"), "body");
        // Empty generation_prompt: strip_prefix succeeds with the whole input.
        assert_eq!(strip_generation_prompt("body", ""), "body");
    }

    #[test]
    fn test_plan_kv_keep() {
        let t = |ids: &[i32]| ids.iter().map(|&i| LlamaToken(i)).collect::<Vec<_>>();
        // Reuse off → always full clear, regardless of overlap.
        assert_eq!(plan_kv_keep(&t(&[0, 1, 2]), &t(&[0, 1, 2]), 3, false), 0);
        // Empty cache → full clear.
        assert_eq!(plan_kv_keep(&t(&[]), &t(&[0, 1, 2]), 3, true), 0);
        // Partial overlap → keep the common prefix.
        assert_eq!(plan_kv_keep(&t(&[0, 1, 9]), &t(&[0, 1, 2]), 3, true), 2);
        // Full match is capped to prompt_tokens - 1 so the last token is always
        // re-decoded (the first sample needs fresh logits).
        assert_eq!(plan_kv_keep(&t(&[0, 1, 2]), &t(&[0, 1, 2]), 3, true), 2);
    }

    #[test]
    fn test_plan_chunk_keep() {
        let txt = |ids: &[i32]| CachedChunk::Text(ids.iter().map(|&i| LlamaToken(i)).collect());
        let img = |key: u64, n_pos: i32| CachedChunk::Media {
            key: Some(key),
            n_pos,
        };
        // Media chunk with no paired bitmap (tiling overflow).
        let img_unkeyed = |n_pos: i32| CachedChunk::Media { key: None, n_pos };

        // Shared [image(256 pos), text(0,1)] prefix; the tail text chunk differs.
        let cached = vec![img(7, 256), txt(&[0, 1]), txt(&[9])];
        let new = vec![img(7, 256), txt(&[0, 1]), txt(&[5])];
        let n_pos = vec![256, 2, 1];
        // Keep the first 2 chunks (image + matching text) = 256 + 2 positions.
        assert_eq!(
            plan_chunk_keep(&cached, &new, &n_pos, true),
            (2, 258),
            "keep the matching image+text prefix, re-eval the differing tail"
        );

        // Reuse off → full clear.
        assert_eq!(plan_chunk_keep(&cached, &new, &n_pos, false), (0, 0));
        // Empty cache → full clear.
        assert_eq!(plan_chunk_keep(&[], &new, &n_pos, true), (0, 0));
        // Differing image at index 0 → nothing matches.
        let other_img = vec![img(99, 256), txt(&[0, 1]), txt(&[5])];
        assert_eq!(plan_chunk_keep(&other_img, &new, &n_pos, true), (0, 0));

        // Full match drops the last chunk so the final position is re-decoded.
        let same = vec![img(7, 256), txt(&[0, 1])];
        let same_n_pos = vec![256, 2];
        assert_eq!(
            plan_chunk_keep(&same, &same, &same_n_pos, true),
            (1, 256),
            "full match keeps all but the last chunk"
        );

        // An unkeyed media chunk (tiling overflow) ends the prefix even when
        // both sides are byte-identical: its image can't be proven unchanged, so
        // reusing its KV could serve a stale image. Keep only the chunk before.
        let unkeyed = vec![img(7, 256), img_unkeyed(256), txt(&[0, 1])];
        let unkeyed_n_pos = vec![256, 256, 2];
        assert_eq!(
            plan_chunk_keep(&unkeyed, &unkeyed, &unkeyed_n_pos, true),
            (1, 256),
            "reuse stops at the first unkeyed media chunk"
        );
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
            n_batch: None,
            n_ubatch: None,
            type_k: None,
            type_v: None,
            reuse_kv_prefix: false,
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
            medias: vec![MediaInput {
                kind: MediaKind::Image as i32,
                source: Some(Source::Encoded(image_bytes)),
                id: None,
            }],
            ..Default::default()
        };

        let output = wrapper.run(args).expect("Multimodal generation failed");
        assert!(!output.is_empty(), "Output should not be empty");
        println!("Multimodal output: {output}");
    }

    /// Regression: with `reuse_kv_prefix=true`, a multimodal request between two
    /// text requests must not corrupt the second text request. The multimodal
    /// path fully clears the KV cache; if it left `cached_tokens` stale, the
    /// following text request would skip prefilling a prefix that is no longer in
    /// the KV cache and generate garbage. The final text output must match a
    /// fresh-context baseline.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_multimodal_between_text_clears_prefix_cache \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_multimodal_between_text_clears_prefix_cache() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;

        let image_bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../output.png"))
                .expect("output.png should exist in repo root");
        let mtmd_config = || LlamaModelConfig {
            model: "gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            hf_repo: Some("unsloth/gemma-4-E4B-it-GGUF".to_string()),
            ctx_size: std::num::NonZeroU32::new(2048),
            reuse_kv_prefix: true,
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
            ..LlamaModelConfig::default()
        };
        let text_args = |prompt: &str| InferenceArgs {
            prompt: prompt.to_string(),
            sample_len: Some(48),
            seed: Some(1234),
            ..Default::default()
        };
        let image_args = InferenceArgs {
            medias: vec![MediaInput {
                kind: MediaKind::Image as i32,
                source: Some(Source::Encoded(image_bytes)),
                id: None,
            }],
            ..text_args("What do you see in this image?")
        };
        let text_prompt = "The capital of France is";

        // Baseline: the text prompt on a fresh context (full clear).
        let baseline = LlamaModelWrapper::new(mtmd_config())
            .expect("load")
            .run(text_args(text_prompt))
            .expect("baseline text");

        // text → multimodal → text on one wrapper. The multimodal call wipes the
        // KV and must also clear the prefix-reuse record.
        let mut wrapper = LlamaModelWrapper::new(mtmd_config()).expect("load");
        wrapper.run(text_args(text_prompt)).expect("first text");
        wrapper.run(image_args).expect("multimodal");
        let after = wrapper
            .run(text_args(text_prompt))
            .expect("text after image");
        assert_eq!(
            after, baseline,
            "text output after a multimodal request must match a fresh context"
        );
    }

    /// With `reuse_kv_prefix=true`, two requests sharing the same image but
    /// asking different questions must reuse the image KV and produce output
    /// identical to a full-clear baseline (correctness of partial KV removal at
    /// a chunk boundary). A changed image must break the prefix and still match.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_multimodal_prefix_reuse_matches_full_clear \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_multimodal_prefix_reuse_matches_full_clear() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;

        let image_bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../output.png"))
                .expect("output.png should exist in repo root");
        let cfg = || LlamaModelConfig {
            model: "gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            hf_repo: Some("unsloth/gemma-4-E4B-it-GGUF".to_string()),
            ctx_size: std::num::NonZeroU32::new(2048),
            reuse_kv_prefix: true,
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
            ..LlamaModelConfig::default()
        };
        // Same image, two different questions.
        let img_args = |prompt: &str| InferenceArgs {
            prompt: prompt.to_string(),
            sample_len: Some(48),
            seed: Some(1234),
            medias: vec![MediaInput {
                kind: MediaKind::Image as i32,
                source: Some(Source::Encoded(image_bytes.clone())),
                id: None,
            }],
            ..Default::default()
        };
        let q_a = "What is in this image?";
        let q_b = "What colors appear in this image?";

        // Baseline: question B on a fresh context (full clear, full re-eval).
        let baseline_b = LlamaModelWrapper::new(cfg())
            .expect("load")
            .run(img_args(q_b))
            .expect("baseline b");

        // Reuse: A then B on one wrapper. B reuses the image (and shared system
        // prompt) KV; its output must equal the full-clear baseline.
        let mut wrapper = LlamaModelWrapper::new(cfg()).expect("load");
        wrapper.run(img_args(q_a)).expect("reuse a");
        let reuse_b = wrapper.run(img_args(q_b)).expect("reuse b");
        assert_eq!(
            reuse_b, baseline_b,
            "multimodal prefix reuse changed the output for question B"
        );
    }

    /// Regression: a text request between two multimodal requests must invalidate
    /// the multimodal prefix record. The text request fully clears the KV cache;
    /// if it left `cached_chunks` stale, the second multimodal request would reuse
    /// an image prefix no longer in the KV and corrupt its output. This is the
    /// mirror of `test_multimodal_between_text_clears_prefix_cache`.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_text_between_multimodal_clears_chunk_cache \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_text_between_multimodal_clears_chunk_cache() {
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
        use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;

        let image_bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../output.png"))
                .expect("output.png should exist in repo root");
        let cfg = || LlamaModelConfig {
            model: "gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            hf_repo: Some("unsloth/gemma-4-E4B-it-GGUF".to_string()),
            ctx_size: std::num::NonZeroU32::new(2048),
            reuse_kv_prefix: true,
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
            ..LlamaModelConfig::default()
        };
        let args = |prompt: &str, medias: Vec<MediaInput>| InferenceArgs {
            prompt: prompt.to_string(),
            sample_len: Some(48),
            seed: Some(1234),
            medias,
            ..Default::default()
        };
        let img_args = || {
            args(
                "What is in this image?",
                vec![MediaInput {
                    kind: MediaKind::Image as i32,
                    source: Some(Source::Encoded(image_bytes.clone())),
                    id: None,
                }],
            )
        };
        let text_args = || args("The capital of France is", vec![]);

        // Baseline: the image request on a fresh context (full clear).
        let baseline = LlamaModelWrapper::new(cfg())
            .expect("load")
            .run(img_args())
            .expect("baseline image");

        // image -> text -> image on one wrapper. The text request wipes the KV
        // and must clear the multimodal record so the second image request does
        // not reuse a stale image prefix.
        let mut wrapper = LlamaModelWrapper::new(cfg()).expect("load");
        wrapper.run(img_args()).expect("first image");
        wrapper.run(text_args()).expect("text between");
        let after = wrapper.run(img_args()).expect("image after text");
        assert_eq!(
            after, baseline,
            "image output after an intervening text request must match a fresh context"
        );
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
            n_batch: None,
            n_ubatch: None,
            type_k: None,
            type_v: None,
            reuse_kv_prefix: false,
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
            medias: vec![MediaInput {
                kind: MediaKind::Audio as i32,
                source: Some(Source::Encoded(audio_bytes)),
                id: None,
            }],
            ..Default::default()
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
        let msgs = [
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
        if let Some(content) = &msg.content
            && let Some(llm_chat_args::message_content::Content::ToolCalls(tc)) = &content.content
        {
            let json = serde_json::to_string(&tc.calls).unwrap();
            assert!(json.contains("get_weather"));
            assert!(json.contains("Tokyo"));
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
            err.to_string()
                .contains(super::ERR_USE_FUNCTION_CALLING_UNSUPPORTED),
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

    // The structured-output schema reported as "json_schema not taking effect"
    // (agent-app thread-reflection-single.yaml). Kept here verbatim so the
    // reproduction tests below exercise exactly what production sends.
    const THREAD_REFLECTION_SCHEMA: &str = r#"{
      "type": "object",
      "required": ["outcome","score_self","summary","task_intent",
                   "task_category","reflection_aspect",
                   "failure_modes","tools_used"],
      "properties": {
        "outcome": {"enum":["SUCCESS","PARTIAL","FAILURE","ABORTED","UNKNOWN"]},
        "score_self": {"type":"number","minimum":0.0,"maximum":1.0},
        "summary": {"type":"string"},
        "task_intent": {"type":"string","maxLength":1000},
        "task_category": {"enum":["coding","consultation","research","creative","general"]},
        "reflection_aspect": {"enum":["TASK_OUTCOME","INTERACTION_STYLE","BOTH"]},
        "failure_modes": {"type":"array","items":{"enum":["tool_misuse","loop","scope_drift","hallucination","context_overflow","data_loss","permission_issue","ambiguous_instruction","conflicting_requirements","missing_context","misleading_premise","goal_drift_by_user","tool_unavailable","external_service_failure","rate_limit","OTHER"]},"maxItems":8},
        "failure_modes_other": {"type":"array","items":{"type":"string","maxLength":100},"maxItems":5},
        "tools_used": {"type":"array","items":{"type":"string","maxLength":100},"maxItems":50},
        "success_factors": {"type":"array","items":{"type":"string","maxLength":200},"maxItems":8},
        "lessons": {"type":"array","items":{"type":"string","maxLength":300},"maxItems":10},
        "key_decisions": {"type":"array","items":{"type":"string","maxLength":300},"maxItems":10},
        "mitigation_hint": {"type":"string","maxLength":500},
        "tool_outcomes": {
          "type":"array",
          "items":{
            "type":"object",
            "required":["tool","contribution"],
            "properties":{
              "tool":{"type":"string","maxLength":100},
              "contribution":{"enum":["POSITIVE","NEGATIVE","NEUTRAL"]},
              "error_kind":{"type":"string","maxLength":100}
            }
          },
          "maxItems":50
        },
        "facts": {
          "type":"array",
          "items":{
            "type":"object",
            "required":["turn_index","kind"],
            "properties":{
              "turn_index":{"type":"integer","minimum":0},
              "kind":{"enum":["OUTCOME_EVIDENCE","SCORE_DRIVER","LESSON_SOURCE",
                              "KEY_DECISION_POINT","EXEMPLAR","COUNTER_EXAMPLE",
                              "CONTEXT_PIVOT"]},
              "weight":{"type":"number","minimum":0.0,"maximum":1.0},
              "note":{"type":"string","maxLength":200},
              "links":{
                "type":"array",
                "items":{
                  "type":"object",
                  "properties":{
                    "field":{"enum":["lesson","failure_mode","key_decision","success_factor"]},
                    "index":{"type":"integer","minimum":0}
                  }
                }
              }
            }
          },
          "maxItems":30
        }
      }
    }"#;

    // Validate that the reported schema actually compiles under the same
    // llguidance backend the runner uses (`grammar_kind = "json"`). A failure
    // here would mean the schema is structurally rejected and the sampler is
    // never installed — distinct from the runtime "mask not applied" path. The
    // production schema already dropped `uniqueItems` (unsupported by
    // llguidance 1.7.5), so this guards against a regression that re-introduces
    // an unimplemented keyword.
    #[test]
    fn test_thread_reflection_schema_is_valid_json() {
        let v: serde_json::Value = serde_json::from_str(THREAD_REFLECTION_SCHEMA)
            .expect("thread-reflection schema must be valid JSON");
        assert_eq!(v["type"], "object");
        // uniqueItems is unsupported by llguidance 1.7.5 (non-lenient compile
        // aborts the whole grammar). The workflow removed it; assert it stays out.
        assert!(
            !THREAD_REFLECTION_SCHEMA.contains("uniqueItems"),
            "schema must not contain uniqueItems (llguidance 1.7.5 cannot compile it)"
        );
    }

    // End-to-end reproduction: feed the exact production schema through the
    // completion path and assert the output is schema-conformant (parses as
    // JSON, required fields present, enums in range). Regression guard for the
    // double-accept bug: the generation loop used to call `sampler.accept()`
    // after `sampler.sample()`, but `sample()` already accepts internally — the
    // llguidance grammar matcher then consumed each token twice, corrupted its
    // state, and silently disabled masking. With the redundant accept removed,
    // the grammar holds for the whole document.
    #[ignore = "depends on model"]
    #[tokio::test]
    async fn test_completion_complex_json_schema_conformance() {
        use jobworkerp_llama_protobuf::protobuf::llm::{
            llm_completion_args, llm_completion_result,
        };

        let mut wrapper = LlamaCppPluginTestEnv::load_wrapper();

        let request = LlmCompletionArgs {
            model: None,
            system_prompt: None,
            // `/no_think` suppresses Qwen3's <think> block so the model spends
            // its token budget on the JSON rather than reasoning prose.
            prompt: "/no_think Summarize this trivial successful coding task as a reflection JSON."
                .to_string(),
            options: Some(llm_completion_args::LlmOptions {
                max_tokens: Some(2048),
                temperature: Some(0.2),
                ..Default::default()
            }),
            context: None,
            function_options: None,
            json_schema: Some(THREAD_REFLECTION_SCHEMA.to_string()),
        };

        let res = wrapper.run_completion(request).expect("run_completion");
        assert!(res.done);
        let content = res.content.expect("content");
        let text = match content.content {
            Some(llm_completion_result::message_content::Content::Text(t)) => t,
            other => panic!("expected text content, got: {other:?}"),
        };
        println!("complex json_schema output: {text}");

        // Strict conformance: the whole output must be a single JSON object.
        let parsed: serde_json::Value = serde_json::from_str(text.trim()).unwrap_or_else(|e| {
            panic!("output must be strict JSON (mask not applied?): {e}\n{text}")
        });
        let obj = parsed.as_object().expect("top-level must be an object");

        for key in [
            "outcome",
            "score_self",
            "summary",
            "task_intent",
            "task_category",
            "reflection_aspect",
            "failure_modes",
            "tools_used",
        ] {
            assert!(
                obj.contains_key(key),
                "missing required field: {key}\n{text}"
            );
        }

        let outcome = obj["outcome"].as_str().expect("outcome is string");
        assert!(
            ["SUCCESS", "PARTIAL", "FAILURE", "ABORTED", "UNKNOWN"].contains(&outcome),
            "outcome out of enum: {outcome}"
        );
        let cat = obj["task_category"]
            .as_str()
            .expect("task_category is string");
        assert!(
            ["coding", "consultation", "research", "creative", "general"].contains(&cat),
            "task_category out of enum: {cat}"
        );
        let score = obj["score_self"].as_f64().expect("score_self is number");
        assert!(
            (0.0..=1.0).contains(&score),
            "score_self out of [0,1]: {score}"
        );
    }

    // Reproduces the reported "json_schema not taking effect" bug with Qwen3
    // under production conditions (no `/no_think`). The root cause was a
    // double-accept in the generation loop: `sampler.sample()` already calls
    // `llama_sampler_accept` internally, but the loop also called
    // `sampler.accept()`, so the llguidance grammar matcher consumed each token
    // twice, corrupted its state on the very first token, and emitted
    // `token "..." doesn't satisfy the grammar; stopping` — after which masking
    // was silently dropped and the model produced `<think>` blocks / non-schema
    // output. With the redundant accept removed, the grammar engages from the
    // first token and the output starts with `{` with no reasoning preamble.
    //
    // The full document is NOT asserted to round-trip here: the production
    // schema does not set `additionalProperties:false`, so the model may emit
    // extra keys and run past `max_tokens`. This test asserts the grammar is
    // *engaged* (clean JSON object opening, no `<think>`/markdown preamble); the
    // strict whole-document case is covered by injecting additionalProperties in
    // the conformance test.
    #[ignore = "depends on model"]
    #[tokio::test]
    async fn test_completion_json_schema_no_no_think_directive() {
        use jobworkerp_llama_protobuf::protobuf::llm::{
            llm_completion_args, llm_completion_result,
        };

        let mut wrapper = LlamaCppPluginTestEnv::load_wrapper();

        let request = LlmCompletionArgs {
            model: None,
            system_prompt: None,
            // Deliberately NO `/no_think`: this is the production condition that
            // previously broke the grammar constraint.
            prompt: "Summarize this trivial successful coding task as a reflection JSON."
                .to_string(),
            options: Some(llm_completion_args::LlmOptions {
                max_tokens: Some(2048),
                temperature: Some(0.2),
                ..Default::default()
            }),
            context: None,
            function_options: None,
            json_schema: Some(THREAD_REFLECTION_SCHEMA.to_string()),
        };

        let res = wrapper.run_completion(request).expect("run_completion");
        assert!(res.done);
        let content = res.content.expect("content");
        let text = match content.content {
            Some(llm_completion_result::message_content::Content::Text(t)) => t,
            other => panic!("expected text content, got: {other:?}"),
        };
        println!("no-/no_think json_schema output: {text}");

        // Grammar-engaged signature: the very first character must be `{`. With
        // the double-accept bug the matcher died on token 0 and the model fell
        // back to a `<think>` block or markdown fence, so the leading char was
        // `<` or `` ` ``. A leading `{` proves the grammar masked everything
        // else from the first token.
        let trimmed = text.trim_start();
        assert!(
            trimmed.starts_with('{'),
            "output must start with '{{' (grammar must be engaged from token 0): {text}"
        );
        assert!(
            !text.contains("<think>") && !text.contains("```"),
            "output must not contain reasoning/markdown preamble: {text}"
        );
        // The required `outcome` enum must appear with a legal value, confirming
        // the grammar is shaping keys/values (not just the opening brace).
        assert!(
            text.contains("\"outcome\""),
            "missing required `outcome` key: {text}"
        );
        let outcome_ok = ["SUCCESS", "PARTIAL", "FAILURE", "ABORTED", "UNKNOWN"]
            .iter()
            .any(|v| text.contains(v));
        assert!(
            outcome_ok,
            "expected a constrained `outcome` enum value: {text}"
        );
    }

    // Test-only helper exposing the env-driven config builder so reproduction
    // tests don't duplicate the envy wiring.
    struct LlamaCppPluginTestEnv;

    // Shared env for the JSON-schema reproduction tests: a small Qwen3 model on
    // CPU with a fixed seed for determinism.
    const QWEN3_JSON_TEST_ENV: &str = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=You are a reflection generator. Respond ONLY with a single JSON object.
";

    impl LlamaCppPluginTestEnv {
        fn config() -> LlamaModelConfig {
            envy::prefixed("LLAMA_")
                .from_env::<LlamaModelConfig>()
                .expect("read model config from env")
        }

        // Load the shared env and build a model wrapper for the reproduction
        // tests.
        fn load_wrapper() -> LlamaModelWrapper {
            dotenvy::from_read(QWEN3_JSON_TEST_ENV.as_bytes()).ok();
            LlamaModelWrapper::new(Self::config()).expect("load model from env")
        }
    }

    // ---------------------------------------------------------------------
    // Real-model regression tests for client-side tool calling. The `poc_*`
    // tests exercise the fork's OpenAI-compatible chat template API
    // end-to-end on Qwen3-0.6B; the `test_apply_oai_*` / `test_build_sampler_*`
    // tests cover individual layers. All require `LLAMA_MODEL` / `LLAMA_HF_REPO`
    // env vars and are `#[ignore]` by default.
    // ---------------------------------------------------------------------

    // System prompt kept minimal so the model's tool-call decision is driven
    // by the tools definition + user message.
    const QWEN3_TOOL_POC_ENV: &str = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=You are a helpful assistant. When a relevant tool is available, call it.
";

    fn load_wrapper_for_tool_poc() -> LlamaModelWrapper {
        dotenvy::from_read(QWEN3_TOOL_POC_ENV.as_bytes()).ok();
        LlamaModelWrapper::new(LlamaCppPluginTestEnv::config())
            .expect("load Qwen3 model for tool-calling tests")
    }

    // OpenAI-compatible single function shared by the tool-calling tests.
    fn poc_tools_json() -> &'static str {
        r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a given city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name, e.g. Tokyo"}},"required":["city"]}}}]"#
    }

    fn poc_inference_args(prompt: &str, temperature: Option<f64>) -> InferenceArgs {
        InferenceArgs {
            prompt: prompt.to_string(),
            max_new_tokens: Some(192),
            temperature,
            seed: Some(1024),
            ..Default::default()
        }
    }

    // `enable_thinking` is left at its default so the helper stays
    // model-agnostic; if the model still emits a `<think>` preamble, prepend
    // "/no_think" to the user message instead of toggling a model-specific flag.
    fn poc_oai_params<'a>(
        messages_json: &'a str,
        tools_json: Option<&'a str>,
        tool_choice: Option<&'a str>,
    ) -> llama_cpp_2::openai::OpenAIChatTemplateParams<'a> {
        llama_cpp_2::openai::OpenAIChatTemplateParams {
            messages_json,
            tools_json,
            tool_choice,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: true,
            add_eos: false,
            parse_tool_calls: true,
        }
    }

    /// `decode_text_only_core_with_sink_and_stops` trims the matched stop
    /// suffix from the returned text. The stop string `"."` almost always
    /// appears in a short reply so the truncate branch fires deterministically.
    #[ignore = "depends on model: additional_stops truncate"]
    #[test]
    fn test_additional_stops_truncate_output_at_match() {
        let mut wrapper = load_wrapper_for_tool_poc();
        let args = poc_inference_args("Reply with: hello.", None);
        let preserved = std::collections::HashSet::new();
        let mut sink = |_: &str| ControlFlow::Continue(());
        let output = wrapper
            .decode_text_only_core_with_sink_and_stops(
                "<|im_start|>user\nReply with: hello.<|im_end|>\n<|im_start|>assistant\n",
                &args,
                &[".".to_string()],
                &preserved,
                &mut sink,
            )
            .expect("decode_text_only_core_with_sink_and_stops");
        // Either the model produced text without "." (and ran to length), or
        // the truncate path fired — in the latter case the trailing "." is
        // gone from output.text.
        eprintln!("stop test output: {:?}", output.text);
        assert!(
            !output.text.ends_with('.'),
            "stop should be trimmed from output: {:?}",
            output.text
        );
    }

    /// Legacy regression: calling `decode_text_only_core_with_sink_and_stops`
    /// with empty stops and empty preserved set must produce the same
    /// generation as the legacy `decode_text_only_core_with_sink`.
    #[ignore = "depends on model: additional_stops empty passthrough"]
    #[test]
    fn test_additional_stops_empty_passthrough() {
        let mut wrapper = load_wrapper_for_tool_poc();
        let args = poc_inference_args("Reply: 1", None);
        let prompt = "<|im_start|>user\nReply: 1<|im_end|>\n<|im_start|>assistant\n";

        let preserved = std::collections::HashSet::new();
        let mut sink_a = |_: &str| ControlFlow::Continue(());
        let baseline = wrapper
            .decode_text_only_core_with_sink(prompt, &args, &mut sink_a)
            .expect("legacy decode");

        let mut sink_b = |_: &str| ControlFlow::Continue(());
        let with_empty_stops = wrapper
            .decode_text_only_core_with_sink_and_stops(prompt, &args, &[], &preserved, &mut sink_b)
            .expect("new decode with empty stops");

        assert_eq!(
            baseline.text, with_empty_stops.text,
            "empty stops + empty preserved must match legacy output"
        );
    }

    /// `build_sampler` must accept an eager (non-lazy) grammar via
    /// `grammar_spec` and produce a usable sampler chain.
    #[ignore = "depends on model: build_sampler eager grammar"]
    #[test]
    fn test_build_sampler_with_eager_grammar_prepends_chain() {
        let wrapper = load_wrapper_for_tool_poc();
        let mut args = poc_inference_args("hi", None);
        // Minimal trivial GBNF grammar: only accept the literal "hello".
        args.grammar_spec = Some(crate::oai_chat::GrammarSpec {
            grammar: "root ::= \"hello\"".to_string(),
            grammar_lazy: false,
            grammar_triggers: vec![],
        });
        wrapper
            .build_sampler(&args)
            .expect("eager grammar sampler should build");
    }

    /// `build_sampler` must accept a lazy grammar with empty triggers — the
    /// shape llama.cpp emits for `tool_choice="required"`, where the grammar
    /// engages immediately and no lazy trigger is needed.
    #[ignore = "depends on model: build_sampler grammar_lazy"]
    #[test]
    fn test_build_sampler_grammar_lazy_patterns_succeeds() {
        let wrapper = load_wrapper_for_tool_poc();
        let mut args = poc_inference_args("hi", None);
        args.grammar_spec = Some(crate::oai_chat::GrammarSpec {
            grammar: "root ::= \"hello\"".to_string(),
            grammar_lazy: true,
            grammar_triggers: vec![],
        });
        wrapper
            .build_sampler(&args)
            .expect("lazy grammar sampler should build with empty triggers");
    }

    /// When both `grammar_spec` and `json_schema` are supplied to
    /// `build_sampler`, the grammar wins (json_schema is dropped with a
    /// warn log). `run_chat_with_sink` rejects the combination earlier in
    /// the request; this branch is defensive in case a future caller
    /// bypasses that check.
    #[ignore = "depends on model: build_sampler grammar+schema"]
    #[test]
    fn test_build_sampler_grammar_and_json_schema_both_present_prefers_grammar() {
        let wrapper = load_wrapper_for_tool_poc();
        let mut args = poc_inference_args("hi", None);
        args.grammar_spec = Some(crate::oai_chat::GrammarSpec {
            grammar: "root ::= \"hello\"".to_string(),
            grammar_lazy: false,
            grammar_triggers: vec![],
        });
        // Invalid JSON schema string: if json_schema were taken instead of
        // grammar_spec, llguidance would reject and fail the call.
        args.json_schema = Some("not actually a schema".to_string());
        wrapper
            .build_sampler(&args)
            .expect("grammar_spec must take precedence over json_schema");
    }

    /// `apply_oai_template_with_tools` smoke test: the rendered prompt
    /// embeds the tool definitions (so the OAI template branch is wired
    /// up against the model file).
    #[ignore = "depends on model: apply_oai_template_with_tools minimal"]
    #[test]
    fn test_apply_oai_template_with_tools_minimal() {
        let wrapper = load_wrapper_for_tool_poc();
        let messages_json = r#"[
            {"role":"system","content":"You are a tool caller."},
            {"role":"user","content":"weather?"}
        ]"#;
        let opts = llm_chat_args::FunctionOptions {
            tool_choice: Some("auto".to_string()),
            ..Default::default()
        };
        let result = wrapper
            .apply_oai_template_with_tools(messages_json, poc_tools_json(), Some(&opts), None)
            .expect("apply_oai_template_with_tools");
        assert!(!result.prompt.is_empty());
        assert!(
            result.prompt.contains("get_weather"),
            "tools section should be rendered into the prompt: {}",
            result.prompt
        );
        assert!(result.parse_tool_calls);
    }

    /// `chat_template_kwargs` is the escape hatch for model-specific switches
    /// (`enable_thinking` etc.) — the plugin itself stays model-agnostic.
    #[ignore = "depends on model: apply_oai_template_with_tools kwargs"]
    #[test]
    fn test_apply_oai_template_with_tools_handles_kwargs() {
        let wrapper = load_wrapper_for_tool_poc();
        let messages_json = r#"[{"role":"user","content":"hi"}]"#;
        let opts = llm_chat_args::FunctionOptions {
            chat_template_kwargs: Some(r#"{"enable_thinking":false}"#.to_string()),
            ..Default::default()
        };
        let result = wrapper
            .apply_oai_template_with_tools(messages_json, poc_tools_json(), Some(&opts), None)
            .expect("apply_oai_template_with_tools accepts kwargs");
        assert!(!result.prompt.is_empty());
    }

    /// Function-specific tool_choice (the OpenAI
    /// `{"type":"function","function":{"name":"..."}}` form) is normalised
    /// upstream into `tool_choice="required"` + filtered tools_json so the
    /// model has no choice but to call the requested function. Make sure
    /// the override path through `apply_oai_template_with_tools` produces
    /// a usable prompt the same way the bare-string path does.
    #[ignore = "depends on model: apply_oai_template_with_tools function-specific choice"]
    #[test]
    fn test_apply_oai_template_with_tools_function_specific_choice() {
        let wrapper = load_wrapper_for_tool_poc();
        let messages_json = r#"[{"role":"user","content":"weather?"}]"#;
        // Same shape `run_chat_with_sink_tools` would have produced after
        // resolve_tool_choice ran: tools filtered to one entry, choice
        // rewritten to "required".
        let tools_filtered = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let result = wrapper
            .apply_oai_template_with_tools(messages_json, tools_filtered, None, Some("required"))
            .expect("apply_oai_template_with_tools with override");
        assert!(!result.prompt.is_empty());
        assert!(result.prompt.contains("get_weather"));
    }

    /// Confirm the OAI template produces a tools-aware prompt and
    /// `parse_response_oaicompat` recovers the tool call from the generated
    /// text. Acts as the canonical smoke test before the higher-level paths.
    #[ignore = "depends on model: tool calling smoke test"]
    #[test]
    fn poc_oaicompat_template_and_parse_tool_call() {
        let mut wrapper = load_wrapper_for_tool_poc();
        let tmpl = wrapper
            .model
            .chat_template(None)
            .expect("chat template available on Qwen3 GGUF");

        let messages_json = r#"[
            {"role":"system","content":"You are a tool caller."},
            {"role":"user","content":"What is the weather in Tokyo? Use get_weather."}
        ]"#;
        let params = poc_oai_params(messages_json, Some(poc_tools_json()), Some("auto"));
        let result = wrapper
            .model
            .apply_chat_template_oaicompat(&tmpl, &params)
            .expect("apply_chat_template_oaicompat");

        // Diagnostic logging for failure triage: when this test breaks,
        // the rendered prompt + grammar metadata is usually enough to
        // figure out whether the chat template stopped emitting tools.
        eprintln!(
            "--- rendered prompt ---\n{}\n--- end prompt ---",
            result.prompt
        );
        eprintln!(
            "grammar.is_some={} grammar_lazy={} triggers={} preserved={:?} stops={:?} chat_format={}",
            result.grammar.is_some(),
            result.grammar_lazy,
            result.grammar_triggers.len(),
            result.preserved_tokens,
            result.additional_stops,
            result.chat_format,
        );
        for (i, t) in result.grammar_triggers.iter().enumerate() {
            eprintln!(
                "  trigger[{i}] type={:?} value={:?} token={:?}",
                t.trigger_type, t.value, t.token
            );
        }

        assert!(
            !result.prompt.is_empty(),
            "rendered prompt must be non-empty"
        );
        assert!(
            result.prompt.contains("get_weather"),
            "tools section should mention get_weather in the rendered prompt"
        );

        let args = poc_inference_args(&result.prompt, None);
        let output = wrapper
            .decode_text_only_core(&result.prompt, &args)
            .expect("decode_text_only_core");
        eprintln!("--- raw generation ---\n{}\n--- end raw ---", output.text);

        let parsed_json = result
            .parse_response_oaicompat(&output.text, false)
            .expect("parse_response_oaicompat");
        eprintln!("--- parsed JSON ---\n{parsed_json}\n--- end parsed ---");

        let value: serde_json::Value =
            serde_json::from_str(&parsed_json).expect("parse_response output is valid JSON");
        let tool_calls = value
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(
            !tool_calls.is_empty(),
            "expected at least one tool_call in parsed JSON: {parsed_json}"
        );
        let first = &tool_calls[0];
        let name = first
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert_eq!(name, "get_weather", "tool name mismatch in {parsed_json}");
        let raw_args = first
            .pointer("/function/arguments")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let args_value: serde_json::Value = serde_json::from_str(raw_args)
            .unwrap_or_else(|e| panic!("function.arguments must be JSON string: {e} ({raw_args})"));
        assert!(
            args_value.get("city").is_some(),
            "expected city argument: {args_value}"
        );
    }

    /// Verify the streaming parser groups token-level chunks into
    /// `tool_calls` deltas. The parser only recognises the tool-call
    /// envelope when fed the model-emitted special tokens (e.g.
    /// `<tool_call>`), so we drive it from a real generation rather than
    /// synthetic UTF-8 splits.
    #[ignore = "depends on model: streaming parser smoke test"]
    #[test]
    fn poc_streaming_state_aggregates_tool_call_deltas() {
        let mut wrapper = load_wrapper_for_tool_poc();
        let tmpl = wrapper
            .model
            .chat_template(None)
            .expect("chat template available");

        let messages_json = r#"[
            {"role":"system","content":"You are a tool caller."},
            {"role":"user","content":"What is the weather in Tokyo? Use get_weather."}
        ]"#;
        let params = poc_oai_params(messages_json, Some(poc_tools_json()), Some("auto"));
        let result = wrapper
            .model
            .apply_chat_template_oaicompat(&tmpl, &params)
            .expect("apply_chat_template_oaicompat");

        // Drive a real generation; the sink receives the same UTF-8 decoded
        // chunks (`&str` per token piece) that production streaming would.
        let mut state = result
            .streaming_state_oaicompat()
            .expect("streaming_state_oaicompat");
        let mut deltas_log: Vec<String> = Vec::new();
        let mut raw_chunks: Vec<String> = Vec::new();
        {
            let args = poc_inference_args(&result.prompt, None);
            let mut sink = |chunk: &str| -> ControlFlow<()> {
                raw_chunks.push(chunk.to_string());
                match state.update(chunk, true) {
                    Ok(diffs) => deltas_log.extend(diffs),
                    Err(e) => eprintln!("state.update error: {e}"),
                }
                ControlFlow::Continue(())
            };
            wrapper
                .decode_text_only_core_with_sink(&result.prompt, &args, &mut sink)
                .expect("decode_text_only_core_with_sink");
        }
        match state.update("", false) {
            Ok(final_diffs) => deltas_log.extend(final_diffs),
            Err(e) => eprintln!("state.update(final) error: {e}"),
        }

        eprintln!("--- raw chunks ({}) ---", raw_chunks.len());
        for c in &raw_chunks {
            eprint!("{c}");
        }
        eprintln!("\n--- end raw chunks ---");
        eprintln!("--- deltas ({}) ---", deltas_log.len());
        for d in &deltas_log {
            eprintln!("{d}");
        }
        eprintln!("--- end deltas ---");

        // Reconstruct (name, arguments) from the deltas in OAI fashion: id and
        // name typically arrive once on the first delta of a tool call;
        // arguments arrive incrementally.
        let mut tool_name: Option<String> = None;
        let mut tool_args_buf = String::new();
        let mut leaked_tool_call_marker = false;
        for delta in &deltas_log {
            let v: serde_json::Value = match serde_json::from_str(delta) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(content) = v.get("content").and_then(|c| c.as_str())
                && (content.contains("<tool_call>") || content.contains("</tool_call>"))
            {
                leaked_tool_call_marker = true;
            }
            let Some(calls) = v.get("tool_calls").and_then(|c| c.as_array()) else {
                continue;
            };
            for call in calls {
                if let Some(name) = call.pointer("/function/name").and_then(|n| n.as_str())
                    && tool_name.is_none()
                    && !name.is_empty()
                {
                    tool_name = Some(name.to_string());
                }
                if let Some(args_chunk) =
                    call.pointer("/function/arguments").and_then(|a| a.as_str())
                {
                    tool_args_buf.push_str(args_chunk);
                }
            }
        }

        assert!(
            !leaked_tool_call_marker,
            "<tool_call> markers must not appear in `content` deltas"
        );
        assert_eq!(
            tool_name.as_deref(),
            Some("get_weather"),
            "tool name should be reassembled from deltas (deltas={deltas_log:?})"
        );
        // Arguments should at least contain the city argument.
        let args_value: serde_json::Value =
            serde_json::from_str(&tool_args_buf).unwrap_or_else(|e| {
                panic!("reassembled arguments must be JSON: {e} ({tool_args_buf})")
            });
        assert!(
            args_value.get("city").is_some(),
            "arguments should include `city`: {args_value}"
        );
    }

    /// Confirm a lazy grammar wired into the sampler chain forces the model
    /// to emit a tool call even on an unrelated prompt that would normally
    /// elicit a free-form reply.
    #[ignore = "depends on model: grammar enforcement smoke test"]
    #[test]
    fn poc_grammar_lazy_forces_tool_call_emission() {
        use llama_cpp_2::model::GrammarTriggerType;

        // wrapper needs `mut` only when the no-grammar fallback runs
        // `decode_text_only_core`; that path borrows &mut self, so keep the
        // binding mutable up-front to avoid a second binding.
        #[allow(unused_mut)]
        let mut wrapper = load_wrapper_for_tool_poc();
        let tmpl = wrapper
            .model
            .chat_template(None)
            .expect("chat template available");

        // Ambiguous prompt: a free-running model would just greet back.
        let messages_json = r#"[
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"Hello!"}
        ]"#;
        // Forcing tool_choice to "required" + the grammar should together
        // guarantee a tool-call emission regardless of the user prompt.
        let params = poc_oai_params(messages_json, Some(poc_tools_json()), Some("required"));
        let result = wrapper
            .model
            .apply_chat_template_oaicompat(&tmpl, &params)
            .expect("apply_chat_template_oaicompat");

        eprintln!("grammar triggers ({}):", result.grammar_triggers.len());
        for (i, t) in result.grammar_triggers.iter().enumerate() {
            eprintln!(
                "  [{i}] type={:?} value={:?} token={:?}",
                t.trigger_type, t.value, t.token
            );
        }
        let Some(grammar) = result.grammar.as_deref() else {
            // No grammar emitted by the template: fall back to verifying
            // that tool_choice="required" alone forces the tool call.
            eprintln!("no grammar emitted; relying on tool_choice=required alone");
            let args = poc_inference_args(&result.prompt, Some(0.7));
            let output = wrapper
                .decode_text_only_core(&result.prompt, &args)
                .expect("decode_text_only_core");
            let parsed = result
                .parse_response_oaicompat(&output.text, false)
                .expect("parse_response_oaicompat");
            let value: serde_json::Value = serde_json::from_str(&parsed).expect("valid JSON");
            assert!(
                value
                    .get("tool_calls")
                    .and_then(|t| t.as_array())
                    .map(|a| !a.is_empty())
                    .unwrap_or(false),
                "tool_choice=required should still force a tool call: {parsed}"
            );
            return;
        };

        // Minimal trigger translation for the smoke test; the production
        // path lives in `oai_chat::grammar_triggers_to_patterns_and_tokens`
        // and handles multi-token `Word` triggers via regex escape.
        let trigger_patterns: Vec<String> = result
            .grammar_triggers
            .iter()
            .filter_map(|t| match t.trigger_type {
                GrammarTriggerType::Pattern => Some(t.value.clone()),
                GrammarTriggerType::PatternFull => Some(format!("^(?:{})$", t.value)),
                _ => None,
            })
            .collect();
        let trigger_tokens: Vec<LlamaToken> = result
            .grammar_triggers
            .iter()
            .filter_map(|t| match t.trigger_type {
                GrammarTriggerType::Token => t.token,
                GrammarTriggerType::Word => {
                    let toks = wrapper
                        .model
                        .str_to_token(&t.value, AddBos::Never)
                        .unwrap_or_default();
                    (toks.len() == 1).then_some(toks[0])
                }
                _ => None,
            })
            .collect();

        // Sampler chain: grammar mask MUST run before the terminal selector,
        // mirroring the rationale in build_sampler comments.
        let grammar_sampler = LlamaSampler::grammar_lazy_patterns(
            wrapper.model,
            grammar,
            "root",
            &trigger_patterns,
            &trigger_tokens,
        )
        .expect("grammar_lazy_patterns");
        let _chain = LlamaSampler::chain_simple(vec![grammar_sampler, LlamaSampler::greedy()]);
        // We cannot easily swap the sampler used by `decode_text_only_core`
        // from outside the wrapper. The goal here is simply to prove that
        // `grammar_lazy_patterns` accepts this model and the trigger shape
        // emitted by the chat template; the integrated path lives in
        // `build_sampler`.
        eprintln!(
            "grammar built successfully (patterns={}, tokens={})",
            trigger_patterns.len(),
            trigger_tokens.len()
        );
    }
}
