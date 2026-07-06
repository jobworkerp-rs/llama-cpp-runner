use anyhow::{Context, Result, anyhow, bail};
use jobworkerp_llama_protobuf::protobuf::llm::{
    LlmChatArgs, LlmChatResult, LlmCompletionArgs, LlmCompletionResult, llm_chat_args,
    llm_chat_result, llm_completion_result,
};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    speculative::MtpSpeculative,
    token::LlamaToken,
};
use llama_cpp_sys_2::{LLAMA_FLASH_ATTN_TYPE_DISABLED, LLAMA_FLASH_ATTN_TYPE_ENABLED};
use mtmd_support::{MediaLimits, MtmdRuntime};
use std::{
    ffi::CString,
    num::NonZeroU32,
    ops::ControlFlow,
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

mod config;
mod decode;
mod helpers;
#[cfg(test)]
mod test_support;
mod wrapper;

// Re-glob so sibling submodules (`use super::*`) see the extracted items unqualified.
use config::*;
use helpers::*;

pub(crate) use config::{
    ERR_CLIENT_TOOLS_WITH_FUNCTION_CALLING, ERR_CLIENT_TOOLS_WITH_JSON_SCHEMA,
    ERR_TOOL_EXECUTION_REQUESTS_REJECTED, ERR_USE_FUNCTION_CALLING_UNSUPPORTED,
};
// `InferenceArgs` is re-exported because it appears in the public signature of
// `LlamaModelWrapper::run`; pre-split it lived at `crate::model::InferenceArgs`
// and external callers may name it that way.
pub use config::{InferenceArgs, LlamaModelConfig};
pub use wrapper::LlamaModelWrapper;
