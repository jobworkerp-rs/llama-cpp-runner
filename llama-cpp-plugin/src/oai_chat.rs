//! OpenAI-compatible helpers for the client-side tool-calling path.
//!
//! This module is intentionally kept separate from `model.rs`: the legacy
//! `build_chat_messages` path (no tools) continues to build `LlamaChatMessage`
//! and let llama.cpp apply the chat template the legacy way, while the
//! tool-aware path defined here renders the supported templates in Rust from a
//! JSON-encoded messages array structured in OpenAI format.
//!
//! The implementation is split across three submodules:
//! - [`common`]: model-family-agnostic shared types, OAI/wire conversions, the
//!   GBNF/stream helpers, and the `OaiStreamParser` dispatcher.
//! - [`qwen`]: the Qwen3.5 tagged tool-call format.
//! - [`gemma4`]: the Gemma4 tool-call format.
//!
//! `model.rs`/`lib.rs` only reach into this module through the `oai_chat::*`
//! path, so the symbols they use are re-exported here while their definitions
//! stay scoped to `crate::oai_chat` (`pub(in crate::oai_chat)`).

mod common;
mod gemma4;
mod qwen;

// Shared symbols referenced by `model.rs`/`lib.rs` through the `oai_chat::*`
// path. Re-exported here so those callers stay unchanged. These are the only
// `common` items that need `pub(crate)`; everything else stays scoped to
// `crate::oai_chat` (`pub(in crate::oai_chat)`).
pub(crate) use common::{
    GrammarSpec, OaiStreamParser, OaiStreamUpdate, ParsedChatMsg, ResolvedToolChoice, THINK_CLOSE,
    THINK_OPEN, ToolCallDelta, ToolChatTemplateResult, build_chat_result_from_oai_json,
    build_oai_messages_json, compute_preserved_token_set, decode_oai_deltas,
    extract_enable_thinking, grammar_triggers_to_patterns_and_tokens, is_gemma4_rust_parser,
    is_qwen_rust_parser, parsed_msg_to_oai_json, resolve_tool_choice,
};
// `GEMMA4_CHAT_FORMAT`, `QWEN_TAGGED_CHAT_FORMAT`, `ToolGrammarTrigger`, and
// `ToolGrammarTriggerType` are reached from `model.rs` only through its
// `#[cfg(test)]` assertions, so gating their re-export keeps a plain
// `cargo build` free of unused-import warnings.
#[cfg(test)]
pub(crate) use common::{
    GEMMA4_CHAT_FORMAT, QWEN_TAGGED_CHAT_FORMAT, ToolGrammarTrigger, ToolGrammarTriggerType,
};

// Gemma4 symbols referenced by `model.rs` through the `oai_chat::*` path.
pub(crate) use gemma4::{
    is_gemma4_tool_template, parse_gemma4_response, render_gemma4_tool_chat_template,
};

// Qwen3.5 tagged symbols referenced by `model.rs` through the `oai_chat::*`
// path.
pub(crate) use qwen::{
    build_qwen_tool_param_index, is_qwen_tagged_tool_template, parse_qwen_tagged_response,
    render_qwen_tool_chat_template,
};
// `build_qwen_tool_call_grammar_spec` is only reached from `model.rs`'s
// `oai_chat::*` path in a `#[cfg(test)]` proof-of-concept, so gating the
// re-export keeps a plain `cargo build` free of an unused-import warning.
#[cfg(test)]
pub(crate) use qwen::build_qwen_tool_call_grammar_spec;
