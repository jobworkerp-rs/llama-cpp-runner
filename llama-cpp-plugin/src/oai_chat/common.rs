//! Shared building blocks for the client-side tool-calling path.
//!
//! These are the model-family-agnostic pieces — the wire/OAI JSON conversions,
//! the GBNF/stream helpers, the grammar-trigger plumbing, and the
//! `OaiStreamParser` dispatcher — that the Qwen3.5 tagged (`qwen`) and Gemma4
//! (`gemma4`) renderers/parsers build on. The root `oai_chat` module re-exports
//! the subset of these that `model.rs`/`lib.rs` reach through the `oai_chat::*`
//! path, so the implementation details stay scoped to `crate::oai_chat`.

use super::{gemma4, qwen};
use anyhow::{Result, anyhow, bail};
use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args;
use jobworkerp_llama_protobuf::protobuf::llm::{
    LlmChatResult, PendingToolCalls, ToolCallRequest, llm_chat_result,
};
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde_json::{Value, json};

/// `common_chat_format` value for the Qwen tagged tool-call format
/// (`COMMON_CHAT_FORMAT_CONTENT_ONLY` = 0, `..._GENERIC` = 1, `..._QWEN` = 2 in
/// llama.cpp). The legacy parser keyed off this enum, so the Rust renderer must
/// report the same value while the legacy parser is still spliced in.
pub(crate) const QWEN_TAGGED_CHAT_FORMAT: i32 = 2;
/// Plugin-local `chat_format` tag for the Gemma4 tool-call format, assigned the
/// next value after the legacy llama.cpp formats (0/1/2) so the renderer can
/// route to the Rust Gemma4 parser. Same role as [`QWEN_TAGGED_CHAT_FORMAT`].
pub(crate) const GEMMA4_CHAT_FORMAT: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum ToolGrammarTriggerType {
    Token,
    Word,
    Pattern,
    PatternFull,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ToolGrammarTrigger {
    pub trigger_type: ToolGrammarTriggerType,
    pub value: String,
    pub token: Option<LlamaToken>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ToolChatTemplateResult {
    pub prompt: String,
    pub grammar: Option<String>,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<ToolGrammarTrigger>,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
    pub chat_format: i32,
    pub parser: Option<String>,
    pub generation_prompt: String,
    pub parse_tool_calls: bool,
}

pub(crate) const THINK_OPEN: &str = "<think>";
pub(crate) const THINK_CLOSE: &str = "</think>";

/// Aggregated chunk forwarded from the OAI streaming parser to the worker
/// thread. Built by [`decode_oai_deltas`] from one batch of `update()` output.
#[derive(Clone, Debug, Default)]
pub(crate) struct OaiStreamUpdate {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCallDelta>,
}

/// Parse a batch of OAI-compatible delta JSON strings into a structured update.
/// Unknown delta shapes are silently ignored so a future parser variant cannot
/// kill the request. Uses typed serde deserialisation to avoid the `Value` heap
/// allocation overhead — this runs on every generated token.
pub(crate) fn decode_oai_deltas(deltas: &[String]) -> OaiStreamUpdate {
    #[derive(serde::Deserialize)]
    struct RawDelta {
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<RawToolCall>>,
    }
    #[derive(serde::Deserialize)]
    struct RawToolCall {
        #[serde(default)]
        index: u32,
        id: Option<String>,
        function: Option<RawFunction>,
    }
    #[derive(serde::Deserialize)]
    struct RawFunction {
        name: Option<String>,
        arguments: Option<String>,
    }

    let mut out = OaiStreamUpdate::default();
    for delta in deltas {
        let Ok(raw) = serde_json::from_str::<RawDelta>(delta) else {
            continue;
        };
        if let Some(s) = raw.content {
            out.text.push_str(&s);
        }
        if let Some(s) = raw.reasoning_content {
            out.reasoning.push_str(&s);
        }
        if let Some(calls) = raw.tool_calls {
            for call in calls {
                let (fn_name, arguments_chunk) = call
                    .function
                    .map(|f| (f.name, f.arguments.unwrap_or_default()))
                    .unwrap_or((None, String::new()));
                out.tool_calls.push(ToolCallDelta {
                    index: call.index,
                    id: call.id.filter(|s| !s.is_empty()),
                    fn_name: fn_name.filter(|s| !s.is_empty()),
                    arguments_chunk,
                });
            }
        }
    }
    out
}

/// Single tool-call delta emitted by the Rust streaming parsers. Mirrors OpenAI
/// Chat Completions streaming semantics: `id` and `fn_name` are populated only
/// on the first delta of a given `index`; subsequent deltas with the same
/// `index` only carry `arguments_chunk`. The consumer is responsible for
/// accumulating arguments string fragments.
#[derive(Clone, Debug, Default)]
pub(crate) struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub fn_name: Option<String>,
    pub arguments_chunk: String,
}

/// Per-call accumulator for streaming tool deltas. Matches the OpenAI Chat
/// Completions accumulation contract: keep the first non-empty `id` and
/// `fn_name`, concatenate `arguments_chunk` in order.
///
/// The v2 ABI does not use this accumulator inside the plugin (the host
/// aggregates streamed tool deltas), but we keep it here for unit tests and
/// as a reference implementation that consumers can reuse.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub(in crate::oai_chat) struct ToolCallAccumulator {
    by_index: std::collections::BTreeMap<u32, AccumulatedCall>,
}

#[allow(dead_code)]
#[derive(Debug, Default)]
struct AccumulatedCall {
    id: Option<String>,
    fn_name: Option<String>,
    arguments_buf: String,
}

#[allow(dead_code)]
impl ToolCallAccumulator {
    pub(in crate::oai_chat) fn new() -> Self {
        Self::default()
    }

    pub(in crate::oai_chat) fn ingest(&mut self, delta: &ToolCallDelta) {
        let entry = self.by_index.entry(delta.index).or_default();
        if let Some(id) = &delta.id
            && entry.id.is_none()
        {
            entry.id = Some(id.clone());
        }
        if let Some(name) = &delta.fn_name
            && entry.fn_name.is_none()
        {
            entry.fn_name = Some(name.clone());
        }
        entry.arguments_buf.push_str(&delta.arguments_chunk);
    }

    pub(in crate::oai_chat) fn is_empty(&self) -> bool {
        self.by_index.is_empty()
    }

    /// The next unused OAI call index — the position a brand-new tool call
    /// would be assigned. `append_chat_chunk` uses this to fold proto
    /// chunks (which carry no explicit index) into the latest open call.
    pub(in crate::oai_chat) fn next_free_index(&self) -> u32 {
        self.by_index.keys().next_back().map_or(0, |i| i + 1)
    }

    /// Finalise into the proto `ToolCallRequest` list, in ascending index
    /// order. Missing ids are filled with the same synthesis used by the
    /// non-streaming path so the client gets a deterministic correlator.
    pub(in crate::oai_chat) fn finalize(self) -> Vec<ToolCallRequest> {
        self.by_index
            .into_iter()
            .enumerate()
            .map(|(seq, (_, call))| ToolCallRequest {
                call_id: call.id.unwrap_or_else(|| synthesize_tool_call_id(seq)),
                fn_name: call.fn_name.unwrap_or_default(),
                fn_arguments: call.arguments_buf,
            })
            .collect()
    }
}

/// Grammar block derived from a `ChatTemplateResult`. Kept Clone so the
/// `InferenceArgs` that carries it can be passed around alongside the
/// owning `LlamaModelWrapper`. `preserved_tokens` lives separately as a
/// `HashSet<LlamaToken>` (see [`compute_preserved_token_set`]) because it
/// is consumed by the decode loop, not by `build_sampler`.
#[derive(Clone, Debug)]
pub(crate) struct GrammarSpec {
    pub grammar: String,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<ToolGrammarTrigger>,
}

// Mint a collision-free GBNF rule name from `base`: the first occurrence keeps
// `base`, later occurrences get `{base}-{n}`. Centralized so the tool-rule and
// parameter-rule namers cannot drift in their suffix scheme — divergence there
// silently reintroduces the rule-name collisions this guards against.
pub(in crate::oai_chat) fn dedup_rule_name(
    seen: &mut std::collections::BTreeMap<String, usize>,
    base: String,
) -> String {
    let count = seen.entry(base.clone()).or_insert(0usize);
    let name = if *count == 0 {
        base
    } else {
        format!("{base}-{count}")
    };
    *count += 1;
    name
}

// Extract and validate a tool's `function.name`. Single source so the rule-name
// builder and the grammar body emitter cannot disagree on what counts as a
// valid name.
pub(in crate::oai_chat) fn qwen_tool_function_name(tool: &Value) -> Result<&str> {
    tool.pointer("/function/name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("tool definition is missing function.name"))
}

// Render a chat template with the plugin-controlled context keys. `setup_env`
// lets each model family register its own jinja filters/functions (e.g. the
// Qwen-only `tojson`/`raise_exception`) without duplicating the ctx assembly,
// which must stay identical so the rendered prompt and the grammar/parser
// inputs cannot desync.
pub(in crate::oai_chat) fn render_template_once(
    template: &str,
    messages: &Value,
    tools: Option<&Value>,
    kwargs: Option<&Value>,
    enable_thinking: bool,
    add_generation_prompt: bool,
    setup_env: impl FnOnce(&mut minijinja::Environment<'_>),
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    setup_env(&mut env);
    let tmpl = env
        .template_from_str(template)
        .map_err(|e| anyhow!("failed to compile chat template: {e}"))?;
    let mut ctx = serde_json::Map::new();
    // Seed caller kwargs first so the plugin-controlled keys below always win.
    // Otherwise a caller could override `messages`/`tools`/`enable_thinking`
    // via chat_template_kwargs and desync the rendered prompt from the grammar
    // (which is built from the dedicated `tools_json`/`enable_thinking` inputs).
    if let Some(Value::Object(extra)) = kwargs {
        for (key, value) in extra {
            ctx.insert(key.clone(), value.clone());
        }
    }
    ctx.insert("messages".to_string(), messages.clone());
    ctx.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
    ctx.insert(
        "add_generation_prompt".to_string(),
        Value::Bool(add_generation_prompt),
    );
    ctx.insert("bos_token".to_string(), Value::String(String::new()));
    ctx.insert("eos_token".to_string(), Value::String(String::new()));
    if let Some(tools) = tools {
        ctx.insert("tools".to_string(), tools.clone());
    }
    tmpl.render(Value::Object(ctx))
        .map_err(|e| anyhow!("failed to render chat template: {e}"))
}

pub(in crate::oai_chat) fn qwen_tool_rule_names(tools: &[Value]) -> Result<Vec<String>> {
    let mut seen = std::collections::BTreeMap::new();
    tools
        .iter()
        .map(|tool| {
            let name = qwen_tool_function_name(tool)?;
            let base = format!("tool-{}", grammar_rule_suffix(name)?);
            Ok(dedup_rule_name(&mut seen, base))
        })
        .collect()
}

pub(in crate::oai_chat) fn grammar_rule_suffix(raw: &str) -> Result<String> {
    let mut suffix = String::new();
    let mut last_was_dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            suffix.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            suffix.push('-');
            last_was_dash = true;
        }
    }
    let trimmed = suffix.trim_matches('-');
    if trimmed.is_empty() {
        bail!("tool and parameter names must contain at least one ASCII alphanumeric character");
    }
    Ok(trimmed.to_string())
}

// A tool call extracted from a Qwen3.5 tagged response. `arguments` is the
// OAI-shaped JSON-encoded string (matching `function.arguments`), and `id` is
// left `None` here because the tag stream carries no id — it is synthesized at
// the OAI-JSON boundary (`parsed_msg_to_oai_json`), mirroring the fork API.
// Wired into the main path in phase 3-integration; defined here with the
// generator so the parse/grammar contract stays co-located.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

// The Rust counterpart of llama.cpp `common_chat_msg`: the structured result of
// parsing a Qwen3.5 tagged assistant turn. Streaming diffs compare two of these
// (see phase 3-2), and the non-streaming path converts one into the OAI JSON
// shape `build_chat_result_from_oai_json` already consumes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParsedChatMsg {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

// Convert a `ParsedChatMsg` into the OAI JSON string shape
// `build_chat_result_from_oai_json` consumes. Tool-call ids are synthesized
// here because the tag stream carries none.
pub(crate) fn parsed_msg_to_oai_json(msg: &ParsedChatMsg) -> Result<String> {
    let calls: Vec<Value> = msg
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, call)| {
            let id = call
                .id
                .clone()
                .unwrap_or_else(|| synthesize_tool_call_id(i));
            json!({
                "type": "function",
                "id": id,
                "function": { "name": call.name, "arguments": call.arguments },
            })
        })
        .collect();
    let mut obj = serde_json::Map::new();
    obj.insert("role".to_string(), Value::String("assistant".to_string()));
    obj.insert("content".to_string(), Value::String(msg.content.clone()));
    if let Some(reasoning) = &msg.reasoning_content {
        obj.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning.clone()),
        );
    }
    if !calls.is_empty() {
        obj.insert("tool_calls".to_string(), Value::Array(calls));
    }
    serde_json::to_string(&Value::Object(obj)).map_err(Into::into)
}

#[derive(Debug, Default)]
pub(in crate::oai_chat) struct ParsedStreamDeltaState {
    previous: ParsedChatMsg,
    generation_prompt_content: String,
    generation_prompt_emitted: bool,
    call_ids: Vec<String>,
}

impl ParsedStreamDeltaState {
    pub(in crate::oai_chat) fn new(generation_prompt: &str) -> Self {
        Self {
            previous: ParsedChatMsg::default(),
            generation_prompt_content: generation_prompt_body(generation_prompt).to_string(),
            generation_prompt_emitted: false,
            call_ids: Vec::new(),
        }
    }
}

/// Streaming OAI-delta parser backing the `chat` path. Owning the choice as one
/// type keeps "exactly one parser is active" a type-level invariant and gives
/// both the per-chunk and the flush call sites a single `update` entry point.
pub(crate) enum OaiStreamParser {
    PlainText,
    Qwen(qwen::QwenTaggedStreamState),
    Gemma4(gemma4::Gemma4StreamState),
}

impl OaiStreamParser {
    /// Pick the parser implied by the rendered template. Unsupported formats
    /// fail here instead of falling back to fork-only OAI parser state.
    pub(crate) fn for_template(
        tmpl_result: &ToolChatTemplateResult,
        tools_json: &str,
    ) -> Result<Self> {
        if !tmpl_result.parse_tool_calls {
            Ok(Self::PlainText)
        } else if is_qwen_rust_parser(tmpl_result) {
            Ok(Self::Qwen(qwen::QwenTaggedStreamState::new(
                tools_json,
                &tmpl_result.generation_prompt,
            )?))
        } else if is_gemma4_rust_parser(tmpl_result) {
            Ok(Self::Gemma4(gemma4::Gemma4StreamState::new(
                &tmpl_result.generation_prompt,
            )))
        } else {
            bail!(
                "unsupported tool-calling chat_format {} for Rust streaming parser",
                tmpl_result.chat_format
            )
        }
    }

    pub(crate) fn update(&mut self, chunk: &str, is_streaming: bool) -> Result<Vec<String>> {
        match self {
            Self::PlainText => {
                let mut deltas = Vec::new();
                if is_streaming && !chunk.is_empty() {
                    push_text_delta(&mut deltas, "content", chunk)?;
                }
                Ok(deltas)
            }
            Self::Qwen(state) => state.update(chunk, is_streaming),
            Self::Gemma4(state) => state.update(chunk, is_streaming),
        }
    }
}

pub(crate) fn is_qwen_rust_parser(tmpl_result: &ToolChatTemplateResult) -> bool {
    tmpl_result.chat_format == QWEN_TAGGED_CHAT_FORMAT
}

pub(crate) fn is_gemma4_rust_parser(tmpl_result: &ToolChatTemplateResult) -> bool {
    tmpl_result.chat_format == GEMMA4_CHAT_FORMAT
}

// Return the body of the generation prompt: everything after the role-header
// line (e.g. the empty `<think>\n\n</think>\n\n` block). The streaming path
// never sees this echoed back by the model, so it must re-emit it as leading
// content. This is the complement of `strip_generation_prompt` in model.rs,
// which keeps the header and drops it from a model response.
fn generation_prompt_body(generation_prompt: &str) -> &str {
    generation_prompt
        .split_once('\n')
        .map_or("", |(_, body)| body)
}

// The five escapes that JSON string syntax and GBNF string literals share.
// Returns the escape sequence for `ch`, or `None` if it needs no escaping by
// this common rule (callers decide how to handle the remainder).
fn json_basic_escape(ch: char) -> Option<&'static str> {
    match ch {
        '"' => Some("\\\""),
        '\\' => Some("\\\\"),
        '\n' => Some("\\n"),
        '\r' => Some("\\r"),
        '\t' => Some("\\t"),
        _ => None,
    }
}

pub(in crate::oai_chat) fn push_json_string_fragment(out: &mut String, raw: &str) {
    for ch in raw.chars() {
        if let Some(esc) = json_basic_escape(ch) {
            out.push_str(esc);
        } else if ch.is_control() {
            use std::fmt::Write as _;
            let _ = write!(out, "\\u{:04x}", ch as u32);
        } else {
            out.push(ch);
        }
    }
}

// Serialize a single-key text delta (`{"<key>": "<text>"}`) as the OAI sink
// expects. Skips empty text so callers don't emit no-op chunks.
fn push_text_delta(deltas: &mut Vec<String>, key: &str, text: &str) -> Result<()> {
    if !text.is_empty() {
        deltas.push(serde_json::to_string(&json!({ key: text }))?);
    }
    Ok(())
}

// Append the new suffix of `now` past `previous` as a text delta, if any. This
// is the "diff two accumulated strings and emit only what grew" primitive
// shared by the content and reasoning streams.
fn push_grown_text_delta(
    deltas: &mut Vec<String>,
    key: &str,
    previous: &str,
    now: &str,
) -> Result<()> {
    if let Some(delta) = now.strip_prefix(previous) {
        push_text_delta(deltas, key, delta)?;
    }
    Ok(())
}

// The OAI streaming tool-call convention: the first delta of a call carries
// `id`/`type`/`function.name`; later deltas only append `function.arguments`.
fn tool_call_delta_json(
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> Result<String> {
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), json!(name));
    }
    function.insert("arguments".to_string(), json!(arguments));
    let mut call = serde_json::Map::new();
    call.insert("index".to_string(), json!(index));
    if let Some(id) = id {
        call.insert("id".to_string(), json!(id));
        call.insert("type".to_string(), json!("function"));
    }
    call.insert("function".to_string(), Value::Object(function));
    Ok(serde_json::to_string(&json!({ "tool_calls": [call] }))?)
}

pub(in crate::oai_chat) fn parsed_stream_deltas(
    state: &mut ParsedStreamDeltaState,
    now: &ParsedChatMsg,
) -> Result<Option<Vec<String>>> {
    if now.content.len() < state.previous.content.len()
        || now.tool_calls.len() < state.previous.tool_calls.len()
        || now
            .tool_calls
            .iter()
            .zip(&state.previous.tool_calls)
            .any(|(current, previous)| current.arguments.len() < previous.arguments.len())
    {
        return Ok(None);
    }

    let mut deltas = Vec::new();
    if !state.generation_prompt_emitted {
        state.generation_prompt_emitted = true;
        push_text_delta(&mut deltas, "content", &state.generation_prompt_content)?;
    }

    push_grown_text_delta(
        &mut deltas,
        "content",
        &state.previous.content,
        &now.content,
    )?;
    if let Some(reasoning) = &now.reasoning_content {
        let previous = state.previous.reasoning_content.as_deref().unwrap_or("");
        push_grown_text_delta(&mut deltas, "reasoning_content", previous, reasoning)?;
    }

    for (index, call) in now.tool_calls.iter().enumerate() {
        let previous = state.previous.tool_calls.get(index);
        let previous_arguments = previous.map_or("", |call| call.arguments.as_str());
        let Some(argument_delta) = call.arguments.strip_prefix(previous_arguments) else {
            return Ok(None);
        };
        if previous.is_none() {
            // First delta for this call: assign and cache a stable synthesized id.
            if state.call_ids.len() <= index {
                state.call_ids.push(synthesize_tool_call_id(index));
            }
            deltas.push(tool_call_delta_json(
                index,
                Some(&state.call_ids[index]),
                Some(&call.name),
                argument_delta,
            )?);
        } else if !argument_delta.is_empty() {
            deltas.push(tool_call_delta_json(index, None, None, argument_delta)?);
        }
    }
    state.previous = now.clone();
    Ok(Some(deltas))
}

pub(in crate::oai_chat) fn escape_gbnf_literal(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match json_basic_escape(ch) {
            Some(esc) => out.push_str(esc),
            None => out.push(ch),
        }
    }
    out
}

/// Translate `ChatTemplateResult.grammar_triggers` into the two-track form
/// `LlamaSampler::grammar_lazy_patterns` expects. Mirrors the conversion in
/// the upstream `openai_stream.rs` example:
/// - `Token` → push to `trigger_tokens`
/// - `Word` → try `str_to_token`; if it tokenizes to a single token, push to
///   `trigger_tokens`; otherwise escape it as a regex literal and push to
///   `trigger_patterns`
/// - `Pattern` → push to `trigger_patterns` verbatim
/// - `PatternFull` → wrap with `^(?:...)$` to anchor the regex
pub(crate) fn grammar_triggers_to_patterns_and_tokens(
    model: &LlamaModel,
    triggers: &[ToolGrammarTrigger],
) -> (Vec<String>, Vec<LlamaToken>) {
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();
    for t in triggers {
        match t.trigger_type {
            ToolGrammarTriggerType::Token => {
                if let Some(tok) = t.token {
                    tokens.push(tok);
                }
            }
            ToolGrammarTriggerType::Word => match model.str_to_token(&t.value, AddBos::Never) {
                Ok(toks) if toks.len() == 1 => tokens.push(toks[0]),
                _ => patterns.push(regex::escape(&t.value)),
            },
            ToolGrammarTriggerType::Pattern => patterns.push(t.value.clone()),
            ToolGrammarTriggerType::PatternFull => {
                patterns.push(format!("^(?:{})$", t.value));
            }
        }
    }
    (patterns, tokens)
}

/// Compute the set of token IDs that must be sampled with `Special::Tokenize`
/// (so the model can actually emit `<tool_call>` etc instead of treating them
/// as plain text). Returns an owned HashSet so it can outlive `GrammarSpec`
/// borrowing.
pub(crate) fn compute_preserved_token_set(
    model: &LlamaModel,
    preserved: &[String],
) -> std::collections::HashSet<LlamaToken> {
    preserved
        .iter()
        .filter_map(|s| {
            let toks = model.str_to_token(s, AddBos::Never).ok()?;
            (toks.len() == 1).then_some(toks[0])
        })
        .collect()
}

/// Extract the `enable_thinking` boolean from a `chat_template_kwargs`
/// JSON object string. The kwargs payload is forwarded to the Rust jinja
/// renderer, and the parsed boolean is also used by the Rust parser. Without
/// this extraction the two channels can disagree (Qwen3 think mode is a
/// notable example).
///
/// Returns `None` when the kwargs is absent, not a JSON object, or the
/// `enable_thinking` key is missing / not a boolean. Callers should fall
/// back to a sensible default (typically `false`) in that case.
pub(crate) fn extract_enable_thinking(chat_template_kwargs: Option<&str>) -> Option<bool> {
    let raw = chat_template_kwargs?;
    let value: Value = serde_json::from_str(raw).ok()?;
    value.get("enable_thinking")?.as_bool()
}

/// Convert the wire-level `ChatMessage` list into an OpenAI-compatible
/// messages JSON array string.
///
/// - When the request omits a system message and `system_prompt_fallback` is
///   non-empty, prepend a `{"role":"system","content":...}` entry.
/// - `assistant` messages carrying `ToolCalls` become
///   `{"role":"assistant","content":null,"tool_calls":[...]}`.
/// - `tool` messages carrying `ToolResults` fan out into one
///   `{"role":"tool","tool_call_id":..,"content":..}` per result; when
///   `is_error` is set, `"[ERROR] "` is prepended to `content` (OAI/jinja
///   templates have no native error field).
/// - `Image` content is rejected: the tools path rejects multimodal input
///   upstream so it should never reach here.
/// - `UNSPECIFIED` role is rejected.
pub(crate) fn build_oai_messages_json(
    system_prompt_fallback: &str,
    messages: &[llm_chat_args::ChatMessage],
) -> Result<String> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len() + 1);

    let has_system = messages
        .iter()
        .any(|m| llm_chat_args::ChatRole::try_from(m.role) == Ok(llm_chat_args::ChatRole::System));
    if !has_system && !system_prompt_fallback.is_empty() {
        out.push(json!({"role": "system", "content": system_prompt_fallback}));
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

        match msg.content.as_ref().and_then(|c| c.content.as_ref()) {
            Some(llm_chat_args::message_content::Content::Text(t)) => {
                out.push(json!({"role": role, "content": t}));
            }
            Some(llm_chat_args::message_content::Content::Image(_)) => {
                bail!("Image content is not supported on the client-tools path");
            }
            Some(llm_chat_args::message_content::Content::ToolCalls(tc)) => {
                if role != "assistant" {
                    bail!("ToolCalls content is only valid on assistant messages (got {role})");
                }
                let calls: Vec<Value> = tc
                    .calls
                    .iter()
                    .map(|c| {
                        json!({
                            "id": c.call_id,
                            "type": "function",
                            "function": {
                                "name": c.fn_name,
                                "arguments": c.fn_arguments,
                            },
                        })
                    })
                    .collect();
                // OAI permits content=null when the assistant turn is purely a
                // tool invocation; we keep it null to avoid confusing the
                // template renderer with empty strings.
                out.push(json!({
                    "role": "assistant",
                    "content": Value::Null,
                    "tool_calls": calls,
                }));
            }
            Some(llm_chat_args::message_content::Content::ToolExecutionRequests(_)) => {
                bail!(crate::model::ERR_TOOL_EXECUTION_REQUESTS_REJECTED);
            }
            Some(llm_chat_args::message_content::Content::ToolResults(tr)) => {
                if role != "tool" {
                    bail!("ToolResults content is only valid on tool messages (got {role})");
                }
                if tr.results.is_empty() {
                    bail!("ToolResults must contain at least one entry");
                }
                for r in &tr.results {
                    if r.call_id.is_empty() {
                        bail!("ToolResult.call_id must not be empty");
                    }
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": r.call_id,
                        "content": if r.is_error {
                            format!("[ERROR] {}", r.content)
                        } else {
                            r.content.clone()
                        },
                    }));
                }
            }
            None => {
                out.push(json!({"role": role, "content": ""}));
            }
        }
    }

    if out.is_empty() {
        bail!("no chat messages to encode (after system fallback resolution)");
    }

    serde_json::to_string(&Value::Array(out))
        .map_err(|e| anyhow!("failed to serialize OAI messages: {e}"))
}

/// Outcome of [`resolve_tool_choice`]. `Passthrough` means the caller can
/// forward the original `tools_json` / `tool_choice` to llama.cpp without
/// any rewrite (the common `auto`/`none`/`required` / unset case);
/// `FunctionSpecific` carries the rewritten pair so OpenAI's
/// `{"type":"function","function":{"name":"..."}}` shape becomes the
/// (filtered tools + `"required"`) pair that llama.cpp accepts.
#[derive(Debug)]
pub(crate) enum ResolvedToolChoice {
    Passthrough,
    FunctionSpecific {
        tools_json: String,
        tool_choice: String,
    },
}

/// Normalise the OpenAI `tool_choice` field for llama.cpp consumption.
///
/// llama.cpp only accepts the bare strings `"auto"`, `"none"`, `"required"`
/// (everything else throws). The OpenAI spec also allows
/// `{"type":"function","function":{"name":"<n>"}}` to force one specific
/// tool. We honour that form by filtering `tools_json` down to the named
/// function and substituting `"required"` so the model must call it.
/// Unknown function names become an error.
pub(crate) fn resolve_tool_choice(
    tools_json: &str,
    tool_choice: Option<&str>,
) -> Result<ResolvedToolChoice> {
    let Some(raw) = tool_choice else {
        return Ok(ResolvedToolChoice::Passthrough);
    };
    let trimmed = raw.trim();
    if matches!(trimmed, "auto" | "none" | "required") {
        return Ok(ResolvedToolChoice::Passthrough);
    }
    // A JSON object must start with '{' once trimmed; bare unknown strings
    // are surfaced as errors rather than silently forwarded so llama.cpp's
    // `invalid_argument` doesn't leak to the caller.
    if !trimmed.starts_with('{') {
        bail!(
            "unsupported tool_choice {raw:?}: expected \"auto\" | \"none\" | \"required\" \
             or a {{\"type\":\"function\",\"function\":{{\"name\":\"...\"}}}} object"
        );
    }
    let parsed: Value = serde_json::from_str(trimmed)
        .map_err(|e| anyhow!("tool_choice is not valid JSON: {e} ({raw})"))?;
    let target_name = parsed
        .pointer("/function/name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow!("tool_choice object must carry function.name: {raw} (got {parsed})")
        })?;

    let tools: Value = serde_json::from_str(tools_json)
        .map_err(|e| anyhow!("client_tools_json is not valid JSON array: {e}"))?;
    let arr = tools
        .as_array()
        .ok_or_else(|| anyhow!("client_tools_json must be a JSON array"))?;
    // Linear scan that short-circuits on the first match; tools arrays
    // are typically small but `parallel_tool_calls=true` callers can
    // register dozens of them.
    let found = arr
        .iter()
        .find(|t| t.pointer("/function/name").and_then(|v| v.as_str()) == Some(target_name))
        .ok_or_else(|| {
            anyhow!(
                "tool_choice requested function {target_name:?} but it is not present in client_tools_json"
            )
        })?;
    let filtered_tools_json = serde_json::to_string(&json!([found]))
        .map_err(|e| anyhow!("failed to serialize filtered tools_json: {e}"))?;
    // Honour OpenAI's function-specific semantics literally: the eager
    // grammar enabled by "required" makes the tool call unconditional.
    // The caller may need to strip the chat template's generation_prompt
    // from the raw output before parsing because the grammar re-emits it.
    Ok(ResolvedToolChoice::FunctionSpecific {
        tools_json: filtered_tools_json,
        tool_choice: "required".to_string(),
    })
}

/// Best-effort fallback parser for raw tool-call output when llama.cpp's
/// PEG parser rejects an eager-grammar emission. Recognises the
/// `<tool_call>{...}</tool_call>` shape that ChatML-derived templates
/// (Qwen, etc.) use and reconstructs the same `{"role":"assistant",...}`
/// JSON shape the OAI parser would have produced. Returns `None` when no
/// well-formed tool calls are found so the caller can surface the
/// original parser error verbatim.
#[cfg(test)]
fn fallback_parse_tool_calls(raw: &str) -> Option<String> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls: Vec<Value> = Vec::new();
    let mut cursor = 0;
    while let Some(start) = raw[cursor..].find(OPEN) {
        let body_start = cursor + start + OPEN.len();
        let Some(end_rel) = raw[body_start..].find(CLOSE) else {
            break;
        };
        let body = raw[body_start..body_start + end_rel].trim();
        if let Ok(parsed) = serde_json::from_str::<Value>(body) {
            let name = parsed
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            let arguments = parsed
                .get("arguments")
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_default();
            if !name.is_empty() {
                calls.push(json!({
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                }));
            }
        }
        cursor = body_start + end_rel + CLOSE.len();
    }
    if calls.is_empty() {
        return None;
    }
    serde_json::to_string(&json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": calls,
    }))
    .ok()
}

/// Generate a fallback tool-call id when the OAI parser did not provide one.
/// Uses index + a coarse epoch suffix so concurrent requests in the same
/// process don't collide. Production paths always receive a parser-supplied
/// id, so this branch only fires for misbehaving templates.
fn synthesize_tool_call_id(index: usize) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("call_{index}_{:016x}", nanos as u64)
}

/// Convert an OAI-shaped parsed assistant message JSON string into a wire-level
/// `LlmChatResult`. Handles:
/// - `content` (plain text) → `MessageContent::Text(...)`
/// - `tool_calls[*]` (OAI shape: `{id?, type, function: {name, arguments}}`)
///   → `MessageContent::ToolCalls(...)` plus `pending_tool_calls` and
///   `requires_tool_execution = Some(true)`
/// - `reasoning_content` → `LlmChatResult.reasoning_content`
/// - id fallback: when an entry has no `id`, a synthetic
///   `call_<index>_<epoch_ns>` id is generated so the client can correlate
///   the eventual `tool` response back to this call
pub(crate) fn build_chat_result_from_oai_json(
    parsed_json: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_prompt_time_sec: f32,
    total_completion_time_sec: f32,
    model_name: &str,
) -> Result<LlmChatResult> {
    let parsed: Value = serde_json::from_str(parsed_json)
        .map_err(|e| anyhow!("tool response parser returned invalid JSON: {e}: {parsed_json}"))?;

    let reasoning_content = parsed
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let tool_calls = parsed
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let usage = Some(llm_chat_result::Usage {
        model: model_name.to_string(),
        prompt_tokens: Some(prompt_tokens),
        completion_tokens: Some(completion_tokens),
        total_prompt_time_sec: Some(total_prompt_time_sec),
        total_completion_time_sec: Some(total_completion_time_sec),
    });

    if tool_calls.is_empty() {
        // Plain assistant text turn.
        let text = parsed
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        return Ok(LlmChatResult {
            content: Some(llm_chat_result::MessageContent {
                content: Some(llm_chat_result::message_content::Content::Text(text)),
            }),
            reasoning_content,
            done: true,
            usage,
            pending_tool_calls: None,
            requires_tool_execution: None,
            tool_execution_results: Vec::new(),
            tool_execution_started: None,
        });
    }

    // Build the canonical proto representation once; the chat-result side
    // mirrors it (the two structs are identical apart from the type name).
    // OAI defines `function.arguments` as a JSON-encoded string, so we keep
    // it verbatim.
    let pending_calls: Vec<ToolCallRequest> = tool_calls
        .iter()
        .enumerate()
        .map(|(i, call)| ToolCallRequest {
            call_id: call
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| synthesize_tool_call_id(i)),
            fn_name: call
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            fn_arguments: call
                .pointer("/function/arguments")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        })
        .collect();
    let result_calls: Vec<llm_chat_result::message_content::ToolCall> = pending_calls
        .iter()
        .enumerate()
        .map(|(i, c)| llm_chat_result::message_content::ToolCall {
            call_id: c.call_id.clone(),
            fn_name: c.fn_name.clone(),
            fn_arguments: c.fn_arguments.clone(),
            // Position in the aggregated list doubles as the OAI index so
            // clients with a single accumulator can treat streaming and
            // non-streaming responses uniformly.
            delta_index: Some(i as u32),
        })
        .collect();

    Ok(LlmChatResult {
        content: Some(llm_chat_result::MessageContent {
            content: Some(llm_chat_result::message_content::Content::ToolCalls(
                llm_chat_result::message_content::ToolCalls {
                    calls: result_calls,
                },
            )),
        }),
        reasoning_content,
        done: true,
        usage,
        pending_tool_calls: Some(PendingToolCalls {
            calls: pending_calls,
        }),
        requires_tool_execution: Some(true),
        tool_execution_results: Vec::new(),
        tool_execution_started: None,
    })
}

/// Test-only helper shared by the `qwen` and `gemma4` stream-delta tests:
/// rewrites every `tool_calls[*].id` to a fixed `<id>` so golden comparisons
/// ignore the synthesized, non-deterministic call ids.
#[cfg(test)]
pub(in crate::oai_chat) fn normalize_stream_delta_ids(values: &mut [serde_json::Value]) {
    for value in values {
        let Some(calls) = value.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls {
            if let Some(obj) = call.as_object_mut()
                && obj.contains_key("id")
            {
                obj.insert("id".to_string(), Value::String("<id>".to_string()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args::{
        ChatMessage, ChatRole, MessageContent,
        message_content::{
            Content, ToolCall, ToolCalls, ToolExecutionRequest, ToolExecutionRequests, ToolResult,
            ToolResults,
        },
    };

    fn text_msg(role: ChatRole, text: &str) -> ChatMessage {
        ChatMessage {
            role: role as i32,
            content: Some(MessageContent {
                content: Some(Content::Text(text.to_string())),
            }),
        }
    }

    fn build(system_fallback: &str, msgs: &[ChatMessage]) -> Value {
        let raw = build_oai_messages_json(system_fallback, msgs).expect("build");
        serde_json::from_str(&raw).expect("valid JSON array")
    }

    #[test]
    fn test_build_oai_messages_basic_user_only() {
        let arr = build("", &[text_msg(ChatRole::User, "hello")]);
        let arr = arr.as_array().expect("array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[0]["content"], "hello");
    }

    #[test]
    fn test_build_oai_messages_prepends_system_fallback_when_absent() {
        let v = build("be terse", &[text_msg(ChatRole::User, "hi")]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "be terse");
        assert_eq!(arr[1]["role"], "user");
    }

    #[test]
    fn test_build_oai_messages_skips_system_fallback_when_present() {
        let v = build(
            "ignored",
            &[
                text_msg(ChatRole::System, "explicit"),
                text_msg(ChatRole::User, "hi"),
            ],
        );
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["content"], "explicit");
    }

    #[test]
    fn test_build_oai_messages_assistant_tool_calls_serialized_correctly() {
        let assistant_msg = ChatMessage {
            role: ChatRole::Assistant as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolCalls(ToolCalls {
                    calls: vec![ToolCall {
                        call_id: "c1".to_string(),
                        fn_name: "get_weather".to_string(),
                        fn_arguments: r#"{"city":"Tokyo"}"#.to_string(),
                    }],
                })),
            }),
        };
        let v = build("", &[assistant_msg]);
        let entry = &v[0];
        assert_eq!(entry["role"], "assistant");
        assert!(entry["content"].is_null());
        let calls = entry["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "c1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Tokyo"}"#);
    }

    #[test]
    fn test_build_oai_messages_tool_results_fans_out() {
        let tool_msg = ChatMessage {
            role: ChatRole::Tool as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolResults(ToolResults {
                    results: vec![
                        ToolResult {
                            call_id: "c1".to_string(),
                            fn_name: "get_weather".to_string(),
                            content: r#"{"temp":22}"#.to_string(),
                            is_error: false,
                        },
                        ToolResult {
                            call_id: "c2".to_string(),
                            fn_name: "get_time".to_string(),
                            content: r#"{"hour":12}"#.to_string(),
                            is_error: false,
                        },
                    ],
                })),
            }),
        };
        let v = build("", &[tool_msg]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "tool");
        assert_eq!(arr[0]["tool_call_id"], "c1");
        assert_eq!(arr[0]["content"], r#"{"temp":22}"#);
        assert_eq!(arr[1]["tool_call_id"], "c2");
        assert_eq!(arr[1]["content"], r#"{"hour":12}"#);
    }

    #[test]
    fn test_build_oai_messages_tool_results_is_error_prefix() {
        let tool_msg = ChatMessage {
            role: ChatRole::Tool as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolResults(ToolResults {
                    results: vec![ToolResult {
                        call_id: "c1".to_string(),
                        fn_name: String::new(),
                        content: "boom".to_string(),
                        is_error: true,
                    }],
                })),
            }),
        };
        let v = build("", &[tool_msg]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["content"], "[ERROR] boom");
    }

    #[test]
    fn test_build_oai_messages_tool_results_empty_results_errors() {
        let tool_msg = ChatMessage {
            role: ChatRole::Tool as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolResults(ToolResults { results: vec![] })),
            }),
        };
        let err = build_oai_messages_json("", &[tool_msg]).unwrap_err();
        assert!(
            err.to_string().contains("at least one entry"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_build_oai_messages_tool_results_empty_call_id_errors() {
        let tool_msg = ChatMessage {
            role: ChatRole::Tool as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolResults(ToolResults {
                    results: vec![ToolResult {
                        call_id: String::new(),
                        fn_name: String::new(),
                        content: "ok".to_string(),
                        is_error: false,
                    }],
                })),
            }),
        };
        let err = build_oai_messages_json("", &[tool_msg]).unwrap_err();
        assert!(
            err.to_string().contains("call_id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_build_oai_messages_tool_results_rejected_on_non_tool_role() {
        let bogus = ChatMessage {
            role: ChatRole::User as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolResults(ToolResults {
                    results: vec![ToolResult {
                        call_id: "c1".to_string(),
                        fn_name: String::new(),
                        content: "ok".to_string(),
                        is_error: false,
                    }],
                })),
            }),
        };
        let err = build_oai_messages_json("", &[bogus]).unwrap_err();
        assert!(err.to_string().contains("tool"), "unexpected error: {err}");
    }

    #[test]
    fn test_build_oai_messages_tool_execution_requests_now_rejected() {
        let tool_msg = ChatMessage {
            role: ChatRole::Tool as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolExecutionRequests(ToolExecutionRequests {
                    requests: vec![ToolExecutionRequest {
                        call_id: "c1".to_string(),
                        fn_name: "get_weather".to_string(),
                        fn_arguments: r#"{"temp":22}"#.to_string(),
                    }],
                })),
            }),
        };
        let err = build_oai_messages_json("", &[tool_msg]).unwrap_err();
        let msg = err.to_string();
        // Error should hint at the new ToolResults variant for migration.
        assert!(msg.contains("ToolResults"), "unexpected error: {msg}");
    }

    #[test]
    fn test_build_oai_messages_unspecified_role_errors() {
        let msg = ChatMessage {
            role: ChatRole::Unspecified as i32,
            content: Some(MessageContent {
                content: Some(Content::Text("x".to_string())),
            }),
        };
        let err = build_oai_messages_json("", &[msg]).unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[test]
    fn test_build_oai_messages_rejects_image_content() {
        use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args::message_content::{
            Image, ImageSource,
        };
        let img_msg = ChatMessage {
            role: ChatRole::User as i32,
            content: Some(MessageContent {
                content: Some(Content::Image(Image {
                    content_type: "image/jpeg".to_string(),
                    source: Some(ImageSource {
                        url: String::new(),
                        base64: "YWJj".to_string(),
                    }),
                })),
            }),
        };
        let err = build_oai_messages_json("", &[img_msg]).unwrap_err();
        assert!(err.to_string().contains("Image"));
    }

    #[test]
    fn test_build_oai_messages_rejects_tool_calls_on_user_role() {
        let bogus = ChatMessage {
            role: ChatRole::User as i32,
            content: Some(MessageContent {
                content: Some(Content::ToolCalls(ToolCalls {
                    calls: vec![ToolCall {
                        call_id: "c1".to_string(),
                        fn_name: "f".to_string(),
                        fn_arguments: "{}".to_string(),
                    }],
                })),
            }),
        };
        let err = build_oai_messages_json("", &[bogus]).unwrap_err();
        assert!(err.to_string().contains("assistant"));
    }

    // build_chat_result_from_oai_json: plain text branch.
    #[test]
    fn test_oai_json_to_chat_result_text_only() {
        let parsed = r#"{"role":"assistant","content":"hi"}"#;
        let r = build_chat_result_from_oai_json(parsed, 4, 1, 0.1, 0.05, "model").unwrap();
        assert!(r.done);
        assert!(r.pending_tool_calls.is_none());
        assert!(r.requires_tool_execution.is_none());
        let content = r.content.unwrap().content.unwrap();
        match content {
            llm_chat_result::message_content::Content::Text(t) => assert_eq!(t, "hi"),
            other => panic!("expected Text, got {other:?}"),
        }
        let usage = r.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(4));
        assert_eq!(usage.completion_tokens, Some(1));
    }

    // build_chat_result_from_oai_json: tool_calls branch with parser-provided id.
    #[test]
    fn test_oai_json_to_chat_result_with_tool_calls() {
        let parsed = r#"{
            "role":"assistant",
            "content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]
        }"#;
        let r = build_chat_result_from_oai_json(parsed, 4, 8, 0.1, 0.2, "model").unwrap();
        assert_eq!(r.requires_tool_execution, Some(true));
        let pending = r.pending_tool_calls.unwrap();
        assert_eq!(pending.calls.len(), 1);
        assert_eq!(pending.calls[0].call_id, "c1");
        assert_eq!(pending.calls[0].fn_name, "f");
        assert_eq!(pending.calls[0].fn_arguments, r#"{"x":1}"#);
        let calls = match r.content.unwrap().content.unwrap() {
            llm_chat_result::message_content::Content::ToolCalls(t) => t.calls,
            other => panic!("expected ToolCalls, got {other:?}"),
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "c1");
        assert_eq!(calls[0].delta_index, Some(0));
    }

    // build_chat_result_from_oai_json: missing id is synthesized.
    #[test]
    fn test_oai_json_to_chat_result_synthesizes_missing_id() {
        let parsed = r#"{
            "role":"assistant",
            "content":null,
            "tool_calls":[{"type":"function","function":{"name":"f","arguments":"{}"}}]
        }"#;
        let r = build_chat_result_from_oai_json(parsed, 4, 4, 0.05, 0.05, "model").unwrap();
        let pending = r.pending_tool_calls.unwrap();
        assert_eq!(pending.calls.len(), 1);
        let id = &pending.calls[0].call_id;
        assert!(id.starts_with("call_0_"), "synthesized id: {id}");
        // Also reflected in MessageContent::ToolCalls
        let calls = match r.content.unwrap().content.unwrap() {
            llm_chat_result::message_content::Content::ToolCalls(t) => t.calls,
            other => panic!("expected ToolCalls, got {other:?}"),
        };
        assert_eq!(calls[0].call_id, *id);
    }

    // build_chat_result_from_oai_json: reasoning_content extraction.
    #[test]
    fn test_oai_json_to_chat_result_with_reasoning() {
        let parsed = r#"{"role":"assistant","content":"final","reasoning_content":"thinking"}"#;
        let r = build_chat_result_from_oai_json(parsed, 4, 4, 0.0, 0.0, "model").unwrap();
        assert_eq!(r.reasoning_content.as_deref(), Some("thinking"));
        match r.content.unwrap().content.unwrap() {
            llm_chat_result::message_content::Content::Text(t) => assert_eq!(t, "final"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // build_chat_result_from_oai_json: malformed JSON returns error.
    #[test]
    fn test_oai_json_to_chat_result_malformed_returns_error() {
        let err = build_chat_result_from_oai_json("not json", 0, 0, 0.0, 0.0, "model").unwrap_err();
        assert!(err.to_string().contains("tool response parser"));
    }

    // resolve_tool_choice: bare strings pass through unchanged so the call
    // site can forward inputs verbatim.
    #[test]
    fn test_resolve_tool_choice_string_passthrough() {
        let tools = r#"[{"type":"function","function":{"name":"f"}}]"#;
        for choice in [None, Some("auto"), Some("none"), Some("required")] {
            let r = resolve_tool_choice(tools, choice).expect("string passthrough");
            assert!(
                matches!(r, ResolvedToolChoice::Passthrough),
                "expected Passthrough for {choice:?}"
            );
        }
    }

    // resolve_tool_choice: OpenAI function-object form filters the tools list
    // to the named entry and forces the call via "required" — the model is
    // not allowed to fall back to plain text, matching OpenAI's contract.
    #[test]
    fn test_resolve_tool_choice_function_object_filters_tools() {
        let tools = r#"[
            {"type":"function","function":{"name":"a","parameters":{}}},
            {"type":"function","function":{"name":"b","parameters":{}}}
        ]"#;
        let r = resolve_tool_choice(
            tools,
            Some(r#"{"type":"function","function":{"name":"b"}}"#),
        )
        .expect("function-object resolution");
        let ResolvedToolChoice::FunctionSpecific {
            tools_json,
            tool_choice,
        } = r
        else {
            panic!("expected FunctionSpecific");
        };
        assert_eq!(tool_choice, "required");
        let filtered: Value = serde_json::from_str(&tools_json).unwrap();
        let arr = filtered.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["function"]["name"], "b");
    }

    // resolve_tool_choice: requesting a function that does not appear in
    // tools_json is an error rather than a silent empty list.
    #[test]
    fn test_resolve_tool_choice_function_not_found_errors() {
        let tools = r#"[{"type":"function","function":{"name":"a"}}]"#;
        let err = resolve_tool_choice(
            tools,
            Some(r#"{"type":"function","function":{"name":"missing"}}"#),
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    // resolve_tool_choice: bare unknown strings are rejected up front so
    // llama.cpp's `invalid_argument` never reaches the caller.
    #[test]
    fn test_resolve_tool_choice_unknown_bare_string_errors() {
        let tools = r#"[{"type":"function","function":{"name":"a"}}]"#;
        let err = resolve_tool_choice(tools, Some("nope")).unwrap_err();
        assert!(err.to_string().contains("unsupported tool_choice"));
    }

    // resolve_tool_choice: object missing function.name is rejected with a
    // pointer to the actual schema the caller should follow.
    #[test]
    fn test_resolve_tool_choice_object_missing_name_errors() {
        let tools = r#"[{"type":"function","function":{"name":"a"}}]"#;
        let err = resolve_tool_choice(tools, Some(r#"{"type":"function"}"#)).unwrap_err();
        assert!(err.to_string().contains("function.name"));
    }

    // fallback_parse_tool_calls: recognises a single <tool_call>...</tool_call>
    // envelope and produces the same JSON shape as the OAI parser would.
    #[test]
    fn test_fallback_parse_tool_calls_single() {
        let raw = "<|im_start|>assistant\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}\n</tool_call>";
        let json = fallback_parse_tool_calls(raw).expect("recovered tool call");
        let v: Value = serde_json::from_str(&json).unwrap();
        let calls = v["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Tokyo"}"#);
        assert!(v["content"].is_null());
    }

    // fallback_parse_tool_calls: extracts multiple envelopes when the model
    // emits parallel calls in sequence.
    #[test]
    fn test_fallback_parse_tool_calls_multiple() {
        let raw = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
                   <tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let json = fallback_parse_tool_calls(raw).expect("recovered");
        let v: Value = serde_json::from_str(&json).unwrap();
        let calls = v["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[1]["function"]["name"], "b");
        assert_eq!(calls[1]["function"]["arguments"], r#"{"x":1}"#);
    }

    // fallback_parse_tool_calls: ignores malformed envelopes (e.g. body
    // that isn't valid JSON) and returns None when nothing valid remains.
    #[test]
    fn test_fallback_parse_tool_calls_returns_none_on_no_match() {
        assert!(fallback_parse_tool_calls("plain text reply").is_none());
        assert!(
            fallback_parse_tool_calls("<tool_call>not json</tool_call>").is_none(),
            "malformed JSON should not produce a recovered call"
        );
    }

    // decode_oai_deltas: plain content delta.
    #[test]
    fn test_decode_oai_deltas_text_content() {
        let u = decode_oai_deltas(&[r#"{"content":"hello"}"#.to_string()]);
        assert_eq!(u.text, "hello");
        assert!(u.reasoning.is_empty());
        assert!(u.tool_calls.is_empty());
    }

    // decode_oai_deltas: reasoning split.
    #[test]
    fn test_decode_oai_deltas_reasoning() {
        let u = decode_oai_deltas(&[r#"{"reasoning_content":"thinking"}"#.to_string()]);
        assert!(u.text.is_empty());
        assert_eq!(u.reasoning, "thinking");
    }

    // decode_oai_deltas: first chunk of a tool call carries id + name + first
    // arguments fragment.
    #[test]
    fn test_decode_oai_deltas_tool_call_first_chunk() {
        let u = decode_oai_deltas(&[
            r#"{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{"}}]}"#.to_string(),
        ]);
        assert_eq!(u.tool_calls.len(), 1);
        let tc = &u.tool_calls[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("c1"));
        assert_eq!(tc.fn_name.as_deref(), Some("f"));
        assert_eq!(tc.arguments_chunk, "{");
    }

    // decode_oai_deltas: subsequent argument fragments lack id/name.
    #[test]
    fn test_decode_oai_deltas_tool_call_argument_chunks() {
        let u = decode_oai_deltas(&[
            r#"{"tool_calls":[{"index":0,"function":{"arguments":"\"x\""}}]}"#.to_string(),
            r#"{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}"#.to_string(),
        ]);
        assert_eq!(u.tool_calls.len(), 2);
        assert_eq!(u.tool_calls[0].id, None);
        assert_eq!(u.tool_calls[0].fn_name, None);
        assert_eq!(u.tool_calls[0].arguments_chunk, "\"x\"");
        assert_eq!(u.tool_calls[1].arguments_chunk, ":1}");
    }

    #[test]
    fn test_parsed_stream_deltas_emits_first_tool_call_with_empty_arguments() {
        let mut state = ParsedStreamDeltaState::new("");
        let now = ParsedChatMsg {
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![ParsedToolCall {
                id: None,
                name: "ping".to_string(),
                arguments: String::new(),
            }],
        };

        let deltas = parsed_stream_deltas(&mut state, &now)
            .expect("stream delta")
            .expect("non-regressing stream");

        assert_eq!(deltas.len(), 1);
        let delta: Value = serde_json::from_str(&deltas[0]).expect("delta JSON");
        let call = &delta["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert!(call["id"].as_str().is_some_and(|s| !s.is_empty()));
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "ping");
        assert_eq!(call["function"]["arguments"], "");
    }

    // ToolCallAccumulator: id/name carry only on the first delta of an index,
    // arguments concatenate in delta order, finalise produces sorted output.
    #[test]
    fn test_tool_call_accumulator_merges_partial_id_and_args() {
        let mut acc = ToolCallAccumulator::new();
        acc.ingest(&ToolCallDelta {
            index: 0,
            id: Some("c1".to_string()),
            fn_name: Some("get_weather".to_string()),
            arguments_chunk: "{\"city\":\"".to_string(),
        });
        acc.ingest(&ToolCallDelta {
            index: 0,
            id: None,
            fn_name: None,
            arguments_chunk: "Tokyo\"}".to_string(),
        });
        let calls = acc.finalize();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "c1");
        assert_eq!(calls[0].fn_name, "get_weather");
        assert_eq!(calls[0].fn_arguments, "{\"city\":\"Tokyo\"}");
    }

    // ToolCallAccumulator: id synthesis when the OAI parser never emitted one.
    #[test]
    fn test_tool_call_accumulator_assigns_uuid_when_id_missing() {
        let mut acc = ToolCallAccumulator::new();
        acc.ingest(&ToolCallDelta {
            index: 0,
            id: None,
            fn_name: Some("f".to_string()),
            arguments_chunk: "{}".to_string(),
        });
        let calls = acc.finalize();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].call_id.starts_with("call_0_"));
        assert_eq!(calls[0].fn_name, "f");
    }

    // decode_oai_deltas: malformed JSON entries are skipped, not fatal.
    #[test]
    fn test_decode_oai_deltas_skips_malformed_entries() {
        let u = decode_oai_deltas(&["not json".to_string(), r#"{"content":"ok"}"#.to_string()]);
        assert_eq!(u.text, "ok");
    }

    // ----------------- extract_enable_thinking -----------------

    #[test]
    fn test_extract_enable_thinking_returns_none_when_kwargs_absent() {
        assert!(extract_enable_thinking(None).is_none());
    }

    #[test]
    fn test_extract_enable_thinking_extracts_true_value() {
        let v = extract_enable_thinking(Some(r#"{"enable_thinking":true}"#));
        assert_eq!(v, Some(true));
    }

    #[test]
    fn test_extract_enable_thinking_extracts_false_value() {
        let v = extract_enable_thinking(Some(r#"{"enable_thinking":false}"#));
        assert_eq!(v, Some(false));
    }

    #[test]
    fn test_extract_enable_thinking_returns_none_when_key_missing() {
        let v = extract_enable_thinking(Some(r#"{"other_flag":true}"#));
        assert!(v.is_none());
    }

    #[test]
    fn test_extract_enable_thinking_returns_none_when_value_not_bool() {
        // String value: rejected (we don't coerce "true" to true).
        let v = extract_enable_thinking(Some(r#"{"enable_thinking":"true"}"#));
        assert!(v.is_none());
    }

    #[test]
    fn test_extract_enable_thinking_returns_none_on_invalid_json() {
        let v = extract_enable_thinking(Some("not json"));
        assert!(v.is_none());
    }

    #[test]
    fn test_extract_enable_thinking_returns_none_for_non_object_root() {
        // A bare boolean is not the documented kwargs shape (must be a
        // JSON object). Reject so the call falls back to the default.
        let v = extract_enable_thinking(Some("true"));
        assert!(v.is_none());
    }

    #[test]
    fn test_extract_enable_thinking_coexists_with_other_keys() {
        let v = extract_enable_thinking(Some(
            r#"{"some_other":42,"enable_thinking":true,"more":"x"}"#,
        ));
        assert_eq!(v, Some(true));
    }

    // parsed_msg_to_oai_json must emit the shape build_chat_result_from_oai_json
    // consumes, with a synthesized non-empty id.
    #[test]
    fn test_parsed_msg_to_oai_json_shape() {
        let msg = ParsedChatMsg {
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![ParsedToolCall {
                id: None,
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Tokyo"}"#.to_string(),
            }],
        };
        let json: Value = serde_json::from_str(&parsed_msg_to_oai_json(&msg).unwrap()).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            json["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Tokyo"}"#
        );
        assert!(
            json["tool_calls"][0]["id"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "id must be synthesized when absent"
        );
    }

    #[test]
    fn test_oai_stream_parser_rejects_unsupported_format() {
        let tmpl_result = ToolChatTemplateResult {
            prompt: String::new(),
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            chat_format: 99,
            parser: None,
            generation_prompt: String::new(),
            parse_tool_calls: true,
        };

        let err = match OaiStreamParser::for_template(&tmpl_result, "[]") {
            Ok(_) => panic!("unsupported chat_format should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("unsupported tool-calling chat_format 99"),
            "{err:?}"
        );
    }

    #[test]
    fn test_oai_stream_parser_respects_parse_tool_calls_false() {
        let tmpl_result = ToolChatTemplateResult {
            prompt: String::new(),
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            chat_format: QWEN_TAGGED_CHAT_FORMAT,
            parser: None,
            generation_prompt: "<|im_start|>assistant\n".to_string(),
            parse_tool_calls: false,
        };
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
        let mut parser = OaiStreamParser::for_template(&tmpl_result, "[]").expect("parser");

        let deltas = parser.update(raw, true).expect("stream update");
        let update = decode_oai_deltas(&deltas);

        assert_eq!(update.text, raw);
        assert!(update.tool_calls.is_empty());
    }
}
