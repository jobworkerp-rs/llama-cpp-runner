//! OpenAI-compatible helpers for the client-side tool-calling path.
//!
//! This module is intentionally kept separate from `model.rs`: the legacy
//! `build_chat_messages` path (no tools) continues to build `LlamaChatMessage`
//! and let llama.cpp apply the chat template the legacy way, while the
//! tool-aware path defined here goes through `apply_chat_template_oaicompat`
//! which requires a JSON-encoded messages array structured in OpenAI format.

use anyhow::{Result, anyhow, bail};
use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args;
use jobworkerp_llama_protobuf::protobuf::llm::{
    LlmChatResult, PendingToolCalls, ToolCallRequest, llm_chat_result,
};
use llama_cpp_2::model::{AddBos, GrammarTrigger, GrammarTriggerType, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde_json::{Value, json};

/// Aggregated chunk forwarded from the OAI streaming parser to the worker
/// thread. Built by [`decode_oai_deltas`] from one batch of `update()` output.
#[derive(Clone, Debug, Default)]
pub(crate) struct OaiStreamUpdate {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCallDelta>,
}

/// Parse a batch of OAI-compatible delta JSON strings (the slice returned by
/// `ChatParseStateOaicompat::update`) into a structured update. Unknown
/// delta shapes are silently ignored so a future parser variant cannot kill
/// the request. Uses typed serde deserialisation to avoid the `Value` heap
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

/// Single tool-call delta emitted by `ChatParseStateOaicompat::update`.
/// Mirrors OpenAI Chat Completions streaming semantics: `id` and `fn_name`
/// are populated only on the first delta of a given `index`; subsequent
/// deltas with the same `index` only carry `arguments_chunk`. The consumer
/// is responsible for accumulating arguments string fragments.
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
pub(crate) struct ToolCallAccumulator {
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
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn ingest(&mut self, delta: &ToolCallDelta) {
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

    pub(crate) fn is_empty(&self) -> bool {
        self.by_index.is_empty()
    }

    /// The next unused OAI call index — the position a brand-new tool call
    /// would be assigned. `append_chat_chunk` uses this to fold proto
    /// chunks (which carry no explicit index) into the latest open call.
    pub(crate) fn next_free_index(&self) -> u32 {
        self.by_index.keys().next_back().map_or(0, |i| i + 1)
    }

    /// Finalise into the proto `ToolCallRequest` list, in ascending index
    /// order. Missing ids are filled with the same synthesis used by the
    /// non-streaming path so the client gets a deterministic correlator.
    pub(crate) fn finalize(self) -> Vec<ToolCallRequest> {
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
    pub grammar_triggers: Vec<GrammarTrigger>,
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
    triggers: &[GrammarTrigger],
) -> (Vec<String>, Vec<LlamaToken>) {
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();
    for t in triggers {
        match t.trigger_type {
            GrammarTriggerType::Token => {
                if let Some(tok) = t.token {
                    tokens.push(tok);
                }
            }
            GrammarTriggerType::Word => match model.str_to_token(&t.value, AddBos::Never) {
                Ok(toks) if toks.len() == 1 => tokens.push(toks[0]),
                _ => patterns.push(regex::escape(&t.value)),
            },
            GrammarTriggerType::Pattern => patterns.push(t.value.clone()),
            GrammarTriggerType::PatternFull => {
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
    let mut set = std::collections::HashSet::new();
    for s in preserved {
        if let Ok(toks) = model.str_to_token(s, AddBos::Never)
            && toks.len() == 1
        {
            set.insert(toks[0]);
        }
    }
    set
}

/// Extract the `enable_thinking` boolean from a `chat_template_kwargs`
/// JSON object string. The kwargs payload is still forwarded verbatim to
/// `apply_chat_template_oaicompat` (so the jinja template sees the same
/// value), but the C++ side ALSO has a dedicated `enable_thinking`
/// parameter that controls grammar/parser behaviour. Without this
/// extraction the two channels can disagree (Qwen3 think mode is a
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
                bail!(
                    "ToolExecutionRequests is no longer accepted; \
                     use ToolResults on a TOOL message"
                );
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
pub(crate) fn fallback_parse_tool_calls(raw: &str) -> Option<String> {
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

/// Convert the JSON string produced by `ChatTemplateResult::parse_response_oaicompat`
/// into a wire-level `LlmChatResult`. Handles:
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
    let parsed: Value = serde_json::from_str(parsed_json).map_err(|e| {
        anyhow!("parse_response_oaicompat returned invalid JSON: {e}: {parsed_json}")
    })?;

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
        assert!(err.to_string().contains("parse_response_oaicompat"));
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
}
