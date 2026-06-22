//! Gemma4 tool-calling format: template renderer, GBNF grammar builder, and the
//! non-streaming/streaming response parsers. Split out of the root `oai_chat`
//! module so the Gemma4-specific tag handling lives next to the contract it
//! enforces. Shared building blocks (`ToolChatTemplateResult`, `GrammarSpec`,
//! `ParsedChatMsg`, the GBNF/stream helpers, ...) remain in the parent module
//! and are imported here.

use super::common::{
    GEMMA4_CHAT_FORMAT, GrammarSpec, ParsedChatMsg, ParsedStreamDeltaState, ParsedToolCall,
    ToolChatTemplateResult, ToolGrammarTrigger, ToolGrammarTriggerType, escape_gbnf_literal,
    parsed_stream_deltas, qwen_tool_function_name, qwen_tool_rule_names, render_template_once,
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

// The tool-call opener is the marker `<|tool_call>` followed by `call:`; the
// marker alone is matched when hiding a half-streamed opener, so derive the
// full opener from it to keep one source of truth.
const GEMMA4_TOOL_CALL_MARKER: &str = "<|tool_call>";
const GEMMA4_TOOL_CALL_OPEN: &str = "<|tool_call>call:";
const GEMMA4_TOOL_CALL_CLOSE: &str = "<tool_call|>";
const GEMMA4_STRING_DELIM: &str = "<|\"|>";
const GEMMA4_CHANNEL_OPEN: &str = "<|channel>";
// `GEMMA4_THOUGHT_OPEN` is the channel opener specialised to the thought
// channel, so derive it from the opener to keep one source of truth.
const GEMMA4_THOUGHT_OPEN: &str = "<|channel>thought";
const GEMMA4_CHANNEL_CLOSE: &str = "<channel|>";
const GEMMA4_TURN_MARKER: &str = "<|turn>";
// Tool-declaration block keyword inside the Gemma4 jinja template.
const GEMMA4_DECLARATION_KW: &str = "declaration:";
const GEMMA4_PRESERVED_TOKENS: &[&str] = &[
    GEMMA4_CHANNEL_OPEN,
    GEMMA4_CHANNEL_CLOSE,
    GEMMA4_TOOL_CALL_MARKER,
    GEMMA4_TOOL_CALL_CLOSE,
    GEMMA4_TURN_MARKER,
];

// Builds the Gemma4 tool-call GBNF grammar. Unlike the Qwen tagged grammar
// (`<function>/<parameter>` tags), Gemma4 emits each call as
// `<|tool_call>call:NAME{...}<tool_call|>` where the argument object is a
// JSON-ish dict whose string values are wrapped in `<|"|>` delimiters rather
// than double quotes — hence the dedicated `gemma4-*` value rules below.
//
// `require_tools` (tool_choice == "required") and `parallel_tool_calls` select
// the quantifier on the call envelope, mirroring the Qwen builder:
//   - auto, single      -> `(call)?`  (optional, lazy trigger)
//   - auto, parallel    -> `(call)*`
//   - required, single  -> `call`     (mandatory, eager)
//   - required, parallel-> `(call)+`
pub(in crate::oai_chat) fn build_gemma4_tool_call_grammar_spec(
    tools_json: &str,
    require_tools: bool,
    parallel_tool_calls: bool,
) -> Result<GrammarSpec> {
    let tools: Vec<Value> =
        serde_json::from_str(tools_json).map_err(|e| anyhow!("invalid tools_json: {e}"))?;
    if tools.is_empty() {
        bail!("tools_json must contain at least one tool");
    }
    let tool_rule_names = qwen_tool_rule_names(&tools)?;
    let mut grammar = String::from(
        "gemma4-array ::= \"[\" space (\"]\" | gemma4-value (\",\" space gemma4-value)* space \"]\")\n\
gemma4-bool ::= json-bool\n\
gemma4-dict ::= \"{\" space (\"}\" | gemma4-dict-kv (\",\" space gemma4-dict-kv)* space \"}\")\n\
gemma4-dict-key ::= gemma4-dict-key-name \":\"\n\
gemma4-dict-key-name ::= [^:}]+\n\
gemma4-dict-kv ::= gemma4-dict-key space gemma4-value\n\
gemma4-null ::= json-null\n\
gemma4-number ::= json-number\n\
gemma4-string ::= \"<|\\\"|>\" gemma4-string-content \"<|\\\"|>\"\n\
gemma4-string-content ::= ([^<] | \"<\" [^|] | \"<|\" [^\"] | \"<|\\\"\" [^|] | \"<|\\\"|\" [^>])*\n\
gemma4-value ::= gemma4-string | gemma4-dict | gemma4-array | gemma4-number | gemma4-bool | gemma4-null\n\
json-bool ::= (\"true\" | \"false\") space\n\
json-null ::= \"null\" space\n\
json-number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ((\"e\" | \"E\") [+-]? [0-9]+)? space\n\
root ::= tool-call\n\
space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n",
    );
    let alternatives = format!("({})", tool_rule_names.join(" | "));
    // The open/close markers carry no GBNF-special characters, so reference the
    // shared constants instead of re-spelling them and risking a desync.
    let call = format!("\"{GEMMA4_TOOL_CALL_OPEN}\" {alternatives} \"{GEMMA4_TOOL_CALL_CLOSE}\"");
    let tool_call = match (require_tools, parallel_tool_calls) {
        (false, false) => format!("({call})?"),
        (false, true) => format!("({call})*"),
        (true, false) => call,
        (true, true) => format!("({call})+"),
    };
    grammar.push_str(&format!("tool-call ::= {tool_call}\n"));
    for (tool, rule_name) in tools.iter().zip(tool_rule_names) {
        let name = qwen_tool_function_name(tool)?;
        grammar.push_str(&format!(
            "{rule_name} ::= (\"{}\") gemma4-dict\n",
            escape_gbnf_literal(name)
        ));
    }
    Ok(GrammarSpec {
        grammar,
        grammar_lazy: !require_tools,
        grammar_triggers: if require_tools {
            Vec::new()
        } else {
            vec![ToolGrammarTrigger {
                trigger_type: ToolGrammarTriggerType::Word,
                value: GEMMA4_TOOL_CALL_MARKER.to_string(),
                token: None,
            }]
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_gemma4_tool_chat_template(
    template: &str,
    messages_json: &str,
    tools_json: Option<&str>,
    tool_choice: Option<&str>,
    chat_template_kwargs: Option<&str>,
    add_generation_prompt: bool,
    enable_thinking: bool,
    parallel_tool_calls: bool,
) -> Result<ToolChatTemplateResult> {
    if tools_json.is_some() && !is_gemma4_tool_template(template) {
        bail!("unsupported chat template for Gemma4 tool calling");
    }

    // `tool_choice: "none"` keeps the tool declarations out of the prompt and
    // disables the tool-call grammar/parser, so gate every tool-dependent step
    // on a single flag rather than re-checking the string in each branch.
    let tools_enabled = tool_choice != Some("none");
    let messages: Value =
        serde_json::from_str(messages_json).map_err(|e| anyhow!("invalid messages_json: {e}"))?;
    let tools = match tools_json {
        Some(tools) if tools_enabled => Some(
            serde_json::from_str::<Value>(tools).map_err(|e| anyhow!("invalid tools_json: {e}"))?,
        ),
        _ => None,
    };
    let kwargs = chat_template_kwargs
        .map(serde_json::from_str::<Value>)
        .transpose()
        .map_err(|e| anyhow!("invalid chat_template_kwargs: {e}"))?;

    let prompt_without_generation = render_gemma4_template_once(
        template,
        &messages,
        tools.as_ref(),
        kwargs.as_ref(),
        enable_thinking,
        false,
    )?;
    let prompt_with_generation =
        trim_gemma4_empty_generation_thought(&render_gemma4_template_once(
            template,
            &messages,
            tools.as_ref(),
            kwargs.as_ref(),
            enable_thinking,
            true,
        )?)
        .to_string();
    // Compute the generation-prompt diff (both prompts borrowed) before moving
    // the selected one into `prompt`, so neither prompt needs to be cloned.
    let generation_prompt = prompt_with_generation
        .strip_prefix(&prompt_without_generation)
        .context("generation prompt is not an append-only suffix of the base prompt")?
        .to_string();
    let prompt = if add_generation_prompt {
        prompt_with_generation
    } else {
        prompt_without_generation
    };
    let require_tools = tool_choice == Some("required");
    let grammar_spec = tools_json
        .filter(|_| tools_enabled)
        .map(|tools| build_gemma4_tool_call_grammar_spec(tools, require_tools, parallel_tool_calls))
        .transpose()?;
    let (grammar, grammar_lazy, grammar_triggers) = grammar_spec
        .map(|spec| (Some(spec.grammar), spec.grammar_lazy, spec.grammar_triggers))
        .unwrap_or_default();

    Ok(ToolChatTemplateResult {
        prompt,
        grammar,
        grammar_lazy,
        grammar_triggers,
        preserved_tokens: GEMMA4_PRESERVED_TOKENS
            .iter()
            .map(|token| (*token).to_string())
            .collect(),
        additional_stops: Vec::new(),
        chat_format: GEMMA4_CHAT_FORMAT,
        parser: None,
        generation_prompt,
        parse_tool_calls: tools_json.is_some() && tools_enabled,
    })
}

// Gemma4 templates append an empty thought channel (`<|channel>thought\n
// <channel|>`) to the generation prompt. We strip it so the prompt ends at the
// model turn: keeping it would (a) break the append-only `strip_prefix` invariant
// the caller relies on to derive the generation prompt, and (b) pre-seed an empty
// reasoning block the model would otherwise fill on its own.
fn trim_gemma4_empty_generation_thought(prompt: &str) -> &str {
    prompt.strip_suffix(GEMMA4_EMPTY_THOUGHT).unwrap_or(prompt)
}

// The empty thought block `<|channel>thought\n<channel|>` = GEMMA4_THOUGHT_OPEN
// + "\n" + GEMMA4_CHANNEL_CLOSE. `concat!` takes literals only, so the parts are
// spelled out here; a compile-time assert below keeps them in sync with the
// constants that are the single source of truth.
const GEMMA4_EMPTY_THOUGHT: &str = "<|channel>thought\n<channel|>";
const _: () = assert!(
    GEMMA4_EMPTY_THOUGHT.len() == GEMMA4_THOUGHT_OPEN.len() + 1 + GEMMA4_CHANNEL_CLOSE.len()
);

/// Whether `template` is a Gemma4 tool template. Like
/// `is_qwen_tagged_tool_template`, the tag set is the heuristic: the Gemma4
/// call opener, the `declaration:` tool-block keyword, the turn marker, and the
/// `<|"|>` string delimiter together are specific enough to distinguish Gemma4
/// from the other tag-based templates the renderer routes to the legacy path.
pub(crate) fn is_gemma4_tool_template(template: &str) -> bool {
    template.contains(GEMMA4_TOOL_CALL_OPEN)
        && template.contains(GEMMA4_DECLARATION_KW)
        && template.contains(GEMMA4_TURN_MARKER)
        && template.contains(GEMMA4_STRING_DELIM)
}

// Gemma4 templates call neither the Qwen-only `tojson` filter nor
// `raise_exception`, so its env needs no extra setup.
fn render_gemma4_template_once(
    template: &str,
    messages: &Value,
    tools: Option<&Value>,
    kwargs: Option<&Value>,
    enable_thinking: bool,
    add_generation_prompt: bool,
) -> Result<String> {
    render_template_once(
        template,
        messages,
        tools,
        kwargs,
        enable_thinking,
        add_generation_prompt,
        |_env| {},
    )
}

struct Gemma4ValueParser<'a> {
    raw: &'a str,
    cursor: usize,
    allow_partial: bool,
}

impl<'a> Gemma4ValueParser<'a> {
    fn new(raw: &'a str, allow_partial: bool) -> Self {
        Self {
            raw,
            cursor: 0,
            allow_partial,
        }
    }

    fn parse_value(&mut self) -> Result<String> {
        self.skip_space();
        let rem = self.remaining();
        if rem.starts_with(GEMMA4_STRING_DELIM) {
            self.parse_string()
        } else if rem.starts_with('{') {
            self.parse_dict()
        } else if rem.starts_with('[') {
            self.parse_array()
        } else {
            self.parse_scalar()
        }
    }

    fn parse_dict(&mut self) -> Result<String> {
        self.expect_char('{')?;
        self.skip_space();
        let mut fields = Vec::new();
        if self.at_end() && self.allow_partial {
            return Ok("{".to_string());
        }
        if self.consume_char('}') {
            return Ok("{}".to_string());
        }
        loop {
            self.skip_space();
            if self.at_end() {
                return self.partial_or_err("{...} Gemma4 dict");
            }
            let (key, key_closed) = self.parse_key()?;
            if !key_closed && self.allow_partial {
                return Ok(format!("{{{}", json_string_unterminated(&key)));
            }
            self.skip_space();
            self.expect_char(':')?;
            self.skip_space();
            if self.at_end() && self.allow_partial {
                return Ok(format!("{{{}", fields.join(",")));
            }
            let value = self.parse_value()?;
            fields.push(format!("{}:{value}", json_string(&key)));
            self.skip_space();
            if self.consume_char(',') {
                continue;
            }
            if self.consume_char('}') {
                return Ok(format!("{{{}}}", fields.join(",")));
            }
            if self.at_end() && self.allow_partial {
                return Ok(format!("{{{}", fields.join(",")));
            }
            bail!("invalid Gemma4 dict near {:?}", self.remaining());
        }
    }

    fn parse_array(&mut self) -> Result<String> {
        self.expect_char('[')?;
        self.skip_space();
        let mut values = Vec::new();
        if self.at_end() && self.allow_partial {
            return Ok("[".to_string());
        }
        if self.consume_char(']') {
            return Ok("[]".to_string());
        }
        loop {
            self.skip_space();
            if self.at_end() {
                return self.partial_or_err("[...] Gemma4 array");
            }
            values.push(self.parse_value()?);
            self.skip_space();
            if self.consume_char(',') {
                continue;
            }
            if self.consume_char(']') {
                return Ok(format!("[{}]", values.join(",")));
            }
            if self.at_end() && self.allow_partial {
                return Ok(format!("[{}", values.join(",")));
            }
            bail!("invalid Gemma4 array near {:?}", self.remaining());
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.cursor += GEMMA4_STRING_DELIM.len();
        if let Some(close_rel) = self.remaining().find(GEMMA4_STRING_DELIM) {
            let value = &self.raw[self.cursor..self.cursor + close_rel];
            self.cursor += close_rel + GEMMA4_STRING_DELIM.len();
            return Ok(json_string(value));
        }
        if self.allow_partial {
            let value = self.remaining();
            self.cursor = self.raw.len();
            return Ok(json_string_unterminated(value));
        }
        bail!("unterminated Gemma4 string")
    }

    fn parse_scalar(&mut self) -> Result<String> {
        let start = self.cursor;
        while let Some(ch) = self.peek_char() {
            if ch == ',' || ch == '}' || ch == ']' || ch.is_whitespace() {
                break;
            }
            self.cursor += ch.len_utf8();
        }
        if start == self.cursor {
            return self.partial_or_err("Gemma4 scalar");
        }
        let scalar = &self.raw[start..self.cursor];
        match scalar {
            "true" | "false" | "null" => Ok(scalar.to_string()),
            _ => {
                // Validate complete scalars; partial numeric fragments are
                // still useful for streaming JSON argument deltas.
                if self.allow_partial && self.at_end() {
                    return Ok(scalar.to_string());
                }
                serde_json::from_str::<Value>(scalar)
                    .with_context(|| format!("invalid Gemma4 scalar `{scalar}`"))?;
                Ok(scalar.to_string())
            }
        }
    }

    fn parse_key(&mut self) -> Result<(String, bool)> {
        let start = self.cursor;
        let mut closed = false;
        while let Some(ch) = self.peek_char() {
            if ch == ':' || ch == '}' {
                closed = ch == ':';
                break;
            }
            self.cursor += ch.len_utf8();
        }
        if start == self.cursor {
            return self.partial_or_err("Gemma4 dict key");
        }
        Ok((self.raw[start..self.cursor].to_string(), closed))
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        if self.consume_char(expected) {
            return Ok(());
        }
        if self.allow_partial && self.at_end() {
            return Ok(());
        }
        bail!(
            "expected `{expected}` in Gemma4 value near {:?}",
            self.remaining()
        )
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.cursor += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn skip_space(&mut self) {
        while let Some(ch) = self.peek_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.cursor += ch.len_utf8();
        }
    }

    fn partial_or_err<T>(&self, what: &str) -> Result<T> {
        if self.allow_partial {
            bail!("partial {what}")
        }
        bail!("unterminated {what}")
    }

    fn remaining(&self) -> &'a str {
        &self.raw[self.cursor..]
    }

    fn at_end(&self) -> bool {
        self.cursor >= self.raw.len()
    }

    fn peek_char(&self) -> Option<char> {
        self.remaining().chars().next()
    }
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).expect("serializing string cannot fail")
}

// JSON-encode a string but drop the closing quote, for a mid-stream string
// value whose end has not arrived yet (the next chunk continues it).
fn json_string_unterminated(value: &str) -> String {
    let quoted = json_string(value);
    quoted.strip_suffix('"').unwrap_or(&quoted).to_string()
}

fn split_gemma4_reasoning(raw: &str) -> (String, Option<String>) {
    let Some(open) = raw.find(GEMMA4_THOUGHT_OPEN) else {
        return (raw.to_string(), None);
    };
    let body_start = open + GEMMA4_THOUGHT_OPEN.len();
    let body_start = raw[body_start..]
        .find('\n')
        .map_or(body_start, |rel| body_start + rel + 1);
    let Some(close_rel) = raw[body_start..].find(GEMMA4_CHANNEL_CLOSE) else {
        let reasoning = raw[body_start..].trim().to_string();
        let content = raw[..open].trim_end().to_string();
        return (content, (!reasoning.is_empty()).then_some(reasoning));
    };
    let close = body_start + close_rel;
    let reasoning = raw[body_start..close].trim().to_string();
    // Splice the text around the channel block back together. trim_end/trim_start
    // drop the newlines that framed `<|channel>thought ... <channel|>` so the
    // rejoined content has no blank gap where the reasoning was removed.
    let mut content = String::new();
    content.push_str(raw[..open].trim_end());
    content.push_str(raw[close + GEMMA4_CHANNEL_CLOSE.len()..].trim_start());
    (content, (!reasoning.is_empty()).then_some(reasoning))
}

fn parse_gemma4_tool_call(
    raw: &str,
    start: usize,
    allow_partial: bool,
) -> Result<(Option<ParsedToolCall>, usize)> {
    let name_start = start + GEMMA4_TOOL_CALL_OPEN.len();
    let Some(args_rel) = raw[name_start..].find('{') else {
        if allow_partial {
            return Ok((None, raw.len()));
        }
        bail!("Gemma4 tool call has no argument dict");
    };
    let args_start = name_start + args_rel;
    let name = raw[name_start..args_start].to_string();
    if name.is_empty() {
        bail!("Gemma4 tool call has an empty name");
    }
    let envelope_end = raw[args_start..]
        .find(GEMMA4_TOOL_CALL_CLOSE)
        .map(|rel| args_start + rel);
    let args_end = envelope_end.unwrap_or(raw.len());
    let mut parser = Gemma4ValueParser::new(&raw[args_start..args_end], allow_partial);
    let arguments = parser.parse_value()?;
    let next = envelope_end.map_or(raw.len(), |end| end + GEMMA4_TOOL_CALL_CLOSE.len());
    Ok((
        Some(ParsedToolCall {
            id: None,
            name,
            arguments,
        }),
        next,
    ))
}

pub(crate) fn parse_gemma4_response(raw: &str, allow_partial: bool) -> Result<ParsedChatMsg> {
    let (without_reasoning, reasoning_content) = split_gemma4_reasoning(raw);
    let mut tool_calls = Vec::new();
    let mut cursor = 0;
    let mut first_tool_start = None;
    while let Some(rel) = without_reasoning[cursor..].find(GEMMA4_TOOL_CALL_OPEN) {
        let start = cursor + rel;
        first_tool_start.get_or_insert(start);
        let (call, next) = parse_gemma4_tool_call(&without_reasoning, start, allow_partial)?;
        // parse_gemma4_tool_call already bails on an empty name, so any Some is
        // a usable call.
        if let Some(call) = call {
            tool_calls.push(call);
        }
        if next <= cursor {
            break;
        }
        cursor = next;
    }
    let content = match first_tool_start {
        Some(start) => without_reasoning[..start].to_string(),
        None => without_reasoning,
    };
    Ok(ParsedChatMsg {
        content,
        reasoning_content,
        tool_calls,
    })
}

#[derive(Debug)]
pub(crate) struct Gemma4StreamState {
    raw: String,
    diff: ParsedStreamDeltaState,
}

impl Gemma4StreamState {
    pub(in crate::oai_chat) fn new(generation_prompt: &str) -> Self {
        Self {
            raw: String::new(),
            diff: ParsedStreamDeltaState::new(generation_prompt),
        }
    }

    pub(in crate::oai_chat) fn update(
        &mut self,
        chunk: &str,
        is_streaming: bool,
    ) -> Result<Vec<String>> {
        self.raw.push_str(chunk);
        let now = parse_gemma4_stream_msg(&self.raw, is_streaming)?;
        let Some(deltas) = parsed_stream_deltas(&mut self.diff, &now)? else {
            return Ok(Vec::new());
        };
        Ok(deltas)
    }
}

fn parse_gemma4_stream_msg(raw: &str, allow_partial: bool) -> Result<ParsedChatMsg> {
    let mut msg = parse_gemma4_response(raw, allow_partial)?;
    if msg.tool_calls.is_empty() {
        // Hide a pending tool-call opener in place (no realloc), mirroring the
        // Qwen path.
        let keep = trim_pending_gemma4_tool_call_prefix(&msg.content).len();
        msg.content.truncate(keep);
    }
    Ok(msg)
}

fn trim_pending_gemma4_tool_call_prefix(content: &str) -> &str {
    debug_assert!(GEMMA4_TOOL_CALL_OPEN.starts_with(GEMMA4_TOOL_CALL_MARKER));
    if let Some(open_start) = content.rfind(GEMMA4_TOOL_CALL_MARKER) {
        let pending = &content[open_start..];
        if GEMMA4_TOOL_CALL_OPEN.starts_with(pending) || pending.starts_with(GEMMA4_TOOL_CALL_OPEN)
        {
            return &content[..open_start];
        }
    }
    for prefix_len in (1..=GEMMA4_TOOL_CALL_OPEN.len()).rev() {
        let prefix = &GEMMA4_TOOL_CALL_OPEN[..prefix_len];
        if content.ends_with(prefix) {
            return &content[..content.len() - prefix_len];
        }
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oai_chat::common::normalize_stream_delta_ids;
    use crate::oai_chat::extract_enable_thinking;

    #[test]
    fn test_build_gemma4_tool_call_grammar_spec_matches_golden() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/gemma4_e4b_auto_tool_call.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("read gemma4 golden fixture");
        let fixture: Value = serde_json::from_str(&raw).expect("golden fixture JSON");
        let expected_grammar = fixture
            .pointer("/template/grammar")
            .and_then(Value::as_str)
            .expect("fixture template.grammar");
        let expected_trigger = fixture
            .pointer("/template/grammar_triggers/0/value")
            .and_then(Value::as_str)
            .expect("fixture grammar trigger value");
        let tools_json =
            serde_json::to_string(fixture.pointer("/params/tools_json").unwrap()).unwrap();

        let spec = build_gemma4_tool_call_grammar_spec(&tools_json, false, false)
            .expect("gemma4 tool grammar");

        assert_eq!(spec.grammar.trim_end(), expected_grammar.trim_end());
        assert!(spec.grammar_lazy);
        assert_eq!(spec.grammar_triggers.len(), 1);
        assert_eq!(spec.grammar_triggers[0].value, expected_trigger);
    }

    #[test]
    fn test_build_gemma4_tool_call_grammar_spec_rejects_missing_name() {
        let tools = r#"[{"type":"function","function":{"parameters":{"type":"object"}}}]"#;

        let err = build_gemma4_tool_call_grammar_spec(tools, false, false).unwrap_err();

        assert!(err.to_string().contains("missing function.name"));
    }

    #[test]
    fn test_build_gemma4_tool_call_grammar_spec_rejects_empty_tools() {
        let err = build_gemma4_tool_call_grammar_spec("[]", false, false).unwrap_err();

        assert!(err.to_string().contains("at least one tool"));
    }

    #[test]
    fn test_build_gemma4_tool_call_grammar_spec_escapes_function_name_literal() {
        let tools = r#"[{"type":"function","function":{"name":"quote\"and\\slash","description":"Escaped name.","parameters":{"type":"object","properties":{}}}}]"#;

        let spec = build_gemma4_tool_call_grammar_spec(tools, false, false)
            .expect("gemma4 escaped-name grammar");

        assert!(
            spec.grammar
                .contains(r#"tool-quote-and-slash ::= ("quote\"and\\slash") gemma4-dict"#),
            "{}",
            spec.grammar
        );
    }

    #[test]
    fn test_render_gemma4_tool_chat_template_matches_golden() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/gemma4_e4b_auto_tool_call.json"
        );
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/google-gemma-4-31B-it.jinja"
        );
        let fixture_raw = std::fs::read_to_string(fixture_path).expect("read gemma4 fixture");
        let template = std::fs::read_to_string(template_path).expect("read gemma4 template");
        let fixture: Value = serde_json::from_str(&fixture_raw).expect("fixture JSON");
        let messages_json =
            serde_json::to_string(fixture.pointer("/params/messages_json").unwrap()).unwrap();
        let tools_json =
            serde_json::to_string(fixture.pointer("/params/tools_json").unwrap()).unwrap();
        let kwargs_json =
            serde_json::to_string(fixture.pointer("/params/chat_template_kwargs").unwrap())
                .unwrap();

        let rendered = render_gemma4_tool_chat_template(
            &template,
            &messages_json,
            Some(&tools_json),
            fixture
                .pointer("/params/tool_choice")
                .and_then(Value::as_str),
            Some(&kwargs_json),
            true,
            extract_enable_thinking(Some(&kwargs_json)).unwrap_or(false),
            false,
        )
        .expect("render gemma4 template");

        assert_eq!(
            rendered.prompt,
            fixture
                .pointer("/template/prompt")
                .unwrap()
                .as_str()
                .unwrap()
        );
        assert_eq!(
            rendered.grammar.as_deref().map(str::trim_end),
            fixture
                .pointer("/template/grammar")
                .unwrap()
                .as_str()
                .map(str::trim_end)
        );
        assert!(rendered.grammar_lazy);
        assert_eq!(rendered.grammar_triggers.len(), 1);
        assert_eq!(
            rendered.grammar_triggers[0].value,
            fixture
                .pointer("/template/grammar_triggers/0/value")
                .unwrap()
                .as_str()
                .unwrap()
        );
        assert_eq!(
            rendered.preserved_tokens,
            serde_json::from_value::<Vec<String>>(
                fixture
                    .pointer("/template/preserved_tokens")
                    .unwrap()
                    .clone()
            )
            .unwrap()
        );
        assert_eq!(
            rendered.generation_prompt,
            fixture
                .pointer("/template/generation_prompt")
                .unwrap()
                .as_str()
                .unwrap()
        );
        assert_eq!(rendered.additional_stops, Vec::<String>::new());
        assert_eq!(rendered.chat_format, GEMMA4_CHAT_FORMAT);
        assert!(rendered.parser.is_none());
        assert!(rendered.parse_tool_calls);
    }

    #[test]
    fn test_render_gemma4_tool_chat_template_forwards_required_and_parallel() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/google-gemma-4-31B-it.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read gemma4 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping","description":"Ping.","parameters":{"type":"object","properties":{}}}}]"#;

        let rendered = render_gemma4_tool_chat_template(
            &template,
            messages,
            Some(tools),
            Some("required"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            true,
        )
        .expect("render required+parallel gemma4");

        assert!(!rendered.grammar_lazy, "required must be eager");
        assert!(rendered.grammar_triggers.is_empty());
        let grammar = rendered.grammar.expect("required tools yield a grammar");
        assert!(
            grammar.contains("tool-call ::= (\"<|tool_call>call:\" (tool-ping) \"<tool_call|>\")+")
        );
    }

    #[test]
    fn test_render_gemma4_tool_chat_template_tool_choice_none_disables_tool_grammar() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/google-gemma-4-31B-it.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read gemma4 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping","description":"Ping.","parameters":{"type":"object","properties":{}}}}]"#;

        let rendered = render_gemma4_tool_chat_template(
            &template,
            messages,
            Some(tools),
            Some("none"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            false,
        )
        .expect("render gemma4 without tool grammar");

        assert!(!rendered.prompt.contains("<|tool>"));
        assert!(rendered.grammar.is_none());
        assert!(!rendered.grammar_lazy);
        assert!(rendered.grammar_triggers.is_empty());
        assert!(!rendered.parse_tool_calls);
        assert_eq!(rendered.chat_format, GEMMA4_CHAT_FORMAT);
    }

    #[test]
    fn test_render_gemma4_tool_chat_template_kwargs_cannot_override_reserved_keys() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/google-gemma-4-31B-it.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read gemma4 template");
        let messages = r#"[{"role":"user","content":"keep me"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping","description":"Ping.","parameters":{"type":"object","properties":{}}}}]"#;
        let kwargs = r#"{"messages":[{"role":"user","content":"INJECTED"}],"tools":[],"enable_thinking":true}"#;

        let rendered = render_gemma4_tool_chat_template(
            &template,
            messages,
            Some(tools),
            Some("auto"),
            Some(kwargs),
            true,
            false,
            false,
        )
        .expect("render with hostile kwargs");

        assert!(rendered.prompt.contains("keep me"));
        assert!(!rendered.prompt.contains("INJECTED"));
        assert!(rendered.prompt.contains("<|tool>declaration:ping"));
        assert!(rendered.prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn test_parse_gemma4_single_tool_matches_golden() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/gemma4_e4b_auto_tool_call.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("read gemma4 golden fixture");
        let fixture: Value = serde_json::from_str(&raw).expect("golden fixture JSON");
        let raw_response = fixture
            .pointer("/parse/0/raw_response")
            .and_then(Value::as_str)
            .expect("fixture parse[0].raw_response");
        let expected = fixture
            .pointer("/parse/0/parsed_json")
            .expect("fixture parse[0].parsed_json");

        let msg = parse_gemma4_response(raw_response, false).expect("parse gemma4 response");

        assert_eq!(msg.tool_calls.len(), 1);
        let call = &msg.tool_calls[0];
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"city":"Tokyo"}"#);
        assert_eq!(
            call.name,
            expected
                .pointer("/tool_calls/0/function/name")
                .and_then(Value::as_str)
                .unwrap()
        );
        assert_eq!(
            call.arguments,
            expected
                .pointer("/tool_calls/0/function/arguments")
                .and_then(Value::as_str)
                .unwrap()
        );
    }

    #[test]
    fn test_parse_gemma4_plain_text_no_tool() {
        let msg = parse_gemma4_response("Hello from Gemma", false).expect("parse plain gemma4");

        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.content, "Hello from Gemma");
        assert!(msg.reasoning_content.is_none());
    }

    #[test]
    fn test_parse_gemma4_values_to_json_arguments() {
        let raw = concat!(
            "<|tool_call>call:mix{",
            "text:<|\"|>a <tag> value<|\"|>,",
            "count:42,",
            "ok:true,",
            "none:null,",
            "items:[<|\"|>x<|\"|>,3],",
            "meta:{nested:<|\"|>yes<|\"|>}",
            "}<tool_call|>"
        );

        let msg = parse_gemma4_response(raw, false).expect("parse mixed gemma4 values");

        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "mix");
        assert_eq!(
            msg.tool_calls[0].arguments,
            r#"{"text":"a <tag> value","count":42,"ok":true,"none":null,"items":["x",3],"meta":{"nested":"yes"}}"#
        );
    }

    #[test]
    fn test_parse_gemma4_partial_tool_call_is_best_effort() {
        let raw = "<|tool_call>call:get_weather{city:<|\"|>Tok";

        let msg = parse_gemma4_response(raw, true).expect("partial gemma4 parse");

        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "get_weather");
        assert_eq!(msg.tool_calls[0].arguments, r#"{"city":"Tok"#);
    }

    #[test]
    fn test_parse_gemma4_partial_dict_key_waits_for_value() {
        let raw = "<|tool_call>call:get_weather{city:";

        let msg = parse_gemma4_response(raw, true).expect("partial gemma4 parse");

        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "get_weather");
        assert_eq!(msg.tool_calls[0].arguments, "{");
    }

    #[test]
    fn test_gemma4_stream_replays_golden_chunks() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/gemma4_e4b_auto_tool_call.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("read gemma4 golden fixture");
        let fixture: Value = serde_json::from_str(&raw).expect("golden fixture JSON");
        let generation_prompt = fixture
            .pointer("/template/generation_prompt")
            .and_then(Value::as_str)
            .expect("fixture generation_prompt");
        let chunks = fixture
            .pointer("/streaming/0/chunks")
            .and_then(Value::as_array)
            .expect("fixture chunks");
        let expected = fixture
            .pointer("/streaming/0/deltas")
            .and_then(Value::as_array)
            .expect("fixture deltas");
        let mut state = Gemma4StreamState::new(generation_prompt);

        for (chunk, expected_deltas) in chunks.iter().zip(expected) {
            let chunk = chunk.as_str().expect("chunk string");
            let deltas = state.update(chunk, true).expect("stream update");
            let mut actual: Vec<Value> = deltas
                .iter()
                .map(|s| serde_json::from_str(s).expect("delta JSON"))
                .collect();
            let mut expected_deltas = expected_deltas.as_array().unwrap().clone();
            normalize_stream_delta_ids(&mut actual);
            normalize_stream_delta_ids(&mut expected_deltas);
            assert_eq!(actual, expected_deltas, "chunk {chunk:?}");
        }
    }
}
