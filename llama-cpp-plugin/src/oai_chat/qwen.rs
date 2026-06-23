//! Qwen3.5 tagged tool-calling format: template renderer, GBNF grammar builder,
//! and the non-streaming/streaming response parsers. Split out of the root
//! `oai_chat` module so the Qwen-specific `<tool_call>/<function=>/<parameter=>`
//! tag handling lives next to the contract it enforces. Shared building blocks
//! (`ToolChatTemplateResult`, `GrammarSpec`, `ParsedChatMsg`, the GBNF/stream
//! helpers, ...) remain in the parent module and are imported here.

use super::common::{
    GrammarSpec, ParsedChatMsg, ParsedStreamDeltaState, ParsedToolCall, QWEN_TAGGED_CHAT_FORMAT,
    THINK_CLOSE, THINK_OPEN, ToolChatTemplateResult, ToolGrammarTrigger, ToolGrammarTriggerType,
    dedup_rule_name, escape_gbnf_literal, grammar_rule_suffix, parsed_stream_deltas,
    push_json_string_fragment, qwen_tool_function_name, qwen_tool_rule_names, render_template_once,
};
use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::json_schema_to_grammar;
use serde_json::Value;

// Qwen3.5 tagged-response markers. Single source of truth shared by the grammar
// builder, the preserved-token table, and both the non-streaming and streaming
// tag walkers, so a tag-shape change cannot desync one path from another.
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

const QWEN_TAGGED_PRESERVED_TOKENS: &[&str] = &[
    THINK_OPEN,
    THINK_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    FUNCTION_OPEN,
    ">",
    FUNCTION_CLOSE,
    PARAMETER_OPEN,
    PARAMETER_CLOSE,
];

// Grammar rule body that matches "any text not ending the </parameter>\n
// closer". Each alternative consumes one char as long as it cannot start the
// closing tag at that position — a hand-rolled prefix automaton for the literal
// `</parameter>\n`. Kept as a named constant so the per-character expansion can
// be verified at a glance and changed in one place if the closer ever changes.
const UNTIL_PARAMETER_CLOSE_RULE: &str = concat!(
    "until-suffix ::= (",
    "[^<]",
    " | \"<\" [^/]",
    " | \"</\" [^p]",
    " | \"</p\" [^a]",
    " | \"</pa\" [^r]",
    " | \"</par\" [^a]",
    " | \"</para\" [^m]",
    " | \"</param\" [^e]",
    " | \"</parame\" [^t]",
    " | \"</paramet\" [^e]",
    " | \"</paramete\" [^r]",
    " | \"</parameter\" [^>]",
    " | \"</parameter>\" [^\\n]",
    ")*\n",
);

// The value-representation contract shared between grammar generation
// (`qwen_parameter_value_rule`) and parse-back (`qwen_tagged_arguments_to_json`),
// mirroring how the reference Qwen3.5 tagged template treats each argument.
// In the reference impl (chat.cpp `common_chat_params_init_qwen3_coder`-style
// builder) a parameter is emitted as raw `until(</parameter>)` text exactly when
// `common_schema_info::resolves_to_string` is true; otherwise it is emitted via
// `p.json(schema)`. We mirror that split here:
//   - `RawString`: `resolves_to_string` is true (a plain `string` type, a
//     `string` const/enum, a `pattern`/`minLength`/`maxLength` schema, or a
//     string `format`). The body is verbatim bare text — no quotes on the wire —
//     and is wrapped into a JSON string as-is on parse-back. Per the reference
//     impl the grammar does NOT constrain the value range even for enum/const;
//     out-of-range values are left to downstream validation.
//   - `JsonLiteral`: the body is itself JSON (number/bool/object/array, or any
//     non-string const/enum) — parsed back into the corresponding JSON value.
// The grammar and the parser must agree on this kind per parameter, otherwise a
// value the grammar accepted cannot be reconstructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QwenToolParameterValueKind {
    RawString,
    JsonLiteral,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenToolParameterSpec {
    name: String,
    value_kind: QwenToolParameterValueKind,
    // Whether the grammar emits this parameter unconditionally. Optional
    // parameters are `*`-quantified (0-or-more) in the grammar, so the model may
    // omit them; parse-back must skip an absent optional rather than error.
    required: bool,
}

#[derive(Clone, Debug)]
struct QwenToolGrammarParameter {
    spec: QwenToolParameterSpec,
    parameter_rule_name: String,
    // What the parameter rule references inline for the value: either a value
    // rule name (e.g. `tool-x-arg-y-value`) or a self-contained GBNF fragment
    // (e.g. `"" until-suffix` for unconstrained strings).
    value_expression: String,
    // Standalone rule line(s) to append to the grammar that define the value
    // rule. Empty when `value_expression` is an inline fragment that needs no
    // separate definition.
    value_rule_definition: String,
}

// Phase 1 of the tool-calling Rust-ification (see
// `ai-docs/llama-cpp-rs-upgrade-plan.md`): builds the Qwen3.5-style
// `<tool_call>/<function>/<parameter>` GBNF grammar without the fork-only OAI
// API. Reached from `render_qwen_tool_chat_template`, which
// `apply_oai_template_with_tools` now routes the prompt/grammar through.
//
// `require_tools` (tool_choice == "required") and `parallel_tool_calls` mirror
// the reference builder in llama.cpp `common/chat.cpp`:
//   - required  -> the tool call is mandatory (no leading free text) and the
//                  grammar is eager (constrains from `root`, no lazy trigger),
//                  so the model cannot answer with plain text.
//   - auto      -> the tool call is optional and the grammar is lazy, armed by
//                  the `<tool_call>\n` trigger; free text is allowed until then.
//   - parallel  -> the `<tool_call>` envelope may hold more than one function
//                  invocation (`call (space call)*`) instead of exactly one.
pub(crate) fn build_qwen_tool_call_grammar_spec(
    tools_json: &str,
    require_tools: bool,
    parallel_tool_calls: bool,
) -> Result<GrammarSpec> {
    let tools: Value =
        serde_json::from_str(tools_json).map_err(|e| anyhow!("invalid tools_json: {e}"))?;
    let tools = tools
        .as_array()
        .ok_or_else(|| anyhow!("tools_json must be an array"))?;
    if tools.is_empty() {
        bail!("tools_json must contain at least one tool");
    }

    let tool_rule_names = qwen_tool_rule_names(tools)?;
    let mut tool_rules = Vec::with_capacity(tools.len());
    for (tool_index, tool) in tools.iter().enumerate() {
        let name = qwen_tool_function_name(tool)?;
        let rule_name = &tool_rule_names[tool_index];
        let parameters = qwen_tool_grammar_parameters(tool, rule_name)?;

        let mut rule = format!(
            "{rule_name} ::= (\"<function=\" \"{}\" \">\\n\") space",
            escape_gbnf_literal(name)
        );
        let mut param_rule_defs = String::new();
        let mut optional_rule_names = Vec::new();
        let mut required_count = 0;
        for parameter in &parameters {
            if parameter.spec.required {
                if required_count > 0 {
                    rule.push_str(" space");
                }
                rule.push(' ');
                rule.push_str(&parameter.parameter_rule_name);
                required_count += 1;
            } else {
                optional_rule_names.push(parameter.parameter_rule_name.as_str());
            }
            let parameter_name = &parameter.spec.name;
            param_rule_defs.push_str(&format!(
                "{} ::= (\"<parameter=\" \"{}\" \">\\n\") {} \"</parameter>\\n\"\n",
                parameter.parameter_rule_name,
                escape_gbnf_literal(parameter_name),
                parameter.value_expression
            ));
            param_rule_defs.push_str(&parameter.value_rule_definition);
        }
        if !optional_rule_names.is_empty() {
            let optional_rule_name = format!("{rule_name}-optional-parameters");
            param_rule_defs.push_str(&format!(
                "{optional_rule_name} ::= ({})\n",
                optional_rule_names.join(" | ")
            ));
            if required_count > 0 {
                rule.push_str(" space");
            }
            rule.push(' ');
            rule.push_str(&optional_rule_name);
            rule.push('*');
        }
        rule.push_str(" space \"</function>\\n\"\n");
        rule.push_str(&param_rule_defs);

        tool_rules.push(rule);
    }

    let alternatives = tool_rule_names.join(" | ");
    // A single tool-call envelope. With parallel_tool_calls the envelope may
    // carry more than one function invocation, matching the reference builder's
    // `call (space call)*`.
    let calls = if parallel_tool_calls {
        format!("({alternatives}) (space ({alternatives}))*")
    } else {
        format!("({alternatives})")
    };
    // `root` references `tool-call` for both required and auto. The difference
    // is the lazy machinery, not the rule body: when lazy (auto) the grammar
    // only constrains after the `<tool_call>\n` trigger fires, so free text
    // before it stands in for the reference builder's `optional(tool-call)`.
    // When eager (required) `root ::= tool-call` constrains from the start, so
    // a tool call is mandatory. This matches the C++ golden, which emits
    // `root ::= tool-call` for auto as well.
    let mut grammar = format!(
        "root ::= tool-call\n\
         space ::= | \" \" | \"\\n\"{{1,2}} [ \\t]{{0,20}}\n\
         tool-call ::= \"<tool_call>\\n\" space {calls} space \"</tool_call>\"\n"
    );
    for rule in tool_rules {
        grammar.push_str(&rule);
    }
    grammar.push_str(UNTIL_PARAMETER_CLOSE_RULE);

    Ok(GrammarSpec {
        grammar,
        // required is eager (constrains from root); auto stays lazy.
        grammar_lazy: !require_tools,
        grammar_triggers: if require_tools {
            Vec::new()
        } else {
            vec![ToolGrammarTrigger {
                trigger_type: ToolGrammarTriggerType::Word,
                value: "<tool_call>\n".to_string(),
                token: None,
            }]
        },
    })
}

// The parameters mirror the subset of tool-template options this renderer
// consumes; the caller already holds them as separate fields, so threading them
// individually avoids an extra adapter struct.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_qwen_tool_chat_template(
    template: &str,
    messages_json: &str,
    tools_json: Option<&str>,
    tool_choice: Option<&str>,
    chat_template_kwargs: Option<&str>,
    add_generation_prompt: bool,
    enable_thinking: bool,
    parallel_tool_calls: bool,
) -> Result<ToolChatTemplateResult> {
    if tools_json.is_some() && !is_qwen_tagged_tool_template(template) {
        bail!("unsupported chat template for Qwen tagged tool calling");
    }

    let messages: Value =
        serde_json::from_str(messages_json).map_err(|e| anyhow!("invalid messages_json: {e}"))?;
    let tools = match tools_json {
        Some(tools) if tool_choice != Some("none") => Some(
            serde_json::from_str::<Value>(tools).map_err(|e| anyhow!("invalid tools_json: {e}"))?,
        ),
        _ => None,
    };
    let kwargs = chat_template_kwargs
        .map(serde_json::from_str::<Value>)
        .transpose()
        .map_err(|e| anyhow!("invalid chat_template_kwargs: {e}"))?;

    // `prompt` and the generation-prompt diff share the two renderings below:
    // with/without the generation prefix. Reusing `prompt_with_generation` for
    // `prompt` (when the caller wants the generation prompt) avoids a third,
    // identical render and keeps `prompt` consistent with the diff it is
    // measured against.
    let prompt_without_generation = render_qwen_template_once(
        template,
        &messages,
        tools.as_ref(),
        kwargs.as_ref(),
        enable_thinking,
        false,
    )?;
    let prompt_with_generation = render_qwen_template_once(
        template,
        &messages,
        tools.as_ref(),
        kwargs.as_ref(),
        enable_thinking,
        true,
    )?;
    // strip_prefix returns None when `prompt_without_generation` is not a
    // prefix of `prompt_with_generation`. For the Qwen tagged templates this
    // path supports, the generation prompt is purely appended (no common
    // suffix), so a prefix mismatch signals an unexpected template shape we
    // must not silently paper over. Compute it before moving either prompt
    // out so neither needs to be cloned.
    let generation_prompt = prompt_with_generation
        .strip_prefix(&prompt_without_generation)
        .context("generation prompt is not an append-only suffix of the base prompt")?
        .to_string();
    let prompt = if add_generation_prompt {
        prompt_with_generation
    } else {
        prompt_without_generation
    };
    // resolve_tool_choice normalises function-specific choices to "required",
    // so a bare "required" here is the only mandatory-tool signal we need.
    let require_tools = tool_choice == Some("required");
    let grammar_spec = tools_json
        .filter(|_| tool_choice != Some("none"))
        .map(|tools| build_qwen_tool_call_grammar_spec(tools, require_tools, parallel_tool_calls))
        .transpose()?;
    let (grammar, grammar_lazy, grammar_triggers) = grammar_spec
        .map(|spec| (Some(spec.grammar), spec.grammar_lazy, spec.grammar_triggers))
        .unwrap_or_default();

    Ok(ToolChatTemplateResult {
        prompt,
        grammar,
        grammar_lazy,
        grammar_triggers,
        preserved_tokens: QWEN_TAGGED_PRESERVED_TOKENS
            .iter()
            .map(|token| (*token).to_string())
            .collect(),
        additional_stops: Vec::new(),
        chat_format: QWEN_TAGGED_CHAT_FORMAT,
        parser: None,
        generation_prompt,
        parse_tool_calls: tools_json.is_some() && tool_choice != Some("none"),
    })
}

/// Whether `template` is a Qwen3.5-style tagged tool template, the only family
/// the Rust renderer currently builds a grammar for. Callers route non-matching
/// templates (Llama/Mistral/etc.) back to the legacy OAI renderer instead of
/// failing, so existing tool-capable models keep working.
///
/// The tag set alone is not enough: other templates reuse the same
/// `<tool_call>/<function=>/<parameter=>/<tools>` markers. StepFun3.5, for one,
/// also calls `tojson(ensure_ascii=False)`, which this renderer's `tojson` filter
/// cannot accept (it would fail in `assert_all_used`). So we additionally require
/// the Qwen3.5-specific `enable_thinking` switch (which this renderer drives) and
/// exclude templates that pass `ensure_ascii` to `tojson`. Anything failing these
/// checks falls back to the legacy renderer rather than erroring mid-render.
pub(crate) fn is_qwen_tagged_tool_template(template: &str) -> bool {
    template.contains("<tool_call>")
        && template.contains("<function=")
        && template.contains("<parameter=")
        && template.contains("<tools>")
        && template.contains("enable_thinking")
        && !template.contains("tojson(ensure_ascii")
}

fn render_qwen_template_once(
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
        |env| {
            env.add_filter("tojson", qwen_tojson_filter);
            env.add_function(
                "raise_exception",
                |msg: String| -> std::result::Result<String, minijinja::Error> {
                    Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        msg,
                    ))
                },
            );
        },
    )
}

fn qwen_tojson_filter(
    value: &minijinja::value::Value,
    indent: Option<minijinja::value::Value>,
    args: minijinja::value::Kwargs,
) -> std::result::Result<minijinja::value::Value, minijinja::Error> {
    let indent = indent
        .or_else(|| {
            args.get::<Option<minijinja::value::Value>>("indent")
                .ok()
                .flatten()
        })
        .and_then(|value| usize::try_from(value).ok());
    args.assert_all_used()?;
    let value = serde_json::to_value(value).map_err(|err| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "cannot serialize to JSON",
        )
        .with_source(err)
    })?;
    let json = if let Some(indent) = indent {
        let mut out = Vec::new();
        let indentation = " ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
        let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
        serde::Serialize::serialize(&value, &mut serializer)
            .map_err(|err| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "cannot serialize to JSON",
                )
                .with_source(err)
            })
            .map(|()| String::from_utf8(out).expect("serde_json only emits UTF-8"))?
    } else {
        qwen_compact_json(&value)
    };
    Ok(minijinja::value::Value::from_safe_string(json))
}

fn qwen_compact_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).expect("serializing scalar JSON cannot fail")
        }
        Value::Array(values) => {
            let items = values
                .iter()
                .map(qwen_compact_json)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{items}]")
        }
        Value::Object(map) => {
            let keys = qwen_ordered_json_keys(map);
            let items = keys
                .into_iter()
                .map(|key| {
                    let key_json =
                        serde_json::to_string(&key).expect("serializing key cannot fail");
                    let value_json = qwen_compact_json(&map[&key]);
                    format!("{key_json}: {value_json}")
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{items}}}")
        }
    }
}

fn qwen_ordered_json_keys(map: &serde_json::Map<String, Value>) -> Vec<String> {
    let preferred = [
        "type",
        "function",
        "name",
        "description",
        "parameters",
        "properties",
        "required",
        "items",
        "enum",
        "const",
        "format",
        "minimum",
        "maximum",
    ];
    let mut keys = Vec::with_capacity(map.len());
    for key in preferred {
        if map.contains_key(key) {
            keys.push(key.to_string());
        }
    }
    keys.extend(
        map.keys()
            .filter(|key| !preferred.contains(&key.as_str()))
            .cloned(),
    );
    keys
}

fn qwen_tool_grammar_parameters(
    tool: &Value,
    tool_rule_name: &str,
) -> Result<Vec<QwenToolGrammarParameter>> {
    // Collect `required` in declared order, dropping empties and intra-array
    // duplicates in one pass so the ordering loop below never emits a name twice.
    let mut required_names: Vec<String> = Vec::new();
    let mut required_set = std::collections::BTreeSet::new();
    for name in tool
        .pointer("/function/parameters/required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|name| !name.is_empty())
    {
        if required_set.insert(name.to_string()) {
            required_names.push(name.to_string());
        }
    }
    // A missing or empty `properties` is a valid parameterless tool (e.g.
    // `{"function":{"name":"ping"}}`): the reference Qwen3.5 builder emits the
    // function rule with no argument parsers in that case. Treat absence as an
    // empty property map rather than an error.
    let empty_props = serde_json::Map::new();
    let properties = tool
        .pointer("/function/parameters/properties")
        .and_then(Value::as_object)
        .unwrap_or(&empty_props);

    // Required parameters come first (preserving their declared order), then the
    // remaining properties. `required_set` keeps the second pass from re-emitting
    // a property already placed by the first.
    let mut ordered_names = Vec::with_capacity(properties.len());
    for name in &required_names {
        if !properties.contains_key(name) {
            bail!("required parameter `{name}` is missing from properties");
        }
        ordered_names.push(name.clone());
    }
    for name in properties.keys() {
        if !name.is_empty() && !required_set.contains(name.as_str()) {
            ordered_names.push(name.clone());
        }
    }

    let mut rule_base_counts = std::collections::BTreeMap::new();
    ordered_names
        .into_iter()
        .map(|name| {
            let schema = properties
                .get(&name)
                .ok_or_else(|| anyhow!("parameter `{name}` is missing schema"))?;
            let required = required_set.contains(name.as_str());
            let parameter_rule_base =
                format!("{tool_rule_name}-arg-{}", grammar_rule_suffix(&name)?);
            let parameter_rule_name = dedup_rule_name(&mut rule_base_counts, parameter_rule_base);
            let value_rule_name = format!("{parameter_rule_name}-value");
            let (value_kind, value_expression, value_rule_definition) =
                qwen_parameter_value_rule(schema, &value_rule_name)?;
            Ok(QwenToolGrammarParameter {
                spec: QwenToolParameterSpec {
                    name,
                    value_kind,
                    required,
                },
                parameter_rule_name,
                value_expression,
                value_rule_definition,
            })
        })
        .collect()
}

fn qwen_parameter_value_rule(
    schema: &Value,
    value_rule_name: &str,
) -> Result<(QwenToolParameterValueKind, String, String)> {
    // Mirror the reference Qwen3.5 builder: anything `resolves_to_string`
    // (plain string, string const/enum, pattern/min/maxLength, string format)
    // is emitted as raw `until(</parameter>)` text with no value-range
    // constraint in the grammar. Non-string schemas fall through to
    // `json_schema_to_grammar`.
    if qwen_schema_resolves_to_string(schema) {
        return Ok((
            QwenToolParameterValueKind::RawString,
            "\"\" until-suffix".to_string(),
            String::new(),
        ));
    }

    let schema_json = serde_json::to_string(schema)?;
    let grammar = json_schema_to_grammar(&schema_json)
        .map_err(|e| anyhow!("json_schema_to_grammar failed for parameter schema: {e}"))?;
    Ok((
        QwenToolParameterValueKind::JsonLiteral,
        value_rule_name.to_string(),
        prefix_json_schema_grammar(&grammar, value_rule_name),
    ))
}

// Port of the reference impl's `common_schema_info::resolves_to_string`
// (json-schema-to-grammar.cpp). The Qwen3.5 tagged template decides "raw text
// vs JSON value" with exactly this predicate, so the grammar must agree or it
// would demand a quoted JSON string where the model emits bare text (and vice
// versa). `$ref`/`oneOf`/`anyOf`/`allOf` are not resolved here because the
// plugin's tool schemas are flat; if those appear they conservatively fall
// through to the JSON-literal path (a quoted string would simply be rejected,
// surfacing the gap rather than silently mis-parsing).
fn qwen_schema_resolves_to_string(schema: &Value) -> bool {
    if !schema.is_object() {
        return false;
    }
    match schema.get("type") {
        Some(Value::String(t)) if t == "string" => return true,
        Some(Value::Array(types)) if types.iter().any(|t| t.as_str() == Some("string")) => {
            return true;
        }
        _ => {}
    }
    if schema.get("const").is_some_and(Value::is_string) {
        return true;
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array)
        && values.iter().any(Value::is_string)
    {
        return true;
    }
    if schema.get("pattern").is_some()
        || schema.get("minLength").is_some()
        || schema.get("maxLength").is_some()
    {
        return true;
    }
    if let Some(Value::String(fmt)) = schema.get("format") {
        return matches!(
            fmt.as_str(),
            "date" | "time" | "date-time" | "uri" | "email" | "hostname" | "ipv4" | "ipv6"
        ) || fmt.starts_with("uuid");
    }
    false
}

// `json_schema_to_grammar` returns a self-contained GBNF whose rules all live in
// a flat namespace (`root`, `integral-part`, ...). Splicing several of these into
// one combined grammar (one per JSON-literal parameter) would collide on those
// shared names, so every rule is renamed into a per-parameter namespace. `root`
// is renamed to `value_rule_name` itself (not `value_rule_name-root`) because the
// parent parameter rule references the value by exactly `value_rule_name` — that
// rename is the splice point, and is load-bearing.
fn prefix_json_schema_grammar(grammar: &str, value_rule_name: &str) -> String {
    // First pass: map each defined rule name to its namespaced form. A reference
    // may precede its definition, so the full map must exist before rewriting.
    let name_map: std::collections::BTreeMap<String, String> = grammar
        .lines()
        .filter_map(|line| line.split_once("::=").map(|(name, _)| name.trim()))
        .filter(|name| !name.is_empty())
        .map(|name| {
            let replacement = if name == "root" {
                value_rule_name.to_string()
            } else {
                format!("{value_rule_name}-{name}")
            };
            (name.to_string(), replacement)
        })
        .collect();

    let mut out = String::with_capacity(grammar.len());
    for line in grammar.lines() {
        out.push_str(&prefix_gbnf_rule_refs(line, &name_map));
        out.push('\n');
    }
    out
}

// Rewrite bare rule-name references on a single GBNF line according to
// `name_map`. A naive string replace cannot be used: an identifier that matches
// a rule name may also appear *inside* a `"..."` string literal or a `[...]`
// char class, where it is data, not a rule reference, and must be left
// untouched. Hence the explicit string/char-class state machine with escape
// handling — do not "simplify" it into a regex/replace, which would corrupt
// such literals.
fn prefix_gbnf_rule_refs(
    line: &str,
    name_map: &std::collections::BTreeMap<String, String>,
) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.char_indices().peekable();
    let mut in_string = false;
    let mut in_char_class = false;
    while let Some((_, ch)) = chars.next() {
        if in_string {
            out.push(ch);
            if ch == '\\' {
                if let Some((_, next)) = chars.next() {
                    out.push(next);
                }
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if in_char_class {
            out.push(ch);
            if ch == '\\' {
                if let Some((_, next)) = chars.next() {
                    out.push(next);
                }
            } else if ch == ']' {
                in_char_class = false;
            }
            continue;
        }
        match ch {
            '"' => {
                in_string = true;
                out.push(ch);
            }
            '[' => {
                in_char_class = true;
                out.push(ch);
            }
            c if is_gbnf_rule_name_char(c) => {
                let mut ident = String::from(c);
                while let Some((_, next)) = chars.peek().copied() {
                    if is_gbnf_rule_name_char(next) {
                        ident.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Some(replacement) = name_map.get(&ident) {
                    out.push_str(replacement);
                } else {
                    out.push_str(&ident);
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

fn is_gbnf_rule_name_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '-'
}

// Parse-back counterpart of the grammar builders above: reconstruct the
// arguments JSON object from the raw tag bodies the model emitted, using the
// same per-parameter `value_kind` contract the grammar enforced. Co-located
// with the generator so both halves of the contract stay in sync.
//
// Output keys follow declared parameter order. Absence handling mirrors the
// grammar: a required parameter is always emitted by the grammar, so its
// absence is a hard error; an optional parameter is `*`-quantified (0-or-more),
// so an absent one is silently skipped rather than errored. Because the grammar
// permits an optional parameter to appear more than once, a duplicate is
// resolved last-wins (the final emitted body for that name takes effect).
fn qwen_tagged_arguments_to_json<'a>(
    parameters: &[QwenToolParameterSpec],
    tagged_values: &[(&'a str, &'a str)],
) -> Result<String> {
    // Linear scan rather than a lookup map: a tool call carries only a handful
    // of parameters, so an index would cost more (allocation) than it saves.
    let mut out = serde_json::Map::new();
    for parameter in parameters {
        // Last-wins on duplicates: take the final matching body so a repeated
        // optional parameter resolves to its last occurrence.
        let raw = tagged_values
            .iter()
            .rev()
            .find(|(name, _)| *name == parameter.name)
            .map(|(_, value)| *value);
        let raw = match raw {
            Some(raw) => raw,
            None if parameter.required => {
                bail!("missing required tagged parameter `{}`", parameter.name)
            }
            // Optional parameter the model chose to omit: not in the output.
            None => continue,
        };
        let value = match parameter.value_kind {
            QwenToolParameterValueKind::RawString => Value::String(raw.to_string()),
            QwenToolParameterValueKind::JsonLiteral => serde_json::from_str(raw)
                .with_context(|| format!("invalid JSON literal for `{}`", parameter.name))?,
        };
        out.insert(parameter.name.clone(), value);
    }
    serde_json::to_string(&Value::Object(out)).map_err(Into::into)
}

// Maps a tool's `function.name` to its ordered parameter specs. The parser
// needs the per-parameter `value_kind` (and `required`, via the specs) to turn
// a raw tag body back into the correct JSON value, exactly as the grammar
// emitted it. Built once per request from the same `tools_json` the grammar
// builder consumes so the two halves cannot disagree.
pub(crate) type QwenToolParamIndex = std::collections::BTreeMap<String, Vec<QwenToolParameterSpec>>;

// Build the name→specs index from the OAI tools JSON, reusing the grammar
// builder's parameter derivation so parse-back and grammar share one source of
// truth for ordering and value kinds.
pub(crate) fn build_qwen_tool_param_index(tools_json: &str) -> Result<QwenToolParamIndex> {
    let tools: Value =
        serde_json::from_str(tools_json).map_err(|e| anyhow!("invalid tools_json: {e}"))?;
    let tools = tools
        .as_array()
        .ok_or_else(|| anyhow!("tools_json must be an array"))?;
    let tool_rule_names = qwen_tool_rule_names(tools)?;
    let mut index = QwenToolParamIndex::new();
    for (tool_index, tool) in tools.iter().enumerate() {
        let name = qwen_tool_function_name(tool)?.to_string();
        let specs = qwen_tool_grammar_parameters(tool, &tool_rule_names[tool_index])?
            .into_iter()
            .map(|p| p.spec)
            .collect();
        index.insert(name, specs);
    }
    Ok(index)
}

// Find `marker` at or after `from`, returning the byte offset just past it
// (e.g. past `"<function="`). Helper for the tag walker.
fn find_after(haystack: &str, marker: &str, from: usize) -> Option<usize> {
    haystack[from..]
        .find(marker)
        .map(|rel| from + rel + marker.len())
}

fn find_marker_outside_parameter(
    haystack: &str,
    marker: &str,
    from: usize,
    limit: usize,
) -> Option<usize> {
    let mut cursor = from.min(limit);
    while cursor < limit {
        let marker_pos = haystack[cursor..limit].find(marker).map(|rel| cursor + rel);
        let parameter_pos = haystack[cursor..limit]
            .find(PARAMETER_OPEN)
            .map(|rel| cursor + rel);
        // Honour the marker only when no `<parameter>` block sits strictly
        // before it; otherwise jump past that block so its body text can't
        // false-match.
        match parameter_pos {
            Some(parameter_pos) if marker_pos.is_none_or(|m| parameter_pos < m) => {
                cursor = skip_parameter_value(haystack, parameter_pos, limit)?;
            }
            _ => return marker_pos,
        }
    }
    None
}

// Given the start of a `<parameter=...>` tag, return the offset just past its
// `</parameter>` close (or `None` if the value is still streaming). Used to step
// over a parameter body so its literal text can't false-match an outer marker.
fn skip_parameter_value(haystack: &str, parameter_pos: usize, limit: usize) -> Option<usize> {
    let value_start = haystack[parameter_pos..limit]
        .find('>')
        .map_or(parameter_pos + PARAMETER_OPEN.len(), |rel| {
            parameter_pos + rel + 1
        });
    let close_rel = haystack[value_start..limit].find(PARAMETER_CLOSE)?;
    Some(value_start + close_rel + PARAMETER_CLOSE.len())
}

fn find_structured_tool_call_start(raw: &str) -> Option<usize> {
    let mut cursor = 0;
    while let Some(tc_after_open) = find_after(raw, TOOL_CALL_OPEN, cursor) {
        let tc_start = tc_after_open - TOOL_CALL_OPEN.len();
        let tc_end = find_marker_outside_parameter(raw, TOOL_CALL_CLOSE, tc_after_open, raw.len())
            .unwrap_or(raw.len());
        if raw[tc_after_open..tc_end]
            .trim_start()
            .starts_with(FUNCTION_OPEN)
        {
            return Some(tc_start);
        }
        cursor = tc_after_open;
    }
    None
}

// Parse one `<function=NAME>...</function>` block. `name_start` points just past
// the `<function=` marker. Returns the call plus the offset past `</function>`.
// With `allow_partial`, an unterminated block yields what was parsed so far (or
// `None` when the name is not yet known) so streaming can re-parse next chunk.
fn parse_qwen_function_block(
    raw: &str,
    name_start: usize,
    index: &QwenToolParamIndex,
    allow_partial: bool,
) -> Result<(Option<ParsedToolCall>, usize)> {
    // `<function=NAME>` — name runs to the closing `>`.
    let Some(name_close_rel) = raw[name_start..].find('>') else {
        if allow_partial {
            return Ok((None, raw.len()));
        }
        bail!("unterminated <function=...> tag");
    };
    let name = raw[name_start..name_start + name_close_rel]
        .trim()
        .to_string();
    let cursor = name_start + name_close_rel + 1; // past '>'

    let func_close = find_marker_outside_parameter(raw, FUNCTION_CLOSE, cursor, raw.len());
    let body_end = func_close.unwrap_or(raw.len());

    // Collect `(key, value)` pairs in emission order; duplicates are kept so
    // `qwen_tagged_arguments_to_json` can apply last-wins.
    let mut tagged: Vec<(&str, &str)> = Vec::new();
    let mut pcur = cursor;
    while let Some(key_start) = find_after(&raw[..body_end], "<parameter=", pcur) {
        let Some(key_close_rel) = raw[key_start..body_end].find('>') else {
            break; // partial: parameter name still streaming
        };
        let key = raw[key_start..key_start + key_close_rel].trim();
        let value_start = key_start + key_close_rel + 1;
        // Value runs until `</parameter>`. The grammar frames the body with a
        // leading newline after `>` and a trailing newline before
        // `</parameter>`; strip exactly those (matching the until-suffix body).
        let Some(val_close_rel) = raw[value_start..body_end].find(PARAMETER_CLOSE) else {
            break; // partial: value still streaming
        };
        let raw_value = &raw[value_start..value_start + val_close_rel];
        let value = raw_value.strip_prefix('\n').unwrap_or(raw_value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        tagged.push((key, value));
        pcur = value_start + val_close_rel + PARAMETER_CLOSE.len();
    }
    let next = func_close.map_or(raw.len(), |c| c + FUNCTION_CLOSE.len());

    if name.is_empty() {
        if allow_partial {
            return Ok((None, next));
        }
        bail!("<function=...> tag has an empty name");
    }

    // Reconstruct arguments using the tool's specs when known. Unknown tools
    // (not in `index`) fall back to treating every emitted value as a raw
    // string so a response is still surfaced rather than dropped.
    let build = || -> Result<String> {
        match index.get(&name) {
            Some(specs) => qwen_tagged_arguments_to_json(specs, &tagged),
            None => {
                let mut out = serde_json::Map::new();
                for (k, v) in &tagged {
                    out.insert((*k).to_string(), Value::String((*v).to_string()));
                }
                Ok(serde_json::to_string(&Value::Object(out))?)
            }
        }
    };
    // While partial, a required parameter may simply not have streamed yet, so
    // a "missing required" error is expected — fall back to an empty object
    // rather than failing; the next chunk re-parses with more bytes.
    let arguments = match build() {
        Ok(args) => args,
        Err(_) if allow_partial => "{}".to_string(),
        Err(e) => return Err(e),
    };
    Ok((
        Some(ParsedToolCall {
            id: None,
            name,
            arguments,
        }),
        next,
    ))
}

// Parse a Qwen3.5 tagged assistant turn into a `ParsedChatMsg`. Splits the
// `<think>...</think>` reasoning block, then walks each `<tool_call>` envelope
// for `<function=...>` blocks. A turn with no tool calls becomes a plain
// `content` message. With `allow_partial`, unterminated tags are tolerated so
// the same code drives streaming (re-parse the growing buffer each chunk).
//
// Ported from the reference `chat-peg-parser.cpp` mapper: whitespace-only
// reasoning is dropped; a function block whose name is not yet known is omitted
// while partial.
pub(crate) fn parse_qwen_tagged_response(
    raw: &str,
    index: &QwenToolParamIndex,
    allow_partial: bool,
) -> Result<ParsedChatMsg> {
    let (without_think, reasoning) = crate::model::LlamaModelWrapper::extract_reasoning(raw);
    // Drop whitespace-only reasoning (the empty `<think>\n\n</think>` Qwen emits
    // when thinking is disabled) so it does not surface as reasoning_content.
    let reasoning_content = reasoning.filter(|r| !r.trim().is_empty());

    let mut tool_calls = Vec::new();
    let mut cursor = 0;
    while let Some(tc_open) = find_after(&without_think, TOOL_CALL_OPEN, cursor) {
        // Each `<tool_call>` may hold one or more `<function=...>` blocks
        // (parallel calls share one envelope).
        let tc_end = find_marker_outside_parameter(
            &without_think,
            TOOL_CALL_CLOSE,
            tc_open,
            without_think.len(),
        );
        let block_end = tc_end.unwrap_or(without_think.len());
        if !without_think[tc_open..block_end]
            .trim_start()
            .starts_with(FUNCTION_OPEN)
        {
            cursor = tc_end.map_or(block_end, |e| e + TOOL_CALL_CLOSE.len());
            continue;
        }
        let mut fcur = tc_open;
        while let Some(name_start) = find_after(&without_think[..block_end], FUNCTION_OPEN, fcur) {
            let (call, next) = parse_qwen_function_block(
                &without_think[..block_end],
                name_start,
                index,
                allow_partial,
            )?;
            if let Some(call) = call {
                tool_calls.push(call);
            }
            if next <= fcur {
                break; // no progress (partial); stop to avoid a spin
            }
            fcur = next;
        }
        cursor = tc_end.map_or(without_think.len(), |e| e + TOOL_CALL_CLOSE.len());
    }

    // With tool calls present the fork API kept content think-only/empty, and
    // downstream only reads content on the no-tool branch, so keeping the
    // think-stripped body is harmless and preserves any leading text.
    Ok(ParsedChatMsg {
        content: without_think,
        reasoning_content,
        tool_calls,
    })
}

/// Incremental Qwen3.5 tagged stream parser. The reference parser re-parses the
/// accumulated assistant turn and emits OpenAI-compatible diffs; this mirrors
/// that shape so chunk boundaries can split tags and JSON arguments freely.
#[derive(Debug)]
pub(crate) struct QwenTaggedStreamState {
    raw: String,
    index: QwenToolParamIndex,
    diff: ParsedStreamDeltaState,
}

impl QwenTaggedStreamState {
    pub(in crate::oai_chat) fn new(tools_json: &str, generation_prompt: &str) -> Result<Self> {
        Ok(Self {
            raw: String::new(),
            index: build_qwen_tool_param_index(tools_json)?,
            diff: ParsedStreamDeltaState::new(generation_prompt),
        })
    }

    pub(in crate::oai_chat) fn update(
        &mut self,
        chunk: &str,
        is_streaming: bool,
    ) -> Result<Vec<String>> {
        self.raw.push_str(chunk);
        let now = parse_qwen_stream_msg(&self.raw, &self.index, is_streaming)?;
        let Some(deltas) = parsed_stream_deltas(&mut self.diff, &now)? else {
            return Ok(Vec::new());
        };
        Ok(deltas)
    }
}

fn parse_qwen_stream_msg(
    raw: &str,
    index: &QwenToolParamIndex,
    _allow_partial: bool,
) -> Result<ParsedChatMsg> {
    let (mut without_think, reasoning) = crate::model::LlamaModelWrapper::extract_reasoning(raw);
    let reasoning_content = reasoning.filter(|r| !r.trim().is_empty());
    if let Some(tool_call_start) = find_structured_tool_call_start(&without_think) {
        // Stream-only branch: re-derive partial argument fragments so chunk
        // boundaries can split JSON values; the non-streaming walker only
        // emits complete arguments.
        without_think.truncate(tool_call_start);
        return Ok(ParsedChatMsg {
            content: without_think,
            reasoning_content,
            tool_calls: parse_qwen_stream_tool_calls(raw, index),
        });
    }
    // No structured tool call yet: the body is plain content. Hide a pending
    // `<tool_call>` opener (in place, no realloc) so it never surfaces as
    // content until the following bytes prove structure vs literal text.
    without_think.truncate(trim_pending_tool_call_open_prefix(&without_think).len());
    Ok(ParsedChatMsg {
        content: without_think,
        reasoning_content,
        tool_calls: Vec::new(),
    })
}

fn trim_pending_tool_call_open_prefix(content: &str) -> &str {
    if let Some(open_start) = content.rfind(TOOL_CALL_OPEN) {
        let after_open = open_start + TOOL_CALL_OPEN.len();
        let pending_body = content[after_open..].trim_start();
        if pending_body.is_empty() || FUNCTION_OPEN.starts_with(pending_body) {
            return &content[..open_start];
        }
    }
    for prefix_len in (1..=TOOL_CALL_OPEN.len()).rev() {
        let prefix = &TOOL_CALL_OPEN[..prefix_len];
        if content.ends_with(prefix) {
            return &content[..content.len() - prefix_len];
        }
    }
    content
}

fn parse_qwen_stream_tool_calls(raw: &str, index: &QwenToolParamIndex) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(tc_open) = find_after(raw, TOOL_CALL_OPEN, cursor) {
        let tc_end = find_marker_outside_parameter(raw, TOOL_CALL_CLOSE, tc_open, raw.len());
        let block_end = tc_end.unwrap_or(raw.len());
        let mut fcur = tc_open;
        while let Some(name_start) = find_after(&raw[..block_end], FUNCTION_OPEN, fcur) {
            let Some(name_close_rel) = raw[name_start..block_end].find('>') else {
                break;
            };
            let name_end = name_start + name_close_rel;
            let name = raw[name_start..name_end].trim();
            let body_start = name_end + 1;
            if body_start >= block_end {
                break;
            }
            if !name.is_empty() {
                let function_close =
                    find_marker_outside_parameter(raw, FUNCTION_CLOSE, body_start, block_end);
                let function_end = function_close.unwrap_or(block_end);
                // A trailing newline after `</function>` is the grammar's framing
                // for "block fully closed", so the JSON object can be sealed.
                let close_object = function_close.is_some_and(|end| {
                    raw[end + FUNCTION_CLOSE.len()..block_end].starts_with('\n')
                });
                let specs = index.get(name).map(Vec::as_slice).unwrap_or(&[]);
                let arguments = qwen_partial_arguments_json(
                    &raw[body_start..function_end],
                    specs,
                    close_object,
                );
                calls.push(ParsedToolCall {
                    id: None,
                    name: name.to_string(),
                    arguments,
                });
                let next = function_close.map_or(block_end, |end| end + FUNCTION_CLOSE.len());
                if next <= fcur {
                    break;
                }
                fcur = next;
            } else {
                fcur = body_start;
            }
        }
        cursor = tc_end.map_or(raw.len(), |end| end + TOOL_CALL_CLOSE.len());
    }
    calls
}

fn qwen_partial_arguments_json(
    function_body: &str,
    parameters: &[QwenToolParameterSpec],
    close_object: bool,
) -> String {
    let mut out = String::from("{");
    let mut first = true;
    let mut cursor = 0;
    while let Some(key_start) = find_after(function_body, PARAMETER_OPEN, cursor) {
        let Some(key_close_rel) = function_body[key_start..].find('>') else {
            break;
        };
        let key_end = key_start + key_close_rel;
        let key = function_body[key_start..key_end].trim();
        if key.is_empty() {
            break;
        }
        let value_start = key_end + 1;
        if value_start >= function_body.len() {
            break;
        }
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string()));
        out.push(':');
        let value_kind = qwen_stream_value_kind(parameters, key);
        let value_end = function_body[value_start..]
            .find(PARAMETER_CLOSE)
            .map(|rel| value_start + rel);
        // A trailing newline after `</parameter>` marks the value fully closed,
        // so a string value can receive its closing quote.
        let value_closed = value_end
            .is_some_and(|end| function_body[end + PARAMETER_CLOSE.len()..].starts_with('\n'));
        let raw_value = match value_end {
            Some(end) => &function_body[value_start..end],
            None => &function_body[value_start..],
        };
        let value = raw_value.strip_prefix('\n').unwrap_or(raw_value);
        let value = if value_closed {
            value.strip_suffix('\n').unwrap_or(value)
        } else {
            trim_pending_parameter_close_prefix(value)
        };
        match value_kind {
            QwenToolParameterValueKind::RawString => {
                out.push('"');
                push_json_string_fragment(&mut out, value);
                if value_closed {
                    out.push('"');
                }
            }
            QwenToolParameterValueKind::JsonLiteral => out.push_str(value),
        }
        cursor = value_end.map_or(function_body.len(), |end| end + PARAMETER_CLOSE.len());
    }
    if close_object {
        out.push('}');
    }
    out
}

fn trim_pending_parameter_close_prefix(value: &str) -> &str {
    // The grammar frames a value with a trailing newline before `</parameter>`;
    // hide any prefix of that closing sequence that has streamed so far. Unlike
    // the tool-call opener, a complete `\n</parameter>` should also be dropped,
    // hence the inclusive upper bound.
    // Kept as a literal (const &str can't `concat!` a const), but tied to
    // PARAMETER_CLOSE so a tag-shape change is caught in debug builds.
    const CLOSE_PREFIX: &str = "\n</parameter>";
    debug_assert!(CLOSE_PREFIX.ends_with(PARAMETER_CLOSE));
    for prefix_len in (1..=CLOSE_PREFIX.len()).rev() {
        let prefix = &CLOSE_PREFIX[..prefix_len];
        if value.ends_with(prefix) {
            return &value[..value.len() - prefix_len];
        }
    }
    value
}

fn qwen_stream_value_kind(
    parameters: &[QwenToolParameterSpec],
    key: &str,
) -> QwenToolParameterValueKind {
    parameters
        .iter()
        .find(|parameter| parameter.name == key)
        .map(|parameter| parameter.value_kind)
        .unwrap_or(QwenToolParameterValueKind::RawString)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oai_chat::common::normalize_stream_delta_ids;
    use crate::oai_chat::{decode_oai_deltas, extract_enable_thinking};

    // build_qwen_tool_call_grammar_spec: emits the Qwen3.5-style
    // <tool_call>/<function>/<parameter> grammar shape captured in the golden
    // fixtures, including an OR over multiple tools.
    #[test]
    fn test_build_qwen_tool_call_grammar_spec_multiple_tools() {
        let tools = r#"[
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
            {"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        assert!(spec.grammar_lazy);
        assert!(spec.grammar.contains("root ::= tool-call"));
        assert!(spec.grammar.contains("(tool-get-weather | tool-get-time)"));
        assert!(spec.grammar.contains("\"<function=\" \"get_weather\""));
        assert!(spec.grammar.contains("\"<parameter=\" \"city\""));
        assert_eq!(spec.grammar_triggers.len(), 1);
        assert_eq!(
            spec.grammar_triggers[0].trigger_type,
            ToolGrammarTriggerType::Word
        );
        assert_eq!(spec.grammar_triggers[0].value, "<tool_call>\n");
    }

    // tool_choice=required must constrain from `root` (eager, no lazy trigger),
    // otherwise the model could answer with plain text and never satisfy the
    // "required" contract.
    #[test]
    fn test_build_qwen_tool_call_grammar_spec_required_is_eager() {
        let tools = r#"[
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, true, false).expect("qwen tool grammar");

        assert!(!spec.grammar_lazy, "required must be eager");
        assert!(
            spec.grammar_triggers.is_empty(),
            "eager grammar carries no lazy trigger"
        );
        assert!(spec.grammar.contains("root ::= tool-call"));
    }

    // parallel_tool_calls must let one <tool_call> envelope hold more than one
    // function invocation; without it the grammar caps the call count at one.
    #[test]
    fn test_build_qwen_tool_call_grammar_spec_parallel_allows_repeated_calls() {
        let tools = r#"[
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
            {"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        ]"#;

        let parallel =
            build_qwen_tool_call_grammar_spec(tools, false, true).expect("qwen tool grammar");
        let single =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        let alternatives = "(tool-get-weather | tool-get-time)";
        // Parallel repeats the alternation; single does not.
        assert!(
            parallel
                .grammar
                .contains(&format!("{alternatives} (space {alternatives})*"))
        );
        assert!(!single.grammar.contains(&format!("(space {alternatives})*")));
    }

    // build_qwen_tool_call_grammar_spec: malformed tool definitions fail
    // before reaching the sampler, instead of producing an unconstrained grammar.
    #[test]
    fn test_build_qwen_tool_call_grammar_spec_rejects_missing_name() {
        let tools = r#"[{"type":"function","function":{"parameters":{"type":"object"}}}]"#;
        let err = build_qwen_tool_call_grammar_spec(tools, false, false).unwrap_err();
        assert!(err.to_string().contains("function.name"));
    }

    #[test]
    fn test_build_qwen_tool_call_grammar_spec_keeps_optional_parameters_optional() {
        let tools = r#"[
            {"type":"function","function":{"name":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["city"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        assert!(
            spec.grammar
                .contains("tool-forecast-optional-parameters ::= ("),
            "optional parameters should be grouped behind a 0-or-more rule:\n{}",
            spec.grammar
        );
        assert!(
            spec.grammar.contains("tool-forecast-optional-parameters*"),
            "function rule should not require every optional property:\n{}",
            spec.grammar
        );
    }

    #[test]
    fn test_build_qwen_tool_call_grammar_spec_uses_schema_specific_value_rules() {
        let tools = r#"[
            {"type":"function","function":{"name":"set_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer"},"rain":{"type":"boolean"},"metadata":{"type":"object","properties":{"source":{"type":"string"}}},"tags":{"type":"array","items":{"type":"string"}},"fixed":{"type":"string","const":"celsius"}},"required":["city","unit","days","rain","metadata","tags","fixed"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        // String-resolving schemas (plain string, string enum, string const) all
        // map to raw `until-suffix` text with no value-range constraint, matching
        // the reference Qwen3.5 builder's `resolves_to_string` path. They share
        // the inline `"" until-suffix` value expression and emit no `-value` rule.
        for raw_string_arg in ["city", "unit", "fixed"] {
            assert!(
                spec.grammar.contains(&format!(
                    "tool-set-weather-arg-{raw_string_arg} ::= (\"<parameter=\" \"{raw_string_arg}\" \">\\n\") \"\" until-suffix"
                )),
                "string-resolving arg `{raw_string_arg}` should use raw until-suffix:\n{}",
                spec.grammar
            );
            assert!(
                !spec
                    .grammar
                    .contains(&format!("tool-set-weather-arg-{raw_string_arg}-value")),
                "string-resolving arg `{raw_string_arg}` should not emit a JSON value rule:\n{}",
                spec.grammar
            );
        }
        // The reference impl does NOT constrain enum/const ranges in the grammar
        // (out-of-range is left to validation), so no literal alternatives appear.
        assert!(
            !spec.grammar.contains("\"celsius\" | \"fahrenheit\""),
            "enum range must not be enforced in the grammar:\n{}",
            spec.grammar
        );
        // Non-string schemas still go through json_schema_to_grammar.
        assert!(
            spec.grammar.contains("tool-set-weather-arg-days-value")
                && spec.grammar.contains("integral-part"),
            "integer parameter should use a JSON-schema-derived value rule:\n{}",
            spec.grammar
        );
        assert!(
            spec.grammar
                .contains("tool-set-weather-arg-rain-value ::= (\"true\" | \"false\")"),
            "boolean parameter should use a JSON literal value rule:\n{}",
            spec.grammar
        );
        assert!(
            spec.grammar.contains("tool-set-weather-arg-metadata-value")
                && spec.grammar.contains("tool-set-weather-arg-tags-value"),
            "object/array parameters should use JSON literal value rules:\n{}",
            spec.grammar
        );
    }

    // Parameterless tools (no `properties`, or `parameters` omitted entirely)
    // are valid OpenAI tool definitions; the reference Qwen3.5 builder emits the
    // function rule with no argument parsers. Must not be rejected.
    #[test]
    fn test_build_qwen_tool_call_grammar_spec_allows_parameterless_tool() {
        let tools = r#"[
            {"type":"function","function":{"name":"ping"}},
            {"type":"function","function":{"name":"noop","parameters":{}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        assert!(spec.grammar.contains("\"<function=\" \"ping\""));
        assert!(spec.grammar.contains("\"<function=\" \"noop\""));
        // No `<parameter=` should be emitted for a parameterless tool.
        assert!(
            !spec.grammar.contains("tool-ping-arg-"),
            "parameterless tool must not emit argument rules:\n{}",
            spec.grammar
        );
    }

    #[test]
    fn test_build_qwen_tool_call_grammar_spec_uses_collision_free_rule_names() {
        let tools = r#"[
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"Foo":{"type":"string"},"foo":{"type":"string"}},"required":["Foo","foo"]}}},
            {"type":"function","function":{"name":"get-weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        assert!(
            spec.grammar
                .contains("(tool-get-weather | tool-get-weather-1)")
        );
        assert!(spec.grammar.contains("tool-get-weather-arg-foo ::="));
        assert!(spec.grammar.contains("tool-get-weather-arg-foo-1 ::="));
        assert!(spec.grammar.contains("\"<function=\" \"get_weather\""));
        assert!(spec.grammar.contains("\"<function=\" \"get-weather\""));
    }

    #[test]
    fn test_build_qwen_tool_call_grammar_spec_required_parameter_spacing_is_stable() {
        let tools = r#"[
            {"type":"function","function":{"name":"book_trip","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"}},"required":["origin","destination"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        assert!(
            spec.grammar.contains("tool-book-trip ::= (\"<function=\" \"book_trip\" \">\\n\") space tool-book-trip-arg-origin space tool-book-trip-arg-destination space \"</function>\\n\""),
            "required parameters must be separated by exactly one space rule:\n{}",
            spec.grammar
        );
    }

    #[test]
    fn test_build_qwen_tool_call_grammar_spec_optional_star_follows_required_parameters() {
        let tools = r#"[
            {"type":"function","function":{"name":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        assert!(
            spec.grammar.contains("tool-forecast ::= (\"<function=\" \"forecast\" \">\\n\") space tool-forecast-arg-city space tool-forecast-optional-parameters* space \"</function>\\n\""),
            "optional parameter group must remain repeatable after required args:\n{}",
            spec.grammar
        );
        assert!(
            spec.grammar.contains("tool-forecast-optional-parameters ::= (tool-forecast-arg-unit | tool-forecast-arg-days)"),
            "optional group alternatives must keep property order:\n{}",
            spec.grammar
        );
    }

    #[test]
    fn test_build_qwen_tool_call_grammar_spec_parallel_tool_call_rule_is_stable() {
        let tools = r#"[
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
            {"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        ]"#;

        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, true).expect("qwen tool grammar");

        assert!(
            spec.grammar.contains(
                "tool-call ::= \"<tool_call>\\n\" space (tool-get-weather | tool-get-time) \
                 (space (tool-get-weather | tool-get-time))* space \"</tool_call>\""
            ),
            "parallel tool-call envelope must repeat the full alternative set:\n{}",
            spec.grammar
        );
    }

    // Test helper: a spec with the given kind, marked required. Optional-aware
    // tests opt out by editing `required` on the returned value.
    fn spec(name: &str, value_kind: QwenToolParameterValueKind) -> QwenToolParameterSpec {
        QwenToolParameterSpec {
            name: name.to_string(),
            value_kind,
            required: true,
        }
    }

    #[test]
    fn test_qwen_tagged_arguments_to_json_preserves_value_contract() {
        use QwenToolParameterValueKind::{JsonLiteral, RawString};
        let parameters = [
            spec("city", RawString),
            spec("unit", RawString),
            spec("days", JsonLiteral),
            spec("rain", JsonLiteral),
            spec("metadata", JsonLiteral),
            spec("tags", JsonLiteral),
        ];
        let tagged_values = [
            ("city", "To\"kyo\nShibuya"),
            ("unit", "celsius"),
            ("days", "3"),
            ("rain", "true"),
            ("metadata", r#"{"source":"api"}"#),
            ("tags", r#"["coastal","night"]"#),
        ];

        let arguments =
            qwen_tagged_arguments_to_json(&parameters, &tagged_values).expect("arguments JSON");
        let value: Value = serde_json::from_str(&arguments).expect("arguments object");

        assert_eq!(value["city"], "To\"kyo\nShibuya");
        assert_eq!(value["unit"], "celsius");
        assert_eq!(value["days"], 3);
        assert_eq!(value["rain"], true);
        assert_eq!(value["metadata"]["source"], "api");
        assert_eq!(value["tags"][0], "coastal");
    }

    // An optional parameter the model omitted must be skipped, not errored: the
    // grammar quantifies optionals with `*`, so a valid tool call can leave them
    // out entirely.
    #[test]
    fn test_qwen_tagged_arguments_omits_absent_optional() {
        use QwenToolParameterValueKind::RawString;
        let mut unit = spec("unit", RawString);
        unit.required = false;
        let parameters = [spec("city", RawString), unit];
        // Only the required `city` was emitted.
        let tagged_values = [("city", "Tokyo")];

        let arguments =
            qwen_tagged_arguments_to_json(&parameters, &tagged_values).expect("arguments JSON");
        let value: Value = serde_json::from_str(&arguments).expect("arguments object");

        assert_eq!(value["city"], "Tokyo");
        assert!(
            value.get("unit").is_none(),
            "absent optional must not appear in the arguments object"
        );
    }

    // A required parameter the grammar guarantees must be present; its absence
    // signals a parse/grammar desync and is a hard error.
    #[test]
    fn test_qwen_tagged_arguments_bails_on_absent_required() {
        use QwenToolParameterValueKind::RawString;
        let parameters = [spec("city", RawString), spec("unit", RawString)];
        let tagged_values = [("city", "Tokyo")];

        let err = qwen_tagged_arguments_to_json(&parameters, &tagged_values)
            .expect_err("absent required parameter must error");
        assert!(
            err.to_string().contains("unit"),
            "error should name the missing required parameter: {err}"
        );
    }

    // The grammar allows an optional parameter to be emitted more than once
    // (`*`), so parse-back resolves duplicates last-wins.
    #[test]
    fn test_qwen_tagged_arguments_duplicate_optional_last_wins() {
        use QwenToolParameterValueKind::RawString;
        let mut unit = spec("unit", RawString);
        unit.required = false;
        let parameters = [unit];
        let tagged_values = [("unit", "celsius"), ("unit", "fahrenheit")];

        let arguments =
            qwen_tagged_arguments_to_json(&parameters, &tagged_values).expect("arguments JSON");
        let value: Value = serde_json::from_str(&arguments).expect("arguments object");

        assert_eq!(value["unit"], "fahrenheit");
    }

    // build_qwen_tool_call_grammar_spec must reproduce, byte-for-byte, the
    // grammar that the fork-only OAI API captured into the golden fixture. This
    // is the contract Phase 1 exists to satisfy: once the OAI API is gone
    // (Phase 4), the Rust builder is the only producer of this grammar, so it
    // must match what the fork produced today. Reads the committed Qwen3.5
    // fixture so the assertion tracks the real captured output, not a paraphrase.
    #[test]
    fn test_build_qwen_tool_call_grammar_spec_matches_qwen35_golden() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/qwen35_4b_auto_tool_call.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("read qwen35 golden fixture");
        let fixture: Value = serde_json::from_str(&raw).expect("golden fixture JSON");
        let expected_grammar = fixture
            .pointer("/template/grammar")
            .and_then(Value::as_str)
            .expect("fixture template.grammar");
        let expected_trigger = fixture
            .pointer("/template/grammar_triggers/0/value")
            .and_then(Value::as_str)
            .expect("fixture grammar trigger value");

        // Same single tool the fixture was captured with: get_weather(city).
        let tools = r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a given city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name, e.g. Tokyo"}},"required":["city"]}}}]"#;
        let spec =
            build_qwen_tool_call_grammar_spec(tools, false, false).expect("qwen tool grammar");

        // The fixture serializes grammar without the trailing newline the
        // builder appends after the last rule; compare on trimmed text.
        assert_eq!(
            spec.grammar.trim_end(),
            expected_grammar.trim_end(),
            "Rust-built grammar must match the captured OAI grammar"
        );
        assert_eq!(spec.grammar_triggers.len(), 1);
        assert_eq!(spec.grammar_triggers[0].value, expected_trigger);
    }

    #[test]
    fn test_render_qwen_tool_chat_template_matches_qwen35_golden() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/qwen35_4b_auto_tool_call.json"
        );
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let fixture_raw = std::fs::read_to_string(fixture_path).expect("read qwen35 fixture");
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let fixture: Value = serde_json::from_str(&fixture_raw).expect("fixture JSON");
        let messages_json =
            serde_json::to_string(fixture.pointer("/params/messages_json").unwrap()).unwrap();
        let tools_json =
            serde_json::to_string(fixture.pointer("/params/tools_json").unwrap()).unwrap();
        let kwargs_json =
            serde_json::to_string(fixture.pointer("/params/chat_template_kwargs").unwrap())
                .unwrap();

        let rendered = render_qwen_tool_chat_template(
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
        .expect("render qwen template");

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
        assert_eq!(
            rendered.additional_stops,
            Vec::<String>::new(),
            "Qwen3.5 fixture records no additional stops"
        );
        assert_eq!(rendered.chat_format, QWEN_TAGGED_CHAT_FORMAT);
        assert!(rendered.parse_tool_calls);
    }

    #[test]
    fn test_render_qwen_tool_chat_template_rejects_unsupported_tool_template() {
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping"}}]"#;
        let err = render_qwen_tool_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}",
            messages,
            Some(tools),
            Some("auto"),
            None,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unsupported chat template"));
    }

    // The Qwen detector must accept the real Qwen3.5 template but reject other
    // templates that reuse the same tags. StepFun3.5 shares the tag set yet uses
    // `tojson(ensure_ascii=False)`, which this renderer cannot evaluate, so it
    // must fall back to the legacy renderer instead of being treated as Qwen.
    #[test]
    fn test_is_qwen_tagged_tool_template_excludes_lookalike_templates() {
        let templates_dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates"
        );
        let qwen =
            std::fs::read_to_string(format!("{templates_dir}/Qwen3.5-4B.jinja")).expect("qwen");
        let stepfun = std::fs::read_to_string(format!("{templates_dir}/StepFun3.5-Flash.jinja"))
            .expect("stepfun");

        assert!(is_qwen_tagged_tool_template(&qwen));
        assert!(
            !is_qwen_tagged_tool_template(&stepfun),
            "StepFun shares the Qwen tags but must not be routed to the Qwen renderer"
        );
    }

    #[test]
    fn test_render_qwen_tool_chat_template_tool_choice_none_disables_tool_grammar() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping"}}]"#;

        let rendered = render_qwen_tool_chat_template(
            &template,
            messages,
            Some(tools),
            Some("none"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            false,
        )
        .expect("render without tool grammar");

        assert!(!rendered.prompt.contains("<tools>"));
        assert!(rendered.grammar.is_none());
        assert!(rendered.grammar_triggers.is_empty());
        assert!(!rendered.parse_tool_calls);
    }

    // The caller owns the enable_thinking default (it shares one value with the
    // legacy parser); when kwargs omit the key the renderer must honour the
    // bool the caller passed, not pick its own default. A desync here would
    // open `<think>` in the prompt while the parser expected the closed form
    // (or vice versa) and break tool-call parsing.
    #[test]
    fn test_render_qwen_tool_chat_template_honours_caller_enable_thinking_default() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;

        let thinking_off = render_qwen_tool_chat_template(
            &template, messages, None, None, None, true, false, false,
        )
        .expect("render with thinking off");
        let thinking_on = render_qwen_tool_chat_template(
            &template, messages, None, None, None, true, true, false,
        )
        .expect("render with thinking on");

        // enable_thinking=false closes the block up front; =true leaves it open.
        assert!(thinking_off.prompt.ends_with("<think>\n\n</think>\n\n"));
        assert!(thinking_on.prompt.ends_with("<think>\n"));
        assert!(!thinking_on.prompt.ends_with("</think>\n\n"));
    }

    #[test]
    fn test_render_qwen_tool_chat_template_without_tools_has_no_tool_grammar() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;

        let rendered = render_qwen_tool_chat_template(
            &template,
            messages,
            None,
            Some("auto"),
            None,
            true,
            false,
            false,
        )
        .expect("render without tools");

        assert!(!rendered.prompt.contains("<tools>"));
        assert!(rendered.grammar.is_none());
        assert!(!rendered.grammar_lazy);
        assert!(rendered.grammar_triggers.is_empty());
        assert!(!rendered.parse_tool_calls);
        assert_eq!(rendered.chat_format, QWEN_TAGGED_CHAT_FORMAT);
    }

    #[test]
    fn test_render_qwen_tool_chat_template_thinking_enabled_keeps_tool_grammar() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]"#;

        let rendered = render_qwen_tool_chat_template(
            &template,
            messages,
            Some(tools),
            Some("auto"),
            Some(r#"{"enable_thinking":true}"#),
            true,
            true,
            false,
        )
        .expect("render thinking-on tools");

        assert!(rendered.prompt.contains("<tools>"));
        assert!(rendered.prompt.ends_with("<think>\n"));
        assert!(rendered.grammar.is_some());
        assert!(rendered.grammar_lazy);
        assert_eq!(rendered.grammar_triggers.len(), 1);
        assert!(rendered.parse_tool_calls);
        assert_eq!(rendered.chat_format, QWEN_TAGGED_CHAT_FORMAT);
    }

    // chat_template_kwargs must not let the caller override the plugin-owned
    // context keys (messages/tools/enable_thinking), or the rendered prompt
    // would desync from the grammar built off the dedicated inputs.
    #[test]
    fn test_render_qwen_tool_chat_template_kwargs_cannot_override_reserved_keys() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let messages = r#"[{"role":"user","content":"keep me"}]"#;
        // kwargs tries to swap in different messages and flip enable_thinking.
        let kwargs =
            r#"{"messages":[{"role":"user","content":"INJECTED"}],"enable_thinking":true}"#;

        let rendered = render_qwen_tool_chat_template(
            &template,
            messages,
            None,
            None,
            Some(kwargs),
            true,
            false,
            false,
        )
        .expect("render with hostile kwargs");

        assert!(rendered.prompt.contains("keep me"));
        assert!(!rendered.prompt.contains("INJECTED"));
        // enable_thinking=false (the caller's value) must win over kwargs=true.
        assert!(rendered.prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    // tool_choice=required propagates through the renderer to an eager grammar,
    // and parallel_tool_calls reaches the grammar's call-count rule.
    #[test]
    fn test_render_qwen_tool_chat_template_forwards_required_and_parallel() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]"#;

        let rendered = render_qwen_tool_chat_template(
            &template,
            messages,
            Some(tools),
            Some("required"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            true,
        )
        .expect("render required+parallel");

        assert!(!rendered.grammar_lazy, "required must be eager");
        assert!(rendered.grammar_triggers.is_empty());
        let grammar = rendered.grammar.expect("required tools yield a grammar");
        assert!(grammar.contains("(tool-ping) (space (tool-ping))*"));
    }

    // Malformed inputs surface a typed error instead of panicking or rendering
    // a corrupt prompt.
    #[test]
    fn test_render_qwen_tool_chat_template_rejects_invalid_inputs() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/Qwen3.5-4B.jinja"
        );
        let template = std::fs::read_to_string(template_path).expect("read qwen35 template");

        let bad_messages = render_qwen_tool_chat_template(
            &template, "not json", None, None, None, true, false, false,
        )
        .unwrap_err();
        assert!(bad_messages.to_string().contains("invalid messages_json"));

        let bad_kwargs = render_qwen_tool_chat_template(
            &template,
            r#"[{"role":"user","content":"hi"}]"#,
            None,
            None,
            Some("not json"),
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(
            bad_kwargs
                .to_string()
                .contains("invalid chat_template_kwargs")
        );
    }

    // The single-tool `get_weather(city)` index used across the parser tests,
    // mirroring the golden fixture's tools_json.
    fn weather_param_index() -> QwenToolParamIndex {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        build_qwen_tool_param_index(tools).expect("param index")
    }

    // The Rust parser must reproduce the tool name/arguments the fork's
    // parse_response_oaicompat captured into the golden fixture (id is
    // synthesized, so compare everything except id).
    #[test]
    fn test_parse_qwen_tagged_single_tool_matches_golden() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/qwen35_4b_auto_tool_call.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("read qwen35 golden fixture");
        let fixture: Value = serde_json::from_str(&raw).expect("golden fixture JSON");
        let raw_response = fixture
            .pointer("/parse/0/raw_response")
            .and_then(Value::as_str)
            .expect("fixture parse[0].raw_response");
        let expected = fixture
            .pointer("/parse/0/parsed_json")
            .expect("fixture parse[0].parsed_json");

        let msg = parse_qwen_tagged_response(raw_response, &weather_param_index(), false)
            .expect("parse tagged response");
        assert_eq!(msg.tool_calls.len(), 1);
        let call = &msg.tool_calls[0];
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"city":"Tokyo"}"#);
        // Cross-check against the fixture's captured OAI shape (minus id).
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

    // A turn with no tool call is a plain text response.
    #[test]
    fn test_parse_qwen_tagged_plain_text_no_tool() {
        let msg =
            parse_qwen_tagged_response("Hello! How can I help you?", &weather_param_index(), false)
                .expect("parse plain text");
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.content, "Hello! How can I help you?");
        assert!(msg.reasoning_content.is_none());
    }

    #[test]
    fn test_parse_qwen_tagged_literal_tool_tags_are_content() {
        let raw = "Use <tool_call> and <function=get_weather> literally.";

        let msg = parse_qwen_tagged_response(raw, &weather_param_index(), false)
            .expect("parse literal tags");

        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.content, raw);
    }

    // Non-whitespace `<think>` content becomes reasoning_content; an empty
    // `<think>\n\n</think>` is dropped.
    #[test]
    fn test_parse_qwen_tagged_reasoning() {
        let msg = parse_qwen_tagged_response(
            "<think>step by step</think>The answer.",
            &weather_param_index(),
            false,
        )
        .expect("parse reasoning");
        assert_eq!(msg.reasoning_content.as_deref(), Some("step by step"));
        assert_eq!(msg.content, "The answer.");

        let empty =
            parse_qwen_tagged_response("<think>\n\n</think>\n\nHi", &weather_param_index(), false)
                .expect("parse empty think");
        assert!(empty.reasoning_content.is_none());
    }

    // A parameterless tool yields an empty arguments object.
    #[test]
    fn test_parse_qwen_tagged_empty_arguments() {
        let tools = r#"[{"type":"function","function":{"name":"ping"}}]"#;
        let index = build_qwen_tool_param_index(tools).expect("param index");
        let raw = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
        let msg = parse_qwen_tagged_response(raw, &index, false).expect("parse parameterless");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "ping");
        assert_eq!(msg.tool_calls[0].arguments, "{}");
    }

    // Two functions in one envelope produce two tool calls (parallel).
    #[test]
    fn test_parse_qwen_tagged_parallel() {
        let raw = "<tool_call>\n\
            <function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n\
            <function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n\
            </tool_call>";
        let msg =
            parse_qwen_tagged_response(raw, &weather_param_index(), false).expect("parse parallel");
        assert_eq!(msg.tool_calls.len(), 2);
        assert_eq!(msg.tool_calls[0].arguments, r#"{"city":"Tokyo"}"#);
        assert_eq!(msg.tool_calls[1].arguments, r#"{"city":"Osaka"}"#);
    }

    #[test]
    fn test_qwen_tagged_stream_parallel_ignores_function_marker_inside_body() {
        let tools = r#"[
            {"type":"function","function":{"name":"echo","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}},
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        ]"#;
        let index = build_qwen_tool_param_index(tools).expect("param index");
        let raw = "<tool_call>\n\
            <function=echo>\n<parameter=text>\nliteral <function=get_weather> text\n</parameter>\n</function>\n\
            <function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n\
            </tool_call>";

        let calls = parse_qwen_stream_tool_calls(raw, &index);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "echo");
        assert_eq!(
            calls[0].arguments,
            r#"{"text":"literal <function=get_weather> text"}"#
        );
        assert_eq!(calls[1].name, "get_weather");
        assert_eq!(calls[1].arguments, r#"{"city":"Osaka"}"#);
    }

    // An omitted optional parameter does not error and is absent from output.
    #[test]
    fn test_parse_qwen_tagged_optional_omitted() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string"}},"required":["city"]}}}]"#;
        let index = build_qwen_tool_param_index(tools).expect("param index");
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
        let msg = parse_qwen_tagged_response(raw, &index, false).expect("parse optional omitted");
        assert_eq!(msg.tool_calls[0].arguments, r#"{"city":"Tokyo"}"#);
    }

    // A non-string (JSON-literal) parameter is reconstructed as the JSON value,
    // not a quoted string.
    #[test]
    fn test_parse_qwen_tagged_json_literal_value() {
        let tools = r#"[{"type":"function","function":{"name":"set_count","parameters":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]}}}]"#;
        let index = build_qwen_tool_param_index(tools).expect("param index");
        let raw = "<tool_call>\n<function=set_count>\n<parameter=count>\n42\n</parameter>\n</function>\n</tool_call>";
        let msg = parse_qwen_tagged_response(raw, &index, false).expect("parse json literal");
        assert_eq!(msg.tool_calls[0].arguments, r#"{"count":42}"#);
    }

    #[test]
    fn test_parse_qwen_tagged_raw_string_allows_function_like_text() {
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nUse </function> and </tool_call> literally\n</parameter>\n</function>\n</tool_call>";

        let msg = parse_qwen_tagged_response(raw, &weather_param_index(), false)
            .expect("parse raw string with tag-like text");

        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(
            msg.tool_calls[0].arguments,
            r#"{"city":"Use </function> and </tool_call> literally"}"#
        );
    }

    // A truncated response (missing closing tags) must not panic when partial
    // parsing is allowed; a function whose name is not yet known is omitted.
    #[test]
    fn test_parse_qwen_tagged_partial_unclosed() {
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTok";
        let msg = parse_qwen_tagged_response(raw, &weather_param_index(), true)
            .expect("partial parse must not error");
        // city value not yet closed, so no complete parameter — arguments empty.
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "get_weather");

        let no_name = "<tool_call>\n<function=";
        let msg2 = parse_qwen_tagged_response(no_name, &weather_param_index(), true)
            .expect("partial parse with no name");
        assert!(msg2.tool_calls.is_empty());
    }

    #[test]
    fn test_qwen_tagged_stream_replays_qwen35_golden_chunks() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tool_calling_golden/qwen35_4b_auto_tool_call.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("read qwen35 golden fixture");
        let fixture: Value = serde_json::from_str(&raw).expect("golden fixture JSON");
        let tools_json =
            serde_json::to_string(fixture.pointer("/params/tools_json").unwrap()).unwrap();
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

        let mut state =
            QwenTaggedStreamState::new(&tools_json, generation_prompt).expect("stream state");

        assert_eq!(chunks.len(), expected.len());
        for (chunk_index, (chunk, expected_deltas)) in chunks.iter().zip(expected).enumerate() {
            let deltas = state
                .update(chunk.as_str().expect("chunk string"), true)
                .expect("stream update");
            let mut actual: Vec<Value> = deltas
                .iter()
                .map(|delta| serde_json::from_str(delta).expect("delta JSON"))
                .collect();
            let mut expected_deltas = expected_deltas.as_array().expect("delta array").to_vec();
            normalize_stream_delta_ids(&mut actual);
            normalize_stream_delta_ids(&mut expected_deltas);
            assert_eq!(
                actual, expected_deltas,
                "delta mismatch for chunk #{chunk_index} {chunk:?}"
            );
        }
        let final_deltas = state.update("", false).expect("flush");
        assert!(
            final_deltas.is_empty(),
            "golden fixture already flushes all deltas"
        );
    }

    #[test]
    fn test_qwen_tagged_stream_handles_partial_and_regression_inputs() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let mut state =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");

        assert_eq!(
            state.update("<tool_call>\n<function=", true).unwrap().len(),
            0
        );
        let partial = state
            .update("get_weather>\n<parameter=city>\nTok", true)
            .unwrap();
        let partial_update = decode_oai_deltas(&partial);
        assert_eq!(partial_update.tool_calls.len(), 1);
        assert_eq!(
            partial_update.tool_calls[0].arguments_chunk,
            r#"{"city":"Tok"#
        );
        let deltas = state
            .update("yo\n</parameter>\n</function>\n</tool_call>", true)
            .unwrap();
        let update = decode_oai_deltas(&deltas);
        assert_eq!(update.tool_calls.len(), 1);
        assert_eq!(update.tool_calls[0].fn_name, None);
        assert_eq!(update.tool_calls[0].arguments_chunk, "yo\"}");

        let mut regressed =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");
        regressed
            .update(
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>",
                true,
            )
            .unwrap();
        regressed.raw = "<tool_call>\n<function=get_weather>".to_string();
        assert!(regressed.update("", true).unwrap().is_empty());
    }

    #[test]
    fn test_qwen_tagged_stream_split_tool_call_prefix_does_not_stall() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let mut state =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");

        assert!(state.update("<", true).unwrap().is_empty());
        assert!(state.update("tool_call", true).unwrap().is_empty());
        assert!(state.update(">", true).unwrap().is_empty());
        let deltas = state
            .update("\n<function=get_weather>\n", true)
            .expect("function opening emits call");
        let update = decode_oai_deltas(&deltas);

        assert_eq!(update.tool_calls.len(), 1);
        assert_eq!(update.tool_calls[0].fn_name.as_deref(), Some("get_weather"));
        assert_eq!(update.tool_calls[0].arguments_chunk, "{");
    }

    #[test]
    fn test_qwen_tagged_stream_keeps_text_before_tool_call() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let mut state =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");

        let text = state.update("Let me check ", true).unwrap();
        assert_eq!(decode_oai_deltas(&text).text, "Let me check ");
        assert!(state.update("<", true).unwrap().is_empty());
        assert!(state.update("tool_call", true).unwrap().is_empty());
        assert!(state.update(">", true).unwrap().is_empty());
        let deltas = state
            .update("\n<function=get_weather>\n", true)
            .expect("function opening emits call");
        let update = decode_oai_deltas(&deltas);

        assert!(update.text.is_empty());
        assert_eq!(update.tool_calls.len(), 1);
        assert_eq!(update.tool_calls[0].fn_name.as_deref(), Some("get_weather"));
        assert_eq!(update.tool_calls[0].arguments_chunk, "{");
    }

    #[test]
    fn test_qwen_tagged_stream_keeps_literal_tool_call_text_as_content() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let mut state =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");

        let first = state.update("Use <tool_call>", true).unwrap();
        assert_eq!(decode_oai_deltas(&first).text, "Use ");
        let rest = state.update(" literally in docs.", true).unwrap();
        let update = decode_oai_deltas(&rest);

        assert_eq!(update.text, "<tool_call> literally in docs.");
        assert!(update.tool_calls.is_empty());
    }

    #[test]
    fn test_qwen_tagged_stream_keeps_literal_function_text_as_content() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let mut state =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");

        let first = state.update("Use <tool_call>", true).unwrap();
        assert_eq!(decode_oai_deltas(&first).text, "Use ");
        let rest = state
            .update(" and <function=get_weather> literally.", true)
            .unwrap();
        let update = decode_oai_deltas(&rest);

        assert_eq!(
            update.text,
            "<tool_call> and <function=get_weather> literally."
        );
        assert!(update.tool_calls.is_empty());
    }

    #[test]
    fn test_qwen_tagged_stream_raw_string_allows_function_like_text() {
        let tools = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
        let mut state =
            QwenTaggedStreamState::new(tools, "<|im_start|>assistant\n").expect("stream state");

        let deltas = state
            .update(
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nUse </function> and </tool_call> literally\n</parameter>\n</function>\n</tool_call>",
                true,
            )
            .unwrap();
        let update = decode_oai_deltas(&deltas);

        assert_eq!(update.tool_calls.len(), 1);
        assert_eq!(update.tool_calls[0].fn_name.as_deref(), Some("get_weather"));
        assert_eq!(
            update.tool_calls[0].arguments_chunk,
            r#"{"city":"Use </function> and </tool_call> literally"}"#
        );
    }
}
