//! Shared `#[cfg(test)]` fixtures and helpers used by the per-module test
//! suites in `config`, `wrapper`, `decode`, and `helpers`.
//!
//! Centralising these mirrors how `oai_chat`'s split keeps shared test helpers
//! in one place. Everything is `pub(in crate::model)` so each submodule's
//! `mod tests` can pull them in via `use crate::model::test_support::*`.

#![cfg(test)]

use super::*;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaInput;
use jobworkerp_llama_protobuf::protobuf::llm::{LlmCompletionArgs, llm_chat_args};

pub(in crate::model) fn make_args(prompt: &str, medias: Vec<MediaInput>) -> InferenceArgs {
    InferenceArgs {
        prompt: prompt.to_string(),
        sample_len: Some(128),
        medias,
        ..Default::default()
    }
}

pub(in crate::model) fn dummy_media() -> MediaInput {
    use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaKind;
    use jobworkerp_llama_protobuf::protobuf::llama_cpp::media_input::Source;
    MediaInput {
        kind: MediaKind::Image as i32,
        source: Some(Source::Encoded(vec![0xFF, 0xD8, 0xFF])),
        id: None,
    }
}

pub(in crate::model) fn make_chat_msg(
    role: llm_chat_args::ChatRole,
    text: &str,
) -> llm_chat_args::ChatMessage {
    llm_chat_args::ChatMessage {
        role: role as i32,
        content: Some(llm_chat_args::MessageContent {
            content: Some(llm_chat_args::message_content::Content::Text(
                text.to_string(),
            )),
        }),
    }
}

pub(in crate::model) fn make_image_msg(base64_data: &str) -> llm_chat_args::ChatMessage {
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

pub(in crate::model) fn make_completion_args(prompt: &str) -> LlmCompletionArgs {
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

// The structured-output schema reported as "json_schema not taking effect"
// (agent-app thread-reflection-single.yaml). Kept here verbatim so the
// reproduction tests below exercise exactly what production sends.
pub(in crate::model) const THREAD_REFLECTION_SCHEMA: &str = r#"{
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

// Test-only helper exposing the env-driven config builder so reproduction
// tests don't duplicate the envy wiring.
pub(in crate::model) struct LlamaCppPluginTestEnv;

// Shared env for the JSON-schema reproduction tests: a small Qwen3 model on
// CPU with a fixed seed for determinism.
pub(in crate::model) const QWEN3_JSON_TEST_ENV: &str = "
LLAMA_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LLAMA_HF_REPO=unsloth/Qwen3-0.6B-GGUF
LLAMA_DISABLE_GPU=true
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=false
LLAMA_SYSTEM_PROMPT=You are a reflection generator. Respond ONLY with a single JSON object.
";

impl LlamaCppPluginTestEnv {
    pub(in crate::model) fn config() -> LlamaModelConfig {
        envy::prefixed("LLAMA_")
            .from_env::<LlamaModelConfig>()
            .expect("read model config from env")
    }

    // Load the shared env and build a model wrapper for the reproduction
    // tests.
    pub(in crate::model) fn load_wrapper() -> LlamaModelWrapper {
        dotenvy::from_read(QWEN3_JSON_TEST_ENV.as_bytes()).ok();
        LlamaModelWrapper::new(Self::config()).expect("load model from env")
    }
}

// ---------------------------------------------------------------------
// Real-model regression tests for client-side tool calling. The `poc_*`
// tests exercise the Rust renderer/parser path end-to-end; the
// `test_apply_oai_*` / `test_build_sampler_*` tests cover individual
// layers. All require `LLAMA_MODEL` / `LLAMA_HF_REPO` env vars and are
// `#[ignore]` by default.
// ---------------------------------------------------------------------

pub(in crate::model) const QWEN35_TOOL_GOLDEN_ENV: &str = "
LLAMA_MODEL=Qwen3.5-4B-UD-Q4_K_XL.gguf
LLAMA_HF_REPO=unsloth/Qwen3.5-4B-GGUF
LLAMA_DISABLE_GPU=false
LLAMA_SEED=1024
LLAMA_THREADS=8
LLAMA_USE_FLASH_ATTENTION=true
LLAMA_SYSTEM_PROMPT=You are a helpful assistant. When a relevant tool is available, call it.
";

pub(in crate::model) fn load_wrapper_for_tool_poc() -> LlamaModelWrapper {
    dotenvy::from_read_override(QWEN35_TOOL_GOLDEN_ENV.as_bytes()).ok();
    LlamaModelWrapper::new(LlamaCppPluginTestEnv::config())
        .expect("load Qwen3.5 model for tool-calling tests")
}

pub(in crate::model) fn load_wrapper_from_tool_env(env: &str) -> LlamaModelWrapper {
    dotenvy::from_read_override(env.as_bytes()).ok();
    LlamaModelWrapper::new(LlamaCppPluginTestEnv::config())
        .expect("load model for tool-calling golden capture")
}

// OpenAI-compatible single function shared by the tool-calling tests.
pub(in crate::model) fn poc_tools_json() -> &'static str {
    r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a given city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name, e.g. Tokyo"}},"required":["city"]}}}]"#
}

pub(in crate::model) fn poc_ping_tools_json() -> &'static str {
    r#"[{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]"#
}

pub(in crate::model) fn template_fixture(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../modules/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/models/templates/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|err| panic!("read template fixture {name}: {err}"))
}

pub(in crate::model) fn fake_legacy_template_result() -> crate::oai_chat::ToolChatTemplateResult {
    crate::oai_chat::ToolChatTemplateResult {
        prompt: "legacy prompt".to_string(),
        grammar: Some("legacy grammar".to_string()),
        grammar_lazy: true,
        grammar_triggers: vec![crate::oai_chat::ToolGrammarTrigger {
            trigger_type: crate::oai_chat::ToolGrammarTriggerType::Word,
            value: "legacy".to_string(),
            token: None,
        }],
        preserved_tokens: vec!["legacy-token".to_string()],
        additional_stops: vec!["legacy-stop".to_string()],
        chat_format: 99,
        parser: Some("legacy parser".to_string()),
        generation_prompt: "legacy generation".to_string(),
        parse_tool_calls: true,
    }
}

pub(in crate::model) const TOOL_CALLING_GOLDEN_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tool_calling_golden"
);

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(in crate::model) struct ToolCallingGoldenFixture {
    fixture_version: u32,
    is_synthetic: bool,
    model_family: String,
    model_repo: String,
    model_file: String,
    scenario: String,
    params: GoldenTemplateParams,
    template: GoldenTemplateResult,
    parse: Vec<GoldenParseCase>,
    streaming: Vec<GoldenStreamingCase>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(in crate::model) struct GoldenTemplateParams {
    messages_json: serde_json::Value,
    tools_json: Option<serde_json::Value>,
    tool_choice: Option<String>,
    parallel_tool_calls: bool,
    enable_thinking: bool,
    chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(in crate::model) struct GoldenTemplateResult {
    prompt: String,
    grammar: Option<String>,
    grammar_lazy: bool,
    grammar_triggers: Vec<GoldenGrammarTrigger>,
    preserved_tokens: Vec<String>,
    additional_stops: Vec<String>,
    generation_prompt: String,
    chat_format: i32,
    parser: Option<String>,
    parse_tool_calls: bool,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(in crate::model) struct GoldenGrammarTrigger {
    trigger_type: String,
    value: String,
    token: Option<i32>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(in crate::model) struct GoldenParseCase {
    name: String,
    raw_response: String,
    parsed_json: serde_json::Value,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(in crate::model) struct GoldenStreamingCase {
    name: String,
    chunks: Vec<String>,
    deltas: Vec<serde_json::Value>,
}

pub(in crate::model) fn assert_valid_golden_fixture(fixture: &ToolCallingGoldenFixture) {
    assert_eq!(fixture.fixture_version, 1);
    assert!(!fixture.model_family.trim().is_empty());
    assert!(!fixture.model_repo.trim().is_empty());
    assert!(!fixture.model_file.trim().is_empty());
    assert!(!fixture.scenario.trim().is_empty());
    assert!(
        fixture.params.messages_json.is_array(),
        "messages_json must be an array"
    );
    if let Some(tools) = &fixture.params.tools_json {
        assert!(tools.is_array(), "tools_json must be an array when present");
    }
    assert!(!fixture.template.prompt.is_empty());
    assert!(
        fixture.template.parse_tool_calls || fixture.params.tools_json.is_none(),
        "tool fixtures must record a parser-enabled template result"
    );
    for parse_case in &fixture.parse {
        assert!(!parse_case.name.trim().is_empty());
        assert!(
            parse_case.parsed_json.is_object(),
            "parsed_json must be an object"
        );
        if fixture.params.tools_json.is_some() && !fixture.is_synthetic {
            assert!(
                parse_case
                    .parsed_json
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .is_some_and(|calls| !calls.is_empty()),
                "real tool fixture parse case must contain non-empty tool_calls: {}",
                parse_case.name
            );
        }
    }
    for stream_case in &fixture.streaming {
        assert!(!stream_case.name.trim().is_empty());
        assert_eq!(
            stream_case.chunks.len(),
            stream_case.deltas.len(),
            "streaming golden keeps one delta batch per input chunk"
        );
        if fixture.params.tools_json.is_some() && !fixture.is_synthetic {
            assert!(
                streaming_case_has_tool_call_delta(stream_case),
                "real tool fixture streaming case must contain at least one tool_call delta: {}",
                stream_case.name
            );
        }
    }
}

pub(in crate::model) fn streaming_case_has_tool_call_delta(
    stream_case: &GoldenStreamingCase,
) -> bool {
    stream_case
        .deltas
        .iter()
        .filter_map(|batch| batch.as_array())
        .flatten()
        .any(|delta| {
            delta
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .is_some_and(|calls| !calls.is_empty())
        })
}

pub(in crate::model) fn load_checked_golden_fixture(
    path: &std::path::Path,
) -> ToolCallingGoldenFixture {
    let raw = std::fs::read_to_string(path).expect("read golden fixture");
    let fixture: ToolCallingGoldenFixture =
        serde_json::from_str(&raw).expect("golden fixture JSON shape");
    assert_valid_golden_fixture(&fixture);
    fixture
}

pub(in crate::model) fn poc_inference_args(
    prompt: &str,
    temperature: Option<f64>,
) -> InferenceArgs {
    InferenceArgs {
        prompt: prompt.to_string(),
        max_new_tokens: Some(192),
        temperature,
        seed: Some(1024),
        ..Default::default()
    }
}
