//! Loosely-coupled free helpers for the `model` module.
//!
//! These functions take no `&self` and are split out of `model.rs` so the
//! decode/wrapper code paths can call them while the context is borrowed out of
//! `self` (see `check_token_length`) and so the KV-cache planning, tokenization,
//! and tool-response parsing helpers can be read and tested independently of the
//! large `LlamaModelWrapper` impl. Behavior is unchanged from the pre-split
//! single-file layout — this is a pure move.

use super::CachedChunk;
use anyhow::{Context, Result, bail};
use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaInput;
use jobworkerp_llama_protobuf::protobuf::llm::{LlmCompletionArgs, llm_chat_args};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use std::ops::ControlFlow;

// Routes a chat template to its Rust renderer (Gemma4 first, then Qwen),
// returning `None` when the template is not a ported format. Kept as a free
// function so the routing decision is testable without loading a GGUF model.
#[allow(clippy::too_many_arguments)]
pub(in crate::model) fn render_ported_tool_template_result(
    template_str: &str,
    oai_messages_json: &str,
    tools_json: &str,
    tool_choice: Option<&str>,
    chat_template_kwargs: Option<&str>,
    add_generation_prompt: bool,
    enable_thinking: bool,
    parallel_tool_calls: bool,
) -> Result<Option<crate::oai_chat::ToolChatTemplateResult>> {
    if crate::oai_chat::is_gemma4_tool_template(template_str) {
        return crate::oai_chat::render_gemma4_tool_chat_template(
            template_str,
            oai_messages_json,
            Some(tools_json),
            tool_choice,
            chat_template_kwargs,
            add_generation_prompt,
            enable_thinking,
            parallel_tool_calls,
        )
        .map(Some);
    }
    if crate::oai_chat::is_qwen_tagged_tool_template(template_str) {
        return crate::oai_chat::render_qwen_tool_chat_template(
            template_str,
            oai_messages_json,
            Some(tools_json),
            tool_choice,
            chat_template_kwargs,
            add_generation_prompt,
            // Share the single `enable_thinking` value resolved by the caller so
            // the Rust-rendered prompt and parser agree on whether the assistant
            // turn opens a <think> block.
            enable_thinking,
            parallel_tool_calls,
        )
        .map(Some);
    }
    Ok(None)
}

/// Length of the longest common prefix of two token slices. Used for KV prefix
/// reuse: tokens up to this length are already in the KV cache and can be kept.
pub(in crate::model) fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Number of KV-cached prompt tokens to keep for prefix reuse. Returns 0 (full
/// clear) when reuse is off or there is no cache. Capped at `prompt_tokens - 1`
/// so the last prompt token is always re-decoded — the first sample needs fresh
/// logits, and an empty prefill range would leave none. `prompt_tokens >= 1` is
/// guaranteed by `AddBos::Always` plus the empty-prompt bail in `run`.
pub(in crate::model) fn plan_kv_keep(
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
pub(in crate::model) fn plan_chunk_keep(
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
pub(in crate::model) fn describe_chunks(
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
pub(in crate::model) fn check_token_length(
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

/// Returns `true` when the streaming sink asked to stop generation. Used by
/// both core loops so the "Break → cancel + suppress KV write-back" rule is
/// expressed in one place.
pub(in crate::model) fn sink_requests_stop(
    sink: &mut dyn FnMut(&str) -> ControlFlow<()>,
    chunk: &str,
) -> bool {
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
pub(in crate::model) fn strip_generation_prompt<'a>(
    s: &'a str,
    generation_prompt: &str,
) -> &'a str {
    if let Some(rest) = s.strip_prefix(generation_prompt) {
        return rest;
    }
    let mut longest_line_prefix = None;
    for (idx, _) in generation_prompt.match_indices('\n') {
        let end = idx + 1;
        if s.starts_with(&generation_prompt[..end]) {
            longest_line_prefix = Some(end);
        }
    }
    if let Some(end) = longest_line_prefix {
        return &s[end..];
    }
    let header_end = match generation_prompt.find('\n') {
        Some(i) => i + 1,
        None => return s,
    };
    let header = &generation_prompt[..header_end];
    s.strip_prefix(header).unwrap_or(s)
}

/// Try the OAI parser on the raw output first; on failure peel off the
/// regenerated assistant header (eager-grammar paths like
/// `tool_choice="required"` make the model echo `generation_prompt`); on
/// final failure recover the tool calls with a tag-based fallback.
/// Returning Err means none of the three approaches found a structured
/// reply, which is genuinely unrecoverable.
pub(in crate::model) fn parse_tool_response_with_recovery(
    tmpl_result: &crate::oai_chat::ToolChatTemplateResult,
    tools_json: &str,
    raw: &str,
) -> Result<String> {
    let stripped = strip_generation_prompt(raw, &tmpl_result.generation_prompt);
    if !tmpl_result.parse_tool_calls {
        let msg = crate::oai_chat::ParsedChatMsg {
            content: stripped.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
        };
        return crate::oai_chat::parsed_msg_to_oai_json(&msg);
    }
    if crate::oai_chat::is_qwen_rust_parser(tmpl_result) {
        let index = crate::oai_chat::build_qwen_tool_param_index(tools_json)?;
        let msg = crate::oai_chat::parse_qwen_tagged_response(stripped, &index, false)?;
        return crate::oai_chat::parsed_msg_to_oai_json(&msg);
    }
    if crate::oai_chat::is_gemma4_rust_parser(tmpl_result) {
        let msg = crate::oai_chat::parse_gemma4_response(stripped, false)?;
        return crate::oai_chat::parsed_msg_to_oai_json(&msg);
    }
    bail!(
        "unsupported tool-calling chat_format {} for Rust response parser",
        tmpl_result.chat_format
    )
}

/// Token-to-piece with a one-shot retry on `InsufficientBufferSpace`. `special`
/// controls whether the token is rendered as a special (literal `<tool_call>`
/// etc) or plaintext: the tools path passes `false` for tokens outside
/// `preserved_tokens` so user-facing whitespace isn't double-escaped; the
/// non-tools path passes `true`.
pub(in crate::model) fn token_to_piece_bytes_retry_special(
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
pub(in crate::model) fn decode_image_to_media(
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
pub(in crate::model) fn validate_completion_args(args: &LlmCompletionArgs) -> Result<()> {
    if args.prompt.is_empty() {
        bail!("prompt is empty");
    }
    if let Some(fo) = &args.function_options
        && fo.use_function_calling
    {
        bail!(super::ERR_USE_FUNCTION_CALLING_UNSUPPORTED);
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
    use crate::model::test_support::*;

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
        // Future templates may have multi-line assistant preambles where the
        // model regenerates only a proper prefix of `generation_prompt`.
        let multi_line = "<|im_start|>assistant\n<think>\nplanning\n</think>\n\n";
        let partial_multi_line = "<|im_start|>assistant\n<think>\n<tool_call>{}</tool_call>";
        assert_eq!(
            strip_generation_prompt(partial_multi_line, multi_line),
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
                .contains(crate::model::config::ERR_USE_FUNCTION_CALLING_UNSUPPORTED),
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

    #[test]
    fn test_render_tool_template_result_qwen_required_parallel_uses_rust_renderer() {
        let template = template_fixture("Qwen3.5-4B.jinja");
        let result = render_ported_tool_template_result(
            &template,
            r#"[{"role":"user","content":"ping"}]"#,
            poc_ping_tools_json(),
            Some("required"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            true,
        )
        .expect("qwen template should be ported")
        .expect("render qwen tool template");

        assert!(result.parser.is_none());
        assert!(!result.grammar_lazy, "required tool choice must be eager");
        assert!(result.grammar_triggers.is_empty());
        assert!(result.parse_tool_calls);
        let grammar = result.grammar.expect("qwen tools yield grammar");
        assert!(grammar.contains("(tool-ping) (space (tool-ping))*"));
    }

    #[test]
    fn test_render_ported_tool_template_result_qwen_does_not_need_legacy_result() {
        let template = template_fixture("Qwen3.5-4B.jinja");
        let result = render_ported_tool_template_result(
            &template,
            r#"[{"role":"user","content":"ping"}]"#,
            poc_ping_tools_json(),
            Some("required"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            true,
        )
        .expect("qwen template should be ported")
        .expect("render qwen tool template");

        assert!(result.parser.is_none());
        assert!(!result.grammar_lazy);
        assert_eq!(result.chat_format, crate::oai_chat::QWEN_TAGGED_CHAT_FORMAT);
        assert!(result.parse_tool_calls);
    }

    #[test]
    fn test_render_tool_template_result_tool_choice_none_disables_qwen_tools() {
        let template = template_fixture("Qwen3.5-4B.jinja");
        let result = render_ported_tool_template_result(
            &template,
            r#"[{"role":"user","content":"ping"}]"#,
            poc_ping_tools_json(),
            Some("none"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            true,
        )
        .expect("qwen template should be ported")
        .expect("render qwen tool_choice none");

        assert!(result.parser.is_none());
        assert!(result.grammar.is_none());
        assert!(!result.grammar_lazy);
        assert!(result.grammar_triggers.is_empty());
        assert!(!result.parse_tool_calls);
        assert!(!result.prompt.contains("<tools>"));
    }

    #[test]
    fn test_render_tool_template_result_gemma4_uses_rust_renderer() {
        let template = template_fixture("google-gemma-4-31B-it.jinja");
        let result = render_ported_tool_template_result(
            &template,
            r#"[{"role":"user","content":"ping"}]"#,
            r#"[{"type":"function","function":{"name":"ping","description":"Ping.","parameters":{"type":"object","properties":{}}}}]"#,
            Some("auto"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            false,
        )
        .expect("gemma4 template should be ported")
        .expect("render gemma4 tool template");

        assert!(result.parser.is_none());
        assert!(result.grammar.is_some());
        assert!(result.grammar_lazy);
        assert_eq!(result.grammar_triggers.len(), 1);
        assert_eq!(result.chat_format, crate::oai_chat::GEMMA4_CHAT_FORMAT);
        assert!(result.parse_tool_calls);
    }

    #[test]
    fn test_render_ported_tool_template_result_gemma4_does_not_need_legacy_result() {
        let template = template_fixture("google-gemma-4-31B-it.jinja");
        let result = render_ported_tool_template_result(
            &template,
            r#"[{"role":"user","content":"ping"}]"#,
            r#"[{"type":"function","function":{"name":"ping","description":"Ping.","parameters":{"type":"object","properties":{}}}}]"#,
            Some("auto"),
            Some(r#"{"enable_thinking":false}"#),
            true,
            false,
            false,
        )
        .expect("gemma4 template should be ported")
        .expect("render gemma4 tool template");

        assert!(result.parser.is_none());
        assert!(result.grammar.is_some());
        assert_eq!(result.chat_format, crate::oai_chat::GEMMA4_CHAT_FORMAT);
        assert!(result.parse_tool_calls);
    }

    #[test]
    fn test_render_ported_tool_template_result_returns_none_for_legacy_template() {
        let template = template_fixture("StepFun3.5-Flash.jinja");
        let result = render_ported_tool_template_result(
            &template,
            r#"[{"role":"user","content":"ping"}]"#,
            poc_ping_tools_json(),
            Some("auto"),
            None,
            true,
            false,
            true,
        )
        .expect("legacy template should be skipped");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_tool_response_with_recovery_uses_qwen_rust_parser() {
        let tmpl_result = crate::oai_chat::ToolChatTemplateResult {
            chat_format: crate::oai_chat::QWEN_TAGGED_CHAT_FORMAT,
            generation_prompt: "<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string(),
            parser: None,
            ..fake_legacy_template_result()
        };
        let raw = "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";

        let parsed =
            parse_tool_response_with_recovery(&tmpl_result, poc_tools_json(), raw).unwrap();
        let value: serde_json::Value = serde_json::from_str(&parsed).unwrap();

        assert_eq!(value["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Tokyo"}"#
        );
    }

    #[test]
    fn test_parse_tool_response_with_recovery_qwen_plain_text() {
        let tmpl_result = crate::oai_chat::ToolChatTemplateResult {
            chat_format: crate::oai_chat::QWEN_TAGGED_CHAT_FORMAT,
            generation_prompt: "<|im_start|>assistant\n".to_string(),
            parser: None,
            ..fake_legacy_template_result()
        };

        let parsed =
            parse_tool_response_with_recovery(&tmpl_result, poc_tools_json(), "Hello").unwrap();
        let value: serde_json::Value = serde_json::from_str(&parsed).unwrap();

        assert_eq!(value["content"], "Hello");
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn test_parse_tool_response_with_recovery_respects_parse_tool_calls_false() {
        let tmpl_result = crate::oai_chat::ToolChatTemplateResult {
            chat_format: crate::oai_chat::QWEN_TAGGED_CHAT_FORMAT,
            generation_prompt: "<|im_start|>assistant\n".to_string(),
            parser: None,
            parse_tool_calls: false,
            ..fake_legacy_template_result()
        };
        let raw = "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";

        let parsed =
            parse_tool_response_with_recovery(&tmpl_result, poc_tools_json(), raw).unwrap();
        let value: serde_json::Value = serde_json::from_str(&parsed).unwrap();

        assert_eq!(
            value["content"],
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
        );
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn test_parse_tool_response_with_recovery_uses_gemma4_rust_parser() {
        let tmpl_result = crate::oai_chat::ToolChatTemplateResult {
            chat_format: crate::oai_chat::GEMMA4_CHAT_FORMAT,
            generation_prompt: "<|turn>model\n".to_string(),
            parser: None,
            ..fake_legacy_template_result()
        };
        let raw = "<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>";

        let parsed =
            parse_tool_response_with_recovery(&tmpl_result, poc_tools_json(), raw).unwrap();
        let value: serde_json::Value = serde_json::from_str(&parsed).unwrap();

        assert_eq!(value["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Tokyo"}"#
        );
    }

    #[test]
    fn test_parse_tool_response_with_recovery_rejects_unsupported_format() {
        let tmpl_result = crate::oai_chat::ToolChatTemplateResult {
            chat_format: 99,
            ..fake_legacy_template_result()
        };

        let err =
            parse_tool_response_with_recovery(&tmpl_result, poc_tools_json(), "plain").unwrap_err();

        assert!(
            err.to_string()
                .contains("unsupported tool-calling chat_format 99"),
            "{err:?}"
        );
    }

    #[test]
    fn test_tool_calling_golden_fixtures_have_expected_shape() {
        let mut checked = 0;
        for entry in std::fs::read_dir(TOOL_CALLING_GOLDEN_DIR).expect("golden fixture dir") {
            let path = entry.expect("fixture entry").path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            load_checked_golden_fixture(&path);
            checked += 1;
        }
        assert!(
            checked > 0,
            "at least one checked-in golden fixture is required"
        );
    }
}
