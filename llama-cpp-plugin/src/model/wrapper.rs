//! `LlamaModelWrapper`: the plugin's model handle and its high-level entry
//! points (initialization, chat-template construction, and the `run` / `chat` /
//! `completion` API surface that `lib.rs` drives).
//!
//! The struct definition lives here; its decode loop is a second
//! `impl LlamaModelWrapper` block in `decode`. Behavior is unchanged from the
//! pre-split single-file layout — this is a pure move.

use super::*;

pub struct LlamaModelWrapper {
    pub(in crate::model) model: &'static LlamaModel,
    pub(in crate::model) backend: &'static LlamaBackend,
    pub(in crate::model) ctx_params: LlamaContextParams,
    pub(in crate::model) system_prompt: String,
    /// Reused across requests: built lazily on the first decode, then kept so
    /// the KV-cache allocation happens once. KV is cleared at the start of each
    /// request for isolation. `None` until the first decode.
    ///
    /// No Mutex needed: all access is through `&mut self` on `run()`, which
    /// provides exclusive access at the Rust borrow-checker level.
    pub(in crate::model) context: Option<SyncContext>,
    /// Runner-level KV prefix reuse policy. None lets each request decide.
    pub(in crate::model) runner_reuse_kv_prefix: Option<bool>,
    pub(in crate::model) mtmd: Option<MtmdRuntime>,
    pub(in crate::model) media_limits: MediaLimits,
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
            runner_reuse_kv_prefix: config.reuse_kv_prefix,
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

    pub(in crate::model) fn build_sampler(&self, args: &InferenceArgs) -> Result<LlamaSampler> {
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
    pub(in crate::model) fn ensure_context(&mut self) -> Result<()> {
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
    /// request through the Rust tool-template renderers so that the resulting
    /// `LlmChatResult` can report parsed tool calls in
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
        let mut stream_parser =
            crate::oai_chat::OaiStreamParser::for_template(&tmpl_result, effective_tools_json)?;
        let output = {
            let mut wrapped_sink = |chunk: &str| -> ControlFlow<()> {
                // Always forward the raw chunk first; the legacy receiver
                // uses it for cancel-driven backpressure and for any other
                // downstream consumer that cares about per-token output.
                let raw_flow = sink(chunk);
                let deltas = stream_parser.update(chunk, true);
                match deltas {
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
                        tracing::warn!("tool-call stream parser failed mid-stream: {e:?}");
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
        let final_deltas = stream_parser.update("", false);
        if let Ok(final_deltas) = final_deltas
            && !final_deltas.is_empty()
        {
            let upd = crate::oai_chat::decode_oai_deltas(&final_deltas);
            if !upd.text.is_empty() || !upd.reasoning.is_empty() || !upd.tool_calls.is_empty() {
                oai_sink(upd);
            }
        }

        let parsed_json =
            parse_tool_response_with_recovery(&tmpl_result, effective_tools_json, &output.text)?;
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
                        bail!(ERR_TOOL_EXECUTION_REQUESTS_REJECTED);
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
    pub(in crate::model) fn format_single_turn(&self, user_prompt: &str) -> Result<String> {
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

    /// Apply the model's tool-aware chat template through the Rust renderers.
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
    pub(in crate::model) fn apply_oai_template_with_tools(
        &self,
        oai_messages_json: &str,
        tools_json: &str,
        tool_opts: Option<&llm_chat_args::FunctionOptions>,
        tool_choice_override: Option<&str>,
    ) -> Result<crate::oai_chat::ToolChatTemplateResult> {
        let tmpl = self
            .model
            .chat_template(None)
            .context("model does not expose a chat template required for tool calling")?;
        let chat_template_kwargs = tool_opts.and_then(|fo| fo.chat_template_kwargs.as_deref());
        // Keep one resolved value for both jinja rendering and Rust parsing so
        // the prompt and parser agree on whether the assistant turn opens a
        // <think> block.
        let enable_thinking =
            crate::oai_chat::extract_enable_thinking(chat_template_kwargs).unwrap_or(false);
        let tool_choice =
            tool_choice_override.or_else(|| tool_opts.and_then(|fo| fo.tool_choice.as_deref()));
        let parallel_tool_calls = tool_opts
            .and_then(|fo| fo.parallel_tool_calls)
            .unwrap_or(false);
        let template_str = tmpl
            .to_str()
            .context("model chat template is not valid UTF-8")?;
        if let Some(result) = render_ported_tool_template_result(
            template_str,
            oai_messages_json,
            tools_json,
            tool_choice,
            chat_template_kwargs,
            true,
            enable_thinking,
            parallel_tool_calls,
        )? {
            return Ok(result);
        }
        bail!("unsupported chat template for Rust tool calling")
    }

    pub(in crate::model) fn fallback_format_messages(raw_messages: &[(String, String)]) -> String {
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
        let Some(start) = output.find(crate::oai_chat::THINK_OPEN) else {
            return (output.to_string(), None);
        };
        let after_open = start + crate::oai_chat::THINK_OPEN.len();
        if let Some(rel_end) = output[after_open..].find(crate::oai_chat::THINK_CLOSE) {
            let end = after_open + rel_end;
            let reasoning = output[after_open..end].trim().to_string();
            let mut text = String::with_capacity(output.len());
            text.push_str(&output[..start]);
            text.push_str(&output[end + crate::oai_chat::THINK_CLOSE.len()..]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_support::*;
    use jobworkerp_llama_protobuf::protobuf::llm::llm_chat_args;

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

    /// `apply_oai_template_with_tools` smoke test: the rendered prompt embeds
    /// the tool definitions through the Rust renderer for the real Qwen tagged
    /// template loaded from the model file.
    #[ignore = "depends on model: apply_oai_template_with_tools minimal"]
    #[test]
    fn test_apply_oai_template_with_tools_minimal() {
        let wrapper = load_wrapper_from_tool_env(QWEN35_TOOL_GOLDEN_ENV);
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
        let wrapper = load_wrapper_from_tool_env(QWEN35_TOOL_GOLDEN_ENV);
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
        let wrapper = load_wrapper_from_tool_env(QWEN35_TOOL_GOLDEN_ENV);
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

    /// Confirm the Rust tool template produces a tools-aware prompt and Rust
    /// parse-back recovers the tool call from generated text.
    #[ignore = "depends on model: tool calling smoke test"]
    #[test]
    fn poc_rust_template_and_parse_tool_call() {
        let mut wrapper = load_wrapper_for_tool_poc();

        let messages_json = r#"[
            {"role":"system","content":"You are a tool caller."},
            {"role":"user","content":"What is the weather in Tokyo? Use get_weather."}
        ]"#;
        let opts = llm_chat_args::FunctionOptions {
            tool_choice: Some("auto".to_string()),
            chat_template_kwargs: Some(r#"{"enable_thinking":false}"#.to_string()),
            ..Default::default()
        };
        let result = wrapper
            .apply_oai_template_with_tools(messages_json, poc_tools_json(), Some(&opts), None)
            .expect("apply_oai_template_with_tools");

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

        let parsed_json =
            parse_tool_response_with_recovery(&result, poc_tools_json(), &output.text)
                .expect("Rust parse-back");
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

        let messages_json = r#"[
            {"role":"system","content":"You are a tool caller."},
            {"role":"user","content":"What is the weather in Tokyo? Use get_weather."}
        ]"#;
        let opts = llm_chat_args::FunctionOptions {
            tool_choice: Some("auto".to_string()),
            chat_template_kwargs: Some(r#"{"enable_thinking":false}"#.to_string()),
            ..Default::default()
        };
        let result = wrapper
            .apply_oai_template_with_tools(messages_json, poc_tools_json(), Some(&opts), None)
            .expect("apply_oai_template_with_tools");

        // Drive a real generation; the sink receives the same UTF-8 decoded
        // chunks (`&str` per token piece) that production streaming would.
        let mut state = crate::oai_chat::OaiStreamParser::for_template(&result, poc_tools_json())
            .expect("Rust streaming parser");
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
        let wrapper = load_wrapper_for_tool_poc();

        // Ambiguous prompt: a free-running model would just greet back.
        let messages_json = r#"[
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"Hello!"}
        ]"#;
        // Forcing tool_choice to "required" + the grammar should together
        // guarantee a tool-call emission regardless of the user prompt.
        let opts = llm_chat_args::FunctionOptions {
            tool_choice: Some("required".to_string()),
            chat_template_kwargs: Some(r#"{"enable_thinking":false}"#.to_string()),
            ..Default::default()
        };
        let result = wrapper
            .apply_oai_template_with_tools(messages_json, poc_tools_json(), Some(&opts), None)
            .expect("apply_oai_template_with_tools");

        eprintln!("grammar triggers ({}):", result.grammar_triggers.len());
        for (i, t) in result.grammar_triggers.iter().enumerate() {
            eprintln!(
                "  [{i}] type={:?} value={:?} token={:?}",
                t.trigger_type, t.value, t.token
            );
        }
        let grammar = result
            .grammar
            .as_deref()
            .expect("required tools yield grammar");

        let (trigger_patterns, trigger_tokens) =
            crate::oai_chat::grammar_triggers_to_patterns_and_tokens(
                wrapper.model,
                &result.grammar_triggers,
            );

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

    // Phase 1 end-to-end PoC (see `ai-docs/llama-cpp-rs-upgrade-plan.md`):
    // proves the plugin-local Rust grammar builder
    // (`oai_chat::build_qwen_tool_call_grammar_spec`) — not the fork-only OAI
    // template grammar — actually forces Qwen3.5 to emit its tagged tool-call
    // syntax during a real decode. The grammar is wired in through the normal
    // `InferenceArgs.grammar_spec` slot consumed by `build_sampler`, so no
    // production code change is needed to exercise it. Only the OAI template's
    // `prompt` (with tool definitions injected) is reused; its grammar is
    // discarded and replaced by the Rust builder's output.
    //
    // With tool_choice=auto the builder emits `grammar_lazy=true` (a
    // `<tool_call>\n` trigger arms the constraint only after the model decides
    // to emit it). Whether the model *self-triggers* is a prompt/model-
    // conditioning question, not a grammar-correctness one, and is deferred to
    // Phase 2 (when the Rust path becomes primary and owns prompt shaping). To
    // isolate *grammar enforcement* here, the PoC builds the grammar with
    // `require_tools=true`, which makes the builder emit an eager grammar (no
    // lazy trigger): the decode must then follow `root` from the very first
    // token, which is exactly the property Phase 1 needs to confirm.
    #[ignore = "depends on model: Qwen3.5-4B Rust-grammar tool-call enforcement"]
    #[test]
    fn poc_qwen35_rust_grammar_eager_forces_tool_call_emission() {
        let mut wrapper = load_wrapper_from_tool_env(QWEN35_TOOL_GOLDEN_ENV);

        let messages_json = r#"[
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"What is the weather in Tokyo?"}
        ]"#;
        let opts = llm_chat_args::FunctionOptions {
            tool_choice: Some("required".to_string()),
            chat_template_kwargs: Some(r#"{"enable_thinking":false}"#.to_string()),
            ..Default::default()
        };
        let template_result = wrapper
            .apply_oai_template_with_tools(messages_json, poc_tools_json(), Some(&opts), None)
            .expect("apply_oai_template_with_tools");

        // The load-bearing substitution: discard the OAI template grammar and
        // drive the decode with the Rust builder's grammar instead. `require_tools`
        // makes the builder emit an eager grammar (no lazy trigger), so the
        // constraint applies from `root` immediately — see the doc comment above
        // for why lazy self-triggering is out of scope here.
        let spec =
            crate::oai_chat::build_qwen_tool_call_grammar_spec(poc_tools_json(), true, false)
                .expect("build_qwen_tool_call_grammar_spec");
        eprintln!("rust grammar:\n{}", spec.grammar);

        let mut args = poc_inference_args(&template_result.prompt, Some(0.7));
        args.grammar_spec = Some(spec);
        let output = wrapper
            .decode_text_only_core(&template_result.prompt, &args)
            .expect("decode_text_only_core");
        eprintln!("decoded text:\n{}", output.text);

        // Qwen3.5 uses the tagged <function=...>/<parameter=...> format, not a
        // JSON envelope. Asserting these substrings confirms the Rust grammar
        // forced that exact shape end-to-end.
        assert!(
            output.text.contains("<tool_call>"),
            "expected <tool_call> opener in: {}",
            output.text
        );
        assert!(
            output.text.contains("<function=get_weather"),
            "expected <function=get_weather in: {}",
            output.text
        );
        assert!(
            output.text.contains("<parameter=city"),
            "expected <parameter=city in: {}",
            output.text
        );
    }
}
