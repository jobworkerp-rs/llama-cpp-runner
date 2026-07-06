//! The low-level decode engine: the text-only and multimodal generation loops
//! that drive `LlamaContext`/`LlamaBatch`/`LlamaSampler`.
//!
//! Split out of `model.rs` as a second `impl LlamaModelWrapper` block. These
//! methods are the most self-contained part of the wrapper (the actual token
//! loop, KV-cache prefix reuse, and streaming/cancellation handling) and are
//! kept together so the high-level entry points in `wrapper` stay readable.
//! Behavior is unchanged from the pre-split single-file layout — this is a pure
//! move; method bodies are byte-for-byte identical.

use super::*;

impl LlamaModelWrapper {
    pub(in crate::model) fn decode(&mut self, args: InferenceArgs) -> Result<DecodeOutput> {
        match (args.medias.is_empty(), self.mtmd.is_some()) {
            (true, _) => self.decode_text_only(args),
            (false, false) => bail!("multimodal input given but mmproj is not configured"),
            (false, true) => self.decode_multimodal(args),
        }
    }

    /// Resolve the effective KV-prefix-reuse policy for this request, warning
    /// once when an explicit runner value overrides a conflicting request value.
    /// Keeps the warn wording single-sourced across both decode paths.
    fn resolve_reuse_kv_prefix(&self, args: &InferenceArgs) -> bool {
        let resolved = resolve_reuse_kv_prefix(self.runner_reuse_kv_prefix, args.reuse_kv_prefix);
        if resolved.conflict {
            tracing::warn!(
                runner = self.runner_reuse_kv_prefix,
                request = args.reuse_kv_prefix,
                "request reuse_kv_prefix ignored because runner settings are explicit"
            );
        }
        resolved.value
    }

    /// Legacy entry point: wraps the user prompt in a system+user chat template
    /// (or falls back to plain concatenation) before delegating to the core
    /// generation loop. Used by the `run` (LlamaArg) path.
    pub(in crate::model) fn decode_text_only(
        &mut self,
        args: InferenceArgs,
    ) -> Result<DecodeOutput> {
        let formatted = self.format_single_turn(&args.prompt)?;
        self.decode_text_only_core(&formatted, &args)
    }

    /// Core text-only generation loop. Receives a `formatted_prompt` that the
    /// caller is responsible for templating; no chat template or system/user
    /// wrapping happens here. This keeps `run_chat` / `run_completion` from
    /// applying the template twice.
    pub(in crate::model) fn decode_text_only_core(
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
    pub(in crate::model) fn decode_text_only_core_with_sink(
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
    pub(in crate::model) fn decode_text_only_core_with_sink_and_stops(
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
        let reuse_kv_prefix = self.resolve_reuse_kv_prefix(args);
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
                // the stop marker (the tool parser expects the assistant
                // turn to end at the stop). The sink already saw the stop
                // bytes; streaming consumers receive the same raw chunks as
                // before. Break the loop without flagging
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
    pub(in crate::model) fn decode_multimodal(
        &mut self,
        args: InferenceArgs,
    ) -> Result<DecodeOutput> {
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
    pub(in crate::model) fn decode_multimodal_core_with_sink(
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

        let reuse_kv_prefix = self.resolve_reuse_kv_prefix(args);
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

            let output_bytes = token_to_piece_bytes_retry_special(model, token, true)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_support::*;
    use jobworkerp_llama_protobuf::protobuf::llama_cpp::MediaInput;

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
    fn test_text_decode_loop_polls_cancel_before_sampling() {
        let source = include_str!("decode.rs");

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
            reuse_kv_prefix: Some(true),
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

    /// Request-level `reuse_kv_prefix=true` must behave like runner-level reuse
    /// when the runner setting is omitted.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_request_reuse_kv_prefix_matches_full_clear \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_request_reuse_kv_prefix_matches_full_clear() {
        fn args(prompt: &str, reuse_kv_prefix: Option<bool>) -> InferenceArgs {
            InferenceArgs {
                prompt: prompt.to_string(),
                sample_len: Some(48),
                seed: Some(1234),
                reuse_kv_prefix,
                ..Default::default()
            }
        }

        let shared = "You are a helpful assistant. The user asks: ";
        let prompt_a = format!("{shared}what is the capital of France?");
        let prompt_b = format!("{shared}what is two plus two?");

        let baseline_b = LlamaModelWrapper::new(LlamaModelConfig::default())
            .expect("load")
            .run(args(&prompt_b, None))
            .expect("baseline b");

        let mut wrapper = LlamaModelWrapper::new(LlamaModelConfig::default()).expect("load");
        wrapper
            .run(args(&prompt_a, Some(true)))
            .expect("request reuse a");
        let reuse_b = wrapper
            .run(args(&prompt_b, Some(true)))
            .expect("request reuse b");
        assert_eq!(
            reuse_b, baseline_b,
            "request-level prefix reuse changed prompt B output"
        );
    }

    /// Alternating request-level reuse must never reuse a stale cache record.
    ///
    /// ```bash
    /// cargo test -p jobworkerp-llama-cpp-plugin test_request_reuse_kv_prefix_alternates_safely \
    ///     --release -- --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore]
    fn test_request_reuse_kv_prefix_alternates_safely() {
        fn args(prompt: &str, reuse_kv_prefix: Option<bool>) -> InferenceArgs {
            InferenceArgs {
                prompt: prompt.to_string(),
                sample_len: Some(48),
                seed: Some(1234),
                reuse_kv_prefix,
                ..Default::default()
            }
        }

        let shared = "You are a helpful assistant. The user asks: ";
        let prompt_a = format!("{shared}what is the capital of France?");
        let prompt_b = format!("{shared}what is two plus two?");
        let prompt_c = format!("{shared}name one primary color.");

        let baseline_b = LlamaModelWrapper::new(LlamaModelConfig::default())
            .expect("load")
            .run(args(&prompt_b, None))
            .expect("baseline b");
        let baseline_c = LlamaModelWrapper::new(LlamaModelConfig::default())
            .expect("load")
            .run(args(&prompt_c, None))
            .expect("baseline c");

        let mut wrapper = LlamaModelWrapper::new(LlamaModelConfig::default()).expect("load");
        wrapper.run(args(&prompt_a, Some(true))).expect("reuse a");
        let no_reuse_b = wrapper
            .run(args(&prompt_b, Some(false)))
            .expect("no reuse b");
        let reuse_c = wrapper.run(args(&prompt_c, Some(true))).expect("reuse c");

        assert_eq!(
            no_reuse_b, baseline_b,
            "reuse=false request must full-clear"
        );
        assert_eq!(
            reuse_c, baseline_c,
            "reuse=true after reuse=false must not use stale cache"
        );
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
            reuse_kv_prefix: Some(false),
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
            reuse_kv_prefix: Some(true),
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
            reuse_kv_prefix: Some(true),
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
            reuse_kv_prefix: Some(true),
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
            reuse_kv_prefix: Some(false),
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
            max_new_tokens: Some(1024),
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
}
