use super::*;

const MAX_TEXT_COUNT: usize = 256;
const MAX_TEXT_UTF8_BYTES: usize = 1_000_000;
const MAX_TOTAL_TEXT_UTF8_BYTES: usize = 4_000_000;
const DEFAULT_MAX_COMPLETION_TOKENS: u64 = 4096;

impl LlamaModelWrapper {
    /// Count an input using the loaded model without creating a decode context.
    pub fn token_count(&self, args: TokenCountArgs) -> Result<TokenCountResult> {
        match args.target {
            None => Ok(Self::invalid_token_count("token count target is required")),
            Some(token_count_args::Target::Text(request)) => self.count_texts(request.texts),
            Some(token_count_args::Target::Completion(args)) => self.count_completion(args),
            Some(token_count_args::Target::Chat(request)) => match request.request {
                Some(args) => self.count_chat(args),
                None => Ok(Self::invalid_token_count(
                    "chat token count request is required",
                )),
            },
        }
    }

    fn invalid_token_count(detail: &str) -> TokenCountResult {
        TokenCountResult {
            outcome: Some(token_count_result::Outcome::Error(
                token_count_result::Error {
                    code: token_count_result::TokenCountErrorCode::InvalidTokenCountArgument as i32,
                    detail: Some(detail.to_string()),
                },
            )),
        }
    }

    fn count_texts(&self, texts: Vec<String>) -> Result<TokenCountResult> {
        if texts.len() > MAX_TEXT_COUNT {
            return Ok(Self::invalid_token_count("too many text inputs"));
        }
        let mut total_bytes = 0_usize;
        for text in &texts {
            if text.len() > MAX_TEXT_UTF8_BYTES {
                return Ok(Self::invalid_token_count("text input exceeds byte limit"));
            }
            total_bytes = match total_bytes.checked_add(text.len()) {
                Some(total) if total <= MAX_TOTAL_TEXT_UTF8_BYTES => total,
                _ => {
                    return Ok(Self::invalid_token_count(
                        "total text input exceeds byte limit",
                    ));
                }
            };
        }

        let mut counts = Vec::with_capacity(texts.len());
        let mut total_tokens = 0_u64;
        for text in texts {
            let count = u64::try_from(
                self.model
                    .str_to_token_with_special(&text, AddBos::Never, false)
                    .map_err(|error| anyhow!("failed to tokenize text: {error}"))?
                    .len(),
            )
            .map_err(|_| anyhow!("token count does not fit in u64"))?;
            total_tokens = total_tokens
                .checked_add(count)
                .ok_or_else(|| anyhow!("total token count overflow"))?;
            counts.push(count);
        }

        Ok(TokenCountResult {
            outcome: Some(token_count_result::Outcome::TextSuccess(
                token_count_result::TextSuccess {
                    resolved_model: self.resolved_model.clone(),
                    tokenizer_id: None,
                    counts,
                    total_tokens,
                },
            )),
        })
    }

    fn count_completion(&self, args: LlmCompletionArgs) -> Result<TokenCountResult> {
        let max_completion_tokens = match Self::max_completion_tokens(
            args.options.as_ref().and_then(|options| options.max_tokens),
        ) {
            Ok(value) => value,
            Err(result) => return Ok(result),
        };
        validate_completion_args(&args)?;
        let system_prompt = args
            .system_prompt
            .as_deref()
            .unwrap_or(self.system_prompt.as_str());
        let rendered = self.format_single_turn_inner(system_prompt, &args.prompt)?;
        self.count_rendered_text(&rendered, max_completion_tokens)
    }

    fn count_chat(&self, args: LlmChatArgs) -> Result<TokenCountResult> {
        let max_completion_tokens = match Self::max_completion_tokens(
            args.options.as_ref().and_then(|options| options.max_tokens),
        ) {
            Ok(value) => value,
            Err(result) => return Ok(result),
        };
        if args
            .function_options
            .as_ref()
            .is_some_and(|options| options.use_function_calling)
        {
            bail!(ERR_USE_FUNCTION_CALLING_UNSUPPORTED);
        }
        if let Some(tools_json) = args
            .function_options
            .as_ref()
            .and_then(|options| options.client_tools_json.as_deref())
        {
            return self.count_tools_chat(&args, tools_json, max_completion_tokens);
        }

        let (messages, raw_messages, medias) = self.build_chat_messages(&args.messages)?;
        let rendered = self.apply_chat_template_multi(&messages, &raw_messages)?;
        if medias.is_empty() {
            return self.count_rendered_text(&rendered, max_completion_tokens);
        }

        let mtmd = self
            .mtmd
            .as_ref()
            .context("multimodal input given but mmproj is not configured")?;
        let bitmaps = mtmd
            .prepare_bitmaps(&medias, &self.media_limits)
            .map_err(|error| anyhow!(error).context("mtmd: preparing bitmaps"))?;
        let chunks = mtmd
            .tokenize(&rendered, &bitmaps)
            .map_err(|error| anyhow!(error).context("mtmd: tokenize"))?;
        let token_count = u64::try_from(chunks.total_tokens())
            .map_err(|_| anyhow!("multimodal token count does not fit in u64"))?;
        let context_window_tokens = (chunks.total_positions() >= 0
            && u64::try_from(chunks.total_positions()).ok() == Some(token_count))
        .then_some(self.context_window_tokens);
        Ok(self.rendered_success(token_count, max_completion_tokens, context_window_tokens))
    }

    fn count_tools_chat(
        &self,
        args: &LlmChatArgs,
        tools_json: &str,
        max_completion_tokens: u64,
    ) -> Result<TokenCountResult> {
        let prepared = self.prepare_tools_chat_template(args, tools_json)?;
        self.count_rendered_text(&prepared.template.prompt, max_completion_tokens)
    }

    fn max_completion_tokens(
        max_tokens: Option<i32>,
    ) -> std::result::Result<u64, TokenCountResult> {
        match max_tokens {
            Some(value) if value <= 0 => {
                Err(Self::invalid_token_count("max_tokens must be positive"))
            }
            Some(value) => Ok(value as u64),
            None => Ok(DEFAULT_MAX_COMPLETION_TOKENS),
        }
    }

    fn count_rendered_text(
        &self,
        rendered: &str,
        max_completion_tokens: u64,
    ) -> Result<TokenCountResult> {
        let token_count = u64::try_from(
            self.model
                .str_to_token_with_special(rendered, AddBos::Always, true)
                .map_err(|error| anyhow!("failed to tokenize rendered prompt: {error}"))?
                .len(),
        )
        .map_err(|_| anyhow!("rendered token count does not fit in u64"))?;
        Ok(self.rendered_success(
            token_count,
            max_completion_tokens,
            Some(self.context_window_tokens),
        ))
    }

    fn rendered_success(
        &self,
        rendered_input_tokens: u64,
        max_completion_tokens: u64,
        context_window_tokens: Option<u64>,
    ) -> TokenCountResult {
        TokenCountResult {
            outcome: Some(token_count_result::Outcome::RenderedRequestSuccess(
                token_count_result::RenderedRequestSuccess {
                    resolved_model: self.resolved_model.clone(),
                    tokenizer_id: None,
                    context_window_tokens,
                    max_completion_tokens: Some(max_completion_tokens),
                    rendered_input_tokens,
                },
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_tokens_uses_execution_default_and_rejects_non_positive_values() {
        assert_eq!(LlamaModelWrapper::max_completion_tokens(None), Ok(4096));
        assert_eq!(LlamaModelWrapper::max_completion_tokens(Some(12)), Ok(12));
        for value in [0, -1] {
            let result = LlamaModelWrapper::max_completion_tokens(Some(value))
                .expect_err("non-positive max_tokens must be invalid");
            assert!(matches!(
                result.outcome,
                Some(token_count_result::Outcome::Error(token_count_result::Error {
                    code,
                    ..
                })) if code == token_count_result::TokenCountErrorCode::InvalidTokenCountArgument as i32
            ));
        }
    }
}
