//! Stream-aware splitter for `<think>…</think>` reasoning blocks.
//!
//! Mirrors the semantics of `LlamaModelWrapper::extract_reasoning` but operates
//! incrementally on chunks delivered by the streaming sink — a `<think>` or
//! `</think>` tag may straddle chunk boundaries, so the splitter buffers
//! ambiguous bytes until the prefix either completes the tag or is proven not
//! to match it.

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Outside,
    InsideThink,
}

/// Splits a stream of generated text chunks into `(content, reasoning)` deltas.
///
/// When `extract` is `false`, the splitter is a pass-through: every chunk is
/// emitted verbatim as `content` and `reasoning` is always empty. When `true`,
/// the splitter holds back any byte sequence that could still complete the
/// next expected tag (`<think>` while outside, `</think>` while inside) and
/// flushes it as soon as ambiguity is resolved.
#[derive(Debug)]
pub struct ReasoningSplitter {
    state: State,
    /// Held-back tail that might still complete the next tag. Always shorter
    /// than the tag we are watching for.
    buffer: String,
    extract: bool,
}

impl ReasoningSplitter {
    pub fn new(extract: bool) -> Self {
        Self {
            state: State::Outside,
            buffer: String::new(),
            extract,
        }
    }

    /// Feed the next chunk of generated text. Returns
    /// `(content_delta, reasoning_delta)`; either may be empty when the
    /// splitter is still buffering an ambiguous tail. Bytes routed to
    /// reasoning are removed from content and vice versa — never duplicated.
    pub fn feed(&mut self, chunk: &str) -> (String, String) {
        if !self.extract {
            return (chunk.to_string(), String::new());
        }
        if chunk.is_empty() {
            return (String::new(), String::new());
        }

        let mut content = String::new();
        let mut reasoning = String::new();
        // Combine the held-back ambiguous tail with the new chunk; tags are
        // ASCII so byte indices coincide with char boundaries.
        let mut work = std::mem::take(&mut self.buffer);
        work.push_str(chunk);

        loop {
            match self.state {
                State::Outside => match work.find(OPEN_TAG) {
                    Some(idx) => {
                        content.push_str(&work[..idx]);
                        work.drain(..idx + OPEN_TAG.len());
                        self.state = State::InsideThink;
                    }
                    None => {
                        let keep_from = longest_ambiguous_suffix(&work, OPEN_TAG);
                        content.push_str(&work[..keep_from]);
                        self.buffer = work[keep_from..].to_string();
                        break;
                    }
                },
                State::InsideThink => match work.find(CLOSE_TAG) {
                    Some(idx) => {
                        reasoning.push_str(&work[..idx]);
                        work.drain(..idx + CLOSE_TAG.len());
                        self.state = State::Outside;
                    }
                    None => {
                        let keep_from = longest_ambiguous_suffix(&work, CLOSE_TAG);
                        reasoning.push_str(&work[..keep_from]);
                        self.buffer = work[keep_from..].to_string();
                        break;
                    }
                },
            }
        }

        (content, reasoning)
    }

    /// Flush any buffered tail at end-of-stream. Buffered bytes are emitted as
    /// `content` when outside `<think>` (an unfinished open tag is preserved
    /// verbatim because we never confirmed it was a tag) and as `reasoning`
    /// when inside (truncated reasoning, matching `extract_reasoning`'s
    /// open-tag-without-close semantics).
    pub fn flush(&mut self) -> (String, String) {
        if !self.extract {
            return (String::new(), String::new());
        }
        let leftover = std::mem::take(&mut self.buffer);
        match self.state {
            State::Outside => (leftover, String::new()),
            State::InsideThink => (String::new(), leftover),
        }
    }
}

/// Return the smallest index `i` such that `text[i..]` could still be the
/// start of `tag`. If no suffix is a prefix of `tag`, returns `text.len()`
/// (nothing to hold back). The candidate window is bounded by the tag length:
/// any longer trailing substring cannot grow into the tag without first
/// completing it, which would have been caught by `text.find(tag)`.
fn longest_ambiguous_suffix(text: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(text.len());
    for keep in (1..=max).rev() {
        let start = text.len() - keep;
        if text.is_char_boundary(start) && tag.starts_with(&text[start..]) {
            return start;
        }
    }
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(splitter: &mut ReasoningSplitter, chunks: &[&str]) -> (String, String) {
        let mut text = String::new();
        let mut reasoning = String::new();
        for chunk in chunks {
            let (t, r) = splitter.feed(chunk);
            text.push_str(&t);
            reasoning.push_str(&r);
        }
        let (t, r) = splitter.flush();
        text.push_str(&t);
        reasoning.push_str(&r);
        (text, reasoning)
    }

    #[test]
    fn passthrough_when_extract_disabled() {
        let mut s = ReasoningSplitter::new(false);
        let (text, reasoning) = drain(&mut s, &["<think>hidden</think>visible"]);
        assert_eq!(text, "<think>hidden</think>visible");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn single_chunk_extracts_block() {
        let mut s = ReasoningSplitter::new(true);
        let (text, reasoning) = drain(&mut s, &["<think>hello</think> world"]);
        assert_eq!(text, " world");
        assert_eq!(reasoning, "hello");
    }

    #[test]
    fn split_across_chunks_recombines_correctly() {
        let mut s = ReasoningSplitter::new(true);
        let (text, reasoning) = drain(&mut s, &["<th", "ink>he", "llo</thi", "nk> world"]);
        assert_eq!(text, " world");
        assert_eq!(reasoning, "hello");
    }

    #[test]
    fn no_think_block_passes_through() {
        let mut s = ReasoningSplitter::new(true);
        let (text, reasoning) = drain(&mut s, &["plain answer"]);
        assert_eq!(text, "plain answer");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn unclosed_think_routes_tail_to_reasoning() {
        // Matches LlamaModelWrapper::extract_reasoning: an open `<think>` with
        // no matching `</think>` is treated as truncated reasoning so callers
        // never receive a half-open tag in the content field.
        let mut s = ReasoningSplitter::new(true);
        let (text, reasoning) = drain(&mut s, &["<think>incomplete"]);
        assert_eq!(text, "");
        assert_eq!(reasoning, "incomplete");
    }

    #[test]
    fn lone_open_tag_text_is_preserved() {
        let mut s = ReasoningSplitter::new(true);
        let (text, reasoning) = drain(&mut s, &["a<b"]);
        assert_eq!(text, "a<b");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn buffer_holds_partial_tag_until_proven_safe() {
        let mut s = ReasoningSplitter::new(true);
        let (first_t, first_r) = s.feed("hello<th");
        assert_eq!(first_t, "hello");
        assert_eq!(first_r, "");
        let (second_t, second_r) = s.feed("ink>thoughts</think>!");
        assert_eq!(second_t, "!");
        assert_eq!(second_r, "thoughts");
    }

    #[test]
    fn multiple_blocks_in_sequence() {
        let mut s = ReasoningSplitter::new(true);
        let (text, reasoning) = drain(&mut s, &["a<think>x</think>b<think>y</think>c"]);
        assert_eq!(text, "abc");
        assert_eq!(reasoning, "xy");
    }
}
