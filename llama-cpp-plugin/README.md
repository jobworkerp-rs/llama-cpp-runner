# llama-cpp-plugin

[日本語版はこちら](README.ja.md)

`llama-cpp-plugin` is a JobworkerP multi-method plugin runner backed by `llama.cpp`.

## Overview

This crate loads GGUF models through `llama.cpp` and exposes three execution methods through the JobworkerP plugin interface:

- `run`: prompt-based text generation using the crate-specific `llama_cpp.LlamaArg`
- `chat`: conversation-based generation using `jobworkerp.runner.llm.LLMChatArgs`
- `completion`: single-turn text completion using `jobworkerp.runner.llm.LLMCompletionArgs`

The protobuf definitions used by this crate live in `llama-protobuf/`, and shared support code lives in sibling crates such as `mtmd-support/` and `modules/jobworkerp-client/`.

## Features

- Loads local GGUF files or downloads model files from Hugging Face
- Supports text generation with sampling controls
- Supports chat-style requests with multi-turn messages
- Supports single-prompt completion requests compatible with the jobworkerp Ollama/GenAI completion API
- Supports tool calling via the chat method
- Supports structured output through JSON Schema constraints
- Supports multimodal inputs when `MtmdSettings` and media inputs are provided
- Exposes plugin metadata and JSON schema for JobworkerP integration

## Methods

### `run`

The `run` method accepts `llama_cpp.LlamaArg`, which is this plugin's own protobuf argument format for prompt-based generation, and returns the same message type with the generated text written back into `prompt`. This method is useful for simple prompt-in / text-out workflows.

Relevant protobuf:

- settings: `llama-protobuf/protobuf/llama_cpp/llama_cpp_runner.proto`
- args/result: `llama-protobuf/protobuf/llama_cpp/llama_cpp_arg.proto`

### `chat`

The `chat` method accepts `jobworkerp.runner.llm.LLMChatArgs` and returns `jobworkerp.runner.llm.LLMChatResult`. This method supports:

- multi-turn chat messages
- system, user, assistant, and tool roles
- tool call generation and execution handoff
- structured output with `json_schema`
- multimodal message content where supported by the loaded model

Relevant protobuf:

- args: `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_args.proto`
- result: `llama-protobuf/protobuf/jobworkerp/runner/llm/chat_result.proto`

### `completion`

The `completion` method accepts `jobworkerp.runner.llm.LLMCompletionArgs` and returns `jobworkerp.runner.llm.LLMCompletionResult`. It is intended for the single-turn "continue the prompt" use case and is wire-compatible with the jobworkerp Ollama/GenAI completion API.

This method is text-only by design; multimodal inputs must use `chat`.

Differences from the jobworkerp standard completion runner (Ollama / GenAI):

| Field | jobworkerp standard | this plugin |
|---|---|---|
| `output_type` | `Both` (streaming + non-streaming) | `NonStreaming` |
| `context.ollama_context` | persisted as KV cache | dropped with a warn (no KV cache reuse — use `chat` for multi-turn) |
| `model` field | switchable per request | fixed at load time; warn and ignored |
| `function_options.use_function_calling = true` | supported | rejected with an error |
| multimodal input | text-only on both sides; use chat for media | text-only |

Relevant protobuf:

- args: `llama-protobuf/protobuf/jobworkerp/runner/llm/completion_args.proto`
- result: `llama-protobuf/protobuf/jobworkerp/runner/llm/completion_result.proto`

## Cancellation

For streaming `chat` / `completion`, the JobworkerP cancel token is converted into a shared cancel flag and observed by both the output sink and the `llama.cpp` abort callback. The sink stops at token boundaries; the abort callback can interrupt long prompt prefill and in-flight `llama_decode` calls at the granularity supported by the active backend. If cancellation is observed, the plugin exits as cancellation without writing back partial KV-cache metadata.

## Runner Settings

The plugin is initialized with `llama_cpp.LlamaRunnerSettings`.

Key fields:

- `model`: local model path or comma-separated model file names
- `hf_repo`: optional Hugging Face repository used to resolve `model`
- `disable_gpu`: disables GPU offloading
- `threads`: generation thread count
- `threads_batch`: prompt and batch processing thread count
- `ctx_size`: context window size
- `n_batch`: logical batch size for prompt processing. Affects prompt evaluation (time-to-first-token), not per-token generation speed. **When omitted it is set to the effective context length** (`ctx_size`, or the model's trained context when unset) so the whole prompt fits in one decode.
- `n_ubatch`: physical micro-batch size. Setting it lower than `n_batch` reduces peak memory during prompt eval. The default is backend-dependent: **Metal / ROCm builds** follow the effective `n_batch`, capped at 2048 to keep the compute buffer bounded (e.g. n_ubatch stays 2048 even when n_batch is 262144); other backends keep llama.cpp's default (512). An explicit value always wins.
- `type_k`: KV cache data type for K. Defaults to the llama.cpp default (F16) when omitted. Quantizing (e.g. `KV_CACHE_TYPE_Q8_0`) reduces KV cache memory for long contexts.
- `type_v`: KV cache data type for V. Defaults to the llama.cpp default (F16) when omitted. **V-cache quantization typically requires flash attention** (a warning is logged if `use_flash_attention` is disabled).
- `reuse_kv_prefix`: keep the KV of the longest common prefix (shared system prompt / document, or the same image) across requests and prefill only the differing suffix. Text is matched per token; images are matched by a content hash, so the same image followed by a different question reuses the image's KV (skipping its encode/decode). Cuts time-to-first-token for workloads that update a shared context across consecutive requests. Defaults to false, which clears the KV cache before each request so requests stay fully independent and deterministic.
- `mtp`: MTP speculative decoding settings. This is a runner-level setting because the draft context is created with the model context. Omit it or set `enabled=false` to keep the existing decode path. `n_max` defaults to 3, `n_min` to 0, and `p_min` to 0.0. `draft_model` selects a separate MTP draft GGUF; when `draft_hf_repo` is omitted it uses the runner `hf_repo`. `draft_type_k` / `draft_type_v` default to `type_k` / `type_v` when omitted.
- `use_flash_attention`: enables flash attention when supported
- `system_prompt`: default system prompt applied by the runner
- `mtmd`: multimodal projector settings

If `hf_repo` is omitted, `model` is treated as a local path. If `hf_repo` is set, the plugin downloads or reuses cached model files from Hugging Face. Existing snapshot cache entries are reused without a metadata network round-trip, so offline warm starts return immediately. On cache misses, partial `.part` downloads are left resumable and size-mismatched download results are rejected.

### MTP speculative decoding

MTP is currently supported for text-only generation. Requests that include media are rejected with `MTP speculative decoding is not supported with multimodal input yet`; disable `mtp` for multimodal runners.

Example Gemma 4 runner settings:

```json
{
  "hfRepo": "unsloth/gemma-4-12B-it-qat-GGUF",
  "model": "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
  "useFlashAttention": true,
  "mtp": {
    "enabled": true,
    "nMax": 4,
    "draftModel": "mtp-gemma-4-12B-it.gguf"
  }
}
```

Example Qwen 3.6 MTP runner settings:

```json
{
  "hfRepo": "unsloth/Qwen3.6-27B-MTP-GGUF",
  "model": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
  "useFlashAttention": true,
  "mtp": {
    "enabled": true,
    "nMax": 4
  }
}
```

Operational procedure:

1. Load the target and draft GGUF files from the same repo. For Gemma 4, use `unsloth/gemma-4-12B-it-qat-GGUF` with target `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` and draft `mtp-gemma-4-12B-it.gguf`.
2. For same-file MTP models such as `unsloth/Qwen3.6-27B-MTP-GGUF`, load `Qwen3.6-27B-UD-Q4_K_XL.gguf` and omit `mtp.draftModel`.
3. Enable flash attention unless you have a backend-specific reason not to.
4. Set `mtp.enabled=true`; set `mtp.draftModel` only when the model repo provides a separate MTP draft GGUF. Use `nMax=4` for smoke tests, or omit it to use the default `3`.
5. Run text-only `chat` / `completion` / `run` requests and compare output quality against the same runner with `mtp.enabled=false`.
6. Keep multimodal workloads on a separate runner with `mtp` omitted until multimodal+MTP is validated.

## Diagnosing Inference Speed

If generation speed (token/sec) is slower than expected (e.g. on macOS/Metal), measure first to isolate the cause. When running through the plugin (`run` / `chat` / `completion`) with `RUST_LOG=info`, the following logs are emitted:

- `context created in N s (n_batch=..., n_ubatch=...)`: per-request `LlamaContext` creation (KV cache allocation). On Metal this cost is incurred on every request.
- `decoded N tokens in N s, speed N t/s`: actual token generation speed.
- `ctx.timings()`: llama.cpp's own breakdown of prompt eval vs. generation.

Interpretation:

- **Context creation dominates**: a fresh `LlamaContext` is created per request; pronounced for short prompt/short output workloads.
- **Low eval token/sec**: verify Metal is actually used — build with `--features metal` and ensure `disable_gpu` is `false`.
- **Slow prompt eval (TTFT)**: `n_batch` / `n_ubatch` may help, but the default (n_batch=2048) is already large, so the effect is limited.

For comparison, run the same model and prompt with `ollama run --verbose` and compare its `eval rate` against the plugin's `speed N t/s`.

> Note: `cargo run --bin sample` uses its own context setup and decode loop, separate from the plugin path, so the plugin logs above are not emitted there. Measure via JobworkerP to observe plugin behavior.

### Tuning for very long contexts (tens of thousands of tokens)

When handling huge contexts (e.g. ctx_size=200k with tens of thousands of input tokens) on Apple Silicon unified memory, the bottleneck shifts to prompt eval (processing the large input) and KV cache memory bandwidth. The following help:

- **Raise `n_ubatch` (e.g. 2048)**: large inputs are processed in `n_ubatch`-sized chunks; a larger `n_ubatch` means fewer chunks and faster prefill. Apple recommends roughly `-ub 2048` for large-prompt processing. **Metal / ROCm builds auto-follow the effective `n_batch`, capped at 2048**, so you already get 2048 without setting anything; set `n_ubatch` explicitly to go higher (the cap is bypassed for explicit values). The cost is a larger compute buffer, acceptable with ample unified memory.
- **Quantize the KV cache (`type_k` / `type_v` = `KV_CACHE_TYPE_Q8_0`)**: the KV cache grows large for long contexts; F16 → Q8_0 halves its footprint and reduces memory bandwidth. `type_v` quantization requires flash attention (enabled by default here).
- **Enable `reuse_kv_prefix=true` to reuse the shared prefix**: when consecutive requests share a system prompt / document and differ only in the tail, the shared KV is kept and only the suffix is prefilled. This nearly eliminates prompt eval (re-evaluating tens of thousands of tokens), dramatically cutting TTFT — the largest win for this workload. Leave it false (default) if requests must stay fully independent.
- **Caveat: bigger `n_ubatch` is not always faster** — the optimal value depends on GPU cache behavior and can collapse if set too high. Sweep 512 / 1024 / 2048 / 4096 with `llama-bench` on your hardware/model. Note `n_ubatch` ≤ `n_batch`.

## Build

```bash
cargo build -p jobworkerp-llama-cpp-plugin
```

Release build:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin
```

CUDA build:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin --features cuda
```

Metal build:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin --features metal
```

ROCm (AMD GPU) build:

```bash
cargo build --release -p jobworkerp-llama-cpp-plugin --features rocm
```

## Test And Lint

```bash
cargo test -p jobworkerp-llama-cpp-plugin -- --test-threads=1
cargo fmt --all --check
cargo clippy -p jobworkerp-llama-cpp-plugin --all-targets
```

If your change affects shared protobuf contracts or shared support crates, run the workspace-level checks as well.

## CLI Sample

This crate also includes a local sample binary:

```bash
cargo run -p jobworkerp-llama-cpp-plugin --bin sample -- \
  hf-model TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf \
  --prompt "Hello"
```

The sample binary is for local experimentation with `llama-cpp-2`. It is separate from the plugin entrypoint that JobworkerP loads.

## Notes

- The exported plugin name is `LLMPromptRunner`. Renaming it can break systems that already refer to this plugin by name.
- Both methods are currently exposed as non-streaming APIs.
- Media inputs are passed through `llama_cpp.MediaInput`, and prompt markers must align with the attached media entries.
- When tool calling is used in manual mode, the chat result can contain pending tool calls instead of a final assistant response.

## Related Files

- plugin entrypoint: `llama-cpp-plugin/src/lib.rs`
- model runtime: `llama-cpp-plugin/src/model.rs`
- shared protobufs: `llama-protobuf/`

## License

MIT License.
