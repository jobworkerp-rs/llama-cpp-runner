# llama-cpp-plugin

[日本語版はこちら](README.ja.md)

`llama-cpp-plugin` is a JobworkerP multi-method plugin runner backed by `llama.cpp`.

## Overview

This crate loads GGUF models through `llama.cpp` and exposes two execution methods through the JobworkerP plugin interface:

- `run`: prompt-based text generation using the crate-specific `llama_cpp.LlamaArg`
- `chat`: conversation-based generation using `jobworkerp.runner.llm.LLMChatArgs`

The protobuf definitions used by this crate live in `llama-protobuf/`, and shared support code lives in sibling crates such as `mtmd-support/` and `modules/jobworkerp-client/`.

## Features

- Loads local GGUF files or downloads model files from Hugging Face
- Supports text generation with sampling controls
- Supports chat-style requests with multi-turn messages
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

## Runner Settings

The plugin is initialized with `llama_cpp.LlamaRunnerSettings`.

Key fields:

- `model`: local model path or comma-separated model file names
- `hf_repo`: optional Hugging Face repository used to resolve `model`
- `disable_gpu`: disables GPU offloading
- `threads`: generation thread count
- `threads_batch`: prompt and batch processing thread count
- `ctx_size`: context window size
- `use_flash_attention`: enables flash attention when supported
- `system_prompt`: default system prompt applied by the runner
- `mtmd`: multimodal projector settings

If `hf_repo` is omitted, `model` is treated as a local path. If `hf_repo` is set, the plugin downloads or reuses cached model files from Hugging Face.

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
