# llama-plugins

[日本語版はこちら](README.ja.md)

Rust workspace for JobworkerP plugins and shared helper crates built on top of `llama.cpp`.

## Overview

This repository contains plugin runners, shared protobuf definitions, and helper crates for using `llama.cpp` models from JobworkerP workflows. It is organized as a Rust workspace so each crate can reuse protobuf types, client code, and utility code without copying the same logic into multiple places.

## Workspace Layout

- `llama-cpp-plugin/`: text generation plugin runner for `llama.cpp`
- `embedding-llm/`: embedding generation plugin backed by `llama.cpp`
- `reranker-runner/`: reranker plugin runner backed by `llama.cpp`
- `llama-protobuf/`: protobuf definitions shared by llama-based plugins
- `mtmd-support/`: shared runtime code for multimodal projector handling
- `modules/jobworkerp-client/`: JobworkerP client code and plugin framework types
- `modules/command-utils/`: shared utility code used across crates
- `gguf-reranker-validation/`: executable crate for validating GGUF reranker behavior

## Build

```bash
cargo build --workspace
```

Release build:

```bash
cargo build --release
```

CUDA-enabled release build (see the CUDA build notes below):

```bash
cargo build --release --features cuda
```

llama.cpp's CPU kernels follow rustc's target CPU, and the default `x86_64` target has no AVX. Build with an explicit CPU level for usable CPU inference, for example `RUSTFLAGS=-Ctarget-cpu=x86-64-v3` (the level of the released binaries) or `-Ctarget-cpu=native` for the build machine only.

## Test And Lint

Run the full workspace test suite with serialized tests:

```bash
cargo test --workspace -- --test-threads=1
```

Check formatting:

```bash
cargo fmt --all --check
```

Run Clippy:

```bash
cargo clippy --workspace --all-targets
```

For iterative work, prefer validating the target crate first before running the full workspace checks.

## Crate-Specific Documentation

- `embedding-llm`: see [embedding-llm/README.md](embedding-llm/README.md)
- `llama-cpp-plugin`: see [llama-cpp-plugin/README.md](llama-cpp-plugin/README.md)
- `embedding-llm` user guide: see [embedding-llm/USER_GUIDE.md](embedding-llm/USER_GUIDE.md)

## Notes

- Hardware acceleration is exposed through crate features such as `cuda`, `metal`, and `openmp` where supported.
- Shared code should stay in `modules/` or dedicated helper crates instead of being copied into plugin crates.
- If you change protobuf contracts or public plugin behavior, update the relevant crate-level documentation as well.
- `modules/llama-cpp-rs` is based on `llama-cpp-rs` `0.1.154`. `llama-cpp-plugin` exposes runner-level MTP speculative decoding settings through `llama_cpp.LlamaRunnerSettings.mtp`; see [llama-cpp-plugin/README.md](llama-cpp-plugin/README.md) for the text-only limitation and Gemma 4 validation procedure.
- **CUDA build notes**: `llama-cpp-sys-2` forwards `CMAKE_*` and `GGML_*` environment variables to llama.cpp's CMake build, and always builds ggml-cuda without NCCL (`GGML_CUDA_NCCL=OFF`), so the plugins never depend on `libnccl.so.2`. Set `CMAKE_POSITION_INDEPENDENT_CODE=ON` (CUDA objects must be position independent to link into the plugin shared libraries) and, to avoid building every GPU architecture, `CMAKE_CUDA_ARCHITECTURES` (for example `89`):
  ```bash
  CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_CUDA_ARCHITECTURES=89 cargo build --release --features cuda
  ```
- Prebuilt plugins are attached to the [GitHub Releases](https://github.com/jobworkerp-rs/llama-cpp-runner/releases) with their runtime requirements.

## License

MIT License
