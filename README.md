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

CUDA-enabled release build:

```bash
cargo build --release --features cuda
```

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
- **CUDA build with NCCL**: When `libnccl` is installed on the build host, `llama-cpp-sys-2` compiles NCCL support into ggml-cuda but does not emit the corresponding linker directive. Combined with a Cargo limitation ([rust-lang/cargo#7506](https://github.com/rust-lang/cargo/issues/7506)) where `build.rs` link directives are not propagated to `bin` targets, binary crates will fail to link. As a workaround, add `-C link-arg=-lnccl` to `RUSTFLAGS` when building with CUDA:
  ```bash
  RUSTFLAGS="-C relocation-model=pic -C link-arg=-lnccl" cargo build --release --features cuda
  ```
  This workaround can be removed once `llama-cpp-sys-2` fixes its `build.rs` to emit `cargo:rustc-link-lib=nccl`, or once Cargo #7506 is resolved.

## License

MIT License
