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
- `modules/llama-cpp-rs` is based on `llama-cpp-rs` `0.1.151`. `llama-cpp-plugin` exposes runner-level MTP speculative decoding settings through `llama_cpp.LlamaRunnerSettings.mtp`; see [llama-cpp-plugin/README.md](llama-cpp-plugin/README.md) for the text-only limitation and Gemma 4 validation procedure.
- **CUDA build / NCCL runtime dependency**: When `libnccl` and `nccl.h` are installed on the build host, `llama.cpp`'s CMake auto-detects NCCL via `find_package(NCCL)` and compiles NCCL support into ggml-cuda. The resulting `.so` then has `libnccl.so.2` as a load-time dependency, so `dlopen()` will fail at plugin load on runtime images that do not ship NCCL (for example `ghcr.io/jobworkerp-rs/grpc-front`).

  This project does not currently use NCCL (no multi-GPU collective operations), so we recommend hiding NCCL from CMake at build time with one of the following:

  1. Temporarily move `libnccl*` and `nccl.h` out of the way on the build host (this is what CI does):
     ```bash
     sudo find /usr /opt/cuda /usr/local/cuda \( -name 'nccl.h' -o -name 'libnccl*' \) \
       -exec mv {} {}.disabled-for-build \;
     cargo build --release --features cuda
     ```
  2. Or simply do not install NCCL on the build host.

  Note: `llama-cpp-sys-2` exposes no Cargo feature to disable NCCL and provides no way to forward an external CMake option (`-DGGML_CUDA_NCCL=OFF`) to the underlying build, so the approaches above are the practical workarounds today.

## License

MIT License
