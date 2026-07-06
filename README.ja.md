# llama-plugins

[English version](README.md)

`llama.cpp` をベースにした JobworkerP 向けプラグイン群と共有補助crateの Rust ワークスペースです。

## 概要

このリポジトリには、`llama.cpp` モデルを JobworkerP のワークフローから利用するためのプラグインランナー、共有 protobuf 定義、補助crateを含みます。Rust ワークスペース構成により、各crateが protobuf 型、クライアント実装、ユーティリティを重複なく再利用できるようにしています。

## ワークスペース構成

- `llama-cpp-plugin/`: `llama.cpp` を使ったテキスト生成用プラグインランナー
- `embedding-llm/`: `llama.cpp` ベースの embedding 生成プラグイン
- `reranker-runner/`: `llama.cpp` ベースの reranker プラグインランナー
- `llama-protobuf/`: llama 系プラグインで共有する protobuf 定義
- `mtmd-support/`: マルチモーダル projector 処理のための共有ランタイムコード
- `modules/jobworkerp-client/`: JobworkerP クライアントコードとプラグインフレームワークの型
- `modules/command-utils/`: 複数crateで使う共有ユーティリティ
- `gguf-reranker-validation/`: GGUF reranker の挙動を検証する実行用crate

## ビルド

```bash
cargo build --workspace
```

リリースビルド:

```bash
cargo build --release
```

CUDA 対応のリリースビルド:

```bash
cargo build --release --features cuda
```

## テストと静的解析

ワークスペース全体のテスト:

```bash
cargo test --workspace -- --test-threads=1
```

フォーマット確認:

```bash
cargo fmt --all --check
```

Clippy:

```bash
cargo clippy --workspace --all-targets
```

日常的な変更では、まず対象crateを個別に確認してからワークスペース全体を検証してください。

## crate別ドキュメント

- `embedding-llm`: [embedding-llm/README.md](embedding-llm/README.md)
- `llama-cpp-plugin`: [llama-cpp-plugin/README.md](llama-cpp-plugin/README.md)
- `embedding-llm` 利用ガイド: [embedding-llm/USER_GUIDE.md](embedding-llm/USER_GUIDE.md)

## 補足

- ハードウェアアクセラレーションは、対応crateで `cuda`、`metal`、`openmp` などの feature として提供しています。
- 共有処理はプラグインcrateへ複製せず、`modules/` または専用補助crateへ集約してください。
- protobuf 契約や公開プラグイン挙動を変更する場合は、関連するcrate単位のドキュメントも更新してください。
- `modules/llama-cpp-rs` は `llama-cpp-rs` `0.1.151` ベースです。`llama-cpp-plugin` は `llama_cpp.LlamaRunnerSettings.mtp` で runner-level の MTP speculative decoding 設定を公開しています。text-only 制限と Gemma 4 検証手順は [llama-cpp-plugin/README.ja.md](llama-cpp-plugin/README.ja.md) を参照してください。
- **CUDA ビルド時の NCCL 依存問題**: ビルドホストに `libnccl` と `nccl.h` がインストールされている場合、`llama.cpp` の CMake が `find_package(NCCL)` で自動検出し、ggml-cuda に NCCL サポートをコンパイルします。これにより生成された `.so` は `libnccl.so.2` をロード時依存として持つため、NCCL がインストールされていない実行環境（例: `ghcr.io/jobworkerp-rs/grpc-front` イメージ）で `dlopen()` が失敗します。

  本プロジェクトは現状 NCCL を使用しない（複数 GPU の集団通信を行わない）ため、以下のいずれかでビルド時に NCCL を検出させないことを推奨します:

  1. ビルドホストから `libnccl*` と `nccl.h` を一時的に隠す（CI で採用）:
     ```bash
     sudo find /usr /opt/cuda /usr/local/cuda \( -name 'nccl.h' -o -name 'libnccl*' \) \
       -exec mv {} {}.disabled-for-build \;
     cargo build --release --features cuda
     ```
  2. または、ビルドホストに NCCL をインストールしない。

  なお、`llama-cpp-sys-2` には NCCL を制御する Cargo feature がなく、外部から CMake オプション (`-DGGML_CUDA_NCCL=OFF`) を渡す手段も提供されていないため、上記の方法が現実的です。

## ライセンス

MIT License
