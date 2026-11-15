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

CUDA 対応のリリースビルド (後述の「CUDA ビルドの注意」を参照):

```bash
cargo build --release --features cuda
```

llama.cpp の CPU カーネルは rustc の target CPU に従い、既定の `x86_64` ターゲットでは AVX を使いません。CPU 推論を実用的な速度にするには CPU レベルを明示してビルドしてください。例: `RUSTFLAGS=-Ctarget-cpu=x86-64-v3` (配布バイナリと同じ水準)、ビルドしたマシン専用なら `-Ctarget-cpu=native`。

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
- `modules/llama-cpp-rs` は `llama-cpp-rs` `0.1.154` ベースです。`llama-cpp-plugin` は `llama_cpp.LlamaRunnerSettings.mtp` で runner-level の MTP speculative decoding 設定を公開しています。text-only 制限と Gemma 4 検証手順は [llama-cpp-plugin/README.ja.md](llama-cpp-plugin/README.ja.md) を参照してください。
- **CUDA ビルドの注意**: `llama-cpp-sys-2` は `CMAKE_*` / `GGML_*` 環境変数を llama.cpp の CMake ビルドへ転送し、ggml-cuda を常に NCCL 無し (`GGML_CUDA_NCCL=OFF`) でビルドするため、プラグインが `libnccl.so.2` に依存することはありません。CUDA のオブジェクトを共有ライブラリにリンクできるよう `CMAKE_POSITION_INDEPENDENT_CODE=ON` を指定し、全 GPU アーキテクチャ向けのビルドを避けるため `CMAKE_CUDA_ARCHITECTURES` (例: `89`) を指定してください:
  ```bash
  CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_CUDA_ARCHITECTURES=89 cargo build --release --features cuda
  ```
- ビルド済みのプラグインは動作要件とともに [GitHub Releases](https://github.com/jobworkerp-rs/llama-cpp-runner/releases) に添付しています。

## ライセンス

MIT License
