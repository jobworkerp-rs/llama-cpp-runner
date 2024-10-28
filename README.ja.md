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
- **CUDA ビルド時の NCCL リンク問題**: ビルドホストに `libnccl` がインストールされている場合、`llama-cpp-sys-2` は ggml-cuda に NCCL サポートをコンパイルしますが、対応するリンカ指示を出力しません。さらに Cargo の制限（[rust-lang/cargo#7506](https://github.com/rust-lang/cargo/issues/7506)）により、`build.rs` のリンク指示が `bin` ターゲットに伝播しないため、バイナリcrateのリンクが失敗します。回避策として、CUDA ビルド時に `RUSTFLAGS` へ `-C link-arg=-lnccl` を追加してください:
  ```bash
  RUSTFLAGS="-C relocation-model=pic -C link-arg=-lnccl" cargo build --release --features cuda
  ```
  この回避策は、`llama-cpp-sys-2` の `build.rs` が `cargo:rustc-link-lib=nccl` を出力するよう修正されるか、Cargo #7506 が解決された時点で不要になります。

## ライセンス

MIT License
