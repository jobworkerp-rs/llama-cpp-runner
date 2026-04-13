use std::io::Result;

fn main() -> Result<()> {
    // `extern_path` delegates types under the `.llama_cpp` proto package to
    // the `jobworkerp_llama_protobuf` crate so we don't re-generate MediaInput
    // / MtmdSettings here. The include path still needs the llama-protobuf
    // directory so protoc can resolve `import "llama_cpp/media_input.proto";`.
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(
            ".",
            "#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]",
        )
        .extern_path(
            ".llama_cpp",
            "::jobworkerp_llama_protobuf::protobuf::llama_cpp",
        )
        .compile_protos(
            &[
                "protobuf/llm_runner_settings.proto",
                "protobuf/embedding_args.proto",
                "protobuf/llm_result.proto",
            ],
            &["protobuf", "../llama-protobuf/protobuf"],
        )?;
    Ok(())
}
