extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(
            ".",
            "#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]",
        )
        .compile_protos(
            &[
                "llama_cpp/media_input.proto",
                "llama_cpp/llama_cpp_runner.proto",
                "llama_cpp/llama_cpp_arg.proto",
                "jobworkerp/runner/llm/chat_args.proto",
                "jobworkerp/runner/llm/chat_result.proto",
                "jobworkerp/runner/llm/completion_args.proto",
                "jobworkerp/runner/llm/completion_result.proto",
            ],
            &["./protobuf"],
        )
        .unwrap();
}
