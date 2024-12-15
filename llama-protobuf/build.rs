extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "llama_cpp/llama_cpp_operation.proto",
                "llama_cpp/llama_cpp_arg.proto",
                "ollama/ollama_operation.proto",
                "ollama/ollama_arg.proto",
            ],
            &["./protobuf"],
        )
        .unwrap();
}
