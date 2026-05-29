extern crate prost_build;
fn main() {
    // prost-build does not emit `rerun-if-changed` directives for the proto
    // files it compiles. Without these, editing a .proto does not trigger a
    // rebuild and stale generated code is kept around in target/.
    let proto_files = [
        "llama_cpp/media_input.proto",
        "llama_cpp/llama_cpp_runner.proto",
        "llama_cpp/llama_cpp_arg.proto",
        "jobworkerp/runner/llm/chat_args.proto",
        "jobworkerp/runner/llm/chat_result.proto",
        "jobworkerp/runner/llm/completion_args.proto",
        "jobworkerp/runner/llm/completion_result.proto",
    ];
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=protobuf");
    for f in &proto_files {
        println!("cargo:rerun-if-changed=protobuf/{f}");
    }
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(
            ".",
            "#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]",
        )
        .compile_protos(&proto_files, &["./protobuf"])
        .unwrap();
}
