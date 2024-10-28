extern crate prost_build;
fn main() {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["llama_cpp_operation.proto", "llama_cpp_arg.proto"],
            &["./protobuf"],
        )
        .unwrap();
}
