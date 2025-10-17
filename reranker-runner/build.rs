// Protocol Buffers compilation configuration
// Uses the same pattern as search-runner/build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::Config::new()
        // Support proto3 optional fields
        .protoc_arg("--experimental_allow_proto3_optional")
        // Auto-derive Serialize/Deserialize/JsonSchema for generated types
        .type_attribute(
            ".",
            "#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]",
        )
        // Compile Protobuf files
        .compile_protos(
            &[
                "reranker_settings.proto",
                "reranker_args.proto",
                "reranker_result.proto",
            ],
            &["protobuf/"],
        )?;

    Ok(())
}
