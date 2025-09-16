use std::io::Result;

fn main() -> Result<()> {
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(
            ".",
            "#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]",
        )
        .compile_protos(
            &[
                "protobuf/llm_runner_settings.proto",
                "protobuf/embedding_args.proto",
                "protobuf/llm_result.proto",
            ],
            &["protobuf"],
        )?;
    Ok(())
}
