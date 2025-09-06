use std::io::Result;

fn main() -> Result<()> {
    // Compile protobuf files
    prost_build::compile_protos(
        &[
            "protobuf/llm_runner_settings.proto",
            "protobuf/embedding_args.proto",
            "protobuf/llm_result.proto",
        ],
        &["protobuf/"],
    )?;
    Ok(())
}