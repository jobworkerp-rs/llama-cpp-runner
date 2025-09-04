use anyhow::{anyhow, Context, Result};
use jobworkerp_client::plugins::PluginRunner;
use jobworkerp_llama_cpp_plugin::LlamaCppPlugin;
use jobworkerp_llama_protobuf::protobuf::llama_cpp::LlamaArg;
use prost::Message;
use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tracing::Level;

fn main() -> Result<()> {
    command_utils::util::tracing::tracing_init_test(Level::INFO);
    dotenvy::dotenv().ok();

    // let args: Vec<String> = env::args().collect();
    // if args.len() < 2 {
    //     println!("Usage: {} <directory_path> [separator]", args[0]);
    //     std::process::exit(1);
    // }
    let args = vec![
        "summary".to_string(),
        "/Users/sutr/mnt/Documents/obsidian/日記/2024/11/".to_string(),
    ];

    let directory = &args[1];
    let separator = args.get(2).map(|s| s.as_str()).unwrap_or("\n---\n");

    let combined_content = collect_and_combine_markdown_files(directory, separator)
        .context("Failed to process markdown files")?;

    let content_len = combined_content.len();
    println!("combined_content length: {}", &content_len);
    let mut plugin = LlamaCppPlugin::new();
    plugin
        .load_model_from_env()
        .expect("failed to load model from env");
    let system_prompt = "以下の文章は、ある特定の年月の日記(Markdownファイル)を結合して作成された文章です。実施したこと、良かった点、改善したい点をそれぞれまとめてください。";
    plugin.set_system_prompt(system_prompt);

    let request = LlamaArg {
        prompt: combined_content,
        sample_len: 5000,
        temperature: Some(0.8),
        top_p: Some(0.9),
        repeat_penalty: Some(0.9),
        repeat_last_n: Some(8),
        seed: Some(32),
        need_print: true,
    };
    let mut buf = Vec::with_capacity(request.encoded_len());
    request.encode(&mut buf).unwrap();
    let res = plugin
        .run(buf, HashMap::new())
        .0
        .expect("failed to run plugin");
    let res = LlamaArg::decode(&mut Cursor::new(res.clone()))
        .map_err(|e| anyhow!("decode error: {}", e))
        .unwrap();

    println!("response: {:?}", res.prompt);

    Ok(())
}

fn collect_markdown_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                collect_markdown_files(&path, files)?;
            } else if path.is_file()
                && path
                    .file_name()
                    .is_some_and(|f| !f.to_string_lossy().starts_with("._"))
                && path.extension().map_or(false, |ext| ext == "md")
            {
                files.push(path);
            }
        }
    }
    Ok(())
}

fn collect_and_combine_markdown_files(directory: &str, separator: &str) -> Result<String> {
    let mut markdown_files = Vec::new();
    collect_markdown_files(Path::new(directory), &mut markdown_files)?;

    markdown_files.sort();

    let mut combined_content = String::new();
    for (i, path) in markdown_files.iter().enumerate() {
        let content =
            fs::read_to_string(path).with_context(|| format!("Failed to read file: {:?}", path))?;

        if i > 0 {
            combined_content.push_str(separator);
        }
        combined_content.push_str(&content);
    }

    Ok(combined_content)
}
