use anyhow::{Context, Result};
use command_utils::util::option::Exists;
use ollama_rs::{
    generation::completion::{request::GenerationRequest, GenerationResponseStream},
    Ollama, error::OllamaError
};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::io::{stdout, AsyncWriteExt};
use tokio_stream::StreamExt;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<()> {
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

    let system_prompt = 
    "以下の文章は、ある特定の年月の日記(Markdownファイル)を結合したものです。その月のまとめとして日記の内容の要約をしてください。";

    let ollama = Ollama::new("http://localhost".to_string(), 11434);
    // let mut context: Option<GenerationContext> = None;

    let request = GenerationRequest::new("llama3.3:70b".into(), combined_content)
        .system(system_prompt.to_string());

    // if let Some(context) = context.clone() {
    //     request = request.context(context);
    // }
    let mut stream: GenerationResponseStream = ollama.generate_stream(request).await.map_err(
        |e| match e {
            OllamaError::Other(s) => anyhow::anyhow!(s),
            e => anyhow::anyhow!(e.to_string()),
        },
    )?;

    let mut stdout = stdout();
    while let Some(Ok(res)) = stream.next().await {
        for ele in res {
            stdout.write_all(ele.response.as_bytes()).await?;
            stdout.flush().await?;

            // if ele.context.is_some() {
            //     context = ele.context;
            // }
        }
    }

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
                    .exists(|f| !f.to_string_lossy().starts_with("._"))
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
