use anyhow::Result;
use async_trait::async_trait;

pub mod protobuf {
    pub mod ollama {
        include!(concat!(env!("OUT_DIR"), "/ollama.rs"));
    }
    pub mod llama_cpp {
        include!(concat!(env!("OUT_DIR"), "/llama_cpp.rs"));
    }
}

#[async_trait]
pub trait PluginRunner: Send + Sync {
    fn name(&self) -> String;
    fn load(&mut self, settings: Vec<u8>) -> Result<()>;
    fn run(&mut self, arg: Vec<u8>) -> Result<Vec<Vec<u8>>>;
    fn cancel(&self) -> bool;
    fn runner_settings_proto(&self) -> String;
    fn job_args_proto(&self) -> String;
    fn result_output_proto(&self) -> Option<String>;
    // if true, use job result of before job, else use job args from request
    fn use_job_result(&self) -> bool;
}
