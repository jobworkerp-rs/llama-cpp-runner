pub mod proto_resolve;

pub mod protobuf {
    pub mod llama_cpp {
        include!(concat!(env!("OUT_DIR"), "/llama_cpp.rs"));
    }
    pub mod llm {
        include!(concat!(env!("OUT_DIR"), "/jobworkerp.runner.llm.rs"));
    }
}

// Convenience re-exports so downstream crates can `use jobworkerp_llama_protobuf::MediaInput;`
// instead of the full protobuf path.
pub use protobuf::llama_cpp::{MediaInput, MediaKind, MtmdSettings, RawAudio, RawImage};
pub use protobuf::llm::{LlmChatArgs, LlmChatResult, LlmCompletionArgs, LlmCompletionResult};
