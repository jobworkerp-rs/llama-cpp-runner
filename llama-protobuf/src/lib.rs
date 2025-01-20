pub mod protobuf {
    pub mod ollama {
        include!(concat!(env!("OUT_DIR"), "/ollama.rs"));
    }
    pub mod llama_cpp {
        include!(concat!(env!("OUT_DIR"), "/llama_cpp.rs"));
    }
}
