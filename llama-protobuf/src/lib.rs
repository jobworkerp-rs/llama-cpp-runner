pub mod protobuf {
    pub mod llama_cpp {
        include!(concat!(env!("OUT_DIR"), "/llama_cpp.rs"));
    }
}

// Convenience re-exports so downstream crates can `use jobworkerp_llama_protobuf::MediaInput;`
// instead of the full protobuf path.
pub use protobuf::llama_cpp::{MediaInput, MediaKind, MtmdSettings, RawAudio, RawImage};
