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
pub use protobuf::llama_cpp::{
    MediaInput, MediaKind, MtmdSettings, MtpSpeculativeSettings, RawAudio, RawImage,
};
pub use protobuf::llm::{
    LlmChatArgs, LlmChatResult, LlmCompletionArgs, LlmCompletionResult, TokenCountArgs,
    TokenCountResult,
};

#[cfg(test)]
mod token_count_wire_tests {
    use super::protobuf::llm::{LlmChatArgs, LlmCompletionArgs, TokenCountArgs, token_count_args};
    use prost::Message;

    #[test]
    fn token_count_target_wire_tags_are_stable() {
        let text = TokenCountArgs {
            target: Some(token_count_args::Target::Text(
                token_count_args::TextCountRequest {
                    texts: vec!["hello".to_string()],
                },
            )),
        };
        let completion = TokenCountArgs {
            target: Some(token_count_args::Target::Completion(
                LlmCompletionArgs::default(),
            )),
        };
        let chat = TokenCountArgs {
            target: Some(token_count_args::Target::Chat(
                token_count_args::ChatCountRequest {
                    request: Some(LlmChatArgs::default()),
                },
            )),
        };

        assert_eq!(text.encode_to_vec()[0], 0x0A, "text = 1");
        assert_eq!(completion.encode_to_vec()[0], 0x12, "completion = 2");
        assert_eq!(chat.encode_to_vec()[0], 0x1A, "chat = 3");
    }
}
