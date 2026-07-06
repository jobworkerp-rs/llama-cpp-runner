// Re-export the generic resolver from command-utils so existing call sites
// (`jobworkerp_llama_protobuf::proto_resolve::resolve_proto_imports`) keep working.
pub use command_utils::protobuf::resolve::resolve_proto_imports;

/// Shared import table: maps import paths to their file contents.
/// All plugins that import `llama_cpp/media_input.proto` share this
/// single definition so call sites never diverge.
pub const MEDIA_INPUT_IMPORT: (&str, &str) = (
    "llama_cpp/media_input.proto",
    include_str!("../protobuf/llama_cpp/media_input.proto"),
);

/// Tests that use real proto files from this workspace or prost-generated
/// types — these cannot live in command-utils.
#[cfg(test)]
mod real_proto_tests {
    use super::*;

    fn no_import_statement(proto: &str) -> bool {
        !proto.lines().any(|l| l.trim().starts_with("import "))
    }

    fn resolve(main: &str, imports: &[(&str, &str)]) -> String {
        resolve_proto_imports(main, imports).expect("resolve failed")
    }

    #[test]
    fn real_llama_cpp_runner_proto() {
        let main = include_str!("../protobuf/llama_cpp/llama_cpp_runner.proto");
        let result = resolve(main, &[MEDIA_INPUT_IMPORT]);

        assert!(no_import_statement(&result), "no import statements");
        assert!(
            result.contains("message LlamaRunnerSettings"),
            "primary message present"
        );
        assert!(
            result.contains("message MtmdSettings"),
            "imported MtmdSettings inlined"
        );

        let first_msg = result
            .lines()
            .find(|l| l.trim().starts_with("message "))
            .unwrap();
        assert!(
            first_msg.contains("LlamaRunnerSettings"),
            "first message is primary"
        );
    }

    #[test]
    fn real_llama_cpp_arg_proto() {
        let main = include_str!("../protobuf/llama_cpp/llama_cpp_arg.proto");
        let result = resolve(main, &[MEDIA_INPUT_IMPORT]);

        assert!(no_import_statement(&result), "no import statements");
        assert!(
            result.contains("message LlamaArg"),
            "primary message present"
        );
        assert!(
            result.contains("message MediaInput"),
            "imported MediaInput inlined"
        );
    }

    #[test]
    fn real_embedding_args_proto() {
        let main = include_str!("../../embedding-llm/protobuf/embedding_args.proto");
        let result = resolve(main, &[MEDIA_INPUT_IMPORT]);

        assert!(no_import_statement(&result), "no import statements");
        assert!(
            result.contains("message EmbeddingArgs"),
            "primary message present"
        );
        assert!(result.contains("MediaInput medias"), "field type resolved");
    }

    #[test]
    fn real_llm_runner_settings_proto() {
        let main = include_str!("../../embedding-llm/protobuf/llm_runner_settings.proto");
        let result = resolve(main, &[MEDIA_INPUT_IMPORT]);

        assert!(no_import_statement(&result), "no import statements");
        assert!(
            result.contains("message EmbeddingLlmRunnerSettings"),
            "primary message present"
        );
        assert!(result.contains("MtmdSettings mtmd"), "field type resolved");
    }
}

/// Binary-compatibility tests: verify that resolved proto strings produce
/// wire-compatible bytes with the prost-generated types.
#[cfg(test)]
mod binary_compat_tests {
    use super::*;
    use command_utils::protobuf::ProtobufDescriptor;
    use prost::Message;

    fn resolve_media_import(main_proto: &str) -> String {
        resolve_proto_imports(main_proto, &[MEDIA_INPUT_IMPORT]).expect("resolve failed")
    }

    #[test]
    fn llama_arg_reflection_encode_prost_decode() {
        let resolved =
            resolve_media_import(include_str!("../protobuf/llama_cpp/llama_cpp_arg.proto"));
        let desc = ProtobufDescriptor::new(&resolved).expect("descriptor creation");
        let msg_desc = desc.get_messages().first().cloned().unwrap();
        assert_eq!(msg_desc.name(), "LlamaArg", "first message must be primary");

        let json = serde_json::json!({
            "prompt": "Hello world",
            "sampleLen": 512,
            "temperature": 0.7,
            "topP": 0.9,
            "repeatPenalty": 1.1,
            "repeatLastN": 64,
            "seed": "42",
            "needPrint": false,
            "medias": [{ "kind": "MEDIA_KIND_IMAGE", "encoded": "AQID" }]
        });
        let bytes = ProtobufDescriptor::json_value_to_message(msg_desc, &json, true, false)
            .expect("reflection encode");

        let decoded =
            crate::protobuf::llama_cpp::LlamaArg::decode(bytes.as_slice()).expect("prost decode");
        assert_eq!(decoded.prompt, "Hello world");
        assert_eq!(decoded.sample_len, 512);
        assert!((decoded.temperature.unwrap() - 0.7).abs() < 1e-6);
        assert!((decoded.top_p.unwrap() - 0.9).abs() < 1e-6);
        assert!((decoded.repeat_penalty.unwrap() - 1.1).abs() < 1e-4);
        assert_eq!(decoded.repeat_last_n, Some(64));
        assert_eq!(decoded.seed, Some(42));
        assert!(!decoded.need_print);
        assert_eq!(decoded.medias.len(), 1);
        assert_eq!(
            decoded.medias[0].kind,
            crate::protobuf::llama_cpp::MediaKind::Image as i32
        );
    }

    #[test]
    fn llama_arg_prost_encode_reflection_decode() {
        let resolved =
            resolve_media_import(include_str!("../protobuf/llama_cpp/llama_cpp_arg.proto"));
        let desc = ProtobufDescriptor::new(&resolved).expect("descriptor creation");
        let msg_desc = desc.get_messages().first().cloned().unwrap();
        assert_eq!(msg_desc.name(), "LlamaArg", "first message must be primary");

        let arg = crate::protobuf::llama_cpp::LlamaArg {
            prompt: "Translate this".into(),
            sample_len: 1024,
            temperature: Some(0.5),
            top_p: Some(0.95),
            repeat_penalty: Some(1.2),
            repeat_last_n: Some(32),
            seed: Some(99),
            need_print: true,
            medias: vec![crate::protobuf::llama_cpp::MediaInput {
                kind: crate::protobuf::llama_cpp::MediaKind::Audio as i32,
                source: Some(crate::protobuf::llama_cpp::media_input::Source::FilePath(
                    "/tmp/test.wav".into(),
                )),
                id: Some("audio-1".into()),
            }],
            reuse_kv_prefix: Some(true),
        };
        let bytes = arg.encode_to_vec();

        let dynamic = ProtobufDescriptor::get_message_from_bytes(msg_desc, &bytes)
            .expect("reflection decode");
        let json = ProtobufDescriptor::message_to_json_value(&dynamic).expect("to json");

        assert_eq!(json["prompt"], "Translate this");
        assert_eq!(json["sampleLen"], 1024);
        assert_eq!(json["needPrint"], true);
        assert_eq!(json["medias"][0]["kind"], "MEDIA_KIND_AUDIO");
        assert_eq!(json["medias"][0]["filePath"], "/tmp/test.wav");
        assert_eq!(json["medias"][0]["id"], "audio-1");
    }

    #[test]
    fn llama_runner_settings_reflection_encode_prost_decode() {
        let resolved =
            resolve_media_import(include_str!("../protobuf/llama_cpp/llama_cpp_runner.proto"));
        let desc = ProtobufDescriptor::new(&resolved).expect("descriptor creation");
        let msg_desc = desc.get_messages().first().cloned().unwrap();
        assert_eq!(
            msg_desc.name(),
            "LlamaRunnerSettings",
            "first message must be primary"
        );

        let json = serde_json::json!({
            "model": "test-model.gguf",
            "hfRepo": "org/repo",
            "disableGpu": true,
            "seed": 42,
            "ctxSize": 4096,
            "typeK": "KV_CACHE_TYPE_Q8_0",
            "typeV": "KV_CACHE_TYPE_F16",
            "useFlashAttention": true,
            "systemPrompt": "You are helpful.",
            "mtmd": {
                "mmproj": "mmproj.gguf",
                "mmprojHfRepo": "org/mmproj",
                "mmprojUseGpu": true,
                "mediaMarker": "<img>",
                "allowUrlFetch": false,
                "maxMediaBytes": 10485760,
                "maxDecodedMediaBytes": "104857600",
                "allowedMediaDirs": ["/data", "/tmp"]
            }
        });
        let bytes = ProtobufDescriptor::json_value_to_message(msg_desc, &json, true, false)
            .expect("reflection encode");

        let decoded = crate::protobuf::llama_cpp::LlamaRunnerSettings::decode(bytes.as_slice())
            .expect("prost decode");
        assert_eq!(decoded.model, "test-model.gguf");
        assert_eq!(decoded.hf_repo, Some("org/repo".into()));
        assert!(decoded.disable_gpu);
        assert_eq!(decoded.seed, Some(42));
        assert_eq!(decoded.ctx_size, Some(4096));
        assert_eq!(
            decoded.type_k,
            Some(crate::protobuf::llama_cpp::KvCacheType::Q80 as i32)
        );
        assert_eq!(
            decoded.type_v,
            Some(crate::protobuf::llama_cpp::KvCacheType::F16 as i32)
        );
        assert_eq!(decoded.use_flash_attention, Some(true));

        let mtmd = decoded.mtmd.expect("mtmd present");
        assert_eq!(mtmd.mmproj, "mmproj.gguf");
        assert_eq!(mtmd.mmproj_hf_repo, Some("org/mmproj".into()));
        assert_eq!(mtmd.mmproj_use_gpu, Some(true));
        assert_eq!(mtmd.media_marker, Some("<img>".into()));
        assert!(!mtmd.allow_url_fetch);
        assert_eq!(mtmd.max_media_bytes, 10_485_760);
        assert_eq!(mtmd.max_decoded_media_bytes, 104_857_600);
        assert_eq!(mtmd.allowed_media_dirs, vec!["/data", "/tmp"]);
    }

    #[test]
    fn llama_runner_settings_prost_encode_reflection_decode() {
        let resolved =
            resolve_media_import(include_str!("../protobuf/llama_cpp/llama_cpp_runner.proto"));
        let desc = ProtobufDescriptor::new(&resolved).expect("descriptor creation");
        let msg_desc = desc.get_messages().first().cloned().unwrap();
        assert_eq!(
            msg_desc.name(),
            "LlamaRunnerSettings",
            "first message must be primary"
        );

        let settings = crate::protobuf::llama_cpp::LlamaRunnerSettings {
            model: "my-model.gguf".into(),
            hf_repo: Some("user/model".into()),
            disable_gpu: true,
            seed: Some(7),
            threads: Some(4),
            threads_batch: Some(2),
            ctx_size: Some(2048),
            n_batch: Some(1024),
            n_ubatch: Some(256),
            type_k: Some(crate::protobuf::llama_cpp::KvCacheType::Q80 as i32),
            type_v: Some(crate::protobuf::llama_cpp::KvCacheType::Q80 as i32),
            reuse_kv_prefix: Some(true),
            use_flash_attention: Some(false),
            system_prompt: Some("Be concise.".into()),
            mtmd: Some(crate::protobuf::llama_cpp::MtmdSettings {
                mmproj: "proj.gguf".into(),
                mmproj_hf_repo: None,
                mmproj_use_gpu: Some(false),
                media_marker: None,
                allow_url_fetch: true,
                max_media_bytes: 5000,
                max_decoded_media_bytes: 50000,
                allowed_media_dirs: vec!["/images".into()],
            }),
        };
        let bytes = settings.encode_to_vec();

        let dynamic = ProtobufDescriptor::get_message_from_bytes(msg_desc, &bytes)
            .expect("reflection decode");
        let json = ProtobufDescriptor::message_to_json_value(&dynamic).expect("to json");

        assert_eq!(json["model"], "my-model.gguf");
        assert_eq!(json["hfRepo"], "user/model");
        assert_eq!(json["disableGpu"], true);
        assert_eq!(json["nBatch"], 1024);
        assert_eq!(json["nUbatch"], 256);
        assert_eq!(json["typeK"], "KV_CACHE_TYPE_Q8_0");
        assert_eq!(json["typeV"], "KV_CACHE_TYPE_Q8_0");
        assert_eq!(json["reuseKvPrefix"], true);
        assert_eq!(json["mtmd"]["mmproj"], "proj.gguf");
        assert_eq!(json["mtmd"]["allowUrlFetch"], true);
        assert_eq!(json["mtmd"]["maxMediaBytes"], 5000);
        assert_eq!(json["mtmd"]["allowedMediaDirs"][0], "/images");
    }
}
