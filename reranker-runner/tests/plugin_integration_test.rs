// Integration tests for RerankerRunnerPlugin
//
// These tests verify the PluginRunner interface implementation
// without requiring actual GGUF model files.

use host_proto::jobworkerp::data::{MethodJsonSchema, MethodSchema};
use jobworkerp_plugin_abi::cancel::FfiCancellationToken;
use jobworkerp_plugin_abi::v2::{CancelToken, PluginV2};
use message_vectordb_reranker_runner::proto::{
    Candidate, RerankerArgs, RerankerOptions, RerankerResult, RerankerSettings,
};
use message_vectordb_reranker_runner::{CANCELLED, METHOD_RUN, RerankerRunnerPlugin};
use prost::Message;
use std::collections::HashMap;

/// Helper: Create default test settings (CPU mode, no actual model loading)
fn create_test_settings() -> RerankerSettings {
    RerankerSettings {
        model_id: "test-model.gguf".to_string(),
        hf_repo: None, // Use local path (will not exist, but ok for interface tests)
        use_cpu: true,
        threads: Some(2),
        threads_batch: None,
        ctx_size: Some(8192),
        n_batch: Some(512),
        use_flash_attention: Some(false),
        default_instruction: Some("Test instruction".to_string()),
        cache_size: Some(100),
        cache_ttl_seconds: Some(60),
        max_document_length_gpu: Some(5000),
        max_document_length_cpu: Some(3000),
    }
}

/// Helper: Create test arguments
fn create_test_args(query: &str, num_candidates: usize) -> RerankerArgs {
    let candidates: Vec<Candidate> = (0..num_candidates)
        .map(|i| Candidate {
            text: format!("Document {} about the query", i),
            original_data: format!(
                r#"{{"id": {}, "title": "Doc {}", "score": 0.{}}}"#,
                i,
                i,
                90 - i.min(90)
            ),
            original_score: Some((0.9 - (i as f32 * 0.05)).max(0.0)),
            id: Some(format!("candidate-{}", i)),
        })
        .collect();

    RerankerArgs {
        query: query.to_string(),
        candidates,
        options: Some(RerankerOptions {
            top_k: Some(5),
            score_threshold: Some(0.1),
            instruction: None,
            score_blend_ratio: Some(0.7), // 70% reranking, 30% original
            batch_size: Some(1),
            use_cache: Some(true),
            max_document_length: None,
        }),
    }
}

#[test]
fn test_plugin_metadata() {
    let plugin = RerankerRunnerPlugin::new();

    // Test name and description
    assert_eq!(plugin.name(), "RerankerRunner");
    assert!(!plugin.description().is_empty());
    println!("Plugin name: {}", plugin.name());
    println!("Plugin description: {}", plugin.description());
}

#[test]
fn test_plugin_proto_schemas() {
    let plugin = RerankerRunnerPlugin::new();

    let settings_proto = plugin.runner_settings_proto();
    let methods = plugin.method_proto_map();
    let run_bytes = methods
        .get(METHOD_RUN)
        .expect("`run` method must be registered");
    let run = MethodSchema::decode(run_bytes.as_slice()).expect("MethodSchema must decode");

    assert!(
        !settings_proto.is_empty(),
        "Settings proto should not be empty"
    );
    assert!(!run.args_proto.is_empty(), "Args proto should not be empty");
    assert!(
        !run.result_proto.is_empty(),
        "Result proto should not be empty"
    );
}

#[test]
fn test_plugin_json_schemas() {
    let plugin = RerankerRunnerPlugin::new();

    let settings_schema = plugin.settings_schema();
    let methods = plugin
        .method_json_schema_map()
        .expect("method_json_schema_map should be Some");
    let run_bytes = methods
        .get(METHOD_RUN)
        .expect("`run` method must be registered");
    let run = MethodJsonSchema::decode(run_bytes.as_slice()).expect("MethodJsonSchema must decode");
    let args_schema = &run.args_schema;
    let output_schema = run
        .result_schema
        .as_ref()
        .expect("result_schema should be Some");

    assert!(
        !settings_schema.is_empty(),
        "Settings schema should not be empty"
    );
    assert!(!args_schema.is_empty(), "Args schema should not be empty");

    // Parse as JSON to verify validity
    let _settings_json: serde_json::Value =
        serde_json::from_str(&settings_schema).expect("Settings schema should be valid JSON");
    let _args_json: serde_json::Value =
        serde_json::from_str(args_schema).expect("Args schema should be valid JSON");
    let _output_json: serde_json::Value =
        serde_json::from_str(output_schema).expect("Output schema should be valid JSON");
}

#[test]
fn test_settings_protobuf_serialization() {
    let settings = create_test_settings();

    // Serialize to protobuf bytes
    let mut buf = Vec::new();
    settings
        .encode(&mut buf)
        .expect("Should serialize settings");

    assert!(!buf.is_empty(), "Serialized settings should not be empty");
    println!("Serialized settings size: {} bytes", buf.len());

    // Deserialize back
    let decoded = RerankerSettings::decode(&buf[..]).expect("Should deserialize settings");

    // Verify fields
    assert_eq!(decoded.model_id, settings.model_id);
    assert_eq!(decoded.use_cpu, settings.use_cpu);
    assert_eq!(decoded.threads, settings.threads);
    assert_eq!(decoded.cache_size, settings.cache_size);
}

#[test]
fn test_args_protobuf_serialization() {
    let args = create_test_args("test query", 3);

    // Serialize to protobuf bytes
    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    assert!(!buf.is_empty(), "Serialized args should not be empty");
    println!("Serialized args size: {} bytes", buf.len());

    // Deserialize back
    let decoded = RerankerArgs::decode(&buf[..]).expect("Should deserialize args");

    // Verify fields
    assert_eq!(decoded.query, args.query);
    assert_eq!(decoded.candidates.len(), 3);
    assert_eq!(decoded.candidates[0].text, args.candidates[0].text);
    assert!(decoded.options.is_some());
}

#[test]
fn test_result_protobuf_serialization() {
    use message_vectordb_reranker_runner::proto::{RankedCandidate, RerankerStats};

    let result = RerankerResult {
        ranked_candidates: vec![
            RankedCandidate {
                reranking_score: 0.95,
                final_score: 0.93,
                rank: 1,
                original_data: r#"{"id": 0}"#.to_string(),
                original_score: Some(0.9),
                id: Some("candidate-0".to_string()),
            },
            RankedCandidate {
                reranking_score: 0.85,
                final_score: 0.83,
                rank: 2,
                original_data: r#"{"id": 1}"#.to_string(),
                original_score: Some(0.8),
                id: Some("candidate-1".to_string()),
            },
        ],
        stats: Some(RerankerStats {
            input_count: 10,
            output_count: 2,
            truncated_count: 0,
            cache_hits: 5,
            cache_misses: 5,
            cache_hit_rate: 0.5,
            processing_time_ms: 1500,
            avg_time_per_candidate_ms: 150.0,
            model_id: "test-model".to_string(),
            device: "CPU".to_string(),
        }),
        success: true,
        error_message: None,
    };

    // Serialize to protobuf bytes
    let mut buf = Vec::new();
    result.encode(&mut buf).expect("Should serialize result");

    assert!(!buf.is_empty(), "Serialized result should not be empty");
    println!("Serialized result size: {} bytes", buf.len());

    // Deserialize back
    let decoded = RerankerResult::decode(&buf[..]).expect("Should deserialize result");

    // Verify fields
    assert_eq!(decoded.ranked_candidates.len(), 2);
    assert_eq!(decoded.ranked_candidates[0].rank, 1);
    assert_eq!(decoded.ranked_candidates[0].reranking_score, 0.95);
    assert!(decoded.stats.is_some());
    assert_eq!(decoded.stats.as_ref().unwrap().input_count, 10);
    assert!(decoded.success);
}

#[tokio::test]
async fn test_plugin_load_with_invalid_settings() {
    let mut plugin = RerankerRunnerPlugin::new();

    // Create settings with invalid model path
    let settings = RerankerSettings {
        model_id: "nonexistent-model.gguf".to_string(),
        hf_repo: None,
        use_cpu: true,
        ..create_test_settings()
    };

    let mut buf = Vec::new();
    settings
        .encode(&mut buf)
        .expect("Should serialize settings");

    // Try to load plugin with invalid model
    let result = plugin.load(buf).await;

    // Should return an error (model not found)
    assert!(result.is_err(), "Loading with invalid model should fail");
    println!("Expected error: {:?}", result.unwrap_err());
}

#[tokio::test]
async fn test_plugin_run_without_load() {
    let mut plugin = RerankerRunnerPlugin::new();

    // Try to run without loading
    let args = create_test_args("test query", 3);
    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    let (result, _metadata) = plugin.run(buf, HashMap::new(), None).await;

    assert!(result.is_err(), "Should fail before reranker is loaded");
}

#[tokio::test]
async fn test_plugin_cancel_operations() {
    let mut plugin = RerankerRunnerPlugin::new();

    // A precancelled token must short-circuit run() with Err(CANCELLED) before
    // any model work is reached, even with no reranker loaded.
    let (ffi, handle) = FfiCancellationToken::new_owned();
    handle.cancel();
    plugin.set_cancellation_token(CancelToken::from_ffi(ffi));

    let args = create_test_args("test query", 3);
    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    let (result, _) = plugin.run(buf, HashMap::new(), None).await;
    assert!(
        matches!(result, Err(ref e) if e == CANCELLED),
        "Precancelled token must surface as Err(\"cancelled\"), got {result:?}"
    );
}

#[tokio::test]
async fn test_empty_query_handling() {
    let mut args = create_test_args("", 3); // Empty query
    args.query = "".to_string();

    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    let mut plugin = RerankerRunnerPlugin::new();
    let (result, _) = plugin.run(buf, HashMap::new(), None).await;

    // The error could be from not being loaded or from empty query validation
    assert!(result.is_err(), "Empty query / not-loaded must fail");
}

#[tokio::test]
async fn test_empty_candidates_handling() {
    let args = RerankerArgs {
        query: "test query".to_string(),
        candidates: vec![], // Empty candidates
        options: Some(RerankerOptions {
            top_k: Some(5),
            ..Default::default()
        }),
    };

    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    let mut plugin = RerankerRunnerPlugin::new();
    let (result, _) = plugin.run(buf, HashMap::new(), None).await;

    assert!(result.is_err(), "Empty candidates / not-loaded must fail");
}

#[test]
fn test_protobuf_round_trip() {
    // Test complete protobuf round-trip for all types

    // 1. Settings
    let settings = create_test_settings();
    let mut buf = Vec::new();
    settings.encode(&mut buf).unwrap();
    let decoded_settings = RerankerSettings::decode(&buf[..]).unwrap();
    assert_eq!(decoded_settings.model_id, settings.model_id);

    // 2. Args
    let args = create_test_args("test", 5);
    let mut buf = Vec::new();
    args.encode(&mut buf).unwrap();
    let decoded_args = RerankerArgs::decode(&buf[..]).unwrap();
    assert_eq!(decoded_args.query, args.query);
    assert_eq!(decoded_args.candidates.len(), 5);

    println!("Protobuf round-trip test passed for all types");
}

#[test]
fn test_large_candidate_list() {
    // Test with 100 candidates
    let args = create_test_args("large test", 100);

    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize large args");

    println!("Serialized 100 candidates: {} bytes", buf.len());

    let decoded = RerankerArgs::decode(&buf[..]).expect("Should deserialize large args");
    assert_eq!(decoded.candidates.len(), 100);
}

#[test]
fn test_unicode_query_and_documents() {
    // Test with Japanese text
    let candidates = vec![
        Candidate {
            text: "これはテストドキュメントです。".to_string(),
            original_data: r#"{"id": 0, "lang": "ja"}"#.to_string(),
            original_score: Some(0.9),
            id: Some("doc-0".to_string()),
        },
        Candidate {
            text: "データベースの設計について".to_string(),
            original_data: r#"{"id": 1, "lang": "ja"}"#.to_string(),
            original_score: Some(0.8),
            id: Some("doc-1".to_string()),
        },
    ];

    let args = RerankerArgs {
        query: "データベース設計".to_string(),
        candidates,
        options: None,
    };

    // Test serialization
    let mut buf = Vec::new();
    args.encode(&mut buf)
        .expect("Should serialize unicode args");

    let decoded = RerankerArgs::decode(&buf[..]).expect("Should deserialize unicode args");
    assert_eq!(decoded.query, "データベース設計");
    assert_eq!(decoded.candidates[0].text, "これはテストドキュメントです。");

    println!("Unicode test passed");
}

#[test]
fn test_score_blend_ratio_boundaries() {
    // Test various score_blend_ratio values
    let ratios = vec![0.0, 0.25, 0.5, 0.75, 1.0];

    for ratio in ratios {
        let mut args = create_test_args("test", 3);
        if let Some(ref mut opts) = args.options {
            opts.score_blend_ratio = Some(ratio);
        }

        let mut buf = Vec::new();
        args.encode(&mut buf).expect("Should serialize");

        let decoded = RerankerArgs::decode(&buf[..]).expect("Should deserialize");
        assert_eq!(
            decoded.options.as_ref().unwrap().score_blend_ratio,
            Some(ratio)
        );
    }

    println!("Score blend ratio boundary test passed");
}

#[test]
fn test_optional_fields() {
    // Test args with minimal optional fields
    let args = RerankerArgs {
        query: "minimal query".to_string(),
        candidates: vec![Candidate {
            text: "doc".to_string(),
            original_data: "{}".to_string(),
            original_score: None, // Optional
            id: None,             // Optional
        }],
        options: None, // Optional
    };

    let mut buf = Vec::new();
    args.encode(&mut buf)
        .expect("Should serialize minimal args");

    let decoded = RerankerArgs::decode(&buf[..]).expect("Should deserialize minimal args");
    assert_eq!(decoded.candidates[0].original_score, None);
    assert!(decoded.options.is_none());

    println!("Optional fields test passed");
}
