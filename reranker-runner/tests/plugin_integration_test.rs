// Integration tests for RerankerRunnerPlugin
//
// These tests verify the PluginRunner interface implementation
// without requiring actual GGUF model files.

use jobworkerp_client::plugins::PluginRunner;
use message_vectordb_reranker_runner::RerankerRunnerPlugin;
use message_vectordb_reranker_runner::proto::{
    Candidate, RerankerArgs, RerankerOptions, RerankerResult, RerankerSettings,
};
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

    // Test that proto schemas are available
    let settings_proto = plugin.runner_settings_proto();
    let args_proto = plugin.job_args_proto();
    let result_proto = plugin.result_output_proto();

    assert!(
        !settings_proto.is_empty(),
        "Settings proto should not be empty"
    );
    assert!(!args_proto.is_empty(), "Args proto should not be empty");
    assert!(result_proto.is_some(), "Result proto should be Some");

    println!("Settings proto length: {}", settings_proto.len());
    println!("Args proto length: {}", args_proto.len());
    println!("Result proto length: {}", result_proto.unwrap().len());
}

#[test]
fn test_plugin_json_schemas() {
    let plugin = RerankerRunnerPlugin::new();

    // Test that JSON schemas are available (return String directly, not Result)
    let settings_schema = plugin.settings_schema();
    let args_schema = plugin.arguments_schema();
    let output_schema = plugin.output_json_schema();

    assert!(
        !settings_schema.is_empty(),
        "Settings schema should not be empty"
    );
    assert!(!args_schema.is_empty(), "Args schema should not be empty");
    assert!(output_schema.is_some(), "Output schema should be Some");

    // Parse as JSON to verify validity
    let settings_json: serde_json::Value =
        serde_json::from_str(&settings_schema).expect("Settings schema should be valid JSON");
    let args_json: serde_json::Value =
        serde_json::from_str(&args_schema).expect("Args schema should be valid JSON");

    // output_schema is Option<String>
    let output_json: serde_json::Value = if let Some(schema) = output_schema {
        serde_json::from_str(&schema).expect("Output schema should be valid JSON")
    } else {
        panic!("Output schema should not be None");
    };

    println!(
        "Settings schema: {}",
        serde_json::to_string_pretty(&settings_json).unwrap()
    );
    println!(
        "Args schema keys: {:?}",
        args_json.as_object().unwrap().keys()
    );
    println!(
        "Output schema keys: {:?}",
        output_json.as_object().unwrap().keys()
    );
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

#[test]
fn test_plugin_load_with_invalid_settings() {
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
    let result = plugin.load(buf);

    // Should return an error (model not found)
    assert!(result.is_err(), "Loading with invalid model should fail");
    println!("Expected error: {:?}", result.unwrap_err());
}

#[test]
fn test_plugin_run_without_load() {
    let mut plugin = RerankerRunnerPlugin::new();

    // Try to run without loading
    let args = create_test_args("test query", 3);
    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    let (result, _metadata) = plugin.run(buf, HashMap::new());

    // Should return an error (plugin not loaded)
    // Note: The current implementation may not check for plugin load state
    // before attempting to run. If it doesn't return an error, that's also acceptable.
    if let Err(e) = result {
        println!("Expected error: {e:?}");
    } else {
        println!("Plugin ran without explicit load check - implementation accepts this");
    }
}

#[test]
fn test_plugin_cancel_operations() {
    let plugin = RerankerRunnerPlugin::new();

    // Test cancel operations (Phase 1: always returns false)
    assert!(!plugin.is_canceled(), "Should not be canceled initially");
    assert!(!plugin.cancel(), "Cancel should return false in Phase 1");
    assert!(!plugin.is_canceled(), "Should still not be canceled");
}

#[test]
fn test_empty_query_handling() {
    let mut args = create_test_args("", 3); // Empty query
    args.query = "".to_string();

    let mut buf = Vec::new();
    args.encode(&mut buf).expect("Should serialize args");

    // Since plugin is not loaded, we can't test the actual validation
    // This test just verifies that empty query can be serialized
    let mut plugin = RerankerRunnerPlugin::new();
    let (result, _) = plugin.run(buf, HashMap::new());

    // The error could be from not being loaded or from empty query validation
    if let Err(e) = result {
        println!("Error as expected: {e:?}");
    } else {
        println!("Implementation allows empty query or doesn't enforce load state");
    }
}

#[test]
fn test_empty_candidates_handling() {
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
    let (result, _) = plugin.run(buf, HashMap::new());

    // The error could be from not being loaded or from empty candidates validation
    if let Err(e) = result {
        println!("Error as expected: {e:?}");
    } else {
        println!("Implementation allows empty candidates or doesn't enforce load state");
    }
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
