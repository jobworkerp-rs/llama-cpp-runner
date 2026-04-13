use regex::Regex;

/// Shared import table: maps import paths to their file contents.
/// All plugins that import `llama_cpp/media_input.proto` share this
/// single definition so call sites never diverge.
pub const MEDIA_INPUT_IMPORT: (&str, &str) = (
    "llama_cpp/media_input.proto",
    include_str!("../protobuf/llama_cpp/media_input.proto"),
);

/// Resolve proto `import` statements by inlining the imported definitions.
///
/// The JobWorkerP plugin system requires each proto string to be
/// self-contained (no `import`), with the first `message` being the primary
/// type. This function takes a main proto and its imports (path→content pairs),
/// strips import lines, and appends the imported definitions after the main
/// content so the primary message stays first.
///
/// Package-prefix removal targets only proto3 type-reference positions
/// (field types, map value types, rpc argument/return types, extend targets)
/// using regex-based matching. Quoted strings, comments, package/syntax
/// declarations, and non-type-reference positions are never modified.
///
/// # Errors
///
/// Returns `Err` if the resolved output still contains unresolved `import`
/// statements, either from the main proto or from any of the imported files
/// (i.e. transitive imports that were not provided in `imports`).
pub fn resolve_proto_imports(main_proto: &str, imports: &[(&str, &str)]) -> Result<String, String> {
    let import_paths: Vec<&str> = imports.iter().map(|(p, _)| *p).collect();

    // Collect package names from imported files to build prefix set
    let imported_packages: Vec<String> = imports
        .iter()
        .filter_map(|(_, content)| extract_package(content))
        .collect();

    // Strip resolved import lines from main proto
    let mut main_lines: Vec<&str> = Vec::new();
    for line in main_proto.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            if let Some(path) = extract_import_path(trimmed) {
                if import_paths.contains(&path.as_str()) {
                    continue;
                }
            }
        }
        main_lines.push(line);
    }

    let mut output = main_lines.join("\n");

    // Append each imported file's definitions.
    // Strip syntax/package lines; strip only resolved import lines (keep
    // unresolved ones so the final check catches transitive imports).
    for (_path, content) in imports {
        output.push('\n');
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("syntax ") || trimmed.starts_with("package ") {
                continue;
            }
            if trimmed.starts_with("import ") {
                if let Some(path) = extract_import_path(trimmed) {
                    if import_paths.contains(&path.as_str()) {
                        continue;
                    }
                }
            }
            output.push_str(line);
            output.push('\n');
        }
    }

    // Replace package-qualified type references with local names,
    // targeting only syntactic positions where types appear in proto3.
    for pkg in &imported_packages {
        output = strip_package_prefix_in_type_refs(&output, pkg);
    }

    // Fail-fast: reject if any import statements remain unresolved
    let unresolved: Vec<&str> = output
        .lines()
        .filter(|l| l.trim().starts_with("import "))
        .collect();
    if !unresolved.is_empty() {
        return Err(format!(
            "unresolved import(s) remain after resolution — \
             all transitive imports must be provided in the `imports` table:\n{}",
            unresolved.join("\n")
        ));
    }

    Ok(output)
}

/// Extract the quoted path from an `import "...";` line.
fn extract_import_path(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let rest = trimmed
        .strip_prefix("import public ")
        .or_else(|| trimmed.strip_prefix("import "))?;
    let start = rest.find('"')?;
    let end = rest[start + 1..].find('"')?;
    Some(rest[start + 1..start + 1 + end].to_string())
}

/// Extract the `package` name from a proto source string.
fn extract_package(proto: &str) -> Option<String> {
    proto.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("package ")
            .and_then(|rest| rest.strip_suffix(';'))
            .map(|pkg| pkg.trim().to_string())
    })
}

/// Remove a package prefix from type references in proto3 syntactic positions.
///
/// Targets these proto3 type-reference contexts:
/// 1. Field types: `(optional|repeated)? pkg.Type field_name = N;`
/// 2. Map value types: `map<KeyType, pkg.Type>`
/// 3. RPC arguments/returns: `rpc Name(pkg.Type) returns (pkg.Type)`
/// 4. Extend targets: `extend pkg.Type {`
///
/// Processing is line-by-line: comment-only lines and trailing comments are
/// preserved verbatim; regex replacement is applied only to the code portion.
/// Nested type names (e.g. `pkg.Outer.Inner`) are supported.
fn strip_package_prefix_in_type_refs(proto: &str, package: &str) -> String {
    let escaped = regex::escape(package);
    let prefix_pattern = format!(r"{escaped}\.");

    let field_re = field_type_regex(&prefix_pattern);
    let map_re = map_value_regex(&prefix_pattern);
    let rpc_re = rpc_type_regex(&prefix_pattern);
    let extend_re = extend_regex(&prefix_pattern);
    let all_regexes: [&Regex; 4] = [&field_re, &map_re, &rpc_re, &extend_re];

    proto
        .lines()
        .map(|line| {
            let trimmed = line.trim();

            // Skip full-line comments entirely
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
                return line.to_string();
            }

            // Split code from trailing comment
            let (code, comment) = split_trailing_comment(line);

            // Apply all regex replacements to code portion only
            let mut replaced = code.to_string();
            for re in all_regexes {
                replaced = re.replace_all(&replaced, "$pre$type$post").to_string();
            }

            match comment {
                Some(c) => format!("{replaced}{c}"),
                None => replaced,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split a line into (code, trailing_comment) at the first `//` that is
/// not inside a quoted string. Handles backslash-escaped quotes (`\"`).
fn split_trailing_comment(line: &str) -> (&str, Option<&str>) {
    let mut in_quotes = false;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if in_quotes && bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2; // skip escaped character
            continue;
        }
        if bytes[i] == b'"' {
            in_quotes = !in_quotes;
        } else if !in_quotes && bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            return (&line[..i], Some(&line[i..]));
        }
        i += 1;
    }
    (line, None)
}

/// Type name pattern that supports nested types (e.g. `Outer.Inner`).
const TYPE_NAME_PATTERN: &str = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*";

fn field_type_regex(prefix_pattern: &str) -> Regex {
    let pattern = format!(
        r"(?P<pre>(?:optional\s+|repeated\s+|required\s+)?){prefix_pattern}(?P<type>{TYPE_NAME_PATTERN})(?P<post>\s+\w+\s*=)"
    );
    Regex::new(&pattern).expect("invalid field regex")
}

fn map_value_regex(prefix_pattern: &str) -> Regex {
    let pattern = format!(
        r"(?P<pre>map\s*<\s*\w+\s*,\s*){prefix_pattern}(?P<type>{TYPE_NAME_PATTERN})(?P<post>\s*>)"
    );
    Regex::new(&pattern).expect("invalid map regex")
}

fn rpc_type_regex(prefix_pattern: &str) -> Regex {
    let pattern =
        format!(r"(?P<pre>\(\s*){prefix_pattern}(?P<type>{TYPE_NAME_PATTERN})(?P<post>\s*\))");
    Regex::new(&pattern).expect("invalid rpc regex")
}

fn extend_regex(prefix_pattern: &str) -> Regex {
    let pattern =
        format!(r"(?P<pre>extend\s+){prefix_pattern}(?P<type>{TYPE_NAME_PATTERN})(?P<post>\s*\{{)");
    Regex::new(&pattern).expect("invalid extend regex")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_import_statement(proto: &str) -> bool {
        !proto.lines().any(|l| l.trim().starts_with("import "))
    }

    const MAIN_WITH_IMPORT: &str = "\
syntax = \"proto3\";

package test_pkg;

import \"llama_cpp/media_input.proto\";

message MainMessage {
  string name = 1;
  llama_cpp.MediaInput media = 2;
}";

    const IMPORTED: &str = "\
syntax = \"proto3\";

package llama_cpp;

message MediaInput {
  bytes data = 1;
}

enum MediaKind {
  MEDIA_KIND_UNSPECIFIED = 0;
}";

    fn resolve(main: &str, imports: &[(&str, &str)]) -> String {
        resolve_proto_imports(main, imports).expect("resolve failed")
    }

    #[test]
    fn import_lines_are_removed() {
        let result = resolve(
            MAIN_WITH_IMPORT,
            &[("llama_cpp/media_input.proto", IMPORTED)],
        );
        assert!(
            !result.contains("import "),
            "resolved proto must not contain import statements"
        );
    }

    #[test]
    fn first_message_is_primary() {
        let result = resolve(
            MAIN_WITH_IMPORT,
            &[("llama_cpp/media_input.proto", IMPORTED)],
        );
        let first_msg = result
            .lines()
            .find(|l| l.trim().starts_with("message "))
            .unwrap();
        assert!(
            first_msg.contains("MainMessage"),
            "first message must be the primary type from main proto, got: {first_msg}"
        );
    }

    #[test]
    fn cross_package_prefix_removed_from_type_refs() {
        let result = resolve(
            MAIN_WITH_IMPORT,
            &[("llama_cpp/media_input.proto", IMPORTED)],
        );
        assert!(
            !result
                .lines()
                .any(|l| l.trim().starts_with("llama_cpp.") || l.contains(" llama_cpp.Media")),
            "cross-package prefix should be stripped from type references"
        );
        assert!(
            result.contains("MediaInput media"),
            "type name without prefix should remain"
        );
    }

    #[test]
    fn package_declaration_preserved() {
        let result = resolve(
            MAIN_WITH_IMPORT,
            &[("llama_cpp/media_input.proto", IMPORTED)],
        );
        assert!(
            result.contains("package test_pkg;"),
            "main package declaration must be preserved"
        );
    }

    #[test]
    fn imported_syntax_and_package_stripped() {
        let result = resolve(
            MAIN_WITH_IMPORT,
            &[("llama_cpp/media_input.proto", IMPORTED)],
        );
        let syntax_count = result.matches("syntax ").count();
        assert_eq!(syntax_count, 1, "only one syntax declaration expected");
        assert!(
            !result.contains("package llama_cpp;"),
            "imported package declaration should be stripped"
        );
    }

    #[test]
    fn imported_definitions_are_present() {
        let result = resolve(
            MAIN_WITH_IMPORT,
            &[("llama_cpp/media_input.proto", IMPORTED)],
        );
        assert!(
            result.contains("message MediaInput"),
            "imported message must be inlined"
        );
        assert!(
            result.contains("enum MediaKind"),
            "imported enum must be inlined"
        );
    }

    #[test]
    fn no_import_returns_unchanged() {
        let no_import = "syntax = \"proto3\";\n\nmessage Simple { string x = 1; }";
        let result = resolve(no_import, &[]);
        assert_eq!(result, no_import);
    }

    #[test]
    fn error_on_unresolved_import() {
        let proto = "\
syntax = \"proto3\";\nimport \"missing/dep.proto\";\nmessage Foo { string x = 1; }";
        let err = resolve_proto_imports(proto, &[]).unwrap_err();
        assert!(err.contains("unresolved import"), "got: {err}");
    }

    #[test]
    fn error_on_transitive_unresolved_import() {
        let main = "\
syntax = \"proto3\";\nimport \"a.proto\";\nmessage Main { A a = 1; }";
        let a_proto = "\
syntax = \"proto3\";\npackage a;\nimport \"b.proto\";\nmessage A { string x = 1; }";
        let err = resolve_proto_imports(main, &[("a.proto", a_proto)]).unwrap_err();
        assert!(err.contains("unresolved import"), "got: {err}");
    }

    #[test]
    fn exact_path_match_no_partial() {
        let proto = "\
syntax = \"proto3\";\n\
import \"llama_cpp/media_input.proto.v2\";\n\
message Msg { string x = 1; }";
        let err =
            resolve_proto_imports(proto, &[("llama_cpp/media_input.proto", IMPORTED)]).unwrap_err();
        assert!(
            err.contains("unresolved import"),
            "partial path match should not resolve: {err}"
        );
    }

    #[test]
    fn exact_path_match_no_prefix() {
        let proto = "\
syntax = \"proto3\";\n\
import \"vendor/llama_cpp/media_input.proto\";\n\
message Msg { string x = 1; }";
        let err =
            resolve_proto_imports(proto, &[("llama_cpp/media_input.proto", IMPORTED)]).unwrap_err();
        assert!(
            err.contains("unresolved import"),
            "prefixed path should not resolve: {err}"
        );
    }

    #[test]
    fn prefix_not_removed_from_comments() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      // This field uses llama_cpp.MediaInput type.\n\
                      message Msg {\n\
                        llama_cpp.MediaInput m = 1;\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("// This field uses llama_cpp.MediaInput type"),
            "prefix in comments should not be stripped, got:\n{result}"
        );
        assert!(
            result.contains("MediaInput m = 1;"),
            "prefix in type reference should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn prefix_not_removed_from_trailing_comment() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        llama_cpp.MediaInput m = 1; // ref llama_cpp.MediaInput\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("// ref llama_cpp.MediaInput"),
            "prefix in trailing comment should not be stripped, got:\n{result}"
        );
        assert!(
            result.contains("MediaInput m = 1;"),
            "prefix in type reference should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn prefix_not_removed_from_quoted_string() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        llama_cpp.MediaInput m = 1;\n\
                        string label = 2 [default = \"llama_cpp.MediaInput\"];\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("\"llama_cpp.MediaInput\""),
            "prefix in quoted string should not be stripped, got:\n{result}"
        );
        assert!(
            result.contains("MediaInput m = 1;"),
            "prefix in type ref should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn prefix_removed_in_rpc_types() {
        let proto = "syntax = \"proto3\";\n\
                      package svc;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      service MediaService {\n\
                        rpc Process(llama_cpp.MediaInput) returns (llama_cpp.MediaInput);\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("rpc Process(MediaInput) returns (MediaInput)"),
            "prefix in rpc types should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn prefix_removed_in_map_value_type() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        map<string, llama_cpp.MediaInput> items = 1;\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("map<string, MediaInput>"),
            "prefix in map value type should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn prefix_removed_in_extend() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      extend llama_cpp.MediaInput {\n\
                        optional string ext_field = 100;\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("extend MediaInput {"),
            "prefix in extend should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn prefix_removed_with_optional_qualifier() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        optional llama_cpp.MediaInput m = 1;\n\
                        repeated llama_cpp.MediaInput ms = 2;\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("optional MediaInput m = 1;"),
            "prefix after optional should be stripped, got:\n{result}"
        );
        assert!(
            result.contains("repeated MediaInput ms = 2;"),
            "prefix after repeated should be stripped, got:\n{result}"
        );
    }

    #[test]
    fn nested_type_name_resolved() {
        let imported_with_nested = "\
syntax = \"proto3\";\n\
package outer_pkg;\n\
message Outer {\n\
  message Inner { string x = 1; }\n\
}";
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"outer.proto\";\n\
                      message Msg {\n\
                        outer_pkg.Outer.Inner nested = 1;\n\
                      }";
        let result = resolve(proto, &[("outer.proto", imported_with_nested)]);
        assert!(
            result.contains("Outer.Inner nested = 1;"),
            "nested type should have package prefix stripped, got:\n{result}"
        );
        assert!(
            !result.contains("outer_pkg.Outer.Inner"),
            "package prefix should be removed"
        );
    }

    #[test]
    fn map_type_in_comment_not_replaced() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        // map<string, llama_cpp.MediaInput> is not used here\n\
                        map<string, llama_cpp.MediaInput> items = 1;\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("// map<string, llama_cpp.MediaInput>"),
            "comment should not be modified, got:\n{result}"
        );
        assert!(
            result.contains("map<string, MediaInput>"),
            "code should have prefix stripped, got:\n{result}"
        );
    }

    #[test]
    fn rpc_type_in_trailing_comment_not_replaced() {
        let proto = "syntax = \"proto3\";\n\
                      package svc;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      service Svc {\n\
                        rpc Do(llama_cpp.MediaInput) returns (llama_cpp.MediaInput); // takes (llama_cpp.MediaInput)\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("rpc Do(MediaInput) returns (MediaInput);"),
            "rpc types should have prefix stripped, got:\n{result}"
        );
        assert!(
            result.contains("// takes (llama_cpp.MediaInput)"),
            "trailing comment should not be modified, got:\n{result}"
        );
    }

    #[test]
    fn import_public_resolved() {
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import public \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        llama_cpp.MediaInput m = 1;\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            no_import_statement(&result),
            "import public should be resolved, got:\n{result}"
        );
        assert!(
            result.contains("MediaInput m = 1;"),
            "type ref should be resolved, got:\n{result}"
        );
    }

    #[test]
    fn escaped_quote_in_string_not_confused() {
        // A string containing \" followed by // should not split at the //
        let proto = "syntax = \"proto3\";\n\
                      package test;\n\
                      import \"llama_cpp/media_input.proto\";\n\
                      message Msg {\n\
                        llama_cpp.MediaInput m = 1;\n\
                        string s = 2 [default = \"val\\\"with//slash\"];\n\
                      }";
        let result = resolve(proto, &[("llama_cpp/media_input.proto", IMPORTED)]);
        assert!(
            result.contains("MediaInput m = 1;"),
            "type ref should still be resolved, got:\n{result}"
        );
        // The string literal with escaped quotes should be preserved intact
        assert!(
            result.contains(r#"[default = "val\"with//slash"]"#),
            "escaped-quote string should be preserved, got:\n{result}"
        );
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
        assert_eq!(json["mtmd"]["mmproj"], "proj.gguf");
        assert_eq!(json["mtmd"]["allowUrlFetch"], true);
        assert_eq!(json["mtmd"]["maxMediaBytes"], 5000);
        assert_eq!(json["mtmd"]["allowedMediaDirs"][0], "/images");
    }
}
