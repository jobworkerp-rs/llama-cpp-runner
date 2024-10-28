use crate::error::MtmdError;

/// Count media markers in `prompt` and reconcile with `n_media`.
///
/// - Marker count == n_media → return prompt unchanged (owned copy).
/// - Marker count == 0 and n_media > 0 → append markers at the end.
/// - Anything else → `MarkerMismatch` error.
pub fn inject_markers(prompt: &str, marker: &str, n_media: usize) -> Result<String, MtmdError> {
    let count = prompt.matches(marker).count();

    match (count, n_media) {
        (c, m) if c == m => Ok(prompt.to_owned()),
        (0, m) if m > 0 => {
            let mut s = String::with_capacity(prompt.len() + m * (marker.len() + 1));
            s.push_str(prompt);
            for i in 0..m {
                // Only insert a newline separator when there is preceding
                // content (non-empty prompt, or a previous marker).
                if i > 0 || !prompt.is_empty() {
                    s.push('\n');
                }
                s.push_str(marker);
            }
            Ok(s)
        }
        (c, m) => Err(MtmdError::MarkerMismatch {
            markers: c,
            medias: m,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MARKER: &str = "<__media__>";

    #[test]
    fn test_markers_match_count() {
        let prompt = "Look at <__media__> and <__media__>";
        let result = inject_markers(prompt, MARKER, 2).unwrap();
        assert_eq!(result, prompt);
    }

    #[test]
    fn test_no_markers_appends() {
        let prompt = "Describe this image";
        let result = inject_markers(prompt, MARKER, 2).unwrap();
        assert_eq!(result, "Describe this image\n<__media__>\n<__media__>");
    }

    #[test]
    fn test_no_markers_no_media_passes() {
        let prompt = "Plain text prompt";
        let result = inject_markers(prompt, MARKER, 0).unwrap();
        assert_eq!(result, prompt);
    }

    #[test]
    fn test_marker_count_mismatch_errors() {
        let prompt = "One <__media__> two <__media__> three <__media__>";
        let result = inject_markers(prompt, MARKER, 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            MtmdError::MarkerMismatch {
                markers: 3,
                medias: 2,
            } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_single_media_no_marker() {
        let prompt = "What is in the picture?";
        let result = inject_markers(prompt, MARKER, 1).unwrap();
        assert_eq!(result, "What is in the picture?\n<__media__>");
    }

    #[test]
    fn test_custom_marker() {
        let marker = "<image>";
        let prompt = "Look at <image>";
        let result = inject_markers(prompt, marker, 1).unwrap();
        assert_eq!(result, prompt);
    }

    #[test]
    fn test_some_markers_but_too_few() {
        let prompt = "Here <__media__>";
        let result = inject_markers(prompt, MARKER, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_text_single_media() {
        let result = inject_markers("", MARKER, 1).unwrap();
        assert_eq!(result, "<__media__>");
    }

    #[test]
    fn test_empty_text_multiple_media() {
        let result = inject_markers("", MARKER, 2).unwrap();
        assert_eq!(result, "<__media__>\n<__media__>");
    }
}
