use jobworkerp_llama_protobuf::MtmdSettings;
use std::path::PathBuf;

/// Per-request limits derived from [`MtmdSettings`].
#[derive(Debug, Clone, Default)]
pub struct MediaLimits {
    /// Whether `MediaInput.url` is allowed.
    /// Phase 1: always returns `UrlFetchDisabled` regardless of this flag.
    pub allow_url_fetch: bool,

    /// Maximum bytes per media item (compressed/on-disk). 0 means unlimited.
    pub max_media_bytes: u64,

    /// Maximum decoded bytes per media item. 0 means unlimited.
    /// Guards against decompression bombs.
    pub max_decoded_media_bytes: u64,

    /// Allowed base directories for `file_path` sources.
    /// Empty means file_path is rejected entirely.
    pub allowed_media_dirs: Vec<PathBuf>,
}

impl MediaLimits {
    pub fn from_settings(s: &MtmdSettings) -> Self {
        Self {
            allow_url_fetch: s.allow_url_fetch,
            max_media_bytes: u64::from(s.max_media_bytes),
            max_decoded_media_bytes: s.max_decoded_media_bytes,
            allowed_media_dirs: s.allowed_media_dirs.iter().map(PathBuf::from).collect(),
        }
    }

    /// Check whether a file path is allowed by the configured base directories.
    /// Returns the canonicalized path on success.
    pub fn check_file_path_allowed(&self, path: &str) -> Result<PathBuf, FilePathDenied> {
        if self.allowed_media_dirs.is_empty() {
            return Err(FilePathDenied::NotOptedIn);
        }

        let canonical = std::fs::canonicalize(path).map_err(|e| FilePathDenied::IoError {
            path: path.to_owned(),
            reason: e.to_string(),
        })?;

        for allowed in &self.allowed_media_dirs {
            let allowed_canonical = match std::fs::canonicalize(allowed) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(
                        "skipping unresolvable allowed_media_dir {}: {e}",
                        allowed.display()
                    );
                    continue;
                }
            };
            if canonical.starts_with(&allowed_canonical) {
                return Ok(canonical);
            }
        }

        Err(FilePathDenied::OutsideAllowed {
            path: canonical.display().to_string(),
        })
    }
}

/// Why a file_path was rejected.
#[derive(Debug, thiserror::Error)]
pub enum FilePathDenied {
    #[error("file_path source is disabled (allowed_media_dirs is empty)")]
    NotOptedIn,

    #[error("file path {path} is outside allowed directories")]
    OutsideAllowed { path: String },

    #[error("cannot resolve path {path}: {reason}")]
    IoError { path: String, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_settings_basic() {
        let s = MtmdSettings {
            mmproj: String::new(),
            mmproj_hf_repo: None,
            mmproj_use_gpu: None,
            media_marker: None,
            allow_url_fetch: true,
            max_media_bytes: 10_000_000,
            max_decoded_media_bytes: 100_000_000,
            allowed_media_dirs: vec!["/tmp".to_string()],
        };
        let limits = MediaLimits::from_settings(&s);
        assert!(limits.allow_url_fetch);
        assert_eq!(limits.max_media_bytes, 10_000_000);
        assert_eq!(limits.max_decoded_media_bytes, 100_000_000);
        assert_eq!(limits.allowed_media_dirs.len(), 1);
    }

    #[test]
    fn test_default_is_restrictive() {
        let limits = MediaLimits::default();
        assert!(!limits.allow_url_fetch);
        assert_eq!(limits.max_media_bytes, 0);
        assert_eq!(limits.max_decoded_media_bytes, 0);
        assert!(limits.allowed_media_dirs.is_empty());
    }

    #[test]
    fn test_file_path_denied_when_no_dirs_configured() {
        let limits = MediaLimits::default();
        let result = limits.check_file_path_allowed("/etc/passwd");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FilePathDenied::NotOptedIn));
    }

    #[test]
    fn test_file_path_allowed_within_dir() {
        // Use /tmp which should exist on all unix systems
        let limits = MediaLimits {
            allowed_media_dirs: vec![PathBuf::from("/tmp")],
            ..Default::default()
        };

        // Create a temp file for testing
        let test_file = "/tmp/mtmd_test_file_path_check";
        std::fs::write(test_file, b"test").unwrap();

        let result = limits.check_file_path_allowed(test_file);
        assert!(result.is_ok());

        std::fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_file_path_rejected_outside_dir() {
        let limits = MediaLimits {
            allowed_media_dirs: vec![PathBuf::from("/tmp")],
            ..Default::default()
        };

        // /etc/hostname should exist but is outside /tmp
        let result = limits.check_file_path_allowed("/etc/hostname");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FilePathDenied::OutsideAllowed { .. }
        ));
    }
}
