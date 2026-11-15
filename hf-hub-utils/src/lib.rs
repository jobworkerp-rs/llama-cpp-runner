//! Shared Hugging Face Hub download helpers with cache-integrity checks.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, CacheRepo};

/// Download or reuse a Hugging Face Hub file.
///
/// A warm cache reuses only a regular snapshot file without a network request.
/// Other cache entries are rejected. On a cache miss, this helper validates the
/// downloaded file size using remote metadata. The `hf-hub` crate already
/// resumes `<etag>.part` files.
pub fn download_hf_file(repo: &str, filename: &str) -> Result<PathBuf> {
    let repo_id = repo.to_string();
    let cache = Cache::from_env();
    let api = api_builder_from_cache_and_env(cache.clone())
        .build()
        .context("unable to create huggingface api")?;
    let api_repo = api.model(repo_id.clone());
    let cache_repo = cache.model(repo_id);

    download_hf_file_with_ops(
        &cache,
        &cache_repo,
        repo,
        filename,
        || {
            api.metadata(&api_repo.url(filename))
                .map(|metadata| RemoteMetadata {
                    commit_hash: metadata.commit_hash().to_string(),
                    etag: metadata.etag().to_string(),
                    size: metadata.size() as u64,
                })
                .map_err(anyhow::Error::from)
        },
        || api_repo.get(filename).map_err(anyhow::Error::from),
        || api_repo.download(filename).map_err(anyhow::Error::from),
    )
}

fn api_builder_from_cache_and_env(cache: Cache) -> ApiBuilder {
    let mut builder = ApiBuilder::from_cache(cache).with_progress(false);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        builder = builder.with_endpoint(endpoint);
    }
    builder
}

#[derive(Debug)]
struct RemoteMetadata {
    commit_hash: String,
    etag: String,
    size: u64,
}

fn download_hf_file_with_ops(
    cache: &Cache,
    cache_repo: &CacheRepo,
    repo: &str,
    filename: &str,
    mut metadata: impl FnMut() -> Result<RemoteMetadata>,
    mut get: impl FnMut() -> Result<PathBuf>,
    mut download: impl FnMut() -> Result<PathBuf>,
) -> Result<PathBuf> {
    if let Some(cached) = cache_repo.get(filename) {
        return validate_warm_cache_file(cache, cache_repo, repo, filename, cached);
    }

    let metadata = match metadata() {
        Ok(metadata) => metadata,
        Err(err) => {
            tracing::warn!(
                repo,
                filename,
                error = %err,
                "unable to fetch huggingface metadata; falling back to hf-hub cache lookup"
            );
            return get().with_context(|| {
                format!("unable to use cached or download huggingface file: {repo}/{filename}")
            });
        }
    };

    let pointer_path = pointer_path(cache_repo, &metadata.commit_hash, filename);
    if cached_file_has_size(&pointer_path, metadata.size)? {
        return Ok(pointer_path);
    }

    remove_incomplete_cache_entries(cache_repo, &metadata.commit_hash, &metadata.etag, filename)?;

    let downloaded = download()
        .with_context(|| format!("unable to download huggingface file: {repo}/{filename}"))?;
    if let Err(error) = ensure_file_size(&downloaded, metadata.size) {
        remove_incomplete_cache_entries(
            cache_repo,
            &metadata.commit_hash,
            &metadata.etag,
            filename,
        )?;
        return Err(error);
    }
    Ok(downloaded)
}

fn pointer_path(cache_repo: &CacheRepo, commit_hash: &str, filename: &str) -> PathBuf {
    let mut path = cache_repo.pointer_path(commit_hash);
    path.push(filename);
    path
}

fn blob_part_path(cache_repo: &CacheRepo, etag: &str) -> PathBuf {
    let mut path = cache_repo.blob_path(etag);
    path.set_extension("part");
    path
}

fn validate_warm_cache_file(
    cache: &Cache,
    cache_repo: &CacheRepo,
    repo: &str,
    filename: &str,
    cached: PathBuf,
) -> Result<PathBuf> {
    let cache_root = cache
        .path()
        .canonicalize()
        .context("unable to resolve huggingface cache root")?;
    let resolved = cached.canonicalize().with_context(|| {
        format!(
            "unable to resolve warm huggingface cache entry for {repo}/{filename}: {}",
            cached.display()
        )
    })?;

    let snapshot_root = cache_repo
        .pointer_path("")
        .canonicalize()
        .with_context(|| {
            format!("unable to resolve huggingface snapshot root for {repo}/{filename}")
        })?;
    let blob_root = cache_repo.blob_path("").canonicalize().ok();

    if !resolved.starts_with(&cache_root) {
        anyhow::bail!(
            "rejecting warm huggingface cache entry outside cache root for {repo}/{filename}: {}",
            cached.display()
        );
    }
    let belongs_to_repo = resolved.starts_with(&snapshot_root)
        || blob_root
            .as_ref()
            .is_some_and(|blob_root| resolved.starts_with(blob_root));
    if !belongs_to_repo {
        anyhow::bail!(
            "rejecting warm huggingface cache entry outside repository cache for {repo}/{filename}: {}",
            cached.display()
        );
    }
    if !resolved.is_file() {
        anyhow::bail!(
            "rejecting non-regular warm huggingface cache entry for {repo}/{filename}: {}",
            cached.display()
        );
    }
    Ok(cached)
}

#[derive(Debug, PartialEq, Eq)]
enum RegularFileSize {
    Matches,
    Mismatch { actual_size: u64 },
}

fn regular_file_size(path: &Path, expected_size: u64) -> std::io::Result<RegularFileSize> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() && metadata.len() == expected_size {
        Ok(RegularFileSize::Matches)
    } else {
        Ok(RegularFileSize::Mismatch {
            actual_size: metadata.len(),
        })
    }
}

fn cached_file_has_size(path: &Path, expected_size: u64) -> Result<bool> {
    match regular_file_size(path, expected_size) {
        Ok(RegularFileSize::Matches) => Ok(true),
        Ok(RegularFileSize::Mismatch { .. }) => Ok(false),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => {
            Err(err).with_context(|| format!("unable to inspect cached file: {}", path.display()))
        }
    }
}

fn remove_incomplete_cache_entries(
    cache_repo: &CacheRepo,
    commit_hash: &str,
    etag: &str,
    filename: &str,
) -> Result<()> {
    remove_file_or_symlink_if_present(&pointer_path(cache_repo, commit_hash, filename))?;

    let blob_path = cache_repo.blob_path(etag);
    if blob_path.exists() {
        // Repairing a size-mismatched cache is this helper's expected job, so
        // this is a routine info-level event, not an unexpected warning.
        tracing::info!(
            path = %blob_path.display(),
            "removing incomplete huggingface blob before retrying download"
        );
        remove_file_or_symlink_if_present(&blob_path)?;
    }

    let part_path = blob_part_path(cache_repo, etag);
    if part_path.exists() {
        tracing::debug!(
            path = %part_path.display(),
            "keeping huggingface partial download for resume"
        );
    }
    Ok(())
}

fn remove_file_or_symlink_if_present(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() || metadata.file_type().is_symlink() => {
            fs::remove_file(path)
                .with_context(|| format!("unable to remove cached file: {}", path.display()))?;
            Ok(())
        }
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => {
            Err(err).with_context(|| format!("unable to inspect cached file: {}", path.display()))
        }
    }
}

fn ensure_file_size(path: &Path, expected_size: u64) -> Result<()> {
    match regular_file_size(path, expected_size)
        .with_context(|| format!("unable to inspect downloaded file: {}", path.display()))?
    {
        RegularFileSize::Matches => Ok(()),
        RegularFileSize::Mismatch { actual_size } => {
            anyhow::bail!(
                "downloaded file size mismatch for {}: expected {}, got {}",
                path.display(),
                expected_size,
                actual_size
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn cache_repo(tempdir: &tempfile::TempDir) -> (Cache, CacheRepo) {
        let cache = Cache::new(tempdir.path().to_path_buf());
        let repo = cache.model("org/model".to_string());
        (cache, repo)
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
        let mut file = fs::File::create(path).expect("create file");
        file.write_all(bytes).expect("write file");
    }

    fn with_env_var<T>(key: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let previous = std::env::var_os(key);
        match value {
            Some(value) => unsafe { std::env::set_var(key, value) },
            None => unsafe { std::env::remove_var(key) },
        }

        let result = f();

        match previous {
            Some(previous) => unsafe { std::env::set_var(key, previous) },
            None => unsafe { std::env::remove_var(key) },
        }
        result
    }

    #[test]
    fn api_builder_from_cache_and_env_preserves_hf_endpoint() {
        with_env_var(
            "HF_ENDPOINT",
            Some("https://hf-mirror.example.test"),
            || {
                let tempdir = tempfile::tempdir().expect("tempdir");
                let cache = Cache::new(tempdir.path().to_path_buf());
                let api = api_builder_from_cache_and_env(cache)
                    .build()
                    .expect("build api");

                let url = api.model("org/model".to_string()).url("model.gguf");

                assert_eq!(
                    url,
                    "https://hf-mirror.example.test/org/model/resolve/main/model.gguf"
                );
            },
        );
    }

    #[test]
    fn download_hf_file_with_ops_returns_a_warm_cache_entry_without_remote_calls() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        write_file(&pointer, b"complete");
        repo.create_ref("commit").expect("create ref");

        let resolved = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || panic!("metadata must not be called for warm cache"),
            || panic!("get must not be called for warm cache"),
            || panic!("download must not be called for warm cache"),
        )
        .expect("resolve");

        assert_eq!(resolved, pointer);
    }

    #[test]
    fn download_hf_file_with_ops_rejects_a_directory_warm_cache_without_remote_calls() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        fs::create_dir_all(&pointer).expect("create directory pointer");
        repo.create_ref("commit").expect("create ref");

        let error = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || panic!("metadata must not be called for invalid warm cache"),
            || panic!("get must not be called for invalid warm cache"),
            || panic!("download must not be called for invalid warm cache"),
        )
        .expect_err("directory warm cache must be rejected");

        assert!(format!("{error:#}").contains("non-regular"));
        assert!(pointer.is_dir(), "directory pointer must not be removed");
    }

    #[test]
    fn download_hf_file_with_ops_rejects_an_external_file_from_an_absolute_ref() {
        let cache_tempdir = tempfile::tempdir().expect("cache tempdir");
        let external_tempdir = tempfile::tempdir().expect("external tempdir");
        let (cache, repo) = cache_repo(&cache_tempdir);
        fs::create_dir_all(repo.pointer_path("")).expect("create snapshot root");
        let external_file = external_tempdir.path().join("model.gguf");
        write_file(&external_file, b"secret");
        repo.create_ref(external_tempdir.path().to_str().expect("utf-8 path"))
            .expect("create absolute ref");

        let error = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || panic!("metadata must not be called for external warm cache"),
            || panic!("get must not be called for external warm cache"),
            || panic!("download must not be called for external warm cache"),
        )
        .expect_err("external warm cache must be rejected");

        assert!(format!("{error:#}").contains("cache root"));
        assert_eq!(
            fs::read(&external_file).expect("read external file"),
            b"secret"
        );
    }

    #[test]
    fn download_hf_file_with_ops_rejects_another_repositorys_file_from_an_absolute_ref() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        fs::create_dir_all(repo.pointer_path("")).expect("create snapshot root");
        let other_repo = cache.model("other/model".to_string());
        let other_pointer = pointer_path(&other_repo, "commit", "model.gguf");
        write_file(&other_pointer, b"other model");
        let other_snapshot = other_repo.pointer_path("commit");
        repo.create_ref(other_snapshot.to_str().expect("utf-8 path"))
            .expect("create absolute ref");

        let error = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || panic!("metadata must not be called for another repository cache"),
            || panic!("get must not be called for another repository cache"),
            || panic!("download must not be called for another repository cache"),
        )
        .expect_err("another repository cache must be rejected");

        assert!(format!("{error:#}").contains("repository cache"));
        assert_eq!(
            fs::read(&other_pointer).expect("read other repository file"),
            b"other model"
        );
    }

    #[test]
    fn download_hf_file_with_ops_rejects_an_external_directory_from_an_absolute_ref() {
        let cache_tempdir = tempfile::tempdir().expect("cache tempdir");
        let external_tempdir = tempfile::tempdir().expect("external tempdir");
        let (cache, repo) = cache_repo(&cache_tempdir);
        fs::create_dir_all(repo.pointer_path("")).expect("create snapshot root");
        let external_directory = external_tempdir.path().join("model.gguf");
        fs::create_dir(&external_directory).expect("create external directory");
        repo.create_ref(external_tempdir.path().to_str().expect("utf-8 path"))
            .expect("create absolute ref");

        let error = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || panic!("metadata must not be called for external warm cache"),
            || panic!("get must not be called for external warm cache"),
            || panic!("download must not be called for external warm cache"),
        )
        .expect_err("external warm cache must be rejected");

        assert!(format!("{error:#}").contains("cache root"));
        assert!(
            external_directory.is_dir(),
            "external directory must not be removed"
        );
    }

    #[test]
    fn download_hf_file_with_ops_removes_an_invalid_download_from_the_cache() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        let blob = repo.blob_path("etag");

        let error = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || {
                Ok(RemoteMetadata {
                    commit_hash: "commit".to_string(),
                    etag: "etag".to_string(),
                    size: 8,
                })
            },
            || panic!("get fallback must not be called when metadata succeeds"),
            || {
                write_file(&pointer, b"short");
                write_file(&blob, b"short");
                repo.create_ref("commit").expect("create ref");
                Ok(pointer.clone())
            },
        )
        .expect_err("invalid download must fail");

        assert!(error.to_string().contains("downloaded file size mismatch"));
        assert!(!pointer.exists(), "invalid pointer must be removed");
        assert!(!blob.exists(), "invalid blob must be removed");
    }

    #[test]
    fn download_hf_file_with_ops_downloads_when_pointer_absent() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        let downloaded = tempdir.path().join("downloaded.gguf");
        write_file(&downloaded, b"complete");

        let resolved = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || {
                Ok(RemoteMetadata {
                    commit_hash: "commit".to_string(),
                    etag: "etag".to_string(),
                    size: 8,
                })
            },
            || panic!("get fallback must not be called when metadata succeeds"),
            || Ok(downloaded.clone()),
        )
        .expect("download");

        assert_eq!(resolved, downloaded);
    }

    /// The crate's reason to exist: a snapshot pointer whose blob size no longer
    /// matches the remote metadata is treated as a cache miss, removed, and
    /// re-downloaded rather than served as-is.
    #[test]
    fn download_hf_file_with_ops_repairs_size_mismatched_cache() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        // `create_ref` would make `cache_repo.get` short-circuit the warm path,
        // so leave the ref absent: the stale pointer must be reached through the
        // metadata/size-check branch, not the warm-cache lookup.
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        let blob = repo.blob_path("etag");
        write_file(&pointer, b"stale"); // 5 bytes — mismatches remote size 8
        write_file(&blob, b"stale");
        let downloaded = tempdir.path().join("downloaded.gguf");
        write_file(&downloaded, b"complete"); // 8 bytes — matches remote size

        let resolved = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || {
                Ok(RemoteMetadata {
                    commit_hash: "commit".to_string(),
                    etag: "etag".to_string(),
                    size: 8,
                })
            },
            || panic!("get fallback must not be called when metadata succeeds"),
            || Ok(downloaded.clone()),
        )
        .expect("repair-and-download");

        assert_eq!(resolved, downloaded);
        // The mismatched pointer and blob were removed before re-download.
        assert!(!pointer.exists(), "stale pointer must be removed");
        assert!(!blob.exists(), "stale blob must be removed");
    }

    /// A cached pointer whose size already matches the remote metadata is served
    /// directly without invoking the download closure.
    #[test]
    fn download_hf_file_with_ops_reuses_size_matched_pointer() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        write_file(&pointer, b"complete"); // 8 bytes — matches remote size

        let resolved = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || {
                Ok(RemoteMetadata {
                    commit_hash: "commit".to_string(),
                    etag: "etag".to_string(),
                    size: 8,
                })
            },
            || panic!("get fallback must not be called when metadata succeeds"),
            || panic!("download must not be called when the cached size matches"),
        )
        .expect("reuse");

        assert_eq!(resolved, pointer);
    }

    #[test]
    fn download_hf_file_with_ops_falls_back_to_get_when_metadata_fails() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (cache, repo) = cache_repo(&tempdir);
        let fallback = tempdir.path().join("fallback.gguf");
        write_file(&fallback, b"cached");

        let resolved = download_hf_file_with_ops(
            &cache,
            &repo,
            "org/model",
            "model.gguf",
            || anyhow::bail!("offline"),
            || Ok(fallback.clone()),
            || panic!("download must not be called from metadata fallback"),
        )
        .expect("fallback");

        assert_eq!(resolved, fallback);
    }

    #[test]
    fn cached_file_has_size_accepts_complete_file() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("model.gguf");
        write_file(&path, b"abcd");

        assert!(cached_file_has_size(&path, 4).expect("inspect"));
        assert!(!cached_file_has_size(&path, 3).expect("inspect"));
    }

    #[test]
    fn cached_file_has_size_treats_missing_file_as_absent() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("missing.gguf");

        assert!(!cached_file_has_size(&path, 4).expect("inspect"));
    }

    #[test]
    fn cached_file_has_size_rejects_a_directory() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("model.gguf");
        fs::create_dir(&path).expect("create directory");

        assert!(!cached_file_has_size(&path, 0).expect("inspect"));
    }

    #[test]
    fn ensure_file_size_accepts_a_regular_file_of_the_expected_size() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("model.gguf");
        write_file(&path, b"abcd");

        ensure_file_size(&path, 4).expect("validate");
    }

    #[test]
    fn ensure_file_size_rejects_a_mismatched_regular_file() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("model.gguf");
        write_file(&path, b"abc");

        let error = ensure_file_size(&path, 4).expect_err("size mismatch");

        assert!(error.to_string().contains("expected 4, got 3"));
    }

    #[test]
    fn ensure_file_size_rejects_a_directory() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("model.gguf");
        fs::create_dir(&path).expect("create directory");

        let error = ensure_file_size(&path, 0).expect_err("directory mismatch");

        assert!(error.to_string().contains("downloaded file size mismatch"));
    }

    #[test]
    fn ensure_file_size_preserves_missing_file_context() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join("missing.gguf");

        let error = ensure_file_size(&path, 4).expect_err("missing file");

        assert!(
            error
                .to_string()
                .contains("unable to inspect downloaded file")
        );
    }

    #[test]
    fn remove_incomplete_cache_entries_removes_pointer_and_blob_but_keeps_part() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (_cache, repo) = cache_repo(&tempdir);
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        let blob = repo.blob_path("etag");
        let part = blob_part_path(&repo, "etag");
        write_file(&pointer, b"bad");
        write_file(&blob, b"bad");
        write_file(&part, b"partial");

        remove_incomplete_cache_entries(&repo, "commit", "etag", "model.gguf").expect("remove");

        assert!(!pointer.exists());
        assert!(!blob.exists());
        assert!(part.exists());
    }

    #[cfg(unix)]
    #[test]
    fn remove_incomplete_cache_entries_removes_broken_pointer_symlink() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let (_cache, repo) = cache_repo(&tempdir);
        let pointer = pointer_path(&repo, "commit", "model.gguf");
        fs::create_dir_all(pointer.parent().expect("parent")).expect("create parent");
        std::os::unix::fs::symlink("missing-blob", &pointer).expect("symlink");

        remove_incomplete_cache_entries(&repo, "commit", "etag", "model.gguf").expect("remove");

        assert!(fs::symlink_metadata(pointer).is_err());
    }
}
