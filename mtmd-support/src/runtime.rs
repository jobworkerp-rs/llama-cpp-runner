use std::ffi::CString;

use llama_cpp_2::{
    context::LlamaContext,
    model::LlamaModel,
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
};

use jobworkerp_llama_protobuf::protobuf::llama_cpp::{MediaInput, MediaKind, media_input};

use crate::{error::MtmdError, limits::MediaLimits, marker::inject_markers};

/// Thin runtime wrapper around [`MtmdContext`] that provides validated
/// bitmap construction, marker injection, and prefill in a single call
/// sequence.
///
/// Designed to be shared between `llama-cpp-plugin` (generation) and
/// `embedding-llm` (embedding). Thread-safety: `MtmdContext` is `Send +
/// Sync` but `eval_chunks` is not safe for concurrent invocation on the
/// same context. Callers must synchronize externally (e.g. via `Mutex`).
pub struct MtmdRuntime {
    ctx: MtmdContext,
    marker: String,
    support_vision: bool,
    support_audio: bool,
    audio_sample_rate: Option<u32>,
}

impl MtmdRuntime {
    /// Create a runtime from an mmproj file path and the already-loaded
    /// text model.
    pub fn from_settings(
        mmproj_path: &str,
        model: &LlamaModel,
        use_gpu: bool,
        media_marker: Option<&str>,
    ) -> Result<Self, MtmdError> {
        let marker = media_marker
            .unwrap_or(llama_cpp_2::mtmd::mtmd_default_marker())
            .to_owned();

        let params = MtmdContextParams {
            use_gpu,
            print_timings: false,
            n_threads: 0, // auto-detect
            media_marker: CString::new(marker.as_str())
                .map_err(|e| MtmdError::MtmdInit(format!("invalid media_marker: {e}")))?,
        };

        let ctx = MtmdContext::init_from_file(mmproj_path, model, &params)
            .map_err(|e| MtmdError::MtmdInit(e.to_string()))?;

        let support_vision = ctx.support_vision();
        let support_audio = ctx.support_audio();
        let audio_sample_rate = ctx.get_audio_sample_rate();

        tracing::info!(
            "MtmdRuntime initialized: vision={support_vision}, audio={support_audio}, \
             audio_sample_rate={audio_sample_rate:?}, marker={marker:?}"
        );

        Ok(Self {
            ctx,
            marker,
            support_vision,
            support_audio,
            audio_sample_rate,
        })
    }

    // -- Accessors -----------------------------------------------------------

    #[must_use]
    pub fn support_vision(&self) -> bool {
        self.support_vision
    }

    #[must_use]
    pub fn support_audio(&self) -> bool {
        self.support_audio
    }

    #[must_use]
    pub fn audio_sample_rate(&self) -> Option<u32> {
        self.audio_sample_rate
    }

    #[must_use]
    pub fn media_marker(&self) -> &str {
        &self.marker
    }

    // -- Bitmap preparation --------------------------------------------------

    /// Convert a slice of protobuf [`MediaInput`] into validated
    /// [`MtmdBitmap`] instances.
    ///
    /// Performs all validation described in the specification error matrix
    /// (section 5): kind checks, capability checks, size limits, kind/source
    /// mismatch detection, and post-decode `is_audio()` cross-validation.
    pub fn prepare_bitmaps(
        &self,
        inputs: &[MediaInput],
        limits: &MediaLimits,
    ) -> Result<Vec<MtmdBitmap>, MtmdError> {
        let mut bitmaps = Vec::with_capacity(inputs.len());

        for (i, input) in inputs.iter().enumerate() {
            // 1. kind validation
            let kind = MediaKind::try_from(input.kind).unwrap_or(MediaKind::Unspecified);
            if kind == MediaKind::Unspecified {
                return Err(MtmdError::KindUnspecified { index: i });
            }
            if kind == MediaKind::Image && !self.support_vision {
                return Err(MtmdError::UnsupportedKindForModel { index: i, kind });
            }
            if kind == MediaKind::Audio && !self.support_audio {
                return Err(MtmdError::UnsupportedKindForModel { index: i, kind });
            }

            // 2. source validation + bitmap creation
            let source = input
                .source
                .as_ref()
                .ok_or(MtmdError::SourceNotSet { index: i })?;

            let bitmap = match source {
                media_input::Source::RawImage(raw) => {
                    if kind != MediaKind::Image {
                        return Err(MtmdError::KindSourceMismatch { index: i });
                    }
                    self.check_raw_size(i, (raw.nx as u64) * (raw.ny as u64) * 3, limits)?;
                    MtmdBitmap::from_image_data(raw.nx, raw.ny, &raw.rgb).map_err(|e| {
                        MtmdError::ImageDecodeFailed {
                            index: i,
                            reason: e.to_string(),
                        }
                    })?
                }

                media_input::Source::RawAudio(raw) => {
                    if kind != MediaKind::Audio {
                        return Err(MtmdError::KindSourceMismatch { index: i });
                    }
                    if raw.channels != 1 {
                        return Err(MtmdError::InvalidAudioChannels {
                            index: i,
                            channels: raw.channels,
                        });
                    }
                    if let Some(expected) = self.audio_sample_rate
                        && raw.sample_rate_hz != expected
                    {
                        return Err(MtmdError::ResamplingRequired {
                            index: i,
                            expected,
                            given: raw.sample_rate_hz,
                        });
                    }
                    self.check_raw_size(i, (raw.samples.len() as u64) * 4, limits)?;
                    MtmdBitmap::from_audio_data(&raw.samples).map_err(|e| {
                        MtmdError::AudioDecodeFailed {
                            index: i,
                            reason: e.to_string(),
                        }
                    })?
                }

                media_input::Source::Encoded(bytes) => {
                    self.check_encoded_size(i, bytes.len() as u64, limits)?;
                    let bitmap = MtmdBitmap::from_buffer(&self.ctx, bytes)
                        .map_err(|e| self.decode_error(i, kind, &e))?;
                    self.verify_kind_matches_decoded(i, kind, &bitmap)?;
                    self.check_decoded_size(i, &bitmap, limits)?;
                    bitmap
                }

                media_input::Source::FilePath(path) => {
                    // Validate path against allowed directories
                    let canonical = limits.check_file_path_allowed(path).map_err(|e| {
                        MtmdError::FilePathDenied {
                            index: i,
                            reason: e.to_string(),
                        }
                    })?;

                    // Open once, stat via fd, then read — eliminates TOCTOU
                    // between size check and file read.
                    let io_err = |reason: String| {
                        if kind == MediaKind::Audio {
                            MtmdError::AudioDecodeFailed { index: i, reason }
                        } else {
                            MtmdError::ImageDecodeFailed { index: i, reason }
                        }
                    };
                    let file = std::fs::File::open(&canonical)
                        .map_err(|e| io_err(format!("cannot open {}: {e}", canonical.display())))?;
                    let file_len = file
                        .metadata()
                        .map_err(|e| io_err(format!("cannot stat {}: {e}", canonical.display())))?
                        .len();

                    self.check_encoded_size(i, file_len, limits)?;

                    let mut bytes = Vec::with_capacity(file_len as usize);
                    std::io::Read::read_to_end(&mut &file, &mut bytes)
                        .map_err(|e| io_err(format!("cannot read {}: {e}", canonical.display())))?;

                    let bitmap = MtmdBitmap::from_buffer(&self.ctx, &bytes)
                        .map_err(|e| self.decode_error(i, kind, &e))?;
                    self.verify_kind_matches_decoded(i, kind, &bitmap)?;
                    self.check_decoded_size(i, &bitmap, limits)?;
                    bitmap
                }

                media_input::Source::Url(_) => {
                    // Phase 1: always reject URL sources regardless of
                    // allow_url_fetch setting (HTTP client not linked).
                    return Err(MtmdError::UrlFetchDisabled { index: i });
                }
            };

            // 3. set optional ID for KV-cache reuse
            if let Some(id) = &input.id {
                bitmap
                    .set_id(id)
                    .map_err(|_| MtmdError::IdHasNulByte { index: i })?;
            }

            bitmaps.push(bitmap);
        }

        Ok(bitmaps)
    }

    // -- Marker injection (delegates to marker module) -----------------------

    /// Insert or validate media markers in the prompt text.
    pub fn inject_markers(&self, prompt: &str, n_media: usize) -> Result<String, MtmdError> {
        inject_markers(prompt, &self.marker, n_media)
    }

    // -- Tokenize + prefill --------------------------------------------------

    /// Tokenize text+bitmaps and prefill the KV cache via `eval_chunks`.
    ///
    /// Returns the new `n_past` position (= number of KV slots consumed).
    pub fn tokenize_and_prefill(
        &self,
        llama_ctx: &mut LlamaContext,
        prompt: &str,
        bitmaps: &[MtmdBitmap],
        n_batch: i32,
    ) -> Result<i32, MtmdError> {
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let input_text = MtmdInputText {
            text: prompt.to_owned(),
            add_special: true,
            parse_special: true,
        };

        let chunks: MtmdInputChunks = self
            .ctx
            .tokenize(input_text, &bitmap_refs)
            .map_err(|e| MtmdError::Tokenize(e.to_string()))?;

        let n_past = chunks
            .eval_chunks(
                &self.ctx, llama_ctx, /* n_past = */ 0, /* seq_id = */ 0, n_batch,
                /* logits_last = */ true,
            )
            .map_err(|e| MtmdError::Eval(e.to_string()))?;

        tracing::debug!(
            "tokenize_and_prefill: total_positions={}, n_past={n_past}",
            chunks.total_positions()
        );

        Ok(n_past)
    }

    // -- Internal helpers ----------------------------------------------------

    fn check_raw_size(
        &self,
        index: usize,
        byte_count: u64,
        limits: &MediaLimits,
    ) -> Result<(), MtmdError> {
        if limits.max_media_bytes > 0 && byte_count > limits.max_media_bytes {
            return Err(MtmdError::MediaTooLarge {
                index,
                actual: byte_count,
                limit: limits.max_media_bytes,
            });
        }
        Ok(())
    }

    fn check_encoded_size(
        &self,
        index: usize,
        byte_count: u64,
        limits: &MediaLimits,
    ) -> Result<(), MtmdError> {
        if limits.max_media_bytes > 0 && byte_count > limits.max_media_bytes {
            return Err(MtmdError::MediaTooLarge {
                index,
                actual: byte_count,
                limit: limits.max_media_bytes,
            });
        }
        Ok(())
    }

    fn check_decoded_size(
        &self,
        index: usize,
        bitmap: &MtmdBitmap,
        limits: &MediaLimits,
    ) -> Result<(), MtmdError> {
        if limits.max_decoded_media_bytes > 0 {
            let decoded_bytes = bitmap.data().len() as u64;
            if decoded_bytes > limits.max_decoded_media_bytes {
                return Err(MtmdError::DecodedMediaTooLarge {
                    index,
                    actual: decoded_bytes,
                    limit: limits.max_decoded_media_bytes,
                });
            }
        }
        Ok(())
    }

    /// Verify that the declared `kind` matches what llama.cpp actually
    /// decoded (only for encoded/file_path paths where auto-detection
    /// applies).
    fn verify_kind_matches_decoded(
        &self,
        index: usize,
        kind: MediaKind,
        bitmap: &MtmdBitmap,
    ) -> Result<(), MtmdError> {
        let is_audio = bitmap.is_audio();
        match (kind, is_audio) {
            (MediaKind::Image, false) | (MediaKind::Audio, true) => Ok(()),
            (MediaKind::Image, true) => Err(MtmdError::KindContentMismatch {
                index,
                declared: kind,
                actual: "audio",
            }),
            (MediaKind::Audio, false) => Err(MtmdError::KindContentMismatch {
                index,
                declared: kind,
                actual: "image",
            }),
            _ => Ok(()), // UNSPECIFIED already filtered earlier
        }
    }

    fn decode_error(
        &self,
        index: usize,
        kind: MediaKind,
        err: &llama_cpp_2::mtmd::MtmdBitmapError,
    ) -> MtmdError {
        if kind == MediaKind::Audio {
            MtmdError::AudioDecodeFailed {
                index,
                reason: err.to_string(),
            }
        } else {
            MtmdError::ImageDecodeFailed {
                index,
                reason: err.to_string(),
            }
        }
    }
}
