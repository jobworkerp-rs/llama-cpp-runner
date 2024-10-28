use jobworkerp_llama_protobuf::MediaKind;

/// Errors produced by the multimodal runtime layer.
///
/// Every variant that relates to a specific media item carries the `index`
/// field so callers can pinpoint the offending input.
#[derive(thiserror::Error, Debug)]
pub enum MtmdError {
    #[error("marker/media count mismatch (markers={markers}, medias={medias})")]
    MarkerMismatch { markers: usize, medias: usize },

    #[error("url fetch disabled for medias[{index}]")]
    UrlFetchDisabled { index: usize },

    #[error("media too large at medias[{index}]: {actual} bytes > {limit} byte limit")]
    MediaTooLarge {
        index: usize,
        actual: u64,
        limit: u64,
    },

    #[error("{kind:?} not supported by mmproj at medias[{index}]")]
    UnsupportedKindForModel { index: usize, kind: MediaKind },

    #[error("resampling required at medias[{index}]: expected={expected} Hz, given={given} Hz")]
    ResamplingRequired {
        index: usize,
        expected: u32,
        given: u32,
    },

    #[error("image decode failed at medias[{index}]: {reason}")]
    ImageDecodeFailed { index: usize, reason: String },

    #[error("audio decode failed at medias[{index}]: {reason}")]
    AudioDecodeFailed { index: usize, reason: String },

    #[error("kind/source mismatch at medias[{index}]")]
    KindSourceMismatch { index: usize },

    #[error("kind={declared:?} but decoded as {actual} at medias[{index}]")]
    KindContentMismatch {
        index: usize,
        declared: MediaKind,
        actual: &'static str,
    },

    #[error("kind must be specified for medias[{index}]")]
    KindUnspecified { index: usize },

    #[error("id contains nul byte at medias[{index}]")]
    IdHasNulByte { index: usize },

    #[error("only mono PCM supported (got {channels} channels) at medias[{index}]")]
    InvalidAudioChannels { index: usize, channels: u32 },

    #[error("instruction must not contain media marker")]
    InstructionContainsMarker,

    #[error("mtmd init failed: {0}")]
    MtmdInit(String),

    #[error("tokenize failed: {0}")]
    Tokenize(String),

    #[error("eval_chunks failed: {0}")]
    Eval(String),

    #[error("source field is not set for medias[{index}]")]
    SourceNotSet { index: usize },

    #[error("file_path denied for medias[{index}]: {reason}")]
    FilePathDenied { index: usize, reason: String },

    #[error("decoded media too large at medias[{index}]: {actual} bytes > {limit} byte limit")]
    DecodedMediaTooLarge {
        index: usize,
        actual: u64,
        limit: u64,
    },
}
