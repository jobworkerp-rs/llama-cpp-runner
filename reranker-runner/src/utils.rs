/// Generate cache key for query-document pair
pub fn generate_cache_key(query: &str, document: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    document.hash(&mut hasher);

    format!("rerank_{:x}", hasher.finish())
}

/// Blend original score with reranking score
pub fn blend_scores(original: f32, reranking: f32, blend_ratio: f32) -> f32 {
    let ratio = blend_ratio.clamp(0.0, 1.0);
    original * (1.0 - ratio) + reranking * ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_scores() {
        // Test pure original score
        assert_eq!(blend_scores(0.8, 0.3, 0.0), 0.8);

        // Test pure reranking score
        assert_eq!(blend_scores(0.8, 0.3, 1.0), 0.3);

        // Test blended score
        let blended = blend_scores(0.8, 0.4, 0.5);
        assert_eq!(blended, 0.6); // (0.8 * 0.5) + (0.4 * 0.5)
    }

    #[test]
    fn test_generate_cache_key() {
        let key1 = generate_cache_key("query1", "doc1");
        let key2 = generate_cache_key("query1", "doc1");
        let key3 = generate_cache_key("query2", "doc1");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert!(key1.starts_with("rerank_"));
    }
}
