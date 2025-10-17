use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cache for reranking results
pub struct RerankerCache {
    cache: Arc<RwLock<LruCache<String, CachedScore>>>,
    ttl: Duration,
}

#[derive(Clone)]
struct CachedScore {
    score: f32,
    timestamp: Instant,
}

impl RerankerCache {
    /// Create a new cache with specified capacity and TTL
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        let cache = LruCache::new(NonZeroUsize::new(capacity).unwrap());
        Self {
            cache: Arc::new(RwLock::new(cache)),
            ttl,
        }
    }

    /// Get cached score if available and not expired
    pub async fn get(&self, key: &str) -> Option<f32> {
        let mut cache_guard = self.cache.write().await;
        if let Some(cached) = cache_guard.get(key) {
            // Check TTL
            if cached.timestamp.elapsed() < self.ttl {
                Some(cached.score)
            } else {
                // Remove expired entry
                cache_guard.pop(key);
                None
            }
        } else {
            None
        }
    }

    /// Store score in cache
    pub async fn put(&self, key: String, score: f32) {
        let mut cache_guard = self.cache.write().await;
        cache_guard.put(
            key,
            CachedScore {
                score,
                timestamp: Instant::now(),
            },
        );
    }

    /// Get cached value or compute and store new value
    pub async fn get_or_compute<F, Fut>(&self, key: &str, compute_fn: F) -> anyhow::Result<f32>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<f32>>,
    {
        // First, try to get from cache
        if let Some(cached_score) = self.get(key).await {
            return Ok(cached_score);
        }

        // Not in cache or expired, compute new value
        let score = compute_fn().await?;

        // Store in cache
        self.put(key.to_string(), score).await;

        Ok(score)
    }

    /// Clear all cached entries
    pub async fn clear(&mut self) {
        let mut cache_guard = self.cache.write().await;
        cache_guard.clear();
    }

    /// Get cache statistics
    pub async fn stats(&self) -> (usize, usize) {
        let cache_guard = self.cache.read().await;
        (cache_guard.len(), cache_guard.cap().get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = RerankerCache::new(10, Duration::from_secs(60));

        // Test put and get
        cache.put("key1".to_string(), 0.8).await;
        assert_eq!(cache.get("key1").await, Some(0.8));

        // Test miss
        assert_eq!(cache.get("key2").await, None);

        // Test stats
        let (size, capacity) = cache.stats().await;
        assert_eq!(size, 1);
        assert_eq!(capacity, 10);
    }

    #[tokio::test]
    async fn test_cache_ttl_expiration() {
        let cache = RerankerCache::new(10, Duration::from_millis(100));

        // Store value
        cache.put("key1".to_string(), 0.8).await;

        // Within TTL: should hit
        assert_eq!(cache.get("key1").await, Some(0.8));

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // After TTL: should miss (expired entry removed)
        assert_eq!(cache.get("key1").await, None);

        // Cache size should be 0 after expiration
        let (size, _) = cache.stats().await;
        assert_eq!(size, 0);
    }

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let cache = RerankerCache::new(3, Duration::from_secs(60));

        // Fill cache to capacity
        cache.put("key1".to_string(), 0.1).await;
        cache.put("key2".to_string(), 0.2).await;
        cache.put("key3".to_string(), 0.3).await;

        let (size, _) = cache.stats().await;
        assert_eq!(size, 3);

        // Add one more to trigger LRU eviction
        cache.put("key4".to_string(), 0.4).await;

        // key1 (least recently used) should be evicted
        assert_eq!(cache.get("key1").await, None);
        assert_eq!(cache.get("key4").await, Some(0.4));

        let (size, _) = cache.stats().await;
        assert_eq!(size, 3);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let mut cache = RerankerCache::new(10, Duration::from_secs(60));

        // Add entries
        cache.put("key1".to_string(), 0.1).await;
        cache.put("key2".to_string(), 0.2).await;

        let (size, _) = cache.stats().await;
        assert_eq!(size, 2);

        // Clear cache
        cache.clear().await;

        // All entries should be gone
        let (size, _) = cache.stats().await;
        assert_eq!(size, 0);
        assert_eq!(cache.get("key1").await, None);
    }

    #[tokio::test]
    async fn test_cache_get_or_compute() {
        let cache = RerankerCache::new(10, Duration::from_secs(60));

        // First call: compute
        let mut call_count = 0;
        let result = cache
            .get_or_compute("key1", || async {
                call_count += 1;
                Ok(0.5)
            })
            .await
            .unwrap();
        assert_eq!(result, 0.5);
        assert_eq!(call_count, 1);

        // Second call: cache hit (compute function not called)
        let result = cache
            .get_or_compute("key1", || async {
                call_count += 1;
                Ok(0.9)
            })
            .await
            .unwrap();
        assert_eq!(result, 0.5); // Cached value
        assert_eq!(call_count, 1); // Not incremented
    }
}
