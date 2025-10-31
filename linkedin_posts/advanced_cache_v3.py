"""
🚀 Advanced Caching & Optimization Layer v3.0
=============================================

Intelligent caching strategies with predictive loading and adaptive optimization.
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Union
from collections import OrderedDict
from functools import lru_cache
import aioredis
import pickle
import zlib

class IntelligentCache:
    """Advanced intelligent caching system for v3.0."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.access_count = {}
        self.last_access = {}
        self.redis_client = None
        self.compression_enabled = True
        
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis connection."""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
            await self.redis_client.ping()
            print("✅ Redis connection established")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
    
    def _generate_key(self, content: str, strategy: str) -> str:
        """Generate cache key with content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"linkedin_opt:{strategy}:{content_hash}"
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.compression_enabled:
            return pickle.dumps(data)
        
        serialized = pickle.dumps(data)
        compressed = zlib.compress(serialized, level=6)
        return compressed
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage."""
        try:
            if compressed_data.startswith(b'\x78\x9c'):  # zlib header
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(compressed_data)
        except Exception:
            return None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent fallback."""
        # Try memory cache first
        if key in self.cache:
            value = self.cache[key]
            self._update_access(key)
            return value
        
        # Try Redis if available
        if self.redis_client:
            try:
                redis_value = await self.redis_client.get(key)
                if redis_value:
                    decompressed = self._decompress_data(redis_value)
                    if decompressed:
                        # Update memory cache
                        self._set_memory(key, decompressed)
                        return decompressed
            except Exception as e:
                print(f"Redis get failed: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with intelligent storage."""
        success = True
        
        # Set in memory cache
        self._set_memory(key, value)
        
        # Set in Redis if available
        if self.redis_client:
            try:
                compressed = self._compress_data(value)
                redis_ttl = ttl or self.ttl
                await self.redis_client.setex(key, redis_ttl, compressed)
            except Exception as e:
                print(f"Redis set failed: {e}")
                success = False
        
        return success
    
    def _set_memory(self, key: str, value: Any):
        """Set value in memory cache with LRU eviction."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict_least_used()
            
            # Add new entry
            self.cache[key] = value
        
        self._update_access(key)
    
    def _update_access(self, key: str):
        """Update access statistics."""
        current_time = time.time()
        self.access_count[key] = self.access_count.get(key, 0) + 1
        self.last_access[key] = current_time
    
    def _evict_least_used(self):
        """Evict least used items based on access patterns."""
        if not self.cache:
            return
        
        # Calculate access score (frequency + recency)
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            access_count = self.access_count.get(key, 0)
            last_access = self.last_access.get(key, current_time)
            time_factor = 1.0 / (current_time - last_access + 1)
            scores[key] = access_count * time_factor
        
        # Find least valuable item
        least_valuable = min(scores.items(), key=lambda x: x[1])[0]
        
        # Remove from cache
        del self.cache[least_valuable]
        del self.access_count[least_valuable]
        del self.last_access[least_valuable]
    
    async def get_or_set(self, key: str, getter_func, ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and store."""
        # Try to get from cache
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute value
        value = await getter_func()
        
        # Store in cache
        await self.set(key, value, ttl)
        
        return value
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # Clear memory cache entries
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.cache[key]
            del self.access_count[key]
            del self.last_access[key]
        
        # Clear Redis entries if available
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Redis pattern invalidation failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'memory_size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self._calculate_hit_rate(),
            'compression_enabled': self.compression_enabled,
            'redis_available': self.redis_client is not None
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(self.access_count.values())
        if total_accesses == 0:
            return 0.0
        return len(self.cache) / total_accesses if total_accesses > 0 else 0.0

class PredictiveCache:
    """Predictive caching with ML-based content analysis."""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.content_patterns = {}
        self.access_patterns = {}
        self.prediction_model = None
        
    async def predict_and_preload(self, content: str, strategy: str):
        """Predict and preload related content."""
        # Analyze content patterns
        content_features = self._extract_features(content)
        
        # Predict related content
        related_keys = self._predict_related_keys(content_features, strategy)
        
        # Preload predicted content
        for key in related_keys:
            if not await self.cache.get(key):
                # Trigger background preloading
                asyncio.create_task(self._preload_content(key))
    
    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract content features for prediction."""
        return {
            'length': len(content),
            'hashtag_count': content.count('#'),
            'mention_count': content.count('@'),
            'link_count': content.count('http'),
            'word_count': len(content.split()),
            'avg_word_length': sum(len(word) for word in content.split()) / len(content.split()) if content.split() else 0
        }
    
    def _predict_related_keys(self, features: Dict[str, Any], strategy: str) -> List[str]:
        """Predict related cache keys based on features."""
        # Simple similarity-based prediction
        related_keys = []
        
        # Find similar content patterns
        for pattern, pattern_features in self.content_patterns.items():
            similarity = self._calculate_similarity(features, pattern_features)
            if similarity > 0.7:  # 70% similarity threshold
                related_keys.append(f"{strategy}:{pattern}")
        
        return related_keys[:5]  # Limit to top 5 predictions
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature sets."""
        if not features1 or not features2:
            return 0.0
        
        # Normalize features
        max_values = {}
        for key in set(features1.keys()) | set(features2.keys()):
            max_values[key] = max(
                features1.get(key, 0),
                features2.get(key, 0)
            )
        
        # Calculate cosine similarity
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for key in max_values:
            val1 = features1.get(key, 0) / max_values[key] if max_values[key] > 0 else 0
            val2 = features2.get(key, 0) / max_values[key] if max_values[key] > 0 else 0
            
            dot_product += val1 * val2
            norm1 += val1 * val1
            norm2 += val2 * val2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 ** 0.5 * norm2 ** 0.5)
    
    async def _preload_content(self, key: str):
        """Preload content in background."""
        try:
            # Simulate content generation
            await asyncio.sleep(0.1)
            
            # Store placeholder in cache
            placeholder = {
                'type': 'preloaded',
                'key': key,
                'timestamp': time.time()
            }
            await self.cache.set(key, placeholder, ttl=300)  # 5 minutes TTL
            
        except Exception as e:
            print(f"Preload failed for {key}: {e}")

class AdaptiveOptimizer:
    """Adaptive optimization based on cache performance."""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.performance_history = []
        self.optimization_rules = []
        
    def add_optimization_rule(self, condition, action):
        """Add optimization rule."""
        self.optimization_rules.append((condition, action))
    
    async def optimize_cache(self):
        """Apply adaptive optimizations."""
        stats = self.cache.get_stats()
        self.performance_history.append({
            'timestamp': time.time(),
            'stats': stats
        })
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Apply optimization rules
        for condition, action in self.optimization_rules:
            if condition(stats, self.performance_history):
                await action(self.cache)
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on performance."""
        suggestions = []
        stats = self.cache.get_stats()
        
        if stats['hit_rate'] < 0.5:
            suggestions.append("Increase cache size to improve hit rate")
        
        if len(self.cache.cache) > self.cache.max_size * 0.9:
            suggestions.append("Cache is nearly full, consider increasing max_size")
        
        if not stats['redis_available']:
            suggestions.append("Enable Redis for distributed caching")
        
        return suggestions

# Usage example
async def demo_advanced_caching():
    """Demonstrate advanced caching capabilities."""
    print("🚀 Advanced Caching & Optimization Demo v3.0")
    print("=" * 50)
    
    # Initialize cache
    cache = IntelligentCache(max_size=1000, ttl=1800)
    await cache.init_redis()
    
    # Initialize predictive cache
    predictive = PredictiveCache(cache)
    
    # Initialize adaptive optimizer
    optimizer = AdaptiveOptimizer(cache)
    
    # Add optimization rules
    optimizer.add_optimization_rule(
        lambda stats, history: stats['hit_rate'] < 0.3,
        lambda cache: asyncio.create_task(cache.invalidate_pattern("old"))
    )
    
    # Test content
    test_contents = [
        "AI breakthrough in machine learning! #ai #ml #innovation",
        "Revolutionary approach to deep learning! #deeplearning #ai",
        "Transforming the future of technology! #innovation #tech #future",
        "Next-generation optimization algorithms! #optimization #algorithms #ml"
    ]
    
    strategies = ["ENGAGEMENT", "REACH", "BRAND_AWARENESS"]
    
    print("📝 Testing content optimization with caching...")
    
    for i, content in enumerate(test_contents):
        print(f"\n{i+1}. Content: {content[:50]}...")
        
        for strategy in strategies:
            # Generate cache key
            key = cache._generate_key(content, strategy)
            
            # Simulate optimization
            start_time = time.time()
            
            # Try to get from cache first
            cached_result = await cache.get(key)
            
            if cached_result:
                print(f"   ✅ {strategy}: Cached result (instant)")
            else:
                # Simulate computation
                await asyncio.sleep(0.1)
                
                # Store result
                result = {
                    'content': content,
                    'strategy': strategy,
                    'optimization_score': 85 + i * 2,
                    'confidence_score': 0.9 + i * 0.02,
                    'timestamp': time.time()
                }
                
                await cache.set(key, result)
                print(f"   🆕 {strategy}: Computed and cached")
            
            # Trigger predictive preloading
            await predictive.predict_and_preload(content, strategy)
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Memory Size: {stats['memory_size']}")
    print(f"   Hit Rate: {stats['hit_rate']:.2%}")
    print(f"   Redis Available: {stats['redis_available']}")
    
    # Get optimization suggestions
    suggestions = optimizer.get_optimization_suggestions()
    if suggestions:
        print(f"\n💡 Optimization Suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
    
    # Apply optimizations
    await optimizer.optimize_cache()
    
    print("\n🎉 Advanced caching demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_advanced_caching())
