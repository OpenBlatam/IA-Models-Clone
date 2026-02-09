"""
Test Result Cache
Caches test results for faster access and comparison
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pickle

class TestResultCache:
    """Cache for test results"""
    
    def __init__(self, cache_dir: Path = None, ttl_hours: int = 24):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".test_cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.cache"
    
    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache entry is expired"""
        if not cache_path.exists():
            return True
        
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists() or self._is_expired(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data.get('value')
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        cache_path = self._get_cache_path(key)
        
        cache_data = {
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"⚠️  Failed to cache {key}: {e}")
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self):
        """Clear all cache entries"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        expired = sum(1 for f in cache_files if self._is_expired(f))
        
        return {
            'total_entries': len(cache_files),
            'expired_entries': expired,
            'active_entries': len(cache_files) - expired,
            'total_size_mb': total_size / (1024 * 1024)
        }

class TestResultCacheManager:
    """Manager for test result caching"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache = TestResultCache(cache_dir)
    
    def cache_test_run(self, test_results: Dict, run_name: str):
        """Cache test run results"""
        cache_key = f"test_run:{run_name}"
        self.cache.set(cache_key, test_results)
    
    def get_cached_run(self, run_name: str) -> Optional[Dict]:
        """Get cached test run"""
        cache_key = f"test_run:{run_name}"
        return self.cache.get(cache_key)
    
    def cache_comparison(self, run1: str, run2: str, comparison: Dict):
        """Cache comparison result"""
        cache_key = f"comparison:{run1}:{run2}"
        self.cache.set(cache_key, comparison)
    
    def get_cached_comparison(self, run1: str, run2: str) -> Optional[Dict]:
        """Get cached comparison"""
        cache_key = f"comparison:{run1}:{run2}"
        return self.cache.get(cache_key)
    
    def cache_statistics(self, stats: Dict):
        """Cache statistics"""
        self.cache.set("statistics", stats)
    
    def get_cached_statistics(self) -> Optional[Dict]:
        """Get cached statistics"""
        return self.cache.get("statistics")

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    manager = TestResultCacheManager(project_root / ".test_cache")
    
    # Cache test results
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'success_rate': 98.0
    }
    
    manager.cache_test_run(test_results, "test_run_001")
    print("✅ Cached test run")
    
    # Retrieve from cache
    cached = manager.get_cached_run("test_run_001")
    if cached:
        print(f"✅ Retrieved from cache: {cached['total_tests']} tests")
    
    # Cache statistics
    stats = manager.cache.get_stats()
    print(f"\n📊 Cache Stats:")
    print(f"  Total Entries: {stats['total_entries']}")
    print(f"  Active Entries: {stats['active_entries']}")
    print(f"  Size: {stats['total_size_mb']:.2f} MB")

if __name__ == "__main__":
    main()







