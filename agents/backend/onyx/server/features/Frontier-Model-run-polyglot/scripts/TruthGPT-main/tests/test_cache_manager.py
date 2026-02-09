"""
Intelligent Test Cache Manager
Caches test results and skips unchanged tests
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import subprocess


class TestCacheManager:
    """Manages intelligent caching of test results"""
    
    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".test_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "test_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _get_test_dependencies(self, test_file: Path, project_root: Path) -> Set[Path]:
        """Get files that a test depends on"""
        dependencies = set()
        
        try:
            import ast
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        # Try to find corresponding file
                        module_parts = node.module.split('.')
                        potential_path = project_root.parent / Path(*module_parts)
                        potential_path = potential_path.with_suffix('.py')
                        
                        if potential_path.exists():
                            dependencies.add(potential_path)
                        else:
                            # Try __init__.py
                            init_path = potential_path.parent / "__init__.py"
                            if init_path.exists():
                                dependencies.add(init_path)
        except Exception as e:
            print(f"Warning: Could not parse dependencies for {test_file}: {e}")
        
        # Always include the test file itself
        dependencies.add(test_file)
        return dependencies
    
    def get_test_hash(self, test_file: Path, project_root: Path) -> str:
        """Calculate hash for test and its dependencies"""
        dependencies = self._get_test_dependencies(test_file, project_root)
        
        hashes = []
        for dep in sorted(dependencies):
            if dep.exists():
                hashes.append(f"{dep}:{self._get_file_hash(dep)}")
        
        combined = "\n".join(hashes)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def is_test_cached(self, test_file: Path, project_root: Path) -> Tuple[bool, Optional[Dict]]:
        """Check if test result is cached and valid"""
        test_key = str(test_file.relative_to(project_root))
        current_hash = self.get_test_hash(test_file, project_root)
        
        if test_key in self.cache:
            cached_data = self.cache[test_key]
            if cached_data.get('hash') == current_hash:
                # Check if cache is still valid (not too old)
                cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                # Cache valid for 24 hours by default
                max_age = cached_data.get('max_age_hours', 24)
                if age_hours < max_age:
                    return True, cached_data
        
        return False, None
    
    def cache_test_result(
        self,
        test_file: Path,
        project_root: Path,
        status: str,
        duration: float,
        output: str = "",
        max_age_hours: int = 24
    ):
        """Cache test result"""
        test_key = str(test_file.relative_to(project_root))
        test_hash = self.get_test_hash(test_file, project_root)
        
        self.cache[test_key] = {
            'hash': test_hash,
            'status': status,
            'duration': duration,
            'output': output[:1000],  # Limit output size
            'timestamp': datetime.now().isoformat(),
            'max_age_hours': max_age_hours
        }
        
        self._save_cache()
    
    def get_cached_tests(self, test_files: List[Path], project_root: Path) -> Tuple[List[Path], List[Path]]:
        """Separate tests into cached (skip) and uncached (run)"""
        cached = []
        uncached = []
        
        for test_file in test_files:
            is_cached, _ = self.is_test_cached(test_file, project_root)
            if is_cached:
                cached.append(test_file)
            else:
                uncached.append(test_file)
        
        return cached, uncached
    
    def invalidate_cache(self, test_file: Path = None, project_root: Path = None):
        """Invalidate cache for specific test or all tests"""
        if test_file and project_root:
            test_key = str(test_file.relative_to(project_root))
            if test_key in self.cache:
                del self.cache[test_key]
                self._save_cache()
                print(f"✅ Invalidated cache for {test_key}")
        else:
            # Clear all cache
            self.cache = {}
            self._save_cache()
            print("✅ Cleared all test cache")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = len(self.cache)
        if total == 0:
            return {
                'total_cached': 0,
                'total_size_mb': 0,
                'oldest_cache': None,
                'newest_cache': None
            }
        
        timestamps = [
            datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            for data in self.cache.values()
        ]
        
        cache_size = self.cache_file.stat().st_size / (1024 * 1024)  # MB
        
        return {
            'total_cached': total,
            'total_size_mb': round(cache_size, 2),
            'oldest_cache': min(timestamps).isoformat(),
            'newest_cache': max(timestamps).isoformat()
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Cache Manager')
    parser.add_argument('--clear', action='store_true', help='Clear all cache')
    parser.add_argument('--invalidate', type=str, help='Invalidate cache for specific test')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--cache-dir', type=str, help='Cache directory')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    manager = TestCacheManager(cache_dir)
    
    if args.clear:
        manager.invalidate_cache()
    elif args.invalidate:
        project_root = Path(__file__).parent
        test_file = project_root / args.invalidate
        manager.invalidate_cache(test_file, project_root)
    elif args.stats:
        stats = manager.get_cache_stats()
        print("📊 Cache Statistics:")
        print(f"  Total Cached Tests: {stats['total_cached']}")
        print(f"  Cache Size: {stats['total_size_mb']} MB")
        print(f"  Oldest Cache: {stats['oldest_cache']}")
        print(f"  Newest Cache: {stats['newest_cache']}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

