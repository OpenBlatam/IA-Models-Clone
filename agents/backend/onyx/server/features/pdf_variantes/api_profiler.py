#!/usr/bin/env python3
"""
API Profiler
============
Performance profiling tool for the API.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with profiling capabilities
"""
import warnings

warnings.warn(
    "api_profiler.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import requests
from contextlib import contextmanager


class APIProfiler:
    """API performance profiler."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.profiles: Dict[str, cProfile.Profile] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def profile_endpoint(self, name: str, method: str = "GET", endpoint: str = "/health", **kwargs):
        """Profile an endpoint call."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", **kwargs)
            elif method.upper() == "POST":
                response = requests.post(f"{self.base_url}{endpoint}", **kwargs)
            elif method.upper() == "PUT":
                response = requests.put(f"{self.base_url}{endpoint}", **kwargs)
            elif method.upper() == "DELETE":
                response = requests.delete(f"{self.base_url}{endpoint}", **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            elapsed_time = time.time() - start_time
            
            profiler.disable()
            
            # Get stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            self.profiles[name] = profiler
            self.results[name] = {
                "endpoint": endpoint,
                "method": method,
                "status": response.status_code,
                "elapsed_time": elapsed_time,
                "stats": stats_stream.getvalue(),
                "timestamp": datetime.now().isoformat()
            }
            
            yield response
            
        except Exception as e:
            profiler.disable()
            self.results[name] = {
                "endpoint": endpoint,
                "method": method,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            raise
    
    def profile_multiple(self, endpoints: List[Dict[str, Any]], iterations: int = 10):
        """Profile multiple endpoints."""
        for endpoint_config in endpoints:
            name = endpoint_config.get("name", endpoint_config.get("endpoint"))
            method = endpoint_config.get("method", "GET")
            endpoint = endpoint_config.get("endpoint", "/health")
            
            print(f"Profiling {method} {endpoint} ({iterations} iterations)...")
            
            times = []
            for i in range(iterations):
                with self.profile_endpoint(f"{name}_{i}", method, endpoint):
                    times.append(self.results[f"{name}_{i}"]["elapsed_time"])
            
            avg_time = sum(times) / len(times)
            print(f"  Average time: {avg_time:.3f}s")
    
    def get_profile_stats(self, name: str) -> str:
        """Get profile statistics for a named profile."""
        if name not in self.profiles:
            return f"No profile found for: {name}"
        
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiles[name], stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats()
        
        return stats_stream.getvalue()
    
    def save_profile(self, name: str, file_path: Path):
        """Save profile to file."""
        if name not in self.profiles:
            print(f"❌ No profile found for: {name}")
            return
        
        self.profiles[name].dump_stats(str(file_path))
        print(f"✅ Profile saved to {file_path}")
    
    def save_results(self, file_path: Path):
        """Save profiling results to file."""
        import json
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                k: v for k, v in result.items()
                if k != "stats"  # Skip stats for JSON
            }
            # Save stats separately if needed
            if "stats" in result:
                stats_file = file_path.parent / f"{name}_stats.txt"
                stats_file.write_text(result["stats"])
        
        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to {file_path}")
    
    def print_summary(self):
        """Print profiling summary."""
        print("\n" + "=" * 60)
        print("📊 Profiling Summary")
        print("=" * 60)
        
        for name, result in self.results.items():
            if "error" in result:
                print(f"❌ {name}: Error - {result['error']}")
            else:
                print(f"✅ {name}:")
                print(f"   Endpoint: {result['method']} {result['endpoint']}")
                print(f"   Status: {result['status']}")
                print(f"   Time: {result['elapsed_time']:.3f}s")
        
        print("=" * 60)


def main():
    """Main entry point."""
    import sys
    
    profiler = APIProfiler()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            with profiler.profile_endpoint("health_check", "GET", "/health"):
                pass
            profiler.print_summary()
        
        elif sys.argv[1] == "endpoint" and len(sys.argv) > 3:
            method = sys.argv[2]
            endpoint = sys.argv[3]
            name = sys.argv[4] if len(sys.argv) > 4 else endpoint.replace("/", "_")
            
            with profiler.profile_endpoint(name, method, endpoint):
                pass
            
            print(profiler.get_profile_stats(name))
            profiler.print_summary()
        
        else:
            print("Usage:")
            print("  python api_profiler.py health                    - Profile health endpoint")
            print("  python api_profiler.py endpoint GET /health      - Profile specific endpoint")
    else:
        # Interactive mode
        print("=" * 60)
        print("📊 API Profiler")
        print("=" * 60)
        print("Profiling health endpoint...")
        
        with profiler.profile_endpoint("health_check", "GET", "/health"):
            pass
        
        profiler.print_summary()
        print("\nTop functions:")
        print(profiler.get_profile_stats("health_check"))


if __name__ == "__main__":
    main()



