from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import logging
from lazy_loading_exploit_modules import (
        import shutil
        import shutil
        import shutil
from typing import Any, List, Dict, Optional
"""
Lazy Loading Demo
Demonstrates lazy loading of heavy modules like exploit databases, vulnerability databases, and ML models
"""


# Import lazy loading components
    LazyModuleManager, ModuleConfig, ModuleType, LoadingStrategy,
    ExploitDatabaseModule, VulnerabilityDatabaseModule, MachineLearningModelModule
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_sample_data():
    """Create sample data files for demo"""
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Sample exploit database
    exploits_data = {
        "exploits": [
            {
                "id": "EXP-001",
                "name": "Buffer Overflow in Web Server",
                "description": "Buffer overflow vulnerability in web server allowing remote code execution",
                "cve_id": "CVE-2024-1234",
                "severity": "high",
                "affected_versions": ["1.0.0", "1.1.0"],
                "exploit_code": "print('exploit code here')"
            },
            {
                "id": "EXP-002",
                "name": "SQL Injection in Login Form",
                "description": "SQL injection vulnerability in login form allowing authentication bypass",
                "cve_id": "CVE-2024-5678",
                "severity": "medium",
                "affected_versions": ["2.0.0"],
                "exploit_code": "SELECT * FROM users WHERE id = 1 OR 1=1"
            },
            {
                "id": "EXP-003",
                "name": "XSS in Comment System",
                "description": "Cross-site scripting vulnerability in comment system",
                "cve_id": "CVE-2024-9012",
                "severity": "low",
                "affected_versions": ["1.5.0", "1.6.0"],
                "exploit_code": "<script>alert('XSS')</script>"
            }
        ]
    }
    
    # Sample vulnerability database
    vulnerabilities_data = {
        "vulnerabilities": [
            {
                "cve_id": "CVE-2024-1234",
                "title": "Buffer Overflow in Web Server",
                "description": "A buffer overflow vulnerability exists in the web server component",
                "severity": "high",
                "cvss_score": 9.8,
                "affected_products": ["WebServer Pro", "WebServer Lite"],
                "patch_available": True,
                "patch_date": "2024-01-15"
            },
            {
                "cve_id": "CVE-2024-5678",
                "title": "SQL Injection in Login Form",
                "description": "SQL injection vulnerability in the login form",
                "severity": "medium",
                "cvss_score": 7.5,
                "affected_products": ["LoginSystem v2.0"],
                "patch_available": False,
                "patch_date": None
            },
            {
                "cve_id": "CVE-2024-9012",
                "title": "XSS in Comment System",
                "description": "Cross-site scripting vulnerability in comment system",
                "severity": "low",
                "cvss_score": 4.2,
                "affected_products": ["CommentSystem v1.5"],
                "patch_available": True,
                "patch_date": "2024-02-01"
            }
        ]
    }
    
    # Write sample data files
    exploits_file = temp_dir / "exploits.json"
    vulns_file = temp_dir / "vulnerabilities.json"
    
    with open(exploits_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(exploits_data, f, indent=2)
    
    with open(vulns_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(vulnerabilities_data, f, indent=2)
    
    return temp_dir, str(exploits_file), str(vulns_file)

async def demo_lazy_loading_performance():
    """Demonstrate lazy loading performance benefits"""
    print("=== Lazy Loading Performance Demo ===\n")
    
    # Create sample data
    temp_dir, exploits_file, vulns_file = await create_sample_data()
    
    try:
        # Create module manager
        manager = LazyModuleManager()
        
        print("1. Registering modules with different loading strategies...")
        
        # Module 1: On-demand loading
        exploit_config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path=exploits_file,
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=10,
            cache_ttl=1800
        )
        await manager.register_module("exploits_ondemand", exploit_config)
        
        # Module 2: Preload loading
        vuln_config = ModuleConfig(
            module_type=ModuleType.VULNERABILITY_DATABASE,
            module_path=vulns_file,
            loading_strategy=LoadingStrategy.PRELOAD,
            cache_size=20,
            cache_ttl=3600
        )
        await manager.register_module("vulnerabilities_preload", vuln_config)
        
        # Module 3: ML model (simulated)
        ml_config = ModuleConfig(
            module_type=ModuleType.MACHINE_LEARNING_MODEL,
            module_path="models/malware_detector.pkl",
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=5,
            cache_ttl=7200
        )
        await manager.register_module("ml_model", ml_config)
        
        print("   Modules registered successfully")
        print()
        
        # Test on-demand loading
        print("2. Testing on-demand loading...")
        print("   Accessing exploit database (should trigger loading)...")
        
        start_time = time.time()
        exploit_module = await manager.get_module("exploits_ondemand")
        exploits = await exploit_module.search_exploits("buffer overflow")
        load_time = time.time() - start_time
        
        print(f"   First access: {load_time:.3f}s, found {len(exploits)} exploits")
        
        # Test cached access
        print("   Accessing exploit database again (should use cache)...")
        start_time = time.time()
        exploits2 = await exploit_module.search_exploits("sql injection")
        cache_time = time.time() - start_time
        
        print(f"   Cached access: {cache_time:.3f}s, found {len(exploits2)} exploits")
        print(f"   Performance improvement: {load_time/cache_time:.1f}x faster")
        print()
        
        # Test preloaded module
        print("3. Testing preloaded module...")
        print("   Accessing vulnerability database (should be preloaded)...")
        
        start_time = time.time()
        vuln_module = await manager.get_module("vulnerabilities_preload")
        vuln = await vuln_module.get_vulnerability("CVE-2024-1234")
        access_time = time.time() - start_time
        
        print(f"   Preloaded access: {access_time:.3f}s")
        print(f"   Vulnerability: {vuln['title'] if vuln else 'Not found'}")
        print()
        
        # Test ML model loading
        print("4. Testing ML model loading...")
        print("   Accessing ML model (should trigger loading)...")
        
        start_time = time.time()
        ml_module = await manager.get_module("ml_model")
        prediction = await ml_module.predict("suspicious_file.exe")
        ml_load_time = time.time() - start_time
        
        print(f"   ML model load time: {ml_load_time:.3f}s")
        print(f"   Prediction: {prediction}")
        print()
        
        # Test memory management
        print("5. Testing memory management...")
        
        # Get initial metrics
        initial_metrics = await manager.get_all_metrics()
        print(f"   Initial modules loaded: {len([m for m in initial_metrics.values() if m.loaded])}")
        
        # Unload a module
        print("   Unloading exploit database...")
        await manager.unload_module("exploits_ondemand")
        
        # Get final metrics
        final_metrics = await manager.get_all_metrics()
        print(f"   Modules loaded after unload: {len([m for m in final_metrics.values() if m.loaded])}")
        print()
        
        # Test error handling
        print("6. Testing error handling...")
        
        # Try to register invalid module
        try:
            invalid_config = ModuleConfig(
                module_type=ModuleType.EXPLOIT_DATABASE,
                module_path="nonexistent_file.json",
                loading_strategy=LoadingStrategy.ON_DEMAND
            )
            await manager.register_module("invalid_module", invalid_config)
            
            # Try to access it
            invalid_module = await manager.get_module("invalid_module")
            await invalid_module.get_module()
            
        except Exception as e:
            print(f"   Expected error caught: {type(e).__name__}: {e}")
        
        print()
        
        # Performance summary
        print("7. Performance Summary:")
        all_metrics = await manager.get_all_metrics()
        
        total_load_time = sum(m.load_time for m in all_metrics.values())
        total_access_count = sum(m.access_count for m in all_metrics.values())
        total_cache_hits = sum(m.cache_hits for m in all_metrics.values())
        total_cache_misses = sum(m.cache_misses for m in all_metrics.values())
        
        print(f"   Total load time: {total_load_time:.3f}s")
        print(f"   Total access count: {total_access_count}")
        print(f"   Cache hit rate: {total_cache_hits/(total_cache_hits + total_cache_misses)*100:.1f}%")
        print(f"   Average load time per module: {total_load_time/len(all_metrics):.3f}s")
        
    finally:
        # Cleanup
        await manager.cleanup_all()
        
        # Remove temporary files
        shutil.rmtree(temp_dir)
    
    print("\n=== Lazy Loading Performance Demo Completed! ===")

async def demo_lazy_loading_strategies():
    """Demonstrate different loading strategies"""
    print("=== Lazy Loading Strategies Demo ===\n")
    
    # Create sample data
    temp_dir, exploits_file, vulns_file = await create_sample_data()
    
    try:
        manager = LazyModuleManager()
        
        print("1. Testing different loading strategies...")
        
        # Strategy 1: On-demand loading
        print("   Strategy 1: On-demand loading")
        ondemand_config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path=exploits_file,
            loading_strategy=LoadingStrategy.ON_DEMAND
        )
        await manager.register_module("ondemand", ondemand_config)
        
        # Strategy 2: Preload loading
        print("   Strategy 2: Preload loading")
        preload_config = ModuleConfig(
            module_type=ModuleType.VULNERABILITY_DATABASE,
            module_path=vulns_file,
            loading_strategy=LoadingStrategy.PRELOAD
        )
        await manager.register_module("preload", preload_config)
        
        print("   Modules registered with different strategies")
        print()
        
        # Test on-demand strategy
        print("2. On-demand strategy test:")
        print("   Module not loaded yet...")
        
        ondemand_module = await manager.get_module("ondemand")
        print(f"   Module loaded: {ondemand_module._loaded}")
        
        print("   Accessing module (triggers loading)...")
        start_time = time.time()
        exploits = await ondemand_module.search_exploits("buffer")
        load_time = time.time() - start_time
        
        print(f"   Load time: {load_time:.3f}s")
        print(f"   Module loaded: {ondemand_module._loaded}")
        print()
        
        # Test preload strategy
        print("3. Preload strategy test:")
        print("   Waiting for preload to complete...")
        await asyncio.sleep(1)  # Give preload time to complete
        
        preload_module = await manager.get_module("preload")
        print(f"   Module loaded: {preload_module._loaded}")
        
        print("   Accessing module (should be instant)...")
        start_time = time.time()
        vuln = await preload_module.get_vulnerability("CVE-2024-1234")
        access_time = time.time() - start_time
        
        print(f"   Access time: {access_time:.3f}s")
        print(f"   Vulnerability found: {vuln is not None}")
        print()
        
        # Strategy comparison
        print("4. Strategy comparison:")
        print(f"   On-demand: Load time {load_time:.3f}s, memory efficient")
        print(f"   Preload: Access time {access_time:.3f}s, faster access")
        print("   On-demand is better for rarely used modules")
        print("   Preload is better for frequently used modules")
        
    finally:
        # Cleanup
        await manager.cleanup_all()
        
        # Remove temporary files
        shutil.rmtree(temp_dir)
    
    print("\n=== Lazy Loading Strategies Demo Completed! ===")

async def demo_caching_behavior():
    """Demonstrate caching behavior"""
    print("=== Caching Behavior Demo ===\n")
    
    # Create sample data
    temp_dir, exploits_file, vulns_file = await create_sample_data()
    
    try:
        manager = LazyModuleManager()
        
        print("1. Setting up module with caching...")
        
        cache_config = ModuleConfig(
            module_type=ModuleType.EXPLOIT_DATABASE,
            module_path=exploits_file,
            loading_strategy=LoadingStrategy.ON_DEMAND,
            cache_size=5,  # Small cache for demo
            cache_ttl=10   # Short TTL for demo
        )
        await manager.register_module("cached_module", cache_config)
        
        module = await manager.get_module("cached_module")
        print("   Module configured with cache size: 5, TTL: 10s")
        print()
        
        # Test cache hits and misses
        print("2. Testing cache behavior...")
        
        # First access (cache miss)
        print("   First access to exploit EXP-001 (cache miss)...")
        start_time = time.time()
        exploit1 = await module.get_exploit_by_id("EXP-001")
        first_time = time.time() - start_time
        print(f"   Time: {first_time:.3f}s, Found: {exploit1 is not None}")
        
        # Second access (cache hit)
        print("   Second access to exploit EXP-001 (cache hit)...")
        start_time = time.time()
        exploit1_cached = await module.get_exploit_by_id("EXP-001")
        second_time = time.time() - start_time
        print(f"   Time: {second_time:.3f}s, Found: {exploit1_cached is not None}")
        
        # Different exploit (cache miss)
        print("   First access to exploit EXP-002 (cache miss)...")
        start_time = time.time()
        exploit2 = await module.get_exploit_by_id("EXP-002")
        third_time = time.time() - start_time
        print(f"   Time: {third_time:.3f}s, Found: {exploit2 is not None}")
        
        print()
        
        # Test cache performance
        print("3. Cache performance analysis:")
        speedup = first_time / second_time if second_time > 0 else float('inf')
        print(f"   Cache speedup: {speedup:.1f}x faster")
        print(f"   Cache hit time: {second_time:.3f}s")
        print(f"   Cache miss time: {first_time:.3f}s")
        print()
        
        # Test cache eviction
        print("4. Testing cache eviction...")
        print("   Adding multiple exploits to trigger cache eviction...")
        
        for i in range(10):
            exploit = await module.get_exploit_by_id(f"EXP-00{i+1}")
            print(f"   Added exploit {i+1} to cache")
        
        # Check cache size
        cache_size = len(module._cache)
        print(f"   Final cache size: {cache_size} (should be <= 5)")
        print()
        
        # Test TTL expiration
        print("5. Testing TTL expiration...")
        print("   Waiting for cache TTL to expire (10s)...")
        await asyncio.sleep(11)
        
        print("   Accessing exploit after TTL expiration...")
        start_time = time.time()
        exploit_expired = await module.get_exploit_by_id("EXP-001")
        expired_time = time.time() - start_time
        print(f"   Time after TTL: {expired_time:.3f}s (should be similar to cache miss)")
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"   Cache hits: {metrics.cache_hits}")
        print(f"   Cache misses: {metrics.cache_misses}")
        hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) * 100
        print(f"   Cache hit rate: {hit_rate:.1f}%")
        
    finally:
        # Cleanup
        await manager.cleanup_all()
        
        # Remove temporary files
        shutil.rmtree(temp_dir)
    
    print("\n=== Caching Behavior Demo Completed! ===")

async def main():
    """Run all lazy loading demos"""
    print("Lazy Loading System Demo")
    print("=" * 50)
    
    # Run performance demo
    await demo_lazy_loading_performance()
    print("\n" + "=" * 50 + "\n")
    
    # Run strategies demo
    await demo_lazy_loading_strategies()
    print("\n" + "=" * 50 + "\n")
    
    # Run caching demo
    await demo_caching_behavior()
    
    print("\n" + "=" * 50)
    print("All Lazy Loading Demos Completed!")

match __name__:
    case "__main__":
    asyncio.run(main()) 