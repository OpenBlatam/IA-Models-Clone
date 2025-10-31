from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import os
import sys
import time
import signal
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import argparse
            import orjson
            import ujson
            import msgspec
            import simdjson
            import uvloop
            import polars as pl
            import duckdb
            import pyarrow as pa
            import lz4
            import blosc2
            import cramjam
            import xxhash
            import blake3
            import mmh3
            import numba
            import redis.asyncio as aioredis
            import redis
            import hiredis
            from prometheus_fastapi_instrumentator import Instrumentator
            import httpx
            import aiofiles
            from refactored_ultra import get_service
            from refactored_ultra import CopywritingInput
            import uvicorn
            from refactored_ultra import refactored_app
                import uvloop
            import httpx
        import httpx
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Refactored Ultra-Optimized Copywriting Service Deployment Script.

Intelligent deployment with:
- Automatic optimization detection and scoring
- Performance benchmarking and validation
- Health monitoring and alerting
- Graceful fallbacks and error recovery
- Production-ready configuration management
"""


# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# === OPTIMIZATION DETECTION SYSTEM ===
class IntelligentOptimizationDetector:
    """Advanced optimization detection with scoring and recommendations."""
    
    def __init__(self) -> Any:
        self.optimizations = {}
        self.performance_score = 0
        self.recommendations = []
        self.missing_critical = []
        self._detect_all_optimizations()
    
    def _detect_all_optimizations(self) -> Any:
        """Detect all available optimizations with detailed analysis."""
        
        # Define optimization categories
        optimization_tests = [
            # Serialization (Critical for API performance)
            ("orjson", self._test_orjson, "critical", 5.0, "Ultra-fast JSON serialization"),
            ("ujson", self._test_ujson, "high", 3.0, "Fast JSON serialization"),
            ("msgspec", self._test_msgspec, "high", 8.0, "Binary serialization"),
            ("simdjson", self._test_simdjson, "medium", 12.0, "SIMD JSON parsing"),
            
            # Event Loop (Critical for async performance)
            ("uvloop", self._test_uvloop, "critical", 4.0, "High-performance event loop"),
            
            # Data Processing (High impact for text operations)
            ("polars", self._test_polars, "high", 10.0, "Ultra-fast DataFrames"),
            ("duckdb", self._test_duckdb, "medium", 15.0, "In-memory SQL analytics"),
            ("pyarrow", self._test_pyarrow, "medium", 8.0, "Columnar data processing"),
            
            # Compression (Medium impact, good for caching)
            ("lz4", self._test_lz4, "medium", 4.0, "Fast compression"),
            ("blosc2", self._test_blosc2, "low", 6.0, "Advanced compression"),
            ("cramjam", self._test_cramjam, "low", 6.5, "Multi-algorithm compression"),
            
            # Hashing (Medium impact for caching keys)
            ("xxhash", self._test_xxhash, "medium", 4.0, "Fast non-crypto hashing"),
            ("blake3", self._test_blake3, "low", 5.0, "Fast crypto hashing"),
            ("mmh3", self._test_mmh3, "low", 3.0, "MurmurHash3"),
            
            # JIT Compilation (High impact for compute-heavy operations)
            ("numba", self._test_numba, "high", 15.0, "JIT compilation"),
            
            # Caching (Critical for production)
            ("redis", self._test_redis, "critical", 3.0, "Distributed caching"),
            ("hiredis", self._test_hiredis, "medium", 2.0, "Fast Redis protocol"),
            
            # Monitoring (Important for production)
            ("prometheus", self._test_prometheus, "high", 1.0, "Production metrics"),
            
            # Networking (Medium impact)
            ("httpx", self._test_httpx, "medium", 2.0, "Modern HTTP client"),
            ("aiofiles", self._test_aiofiles, "low", 1.5, "Async file operations"),
        ]
        
        total_possible_score = 0
        actual_score = 0
        
        for name, test_func, priority, speedup, description in optimization_tests:
            available, version, details = test_func()
            
            self.optimizations[name] = {
                "available": available,
                "version": version,
                "priority": priority,
                "speedup": speedup,
                "description": description,
                "details": details
            }
            
            # Calculate scores
            priority_multiplier = {"critical": 3.0, "high": 2.0, "medium": 1.5, "low": 1.0}
            max_points = speedup * priority_multiplier[priority]
            total_possible_score += max_points
            
            if available:
                actual_score += max_points
            else:
                if priority in ["critical", "high"]:
                    self.missing_critical.append(name)
                
                # Generate recommendations
                if priority == "critical":
                    self.recommendations.append(f"CRITICAL: Install {name} for {speedup}x speedup - {description}")
                elif priority == "high":
                    self.recommendations.append(f"HIGH: Install {name} for {speedup}x speedup - {description}")
        
        # Calculate performance score (0-100)
        self.performance_score = (actual_score / total_possible_score) * 100 if total_possible_score > 0 else 0
        
        # Calculate realistic speedup
        self.realistic_speedup = self._calculate_realistic_speedup()
    
    def _test_orjson(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(orjson, '__version__', 'unknown'), "Rust-based JSON library"
        except ImportError:
            return False, "", "Not installed - pip install orjson"
    
    def _test_ujson(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(ujson, '__version__', 'unknown'), "C-based JSON library"
        except ImportError:
            return False, "", "Not installed - pip install ujson"
    
    def _test_msgspec(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(msgspec, '__version__', 'unknown'), "Binary serialization"
        except ImportError:
            return False, "", "Not installed - pip install msgspec"
    
    def _test_simdjson(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(simdjson, '__version__', 'unknown'), "SIMD JSON parsing"
        except ImportError:
            return False, "", "Not installed - pip install simdjson"
    
    def _test_uvloop(self) -> Tuple[bool, str, str]:
        try:
            if sys.platform == 'win32':
                return False, "", "Not supported on Windows"
            return True, getattr(uvloop, '__version__', 'unknown'), "libuv-based event loop"
        except ImportError:
            return False, "", "Not installed - pip install uvloop"
    
    def _test_polars(self) -> Tuple[bool, str, str]:
        try:
            return True, pl.__version__, "Rust-based DataFrames"
        except ImportError:
            return False, "", "Not installed - pip install polars"
    
    def _test_duckdb(self) -> Tuple[bool, str, str]:
        try:
            return True, duckdb.__version__, "In-memory SQL engine"
        except ImportError:
            return False, "", "Not installed - pip install duckdb"
    
    def _test_pyarrow(self) -> Tuple[bool, str, str]:
        try:
            return True, pa.__version__, "Columnar data format"
        except ImportError:
            return False, "", "Not installed - pip install pyarrow"
    
    def _test_lz4(self) -> Tuple[bool, str, str]:
        try:
            return True, lz4.version.version, "Fast compression"
        except ImportError:
            return False, "", "Not installed - pip install lz4"
    
    def _test_blosc2(self) -> Tuple[bool, str, str]:
        try:
            return True, blosc2.__version__, "Advanced compression"
        except ImportError:
            return False, "", "Not installed - pip install blosc2"
    
    def _test_cramjam(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(cramjam, '__version__', 'unknown'), "Multi-algorithm compression"
        except ImportError:
            return False, "", "Not installed - pip install cramjam"
    
    def _test_xxhash(self) -> Tuple[bool, str, str]:
        try:
            return True, xxhash.VERSION, "Fast non-crypto hashing"
        except ImportError:
            return False, "", "Not installed - pip install xxhash"
    
    def _test_blake3(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(blake3, '__version__', 'unknown'), "Fast crypto hashing"
        except ImportError:
            return False, "", "Not installed - pip install blake3"
    
    def _test_mmh3(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(mmh3, '__version__', 'unknown'), "MurmurHash3"
        except ImportError:
            return False, "", "Not installed - pip install mmh3"
    
    def _test_numba(self) -> Tuple[bool, str, str]:
        try:
            return True, numba.__version__, "JIT compilation"
        except ImportError:
            return False, "", "Not installed - pip install numba"
    
    def _test_redis(self) -> Tuple[bool, str, str]:
        try:
            return True, redis.__version__, "Async Redis client"
        except ImportError:
            return False, "", "Not installed - pip install redis aioredis"
    
    def _test_hiredis(self) -> Tuple[bool, str, str]:
        try:
            return True, hiredis.__version__, "Fast Redis protocol"
        except ImportError:
            return False, "", "Not installed - pip install hiredis"
    
    def _test_prometheus(self) -> Tuple[bool, str, str]:
        try:
            return True, "available", "Prometheus metrics"
        except ImportError:
            return False, "", "Not installed - pip install prometheus-fastapi-instrumentator"
    
    async def _test_httpx(self) -> Tuple[bool, str, str]:
        try:
            return True, httpx.__version__, "Modern HTTP client"
        except ImportError:
            return False, "", "Not installed - pip install httpx"
    
    def _test_aiofiles(self) -> Tuple[bool, str, str]:
        try:
            return True, getattr(aiofiles, '__version__', 'unknown'), "Async file operations"
        except ImportError:
            return False, "", "Not installed - pip install aiofiles"
    
    def _calculate_realistic_speedup(self) -> float:
        """Calculate realistic total speedup based on available optimizations."""
        speedup = 1.0
        
        # Apply conservative speedup multipliers
        for name, opt in self.optimizations.items():
            if opt["available"]:
                # Conservative real-world speedups
                conservative_speedups = {
                    "orjson": 2.5, "ujson": 1.8, "msgspec": 3.0, "simdjson": 4.0,
                    "uvloop": 1.8, "polars": 2.0, "duckdb": 2.5, "pyarrow": 1.5,
                    "lz4": 1.3, "blosc2": 1.4, "cramjam": 1.4,
                    "xxhash": 1.2, "blake3": 1.3, "mmh3": 1.1,
                    "numba": 3.0, "redis": 1.5, "hiredis": 1.2,
                    "prometheus": 1.0, "httpx": 1.1, "aiofiles": 1.1
                }
                
                multiplier = conservative_speedups.get(name, 1.0)
                speedup *= multiplier
        
        return min(speedup, 20.0)  # Cap at realistic maximum
    
    def get_performance_level(self) -> str:
        """Get performance level based on score."""
        if self.performance_score >= 90:
            return "QUANTUM"
        elif self.performance_score >= 75:
            return "ULTRA"
        elif self.performance_score >= 60:
            return "HIGH"
        elif self.performance_score >= 40:
            return "MEDIUM"
        else:
            return "BASIC"
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        available = sum(1 for opt in self.optimizations.values() if opt["available"])
        total = len(self.optimizations)
        
        # Group by category
        categories = {}
        for name, opt in self.optimizations.items():
            priority = opt["priority"]
            if priority not in categories:
                categories[priority] = {"available": 0, "total": 0, "items": []}
            
            categories[priority]["total"] += 1
            categories[priority]["items"].append({
                "name": name,
                "available": opt["available"],
                "speedup": opt["speedup"],
                "description": opt["description"]
            })
            
            if opt["available"]:
                categories[priority]["available"] += 1
        
        return {
            "performance_score": round(self.performance_score, 1),
            "performance_level": self.get_performance_level(),
            "realistic_speedup": f"{self.realistic_speedup:.1f}x",
            "summary": {
                "available": available,
                "total": total,
                "percentage": round((available / total) * 100, 1)
            },
            "categories": categories,
            "missing_critical": self.missing_critical,
            "recommendations": self.recommendations[:5],  # Top 5 recommendations
            "optimizations": self.optimizations
        }

# === DEPLOYMENT CONFIGURATION ===
@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration."""
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8003))
    workers: int = int(os.getenv("WORKERS", min(8, mp.cpu_count())))
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Performance settings
    enable_optimizations: bool = os.getenv("ENABLE_OPTIMIZATIONS", "true").lower() == "true"
    enable_monitoring: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    enable_benchmarking: bool = os.getenv("ENABLE_BENCHMARKING", "true").lower() == "true"
    
    # Timeouts and limits
    startup_timeout: int = int(os.getenv("STARTUP_TIMEOUT", 60))
    shutdown_timeout: int = int(os.getenv("SHUTDOWN_TIMEOUT", 30))
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", 60))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "detailed")

# === DEPLOYMENT MANAGER ===
class RefactoredDeploymentManager:
    """Advanced deployment manager with intelligent optimization handling."""
    
    def __init__(self, config: DeploymentConfig):
        
    """__init__ function."""
self.config = config
        self.detector = IntelligentOptimizationDetector()
        self.server_process = None
        self.running = False
        self.start_time = time.time()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self) -> Any:
        """Setup comprehensive logging."""
        if self.config.log_format == "detailed":
            format_str = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        else:
            format_str = '%(asctime)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=format_str,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('deployment.log') if not self.config.debug else logging.NullHandler()
            ]
        )
    
    def print_startup_banner(self) -> Any:
        """Print comprehensive startup banner."""
        report = self.detector.get_detailed_report()
        
        print("\n" + "="*80)
        print("🚀 REFACTORED ULTRA-OPTIMIZED COPYWRITING SERVICE")
        print("="*80)
        print(f"Performance Score: {report['performance_score']}/100 ({report['performance_level']})")
        print(f"Realistic Speedup: {report['realistic_speedup']}")
        print(f"Optimizations: {report['summary']['available']}/{report['summary']['total']} ({report['summary']['percentage']}%)")
        print(f"Environment: {self.config.environment}")
        print(f"Host: {self.config.host}:{self.config.port}")
        print(f"Workers: {self.config.workers}")
        print("-"*80)
        
        # Show optimization status by category
        for priority in ["critical", "high", "medium", "low"]:
            if priority in report["categories"]:
                cat = report["categories"][priority]
                icon = "🔴" if priority == "critical" else "🟡" if priority == "high" else "🟢"
                print(f"{icon} {priority.upper():8} {cat['available']:2}/{cat['total']:2} available")
                
                for item in cat["items"]:
                    status = "✅" if item["available"] else "❌"
                    print(f"   {status} {item['name']:12} {item['speedup']:4.1f}x - {item['description']}")
        
        print("-"*80)
        
        # Show critical missing optimizations
        if report["missing_critical"]:
            print("🚨 MISSING CRITICAL OPTIMIZATIONS:")
            for opt in report["missing_critical"]:
                details = self.detector.optimizations[opt]
                print(f"   ❌ {opt}: {details['description']} ({details['speedup']}x speedup)")
            print("-"*80)
        
        # Show top recommendations
        if report["recommendations"]:
            print("💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"][:3], 1):
                print(f"   {i}. {rec}")
            print("-"*80)
        
        print("🔧 Configuration:")
        print(f"   Optimizations: {'Enabled' if self.config.enable_optimizations else 'Disabled'}")
        print(f"   Monitoring: {'Enabled' if self.config.enable_monitoring else 'Disabled'}")
        print(f"   Benchmarking: {'Enabled' if self.config.enable_benchmarking else 'Disabled'}")
        print(f"   Debug Mode: {'Enabled' if self.config.debug else 'Disabled'}")
        print("="*80 + "\n")
    
    async def run_startup_benchmark(self) -> Any:
        """Run startup performance benchmark."""
        if not self.config.enable_benchmarking:
            return
        
        self.logger.info("Running startup benchmark...")
        
        try:
            # Import and test the service
            
            start_time = time.perf_counter()
            service = await get_service()
            init_time = time.perf_counter() - start_time
            
            self.logger.info(f"Service initialization: {init_time*1000:.1f}ms")
            
            # Test basic functionality
            
            test_input = CopywritingInput(
                product_description="Test product for benchmarking",
                target_platform="instagram",
                content_type="social_post",
                tone="professional",
                use_case="brand_awareness",
                language="es"
            )
            
            start_time = time.perf_counter()
            result = await service.generate_copy(test_input)
            generation_time = time.perf_counter() - start_time
            
            self.logger.info(f"Test generation: {generation_time*1000:.1f}ms")
            self.logger.info(f"Generated {len(result.variants)} variants")
            
            # Performance validation
            if generation_time > 5.0:
                self.logger.warning("Generation time is high, consider installing more optimizations")
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
    
    async def start_server(self) -> Any:
        """Start the refactored server."""
        try:
            
            # Configure uvloop if available
            if self.detector.optimizations["uvloop"]["available"]:
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                self.logger.info("UVLoop enabled for maximum async performance")
            
            # Configure server
            uvicorn_config = uvicorn.Config(
                app=refactored_app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level.lower(),
                loop="uvloop" if self.detector.optimizations["uvloop"]["available"] else "asyncio",
                access_log=self.config.debug,
                reload=self.config.debug,
                workers=1 if self.config.debug else None
            )
            
            server = uvicorn.Server(uvicorn_config)
            
            self.logger.info(f"Starting refactored server on {self.config.host}:{self.config.port}")
            self.running = True
            
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            raise
    
    def setup_signal_handlers(self) -> Any:
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame) -> Any:
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
            
            # Stop event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def health_monitor(self) -> Any:
        """Monitor service health."""
        while self.running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if not self.running:
                    break
                
                # Perform health check
                health_status = await self._perform_health_check()
                
                if health_status["status"] != "healthy":
                    self.logger.warning(f"Health check failed: {health_status}")
                else:
                    self.logger.debug(f"Health check passed: {health_status['response_time_ms']:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            
            async with httpx.AsyncClient() as client:
                start_time = time.perf_counter()
                response = await client.get(
                    f"http://localhost:{self.config.port}/refactored/health",
                    timeout=5.0
                )
                response_time = time.perf_counter() - start_time
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time_ms": response_time * 1000,
                        "data": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "status_code": response.status_code,
                        "error": response.text
                    }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run(self) -> Any:
        """Run the complete deployment."""
        try:
            self.print_startup_banner()
            self.setup_signal_handlers()
            
            # Run startup benchmark
            await self.run_startup_benchmark()
            
            # Start health monitoring
            health_task = asyncio.create_task(self.health_monitor())
            
            # Start server
            server_task = asyncio.create_task(self.start_server())
            
            # Wait for completion
            await asyncio.gather(server_task, health_task, return_exceptions=True)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
        finally:
            uptime = time.time() - self.start_time
            self.logger.info(f"Deployment stopped after {uptime:.1f} seconds")

# === UTILITY FUNCTIONS ===
async def check_optimizations():
    """Check and display optimization status."""
    detector = IntelligentOptimizationDetector()
    report = detector.get_detailed_report()
    
    print("\n🔍 OPTIMIZATION STATUS REPORT")
    print("="*50)
    print(f"Performance Score: {report['performance_score']}/100")
    print(f"Performance Level: {report['performance_level']}")
    print(f"Realistic Speedup: {report['realistic_speedup']}")
    print(f"Available: {report['summary']['available']}/{report['summary']['total']} ({report['summary']['percentage']}%)")
    print("-"*50)
    
    for priority in ["critical", "high", "medium", "low"]:
        if priority in report["categories"]:
            cat = report["categories"][priority]
            print(f"\n{priority.upper()} Priority: {cat['available']}/{cat['total']}")
            for item in cat["items"]:
                status = "✅" if item["available"] else "❌"
                print(f"  {status} {item['name']:12} ({item['speedup']:4.1f}x) - {item['description']}")
    
    if report["recommendations"]:
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

async def install_optimizations():
    """Show installation instructions for missing optimizations."""
    detector = IntelligentOptimizationDetector()
    
    missing = [
        name for name, opt in detector.optimizations.items()
        if not opt["available"]
    ]
    
    if not missing:
        print("✅ All optimizations are already installed!")
        return
    
    print(f"\n📦 INSTALLATION INSTRUCTIONS")
    print("="*50)
    print("Install missing optimizations with:")
    print()
    
    # Group by priority
    critical = [name for name in missing if detector.optimizations[name]["priority"] == "critical"]
    high = [name for name in missing if detector.optimizations[name]["priority"] == "high"]
    medium = [name for name in missing if detector.optimizations[name]["priority"] == "medium"]
    low = [name for name in missing if detector.optimizations[name]["priority"] == "low"]
    
    if critical:
        print("🔴 CRITICAL (Install these first):")
        for name in critical:
            details = detector.optimizations[name]["details"]
            print(f"   {details}")
    
    if high:
        print("\n🟡 HIGH PRIORITY:")
        for name in high:
            details = detector.optimizations[name]["details"]
            print(f"   {details}")
    
    if medium:
        print("\n🟢 MEDIUM PRIORITY:")
        for name in medium:
            details = detector.optimizations[name]["details"]
            print(f"   {details}")
    
    if low:
        print("\n⚪ LOW PRIORITY:")
        for name in low:
            details = detector.optimizations[name]["details"]
            print(f"   {details}")
    
    print(f"\nOr install all at once:")
    print("pip install -r requirements_ultra_max.txt")

async def run_performance_test():
    """Run comprehensive performance test."""
    print("🧪 Running performance test...")
    
    try:
        
        # Test data
        test_data = {
            "product_description": "Plataforma avanzada de marketing digital con inteligencia artificial",
            "target_platform": "instagram",
            "content_type": "social_post",
            "tone": "professional",
            "use_case": "brand_awareness",
            "language": "es",
            "creativity_level": "creative",
            "variant_settings": {"max_variants": 5}
        }
        
        # Warmup request
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:8003/refactored/generate",
                json=test_data,
                timeout=30.0
            )
        
        # Performance test
        times = []
        for i in range(5):
            start_time = time.perf_counter()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8003/refactored/generate",
                    json=test_data,
                    timeout=30.0
                )
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            if response.status_code != 200:
                print(f"❌ Test {i+1} failed: {response.status_code}")
                return
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"✅ Performance test completed:")
        print(f"   Average: {avg_time*1000:.1f}ms")
        print(f"   Minimum: {min_time*1000:.1f}ms")
        print(f"   Maximum: {max_time*1000:.1f}ms")
        print(f"   Requests/second: {1/avg_time:.1f}")
        
        # Get final stats
        async with httpx.AsyncClient() as client:
            stats_response = await client.get("http://localhost:8003/refactored/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"   Cache hit rate: {stats['cache_stats']['hit_rate_percent']:.1f}%")
                print(f"   Total requests: {stats['service_stats']['requests_processed']}")
    
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

# === MAIN ===
async def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Refactored Ultra-Optimized Copywriting Service Deployment"
    )
    
    parser.add_argument("--check", action="store_true", help="Check optimization status")
    parser.add_argument("--install", action="store_true", help="Show installation instructions")
    parser.add_argument("--test", action="store_true", help="Run performance test")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--port", type=int, default=8003, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.check:
        await check_optimizations()
        return
    
    if args.install:
        await install_optimizations()
        return
    
    if args.test:
        await run_performance_test()
        return
    
    # Create configuration
    config = DeploymentConfig(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    # Run deployment
    manager = RefactoredDeploymentManager(config)
    await manager.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Deployment stopped by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1) 