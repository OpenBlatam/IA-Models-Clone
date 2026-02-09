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
import signal
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
            import orjson
            import uvloop
            import redis.asyncio as aioredis
            import polars as pl
            from prometheus_fastapi_instrumentator import Instrumentator
            import lz4
            import xxhash
            import httpx
            import uvicorn
            from final_main import app
                import uvloop
        import httpx
        import time
    import argparse
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Final Production Deployment Script for Copywriting Service.

Clean, production-ready deployment with:
- Intelligent optimization detection
- Performance monitoring
- Graceful shutdown
- Health checks
- Auto-configuration
"""


# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# === OPTIMIZATION DETECTION ===
class OptimizationDetector:
    """Detect available optimization libraries."""
    
    def __init__(self) -> Any:
        self.optimizations = {}
        self.total_speedup = 1.0
        self.performance_level = "BASIC"
        self._detect_all()
    
    def _detect_all(self) -> Any:
        """Detect all optimizations."""
        detections = [
            ("orjson", self._detect_orjson, 3.0, "JSON processing"),
            ("uvloop", self._detect_uvloop, 2.0, "Event loop"),
            ("redis", self._detect_redis, 2.0, "Caching"),
            ("polars", self._detect_polars, 1.5, "Data processing"),
            ("prometheus", self._detect_prometheus, 1.0, "Monitoring"),
            ("lz4", self._detect_lz4, 1.2, "Compression"),
            ("xxhash", self._detect_xxhash, 1.1, "Hashing"),
        ]
        
        available_count = 0
        
        for name, detector, speedup, description in detections:
            available, details = detector()
            self.optimizations[name] = {
                "available": available,
                "speedup": speedup if available else 1.0,
                "description": description,
                "details": details
            }
            
            if available:
                available_count += 1
                self.total_speedup *= speedup
        
        # Calculate performance level
        if available_count >= 6:
            self.performance_level = "ULTRA"
        elif available_count >= 4:
            self.performance_level = "HIGH"
        elif available_count >= 2:
            self.performance_level = "MEDIUM"
        else:
            self.performance_level = "BASIC"
        
        # Cap realistic speedup
        self.total_speedup = min(self.total_speedup, 20.0)
    
    def _detect_orjson(self) -> tuple[bool, str]:
        try:
            return True, f"v{orjson.__version__}"
        except ImportError:
            return False, "Not installed"
    
    def _detect_uvloop(self) -> tuple[bool, str]:
        try:
            if sys.platform == 'win32':
                return False, "Not supported on Windows"
            return True, f"v{uvloop.__version__}"
        except ImportError:
            return False, "Not installed"
    
    def _detect_redis(self) -> tuple[bool, str]:
        try:
            return True, "Available"
        except ImportError:
            return False, "Not installed"
    
    def _detect_polars(self) -> tuple[bool, str]:
        try:
            return True, f"v{pl.__version__}"
        except ImportError:
            return False, "Not installed"
    
    def _detect_prometheus(self) -> tuple[bool, str]:
        try:
            return True, "Available"
        except ImportError:
            return False, "Not installed"
    
    def _detect_lz4(self) -> tuple[bool, str]:
        try:
            return True, f"v{lz4.version.version}"
        except ImportError:
            return False, "Not installed"
    
    def _detect_xxhash(self) -> tuple[bool, str]:
        try:
            return True, f"v{xxhash.VERSION}"
        except ImportError:
            return False, "Not installed"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        available = sum(1 for opt in self.optimizations.values() if opt["available"])
        total = len(self.optimizations)
        
        return {
            "performance_level": self.performance_level,
            "total_speedup": f"{self.total_speedup:.1f}x",
            "optimizations_available": f"{available}/{total}",
            "optimizations": self.optimizations
        }

# === CONFIGURATION ===
class FinalDeploymentConfig:
    """Final deployment configuration."""
    
    def __init__(self) -> Any:
        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8000))
        self.workers = int(os.getenv("WORKERS", min(16, mp.cpu_count() * 2)))
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Performance
        self.enable_uvloop = os.getenv("ENABLE_UVLOOP", "true").lower() == "true"
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        
        # Security
        self.api_key = os.getenv("COPYWRITING_API_KEY", "final-optimized-2024")
        
        # Timeouts
        self.startup_timeout = int(os.getenv("STARTUP_TIMEOUT", 30))
        self.shutdown_timeout = int(os.getenv("SHUTDOWN_TIMEOUT", 15))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

# === HEALTH CHECKER ===
class HealthChecker:
    """Health check utilities."""
    
    @staticmethod
    async def check_service_health() -> Dict[str, Any]:
        """Check service health."""
        try:
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8000/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
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
    
    @staticmethod
    async def wait_for_service(max_attempts: int = 30) -> bool:
        """Wait for service to be ready."""
        for attempt in range(max_attempts):
            health = await HealthChecker.check_service_health()
            
            if health["status"] == "healthy":
                return True
            
            print(f"Waiting for service... (attempt {attempt + 1}/{max_attempts})")
            await asyncio.sleep(1)
        
        return False

# === DEPLOYMENT MANAGER ===
class FinalDeploymentManager:
    """Final deployment manager."""
    
    def __init__(self) -> Any:
        self.config = FinalDeploymentConfig()
        self.detector = OptimizationDetector()
        self.server_process = None
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def print_banner(self) -> Any:
        """Print startup banner."""
        summary = self.detector.get_summary()
        
        print("\n" + "="*60)
        print("🚀 FINAL OPTIMIZED COPYWRITING SERVICE")
        print("="*60)
        print(f"Performance Level: {summary['performance_level']}")
        print(f"Total Speedup: {summary['total_speedup']}")
        print(f"Optimizations: {summary['optimizations_available']}")
        print(f"Environment: {self.config.environment}")
        print(f"Host: {self.config.host}:{self.config.port}")
        print(f"Workers: {self.config.workers}")
        print("-"*60)
        
        # Show optimizations
        for name, opt in summary['optimizations'].items():
            status = "✅" if opt['available'] else "❌"
            speedup = f"{opt['speedup']:.1f}x" if opt['available'] else "1.0x"
            print(f"{status} {name.upper():12} {speedup:>6} - {opt['description']}")
            if opt['details'] != "Not installed":
                print(f"   {opt['details']}")
        
        print("="*60)
        print("🔧 Configuration:")
        print(f"   Cache: {'Enabled' if self.config.enable_cache else 'Disabled'}")
        print(f"   Metrics: {'Enabled' if self.config.enable_metrics else 'Disabled'}")
        print(f"   UVLoop: {'Enabled' if self.config.enable_uvloop else 'Disabled'}")
        print(f"   Debug: {'Enabled' if self.config.debug else 'Disabled'}")
        print("="*60 + "\n")
    
    async def start_server(self) -> Any:
        """Start the server."""
        try:
            
            # Configure uvloop
            if self.config.enable_uvloop and self.detector.optimizations["uvloop"]["available"]:
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                self.logger.info("UVLoop enabled")
            
            # Configure uvicorn
            uvicorn_config = uvicorn.Config(
                app=app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level.lower(),
                loop="uvloop" if (self.config.enable_uvloop and 
                                self.detector.optimizations["uvloop"]["available"]) else "asyncio",
                access_log=self.config.debug,
                reload=self.config.debug,
                workers=1 if self.config.debug else None  # Single worker for debug
            )
            
            server = uvicorn.Server(uvicorn_config)
            
            self.logger.info(f"Starting server on {self.config.host}:{self.config.port}")
            self.running = True
            
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    def setup_signal_handlers(self) -> Any:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame) -> Any:
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
            
            # Stop the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_health_monitor(self) -> Any:
        """Run health monitoring."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.running:
                    break
                    
                health = await HealthChecker.check_service_health()
                
                if health["status"] != "healthy":
                    self.logger.warning(f"Health check failed: {health}")
                else:
                    self.logger.debug(f"Health check passed: {health['response_time_ms']:.1f}ms")
                    
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    async def run(self) -> Any:
        """Run the deployment."""
        try:
            self.print_banner()
            self.setup_signal_handlers()
            
            # Start health monitor
            health_task = asyncio.create_task(self.run_health_monitor())
            
            # Start server
            server_task = asyncio.create_task(self.start_server())
            
            # Wait for completion
            await asyncio.gather(server_task, health_task, return_exceptions=True)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Deployment error: {e}")
            raise
        finally:
            self.logger.info("Deployment stopped")

# === UTILITY FUNCTIONS ===
async def install_missing_optimizations():
    """Install missing optimization libraries."""
    detector = OptimizationDetector()
    missing = [
        name for name, opt in detector.optimizations.items() 
        if not opt["available"] and opt["details"] == "Not installed"
    ]
    
    if not missing:
        print("✅ All optimizations are available!")
        return
    
    print(f"❌ Missing optimizations: {', '.join(missing)}")
    print("\nTo install missing optimizations:")
    print("pip install -r requirements_final.txt")
    
    return missing

async def run_performance_test():
    """Run performance test."""
    print("🧪 Running performance test...")
    
    try:
        
        # Test data
        test_data = {
            "product_description": "Plataforma de marketing digital con IA",
            "target_platform": "instagram",
            "content_type": "social_post",
            "tone": "professional",
            "use_case": "brand_awareness",
            "language": "es"
        }
        
        # Run test
        start_time = time.perf_counter()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/generate",
                json=test_data,
                headers={"X-API-Key": "final-optimized-2024"},
                timeout=30.0
            )
        
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Test passed in {(end_time - start_time)*1000:.1f}ms")
            print(f"   Performance level: {data.get('performance_metrics', {}).get('performance_level', 'Unknown')}")
            print(f"   Total speedup: {data.get('performance_metrics', {}).get('total_speedup', 'Unknown')}")
            print(f"   Variants generated: {len(data.get('variants', []))}")
        else:
            print(f"❌ Test failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ Test error: {e}")

# === MAIN ===
async def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Final Copywriting Service Deployment")
    parser.add_argument("--check", action="store_true", help="Check optimizations")
    parser.add_argument("--test", action="store_true", help="Run performance test")
    parser.add_argument("--install", action="store_true", help="Show installation instructions")
    
    args = parser.parse_args()
    
    if args.check:
        detector = OptimizationDetector()
        summary = detector.get_summary()
        
        print("\n🔍 OPTIMIZATION STATUS")
        print("="*40)
        print(f"Performance Level: {summary['performance_level']}")
        print(f"Total Speedup: {summary['total_speedup']}")
        print(f"Available: {summary['optimizations_available']}")
        print("-"*40)
        
        for name, opt in summary['optimizations'].items():
            status = "✅" if opt['available'] else "❌"
            print(f"{status} {name.upper():12} - {opt['description']}")
        
        return
    
    if args.install:
        await install_missing_optimizations()
        return
    
    if args.test:
        await run_performance_test()
        return
    
    # Run deployment
    manager = FinalDeploymentManager()
    await manager.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1) 