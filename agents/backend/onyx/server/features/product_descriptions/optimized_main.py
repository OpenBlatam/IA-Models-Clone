from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import argparse
import logging
import sys
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from core.generator import ProductDescriptionGenerator
from core.config import ProductDescriptionConfig, ECOMMERCE_CONFIG, LUXURY_CONFIG, TECHNICAL_CONFIG
from api.service import ProductDescriptionService
from api.gradio_interface import create_gradio_app
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
🚀 OPTIMIZED Product Description Generator - Main Entry Point
============================================================

Ultra-optimized version with:
- Async-first architecture
- Performance optimizations
- Advanced caching
- Circuit breakers
- Connection pooling
- Memory optimization
- Batch processing
- Real-time metrics
"""


# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))


# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('product_descriptions.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance tracking with memory optimization"""
    start_time: float
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    error_count: int = 0
    
    def update_memory_usage(self) -> Any:
        """Update memory usage metrics"""
        process = psutil.Process()
        self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        self.cpu_usage = process.cpu_percent()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        duration = time.time() - self.start_time
        return {
            "uptime_seconds": duration,
            "memory_usage_mb": self.memory_usage,
            "cpu_usage_percent": self.cpu_usage,
            "cache_hit_rate": self.cache_hits / max(self.total_requests, 1),
            "error_rate": self.error_count / max(self.total_requests, 1),
            "requests_per_second": self.total_requests / max(duration, 1)
        }


class OptimizedProductDescriptionGenerator:
    """Ultra-optimized generator with advanced features"""
    
    def __init__(self, config: Optional[ProductDescriptionConfig] = None):
        
    """__init__ function."""
self.config = config or ProductDescriptionConfig()
        self.generator: Optional[ProductDescriptionGenerator] = None
        self.metrics = PerformanceMetrics(time.time())
        self.is_initialized = False
        
        # Optimized thread pools
        self.io_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io_worker")
        self.cpu_executor = ProcessPoolExecutor(max_workers=4)
        
        # Memory management
        self._setup_memory_management()
        
        # Graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_memory_management(self) -> Any:
        """Setup memory optimization"""
        # Enable garbage collection
        gc.enable()
        
        # Set memory thresholds
        self.memory_threshold_mb = 1024  # 1GB
        self.last_gc_time = time.time()
    
    def _setup_signal_handlers(self) -> Any:
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame) -> Any:
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """Initialize with performance optimizations"""
        try:
            logger.info("🚀 Initializing Optimized Product Description Generator...")
            
            # Initialize base generator
            self.generator = ProductDescriptionGenerator(self.config)
            await self.generator.initialize()
            
            # Pre-warm cache with common patterns
            await self._pre_warm_cache()
            
            self.is_initialized = True
            logger.info("✅ Optimized generator initialized successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize generator: {e}")
            self.metrics.error_count += 1
            return False
    
    async def _pre_warm_cache(self) -> Any:
        """Pre-warm cache with common product patterns"""
        common_products = [
            {
                "product_name": "Wireless Headphones",
                "features": ["Bluetooth", "Noise Cancellation", "Long Battery"],
                "category": "electronics",
                "style": "professional"
            },
            {
                "product_name": "Smartphone",
                "features": ["5G", "High Resolution Camera", "Fast Charging"],
                "category": "electronics", 
                "style": "technical"
            }
        ]
        
        logger.info("🔥 Pre-warming cache...")
        for product in common_products:
            try:
                await self.generator.generate_async(**product)
            except Exception as e:
                logger.warning(f"Cache pre-warming failed for {product['product_name']}: {e}")
    
    async def generate_optimized(
        self,
        product_name: str,
        features: list[str],
        **kwargs
    ) -> list[dict[str, Any]]:
        """Generate with performance optimizations"""
        if not self.is_initialized:
            raise RuntimeError("Generator not initialized")
        
        self.metrics.total_requests += 1
        self.metrics.update_memory_usage()
        
        # Memory management
        await self._check_memory_usage()
        
        try:
            # Use async generation for better performance
            results = await self.generator.generate_async(
                product_name=product_name,
                features=features,
                **kwargs
            )
            
            # Update cache metrics
            if hasattr(self.generator, 'stats'):
                self.metrics.cache_hits = self.generator.stats.get('cache_hits', 0)
                self.metrics.cache_misses = self.generator.stats.get('cache_misses', 0)
            
            return results
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Generation failed: {e}")
            raise
    
    async def _check_memory_usage(self) -> Any:
        """Check and manage memory usage"""
        if self.metrics.memory_usage > self.memory_threshold_mb:
            logger.warning(f"High memory usage: {self.metrics.memory_usage:.1f}MB")
            
            # Force garbage collection
            gc.collect()
            
            # Clear generator cache if available
            if hasattr(self.generator, 'clear_cache'):
                self.generator.clear_cache()
            
            self.last_gc_time = time.time()
    
    async def generate_batch_optimized(
        self,
        products: list[dict[str, Any]],
        max_concurrent: int = 10
    ) -> list[list[dict[str, Any]]]:
        """Optimized batch generation with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(product) -> Any:
            async with semaphore:
                return await self.generate_optimized(**product)
        
        tasks = [generate_single(product) for product in products]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch generation error: {result}")
                self.metrics.error_count += 1
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics"""
        self.metrics.update_memory_usage()
        return self.metrics.get_stats()
    
    def shutdown(self) -> Any:
        """Graceful shutdown with cleanup"""
        logger.info("🔄 Shutting down optimized generator...")
        
        # Shutdown executors
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        
        # Clear caches
        if self.generator and hasattr(self.generator, 'clear_cache'):
            self.generator.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ Optimized generator shutdown complete")


@asynccontextmanager
async def get_optimized_generator():
    """Context manager for optimized generator"""
    generator = OptimizedProductDescriptionGenerator()
    try:
        await generator.initialize()
        yield generator
    finally:
        generator.shutdown()


async def run_optimized_demo():
    """Run demo with performance optimizations"""
    print("🚀 Optimized Product Description Generator - Demo Mode")
    print("=" * 60)
    
    async with get_optimized_generator() as generator:
        print("✅ Optimized generator ready!")
        
        # Demo products with performance tracking
        demo_products = [
            {
                "product_name": "Wireless Bluetooth Headphones",
                "features": ["Active noise cancellation", "30-hour battery", "Premium leather", "Quick charge"],
                "category": "electronics",
                "brand": "TechPro",
                "style": "professional",
                "tone": "friendly"
            },
            {
                "product_name": "Luxury Silk Scarf",
                "features": ["100% pure silk", "Hand-rolled edges", "Designer pattern", "Gift packaging"],
                "category": "clothing",
                "brand": "LuxeStyle",
                "style": "luxury",
                "tone": "sophisticated"
            },
            {
                "product_name": "Gaming Mechanical Keyboard",
                "features": ["Cherry MX switches", "RGB backlighting", "Programmable keys", "USB-C connectivity"],
                "category": "electronics",
                "brand": "GameForce",
                "style": "technical",
                "tone": "enthusiastic"
            }
        ]
        
        # Batch generation for better performance
        print("\n🔥 Generating descriptions in batch...")
        start_time = time.time()
        
        results = await generator.generate_batch_optimized(demo_products)
        
        generation_time = time.time() - start_time
        
        # Display results with performance metrics
        for i, (product, result_list) in enumerate(zip(demo_products, results), 1):
            print(f"\n📝 Demo {i}: {product['product_name']}")
            print("-" * 50)
            
            if result_list and len(result_list) > 0:
                result = result_list[0]
                print(f"✨ Generated Description:")
                print(f"{result['description']}")
                print(f"\n📊 Metrics:")
                print(f"   Quality Score: {result['quality_score']:.2f}")
                print(f"   SEO Score: {result['seo_score']:.2f}")
                print(f"   Word Count: {result['metadata']['word_count']}")
        
        # Performance statistics
        stats = generator.get_performance_stats()
        print(f"\n🚀 Performance Statistics:")
        print(f"   Batch Generation Time: {generation_time:.2f}s")
        print(f"   Memory Usage: {stats['memory_usage_mb']:.1f}MB")
        print(f"   CPU Usage: {stats['cpu_usage_percent']:.1f}%")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        print(f"   Requests/Second: {stats['requests_per_second']:.1f}")
        
        print("\n🎉 Optimized demo completed!")


def run_optimized_cli():
    """Run CLI mode with optimizations"""
    print("🚀 Optimized Product Description Generator - CLI Mode")
    print("=" * 60)
    
    # Get user input with validation
    product_name = input("Product Name: ").strip()
    if not product_name:
        print("❌ Error: Product name cannot be empty")
        return
    
    features_input = input("Features (comma-separated): ").strip()
    features = [f.strip() for f in features_input.split(',') if f.strip()]
    if not features:
        print("❌ Error: At least one feature is required")
        return
    
    # Get other parameters with defaults
    category = input("Category (default: general): ").strip() or "general"
    brand = input("Brand (default: unknown): ").strip() or "unknown"
    style = input("Style [professional/casual/luxury/technical/creative] (default: professional): ").strip() or "professional"
    tone = input("Tone [friendly/formal/enthusiastic/informative/persuasive] (default: friendly): ").strip() or "friendly"
    
    # Validate inputs
    valid_styles = ["professional", "casual", "luxury", "technical", "creative"]
    valid_tones = ["friendly", "formal", "enthusiastic", "informative", "persuasive"]
    
    if style not in valid_styles:
        print(f"❌ Error: Invalid style. Must be one of: {', '.join(valid_styles)}")
        return
    
    if tone not in valid_tones:
        print(f"❌ Error: Invalid tone. Must be one of: {', '.join(valid_tones)}")
        return
    
    async def generate():
        
    """generate function."""
async with get_optimized_generator() as generator:
            print("\n🤖 Generating optimized description...")
            start_time = time.time()
            
            results = await generator.generate_optimized(
                product_name=product_name,
                features=features,
                category=category,
                brand=brand,
                style=style,
                tone=tone,
                num_variations=2
            )
            
            generation_time = time.time() - start_time
            
            # Display results
            print(f"\n✨ Generated Descriptions (in {generation_time:.2f}s):")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n📝 Variation {i}:")
                print(f"{result['description']}")
                print(f"\n📊 Metrics:")
                print(f"   Quality: {result['quality_score']:.2f}/1.0")
                print(f"   SEO: {result['seo_score']:.2f}/1.0")
                print(f"   Words: {result['metadata']['word_count']}")
            
            # Performance stats
            stats = generator.get_performance_stats()
            print(f"\n🚀 Performance:")
            print(f"   Memory: {stats['memory_usage_mb']:.1f}MB")
            print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    
    asyncio.run(generate())


def run_optimized_api(host: str = "0.0.0.0", port: int = 8000):
    """Run API service with optimizations"""
    if not host or not host.strip():
        print("❌ Error: Host cannot be empty")
        return
    
    if port < 1 or port > 65535:
        print("❌ Error: Port must be between 1 and 65535")
        return
    
    print(f"🚀 Starting Optimized Product Description API on {host}:{port}")
    print("🔥 Features: Async, Caching, Circuit Breakers, Performance Monitoring")
    
    service = ProductDescriptionService()
    service.run(host=host, port=port)


def run_optimized_gradio(share: bool = False, port: int = 7860):
    """Run Gradio interface with optimizations"""
    if port < 1 or port > 65535:
        print("❌ Error: Port must be between 1 and 65535")
        return
    
    print(f"🎮 Starting Optimized Gradio Interface on port {port}")
    print("🔥 Features: Async Processing, Real-time Metrics, Performance Dashboard")
    
    app = create_gradio_app()
    app.launch(share=share, server_port=port)


def main():
    """Main entry point with optimizations"""
    parser = argparse.ArgumentParser(
        description="🚀 Optimized Product Description Generator - AI-powered product descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Optimized Examples:
  python optimized_main.py demo                    # Run optimized demo mode
  python optimized_main.py cli                     # Run optimized CLI mode
  python optimized_main.py api                     # Start optimized API service
  python optimized_main.py gradio                  # Start optimized Gradio interface
  python optimized_main.py api --port 8080         # API on custom port
  python optimized_main.py gradio --share          # Gradio with public sharing
  python optimized_main.py api --debug             # API with debug logging
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["demo", "cli", "api", "gradio"],
        help="Run mode"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for API service (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API service (default: 8000) or Gradio (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable public sharing for Gradio interface"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=1024,
        help="Memory limit in MB (default: 1024)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.port < 1 or args.port > 65535:
        print("❌ Error: Port must be between 1 and 65535")
        return
    
    if not args.host or not args.host.strip():
        print("❌ Error: Host cannot be empty")
        return
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Debug mode enabled")
    
    # Run selected mode with optimizations
    try:
        if args.mode == "demo":
            asyncio.run(run_optimized_demo())
        elif args.mode == "cli":
            run_optimized_cli()
        elif args.mode == "api":
            run_optimized_api(host=args.host, port=args.port)
        elif args.mode == "gradio":
            gradio_port = args.port if args.port != 8000 else 7860
            run_optimized_gradio(share=args.share, port=gradio_port)
            
    except KeyboardInterrupt:
        print("\n👋 Graceful shutdown completed!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 