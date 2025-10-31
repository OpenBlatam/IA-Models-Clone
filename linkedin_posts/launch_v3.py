#!/usr/bin/env python3
"""
🚀 Next-Generation Launcher for Ultra-Optimized LinkedIn Posts Optimization v3.0
===============================================================================

Main launcher script that orchestrates all components of the revolutionary v3.0 system.
"""

import asyncio
import sys
import signal
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config_v3 import ConfigManager, create_default_config
    from api_gateway_v3 import NextGenAPIGateway
    from deployment_v3 import NextGenDeploymentManager
    from ultra_optimized_linkedin_optimizer_v3 import create_nextgen_service
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required files are available.")
    sys.exit(1)

class NextGenLauncher:
    """Main launcher for the next-generation v3.0 system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_manager = None
        self.api_gateway = None
        self.service = None
        self.deployment_manager = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            print("🚀 Initializing Next-Generation LinkedIn Optimizer v3.0...")
            
            # Initialize configuration
            print("📋 Loading configuration...")
            self.config_manager = ConfigManager(self.config_path)
            config = self.config_manager.get_config()
            
            print(f"✅ Configuration loaded: {config.environment.value} environment")
            print(f"🔧 Features enabled: {', '.join([k for k, v in config.features.items() if v])}")
            
            # Initialize core service
            print("🧠 Initializing core optimization service...")
            self.service = create_nextgen_service()
            
            # Initialize API gateway
            print("🌐 Initializing API gateway...")
            self.api_gateway = NextGenAPIGateway()
            
            # Initialize deployment manager
            print("🚀 Initializing deployment manager...")
            self.deployment_manager = NextGenDeploymentManager()
            
            print("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def start_services(self):
        """Start all system services."""
        try:
            print("🚀 Starting Next-Generation LinkedIn Optimizer v3.0 services...")
            
            # Start background tasks
            background_tasks = []
            
            # Start monitoring if enabled
            if self.config_manager.get_config().monitoring.enable_metrics:
                print("📊 Starting metrics collection...")
                background_tasks.append(self.start_metrics_collection())
            
            # Start real-time learning if enabled
            if self.config_manager.get_feature_flag("real_time_learning"):
                print("🧠 Starting real-time learning engine...")
                background_tasks.append(self.start_learning_engine())
            
            # Start A/B testing if enabled
            if self.config_manager.get_feature_flag("ab_testing"):
                print("🧪 Starting A/B testing engine...")
                background_tasks.append(self.start_ab_testing_engine())
            
            # Start distributed processing if enabled
            if self.config_manager.get_feature_flag("distributed_processing"):
                print("⚡ Starting distributed processing engine...")
                background_tasks.append(self.start_distributed_engine())
            
            # Start all background tasks
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            
            print("✅ All services started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Service startup failed: {e}")
            return False
    
    async def start_metrics_collection(self):
        """Start metrics collection service."""
        try:
            while not self.shutdown_event.is_set():
                # Collect system metrics
                if hasattr(self.service, 'monitor'):
                    stats = self.service.monitor.get_stats()
                    # Log key metrics
                    if stats:
                        print(f"📊 System Stats - Uptime: {stats.get('total_uptime', 0):.1f}s, "
                              f"Operations: {len(stats.get('operations', {}))}")
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
        except asyncio.CancelledError:
            print("📊 Metrics collection stopped")
        except Exception as e:
            print(f"❌ Metrics collection error: {e}")
    
    async def start_learning_engine(self):
        """Start real-time learning engine."""
        try:
            while not self.shutdown_event.is_set():
                # Process learning insights
                if hasattr(self.service, 'learning_engine'):
                    insights = await self.service.get_learning_insights()
                    if insights:
                        print(f"🧠 Learning Insights: {len(insights)} new insights processed")
                
                await asyncio.sleep(60)  # Process every minute
                
        except asyncio.CancelledError:
            print("🧠 Learning engine stopped")
        except Exception as e:
            print(f"❌ Learning engine error: {e}")
    
    async def start_ab_testing_engine(self):
        """Start A/B testing engine."""
        try:
            while not self.shutdown_event.is_set():
                # Check A/B test status
                if hasattr(self.service, 'ab_testing_engine'):
                    active_tests = len(self.service.ab_testing_engine.active_tests)
                    if active_tests > 0:
                        print(f"🧪 A/B Testing: {active_tests} active tests")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
        except asyncio.CancelledError:
            print("🧪 A/B testing engine stopped")
        except Exception as e:
            print(f"❌ A/B testing engine error: {e}")
    
    async def start_distributed_engine(self):
        """Start distributed processing engine."""
        try:
            while not self.shutdown_event.is_set():
                # Check distributed engine status
                if hasattr(self.service, 'distributed_engine'):
                    if self.service.distributed_engine.is_running:
                        worker_count = len(self.service.distributed_engine.workers)
                        print(f"⚡ Distributed Processing: {worker_count} workers active")
                
                await asyncio.sleep(180)  # Check every 3 minutes
                
        except asyncio.CancelledError:
            print("⚡ Distributed processing engine stopped")
        except Exception as e:
            print(f"❌ Distributed processing engine error: {e}")
    
    async def run_api_gateway(self):
        """Run the API gateway."""
        try:
            config = self.config_manager.get_config()
            
            print(f"🌐 Starting API gateway on {config.api_host}:{config.api_port}")
            print(f"📚 API Documentation: http://{config.api_host}:{config.api_port}/docs")
            print(f"📊 Metrics: http://{config.api_host}:{config.api_port}/metrics")
            print(f"🏥 Health Check: http://{config.api_host}:{config.api_port}/health")
            
            # Start API gateway
            server = uvicorn.Server(
                uvicorn.Config(
                    self.api_gateway.app,
                    host=config.api_host,
                    port=config.api_port,
                    workers=config.api_workers,
                    reload=config.api_reload,
                    log_level="info"
                )
            )
            
            # Run server
            await server.serve()
            
        except Exception as e:
            print(f"❌ API gateway error: {e}")
    
    async def run_interactive_mode(self):
        """Run in interactive mode for testing."""
        try:
            print("🧪 Interactive Mode - Testing Next-Generation System")
            print("=" * 60)
            
            # Test basic functionality
            print("\n1️⃣ Testing content optimization...")
            test_content = "Just completed an amazing AI project! #artificialintelligence #machinelearning"
            
            result = await self.service.optimize_linkedin_post(
                test_content,
                "ENGAGEMENT"
            )
            
            print(f"✅ Optimization completed!")
            print(f"   Score: {result.optimization_score:.1f}%")
            print(f"   Confidence: {result.confidence_score:.1f}%")
            print(f"   Processing Time: {result.processing_time:.3f}s")
            
            # Test multi-language
            print("\n2️⃣ Testing multi-language optimization...")
            try:
                multi_lang_result = await self.service.optimize_linkedin_post(
                    test_content,
                    "ENGAGEMENT",
                    target_language="SPANISH"
                )
                print(f"✅ Multi-language optimization completed!")
                print(f"   Target Language: {multi_lang_result.optimized_content.language}")
            except Exception as e:
                print(f"⚠️ Multi-language test failed: {e}")
            
            # Test A/B testing
            print("\n3️⃣ Testing A/B testing...")
            try:
                ab_test_result = await self.service.optimize_linkedin_post(
                    test_content,
                    "ENGAGEMENT",
                    enable_ab_testing=True
                )
                print(f"✅ A/B testing completed!")
                if ab_test_result.ab_test_results:
                    print(f"   Test ID: {ab_test_result.ab_test_results['test_id']}")
            except Exception as e:
                print(f"⚠️ A/B testing failed: {e}")
            
            # Test real-time learning
            print("\n4️⃣ Testing real-time learning...")
            try:
                learning_result = await self.service.optimize_linkedin_post(
                    test_content,
                    "ENGAGEMENT",
                    enable_learning=True
                )
                print(f"✅ Real-time learning completed!")
                insights = await self.service.get_learning_insights()
                print(f"   Learning Insights: {len(insights)}")
            except Exception as e:
                print(f"⚠️ Real-time learning failed: {e}")
            
            print("\n🎉 Interactive testing completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Interactive mode error: {e}")
    
    async def run_benchmark_mode(self):
        """Run in benchmark mode for performance testing."""
        try:
            print("⚡ Benchmark Mode - Performance Testing Next-Generation System")
            print("=" * 60)
            
            # Test content
            test_contents = [
                "Just completed an amazing AI project! #artificialintelligence #machinelearning",
                "Excited to share our latest breakthrough in natural language processing! #nlp #ai",
                "The future of machine learning is here! #ml #innovation #tech",
                "Revolutionary approach to deep learning optimization! #deeplearning #optimization",
                "Transforming the way we think about AI! #artificialintelligence #innovation"
            ]
            
            strategies = ["ENGAGEMENT", "REACH", "BRAND_AWARENESS", "LEAD_GENERATION"]
            
            print(f"\n📊 Running benchmarks with {len(test_contents)} contents and {len(strategies)} strategies...")
            
            total_start_time = time.time()
            total_optimizations = 0
            total_processing_time = 0
            
            for i, content in enumerate(test_contents, 1):
                print(f"\n📝 Content {i}/{len(test_contents)}: {content[:50]}...")
                
                for j, strategy in enumerate(strategies, 1):
                    print(f"   🎯 Strategy {j}/{len(strategies)}: {strategy}")
                    
                    start_time = time.time()
                    try:
                        result = await self.service.optimize_linkedin_post(
                            content,
                            strategy
                        )
                        end_time = time.time()
                        
                        processing_time = end_time - start_time
                        total_processing_time += processing_time
                        total_optimizations += 1
                        
                        print(f"      ✅ Completed in {processing_time:.3f}s")
                        print(f"         Score: {result.optimization_score:.1f}%")
                        print(f"         Confidence: {result.confidence_score:.1f}%")
                        
                    except Exception as e:
                        print(f"      ❌ Failed: {e}")
            
            total_time = time.time() - total_start_time
            
            print("\n" + "=" * 60)
            print("📊 BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total Optimizations: {total_optimizations}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Average Processing Time: {total_processing_time/total_optimizations:.3f}s per optimization")
            print(f"Throughput: {total_optimizations/total_time:.2f} optimizations/second")
            print(f"Success Rate: {(total_optimizations/(len(test_contents)*len(strategies)))*100:.1f}%")
            
            print("\n🎉 Benchmark completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Benchmark mode error: {e}")
    
    async def run(self, mode: str = "api"):
        """Run the launcher in specified mode."""
        try:
            # Initialize system
            if not await self.initialize():
                return False
            
            # Start services
            if not await self.start_services():
                return False
            
            self.running = True
            print("🚀 Next-Generation LinkedIn Optimizer v3.0 is now running!")
            
            # Run in specified mode
            if mode == "api":
                await self.run_api_gateway()
            elif mode == "interactive":
                await self.run_interactive_mode()
            elif mode == "benchmark":
                await self.run_benchmark_mode()
            else:
                print(f"❌ Unknown mode: {mode}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Runtime error: {e}")
            return False
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        if not self.running:
            return
        
        print("\n🛑 Shutting down Next-Generation LinkedIn Optimizer v3.0...")
        self.running = False
        
        try:
            # Shutdown service
            if self.service and hasattr(self.service, 'shutdown'):
                await self.service.shutdown()
                print("✅ Core service shutdown completed")
            
            # Shutdown API gateway
            if self.api_gateway:
                print("✅ API gateway shutdown completed")
            
            # Shutdown deployment manager
            if self.deployment_manager:
                print("✅ Deployment manager shutdown completed")
            
            print("🎉 Graceful shutdown completed!")
            
        except Exception as e:
            print(f"⚠️ Shutdown warning: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Next-Generation LinkedIn Optimizer v3.0 - Revolutionary Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run API gateway
  python launch_v3.py --mode api
  
  # Run in interactive mode for testing
  python launch_v3.py --mode interactive
  
  # Run performance benchmarks
  python launch_v3.py --mode benchmark
  
  # Create default configuration
  python launch_v3.py --create-config
  
  # Deploy to Kubernetes
  python launch_v3.py --deploy kubernetes
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "interactive", "benchmark"],
        default="api",
        help="Operation mode (default: api)"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file"
    )
    
    parser.add_argument(
        "--deploy",
        choices=["docker", "kubernetes", "helm", "terraform"],
        help="Deploy system to specified target"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        print("📋 Creating default configuration file...")
        default_config = create_default_config()
        with open("config_v3.yaml", "w") as f:
            f.write(default_config)
        print("✅ Default configuration file created: config_v3.yaml")
        return
    
    if args.deploy:
        print(f"🚀 Deploying to {args.deploy}...")
        deployment_manager = NextGenDeploymentManager()
        success = deployment_manager.deploy(args.deploy)
        if success:
            print(f"✅ Deployment to {args.deploy} completed successfully!")
        else:
            print(f"❌ Deployment to {args.deploy} failed!")
        return
    
    # Run launcher
    print("🚀 Next-Generation LinkedIn Optimizer v3.0 - Revolutionary Launcher")
    print("=" * 70)
    
    launcher = NextGenLauncher(args.config)
    
    try:
        # Set configuration overrides
        if args.host != "0.0.0.0" or args.port != 8000:
            launcher.config_manager.update_config({
                "api_host": args.host,
                "api_port": args.port
            })
        
        # Run launcher
        success = asyncio.run(launcher.run(args.mode))
        
        if success:
            print("🎉 Launcher completed successfully!")
            sys.exit(0)
        else:
            print("💥 Launcher failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
