#!/usr/bin/env python3
"""
Ultimate Facebook Posts System v4.0 - Final Launch Script
Complete system integration with all advanced features
"""

import asyncio
import argparse
import logging
import sys
import time
import signal
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ultimate_system.log')
    ]
)

logger = logging.getLogger(__name__)

# Import all system components
try:
    from core.ultimate_integration import get_ultimate_integration_system
    from core.enterprise_features import get_enterprise_features_system
    from core.predictive_analytics import get_predictive_analytics_system
    from core.real_time_dashboard import get_real_time_dashboard
    from core.intelligent_cache import get_intelligent_cache
    from core.auto_scaling import get_auto_scaling_system
    from core.advanced_security import get_advanced_security_system
    from core.performance_optimizer import get_performance_optimizer
    from core.advanced_monitoring import get_monitoring_system
    from services.advanced_ai_enhancer import get_ai_enhancer
    from deployment.ultimate_deployment import get_ultimate_deployment_system
    from app import create_app
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all dependencies are installed and paths are correct")
    sys.exit(1)


class UltimateSystemLauncher:
    """Ultimate System Launcher with complete integration"""
    
    def __init__(self, mode: str = "production"):
        self.mode = mode
        self.systems = {}
        self.is_running = False
        self.startup_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def initialize_all_systems(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Ultimate Facebook Posts System v4.0...")
            self.startup_time = time.time()
            
            # Initialize core systems
            logger.info("📊 Initializing Predictive Analytics System...")
            self.systems['predictive_analytics'] = await get_predictive_analytics_system()
            
            logger.info("📈 Initializing Real-time Dashboard System...")
            self.systems['dashboard'] = await get_real_time_dashboard()
            
            logger.info("🧠 Initializing Intelligent Cache System...")
            self.systems['cache'] = await get_intelligent_cache()
            
            logger.info("⚡ Initializing Auto-scaling System...")
            self.systems['auto_scaling'] = await get_auto_scaling_system()
            
            logger.info("🔒 Initializing Advanced Security System...")
            self.systems['security'] = await get_advanced_security_system()
            
            logger.info("🎯 Initializing Performance Optimizer...")
            self.systems['performance'] = await get_performance_optimizer()
            
            logger.info("📊 Initializing Advanced Monitoring System...")
            self.systems['monitoring'] = await get_monitoring_system()
            
            logger.info("🤖 Initializing AI Enhancer...")
            self.systems['ai_enhancer'] = await get_ai_enhancer()
            
            logger.info("🏢 Initializing Enterprise Features System...")
            self.systems['enterprise'] = await get_enterprise_features_system()
            
            logger.info("🔗 Initializing Ultimate Integration System...")
            self.systems['integration'] = await get_ultimate_integration_system()
            
            logger.info("🚀 Initializing Ultimate Deployment System...")
            self.systems['deployment'] = await get_ultimate_deployment_system()
            
            # Create FastAPI app
            logger.info("🌐 Creating FastAPI Application...")
            self.systems['app'] = create_app()
            
            initialization_time = time.time() - self.startup_time
            logger.info(f"✅ All systems initialized successfully in {initialization_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize systems: {str(e)}")
            return False
    
    async def start_all_systems(self) -> bool:
        """Start all system components"""
        try:
            logger.info("🚀 Starting all system components...")
            
            # Start systems that have start methods
            systems_to_start = [
                'predictive_analytics', 'dashboard', 'cache', 'auto_scaling',
                'security', 'performance', 'monitoring', 'integration'
            ]
            
            for system_name in systems_to_start:
                if system_name in self.systems:
                    system = self.systems[system_name]
                    if hasattr(system, 'start'):
                        await system.start()
                        logger.info(f"✅ Started {system_name}")
            
            self.is_running = True
            logger.info("🎉 All systems started successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start systems: {str(e)}")
            return False
    
    async def run_system_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        try:
            logger.info("🔍 Running system health check...")
            
            health_status = {
                "overall_status": "healthy",
                "components": {},
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": time.time() - self.startup_time if self.startup_time else 0
            }
            
            # Check each system
            for system_name, system in self.systems.items():
                try:
                    if hasattr(system, 'get_health_status'):
                        component_health = system.get_health_status()
                    elif hasattr(system, 'get_integration_statistics'):
                        stats = system.get_integration_statistics()
                        component_health = {
                            "status": "healthy" if stats.get("is_running", False) else "unhealthy",
                            "details": stats
                        }
                    else:
                        component_health = {"status": "unknown", "details": "No health check available"}
                    
                    health_status["components"][system_name] = component_health
                    
                except Exception as e:
                    health_status["components"][system_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Determine overall status
            component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
            if "error" in component_statuses:
                health_status["overall_status"] = "degraded"
            elif "unhealthy" in component_statuses:
                health_status["overall_status"] = "degraded"
            
            logger.info(f"🏥 System health check completed: {health_status['overall_status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {"overall_status": "error", "error": str(e)}
    
    async def display_system_status(self) -> None:
        """Display comprehensive system status"""
        try:
            print("\n" + "="*80)
            print("🚀 ULTIMATE FACEBOOK POSTS SYSTEM v4.0 - STATUS REPORT")
            print("="*80)
            
            # System overview
            uptime = time.time() - self.startup_time if self.startup_time else 0
            print(f"⏱️  Uptime: {uptime:.2f} seconds")
            print(f"🎯 Mode: {self.mode.upper()}")
            print(f"🔄 Status: {'RUNNING' if self.is_running else 'STOPPED'}")
            
            # Component status
            print("\n📊 COMPONENT STATUS:")
            print("-" * 50)
            
            for system_name, system in self.systems.items():
                try:
                    if hasattr(system, 'get_integration_statistics'):
                        stats = system.get_integration_statistics()
                        status = "🟢 RUNNING" if stats.get("is_running", False) else "🔴 STOPPED"
                        print(f"  {system_name:20} {status}")
                    elif hasattr(system, 'get_health_status'):
                        health = system.get_health_status()
                        status = "🟢 HEALTHY" if health.get("status") == "healthy" else "🟡 DEGRADED"
                        print(f"  {system_name:20} {status}")
                    else:
                        print(f"  {system_name:20} 🟡 UNKNOWN")
                except Exception as e:
                    print(f"  {system_name:20} 🔴 ERROR: {str(e)[:30]}...")
            
            # Performance metrics
            print("\n📈 PERFORMANCE METRICS:")
            print("-" * 50)
            
            if 'performance' in self.systems:
                try:
                    perf_summary = self.systems['performance'].get_performance_summary()
                    print(f"  CPU Usage:     {perf_summary.get('cpu_usage', 'N/A'):>10}%")
                    print(f"  Memory Usage:  {perf_summary.get('memory_usage', 'N/A'):>10}%")
                    print(f"  Response Time: {perf_summary.get('response_time', 'N/A'):>10}ms")
                    print(f"  Throughput:    {perf_summary.get('throughput', 'N/A'):>10} req/s")
                except Exception as e:
                    print(f"  Performance metrics unavailable: {str(e)}")
            
            # Security status
            print("\n🔒 SECURITY STATUS:")
            print("-" * 50)
            
            if 'security' in self.systems:
                try:
                    security_stats = self.systems['security'].get_security_statistics()
                    print(f"  Blocked IPs:     {security_stats.get('blocked_ips', 0):>10}")
                    print(f"  Active API Keys: {security_stats.get('active_api_keys', 0):>10}")
                    print(f"  Security Rules:  {security_stats.get('security_rules', 0):>10}")
                    print(f"  Recent Events:   {len(security_stats.get('recent_events', [])):>10}")
                except Exception as e:
                    print(f"  Security metrics unavailable: {str(e)}")
            
            # Cache status
            print("\n🧠 CACHE STATUS:")
            print("-" * 50)
            
            if 'cache' in self.systems:
                try:
                    cache_metrics = self.systems['cache'].get_cache_metrics()
                    print(f"  Total Items:     {cache_metrics.total_items:>10}")
                    print(f"  Hit Rate:        {cache_metrics.hit_rate*100:>9.1f}%")
                    print(f"  Memory Usage:    {cache_metrics.memory_usage_percent:>9.1f}%")
                    print(f"  Avg Access Time: {cache_metrics.average_access_time*1000:>9.1f}ms")
                except Exception as e:
                    print(f"  Cache metrics unavailable: {str(e)}")
            
            print("\n" + "="*80)
            print("🎉 Ultimate System is fully operational!")
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Error displaying system status: {str(e)}")
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark"""
        try:
            logger.info("🏃 Running performance benchmark...")
            
            benchmark_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "tests": {},
                "overall_score": 0
            }
            
            # Test cache performance
            if 'cache' in self.systems:
                start_time = time.time()
                for i in range(1000):
                    await self.systems['cache'].set(f"benchmark_{i}", f"value_{i}", "post_content")
                cache_time = time.time() - start_time
                benchmark_results["tests"]["cache_write"] = {
                    "operations": 1000,
                    "time_seconds": cache_time,
                    "ops_per_second": 1000 / cache_time
                }
            
            # Test prediction performance
            if 'predictive_analytics' in self.systems:
                start_time = time.time()
                for i in range(100):
                    await self.systems['predictive_analytics'].predict(
                        "engagement", f"Test content {i}", datetime.utcnow(), "general"
                    )
                prediction_time = time.time() - start_time
                benchmark_results["tests"]["predictions"] = {
                    "operations": 100,
                    "time_seconds": prediction_time,
                    "ops_per_second": 100 / prediction_time
                }
            
            # Calculate overall score
            if benchmark_results["tests"]:
                scores = [test["ops_per_second"] for test in benchmark_results["tests"].values()]
                benchmark_results["overall_score"] = sum(scores) / len(scores)
            
            logger.info(f"✅ Performance benchmark completed. Overall score: {benchmark_results['overall_score']:.2f} ops/s")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {str(e)}")
            return {"error": str(e)}
    
    async def stop_all_systems(self) -> None:
        """Stop all system components"""
        try:
            logger.info("🛑 Stopping all system components...")
            
            # Stop systems in reverse order
            systems_to_stop = [
                'integration', 'monitoring', 'performance', 'security',
                'auto_scaling', 'cache', 'dashboard', 'predictive_analytics'
            ]
            
            for system_name in systems_to_stop:
                if system_name in self.systems:
                    system = self.systems[system_name]
                    if hasattr(system, 'stop'):
                        await system.stop()
                        logger.info(f"✅ Stopped {system_name}")
            
            self.is_running = False
            logger.info("🛑 All systems stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping systems: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop_all_systems())
    
    async def run_ultimate_system(self) -> None:
        """Run the complete ultimate system"""
        try:
            # Initialize systems
            if not await self.initialize_all_systems():
                logger.error("❌ Failed to initialize systems")
                return
            
            # Start systems
            if not await self.start_all_systems():
                logger.error("❌ Failed to start systems")
                return
            
            # Display initial status
            await self.display_system_status()
            
            # Run health check
            health_status = await self.run_system_health_check()
            logger.info(f"🏥 System health: {health_status['overall_status']}")
            
            # Run performance benchmark
            if self.mode == "development":
                benchmark_results = await self.run_performance_benchmark()
                logger.info(f"🏃 Performance benchmark: {benchmark_results.get('overall_score', 0):.2f} ops/s")
            
            # Keep system running
            logger.info("🎉 Ultimate Facebook Posts System v4.0 is now running!")
            logger.info("Press Ctrl+C to stop the system")
            
            # Main loop
            while self.is_running:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Periodic health check
                health_status = await self.run_system_health_check()
                if health_status['overall_status'] != 'healthy':
                    logger.warning(f"⚠️  System health degraded: {health_status['overall_status']}")
            
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
        finally:
            await self.stop_all_systems()
            logger.info("👋 Ultimate System shutdown complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ultimate Facebook Posts System v4.0")
    parser.add_argument(
        "--mode",
        choices=["development", "staging", "production"],
        default="production",
        help="System mode"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and exit"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark and exit"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and exit"
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = UltimateSystemLauncher(mode=args.mode)
    
    try:
        # Initialize systems
        if not await launcher.initialize_all_systems():
            logger.error("❌ Failed to initialize systems")
            sys.exit(1)
        
        if args.health_check:
            # Run health check
            health_status = await launcher.run_system_health_check()
            print(json.dumps(health_status, indent=2))
            sys.exit(0 if health_status['overall_status'] == 'healthy' else 1)
        
        if args.benchmark:
            # Run benchmark
            benchmark_results = await launcher.run_performance_benchmark()
            print(json.dumps(benchmark_results, indent=2))
            sys.exit(0)
        
        if args.status:
            # Show status
            await launcher.display_system_status()
            sys.exit(0)
        
        # Start systems
        if not await launcher.start_all_systems():
            logger.error("❌ Failed to start systems")
            sys.exit(1)
        
        # Run ultimate system
        await launcher.run_ultimate_system()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Import json for CLI output
    import json
    
    # Run the main function
    asyncio.run(main())

