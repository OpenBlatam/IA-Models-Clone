"""
Ultimate Blaze AI System Demo - Showcasing All Advanced Features

This demo demonstrates the complete Blaze AI system with all advanced features:
- Advanced caching with Redis, compression, and encryption
- Comprehensive security with JWT, RBAC, and monitoring
- High-performance task scheduling and worker pools
- Advanced metrics and alerting
- REST API and web dashboard
- Performance optimization and monitoring
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import argparse
import threading

from core.interfaces import CoreConfig, create_development_config
from engines import get_engine_manager
from utils.metrics import get_advanced_metrics_collector
from utils.alerting import get_intelligent_alerting_engine
from utils.advanced_cache import get_advanced_cache, CacheConfig, CacheBackend, CompressionType, EvictionPolicy
from security.security_manager import get_security_manager, SecurityConfig, SecurityLevel, Permission
from performance.task_scheduler import get_task_scheduler, SchedulerConfig, TaskPriority
from api.rest_api import AdvancedRESTAPI
from web.dashboard import AdvancedWebDashboard

# =============================================================================
# Demo Scenarios
# =============================================================================

class UltimateDemoScenarios:
    """Comprehensive demo scenarios showcasing all advanced features."""
    
    def __init__(self, engine_manager, metrics_collector, alerting_engine, 
                 advanced_cache, security_manager, task_scheduler):
        self.engine_manager = engine_manager
        self.metrics_collector = metrics_collector
        self.alerting_engine = alerting_engine
        self.advanced_cache = advanced_cache
        self.security_manager = security_manager
        self.task_scheduler = task_scheduler
        self.logger = None  # Will be set by setup_logging
    
    async def demo_advanced_caching(self):
        """Demonstrate advanced caching features."""
        print("\n🚀 **Advanced Caching System Demo**")
        print("=" * 50)
        
        # Test different cache backends and compression
        test_data = {
            "large_text": "A" * 10000,  # 10KB of data
            "numbers": list(range(10000)),
            "nested": {"level1": {"level2": {"level3": "deep_value"}}}
        }
        
        # Test memory cache
        print("\n📦 Testing Memory Cache Backend...")
        await self.advanced_cache.set("test_memory", test_data, ttl=60)
        cached_data = await self.advanced_cache.get("test_memory")
        print(f"✅ Memory cache: {'Data retrieved successfully' if cached_data else 'Failed'}")
        
        # Test compression
        print("\n🗜️ Testing Compression...")
        await self.advanced_cache.set("compressed_data", test_data, ttl=60)
        cache_stats = await self.advanced_cache.get_stats()
        print(f"✅ Compression ratio: {cache_stats.compression_ratio:.2f}")
        
        # Test encryption
        print("\n🔐 Testing Encryption...")
        await self.advanced_cache.set("encrypted_data", {"secret": "sensitive_info"}, ttl=60)
        encrypted_data = await self.advanced_cache.get("encrypted_data")
        print(f"✅ Encryption: {'Data retrieved successfully' if encrypted_data else 'Failed'}")
        
        # Performance test
        print("\n⚡ Performance Test...")
        start_time = time.time()
        for i in range(100):
            await self.advanced_cache.set(f"perf_test_{i}", f"data_{i}", ttl=60)
        
        for i in range(100):
            await self.advanced_cache.get(f"perf_test_{i}")
        
        duration = time.time() - start_time
        print(f"✅ 200 operations completed in {duration:.3f}s ({200/duration:.0f} ops/sec)")
        
        # Show cache statistics
        stats = await self.advanced_cache.get_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   Hits: {stats.hits}")
        print(f"   Misses: {stats.misses}")
        print(f"   Hit Rate: {stats.hit_rate:.2%}")
        print(f"   Memory Usage: {stats.memory_usage} entries")
    
    async def demo_security_system(self):
        """Demonstrate comprehensive security features."""
        print("\n🛡️ **Advanced Security System Demo**")
        print("=" * 50)
        
        # Test user authentication
        print("\n🔑 Testing User Authentication...")
        session = await self.security_manager.authenticate_user(
            username="admin",
            password="admin_password",
            ip_address="192.168.1.100",
            user_agent="Demo Browser"
        )
        
        if session:
            print(f"✅ Admin user authenticated successfully")
            print(f"   Session ID: {session.id}")
            print(f"   Token: {session.token[:20]}...")
            print(f"   Expires: {session.expires_at}")
        else:
            print("❌ Authentication failed")
            return
        
        # Test authorization
        print("\n🔒 Testing Authorization...")
        user = await self.security_manager.validate_request(
            token=session.token,
            required_permission=Permission.ADMIN,
            required_level=SecurityLevel.ADMIN,
            ip_address="192.168.1.100",
            user_agent="Demo Browser"
        )
        
        if user:
            print(f"✅ Authorization successful for user: {user.username}")
            print(f"   Roles: {', '.join(user.roles)}")
            print(f"   Permissions: {[p.value for p in user.permissions]}")
        else:
            print("❌ Authorization failed")
        
        # Test rate limiting
        print("\n⏱️ Testing Rate Limiting...")
        for i in range(5):
            allowed = await self.security_manager.check_rate_limit(
                identifier="192.168.1.100",
                limit_type="ip"
            )
            print(f"   Request {i+1}: {'Allowed' if allowed else 'Rate Limited'}")
        
        # Show security status
        security_status = await self.security_manager.get_security_status()
        print(f"\n📊 Security Status:")
        print(f"   Active Sessions: {security_status['authentication']['active_sessions']}")
        print(f"   Total Users: {security_status['authentication']['total_users']}")
        print(f"   Locked Users: {security_status['authentication']['locked_users']}")
        print(f"   Total Roles: {security_status['authorization']['total_roles']}")
    
    async def demo_task_scheduler(self):
        """Demonstrate high-performance task scheduling."""
        print("\n⚙️ **Advanced Task Scheduler Demo**")
        print("=" * 50)
        
        # Define sample tasks
        async def sample_task(name: str, duration: float) -> str:
            """Sample async task."""
            await asyncio.sleep(duration)
            return f"Task {name} completed in {duration}s"
        
        def sample_sync_task(name: str, duration: float) -> str:
            """Sample sync task."""
            time.sleep(duration)
            return f"Sync task {name} completed in {duration}s"
        
        # Submit tasks with different priorities
        print("\n📋 Submitting Tasks with Different Priorities...")
        
        task_ids = []
        
        # High priority task
        task_id = await self.task_scheduler.submit_task(
            name="High Priority Task",
            func=sample_task,
            "high_priority",
            0.5,
            priority=TaskPriority.HIGH
        )
        task_ids.append(task_id)
        print(f"   High Priority Task submitted: {task_id}")
        
        # Normal priority task
        task_id = await self.task_scheduler.submit_task(
            name="Normal Priority Task",
            func=sample_task,
            "normal_priority",
            1.0,
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
        print(f"   Normal Priority Task submitted: {task_id}")
        
        # Background task
        task_id = await self.task_scheduler.submit_task(
            name="Background Task",
            func=sample_task,
            "background",
            2.0,
            priority=TaskPriority.BACKGROUND
        )
        task_ids.append(task_id)
        print(f"   Background Task submitted: {task_id}")
        
        # Submit sync task
        task_id = await self.task_scheduler.submit_task_sync(
            name="Sync Task",
            func=sample_sync_task,
            "sync_task",
            0.3,
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
        print(f"   Sync Task submitted: {task_id}")
        
        # Wait for all tasks to complete
        print("\n⏳ Waiting for tasks to complete...")
        results = []
        for task_id in task_ids:
            try:
                result = await self.task_scheduler.wait_for_task(task_id, timeout=10)
                results.append(result)
                print(f"   ✅ {result}")
            except Exception as e:
                print(f"   ❌ Task {task_id} failed: {e}")
        
        # Show scheduler statistics
        scheduler_stats = await self.task_scheduler.get_scheduler_stats()
        print(f"\n📊 Scheduler Statistics:")
        print(f"   Queue Size: {scheduler_stats['queue']['size']}")
        print(f"   Total Workers: {scheduler_stats['workers']['total_workers']}")
        print(f"   Available Workers: {scheduler_stats['workers']['available_workers']}")
        print(f"   Worker Utilization: {scheduler_stats['workers']['utilization']:.2%}")
        print(f"   Total Tasks: {scheduler_stats['tasks']['total']}")
        print(f"   Completed Tasks: {scheduler_stats['tasks']['completed']}")
        print(f"   Failed Tasks: {scheduler_stats['tasks']['failed']}")
    
    async def demo_integration_features(self):
        """Demonstrate integration between all systems."""
        print("\n🔗 **System Integration Demo**")
        print("=" * 50)
        
        # Create a complex workflow
        print("\n🔄 Creating Complex Workflow...")
        
        # Step 1: Cache some data
        workflow_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": str(uuid.uuid4()),
            "steps": ["cache", "process", "store", "monitor"]
        }
        
        await self.advanced_cache.set("workflow_data", workflow_data, ttl=300)
        print("   ✅ Step 1: Data cached")
        
        # Step 2: Process with task scheduler
        async def process_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
            """Process workflow data."""
            await asyncio.sleep(0.5)  # Simulate processing
            data["processed"] = True
            data["processing_time"] = datetime.utcnow().isoformat()
            return data
        
        task_id = await self.task_scheduler.submit_task(
            name="Workflow Processing",
            func=process_workflow,
            workflow_data,
            priority=TaskPriority.HIGH
        )
        
        result = await self.task_scheduler.wait_for_task(task_id, timeout=10)
        print("   ✅ Step 2: Data processed")
        
        # Step 3: Store processed data
        await self.advanced_cache.set("processed_workflow", result, ttl=600)
        print("   ✅ Step 3: Data stored")
        
        # Step 4: Monitor with metrics
        self.metrics_collector.increment_counter(
            "workflow_completed",
            "workflow",
            {"status": "success", "workflow_id": workflow_data["workflow_id"]}
        )
        
        self.metrics_collector.set_gauge(
            "workflow_duration",
            "workflow",
            0.5,  # Simulated duration
            {"workflow_id": workflow_data["workflow_id"]}
        )
        
        print("   ✅ Step 4: Metrics recorded")
        
        # Step 5: Security audit
        await self.security_manager.security_monitor.log_event(
            event_type="workflow_completed",
            user_id="admin",
            ip_address="192.168.1.100",
            user_agent="Demo System",
            details={"workflow_id": workflow_data["workflow_id"], "status": "success"},
            severity="info"
        )
        print("   ✅ Step 5: Security audit logged")
        
        print("\n🎉 Workflow completed successfully!")
        
        # Show integration metrics
        print(f"\n📊 Integration Metrics:")
        cache_stats = await self.advanced_cache.get_stats()
        print(f"   Cache Hit Rate: {cache_stats.hit_rate:.2%}")
        
        scheduler_stats = await self.task_scheduler.get_scheduler_stats()
        print(f"   Task Success Rate: {scheduler_stats['workers']['workers']['worker_0']['success_rate']:.2%}")
        
        security_status = await self.security_manager.get_security_status()
        print(f"   Security Events: {security_status['security_monitoring']['total_events']}")
    
    async def run_comprehensive_demo(self):
        """Run the complete comprehensive demo."""
        print("🚀 **ULTIMATE BLAZE AI SYSTEM DEMO**")
        print("=" * 60)
        print("This demo showcases ALL advanced features:")
        print("• Advanced Caching (Redis, Compression, Encryption)")
        print("• Comprehensive Security (JWT, RBAC, Monitoring)")
        print("• High-Performance Task Scheduling")
        print("• Advanced Metrics and Alerting")
        print("• REST API and Web Dashboard")
        print("• Performance Optimization")
        print("=" * 60)
        
        try:
            # Run all demo scenarios
            await self.demo_advanced_caching()
            await self.demo_security_system()
            await self.demo_task_scheduler()
            await self.demo_integration_features()
            
            print("\n🎉 **ALL DEMOS COMPLETED SUCCESSFULLY!**")
            print("The Blaze AI system is now running with all advanced features!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

# =============================================================================
# Service Manager
# =============================================================================

class UltimateServiceManager:
    """Manages all system services for the ultimate demo."""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.logger = None
        self.rest_api = None
        self.web_dashboard = None
        self.rest_api_thread = None
        self.dashboard_thread = None
        
        # Initialize core services
        self.engine_manager = None
        self.metrics_collector = None
        self.alerting_engine = None
        self.advanced_cache = None
        self.security_manager = None
        self.task_scheduler = None
    
    async def initialize_services(self):
        """Initialize all system services."""
        print("🔧 Initializing Ultimate Blaze AI System...")
        
        # Initialize core services
        self.engine_manager = get_engine_manager(self.config)
        self.metrics_collector = get_advanced_metrics_collector(self.config)
        self.alerting_engine = get_intelligent_alerting_engine(self.config)
        
        # Initialize advanced cache with Redis support
        cache_config = CacheConfig(
            backend=CacheBackend.HYBRID,
            compression=CompressionType.GZIP,
            encryption=True,
            eviction_policy=EvictionPolicy.HYBRID
        )
        self.advanced_cache = get_advanced_cache(cache_config)
        
        # Initialize security manager
        security_config = SecurityConfig(
            rate_limit_enabled=True,
            enable_audit_logging=True,
            enable_security_monitoring=True
        )
        self.security_manager = get_security_manager(security_config)
        
        # Initialize task scheduler
        scheduler_config = SchedulerConfig(
            max_workers=8,
            enable_worker_autoscaling=True,
            enable_priority_queue=True
        )
        self.task_scheduler = get_task_scheduler(scheduler_config)
        
        # Initialize REST API
        self.rest_api = AdvancedRESTAPI(self.config)
        
        # Initialize web dashboard
        self.web_dashboard = AdvancedWebDashboard(self.config)
        
        print("✅ All services initialized successfully!")
    
    def start_rest_api(self):
        """Start REST API in background thread."""
        if self.rest_api:
            self.rest_api_thread = threading.Thread(
                target=self.rest_api.start,
                daemon=True
            )
            self.rest_api_thread.start()
            print("🌐 REST API started in background")
    
    def start_web_dashboard(self):
        """Start web dashboard in background thread."""
        if self.web_dashboard:
            self.dashboard_thread = threading.Thread(
                target=self.web_dashboard.start,
                daemon=True
            )
            self.dashboard_thread.start()
            print("📊 Web Dashboard started in background")
    
    async def shutdown_services(self):
        """Shutdown all services gracefully."""
        print("\n🔄 Shutting down services...")
        
        if self.task_scheduler:
            await self.task_scheduler.shutdown()
        
        if self.security_manager:
            await self.security_manager.shutdown()
        
        if self.advanced_cache:
            await self.advanced_cache.shutdown()
        
        if self.alerting_engine:
            await self.alerting_engine.shutdown()
        
        if self.metrics_collector:
            await self.metrics_collector.shutdown()
        
        if self.engine_manager:
            await self.engine_manager.shutdown()
        
        print("✅ All services shut down successfully")

# =============================================================================
# Main Demo Function
# =============================================================================

async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Ultimate Blaze AI System Demo")
    parser.add_argument("--start-services", action="store_true", 
                       help="Start REST API and web dashboard")
    parser.add_argument("--demo", choices=["all", "cache", "security", "scheduler", "integration"],
                       default="all", help="Specific demo to run")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = create_development_config()
    
    # Initialize service manager
    service_manager = UltimateServiceManager(config)
    await service_manager.initialize_services()
    
    # Start services if requested
    if args.start_services:
        service_manager.start_rest_api()
        service_manager.start_web_dashboard()
        print("\n🌐 Services started! Access:")
        print("   REST API: http://localhost:8000")
        print("   Web Dashboard: http://localhost:5000")
        print("   API Docs: http://localhost:8000/docs")
    
    # Create demo scenarios
    demo_scenarios = UltimateDemoScenarios(
        service_manager.engine_manager,
        service_manager.metrics_collector,
        service_manager.alerting_engine,
        service_manager.advanced_cache,
        service_manager.security_manager,
        service_manager.task_scheduler
    )
    
    # Run specific demo or all
    if args.demo == "all":
        await demo_scenarios.run_comprehensive_demo()
    elif args.demo == "cache":
        await demo_scenarios.demo_advanced_caching()
    elif args.demo == "security":
        await demo_scenarios.demo_security_system()
    elif args.demo == "scheduler":
        await demo_scenarios.demo_task_scheduler()
    elif args.demo == "integration":
        await demo_scenarios.demo_integration_features()
    
    # Save demo results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "demo_type": args.demo,
        "services_started": args.start_services,
        "status": "completed"
    }
    
    with open("ultimate_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Demo results saved to: ultimate_demo_results.json")
    
    # Keep services running if started
    if args.start_services:
        print("\n🔄 Services are running. Press Ctrl+C to stop...")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    # Shutdown services
    await service_manager.shutdown_services()
    print("\n👋 Ultimate Blaze AI System Demo completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


