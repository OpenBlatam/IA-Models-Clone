"""
Blaze AI Distributed Processing Module Example

This example demonstrates how to use the Distributed Processing module for
distributed computing, load balancing, fault tolerance, and cluster management.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ..modules import (
    ModuleRegistry,
    create_module_registry,
    create_distributed_processing_module,
    create_cache_module,
    create_monitoring_module,
    create_security_module
)
from ..modules.distributed_processing import (
    TaskPriority,
    TaskStatus,
    NodeStatus,
    LoadBalancingStrategy,
    FaultToleranceStrategy
)

logger = logging.getLogger(__name__)

# ============================================================================
# DISTRIBUTED PROCESSING DEMONSTRATION
# ============================================================================

class DistributedProcessingDemo:
    """Demonstrates various distributed processing features."""
    
    def __init__(self):
        self.registry = None
        self.distributed_module = None
        self.cache_module = None
        self.monitoring_module = None
        self.security_module = None
        
        # Task tracking
        self.submitted_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        
    async def setup_system(self):
        """Setup the distributed processing system."""
        logger.info("🚀 Setting up Distributed Processing System...")
        
        # Create module registry
        self.registry = create_module_registry()
        
        # Create distributed processing module with enhanced configuration
        self.distributed_module = create_distributed_processing_module(
            name="blaze_distributed",
            node_id="demo_node_001",
            node_name="Demo Distributed Node",
            node_capacity=200,
            node_weight=1.5,
            discovery_port=8888,
            communication_port=8889,
            heartbeat_interval=3.0,
            node_timeout=20.0,
            max_task_retries=5,
            task_timeout=600.0,
            batch_size=20,
            enable_checkpointing=True,
            checkpoint_interval=30.0,
            load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
            enable_auto_scaling=True,
            min_nodes=1,
            max_nodes=50,
            scaling_threshold=0.75,
            fault_tolerance_strategy=FaultToleranceStrategy.CIRCUIT_BREAKER,
            replication_factor=3,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=45.0,
            enable_persistence=True,
            backup_interval=1800.0,
            priority=1
        )
        
        # Create supporting modules
        self.cache_module = create_cache_module("distributed_cache", max_size=2000, priority=2)
        self.monitoring_module = create_monitoring_module("distributed_monitoring", collection_interval=3.0, priority=3)
        self.security_module = create_security_module("distributed_security", priority=4)
        
        # Register modules
        await self.registry.register_module(self.distributed_module)
        await self.registry.register_module(self.cache_module)
        await self.registry.register_module(self.monitoring_module)
        await self.registry.register_module(self.security_module)
        
        logger.info("✅ Distributed processing system setup completed")
    
    async def demonstrate_task_submission(self):
        """Demonstrate various types of task submission."""
        logger.info("📝 Demonstrating Task Submission...")
        
        # Submit different types of tasks with various priorities
        task_types = [
            ("data_processing", {"dataset": "large_dataset.csv", "operation": "aggregation"}, TaskPriority.HIGH),
            ("model_training", {"model": "transformer", "epochs": 100}, TaskPriority.CRITICAL),
            ("inference", {"model_id": "model_123", "input_data": "sample_data"}, TaskPriority.NORMAL),
            ("data_cleaning", {"dataset": "raw_data.json", "filters": ["null", "duplicate"]}, TaskPriority.LOW),
            ("report_generation", {"template": "monthly_report", "data_source": "analytics_db"}, TaskPriority.BACKGROUND)
        ]
        
        for task_type, task_data, priority in task_types:
            task_id = await self.distributed_module.submit_distributed_task(
                task_type=task_type,
                task_data=task_data,
                priority=priority
            )
            
            if task_id:
                self.submitted_tasks.append(task_id)
                logger.info(f"✅ Submitted {priority.name} priority task: {task_type} (ID: {task_id})")
            else:
                logger.warning(f"⚠️ Failed to submit task: {task_type}")
        
        logger.info(f"📊 Total tasks submitted: {len(self.submitted_tasks)}")
    
    async def demonstrate_task_monitoring(self):
        """Demonstrate task monitoring and status tracking."""
        logger.info("👀 Demonstrating Task Monitoring...")
        
        # Monitor task statuses
        for task_id in self.submitted_tasks:
            status = await self.distributed_module.get_task_status(task_id)
            if status:
                logger.info(f"📋 Task {task_id}: {status.value}")
            else:
                logger.warning(f"⚠️ Could not get status for task: {task_id}")
        
        # Simulate some task completion
        logger.info("🔄 Simulating task completion...")
        await asyncio.sleep(2)
        
        # Check cluster status
        cluster_status = await self.distributed_module.get_cluster_status()
        logger.info("📊 Cluster Status:")
        logger.info(f"   Total Nodes: {cluster_status.get('metrics', {}).get('total_nodes', 0)}")
        logger.info(f"   Active Nodes: {cluster_status.get('metrics', {}).get('active_nodes', 0)}")
        logger.info(f"   Total Tasks: {cluster_status.get('metrics', {}).get('total_tasks', 0)}")
        logger.info(f"   Cluster Load: {cluster_status.get('metrics', {}).get('cluster_load', 0.0):.2f}")
        
        tasks_info = cluster_status.get('tasks', {})
        logger.info(f"   Pending Tasks: {tasks_info.get('pending', 0)}")
        logger.info(f"   Running Tasks: {tasks_info.get('running', 0)}")
        logger.info(f"   Completed Tasks: {tasks_info.get('completed', 0)}")
        logger.info(f"   Failed Tasks: {tasks_info.get('failed', 0)}")
    
    async def demonstrate_load_balancing(self):
        """Demonstrate load balancing strategies."""
        logger.info("⚖️ Demonstrating Load Balancing Strategies...")
        
        # Test different load balancing strategies
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            LoadBalancingStrategy.CONSISTENT_HASH,
            LoadBalancingStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            logger.info(f"🔧 Testing {strategy.value} strategy...")
            
            # Submit a test task
            task_id = await self.distributed_module.submit_distributed_task(
                task_type="load_balance_test",
                task_data={"strategy": strategy.value, "test_data": "load_balance_demo"},
                priority=TaskPriority.NORMAL
            )
            
            if task_id:
                logger.info(f"   ✅ Task submitted with {strategy.value}: {task_id}")
                self.submitted_tasks.append(task_id)
            else:
                logger.warning(f"   ⚠️ Failed to submit task with {strategy.value}")
            
            await asyncio.sleep(1)
        
        logger.info("📈 Load balancing demonstration completed")
    
    async def demonstrate_fault_tolerance(self):
        """Demonstrate fault tolerance mechanisms."""
        logger.info("🛡️ Demonstrating Fault Tolerance Mechanisms...")
        
        # Test circuit breaker pattern
        logger.info("🔌 Testing Circuit Breaker Pattern...")
        
        # Submit multiple tasks to test fault tolerance
        fault_tolerance_tasks = []
        for i in range(10):
            task_id = await self.distributed_module.submit_distributed_task(
                task_type="fault_tolerance_test",
                task_data={"test_id": i, "mechanism": "circuit_breaker", "data": f"test_data_{i}"},
                priority=TaskPriority.HIGH
            )
            
            if task_id:
                fault_tolerance_tasks.append(task_id)
                logger.info(f"   ✅ Fault tolerance test task {i+1}: {task_id}")
        
        logger.info(f"📊 Fault tolerance test tasks submitted: {len(fault_tolerance_tasks)}")
        
        # Test task cancellation
        logger.info("❌ Testing Task Cancellation...")
        if fault_tolerance_tasks:
            task_to_cancel = fault_tolerance_tasks[0]
            success = await self.distributed_module.cancel_task(task_to_cancel)
            
            if success:
                logger.info(f"   ✅ Successfully cancelled task: {task_to_cancel}")
                fault_tolerance_tasks.remove(task_to_cancel)
            else:
                logger.warning(f"   ⚠️ Failed to cancel task: {task_to_cancel}")
        
        # Test retry mechanism
        logger.info("🔄 Testing Retry Mechanism...")
        retry_task = await self.distributed_module.submit_distributed_task(
            task_type="retry_test",
            task_data={"test_type": "retry_mechanism", "max_retries": 3},
            priority=TaskPriority.NORMAL
        )
        
        if retry_task:
            logger.info(f"   ✅ Retry test task submitted: {retry_task}")
            self.submitted_tasks.append(retry_task)
    
    async def demonstrate_cluster_management(self):
        """Demonstrate cluster management features."""
        logger.info("🏗️ Demonstrating Cluster Management...")
        
        # Get detailed cluster information
        cluster_status = await self.distributed_module.get_cluster_status()
        
        logger.info("🌐 Cluster Information:")
        nodes = cluster_status.get('nodes', [])
        for node in nodes:
            logger.info(f"   Node: {node.get('node_name', 'Unknown')} ({node.get('node_id', 'Unknown')})")
            logger.info(f"     Status: {node.get('node_status', 'Unknown')}")
            logger.info(f"     Load: {node.get('current_load', 0)}/{node.get('node_capacity', 0)}")
            logger.info(f"     Weight: {node.get('node_weight', 0.0)}")
            logger.info(f"     Last Heartbeat: {node.get('last_heartbeat', 'Unknown')}")
        
        # Test node health monitoring
        logger.info("🏥 Testing Node Health Monitoring...")
        health_status = await self.distributed_module.health_check()
        
        logger.info("📊 Module Health Status:")
        for key, value in health_status.items():
            logger.info(f"   {key}: {value}")
        
        # Test metrics collection
        logger.info("📈 Testing Metrics Collection...")
        metrics = await self.distributed_module.get_metrics()
        
        logger.info("📊 Cluster Metrics:")
        logger.info(f"   Total Nodes: {metrics.total_nodes}")
        logger.info(f"   Active Nodes: {metrics.active_nodes}")
        logger.info(f"   Total Tasks: {metrics.total_tasks}")
        logger.info(f"   Completed Tasks: {metrics.completed_tasks}")
        logger.info(f"   Failed Tasks: {metrics.failed_tasks}")
        logger.info(f"   Average Response Time: {metrics.average_response_time:.2f}s")
        logger.info(f"   Cluster Load: {metrics.cluster_load:.2f}")
        logger.info(f"   Network Latency: {metrics.network_latency:.2f}ms")
        logger.info(f"   Last Updated: {metrics.last_updated}")
    
    async def demonstrate_auto_scaling(self):
        """Demonstrate automatic scaling capabilities."""
        logger.info("📏 Demonstrating Auto-Scaling...")
        
        # Submit many tasks to trigger scaling
        logger.info("🚀 Submitting high-load tasks to trigger scaling...")
        
        scaling_tasks = []
        for i in range(50):
            task_id = await self.distributed_module.submit_distributed_task(
                task_type="scaling_test",
                task_data={"test_id": i, "load_type": "high", "complexity": "intensive"},
                priority=TaskPriority.HIGH
            )
            
            if task_id:
                scaling_tasks.append(task_id)
                if i % 10 == 0:
                    logger.info(f"   ✅ Submitted scaling test task {i+1}/50: {task_id}")
        
        logger.info(f"📊 High-load tasks submitted: {len(scaling_tasks)}")
        
        # Monitor scaling behavior
        logger.info("📈 Monitoring scaling behavior...")
        for i in range(5):
            cluster_status = await self.distributed_module.get_cluster_status()
            cluster_load = cluster_status.get('metrics', {}).get('cluster_load', 0.0)
            
            logger.info(f"   📊 Cluster load at iteration {i+1}: {cluster_load:.2f}")
            
            if cluster_load > 0.8:
                logger.info("   🚨 High load detected - scaling should be triggered!")
            elif cluster_load < 0.2:
                logger.info("   📉 Low load detected - scaling down may occur")
            else:
                logger.info("   ✅ Load is within normal range")
            
            await asyncio.sleep(2)
        
        # Add scaling tasks to main list
        self.submitted_tasks.extend(scaling_tasks)
    
    async def demonstrate_performance_benchmarks(self):
        """Demonstrate performance benchmarks."""
        logger.info("⚡ Demonstrating Performance Benchmarks...")
        
        # Test task submission performance
        logger.info("📝 Testing Task Submission Performance...")
        start_time = time.time()
        
        performance_tasks = []
        for i in range(100):
            task_id = await self.distributed_module.submit_distributed_task(
                task_type="performance_test",
                task_data={"test_id": i, "benchmark": "submission_speed", "data_size": "small"},
                priority=TaskPriority.NORMAL
            )
            
            if task_id:
                performance_tasks.append(task_id)
        
        submission_time = time.time() - start_time
        logger.info(f"   ⚡ Submitted {len(performance_tasks)} tasks in {submission_time:.3f} seconds")
        logger.info(f"   📊 Average submission rate: {len(performance_tasks)/submission_time:.1f} tasks/second")
        
        # Test status checking performance
        logger.info("👀 Testing Status Checking Performance...")
        start_time = time.time()
        
        status_checks = 0
        for task_id in performance_tasks[:20]:  # Check first 20 tasks
            status = await self.distributed_module.get_task_status(task_id)
            if status:
                status_checks += 1
        
        status_time = time.time() - start_time
        logger.info(f"   ⚡ Checked {status_checks} task statuses in {status_time:.3f} seconds")
        logger.info(f"   📊 Average status check rate: {status_checks/status_time:.1f} checks/second")
        
        # Add performance tasks to main list
        self.submitted_tasks.extend(performance_tasks)
        
        logger.info("🎯 Performance benchmarks completed")
    
    async def demonstrate_integration_features(self):
        """Demonstrate integration with other modules."""
        logger.info("🔗 Demonstrating Module Integration...")
        
        # Test integration with cache module
        logger.info("💾 Testing Cache Integration...")
        cache_key = "distributed_demo_cache"
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "node_id": self.distributed_module.config.node_id,
            "total_tasks": len(self.submitted_tasks),
            "integration_test": True
        }
        
        await self.cache_module.set(cache_key, cache_data, tags=["distributed", "demo"])
        cached_data = await self.cache_module.get(cache_key)
        
        if cached_data:
            logger.info(f"   ✅ Cache integration successful: {cached_data}")
        else:
            logger.warning("   ⚠️ Cache integration failed")
        
        # Test integration with monitoring module
        logger.info("📊 Testing Monitoring Integration...")
        monitoring_metrics = await self.monitoring_module.get_metrics()
        logger.info(f"   📈 System metrics collected: {len(monitoring_metrics)} metrics")
        
        # Test integration with security module
        logger.info("🔐 Testing Security Integration...")
        security_metrics = await self.security_module.get_metrics()
        logger.info(f"   🛡️ Security metrics: {security_metrics.total_users} users, {security_metrics.active_sessions} sessions")
        
        logger.info("🔗 Module integration demonstration completed")
    
    async def run_demo(self):
        """Run the complete distributed processing demonstration."""
        try:
            logger.info("🚀 Starting Blaze AI Distributed Processing Module Demonstration")
            
            # Setup system
            await self.setup_system()
            
            # Wait for modules to be ready
            await asyncio.sleep(3)
            
            # Run demonstrations
            await self.demonstrate_task_submission()
            await asyncio.sleep(2)
            
            await self.demonstrate_task_monitoring()
            await asyncio.sleep(2)
            
            await self.demonstrate_load_balancing()
            await asyncio.sleep(2)
            
            await self.demonstrate_fault_tolerance()
            await asyncio.sleep(2)
            
            await self.demonstrate_cluster_management()
            await asyncio.sleep(2)
            
            await self.demonstrate_auto_scaling()
            await asyncio.sleep(3)
            
            await self.demonstrate_performance_benchmarks()
            await asyncio.sleep(2)
            
            await self.demonstrate_integration_features()
            
            logger.info("🎉 Distributed processing demonstration completed successfully!")
            logger.info("🌐 Distributed system is now running and managing cluster operations")
            
            # Final status report
            final_status = await self.distributed_module.get_cluster_status()
            logger.info("📊 Final System Status:")
            logger.info(f"   Total Tasks: {final_status.get('metrics', {}).get('total_tasks', 0)}")
            logger.info(f"   Active Nodes: {final_status.get('metrics', {}).get('active_nodes', 0)}")
            logger.info(f"   Cluster Load: {final_status.get('metrics', {}).get('cluster_load', 0.0):.2f}")
            
            # Keep running for a while to see background processes
            logger.info("⏸️ System will continue running for 30 seconds to demonstrate background processes...")
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Distributed processing demonstration failed: {e}")
            raise
        
        finally:
            # Shutdown system
            if self.registry:
                logger.info("🔄 Shutting down distributed processing system...")
                await self.registry.shutdown()
                logger.info("✅ Distributed processing system shutdown completed")

# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    """Main example function."""
    demo = DistributedProcessingDemo()
    await demo.run_demo()

# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())

