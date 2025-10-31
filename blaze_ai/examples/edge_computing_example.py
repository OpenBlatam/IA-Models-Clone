"""
Blaze AI Edge Computing Module Example

This example demonstrates how to use the Edge Computing module for
IoT devices, edge computing, and local processing capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ..modules import (
    ModuleRegistry,
    create_module_registry,
    create_edge_computing_module,
    create_cache_module,
    create_monitoring_module,
    create_security_module
)
from ..modules.edge_computing import (
    EdgeNodeType,
    ResourceLevel,
    SyncStrategy,
    OfflineMode
)

logger = logging.getLogger(__name__)

class EdgeComputingDemo:
    def __init__(self):
        self.registry = None
        self.edge_module = None
        self.cache_module = None
        self.monitoring_module = None
        self.security_module = None
        self.submitted_tasks: List[str] = []
        self.completed_tasks: List[str] = []

    async def setup_system(self):
        """Set up the edge computing system with all necessary modules."""
        logger.info("Setting up Edge Computing System...")
        
        # Create module registry
        self.registry = create_module_registry()
        
        # Create Edge Computing module with custom configuration
        edge_config = {
            "node_name": "demo-edge-node",
            "node_type": EdgeNodeType.EDGE_SERVER,
            "max_cpu_usage": 70.0,
            "max_memory_usage": 80.0,
            "sync_strategy": SyncStrategy.BATCH,
            "offline_mode": OfflineMode.CACHED_OPERATIONS,
            "enable_local_ml": True,
            "enable_local_cache": True
        }
        
        self.edge_module = create_edge_computing_module_with_defaults(**edge_config)
        
        # Create supporting modules
        self.cache_module = create_cache_module()
        self.monitoring_module = create_monitoring_module()
        self.security_module = create_security_module()
        
        # Register all modules
        await self.registry.register_module(self.edge_module)
        await self.registry.register_module(self.cache_module)
        await self.registry.register_module(self.monitoring_module)
        await self.registry.register_module(self.security_module)
        
        # Initialize all modules
        await self.registry.initialize_all()
        
        logger.info("Edge Computing System setup complete!")

    async def demonstrate_edge_node_capabilities(self):
        """Demonstrate the edge node's capabilities and information."""
        logger.info("\n=== Edge Node Capabilities ===")
        
        # Get node information
        node_info = await self.edge_module.get_node_info()
        logger.info(f"Node ID: {node_info.node_id}")
        logger.info(f"Node Name: {node_info.node_name}")
        logger.info(f"Node Type: {node_info.node_type.value}")
        logger.info(f"Platform: {node_info.platform}")
        logger.info(f"Architecture: {node_info.architecture}")
        logger.info(f"Python Version: {node_info.python_version}")
        logger.info(f"Local ML Enabled: {node_info.local_ml_enabled}")
        logger.info(f"Local Cache Enabled: {node_info.local_cache_enabled}")
        logger.info(f"Offline Capable: {node_info.offline_capable}")

    async def demonstrate_resource_monitoring(self):
        """Demonstrate resource monitoring and optimization."""
        logger.info("\n=== Resource Monitoring ===")
        
        # Get current resource status
        resource_status = await self.edge_module.get_resource_status()
        
        logger.info("Current Resource Usage:")
        resources = resource_status["resources"]
        logger.info(f"  CPU: {resources['cpu_usage']:.1f}%")
        logger.info(f"  Memory: {resources['memory_usage']:.1f}%")
        logger.info(f"  Disk: {resources['disk_usage']:.1f}%")
        logger.info(f"  Network: {resources['network_usage']:.1f} MB")
        
        logger.info(f"Resource Level: {resource_status['resource_level']}")
        
        if resource_status["recommendations"]:
            logger.info("Optimization Recommendations:")
            for rec in resource_status["recommendations"]:
                logger.info(f"  - {rec}")

    async def demonstrate_local_data_storage(self):
        """Demonstrate local data storage capabilities."""
        logger.info("\n=== Local Data Storage ===")
        
        # Store various types of data locally
        test_data = {
            "sensor_readings": [23.5, 24.1, 23.8, 24.3],
            "device_status": "operational",
            "last_maintenance": "2024-01-15",
            "performance_metrics": {"uptime": 99.8, "efficiency": 94.2}
        }
        
        # Store with metadata
        metadata = {
            "source": "temperature_sensor",
            "timestamp": datetime.now().isoformat(),
            "priority": "high"
        }
        
        success = await self.edge_module.store_local_data(
            "sensor_data_001", 
            test_data, 
            metadata
        )
        
        if success:
            logger.info("Data stored locally successfully")
            
            # Retrieve the data
            retrieved_data = await self.edge_module.retrieve_local_data("sensor_data_001")
            if retrieved_data:
                logger.info("Data retrieved successfully:")
                logger.info(f"  Sensor readings: {retrieved_data['sensor_readings']}")
                logger.info(f"  Device status: {retrieved_data['device_status']}")
        else:
            logger.error("Failed to store data locally")

    async def demonstrate_edge_task_execution(self):
        """Demonstrate edge task execution capabilities."""
        logger.info("\n=== Edge Task Execution ===")
        
        # Submit different types of tasks
        task_types = [
            ("data_processing", {"input_data": [1, 2, 3, 4, 5], "operation": "sum"}),
            ("ml_inference", {"model_input": "sample_data", "model_type": "classification"}),
            ("data_aggregation", {"values": [10, 20, 30, 40, 50], "aggregation": "average"}),
            ("local_analysis", {"data_points": 100, "analysis_type": "trend_analysis"})
        ]
        
        for task_type, task_data in task_types:
            task_id = await self.edge_module.submit_edge_task(
                task_type=task_type,
                task_data=task_data,
                priority=1
            )
            
            if task_id:
                self.submitted_tasks.append(task_id)
                logger.info(f"Submitted {task_type} task: {task_id}")
            else:
                logger.error(f"Failed to submit {task_type} task")
        
        # Wait for tasks to complete
        logger.info("Waiting for tasks to complete...")
        await asyncio.sleep(2)
        
        # Check task statuses and results
        for task_id in self.submitted_tasks:
            status = await self.edge_module.get_task_status(task_id)
            logger.info(f"Task {task_id}: {status}")
            
            if status == "completed":
                result = await self.edge_module.get_task_result(task_id)
                if result:
                    logger.info(f"  Result: {result}")
                    self.completed_tasks.append(task_id)

    async def demonstrate_cluster_connectivity(self):
        """Demonstrate cluster connectivity and synchronization."""
        logger.info("\n=== Cluster Connectivity ===")
        
        # Get cluster status
        cluster_status = await self.edge_module.get_cluster_status()
        logger.info(f"Cluster Status: {cluster_status['status']}")
        
        if cluster_status['status'] == 'connected':
            logger.info(f"Last Sync: {cluster_status['last_sync']}")
            logger.info(f"Sync Errors: {cluster_status['sync_errors']}")
            logger.info(f"Cluster Health: {cluster_status['cluster_health']}")
        
        # Force synchronization
        logger.info("Forcing synchronization with cluster...")
        sync_success = await self.edge_module.force_sync()
        
        if sync_success:
            logger.info("Synchronization successful")
        else:
            logger.warning("Synchronization failed")

    async def demonstrate_offline_capabilities(self):
        """Demonstrate offline operation capabilities."""
        logger.info("\n=== Offline Capabilities ===")
        
        # Simulate offline mode by disconnecting from cluster
        logger.info("Simulating offline mode...")
        
        # Store data while "offline"
        offline_data = {
            "offline_operation": True,
            "timestamp": datetime.now().isoformat(),
            "data": "This data was collected while offline"
        }
        
        success = await self.edge_module.store_local_data("offline_data", offline_data)
        if success:
            logger.info("Data stored while offline")
        
        # Process tasks while "offline"
        offline_task_id = await self.edge_module.submit_edge_task(
            "data_processing",
            {"offline": True, "data": [1, 2, 3, 4, 5]},
            priority=2
        )
        
        if offline_task_id:
            logger.info(f"Offline task submitted: {offline_task_id}")
            
            # Wait for completion
            await asyncio.sleep(1)
            
            status = await self.edge_module.get_task_status(offline_task_id)
            logger.info(f"Offline task status: {status}")
            
            if status == "completed":
                result = await self.edge_module.get_task_result(offline_task_id)
                logger.info(f"Offline task result: {result}")

    async def demonstrate_performance_metrics(self):
        """Demonstrate performance monitoring and metrics."""
        logger.info("\n=== Performance Metrics ===")
        
        # Get current metrics
        metrics = await self.edge_module.get_metrics()
        
        logger.info("Edge Computing Metrics:")
        logger.info(f"  Total Tasks: {metrics.total_tasks}")
        logger.info(f"  Completed Tasks: {metrics.completed_tasks}")
        logger.info(f"  Failed Tasks: {metrics.failed_tasks}")
        logger.info(f"  Pending Tasks: {metrics.pending_tasks}")
        logger.info(f"  Average Task Duration: {metrics.avg_task_duration:.3f}s")
        logger.info(f"  Sync Operations: {metrics.sync_operations}")
        logger.info(f"  Offline Operations: {metrics.offline_operations}")
        logger.info(f"  Last Updated: {metrics.last_updated.isoformat()}")

    async def demonstrate_integration_features(self):
        """Demonstrate integration with other modules."""
        logger.info("\n=== Module Integration ===")
        
        # Test integration with Cache module
        cache_key = "edge_integration_test"
        cache_data = {"integration": True, "timestamp": datetime.now().isoformat()}
        
        await self.cache_module.set(cache_key, cache_data, ttl=300)
        retrieved_cache = await self.cache_module.get(cache_key)
        
        if retrieved_cache:
            logger.info("Cache integration working")
        
        # Test integration with Monitoring module
        monitoring_data = {
            "module": "edge_computing",
            "status": "operational",
            "metrics": await self.edge_module.get_metrics().__dict__
        }
        
        await self.monitoring_module.record_metric("edge_computing_status", monitoring_data)
        logger.info("Monitoring integration working")
        
        # Test integration with Security module
        security_status = await self.security_module.health_check()
        logger.info(f"Security module status: {security_status['status']}")

    async def demonstrate_health_monitoring(self):
        """Demonstrate comprehensive health monitoring."""
        logger.info("\n=== Health Monitoring ===")
        
        # Get comprehensive health status
        health_status = await self.edge_module.health_check()
        
        logger.info("Edge Computing Module Health:")
        logger.info(f"  Status: {health_status['status']}")
        logger.info(f"  Active Tasks: {health_status['active_tasks']}")
        logger.info(f"  Last Sync: {health_status['last_sync']}")
        
        # Check resource status
        resource_status = health_status['resource_status']
        logger.info(f"  Resource Level: {resource_status['resource_level']}")
        
        # Check cluster status
        cluster_status = health_status['cluster_status']
        logger.info(f"  Cluster Status: {cluster_status['status']}")

    async def run_demo(self):
        """Run the complete edge computing demonstration."""
        try:
            logger.info("🚀 Starting Blaze AI Edge Computing Demo")
            logger.info("=" * 50)
            
            # Setup system
            await self.setup_system()
            
            # Wait for modules to be ready
            await asyncio.sleep(1)
            
            # Run demonstrations
            await self.demonstrate_edge_node_capabilities()
            await self.demonstrate_resource_monitoring()
            await self.demonstrate_local_data_storage()
            await self.demonstrate_edge_task_execution()
            await self.demonstrate_cluster_connectivity()
            await self.demonstrate_offline_capabilities()
            await self.demonstrate_performance_metrics()
            await self.demonstrate_integration_features()
            await self.demonstrate_health_monitoring()
            
            logger.info("\n" + "=" * 50)
            logger.info("✅ Edge Computing Demo completed successfully!")
            logger.info(f"📊 Total tasks submitted: {len(self.submitted_tasks)}")
            logger.info(f"✅ Tasks completed: {len(self.completed_tasks)}")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # Cleanup
            if self.registry:
                await self.registry.shutdown_all()
                logger.info("System shutdown complete")

async def main():
    """Main function to run the edge computing demo."""
    demo = EdgeComputingDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())

