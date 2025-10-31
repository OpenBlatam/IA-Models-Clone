"""
Blaze AI Edge Computing Module v7.6.0

Advanced edge computing system for IoT devices, providing local processing,
resource optimization, and seamless integration with the main Blaze AI cluster.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
import psutil
import platform
from pathlib import Path
import aiofiles
import asyncio_mqtt
import aioredis

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# Enums
class EdgeNodeType(Enum):
    """Types of edge computing nodes."""
    IOT_DEVICE = "iot_device"
    EDGE_SERVER = "edge_server"
    GATEWAY = "gateway"
    MOBILE_DEVICE = "mobile_device"
    EMBEDDED_SYSTEM = "embedded_system"

class ResourceLevel(Enum):
    """Resource availability levels."""
    CRITICAL = "critical"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OPTIMAL = "optimal"

class SyncStrategy(Enum):
    """Data synchronization strategies."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"

class OfflineMode(Enum):
    """Offline operation modes."""
    DISABLED = "disabled"
    LOCAL_ONLY = "local_only"
    CACHED_OPERATIONS = "cached_operations"
    FULL_OFFLINE = "full_offline"

# Configuration and Data Classes
@dataclass
class EdgeComputingConfig(ModuleConfig):
    """Configuration for Edge Computing module."""
    
    # Node identification
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_name: str = "edge-node"
    node_type: EdgeNodeType = EdgeNodeType.EDGE_SERVER
    
    # Resource management
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 85.0
    max_disk_usage: float = 90.0
    max_network_usage: float = 75.0
    
    # Synchronization settings
    sync_interval: float = 30.0  # seconds
    sync_strategy: SyncStrategy = SyncStrategy.BATCH
    offline_mode: OfflineMode = OfflineMode.CACHED_OPERATIONS
    
    # Local storage
    local_data_path: str = "./edge_data"
    max_local_storage: int = 1024 * 1024 * 1024  # 1GB
    data_retention_days: int = 7
    
    # Network settings
    cluster_endpoint: str = "http://localhost:8000"
    heartbeat_interval: float = 10.0
    connection_timeout: float = 30.0
    
    # Processing capabilities
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0
    enable_local_ml: bool = True
    enable_local_cache: bool = True
    
    # Security
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    ssl_verify: bool = True

@dataclass
class EdgeNodeInfo:
    """Information about an edge computing node."""
    
    node_id: str
    node_name: str
    node_type: EdgeNodeType
    status: str = "online"
    
    # System information
    platform: str = ""
    architecture: str = ""
    python_version: str = ""
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_usage: float = 0.0
    
    # Capabilities
    local_ml_enabled: bool = False
    local_cache_enabled: bool = False
    offline_capable: bool = False
    
    # Timestamps
    last_seen: datetime = field(default_factory=datetime.now)
    last_sync: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "node_type": self.node_type.value,
            "status": self.status,
            "platform": self.platform,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "network_usage": self.network_usage,
            "local_ml_enabled": self.local_ml_enabled,
            "local_cache_enabled": self.local_cache_enabled,
            "offline_capable": self.offline_capable,
            "last_seen": self.last_seen.isoformat(),
            "last_sync": self.last_sync.isoformat()
        }

@dataclass
class EdgeTask:
    """Task to be executed on edge node."""
    
    task_id: str
    task_type: str
    task_data: Dict[str, Any]
    priority: int = 1
    
    # Execution settings
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    
    # Status tracking
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_data": self.task_data,
            "priority": self.priority,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error
        }

@dataclass
class EdgeMetrics:
    """Performance metrics for edge computing operations."""
    
    # Task metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    
    # Performance metrics
    avg_task_duration: float = 0.0
    total_processing_time: float = 0.0
    
    # Resource metrics
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    avg_disk_usage: float = 0.0
    
    # Network metrics
    sync_operations: int = 0
    data_transferred: int = 0
    offline_operations: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_task_metrics(self, duration: float, success: bool):
        """Update task-related metrics."""
        self.total_tasks += 1
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.total_processing_time += duration
        self.avg_task_duration = self.total_processing_time / self.total_tasks

# Core Components
class ResourceMonitor:
    """Monitors system resources and provides optimization recommendations."""
    
    def __init__(self, config: EdgeComputingConfig):
        self.config = config
        self.last_check = datetime.now()
        self.resource_history: List[Dict[str, float]] = []
    
    async def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network usage (simplified)
            network_io = psutil.net_io_counters()
            network_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # MB
            
            resources = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "network_usage": network_usage
            }
            
            self.resource_history.append(resources)
            if len(self.resource_history) > 100:  # Keep last 100 measurements
                self.resource_history.pop(0)
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "network_usage": 0.0
            }
    
    def get_resource_level(self, resources: Dict[str, float]) -> ResourceLevel:
        """Determine resource availability level."""
        max_usage = max(
            resources.get("cpu_usage", 0),
            resources.get("memory_usage", 0),
            resources.get("disk_usage", 0)
        )
        
        if max_usage >= 90:
            return ResourceLevel.CRITICAL
        elif max_usage >= 80:
            return ResourceLevel.LOW
        elif max_usage >= 60:
            return ResourceLevel.MEDIUM
        elif max_usage >= 40:
            return ResourceLevel.HIGH
        else:
            return ResourceLevel.OPTIMAL
    
    async def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for resource optimization."""
        recommendations = []
        resources = await self.get_system_resources()
        
        if resources["cpu_usage"] > self.config.max_cpu_usage:
            recommendations.append("CPU usage high - consider reducing concurrent tasks")
        
        if resources["memory_usage"] > self.config.max_memory_usage:
            recommendations.append("Memory usage high - consider clearing cache")
        
        if resources["disk_usage"] > self.config.max_disk_usage:
            recommendations.append("Disk usage high - consider cleaning old data")
        
        return recommendations

class LocalDataManager:
    """Manages local data storage and synchronization."""
    
    def __init__(self, config: EdgeComputingConfig):
        self.config = config
        self.data_path = Path(config.local_data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.sync_queue: List[Dict[str, Any]] = []
    
    async def store_local_data(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data locally with optional metadata."""
        try:
            file_path = self.data_path / f"{key}.json"
            
            # Check storage limits
            if await self._check_storage_limits():
                await self._cleanup_old_data()
            
            # Prepare data for storage
            storage_data = {
                "data": data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "size": len(str(data))
            }
            
            # Encrypt if enabled
            if self.config.enable_encryption and self.config.encryption_key:
                storage_data = await self._encrypt_data(storage_data)
            
            # Write to file
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(storage_data, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing local data: {e}")
            return False
    
    async def retrieve_local_data(self, key: str) -> Optional[Any]:
        """Retrieve data from local storage."""
        try:
            file_path = self.data_path / f"{key}.json"
            
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                storage_data = json.loads(content)
            
            # Decrypt if needed
            if self.config.enable_encryption and self.config.encryption_key:
                storage_data = await self._decrypt_data(storage_data)
            
            return storage_data.get("data")
            
        except Exception as e:
            logger.error(f"Error retrieving local data: {e}")
            return None
    
    async def queue_for_sync(self, data: Dict[str, Any]):
        """Queue data for synchronization with main cluster."""
        self.sync_queue.append({
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Limit queue size
        if len(self.sync_queue) > 1000:
            self.sync_queue = self.sync_queue[-500:]
    
    async def _check_storage_limits(self) -> bool:
        """Check if storage limits are exceeded."""
        try:
            total_size = sum(
                f.stat().st_size for f in self.data_path.rglob('*') if f.is_file()
            )
            return total_size > self.config.max_local_storage
        except Exception:
            return False
    
    async def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            
            for file_path in self.data_path.rglob('*.json'):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt data for local storage."""
        # Simplified encryption - in production, use proper encryption
        if self.config.encryption_key:
            import hashlib
            encrypted = hashlib.sha256(str(data).encode()).hexdigest()
            return {"encrypted": True, "hash": encrypted, "data": str(data)}
        return data
    
    async def _decrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt data from local storage."""
        # Simplified decryption
        if data.get("encrypted"):
            return {"data": data.get("data", "")}
        return data

class TaskExecutor:
    """Executes tasks locally on edge nodes."""
    
    def __init__(self, config: EdgeComputingConfig):
        self.config = config
        self.active_tasks: Dict[str, EdgeTask] = {}
        self.task_results: Dict[str, Any] = {}
        self.execution_history: List[EdgeTask] = []
    
    async def submit_task(self, task: EdgeTask) -> bool:
        """Submit a task for execution."""
        try:
            if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                logger.warning("Maximum concurrent tasks reached")
                return False
            
            self.active_tasks[task.task_id] = task
            task.status = "queued"
            
            # Start execution
            asyncio.create_task(self._execute_task(task))
            return True
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return False
    
    async def _execute_task(self, task: EdgeTask):
        """Execute a task asynchronously."""
        try:
            task.status = "running"
            task.started_at = datetime.now()
            
            # Execute based on task type
            result = await self._process_task(task)
            
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            # Store result
            self.task_results[task.task_id] = result
            
            # Move to history
            self.execution_history.append(task)
            del self.active_tasks[task.task_id]
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self.submit_task(task)
            else:
                self.execution_history.append(task)
                del self.active_tasks[task.task_id]
    
    async def _process_task(self, task: EdgeTask) -> Any:
        """Process different types of tasks."""
        task_type = task.task_type.lower()
        
        if task_type == "data_processing":
            return await self._process_data_processing(task.task_data)
        elif task_type == "ml_inference":
            return await self._process_ml_inference(task.task_data)
        elif task_type == "data_aggregation":
            return await self._process_data_aggregation(task.task_data)
        elif task_type == "local_analysis":
            return await self._process_local_analysis(task.task_data)
        else:
            # Generic task processing
            return await self._process_generic_task(task.task_data)
    
    async def _process_data_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data locally."""
        # Simulate data processing
        await asyncio.sleep(0.1)
        return {
            "processed": True,
            "input_size": len(str(data)),
            "output_size": len(str(data)) * 2,
            "processing_time": time.time()
        }
    
    async def _process_ml_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML inference locally."""
        # Simulate ML inference
        await asyncio.sleep(0.2)
        return {
            "inference_complete": True,
            "confidence": 0.85,
            "prediction": "sample_prediction",
            "model_version": "local_v1.0"
        }
    
    async def _process_data_aggregation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data locally."""
        # Simulate data aggregation
        await asyncio.sleep(0.15)
        return {
            "aggregated": True,
            "data_points": len(data.get("values", [])),
            "summary": "aggregated_summary",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_local_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform local data analysis."""
        # Simulate local analysis
        await asyncio.sleep(0.25)
        return {
            "analysis_complete": True,
            "insights": ["insight_1", "insight_2"],
            "metrics": {"metric1": 42, "metric2": 84},
            "recommendations": ["rec1", "rec2"]
        }
    
    async def _process_generic_task(self, data: Dict[str, Any]) -> Any:
        """Process generic tasks."""
        # Simulate generic processing
        await asyncio.sleep(0.1)
        return {
            "generic_processed": True,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a specific task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].status
        
        # Check history
        for task in self.execution_history:
            if task.task_id == task_id:
                return task.status
        
        return None
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task."""
        return self.task_results.get(task_id)

class ClusterConnector:
    """Manages connection and synchronization with main cluster."""
    
    def __init__(self, config: EdgeComputingConfig):
        self.config = config
        self.connected = False
        self.last_sync = datetime.now()
        self.sync_errors = 0
        self.connection_task: Optional[asyncio.Task] = None
    
    async def connect_to_cluster(self) -> bool:
        """Establish connection to main cluster."""
        try:
            # Simulate connection
            await asyncio.sleep(0.1)
            self.connected = True
            self.sync_errors = 0
            logger.info("Connected to main cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to cluster: {e}")
            self.connected = False
            return False
    
    async def disconnect_from_cluster(self):
        """Disconnect from main cluster."""
        self.connected = False
        if self.connection_task:
            self.connection_task.cancel()
        logger.info("Disconnected from main cluster")
    
    async def sync_with_cluster(self, data: Dict[str, Any]) -> bool:
        """Synchronize data with main cluster."""
        if not self.connected:
            return False
        
        try:
            # Simulate sync operation
            await asyncio.sleep(0.05)
            self.last_sync = datetime.now()
            self.sync_errors = 0
            return True
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.sync_errors += 1
            return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of main cluster."""
        if not self.connected:
            return {"status": "disconnected"}
        
        try:
            # Simulate cluster status check
            await asyncio.sleep(0.02)
            return {
                "status": "connected",
                "last_sync": self.last_sync.isoformat(),
                "sync_errors": self.sync_errors,
                "cluster_health": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"status": "error", "error": str(e)}

# Main Module
class EdgeComputingModule(BaseModule):
    """Advanced edge computing module for Blaze AI system."""
    
    def __init__(self, config: EdgeComputingConfig):
        super().__init__(config)
        self.config = config
        
        # Core components
        self.resource_monitor = ResourceMonitor(config)
        self.data_manager = LocalDataManager(config)
        self.task_executor = TaskExecutor(config)
        self.cluster_connector = ClusterConnector(config)
        
        # Node information
        self.node_info = EdgeNodeInfo(
            node_id=config.node_id,
            node_name=config.node_name,
            node_type=config.node_type,
            platform=platform.system(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            local_ml_enabled=config.enable_local_ml,
            local_cache_enabled=config.enable_local_cache,
            offline_capable=config.offline_mode != OfflineMode.DISABLED
        )
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.sync_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = EdgeMetrics()
    
    async def initialize(self) -> bool:
        """Initialize the edge computing module."""
        try:
            logger.info("Initializing Edge Computing Module")
            
            # Initialize components
            await self.data_manager.store_local_data("node_info", self.node_info.to_dict())
            
            # Connect to cluster
            await self.cluster_connector.connect_to_cluster()
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.sync_task = asyncio.create_task(self._sync_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.status = ModuleStatus.RUNNING
            logger.info("Edge Computing Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Edge Computing Module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the edge computing module."""
        try:
            logger.info("Shutting down Edge Computing Module")
            
            # Cancel background tasks
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.sync_task:
                self.sync_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Disconnect from cluster
            await self.cluster_connector.disconnect_from_cluster()
            
            # Save final metrics
            await self.data_manager.store_local_data("final_metrics", self.metrics.__dict__)
            
            self.status = ModuleStatus.STOPPED
            logger.info("Edge Computing Module shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
    async def submit_edge_task(self, task_type: str, task_data: Dict[str, Any], 
                              priority: int = 1, timeout: float = None) -> Optional[str]:
        """Submit a task for execution on the edge node."""
        try:
            task = EdgeTask(
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                task_data=task_data,
                priority=priority,
                timeout=timeout or self.config.task_timeout
            )
            
            success = await self.task_executor.submit_task(task)
            if success:
                self.metrics.pending_tasks += 1
                return task.task_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error submitting edge task: {e}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a specific task."""
        return await self.task_executor.get_task_status(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task."""
        return await self.task_executor.get_task_result(task_id)
    
    async def store_local_data(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data locally on the edge node."""
        return await self.data_manager.store_local_data(key, data, metadata)
    
    async def retrieve_local_data(self, key: str) -> Optional[Any]:
        """Retrieve data from local storage."""
        return await self.data_manager.retrieve_local_data(key)
    
    async def get_node_info(self) -> EdgeNodeInfo:
        """Get current node information."""
        # Update resource metrics
        resources = await self.resource_monitor.get_system_resources()
        self.node_info.cpu_usage = resources["cpu_usage"]
        self.node_info.memory_usage = resources["memory_usage"]
        self.node_info.disk_usage = resources["disk_usage"]
        self.node_info.network_usage = resources["network_usage"]
        self.node_info.last_seen = datetime.now()
        
        return self.node_info
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status and recommendations."""
        resources = await self.resource_monitor.get_system_resources()
        recommendations = await self.resource_monitor.get_optimization_recommendations()
        
        return {
            "resources": resources,
            "resource_level": self.resource_monitor.get_resource_level(resources).value,
            "recommendations": recommendations,
            "limits": {
                "max_cpu": self.config.max_cpu_usage,
                "max_memory": self.config.max_memory_usage,
                "max_disk": self.config.max_disk_usage,
                "max_network": self.config.max_network_usage
            }
        }
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of connection to main cluster."""
        return await self.cluster_connector.get_cluster_status()
    
    async def force_sync(self) -> bool:
        """Force synchronization with main cluster."""
        try:
            # Sync node info
            node_info = await self.get_node_info()
            success = await self.cluster_connector.sync_with_cluster({
                "type": "node_update",
                "data": node_info.to_dict()
            })
            
            if success:
                self.node_info.last_sync = datetime.now()
                self.metrics.sync_operations += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Force sync failed: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to main cluster."""
        while True:
            try:
                if self.cluster_connector.connected:
                    await self.cluster_connector.sync_with_cluster({
                        "type": "heartbeat",
                        "node_id": self.config.node_id,
                        "timestamp": datetime.now().isoformat()
                    })
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _sync_loop(self):
        """Periodic synchronization with main cluster."""
        while True:
            try:
                if self.cluster_connector.connected:
                    await self.force_sync()
                
                await asyncio.sleep(self.config.sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data and metrics."""
        while True:
            try:
                # Cleanup old data
                await self.data_manager._cleanup_old_data()
                
                # Update metrics
                self.metrics.last_updated = datetime.now()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def get_metrics(self) -> EdgeMetrics:
        """Get current edge computing metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Get module health status."""
        try:
            resources = await self.resource_monitor.get_system_resources()
            cluster_status = await self.cluster_connector.get_cluster_status()
            
            return {
                "status": self.status.value,
                "node_info": await self.get_node_info(),
                "resource_status": await self.get_resource_status(),
                "cluster_status": cluster_status,
                "metrics": self.metrics.__dict__,
                "active_tasks": len(self.task_executor.active_tasks),
                "last_sync": self.node_info.last_sync.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Factory Functions
def create_edge_computing_module(config: Optional[EdgeComputingConfig] = None) -> EdgeComputingModule:
    """Create an Edge Computing module with the given configuration."""
    if config is None:
        config = EdgeComputingConfig()
    return EdgeComputingModule(config)

def create_edge_computing_module_with_defaults(**kwargs) -> EdgeComputingModule:
    """Create an Edge Computing module with default configuration and custom overrides."""
    config = EdgeComputingConfig(**kwargs)
    return EdgeComputingModule(config)

__all__ = [
    # Enums
    "EdgeNodeType", "ResourceLevel", "SyncStrategy", "OfflineMode",
    
    # Configuration and Data Classes
    "EdgeComputingConfig", "EdgeNodeInfo", "EdgeTask", "EdgeMetrics",
    
    # Core Components
    "ResourceMonitor", "LocalDataManager", "TaskExecutor", "ClusterConnector",
    
    # Main Module
    "EdgeComputingModule",
    
    # Factory Functions
    "create_edge_computing_module", "create_edge_computing_module_with_defaults"
]

