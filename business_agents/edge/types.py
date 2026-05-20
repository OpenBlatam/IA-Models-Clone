"""
Edge Computing Types and Definitions
====================================

Type definitions for edge computing and fog computing.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import uuid

class NodeType(Enum):
    """Edge node types."""
    EDGE_SERVER = "edge_server"
    FOG_NODE = "fog_node"
    IOT_DEVICE = "iot_device"
    MOBILE_DEVICE = "mobile_device"
    GATEWAY = "gateway"
    ROUTER = "router"
    SWITCH = "switch"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CLOUD_NODE = "cloud_node"

class ComputeTier(Enum):
    """Compute tier levels."""
    TIER_1 = "tier_1"  # Cloud
    TIER_2 = "tier_2"  # Fog
    TIER_3 = "tier_3"  # Edge
    TIER_4 = "tier_4"  # IoT/Embedded

class TaskType(Enum):
    """Task types for edge computing."""
    COMPUTE_INTENSIVE = "compute_intensive"
    DATA_INTENSIVE = "data_intensive"
    LATENCY_CRITICAL = "latency_critical"
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    ML_INFERENCE = "ml_inference"
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    ANALYTICS = "analytics"

class ResourceType(Enum):
    """Resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    TPU = "tpu"
    FPGA = "fpga"
    POWER = "power"
    BANDWIDTH = "bandwidth"
    SENSOR = "sensor"

class NetworkProtocol(Enum):
    """Network protocols for edge computing."""
    HTTP = "http"
    HTTPS = "https"
    MQTT = "mqtt"
    COAP = "coap"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    UDP = "udp"
    TCP = "tcp"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    LORA = "lora"
    NB_IOT = "nb_iot"
    LTE_M = "lte_m"

@dataclass
class EdgeResource:
    """Edge computing resource definition."""
    id: str
    resource_type: ResourceType
    capacity: float
    available: float
    unit: str
    cost_per_unit: float = 0.0
    power_consumption: float = 0.0
    location: Tuple[float, float] = (0.0, 0.0)  # lat, lon
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def utilization(self) -> float:
        """Get resource utilization percentage."""
        if self.capacity > 0:
            return (self.capacity - self.available) / self.capacity * 100
        return 0.0
    
    def is_available(self, required: float) -> bool:
        """Check if required amount is available."""
        return self.available >= required
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource amount."""
        if self.is_available(amount):
            self.available -= amount
            self.updated_at = datetime.now()
            return True
        return False
    
    def deallocate(self, amount: float):
        """Deallocate resource amount."""
        self.available = min(self.capacity, self.available + amount)
        self.updated_at = datetime.now()

@dataclass
class EdgeNode:
    """Edge computing node."""
    id: str
    name: str
    node_type: NodeType
    compute_tier: ComputeTier
    location: Tuple[float, float] = (0.0, 0.0)  # lat, lon
    resources: Dict[ResourceType, EdgeResource] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    network_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "online"  # online, offline, maintenance, error
    last_heartbeat: datetime = field(default_factory=datetime.now)
    power_status: str = "on"  # on, off, low_power, sleep
    temperature: float = 25.0  # Celsius
    uptime: float = 0.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_resource(self, resource_type: ResourceType) -> Optional[EdgeResource]:
        """Get resource by type."""
        return self.resources.get(resource_type)
    
    def add_resource(self, resource: EdgeResource):
        """Add resource to node."""
        self.resources[resource.resource_type] = resource
        self.updated_at = datetime.now()
    
    def get_total_capacity(self, resource_type: ResourceType) -> float:
        """Get total capacity for resource type."""
        resource = self.get_resource(resource_type)
        return resource.capacity if resource else 0.0
    
    def get_available_capacity(self, resource_type: ResourceType) -> float:
        """Get available capacity for resource type."""
        resource = self.get_resource(resource_type)
        return resource.available if resource else 0.0
    
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (self.status == "online" and 
                self.power_status == "on" and
                self.temperature < 80.0 and
                (datetime.now() - self.last_heartbeat).total_seconds() < 300)

@dataclass
class FogNode(EdgeNode):
    """Fog computing node (extends EdgeNode)."""
    fog_level: int = 2  # 1-4, where 1 is closest to cloud
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    aggregation_capabilities: List[str] = field(default_factory=list)
    caching_capabilities: List[str] = field(default_factory=list)
    data_processing_capabilities: List[str] = field(default_factory=list)

@dataclass
class IoTDevice(EdgeNode):
    """IoT device (extends EdgeNode)."""
    device_type: str = "sensor"
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    battery_level: float = 100.0
    signal_strength: float = 0.0
    data_rate: float = 0.0  # bytes per second
    sleep_mode: bool = False
    wake_interval: int = 60  # seconds
    sensors: List[Dict[str, Any]] = field(default_factory=list)
    actuators: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EdgeTask:
    """Edge computing task."""
    id: str
    name: str
    task_type: TaskType
    priority: int = 5  # 1-10, 10 being highest
    requirements: Dict[ResourceType, float] = field(default_factory=dict)
    data_size: int = 0  # bytes
    estimated_duration: float = 0.0  # seconds
    deadline: Optional[datetime] = None
    source_node: str = ""
    target_nodes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed, cancelled
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_node: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_run_on_node(self, node: EdgeNode) -> bool:
        """Check if task can run on given node."""
        for resource_type, required in self.requirements.items():
            available = node.get_available_capacity(resource_type)
            if available < required:
                return False
        return True
    
    def estimate_cost(self, node: EdgeNode) -> float:
        """Estimate cost of running task on node."""
        total_cost = 0.0
        for resource_type, required in self.requirements.items():
            resource = node.get_resource(resource_type)
            if resource:
                total_cost += required * resource.cost_per_unit
        return total_cost

@dataclass
class EdgeCluster:
    """Edge computing cluster."""
    id: str
    name: str
    description: str
    nodes: List[EdgeNode] = field(default_factory=list)
    cluster_type: str = "homogeneous"  # homogeneous, heterogeneous
    load_balancing_strategy: str = "round_robin"
    failover_strategy: str = "automatic"
    auto_scaling: bool = False
    min_nodes: int = 1
    max_nodes: int = 10
    health_check_interval: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_healthy_nodes(self) -> List[EdgeNode]:
        """Get list of healthy nodes."""
        return [node for node in self.nodes if node.is_healthy()]
    
    def get_total_capacity(self, resource_type: ResourceType) -> float:
        """Get total capacity across all nodes."""
        return sum(node.get_total_capacity(resource_type) for node in self.nodes)
    
    def get_available_capacity(self, resource_type: ResourceType) -> float:
        """Get available capacity across all nodes."""
        return sum(node.get_available_capacity(resource_type) for node in self.nodes)
    
    def add_node(self, node: EdgeNode):
        """Add node to cluster."""
        self.nodes.append(node)
        self.updated_at = datetime.now()
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from cluster."""
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                del self.nodes[i]
                self.updated_at = datetime.now()
                return True
        return False

@dataclass
class NetworkTopology:
    """Network topology for edge computing."""
    id: str
    name: str
    description: str
    nodes: List[EdgeNode] = field(default_factory=list)
    connections: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)  # (from_node, to_node, connection_info)
    protocols: List[NetworkProtocol] = field(default_factory=list)
    bandwidth_capacity: float = 0.0  # Mbps
    latency: float = 0.0  # milliseconds
    reliability: float = 1.0  # 0-1
    security_level: str = "standard"  # low, standard, high, military
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_connection(self, from_node_id: str, to_node_id: str, connection_info: Dict[str, Any]):
        """Add connection between nodes."""
        connection = (from_node_id, to_node_id, connection_info)
        if connection not in self.connections:
            self.connections.append(connection)
            self.updated_at = datetime.now()
    
    def remove_connection(self, from_node_id: str, to_node_id: str):
        """Remove connection between nodes."""
        self.connections = [
            conn for conn in self.connections 
            if not (conn[0] == from_node_id and conn[1] == to_node_id)
        ]
        self.updated_at = datetime.now()
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get list of nodes connected to given node."""
        connected = []
        for from_node, to_node, _ in self.connections:
            if from_node == node_id:
                connected.append(to_node)
            elif to_node == node_id:
                connected.append(from_node)
        return connected

@dataclass
class EdgeWorkload:
    """Edge computing workload."""
    id: str
    name: str
    description: str
    tasks: List[EdgeTask] = field(default_factory=list)
    data_dependencies: List[Tuple[str, str]] = field(default_factory=list)  # (task_id, data_source)
    workflow: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 5
    deadline: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_requirements(self) -> Dict[ResourceType, float]:
        """Get total resource requirements for all tasks."""
        total_requirements = {}
        for task in self.tasks:
            for resource_type, amount in task.requirements.items():
                total_requirements[resource_type] = total_requirements.get(resource_type, 0) + amount
        return total_requirements
    
    def estimate_total_duration(self) -> float:
        """Estimate total workload duration."""
        return sum(task.estimated_duration for task in self.tasks)

@dataclass
class EdgeMetrics:
    """Edge computing metrics."""
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    average_task_duration: float = 0.0
    average_latency: float = 0.0
    network_utilization: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0
    power_consumption: float = 0.0
    data_processed: int = 0  # bytes
    energy_efficiency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EdgePolicy:
    """Edge computing policy."""
    id: str
    name: str
    description: str
    policy_type: str  # resource_allocation, task_scheduling, data_placement, security
    rules: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 5
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
