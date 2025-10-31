"""
Blaze AI Cloud Integration Module v8.0.0

This module provides comprehensive cloud integration capabilities for multi-cloud
deployment, resource management, and distributed computing across cloud providers.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import json
import hashlib
import boto3
import google.cloud.compute_v1 as compute_v1
from azure.mgmt.compute import ComputeManagementClient
from azure.identity import DefaultAzureCredential
import kubernetes
from kubernetes import client, config

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITALOCEAN = "digitalocean"
    VULTR = "vultr"
    CUSTOM = "custom"

class ResourceType(Enum):
    """Cloud resource types."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    CONTAINER = "container"
    FUNCTION = "function"

class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    FAILED = "failed"
    TERMINATED = "terminated"

class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    MANUAL = "manual"
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    CUSTOM_METRICS = "custom_metrics"
    SCHEDULE_BASED = "schedule_based"

@dataclass
class CloudIntegrationConfig:
    """Configuration for Cloud Integration module."""
    # Basic settings
    name: str = "cloud_integration"
    enabled_providers: List[CloudProvider] = field(default_factory=lambda: [CloudProvider.AWS])
    auto_scaling: bool = True
    load_balancing: bool = True
    
    # AWS settings
    aws_region: str = "us-east-1"
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    
    # Azure settings
    azure_subscription_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_location: str = "East US"
    
    # GCP settings
    gcp_project_id: Optional[str] = None
    gcp_zone: str = "us-central1-a"
    
    # Scaling settings
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    
    # Monitoring settings
    monitoring_interval: float = 60.0  # 1 minute
    health_check_interval: float = 30.0  # 30 seconds

@dataclass
class CloudResource:
    """Cloud resource information."""
    resource_id: str
    provider: CloudProvider
    resource_type: ResourceType
    name: str
    region: str
    status: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    deployment_id: str
    name: str
    provider: CloudProvider
    region: str
    instance_type: str
    image_id: str
    min_instances: int
    max_instances: int
    scaling_policy: ScalingPolicy
    load_balancer: bool
    health_check_path: str = "/health"
    environment_variables: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentStatus:
    """Deployment status information."""
    deployment_id: str
    status: DeploymentStatus
    current_instances: int
    target_instances: int
    cpu_utilization: float
    memory_utilization: float
    network_in: float
    network_out: float
    last_updated: datetime
    health_status: str

@dataclass
class CloudMetrics:
    """Cloud integration metrics."""
    total_deployments: int = 0
    active_deployments: int = 0
    total_instances: int = 0
    active_instances: int = 0
    total_cost: float = 0.0
    monthly_cost: float = 0.0
    scaling_events: int = 0
    failed_deployments: int = 0

class AWSManager:
    """AWS cloud provider manager."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.ec2_client = None
        self.ec2_resource = None
        self.autoscaling_client = None
        self.elbv2_client = None
        
    async def initialize(self):
        """Initialize AWS clients."""
        try:
            session = boto3.Session(
                aws_access_key_id=self.config.aws_access_key,
                aws_secret_access_key=self.config.aws_secret_key,
                region_name=self.config.aws_region
            )
            
            self.ec2_client = session.client('ec2')
            self.ec2_resource = session.resource('ec2')
            self.autoscaling_client = session.client('autoscaling')
            self.elbv2_client = session.client('elbv2')
            
            logger.info("AWS clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    async def create_instance(self, config: Dict[str, Any]) -> str:
        """Create an EC2 instance."""
        try:
            response = self.ec2_client.run_instances(
                ImageId=config['image_id'],
                MinCount=1,
                MaxCount=1,
                InstanceType=config['instance_type'],
                KeyName=config.get('key_name'),
                SecurityGroupIds=config.get('security_groups', []),
                SubnetId=config.get('subnet_id'),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': 'Name', 'Value': config['name']}]
                }]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"Created AWS instance: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create AWS instance: {e}")
            raise
    
    async def create_autoscaling_group(self, config: Dict[str, Any]) -> str:
        """Create an Auto Scaling Group."""
        try:
            response = self.autoscaling_client.create_auto_scaling_group(
                AutoScalingGroupName=config['name'],
                MinSize=config['min_size'],
                MaxSize=config['max_size'],
                DesiredCapacity=config['desired_capacity'],
                LaunchTemplate={
                    'LaunchTemplateId': config['launch_template_id'],
                    'Version': '$Latest'
                },
                VPCZoneIdentifier=config['subnet_ids'],
                TargetGroupARNs=config.get('target_group_arns', [])
            )
            
            logger.info(f"Created AWS Auto Scaling Group: {config['name']}")
            return config['name']
            
        except Exception as e:
            logger.error(f"Failed to create AWS Auto Scaling Group: {e}")
            raise

class AzureManager:
    """Azure cloud provider manager."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.compute_client = None
        self.network_client = None
        
    async def initialize(self):
        """Initialize Azure clients."""
        try:
            credential = DefaultAzureCredential()
            self.compute_client = ComputeManagementClient(
                credential, 
                self.config.azure_subscription_id
            )
            
            logger.info("Azure clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {e}")
            raise
    
    async def create_vm(self, config: Dict[str, Any]) -> str:
        """Create an Azure VM."""
        try:
            vm_parameters = {
                'location': self.config.azure_location,
                'hardware_profile': {
                    'vm_size': config['vm_size']
                },
                'storage_profile': {
                    'image_reference': {
                        'publisher': config['publisher'],
                        'offer': config['offer'],
                        'sku': config['sku'],
                        'version': config['version']
                    }
                },
                'network_profile': {
                    'network_interfaces': [{
                        'id': config['network_interface_id']
                    }]
                }
            }
            
            poller = self.compute_client.virtual_machines.begin_create_or_update(
                config['resource_group'],
                config['vm_name'],
                vm_parameters
            )
            
            vm = poller.result()
            logger.info(f"Created Azure VM: {vm.name}")
            return vm.name
            
        except Exception as e:
            logger.error(f"Failed to create Azure VM: {e}")
            raise

class GCPManager:
    """Google Cloud Platform manager."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.compute_client = None
        
    async def initialize(self):
        """Initialize GCP clients."""
        try:
            self.compute_client = compute_v1.InstancesClient()
            logger.info("GCP clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            raise
    
    async def create_instance(self, config: Dict[str, Any]) -> str:
        """Create a GCP compute instance."""
        try:
            instance = compute_v1.Instance(
                name=config['name'],
                machine_type=f"zones/{self.config.gcp_zone}/machineTypes/{config['machine_type']}",
                disks=[{
                    'boot': True,
                    'auto_delete': True,
                    'device_name': config['name'],
                    'initialize_params': {
                        'source_image': config['source_image']
                    }
                }],
                network_interfaces=[{
                    'network': 'global/networks/default',
                    'access_configs': [{
                        'name': 'External NAT',
                        'type_': 'ONE_TO_ONE_NAT'
                    }]
                }]
            )
            
            operation = self.compute_client.insert(
                project=self.config.gcp_project_id,
                zone=self.config.gcp_zone,
                instance_resource=instance
            )
            
            operation.result()
            logger.info(f"Created GCP instance: {config['name']}")
            return config['name']
            
        except Exception as e:
            logger.error(f"Failed to create GCP instance: {e}")
            raise

class KubernetesManager:
    """Kubernetes cluster manager."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.api_client = None
        
    async def initialize(self):
        """Initialize Kubernetes client."""
        try:
            # Try to load in-cluster config first, then kubeconfig
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            
            self.api_client = client.ApiClient()
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    async def create_deployment(self, config: Dict[str, Any]) -> str:
        """Create a Kubernetes deployment."""
        try:
            apps_v1 = client.AppsV1Api(self.api_client)
            
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=config['name']),
                spec=client.V1DeploymentSpec(
                    replicas=config['replicas'],
                    selector=client.V1LabelSelector(
                        match_labels={"app": config['name']}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": config['name']}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=config['name'],
                                    image=config['image'],
                                    ports=[client.V1ContainerPort(container_port=config['port'])]
                                )
                            ]
                        )
                    )
                )
            )
            
            response = apps_v1.create_namespaced_deployment(
                namespace=config.get('namespace', 'default'),
                body=deployment
            )
            
            logger.info(f"Created Kubernetes deployment: {response.metadata.name}")
            return response.metadata.name
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes deployment: {e}")
            raise

class LoadBalancer:
    """Load balancer manager."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.targets: Dict[str, List[str]] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        
    async def add_target(self, lb_name: str, target: str):
        """Add a target to a load balancer."""
        if lb_name not in self.targets:
            self.targets[lb_name] = []
        
        if target not in self.targets[lb_name]:
            self.targets[lb_name].append(target)
            logger.info(f"Added target {target} to load balancer {lb_name}")
    
    async def remove_target(self, lb_name: str, target: str):
        """Remove a target from a load balancer."""
        if lb_name in self.targets and target in self.targets[lb_name]:
            self.targets[lb_name].remove(target)
            logger.info(f"Removed target {target} from load balancer {lb_name}")
    
    async def get_targets(self, lb_name: str) -> List[str]:
        """Get targets for a load balancer."""
        return self.targets.get(lb_name, [])
    
    async def health_check(self, target: str) -> bool:
        """Perform health check on a target."""
        # Simple health check - in production, this would be more sophisticated
        return True

class AutoScaler:
    """Auto-scaling manager."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    async def add_scaling_policy(self, deployment_id: str, policy: Dict[str, Any]):
        """Add a scaling policy for a deployment."""
        self.scaling_policies[deployment_id] = policy
        logger.info(f"Added scaling policy for deployment {deployment_id}")
    
    async def evaluate_scaling(self, deployment_id: str, metrics: Dict[str, Any]) -> Optional[str]:
        """Evaluate if scaling is needed."""
        if deployment_id not in self.scaling_policies:
            return None
        
        policy = self.scaling_policies[deployment_id]
        current_instances = metrics.get('current_instances', 1)
        
        # Store metrics for analysis
        self.metrics_history[deployment_id].append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Keep only last 100 metrics
        if len(self.metrics_history[deployment_id]) > 100:
            self.metrics_history[deployment_id] = self.metrics_history[deployment_id][-100:]
        
        # CPU-based scaling
        if policy.get('scaling_policy') == ScalingPolicy.CPU_BASED:
            cpu_utilization = metrics.get('cpu_utilization', 0)
            target_cpu = policy.get('target_cpu_utilization', 70.0)
            
            if cpu_utilization > target_cpu and current_instances < policy.get('max_instances', 10):
                return 'scale_up'
            elif cpu_utilization < target_cpu * 0.5 and current_instances > policy.get('min_instances', 1):
                return 'scale_down'
        
        # Memory-based scaling
        elif policy.get('scaling_policy') == ScalingPolicy.MEMORY_BASED:
            memory_utilization = metrics.get('memory_utilization', 0)
            target_memory = policy.get('target_memory_utilization', 80.0)
            
            if memory_utilization > target_memory and current_instances < policy.get('max_instances', 10):
                return 'scale_up'
            elif memory_utilization < target_memory * 0.5 and current_instances > policy.get('min_instances', 1):
                return 'scale_down'
        
        return None

class CloudIntegrationModule:
    """Cloud Integration module for Blaze AI system."""
    
    def __init__(self, config: CloudIntegrationConfig):
        self.config = config
        self.status = "uninitialized"
        
        # Cloud providers
        self.aws_manager: Optional[AWSManager] = None
        self.azure_manager: Optional[AzureManager] = None
        self.gcp_manager: Optional[GCPManager] = None
        self.k8s_manager: Optional[KubernetesManager] = None
        
        # Managers
        self.load_balancer = LoadBalancer(config)
        self.auto_scaler = AutoScaler(config)
        
        # State
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.deployment_statuses: Dict[str, DeploymentStatus] = {}
        self.resources: Dict[str, CloudResource] = {}
        self.metrics = CloudMetrics()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the cloud integration module."""
        try:
            logger.info("Initializing Cloud Integration Module")
            
            # Initialize enabled cloud providers
            if CloudProvider.AWS in self.config.enabled_providers:
                self.aws_manager = AWSManager(self.config)
                await self.aws_manager.initialize()
            
            if CloudProvider.AZURE in self.config.enabled_providers:
                self.azure_manager = AzureManager(self.config)
                await self.azure_manager.initialize()
            
            if CloudProvider.GCP in self.config.enabled_providers:
                self.gcp_manager = GCPManager(self.config)
                await self.gcp_manager.initialize()
            
            # Initialize Kubernetes if needed
            self.k8s_manager = KubernetesManager(self.config)
            await self.k8s_manager.initialize()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.status = "active"
            logger.info("Cloud Integration Module initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Integration Module: {e}")
            self.status = "error"
            raise
    
    async def shutdown(self):
        """Shutdown the cloud integration module."""
        try:
            logger.info("Shutting down Cloud Integration Module")
            
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._health_check_task:
                self._health_check_task.cancel()
            
            self.status = "shutdown"
            logger.info("Cloud Integration Module shut down successfully")
            
        except Exception as e:
            logger.error(f"Failed to shutdown Cloud Integration Module: {e}")
            raise
    
    async def deploy_to_cloud(self, deployment_config: Dict[str, Any]) -> str:
        """Deploy an application to the cloud."""
        try:
            deployment_id = str(uuid.uuid4())
            provider = CloudProvider(deployment_config['provider'])
            
            # Create deployment config
            config = DeploymentConfig(
                deployment_id=deployment_id,
                name=deployment_config['name'],
                provider=provider,
                region=deployment_config['region'],
                instance_type=deployment_config['instance_type'],
                image_id=deployment_config['image_id'],
                min_instances=deployment_config.get('min_instances', self.config.min_instances),
                max_instances=deployment_config.get('max_instances', self.config.max_instances),
                scaling_policy=ScalingPolicy(deployment_config.get('scaling_policy', 'cpu_based')),
                load_balancer=deployment_config.get('load_balancer', True),
                environment_variables=deployment_config.get('environment_variables', {})
            )
            
            # Deploy based on provider
            if provider == CloudProvider.AWS and self.aws_manager:
                instance_id = await self.aws_manager.create_instance({
                    'name': config.name,
                    'image_id': config.image_id,
                    'instance_type': config.instance_type,
                    'key_name': deployment_config.get('key_name'),
                    'security_groups': deployment_config.get('security_groups', []),
                    'subnet_id': deployment_config.get('subnet_id')
                })
                
                # Create auto scaling group if needed
                if config.scaling_policy != ScalingPolicy.MANUAL:
                    await self.aws_manager.create_autoscaling_group({
                        'name': f"{config.name}-asg",
                        'min_size': config.min_instances,
                        'max_size': config.max_instances,
                        'desired_capacity': config.min_instances,
                        'launch_template_id': deployment_config.get('launch_template_id'),
                        'subnet_ids': deployment_config.get('subnet_ids', [])
                    })
                
            elif provider == CloudProvider.AZURE and self.azure_manager:
                vm_name = await self.azure_manager.create_vm({
                    'vm_name': config.name,
                    'vm_size': config.instance_type,
                    'publisher': deployment_config.get('publisher'),
                    'offer': deployment_config.get('offer'),
                    'sku': deployment_config.get('sku'),
                    'version': deployment_config.get('version'),
                    'resource_group': deployment_config.get('resource_group'),
                    'network_interface_id': deployment_config.get('network_interface_id')
                })
                
            elif provider == CloudProvider.GCP and self.gcp_manager:
                instance_name = await self.gcp_manager.create_instance({
                    'name': config.name,
                    'machine_type': config.instance_type,
                    'source_image': config.image_id
                })
            
            # Store deployment configuration
            self.deployments[deployment_id] = config
            
            # Initialize deployment status
            self.deployment_statuses[deployment_id] = DeploymentStatus(
                deployment_id=deployment_id,
                status=DeploymentStatus.RUNNING,
                current_instances=config.min_instances,
                target_instances=config.min_instances,
                cpu_utilization=0.0,
                memory_utilization=0.0,
                network_in=0.0,
                network_out=0.0,
                last_updated=datetime.now(),
                health_status="healthy"
            )
            
            # Add scaling policy
            await self.auto_scaler.add_scaling_policy(deployment_id, {
                'scaling_policy': config.scaling_policy,
                'min_instances': config.min_instances,
                'max_instances': config.max_instances,
                'target_cpu_utilization': self.config.target_cpu_utilization,
                'target_memory_utilization': self.config.target_memory_utilization
            })
            
            # Update metrics
            self.metrics.total_deployments += 1
            self.metrics.active_deployments += 1
            
            logger.info(f"Deployed to {provider.value}: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to deploy to cloud: {e}")
            self.metrics.failed_deployments += 1
            raise
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a deployment."""
        if deployment_id not in self.deployment_statuses:
            return None
        
        status = self.deployment_statuses[deployment_id]
        config = self.deployments.get(deployment_id)
        
        return {
            "deployment_id": deployment_id,
            "name": config.name if config else "Unknown",
            "provider": config.provider.value if config else "Unknown",
            "status": status.status.value,
            "current_instances": status.current_instances,
            "target_instances": status.target_instances,
            "cpu_utilization": status.cpu_utilization,
            "memory_utilization": status.memory_utilization,
            "network_in": status.network_in,
            "network_out": status.network_out,
            "last_updated": status.last_updated.isoformat(),
            "health_status": status.health_status
        }
    
    async def scale_deployment(self, deployment_id: str, target_instances: int) -> bool:
        """Scale a deployment to target number of instances."""
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            config = self.deployments[deployment_id]
            status = self.deployment_statuses[deployment_id]
            
            # Validate scaling limits
            if target_instances < config.min_instances or target_instances > config.max_instances:
                raise ValueError(f"Target instances {target_instances} outside allowed range [{config.min_instances}, {config.max_instances}]")
            
            # Update target instances
            status.target_instances = target_instances
            status.status = DeploymentStatus.SCALING
            status.last_updated = datetime.now()
            
            # In a real implementation, this would trigger actual scaling operations
            # For now, we just update the status
            await asyncio.sleep(2)  # Simulate scaling time
            status.current_instances = target_instances
            status.status = DeploymentStatus.RUNNING
            
            self.metrics.scaling_events += 1
            logger.info(f"Scaled deployment {deployment_id} to {target_instances} instances")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_id}: {e}")
            return False
    
    async def get_metrics(self) -> CloudMetrics:
        """Get cloud integration metrics."""
        # Update active counts
        self.metrics.active_deployments = len([
            d for d in self.deployment_statuses.values() 
            if d.status in [DeploymentStatus.RUNNING, DeploymentStatus.SCALING]
        ])
        
        self.metrics.active_instances = sum(
            d.current_instances for d in self.deployment_statuses.values()
            if d.status in [DeploymentStatus.RUNNING, DeploymentStatus.SCALING]
        )
        
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the cloud integration module."""
        return {
            "status": self.status,
            "enabled_providers": [p.value for p in self.config.enabled_providers],
            "total_deployments": self.metrics.total_deployments,
            "active_deployments": self.metrics.active_deployments,
            "total_instances": self.metrics.total_instances,
            "active_instances": self.metrics.active_instances,
            "auto_scaling": self.config.auto_scaling,
            "load_balancing": self.config.load_balancing
        }
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.status == "active":
            try:
                # Update deployment metrics (simulated)
                for deployment_id, status in self.deployment_statuses.items():
                    if status.status == DeploymentStatus.RUNNING:
                        # Simulate metric updates
                        status.cpu_utilization = 50.0 + (hash(deployment_id) % 40)
                        status.memory_utilization = 60.0 + (hash(deployment_id) % 30)
                        status.network_in = 100.0 + (hash(deployment_id) % 200)
                        status.network_out = 50.0 + (hash(deployment_id) % 150)
                        status.last_updated = datetime.now()
                        
                        # Check if scaling is needed
                        scaling_action = await self.auto_scaler.evaluate_scaling(
                            deployment_id, {
                                'current_instances': status.current_instances,
                                'cpu_utilization': status.cpu_utilization,
                                'memory_utilization': status.memory_utilization
                            }
                        )
                        
                        if scaling_action:
                            if scaling_action == 'scale_up':
                                await self.scale_deployment(deployment_id, status.current_instances + 1)
                            elif scaling_action == 'scale_down':
                                await self.scale_deployment(deployment_id, status.current_instances - 1)
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.status == "active":
            try:
                # Perform health checks on all deployments
                for deployment_id, status in self.deployment_statuses.items():
                    if status.status == DeploymentStatus.RUNNING:
                        # Simple health check - in production, this would check actual endpoints
                        is_healthy = await self.load_balancer.health_check(deployment_id)
                        status.health_status = "healthy" if is_healthy else "unhealthy"
                        
                        if not is_healthy:
                            logger.warning(f"Deployment {deployment_id} health check failed")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)

# Factory functions
async def create_cloud_integration_module(config: CloudIntegrationConfig) -> CloudIntegrationModule:
    """Create a Cloud Integration module with the given configuration."""
    module = CloudIntegrationModule(config)
    await module.initialize()
    return module

async def create_cloud_integration_module_with_defaults(**overrides) -> CloudIntegrationModule:
    """Create a Cloud Integration module with default configuration and custom overrides."""
    config = CloudIntegrationConfig()
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return await create_cloud_integration_module(config)

