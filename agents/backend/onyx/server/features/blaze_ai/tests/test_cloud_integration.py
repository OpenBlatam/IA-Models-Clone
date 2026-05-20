"""
Blaze AI Cloud Integration Module Tests

This file provides comprehensive tests for the Cloud Integration Module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from blaze_ai.modules.cloud_integration import (
    CloudIntegrationModule,
    CloudIntegrationConfig,
    CloudProvider,
    ResourceType,
    ScalingPolicy,
    DeploymentStatus,
    AWSManager,
    AzureManager,
    GCPManager,
    KubernetesManager,
    LoadBalancer,
    AutoScaler,
    create_cloud_integration_module,
    create_cloud_integration_module_with_defaults
)

@pytest.fixture
def cloud_config():
    """Create a basic cloud integration configuration."""
    return CloudIntegrationConfig(
        name="test_cloud_integration",
        enabled_providers=[CloudProvider.AWS],
        auto_scaling=True,
        load_balancing=True,
        min_instances=2,
        max_instances=10
    )

@pytest.fixture
async def cloud_module(cloud_config):
    """Create a cloud integration module for testing."""
    module = CloudIntegrationModule(cloud_config)
    # Mock the cloud provider managers to avoid actual API calls
    with patch.object(module, 'aws_manager'), \
         patch.object(module, 'azure_manager'), \
         patch.object(module, 'gcp_manager'), \
         patch.object(module, 'k8s_manager'):
        await module.initialize()
        yield module
        await module.shutdown()

class TestCloudIntegrationConfig:
    """Test CloudIntegrationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CloudIntegrationConfig()
        assert config.name == "cloud_integration"
        assert CloudProvider.AWS in config.enabled_providers
        assert config.auto_scaling is True
        assert config.load_balancing is True
        assert config.min_instances == 1
        assert config.max_instances == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CloudIntegrationConfig(
            name="custom_cloud",
            enabled_providers=[CloudProvider.AZURE, CloudProvider.GCP],
            auto_scaling=False,
            min_instances=5,
            max_instances=20
        )
        assert config.name == "custom_cloud"
        assert CloudProvider.AZURE in config.enabled_providers
        assert CloudProvider.GCP in config.enabled_providers
        assert config.auto_scaling is False
        assert config.min_instances == 5
        assert config.max_instances == 20

class TestAWSManager:
    """Test AWS cloud provider manager."""
    
    @pytest.fixture
    def aws_manager(self, cloud_config):
        """Create an AWS manager instance."""
        return AWSManager(cloud_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, aws_manager):
        """Test AWS manager initialization."""
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.client.return_value = Mock()
            mock_session.return_value.resource.return_value = Mock()
            
            await aws_manager.initialize()
            
            assert aws_manager.ec2_client is not None
            assert aws_manager.ec2_resource is not None
            assert aws_manager.autoscaling_client is not None
            assert aws_manager.elbv2_client is not None
    
    @pytest.mark.asyncio
    async def test_create_instance(self, aws_manager):
        """Test EC2 instance creation."""
        with patch.object(aws_manager, 'ec2_client') as mock_client:
            mock_client.run_instances.return_value = {
                'Instances': [{'InstanceId': 'i-1234567890abcdef0'}]
            }
            
            config = {
                'name': 'test-instance',
                'image_id': 'ami-12345678',
                'instance_type': 't3.micro',
                'key_name': 'test-key',
                'security_groups': ['sg-12345678'],
                'subnet_id': 'subnet-12345678'
            }
            
            instance_id = await aws_manager.create_instance(config)
            
            assert instance_id == 'i-1234567890abcdef0'
            mock_client.run_instances.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_autoscaling_group(self, aws_manager):
        """Test Auto Scaling Group creation."""
        with patch.object(aws_manager, 'autoscaling_client') as mock_client:
            mock_client.create_auto_scaling_group.return_value = {}
            
            config = {
                'name': 'test-asg',
                'min_size': 1,
                'max_size': 5,
                'desired_capacity': 2,
                'launch_template_id': 'lt-12345678',
                'subnet_ids': ['subnet-1', 'subnet-2']
            }
            
            asg_name = await aws_manager.create_autoscaling_group(config)
            
            assert asg_name == 'test-asg'
            mock_client.create_auto_scaling_group.assert_called_once()

class TestAzureManager:
    """Test Azure cloud provider manager."""
    
    @pytest.fixture
    def azure_manager(self, cloud_config):
        """Create an Azure manager instance."""
        return AzureManager(cloud_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, azure_manager):
        """Test Azure manager initialization."""
        with patch('azure.identity.DefaultAzureCredential') as mock_credential, \
             patch('azure.mgmt.compute.ComputeManagementClient') as mock_client:
            mock_credential.return_value = Mock()
            mock_client.return_value = Mock()
            
            await azure_manager.initialize()
            
            assert azure_manager.compute_client is not None
    
    @pytest.mark.asyncio
    async def test_create_vm(self, azure_manager):
        """Test Azure VM creation."""
        with patch.object(azure_manager, 'compute_client') as mock_client:
            mock_poller = Mock()
            mock_poller.result.return_value = Mock()
            mock_poller.result.return_value.name = 'test-vm'
            mock_client.virtual_machines.begin_create_or_update.return_value = mock_poller
            
            config = {
                'vm_name': 'test-vm',
                'vm_size': 'Standard_D2s_v3',
                'publisher': 'Canonical',
                'offer': 'UbuntuServer',
                'sku': '18.04-LTS',
                'version': 'latest',
                'resource_group': 'test-rg',
                'network_interface_id': 'nic-12345678'
            }
            
            vm_name = await azure_manager.create_vm(config)
            
            assert vm_name == 'test-vm'
            mock_client.virtual_machines.begin_create_or_update.assert_called_once()

class TestGCPManager:
    """Test Google Cloud Platform manager."""
    
    @pytest.fixture
    def gcp_manager(self, cloud_config):
        """Create a GCP manager instance."""
        return GCPManager(cloud_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, gcp_manager):
        """Test GCP manager initialization."""
        with patch('google.cloud.compute_v1.InstancesClient') as mock_client:
            mock_client.return_value = Mock()
            
            await gcp_manager.initialize()
            
            assert gcp_manager.compute_client is not None
    
    @pytest.mark.asyncio
    async def test_create_instance(self, gcp_manager):
        """Test GCP compute instance creation."""
        with patch.object(gcp_manager, 'compute_client') as mock_client:
            mock_operation = Mock()
            mock_operation.result.return_value = None
            mock_client.insert.return_value = mock_operation
            
            config = {
                'name': 'test-instance',
                'machine_type': 'n1-standard-1',
                'source_image': 'projects/deeplearning-platform-release/global/images/family/tf-latest-gpu'
            }
            
            instance_name = await gcp_manager.create_instance(config)
            
            assert instance_name == 'test-instance'
            mock_client.insert.assert_called_once()

class TestKubernetesManager:
    """Test Kubernetes cluster manager."""
    
    @pytest.fixture
    def k8s_manager(self, cloud_config):
        """Create a Kubernetes manager instance."""
        return KubernetesManager(cloud_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, k8s_manager):
        """Test Kubernetes manager initialization."""
        with patch('kubernetes.config') as mock_config, \
             patch('kubernetes.client.ApiClient') as mock_client:
            mock_config.load_incluster_config.side_effect = Exception()
            mock_config.load_kube_config.return_value = None
            mock_client.return_value = Mock()
            
            await k8s_manager.initialize()
            
            assert k8s_manager.api_client is not None
    
    @pytest.mark.asyncio
    async def test_create_deployment(self, k8s_manager):
        """Test Kubernetes deployment creation."""
        with patch.object(k8s_manager, 'api_client') as mock_client, \
             patch('kubernetes.client.AppsV1Api') as mock_apps_api:
            mock_response = Mock()
            mock_response.metadata.name = 'test-deployment'
            mock_apps_api.return_value.create_namespaced_deployment.return_value = mock_response
            
            config = {
                'name': 'test-app',
                'image': 'test-image:latest',
                'port': 8080,
                'replicas': 3,
                'namespace': 'default'
            }
            
            deployment_name = await k8s_manager.create_deployment(config)
            
            assert deployment_name == 'test-deployment'
            mock_apps_api.return_value.create_namespaced_deployment.assert_called_once()

class TestLoadBalancer:
    """Test load balancer manager."""
    
    @pytest.fixture
    def load_balancer(self, cloud_config):
        """Create a load balancer instance."""
        return LoadBalancer(cloud_config)
    
    @pytest.mark.asyncio
    async def test_add_target(self, load_balancer):
        """Test adding targets to load balancer."""
        lb_name = "test-lb"
        target = "target-1"
        
        await load_balancer.add_target(lb_name, target)
        
        assert target in load_balancer.targets[lb_name]
    
    @pytest.mark.asyncio
    async def test_remove_target(self, load_balancer):
        """Test removing targets from load balancer."""
        lb_name = "test-lb"
        target = "target-1"
        
        # Add target first
        await load_balancer.add_target(lb_name, target)
        assert target in load_balancer.targets[lb_name]
        
        # Remove target
        await load_balancer.remove_target(lb_name, target)
        assert target not in load_balancer.targets[lb_name]
    
    @pytest.mark.asyncio
    async def test_get_targets(self, load_balancer):
        """Test getting targets from load balancer."""
        lb_name = "test-lb"
        targets = ["target-1", "target-2", "target-3"]
        
        for target in targets:
            await load_balancer.add_target(lb_name, target)
        
        retrieved_targets = await load_balancer.get_targets(lb_name)
        assert set(retrieved_targets) == set(targets)
    
    @pytest.mark.asyncio
    async def test_health_check(self, load_balancer):
        """Test health check functionality."""
        target = "test-target"
        
        is_healthy = await load_balancer.health_check(target)
        
        # Simple health check always returns True in this implementation
        assert is_healthy is True

class TestAutoScaler:
    """Test auto-scaling manager."""
    
    @pytest.fixture
    def auto_scaler(self, cloud_config):
        """Create an auto-scaler instance."""
        return AutoScaler(cloud_config)
    
    @pytest.mark.asyncio
    async def test_add_scaling_policy(self, auto_scaler):
        """Test adding scaling policies."""
        deployment_id = "deployment-1"
        policy = {
            'scaling_policy': ScalingPolicy.CPU_BASED,
            'min_instances': 1,
            'max_instances': 10,
            'target_cpu_utilization': 70.0
        }
        
        await auto_scaler.add_scaling_policy(deployment_id, policy)
        
        assert deployment_id in auto_scaler.scaling_policies
        assert auto_scaler.scaling_policies[deployment_id] == policy
    
    @pytest.mark.asyncio
    async def test_evaluate_scaling_cpu_based_scale_up(self, auto_scaler):
        """Test CPU-based scaling up evaluation."""
        deployment_id = "deployment-1"
        policy = {
            'scaling_policy': ScalingPolicy.CPU_BASED,
            'min_instances': 1,
            'max_instances': 10,
            'target_cpu_utilization': 70.0
        }
        
        await auto_scaler.add_scaling_policy(deployment_id, policy)
        
        metrics = {
            'current_instances': 2,
            'cpu_utilization': 85.0  # Above target
        }
        
        action = await auto_scaler.evaluate_scaling(deployment_id, metrics)
        
        assert action == 'scale_up'
    
    @pytest.mark.asyncio
    async def test_evaluate_scaling_cpu_based_scale_down(self, auto_scaler):
        """Test CPU-based scaling down evaluation."""
        deployment_id = "deployment-1"
        policy = {
            'scaling_policy': ScalingPolicy.CPU_BASED,
            'min_instances': 1,
            'max_instances': 10,
            'target_cpu_utilization': 70.0
        }
        
        await auto_scaler.add_scaling_policy(deployment_id, policy)
        
        metrics = {
            'current_instances': 3,
            'cpu_utilization': 25.0  # Below 50% of target
        }
        
        action = await auto_scaler.evaluate_scaling(deployment_id, metrics)
        
        assert action == 'scale_down'
    
    @pytest.mark.asyncio
    async def test_evaluate_scaling_no_action(self, auto_scaler):
        """Test scaling evaluation when no action is needed."""
        deployment_id = "deployment-1"
        policy = {
            'scaling_policy': ScalingPolicy.CPU_BASED,
            'min_instances': 1,
            'max_instances': 10,
            'target_cpu_utilization': 70.0
        }
        
        await auto_scaler.add_scaling_policy(deployment_id, policy)
        
        metrics = {
            'current_instances': 2,
            'cpu_utilization': 60.0  # Within acceptable range
        }
        
        action = await auto_scaler.evaluate_scaling(deployment_id, metrics)
        
        assert action is None

class TestCloudIntegrationModule:
    """Test the main Cloud Integration module."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, cloud_config):
        """Test module initialization."""
        with patch.object(AWSManager, 'initialize'), \
             patch.object(AzureManager, 'initialize'), \
             patch.object(GCPManager, 'initialize'), \
             patch.object(KubernetesManager, 'initialize'):
            
            module = CloudIntegrationModule(cloud_config)
            await module.initialize()
            
            assert module.status == "active"
            assert module.aws_manager is not None
            assert module.load_balancer is not None
            assert module.auto_scaler is not None
    
    @pytest.mark.asyncio
    async def test_shutdown(self, cloud_module):
        """Test module shutdown."""
        await cloud_module.shutdown()
        
        assert cloud_module.status == "shutdown"
    
    @pytest.mark.asyncio
    async def test_deploy_to_cloud_aws(self, cloud_module):
        """Test deploying to AWS cloud."""
        deployment_config = {
            "name": "test-app",
            "provider": "aws",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "image_id": "ami-12345678",
            "min_instances": 2,
            "max_instances": 8,
            "scaling_policy": "cpu_based",
            "load_balancer": True
        }
        
        with patch.object(cloud_module.aws_manager, 'create_instance') as mock_create:
            mock_create.return_value = "i-1234567890abcdef0"
            
            deployment_id = await cloud_module.deploy_to_cloud(deployment_config)
            
            assert deployment_id is not None
            assert deployment_id in cloud_module.deployments
            assert deployment_id in cloud_module.deployment_statuses
    
    @pytest.mark.asyncio
    async def test_get_deployment_status(self, cloud_module):
        """Test getting deployment status."""
        # First deploy something
        deployment_config = {
            "name": "test-app",
            "provider": "aws",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "image_id": "ami-12345678",
            "min_instances": 2,
            "max_instances": 8,
            "scaling_policy": "cpu_based",
            "load_balancer": True
        }
        
        with patch.object(cloud_module.aws_manager, 'create_instance'):
            deployment_id = await cloud_module.deploy_to_cloud(deployment_config)
        
        # Get status
        status = await cloud_module.get_deployment_status(deployment_id)
        
        assert status is not None
        assert status["deployment_id"] == deployment_id
        assert status["name"] == "test-app"
        assert status["provider"] == "aws"
    
    @pytest.mark.asyncio
    async def test_scale_deployment(self, cloud_module):
        """Test scaling a deployment."""
        # First deploy something
        deployment_config = {
            "name": "test-app",
            "provider": "aws",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "image_id": "ami-12345678",
            "min_instances": 2,
            "max_instances": 8,
            "scaling_policy": "cpu_based",
            "load_balancer": True
        }
        
        with patch.object(cloud_module.aws_manager, 'create_instance'):
            deployment_id = await cloud_module.deploy_to_cloud(deployment_config)
        
        # Scale up
        success = await cloud_module.scale_deployment(deployment_id, 5)
        
        assert success is True
        
        # Check status
        status = await cloud_module.get_deployment_status(deployment_id)
        assert status["current_instances"] == 5
        assert status["target_instances"] == 5
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, cloud_module):
        """Test getting module metrics."""
        metrics = await cloud_module.get_metrics()
        
        assert metrics is not None
        assert hasattr(metrics, 'total_deployments')
        assert hasattr(metrics, 'active_deployments')
        assert hasattr(metrics, 'total_instances')
        assert hasattr(metrics, 'active_instances')
    
    @pytest.mark.asyncio
    async def test_health_check(self, cloud_module):
        """Test module health check."""
        health = await cloud_module.health_check()
        
        assert health is not None
        assert "status" in health
        assert "enabled_providers" in health
        assert "auto_scaling" in health
        assert "load_balancing" in health

class TestFactoryFunctions:
    """Test factory functions."""
    
    @pytest.mark.asyncio
    async def test_create_cloud_integration_module(self, cloud_config):
        """Test creating module with explicit config."""
        with patch.object(CloudIntegrationModule, 'initialize'):
            module = await create_cloud_integration_module(cloud_config)
            
            assert isinstance(module, CloudIntegrationModule)
            assert module.config == cloud_config
    
    @pytest.mark.asyncio
    async def test_create_cloud_integration_module_with_defaults(self):
        """Test creating module with default config and overrides."""
        with patch.object(CloudIntegrationModule, 'initialize'):
            module = await create_cloud_integration_module_with_defaults(
                enabled_providers=[CloudProvider.AZURE],
                auto_scaling=False
            )
            
            assert isinstance(module, CloudIntegrationModule)
            assert CloudProvider.AZURE in module.config.enabled_providers
            assert module.config.auto_scaling is False

@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a complete integration scenario."""
    # Create module
    config = CloudIntegrationConfig(
        enabled_providers=[CloudProvider.AWS],
        auto_scaling=True,
        load_balancing=True
    )
    
    with patch.object(AWSManager, 'initialize'), \
         patch.object(AzureManager, 'initialize'), \
         patch.object(GCPManager, 'initialize'), \
         patch.object(KubernetesManager, 'initialize'):
        
        module = CloudIntegrationModule(config)
        await module.initialize()
        
        # Deploy application
        deployment_config = {
            "name": "integration-test-app",
            "provider": "aws",
            "region": "us-east-1",
            "instance_type": "t3.small",
            "image_id": "ami-12345678",
            "min_instances": 2,
            "max_instances": 10,
            "scaling_policy": "cpu_based",
            "load_balancer": True
        }
        
        with patch.object(module.aws_manager, 'create_instance'):
            deployment_id = await module.deploy_to_cloud(deployment_config)
        
        # Verify deployment
        assert deployment_id in module.deployments
        assert deployment_id in module.deployment_statuses
        
        # Check status
        status = await module.get_deployment_status(deployment_id)
        assert status is not None
        assert status["name"] == "integration-test-app"
        
        # Scale deployment
        success = await module.scale_deployment(deployment_id, 4)
        assert success is True
        
        # Get metrics
        metrics = await module.get_metrics()
        assert metrics.total_deployments > 0
        assert metrics.scaling_events > 0
        
        # Health check
        health = await module.health_check()
        assert health["status"] == "active"
        
        # Cleanup
        await module.shutdown()
        assert module.status == "shutdown"

if __name__ == "__main__":
    pytest.main([__file__])

