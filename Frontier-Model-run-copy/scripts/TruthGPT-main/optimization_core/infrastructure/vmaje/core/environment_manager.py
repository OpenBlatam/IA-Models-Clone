"""
Environment Manager
===================

Advanced environment management system with:
- Multi-environment support (dev, staging, prod)
- Environment isolation and security
- Configuration management
- Resource provisioning
- Environment templates
- Automated environment setup
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import yaml
import threading
from pathlib import Path


class EnvironmentType(Enum):
    """Environment type enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"


class EnvironmentStatus(Enum):
    """Environment status enumeration"""
    CREATING = "creating"
    ACTIVE = "active"
    UPDATING = "updating"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    name: str
    env_type: EnvironmentType
    region: str
    vm_count: int = 1
    vm_size: str = "Standard_D2s_v3"
    gpu_enabled: bool = False
    gpu_count: int = 0
    storage_size: int = 100  # GB
    network_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Environment:
    """Environment representation"""
    id: str
    config: EnvironmentConfig
    status: EnvironmentStatus
    created_at: datetime
    updated_at: datetime
    vms: List[str] = field(default_factory=list)  # VM IDs
    resources: Dict[str, Any] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentManager:
    """
    Advanced Environment Manager.
    
    Features:
    - Multi-environment support
    - Environment isolation and security
    - Configuration management
    - Resource provisioning
    - Environment templates
    - Automated environment setup
    """
    
    def __init__(self, vm_manager, config: Optional[Dict[str, Any]] = None):
        self.vm_manager = vm_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Environment storage
        self.environments: Dict[str, Environment] = {}
        self.environment_templates: Dict[str, EnvironmentConfig] = {}
        
        # Environment configurations
        self._load_environment_templates()
        
        # Threading
        self.monitoring_thread = None
        self.running = False
        
        # Start monitoring
        self.start_monitoring()
    
    def _load_environment_templates(self):
        """Load environment templates"""
        # Development template
        self.environment_templates['development'] = EnvironmentConfig(
            name='dev-template',
            env_type=EnvironmentType.DEVELOPMENT,
            region='eastus',
            vm_count=1,
            vm_size='Standard_D2s_v3',
            gpu_enabled=False,
            storage_size=50,
            tags={'environment': 'development', 'team': 'ml-engineering'}
        )
        
        # Staging template
        self.environment_templates['staging'] = EnvironmentConfig(
            name='staging-template',
            env_type=EnvironmentType.STAGING,
            region='eastus',
            vm_count=2,
            vm_size='Standard_D4s_v3',
            gpu_enabled=True,
            gpu_count=1,
            storage_size=100,
            tags={'environment': 'staging', 'team': 'ml-engineering'}
        )
        
        # Production template
        self.environment_templates['production'] = EnvironmentConfig(
            name='prod-template',
            env_type=EnvironmentType.PRODUCTION,
            region='eastus',
            vm_count=5,
            vm_size='Standard_D8s_v3',
            gpu_enabled=True,
            gpu_count=2,
            storage_size=500,
            tags={'environment': 'production', 'team': 'ml-engineering'}
        )
    
    def start_monitoring(self):
        """Start environment monitoring"""
        if self.monitoring_thread:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Environment monitoring started")
    
    def stop_monitoring(self):
        """Stop environment monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Environment monitoring stopped")
    
    def _monitoring_loop(self):
        """Environment monitoring loop"""
        while self.running:
            try:
                self._update_environment_statuses()
                time.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"Error in environment monitoring: {e}")
                time.sleep(60)
    
    def _update_environment_statuses(self):
        """Update environment statuses"""
        for env_id, env in self.environments.items():
            try:
                # Check VM statuses
                vm_statuses = []
                for vm_id in env.vms:
                    vm = self.vm_manager.get_vm(vm_id)
                    if vm:
                        vm_statuses.append(vm.status.value)
                
                # Update environment status based on VM statuses
                if not vm_statuses:
                    env.status = EnvironmentStatus.STOPPED
                elif all(status == 'running' for status in vm_statuses):
                    env.status = EnvironmentStatus.ACTIVE
                elif any(status == 'error' for status in vm_statuses):
                    env.status = EnvironmentStatus.ERROR
                else:
                    env.status = EnvironmentStatus.UPDATING
                
                env.updated_at = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Failed to update environment {env.name}: {e}")
    
    async def create_environment(self, config: EnvironmentConfig) -> Environment:
        """Create a new environment"""
        self.logger.info(f"Creating environment: {config.name}")
        
        env_id = str(uuid.uuid4())
        
        environment = Environment(
            id=env_id,
            config=config,
            status=EnvironmentStatus.CREATING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store environment
        self.environments[env_id] = environment
        
        try:
            # Create VMs for environment
            await self._provision_environment_vms(environment)
            
            # Configure environment
            await self._configure_environment(environment)
            
            # Update status
            environment.status = EnvironmentStatus.ACTIVE
            environment.updated_at = datetime.now()
            
            self.logger.info(f"Environment created successfully: {config.name}")
            return environment
            
        except Exception as e:
            environment.status = EnvironmentStatus.ERROR
            environment.updated_at = datetime.now()
            self.logger.error(f"Failed to create environment {config.name}: {e}")
            raise
    
    async def _provision_environment_vms(self, environment: Environment):
        """Provision VMs for environment"""
        for i in range(environment.config.vm_count):
            vm_name = f"{environment.config.name}-vm-{i+1}"
            
            # Create VM config
            vm_config = self._create_vm_config(vm_name, environment.config)
            
            # Create VM
            vm = await self.vm_manager.create_vm(vm_config)
            environment.vms.append(vm.id)
            
            self.logger.info(f"Created VM {vm_name} for environment {environment.config.name}")
    
    def _create_vm_config(self, vm_name: str, env_config: EnvironmentConfig):
        """Create VM config for environment"""
        from .vm_manager import VMConfig, VMType, CloudProvider
        
        # Determine VM type based on GPU requirements
        if env_config.gpu_enabled:
            vm_type = VMType.GPU_OPTIMIZED
        else:
            vm_type = VMType.GENERAL_PURPOSE
        
        return VMConfig(
            name=vm_name,
            provider=CloudProvider.AZURE,  # Default to Azure
            vm_type=vm_type,
            size=env_config.vm_size,
            region=env_config.region,
            image="Ubuntu2204",  # Default image
            disk_size=env_config.storage_size,
            gpu_count=env_config.gpu_count if env_config.gpu_enabled else 0,
            tags=env_config.tags,
            metadata={'environment_id': env_config.name}
        )
    
    async def _configure_environment(self, environment: Environment):
        """Configure environment after VM creation"""
        # This would include:
        # - Installing required software
        # - Configuring networking
        # - Setting up monitoring
        # - Configuring security
        # - Setting up storage
        
        self.logger.info(f"Configuring environment: {environment.config.name}")
        
        # Placeholder for configuration logic
        await asyncio.sleep(1)  # Simulate configuration time
    
    async def start_environment(self, env_id: str) -> bool:
        """Start an environment"""
        if env_id not in self.environments:
            raise ValueError(f"Environment not found: {env_id}")
        
        environment = self.environments[env_id]
        self.logger.info(f"Starting environment: {environment.config.name}")
        
        try:
            # Start all VMs in environment
            for vm_id in environment.vms:
                await self.vm_manager.start_vm(vm_id)
            
            environment.status = EnvironmentStatus.ACTIVE
            environment.updated_at = datetime.now()
            
            self.logger.info(f"Environment started successfully: {environment.config.name}")
            return True
            
        except Exception as e:
            environment.status = EnvironmentStatus.ERROR
            environment.updated_at = datetime.now()
            self.logger.error(f"Failed to start environment {environment.config.name}: {e}")
            return False
    
    async def stop_environment(self, env_id: str) -> bool:
        """Stop an environment"""
        if env_id not in self.environments:
            raise ValueError(f"Environment not found: {env_id}")
        
        environment = self.environments[env_id]
        self.logger.info(f"Stopping environment: {environment.config.name}")
        
        try:
            # Stop all VMs in environment
            for vm_id in environment.vms:
                await self.vm_manager.stop_vm(vm_id)
            
            environment.status = EnvironmentStatus.STOPPED
            environment.updated_at = datetime.now()
            
            self.logger.info(f"Environment stopped successfully: {environment.config.name}")
            return True
            
        except Exception as e:
            environment.status = EnvironmentStatus.ERROR
            environment.updated_at = datetime.now()
            self.logger.error(f"Failed to stop environment {environment.config.name}: {e}")
            return False
    
    async def delete_environment(self, env_id: str) -> bool:
        """Delete an environment"""
        if env_id not in self.environments:
            raise ValueError(f"Environment not found: {env_id}")
        
        environment = self.environments[env_id]
        self.logger.info(f"Deleting environment: {environment.config.name}")
        
        try:
            # Delete all VMs in environment
            for vm_id in environment.vms:
                await self.vm_manager.delete_vm(vm_id)
            
            environment.status = EnvironmentStatus.DELETED
            environment.updated_at = datetime.now()
            
            # Remove from storage
            del self.environments[env_id]
            
            self.logger.info(f"Environment deleted successfully: {environment.config.name}")
            return True
            
        except Exception as e:
            environment.status = EnvironmentStatus.ERROR
            environment.updated_at = datetime.now()
            self.logger.error(f"Failed to delete environment {environment.config.name}: {e}")
            return False
    
    def get_environment(self, env_id: str) -> Optional[Environment]:
        """Get environment by ID"""
        return self.environments.get(env_id)
    
    def list_environments(self, env_type: Optional[EnvironmentType] = None) -> List[Environment]:
        """List environments"""
        environments = list(self.environments.values())
        
        if env_type:
            environments = [env for env in environments if env.config.env_type == env_type]
        
        return environments
    
    def get_environment_template(self, template_name: str) -> Optional[EnvironmentConfig]:
        """Get environment template"""
        return self.environment_templates.get(template_name)
    
    def create_environment_from_template(self, template_name: str, name: str, **overrides) -> Environment:
        """Create environment from template"""
        template = self.get_environment_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Create config from template with overrides
        config_dict = template.__dict__.copy()
        config_dict.update(overrides)
        config_dict['name'] = name
        
        config = EnvironmentConfig(**config_dict)
        
        # Create environment
        return asyncio.create_task(self.create_environment(config))
    
    def get_environment_metrics(self, env_id: str) -> Dict[str, Any]:
        """Get environment metrics"""
        if env_id not in self.environments:
            raise ValueError(f"Environment not found: {env_id}")
        
        environment = self.environments[env_id]
        
        # Get VM metrics
        vm_metrics = []
        for vm_id in environment.vms:
            vm = self.vm_manager.get_vm(vm_id)
            if vm:
                vm_metrics.append(self.vm_manager.get_vm_metrics(vm_id))
        
        return {
            'id': environment.id,
            'name': environment.config.name,
            'type': environment.config.env_type.value,
            'status': environment.status.value,
            'vm_count': len(environment.vms),
            'vm_metrics': vm_metrics,
            'created_at': environment.created_at.isoformat(),
            'updated_at': environment.updated_at.isoformat(),
            'costs': environment.costs
        }
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get environment summary"""
        total_environments = len(self.environments)
        active_environments = len([env for env in self.environments.values() if env.status == EnvironmentStatus.ACTIVE])
        stopped_environments = len([env for env in self.environments.values() if env.status == EnvironmentStatus.STOPPED])
        
        # Count by type
        type_counts = {}
        for env in self.environments.values():
            env_type = env.config.env_type.value
            type_counts[env_type] = type_counts.get(env_type, 0) + 1
        
        return {
            'total_environments': total_environments,
            'active_environments': active_environments,
            'stopped_environments': stopped_environments,
            'type_counts': type_counts,
            'templates_available': list(self.environment_templates.keys())
        }
    
    def export_environment_config(self, env_id: str) -> Dict[str, Any]:
        """Export environment configuration"""
        if env_id not in self.environments:
            raise ValueError(f"Environment not found: {env_id}")
        
        environment = self.environments[env_id]
        
        return {
            'id': environment.id,
            'config': environment.config.__dict__,
            'status': environment.status.value,
            'vms': environment.vms,
            'resources': environment.resources,
            'created_at': environment.created_at.isoformat(),
            'updated_at': environment.updated_at.isoformat()
        }
    
    def import_environment_config(self, config_data: Dict[str, Any]) -> Environment:
        """Import environment configuration"""
        # This would recreate an environment from exported config
        # Implementation would depend on specific requirements
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        self.logger.info("EnvironmentManager cleanup completed")


