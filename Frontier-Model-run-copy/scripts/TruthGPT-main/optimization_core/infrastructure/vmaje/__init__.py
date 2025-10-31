"""
VMAJE - Virtual Machine, Application, Job, and Environment Management
====================================================================

Advanced VM orchestration system for TruthGPT optimization framework with:
- Dynamic VM provisioning and scaling
- Intelligent job scheduling and queuing
- Environment management and isolation
- Resource optimization and cost management
- Multi-cloud support (Azure, AWS, GCP)
- GPU acceleration and specialized workloads
"""

from .core.vm_manager import VMManager
from .core.job_scheduler import JobScheduler
from .core.environment_manager import EnvironmentManager
from .core.resource_optimizer import ResourceOptimizer
from .core.cost_manager import CostManager
from .core.monitoring import VMAJEMonitor

__version__ = "1.0.0"
__author__ = "TruthGPT Team"
__license__ = "MIT"

# Main VMAJE components
vm_manager = VMManager()
job_scheduler = JobScheduler()
environment_manager = EnvironmentManager()
resource_optimizer = ResourceOptimizer()
cost_manager = CostManager()
monitor = VMAJEMonitor()

__all__ = [
    'VMManager',
    'JobScheduler', 
    'EnvironmentManager',
    'ResourceOptimizer',
    'CostManager',
    'VMAJEMonitor',
    'vm_manager',
    'job_scheduler',
    'environment_manager',
    'resource_optimizer',
    'cost_manager',
    'monitor'
]


