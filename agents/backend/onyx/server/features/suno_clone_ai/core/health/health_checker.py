"""
Health Checker

Utilities for checking system and model health.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import psutil
import os

logger = logging.getLogger(__name__)


class HealthChecker:
    """Check system and model health."""
    
    def __init__(self):
        """Initialize health checker."""
        pass
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check system health.
        
        Returns:
            System health status
        """
        health = {
            'status': 'healthy',
            'checks': {}
        }
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        health['checks']['cpu'] = {
            'usage_percent': cpu_percent,
            'status': 'healthy' if cpu_percent < 90 else 'warning'
        }
        
        # Memory check
        memory = psutil.virtual_memory()
        health['checks']['memory'] = {
            'usage_percent': memory.percent,
            'available_gb': memory.available / (1024 ** 3),
            'status': 'healthy' if memory.percent < 90 else 'warning'
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        health['checks']['disk'] = {
            'usage_percent': disk.percent,
            'free_gb': disk.free / (1024 ** 3),
            'status': 'healthy' if disk.percent < 90 else 'warning'
        }
        
        # GPU check
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            health['checks']['gpu'] = {
                'available': True,
                'memory_usage_percent': gpu_memory * 100,
                'status': 'healthy' if gpu_memory < 0.9 else 'warning'
            }
        else:
            health['checks']['gpu'] = {
                'available': False,
                'status': 'not_available'
            }
        
        # Overall status
        if any(check['status'] == 'warning' for check in health['checks'].values()):
            health['status'] = 'warning'
        
        return health
    
    def check_model_health(
        self,
        model: nn.Module,
        input_shape: tuple
    ) -> Dict[str, Any]:
        """
        Check model health.
        
        Args:
            model: Model to check
            input_shape: Input tensor shape
            
        Returns:
            Model health status
        """
        health = {
            'status': 'healthy',
            'checks': {}
        }
        
        try:
            # Check forward pass
            model.eval()
            dummy_input = torch.randn(input_shape)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            health['checks']['forward_pass'] = {
                'status': 'healthy',
                'output_shape': list(output.shape) if isinstance(output, torch.Tensor) else None
            }
            
            # Check for NaN/Inf
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                
                health['checks']['nan_inf'] = {
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'status': 'healthy' if not (has_nan or has_inf) else 'unhealthy'
                }
            
            # Check parameters
            from core.debugging import count_parameters
            param_counts = count_parameters(model)
            
            health['checks']['parameters'] = {
                'total': param_counts['total'],
                'trainable': param_counts['trainable'],
                'status': 'healthy'
            }
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['checks']['error'] = {
                'message': str(e),
                'status': 'unhealthy'
            }
        
        # Overall status
        if any(check.get('status') == 'unhealthy' for check in health['checks'].values()):
            health['status'] = 'unhealthy'
        
        return health
    
    def check_resource_health(self) -> Dict[str, Any]:
        """
        Check resource health.
        
        Returns:
            Resource health status
        """
        health = {
            'status': 'healthy',
            'resources': {}
        }
        
        # Process resources
        process = psutil.Process(os.getpid())
        health['resources']['process'] = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / (1024 ** 2),
            'num_threads': process.num_threads()
        }
        
        # GPU resources
        if torch.cuda.is_available():
            health['resources']['gpu'] = {
                'memory_allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                'memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
                'device_count': torch.cuda.device_count()
            }
        
        return health


def check_system_health() -> Dict[str, Any]:
    """Check system health."""
    checker = HealthChecker()
    return checker.check_system_health()


def check_model_health(
    model: nn.Module,
    input_shape: tuple
) -> Dict[str, Any]:
    """Check model health."""
    checker = HealthChecker()
    return checker.check_model_health(model, input_shape)


def check_resource_health() -> Dict[str, Any]:
    """Check resource health."""
    checker = HealthChecker()
    return checker.check_resource_health()



