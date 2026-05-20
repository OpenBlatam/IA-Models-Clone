"""
Compatibility Utilities
Version compatibility and migration utilities
"""

import torch
import sys
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CompatibilityChecker:
    """
    Check compatibility between versions
    """
    
    @staticmethod
    def check_pytorch_version(min_version: str = "1.9.0") -> Tuple[bool, str]:
        """
        Check PyTorch version
        
        Args:
            min_version: Minimum required version
            
        Returns:
            Tuple of (is_compatible, current_version)
        """
        current_version = torch.__version__
        
        def version_tuple(v):
            return tuple(map(int, v.split('.')[:3]))
        
        is_compatible = version_tuple(current_version) >= version_tuple(min_version)
        
        return is_compatible, current_version
    
    @staticmethod
    def check_python_version(min_version: str = "3.7") -> Tuple[bool, str]:
        """
        Check Python version
        
        Args:
            min_version: Minimum required version
            
        Returns:
            Tuple of (is_compatible, current_version)
        """
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        def version_tuple(v):
            return tuple(map(int, v.split('.')))
        
        is_compatible = version_tuple(current_version) >= version_tuple(min_version)
        
        return is_compatible, current_version
    
    @staticmethod
    def check_cuda_compatibility() -> Dict[str, Any]:
        """
        Check CUDA compatibility
        
        Returns:
            Dictionary with CUDA info
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'cudnn_version': None,
            'device_count': 0,
        }
        
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['device_count'] = torch.cuda.device_count()
        
        return info
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information
        
        Returns:
            Dictionary with system info
        """
        pytorch_compat, pytorch_version = CompatibilityChecker.check_pytorch_version()
        python_compat, python_version = CompatibilityChecker.check_python_version()
        cuda_info = CompatibilityChecker.check_cuda_compatibility()
        
        return {
            'python_version': python_version,
            'python_compatible': python_compat,
            'pytorch_version': pytorch_version,
            'pytorch_compatible': pytorch_compat,
            'cuda': cuda_info,
        }
    
    @staticmethod
    def print_compatibility_report() -> None:
        """Print compatibility report"""
        info = CompatibilityChecker.get_system_info()
        
        print("\n" + "="*60)
        print("Compatibility Report")
        print("="*60)
        print(f"Python: {info['python_version']} {'✓' if info['python_compatible'] else '✗'}")
        print(f"PyTorch: {info['pytorch_version']} {'✓' if info['pytorch_compatible'] else '✗'}")
        
        if info['cuda']['cuda_available']:
            print(f"CUDA: {info['cuda']['cuda_version']} ✓")
            print(f"cuDNN: {info['cuda']['cudnn_version']} ✓")
            print(f"GPU Count: {info['cuda']['device_count']}")
        else:
            print("CUDA: Not available")
        
        print("="*60 + "\n")



