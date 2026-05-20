"""
Dependency Checker
Check and verify dependencies
"""

import sys
import importlib
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DependencyChecker:
    """
    Check and verify dependencies
    """
    
    REQUIRED_PACKAGES = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
    }
    
    OPTIONAL_PACKAGES = {
        'transformers': 'Transformers',
        'diffusers': 'Diffusers',
        'gradio': 'Gradio',
        'wandb': 'Weights & Biases',
        'tensorboard': 'TensorBoard',
        'optuna': 'Optuna',
        'seaborn': 'Seaborn',
        'matplotlib': 'Matplotlib',
    }
    
    @staticmethod
    def check_package(package_name: str) -> Tuple[bool, str]:
        """
        Check if package is installed
        
        Args:
            package_name: Package name
            
        Returns:
            Tuple of (is_installed, version)
        """
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, None
    
    @staticmethod
    def check_all() -> Dict[str, Dict[str, any]]:
        """
        Check all dependencies
        
        Returns:
            Dictionary with dependency status
        """
        results = {
            'required': {},
            'optional': {},
        }
        
        # Check required packages
        for package, display_name in DependencyChecker.REQUIRED_PACKAGES.items():
            is_installed, version = DependencyChecker.check_package(package)
            results['required'][package] = {
                'installed': is_installed,
                'version': version,
                'name': display_name,
            }
            if not is_installed:
                logger.warning(f"Required package {display_name} ({package}) is not installed")
        
        # Check optional packages
        for package, display_name in DependencyChecker.OPTIONAL_PACKAGES.items():
            is_installed, version = DependencyChecker.check_package(package)
            results['optional'][package] = {
                'installed': is_installed,
                'version': version,
                'name': display_name,
            }
        
        return results
    
    @staticmethod
    def print_report() -> None:
        """Print dependency report"""
        results = DependencyChecker.check_all()
        
        print("\n" + "="*60)
        print("Dependency Check Report")
        print("="*60)
        
        print("\nRequired Packages:")
        for package, info in results['required'].items():
            status = "✓" if info['installed'] else "✗"
            version = f" (v{info['version']})" if info['version'] else ""
            print(f"  {status} {info['name']}: {package}{version}")
        
        print("\nOptional Packages:")
        for package, info in results['optional'].items():
            status = "✓" if info['installed'] else "○"
            version = f" (v{info['version']})" if info['version'] else ""
            print(f"  {status} {info['name']}: {package}{version}")
        
        print("="*60 + "\n")
    
    @staticmethod
    def verify_requirements() -> bool:
        """
        Verify all required packages are installed
        
        Returns:
            True if all required packages are installed
        """
        results = DependencyChecker.check_all()
        all_installed = all(
            info['installed'] for info in results['required'].values()
        )
        
        if not all_installed:
            missing = [
                info['name'] for package, info in results['required'].items()
                if not info['installed']
            ]
            logger.error(f"Missing required packages: {', '.join(missing)}")
        
        return all_installed



