"""
Configuration manager for Ultimate Enhanced Supreme Production system
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.logger = logger
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        return str(Path(__file__).parent.parent.parent / 'ultimate_enhanced_supreme_production_config.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"📁 Configuration loaded from {self.config_path}")
                return config
            else:
                self.logger.warning(f"⚠️ Configuration file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'supreme_optimization_level': 'supreme_omnipotent',
            'ultra_fast_level': 'infinity',
            'refactored_ultimate_hybrid_level': 'ultimate_hybrid',
            'cuda_kernel_level': 'ultimate',
            'gpu_utilization_level': 'ultimate',
            'memory_optimization_level': 'ultimate',
            'reward_function_level': 'ultimate',
            'truthgpt_adapter_level': 'ultimate',
            'microservices_level': 'ultimate',
            'max_concurrent_generations': 10000,
            'max_documents_per_query': 1000000,
            'max_continuous_documents': 10000000,
            'generation_timeout': 300.0,
            'optimization_timeout': 60.0,
            'monitoring_interval': 1.0,
            'health_check_interval': 5.0,
            'target_speedup': 1000000000000000000000.0,
            'target_memory_reduction': 0.999999999,
            'target_accuracy_preservation': 0.999999999,
            'target_energy_efficiency': 0.999999999
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration."""
        try:
            # Update configuration
            self.config.update(config_data)
            
            # Save to file
            self._save_config()
            
            self.logger.info("✅ Configuration updated successfully")
            return self.config.copy()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating configuration: {e}")
            return {}
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"💾 Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"❌ Error saving configuration: {e}")
    
    def get_supreme_optimization_level(self) -> str:
        """Get Supreme optimization level."""
        return self.config.get('supreme_optimization_level', 'supreme_omnipotent')
    
    def get_ultra_fast_level(self) -> str:
        """Get Ultra-Fast level."""
        return self.config.get('ultra_fast_level', 'infinity')
    
    def get_refactored_ultimate_hybrid_level(self) -> str:
        """Get Refactored Ultimate Hybrid level."""
        return self.config.get('refactored_ultimate_hybrid_level', 'ultimate_hybrid')
    
    def get_cuda_kernel_level(self) -> str:
        """Get CUDA Kernel level."""
        return self.config.get('cuda_kernel_level', 'ultimate')
    
    def get_gpu_utilization_level(self) -> str:
        """Get GPU Utilization level."""
        return self.config.get('gpu_utilization_level', 'ultimate')
    
    def get_memory_optimization_level(self) -> str:
        """Get Memory Optimization level."""
        return self.config.get('memory_optimization_level', 'ultimate')
    
    def get_reward_function_level(self) -> str:
        """Get Reward Function level."""
        return self.config.get('reward_function_level', 'ultimate')
    
    def get_truthgpt_adapter_level(self) -> str:
        """Get TruthGPT Adapter level."""
        return self.config.get('truthgpt_adapter_level', 'ultimate')
    
    def get_microservices_level(self) -> str:
        """Get Microservices level."""
        return self.config.get('microservices_level', 'ultimate')
    
    def get_max_concurrent_generations(self) -> int:
        """Get maximum concurrent generations."""
        return self.config.get('max_concurrent_generations', 10000)
    
    def get_max_documents_per_query(self) -> int:
        """Get maximum documents per query."""
        return self.config.get('max_documents_per_query', 1000000)
    
    def get_max_continuous_documents(self) -> int:
        """Get maximum continuous documents."""
        return self.config.get('max_continuous_documents', 10000000)
    
    def get_generation_timeout(self) -> float:
        """Get generation timeout."""
        return self.config.get('generation_timeout', 300.0)
    
    def get_optimization_timeout(self) -> float:
        """Get optimization timeout."""
        return self.config.get('optimization_timeout', 60.0)
    
    def get_monitoring_interval(self) -> float:
        """Get monitoring interval."""
        return self.config.get('monitoring_interval', 1.0)
    
    def get_health_check_interval(self) -> float:
        """Get health check interval."""
        return self.config.get('health_check_interval', 5.0)
    
    def get_target_speedup(self) -> float:
        """Get target speedup."""
        return self.config.get('target_speedup', 1000000000000000000000.0)
    
    def get_target_memory_reduction(self) -> float:
        """Get target memory reduction."""
        return self.config.get('target_memory_reduction', 0.999999999)
    
    def get_target_accuracy_preservation(self) -> float:
        """Get target accuracy preservation."""
        return self.config.get('target_accuracy_preservation', 0.999999999)
    
    def get_target_energy_efficiency(self) -> float:
        """Get target energy efficiency."""
        return self.config.get('target_energy_efficiency', 0.999999999)