"""
Configuration Validator
======================

Validate and optimize configuration settings.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validate and optimize configuration.
    
    Features:
    - Configuration validation
    - Auto-correction
    - Optimization suggestions
    - Environment detection
    """
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and return issues.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation result with issues and suggestions
        """
        issues = []
        suggestions = []
        warnings = []
        
        # Validate scale factor
        scale_factor = config.get("default_scale_factor", 2.0)
        if scale_factor < 1.0 or scale_factor > 8.0:
            issues.append(f"Invalid scale factor: {scale_factor} (should be 1.0-8.0)")
            suggestions.append("Set scale_factor to 2.0 or 4.0")
        
        # Validate quality mode
        quality_mode = config.get("quality_mode", "high")
        valid_modes = ["fast", "balanced", "high", "ultra"]
        if quality_mode not in valid_modes:
            issues.append(f"Invalid quality mode: {quality_mode}")
            suggestions.append(f"Use one of: {', '.join(valid_modes)}")
        
        # Check Real-ESRGAN availability
        if config.get("use_realesrgan", False):
            try:
                from ..models.realesrgan_integration import REALESRGAN_AVAILABLE
                if not REALESRGAN_AVAILABLE:
                    warnings.append("Real-ESRGAN requested but not available")
                    suggestions.append("Install with: pip install realesrgan basicsr")
            except ImportError:
                warnings.append("Real-ESRGAN module not found")
        
        # Check OpenRouter
        if config.get("use_ai_enhancement", False):
            api_key = config.get("openrouter", {}).get("api_key")
            if not api_key:
                warnings.append("AI enhancement enabled but no API key provided")
                suggestions.append("Set OPENROUTER_API_KEY environment variable")
        
        # Check paths
        output_dir = config.get("output_dir", "./output")
        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                suggestions.append(f"Created output directory: {output_dir}")
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")
        
        # Check optimization core
        opt_core_path = config.get("optimization_core_path")
        if opt_core_path and not Path(opt_core_path).exists():
            warnings.append(f"Optimization core path not found: {opt_core_path}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    @staticmethod
    def optimize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration based on system resources.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Optimized configuration
        """
        import psutil
        
        optimized = config.copy()
        
        # Detect system resources
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Optimize workers
        if "max_workers" not in optimized or optimized["max_workers"] is None:
            optimized["max_workers"] = min(8, max(2, cpu_count // 2))
        
        # Optimize batch size
        if memory_gb < 4:
            optimized["batch_size"] = 1
        elif memory_gb < 8:
            optimized["batch_size"] = 2
        else:
            optimized["batch_size"] = min(4, optimized.get("batch_size", 2))
        
        # Optimize cache
        if "enable_cache" not in optimized:
            optimized["enable_cache"] = True
        
        if memory_gb < 4:
            optimized["cache_size_mb"] = 500
        elif memory_gb < 8:
            optimized["cache_size_mb"] = 1000
        else:
            optimized["cache_size_mb"] = 2000
        
        return optimized
    
    @staticmethod
    def get_recommended_config(use_case: str = "production") -> Dict[str, Any]:
        """
        Get recommended configuration for use case.
        
        Args:
            use_case: Use case ('production', 'development', 'testing')
            
        Returns:
            Recommended configuration
        """
        configs = {
            "production": {
                "quality_mode": "high",
                "use_realesrgan": True,
                "use_ai_enhancement": True,
                "enable_cache": True,
                "max_workers": 4,
                "batch_size": 2,
            },
            "development": {
                "quality_mode": "balanced",
                "use_realesrgan": False,
                "use_ai_enhancement": False,
                "enable_cache": True,
                "max_workers": 2,
                "batch_size": 1,
            },
            "testing": {
                "quality_mode": "fast",
                "use_realesrgan": False,
                "use_ai_enhancement": False,
                "enable_cache": False,
                "max_workers": 1,
                "batch_size": 1,
            }
        }
        
        return configs.get(use_case, configs["production"])


