"""
Configuration Validator for Imagen Video Enhancer AI
===================================================

Advanced configuration validation.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigValidator:
    """
    Validates configuration with advanced checks.
    
    Features:
    - Type validation
    - Range validation
    - File path validation
    - Dependency validation
    - Environment validation
    """
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate OpenRouter config
        if "openrouter" in config:
            errors.extend(ConfigValidator._validate_openrouter(config["openrouter"]))
        
        # Validate TruthGPT config
        if "truthgpt" in config:
            errors.extend(ConfigValidator._validate_truthgpt(config["truthgpt"]))
        
        # Validate file size limits
        if "max_file_size_mb" in config:
            if not isinstance(config["max_file_size_mb"], (int, float)) or config["max_file_size_mb"] <= 0:
                errors.append("max_file_size_mb must be a positive number")
        
        # Validate output directory
        if "output_dir" in config:
            output_dir = Path(config["output_dir"])
            if not output_dir.parent.exists():
                errors.append(f"Output directory parent does not exist: {output_dir.parent}")
        
        return errors
    
    @staticmethod
    def _validate_openrouter(openrouter_config: Dict[str, Any]) -> List[str]:
        """Validate OpenRouter configuration."""
        errors = []
        
        if "api_key" not in openrouter_config:
            errors.append("OpenRouter API key is required")
        elif not openrouter_config["api_key"] or len(openrouter_config["api_key"]) < 10:
            errors.append("OpenRouter API key appears invalid")
        
        if "base_url" in openrouter_config:
            url = openrouter_config["base_url"]
            if not url.startswith(("http://", "https://")):
                errors.append("OpenRouter base_url must be a valid HTTP/HTTPS URL")
        
        if "timeout" in openrouter_config:
            timeout = openrouter_config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("OpenRouter timeout must be a positive number")
        
        return errors
    
    @staticmethod
    def _validate_truthgpt(truthgpt_config: Dict[str, Any]) -> List[str]:
        """Validate TruthGPT configuration."""
        errors = []
        
        if truthgpt_config.get("enabled", False):
            if "api_key" not in truthgpt_config:
                errors.append("TruthGPT API key is required when enabled")
            elif not truthgpt_config["api_key"] or len(truthgpt_config["api_key"]) < 10:
                errors.append("TruthGPT API key appears invalid")
        
        return errors
    
    @staticmethod
    def validate_file_paths(config: Dict[str, Any]) -> List[str]:
        """
        Validate file paths in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check output directory
        if "output_dir" in config:
            output_dir = Path(config["output_dir"])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                # Check write permissions
                test_file = output_dir / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"Cannot write to output directory: {e}")
        
        return errors
    
    @staticmethod
    def validate_environment() -> List[str]:
        """
        Validate environment requirements.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            errors.append("Python 3.8+ is required")
        
        # Check required packages
        required_packages = [
            "httpx",
            "fastapi",
            "pydantic",
            "PIL"
        ]
        
        for package in required_packages:
            try:
                __import__(package.lower().replace("-", "_"))
            except ImportError:
                errors.append(f"Required package not installed: {package}")
        
        return errors
    
    @staticmethod
    def get_config_recommendations(config: Dict[str, Any]) -> List[str]:
        """
        Get configuration recommendations.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check cache settings
        if "cache" in config:
            cache_config = config["cache"]
            if cache_config.get("default_ttl_hours", 24) < 1:
                recommendations.append("Cache TTL is very short, consider increasing for better performance")
        
        # Check parallel tasks
        if "max_parallel_tasks" in config:
            max_tasks = config["max_parallel_tasks"]
            if max_tasks > 20:
                recommendations.append("High number of parallel tasks may cause resource issues")
            elif max_tasks < 2:
                recommendations.append("Consider increasing parallel tasks for better throughput")
        
        # Check rate limits
        if "rate_limit" in config:
            rate_limit = config["rate_limit"]
            if rate_limit.get("requests_per_second", 10) > 50:
                recommendations.append("Very high rate limit may cause API throttling")
        
        return recommendations




