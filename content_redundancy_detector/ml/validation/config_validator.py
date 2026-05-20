"""
Config Validator
Advanced configuration validation
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Advanced configuration validation
    """
    
    @staticmethod
    def validate_complete_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check required sections
        required_sections = ['model', 'training']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate model config
        if 'model' in config:
            model_config = config['model']
            if 'variant' not in model_config:
                errors.append("Model config missing 'variant'")
            if 'num_classes' not in model_config:
                errors.append("Model config missing 'num_classes'")
            elif not isinstance(model_config['num_classes'], int) or model_config['num_classes'] <= 0:
                errors.append("num_classes must be positive integer")
        
        # Validate training config
        if 'training' in config:
            training_config = config['training']
            if 'learning_rate' in training_config:
                lr = training_config['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0:
                    errors.append("learning_rate must be positive")
                elif lr > 1:
                    warnings.append("learning_rate seems very high (>1)")
            
            if 'batch_size' in training_config:
                bs = training_config['batch_size']
                if not isinstance(bs, int) or bs <= 0:
                    errors.append("batch_size must be positive integer")
                elif bs > 1024:
                    warnings.append("batch_size seems very large (>1024)")
        
        # Validate device config
        if 'device' in config:
            device_config = config['device']
            if 'use_gpu' in device_config and not isinstance(device_config['use_gpu'], bool):
                errors.append("use_gpu must be boolean")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }



