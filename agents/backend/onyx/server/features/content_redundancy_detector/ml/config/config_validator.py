"""
Configuration Validator
Validate configuration files and dictionaries
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validate configurations
    """
    
    def __init__(self):
        """Initialize validator"""
        self.errors = []
        self.warnings = []
    
    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration
        
        Args:
            config: Model configuration
            
        Returns:
            True if valid
        """
        valid = True
        
        # Required fields
        if 'variant' not in config:
            self.errors.append("Model config missing 'variant'")
            valid = False
        
        if 'num_classes' not in config:
            self.errors.append("Model config missing 'num_classes'")
            valid = False
        
        # Validate values
        if 'num_classes' in config:
            if not isinstance(config['num_classes'], int) or config['num_classes'] <= 0:
                self.errors.append("num_classes must be positive integer")
                valid = False
        
        if 'dropout' in config:
            dropout = config['dropout']
            if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
                self.errors.append("dropout must be between 0 and 1")
                valid = False
        
        return valid
    
    def validate_training_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration
        
        Args:
            config: Training configuration
            
        Returns:
            True if valid
        """
        valid = True
        
        # Required fields
        required_fields = ['learning_rate', 'batch_size', 'num_epochs']
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Training config missing '{field}'")
                valid = False
        
        # Validate values
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("learning_rate must be positive")
                valid = False
        
        if 'batch_size' in config:
            bs = config['batch_size']
            if not isinstance(bs, int) or bs <= 0:
                self.errors.append("batch_size must be positive integer")
                valid = False
        
        if 'num_epochs' in config:
            epochs = config['num_epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                self.errors.append("num_epochs must be positive integer")
                valid = False
        
        return valid
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate complete configuration
        
        Args:
            config: Complete configuration
            
        Returns:
            True if valid
        """
        self.errors.clear()
        self.warnings.clear()
        
        valid = True
        
        # Validate model config
        if 'model' in config:
            if not self.validate_model_config(config['model']):
                valid = False
        else:
            self.errors.append("Config missing 'model' section")
            valid = False
        
        # Validate training config
        if 'training' in config:
            if not self.validate_training_config(config['training']):
                valid = False
        else:
            self.errors.append("Config missing 'training' section")
            valid = False
        
        return valid
    
    def get_errors(self) -> List[str]:
        """Get validation errors"""
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings"""
        return self.warnings



