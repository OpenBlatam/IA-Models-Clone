"""
Parameter Mapper
================

Maps TensorFlow-like optimizer parameters to PyTorch optimizer parameters.
"""

from typing import Dict, Any


class OptimizerParameterMapper:
    """
    Maps TensorFlow-like parameters to PyTorch optimizer parameters.
    
    Responsibilities:
    - Converting parameter names from TensorFlow to PyTorch conventions
    - Handling special cases (e.g., beta_1/beta_2 -> betas tuple)
    - Normalizing parameter values
    """
    
    # Mapping from TensorFlow parameter names to PyTorch parameter names
    _PARAMETER_MAPPING = {
        'beta_1': 'betas',
        'beta1': 'betas',
        'beta_2': 'betas',
        'beta2': 'betas',
        'epsilon': 'eps',
        'eps': 'eps',
        'rho': 'alpha',
        'alpha': 'alpha',
        'initial_accumulator_value': 'initial_accumulator_value',
        'centered': 'centered',
        'amsgrad': 'amsgrad',
        'weight_decay': 'weight_decay',
        'momentum': 'momentum',
        'nesterov': 'nesterov',
    }
    
    @classmethod
    def map_to_pytorch(
        cls,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Map TensorFlow-like parameters to PyTorch optimizer parameters.
        
        Args:
            optimizer_type: Type of optimizer (adam, sgd, etc.)
            learning_rate: Learning rate
            **kwargs: TensorFlow-like parameters
        
        Returns:
            Dictionary with PyTorch-compatible parameters
        """
        pytorch_kwargs = {}
        optimizer_type = optimizer_type.lower()
        
        # Map individual parameters
        for key, value in kwargs.items():
            mapped_key = cls._PARAMETER_MAPPING.get(key, key)
            
            if mapped_key == 'betas':
                # Handle beta_1 and beta_2 specially
                if key in ['beta_1', 'beta1']:
                    pytorch_kwargs.setdefault('betas', (value, 0.999))
                    # Update second beta if not already set
                    if 'betas' in pytorch_kwargs and len(pytorch_kwargs['betas']) == 2:
                        pytorch_kwargs['betas'] = (value, pytorch_kwargs['betas'][1])
                elif key in ['beta_2', 'beta2']:
                    pytorch_kwargs.setdefault('betas', (0.9, value))
                    # Update first beta if not already set
                    if 'betas' in pytorch_kwargs and len(pytorch_kwargs['betas']) == 2:
                        pytorch_kwargs['betas'] = (pytorch_kwargs['betas'][0], value)
            else:
                pytorch_kwargs[mapped_key] = value
        
        # Ensure betas tuple for Adam/AdamW
        if optimizer_type in ['adam', 'adamw'] and 'betas' not in pytorch_kwargs:
            beta_1 = kwargs.get('beta_1', kwargs.get('beta1', 0.9))
            beta_2 = kwargs.get('beta_2', kwargs.get('beta2', 0.999))
            pytorch_kwargs['betas'] = (beta_1, beta_2)
        
        return pytorch_kwargs

