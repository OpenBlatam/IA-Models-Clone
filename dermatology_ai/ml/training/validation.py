"""
Training Validation
Validate training setup before starting
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from ml.common.validators import ModelValidator, InputValidator
from ml.common.errors import TrainingError, ValidationError

logger = logging.getLogger(__name__)


class TrainingValidator:
    """Validate training setup"""
    
    @staticmethod
    def validate_training_setup(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Validate complete training setup
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (optional)
            loss_fn: Loss function (optional)
            device: Device
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Validate model
        logger.info("Validating model...")
        model_results = ModelValidator.validate_model(model, device=device)
        results['checks']['model'] = model_results
        if not model_results['valid']:
            results['valid'] = False
            results['errors'].extend(model_results['errors'])
        results['warnings'].extend(model_results['warnings'])
        
        # Validate data loaders
        logger.info("Validating data loaders...")
        try:
            # Check train loader
            batch = next(iter(train_loader))
            if not InputValidator.validate_batch(batch, required_keys=['image']):
                results['valid'] = False
                results['errors'].append("Invalid training batch")
            
            # Check val loader if provided
            if val_loader:
                val_batch = next(iter(val_loader))
                if not InputValidator.validate_batch(val_batch, required_keys=['image']):
                    results['valid'] = False
                    results['errors'].append("Invalid validation batch")
            
            results['checks']['data_loaders'] = {'valid': True}
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Data loader validation failed: {e}")
            results['checks']['data_loaders'] = {'valid': False, 'error': str(e)}
        
        # Validate optimizer
        if optimizer:
            logger.info("Validating optimizer...")
            try:
                # Check if optimizer has model parameters
                model_params = set(model.parameters())
                opt_params = set()
                for param_group in optimizer.param_groups:
                    opt_params.update(param_group['params'])
                
                if not model_params.issubset(opt_params):
                    results['warnings'].append(
                        "Not all model parameters are in optimizer"
                    )
                
                results['checks']['optimizer'] = {'valid': True}
            except Exception as e:
                results['warnings'].append(f"Optimizer validation warning: {e}")
                results['checks']['optimizer'] = {'valid': False, 'warning': str(e)}
        
        # Validate loss function
        if loss_fn:
            logger.info("Validating loss function...")
            try:
                # Test loss function with dummy data
                dummy_output = torch.randn(2, 10)
                dummy_target = torch.randn(2, 10)
                loss = loss_fn(dummy_output, dummy_target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    results['valid'] = False
                    results['errors'].append("Loss function produces NaN/Inf")
                
                results['checks']['loss'] = {'valid': True}
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"Loss function validation failed: {e}")
                results['checks']['loss'] = {'valid': False, 'error': str(e)}
        
        # Check device compatibility
        if device != "cpu" and not torch.cuda.is_available():
            results['warnings'].append(
                f"Device '{device}' requested but CUDA not available, using CPU"
            )
        
        return results
    
    @staticmethod
    def validate_before_training(
        model: nn.Module,
        train_loader: DataLoader,
        **kwargs
    ) -> bool:
        """
        Quick validation before training
        Raises exception if invalid
        
        Args:
            model: Model
            train_loader: Training loader
            **kwargs: Additional arguments
            
        Returns:
            True if valid
            
        Raises:
            TrainingError if invalid
        """
        results = TrainingValidator.validate_training_setup(
            model, train_loader, **kwargs
        )
        
        if not results['valid']:
            error_msg = "Training setup validation failed:\n" + "\n".join(results['errors'])
            raise TrainingError(error_msg)
        
        if results['warnings']:
            for warning in results['warnings']:
                logger.warning(warning)
        
        return True













