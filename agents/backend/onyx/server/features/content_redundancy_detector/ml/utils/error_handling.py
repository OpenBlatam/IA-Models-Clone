"""
Error Handling Utilities
Comprehensive error handling and recovery
"""

import torch
import logging
from typing import Optional, Callable, Any, Dict
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Comprehensive error handling utilities
    """
    
    @staticmethod
    def handle_nan_inf(
        tensor: torch.Tensor,
        replace_nan: float = 0.0,
        replace_inf: float = 1e6,
    ) -> torch.Tensor:
        """
        Handle NaN and Inf values in tensor
        
        Args:
            tensor: Input tensor
            replace_nan: Value to replace NaN
            replace_inf: Value to replace Inf
            
        Returns:
            Cleaned tensor
        """
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, replace_nan), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.full_like(tensor, replace_inf), tensor)
        return tensor
    
    @staticmethod
    def safe_backward(
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Safe backward pass with error handling
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Dictionary with status and info
        """
        result = {
            'success': False,
            'error': None,
            'grad_norm': None,
        }
        
        try:
            # Check loss
            if torch.isnan(loss):
                result['error'] = 'Loss is NaN'
                return result
            
            if torch.isinf(loss):
                result['error'] = 'Loss is Inf'
                return result
            
            # Backward
            loss.backward()
            
            # Clip gradients if specified
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in optimizer.param_groups[0]['params'] if p.grad is not None],
                    max_grad_norm
                )
                result['grad_norm'] = float(grad_norm)
            
            result['success'] = True
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Safe backward failed: {e}", exc_info=True)
        
        return result
    
    @staticmethod
    def retry_on_error(
        max_retries: int = 3,
        delay: float = 1.0,
        exceptions: tuple = (Exception,),
    ):
        """
        Decorator for retrying on error
        
        Args:
            max_retries: Maximum number of retries
            delay: Delay between retries
            exceptions: Exceptions to catch
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed: {e}. Retrying..."
                            )
                            import time
                            time.sleep(delay)
                        else:
                            logger.error(
                                f"All {max_retries} attempts failed. Last error: {e}",
                                exc_info=True
                            )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    def safe_model_forward(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        fallback_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Safe model forward with error handling
        
        Args:
            model: Model
            inputs: Input tensor
            fallback_value: Fallback value on error
            
        Returns:
            Model output or fallback
        """
        try:
            return model(inputs)
        except Exception as e:
            logger.error(f"Model forward failed: {e}", exc_info=True)
            if fallback_value is not None:
                return fallback_value
            raise


class TrainingErrorHandler:
    """
    Error handling for training loops
    """
    
    def __init__(
        self,
        max_nan_losses: int = 5,
        max_inf_losses: int = 5,
        gradient_clip: Optional[float] = 1.0,
    ):
        """
        Initialize training error handler
        
        Args:
            max_nan_losses: Maximum consecutive NaN losses before stopping
            max_inf_losses: Maximum consecutive Inf losses before stopping
            gradient_clip: Gradient clipping value
        """
        self.max_nan_losses = max_nan_losses
        self.max_inf_losses = max_inf_losses
        self.gradient_clip = gradient_clip
        self.nan_loss_count = 0
        self.inf_loss_count = 0
    
    def check_loss(self, loss: torch.Tensor) -> Dict[str, Any]:
        """
        Check loss for issues
        
        Args:
            loss: Loss tensor
            
        Returns:
            Check results
        """
        result = {
            'valid': True,
            'should_stop': False,
            'warnings': [],
        }
        
        if torch.isnan(loss):
            self.nan_loss_count += 1
            result['warnings'].append(f"NaN loss (count: {self.nan_loss_count})")
            result['valid'] = False
            
            if self.nan_loss_count >= self.max_nan_losses:
                result['should_stop'] = True
                result['warnings'].append("Too many NaN losses, stopping training")
        else:
            self.nan_loss_count = 0
        
        if torch.isinf(loss):
            self.inf_loss_count += 1
            result['warnings'].append(f"Inf loss (count: {self.inf_loss_count})")
            result['valid'] = False
            
            if self.inf_loss_count >= self.max_inf_losses:
                result['should_stop'] = True
                result['warnings'].append("Too many Inf losses, stopping training")
        else:
            self.inf_loss_count = 0
        
        return result
    
    def reset(self) -> None:
        """Reset error counters"""
        self.nan_loss_count = 0
        self.inf_loss_count = 0



