"""
Model Compilation Module

Model compilation functionality.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelCompilationMixin:
    """Model compilation mixin."""
    
    def compile_model(self, model: torch.nn.Module, mode: str = "reduce-overhead") -> torch.nn.Module:
        """
        Compile model for faster execution
        
        Args:
            model: Model to compile
            mode: Compilation mode
        
        Returns:
            Compiled model
        """
        if hasattr(torch, 'compile'):
            try:
                compiled = torch.compile(model, mode=mode)
                logger.info(f"Model compiled with mode={mode}")
                return compiled
            except Exception as e:
                logger.warning(f"Model compilation failed: {str(e)}")
                return model
        else:
            logger.warning("torch.compile not available")
            return model



