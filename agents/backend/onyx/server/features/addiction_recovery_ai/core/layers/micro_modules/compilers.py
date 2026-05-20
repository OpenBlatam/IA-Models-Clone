"""
Compilers - Ultra-Specific Model Compilation Components
Each compilation strategy in its own focused implementation
"""

import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompilerBase(ABC):
    """Base class for all compilers"""
    
    def __init__(self, name: str = "Compiler"):
        self.name = name
    
    @abstractmethod
    def compile(self, model: nn.Module) -> nn.Module:
        """Compile model"""
        pass


class TorchCompileCompiler(CompilerBase):
    """torch.compile compiler"""
    
    def __init__(self, mode: str = "reduce-overhead", fullgraph: bool = False):
        super().__init__("TorchCompileCompiler")
        self.mode = mode
        self.fullgraph = fullgraph
    
    def compile(self, model: nn.Module) -> nn.Module:
        """Compile with torch.compile"""
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available")
            return model
        
        try:
            compiled = torch.compile(model, mode=self.mode, fullgraph=self.fullgraph)
            logger.info(f"Model compiled with mode: {self.mode}")
            return compiled
        except Exception as e:
            logger.warning(f"Compilation failed: {e}")
            return model


class TorchScriptCompiler(CompilerBase):
    """TorchScript compiler"""
    
    def __init__(self, example_input: torch.Tensor):
        super().__init__("TorchScriptCompiler")
        self.example_input = example_input
    
    def compile(self, model: nn.Module) -> nn.Module:
        """Compile with TorchScript"""
        try:
            model.eval()
            traced = torch.jit.trace(model, self.example_input)
            logger.info("Model compiled with TorchScript")
            return traced
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
            return model


class TorchScriptScriptCompiler(CompilerBase):
    """TorchScript script compiler"""
    
    def compile(self, model: nn.Module) -> nn.Module:
        """Compile with TorchScript script"""
        try:
            scripted = torch.jit.script(model)
            logger.info("Model compiled with TorchScript script")
            return scripted
        except Exception as e:
            logger.warning(f"TorchScript script compilation failed: {e}")
            return model


class OptimizeForInferenceCompiler(CompilerBase):
    """Optimize model for inference"""
    
    def __init__(self):
        super().__init__("OptimizeForInferenceCompiler")
    
    def compile(self, model: nn.Module) -> nn.Module:
        """Optimize for inference"""
        model.eval()
        # Additional optimizations can be added here
        # e.g., fuse operations, remove dropout, etc.
        logger.info("Model optimized for inference")
        return model


# Factory for compilers
class CompilerFactory:
    """Factory for creating compilers"""
    
    @staticmethod
    def create(
        compiler_type: str,
        **kwargs
    ) -> CompilerBase:
        """Create compiler"""
        compiler_type = compiler_type.lower()
        
        if compiler_type == 'torch_compile':
            return TorchCompileCompiler(**kwargs)
        elif compiler_type == 'torchscript':
            if 'example_input' not in kwargs:
                raise ValueError("example_input required for TorchScript")
            return TorchScriptCompiler(**kwargs)
        elif compiler_type == 'torchscript_script':
            return TorchScriptScriptCompiler(**kwargs)
        elif compiler_type == 'optimize':
            return OptimizeForInferenceCompiler(**kwargs)
        else:
            raise ValueError(f"Unknown compiler type: {compiler_type}")


__all__ = [
    "CompilerBase",
    "TorchCompileCompiler",
    "TorchScriptCompiler",
    "TorchScriptScriptCompiler",
    "OptimizeForInferenceCompiler",
    "CompilerFactory",
]



