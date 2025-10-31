"""
Model management module for loading, saving, and configuring models.
"""
import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.interfaces import BaseModelManager

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class ModelManager(BaseModelManager):
    """
    Manages model loading, saving, and configuration.
    Implements BaseModelManager interface.
    """
    
    def load_model(
        self,
        model_name: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        gradient_checkpointing: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = True,
    ) -> torch.nn.Module:
        """
        Load a model from name or path.
        
        Args:
            model_name: Model name or path
            torch_dtype: Optional dtype for model weights
            device_map: Optional device mapping strategy
            gradient_checkpointing: Enable gradient checkpointing
            lora_config: Optional LoRA configuration
            trust_remote_code: Trust remote code from HuggingFace
        
        Returns:
            Loaded model
        """
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
            
            # Enable gradient checkpointing if requested
            if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.debug("Gradient checkpointing enabled")
            
            # Disable cache during training
            if hasattr(model, "config"):
                try:
                    model.config.use_cache = False
                except Exception:
                    pass
            
            # Apply LoRA if configured
            if lora_config and lora_config.get("enabled", False):
                if not _PEFT_AVAILABLE:
                    raise RuntimeError("PEFT not available but LoRA was requested")
                
                lconf = LoraConfig(
                    r=lora_config.get("r", 16),
                    lora_alpha=lora_config.get("alpha", 32),
                    lora_dropout=lora_config.get("dropout", 0.05),
                    target_modules=[
                        "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj"
                    ],
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lconf)
                logger.info("LoRA applied to model")
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            raise
    
    def save_model(
        self,
        model: torch.nn.Module,
        path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        safe_serialization: bool = True,
        **kwargs
    ) -> None:
        """
        Save model to path.
        
        Args:
            model: Model to save
            path: Save path
            tokenizer: Optional tokenizer to save
            safe_serialization: Use SafeTensors format
            **kwargs: Additional save arguments
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Handle DataParallel
            model_to_save = model
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            elif hasattr(model, "module"):
                model_to_save = model.module
            
            # Save model
            model_to_save.save_pretrained(
                path,
                safe_serialization=safe_serialization,
                **kwargs
            )
            
            # Save tokenizer if provided
            if tokenizer:
                tokenizer.save_pretrained(path)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}", exc_info=True)
            raise
    
    def get_model_device(self, model: torch.nn.Module) -> torch.device:
        """
        Get device where model is located.
        
        Args:
            model: Model instance
        
        Returns:
            Device
        """
        # Get base model if wrapped
        base_model = model
        if isinstance(model, nn.DataParallel):
            base_model = model.module
        
        # Get device from first parameter
        if hasattr(base_model, "parameters"):
            param = next(base_model.parameters(), None)
            if param is not None:
                return param.device
        
        # Fallback
        return torch.device("cpu")
    
    def enable_multi_gpu(
        self,
        model: torch.nn.Module,
        device_ids: Optional[list] = None
    ) -> torch.nn.Module:
        """
        Enable multi-GPU training with DataParallel.
        
        Args:
            model: Model to parallelize
            device_ids: Optional list of device IDs
        
        Returns:
            Parallelized model
        """
        if torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            return nn.DataParallel(model, device_ids=device_ids)
        else:
            logger.warning("Only one GPU available, DataParallel not applied")
            return model
    
    def enable_torch_compile(
        self,
        model: torch.nn.Module,
        mode: str = "default"
    ) -> torch.nn.Module:
        """
        Enable torch.compile optimization.
        
        Args:
            model: Model to compile
            mode: Compilation mode (default|reduce-overhead|max-autotune)
        
        Returns:
            Compiled model
        """
        if hasattr(torch, "compile"):
            try:
                logger.info(f"Compiling model with mode: {mode}")
                return torch.compile(model, mode=mode)
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
                return model
        else:
            logger.warning("torch.compile not available")
            return model
    
    def configure_device_settings(
        self,
        allow_tf32: bool = True,
        matmul_precision: str = "high"
    ) -> None:
        """
        Configure CUDA device settings for optimal performance.
        
        Args:
            allow_tf32: Enable TF32 for Ampere+ GPUs
            matmul_precision: Matrix multiplication precision
        """
        if not torch.cuda.is_available():
            return
        
        try:
            if allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision(matmul_precision)
                logger.debug("TF32 enabled")
            
            # Configure SDPA kernels
            if hasattr(torch.backends.cuda, "sdp_kernel"):
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True
                )
                logger.debug("SDPA kernels configured")
        except Exception as e:
            logger.warning(f"Failed to configure device settings: {e}")


class ModelBuilder:
    """
    Builder pattern for constructing models with various configurations.
    """
    
    def __init__(self):
        self._manager = ModelManager()
        self._model_name: Optional[str] = None
        self._torch_dtype: Optional[torch.dtype] = None
        self._device_map: Optional[str] = None
        self._gradient_checkpointing: bool = True
        self._lora_config: Optional[Dict[str, Any]] = None
        self._multi_gpu: bool = False
        self._torch_compile: bool = False
        self._compile_mode: str = "default"
        self._device_settings: Dict[str, Any] = {}
    
    def with_model_name(self, name: str) -> "ModelBuilder":
        """Set model name or path."""
        self._model_name = name
        return self
    
    def with_dtype(self, dtype: torch.dtype) -> "ModelBuilder":
        """Set model dtype."""
        self._torch_dtype = dtype
        return self
    
    def with_device_map(self, device_map: str) -> "ModelBuilder":
        """Set device mapping strategy."""
        self._device_map = device_map
        return self
    
    def with_gradient_checkpointing(self, enabled: bool = True) -> "ModelBuilder":
        """Enable/disable gradient checkpointing."""
        self._gradient_checkpointing = enabled
        return self
    
    def with_lora(
        self,
        enabled: bool = True,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05
    ) -> "ModelBuilder":
        """Configure LoRA."""
        self._lora_config = {
            "enabled": enabled,
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
        }
        return self
    
    def with_multi_gpu(self, enabled: bool = True) -> "ModelBuilder":
        """Enable multi-GPU training."""
        self._multi_gpu = enabled
        return self
    
    def with_torch_compile(self, enabled: bool = True, mode: str = "default") -> "ModelBuilder":
        """Enable torch.compile."""
        self._torch_compile = enabled
        self._compile_mode = mode
        return self
    
    def with_device_settings(self, allow_tf32: bool = True, matmul_precision: str = "high") -> "ModelBuilder":
        """Configure device settings."""
        self._device_settings = {
            "allow_tf32": allow_tf32,
            "matmul_precision": matmul_precision,
        }
        return self
    
    def build(self) -> torch.nn.Module:
        """Build the configured model."""
        if not self._model_name:
            raise ValueError("model_name must be set")
        
        # Configure device settings first
        if self._device_settings:
            self._manager.configure_device_settings(**self._device_settings)
        
        # Load model
        model = self._manager.load_model(
            model_name=self._model_name,
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
            gradient_checkpointing=self._gradient_checkpointing,
            lora_config=self._lora_config,
        )
        
        # Apply optimizations
        if self._multi_gpu:
            model = self._manager.enable_multi_gpu(model)
        
        if self._torch_compile:
            model = self._manager.enable_torch_compile(model, self._compile_mode)
        
        return model


