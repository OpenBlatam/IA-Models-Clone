"""
Pipeline Manager
=================

Manages Flux2 pipeline initialization, loading, and optimization.
"""

import logging
import os
from typing import Optional
import torch

try:
    from diffusers import FluxPipeline, FluxInpaintPipeline
    from huggingface_hub import login
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages Flux2 pipeline initialization and optimization."""
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/flux2-dev",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_inpainting: bool = True,
    ):
        """
        Initialize pipeline manager.
        
        Args:
            model_id: HuggingFace model ID
            device: Torch device
            dtype: Data type
            use_inpainting: Use inpainting pipeline
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_inpainting = use_inpainting
        self.pipeline: Optional[FluxPipeline] = None
    
    def load_pipeline(self) -> FluxPipeline:
        """
        Load and return the appropriate pipeline.
        
        Returns:
            Loaded pipeline
            
        Raises:
            RuntimeError: If pipeline loading fails
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers library is required. Install with: pip install diffusers transformers"
            )
        
        if self.pipeline is not None:
            return self.pipeline
        
        # Check for HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
                logger.info("HuggingFace token authenticated")
            except Exception as e:
                logger.warning(f"Failed to login with HuggingFace token: {e}")
        
        try:
            # Prepare loading arguments
            load_kwargs = {
                "torch_dtype": self.dtype,
            }
            
            # Add token if available
            if hf_token:
                load_kwargs["token"] = hf_token
            
            # Try to load from local cache first
            cache_dir = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            local_path = os.path.join(cache_dir, "hub", f"models--{self.model_id.replace('/', '--')}")
            
            if os.path.exists(local_path):
                logger.info(f"Attempting to load from local cache: {local_path}")
                load_kwargs["local_files_only"] = True
            
            if self.use_inpainting:
                logger.info(f"Loading FluxInpaintPipeline: {self.model_id}")
                try:
                    self.pipeline = FluxInpaintPipeline.from_pretrained(
                        self.model_id,
                        **load_kwargs
                    )
                except Exception as local_error:
                    if load_kwargs.get("local_files_only"):
                        logger.warning(f"Local load failed, trying remote: {local_error}")
                        load_kwargs.pop("local_files_only", None)
                        self.pipeline = FluxInpaintPipeline.from_pretrained(
                            self.model_id,
                            **load_kwargs
                        )
                    else:
                        raise
            else:
                logger.info(f"Loading FluxPipeline: {self.model_id}")
                try:
                    self.pipeline = FluxPipeline.from_pretrained(
                        self.model_id,
                        **load_kwargs
                    )
                except Exception as local_error:
                    if load_kwargs.get("local_files_only"):
                        logger.warning(f"Local load failed, trying remote: {local_error}")
                        load_kwargs.pop("local_files_only", None)
                        self.pipeline = FluxPipeline.from_pretrained(
                            self.model_id,
                            **load_kwargs
                        )
                    else:
                        raise
            
            # Move to device
            if self.device is not None:
                self.pipeline = self.pipeline.to(self.device)
            
            logger.info(f"Pipeline loaded successfully on {self.device}")
            return self.pipeline
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading pipeline: {error_msg}")
            
            # Provide helpful error messages
            if "not cached locally" in error_msg or "fetch metadata" in error_msg:
                help_msg = (
                    f"\n❌ Error: No se pudo cargar el modelo {self.model_id}\n"
                    f"📋 Posibles soluciones:\n"
                    f"1. Verifica tu conexión a internet\n"
                    f"2. El modelo puede requerir autenticación en HuggingFace\n"
                    f"   - Obtén un token en: https://huggingface.co/settings/tokens\n"
                    f"   - Configúralo con: set HUGGINGFACE_TOKEN=tu_token\n"
                    f"3. Acepta los términos del modelo en: https://huggingface.co/{self.model_id}\n"
                    f"4. Intenta descargar el modelo manualmente primero\n"
                )
                raise RuntimeError(help_msg + f"\nError original: {error_msg}") from e
            elif "401" in error_msg or "Unauthorized" in error_msg:
                help_msg = (
                    f"\n❌ Error: No autorizado para acceder al modelo {self.model_id}\n"
                    f"📋 Solución:\n"
                    f"1. Obtén un token en: https://huggingface.co/settings/tokens\n"
                    f"2. Configúralo con: set HUGGINGFACE_TOKEN=tu_token\n"
                    f"3. Acepta los términos del modelo en: https://huggingface.co/{self.model_id}\n"
                )
                raise RuntimeError(help_msg + f"\nError original: {error_msg}") from e
            else:
                raise RuntimeError(f"Failed to load pipeline: {error_msg}") from e
    
    def apply_optimizations(self, enable_attention_slicing: bool = True) -> None:
        """
        Apply optimizations to the pipeline.
        
        Args:
            enable_attention_slicing: Enable attention slicing
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        try:
            if enable_attention_slicing and hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing(1)
                logger.info("Attention slicing enabled")
            
            try:
                if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("XFormers memory efficient attention enabled")
            except Exception:
                logger.warning("XFormers not available, continuing without it")
            
            try:
                if hasattr(torch, "compile"):
                    self.pipeline = torch.compile(self.pipeline, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
            except Exception:
                logger.warning("torch.compile not available or failed")
        
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
    
    def get_pipeline(self) -> Optional[FluxPipeline]:
        """Get the current pipeline."""
        return self.pipeline

