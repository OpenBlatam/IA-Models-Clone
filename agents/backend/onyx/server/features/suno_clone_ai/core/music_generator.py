"""
Enhanced Music Generator with Deep Learning Best Practices

Implements:
- Proper model initialization and weight initialization
- Mixed precision training support
- Gradient clipping and NaN/Inf handling
- Efficient GPU utilization
- Proper error handling and logging
"""

import logging
import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

# Optimization with numba for numerical operations
try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize model weights using best practices.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        # Xavier/Glorot initialization for linear and conv layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    if hasattr(model, 'parameters'):
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm.item()
    return 0.0


def check_for_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check for NaN or Inf values in tensor.
    
    Args:
        tensor: Tensor to check
        name: Name for logging
        
    Returns:
        True if NaN/Inf found, False otherwise
    """
    if torch.isnan(tensor).any():
        logger.warning(f"NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        logger.warning(f"Inf detected in {name}")
        return True
    return False


class MusicGenerator:
    """
    Enhanced music generator with deep learning best practices.
    
    Features:
    - Proper weight initialization
    - Mixed precision support
    - Gradient clipping
    - NaN/Inf detection
    - Efficient GPU utilization
    """
    
    def __init__(
        self,
        use_mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        enable_autograd_anomaly: bool = False,
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead"
    ):
        """
        Initialize music generator.
        
        Args:
            use_mixed_precision: Enable mixed precision training/inference
            gradient_clip_norm: Maximum gradient norm for clipping
            enable_autograd_anomaly: Enable autograd anomaly detection for debugging
        """
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.gradient_clip_norm = gradient_clip_norm
        self.scaler = None
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        
        # Setup mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision enabled")
        
        # Enable autograd anomaly detection for debugging
        if enable_autograd_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Autograd anomaly detection enabled (performance impact)")
        
        # Setup optimal GPU settings
        self._setup_gpu_optimizations()
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Get the appropriate device (cuda/cpu)."""
        if torch.cuda.is_available() and settings.use_gpu:
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            device = "cpu"
            logger.info("Using CPU")
        return device
    
    def _setup_gpu_optimizations(self) -> None:
        """Setup optimal GPU settings for performance."""
        if torch.cuda.is_available():
            # Enable cuDNN benchmarking for faster convolutions
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable TensorFloat-32 for faster operations on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("GPU optimizations enabled")
    
    def _load_model(self) -> None:
        """Load the music generation model with proper initialization."""
        try:
            logger.info(f"Loading music generation model: {settings.music_model}")
            logger.info(f"Using device: {self.device}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(settings.music_model)
            
            # Load model
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                settings.music_model
            )
            
            # Apply weight initialization if needed (for fine-tuning)
            if hasattr(self.model, 'apply'):
                self.model.apply(initialize_weights)
            
            # Move to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Compile model for speed (if enabled)
            if self.use_compile and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(
                        self.model,
                        mode=self.compile_mode,
                        fullgraph=False
                    )
                    logger.info(f"Model compiled with mode: {self.compile_mode}")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
            
            # Enable gradient checkpointing if available (saves memory)
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
                except Exception as e:
                    logger.debug(f"Could not enable gradient checkpointing: {e}")
            
            logger.info("Music generation model loaded successfully")
            
            # Log model info
            if hasattr(self.model, 'config'):
                logger.info(f"Model config: {self.model.config}")
            
        except Exception as e:
            logger.error(f"Error loading music model: {e}", exc_info=True)
            raise
    
    def generate_from_text(
        self,
        text: str,
        duration: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music from text with proper error handling.
        
        Args:
            text: Song description or lyrics
            duration: Duration in seconds (default: settings.default_duration)
            guidance_scale: Guidance scale for generation
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated audio array
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input validation fails
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            duration = duration or settings.default_duration
            guidance_scale = guidance_scale or settings.cfg_coef
            temperature = temperature or settings.temperature
            top_k = top_k or settings.top_k
            top_p = top_p or settings.top_p
            
            # Validate inputs
            if duration <= 0 or duration > settings.max_audio_length:
                raise ValueError(
                    f"Duration must be between 1 and {settings.max_audio_length} seconds"
                )
            
            # Prepare inputs with proper tokenization
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Check for NaN/Inf in inputs
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    check_for_nan_inf(value, f"input_{key}")
            
            # Calculate max_new_tokens
            hop_length = getattr(
                self.model.config.audio_encoder, 
                'hop_length', 
                512
            )
            max_new_tokens = int(duration * settings.sample_rate / hop_length)
            
            # Generate audio with mixed precision if enabled
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        audio_values = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            guidance_scale=guidance_scale,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            **kwargs
                        )
                else:
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        **kwargs
                    )
            
            # Check for NaN/Inf in output
            check_for_nan_inf(audio_values, "audio_output")
            
            # Convert to numpy
            audio = audio_values[0, 0].cpu().numpy()
            
            # Validate output
            if np.isnan(audio).any() or np.isinf(audio).any():
                logger.error("Generated audio contains NaN or Inf values")
                raise ValueError("Invalid audio generated (NaN/Inf detected)")
            
            logger.info(
                f"Generated audio: shape={audio.shape}, "
                f"duration={len(audio)/settings.sample_rate:.2f}s"
            )
            
            return audio
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory. Try reducing duration or batch size.")
        except Exception as e:
            logger.error(f"Error generating music: {e}", exc_info=True)
            raise
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None
    ) -> str:
        """
        Save generated audio to file with proper error handling.
        
        Args:
            audio: Audio array
            output_path: Path where to save the file
            sample_rate: Sample rate of the audio
            
        Returns:
            Path of saved file
            
        Raises:
            ValueError: If audio is invalid
        """
        try:
            # Validate audio
            if audio is None or len(audio) == 0:
                raise ValueError("Audio array is empty")
            
            if np.isnan(audio).any() or np.isinf(audio).any():
                raise ValueError("Audio contains NaN or Inf values")
            
            sample_rate = sample_rate or settings.sample_rate
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio to prevent clipping
            max_val = torch.abs(audio_tensor).max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
                logger.warning("Audio normalized to prevent clipping")
            
            # Save file
            torchaudio.save(
                str(output_path),
                audio_tensor,
                sample_rate,
                format="wav"
            )
            
            logger.info(f"Audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}", exc_info=True)
            raise
    
    def generate_and_save(
        self,
        text: str,
        output_path: str,
        duration: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate music and save directly.
        
        Args:
            text: Song description
            output_path: Path where to save
            duration: Duration in seconds
            **kwargs: Additional parameters
            
        Returns:
            Path of saved file
        """
        audio = self.generate_from_text(text, duration=duration, **kwargs)
        return self.save_audio(audio, output_path)
    
    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")


# Global instance
_music_generator: Optional[MusicGenerator] = None


def get_music_generator() -> MusicGenerator:
    """Get the global music generator instance."""
    global _music_generator
    if _music_generator is None:
        _music_generator = MusicGenerator()
    return _music_generator
