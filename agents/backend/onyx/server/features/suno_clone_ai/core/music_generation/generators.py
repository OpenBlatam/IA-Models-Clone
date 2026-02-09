"""
Music Generation Models Module

Unified interface for different music generation models.
"""

from typing import Optional, Dict, Any, List, Union
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class BaseMusicGenerator:
    """Base class for music generators."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the model."""
        raise NotImplementedError
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        **kwargs
    ) -> np.ndarray:
        """Generate music from prompt."""
        raise NotImplementedError
    
    def generate_batch(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate music for multiple prompts."""
        raise NotImplementedError


class AudiocraftGenerator(BaseMusicGenerator):
    """
    Audiocraft (Meta) music generator.
    
    Supports:
    - musicgen-large
    - musicgen-stereo-large
    - musicgen-melody
    """
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-large",
        device: Optional[str] = None
    ):
        super().__init__(device)
        self.model_name = model_name
    
    def initialize(self) -> None:
        """Initialize Audiocraft model."""
        try:
            from audiocraft.models import MusicGen
            
            logger.info(f"Loading Audiocraft model: {self.model_name}")
            self.model = MusicGen.get_pretrained(
                self.model_name,
                device=self.device
            )
            self._initialized = True
            logger.info("Audiocraft model loaded successfully")
        except ImportError:
            raise ImportError(
                "audiocraft not installed. Install with: pip install audiocraft"
            )
        except Exception as e:
            logger.error(f"Error loading Audiocraft model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        temperature: float = 1.0,
        top_k: int = 250,
        use_sampling: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Generate music from text prompt."""
        if not self._initialized:
            self.initialize()
        
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            use_sampling=use_sampling
        )
        
        with torch.inference_mode():
            audio = self.model.generate([prompt], progress=False)
            audio = audio[0].cpu().numpy()
        
        return audio
    
    def generate_batch(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate music for multiple prompts."""
        if not self._initialized:
            self.initialize()
        
        self.model.set_generation_params(duration=duration, **kwargs)
        
        with torch.inference_mode():
            audio_batch = self.model.generate(prompts, progress=False)
            audio_list = [audio.cpu().numpy() for audio in audio_batch]
        
        return audio_list


class MusicGenHuggingFaceGenerator(BaseMusicGenerator):
    """
    MusicGen via Hugging Face Transformers.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-large",
        device: Optional[str] = None
    ):
        super().__init__(device)
        self.model_name = model_name
        self.processor = None
    
    def initialize(self) -> None:
        """Initialize MusicGen model from Hugging Face."""
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading MusicGen model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name
            )
            self.model = self.model.to(self.device)
            self._initialized = True
            logger.info("MusicGen model loaded successfully")
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Error loading MusicGen model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        max_new_tokens: int = 1024,
        **kwargs
    ) -> np.ndarray:
        """Generate music from text prompt."""
        if not self._initialized:
            self.initialize()
        
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.inference_mode():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            audio = audio_values[0].cpu().numpy()
        
        return audio
    
    def generate_batch(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate music for multiple prompts."""
        if not self._initialized:
            self.initialize()
        
        inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.inference_mode():
            audio_values = self.model.generate(**inputs, **kwargs)
            audio_list = [audio.cpu().numpy() for audio in audio_values]
        
        return audio_list


class StableAudioGenerator(BaseMusicGenerator):
    """
    Stable Audio generator (Stability AI).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "stable-audio-2.0"
    ):
        super().__init__()
        self.api_key = api_key
        self.model_name = model
        self.client = None
    
    def initialize(self) -> None:
        """Initialize Stable Audio client."""
        try:
            from stability_sdk import client
            import os
            
            api_key = self.api_key or os.environ.get('STABILITY_KEY')
            if not api_key:
                raise ValueError("Stability API key required")
            
            self.client = client.StabilityInference(
                key=api_key,
                verbose=True,
            )
            self._initialized = True
            logger.info("Stable Audio client initialized")
        except ImportError:
            raise ImportError(
                "stability-sdk not installed. Install with: pip install stability-sdk"
            )
        except Exception as e:
            logger.error(f"Error initializing Stable Audio: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        **kwargs
    ) -> np.ndarray:
        """Generate music from text prompt."""
        if not self._initialized:
            self.initialize()
        
        try:
            audio = self.client.generate(
                prompt=prompt,
                duration=duration,
                model=self.model_name,
                **kwargs
            )
            return np.array(audio)
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise


def create_generator(
    generator_type: str = "audiocraft",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseMusicGenerator:
    """
    Factory function to create music generators.
    
    Args:
        generator_type: Type of generator (audiocraft, musicgen_hf, stable_audio)
        model_name: Model name to use
        **kwargs: Additional arguments
        
    Returns:
        Music generator instance
    """
    generators = {
        "audiocraft": AudiocraftGenerator,
        "musicgen_hf": MusicGenHuggingFaceGenerator,
        "stable_audio": StableAudioGenerator,
    }
    
    if generator_type not in generators:
        raise ValueError(
            f"Unknown generator type: {generator_type}. "
            f"Available: {list(generators.keys())}"
        )
    
    generator_class = generators[generator_type]
    
    if model_name:
        kwargs["model_name"] = model_name
    
    return generator_class(**kwargs)















