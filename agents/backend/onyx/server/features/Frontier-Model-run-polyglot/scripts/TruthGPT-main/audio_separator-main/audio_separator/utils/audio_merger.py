"""
Audio merging utilities.

Refactored to consolidate functions into AudioMerger class.
"""

from typing import Dict, Optional
import numpy as np

from ..core.base_component import BaseComponent
from ..separator.base_separator import DEFAULT_SAMPLE_RATE
from ..exceptions import AudioValidationError
from ..logger import logger
from .audio_helpers import pad_audio_to_length, normalize_by_peak


class AudioMerger(BaseComponent):
    """
    Audio merging and blending utilities.
    
    Responsibilities:
    - Merge multiple audio sources
    - Create custom mixes with volume control
    - Blend two audio signals
    
    Single Responsibility: Handle all audio merging and blending operations.
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, name: Optional[str] = None):
        """
        Initialize audio merger.
        
        Args:
            sample_rate: Sample rate for audio operations
            name: Component name (defaults to class name)
        """
        super().__init__(name=name or "AudioMerger")
        self.sample_rate = sample_rate
        self.initialize()
    
    def _do_initialize(self, **kwargs):
        """No initialization needed."""
        pass
    
    def merge_sources(
        self,
        sources: Dict[str, np.ndarray],
        volumes: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Merge multiple audio sources into one.
        
        Args:
            sources: Dictionary of source names to audio arrays
            volumes: Optional volume levels for each source (0.0-1.0)
            normalize: Normalize output to prevent clipping
            
        Returns:
            Merged audio array
            
        Raises:
            AudioValidationError: If no sources provided
        """
        if not sources:
            raise AudioValidationError(
                "No sources provided for merging",
                component=self.name,
                error_code="NO_SOURCES"
            )
        
        # Get maximum length
        max_length = max(len(audio) for audio in sources.values())
        
        # Initialize output
        merged = np.zeros(max_length, dtype=np.float32)
        
        # Default volumes
        if volumes is None:
            volumes = {name: 1.0 for name in sources.keys()}
        
        # Merge sources
        for source_name, source_audio in sources.items():
            volume = volumes.get(source_name, 1.0)
            
            # Ensure same length using helper
            padded = pad_audio_to_length(source_audio, max_length)
            merged += padded * volume
        
        # Normalize if needed using helper
        if normalize:
            merged = normalize_by_peak(merged, target_peak=1.0)
        
        return merged
    
    def create_mix(
        self,
        sources: Dict[str, np.ndarray],
        mix_config: Dict[str, float]
    ) -> np.ndarray:
        """
        Create a custom mix from sources.
        
        Args:
            sources: Dictionary of source names to audio arrays
            mix_config: Mix configuration with volumes and effects
            
        Returns:
            Mixed audio array
        """
        volumes = {}
        for source_name in sources.keys():
            volumes[source_name] = mix_config.get(f"{source_name}_volume", 1.0)
        
        merged = self.merge_sources(sources, volumes, normalize=True)
        
        # Apply fade if specified
        if "fade_in" in mix_config or "fade_out" in mix_config:
            from .audio_enhancement import AudioEnhancer
            enhancer = AudioEnhancer(sample_rate=self.sample_rate)
            merged = enhancer.apply_fade(
                merged,
                fade_in=mix_config.get("fade_in", 0.0),
                fade_out=mix_config.get("fade_out", 0.0)
            )
        
        return merged
    
    def blend(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        blend_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Blend two audio signals.
        
        Args:
            audio1: First audio array
            audio2: Second audio array
            blend_ratio: Blend ratio (0.0 = all audio1, 1.0 = all audio2)
            
        Returns:
            Blended audio array
        """
        # Ensure same length using helper
        max_length = max(len(audio1), len(audio2))
        audio1 = pad_audio_to_length(audio1, max_length)
        audio2 = pad_audio_to_length(audio2, max_length)
        
        # Blend
        blended = audio1 * (1 - blend_ratio) + audio2 * blend_ratio
        
        return blended


# Backward compatibility functions
def merge_sources(
    sources: Dict[str, np.ndarray],
    volumes: Optional[Dict[str, float]] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Merge multiple audio sources into one (backward compatibility).
    
    Args:
        sources: Dictionary of source names to audio arrays
        volumes: Optional volume levels for each source (0.0-1.0)
        normalize: Normalize output to prevent clipping
        
    Returns:
        Merged audio array
    """
    merger = AudioMerger()
    return merger.merge_sources(sources, volumes, normalize)


def create_mix(
    sources: Dict[str, np.ndarray],
    mix_config: Dict[str, float],
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """
    Create a custom mix from sources (backward compatibility).
    
    Args:
        sources: Dictionary of source names to audio arrays
        mix_config: Mix configuration with volumes and effects
        sample_rate: Sample rate
        
    Returns:
        Mixed audio array
    """
    merger = AudioMerger(sample_rate=sample_rate)
    return merger.create_mix(sources, mix_config)


def blend_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    blend_ratio: float = 0.5
) -> np.ndarray:
    """
    Blend two audio signals (backward compatibility).
    
    Args:
        audio1: First audio array
        audio2: Second audio array
        blend_ratio: Blend ratio (0.0 = all audio1, 1.0 = all audio2)
        
    Returns:
        Blended audio array
    """
    merger = AudioMerger()
    return merger.blend(audio1, audio2, blend_ratio)
