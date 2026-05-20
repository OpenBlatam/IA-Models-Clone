"""
Demucs-based audio separation model.
Demucs is a state-of-the-art music source separation model.
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
from pathlib import Path

from .base_separator import BaseSeparatorModel
from .constants import DEFAULT_NUM_SOURCES, DEFAULT_SAMPLE_RATE
from ..separator.constants import DEFAULT_4_STEM_SOURCES
from ..exceptions import AudioModelError
from ..logger import logger

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_VARIANT = "htdemucs"


class DemucsModel(BaseSeparatorModel):
    """
    Demucs model for audio source separation.
    
    Supports multiple variants: htdemucs, htdemucs_ft, mdx, etc.
    """
    
    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        num_sources: int = DEFAULT_NUM_SOURCES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs
    ):
        """
        Initialize Demucs model.
        
        Args:
            variant: Model variant (htdemucs, htdemucs_ft, mdx, etc.)
            num_sources: Number of sources to separate
            sample_rate: Audio sample rate
        """
        super().__init__(num_sources=num_sources, sample_rate=sample_rate, **kwargs)
        self.variant = variant
        self.model = None
        self._source_names: Optional[List[str]] = None
        self._load_model()
        
    def _load_model(self):
        """Load the Demucs model."""
        try:
            import demucs
            from demucs.pretrained import get_model
            
            logger.debug(f"Loading Demucs model variant: {self.variant}")
            self.model = get_model(self.variant)
            self.model.eval()
            
            # Update num_sources and source_names based on model
            if hasattr(self.model, 'sources'):
                self._source_names = list(self.model.sources)
                self.num_sources = len(self._source_names)
                logger.debug(f"Model has {self.num_sources} sources: {self._source_names}")
            else:
                self._source_names = None
                
        except ImportError as e:
            raise AudioModelError(
                "demucs is not installed. Install it with: pip install demucs",
                component="DemucsModel",
                error_code="DEMUCS_NOT_INSTALLED"
            ) from e
        except Exception as e:
            raise AudioModelError(
                f"Failed to load Demucs model '{self.variant}': {str(e)}",
                component="DemucsModel",
                error_code="MODEL_LOAD_FAILED"
            ) from e
    
    def _get_source_names(self) -> List[str]:
        """
        Get source names for the model.
        
        Returns:
            List of source names
        """
        if self._source_names is not None:
            return self._source_names
        
        if hasattr(self.model, 'sources'):
            return list(self.model.sources)
        
        # Fallback to default or generated names
        return DEFAULT_4_STEM_SOURCES[:self.num_sources] if self.num_sources <= len(DEFAULT_4_STEM_SOURCES) else [
            f'source_{i}' for i in range(self.num_sources)
        ]
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Demucs model.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Dictionary of separated sources
            
        Raises:
            AudioModelError: If model is not loaded or separation fails
        """
        if self.model is None:
            raise AudioModelError(
                "Model not loaded. Call _load_model() first.",
                component="DemucsModel",
                error_code="MODEL_NOT_LOADED"
            )
        
        try:
            with torch.no_grad():
                separated = self.model(audio)
            
            # Map to source names
            source_names = self._get_source_names()
            result = {}
            for i, source_name in enumerate(source_names):
                if i < len(separated):
                    result[source_name] = separated[i]
            
            return result
        except Exception as e:
            raise AudioModelError(
                f"Error during forward pass: {str(e)}",
                component="DemucsModel",
                error_code="FORWARD_FAILED"
            ) from e
    
    def _prepare_output_dir(self, audio_path: str, output_dir: Optional[str]) -> Path:
        """
        Prepare output directory for separation results.
        
        Args:
            audio_path: Path to input audio
            output_dir: Desired output directory (None for default)
            
        Returns:
            Prepared output directory Path
        """
        if output_dir is None:
            output_dir = Path(audio_path).parent / "separated"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _map_output_files(
        self,
        output_dir: Path,
        audio_name: str
    ) -> Dict[str, str]:
        """
        Map output files to source names.
        
        Args:
            output_dir: Output directory
            audio_name: Base name of audio file
            
        Returns:
            Dictionary mapping source names to output paths
        """
        result = {}
        source_names = self._get_source_names()
        
        for source_name in source_names:
            output_path = output_dir / self.variant / audio_name / f"{source_name}.wav"
            if output_path.exists():
                result[source_name] = str(output_path)
        
        return result
    
    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio using Demucs.
        
        Args:
            audio_path: Path to input audio
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping source names to output paths
            
        Raises:
            AudioModelError: If separation fails
        """
        try:
            from demucs import separate
        except ImportError as e:
            raise AudioModelError(
                "demucs is not installed. Install it with: pip install demucs",
                component="DemucsModel",
                error_code="DEMUCS_NOT_INSTALLED"
            ) from e
        
        try:
            # Prepare output directory
            output_dir = self._prepare_output_dir(audio_path, output_dir)
            
            # Run separation
            logger.debug(f"Running Demucs separation with variant: {self.variant}")
            separate.track(
                audio_path,
                out=output_dir,
                model=self.variant,
                **kwargs
            )
            
            # Map output files to source names
            audio_name = Path(audio_path).stem
            result = self._map_output_files(output_dir, audio_name)
            
            logger.info(f"Successfully separated into {len(result)} sources")
            return result
            
        except Exception as e:
            raise AudioModelError(
                f"Error during separation: {str(e)}",
                component="DemucsModel",
                error_code="SEPARATION_FAILED"
            ) from e

