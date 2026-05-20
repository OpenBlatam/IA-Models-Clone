"""
Spleeter-based audio separation model.
Spleeter is a deep learning source separation library.
Refactored to use constants.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from pathlib import Path

from .base_separator import BaseSeparatorModel
from .constants import DEFAULT_NUM_SOURCES, DEFAULT_SAMPLE_RATE


class SpleeterModel(BaseSeparatorModel):
    """
    Spleeter model for audio source separation.
    
    Supports 2stems (vocals/ accompaniment) and 4stems (vocals/drums/bass/other).
    """
    
    def __init__(
        self,
        stems: int = DEFAULT_NUM_SOURCES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs
    ):
        """
        Initialize Spleeter model.
        
        Args:
            stems: Number of stems (2 or 4)
            sample_rate: Audio sample rate
        """
        if stems not in [2, 4, 5]:
            raise ValueError("stems must be 2, 4, or 5")
        
        super().__init__(num_sources=stems, sample_rate=sample_rate, **kwargs)
        self.stems = stems
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the Spleeter model."""
        try:
            from spleeter.separator import Separator
            
            model_name = f"spleeter:{self.stems}stems-16kHz"
            self.model = Separator(model_name, stft_backend="tensorflow")
            
            # Map stems to source names
            if self.stems == 2:
                self.source_names = ["vocals", "accompaniment"]
            elif self.stems == 4:
                self.source_names = ["vocals", "drums", "bass", "other"]
            elif self.stems == 5:
                self.source_names = ["vocals", "drums", "bass", "piano", "other"]
                
        except ImportError:
            raise ImportError(
                "spleeter is not installed. Install it with: pip install spleeter"
            )
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Spleeter model.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Dictionary of separated sources
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Convert tensor to numpy for Spleeter
        audio_np = audio.detach().cpu().numpy()
        if audio_np.ndim == 3:
            audio_np = audio_np.squeeze(0)
        
        # Spleeter expects (samples, channels)
        if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
            audio_np = audio_np.T
        
        # Run separation
        prediction = self.model.separate(audio_np)
        
        # Convert back to tensors
        result = {}
        for source_name in self.source_names:
            source_audio = prediction[source_name]
            if source_audio.ndim == 1:
                source_audio = source_audio.unsqueeze(0)
            result[source_name] = torch.from_numpy(source_audio).float()
        
        return result
    
    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio using Spleeter.
        
        Args:
            audio_path: Path to input audio
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping source names to output paths
        """
        try:
            from spleeter.audio.adapter import AudioAdapter
            
            if output_dir is None:
                output_dir = Path(audio_path).parent / "separated"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load audio
            audio_adapter = AudioAdapter.default()
            waveform, sample_rate = audio_adapter.load(audio_path)
            
            # Separate
            prediction = self.model.separate(waveform)
            
            # Save separated sources
            result = {}
            audio_name = Path(audio_path).stem
            
            for source_name, source_audio in prediction.items():
                output_path = output_dir / f"{audio_name}_{source_name}.wav"
                audio_adapter.save(str(output_path), source_audio, sample_rate)
                result[source_name] = str(output_path)
            
            return result
            
        except ImportError:
            raise ImportError("spleeter is not installed")
        except Exception as e:
            raise RuntimeError(f"Error during separation: {str(e)}")

