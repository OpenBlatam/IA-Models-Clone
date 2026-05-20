"""
Audio Adapter - Adapt different audio formats
"""

from typing import Any, Dict
import numpy as np
import logging

from .adapter import BaseAdapter

logger = logging.getLogger(__name__)


class AudioAdapter(BaseAdapter):
    """
    Adapter for different audio formats
    """
    
    def __init__(self):
        super().__init__("audio_file", "numpy_array")
    
    def adapt(self, data: Any) -> np.ndarray:
        """Adapt audio to numpy array"""
        try:
            import librosa
            
            if isinstance(data, str):
                # File path
                y, sr = librosa.load(data, sr=22050)
                return y
            elif isinstance(data, tuple):
                # (audio, sample_rate)
                y, sr = data
                if sr != 22050:
                    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                return y
            elif isinstance(data, np.ndarray):
                # Already numpy array
                return data
            else:
                raise ValueError(f"Unsupported audio format: {type(data)}")
        
        except Exception as e:
            logger.error(f"Error adapting audio: {str(e)}")
            raise








