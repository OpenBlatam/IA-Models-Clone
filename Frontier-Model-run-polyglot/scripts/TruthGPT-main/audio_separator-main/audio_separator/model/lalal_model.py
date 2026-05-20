"""
LALAL.AI-based audio separation model.
Wrapper for LALAL.AI API integration.
Refactored to use constants.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from pathlib import Path

from .base_separator import BaseSeparatorModel
from .constants import DEFAULT_SAMPLE_RATE


class LalalModel(BaseSeparatorModel):
    """
    LALAL.AI model wrapper for audio source separation.
    
    This is a wrapper around the LALAL.AI API service.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        num_sources: int = 2,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs
    ):
        """
        Initialize LALAL.AI model.
        
        Args:
            api_key: LALAL.AI API key
            num_sources: Number of sources (typically 2: vocals/ instrumental)
            sample_rate: Audio sample rate
        """
        super().__init__(num_sources=num_sources, sample_rate=sample_rate, **kwargs)
        self.api_key = api_key or self._get_api_key()
        self.source_names = ["vocals", "instrumental"]
        
    def _get_api_key(self) -> str:
        """Get API key from environment or config."""
        import os
        api_key = os.getenv("LALAL_API_KEY")
        if not api_key:
            raise ValueError(
                "LALAL_API_KEY not found. Set it as environment variable or pass as api_key."
            )
        return api_key
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LALAL.AI model.
        
        Note: LALAL.AI is an API service, so this method
        processes local audio and calls the API.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Dictionary of separated sources
        """
        raise NotImplementedError(
            "LALAL.AI requires API calls. Use separate() method instead."
        )
    
    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio using LALAL.AI API.
        
        Args:
            audio_path: Path to input audio
            output_dir: Output directory
            **kwargs: Additional arguments (filter_type, split_type, etc.)
            
        Returns:
            Dictionary mapping source names to output paths
        """
        try:
            import requests
            import time
            
            if output_dir is None:
                output_dir = Path(audio_path).parent / "separated"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Upload audio
            upload_url = "https://api.lalal.ai/v1/upload"
            with open(audio_path, 'rb') as f:
                files = {'file': f}
                headers = {'Authorization': f'Bearer {self.api_key}'}
                response = requests.post(upload_url, files=files, headers=headers)
                response.raise_for_status()
                upload_data = response.json()
                task_id = upload_data.get('id')
            
            # Wait for processing
            status_url = f"https://api.lalal.ai/v1/task/{task_id}"
            while True:
                response = requests.get(status_url, headers=headers)
                response.raise_for_status()
                status_data = response.json()
                
                if status_data.get('status') == 'completed':
                    break
                elif status_data.get('status') == 'failed':
                    raise RuntimeError("LALAL.AI processing failed")
                
                time.sleep(2)
            
            # Download results
            result = {}
            audio_name = Path(audio_path).stem
            
            for source_name in self.source_names:
                download_url = status_data.get('download_urls', {}).get(source_name)
                if download_url:
                    output_path = output_dir / f"{audio_name}_{source_name}.wav"
                    response = requests.get(download_url)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    result[source_name] = str(output_path)
            
            return result
            
        except ImportError:
            raise ImportError("requests is not installed")
        except Exception as e:
            raise RuntimeError(f"Error during LALAL.AI separation: {str(e)}")

