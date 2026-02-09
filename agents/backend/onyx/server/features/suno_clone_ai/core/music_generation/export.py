"""
Export Module

Export audio to various formats.
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioExporter:
    """
    Export audio to various formats.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def export_wav(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        normalize: bool = True
    ) -> None:
        """
        Export audio to WAV format.
        
        Args:
            audio: Audio array
            output_path: Output file path
            normalize: Normalize audio before export
        """
        try:
            import soundfile as sf
            
            # Normalize if requested
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95  # Leave headroom
            
            # Ensure mono/stereo format
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1).T  # Convert to column vector
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(output_path), audio, self.sample_rate)
            logger.info(f"Exported WAV to {output_path}")
        except ImportError:
            raise ImportError("soundfile not installed. Install with: pip install soundfile")
        except Exception as e:
            logger.error(f"Error exporting WAV: {e}")
            raise
    
    def export_mp3(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        bitrate: str = "192k",
        normalize: bool = True
    ) -> None:
        """
        Export audio to MP3 format.
        
        Args:
            audio: Audio array
            output_path: Output file path
            bitrate: MP3 bitrate
            normalize: Normalize audio before export
        """
        try:
            import soundfile as sf
            import subprocess
            import tempfile
            
            # Normalize
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
            
            # Export to temporary WAV first
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                sf.write(tmp_wav.name, audio, self.sample_rate)
                
                # Convert to MP3 using ffmpeg
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                subprocess.run([
                    "ffmpeg", "-i", tmp_wav.name,
                    "-b:a", bitrate,
                    "-y", str(output_path)
                ], check=True, capture_output=True)
                
                # Cleanup
                Path(tmp_wav.name).unlink()
            
            logger.info(f"Exported MP3 to {output_path}")
        except ImportError:
            raise ImportError("soundfile not installed. Install with: pip install soundfile")
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg not found. Install ffmpeg to export MP3")
        except Exception as e:
            logger.error(f"Error exporting MP3: {e}")
            raise
    
    def export_flac(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        normalize: bool = True
    ) -> None:
        """
        Export audio to FLAC format (lossless).
        
        Args:
            audio: Audio array
            output_path: Output file path
            normalize: Normalize audio before export
        """
        try:
            import soundfile as sf
            
            # Normalize
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(output_path), audio, self.sample_rate, format="FLAC")
            logger.info(f"Exported FLAC to {output_path}")
        except ImportError:
            raise ImportError("soundfile not installed. Install with: pip install soundfile")
        except Exception as e:
            logger.error(f"Error exporting FLAC: {e}")
            raise
    
    def export_ogg(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        quality: int = 5,
        normalize: bool = True
    ) -> None:
        """
        Export audio to OGG Vorbis format.
        
        Args:
            audio: Audio array
            output_path: Output file path
            quality: Quality level (0-10, higher is better)
            normalize: Normalize audio before export
        """
        try:
            import soundfile as sf
            import subprocess
            import tempfile
            
            # Normalize
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
            
            # Export to temporary WAV first
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                sf.write(tmp_wav.name, audio, self.sample_rate)
                
                # Convert to OGG using ffmpeg
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                subprocess.run([
                    "ffmpeg", "-i", tmp_wav.name,
                    "-c:a", "libvorbis",
                    "-q:a", str(quality),
                    "-y", str(output_path)
                ], check=True, capture_output=True)
                
                # Cleanup
                Path(tmp_wav.name).unlink()
            
            logger.info(f"Exported OGG to {output_path}")
        except ImportError:
            raise ImportError("soundfile not installed. Install with: pip install soundfile")
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg not found. Install ffmpeg to export OGG")
        except Exception as e:
            logger.error(f"Error exporting OGG: {e}")
            raise
    
    def export_multiple_formats(
        self,
        audio: np.ndarray,
        base_path: Union[str, Path],
        formats: List[str] = ["wav", "mp3", "flac"],
        **kwargs
    ) -> Dict[str, Path]:
        """
        Export audio to multiple formats.
        
        Args:
            audio: Audio array
            base_path: Base path (without extension)
            formats: List of formats to export
            **kwargs: Additional parameters for export
            
        Returns:
            Dictionary mapping format to output path
        """
        base_path = Path(base_path)
        exported = {}
        
        for fmt in formats:
            output_path = base_path.with_suffix(f".{fmt}")
            
            try:
                if fmt == "wav":
                    self.export_wav(audio, output_path, **kwargs)
                elif fmt == "mp3":
                    self.export_mp3(audio, output_path, **kwargs)
                elif fmt == "flac":
                    self.export_flac(audio, output_path, **kwargs)
                elif fmt == "ogg":
                    self.export_ogg(audio, output_path, **kwargs)
                else:
                    logger.warning(f"Unknown format: {fmt}")
                    continue
                
                exported[fmt] = output_path
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
        
        return exported















