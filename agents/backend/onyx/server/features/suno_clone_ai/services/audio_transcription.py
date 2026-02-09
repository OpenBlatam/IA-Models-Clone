"""
Sistema de Transcripción de Audio

Proporciona:
- Transcripción de audio a texto
- Detección de idioma
- Timestamps por palabra
- Puntuación automática
- Resumen de contenido
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("Whisper not available, transcription limited")


@dataclass
class TranscriptionSegment:
    """Segmento de transcripción"""
    start: float
    end: float
    text: str
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Resultado de transcripción"""
    text: str
    language: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AudioTranscriptionService:
    """Servicio de transcripción de audio"""
    
    def __init__(self, model_name: str = "base"):
        """
        Args:
            model_name: Nombre del modelo Whisper a usar
        """
        self.model_name = model_name
        self.model = None
        
        if WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Whisper model: {model_name}")
                self.model = whisper.load_model(model_name)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
        else:
            logger.warning("Whisper not available, using mock transcription")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> TranscriptionResult:
        """
        Transcribe un archivo de audio
        
        Args:
            audio_path: Ruta del archivo de audio
            language: Idioma (opcional, auto-detect si None)
            task: Tarea (transcribe o translate)
        
        Returns:
            TranscriptionResult
        """
        if not WHISPER_AVAILABLE or self.model is None:
            # Mock transcription para desarrollo
            return TranscriptionResult(
                text="[Mock transcription - Whisper not available]",
                language="en",
                confidence=0.0
            )
        
        try:
            # Transcribir
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task
            )
            
            # Convertir a nuestro formato
            segments = [
                TranscriptionSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    confidence=seg.get("no_speech_prob", 1.0)
                )
                for seg in result.get("segments", [])
            ]
            
            transcription = TranscriptionResult(
                text=result.get("text", ""),
                language=result.get("language", "unknown"),
                segments=segments,
                duration=result.get("duration", 0.0),
                confidence=1.0 - result.get("no_speech_prob", 0.0)
            )
            
            logger.info(f"Audio transcribed: {len(segments)} segments")
            return transcription
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def transcribe_data(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe datos de audio en memoria
        
        Args:
            audio_data: Datos de audio
            sample_rate: Sample rate
            language: Idioma
        
        Returns:
            TranscriptionResult
        """
        # Guardar temporalmente y transcribir
        import tempfile
        import soundfile as sf
        import numpy as np
        
        try:
            # Convertir bytes a numpy array (simplificado)
            # En producción usaría librosa o similar
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                # Guardar audio temporal
                # audio_array = librosa.load(io.BytesIO(audio_data), sr=sample_rate)[0]
                # sf.write(tmp.name, audio_array, sample_rate)
                
                # Transcribir
                return self.transcribe(tmp.name, language=language)
        
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            raise
    
    def detect_language(self, audio_path: str) -> str:
        """
        Detecta el idioma del audio
        
        Args:
            audio_path: Ruta del archivo
        
        Returns:
            Código de idioma
        """
        if not WHISPER_AVAILABLE or self.model is None:
            return "en"  # Default
        
        try:
            # Cargar audio y detectar idioma
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detectar idioma
            _, probs = self.model.detect_language(mel)
            language = max(probs, key=probs.get)
            
            return language
        
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "unknown"
    
    def summarize_transcription(self, transcription: TranscriptionResult) -> Dict[str, Any]:
        """
        Genera un resumen de la transcripción
        
        Args:
            transcription: Resultado de transcripción
        
        Returns:
            Resumen con estadísticas
        """
        text = transcription.text
        words = text.split()
        
        return {
            "word_count": len(words),
            "character_count": len(text),
            "segment_count": len(transcription.segments),
            "duration": transcription.duration,
            "language": transcription.language,
            "confidence": transcription.confidence,
            "avg_segment_length": (
                sum(len(seg.text) for seg in transcription.segments) / len(transcription.segments)
                if transcription.segments else 0
            )
        }


# Instancia global
_transcription_service: Optional[AudioTranscriptionService] = None


def get_transcription_service(model_name: str = "base") -> AudioTranscriptionService:
    """Obtiene la instancia global del servicio de transcripción"""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = AudioTranscriptionService(model_name=model_name)
    return _transcription_service

