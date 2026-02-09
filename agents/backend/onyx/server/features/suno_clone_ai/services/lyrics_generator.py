"""
Sistema de Generación de Letras

Proporciona:
- Generación de letras con IA
- Generación basada en tema/estilo
- Rimas y métricas
- Múltiples idiomas
- Integración con generación de música
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, lyrics generation limited")


@dataclass
class Lyrics:
    """Letra de canción"""
    title: str
    verses: List[str]
    chorus: Optional[str] = None
    bridge: Optional[str] = None
    language: str = "en"
    style: Optional[str] = None
    theme: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class LyricsGenerator:
    """Generador de letras con IA"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Args:
            model_name: Nombre del modelo a usar
        """
        self.model_name = model_name
        self.generator = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading lyrics generator model: {model_name}")
                self.generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name
                )
                logger.info("Lyrics generator model loaded")
            except Exception as e:
                logger.warning(f"Could not load lyrics model: {e}")
    
    def generate_lyrics(
        self,
        theme: str,
        style: Optional[str] = None,
        language: str = "en",
        num_verses: int = 3,
        include_chorus: bool = True
    ) -> Lyrics:
        """
        Genera letras de canción
        
        Args:
            theme: Tema de la canción
            style: Estilo musical (opcional)
            language: Idioma
            num_verses: Número de versos
            include_chorus: Incluir coro
        
        Returns:
            Lyrics generadas
        """
        if not self.generator:
            # Mock lyrics
            return Lyrics(
                title=f"Song about {theme}",
                verses=[
                    f"Verse 1 about {theme}",
                    f"Verse 2 about {theme}",
                    f"Verse 3 about {theme}"
                ],
                chorus=f"Chorus about {theme}" if include_chorus else None,
                language=language,
                style=style,
                theme=theme
            )
        
        try:
            # Construir prompt
            prompt = self._build_prompt(theme, style, language, num_verses, include_chorus)
            
            # Generar texto
            generated = self.generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True
            )
            
            # Parsear resultado
            text = generated[0]["generated_text"]
            lyrics = self._parse_lyrics(text, theme, style, language, include_chorus)
            
            logger.info(f"Lyrics generated for theme: {theme}")
            return lyrics
        
        except Exception as e:
            logger.error(f"Error generating lyrics: {e}")
            # Fallback a mock
            return Lyrics(
                title=f"Song about {theme}",
                verses=[f"Verse about {theme}"] * num_verses,
                chorus=f"Chorus about {theme}" if include_chorus else None,
                language=language,
                style=style,
                theme=theme
            )
    
    def _build_prompt(
        self,
        theme: str,
        style: Optional[str],
        language: str,
        num_verses: int,
        include_chorus: bool
    ) -> str:
        """Construye el prompt para generación"""
        prompt_parts = [f"Write a song about {theme}"]
        
        if style:
            prompt_parts.append(f"in {style} style")
        
        prompt_parts.append(f"with {num_verses} verses")
        
        if include_chorus:
            prompt_parts.append("and a chorus")
        
        return ". ".join(prompt_parts) + ":"
    
    def _parse_lyrics(
        self,
        text: str,
        theme: str,
        style: Optional[str],
        language: str,
        include_chorus: bool
    ) -> Lyrics:
        """Parsea el texto generado en estructura de letras"""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Extraer título (primera línea o basado en tema)
        title = lines[0] if lines else f"Song about {theme}"
        
        # Separar versos y coro (simplificado)
        verses = []
        chorus = None
        
        current_section = []
        for line in lines[1:]:
            if "chorus" in line.lower() or "refrain" in line.lower():
                if current_section:
                    verses.append("\n".join(current_section))
                    current_section = []
                # Siguientes líneas son el coro
                continue
            elif line and not line.startswith("["):
                current_section.append(line)
        
        if current_section:
            verses.append("\n".join(current_section))
        
        # Si no hay coro pero se requiere, usar último verso
        if include_chorus and not chorus and verses:
            chorus = verses[-1]
            verses = verses[:-1]
        
        return Lyrics(
            title=title,
            verses=verses or [f"Verse about {theme}"],
            chorus=chorus if include_chorus else None,
            language=language,
            style=style,
            theme=theme
        )
    
    def generate_from_music(
        self,
        audio_path: str,
        transcription_service: Optional[Any] = None
    ) -> Lyrics:
        """
        Genera letras basadas en música existente (vía transcripción)
        
        Args:
            audio_path: Ruta del audio
            transcription_service: Servicio de transcripción
        
        Returns:
            Lyrics generadas
        """
        if transcription_service:
            # Transcribir audio
            transcription = transcription_service.transcribe(audio_path)
            # Generar letras basadas en transcripción
            return self.generate_lyrics(
                theme=transcription.text[:100],  # Primeros 100 caracteres como tema
                language=transcription.language
            )
        else:
            return Lyrics(
                title="Generated Song",
                verses=["Verse 1", "Verse 2", "Verse 3"],
                chorus="Chorus"
            )


# Instancia global
_lyrics_generator: Optional[LyricsGenerator] = None


def get_lyrics_generator(model_name: str = "gpt2") -> LyricsGenerator:
    """Obtiene la instancia global del generador de letras"""
    global _lyrics_generator
    if _lyrics_generator is None:
        _lyrics_generator = LyricsGenerator(model_name=model_name)
    return _lyrics_generator

