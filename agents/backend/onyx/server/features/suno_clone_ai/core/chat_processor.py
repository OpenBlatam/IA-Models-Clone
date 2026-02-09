"""
Procesador de chat para interpretar solicitudes de canciones
Extrae información de la conversación del usuario y la convierte en prompts para generación
"""

import logging
import re
from typing import Dict, Optional, List
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class ChatProcessor:
    """Procesa mensajes de chat y extrae información para generación de música"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    
    def extract_song_info(self, user_message: str, chat_history: Optional[List[Dict]] = None) -> Dict:
        """
        Extrae información de la canción desde el mensaje del usuario
        
        Args:
            user_message: Mensaje del usuario
            chat_history: Historial de conversación (opcional)
            
        Returns:
            Diccionario con información extraída:
            - prompt: Prompt para generación
            - genre: Género musical
            - mood: Estado de ánimo
            - tempo: Tempo (si se menciona)
            - instruments: Instrumentos mencionados
            - duration: Duración solicitada
        """
        try:
            # Extraer información básica usando regex
            genre = self._extract_genre(user_message)
            mood = self._extract_mood(user_message)
            tempo = self._extract_tempo(user_message)
            instruments = self._extract_instruments(user_message)
            duration = self._extract_duration(user_message)
            
            # Si hay OpenAI disponible, usar para mejorar el prompt
            if self.client:
                enhanced_prompt = self._enhance_prompt_with_ai(user_message, chat_history)
            else:
                enhanced_prompt = self._create_basic_prompt(user_message, genre, mood, tempo, instruments)
            
            return {
                "prompt": enhanced_prompt,
                "genre": genre,
                "mood": mood,
                "tempo": tempo,
                "instruments": instruments,
                "duration": duration,
                "original_message": user_message
            }
            
        except Exception as e:
            logger.error(f"Error extracting song info: {e}")
            # Fallback a prompt básico
            return {
                "prompt": user_message,
                "genre": None,
                "mood": None,
                "tempo": None,
                "instruments": [],
                "duration": None,
                "original_message": user_message
            }
    
    def _extract_genre(self, text: str) -> Optional[str]:
        """Extrae el género musical mencionado"""
        genres = [
            "rock", "pop", "jazz", "classical", "electronic", "hip hop", "rap",
            "country", "blues", "reggae", "metal", "folk", "latin", "r&b",
            "soul", "funk", "disco", "techno", "house", "ambient", "indie"
        ]
        
        text_lower = text.lower()
        for genre in genres:
            if genre in text_lower:
                return genre
        
        return None
    
    def _extract_mood(self, text: str) -> Optional[str]:
        """Extrae el estado de ánimo mencionado"""
        moods = [
            "happy", "sad", "energetic", "calm", "melancholic", "uplifting",
            "dark", "bright", "romantic", "aggressive", "peaceful", "dramatic"
        ]
        
        text_lower = text.lower()
        for mood in moods:
            if mood in text_lower:
                return mood
        
        return None
    
    def _extract_tempo(self, text: str) -> Optional[str]:
        """Extrae el tempo mencionado"""
        tempo_patterns = [
            r"(\d+)\s*bpm",
            r"(slow|fast|medium|moderate)\s*tempo",
            r"tempo\s*(slow|fast|medium|moderate)"
        ]
        
        for pattern in tempo_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return None
    
    def _extract_instruments(self, text: str) -> List[str]:
        """Extrae instrumentos mencionados"""
        instruments = [
            "guitar", "piano", "drums", "bass", "violin", "saxophone",
            "trumpet", "flute", "cello", "viola", "harp", "organ",
            "synthesizer", "synth", "keyboard", "vocals", "voice"
        ]
        
        found = []
        text_lower = text.lower()
        for instrument in instruments:
            if instrument in text_lower:
                found.append(instrument)
        
        return found
    
    def _extract_duration(self, text: str) -> Optional[int]:
        """Extrae la duración solicitada en segundos"""
        duration_patterns = [
            r"(\d+)\s*seconds?",
            r"(\d+)\s*sec",
            r"(\d+)\s*minutes?",
            r"(\d+)\s*min"
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = int(match.group(1))
                if "min" in pattern:
                    value *= 60
                return min(value, settings.max_audio_length)
        
        return None
    
    def _create_basic_prompt(self, text: str, genre: Optional[str], mood: Optional[str],
                            tempo: Optional[str], instruments: List[str]) -> str:
        """Crea un prompt básico sin IA"""
        parts = [text]
        
        if genre:
            parts.append(f"Genre: {genre}")
        if mood:
            parts.append(f"Mood: {mood}")
        if tempo:
            parts.append(f"Tempo: {tempo}")
        if instruments:
            parts.append(f"Instruments: {', '.join(instruments)}")
        
        return ", ".join(parts)
    
    def _enhance_prompt_with_ai(self, user_message: str, chat_history: Optional[List[Dict]]) -> str:
        """Mejora el prompt usando OpenAI"""
        if not self.client:
            return self._create_basic_prompt(user_message, None, None, None, [])
        
        try:
            system_prompt = """You are a music generation assistant. Your task is to convert user requests into detailed prompts for AI music generation.

Extract key information:
- Musical style/genre
- Mood/emotion
- Tempo
- Instruments
- Any specific musical elements

Create a concise but descriptive prompt that will help generate the requested music."""
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if chat_history:
                messages.extend(chat_history[-5:])  # Últimos 5 mensajes para contexto
            
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Error enhancing prompt with AI: {e}")
            return self._create_basic_prompt(user_message, None, None, None, [])


# Instancia global
_chat_processor: Optional[ChatProcessor] = None


def get_chat_processor() -> ChatProcessor:
    """Obtiene la instancia global del procesador de chat"""
    global _chat_processor
    if _chat_processor is None:
        _chat_processor = ChatProcessor()
    return _chat_processor

