"""
Translation Service for Social Video Transcriber AI
Provides automatic translation of transcriptions using OpenRouter
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from ..config.settings import get_settings
from .openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Supported languages for translation"""
    SPANISH = "es"
    ENGLISH = "en"
    PORTUGUESE = "pt"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


LANGUAGE_NAMES = {
    SupportedLanguage.SPANISH: "Español",
    SupportedLanguage.ENGLISH: "English",
    SupportedLanguage.PORTUGUESE: "Português",
    SupportedLanguage.FRENCH: "Français",
    SupportedLanguage.GERMAN: "Deutsch",
    SupportedLanguage.ITALIAN: "Italiano",
    SupportedLanguage.CHINESE: "中文",
    SupportedLanguage.JAPANESE: "日本語",
    SupportedLanguage.KOREAN: "한국어",
    SupportedLanguage.RUSSIAN: "Русский",
    SupportedLanguage.ARABIC: "العربية",
    SupportedLanguage.HINDI: "हिन्दी",
}


@dataclass
class TranslationResult:
    """Translation result"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    word_count_original: int
    word_count_translated: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "word_count_original": self.word_count_original,
            "word_count_translated": self.word_count_translated,
            "confidence": self.confidence,
        }


@dataclass
class SegmentTranslation:
    """Translation of a single segment"""
    segment_id: int
    start_time: float
    end_time: float
    original_text: str
    translated_text: str


class TranslationService:
    """Service for translating transcriptions"""
    
    TRANSLATION_PROMPT = """Traduce el siguiente texto de {source_lang} a {target_lang}.

REGLAS IMPORTANTES:
1. Mantén el significado y tono original
2. Usa expresiones naturales en el idioma destino
3. Preserva los nombres propios y marcas
4. Mantén el formato de párrafos

TEXTO A TRADUCIR:
{text}

Responde SOLO con la traducción, sin explicaciones adicionales."""

    BATCH_TRANSLATION_PROMPT = """Traduce los siguientes segmentos de {source_lang} a {target_lang}.
Mantén el mismo formato JSON con los textos traducidos.

SEGMENTOS:
{segments}

Responde SOLO en JSON válido:
{{
    "translations": [
        {{"id": 0, "text": "traducción del segmento 0"}},
        {{"id": 1, "text": "traducción del segmento 1"}}
    ]
}}"""

    DETECT_LANGUAGE_PROMPT = """Detecta el idioma del siguiente texto y responde con el código ISO 639-1.

TEXTO:
{text}

Responde SOLO con el código de idioma (ej: es, en, pt, fr, de, it, zh, ja, ko, ru, ar, hi)."""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_openrouter_client()
    
    async def detect_language(self, text: str) -> str:
        """
        Detect the language of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO 639-1 language code
        """
        if len(text.strip()) < 10:
            return "unknown"
        
        sample = text[:500] if len(text) > 500 else text
        
        try:
            response = await self.client.complete(
                prompt=self.DETECT_LANGUAGE_PROMPT.format(text=sample),
                system_prompt="Eres un experto en detección de idiomas. Responde solo con el código ISO 639-1.",
                max_tokens=10,
                temperature=0.1,
            )
            
            code = response.strip().lower()[:2]
            
            try:
                SupportedLanguage(code)
                return code
            except ValueError:
                return code
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"
    
    async def translate(
        self,
        text: str,
        target_language: SupportedLanguage,
        source_language: Optional[SupportedLanguage] = None,
    ) -> TranslationResult:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language (auto-detect if None)
            
        Returns:
            TranslationResult
        """
        if not text or len(text.strip()) < 5:
            raise ValueError("Text too short for translation")
        
        if source_language is None:
            detected = await self.detect_language(text)
            source_language = SupportedLanguage(detected) if detected in [l.value for l in SupportedLanguage] else SupportedLanguage.SPANISH
        
        if source_language == target_language:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language.value,
                target_language=target_language.value,
                word_count_original=len(text.split()),
                word_count_translated=len(text.split()),
                confidence=1.0,
            )
        
        logger.info(f"Translating {len(text)} chars: {source_language.value} -> {target_language.value}")
        
        try:
            source_name = LANGUAGE_NAMES.get(source_language, source_language.value)
            target_name = LANGUAGE_NAMES.get(target_language, target_language.value)
            
            translated = await self.client.complete(
                prompt=self.TRANSLATION_PROMPT.format(
                    source_lang=source_name,
                    target_lang=target_name,
                    text=text,
                ),
                system_prompt=f"Eres un traductor experto de {source_name} a {target_name}. Traduce de manera natural y precisa.",
                max_tokens=len(text) * 2,
                temperature=0.3,
            )
            
            return TranslationResult(
                original_text=text,
                translated_text=translated.strip(),
                source_language=source_language.value,
                target_language=target_language.value,
                word_count_original=len(text.split()),
                word_count_translated=len(translated.split()),
                confidence=0.95,
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    async def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        target_language: SupportedLanguage,
        source_language: Optional[SupportedLanguage] = None,
    ) -> List[SegmentTranslation]:
        """
        Translate multiple segments in batch
        
        Args:
            segments: List of segments with id, start_time, end_time, text
            target_language: Target language
            source_language: Source language
            
        Returns:
            List of SegmentTranslation
        """
        if not segments:
            return []
        
        if source_language is None:
            first_text = segments[0].get("text", "")
            detected = await self.detect_language(first_text)
            source_language = SupportedLanguage(detected) if detected in [l.value for l in SupportedLanguage] else SupportedLanguage.SPANISH
        
        if source_language == target_language:
            return [
                SegmentTranslation(
                    segment_id=s.get("id", i),
                    start_time=s.get("start_time", 0),
                    end_time=s.get("end_time", 0),
                    original_text=s.get("text", ""),
                    translated_text=s.get("text", ""),
                )
                for i, s in enumerate(segments)
            ]
        
        logger.info(f"Batch translating {len(segments)} segments")
        
        segments_data = [
            {"id": s.get("id", i), "text": s.get("text", "")}
            for i, s in enumerate(segments)
        ]
        
        try:
            source_name = LANGUAGE_NAMES.get(source_language, source_language.value)
            target_name = LANGUAGE_NAMES.get(target_language, target_language.value)
            
            response = await self.client.complete(
                prompt=self.BATCH_TRANSLATION_PROMPT.format(
                    source_lang=source_name,
                    target_lang=target_name,
                    segments=json.dumps(segments_data, ensure_ascii=False),
                ),
                system_prompt=f"Eres un traductor experto. Traduce cada segmento manteniendo el contexto general.",
                max_tokens=sum(len(s.get("text", "")) for s in segments) * 2,
                temperature=0.3,
            )
            
            data = self._parse_json(response)
            translations_map = {
                t["id"]: t["text"]
                for t in data.get("translations", [])
            }
            
            results = []
            for i, s in enumerate(segments):
                seg_id = s.get("id", i)
                results.append(SegmentTranslation(
                    segment_id=seg_id,
                    start_time=s.get("start_time", 0),
                    end_time=s.get("end_time", 0),
                    original_text=s.get("text", ""),
                    translated_text=translations_map.get(seg_id, s.get("text", "")),
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return [
                SegmentTranslation(
                    segment_id=s.get("id", i),
                    start_time=s.get("start_time", 0),
                    end_time=s.get("end_time", 0),
                    original_text=s.get("text", ""),
                    translated_text=s.get("text", ""),
                )
                for i, s in enumerate(segments)
            ]
    
    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response"""
        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {"translations": []}
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [
            {"code": lang.value, "name": LANGUAGE_NAMES[lang]}
            for lang in SupportedLanguage
        ]


_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """Get translation service singleton"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service












