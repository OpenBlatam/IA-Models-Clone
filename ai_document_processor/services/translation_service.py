"""
Servicio de Traducción
======================

Servicio para traducir documentos entre diferentes idiomas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

# Importaciones de traducción
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from googletrans import Translator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Resultado de traducción"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    translation_method: str
    word_count: int
    character_count: int

@dataclass
class LanguageDetection:
    """Detección de idioma"""
    language: str
    confidence: float
    method: str

class TranslationService:
    """Servicio de traducción de documentos"""
    
    def __init__(self):
        self.openai_client = None
        self.google_translator = None
        self.supported_languages = {
            'es': 'Español',
            'en': 'English',
            'fr': 'Français',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português',
            'ru': 'Русский',
            'ja': '日本語',
            'ko': '한국어',
            'zh': '中文',
            'ar': 'العربية',
            'hi': 'हिन्दी'
        }
        
    async def initialize(self):
        """Inicializa el servicio de traducción"""
        logger.info("Inicializando servicio de traducción...")
        
        # Configurar OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai
            logger.info("✅ OpenAI configurado para traducción")
        
        # Configurar Google Translate
        if GOOGLE_TRANSLATE_AVAILABLE:
            try:
                self.google_translator = Translator()
                logger.info("✅ Google Translate configurado")
            except Exception as e:
                logger.warning(f"Error configurando Google Translate: {e}")
        
        logger.info("Servicio de traducción inicializado")
    
    def detect_language(self, text: str) -> LanguageDetection:
        """Detecta el idioma del texto"""
        try:
            if self.google_translator:
                # Usar Google Translate
                detection = self.google_translator.detect(text[:5000])  # Limitar longitud
                return LanguageDetection(
                    language=detection.lang,
                    confidence=detection.confidence,
                    method="google_translate"
                )
            
            elif self.openai_client:
                # Usar OpenAI para detección
                prompt = f"""
                Detecta el idioma del siguiente texto y responde solo con el código ISO 639-1 del idioma (ej: es, en, fr, de, it, pt, ru, ja, ko, zh, ar, hi).
                
                Texto: {text[:1000]}
                
                Idioma:
                """
                
                response = asyncio.run(self.openai_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                ))
                
                detected_lang = response.choices[0].message.content.strip().lower()
                
                return LanguageDetection(
                    language=detected_lang,
                    confidence=0.8,
                    method="openai"
                )
            
            else:
                # Detección básica por palabras comunes
                text_lower = text.lower()
                
                # Palabras comunes en diferentes idiomas
                language_indicators = {
                    'es': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'más', 'muy', 'ya', 'todo', 'esta', 'ser', 'tiene', 'también', 'fue', 'había', 'me', 'si', 'sin', 'sobre', 'este', 'entre', 'cuando', 'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'le', 'ha', 'me', 'sus', 'ya', 'están'],
                    'en': ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'],
                    'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son', 'que', 'je', 'il', 'à', 'ce', 'ne', 'pas', 'son', 'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son', 'que', 'je', 'il', 'à', 'ce', 'ne', 'pas', 'son'],
                    'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'in', 'kann', 'da', 'gegen', 'vom', 'können', 'schon', 'wenn', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'in', 'kann', 'da', 'gegen', 'vom', 'können', 'schon', 'wenn']
                }
                
                scores = {}
                for lang, words in language_indicators.items():
                    score = sum(1 for word in words if word in text_lower)
                    scores[lang] = score
                
                detected_lang = max(scores, key=scores.get) if scores else 'es'
                confidence = scores[detected_lang] / len(text.split()) if text.split() else 0.1
                
                return LanguageDetection(
                    language=detected_lang,
                    confidence=min(confidence, 1.0),
                    method="basic_pattern"
                )
                
        except Exception as e:
            logger.error(f"Error detectando idioma: {e}")
            return LanguageDetection(
                language='es',
                confidence=0.1,
                method="fallback"
            )
    
    async def translate_text(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> TranslationResult:
        """Traduce texto a un idioma objetivo"""
        try:
            # Detectar idioma fuente si no se proporciona
            if not source_language:
                detection = self.detect_language(text)
                source_language = detection.language
            
            # Validar idiomas
            if source_language not in self.supported_languages:
                raise ValueError(f"Idioma fuente no soportado: {source_language}")
            if target_language not in self.supported_languages:
                raise ValueError(f"Idioma objetivo no soportado: {target_language}")
            
            # Si el idioma fuente y objetivo son iguales, no traducir
            if source_language == target_language:
                return TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=1.0,
                    translation_method="no_translation_needed",
                    word_count=len(text.split()),
                    character_count=len(text)
                )
            
            translated_text = ""
            confidence = 0.0
            method = ""
            
            if self.openai_client:
                # Usar OpenAI para traducción
                source_name = self.supported_languages[source_language]
                target_name = self.supported_languages[target_language]
                
                prompt = f"""
                Traduce el siguiente texto de {source_name} a {target_name}.
                Mantén el formato, estructura y tono del texto original.
                Si el texto contiene elementos técnicos o específicos, tradúcelos apropiadamente.
                
                Texto original:
                {text}
                
                Traducción:
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=len(text.split()) * 2,  # Estimar tokens necesarios
                    temperature=0.3
                )
                
                translated_text = response.choices[0].message.content.strip()
                confidence = 0.9
                method = "openai"
            
            elif self.google_translator:
                # Usar Google Translate
                try:
                    result = self.google_translator.translate(
                        text, 
                        src=source_language, 
                        dest=target_language
                    )
                    translated_text = result.text
                    confidence = 0.8
                    method = "google_translate"
                except Exception as e:
                    logger.warning(f"Error con Google Translate: {e}")
                    raise
            
            else:
                # Traducción básica (solo para demostración)
                translated_text = f"[TRADUCIDO A {target_language.upper()}] {text}"
                confidence = 0.1
                method = "basic_fallback"
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence=confidence,
                translation_method=method,
                word_count=len(translated_text.split()),
                character_count=len(translated_text)
            )
            
        except Exception as e:
            logger.error(f"Error traduciendo texto: {e}")
            raise
    
    async def translate_document(
        self, 
        content: str, 
        target_language: str, 
        source_language: Optional[str] = None,
        preserve_formatting: bool = True
    ) -> TranslationResult:
        """Traduce un documento completo preservando formato"""
        try:
            if preserve_formatting:
                # Dividir en párrafos y traducir cada uno
                paragraphs = content.split('\n\n')
                translated_paragraphs = []
                
                for paragraph in paragraphs:
                    if paragraph.strip():
                        result = await self.translate_text(paragraph, target_language, source_language)
                        translated_paragraphs.append(result.translated_text)
                    else:
                        translated_paragraphs.append(paragraph)
                
                translated_content = '\n\n'.join(translated_paragraphs)
                
                # Detectar idioma fuente
                if not source_language:
                    detection = self.detect_language(content)
                    source_language = detection.language
                
                return TranslationResult(
                    original_text=content,
                    translated_text=translated_content,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.8,
                    translation_method="document_translation",
                    word_count=len(translated_content.split()),
                    character_count=len(translated_content)
                )
            else:
                # Traducir como texto simple
                return await self.translate_text(content, target_language, source_language)
                
        except Exception as e:
            logger.error(f"Error traduciendo documento: {e}")
            raise
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Obtiene la lista de idiomas soportados"""
        return self.supported_languages.copy()
    
    def is_language_supported(self, language_code: str) -> bool:
        """Verifica si un idioma está soportado"""
        return language_code in self.supported_languages
    
    async def translate_batch(
        self, 
        texts: List[str], 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> List[TranslationResult]:
        """Traduce múltiples textos en lote"""
        try:
            tasks = []
            for text in texts:
                task = self.translate_text(text, target_language, source_language)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados y manejar errores
            translation_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error traduciendo texto {i}: {result}")
                    # Crear resultado de error
                    error_result = TranslationResult(
                        original_text=texts[i],
                        translated_text=f"[ERROR DE TRADUCCIÓN] {texts[i]}",
                        source_language=source_language or "unknown",
                        target_language=target_language,
                        confidence=0.0,
                        translation_method="error",
                        word_count=len(texts[i].split()),
                        character_count=len(texts[i])
                    )
                    translation_results.append(error_result)
                else:
                    translation_results.append(result)
            
            return translation_results
            
        except Exception as e:
            logger.error(f"Error en traducción en lote: {e}")
            raise
    
    def get_translation_quality_score(self, original: str, translated: str) -> float:
        """Calcula un score de calidad de la traducción"""
        try:
            # Métricas básicas de calidad
            original_words = len(original.split())
            translated_words = len(translated.split())
            
            # Ratio de palabras (debería ser similar)
            word_ratio = min(original_words, translated_words) / max(original_words, translated_words)
            
            # Longitud de caracteres
            char_ratio = min(len(original), len(translated)) / max(len(original), len(translated))
            
            # Score combinado (0-1)
            quality_score = (word_ratio * 0.6) + (char_ratio * 0.4)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculando calidad de traducción: {e}")
            return 0.5


