"""
Multi-Language Support System
=============================

This module provides multi-language support for content generation,
including language detection, translation, and localized content creation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LanguageConfig:
    """Language configuration"""
    code: str
    name: str
    native_name: str
    prompt_templates: Dict[str, str]
    cultural_adaptations: Dict[str, Any]
    writing_style: Dict[str, Any]

class MultiLanguageManager:
    """Manager for multi-language content generation"""
    
    def __init__(self):
        self.supported_languages: Dict[str, LanguageConfig] = {}
        self._initialize_languages()
    
    def _initialize_languages(self):
        """Initialize supported languages"""
        
        # English
        english_config = LanguageConfig(
            code="en",
            name="English",
            native_name="English",
            prompt_templates={
                "blog_post": "Write a comprehensive blog post about {topic} in English...",
                "tutorial": "Create a detailed tutorial about {topic} in English...",
                "product_description": "Write a product description for {product} in English...",
                "social_media": "Create an engaging social media post about {topic} in English..."
            },
            cultural_adaptations={
                "date_format": "%B %d, %Y",
                "number_format": "1,234.56",
                "currency": "USD",
                "greeting_style": "direct",
                "formality_level": "professional"
            },
            writing_style={
                "sentence_structure": "subject-verb-object",
                "paragraph_length": "medium",
                "tone": "conversational",
                "punctuation_style": "standard"
            }
        )
        
        # Spanish
        spanish_config = LanguageConfig(
            code="es",
            name="Spanish",
            native_name="Español",
            prompt_templates={
                "blog_post": "Escribe un artículo completo sobre {topic} en español...",
                "tutorial": "Crea un tutorial detallado sobre {topic} en español...",
                "product_description": "Escribe una descripción de producto para {product} en español...",
                "social_media": "Crea una publicación atractiva en redes sociales sobre {topic} en español..."
            },
            cultural_adaptations={
                "date_format": "%d de %B de %Y",
                "number_format": "1.234,56",
                "currency": "EUR",
                "greeting_style": "warm",
                "formality_level": "friendly"
            },
            writing_style={
                "sentence_structure": "flexible",
                "paragraph_length": "medium",
                "tone": "warm",
                "punctuation_style": "spanish"
            }
        )
        
        # French
        french_config = LanguageConfig(
            code="fr",
            name="French",
            native_name="Français",
            prompt_templates={
                "blog_post": "Écrivez un article de blog complet sur {topic} en français...",
                "tutorial": "Créez un tutoriel détaillé sur {topic} en français...",
                "product_description": "Écrivez une description de produit pour {product} en français...",
                "social_media": "Créez une publication engageante sur les réseaux sociaux sur {topic} en français..."
            },
            cultural_adaptations={
                "date_format": "%d %B %Y",
                "number_format": "1 234,56",
                "currency": "EUR",
                "greeting_style": "formal",
                "formality_level": "polite"
            },
            writing_style={
                "sentence_structure": "complex",
                "paragraph_length": "long",
                "tone": "elegant",
                "punctuation_style": "french"
            }
        )
        
        # Portuguese
        portuguese_config = LanguageConfig(
            code="pt",
            name="Portuguese",
            native_name="Português",
            prompt_templates={
                "blog_post": "Escreva um artigo completo sobre {topic} em português...",
                "tutorial": "Crie um tutorial detalhado sobre {topic} em português...",
                "product_description": "Escreva uma descrição de produto para {product} em português...",
                "social_media": "Crie uma publicação envolvente nas redes sociais sobre {topic} em português..."
            },
            cultural_adaptations={
                "date_format": "%d de %B de %Y",
                "number_format": "1.234,56",
                "currency": "BRL",
                "greeting_style": "warm",
                "formality_level": "friendly"
            },
            writing_style={
                "sentence_structure": "flexible",
                "paragraph_length": "medium",
                "tone": "warm",
                "punctuation_style": "portuguese"
            }
        )
        
        # German
        german_config = LanguageConfig(
            code="de",
            name="German",
            native_name="Deutsch",
            prompt_templates={
                "blog_post": "Schreiben Sie einen umfassenden Blog-Artikel über {topic} auf Deutsch...",
                "tutorial": "Erstellen Sie ein detailliertes Tutorial über {topic} auf Deutsch...",
                "product_description": "Schreiben Sie eine Produktbeschreibung für {product} auf Deutsch...",
                "social_media": "Erstellen Sie einen ansprechenden Social-Media-Post über {topic} auf Deutsch..."
            },
            cultural_adaptations={
                "date_format": "%d. %B %Y",
                "number_format": "1.234,56",
                "currency": "EUR",
                "greeting_style": "formal",
                "formality_level": "professional"
            },
            writing_style={
                "sentence_structure": "complex",
                "paragraph_length": "long",
                "tone": "precise",
                "punctuation_style": "german"
            }
        )
        
        # Italian
        italian_config = LanguageConfig(
            code="it",
            name="Italian",
            native_name="Italiano",
            prompt_templates={
                "blog_post": "Scrivi un articolo completo su {topic} in italiano...",
                "tutorial": "Crea un tutorial dettagliato su {topic} in italiano...",
                "product_description": "Scrivi una descrizione del prodotto per {product} in italiano...",
                "social_media": "Crea un post coinvolgente sui social media su {topic} in italiano..."
            },
            cultural_adaptations={
                "date_format": "%d %B %Y",
                "number_format": "1.234,56",
                "currency": "EUR",
                "greeting_style": "warm",
                "formality_level": "friendly"
            },
            writing_style={
                "sentence_structure": "flexible",
                "paragraph_length": "medium",
                "tone": "passionate",
                "punctuation_style": "italian"
            }
        )
        
        # Add languages to manager
        self.supported_languages = {
            "en": english_config,
            "es": spanish_config,
            "fr": french_config,
            "pt": portuguese_config,
            "de": german_config,
            "it": italian_config
        }
        
        logger.info(f"Initialized {len(self.supported_languages)} languages")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.supported_languages.keys())
    
    def get_language_config(self, language_code: str) -> Optional[LanguageConfig]:
        """Get language configuration"""
        return self.supported_languages.get(language_code.lower())
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        try:
            text_lower = text.lower()
            
            # Language-specific word patterns
            language_patterns = {
                "es": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son"],
                "fr": ["le", "la", "de", "et", "à", "un", "il", "que", "ne", "se", "ce", "pas", "tout", "plus", "par", "pour"],
                "pt": ["o", "a", "de", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", "na", "por", "mais"],
                "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "ist", "im", "dem", "nicht"],
                "it": ["il", "la", "di", "e", "a", "da", "in", "un", "per", "con", "su", "al", "del", "le", "si", "una", "anche", "se", "ma"],
                "en": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]
            }
            
            scores = {}
            for lang, patterns in language_patterns.items():
                score = sum(1 for pattern in patterns if pattern in text_lower)
                scores[lang] = score
            
            # Return language with highest score
            if scores:
                detected_lang = max(scores, key=scores.get)
                if scores[detected_lang] > 0:
                    return detected_lang
            
            # Default to English if no clear match
            return "en"
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "en"
    
    def adapt_prompt_for_language(
        self,
        base_prompt: str,
        language_code: str,
        content_type: str = "blog_post"
    ) -> str:
        """Adapt a prompt for a specific language"""
        try:
            lang_config = self.get_language_config(language_code)
            if not lang_config:
                logger.warning(f"Language {language_code} not supported, using English")
                return base_prompt
            
            # Get language-specific template
            template = lang_config.prompt_templates.get(content_type, base_prompt)
            
            # Add language-specific instructions
            language_instructions = f"""
            
            IMPORTANT LANGUAGE REQUIREMENTS:
            - Write entirely in {lang_config.native_name} ({lang_config.name})
            - Use {lang_config.writing_style['tone']} tone
            - Follow {lang_config.cultural_adaptations['formality_level']} formality level
            - Use {lang_config.cultural_adaptations['date_format']} date format
            - Use {lang_config.cultural_adaptations['number_format']} number format
            - Ensure cultural appropriateness for {lang_config.name} speakers
            """
            
            return template + language_instructions
            
        except Exception as e:
            logger.error(f"Error adapting prompt for language {language_code}: {str(e)}")
            return base_prompt
    
    def localize_content_metadata(
        self,
        metadata: Dict[str, Any],
        language_code: str
    ) -> Dict[str, Any]:
        """Localize content metadata for a specific language"""
        try:
            lang_config = self.get_language_config(language_code)
            if not lang_config:
                return metadata
            
            localized_metadata = metadata.copy()
            
            # Add language-specific information
            localized_metadata.update({
                "language": {
                    "code": language_code,
                    "name": lang_config.name,
                    "native_name": lang_config.native_name
                },
                "cultural_adaptations": lang_config.cultural_adaptations,
                "writing_style": lang_config.writing_style,
                "localized_at": datetime.now().isoformat()
            })
            
            return localized_metadata
            
        except Exception as e:
            logger.error(f"Error localizing metadata: {str(e)}")
            return metadata
    
    def get_language_specific_keywords(
        self,
        base_keywords: List[str],
        language_code: str
    ) -> List[str]:
        """Get language-specific keywords"""
        try:
            lang_config = self.get_language_config(language_code)
            if not lang_config:
                return base_keywords
            
            # Simple keyword translation mapping
            keyword_translations = {
                "en": {
                    "blog": "blog", "tutorial": "tutorial", "guide": "guide",
                    "tips": "tips", "how to": "how to", "best": "best"
                },
                "es": {
                    "blog": "blog", "tutorial": "tutorial", "guide": "guía",
                    "tips": "consejos", "how to": "cómo", "best": "mejor"
                },
                "fr": {
                    "blog": "blog", "tutorial": "tutoriel", "guide": "guide",
                    "tips": "conseils", "how to": "comment", "best": "meilleur"
                },
                "pt": {
                    "blog": "blog", "tutorial": "tutorial", "guide": "guia",
                    "tips": "dicas", "how to": "como", "best": "melhor"
                },
                "de": {
                    "blog": "blog", "tutorial": "tutorial", "guide": "anleitung",
                    "tips": "tipps", "how to": "wie man", "best": "beste"
                },
                "it": {
                    "blog": "blog", "tutorial": "tutorial", "guide": "guida",
                    "tips": "consigli", "how to": "come", "best": "migliore"
                }
            }
            
            translations = keyword_translations.get(language_code, {})
            localized_keywords = []
            
            for keyword in base_keywords:
                localized_keyword = translations.get(keyword.lower(), keyword)
                localized_keywords.append(localized_keyword)
            
            return localized_keywords
            
        except Exception as e:
            logger.error(f"Error getting language-specific keywords: {str(e)}")
            return base_keywords
    
    def validate_language_support(self, language_code: str) -> bool:
        """Validate if a language is supported"""
        return language_code.lower() in self.supported_languages
    
    def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """Get comprehensive language information"""
        lang_config = self.get_language_config(language_code)
        if not lang_config:
            return {}
        
        return {
            "code": lang_config.code,
            "name": lang_config.name,
            "native_name": lang_config.native_name,
            "cultural_adaptations": lang_config.cultural_adaptations,
            "writing_style": lang_config.writing_style,
            "supported_content_types": list(lang_config.prompt_templates.keys())
        }

# Global multi-language manager
multilang_manager = MultiLanguageManager()

# Example usage
if __name__ == "__main__":
    async def test_multilang():
        print("🧪 Testing Multi-Language Support")
        print("=" * 40)
        
        # Test language detection
        test_texts = {
            "en": "This is a sample text in English for testing language detection.",
            "es": "Este es un texto de muestra en español para probar la detección de idioma.",
            "fr": "Ceci est un texte d'exemple en français pour tester la détection de langue.",
            "de": "Dies ist ein Beispieltext auf Deutsch zum Testen der Spracherkennung."
        }
        
        for expected_lang, text in test_texts.items():
            detected = multilang_manager.detect_language(text)
            print(f"Expected: {expected_lang}, Detected: {detected} - {'✅' if detected == expected_lang else '❌'}")
        
        # Test prompt adaptation
        base_prompt = "Write a blog post about artificial intelligence"
        for lang_code in ["es", "fr", "de"]:
            adapted = multilang_manager.adapt_prompt_for_language(base_prompt, lang_code)
            print(f"\n{lang_code.upper()} adapted prompt:")
            print(f"{adapted[:100]}...")
        
        # Test keyword localization
        keywords = ["blog", "tutorial", "guide", "tips"]
        for lang_code in ["es", "fr", "de"]:
            localized = multilang_manager.get_language_specific_keywords(keywords, lang_code)
            print(f"\n{lang_code.upper()} keywords: {localized}")
    
    asyncio.run(test_multilang())


