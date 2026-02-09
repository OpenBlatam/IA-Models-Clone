"""
Internationalization (i18n) System for Flux2 Clothing Changer
===============================================================

Multi-language support and localization.
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Translation:
    """Translation entry."""
    key: str
    language: str
    value: str
    context: Optional[str] = None


class I18nSystem:
    """Internationalization and localization system."""
    
    def __init__(
        self,
        default_language: str = "en",
        translations_dir: Path = Path("translations"),
    ):
        """
        Initialize i18n system.
        
        Args:
            default_language: Default language code
            translations_dir: Directory for translation files
        """
        self.default_language = default_language
        self.translations_dir = translations_dir
        self.translations_dir.mkdir(parents=True, exist_ok=True)
        
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = default_language
        self.supported_languages = [default_language]
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load translations from files."""
        if not self.translations_dir.exists():
            return
        
        for lang_file in self.translations_dir.glob("*.json"):
            language = lang_file.stem
            
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    self.translations[language] = json.load(f)
                
                if language not in self.supported_languages:
                    self.supported_languages.append(language)
                
                logger.info(f"Loaded translations for {language}")
            except Exception as e:
                logger.warning(f"Failed to load translations for {language}: {e}")
    
    def translate(
        self,
        key: str,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Translate a key.
        
        Args:
            key: Translation key
            language: Language code (None for current)
            **kwargs: Format parameters
            
        Returns:
            Translated string
        """
        lang = language or self.current_language
        
        # Get translation
        translation = self.translations.get(lang, {}).get(key)
        
        if not translation:
            # Fallback to default language
            if lang != self.default_language:
                translation = self.translations.get(self.default_language, {}).get(key)
            
            # Fallback to key itself
            if not translation:
                translation = key
        
        # Format if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except Exception:
                pass
        
        return translation
    
    def set_language(self, language: str) -> bool:
        """
        Set current language.
        
        Args:
            language: Language code
            
        Returns:
            True if language is supported
        """
        if language in self.supported_languages:
            self.current_language = language
            logger.info(f"Language set to {language}")
            return True
        else:
            logger.warning(f"Language {language} not supported")
            return False
    
    def add_translation(
        self,
        key: str,
        language: str,
        value: str,
        save: bool = True,
    ) -> None:
        """
        Add or update translation.
        
        Args:
            key: Translation key
            language: Language code
            value: Translation value
            save: Save to file
        """
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language][key] = value
        
        if language not in self.supported_languages:
            self.supported_languages.append(language)
        
        if save:
            self._save_translations(language)
    
    def _save_translations(self, language: str) -> None:
        """Save translations for a language."""
        if language not in self.translations:
            return
        
        lang_file = self.translations_dir / f"{language}.json"
        
        try:
            with open(lang_file, "w", encoding="utf-8") as f:
                json.dump(self.translations[language], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved translations for {language}")
        except Exception as e:
            logger.error(f"Failed to save translations: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def format_number(self, number: float, language: Optional[str] = None) -> str:
        """
        Format number according to language.
        
        Args:
            number: Number to format
            language: Language code
            
        Returns:
            Formatted number string
        """
        lang = language or self.current_language
        
        # Simple formatting (can be enhanced with locale)
        if lang in ["es", "fr", "de"]:
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{number:,.2f}"
    
    def format_date(
        self,
        timestamp: float,
        language: Optional[str] = None,
        format_type: str = "short",
    ) -> str:
        """
        Format date according to language.
        
        Args:
            timestamp: Unix timestamp
            language: Language code
            format_type: Format type (short, long, datetime)
            
        Returns:
            Formatted date string
        """
        lang = language or self.current_language
        dt = datetime.fromtimestamp(timestamp)
        
        # Simple formatting (can be enhanced with locale)
        if format_type == "short":
            return dt.strftime("%Y-%m-%d")
        elif format_type == "long":
            return dt.strftime("%B %d, %Y")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")


