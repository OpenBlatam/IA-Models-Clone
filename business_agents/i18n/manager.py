"""
I18n Manager
============

Internationalization management and translation handling.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid
from pathlib import Path

from .types import (
    Language, Locale, TranslationKey, TranslationValue, TranslationContext,
    TranslationNamespace, TranslationStats, PluralRule, DateFormat, NumberFormat
)

logger = logging.getLogger(__name__)

class TranslationManager:
    """Translation management system."""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, TranslationValue]] = {}  # namespace -> key -> locale -> value
        self.keys: Dict[str, TranslationKey] = {}  # full_key -> TranslationKey
        self.namespaces: Dict[str, TranslationNamespace] = {}
        self.plural_rules: Dict[Language, PluralRule] = {}
        self.date_formats: Dict[Locale, DateFormat] = {}
        self.number_formats: Dict[Locale, NumberFormat] = {}
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the translation manager."""
        try:
            # Initialize default namespace
            await self.create_namespace("default", "Default translations", Language.ENGLISH)
            
            # Load plural rules
            await self._load_plural_rules()
            
            # Load date and number formats
            await self._load_locale_formats()
            
            logger.info("Translation manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation manager: {str(e)}")
            raise
    
    async def create_namespace(
        self, 
        name: str, 
        description: str, 
        default_language: Language = Language.ENGLISH,
        supported_languages: List[Language] = None
    ) -> str:
        """Create a new translation namespace."""
        try:
            namespace = TranslationNamespace(
                name=name,
                description=description,
                default_language=default_language,
                supported_languages=supported_languages or [default_language]
            )
            
            async with self._lock:
                self.namespaces[name] = namespace
                if name not in self.translations:
                    self.translations[name] = {}
            
            logger.info(f"Created translation namespace: {name}")
            return name
            
        except Exception as e:
            logger.error(f"Failed to create namespace: {str(e)}")
            raise
    
    async def add_translation_key(
        self,
        key: str,
        namespace: str = "default",
        context: Optional[str] = None,
        description: Optional[str] = None,
        plural_forms: Optional[List[str]] = None,
        tags: List[str] = None
    ) -> str:
        """Add a new translation key."""
        try:
            translation_key = TranslationKey(
                key=key,
                namespace=namespace,
                context=context,
                description=description,
                plural_forms=plural_forms,
                tags=tags or []
            )
            
            full_key = translation_key.get_full_key()
            
            async with self._lock:
                self.keys[full_key] = translation_key
                if namespace not in self.translations:
                    self.translations[namespace] = {}
                if full_key not in self.translations[namespace]:
                    self.translations[namespace][full_key] = {}
            
            logger.info(f"Added translation key: {full_key}")
            return full_key
            
        except Exception as e:
            logger.error(f"Failed to add translation key: {str(e)}")
            raise
    
    async def set_translation(
        self,
        key: str,
        locale: Locale,
        value: str,
        namespace: str = "default",
        context: Optional[str] = None,
        plural_form: Optional[str] = None,
        is_approved: bool = False,
        translator: Optional[str] = None
    ) -> bool:
        """Set a translation value."""
        try:
            full_key = self._get_full_key(key, namespace, context)
            
            translation_value = TranslationValue(
                key=full_key,
                locale=locale,
                value=value,
                plural_form=plural_form,
                is_approved=is_approved,
                translator=translator
            )
            
            async with self._lock:
                if namespace not in self.translations:
                    self.translations[namespace] = {}
                if full_key not in self.translations[namespace]:
                    self.translations[namespace][full_key] = {}
                
                self.translations[namespace][full_key][str(locale)] = translation_value
            
            logger.debug(f"Set translation: {full_key} [{locale}] = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set translation: {str(e)}")
            return False
    
    async def get_translation(
        self,
        key: str,
        locale: Locale,
        namespace: str = "default",
        context: Optional[str] = None,
        fallback_locales: List[Locale] = None,
        variables: Dict[str, Any] = None,
        plural_count: Optional[int] = None
    ) -> Optional[str]:
        """Get a translation value."""
        try:
            full_key = self._get_full_key(key, namespace, context)
            
            # Try to get translation for the requested locale
            translation = await self._get_translation_for_locale(full_key, locale, namespace, plural_count)
            if translation:
                return await self._interpolate_variables(translation, variables or {})
            
            # Try fallback locales
            if fallback_locales:
                for fallback_locale in fallback_locales:
                    translation = await self._get_translation_for_locale(full_key, fallback_locale, namespace, plural_count)
                    if translation:
                        return await self._interpolate_variables(translation, variables or {})
            
            # Try default locale for namespace
            if namespace in self.namespaces:
                default_locale = Locale(self.namespaces[namespace].default_language)
                translation = await self._get_translation_for_locale(full_key, default_locale, namespace, plural_count)
                if translation:
                    return await self._interpolate_variables(translation, variables or {})
            
            # Return the key if no translation found
            logger.warning(f"Translation not found: {full_key} [{locale}]")
            return full_key
            
        except Exception as e:
            logger.error(f"Failed to get translation: {str(e)}")
            return key
    
    async def _get_translation_for_locale(
        self, 
        full_key: str, 
        locale: Locale, 
        namespace: str,
        plural_count: Optional[int] = None
    ) -> Optional[str]:
        """Get translation for specific locale."""
        try:
            async with self._lock:
                if (namespace in self.translations and 
                    full_key in self.translations[namespace] and 
                    str(locale) in self.translations[namespace][full_key]):
                    
                    translation_value = self.translations[namespace][full_key][str(locale)]
                    
                    # Handle plural forms
                    if plural_count is not None and translation_value.plural_form:
                        # Get the appropriate plural form
                        plural_form = await self._get_plural_form(locale.language, plural_count)
                        if plural_form:
                            # Look for translation with specific plural form
                            for trans in self.translations[namespace][full_key].values():
                                if (trans.locale == locale and 
                                    trans.plural_form == plural_form):
                                    return trans.value
                    
                    return translation_value.value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get translation for locale: {str(e)}")
            return None
    
    async def _get_plural_form(self, language: Language, count: int) -> Optional[str]:
        """Get plural form for a given count."""
        try:
            if language not in self.plural_rules:
                return None
            
            rule = self.plural_rules[language]
            # This is a simplified implementation
            # In a real system, you'd use proper plural rule evaluation
            
            if count == 0:
                return "zero"
            elif count == 1:
                return "one"
            elif count == 2:
                return "two"
            elif count < 5:
                return "few"
            else:
                return "many"
                
        except Exception as e:
            logger.error(f"Failed to get plural form: {str(e)}")
            return None
    
    async def _interpolate_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Interpolate variables in translation text."""
        try:
            if not variables:
                return text
            
            # Simple variable interpolation: {variable_name}
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                text = text.replace(placeholder, str(value))
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to interpolate variables: {str(e)}")
            return text
    
    def _get_full_key(self, key: str, namespace: str, context: Optional[str]) -> str:
        """Get full translation key."""
        parts = [namespace, key]
        if context:
            parts.append(context)
        return ".".join(parts)
    
    async def get_translation_stats(self, namespace: str = None) -> TranslationStats:
        """Get translation statistics."""
        try:
            stats = TranslationStats()
            
            async with self._lock:
                namespaces_to_check = [namespace] if namespace else list(self.namespaces.keys())
                
                for ns in namespaces_to_check:
                    if ns not in self.translations:
                        continue
                    
                    ns_stats = {
                        "total_keys": 0,
                        "translated_keys": 0,
                        "approved_keys": 0
                    }
                    
                    for full_key, translations in self.translations[ns].items():
                        ns_stats["total_keys"] += 1
                        
                        if translations:
                            ns_stats["translated_keys"] += 1
                            
                            # Check if any translation is approved
                            if any(t.is_approved for t in translations.values()):
                                ns_stats["approved_keys"] += 1
                    
                    stats.namespace_stats[ns] = ns_stats
                    stats.total_keys += ns_stats["total_keys"]
                    stats.translated_keys += ns_stats["translated_keys"]
                    stats.approved_keys += ns_stats["approved_keys"]
                
                if stats.total_keys > 0:
                    stats.completion_percentage = (stats.translated_keys / stats.total_keys) * 100
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get translation stats: {str(e)}")
            return TranslationStats()
    
    async def export_translations(
        self, 
        namespace: str, 
        locale: Locale, 
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export translations in specified format."""
        try:
            async with self._lock:
                if namespace not in self.translations:
                    return {}
                
                translations = {}
                for full_key, locale_translations in self.translations[namespace].items():
                    if str(locale) in locale_translations:
                        translation_value = locale_translations[str(locale)]
                        translations[full_key] = translation_value.value
                
                if format == "json":
                    return translations
                elif format == "po":
                    return await self._export_to_po(translations, locale)
                else:
                    return translations
                    
        except Exception as e:
            logger.error(f"Failed to export translations: {str(e)}")
            return {}
    
    async def import_translations(
        self, 
        namespace: str, 
        locale: Locale, 
        translations: Dict[str, str],
        overwrite: bool = False
    ) -> Dict[str, int]:
        """Import translations."""
        try:
            results = {
                "imported": 0,
                "updated": 0,
                "skipped": 0,
                "errors": 0
            }
            
            async with self._lock:
                if namespace not in self.translations:
                    self.translations[namespace] = {}
                
                for key, value in translations.items():
                    try:
                        full_key = key
                        if namespace not in key:
                            full_key = f"{namespace}.{key}"
                        
                        if full_key not in self.translations[namespace]:
                            self.translations[namespace][full_key] = {}
                        
                        locale_str = str(locale)
                        if locale_str in self.translations[namespace][full_key] and not overwrite:
                            results["skipped"] += 1
                            continue
                        
                        translation_value = TranslationValue(
                            key=full_key,
                            locale=locale,
                            value=value
                        )
                        
                        self.translations[namespace][full_key][locale_str] = translation_value
                        
                        if locale_str in self.translations[namespace][full_key]:
                            results["updated"] += 1
                        else:
                            results["imported"] += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to import translation {key}: {str(e)}")
                        results["errors"] += 1
            
            logger.info(f"Imported translations: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to import translations: {str(e)}")
            return {"imported": 0, "updated": 0, "skipped": 0, "errors": 1}
    
    async def _load_plural_rules(self):
        """Load plural rules for different languages."""
        try:
            # Simplified plural rules - in a real system, you'd load from CLDR data
            self.plural_rules = {
                Language.ENGLISH: PluralRule(Language.ENGLISH, "cardinal", "n != 1"),
                Language.SPANISH: PluralRule(Language.SPANISH, "cardinal", "n != 1"),
                Language.FRENCH: PluralRule(Language.FRENCH, "cardinal", "n > 1"),
                Language.GERMAN: PluralRule(Language.GERMAN, "cardinal", "n != 1"),
                Language.RUSSIAN: PluralRule(Language.RUSSIAN, "cardinal", "n % 10 == 1 && n % 100 != 11 ? 0 : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 10 || n % 100 >= 20) ? 1 : 2"),
            }
            
        except Exception as e:
            logger.error(f"Failed to load plural rules: {str(e)}")
    
    async def _load_locale_formats(self):
        """Load date and number formats for different locales."""
        try:
            # Simplified formats - in a real system, you'd load from CLDR data
            self.date_formats = {
                Locale(Language.ENGLISH): DateFormat(Locale(Language.ENGLISH)),
                Locale(Language.SPANISH): DateFormat(Locale(Language.SPANISH), "%d/%m/%y", "%d %b %Y"),
                Locale(Language.FRENCH): DateFormat(Locale(Language.FRENCH), "%d/%m/%y", "%d %b %Y"),
                Locale(Language.GERMAN): DateFormat(Locale(Language.GERMAN), "%d.%m.%y", "%d. %b %Y"),
            }
            
            self.number_formats = {
                Locale(Language.ENGLISH): NumberFormat(Locale(Language.ENGLISH)),
                Locale(Language.SPANISH): NumberFormat(Locale(Language.SPANISH), ",", "."),
                Locale(Language.FRENCH): NumberFormat(Locale(Language.FRENCH), ",", " "),
                Locale(Language.GERMAN): NumberFormat(Locale(Language.GERMAN), ",", "."),
            }
            
        except Exception as e:
            logger.error(f"Failed to load locale formats: {str(e)}")
    
    async def _export_to_po(self, translations: Dict[str, str], locale: Locale) -> str:
        """Export translations to PO format."""
        try:
            po_content = f'msgid ""\n'
            po_content += f'msgstr ""\n'
            po_content += f'"Content-Type: text/plain; charset=UTF-8\\n"\n'
            po_content += f'"Language: {locale.language.value}\\n"\n'
            po_content += f'"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
            
            for key, value in translations.items():
                po_content += f'msgid "{key}"\n'
                po_content += f'msgstr "{value}"\n\n'
            
            return po_content
            
        except Exception as e:
            logger.error(f"Failed to export to PO format: {str(e)}")
            return ""

class I18nManager:
    """Main internationalization manager."""
    
    def __init__(self):
        self.translation_manager = TranslationManager()
        self.current_locale: Optional[Locale] = None
        self.fallback_locales: List[Locale] = []
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the I18n manager."""
        try:
            await self.translation_manager.initialize()
            
            # Set default locale
            self.current_locale = Locale(Language.ENGLISH)
            self.fallback_locales = [Locale(Language.ENGLISH)]
            
            logger.info("I18n manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize I18n manager: {str(e)}")
            raise
    
    async def set_locale(self, locale: Locale, fallback_locales: List[Locale] = None):
        """Set current locale and fallback locales."""
        try:
            async with self._lock:
                self.current_locale = locale
                self.fallback_locales = fallback_locales or [Locale(Language.ENGLISH)]
            
            logger.info(f"Set locale to: {locale}")
            
        except Exception as e:
            logger.error(f"Failed to set locale: {str(e)}")
    
    async def translate(
        self,
        key: str,
        namespace: str = "default",
        context: Optional[str] = None,
        variables: Dict[str, Any] = None,
        plural_count: Optional[int] = None,
        locale: Optional[Locale] = None
    ) -> str:
        """Translate a key to the current locale."""
        try:
            target_locale = locale or self.current_locale
            if not target_locale:
                return key
            
            return await self.translation_manager.get_translation(
                key=key,
                locale=target_locale,
                namespace=namespace,
                context=context,
                fallback_locales=self.fallback_locales,
                variables=variables,
                plural_count=plural_count
            )
            
        except Exception as e:
            logger.error(f"Failed to translate key {key}: {str(e)}")
            return key
    
    async def format_date(
        self, 
        date: datetime, 
        format_type: str = "medium",
        locale: Optional[Locale] = None
    ) -> str:
        """Format date according to locale."""
        try:
            target_locale = locale or self.current_locale
            if not target_locale:
                return date.strftime("%Y-%m-%d")
            
            date_format = self.translation_manager.date_formats.get(target_locale)
            if not date_format:
                return date.strftime("%Y-%m-%d")
            
            format_string = getattr(date_format, f"{format_type}_date", date_format.medium_date)
            return date.strftime(format_string)
            
        except Exception as e:
            logger.error(f"Failed to format date: {str(e)}")
            return str(date)
    
    async def format_number(
        self, 
        number: Union[int, float], 
        locale: Optional[Locale] = None
    ) -> str:
        """Format number according to locale."""
        try:
            target_locale = locale or self.current_locale
            if not target_locale:
                return str(number)
            
            number_format = self.translation_manager.number_formats.get(target_locale)
            if not number_format:
                return str(number)
            
            # Simple number formatting - in a real system, you'd use proper locale formatting
            formatted = f"{number:,}".replace(",", number_format.thousands_separator)
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format number: {str(e)}")
            return str(number)
    
    async def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages."""
        return list(Language)
    
    async def get_translation_stats(self, namespace: str = None) -> TranslationStats:
        """Get translation statistics."""
        return await self.translation_manager.get_translation_stats(namespace)

# Global I18n manager instance
i18n_manager = I18nManager()
