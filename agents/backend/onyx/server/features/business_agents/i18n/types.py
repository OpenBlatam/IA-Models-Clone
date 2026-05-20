"""
I18n Types and Definitions
==========================

Type definitions for internationalization components.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    POLISH = "pl"
    CZECH = "cs"
    HUNGARIAN = "hu"
    ROMANIAN = "ro"
    BULGARIAN = "bg"
    CROATIAN = "hr"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    ESTONIAN = "et"
    LATVIAN = "lv"
    LITHUANIAN = "lt"
    GREEK = "el"
    TURKISH = "tr"
    HEBREW = "he"
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    FILIPINO = "tl"
    UKRAINIAN = "uk"
    BELARUSIAN = "be"
    SERBIAN = "sr"
    MACEDONIAN = "mk"
    ALBANIAN = "sq"
    BOSNIAN = "bs"
    MONTENEGRIN = "me"

@dataclass
class Locale:
    """Locale information."""
    language: Language
    country: Optional[str] = None
    variant: Optional[str] = None
    script: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of locale."""
        parts = [self.language.value]
        if self.script:
            parts.append(self.script)
        if self.country:
            parts.append(self.country)
        if self.variant:
            parts.append(self.variant)
        return "-".join(parts)
    
    @classmethod
    def from_string(cls, locale_str: str) -> 'Locale':
        """Create locale from string."""
        parts = locale_str.split("-")
        language = Language(parts[0])
        
        country = None
        variant = None
        script = None
        
        if len(parts) > 1:
            if len(parts[1]) == 4:  # Script code
                script = parts[1]
                if len(parts) > 2:
                    country = parts[2]
                if len(parts) > 3:
                    variant = parts[3]
            elif len(parts[1]) == 2:  # Country code
                country = parts[1]
                if len(parts) > 2:
                    variant = parts[2]
        
        return cls(language=language, country=country, variant=variant, script=script)

@dataclass
class TranslationKey:
    """Translation key definition."""
    key: str
    namespace: str = "default"
    context: Optional[str] = None
    plural_forms: Optional[List[str]] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_full_key(self) -> str:
        """Get full translation key."""
        parts = [self.namespace, self.key]
        if self.context:
            parts.append(self.context)
        return ".".join(parts)

@dataclass
class TranslationValue:
    """Translation value definition."""
    key: str
    locale: Locale
    value: str
    plural_form: Optional[str] = None
    is_approved: bool = False
    translator: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranslationNamespace:
    """Translation namespace definition."""
    name: str
    description: str
    default_language: Language = Language.ENGLISH
    supported_languages: List[Language] = field(default_factory=lambda: [Language.ENGLISH])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranslationProject:
    """Translation project definition."""
    id: str
    name: str
    description: str
    source_language: Language = Language.ENGLISH
    target_languages: List[Language] = field(default_factory=list)
    namespaces: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranslationStats:
    """Translation statistics."""
    total_keys: int = 0
    translated_keys: int = 0
    approved_keys: int = 0
    completion_percentage: float = 0.0
    language_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    namespace_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TranslationContext:
    """Translation context information."""
    user_locale: Locale
    fallback_locales: List[Locale] = field(default_factory=list)
    namespace: str = "default"
    context: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    plural_count: Optional[int] = None
    gender: Optional[str] = None
    formality: Optional[str] = None  # formal, informal

@dataclass
class PluralRule:
    """Plural rule definition."""
    language: Language
    rule_type: str  # cardinal, ordinal
    rule_expression: str
    examples: List[str] = field(default_factory=list)

@dataclass
class DateFormat:
    """Date format definition."""
    locale: Locale
    short_date: str = "%m/%d/%y"
    medium_date: str = "%b %d, %Y"
    long_date: str = "%B %d, %Y"
    full_date: str = "%A, %B %d, %Y"
    short_time: str = "%H:%M"
    medium_time: str = "%H:%M:%S"
    long_time: str = "%H:%M:%S %Z"
    full_time: str = "%H:%M:%S %Z %z"
    short_datetime: str = "%m/%d/%y %H:%M"
    medium_datetime: str = "%b %d, %Y %H:%M:%S"
    long_datetime: str = "%B %d, %Y %H:%M:%S"
    full_datetime: str = "%A, %B %d, %Y %H:%M:%S %Z"

@dataclass
class NumberFormat:
    """Number format definition."""
    locale: Locale
    decimal_separator: str = "."
    thousands_separator: str = ","
    grouping: List[int] = field(default_factory=lambda: [3])
    currency_symbol: str = "$"
    currency_position: str = "before"  # before, after
    positive_sign: str = ""
    negative_sign: str = "-"
    percent_symbol: str = "%"
    scientific_notation: str = "E"

@dataclass
class TranslationTemplate:
    """Translation template definition."""
    id: str
    name: str
    description: str
    template: str
    variables: List[str] = field(default_factory=list)
    locale: Locale
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TranslationImport:
    """Translation import definition."""
    id: str
    source: str  # file, api, manual
    format: str  # json, po, xliff, csv
    namespace: str
    locale: Locale
    imported_keys: int = 0
    updated_keys: int = 0
    new_keys: int = 0
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TranslationExport:
    """Translation export definition."""
    id: str
    format: str  # json, po, xliff, csv
    namespace: str
    locales: List[Locale]
    exported_keys: int = 0
    file_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TranslationWorkflow:
    """Translation workflow definition."""
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    due_date: Optional[datetime] = None
    status: str = "draft"  # draft, in_progress, review, approved, completed
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TranslationQuality:
    """Translation quality metrics."""
    key: str
    locale: Locale
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approved: bool = False

@dataclass
class TranslationMemory:
    """Translation memory entry."""
    id: str
    source_text: str
    target_text: str
    source_locale: Locale
    target_locale: Locale
    domain: Optional[str] = None
    confidence: float = 1.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
