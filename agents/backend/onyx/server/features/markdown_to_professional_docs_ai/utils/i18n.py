"""Internationalization support"""
from typing import Dict, Any, Optional
from enum import Enum


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"


# Translation strings
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Document",
        "table": "Table",
        "chart": "Chart",
        "figure": "Figure",
        "page": "Page",
        "of": "of",
        "generated": "Generated",
        "on": "on"
    },
    "es": {
        "title": "Documento",
        "table": "Tabla",
        "chart": "Gráfica",
        "figure": "Figura",
        "page": "Página",
        "of": "de",
        "generated": "Generado",
        "on": "el"
    },
    "fr": {
        "title": "Document",
        "table": "Tableau",
        "chart": "Graphique",
        "figure": "Figure",
        "page": "Page",
        "of": "de",
        "generated": "Généré",
        "on": "le"
    },
    "de": {
        "title": "Dokument",
        "table": "Tabelle",
        "chart": "Diagramm",
        "figure": "Abbildung",
        "page": "Seite",
        "of": "von",
        "generated": "Erstellt",
        "on": "am"
    },
    "pt": {
        "title": "Documento",
        "table": "Tabela",
        "chart": "Gráfico",
        "figure": "Figura",
        "page": "Página",
        "of": "de",
        "generated": "Gerado",
        "on": "em"
    },
    "it": {
        "title": "Documento",
        "table": "Tabella",
        "chart": "Grafico",
        "figure": "Figura",
        "page": "Pagina",
        "of": "di",
        "generated": "Generato",
        "on": "il"
    },
    "zh": {
        "title": "文档",
        "table": "表格",
        "chart": "图表",
        "figure": "图",
        "page": "页",
        "of": "的",
        "generated": "生成",
        "on": "于"
    },
    "ja": {
        "title": "文書",
        "table": "表",
        "chart": "グラフ",
        "figure": "図",
        "page": "ページ",
        "of": "の",
        "generated": "生成",
        "on": "に"
    },
    "ko": {
        "title": "문서",
        "table": "표",
        "chart": "차트",
        "figure": "그림",
        "page": "페이지",
        "of": "의",
        "generated": "생성",
        "on": "에"
    },
    "ru": {
        "title": "Документ",
        "table": "Таблица",
        "chart": "График",
        "figure": "Рисунок",
        "page": "Страница",
        "of": "из",
        "generated": "Создан",
        "on": "на"
    }
}


def get_translation(key: str, language: str = "en") -> str:
    """
    Get translation for key
    
    Args:
        key: Translation key
        language: Language code
        
    Returns:
        Translated string or key if not found
    """
    lang_dict = TRANSLATIONS.get(language, TRANSLATIONS["en"])
    return lang_dict.get(key, key)


def get_all_translations(language: str = "en") -> Dict[str, str]:
    """
    Get all translations for language
    
    Args:
        language: Language code
        
    Returns:
        Dictionary of translations
    """
    return TRANSLATIONS.get(language, TRANSLATIONS["en"])


def detect_language(text: str) -> str:
    """
    Simple language detection based on common words
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected language code
    """
    text_lower = text.lower()
    
    # Common words in different languages
    language_indicators = {
        "es": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para"],
        "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour", "dans", "ce", "son", "une", "sur", "avec"],
        "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "ist", "im", "dem", "nicht", "ein"],
        "pt": ["o", "de", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais"],
        "it": ["il", "di", "che", "e", "la", "a", "per", "è", "in", "un", "sono", "si", "una", "le", "con", "da", "come"],
        "zh": ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说"],
        "ja": ["の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も", "する", "から", "な"],
        "ko": ["이", "가", "을", "를", "에", "의", "와", "과", "도", "로", "으로", "에서", "에게", "한테", "께", "더", "많이"],
        "ru": ["в", "и", "не", "что", "он", "на", "я", "со", "как", "а", "тот", "это", "она", "этот", "который", "мы", "от", "вы"]
    }
    
    # Count matches
    scores = {}
    for lang, words in language_indicators.items():
        score = sum(1 for word in words if word in text_lower)
        scores[lang] = score
    
    # Return language with highest score, default to English
    if scores:
        detected = max(scores.items(), key=lambda x: x[1])
        if detected[1] > 0:
            return detected[0]
    
    return "en"

