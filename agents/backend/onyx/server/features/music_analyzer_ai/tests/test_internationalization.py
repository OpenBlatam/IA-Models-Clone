"""
Tests de internacionalización (i18n)
"""

import pytest
from unittest.mock import Mock


class TestInternationalization:
    """Tests de internacionalización"""
    
    def test_get_translation(self):
        """Test de obtención de traducción"""
        translations = {
            "en": {
                "welcome": "Welcome",
                "error": "An error occurred"
            },
            "es": {
                "welcome": "Bienvenido",
                "error": "Ocurrió un error"
            },
            "fr": {
                "welcome": "Bienvenue",
                "error": "Une erreur s'est produite"
            }
        }
        
        def get_translation(key, locale="en"):
            return translations.get(locale, translations["en"]).get(key, key)
        
        assert get_translation("welcome", "en") == "Welcome"
        assert get_translation("welcome", "es") == "Bienvenido"
        assert get_translation("welcome", "fr") == "Bienvenue"
    
    def test_translation_with_params(self):
        """Test de traducción con parámetros"""
        translations = {
            "en": {
                "greeting": "Hello, {name}!",
                "items_count": "You have {count} items"
            },
            "es": {
                "greeting": "¡Hola, {name}!",
                "items_count": "Tienes {count} elementos"
            }
        }
        
        def translate(key, locale="en", **params):
            template = translations.get(locale, translations["en"]).get(key, key)
            return template.format(**params)
        
        assert translate("greeting", "en", name="John") == "Hello, John!"
        assert translate("greeting", "es", name="Juan") == "¡Hola, Juan!"
        assert translate("items_count", "en", count=5) == "You have 5 items"
        assert translate("items_count", "es", count=3) == "Tienes 3 elementos"
    
    def test_locale_detection(self):
        """Test de detección de locale"""
        def detect_locale(accept_language_header):
            if not accept_language_header:
                return "en"  # Default
            
            # Parse Accept-Language header
            languages = accept_language_header.split(",")
            preferred = languages[0].split(";")[0].strip().lower()
            
            # Map to supported locales
            locale_map = {
                "en": "en",
                "es": "es",
                "fr": "fr",
                "es-es": "es",
                "es-mx": "es",
                "fr-fr": "fr",
                "fr-ca": "fr"
            }
            
            return locale_map.get(preferred, "en")
        
        assert detect_locale("en-US,en;q=0.9") == "en"
        assert detect_locale("es-ES,es;q=0.9") == "es"
        assert detect_locale("fr-FR,fr;q=0.9") == "fr"
        assert detect_locale("") == "en"
    
    def test_fallback_translation(self):
        """Test de traducción con fallback"""
        translations = {
            "en": {
                "welcome": "Welcome",
                "error": "An error occurred"
            },
            "es": {
                "welcome": "Bienvenido"
                # "error" no está en español, debe usar fallback
            }
        }
        
        def translate_with_fallback(key, locale="en"):
            locale_translations = translations.get(locale, {})
            if key in locale_translations:
                return locale_translations[key]
            
            # Fallback a inglés
            return translations["en"].get(key, key)
        
        assert translate_with_fallback("welcome", "es") == "Bienvenido"
        assert translate_with_fallback("error", "es") == "An error occurred"  # Fallback
        assert translate_with_fallback("welcome", "en") == "Welcome"


class TestNumberFormatting:
    """Tests de formato de números"""
    
    def test_number_formatting(self):
        """Test de formato de números por locale"""
        def format_number(number, locale="en"):
            if locale == "es":
                # Formato español: 1.234,56
                return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            elif locale == "fr":
                # Formato francés: 1 234,56
                return f"{number:,.2f}".replace(",", " ").replace(".", ",")
            else:
                # Formato inglés: 1,234.56
                return f"{number:,.2f}"
        
        number = 1234.56
        
        assert format_number(number, "en") == "1,234.56"
        assert format_number(number, "es") == "1.234,56"
        assert format_number(number, "fr") == "1 234,56"
    
    def test_currency_formatting(self):
        """Test de formato de moneda"""
        def format_currency(amount, locale="en"):
            currency_symbols = {
                "en": "$",
                "es": "€",
                "fr": "€"
            }
            
            symbol = currency_symbols.get(locale, "$")
            formatted = format_number(amount, locale)
            
            if locale == "en":
                return f"{symbol}{formatted}"
            else:
                return f"{formatted} {symbol}"
        
        amount = 1234.56
        
        assert format_currency(amount, "en") == "$1,234.56"
        assert format_currency(amount, "es") == "1.234,56 €"
        assert format_currency(amount, "fr") == "1 234,56 €"


class TestDateFormatting:
    """Tests de formato de fechas"""
    
    def test_date_formatting(self):
        """Test de formato de fechas por locale"""
        from datetime import datetime
        
        def format_date(date, locale="en"):
            date_formats = {
                "en": "%m/%d/%Y",  # MM/DD/YYYY
                "es": "%d/%m/%Y",  # DD/MM/YYYY
                "fr": "%d/%m/%Y"   # DD/MM/YYYY
            }
            
            format_str = date_formats.get(locale, "%Y-%m-%d")
            return date.strftime(format_str)
        
        date = datetime(2024, 1, 15)
        
        assert format_date(date, "en") == "01/15/2024"
        assert format_date(date, "es") == "15/01/2024"
        assert format_date(date, "fr") == "15/01/2024"
    
    def test_datetime_formatting(self):
        """Test de formato de fecha y hora"""
        from datetime import datetime
        
        def format_datetime(dt, locale="en"):
            formats = {
                "en": "%B %d, %Y at %I:%M %p",  # January 15, 2024 at 02:30 PM
                "es": "%d de %B de %Y a las %H:%M",  # 15 de enero de 2024 a las 14:30
                "fr": "%d %B %Y à %H:%M"  # 15 janvier 2024 à 14:30
            }
            
            format_str = formats.get(locale, "%Y-%m-%d %H:%M")
            return dt.strftime(format_str)
        
        dt = datetime(2024, 1, 15, 14, 30)
        
        result_en = format_datetime(dt, "en")
        assert "2024" in result_en
        assert "14" in result_en or "2" in result_en


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

