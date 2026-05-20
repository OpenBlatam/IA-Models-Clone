"""
i18n System - Sistema de internacionalización
===============================================
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class I18nSystem:
    """Sistema de internacionalización"""
    
    def __init__(self, default_language: str = "es"):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
    
    def _load_translations(self):
        """Carga traducciones"""
        # Traducciones básicas en código (en producción vendrían de archivos)
        self.translations = {
            "es": {
                "prototype_generated": "Prototipo generado",
                "materials_needed": "Materiales necesarios",
                "total_cost": "Costo total",
                "build_time": "Tiempo de construcción",
                "difficulty": "Dificultad",
                "assembly_instructions": "Instrucciones de ensamblaje",
                "budget_options": "Opciones de presupuesto",
                "low": "Bajo",
                "medium": "Medio",
                "high": "Alto",
                "easy": "Fácil",
                "medium_difficulty": "Media",
                "hard": "Difícil"
            },
            "en": {
                "prototype_generated": "Prototype generated",
                "materials_needed": "Materials needed",
                "total_cost": "Total cost",
                "build_time": "Build time",
                "difficulty": "Difficulty",
                "assembly_instructions": "Assembly instructions",
                "budget_options": "Budget options",
                "low": "Low",
                "medium": "Medium",
                "high": "High",
                "easy": "Easy",
                "medium_difficulty": "Medium",
                "hard": "Hard"
            },
            "pt": {
                "prototype_generated": "Protótipo gerado",
                "materials_needed": "Materiais necessários",
                "total_cost": "Custo total",
                "build_time": "Tempo de construção",
                "difficulty": "Dificuldade",
                "assembly_instructions": "Instruções de montagem",
                "budget_options": "Opções de orçamento",
                "low": "Baixo",
                "medium": "Médio",
                "high": "Alto",
                "easy": "Fácil",
                "medium_difficulty": "Média",
                "hard": "Difícil"
            }
        }
    
    def set_language(self, language: str):
        """Establece el idioma actual"""
        if language in self.translations:
            self.current_language = language
        else:
            logger.warning(f"Idioma {language} no disponible, usando {self.default_language}")
            self.current_language = self.default_language
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Traduce una clave"""
        lang = language or self.current_language
        translations = self.translations.get(lang, self.translations[self.default_language])
        
        text = translations.get(key, key)
        
        # Reemplazar variables
        if kwargs:
            text = text.format(**kwargs)
        
        return text
    
    def t(self, key: str, **kwargs) -> str:
        """Alias corto para translate"""
        return self.translate(key, **kwargs)
    
    def get_available_languages(self) -> list:
        """Obtiene idiomas disponibles"""
        return list(self.translations.keys())
    
    def add_translation(self, language: str, key: str, value: str):
        """Agrega una traducción"""
        if language not in self.translations:
            self.translations[language] = {}
        self.translations[language][key] = value




