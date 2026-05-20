"""
Entity Extractor - Sistema de extracción de entidades nombradas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Tipos de entidades"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Entidad nombrada"""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float


class EntityExtractor:
    """Extractor de entidades nombradas"""

    def __init__(self):
        """Inicializar extractor"""
        # Patrones comunes
        self.patterns = {
            EntityType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            EntityType.URL: re.compile(r'https?://[^\s]+|www\.[^\s]+'),
            EntityType.PHONE: re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
            EntityType.DATE: re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
            EntityType.TIME: re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM|am|pm)?\b'),
            EntityType.MONEY: re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dólares|dollars|USD|EUR|€)'),
            EntityType.PERCENTAGE: re.compile(r'\d+(?:\.\d+)?%'),
        }
        
        # Palabras clave para organizaciones
        self.org_keywords = {
            'inc', 'llc', 'ltd', 'corp', 'company', 'corporation',
            'inc.', 'ltd.', 'corp.', 'co.', 's.a.', 's.l.',
            'empresa', 'corporación', 'sociedad', 'organización'
        }
        
        # Títulos que indican personas
        self.person_titles = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam',
            'señor', 'señora', 'señorita', 'doctor', 'profesor'
        }

    def extract(self, text: str) -> List[Entity]:
        """
        Extraer entidades del texto.

        Args:
            text: Texto

        Returns:
            Lista de entidades
        """
        entities = []
        
        # Extraer entidades con patrones
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # Extraer organizaciones (básico)
        org_entities = self._extract_organizations(text)
        entities.extend(org_entities)
        
        # Extraer personas (básico)
        person_entities = self._extract_persons(text)
        entities.extend(person_entities)
        
        # Extraer ubicaciones (básico)
        location_entities = self._extract_locations(text)
        entities.extend(location_entities)
        
        # Ordenar por posición
        entities.sort(key=lambda e: e.start)
        
        return entities

    def _extract_organizations(self, text: str) -> List[Entity]:
        """Extraer organizaciones"""
        entities = []
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,;:')
            if word_lower in self.org_keywords:
                # Intentar capturar nombre de organización
                if i > 0:
                    org_name = words[i-1] + ' ' + word
                    start = text.find(org_name)
                    if start != -1:
                        entities.append(Entity(
                            text=org_name,
                            type=EntityType.ORGANIZATION,
                            start=start,
                            end=start + len(org_name),
                            confidence=0.7
                        ))
        
        return entities

    def _extract_persons(self, text: str) -> List[Entity]:
        """Extraer personas"""
        entities = []
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,;:')
            if word_lower in self.person_titles and i < len(words) - 1:
                # Capturar nombre después del título
                name = words[i+1]
                person_text = word + ' ' + name
                start = text.find(person_text)
                if start != -1:
                    entities.append(Entity(
                        text=person_text,
                        type=EntityType.PERSON,
                        start=start,
                        end=start + len(person_text),
                        confidence=0.7
                    ))
        
        return entities

    def _extract_locations(self, text: str) -> List[Entity]:
        """Extraer ubicaciones (básico)"""
        entities = []
        # Palabras clave de ubicaciones comunes
        location_keywords = {
            'calle', 'avenida', 'avenue', 'street', 'road', 'boulevard',
            'ciudad', 'city', 'país', 'country', 'estado', 'state',
            'provincia', 'province', 'región', 'region'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,;:')
            if word_lower in location_keywords:
                # Capturar nombre de ubicación
                if i < len(words) - 1:
                    location_name = word + ' ' + words[i+1]
                    start = text.find(location_name)
                    if start != -1:
                        entities.append(Entity(
                            text=location_name,
                            type=EntityType.LOCATION,
                            start=start,
                            end=start + len(location_name),
                            confidence=0.6
                        ))
        
        return entities

    def extract_by_type(self, text: str, entity_type: EntityType) -> List[Entity]:
        """
        Extraer entidades de un tipo específico.

        Args:
            text: Texto
            entity_type: Tipo de entidad

        Returns:
            Lista de entidades
        """
        all_entities = self.extract(text)
        return [e for e in all_entities if e.type == entity_type]

    def get_entities_summary(self, text: str) -> Dict[str, Any]:
        """
        Obtener resumen de entidades.

        Args:
            text: Texto

        Returns:
            Resumen de entidades
        """
        entities = self.extract(text)
        
        by_type = {}
        for entity in entities:
            entity_type = entity.type.value
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity.text)
        
        return {
            "total_entities": len(entities),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "entities": [
                {
                    "text": e.text,
                    "type": e.type.value,
                    "confidence": e.confidence
                }
                for e in entities
            ]
        }






