"""
Detector de Categorías Mejorado
================================

Detección inteligente de categorías usando análisis de texto mejorado.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CategoryDetector:
    """Detector mejorado de categorías de oficios."""
    
    # Keywords con pesos para mejor detección
    CATEGORY_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
        "plomeria": [
            ("agua", 1.0), ("tubería", 2.0), ("tuberia", 2.0), ("grifo", 2.5), ("llave", 2.0),
            ("drenaje", 2.0), ("fuga", 2.5), ("caño", 2.0), ("plomero", 3.0), ("plomería", 3.0),
            ("desagüe", 2.0), ("desague", 2.0), ("inodoro", 2.0), ("baño", 1.5), ("lavabo", 2.0),
            ("ducha", 1.5), ("regadera", 1.5), ("calentador", 1.5), ("caldera", 1.5), ("válvula", 2.0),
            ("valvula", 2.0), ("sifón", 2.0), ("sifon", 2.0), ("cañería", 2.0), ("cañeria", 2.0)
        ],
        "electricidad": [
            ("eléctrico", 2.5), ("electrico", 2.5), ("cable", 2.0), ("interruptor", 2.5),
            ("enchufe", 2.5), ("corriente", 2.0), ("voltaje", 2.0), ("corto", 2.5), ("circuito", 2.5),
            ("fusible", 2.0), ("breaker", 2.0), ("disyuntor", 2.0), ("luz", 1.5), ("bombilla", 1.5),
            ("lámpara", 1.5), ("lampara", 1.5), ("toma", 2.0), ("conexión", 1.5), ("conexion", 1.5),
            ("cableado", 2.0), ("instalación eléctrica", 3.0), ("instalacion electrica", 3.0)
        ],
        "carpinteria": [
            ("madera", 2.5), ("mueble", 2.5), ("tabla", 2.0), ("sierra", 2.0), ("clavo", 1.5),
            ("tornillo", 1.5), ("carpintero", 3.0), ("carpintería", 3.0), ("carpinteria", 3.0),
            ("estante", 2.0), ("repisa", 2.0), ("puerta", 2.0), ("ventana", 2.0), ("marco", 2.0),
            ("ensamblaje", 2.0), ("lijar", 1.5), ("barniz", 1.5), ("barnizar", 1.5)
        ],
        "techos": [
            ("techo", 3.0), ("teja", 2.5), ("gotera", 3.0), ("filtración", 2.5), ("filtracion", 2.5),
            ("cubierta", 2.0), ("canalón", 2.0), ("canalon", 2.0), ("bajante", 2.0), ("humedad", 2.0),
            ("mancha", 1.5), ("goteo", 2.5), ("lluvia", 1.5), ("azotea", 2.0), ("tejado", 2.5)
        ],
        "albanileria": [
            ("ladrillo", 2.5), ("cemento", 2.5), ("pared", 2.0), ("muro", 2.0), ("albañil", 3.0),
            ("albañilería", 3.0), ("albanileria", 3.0), ("mampostería", 2.5), ("mamposteria", 2.5),
            ("mortero", 2.0), ("yeso", 2.0), ("estuco", 2.0), ("grieta", 2.0), ("fisura", 2.0),
            ("construcción", 1.5), ("construccion", 1.5)
        ],
        "pintura": [
            ("pintura", 3.0), ("brocha", 2.0), ("rodillo", 2.0), ("color", 1.5), ("barniz", 2.0),
            ("esmalte", 2.0), ("látex", 2.0), ("latex", 2.0), ("pintar", 2.5), ("pintado", 2.0),
            ("capas", 1.5), ("imprimación", 1.5), ("imprimacion", 1.5)
        ],
        "herreria": [
            ("metal", 2.0), ("hierro", 2.5), ("soldadura", 2.5), ("herrero", 3.0), ("herrería", 3.0),
            ("herrería", 3.0), ("rejilla", 2.0), ("portón", 2.0), ("porton", 2.0), ("reja", 2.0),
            ("estructura metálica", 2.5), ("estructura metalica", 2.5)
        ],
        "jardineria": [
            ("planta", 2.0), ("jardín", 2.5), ("jardin", 2.5), ("tierra", 1.5), ("riego", 2.0),
            ("poda", 2.0), ("césped", 2.0), ("cesped", 2.0), ("árbol", 1.5), ("arbol", 1.5),
            ("flor", 1.5), ("maceta", 1.5), ("abono", 1.5), ("fertilizante", 1.5)
        ]
    }
    
    @classmethod
    def detect_category(
        cls,
        text: str,
        confidence_threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        Detectar categoría con score de confianza.
        
        Args:
            text: Texto a analizar
            confidence_threshold: Umbral mínimo de confianza
        
        Returns:
            Tuple de (categoría, score)
        """
        if not text:
            return "general", 0.0
        
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        
        # Calcular scores por categoría
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0.0
            matches = 0
            
            for keyword, weight in keywords:
                # Buscar palabra completa (no substring)
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches_found = len(re.findall(pattern, text_lower, re.IGNORECASE))
                
                if matches_found > 0:
                    matches += matches_found
                    score += weight * matches_found
            
            # Normalizar score
            if matches > 0:
                scores[category] = score / (matches * 2.0)  # Normalizar
        
        # Si no hay matches, retornar general
        if not scores:
            return "general", 0.0
        
        # Obtener categoría con mayor score
        best_category = max(scores.items(), key=lambda x: x[1])
        
        # Si el score es muy bajo, retornar general
        if best_category[1] < confidence_threshold:
            return "general", best_category[1]
        
        return best_category
    
    @classmethod
    def detect_multiple_categories(
        cls,
        text: str,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Detectar múltiples categorías posibles.
        
        Args:
            text: Texto a analizar
            top_n: Número de categorías a retornar
        
        Returns:
            Lista de (categoría, score) ordenada
        """
        if not text:
            return [("general", 0.0)]
        
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        
        # Calcular scores
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0.0
            matches = 0
            
            for keyword, weight in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches_found = len(re.findall(pattern, text_lower, re.IGNORECASE))
                
                if matches_found > 0:
                    matches += matches_found
                    score += weight * matches_found
            
            if matches > 0:
                scores[category] = score / (matches * 2.0)
        
        # Ordenar y retornar top N
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_categories:
            return [("general", 0.0)]
        
        return sorted_categories[:top_n]




