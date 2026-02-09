"""
LLM Integration - Integración con LLM para descripciones inteligentes
======================================================================
"""

import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class LLMIntegration:
    """Integración con LLM para mejorar descripciones"""
    
    def __init__(self, llm_provider: Optional[str] = None, api_key: Optional[str] = None):
        self.llm_provider = llm_provider or "openai"
        self.api_key = api_key
        self.enabled = bool(api_key)
    
    async def enhance_description(self, description: str) -> Dict[str, Any]:
        """
        Mejora una descripción usando LLM
        
        En producción, esto se conectaría a OpenAI, Anthropic, etc.
        Por ahora, usa reglas heurísticas mejoradas.
        """
        if not self.enabled:
            return self._enhance_with_rules(description)
        
        # Aquí iría la llamada real al LLM
        # Por ahora, usamos reglas mejoradas
        return self._enhance_with_rules(description)
    
    def _enhance_with_rules(self, description: str) -> Dict[str, Any]:
        """Mejora descripción usando reglas heurísticas"""
        enhanced = {
            "original": description,
            "enhanced": description,
            "extracted_features": self._extract_features(description),
            "suggestions": self._generate_suggestions(description),
            "improved_description": self._improve_description(description)
        }
        
        return enhanced
    
    def _extract_features(self, description: str) -> Dict[str, Any]:
        """Extrae características de la descripción"""
        description_lower = description.lower()
        
        features = {
            "size": self._extract_size(description_lower),
            "materials": self._extract_materials(description_lower),
            "functionality": self._extract_functionality(description_lower),
            "style": self._extract_style(description_lower),
            "requirements": self._extract_requirements(description_lower)
        }
        
        return features
    
    def _extract_size(self, text: str) -> Optional[str]:
        """Extrae información de tamaño"""
        size_keywords = {
            "pequeño": ["pequeño", "pequeña", "compacto", "mini"],
            "mediano": ["mediano", "mediana", "estándar", "normal"],
            "grande": ["grande", "grande", "extenso", "amplio"]
        }
        
        for size, keywords in size_keywords.items():
            if any(kw in text for kw in keywords):
                return size
        return None
    
    def _extract_materials(self, text: str) -> List[str]:
        """Extrae materiales mencionados"""
        materials = []
        material_keywords = {
            "acero": ["acero", "metal", "metálico"],
            "plastico": ["plástico", "plastico", "pvc"],
            "vidrio": ["vidrio", "cristal"],
            "madera": ["madera", "wood"],
            "aluminio": ["aluminio", "aluminum"]
        }
        
        for material, keywords in material_keywords.items():
            if any(kw in text for kw in keywords):
                materials.append(material)
        
        return materials
    
    def _extract_functionality(self, text: str) -> List[str]:
        """Extrae funcionalidades mencionadas"""
        functionalities = []
        func_keywords = {
            "portable": ["portable", "portátil", "móvil"],
            "automatico": ["automático", "automatico", "auto"],
            "digital": ["digital", "electrónico", "electronico"],
            "inteligente": ["inteligente", "smart", "ai"]
        }
        
        for func, keywords in func_keywords.items():
            if any(kw in text for kw in keywords):
                functionalities.append(func)
        
        return functionalities
    
    def _extract_style(self, text: str) -> Optional[str]:
        """Extrae estilo mencionado"""
        style_keywords = {
            "moderno": ["moderno", "moderna", "contemporáneo"],
            "clasico": ["clásico", "clasico", "tradicional"],
            "minimalista": ["minimalista", "simple", "sencillo"],
            "elegante": ["elegante", "sofisticado", "refinado"]
        }
        
        for style, keywords in style_keywords.items():
            if any(kw in text for kw in keywords):
                return style
        return None
    
    def _extract_requirements(self, text: str) -> List[str]:
        """Extrae requisitos mencionados"""
        requirements = []
        req_keywords = {
            "potente": ["potente", "fuerte", "powerful"],
            "durable": ["durable", "resistente", "robusto"],
            "facil_limpiar": ["fácil de limpiar", "facil de limpiar", "limpieza fácil"],
            "seguro": ["seguro", "segura", "safety", "safe"],
            "economico": ["económico", "economico", "barato", "barata"]
        }
        
        for req, keywords in req_keywords.items():
            if any(kw in text for kw in keywords):
                requirements.append(req)
        
        return requirements
    
    def _generate_suggestions(self, description: str) -> List[str]:
        """Genera sugerencias para mejorar la descripción"""
        suggestions = []
        
        if len(description) < 20:
            suggestions.append("Considera agregar más detalles sobre el producto")
        
        if "material" not in description.lower() and "materiales" not in description.lower():
            suggestions.append("Menciona los materiales deseados para mejores resultados")
        
        if "presupuesto" not in description.lower() and "costo" not in description.lower():
            suggestions.append("Incluye información de presupuesto si tienes uno")
        
        return suggestions
    
    def _improve_description(self, description: str) -> str:
        """Mejora la descripción agregando contexto"""
        improved = description
        
        # Agregar contexto si falta
        if not any(word in description.lower() for word in ["quiero", "necesito", "deseo"]):
            improved = f"Quiero hacer {improved}"
        
        return improved
    
    async def generate_alternative_descriptions(self, description: str, 
                                               num_alternatives: int = 3) -> List[str]:
        """Genera descripciones alternativas"""
        alternatives = []
        
        base_words = description.split()
        
        # Variaciones simples
        variations = [
            f"Necesito crear {description}",
            f"Deseo diseñar {description}",
            f"Quiero construir {description}",
            f"Busco hacer {description}",
            f"Necesito un {description}"
        ]
        
        return variations[:num_alternatives]
    
    async def analyze_complexity_from_description(self, description: str) -> Dict[str, Any]:
        """Analiza la complejidad esperada desde la descripción"""
        description_lower = description.lower()
        
        complexity_indicators = {
            "high": ["complejo", "avanzado", "profesional", "industrial", "múltiples"],
            "medium": ["estándar", "normal", "regular", "básico"],
            "low": ["simple", "fácil", "básico", "sencillo"]
        }
        
        complexity_score = 0
        detected_complexity = "medium"
        
        for level, keywords in complexity_indicators.items():
            matches = sum(1 for kw in keywords if kw in description_lower)
            if matches > 0:
                if level == "high":
                    complexity_score += matches * 2
                elif level == "low":
                    complexity_score -= matches
                else:
                    complexity_score += matches
        
        if complexity_score > 3:
            detected_complexity = "high"
        elif complexity_score < -1:
            detected_complexity = "low"
        
        return {
            "complexity": detected_complexity,
            "score": complexity_score,
            "indicators_found": matches if 'matches' in locals() else 0
        }




