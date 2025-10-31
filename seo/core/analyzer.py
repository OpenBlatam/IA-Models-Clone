from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
from typing import Dict, List, Any, Optional
from loguru import logger
import orjson
from .interfaces import SEOAnalyzer
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
        from langchain.prompts import ChatPromptTemplate
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO Analyzer ultra-optimizado para el servicio SEO.
Implementación con LangChain y análisis inteligente.
"""




class UltraFastSEOAnalyzer(SEOAnalyzer):
    """Analizador SEO ultra-optimizado con LangChain."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.api_key = self.config.get('api_key')
        self.llm = None
        self._setup_langchain()
    
    def get_analyzer_name(self) -> str:
        return "ultra_fast_langchain"
    
    def _setup_langchain(self) -> Any:
        """Configura LangChain ultra-optimizado."""
        if not self.api_key:
            logger.warning("No API key provided, using fallback analyzer")
            return
        
        try:
            
            self.llm = ChatOpenAI(
                model=self.config.get('model', 'gpt-3.5-turbo'),
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 1000),
                request_timeout=self.config.get('timeout', 30)
            )
            
        except ImportError:
            logger.warning("LangChain not available, using fallback analyzer")
        except Exception as e:
            logger.error(f"Error setting up LangChain: {e}")
    
    async def analyze(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Analiza datos SEO con máxima eficiencia."""
        if not self.llm:
            return self._fallback_analysis(seo_data, url)
        
        try:
            prompt = self._create_optimized_prompt(seo_data, url)
            response = await self.llm.ainvoke(prompt)
            
            # Parsear respuesta con orjson (más rápido)
            analysis = orjson.loads(response.content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LangChain analysis: {e}")
            return self._fallback_analysis(seo_data, url)
    
    def _create_optimized_prompt(self, seo_data: Dict[str, Any], url: str):
        """Crea prompt optimizado para análisis SEO."""
        
        template = """
        Analiza los siguientes datos SEO de {url} y proporciona recomendaciones en formato JSON:
        
        Datos SEO:
        - Título: {title}
        - Meta descripción: {meta_description}
        - Headers H1: {h1_count} elementos
        - Headers H2: {h2_count} elementos
        - Imágenes: {images_count} elementos
        - Enlaces: {links_count} elementos
        - Longitud del contenido: {content_length} caracteres
        - Keywords: {keywords}
        
        Proporciona análisis en este formato JSON:
        {{
            "score": 85,
            "recommendations": ["recomendación 1", "recomendación 2"],
            "strengths": ["fortaleza 1", "fortaleza 2"],
            "weaknesses": ["debilidad 1", "debilidad 2"],
            "priority_actions": ["acción 1", "acción 2"],
            "technical_issues": ["issue 1", "issue 2"],
            "opportunities": ["oportunidad 1", "oportunidad 2"]
        }}
        """
        
        # Preparar datos para el prompt
        prompt_data = {
            "url": url,
            "title": seo_data.get("title", ""),
            "meta_description": seo_data.get("meta_description", ""),
            "h1_count": len(seo_data.get("h1_tags", [])),
            "h2_count": len(seo_data.get("h2_tags", [])),
            "images_count": len(seo_data.get("images", [])),
            "links_count": len(seo_data.get("links", [])),
            "content_length": seo_data.get("content_length", 0),
            "keywords": ", ".join(seo_data.get("keywords", []))
        }
        
        return ChatPromptTemplate.from_template(template).format_messages(**prompt_data)
    
    def _fallback_analysis(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Análisis de fallback ultra-optimizado."""
        score = 50
        recommendations = []
        strengths = []
        weaknesses = []
        technical_issues = []
        opportunities = []
        
        # Análisis básico ultra-rápido
        if seo_data.get("title"):
            score += 10
            strengths.append("Título presente")
            if len(seo_data["title"]) < 30:
                weaknesses.append("Título muy corto")
                recommendations.append("Extender título a 50-60 caracteres")
            elif len(seo_data["title"]) > 60:
                weaknesses.append("Título muy largo")
                recommendations.append("Acortar título a 50-60 caracteres")
        else:
            weaknesses.append("Falta título")
            recommendations.append("Agregar título SEO")
            technical_issues.append("Título faltante")
        
        if seo_data.get("meta_description"):
            score += 10
            strengths.append("Meta descripción presente")
            if len(seo_data["meta_description"]) < 120:
                weaknesses.append("Meta descripción muy corta")
                recommendations.append("Extender meta descripción a 150-160 caracteres")
            elif len(seo_data["meta_description"]) > 160:
                weaknesses.append("Meta descripción muy larga")
                recommendations.append("Acortar meta descripción a 150-160 caracteres")
        else:
            weaknesses.append("Falta meta descripción")
            recommendations.append("Agregar meta descripción")
            technical_issues.append("Meta descripción faltante")
        
        # Análisis de headers
        h1_count = len(seo_data.get("h1_tags", []))
        if h1_count == 0:
            weaknesses.append("Falta header H1")
            recommendations.append("Agregar header H1 principal")
            technical_issues.append("Header H1 faltante")
        elif h1_count == 1:
            score += 5
            strengths.append("Header H1 presente")
        else:
            weaknesses.append(f"Demasiados headers H1 ({h1_count})")
            recommendations.append("Usar solo un header H1 por página")
            technical_issues.append("Múltiples headers H1")
        
        # Análisis de contenido
        content_length = seo_data.get("content_length", 0)
        if content_length > 300:
            score += 10
            strengths.append("Contenido sustancial")
            if content_length > 1000:
                opportunities.append("Contenido extenso - optimizar para long-tail keywords")
        else:
            weaknesses.append("Contenido insuficiente")
            recommendations.append("Aumentar contenido a al menos 300 palabras")
            technical_issues.append("Contenido insuficiente")
        
        # Análisis de imágenes
        images = seo_data.get("images", [])
        images_with_alt = sum(1 for img in images if img.get("alt"))
        if images:
            if images_with_alt == len(images):
                score += 5
                strengths.append("Todas las imágenes tienen alt text")
            else:
                weaknesses.append(f"Faltan alt text en {len(images) - images_with_alt} imágenes")
                recommendations.append("Agregar alt text descriptivo a todas las imágenes")
                technical_issues.append("Alt text faltante en imágenes")
        
        # Análisis de enlaces
        links = seo_data.get("links", [])
        internal_links = sum(1 for link in links if link.get("is_internal", False))
        if links:
            if internal_links > 0:
                score += 5
                strengths.append("Enlaces internos presentes")
                if internal_links < 3:
                    opportunities.append("Aumentar enlaces internos para mejor SEO")
            else:
                weaknesses.append("No hay enlaces internos")
                recommendations.append("Agregar enlaces internos relevantes")
        
        # Análisis de keywords
        keywords = seo_data.get("keywords", [])
        if keywords:
            score += 5
            strengths.append("Keywords definidas")
        else:
            opportunities.append("Definir keywords meta tag")
        
        # Análisis de datos estructurados
        structured_data = seo_data.get("structured_data", [])
        if structured_data:
            score += 5
            strengths.append("Datos estructurados presentes")
        else:
            opportunities.append("Implementar datos estructurados (Schema.org)")
        
        return {
            "score": min(score, 100),
            "recommendations": recommendations[:5],
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "priority_actions": recommendations[:3],
            "technical_issues": technical_issues[:3],
            "opportunities": opportunities[:3],
            "analysis_type": "fallback"
        }


class RuleBasedAnalyzer(SEOAnalyzer):
    """Analizador basado en reglas ultra-optimizado."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.rules = self._load_rules()
    
    def get_analyzer_name(self) -> str:
        return "rule_based"
    
    def _load_rules(self) -> Dict[str, Any]:
        """Carga reglas de análisis SEO."""
        return {
            "title": {
                "min_length": 30,
                "max_length": 60,
                "required": True,
                "score": 15
            },
            "meta_description": {
                "min_length": 120,
                "max_length": 160,
                "required": True,
                "score": 15
            },
            "h1": {
                "count": 1,
                "required": True,
                "score": 10
            },
            "content": {
                "min_length": 300,
                "score": 20
            },
            "images": {
                "alt_required": True,
                "score": 10
            },
            "links": {
                "internal_required": True,
                "score": 10
            },
            "keywords": {
                "required": False,
                "score": 5
            },
            "structured_data": {
                "required": False,
                "score": 5
            }
        }
    
    async def analyze(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Analiza datos SEO usando reglas predefinidas."""
        score = 0
        recommendations = []
        strengths = []
        weaknesses = []
        technical_issues = []
        opportunities = []
        
        # Analizar título
        title = seo_data.get("title", "")
        title_rule = self.rules["title"]
        
        if title:
            strengths.append("Título presente")
            score += title_rule["score"]
            
            if len(title) < title_rule["min_length"]:
                weaknesses.append(f"Título muy corto ({len(title)} caracteres)")
                recommendations.append(f"Extender título a {title_rule['min_length']}-{title_rule['max_length']} caracteres")
            elif len(title) > title_rule["max_length"]:
                weaknesses.append(f"Título muy largo ({len(title)} caracteres)")
                recommendations.append(f"Acortar título a {title_rule['min_length']}-{title_rule['max_length']} caracteres")
        else:
            weaknesses.append("Falta título")
            recommendations.append("Agregar título SEO")
            technical_issues.append("Título faltante")
        
        # Analizar meta descripción
        meta_desc = seo_data.get("meta_description", "")
        meta_rule = self.rules["meta_description"]
        
        if meta_desc:
            strengths.append("Meta descripción presente")
            score += meta_rule["score"]
            
            if len(meta_desc) < meta_rule["min_length"]:
                weaknesses.append(f"Meta descripción muy corta ({len(meta_desc)} caracteres)")
                recommendations.append(f"Extender meta descripción a {meta_rule['min_length']}-{meta_rule['max_length']} caracteres")
            elif len(meta_desc) > meta_rule["max_length"]:
                weaknesses.append(f"Meta descripción muy larga ({len(meta_desc)} caracteres)")
                recommendations.append(f"Acortar meta descripción a {meta_rule['min_length']}-{meta_rule['max_length']} caracteres")
        else:
            weaknesses.append("Falta meta descripción")
            recommendations.append("Agregar meta descripción")
            technical_issues.append("Meta descripción faltante")
        
        # Analizar headers H1
        h1_tags = seo_data.get("h1_tags", [])
        h1_rule = self.rules["h1"]
        
        if len(h1_tags) == h1_rule["count"]:
            strengths.append("Header H1 correcto")
            score += h1_rule["score"]
        elif len(h1_tags) == 0:
            weaknesses.append("Falta header H1")
            recommendations.append("Agregar header H1 principal")
            technical_issues.append("Header H1 faltante")
        else:
            weaknesses.append(f"Demasiados headers H1 ({len(h1_tags)})")
            recommendations.append("Usar solo un header H1 por página")
            technical_issues.append("Múltiples headers H1")
        
        # Analizar contenido
        content_length = seo_data.get("content_length", 0)
        content_rule = self.rules["content"]
        
        if content_length >= content_rule["min_length"]:
            strengths.append("Contenido sustancial")
            score += content_rule["score"]
        else:
            weaknesses.append(f"Contenido insuficiente ({content_length} caracteres)")
            recommendations.append(f"Aumentar contenido a al menos {content_rule['min_length']} caracteres")
            technical_issues.append("Contenido insuficiente")
        
        # Analizar imágenes
        images = seo_data.get("images", [])
        images_rule = self.rules["images"]
        
        if images:
            images_with_alt = sum(1 for img in images if img.get("alt"))
            if images_with_alt == len(images):
                strengths.append("Todas las imágenes tienen alt text")
                score += images_rule["score"]
            else:
                weaknesses.append(f"Faltan alt text en {len(images) - images_with_alt} imágenes")
                recommendations.append("Agregar alt text descriptivo a todas las imágenes")
                technical_issues.append("Alt text faltante en imágenes")
        
        # Analizar enlaces
        links = seo_data.get("links", [])
        links_rule = self.rules["links"]
        
        if links:
            internal_links = sum(1 for link in links if link.get("is_internal", False))
            if internal_links > 0:
                strengths.append("Enlaces internos presentes")
                score += links_rule["score"]
            else:
                weaknesses.append("No hay enlaces internos")
                recommendations.append("Agregar enlaces internos relevantes")
        
        # Analizar keywords
        keywords = seo_data.get("keywords", [])
        keywords_rule = self.rules["keywords"]
        
        if keywords:
            strengths.append("Keywords definidas")
            score += keywords_rule["score"]
        else:
            opportunities.append("Definir keywords meta tag")
        
        # Analizar datos estructurados
        structured_data = seo_data.get("structured_data", [])
        structured_rule = self.rules["structured_data"]
        
        if structured_data:
            strengths.append("Datos estructurados presentes")
            score += structured_rule["score"]
        else:
            opportunities.append("Implementar datos estructurados (Schema.org)")
        
        return {
            "score": min(score, 100),
            "recommendations": recommendations[:5],
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "priority_actions": recommendations[:3],
            "technical_issues": technical_issues[:3],
            "opportunities": opportunities[:3],
            "analysis_type": "rule_based"
        }


class AnalyzerFactory:
    """Factory para crear analizadores SEO."""
    
    @staticmethod
    def create_analyzer(
        analyzer_type: str = "ultra_fast", 
        config: Optional[Dict[str, Any]] = None
    ) -> SEOAnalyzer:
        """Crea un analizador SEO basado en el tipo especificado."""
        if analyzer_type == "ultra_fast":
            return UltraFastSEOAnalyzer(config)
        elif analyzer_type == "rule_based":
            return RuleBasedAnalyzer(config)
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}") 