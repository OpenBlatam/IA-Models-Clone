"""
Content Optimizer - Sistema de optimización de contenido avanzado
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

from ..models import Language, SentimentType
from .embeddings import EmbeddingManager
from .transformer_models import TransformerModelManager

logger = logging.getLogger(__name__)


class OptimizationGoal(Enum):
    """Objetivos de optimización."""
    SEO = "seo"
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    ACCESSIBILITY = "accessibility"
    CLARITY = "clarity"
    PERSUASION = "persuasion"
    TECHNICAL = "technical"


class ContentType(Enum):
    """Tipos de contenido."""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    PRODUCT_DESCRIPTION = "product_description"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    LANDING_PAGE = "landing_page"
    NEWS_LETTER = "news_letter"
    TECHNICAL_DOCUMENT = "technical_document"
    MARKETING_COPY = "marketing_copy"
    EDUCATIONAL_CONTENT = "educational_content"


@dataclass
class OptimizationRule:
    """Regla de optimización."""
    rule_id: str
    name: str
    description: str
    goal: OptimizationGoal
    content_type: ContentType
    priority: int = 1
    weight: float = 1.0
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Sugerencia de optimización."""
    suggestion_id: str
    rule_id: str
    type: str  # "add", "remove", "replace", "reorganize"
    description: str
    original_text: str
    suggested_text: str
    position: int
    impact_score: float
    effort_level: str  # "low", "medium", "high"
    category: str


@dataclass
class ContentOptimization:
    """Optimización de contenido."""
    content_id: str
    original_content: str
    optimized_content: str
    optimization_goal: OptimizationGoal
    content_type: ContentType
    suggestions: List[OptimizationSuggestion]
    overall_score: float
    improvement_percentage: float
    optimized_at: datetime = field(default_factory=datetime.now)


class ContentOptimizer:
    """
    Sistema de optimización de contenido avanzado.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, transformer_manager: TransformerModelManager):
        """Inicializar optimizador de contenido."""
        self.embedding_manager = embedding_manager
        self.transformer_manager = transformer_manager
        
        # Reglas de optimización
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        
        # Cache de optimizaciones
        self.optimization_cache = {}
        
        # Configuración
        self.cache_ttl = 3600  # 1 hora
        self.max_suggestions = 50
        
        # Inicializar reglas por defecto
        self._initialize_default_rules()
        
        logger.info("ContentOptimizer inicializado")
    
    async def initialize(self):
        """Inicializar el optimizador de contenido."""
        try:
            logger.info("ContentOptimizer inicializado exitosamente")
        except Exception as e:
            logger.error(f"Error al inicializar ContentOptimizer: {e}")
            raise
    
    def _initialize_default_rules(self):
        """Inicializar reglas de optimización por defecto."""
        # Reglas SEO
        self.optimization_rules["seo_title_length"] = OptimizationRule(
            rule_id="seo_title_length",
            name="Longitud de título SEO",
            description="Optimizar longitud del título para SEO (50-60 caracteres)",
            goal=OptimizationGoal.SEO,
            content_type=ContentType.BLOG_POST,
            priority=1,
            weight=0.8,
            parameters={"min_length": 50, "max_length": 60}
        )
        
        self.optimization_rules["seo_meta_description"] = OptimizationRule(
            rule_id="seo_meta_description",
            name="Descripción meta SEO",
            description="Optimizar descripción meta (150-160 caracteres)",
            goal=OptimizationGoal.SEO,
            content_type=ContentType.BLOG_POST,
            priority=1,
            weight=0.7,
            parameters={"min_length": 150, "max_length": 160}
        )
        
        self.optimization_rules["seo_keyword_density"] = OptimizationRule(
            rule_id="seo_keyword_density",
            name="Densidad de palabras clave",
            description="Optimizar densidad de palabras clave (1-3%)",
            goal=OptimizationGoal.SEO,
            content_type=ContentType.BLOG_POST,
            priority=2,
            weight=0.6,
            parameters={"min_density": 1.0, "max_density": 3.0}
        )
        
        # Reglas de legibilidad
        self.optimization_rules["readability_sentence_length"] = OptimizationRule(
            rule_id="readability_sentence_length",
            name="Longitud de oraciones",
            description="Optimizar longitud de oraciones para mejor legibilidad",
            goal=OptimizationGoal.READABILITY,
            content_type=ContentType.BLOG_POST,
            priority=1,
            weight=0.8,
            parameters={"max_words": 20, "avg_words": 15}
        )
        
        self.optimization_rules["readability_paragraph_length"] = OptimizationRule(
            rule_id="readability_paragraph_length",
            name="Longitud de párrafos",
            description="Optimizar longitud de párrafos (3-5 oraciones)",
            goal=OptimizationGoal.READABILITY,
            content_type=ContentType.BLOG_POST,
            priority=2,
            weight=0.7,
            parameters={"min_sentences": 3, "max_sentences": 5}
        )
        
        # Reglas de engagement
        self.optimization_rules["engagement_headlines"] = OptimizationRule(
            rule_id="engagement_headlines",
            name="Títulos atractivos",
            description="Crear títulos más atractivos y llamativos",
            goal=OptimizationGoal.ENGAGEMENT,
            content_type=ContentType.BLOG_POST,
            priority=1,
            weight=0.9,
            parameters={"power_words": ["increíble", "sorprendente", "revolucionario", "secreto"]}
        )
        
        self.optimization_rules["engagement_call_to_action"] = OptimizationRule(
            rule_id="engagement_call_to_action",
            name="Llamadas a la acción",
            description="Agregar llamadas a la acción efectivas",
            goal=OptimizationGoal.ENGAGEMENT,
            content_type=ContentType.BLOG_POST,
            priority=2,
            weight=0.8,
            parameters={"cta_phrases": ["Descubre", "Aprende", "Descarga", "Regístrate"]}
        )
        
        # Reglas de conversión
        self.optimization_rules["conversion_urgency"] = OptimizationRule(
            rule_id="conversion_urgency",
            name="Crear urgencia",
            description="Agregar elementos de urgencia para aumentar conversiones",
            goal=OptimizationGoal.CONVERSION,
            content_type=ContentType.MARKETING_COPY,
            priority=1,
            weight=0.9,
            parameters={"urgency_words": ["limitado", "exclusivo", "ahora", "urgente"]}
        )
        
        # Reglas de accesibilidad
        self.optimization_rules["accessibility_alt_text"] = OptimizationRule(
            rule_id="accessibility_alt_text",
            name="Texto alternativo",
            description="Agregar texto alternativo descriptivo para imágenes",
            goal=OptimizationGoal.ACCESSIBILITY,
            content_type=ContentType.BLOG_POST,
            priority=1,
            weight=0.8,
            parameters={"min_length": 10, "max_length": 125}
        )
        
        # Reglas de claridad
        self.optimization_rules["clarity_jargon"] = OptimizationRule(
            rule_id="clarity_jargon",
            name="Reducir jerga técnica",
            description="Simplificar jerga técnica para mayor claridad",
            goal=OptimizationGoal.CLARITY,
            content_type=ContentType.TECHNICAL_DOCUMENT,
            priority=1,
            weight=0.7,
            parameters={"jargon_words": ["paradigma", "sinergia", "escalable", "robusto"]}
        )
    
    async def optimize_content(
        self,
        content: str,
        optimization_goal: OptimizationGoal,
        content_type: ContentType,
        target_keywords: Optional[List[str]] = None,
        custom_rules: Optional[List[OptimizationRule]] = None
    ) -> ContentOptimization:
        """
        Optimizar contenido según objetivos específicos.
        
        Args:
            content: Contenido a optimizar
            optimization_goal: Objetivo de optimización
            content_type: Tipo de contenido
            target_keywords: Palabras clave objetivo
            custom_rules: Reglas personalizadas
            
        Returns:
            Optimización del contenido
        """
        try:
            content_id = str(uuid.uuid4())
            
            # Verificar cache
            cache_key = self._generate_cache_key(content, optimization_goal, content_type)
            if cache_key in self.optimization_cache:
                cached_optimization = self.optimization_cache[cache_key]
                if datetime.now() - cached_optimization['timestamp'] < timedelta(seconds=self.cache_ttl):
                    return cached_optimization['optimization']
            
            # Obtener reglas aplicables
            applicable_rules = self._get_applicable_rules(optimization_goal, content_type, custom_rules)
            
            # Generar sugerencias de optimización
            suggestions = []
            for rule in applicable_rules:
                rule_suggestions = await self._apply_optimization_rule(content, rule, target_keywords)
                suggestions.extend(rule_suggestions)
            
            # Ordenar sugerencias por impacto
            suggestions.sort(key=lambda x: x.impact_score, reverse=True)
            suggestions = suggestions[:self.max_suggestions]
            
            # Aplicar optimizaciones
            optimized_content = await self._apply_optimizations(content, suggestions)
            
            # Calcular puntuación general
            overall_score = await self._calculate_optimization_score(content, optimized_content, optimization_goal)
            
            # Calcular porcentaje de mejora
            improvement_percentage = await self._calculate_improvement_percentage(content, optimized_content)
            
            # Crear optimización
            optimization = ContentOptimization(
                content_id=content_id,
                original_content=content,
                optimized_content=optimized_content,
                optimization_goal=optimization_goal,
                content_type=content_type,
                suggestions=suggestions,
                overall_score=overall_score,
                improvement_percentage=improvement_percentage
            )
            
            # Guardar en cache
            self.optimization_cache[cache_key] = {
                'optimization': optimization,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Contenido {content_id} optimizado exitosamente")
            return optimization
            
        except Exception as e:
            logger.error(f"Error al optimizar contenido: {e}")
            raise
    
    def _get_applicable_rules(
        self,
        optimization_goal: OptimizationGoal,
        content_type: ContentType,
        custom_rules: Optional[List[OptimizationRule]]
    ) -> List[OptimizationRule]:
        """Obtener reglas aplicables para la optimización."""
        applicable_rules = []
        
        # Agregar reglas personalizadas si se proporcionan
        if custom_rules:
            applicable_rules.extend(custom_rules)
        
        # Agregar reglas por defecto que coincidan
        for rule in self.optimization_rules.values():
            if (rule.enabled and 
                rule.goal == optimization_goal and 
                rule.content_type == content_type):
                applicable_rules.append(rule)
        
        # Ordenar por prioridad
        applicable_rules.sort(key=lambda x: x.priority)
        return applicable_rules
    
    async def _apply_optimization_rule(
        self,
        content: str,
        rule: OptimizationRule,
        target_keywords: Optional[List[str]]
    ) -> List[OptimizationSuggestion]:
        """Aplicar una regla de optimización específica."""
        suggestions = []
        
        try:
            if rule.rule_id == "seo_title_length":
                suggestions.extend(await self._optimize_title_length(content, rule))
            elif rule.rule_id == "seo_meta_description":
                suggestions.extend(await self._optimize_meta_description(content, rule))
            elif rule.rule_id == "seo_keyword_density":
                suggestions.extend(await self._optimize_keyword_density(content, rule, target_keywords))
            elif rule.rule_id == "readability_sentence_length":
                suggestions.extend(await self._optimize_sentence_length(content, rule))
            elif rule.rule_id == "readability_paragraph_length":
                suggestions.extend(await self._optimize_paragraph_length(content, rule))
            elif rule.rule_id == "engagement_headlines":
                suggestions.extend(await self._optimize_headlines(content, rule))
            elif rule.rule_id == "engagement_call_to_action":
                suggestions.extend(await self._optimize_call_to_action(content, rule))
            elif rule.rule_id == "conversion_urgency":
                suggestions.extend(await self._optimize_urgency(content, rule))
            elif rule.rule_id == "accessibility_alt_text":
                suggestions.extend(await self._optimize_alt_text(content, rule))
            elif rule.rule_id == "clarity_jargon":
                suggestions.extend(await self._optimize_jargon(content, rule))
            
        except Exception as e:
            logger.error(f"Error al aplicar regla {rule.rule_id}: {e}")
        
        return suggestions
    
    async def _optimize_title_length(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar longitud del título."""
        suggestions = []
        
        # Buscar títulos en el contenido
        title_patterns = [
            r'^#\s+(.+?)$',  # Markdown H1
            r'<h1[^>]*>(.+?)</h1>',  # HTML H1
            r'^(.+?)$'  # Primera línea
        ]
        
        for pattern in title_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                title_length = len(title)
                
                min_length = rule.parameters.get("min_length", 50)
                max_length = rule.parameters.get("max_length", 60)
                
                if title_length < min_length:
                    suggestion = OptimizationSuggestion(
                        suggestion_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        type="replace",
                        description=f"Título muy corto ({title_length} caracteres). Recomendado: {min_length}-{max_length}",
                        original_text=title,
                        suggested_text=title + " - Guía completa y actualizada",
                        position=match.start(),
                        impact_score=0.8,
                        effort_level="low",
                        category="seo"
                    )
                    suggestions.append(suggestion)
                elif title_length > max_length:
                    # Truncar título
                    truncated_title = title[:max_length-3] + "..."
                    suggestion = OptimizationSuggestion(
                        suggestion_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        type="replace",
                        description=f"Título muy largo ({title_length} caracteres). Recomendado: {min_length}-{max_length}",
                        original_text=title,
                        suggested_text=truncated_title,
                        position=match.start(),
                        impact_score=0.7,
                        effort_level="low",
                        category="seo"
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_meta_description(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar descripción meta."""
        suggestions = []
        
        # Buscar descripción meta existente
        meta_pattern = r'<meta\s+name=["\']description["\']\s+content=["\'](.+?)["\']'
        match = re.search(meta_pattern, content, re.IGNORECASE)
        
        min_length = rule.parameters.get("min_length", 150)
        max_length = rule.parameters.get("max_length", 160)
        
        if match:
            description = match.group(1)
            desc_length = len(description)
            
            if desc_length < min_length:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="replace",
                    description=f"Descripción meta muy corta ({desc_length} caracteres)",
                    original_text=description,
                    suggested_text=description + " Descubre más información y recursos útiles.",
                    position=match.start(1),
                    impact_score=0.7,
                    effort_level="low",
                    category="seo"
                )
                suggestions.append(suggestion)
            elif desc_length > max_length:
                truncated_desc = description[:max_length-3] + "..."
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="replace",
                    description=f"Descripción meta muy larga ({desc_length} caracteres)",
                    original_text=description,
                    suggested_text=truncated_desc,
                    position=match.start(1),
                    impact_score=0.6,
                    effort_level="low",
                    category="seo"
                )
                suggestions.append(suggestion)
        else:
            # Agregar descripción meta si no existe
            suggestion = OptimizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                type="add",
                description="Agregar descripción meta para SEO",
                original_text="",
                suggested_text='<meta name="description" content="Descripción optimizada para SEO del contenido">',
                position=0,
                impact_score=0.8,
                effort_level="medium",
                category="seo"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_keyword_density(
        self,
        content: str,
        rule: OptimizationRule,
        target_keywords: Optional[List[str]]
    ) -> List[OptimizationSuggestion]:
        """Optimizar densidad de palabras clave."""
        suggestions = []
        
        if not target_keywords:
            return suggestions
        
        min_density = rule.parameters.get("min_density", 1.0)
        max_density = rule.parameters.get("max_density", 3.0)
        
        words = content.lower().split()
        total_words = len(words)
        
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            keyword_count = words.count(keyword_lower)
            density = (keyword_count / total_words) * 100 if total_words > 0 else 0
            
            if density < min_density:
                # Sugerir agregar más instancias de la palabra clave
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="add",
                    description=f"Densidad de '{keyword}' muy baja ({density:.1f}%). Recomendado: {min_density}-{max_density}%",
                    original_text="",
                    suggested_text=f"Considera agregar más instancias de '{keyword}' naturalmente en el contenido",
                    position=0,
                    impact_score=0.6,
                    effort_level="medium",
                    category="seo"
                )
                suggestions.append(suggestion)
            elif density > max_density:
                # Sugerir reducir la densidad
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="remove",
                    description=f"Densidad de '{keyword}' muy alta ({density:.1f}%). Recomendado: {min_density}-{max_density}%",
                    original_text="",
                    suggested_text=f"Considera reducir algunas instancias de '{keyword}' para evitar keyword stuffing",
                    position=0,
                    impact_score=0.5,
                    effort_level="medium",
                    category="seo"
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_sentence_length(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar longitud de oraciones."""
        suggestions = []
        
        max_words = rule.parameters.get("max_words", 20)
        avg_words = rule.parameters.get("avg_words", 15)
        
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            words = sentence.split()
            word_count = len(words)
            
            if word_count > max_words:
                # Sugerir dividir oración larga
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="replace",
                    description=f"Oración muy larga ({word_count} palabras). Recomendado: máximo {max_words}",
                    original_text=sentence,
                    suggested_text=self._split_long_sentence(sentence, max_words),
                    position=content.find(sentence),
                    impact_score=0.7,
                    effort_level="medium",
                    category="readability"
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_paragraph_length(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar longitud de párrafos."""
        suggestions = []
        
        min_sentences = rule.parameters.get("min_sentences", 3)
        max_sentences = rule.parameters.get("max_sentences", 5)
        
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            sentences = re.split(r'[.!?]+', paragraph)
            sentence_count = len([s for s in sentences if s.strip()])
            
            if sentence_count < min_sentences:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="add",
                    description=f"Párrafo muy corto ({sentence_count} oraciones). Recomendado: {min_sentences}-{max_sentences}",
                    original_text="",
                    suggested_text="Considera expandir este párrafo con más información relevante",
                    position=content.find(paragraph),
                    impact_score=0.5,
                    effort_level="medium",
                    category="readability"
                )
                suggestions.append(suggestion)
            elif sentence_count > max_sentences:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="reorganize",
                    description=f"Párrafo muy largo ({sentence_count} oraciones). Recomendado: {min_sentences}-{max_sentences}",
                    original_text=paragraph,
                    suggested_text=self._split_long_paragraph(paragraph, max_sentences),
                    position=content.find(paragraph),
                    impact_score=0.6,
                    effort_level="medium",
                    category="readability"
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_headlines(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar títulos para mayor engagement."""
        suggestions = []
        
        power_words = rule.parameters.get("power_words", ["increíble", "sorprendente", "revolucionario"])
        
        # Buscar títulos
        title_patterns = [
            r'^#+\s+(.+?)$',  # Markdown headers
            r'<h[1-6][^>]*>(.+?)</h[1-6]>'  # HTML headers
        ]
        
        for pattern in title_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                
                # Verificar si ya tiene palabras poderosas
                has_power_word = any(word.lower() in title.lower() for word in power_words)
                
                if not has_power_word:
                    # Sugerir agregar palabra poderosa
                    enhanced_title = self._enhance_title_with_power_words(title, power_words)
                    suggestion = OptimizationSuggestion(
                        suggestion_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        type="replace",
                        description="Título puede ser más atractivo con palabras poderosas",
                        original_text=title,
                        suggested_text=enhanced_title,
                        position=match.start(1),
                        impact_score=0.8,
                        effort_level="low",
                        category="engagement"
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_call_to_action(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar llamadas a la acción."""
        suggestions = []
        
        cta_phrases = rule.parameters.get("cta_phrases", ["Descubre", "Aprende", "Descarga", "Regístrate"])
        
        # Verificar si hay CTAs existentes
        cta_patterns = [
            r'(?:click|haz clic|descarga|regístrate|aprende|descubre)',
            r'(?:call to action|llamada a la acción)',
            r'(?:button|botón)'
        ]
        
        has_cta = any(re.search(pattern, content, re.IGNORECASE) for pattern in cta_patterns)
        
        if not has_cta:
            # Sugerir agregar CTA
            cta_text = f"¡{cta_phrases[0]} más sobre este tema!"
            suggestion = OptimizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                type="add",
                description="Agregar llamada a la acción para aumentar engagement",
                original_text="",
                suggested_text=cta_text,
                position=len(content),
                impact_score=0.9,
                effort_level="low",
                category="engagement"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_urgency(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar para crear urgencia."""
        suggestions = []
        
        urgency_words = rule.parameters.get("urgency_words", ["limitado", "exclusivo", "ahora", "urgente"])
        
        # Verificar si ya tiene palabras de urgencia
        has_urgency = any(word.lower() in content.lower() for word in urgency_words)
        
        if not has_urgency:
            # Sugerir agregar urgencia
            urgency_text = "¡Oferta limitada! Actúa ahora antes de que se agote."
            suggestion = OptimizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                type="add",
                description="Agregar elementos de urgencia para aumentar conversiones",
                original_text="",
                suggested_text=urgency_text,
                position=len(content),
                impact_score=0.8,
                effort_level="low",
                category="conversion"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_alt_text(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar texto alternativo de imágenes."""
        suggestions = []
        
        min_length = rule.parameters.get("min_length", 10)
        max_length = rule.parameters.get("max_length", 125)
        
        # Buscar imágenes sin alt text o con alt text inadecuado
        img_pattern = r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*>'
        matches = re.finditer(img_pattern, content, re.IGNORECASE)
        
        for match in matches:
            alt_text = match.group(1)
            alt_length = len(alt_text)
            
            if alt_length < min_length:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="replace",
                    description=f"Alt text muy corto ({alt_length} caracteres). Recomendado: {min_length}-{max_length}",
                    original_text=alt_text,
                    suggested_text=alt_text + " - Imagen descriptiva del contenido",
                    position=match.start(1),
                    impact_score=0.6,
                    effort_level="low",
                    category="accessibility"
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _optimize_jargon(self, content: str, rule: OptimizationRule) -> List[OptimizationSuggestion]:
        """Optimizar jerga técnica."""
        suggestions = []
        
        jargon_words = rule.parameters.get("jargon_words", ["paradigma", "sinergia", "escalable", "robusto"])
        
        for jargon in jargon_words:
            if jargon.lower() in content.lower():
                # Sugerir reemplazo más simple
                simple_alternatives = {
                    "paradigma": "enfoque",
                    "sinergia": "colaboración",
                    "escalable": "que puede crecer",
                    "robusto": "fuerte y confiable"
                }
                
                alternative = simple_alternatives.get(jargon.lower(), "término más simple")
                
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    type="replace",
                    description=f"Simplificar jerga técnica '{jargon}' para mayor claridad",
                    original_text=jargon,
                    suggested_text=alternative,
                    position=content.lower().find(jargon.lower()),
                    impact_score=0.7,
                    effort_level="low",
                    category="clarity"
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    # Métodos auxiliares
    def _split_long_sentence(self, sentence: str, max_words: int) -> str:
        """Dividir oración larga en oraciones más cortas."""
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
        
        # Dividir en el punto medio
        mid_point = len(words) // 2
        first_part = ' '.join(words[:mid_point])
        second_part = ' '.join(words[mid_point:])
        
        return f"{first_part}. {second_part.capitalize()}"
    
    def _split_long_paragraph(self, paragraph: str, max_sentences: int) -> str:
        """Dividir párrafo largo en párrafos más cortos."""
        sentences = re.split(r'[.!?]+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return paragraph
        
        # Dividir en el punto medio
        mid_point = len(sentences) // 2
        first_part = '. '.join(sentences[:mid_point]) + '.'
        second_part = '. '.join(sentences[mid_point:]) + '.'
        
        return f"{first_part}\n\n{second_part}"
    
    def _enhance_title_with_power_words(self, title: str, power_words: List[str]) -> str:
        """Mejorar título con palabras poderosas."""
        # Agregar palabra poderosa al inicio
        enhanced_title = f"{power_words[0].title()} {title.lower()}"
        return enhanced_title
    
    async def _apply_optimizations(self, content: str, suggestions: List[OptimizationSuggestion]) -> str:
        """Aplicar optimizaciones al contenido."""
        optimized_content = content
        
        # Aplicar sugerencias en orden inverso para mantener posiciones
        for suggestion in reversed(suggestions):
            if suggestion.type == "replace":
                optimized_content = (
                    optimized_content[:suggestion.position] +
                    suggestion.suggested_text +
                    optimized_content[suggestion.position + len(suggestion.original_text):]
                )
            elif suggestion.type == "add":
                optimized_content = (
                    optimized_content[:suggestion.position] +
                    suggestion.suggested_text +
                    optimized_content[suggestion.position:]
                )
            elif suggestion.type == "remove":
                # Implementar lógica de eliminación si es necesario
                pass
        
        return optimized_content
    
    async def _calculate_optimization_score(
        self,
        original_content: str,
        optimized_content: str,
        optimization_goal: OptimizationGoal
    ) -> float:
        """Calcular puntuación de optimización."""
        # Implementación básica - en producción usar métricas más sofisticadas
        if optimization_goal == OptimizationGoal.SEO:
            return 85.0
        elif optimization_goal == OptimizationGoal.READABILITY:
            return 80.0
        elif optimization_goal == OptimizationGoal.ENGAGEMENT:
            return 90.0
        else:
            return 75.0
    
    async def _calculate_improvement_percentage(
        self,
        original_content: str,
        optimized_content: str
    ) -> float:
        """Calcular porcentaje de mejora."""
        # Implementación básica
        return 15.0  # 15% de mejora promedio
    
    def _generate_cache_key(self, content: str, goal: OptimizationGoal, content_type: ContentType) -> str:
        """Generar clave de cache."""
        import hashlib
        return hashlib.md5(f"{goal.value}:{content_type.value}:{content}".encode()).hexdigest()
    
    async def add_optimization_rule(self, rule: OptimizationRule) -> bool:
        """Agregar nueva regla de optimización."""
        try:
            self.optimization_rules[rule.rule_id] = rule
            logger.info(f"Regla de optimización {rule.rule_id} agregada")
            return True
        except Exception as e:
            logger.error(f"Error al agregar regla de optimización: {e}")
            return False
    
    async def remove_optimization_rule(self, rule_id: str) -> bool:
        """Remover regla de optimización."""
        try:
            if rule_id in self.optimization_rules:
                del self.optimization_rules[rule_id]
                logger.info(f"Regla de optimización {rule_id} removida")
                return True
            return False
        except Exception as e:
            logger.error(f"Error al remover regla de optimización: {e}")
            return False
    
    async def get_optimization_rules(self) -> List[Dict[str, Any]]:
        """Obtener todas las reglas de optimización."""
        return [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "goal": rule.goal.value,
                "content_type": rule.content_type.value,
                "priority": rule.priority,
                "weight": rule.weight,
                "enabled": rule.enabled
            }
            for rule in self.optimization_rules.values()
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del optimizador de contenido."""
        try:
            return {
                "status": "healthy",
                "optimization_rules": len(self.optimization_rules),
                "cache_size": len(self.optimization_cache),
                "embedding_manager_status": await self.embedding_manager.health_check(),
                "transformer_manager_status": await self.transformer_manager.health_check(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check de ContentOptimizer: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




