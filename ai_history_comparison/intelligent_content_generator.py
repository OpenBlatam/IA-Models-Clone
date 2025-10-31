"""
Intelligent Content Generation System for AI History Comparison
Sistema de generación de contenido inteligente para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Tipos de contenido"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    TECHNICAL_DOCUMENT = "technical_document"
    SUMMARY = "summary"
    REPORT = "report"
    PRESENTATION = "presentation"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    NEWS_ITEM = "news_item"
    TUTORIAL = "tutorial"

class ContentStyle(Enum):
    """Estilos de contenido"""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"

class ContentTone(Enum):
    """Tonos de contenido"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    ENTHUSIASTIC = "enthusiastic"
    NEUTRAL = "neutral"
    URGENT = "urgent"
    CALM = "calm"
    INSPIRATIONAL = "inspirational"

@dataclass
class ContentTemplate:
    """Plantilla de contenido"""
    id: str
    name: str
    content_type: ContentType
    style: ContentStyle
    tone: ContentTone
    structure: List[str]
    keywords: List[str]
    target_audience: str
    word_count_range: Tuple[int, int]
    sections: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ContentRequest:
    """Solicitud de generación de contenido"""
    topic: str
    content_type: ContentType
    style: ContentStyle
    tone: ContentTone
    target_audience: str
    word_count: int
    keywords: List[str]
    requirements: List[str]
    context: Optional[Dict[str, Any]] = None
    language: str = "es"

@dataclass
class GeneratedContent:
    """Contenido generado"""
    id: str
    request: ContentRequest
    content: str
    structure: Dict[str, str]
    metadata: Dict[str, Any]
    quality_score: float
    generated_at: datetime = field(default_factory=datetime.now)

class IntelligentContentGenerator:
    """
    Generador inteligente de contenido
    """
    
    def __init__(
        self,
        enable_ai_integration: bool = True,
        enable_quality_control: bool = True,
        enable_optimization: bool = True
    ):
        self.enable_ai_integration = enable_ai_integration
        self.enable_quality_control = enable_quality_control
        self.enable_optimization = enable_optimization
        
        # Plantillas de contenido
        self.templates: Dict[str, ContentTemplate] = {}
        
        # Historial de contenido generado
        self.generated_content: Dict[str, GeneratedContent] = {}
        
        # Patrones de contenido exitoso
        self.success_patterns: Dict[str, List[str]] = {}
        
        # Configuración
        self.config = {
            "max_content_length": 5000,
            "min_content_length": 100,
            "default_word_count": 500,
            "quality_threshold": 0.7,
            "optimization_iterations": 3
        }
        
        # Inicializar plantillas
        self._initialize_templates()
        
        # Inicializar patrones de éxito
        self._initialize_success_patterns()
    
    def _initialize_templates(self):
        """Inicializar plantillas de contenido"""
        
        # Plantilla para artículo técnico
        self.templates["technical_article"] = ContentTemplate(
            id="technical_article",
            name="Artículo Técnico",
            content_type=ContentType.ARTICLE,
            style=ContentStyle.TECHNICAL,
            tone=ContentTone.PROFESSIONAL,
            structure=[
                "introducción",
                "problema",
                "solución",
                "implementación",
                "resultados",
                "conclusión"
            ],
            keywords=["técnico", "implementación", "solución", "resultados"],
            target_audience="desarrolladores y técnicos",
            word_count_range=(800, 2000),
            sections=[
                {"name": "introducción", "min_words": 100, "max_words": 200},
                {"name": "problema", "min_words": 150, "max_words": 300},
                {"name": "solución", "min_words": 200, "max_words": 400},
                {"name": "implementación", "min_words": 200, "max_words": 500},
                {"name": "resultados", "min_words": 150, "max_words": 300},
                {"name": "conclusión", "min_words": 100, "max_words": 200}
            ]
        )
        
        # Plantilla para blog post
        self.templates["blog_post"] = ContentTemplate(
            id="blog_post",
            name="Blog Post",
            content_type=ContentType.BLOG_POST,
            style=ContentStyle.CONVERSATIONAL,
            tone=ContentTone.FRIENDLY,
            structure=[
                "hook",
                "introducción",
                "desarrollo",
                "conclusión",
                "call_to_action"
            ],
            keywords=["blog", "personal", "experiencia", "consejos"],
            target_audience="lectores generales",
            word_count_range=(300, 800),
            sections=[
                {"name": "hook", "min_words": 50, "max_words": 100},
                {"name": "introducción", "min_words": 100, "max_words": 200},
                {"name": "desarrollo", "min_words": 200, "max_words": 400},
                {"name": "conclusión", "min_words": 100, "max_words": 200},
                {"name": "call_to_action", "min_words": 50, "max_words": 100}
            ]
        )
        
        # Plantilla para resumen ejecutivo
        self.templates["executive_summary"] = ContentTemplate(
            id="executive_summary",
            name="Resumen Ejecutivo",
            content_type=ContentType.SUMMARY,
            style=ContentStyle.FORMAL,
            tone=ContentTone.PROFESSIONAL,
            structure=[
                "resumen_ejecutivo",
                "puntos_clave",
                "recomendaciones",
                "próximos_pasos"
            ],
            keywords=["resumen", "ejecutivo", "puntos clave", "recomendaciones"],
            target_audience="ejecutivos y tomadores de decisión",
            word_count_range=(200, 500),
            sections=[
                {"name": "resumen_ejecutivo", "min_words": 100, "max_words": 200},
                {"name": "puntos_clave", "min_words": 150, "max_words": 250},
                {"name": "recomendaciones", "min_words": 100, "max_words": 200},
                {"name": "próximos_pasos", "min_words": 50, "max_words": 100}
            ]
        )
        
        # Plantilla para tutorial
        self.templates["tutorial"] = ContentTemplate(
            id="tutorial",
            name="Tutorial",
            content_type=ContentType.TUTORIAL,
            style=ContentStyle.INFORMATIVE,
            tone=ContentTone.FRIENDLY,
            structure=[
                "introducción",
                "prerrequisitos",
                "paso_a_paso",
                "ejemplos",
                "conclusión"
            ],
            keywords=["tutorial", "paso a paso", "ejemplos", "aprender"],
            target_audience="principiantes y estudiantes",
            word_count_range=(500, 1500),
            sections=[
                {"name": "introducción", "min_words": 100, "max_words": 200},
                {"name": "prerrequisitos", "min_words": 50, "max_words": 150},
                {"name": "paso_a_paso", "min_words": 300, "max_words": 800},
                {"name": "ejemplos", "min_words": 200, "max_words": 400},
                {"name": "conclusión", "min_words": 100, "max_words": 200}
            ]
        )
    
    def _initialize_success_patterns(self):
        """Inicializar patrones de contenido exitoso"""
        self.success_patterns = {
            "high_engagement": [
                "Usar preguntas retóricas",
                "Incluir estadísticas impactantes",
                "Contar historias personales",
                "Usar analogías",
                "Incluir llamadas a la acción claras"
            ],
            "high_readability": [
                "Usar oraciones cortas",
                "Dividir párrafos largos",
                "Usar subtítulos",
                "Incluir listas",
                "Evitar jerga técnica"
            ],
            "high_authority": [
                "Citar fuentes confiables",
                "Incluir datos específicos",
                "Usar lenguaje técnico apropiado",
                "Proporcionar ejemplos concretos",
                "Mencionar credenciales relevantes"
            ],
            "high_shareability": [
                "Crear títulos atractivos",
                "Incluir elementos visuales",
                "Usar hashtags relevantes",
                "Crear contenido evergreen",
                "Incluir elementos emocionales"
            ]
        }
    
    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """
        Generar contenido basado en la solicitud
        
        Args:
            request: Solicitud de generación de contenido
            
        Returns:
            Contenido generado
        """
        try:
            logger.info(f"Generating content for topic: {request.topic}")
            
            # Seleccionar plantilla apropiada
            template = self._select_template(request)
            
            # Generar estructura del contenido
            structure = await self._generate_structure(request, template)
            
            # Generar contenido para cada sección
            content_sections = {}
            for section_name, section_info in structure.items():
                content_sections[section_name] = await self._generate_section_content(
                    request, template, section_name, section_info
                )
            
            # Combinar secciones en contenido final
            full_content = self._combine_sections(content_sections, template)
            
            # Optimizar contenido si está habilitado
            if self.enable_optimization:
                full_content = await self._optimize_content(full_content, request, template)
            
            # Evaluar calidad del contenido
            quality_score = await self._evaluate_content_quality(full_content, request)
            
            # Crear objeto de contenido generado
            content_id = f"content_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            generated_content = GeneratedContent(
                id=content_id,
                request=request,
                content=full_content,
                structure=content_sections,
                metadata={
                    "template_used": template.id,
                    "word_count": len(full_content.split()),
                    "section_count": len(content_sections),
                    "generation_time": datetime.now().isoformat()
                },
                quality_score=quality_score
            )
            
            # Almacenar contenido generado
            self.generated_content[content_id] = generated_content
            
            logger.info(f"Content generated successfully: {content_id}")
            return generated_content
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    def _select_template(self, request: ContentRequest) -> ContentTemplate:
        """Seleccionar plantilla apropiada basada en la solicitud"""
        # Buscar plantilla que coincida con el tipo de contenido
        matching_templates = [
            template for template in self.templates.values()
            if template.content_type == request.content_type
        ]
        
        if not matching_templates:
            # Usar plantilla por defecto
            return self.templates["blog_post"]
        
        # Seleccionar la mejor plantilla basada en estilo y tono
        best_template = matching_templates[0]
        for template in matching_templates:
            if (template.style == request.style and 
                template.tone == request.tone):
                best_template = template
                break
        
        return best_template
    
    async def _generate_structure(
        self,
        request: ContentRequest,
        template: ContentTemplate
    ) -> Dict[str, Dict[str, Any]]:
        """Generar estructura del contenido"""
        structure = {}
        
        for section in template.sections:
            section_name = section["name"]
            
            # Calcular número de palabras para esta sección
            target_words = min(
                max(section["min_words"], request.word_count // len(template.sections)),
                section["max_words"]
            )
            
            structure[section_name] = {
                "target_words": target_words,
                "keywords": self._extract_section_keywords(request, section_name),
                "style_guidelines": self._get_section_style_guidelines(template, section_name)
            }
        
        return structure
    
    def _extract_section_keywords(self, request: ContentRequest, section_name: str) -> List[str]:
        """Extraer palabras clave relevantes para una sección"""
        base_keywords = request.keywords.copy()
        
        # Agregar palabras clave específicas por sección
        section_keywords = {
            "introducción": ["introducción", "overview", "resumen"],
            "conclusión": ["conclusión", "resumen", "finalmente"],
            "problema": ["problema", "desafío", "issue", "challenge"],
            "solución": ["solución", "solución", "approach", "método"],
            "resultados": ["resultados", "outcomes", "impacto", "beneficios"]
        }
        
        if section_name in section_keywords:
            base_keywords.extend(section_keywords[section_name])
        
        return base_keywords
    
    def _get_section_style_guidelines(
        self,
        template: ContentTemplate,
        section_name: str
    ) -> Dict[str, Any]:
        """Obtener guías de estilo para una sección"""
        return {
            "style": template.style.value,
            "tone": template.tone.value,
            "target_audience": template.target_audience,
            "formality_level": self._get_formality_level(template.style),
            "sentence_length": self._get_sentence_length_guideline(template.style),
            "vocabulary_level": self._get_vocabulary_level(template.style)
        }
    
    def _get_formality_level(self, style: ContentStyle) -> str:
        """Obtener nivel de formalidad basado en el estilo"""
        formal_styles = [ContentStyle.FORMAL, ContentStyle.ACADEMIC, ContentStyle.TECHNICAL]
        return "formal" if style in formal_styles else "informal"
    
    def _get_sentence_length_guideline(self, style: ContentStyle) -> str:
        """Obtener guía de longitud de oraciones"""
        if style in [ContentStyle.TECHNICAL, ContentStyle.ACADEMIC]:
            return "medium_to_long"
        elif style in [ContentStyle.CONVERSATIONAL, ContentStyle.CREATIVE]:
            return "short_to_medium"
        else:
            return "medium"
    
    def _get_vocabulary_level(self, style: ContentStyle) -> str:
        """Obtener nivel de vocabulario"""
        if style in [ContentStyle.TECHNICAL, ContentStyle.ACADEMIC]:
            return "advanced"
        elif style in [ContentStyle.CONVERSATIONAL, ContentStyle.CREATIVE]:
            return "simple"
        else:
            return "intermediate"
    
    async def _generate_section_content(
        self,
        request: ContentRequest,
        template: ContentTemplate,
        section_name: str,
        section_info: Dict[str, Any]
    ) -> str:
        """Generar contenido para una sección específica"""
        # Esta es una implementación básica
        # En una implementación real, se integraría con un modelo de IA
        
        target_words = section_info["target_words"]
        keywords = section_info["keywords"]
        style_guidelines = section_info["style_guidelines"]
        
        # Generar contenido basado en el tipo de sección
        if section_name == "introducción":
            content = self._generate_introduction(request, keywords, target_words)
        elif section_name == "conclusión":
            content = self._generate_conclusion(request, keywords, target_words)
        elif section_name == "problema":
            content = self._generate_problem_section(request, keywords, target_words)
        elif section_name == "solución":
            content = self._generate_solution_section(request, keywords, target_words)
        elif section_name == "resultados":
            content = self._generate_results_section(request, keywords, target_words)
        else:
            content = self._generate_generic_section(request, keywords, target_words)
        
        # Ajustar estilo según las guías
        content = self._adjust_content_style(content, style_guidelines)
        
        return content
    
    def _generate_introduction(self, request: ContentRequest, keywords: List[str], target_words: int) -> str:
        """Generar introducción"""
        topic = request.topic
        audience = request.target_audience
        
        introduction_templates = [
            f"En este artículo, exploraremos {topic} y su relevancia para {audience}. "
            f"Analizaremos los aspectos más importantes y proporcionaremos insights valiosos.",
            
            f"{topic} es un tema de gran importancia en la actualidad. "
            f"En este contenido, profundizaremos en los conceptos clave y "
            f"proporcionaremos información práctica para {audience}.",
            
            f"¿Te has preguntado sobre {topic}? En este análisis completo, "
            f"examinaremos todos los aspectos relevantes y te ayudaremos a "
            f"comprender mejor este tema importante."
        ]
        
        base_content = random.choice(introduction_templates)
        
        # Expandir contenido para alcanzar el número de palabras objetivo
        while len(base_content.split()) < target_words:
            expansion_phrases = [
                f"Además, consideraremos las implicaciones prácticas de {topic}.",
                f"También analizaremos las tendencias actuales relacionadas con este tema.",
                f"Es importante entender cómo {topic} afecta a {audience} en el contexto actual.",
                f"Exploraremos diferentes perspectivas sobre {topic} para proporcionar una visión completa."
            ]
            base_content += " " + random.choice(expansion_phrases)
        
        return base_content
    
    def _generate_conclusion(self, request: ContentRequest, keywords: List[str], target_words: int) -> str:
        """Generar conclusión"""
        topic = request.topic
        
        conclusion_templates = [
            f"En conclusión, {topic} representa un aspecto fundamental que debe ser considerado cuidadosamente. "
            f"Los puntos analizados en este contenido proporcionan una base sólida para comprender mejor este tema.",
            
            f"Para resumir, hemos explorado los aspectos más relevantes de {topic}. "
            f"Es importante aplicar estos conocimientos en la práctica y continuar aprendiendo sobre este tema.",
            
            f"En resumen, {topic} ofrece oportunidades significativas para {request.target_audience}. "
            f"Esperamos que este contenido haya sido útil y que puedas aplicar estos insights en tu trabajo."
        ]
        
        base_content = random.choice(conclusion_templates)
        
        # Expandir contenido
        while len(base_content.split()) < target_words:
            expansion_phrases = [
                f"Recuerda que el aprendizaje continuo es clave para dominar {topic}.",
                f"Te recomendamos explorar más recursos sobre este tema para profundizar tu conocimiento.",
                f"Si tienes preguntas sobre {topic}, no dudes en buscar más información o consultar con expertos.",
                f"El futuro de {topic} promete desarrollos interesantes que vale la pena seguir."
            ]
            base_content += " " + random.choice(expansion_phrases)
        
        return base_content
    
    def _generate_problem_section(self, request: ContentRequest, keywords: List[str], target_words: int) -> str:
        """Generar sección de problema"""
        topic = request.topic
        
        problem_templates = [
            f"Uno de los principales desafíos relacionados con {topic} es la complejidad de su implementación. "
            f"Muchos {request.target_audience} enfrentan dificultades para entender y aplicar los conceptos fundamentales.",
            
            f"El problema central con {topic} radica en la falta de comprensión de sus principios básicos. "
            f"Esto puede llevar a implementaciones ineficientes y resultados subóptimos.",
            
            f"Un desafío significativo en {topic} es la evolución constante de las mejores prácticas. "
            f"Los {request.target_audience} deben mantenerse actualizados con los últimos desarrollos."
        ]
        
        base_content = random.choice(problem_templates)
        
        # Expandir con más detalles del problema
        while len(base_content.split()) < target_words:
            expansion_phrases = [
                f"Además, la falta de recursos adecuados puede complicar aún más la situación.",
                f"Otro aspecto problemático es la curva de aprendizaje asociada con {topic}.",
                f"La fragmentación de la información disponible también representa un obstáculo importante.",
                f"Los costos asociados con la implementación pueden ser prohibitivos para algunos casos de uso."
            ]
            base_content += " " + random.choice(expansion_phrases)
        
        return base_content
    
    def _generate_solution_section(self, request: ContentRequest, keywords: List[str], target_words: int) -> str:
        """Generar sección de solución"""
        topic = request.topic
        
        solution_templates = [
            f"La solución más efectiva para abordar {topic} implica un enfoque sistemático y bien planificado. "
            f"Es importante comenzar con una comprensión clara de los objetivos y requerimientos.",
            
            f"Para resolver los desafíos de {topic}, recomendamos implementar una estrategia gradual. "
            f"Esto permite a los {request.target_audience} desarrollar competencias de manera progresiva.",
            
            f"Una aproximación exitosa a {topic} requiere combinar teoría y práctica. "
            f"Los {request.target_audience} deben tener acceso tanto a recursos educativos como a oportunidades de aplicación."
        ]
        
        base_content = random.choice(solution_templates)
        
        # Expandir con detalles de la solución
        while len(base_content.split()) < target_words:
            expansion_phrases = [
                f"Es fundamental establecer métricas claras para medir el progreso y el éxito.",
                f"La colaboración entre diferentes stakeholders es clave para el éxito de la implementación.",
                f"Se recomienda comenzar con proyectos piloto para validar el enfoque antes de una implementación completa.",
                f"El uso de herramientas y tecnologías apropiadas puede facilitar significativamente el proceso."
            ]
            base_content += " " + random.choice(expansion_phrases)
        
        return base_content
    
    def _generate_results_section(self, request: ContentRequest, keywords: List[str], target_words: int) -> str:
        """Generar sección de resultados"""
        topic = request.topic
        
        results_templates = [
            f"Los resultados obtenidos con {topic} han sido significativamente positivos. "
            f"Los {request.target_audience} han reportado mejoras notables en sus procesos y resultados.",
            
            f"La implementación de {topic} ha demostrado ser altamente efectiva. "
            f"Los datos muestran mejoras consistentes en los indicadores clave de rendimiento.",
            
            f"Los resultados de aplicar {topic} superan las expectativas iniciales. "
            f"Los {request.target_audience} han experimentado beneficios tangibles y medibles."
        ]
        
        base_content = random.choice(results_templates)
        
        # Expandir con detalles de resultados
        while len(base_content.split()) < target_words:
            expansion_phrases = [
                f"Las métricas de satisfacción han aumentado considerablemente desde la implementación.",
                f"Se han observado reducciones significativas en los tiempos de procesamiento.",
                f"La calidad de los resultados ha mejorado de manera consistente y sostenible.",
                f"Los costos operativos se han reducido mientras que la eficiencia ha aumentado."
            ]
            base_content += " " + random.choice(expansion_phrases)
        
        return base_content
    
    def _generate_generic_section(self, request: ContentRequest, keywords: List[str], target_words: int) -> str:
        """Generar sección genérica"""
        topic = request.topic
        
        generic_templates = [
            f"Al explorar {topic}, es importante considerar todos los aspectos relevantes. "
            f"Los {request.target_audience} deben tener en cuenta las implicaciones prácticas de cada decisión.",
            
            f"En el contexto de {topic}, es fundamental mantener una perspectiva equilibrada. "
            f"Esto permite a los {request.target_audience} tomar decisiones informadas y efectivas.",
            
            f"El análisis de {topic} revela múltiples dimensiones que deben ser consideradas. "
            f"Cada aspecto contribuye a una comprensión más completa del tema en cuestión."
        ]
        
        base_content = random.choice(generic_templates)
        
        # Expandir contenido genérico
        while len(base_content.split()) < target_words:
            expansion_phrases = [
                f"Es importante evaluar cada opción cuidadosamente antes de tomar una decisión.",
                f"La experiencia práctica es invaluable para comprender completamente este tema.",
                f"Los expertos en el campo recomiendan un enfoque gradual y bien planificado.",
                f"La investigación continua es esencial para mantenerse actualizado con los últimos desarrollos."
            ]
            base_content += " " + random.choice(expansion_phrases)
        
        return base_content
    
    def _adjust_content_style(self, content: str, style_guidelines: Dict[str, Any]) -> str:
        """Ajustar el estilo del contenido según las guías"""
        # Ajustar formalidad
        if style_guidelines["formality_level"] == "formal":
            content = self._make_content_more_formal(content)
        elif style_guidelines["formality_level"] == "informal":
            content = self._make_content_more_informal(content)
        
        # Ajustar longitud de oraciones
        if style_guidelines["sentence_length"] == "short_to_medium":
            content = self._shorten_sentences(content)
        elif style_guidelines["sentence_length"] == "medium_to_long":
            content = self._lengthen_sentences(content)
        
        # Ajustar nivel de vocabulario
        if style_guidelines["vocabulary_level"] == "simple":
            content = self._simplify_vocabulary(content)
        elif style_guidelines["vocabulary_level"] == "advanced":
            content = self._enhance_vocabulary(content)
        
        return content
    
    def _make_content_more_formal(self, content: str) -> str:
        """Hacer el contenido más formal"""
        replacements = {
            "te": "usted",
            "tú": "usted",
            "vas": "va",
            "tienes": "tiene",
            "puedes": "puede",
            "debes": "debe"
        }
        
        for informal, formal in replacements.items():
            content = content.replace(informal, formal)
        
        return content
    
    def _make_content_more_informal(self, content: str) -> str:
        """Hacer el contenido más informal"""
        replacements = {
            "usted": "tú",
            "va": "vas",
            "tiene": "tienes",
            "puede": "puedes",
            "debe": "debes"
        }
        
        for formal, informal in replacements.items():
            content = content.replace(formal, informal)
        
        return content
    
    def _shorten_sentences(self, content: str) -> str:
        """Acortar oraciones largas"""
        sentences = content.split('. ')
        shortened_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:
                # Dividir oraciones muy largas
                words = sentence.split()
                mid_point = len(words) // 2
                first_part = ' '.join(words[:mid_point])
                second_part = ' '.join(words[mid_point:])
                shortened_sentences.append(first_part + '.')
                shortened_sentences.append(second_part)
            else:
                shortened_sentences.append(sentence)
        
        return '. '.join(shortened_sentences)
    
    def _lengthen_sentences(self, content: str) -> str:
        """Alargar oraciones cortas"""
        sentences = content.split('. ')
        lengthened_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 10 and i < len(sentences) - 1:
                # Combinar oraciones cortas
                next_sentence = sentences[i + 1]
                combined = sentence + ' y ' + next_sentence
                lengthened_sentences.append(combined)
                sentences[i + 1] = ""  # Marcar como procesada
            elif sentence:
                lengthened_sentences.append(sentence)
        
        return '. '.join([s for s in lengthened_sentences if s])
    
    def _simplify_vocabulary(self, content: str) -> str:
        """Simplificar vocabulario"""
        replacements = {
            "implementar": "poner en práctica",
            "optimizar": "mejorar",
            "facilitar": "hacer más fácil",
            "efectivo": "que funciona bien",
            "significativo": "importante"
        }
        
        for complex_word, simple_word in replacements.items():
            content = content.replace(complex_word, simple_word)
        
        return content
    
    def _enhance_vocabulary(self, content: str) -> str:
        """Mejorar vocabulario"""
        replacements = {
            "poner en práctica": "implementar",
            "mejorar": "optimizar",
            "hacer más fácil": "facilitar",
            "que funciona bien": "efectivo",
            "importante": "significativo"
        }
        
        for simple_word, complex_word in replacements.items():
            content = content.replace(simple_word, complex_word)
        
        return content
    
    def _combine_sections(self, content_sections: Dict[str, str], template: ContentTemplate) -> str:
        """Combinar secciones en contenido final"""
        full_content = ""
        
        for section_name in template.structure:
            if section_name in content_sections:
                section_content = content_sections[section_name]
                
                # Agregar título de sección
                section_title = section_name.replace('_', ' ').title()
                full_content += f"\n\n## {section_title}\n\n"
                
                # Agregar contenido de la sección
                full_content += section_content
        
        return full_content.strip()
    
    async def _optimize_content(
        self,
        content: str,
        request: ContentRequest,
        template: ContentTemplate
    ) -> str:
        """Optimizar contenido generado"""
        optimized_content = content
        
        # Aplicar patrones de éxito
        for pattern_type, patterns in self.success_patterns.items():
            if pattern_type == "high_engagement":
                optimized_content = self._apply_engagement_patterns(optimized_content)
            elif pattern_type == "high_readability":
                optimized_content = self._apply_readability_patterns(optimized_content)
            elif pattern_type == "high_authority":
                optimized_content = self._apply_authority_patterns(optimized_content)
        
        return optimized_content
    
    def _apply_engagement_patterns(self, content: str) -> str:
        """Aplicar patrones de engagement"""
        # Agregar preguntas retóricas
        if "?" not in content:
            questions = [
                "¿Te has preguntado alguna vez sobre esto?",
                "¿Qué opinas sobre este enfoque?",
                "¿Has considerado esta perspectiva?"
            ]
            content = random.choice(questions) + " " + content
        
        return content
    
    def _apply_readability_patterns(self, content: str) -> str:
        """Aplicar patrones de legibilidad"""
        # Agregar subtítulos si no existen
        if "##" not in content:
            sentences = content.split('. ')
            if len(sentences) > 3:
                mid_point = len(sentences) // 2
                content = '. '.join(sentences[:mid_point]) + '.\n\n## Puntos Clave\n\n' + '. '.join(sentences[mid_point:])
        
        return content
    
    def _apply_authority_patterns(self, content: str) -> str:
        """Aplicar patrones de autoridad"""
        # Agregar datos específicos
        if "estudios" not in content.lower() and "investigación" not in content.lower():
            authority_phrases = [
                "Según estudios recientes,",
                "La investigación muestra que",
                "Los datos indican que",
                "Los expertos sugieren que"
            ]
            content = random.choice(authority_phrases) + " " + content
        
        return content
    
    async def _evaluate_content_quality(
        self,
        content: str,
        request: ContentRequest
    ) -> float:
        """Evaluar calidad del contenido generado"""
        # Métricas básicas de calidad
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        
        # Verificar si cumple con los requisitos
        meets_word_count = abs(word_count - request.word_count) / request.word_count < 0.2
        has_structure = "##" in content or len(sentence_count) > 3
        
        # Calcular score de calidad
        quality_score = 0.0
        quality_score += 0.4 if meets_word_count else 0.0
        quality_score += 0.3 if has_structure else 0.0
        quality_score += 0.3 if len(content) > 100 else 0.0
        
        return quality_score
    
    async def get_content_summary(self) -> Dict[str, Any]:
        """Obtener resumen de contenido generado"""
        if not self.generated_content:
            return {"message": "No content generated yet"}
        
        # Estadísticas generales
        total_content = len(self.generated_content)
        content_types = Counter([content.request.content_type.value for content in self.generated_content.values()])
        quality_scores = [content.quality_score for content in self.generated_content.values()]
        
        return {
            "total_content_generated": total_content,
            "content_types": dict(content_types),
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "templates_available": len(self.templates),
            "last_generation": max([content.generated_at for content in self.generated_content.values()]).isoformat()
        }
    
    async def export_content(self, content_id: str, filepath: str = None) -> str:
        """Exportar contenido generado"""
        if content_id not in self.generated_content:
            raise ValueError(f"Content {content_id} not found")
        
        if filepath is None:
            filepath = f"exports/content_{content_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        content = self.generated_content[content_id]
        
        # Convertir a diccionario
        content_data = {
            "id": content.id,
            "request": {
                "topic": content.request.topic,
                "content_type": content.request.content_type.value,
                "style": content.request.style.value,
                "tone": content.request.tone.value,
                "target_audience": content.request.target_audience,
                "word_count": content.request.word_count,
                "keywords": content.request.keywords,
                "language": content.request.language
            },
            "content": content.content,
            "structure": content.structure,
            "metadata": content.metadata,
            "quality_score": content.quality_score,
            "generated_at": content.generated_at.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Content exported to {filepath}")
        return filepath


























