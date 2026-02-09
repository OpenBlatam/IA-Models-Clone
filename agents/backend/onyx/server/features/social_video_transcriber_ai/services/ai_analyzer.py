"""
AI Analyzer Service
Uses OpenRouter to analyze content framework and structure
"""

import json
import logging
from typing import Optional, Dict, Any, List

from ..config.settings import get_settings
from ..core.models import ContentAnalysis, ContentFramework
from .openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Service for AI-powered content analysis"""
    
    FRAMEWORK_ANALYSIS_PROMPT = """Analiza el siguiente texto transcrito de un video y determina:

1. **Framework de contenido**: ¿Qué estructura o framework de contenido utiliza? Las opciones son:
   - hook_story_offer: Gancho, Historia, Oferta
   - problem_agitate_solve: Problema, Agitación, Solución
   - aida: Atención, Interés, Deseo, Acción
   - star: Situación, Tarea, Acción, Resultado
   - bab: Antes, Después, Puente
   - educational: Contenido educativo/informativo
   - storytelling: Narrativa/Historia
   - listicle: Lista de puntos/tips
   - tutorial: Paso a paso/Tutorial
   - review: Reseña/Review
   - news: Noticias/Actualidad
   - entertainment: Entretenimiento
   - motivational: Motivacional/Inspiracional
   - custom: Otro framework personalizado

2. **Estructura detallada**: Identifica las secciones del contenido con sus tiempos aproximados si los hay.

3. **Puntos clave**: Lista los principales puntos o ideas del contenido.

4. **Tono**: casual, profesional, humorístico, serio, inspiracional, etc.

5. **Audiencia objetivo**: ¿A quién va dirigido este contenido?

6. **Call to Action**: ¿Hay alguna llamada a la acción?

7. **Hashtags sugeridos**: Sugiere 5-10 hashtags relevantes.

TEXTO A ANALIZAR:
{text}

Responde ÚNICAMENTE en formato JSON válido con la siguiente estructura:
{{
    "framework": "nombre_del_framework",
    "framework_confidence": 0.0-1.0,
    "structure": {{
        "section_name": {{
            "text": "resumen de la sección",
            "purpose": "propósito de esta sección"
        }}
    }},
    "key_points": ["punto 1", "punto 2"],
    "tone": "tono detectado",
    "target_audience": "audiencia objetivo",
    "call_to_action": "CTA detectado o null",
    "hashtags_suggested": ["#hashtag1", "#hashtag2"],
    "content_type": "tipo de contenido",
    "language_detected": "código de idioma ISO 639-1"
}}"""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_openrouter_client()
    
    async def analyze_content(
        self,
        text: str,
        include_structure: bool = True,
    ) -> ContentAnalysis:
        """
        Analyze content to determine framework and structure
        
        Args:
            text: Transcribed text to analyze
            include_structure: Whether to include detailed structure analysis
            
        Returns:
            ContentAnalysis with framework and structure details
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for analysis")
        
        logger.info(f"Analyzing content ({len(text)} chars)")
        
        prompt = self.FRAMEWORK_ANALYSIS_PROMPT.format(text=text)
        
        try:
            response = await self.client.complete(
                prompt=prompt,
                system_prompt="Eres un experto en análisis de contenido y copywriting. Analiza contenido de redes sociales y determina su estructura y framework. Responde siempre en JSON válido.",
                max_tokens=self.settings.analysis_max_tokens,
                temperature=0.3,  # Lower temperature for more consistent analysis
            )
            
            # Parse JSON response
            analysis_data = self._parse_json_response(response)
            
            # Calculate word count and reading time
            word_count = len(text.split())
            reading_time = word_count / 200  # Average reading speed
            
            # Map framework string to enum
            framework_str = analysis_data.get('framework', 'custom').lower()
            try:
                framework = ContentFramework(framework_str)
            except ValueError:
                framework = ContentFramework.CUSTOM
            
            analysis = ContentAnalysis(
                framework=framework,
                framework_confidence=float(analysis_data.get('framework_confidence', 0.5)),
                structure=analysis_data.get('structure', {}),
                key_points=analysis_data.get('key_points', []),
                tone=analysis_data.get('tone', 'neutral'),
                target_audience=analysis_data.get('target_audience'),
                call_to_action=analysis_data.get('call_to_action'),
                hashtags_suggested=analysis_data.get('hashtags_suggested', []),
                content_type=analysis_data.get('content_type', 'general'),
                language_detected=analysis_data.get('language_detected', 'es'),
                word_count=word_count,
                estimated_reading_time=reading_time,
            )
            
            logger.info(f"Analysis complete: {framework.value} ({analysis.framework_confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            # Return basic analysis on failure
            return ContentAnalysis(
                framework=ContentFramework.CUSTOM,
                framework_confidence=0.0,
                word_count=len(text.split()),
                estimated_reading_time=len(text.split()) / 200,
            )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response, handling markdown code blocks"""
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            # Remove first and last lines (```json and ```)
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}
    
    async def suggest_improvements(
        self,
        text: str,
        analysis: ContentAnalysis,
    ) -> List[str]:
        """
        Suggest improvements based on content analysis
        
        Args:
            text: Original text
            analysis: Content analysis results
            
        Returns:
            List of improvement suggestions
        """
        prompt = f"""Basándote en el siguiente análisis de contenido, sugiere 3-5 mejoras específicas y accionables:

TEXTO ORIGINAL:
{text}

ANÁLISIS:
- Framework: {analysis.framework.value}
- Tono: {analysis.tone}
- Puntos clave: {', '.join(analysis.key_points[:3]) if analysis.key_points else 'N/A'}

Proporciona sugerencias concretas para mejorar:
1. Engagement inicial (hook)
2. Claridad del mensaje
3. Call to action
4. Estructura general

Responde con una lista de sugerencias, una por línea."""

        try:
            response = await self.client.complete(
                prompt=prompt,
                system_prompt="Eres un experto en marketing de contenido y copywriting para redes sociales.",
                max_tokens=1000,
                temperature=0.7,
            )
            
            # Parse suggestions from response
            suggestions = [
                line.strip().lstrip('0123456789.-) ')
                for line in response.split('\n')
                if line.strip() and len(line.strip()) > 10
            ]
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate improvements: {e}")
            return []


_ai_analyzer: Optional[AIAnalyzer] = None


def get_ai_analyzer() -> AIAnalyzer:
    """Get AI analyzer singleton"""
    global _ai_analyzer
    if _ai_analyzer is None:
        _ai_analyzer = AIAnalyzer()
    return _ai_analyzer












