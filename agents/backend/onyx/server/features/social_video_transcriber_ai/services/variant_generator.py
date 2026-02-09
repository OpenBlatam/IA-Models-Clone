"""
Variant Generator Service
Generates text variants maintaining context, structure, and length
"""

import json
import logging
from typing import Optional, List, Dict, Any
from uuid import uuid4

from ..config.settings import get_settings
from ..core.models import TextVariant, ContentAnalysis
from .openrouter_client import get_openrouter_client
from .ai_analyzer import get_ai_analyzer

logger = logging.getLogger(__name__)


class VariantGenerator:
    """Service for generating text variants"""
    
    VARIANT_GENERATION_PROMPT = """Genera {num_variants} variantes del siguiente texto.

REQUISITOS:
- Mantener el MISMO significado y mensaje principal
- {structure_instruction}
- {length_instruction}
- {tone_instruction}
{custom_instructions}

TEXTO ORIGINAL:
{text}

{analysis_context}

Para cada variante, proporciona:
1. El texto completo de la variante
2. Una puntuación de similitud (0.0-1.0) indicando qué tan similar es al original
3. El tipo de cambio realizado

Responde ÚNICAMENTE en formato JSON válido:
{{
    "variants": [
        {{
            "variant_text": "texto de la variante",
            "similarity_score": 0.85,
            "variant_type": "rewrite",
            "preserves_structure": true,
            "preserves_length": true
        }}
    ]
}}"""

    QUICK_VARIANT_PROMPT = """Genera 3 variantes rápidas del siguiente texto, manteniendo:
- El mismo mensaje y significado
- Una longitud similar (±20%)
- La misma estructura general

TEXTO:
{text}

Responde en JSON:
{{
    "variants": [
        {{
            "variant_text": "variante 1",
            "similarity_score": 0.9
        }},
        {{
            "variant_text": "variante 2", 
            "similarity_score": 0.85
        }},
        {{
            "variant_text": "variante 3",
            "similarity_score": 0.8
        }}
    ]
}}"""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_openrouter_client()
        self.analyzer = get_ai_analyzer()
    
    async def generate_variants(
        self,
        text: str,
        num_variants: int = 3,
        preserve_structure: bool = True,
        preserve_length: bool = True,
        target_tone: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        analysis: Optional[ContentAnalysis] = None,
    ) -> List[TextVariant]:
        """
        Generate text variants maintaining specified constraints
        
        Args:
            text: Original text
            num_variants: Number of variants to generate
            preserve_structure: Whether to maintain the same structure
            preserve_length: Whether to maintain similar length
            target_tone: Target tone for variants
            custom_instructions: Additional instructions
            analysis: Pre-computed content analysis
            
        Returns:
            List of TextVariant objects
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for variant generation")
        
        logger.info(f"Generating {num_variants} variants ({len(text)} chars)")
        
        # Build instruction strings
        structure_instruction = (
            "Mantener la MISMA estructura (secciones, orden de ideas)"
            if preserve_structure else
            "Puedes reorganizar la estructura si mejora el mensaje"
        )
        
        length_instruction = (
            f"Mantener una longitud similar al original (~{len(text.split())} palabras, ±20%)"
            if preserve_length else
            "La longitud puede variar según sea necesario"
        )
        
        tone_instruction = (
            f"Usar un tono {target_tone}"
            if target_tone else
            "Mantener el tono original"
        )
        
        custom_inst = f"- {custom_instructions}" if custom_instructions else ""
        
        # Build analysis context if available
        analysis_context = ""
        if analysis:
            analysis_context = f"""
CONTEXTO DEL ANÁLISIS:
- Framework detectado: {analysis.framework.value}
- Tono original: {analysis.tone}
- Puntos clave: {', '.join(analysis.key_points[:3]) if analysis.key_points else 'N/A'}
"""
        
        prompt = self.VARIANT_GENERATION_PROMPT.format(
            num_variants=num_variants,
            structure_instruction=structure_instruction,
            length_instruction=length_instruction,
            tone_instruction=tone_instruction,
            custom_instructions=custom_inst,
            text=text,
            analysis_context=analysis_context,
        )
        
        try:
            response = await self.client.complete(
                prompt=prompt,
                system_prompt="Eres un experto copywriter especializado en crear variantes de texto que mantienen el mensaje original mientras mejoran el engagement. Siempre respondes en JSON válido.",
                max_tokens=self.settings.variant_max_tokens * num_variants,
                temperature=self.settings.temperature,
            )
            
            # Parse response
            variants_data = self._parse_json_response(response)
            variants_list = variants_data.get('variants', [])
            
            # Convert to TextVariant objects
            variants = []
            for v_data in variants_list[:num_variants]:
                variant = TextVariant(
                    id=uuid4(),
                    original_text=text,
                    variant_text=v_data.get('variant_text', ''),
                    variant_type=v_data.get('variant_type', 'rewrite'),
                    similarity_score=float(v_data.get('similarity_score', 0.8)),
                    preserves_structure=v_data.get('preserves_structure', preserve_structure),
                    preserves_length=v_data.get('preserves_length', preserve_length),
                    tone=target_tone,
                    word_count=len(v_data.get('variant_text', '').split()),
                )
                variants.append(variant)
            
            logger.info(f"Generated {len(variants)} variants successfully")
            return variants
            
        except Exception as e:
            logger.error(f"Variant generation failed: {e}")
            raise
    
    async def generate_quick_variant(
        self,
        text: str,
    ) -> List[TextVariant]:
        """
        Quick one-click variant generation with default settings
        
        Args:
            text: Original text
            
        Returns:
            List of 3 TextVariant objects
        """
        logger.info(f"Quick variant generation ({len(text)} chars)")
        
        prompt = self.QUICK_VARIANT_PROMPT.format(text=text)
        
        try:
            response = await self.client.complete(
                prompt=prompt,
                system_prompt="Eres un copywriter experto. Genera variantes de texto manteniendo el mensaje. Responde solo en JSON.",
                max_tokens=self.settings.variant_max_tokens * 3,
                temperature=0.8,
            )
            
            variants_data = self._parse_json_response(response)
            variants_list = variants_data.get('variants', [])
            
            variants = []
            for v_data in variants_list[:3]:
                variant = TextVariant(
                    id=uuid4(),
                    original_text=text,
                    variant_text=v_data.get('variant_text', ''),
                    variant_type='quick_rewrite',
                    similarity_score=float(v_data.get('similarity_score', 0.85)),
                    preserves_structure=True,
                    preserves_length=True,
                    word_count=len(v_data.get('variant_text', '').split()),
                )
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Quick variant generation failed: {e}")
            raise
    
    async def generate_tone_variant(
        self,
        text: str,
        target_tone: str,
    ) -> TextVariant:
        """
        Generate a variant with a specific tone
        
        Args:
            text: Original text
            target_tone: Target tone (casual, professional, humorous, etc.)
            
        Returns:
            Single TextVariant with new tone
        """
        prompt = f"""Reescribe el siguiente texto con un tono {target_tone}, manteniendo exactamente el mismo mensaje y estructura.

TEXTO ORIGINAL:
{text}

Responde en JSON:
{{
    "variant_text": "texto reescrito con tono {target_tone}",
    "similarity_score": 0.9
}}"""

        try:
            response = await self.client.complete(
                prompt=prompt,
                system_prompt=f"Eres un experto en adaptar textos a diferentes tonos. Tu especialidad es el tono {target_tone}.",
                max_tokens=self.settings.variant_max_tokens,
                temperature=0.7,
            )
            
            data = self._parse_json_response(response)
            
            return TextVariant(
                id=uuid4(),
                original_text=text,
                variant_text=data.get('variant_text', text),
                variant_type='tone_change',
                similarity_score=float(data.get('similarity_score', 0.85)),
                preserves_structure=True,
                preserves_length=True,
                tone=target_tone,
                word_count=len(data.get('variant_text', '').split()),
            )
            
        except Exception as e:
            logger.error(f"Tone variant generation failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response"""
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {'variants': []}


_variant_generator: Optional[VariantGenerator] = None


def get_variant_generator() -> VariantGenerator:
    """Get variant generator singleton"""
    global _variant_generator
    if _variant_generator is None:
        _variant_generator = VariantGenerator()
    return _variant_generator












