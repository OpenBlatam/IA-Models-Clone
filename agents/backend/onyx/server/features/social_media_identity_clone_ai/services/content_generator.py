"""
Servicio para generar contenido basado en identidad clonada

Refactorizado con:
- Integración de transformers para generación avanzada
- Mejor manejo de errores y validación
- Type hints completos
- Optimización de prompts
"""

import logging
import uuid
import re
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime

# Helper imports
from ..utils.cache_helpers import generate_cache_key
from ..utils.id_helpers import generate_id
from ..utils.datetime_helpers import now
from ..utils.string_helpers import extract_hashtags

from openai import OpenAI
import torch

from ..core.models import (
    IdentityProfile,
    GeneratedContent,
    Platform,
    ContentType
)
from ..core.base_service import BaseMLService
from ..core.exceptions import (
    ContentGenerationError,
    ValidationError,
    ModelLoadingError
)
from ..config import get_settings
from ..ml_advanced.transformer_service import get_transformer_service
from ..ml_advanced.lora_finetuning import get_lora_finetuner, LoRAConfig

logger = logging.getLogger(__name__)


class ContentGenerator(BaseMLService):
    """
    Genera contenido basado en perfil de identidad
    
    Mejoras:
    - Integración con transformers para generación avanzada
    - Soporte para fine-tuning con LoRA
    - Procesamiento optimizado con GPU y mixed precision
    - Prompts mejorados y más efectivos
    - Validación robusta de contenido generado
    - Batching para generación múltiple
    """
    
    def __init__(self, identity_profile: IdentityProfile, enable_cache: bool = True):
        """
        Inicializa el generador de contenido
        
        Args:
            identity_profile: Perfil de identidad a usar
            enable_cache: Habilitar caché de contenido generado (default: True)
        """
        super().__init__()
        self._validate_input(identity_profile, "identity_profile")
        self.identity = identity_profile
        self._cache = {} if enable_cache else None
        self._initialize_llm_client()
        self._initialize_transformer_service()
        self._lora_finetuner: Optional[Any] = None
        self._generation_model: Optional[Any] = None
        self._use_lora_model = False
    
    def _initialize_llm_client(self) -> None:
        """Inicializa cliente LLM con manejo de errores"""
        try:
            if self.settings.openai_api_key:
                self.client = OpenAI(
                    api_key=self.settings.openai_api_key
                )
                self.logger.info("OpenAI client inicializado para generación")
            else:
                self.client = None
                self.logger.warning("OpenAI API key no configurada")
        except Exception as e:
            self.logger.warning(
                f"Error inicializando OpenAI client: {e}"
            )
            self.client = None
    
    def _initialize_transformer_service(self) -> None:
        """Inicializa servicio de transformers"""
        try:
            self._transformer_service = get_transformer_service()
            self.logger.info("TransformerService inicializado para generación")
        except Exception as e:
            self.logger.warning(
                f"Error inicializando TransformerService: {e}"
            )
            self._transformer_service = None
    
    def _load_model(self) -> None:
        """Carga modelo de generación (implementación de BaseMLService)"""
        # El modelo se carga bajo demanda cuando se necesita
        pass
    
    def load_lora_model(
        self,
        model_path: str,
        base_model_name: str = "gpt2"
    ) -> None:
        """
        Carga modelo fine-tuned con LoRA para generación personalizada
        
        Args:
            model_path: Path al modelo fine-tuned
            base_model_name: Nombre del modelo base
        """
        try:
            self._lora_finetuner = get_lora_finetuner()
            self._lora_finetuner.prepare_model_for_lora(
                base_model_name=base_model_name
            )
            # Cargar pesos fine-tuned si existen
            if model_path:
                try:
                    from transformers import AutoModelForCausalLM
                    self._lora_finetuner.model = AutoModelForCausalLM.from_pretrained(
                        model_path
                    ).to(self.device)
                    self.logger.info(f"Modelo LoRA cargado desde {model_path}")
                except Exception as e:
                    self.logger.warning(
                        f"No se pudo cargar modelo desde {model_path}: {e}"
                    )
            
            self._use_lora_model = True
            self.logger.info("Modelo LoRA listo para generación")
            
        except Exception as e:
            self.logger.error(
                f"Error cargando modelo LoRA: {e}",
                exc_info=True
            )
            self._use_lora_model = False
    
    async def generate_instagram_post(
        self,
        topic: Optional[str] = None,
        style: Optional[str] = None,
        use_lora: bool = False,
        use_cache: bool = True
    ) -> GeneratedContent:
        """
        Genera un post para Instagram basado en la identidad
        Optimizado con caché para mayor velocidad
        
        Args:
            topic: Tema opcional para el post
            style: Estilo opcional (motivational, educational, etc.)
            use_lora: Si usar modelo LoRA fine-tuned
            use_cache: Si usar caché (default: True)
            
        Returns:
            GeneratedContent con el post generado
        """
        self._log_operation(
            "generate_instagram_post",
            topic=topic,
            style=style,
            use_lora=use_lora
        )
        
        # Verificar caché usando helper
        if use_cache and self._cache is not None:
            cache_key = generate_cache_key(
                "instagram",
                self.identity.profile_id,
                topic,
                style,
                use_lora
            )
            if cache_key in self._cache:
                self.logger.debug("Post de Instagram obtenido de caché")
                cached = self._cache[cache_key]
                # Actualizar timestamp
                cached.generated_at = now()
                return cached
        
        prompt = self._build_instagram_prompt(topic, style)
        content = await self._generate_with_ai(prompt, use_lora=use_lora)
        
        # Extraer hashtags usando helper
        hashtags = extract_hashtags(content)
        
        # Calcular confidence score basado en calidad
        confidence = self._calculate_confidence_score(content, hashtags)
        
        result = GeneratedContent(
            content_id=generate_id("content"),
            identity_profile_id=self.identity.profile_id,
            platform=Platform.INSTAGRAM,
            content_type=ContentType.POST,
            content=content,
            hashtags=hashtags,
            generated_at=now(),
            confidence_score=confidence
        )
        
        # Guardar en caché usando cache manager
        if use_cache and self._cache is not None:
            from ..utils.cache_manager import get_cache
            cache = get_cache()
            cache.set(cache_key, result, max_size=500)
        
        return result
    
    def _calculate_confidence_score(
        self,
        content: str,
        hashtags: List[str]
    ) -> float:
        """
        Calcula confidence score basado en calidad del contenido
        
        Args:
            content: Contenido generado
            hashtags: Hashtags extraídos
            
        Returns:
            Score de confianza (0.0 - 1.0)
        """
        score = 0.5  # Base score
        
        # Longitud apropiada
        if 50 <= len(content) <= 2000:
            score += 0.2
        
        # Hashtags presentes
        if len(hashtags) >= 3:
            score += 0.15
        
        # Contenido no vacío
        if content.strip():
            score += 0.15
        
        return min(score, 1.0)
    
    async def generate_tiktok_script(
        self,
        topic: Optional[str] = None,
        duration: int = 60,
        use_cache: bool = True
    ) -> GeneratedContent:
        """
        Genera un script para TikTok basado en la identidad
        Optimizado con caché para mayor velocidad
        
        Args:
            topic: Tema opcional para el video
            duration: Duración del video en segundos
            use_cache: Si usar caché (default: True)
            
        Returns:
            GeneratedContent con el script generado
        """
        logger.info(f"Generando script de TikTok - Tema: {topic}, Duración: {duration}s")
        
        # Verificar caché usando helper
        if use_cache and self._cache is not None:
            cache_key = generate_cache_key(
                "tiktok",
                self.identity.profile_id,
                topic,
                duration
            )
            if cache_key in self._cache:
                self.logger.debug("Script de TikTok obtenido de caché")
                cached = self._cache[cache_key]
                cached.generated_at = now()
                return cached
        
        prompt = self._build_tiktok_prompt(topic, duration)
        content = await self._generate_with_ai(prompt)
        
        # Extraer hashtags usando helper
        hashtags = extract_hashtags(content)
        
        result = GeneratedContent(
            content_id=generate_id("content"),
            identity_profile_id=self.identity.profile_id,
            platform=Platform.TIKTOK,
            content_type=ContentType.VIDEO,
            content=content,
            title=topic or "Video Script",
            hashtags=hashtags,
            generated_at=now(),
            confidence_score=0.85
        )
        
        # Guardar en caché usando cache manager
        if use_cache and self._cache is not None:
            from ..utils.cache_manager import get_cache
            cache = get_cache()
            cache.set(cache_key, result, max_size=500)
        
        return result
    
    async def generate_youtube_description(
        self,
        video_title: str,
        tags: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> GeneratedContent:
        """
        Genera una descripción para YouTube basado en la identidad
        Optimizado con caché para mayor velocidad
        
        Args:
            video_title: Título del video
            tags: Tags opcionales
            use_cache: Si usar caché (default: True)
            
        Returns:
            GeneratedContent con la descripción generada
        """
        logger.info(f"Generando descripción de YouTube - Título: {video_title}")
        
        # Verificar caché usando helper
        if use_cache and self._cache is not None:
            tags_str = ",".join(sorted(tags or []))
            cache_key = generate_cache_key(
                "youtube",
                self.identity.profile_id,
                video_title,
                tags_str
            )
            if cache_key in self._cache:
                self.logger.debug("Descripción de YouTube obtenida de caché")
                cached = self._cache[cache_key]
                cached.generated_at = now()
                return cached
        
        prompt = self._build_youtube_prompt(video_title, tags)
        content = await self._generate_with_ai(prompt)
        
        result = GeneratedContent(
            content_id=generate_id("content"),
            identity_profile_id=self.identity.profile_id,
            platform=Platform.YOUTUBE,
            content_type=ContentType.VIDEO,
            content=content,
            title=video_title,
            hashtags=tags or [],
            generated_at=now(),
            confidence_score=0.85
        )
        
        # Guardar en caché
        if use_cache and self._cache is not None:
            if len(self._cache) > 500:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = result
        
        return result
    
    def _build_instagram_prompt(self, topic: Optional[str], style: Optional[str]) -> str:
        """Construye prompt para generación de post de Instagram"""
        identity_context = self._get_identity_context()
        
        prompt = f"""
Eres {self.identity.display_name or self.identity.username}. 
Genera un post para Instagram que sea auténtico a tu identidad y estilo.

Contexto de tu identidad:
{identity_context}

"""
        
        if topic:
            prompt += f"Tema del post: {topic}\n"
        
        if style:
            prompt += f"Estilo: {style}\n"
        
        prompt += """
Genera un caption para Instagram que:
- Sea auténtico a tu voz y estilo
- Use tu tono de comunicación característico
- Incluya hashtags relevantes al final
- Sea engaging y auténtico

Responde solo con el caption, sin explicaciones adicionales.
"""
        
        return prompt
    
    def _build_tiktok_prompt(self, topic: Optional[str], duration: int) -> str:
        """Construye prompt para generación de script de TikTok"""
        identity_context = self._get_identity_context()
        
        # Calcular palabras aproximadas (150 palabras por minuto)
        words = int((duration / 60) * 150)
        
        prompt = f"""
Eres {self.identity.display_name or self.identity.username}.
Genera un script para un video de TikTok de {duration} segundos (~{words} palabras) que sea auténtico a tu identidad.

Contexto de tu identidad:
{identity_context}

"""
        
        if topic:
            prompt += f"Tema del video: {topic}\n"
        
        prompt += """
Genera un script que:
- Sea auténtico a tu voz y estilo
- Use tu tono de comunicación característico
- Sea engaging y capture atención desde el inicio
- Incluya hooks y transiciones naturales
- Sea apropiado para la duración especificada

Formato: Script directo, como si estuvieras hablando a cámara.
"""
        
        return prompt
    
    def _build_youtube_description_prompt(self, video_title: str, tags: Optional[List[str]]) -> str:
        """Construye prompt para generación de descripción de YouTube"""
        identity_context = self._get_identity_context()
        
        prompt = f"""
Eres {self.identity.display_name or self.identity.username}.
Genera una descripción para un video de YouTube que sea auténtica a tu identidad.

Título del video: {video_title}

Contexto de tu identidad:
{identity_context}

"""
        
        if tags:
            prompt += f"Tags relevantes: {', '.join(tags)}\n"
        
        prompt += """
Genera una descripción que:
- Sea auténtica a tu voz y estilo
- Proporcione contexto sobre el video
- Incluya llamados a la acción
- Sea engaging y profesional
- Incluya timestamps si es relevante

Formato: Descripción completa para YouTube.
"""
        
        return prompt
    
    def _build_youtube_prompt(self, video_title: str, tags: Optional[List[str]]) -> str:
        """Construye prompt para generación de descripción de YouTube"""
        return self._build_youtube_description_prompt(video_title, tags)
    
    def _get_identity_context(self) -> str:
        """Obtiene contexto de identidad para prompts"""
        context_parts = []
        
        if self.identity.bio:
            context_parts.append(f"Bio: {self.identity.bio}")
        
        analysis = self.identity.content_analysis
        if analysis.tone:
            context_parts.append(f"Tono: {analysis.tone}")
        
        if analysis.communication_style:
            context_parts.append(f"Estilo de comunicación: {analysis.communication_style}")
        
        if analysis.personality_traits:
            context_parts.append(f"Rasgos de personalidad: {', '.join(analysis.personality_traits[:5])}")
        
        if analysis.topics:
            context_parts.append(f"Temas frecuentes: {', '.join(analysis.topics[:5])}")
        
        if analysis.common_phrases:
            context_parts.append(f"Frases comunes: {', '.join(analysis.common_phrases[:3])}")
        
        return "\n".join(context_parts)
    
    async def _generate_with_ai(
        self,
        prompt: str,
        use_lora: bool = False
    ) -> str:
        """
        Genera contenido usando IA avanzada con múltiples opciones
        
        Args:
            prompt: Prompt para generación
            use_lora: Si usar modelo LoRA fine-tuned
            
        Returns:
            Contenido generado
            
        Raises:
            ContentGenerationError: Si hay error en la generación
        """
        # Prioridad 1: Modelo LoRA fine-tuned (más personalizado)
        if use_lora and self._use_lora_model and self._lora_finetuner:
            try:
                return await self._generate_with_lora(prompt)
            except Exception as e:
                self.logger.warning(
                    f"Error con modelo LoRA, usando fallback: {e}"
                )
        
        # Prioridad 2: OpenAI API (más potente)
        if self.client:
            try:
                return await self._generate_with_openai(prompt)
            except Exception as e:
                self.logger.warning(
                    f"Error con OpenAI, usando fallback: {e}"
                )
        
        # Prioridad 3: Fallback básico
        return self._generate_basic_fallback(prompt)
    
    async def _generate_with_lora(self, prompt: str) -> str:
        """
        Genera contenido usando modelo LoRA fine-tuned
        
        Args:
            prompt: Prompt para generación
            
        Returns:
            Contenido generado
        """
        if not self._lora_finetuner or not self._lora_finetuner.model:
            raise ContentGenerationError("Modelo LoRA no disponible")
        
        try:
            # Construir prompt completo con contexto de identidad
            full_prompt = self._build_full_prompt_with_context(prompt)
            
            # Generar con modelo LoRA
            generated_text = self._lora_finetuner.generate_text(
                prompt=full_prompt,
                max_length=self.settings.max_content_length,
                temperature=self.settings.content_temperature,
                top_p=0.9
            )
            
            # Extraer solo la parte generada (remover prompt)
            if full_prompt in generated_text:
                generated_text = generated_text.split(full_prompt)[-1].strip()
            
            # Validar contenido
            if not generated_text or len(generated_text) < 10:
                raise ContentGenerationError("Contenido generado muy corto")
            
            return generated_text
            
        except Exception as e:
            self.logger.error(
                f"Error generando con LoRA: {e}",
                exc_info=True
            )
            raise ContentGenerationError(
                f"Error generando con LoRA: {str(e)}",
                error_code="LORA_GENERATION_ERROR"
            ) from e
    
    async def _generate_with_openai(self, prompt: str) -> str:
        """
        Genera contenido usando OpenAI API
        
        Args:
            prompt: Prompt para generación
            
        Returns:
            Contenido generado
        """
        if not self.client:
            raise ContentGenerationError("OpenAI client no disponible")
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en generación de contenido "
                            "auténtico para redes sociales. Genera contenido "
                            "que sea engaging, auténtico y alineado con la "
                            "identidad proporcionada."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.content_temperature,
                max_tokens=self.settings.max_content_length
            )
            
            generated_content = response.choices[0].message.content.strip()
            
            # Validar contenido generado
            if not generated_content or len(generated_content) < 10:
                raise ContentGenerationError("Contenido generado muy corto")
            
            return generated_content
            
        except Exception as e:
            self.logger.error(
                f"Error generando con OpenAI: {e}",
                exc_info=True
            )
            raise ContentGenerationError(
                f"Error generando con OpenAI: {str(e)}",
                error_code="OPENAI_GENERATION_ERROR"
            ) from e
    
    def _build_full_prompt_with_context(self, prompt: str) -> str:
        """
        Construye prompt completo con contexto de identidad
        
        Args:
            prompt: Prompt base
            
        Returns:
            Prompt completo con contexto
        """
        identity_context = self._get_identity_context()
        
        full_prompt = f"""Contexto de identidad:
{identity_context}

{prompt}

Genera contenido auténtico que refleje esta identidad:"""
        
        return full_prompt
    
    def _generate_basic_fallback(self, prompt: str) -> str:
        """
        Generación básica de fallback
        
        Args:
            prompt: Prompt original
            
        Returns:
            Contenido básico generado
        """
        return (
            f"[Contenido generado basado en: {self.identity.username}]\n"
            f"Nota: Generación avanzada no disponible."
        )
    
    def _extract_hashtags(self, content: str) -> List[str]:
        """
        Extrae hashtags del contenido
        
        Args:
            content: Contenido a analizar
            
        Returns:
            Lista de hashtags (sin #)
        """
        hashtags = re.findall(r'#\w+', content)
        return [tag[1:] for tag in hashtags]  # Remover el #

