"""
AI Content Generator - Generador de Contenido con IA
=====================================================

Generador de contenido usando modelos de IA (OpenAI, Anthropic, etc.)
"""

import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class AIContentGenerator:
    """Generador de contenido usando IA"""
    
    def __init__(self, ai_provider: str = "openai", api_key: Optional[str] = None):
        """
        Inicializar generador de contenido con IA
        
        Args:
            ai_provider: Proveedor de IA (openai, anthropic, etc.)
            api_key: API key del proveedor
        """
        self.ai_provider = ai_provider
        self.api_key = api_key
        self._client = None
        logger.info(f"AI Content Generator inicializado con {ai_provider}")
    
    def _get_client(self):
        """Obtener cliente de IA (lazy loading)"""
        if self._client is None:
            if self.ai_provider == "openai":
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=self.api_key)
                except ImportError:
                    logger.warning("OpenAI no instalado, usando modo fallback")
                    self._client = None
            # Agregar más proveedores aquí
        
        return self._client
    
    def generate_post(
        self,
        topic: str,
        platform: str,
        tone: str = "professional",
        length: str = "medium",
        context: Optional[str] = None
    ) -> str:
        """
        Generar un post usando IA
        
        Args:
            topic: Tema del post
            platform: Plataforma objetivo
            tone: Tono (professional, casual, funny, etc.)
            length: Longitud (short, medium, long)
            context: Contexto adicional
            
        Returns:
            Contenido generado
        """
        client = self._get_client()
        
        if client is None:
            # Modo fallback sin IA
            return self._generate_fallback(topic, platform, tone, length)
        
        try:
            # Construir prompt
            prompt = self._build_prompt(topic, platform, tone, length, context)
            
            # Llamar a la API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en marketing en redes sociales."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self._get_max_tokens(length, platform),
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            logger.info(f"Contenido generado con IA para {platform}")
            return content
            
        except Exception as e:
            logger.error(f"Error generando contenido con IA: {e}")
            return self._generate_fallback(topic, platform, tone, length)
    
    def generate_caption(
        self,
        image_description: str,
        platform: str,
        hashtags: Optional[List[str]] = None
    ) -> str:
        """
        Generar caption para imagen usando IA
        
        Args:
            image_description: Descripción de la imagen
            platform: Plataforma objetivo
            hashtags: Hashtags opcionales
            
        Returns:
            Caption generado
        """
        client = self._get_client()
        
        if client is None:
            return self._generate_caption_fallback(image_description, platform, hashtags)
        
        try:
            prompt = f"""Genera un caption atractivo para {platform} basado en esta descripción de imagen:
            
{image_description}

El caption debe ser:
- Atractivo y engaging
- Apropiado para {platform}
- Incluir un call-to-action si es apropiado
"""
            
            if hashtags:
                prompt += f"\nIncluye estos hashtags: {', '.join(hashtags)}"
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en crear captions para redes sociales."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.8
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando caption con IA: {e}")
            return self._generate_caption_fallback(image_description, platform, hashtags)
    
    def generate_hashtags(
        self,
        content: str,
        platform: str,
        count: int = 10
    ) -> List[str]:
        """
        Generar hashtags relevantes usando IA
        
        Args:
            content: Contenido del post
            platform: Plataforma objetivo
            count: Número de hashtags
            
        Returns:
            Lista de hashtags
        """
        client = self._get_client()
        
        if client is None:
            return self._generate_hashtags_fallback(content, platform, count)
        
        try:
            prompt = f"""Genera {count} hashtags relevantes para este contenido en {platform}:
            
{content}

Los hashtags deben ser:
- Relevantes al contenido
- Populares en {platform}
- Sin el símbolo #
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en hashtags para redes sociales."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            hashtags_text = response.choices[0].message.content
            # Extraer hashtags del texto
            hashtags = [h.strip().replace("#", "") for h in hashtags_text.split() if h.strip()]
            
            return hashtags[:count]
            
        except Exception as e:
            logger.error(f"Error generando hashtags con IA: {e}")
            return self._generate_hashtags_fallback(content, platform, count)
    
    def _build_prompt(
        self,
        topic: str,
        platform: str,
        tone: str,
        length: str,
        context: Optional[str]
    ) -> str:
        """Construir prompt para IA"""
        length_descriptions = {
            "short": "corto (1-2 párrafos)",
            "medium": "medio (3-4 párrafos)",
            "long": "largo (5+ párrafos)"
        }
        
        prompt = f"""Crea un post para {platform} sobre: {topic}

Requisitos:
- Tono: {tone}
- Longitud: {length_descriptions.get(length, 'medio')}
- Optimizado para {platform}
- Engaging y atractivo
- Incluye un call-to-action si es apropiado
"""
        
        if context:
            prompt += f"\nContexto adicional: {context}"
        
        return prompt
    
    def _get_max_tokens(self, length: str, platform: str) -> int:
        """Obtener máximo de tokens según longitud"""
        limits = {
            "short": 150,
            "medium": 300,
            "long": 500
        }
        return limits.get(length, 300)
    
    def _generate_fallback(
        self,
        topic: str,
        platform: str,
        tone: str,
        length: str
    ) -> str:
        """Generar contenido sin IA (fallback)"""
        templates = {
            "short": f"📢 {topic} - ¡No te lo pierdas!",
            "medium": f"🎯 Hoy queremos hablar sobre {topic}. Es un tema importante que afecta a muchos. ¿Qué opinas?",
            "long": f"🌟 Hablemos sobre {topic}\n\nEste es un tema que nos parece muy relevante en la actualidad. Queremos compartir nuestra perspectiva y conocer la tuya.\n\n¿Has tenido alguna experiencia relacionada? Comparte en los comentarios."
        }
        return templates.get(length, templates["medium"])
    
    def _generate_caption_fallback(
        self,
        image_description: str,
        platform: str,
        hashtags: Optional[List[str]]
    ) -> str:
        """Generar caption sin IA (fallback)"""
        caption = f"✨ {image_description}\n\n"
        if hashtags:
            caption += " ".join([f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags])
        return caption
    
    def _generate_hashtags_fallback(
        self,
        content: str,
        platform: str,
        count: int
    ) -> List[str]:
        """Generar hashtags sin IA (fallback)"""
        from ..utils.content_optimizer import ContentOptimizer
        keywords = ContentOptimizer.extract_keywords(content, count)
        return keywords




