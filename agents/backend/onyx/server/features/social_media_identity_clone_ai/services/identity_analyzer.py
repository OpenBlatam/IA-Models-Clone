"""
Servicio para analizar contenido y construir perfil de identidad

Refactorizado con:
- Integración avanzada de transformers
- Mejor manejo de errores
- Type hints completos
- Optimización de procesamiento
"""

import logging
import uuid
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from openai import OpenAI
import torch
import numpy as np

from ..core.models import (
    SocialProfile,
    IdentityProfile,
    ContentAnalysis,
    Platform
)
from ..core.base_service import BaseService
from ..core.exceptions import (
    IdentityAnalysisError,
    ModelLoadingError,
    ValidationError
)
from ..config import get_settings
from ..utils.text_processor import TextProcessor
from ..ml_advanced.transformer_service import get_transformer_service

logger = logging.getLogger(__name__)


class IdentityAnalyzer(BaseService):
    """
    Analiza contenido y construye perfil de identidad
    
    Mejoras:
    - Integración con transformers para análisis avanzado
    - Procesamiento optimizado con GPU
    - Manejo robusto de errores
    - Análisis más profundo y preciso
    """
    
    def __init__(self, enable_cache: bool = True):
        """
        Inicializa el analizador de identidad
        
        Args:
            enable_cache: Habilitar caché de análisis (default: True)
        """
        super().__init__()
        self.text_processor = TextProcessor(enable_cache=enable_cache)
        self._cache = {} if enable_cache else None
        self._initialize_llm_client()
        self._transformer_service = None
        self._initialize_transformer_service()
    
    def _initialize_transformer_service(self) -> None:
        """Inicializa servicio de transformers"""
        try:
            self._transformer_service = get_transformer_service()
            self.logger.info("TransformerService inicializado")
        except Exception as e:
            self.logger.warning(
                f"Error inicializando TransformerService: {e}"
            )
            self._transformer_service = None
    
    def _initialize_llm_client(self) -> None:
        """Inicializa cliente LLM con manejo de errores"""
        try:
            if self.settings.openai_api_key:
                self.client = OpenAI(
                    api_key=self.settings.openai_api_key
                )
                self.logger.info("OpenAI client inicializado")
            else:
                self.client = None
                self.logger.warning("OpenAI API key no configurada")
        except Exception as e:
            self.logger.warning(f"Error inicializando OpenAI client: {e}")
            self.client = None
    
    async def build_identity(
        self,
        tiktok_profile: Optional[SocialProfile] = None,
        instagram_profile: Optional[SocialProfile] = None,
        youtube_profile: Optional[SocialProfile] = None,
        use_cache: bool = True
    ) -> IdentityProfile:
        """
        Construye un perfil de identidad completo a partir de perfiles de redes sociales
        Optimizado con caché y procesamiento paralelo
        
        Args:
            tiktok_profile: Perfil de TikTok
            instagram_profile: Perfil de Instagram
            youtube_profile: Perfil de YouTube
            use_cache: Si usar caché (default: True)
            
        Returns:
            IdentityProfile completo
        """
        logger.info("Construyendo perfil de identidad...")
        
        # Generar clave de caché usando helper
        if use_cache and self._cache is not None:
            from ..utils.cache_helpers import generate_cache_key
            from ..utils.condition_helpers import coalesce
            
            cache_key = generate_cache_key(
                "identity",
                coalesce(
                    tiktok_profile.username if tiktok_profile else None,
                    instagram_profile.username if instagram_profile else None,
                    youtube_profile.username if youtube_profile else None,
                    default=""
                )
            )
            
            if cache_key in self._cache:
                logger.info("Perfil de identidad obtenido de caché")
                return self._cache[cache_key]
        
        # Consolidar contenido (paralelo si es posible)
        all_content = self._consolidate_content(
            tiktok_profile, instagram_profile, youtube_profile
        )
        
        # Analizar contenido (puede ser paralelo)
        content_analysis = await self._analyze_content(all_content)
        
        # Construir base de conocimiento
        knowledge_base = await self._build_knowledge_base(all_content, content_analysis)
        
        # Determinar username principal
        username = self._determine_username(
            tiktok_profile, instagram_profile, youtube_profile
        )
        
        # Construir perfil de identidad usando helpers
        from ..utils.id_helpers import generate_id
        from ..utils.datetime_helpers import now
        from ..utils.dict_helpers import safe_get
        
        identity = IdentityProfile(
            profile_id=generate_id("identity"),
            username=username,
            display_name=self._determine_display_name(
                tiktok_profile, instagram_profile, youtube_profile
            ),
            bio=self._determine_bio(
                tiktok_profile, instagram_profile, youtube_profile
            ),
            tiktok_profile=tiktok_profile,
            instagram_profile=instagram_profile,
            youtube_profile=youtube_profile,
            content_analysis=content_analysis,
            knowledge_base=knowledge_base,
            total_videos=len(safe_get(all_content, "videos", default=[])),
            total_posts=len(safe_get(all_content, "posts", default=[])),
            total_comments=len(safe_get(all_content, "comments", default=[])),
            created_at=now(),
            updated_at=now()
        )
        
        # Guardar en caché
        if use_cache and self._cache is not None:
            if len(self._cache) > 100:  # Limitar tamaño
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = identity
        
        logger.info(f"Perfil de identidad construido: {identity.profile_id}")
        return identity
    
    def _consolidate_content(
        self,
        tiktok_profile: Optional[SocialProfile],
        instagram_profile: Optional[SocialProfile],
        youtube_profile: Optional[SocialProfile],
    ) -> dict:
        """Consolida contenido de todas las plataformas usando helpers"""
        from ..utils.data_consolidation_helpers import (
            consolidate_lists,
            extract_text_fields
        )
        
        # Consolidar videos
        all_videos = consolidate_lists(
            tiktok_profile.videos if tiktok_profile else None,
            youtube_profile.videos if youtube_profile else None
        )
        
        # Consolidar posts
        all_posts = consolidate_lists(
            instagram_profile.posts if instagram_profile else None
        )
        
        # Consolidar comments
        all_comments = consolidate_lists(
            instagram_profile.comments if instagram_profile else None
        )
        
        # Extraer textos de videos
        video_texts = extract_text_fields(
            all_videos,
            [
                lambda v: v.transcript,
                lambda v: v.description
            ]
        )
        
        # Extraer textos de posts
        post_texts = extract_text_fields(
            all_posts,
            [lambda p: p.caption]
        )
        
        # Combinar todos los textos
        all_texts = video_texts + post_texts
        
        return {
            "videos": all_videos,
            "posts": all_posts,
            "comments": all_comments,
            "texts": all_texts
        }
    
    async def _analyze_content(
        self,
        content: Dict[str, Any]
    ) -> ContentAnalysis:
        """
        Analiza contenido usando IA avanzada con transformers
        Optimizado con procesamiento paralelo y caché
        
        Args:
            content: Diccionario con contenido a analizar
            
        Returns:
            ContentAnalysis con resultados del análisis
        """
        all_text = "\n\n".join(content.get("texts", []))
        
        if not all_text or len(all_text) < 100:
            self.logger.warning(
                "Contenido insuficiente para análisis profundo"
            )
            return ContentAnalysis()
        
        # Verificar caché de análisis
        if self._cache is not None:
            import hashlib
            text_hash = hashlib.md5(all_text.encode()).hexdigest()
            cache_key = f"analysis_{text_hash}"
            if cache_key in self._cache:
                self.logger.debug("Análisis obtenido de caché")
                return self._cache[cache_key]
        
        # Procesar análisis en paralelo (transformers + LLM simultáneamente)
        import asyncio
        
        transformer_analysis = None
        llm_analysis = None
        
        # Crear tareas paralelas
        tasks = []
        
        if self._transformer_service:
            tasks.append(self._analyze_with_transformers(content.get("texts", [])))
        else:
            tasks.append(None)
        
        if self.client:
            tasks.append(self._analyze_with_llm(all_text))
        else:
            tasks.append(None)
        
        # Ejecutar en paralelo
        results = await asyncio.gather(*[task for task in tasks if task is not None], return_exceptions=True)
        
        # Procesar resultados
        result_idx = 0
        if self._transformer_service:
            if result_idx < len(results) and not isinstance(results[result_idx], Exception):
                transformer_analysis = results[result_idx]
            elif result_idx < len(results):
                self.logger.warning(f"Error en análisis transformers: {results[result_idx]}")
            result_idx += 1
        
        if self.client:
            if result_idx < len(results) and not isinstance(results[result_idx], Exception):
                llm_analysis = results[result_idx]
            elif result_idx < len(results):
                self.logger.warning(f"Error en análisis LLM: {results[result_idx]}")
        
        # Análisis básico rápido (siempre disponible)
        basic_analysis = self.text_processor.analyze_basic(all_text)
        
        # Combinar análisis
        final_analysis = self._combine_analyses(
            basic_analysis, transformer_analysis, llm_analysis
        )
        
        # Guardar en caché
        if self._cache is not None:
            if len(self._cache) > 200:  # Limitar tamaño
                # Eliminar análisis más antiguos
                keys_to_delete = [k for k in self._cache.keys() if k.startswith("analysis_")][:50]
                for key in keys_to_delete:
                    del self._cache[key]
            self._cache[cache_key] = final_analysis
        
        return final_analysis
    
    async def _analyze_with_transformers(
        self,
        texts: List[str]
    ) -> ContentAnalysis:
        """
        Analiza contenido usando transformers
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            ContentAnalysis con resultados
        """
        if not self._transformer_service:
            raise IdentityAnalysisError("TransformerService no disponible")
        
        try:
            # Análisis de estilo en batch
            style_analyses = self._transformer_service.analyze_text_style_batch(
                texts,
                batch_size=32
            )
            
            # Generar embeddings para análisis semántico
            embeddings = self._transformer_service.generate_embeddings(
                texts,
                use_cache=True,
                batch_size=32
            )
            
            # Extraer información de los análisis
            topics = self._extract_topics_from_embeddings(embeddings, texts)
            tone = self._determine_tone_from_styles(style_analyses)
            personality_traits = self._extract_personality_from_styles(
                style_analyses
            )
            
            return ContentAnalysis(
                topics=topics,
                themes=topics,  # Usar topics como themes inicialmente
                tone=tone,
                personality_traits=personality_traits,
                communication_style=tone,  # Usar tone como communication_style
                sentiment_analysis={
                    "positive": sum(
                        1 for s in style_analyses
                        if s.get("sentiment_score", 0) > 0.5
                    ) / len(style_analyses) if style_analyses else 0.0,
                    "negative": sum(
                        1 for s in style_analyses
                        if s.get("sentiment_score", 0) < 0.5
                    ) / len(style_analyses) if style_analyses else 0.0,
                    "neutral": 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error en análisis transformers: {e}",
                exc_info=True
            )
            raise IdentityAnalysisError(
                f"Error en análisis transformers: {str(e)}",
                error_code="TRANSFORMER_ANALYSIS_ERROR"
            ) from e
    
    def _extract_topics_from_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        n_topics: int = 5
    ) -> List[str]:
        """Extrae topics usando clustering de embeddings"""
        try:
            from sklearn.cluster import KMeans
            
            if len(embeddings) < n_topics:
                return []
            
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Obtener textos representativos de cada cluster
            topics = []
            for i in range(n_topics):
                cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
                if cluster_texts:
                    # Usar el texto más cercano al centroide
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(
                        embeddings[clusters == i] - centroid,
                        axis=1
                    )
                    closest_idx = np.argmin(distances)
                    topics.append(cluster_texts[closest_idx][:50])
            
            return topics
        except ImportError:
            return []
        except Exception as e:
            self.logger.warning(f"Error extrayendo topics: {e}")
            return []
    
    def _determine_tone_from_styles(
        self,
        style_analyses: List[Dict[str, Any]]
    ) -> str:
        """Determina tono general de los análisis de estilo"""
        if not style_analyses:
            return "neutral"
        
        avg_sentiment = np.mean([
            s.get("sentiment_score", 0.5) for s in style_analyses
        ])
        
        if avg_sentiment > 0.7:
            return "positive"
        elif avg_sentiment < 0.3:
            return "negative"
        else:
            return "neutral"
    
    def _extract_personality_from_styles(
        self,
        style_analyses: List[Dict[str, Any]]
    ) -> List[str]:
        """Extrae rasgos de personalidad de los análisis"""
        traits = []
        
        # Analizar características comunes
        has_emojis = sum(
            1 for s in style_analyses
            if s.get("features", {}).get("has_emojis", False)
        ) / len(style_analyses) if style_analyses else 0
        
        has_exclamations = sum(
            1 for s in style_analyses
            if s.get("features", {}).get("has_exclamations", False)
        ) / len(style_analyses) if style_analyses else 0
        
        if has_emojis > 0.3:
            traits.append("expressive")
        if has_exclamations > 0.3:
            traits.append("enthusiastic")
        
        return traits
    
    def _combine_analyses(
        self,
        basic_analysis: ContentAnalysis,
        transformer_analysis: Optional[ContentAnalysis] = None,
        llm_analysis: Optional[ContentAnalysis] = None
    ) -> ContentAnalysis:
        """
        Combina análisis de múltiples fuentes (básico, transformers, LLM)
        Optimizado: prioriza LLM > Transformers > Básico
        """
        # Empezar con análisis básico como base
        combined = ContentAnalysis(
            topics=basic_analysis.topics or [],
            themes=basic_analysis.themes or [],
            tone=basic_analysis.tone,
            personality_traits=basic_analysis.personality_traits or [],
            communication_style=basic_analysis.communication_style,
            common_phrases=basic_analysis.common_phrases or [],
            values=basic_analysis.values or [],
            interests=basic_analysis.interests or [],
            language_patterns=basic_analysis.language_patterns or {},
            sentiment_analysis=basic_analysis.sentiment_analysis or {}
        )
        
        # Enriquecer con transformers si está disponible
        if transformer_analysis:
            combined.topics = list(set(combined.topics + (transformer_analysis.topics or [])))
            combined.themes = list(set(combined.themes + (transformer_analysis.themes or [])))
            combined.tone = combined.tone or transformer_analysis.tone
            combined.personality_traits = list(set(
                combined.personality_traits + (transformer_analysis.personality_traits or [])
            ))
            combined.communication_style = combined.communication_style or transformer_analysis.communication_style
            combined.language_patterns = {**combined.language_patterns, **(transformer_analysis.language_patterns or {})}
            if transformer_analysis.sentiment_analysis:
                combined.sentiment_analysis = transformer_analysis.sentiment_analysis
        
        # Priorizar LLM (más preciso) si está disponible
        if llm_analysis:
            combined.topics = llm_analysis.topics or combined.topics
            combined.themes = llm_analysis.themes or combined.themes
            combined.tone = llm_analysis.tone or combined.tone
            combined.personality_traits = llm_analysis.personality_traits or combined.personality_traits
            combined.communication_style = llm_analysis.communication_style or combined.communication_style
            combined.common_phrases = llm_analysis.common_phrases or combined.common_phrases
            combined.values = llm_analysis.values or combined.values
            combined.interests = llm_analysis.interests or combined.interests
            combined.language_patterns = {**combined.language_patterns, **(llm_analysis.language_patterns or {})}
            combined.sentiment_analysis = llm_analysis.sentiment_analysis or combined.sentiment_analysis
        
        return combined
    
    async def _analyze_with_llm(self, text: str) -> ContentAnalysis:
        """
        Analiza contenido usando LLM avanzado
        
        Args:
            text: Texto a analizar
            
        Returns:
            ContentAnalysis con resultados
        """
        if not self.client:
            raise IdentityAnalysisError("LLM client no disponible")
        
        # Limitar texto para no exceder tokens
        max_text_length = 10000
        truncated_text = text[:max_text_length]
        
        analysis_prompt = f"""
Analiza el siguiente contenido de redes sociales y extrae:

1. Temas principales (topics) - array de strings
2. Temas recurrentes (themes) - array de strings
3. Tono de comunicación (tone) - string: formal, casual, humorístico, etc.
4. Rasgos de personalidad (personality_traits) - array de strings
5. Estilo de comunicación (communication_style) - string
6. Frases comunes (common_phrases) - array de strings
7. Valores expresados (values) - array de strings
8. Intereses mencionados (interests) - array de strings
9. Patrones de lenguaje (language_patterns) - object
10. Análisis de sentimiento (sentiment_analysis) - object con keys: positive, negative, neutral

Contenido:
{truncated_text}

Responde SOLO con JSON válido, sin texto adicional.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en análisis de personalidad "
                            "y comunicación. Responde siempre en JSON válido."
                        )
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parsear JSON de forma segura
            response_content = response.choices[0].message.content
            analysis_data = json.loads(response_content)
            
            return ContentAnalysis(
                topics=analysis_data.get("topics", []),
                themes=analysis_data.get("themes", []),
                tone=analysis_data.get("tone"),
                personality_traits=analysis_data.get("personality_traits", []),
                communication_style=analysis_data.get("communication_style"),
                common_phrases=analysis_data.get("common_phrases", []),
                values=analysis_data.get("values", []),
                interests=analysis_data.get("interests", []),
                language_patterns=analysis_data.get("language_patterns", {}),
                sentiment_analysis=analysis_data.get("sentiment_analysis", {})
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parseando JSON de LLM: {e}")
            raise IdentityAnalysisError(
                "Error parseando respuesta del LLM",
                error_code="JSON_PARSE_ERROR"
            ) from e
        except Exception as e:
            self.logger.error(f"Error en análisis con LLM: {e}", exc_info=True)
            raise IdentityAnalysisError(
                f"Error en análisis LLM: {str(e)}",
                error_code="LLM_ANALYSIS_ERROR"
            ) from e
    
    async def _build_knowledge_base(
        self, content: dict, analysis: ContentAnalysis
    ) -> dict:
        """Construye base de conocimiento del perfil"""
        knowledge_base = {
            "topics": analysis.topics,
            "themes": analysis.themes,
            "personality": {
                "traits": analysis.personality_traits,
                "tone": analysis.tone,
                "communication_style": analysis.communication_style
            },
            "values": analysis.values,
            "interests": analysis.interests,
            "common_phrases": analysis.common_phrases,
            "language_patterns": analysis.language_patterns,
            "content_stats": {
                "total_videos": len(content.get("videos", [])),
                "total_posts": len(content.get("posts", [])),
                "total_text_length": sum(len(t) for t in content.get("texts", []))
            }
        }
        
        return knowledge_base
    
    def _determine_username(
        self,
        tiktok_profile: Optional[SocialProfile],
        instagram_profile: Optional[SocialProfile],
        youtube_profile: Optional[SocialProfile],
    ) -> str:
        """Determina el username principal"""
        if tiktok_profile:
            return tiktok_profile.username
        if instagram_profile:
            return instagram_profile.username
        if youtube_profile:
            return youtube_profile.username
        return "unknown"
    
    def _determine_display_name(
        self,
        tiktok_profile: Optional[SocialProfile],
        instagram_profile: Optional[SocialProfile],
        youtube_profile: Optional[SocialProfile],
    ) -> Optional[str]:
        """Determina el display name principal"""
        if tiktok_profile and tiktok_profile.display_name:
            return tiktok_profile.display_name
        if instagram_profile and instagram_profile.display_name:
            return instagram_profile.display_name
        if youtube_profile and youtube_profile.display_name:
            return youtube_profile.display_name
        return None
    
    def _determine_bio(
        self,
        tiktok_profile: Optional[SocialProfile],
        instagram_profile: Optional[SocialProfile],
        youtube_profile: Optional[SocialProfile],
    ) -> Optional[str]:
        """Determina la bio principal"""
        if instagram_profile and instagram_profile.bio:
            return instagram_profile.bio
        if tiktok_profile and tiktok_profile.bio:
            return tiktok_profile.bio
        if youtube_profile and youtube_profile.bio:
            return youtube_profile.bio
        return None

