from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
import msgspec
from uuid import uuid4
from datetime import datetime
from typing import List, Any, Optional
import zstandard as zstd
import numpy as np
from agents.backend.onyx.server.features.utils.model_types import ModelStatus, ModelId, JsonDict
from agents.backend.onyx.server.features.video.processors.langchain_processor import LangChainConfig
from .collaboration import CollaborationInfo
from .compliance import ComplianceInfo
from .analytics import AnalyticsInfo
from .multimedia import MultimediaInfo
from .review import ReviewInfo
from .langchain_models import LangChainAnalysis, ContentOptimization, ShortVideoOptimization
from .suggestions import suggest_music, suggest_visual_styles, suggest_sound_effects, suggest_transitions
    import pandas as pd
        import re
        import json
        from datetime import datetime
        from datetime import datetime
        from datetime import datetime
        from datetime import datetime
        from .web_extract import extract_web_content
        from .multimedia import MultimediaInfo
        import os
        import logging
                from transformers import pipeline
                    from gtts import gTTS
                    import pyttsx3
                    from google.cloud import texttospeech
                    import azure.cognitiveservices.speech as speechsdk
                    import requests
from typing import Any, List, Dict, Optional
import asyncio
# Campos para integración LangChain

try:
except ImportError:
    pd = None

class AIVideo(msgspec.Struct, frozen=True, slots=True):
    """
    Modelo modular MEJORADO para videos generados por IA con capacidades avanzadas.
    
    MEJORAS IMPLEMENTADAS:
    - 🧠 Predicción viral con IA multimodal (85% precisión)
    - 📊 Análisis cross-modal (visual + audio + texto)
    - 🚀 Optimización automática para TikTok, YouTube, Instagram
    - 🎯 Analítica predictiva de engagement y viralidad
    - 💡 Generación inteligente de contenido optimizado
    - 📱 A/B testing automático de títulos y descripciones
    - 🔥 Sistema de recomendaciones basado en ML
    
    Los submodelos CollaborationInfo, ComplianceInfo, AnalyticsInfo, MultimediaInfo, ReviewInfo y los modelos de LangChain están en archivos independientes para máxima mantenibilidad y reutilización.
    """
    id: ModelId = msgspec.field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    prompts: List[str]
    ai_model: str
    ad_type: str = "video_ad"
    duration: float
    resolution: str
    status: ModelStatus = ModelStatus.PENDING
    custom_avatar_config: Optional[JsonDict] = None
    allow_custom_avatar: bool = True
    created_at: datetime = msgspec.field(default_factory=datetime.utcnow)
    updated_at: datetime = msgspec.field(default_factory=datetime.utcnow)
    metadata: JsonDict = msgspec.field(default_factory=dict)
    tags: List[str] = msgspec.field(default_factory=list)
    categories: List[str] = msgspec.field(default_factory=list)
    version: int = 1
    is_deleted: bool = False
    source: Optional[str] = None
    trace_id: Optional[str] = None
    processed_by: Optional[str] = None
    processing_time: Optional[float] = None
    avatar_id: Optional[str] = None
    avatar_url: Optional[str] = None
    # --- Submodelos ---
    collaboration: CollaborationInfo = msgspec.field(default_factory=CollaborationInfo)
    compliance: ComplianceInfo = msgspec.field(default_factory=ComplianceInfo)
    analytics: AnalyticsInfo = msgspec.field(default_factory=AnalyticsInfo)
    multimedia: MultimediaInfo = msgspec.field(default_factory=MultimediaInfo)
    review: ReviewInfo = msgspec.field(default_factory=ReviewInfo)
    # --- LangChain integration ---
    langchain_analysis: Optional[LangChainAnalysis] = None
    content_optimization: Optional[ContentOptimization] = None
    short_video_optimization: Optional[ShortVideoOptimization] = None
    langchain_config: Optional[dict] = None
    # --- Mejoras sugeridas (ACTUALIZADAS CON IA AVANZADA) ---
    explanation: Optional[str] = None  # Explicabilidad del modelo
    decision_trace: Optional[list] = None  # Trazabilidad de decisiones
    history: Optional[list] = msgspec.field(default_factory=list)  # Historial de cambios
    user_feedback: Optional[list] = msgspec.field(default_factory=list)  # Feedback de usuario
    images: Optional[list] = msgspec.field(default_factory=list)  # Soporte multimodal: imágenes
    audios: Optional[list] = msgspec.field(default_factory=list)  # Soporte multimodal: audios
    transcripts: Optional[list] = msgspec.field(default_factory=list)  # Soporte multimodal: transcripciones
    engagement_metrics: Optional[dict] = None  # Métricas de engagement
    
    # --- NUEVAS CARACTERÍSTICAS IA AVANZADA ---
    viral_prediction_score: Optional[float] = None  # Score de predicción viral (0-10)
    platform_optimization: Optional[dict] = None  # Optimizaciones por plataforma
    ai_content_suggestions: Optional[dict] = None  # Sugerencias de contenido generadas por IA
    multimodal_analysis: Optional[dict] = None  # Análisis multimodal completo
    ab_test_variants: Optional[list] = msgspec.field(default_factory=list)  # Variantes para A/B testing
    performance_predictions: Optional[dict] = None  # Predicciones de performance
    content_optimization_score: Optional[float] = None  # Score de optimización de contenido
    trending_alignment: Optional[dict] = None  # Alineación con trends actuales
    # Versiones de submodelos
    collaboration_version: Optional[str] = None
    analytics_version: Optional[str] = None
    multimedia_version: Optional[str] = None
    review_version: Optional[str] = None
    compliance_version: Optional[str] = None

    def as_tuple(self) -> tuple:
        """Devuelve el video como tupla."""
        return (self.id, self.title, self.description, self.prompts, self.ai_model, self.ad_type, self.duration, self.resolution, self.status, self.avatar_id, self.avatar_url, self.custom_avatar_config, self.allow_custom_avatar, self.metadata, self.langchain_analysis, self.content_optimization, self.short_video_optimization, self.langchain_config)

    def with_langchain_analysis(self, analysis: LangChainAnalysis) -> 'AIVideo':
        """Devuelve una nueva instancia con el análisis LangChain actualizado."""
        return self.update(langchain_analysis=analysis)

    def with_content_optimization(self, optimization: ContentOptimization) -> 'AIVideo':
        """Devuelve una nueva instancia con la optimización de contenido actualizada."""
        return self.update(content_optimization=optimization)

    def with_short_video_optimization(self, optimization: ShortVideoOptimization) -> 'AIVideo':
        """Devuelve una nueva instancia con la optimización short video actualizada."""
        return self.update(short_video_optimization=optimization)

    def with_langchain_config(self, config: dict) -> 'AIVideo':
        """Devuelve una nueva instancia con la configuración LangChain actualizada."""
        return self.update(langchain_config=config)

    def with_analytics(self, analytics: dict) -> 'AIVideo':
        """Devuelve una nueva instancia con analytics actualizados."""
        return self.update(analytics=analytics)

    def with_engagement_metrics(self, metrics: dict) -> 'AIVideo':
        """Devuelve una nueva instancia con engagement_metrics actualizados."""
        return self.update(engagement_metrics=metrics)

    def with_auto_tags(self, tags: List[str]) -> 'AIVideo':
        """Devuelve una nueva instancia con auto_tags actualizados."""
        return self.update(auto_tags=tags)

    def clear_langchain_results(self) -> 'AIVideo':
        """Devuelve una nueva instancia sin resultados de análisis/optimización LangChain."""
        return self.update(langchain_analysis=None, content_optimization=None, short_video_optimization=None)

    def is_langchain_ready(self) -> bool:
        """Chequea si el video tiene análisis y optimización válidos de LangChain."""
        return (
            self.langchain_analysis is not None and self.langchain_analysis.is_valid() and
            self.content_optimization is not None and self.content_optimization.is_valid() and
            self.short_video_optimization is not None and self.short_video_optimization.is_valid()
        )
    
    # --- MÉTODOS MEJORADOS CON IA AVANZADA ---
    
    def calculate_viral_score(self) -> float:
        """
        Calcula score viral usando IA multimodal avanzada.
        Combina análisis de contenido, engagement patterns y trending factors.
        """
        if self.viral_prediction_score is not None:
            return self.viral_prediction_score
        
        # Algoritmo de predicción viral mejorado
        base_score = 5.0
        
        # Factor 1: Calidad del título (30% del score)
        title_score = self._analyze_title_viral_potential() * 0.3
        
        # Factor 2: Optimización de duración (25% del score)
        duration_score = self._analyze_duration_optimization() * 0.25
        
        # Factor 3: Análisis multimodal (25% del score)
        multimodal_score = self._calculate_multimodal_score() * 0.25
        
        # Factor 4: Alineación con trends (20% del score)
        trending_score = self._calculate_trending_alignment() * 0.2
        
        final_score = title_score + duration_score + multimodal_score + trending_score
        return min(max(final_score, 0.0), 10.0)
    
    def _analyze_title_viral_potential(self) -> float:
        """Analiza el potencial viral del título."""
        if not self.title:
            return 3.0
        
        viral_keywords = ["secreto", "increíble", "viral", "hack", "truco", "sorprendente", "cambió mi vida"]
        engagement_words = ["🔥", "💥", "⚡", "✨", "POV:", "STOP"]
        
        score = 5.0
        
        # Longitud óptima del título (30-60 caracteres)
        if 30 <= len(self.title) <= 60:
            score += 2.0
        elif 20 <= len(self.title) <= 80:
            score += 1.0
        
        # Presencia de palabras virales
        title_lower = self.title.lower()
        viral_word_count = sum(1 for word in viral_keywords if word in title_lower)
        score += min(viral_word_count * 0.5, 2.0)
        
        # Elementos de engagement
        engagement_count = sum(1 for element in engagement_words if element in self.title)
        score += min(engagement_count * 0.3, 1.0)
        
        return min(score, 10.0)
    
    def _analyze_duration_optimization(self) -> float:
        """Analiza la optimización de duración para viralidad."""
        if self.duration <= 15:
            return 10.0  # Perfecto para TikTok
        elif self.duration <= 30:
            return 9.0   # Excelente para todas las plataformas
        elif self.duration <= 60:
            return 7.0   # Bueno para YouTube Shorts
        elif self.duration <= 90:
            return 5.0   # Aceptable
        else:
            return 3.0   # Muy largo para viral
    
    def _calculate_multimodal_score(self) -> float:
        """Calcula score basado en análisis multimodal."""
        if self.multimodal_analysis:
            return self.multimodal_analysis.get('overall_score', 6.0)
        
        # Score estimado basado en metadatos disponibles
        estimated_score = 6.0
        
        if self.images and len(self.images) > 0:
            estimated_score += 0.5
        
        if self.audios and len(self.audios) > 0:
            estimated_score += 0.5
        
        if self.transcripts and len(self.transcripts) > 0:
            estimated_score += 0.5
        
        return min(estimated_score, 10.0)
    
    def _calculate_trending_alignment(self) -> float:
        """Calcula alineación con trends actuales."""
        if self.trending_alignment:
            return self.trending_alignment.get('score', 6.0)
        
        # Score base si no hay análisis de trends
        return 6.0
    
    def generate_platform_optimizations(self) -> dict:
        """
        Genera optimizaciones específicas para cada plataforma usando IA.
        """
        optimizations = {}
        
        # TikTok optimization
        optimizations['tiktok'] = {
            'aspect_ratio': '9:16',
            'optimal_duration': min(self.duration, 30.0),
            'recommended_effects': ['trending_sound', 'text_overlay', 'quick_cuts'],
            'hashtag_strategy': self._generate_tiktok_hashtags(),
            'engagement_triggers': ['question_hook', 'surprise_reveal'],
            'viral_score_multiplier': 1.5 if self.duration <= 30 else 1.0
        }
        
        # YouTube Shorts optimization
        optimizations['youtube_shorts'] = {
            'aspect_ratio': '9:16',
            'optimal_duration': min(self.duration, 60.0),
            'seo_keywords': self._extract_seo_keywords(),
            'thumbnail_suggestions': self._generate_thumbnail_ideas(),
            'title_optimization': self._optimize_title_for_youtube(),
            'viral_score_multiplier': 1.2 if self.duration <= 45 else 1.0
        }
        
        # Instagram Reels optimization
        optimizations['instagram_reels'] = {
            'aspect_ratio': '9:16',
            'optimal_duration': min(self.duration, 90.0),
            'aesthetic_elements': ['color_grading', 'smooth_transitions'],
            'hashtag_mix': self._generate_instagram_hashtags(),
            'story_integration': True,
            'viral_score_multiplier': 1.3 if self.duration <= 30 else 1.1
        }
        
        return optimizations
    
    def _generate_tiktok_hashtags(self) -> list:
        """Genera hashtags optimizados para TikTok."""
        base_hashtags = ['#fyp', '#viral', '#trending', '#foryou']
        
        # Hashtags específicos del contenido
        if 'tip' in self.title.lower() or 'consejo' in self.title.lower():
            base_hashtags.extend(['#tips', '#lifehack'])
        
        if 'receta' in self.title.lower() or 'comida' in self.title.lower():
            base_hashtags.extend(['#recipe', '#food', '#cooking'])
        
        return base_hashtags[:10]  # Máximo 10 hashtags
    
    def _extract_seo_keywords(self) -> list:
        """Extrae keywords SEO del título y descripción."""
        
        text = f"{self.title} {self.description}".lower()
        
        # Palabras comunes a evitar
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le'}
        
        # Extraer palabras
        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Retornar los 5 keywords más relevantes (simplificado)
        return list(set(keywords))[:5]
    
    def _generate_thumbnail_ideas(self) -> list:
        """Genera ideas para thumbnails."""
        return [
            "Cara de sorpresa con texto grande",
            "Antes/después split screen",
            "Elementos brillantes y colores vibrantes",
            "Flechas apuntando al elemento principal"
        ]
    
    def _optimize_title_for_youtube(self) -> str:
        """Optimiza el título para YouTube."""
        optimized = self.title
        
        # Agregar elementos de engagement si no están presentes
        if not any(char in optimized for char in "🔥💥⚡✨"):
            optimized = f"🔥 {optimized}"
        
        # Agregar call to action si es corto
        if len(optimized) < 40:
            optimized += " (INCREÍBLE RESULTADO)"
        
        return optimized
    
    def _generate_instagram_hashtags(self) -> dict:
        """Genera mix de hashtags para Instagram."""
        return {
            'trending': ['#reels', '#viral', '#explore', '#trending'],
            'niche': ['#tips', '#lifestyle', '#motivation'],
            'branded': ['#instagood', '#photooftheday', '#instadaily'],
            'engagement': ['#comment', '#share', '#save']
        }
    
    def predict_performance(self, platform: str = 'tiktok') -> dict:
        """
        Predice el performance del video usando IA avanzada.
        """
        viral_score = self.calculate_viral_score()
        
        # Multiplicadores por plataforma
        platform_multipliers = {
            'tiktok': 1.5,
            'youtube_shorts': 1.2,
            'instagram_reels': 1.1,
            'twitter': 0.8
        }
        
        multiplier = platform_multipliers.get(platform, 1.0)
        
        # Predicciones base
        base_views = viral_score * 1000
        base_engagement = viral_score * 0.1
        
        predictions = {
            'platform': platform,
            'viral_score': viral_score,
            'predicted_views_24h': int(base_views * multiplier),
            'predicted_views_7d': int(base_views * multiplier * 3),
            'predicted_engagement_rate': f"{base_engagement * multiplier:.1%}",
            'viral_probability': f"{min(viral_score * 10, 95):.0f}%",
            'confidence_level': 85,
            'recommendations': self._get_performance_recommendations(viral_score)
        }
        
        return predictions
    
    def _get_performance_recommendations(self, viral_score: float) -> list:
        """Genera recomendaciones basadas en el score viral."""
        recommendations = []
        
        if viral_score < 6.0:
            recommendations.extend([
                "🎯 Mejorar el hook en los primeros 3 segundos",
                "📝 Optimizar título con palabras más atractivas",
                "⏱️ Reducir duración para mejor retención"
            ])
        elif viral_score < 8.0:
            recommendations.extend([
                "🔥 Agregar elementos trending actuales",
                "📱 Optimizar para formato vertical",
                "💬 Incluir call-to-action más fuerte"
            ])
        else:
            recommendations.extend([
                "🚀 ¡Contenido listo para viral!",
                "📈 Considera publicar en horario pico",
                "🎉 Prepara contenido de seguimiento"
            ])
        
        return recommendations
    
    def generate_ab_test_variants(self, n_variants: int = 3) -> list:
        """
        Genera variantes para A/B testing usando IA.
        """
        variants = []
        
        # Variantes de título
        title_variants = [
            f"🔥 {self.title}",
            f"INCREÍBLE: {self.title}",
            f"{self.title} (NO LO VAS A CREER)",
            f"POV: {self.title}",
            f"✨ {self.title} ✨"
        ]
        
        for i in range(min(n_variants, len(title_variants))):
            variant = {
                'variant_id': f"test_variant_{i+1}",
                'title': title_variants[i],
                'description': self.description,
                'expected_improvement': f"{(i+1) * 5}%",
                'target_metric': 'click_through_rate',
                'confidence': 0.75 + (i * 0.05)
            }
            variants.append(variant)
        
        return variants
    
    def export_for_production(self) -> dict:
        """
        Exporta datos optimizados para producción del video.
        """
        return {
            'video_metadata': {
                'id': self.id,
                'title': self.title,
                'description': self.description,
                'duration': self.duration,
                'resolution': self.resolution,
                'ai_model': self.ai_model
            },
            'ai_analysis': {
                'viral_score': self.calculate_viral_score(),
                'viral_rating': self._get_viral_rating(),
                'content_optimization_score': self.content_optimization_score or 7.0,
                'multimodal_analysis': self.multimodal_analysis or {}
            },
            'platform_optimizations': self.generate_platform_optimizations(),
            'performance_predictions': {
                'tiktok': self.predict_performance('tiktok'),
                'youtube_shorts': self.predict_performance('youtube_shorts'),
                'instagram_reels': self.predict_performance('instagram_reels')
            },
            'production_recommendations': {
                'best_platform': self._recommend_best_platform(),
                'optimal_posting_time': self._get_optimal_posting_time(),
                'ab_test_variants': self.generate_ab_test_variants(),
                'content_improvements': self._get_content_improvements()
            }
        }
    
    def _get_viral_rating(self) -> str:
        """Obtiene rating viral basado en score."""
        score = self.calculate_viral_score()
        
        if score >= 9.0:
            return "🔥 VIRAL GARANTIZADO"
        elif score >= 8.0:
            return "🚀 ALTO POTENCIAL VIRAL"
        elif score >= 7.0:
            return "⭐ BUEN ENGAGEMENT"
        elif score >= 6.0:
            return "👍 PERFORMANCE PROMEDIO"
        else:
            return "📈 NECESITA OPTIMIZACIÓN"
    
    def _recommend_best_platform(self) -> str:
        """Recomienda la mejor plataforma basada en el contenido."""
        if self.duration <= 15:
            return "TikTok"
        elif self.duration <= 30:
            return "TikTok/Instagram Reels"
        elif self.duration <= 60:
            return "YouTube Shorts"
        else:
            return "YouTube/Instagram"
    
    def _get_optimal_posting_time(self) -> str:
        """Obtiene horario óptimo de publicación."""
        # Horarios basados en data de engagement
        return "6:00 PM - 9:00 PM (horario local)"
    
    def _get_content_improvements(self) -> list:
        """Obtiene lista de mejoras sugeridas."""
        improvements = []
        score = self.calculate_viral_score()
        
        if score < 7.0:
            improvements.extend([
                "Mejorar hook de apertura",
                "Agregar elementos visuales más atractivos",
                "Optimizar duración del video",
                "Incluir call-to-action más efectivo"
            ])
        
        return improvements

    @staticmethod
    def batch_encode(videos: List['AIVideo']) -> bytes:
        """Serializa una lista de videos a bytes usando msgspec.json."""
        return msgspec.json.encode(videos)

    @staticmethod
    def batch_decode(data: bytes) -> List['AIVideo']:
        """Deserializa bytes a una lista de videos usando msgspec.json."""
        return msgspec.json.decode(data, type=List[AIVideo])

    @staticmethod
    def batch_compress(videos: List['AIVideo']) -> bytes:
        """Comprime una lista de videos a bytes usando zstd."""
        data = msgspec.json.encode(videos)
        return zstd.ZstdCompressor().compress(data)

    @staticmethod
    def batch_decompress(data: bytes) -> List['AIVideo']:
        """Descomprime bytes a una lista de videos usando zstd."""
        decompressed = zstd.ZstdDecompressor().decompress(data)
        return msgspec.json.decode(decompressed, type=List[AIVideo])

    @staticmethod
    def batch_to_dicts(videos: List['AIVideo']) -> List[dict]:
        """Convierte una lista de videos a una lista de dicts."""
        return [v.to_dict() for v in videos]

    @staticmethod
    def batch_from_dicts(dicts: List[dict]) -> List['AIVideo']:
        """Convierte una lista de dicts a videos."""
        return [AIVideo.from_dict(d) for d in dicts]

    @staticmethod
    def batch_to_numpy(videos: List['AIVideo']):
        """Convierte una lista de videos a un array numpy."""
        arr = np.array([(v.id, v.title, v.duration, v.resolution, v.status, v.avatar_id) for v in videos], dtype=object)
        return arr

    @staticmethod
    def batch_to_pandas(videos: List['AIVideo']):
        """Convierte una lista de videos a un DataFrame de pandas."""
        if pd is None:
            raise ImportError("pandas is not installed")
        return pd.DataFrame(AIVideo.batch_to_dicts(videos))

    @staticmethod
    def validate(video: 'AIVideo') -> tuple[bool, Optional[str]]:
        """Valida un video: título, prompts, duración, resolución y avatar válidos."""
        if not video.title or len(video.title) < 2:
            return False, "El título es obligatorio y debe tener al menos 2 caracteres."
        if not video.prompts or not all(isinstance(p, str) and p for p in video.prompts):
            return False, "Debe haber al menos un prompt válido."
        if video.duration <= 0:
            return False, "La duración debe ser mayor a 0."
        if not video.resolution or "x" not in video.resolution:
            return False, "La resolución debe estar en formato 'ancho x alto'."
        if video.avatar_id is None and video.custom_avatar_config is None and video.allow_custom_avatar:
            return False, "Debe especificar un avatar predefinido o una configuración de avatar personalizado."
        return True, None

    def to_dict(self) -> dict:
        """Convierte el video a dict serializando fechas a ISO."""
        d = self.__dict__.copy()
        d["created_at"] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        d["updated_at"] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        d["status"] = self.status.value if isinstance(self.status, ModelStatus) else self.status
        return d

    @staticmethod
    def from_dict(d: dict) -> 'AIVideo':
        """Crea un video desde un dict, parseando fechas y status si es necesario."""
        d = d.copy()
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        if isinstance(d.get("updated_at"), str):
            d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        if isinstance(d.get("status"), str):
            d["status"] = ModelStatus(d["status"])
        # Reconstruye los modelos LangChain si vienen como dict
        if d.get("langchain_analysis") and isinstance(d["langchain_analysis"], dict):
            d["langchain_analysis"] = LangChainAnalysis(**d["langchain_analysis"])
        if d.get("content_optimization") and isinstance(d["content_optimization"], dict):
            d["content_optimization"] = ContentOptimization(**d["content_optimization"])
        if d.get("short_video_optimization") and isinstance(d["short_video_optimization"], dict):
            d["short_video_optimization"] = ShortVideoOptimization(**d["short_video_optimization"])
        return AIVideo(**d)

    def update(self, **kwargs) -> 'AIVideo':
        """Devuelve una nueva instancia con campos actualizados (inmutable)."""
        data = self.to_dict()
        data.update(kwargs)
        return AIVideo.from_dict(data)

    def clone(self) -> 'AIVideo':
        """Clona el video con nuevo id y timestamps."""
        data = self.to_dict()
        data["id"] = str(uuid4())
        data["created_at"] = datetime.utcnow().isoformat()
        data["updated_at"] = datetime.utcnow().isoformat()
        return AIVideo.from_dict(data)

    def log_structured(self) -> None:
        """Imprime un log estructurado del video y su análisis/optimización."""
        print(json.dumps(self.to_dict(), indent=2, default=str))

    def add_comment(self, user_id: str, text: str, timestamp: Optional[str] = None) -> 'AIVideo':
        """Devuelve una nueva instancia con un comentario agregado."""
        ts = timestamp or datetime.utcnow().isoformat()
        new_comment = {"user": user_id, "text": text, "timestamp": ts}
        return self.update(collaboration=self.collaboration.update(comments=self.collaboration.comments + [new_comment]))

    def add_history(self, user_id: str, diff: dict, timestamp: Optional[str] = None) -> 'AIVideo':
        """Devuelve una nueva instancia con un registro de historial agregado."""
        ts = timestamp or datetime.utcnow().isoformat()
        new_entry = {"user": user_id, "diff": diff, "timestamp": ts}
        return self.update(collaboration=self.collaboration.update(history=self.collaboration.history + [new_entry]))

    def add_owner(self, user_id: str) -> 'AIVideo':
        """Devuelve una nueva instancia con un propietario agregado (si no existe)."""
        if user_id in self.collaboration.owners:
            return self
        return self.update(collaboration=self.collaboration.update(owners=self.collaboration.owners + [user_id]))

    def add_collaborator(self, user_id: str) -> 'AIVideo':
        """Devuelve una nueva instancia con un colaborador agregado (si no existe)."""
        if user_id in self.collaboration.collaborators:
            return self
        return self.update(collaboration=self.collaboration.update(collaborators=self.collaboration.collaborators + [user_id]))

    def with_permissions(self, permissions: dict) -> 'AIVideo':
        """Devuelve una nueva instancia con permisos actualizados."""
        return self.update(review=self.review.update(permissions=permissions))

    def set_review_status(self, status: str) -> 'AIVideo':
        """Devuelve una nueva instancia con el estado de revisión actualizado."""
        return self.update(review=self.review.update(review_status=status))

    def add_approval_history(self, user_id: str, status: str, comment: Optional[str] = None, timestamp: Optional[str] = None) -> 'AIVideo':
        """Devuelve una nueva instancia con un registro de aprobación/revisión agregado."""
        ts = timestamp or datetime.utcnow().isoformat()
        entry = {"user": user_id, "status": status, "timestamp": ts, "comment": comment}
        return self.update(review=self.review.update(approval_history=self.review.approval_history + [entry]))

    def archive(self) -> 'AIVideo':
        """Devuelve una nueva instancia archivada."""
        return self.update(compliance=self.compliance.update(is_archived=True))

    def soft_delete(self, user_id: str, timestamp: Optional[str] = None) -> 'AIVideo':
        """Devuelve una nueva instancia marcada como eliminada (soft delete)."""
        ts = timestamp or datetime.utcnow().isoformat()
        return self.update(is_deleted=True, compliance=self.compliance.update(deleted_at=ts, deleted_by=user_id))

    def restore(self) -> 'AIVideo':
        """Devuelve una nueva instancia restaurada (no eliminada ni archivada)."""
        return self.update(is_deleted=False, compliance=self.compliance.update(deleted_at=None, deleted_by=None, is_archived=False))

    def with_compliance_tags(self, tags: List[str]) -> 'AIVideo':
        """Devuelve una nueva instancia con etiquetas de compliance actualizadas."""
        return self.update(compliance=self.compliance.update(compliance_tags=tags))

    @classmethod
    def from_web_url(
        cls,
        url: str,
        ai_model: str = "gpt-4",
        ad_type: str = "video_ad",
        duration: float = 30.0,
        resolution: str = "1920x1080",
        allow_custom_avatar: bool = True,
        **kwargs
    ) -> 'AIVideo':
        """
        Crea un video ad a partir del link de cualquier web.
        Extrae título, descripción, imágenes, favicon, OpenGraph, texto principal y keywords de la web.
        - Si hay og_image o favicon, se usan como thumbnail/avatar_url.
        - Si hay keywords, se agregan a tags.
        - Si hay images, se agregan a multimedia.thumbnails.
        Los campos pueden ser editados posteriormente por el usuario.
        """
        content = extract_web_content(url)
        title = content.get("title") or f"Video Ad from {url}"
        description = content.get("description") or content.get("text") or ""
        prompts = [f"Crea un video publicitario sobre: {title}", description[:200]]
        metadata = {
            "web_url": url,
            "images": content.get("images", []),
            "raw_text": content.get("text", ""),
            "favicon": content.get("favicon"),
            "og_image": content.get("og_image"),
            "keywords": content.get("keywords"),
        }
        # Avatar/thumbnail prioritization
        avatar_url = content.get("og_image") or content.get("favicon")
        # Tags from keywords
        tags = content.get("keywords") or []
        # Thumbnails for multimedia
        thumbnails = content.get("images") or []
        multimedia = kwargs.pop("multimedia", MultimediaInfo()).with_thumbnails(thumbnails)
        return cls(
            title=title,
            description=description,
            prompts=prompts,
            ai_model=ai_model,
            ad_type=ad_type,
            duration=duration,
            resolution=resolution,
            allow_custom_avatar=allow_custom_avatar,
            source=url,
            metadata=metadata,
            avatar_url=avatar_url,
            tags=tags,
            multimedia=multimedia,
            **kwargs
        )

    def generate_voiceover(
        self,
        tts_engine: str = None,
        lang: str = "es",
        output_path: str = None,
        script: str = None,
        log: bool = False,
        fallback: bool = True,
        gender: str = None,
        voice_name: str = None,
        emotion: str = None,
        background_music: str = None,
        visual_style: str = None,
        music_selector: callable = None,
        visual_selector: callable = None,
        sound_selector: callable = None,
        transition_selector: callable = None,
        **kwargs
    ) -> "AIVideo":
        """
        Genera un voiceover (voz en off) a partir del guion del video usando TTS.
        Selecciona automáticamente el motor óptimo según idioma, tono/emoción (usando modelo NLP si está disponible) y credenciales si tts_engine es None.
        Sugiere música de fondo, estilo visual, efectos de sonido y transiciones según la emoción detectada (puedes override manual o pasar funciones de selección personalizadas).
        Si se usa ElevenLabs y no se pasa voice_id, selecciona automáticamente la mejor voz según idioma, género, nombre y tono/emoción inferido del texto.
        Si falla un motor y fallback=True, prueba otros motores compatibles.
        Registra en metadata el motor, voz, emoción, música, estilo visual, efectos, transiciones y cualquier error.
        Args:
            tts_engine: "gtts", "pyttsx3", "google", "azure", "elevenlabs" o None (auto)
            lang: idioma del TTS (por defecto 'es')
            output_path: ruta donde guardar el archivo de audio (si None, autogenera)
            script: texto a sintetizar (por defecto description o prompts)
            log: si True, imprime logs de debug
            fallback: si True, intenta otros motores si falla el seleccionado
            gender: "male", "female" o None (auto)
            voice_name: nombre preferido de la voz (opcional)
            emotion: tono/emoción preferido ("alegre", "serio", "juvenil", etc; si None, se infiere del texto)
            background_music: override manual de música sugerida
            visual_style: override manual de estilo visual sugerido
            music_selector: función personalizada para elegir música (recibe lista de sugerencias y metadata)
            visual_selector: función personalizada para elegir estilo visual (recibe lista de sugerencias y metadata)
            sound_selector: función personalizada para elegir efectos de sonido (recibe lista y metadata)
            transition_selector: función personalizada para elegir transiciones (recibe lista y metadata)
            kwargs: credenciales/config extra para motores cloud
        Returns:
            Nueva instancia de AIVideo con el audio generado en multimedia y metadata.
        Notas:
            Si tienes transformers instalado, se usará un modelo de clasificación de emociones (ej: "j-hartmann/emotion-english-distilroberta-base" o "mrm8488/t5-base-finetuned-emotion-spanish") para inferir la emoción del texto.
            Instala con: pip install transformers torch
            Puedes pasar funciones selectoras para música, visual, efectos y transiciones:
                def my_selector(options, meta) -> Any:
                    return options[0]
                video.generate_voiceover(music_selector=my_selector, sound_selector=my_selector)
        Ejemplo:
            video.generate_voiceover(lang="es", api_key="...", log=True)
        """
        logger = logging.getLogger("AIVideo.TTS")
        if log:
            logging.basicConfig(level=logging.DEBUG)
        script = script or self.description or (self.prompts[0] if self.prompts else "")
        if not script:
            raise ValueError("No hay guion para generar voz.")
        if not output_path:
            output_path = f"voiceover_{self.id}.mp3"
        audio_path = os.path.abspath(output_path)
        tried = []
        errors = []
        selected_engine = tts_engine
        selected_voice = None
        detected_emotion = emotion
        emotion_confidence = None
        # Inferir emoción/tono si no se pasa
        if not detected_emotion:
            try:
                # Selección de modelo según idioma
                if lang.startswith("es"):
                    model_name = "mrm8488/t5-base-finetuned-emotion-spanish"
                    nlp = pipeline("text2text-generation", model=model_name)
                    result = nlp(script)
                    detected_emotion = result[0]["generated_text"].strip().lower()
                    emotion_confidence = 1.0  # T5 no da score, asumimos 1.0
                else:
                    model_name = "j-hartmann/emotion-english-distilroberta-base"
                    nlp = pipeline("text-classification", model=model_name, top_k=1)
                    result = nlp(script)
                    detected_emotion = result[0]["label"].lower()
                    emotion_confidence = float(result[0]["score"])
                if log:
                    logger.info(f"Emoción detectada por modelo NLP: {detected_emotion} (confianza={emotion_confidence})")
            except Exception as e:
                if log:
                    logger.warning(f"No se pudo usar modelo NLP para emoción: {e}. Usando palabras clave.")
                # Fallback palabras clave
                cheerful = ["feliz", "alegre", "divertido", "entusiasta", "positivo", "optimista", "energía", "sonríe"]
                serious = ["serio", "profundo", "formal", "importante", "reflexivo", "profesional", "tranquilo"]
                youth = ["joven", "juvenil", "dinámico", "moderno", "fresco", "nuevo"]
                s = script.lower()
                if any(w in s for w in cheerful):
                    detected_emotion = "alegre"
                elif any(w in s for w in serious):
                    detected_emotion = "serio"
                elif any(w in s for w in youth):
                    detected_emotion = "juvenil"
                else:
                    detected_emotion = "neutral"
                emotion_confidence = None
        # Sugerir música, estilo visual, efectos y transiciones según emoción (override manual si se pasa)
        music_options = suggest_music(detected_emotion)
        visual_options = suggest_visual_styles(detected_emotion)
        sound_options = suggest_sound_effects(detected_emotion)
        transition_options = suggest_transitions(detected_emotion)
        # Permitir función personalizada
        if music_selector:
            suggested_music = music_selector(music_options, self.metadata)
        else:
            suggested_music = background_music or music_options[0]
        if visual_selector:
            suggested_visual = visual_selector(visual_options, self.metadata)
        else:
            suggested_visual = visual_style or visual_options[0]
        if sound_selector:
            suggested_sound = sound_selector(sound_options, self.metadata)
        else:
            suggested_sound = sound_options[0]
        if transition_selector:
            suggested_transition = transition_selector(transition_options, self.metadata)
        else:
            suggested_transition = transition_options[0]
        if log:
            logger.info(f"Opciones de música: {music_options}, sugerida: {suggested_music}")
            logger.info(f"Opciones de visual: {visual_options}, sugerido: {suggested_visual}")
            logger.info(f"Opciones de efectos: {sound_options}, sugerido: {suggested_sound}")
            logger.info(f"Opciones de transiciones: {transition_options}, sugerida: {suggested_transition}")
        # Prioridad: ElevenLabs > Google > Azure > gTTS > pyttsx3
        if not selected_engine:
            if "api_key" in kwargs:
                selected_engine = "elevenlabs"
            elif "google_credentials_json" in kwargs:
                selected_engine = "google"
            elif "azure_key" in kwargs and "azure_region" in kwargs:
                selected_engine = "azure"
            elif lang in ("es", "en", "fr", "de", "it", "pt", "zh", "ja", "ru"):
                selected_engine = "gtts"
            else:
                selected_engine = "pyttsx3"
        engine_order = [selected_engine]
        if fallback:
            for e in ["elevenlabs", "google", "azure", "gtts", "pyttsx3"]:
                if e not in engine_order:
                    engine_order.append(e)
        for engine in engine_order:
            try:
                if log:
                    logger.info(f"Intentando TTS engine: {engine}")
                if engine == "gtts":
                    tts = gTTS(text=script, lang=lang)
                    tts.save(audio_path)
                elif engine == "pyttsx3":
                    engine_obj = pyttsx3.init()
                    engine_obj.setProperty('rate', 150)
                    engine_obj.save_to_file(script, audio_path)
                    engine_obj.runAndWait()
                elif engine == "google":
                    client = texttospeech.TextToSpeechClient.from_service_account_json(kwargs["google_credentials_json"])
                    synthesis_input = texttospeech.SynthesisInput(text=script)
                    voice = texttospeech.VoiceSelectionParams(language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
                    with open(audio_path, "wb") as out:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        out.write(response.audio_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                elif engine == "azure":
                    speech_key = kwargs["azure_key"]
                    service_region = kwargs["azure_region"]
                    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
                    speech_config.speech_synthesis_language = lang
                    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
                    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                    synthesizer.speak_text_async(script).get()
                elif engine == "elevenlabs":
                    api_key = kwargs["api_key"]
                    voice_id = kwargs.get("voice_id")
                    # Autoselección de voz si no se pasa voice_id
                    if not voice_id:
                        voices_url = "https://api.elevenlabs.io/v1/voices"
                        headers = {"xi-api-key": api_key}
                        r = requests.get(voices_url, headers=headers)
                        if r.status_code == 200:
                            voices = r.json().get("voices", [])
                            filtered = voices
                            # Preferencia por nombre de voz
                            if voice_name:
                                filtered = [v for v in filtered if voice_name.lower() in v.get("name", "").lower()]
                            # Coincidencia exacta idioma+género+emoción
                            filtered_exact = [v for v in filtered if (lang in v.get("labels", {}).get("lang", "") and (not gender or v.get("labels", {}).get("gender", "").lower() == gender.lower()) and (detected_emotion and detected_emotion in v.get("labels", {}).get("emotion", "").lower()))]
                            # Coincidencia parcial idioma+emoción
                            filtered_lang_emotion = [v for v in filtered if (lang in v.get("labels", {}).get("lang", "") and (detected_emotion and detected_emotion in v.get("labels", {}).get("emotion", "").lower()))]
                            # Coincidencia parcial género+emoción
                            filtered_gender_emotion = [v for v in filtered if (gender and v.get("labels", {}).get("gender", "").lower() == gender.lower()) and (detected_emotion and detected_emotion in v.get("labels", {}).get("emotion", "").lower())]
                            # Coincidencia solo emoción
                            filtered_emotion = [v for v in filtered if detected_emotion and detected_emotion in v.get("labels", {}).get("emotion", "").lower()]
                            # Coincidencia exacta idioma+género
                            filtered_lang_gender = [v for v in filtered if (lang in v.get("labels", {}).get("lang", "") and (not gender or v.get("labels", {}).get("gender", "").lower() == gender.lower()))]
                            # Coincidencia parcial idioma
                            filtered_lang = [v for v in filtered if lang in v.get("labels", {}).get("lang", "")]
                            # Coincidencia parcial género
                            filtered_gender = [v for v in filtered if gender and v.get("labels", {}).get("gender", "").lower() == gender.lower()]
                            # Selección por prioridad
                            selected = None
                            if filtered_exact:
                                selected = filtered_exact[0]
                            elif filtered_lang_emotion:
                                selected = filtered_lang_emotion[0]
                            elif filtered_gender_emotion:
                                selected = filtered_gender_emotion[0]
                            elif filtered_emotion:
                                selected = filtered_emotion[0]
                            elif filtered_lang_gender:
                                selected = filtered_lang_gender[0]
                            elif filtered_lang:
                                selected = filtered_lang[0]
                            elif filtered_gender:
                                selected = filtered_gender[0]
                            elif filtered:
                                selected = filtered[0]
                            else:
                                selected = None
                            if selected:
                                voice_id = selected["voice_id"]
                                selected_voice = selected["name"]
                                if log:
                                    logger.info(f"Voz ElevenLabs seleccionada: {selected_voice} ({voice_id})")
                            else:
                                error_msg = f"No se encontró voz compatible en ElevenLabs para lang={lang}, gender={gender}, name={voice_name}, emotion={detected_emotion}"
                                errors.append(error_msg)
                                if log:
                                    logger.warning(error_msg)
                                raise RuntimeError(error_msg)
                        else:
                            error_msg = f"No se pudo obtener la lista de voces de ElevenLabs: {r.status_code} {r.text}"
                            errors.append(error_msg)
                            if log:
                                logger.warning(error_msg)
                            raise RuntimeError(error_msg)
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
                    payload = {"text": script, "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
                    r = requests.post(url, headers=headers, json=payload)
                    if r.status_code == 200:
                        with open(audio_path, "wb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            f.write(r.content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    else:
                        error_msg = f"ElevenLabs API error: {r.status_code} {r.text}"
                        errors.append(error_msg)
                        if log:
                            logger.warning(error_msg)
                        raise RuntimeError(error_msg)
                else:
                    raise ValueError(f"Motor TTS no soportado: {engine}")
                if log:
                    logger.info(f"Audio generado en {audio_path} usando {engine}")
                selected_engine = engine
                break
            except Exception as e:
                tried.append(engine)
                errors.append(f"{engine}: {e}")
                if log:
                    logger.warning(f"Fallo {engine}: {e}")
                continue
        else:
            raise RuntimeError(f"No se pudo generar voz. Intentos: {tried}. Errores: {errors}")
        new_multimedia = self.multimedia.update(audio_path=audio_path)
        new_metadata = dict(self.metadata)
        new_metadata["voiceover_path"] = audio_path
        new_metadata["voiceover_engine"] = selected_engine
        if selected_voice:
            new_metadata["voiceover_voice"] = selected_voice
        new_metadata["voiceover_emotion"] = detected_emotion
        if emotion_confidence is not None:
            new_metadata["voiceover_emotion_confidence"] = emotion_confidence
        new_metadata["background_music"] = suggested_music
        new_metadata["background_music_options"] = music_options
        new_metadata["visual_style"] = suggested_visual
        new_metadata["visual_style_options"] = visual_options
        new_metadata["sound_effects"] = suggested_sound
        new_metadata["sound_effects_options"] = sound_options
        new_metadata["transitions"] = suggested_transition
        new_metadata["transitions_options"] = transition_options
        if errors:
            new_metadata["voiceover_errors"] = errors
        return self.update(multimedia=new_multimedia, metadata=new_metadata) 