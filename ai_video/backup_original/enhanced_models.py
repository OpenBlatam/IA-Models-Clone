from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
import msgspec
import asyncio
import hashlib
from uuid import uuid4
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from transformers import (
    import cv2
    import mediapipe as mp
    import librosa
    import soundfile as sf
from .models import AIVideo
from agents.backend.onyx.server.features.utils.model_types import ModelStatus, ModelId, JsonDict
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced AI Video Models

Modelo mejorado de video IA con capacidades avanzadas de machine learning,
optimización multimodal, y generación inteligente de contenido viral.
"""


# Enhanced imports
try:
        AutoModel, AutoTokenizer, AutoProcessor,
        CLIPModel, CLIPProcessor,
        BlipProcessor, BlipForConditionalGeneration,
        pipeline
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    COMPUTER_VISION_AVAILABLE = True
except ImportError:
    COMPUTER_VISION_AVAILABLE = False

try:
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# Base imports

# =============================================================================
# ENHANCED ENUMS
# =============================================================================

class VideoAIModel(str, Enum):
    """Enhanced AI models for video generation."""
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4_TURBO = "gpt-4-turbo-preview"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    LLAVA_1_6 = "llava-1.6-34b"
    COGVLM = "cogvlm-chat-17b"
    BLIP2 = "blip2-opt-6.7b"
    # Specialized video models
    RUNWAY_GEN2 = "runway-gen2"
    STABLE_VIDEO = "stable-video-diffusion"
    PIKA_LABS = "pika-labs-1.0"
    ZEROSCOPE = "zeroscope-v2"
    MODELSCOPE = "modelscope-t2v"

class ContentQuality(str, Enum):
    """Content quality levels."""
    VIRAL = "viral"           # 9.0-10.0
    EXCELLENT = "excellent"   # 8.0-8.9
    GOOD = "good"            # 7.0-7.9
    AVERAGE = "average"      # 6.0-6.9
    POOR = "poor"            # 0.0-5.9

class EngagementMetric(str, Enum):
    """Enhanced engagement metrics."""
    HOOK_EFFECTIVENESS = "hook_effectiveness"
    RETENTION_RATE = "retention_rate"
    SHARE_PROBABILITY = "share_probability"
    COMMENT_GENERATION = "comment_generation"
    VIRAL_COEFFICIENT = "viral_coefficient"
    EMOTIONAL_RESONANCE = "emotional_resonance"
    COGNITIVE_LOAD = "cognitive_load"
    ATTENTION_GRABBING = "attention_grabbing"

class PlatformOptimization(str, Enum):
    """Platform-specific optimizations."""
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    YOUTUBE_SHORTS = "youtube_shorts"
    SNAPCHAT_SPOTLIGHT = "snapchat_spotlight"
    TWITTER_VIDEO = "twitter_video"
    LINKEDIN_VIDEO = "linkedin_video"
    FACEBOOK_REELS = "facebook_reels"

# =============================================================================
# ENHANCED MODELS
# =============================================================================

@dataclass(slots=True)
class AIViralPredictor:
    """AI-powered viral prediction model."""
    model_name: str = "viral_predictor_v2"
    confidence_threshold: float = 0.7
    viral_score: float = 0.0
    confidence: float = 0.0
    
    # Detailed predictions
    hook_score: float = 0.0
    retention_score: float = 0.0
    share_score: float = 0.0
    comment_score: float = 0.0
    emotional_score: float = 0.0
    
    # Platform-specific scores
    platform_scores: Dict[str, float] = field(default_factory=dict)
    
    # Prediction metadata
    analyzed_features: List[str] = field(default_factory=list)
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)
    model_version: str = "2.1.0"

@dataclass(slots=True)
class MultimodalAnalysis:
    """Comprehensive multimodal content analysis."""
    # Visual analysis
    visual_features: Dict[str, float] = field(default_factory=dict)
    object_detection: List[Dict] = field(default_factory=list)
    scene_classification: List[str] = field(default_factory=list)
    color_analysis: Dict[str, float] = field(default_factory=dict)
    composition_score: float = 0.0
    
    # Audio analysis
    audio_features: Dict[str, float] = field(default_factory=dict)
    speech_analysis: Dict[str, Any] = field(default_factory=dict)
    music_analysis: Dict[str, Any] = field(default_factory=dict)
    audio_quality_score: float = 0.0
    
    # Text analysis
    text_features: Dict[str, float] = field(default_factory=dict)
    sentiment_analysis: Dict[str, float] = field(default_factory=dict)
    readability_score: float = 0.0
    keyword_relevance: Dict[str, float] = field(default_factory=dict)
    
    # Cross-modal analysis
    audio_visual_sync: float = 0.0
    text_visual_coherence: float = 0.0
    overall_coherence: float = 0.0

@dataclass(slots=True)
class ContentOptimizer:
    """AI-powered content optimization engine."""
    optimization_type: str = "viral_optimization"
    
    # Title optimization
    optimized_titles: List[Dict[str, Union[str, float]]] = field(default_factory=list)
    title_ab_test_variants: List[str] = field(default_factory=list)
    
    # Description optimization
    optimized_descriptions: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    hashtag_suggestions: List[str] = field(default_factory=list)
    
    # Timing optimization
    optimal_posting_times: Dict[str, List[str]] = field(default_factory=dict)
    optimal_video_length: Dict[str, float] = field(default_factory=dict)
    
    # Platform-specific optimizations
    platform_recommendations: Dict[str, Dict] = field(default_factory=dict)
    
    # A/B testing recommendations
    test_variants: List[Dict] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class EnhancedAnalytics:
    """Enhanced analytics with predictive capabilities."""
    # Real-time metrics
    engagement_velocity: float = 0.0
    viral_trajectory: List[float] = field(default_factory=list)
    audience_growth_rate: float = 0.0
    
    # Predictive analytics
    predicted_views_24h: float = 0.0
    predicted_views_7d: float = 0.0
    predicted_engagement_rate: float = 0.0
    viral_probability: float = 0.0
    
    # Audience insights
    demographic_analysis: Dict[str, float] = field(default_factory=dict)
    interest_analysis: Dict[str, float] = field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Performance benchmarking
    industry_percentile: float = 0.0
    creator_percentile: float = 0.0
    similar_content_comparison: Dict[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class AIContentGenerator:
    """AI-powered content generation capabilities."""
    generation_model: str = VideoAIModel.GPT4_VISION
    
    # Text generation
    script_variants: List[str] = field(default_factory=list)
    caption_variants: List[str] = field(default_factory=list)
    hook_suggestions: List[str] = field(default_factory=list)
    
    # Visual generation
    thumbnail_concepts: List[Dict] = field(default_factory=list)
    visual_style_suggestions: List[str] = field(default_factory=list)
    color_palette_suggestions: List[Dict] = field(default_factory=list)
    
    # Audio generation
    background_music_suggestions: List[Dict] = field(default_factory=list)
    sound_effect_suggestions: List[str] = field(default_factory=list)
    voiceover_styles: List[str] = field(default_factory=list)
    
    # Interactive elements
    poll_suggestions: List[Dict] = field(default_factory=list)
    challenge_concepts: List[str] = field(default_factory=list)
    call_to_action_variants: List[str] = field(default_factory=list)

# =============================================================================
# ENHANCED AI VIDEO MODEL
# =============================================================================

class EnhancedAIVideo(msgspec.Struct, frozen=True, slots=True):
    """
    Enhanced AI Video model with advanced machine learning capabilities,
    multimodal analysis, and intelligent optimization features.
    """
    # Base fields from AIVideo
    id: ModelId = msgspec.field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    prompts: List[str]
    ai_model: VideoAIModel = VideoAIModel.GPT4_VISION
    ad_type: str = "enhanced_video_ad"
    duration: float
    resolution: str
    status: ModelStatus = ModelStatus.PENDING
    created_at: datetime = msgspec.field(default_factory=datetime.utcnow)
    updated_at: datetime = msgspec.field(default_factory=datetime.utcnow)
    metadata: JsonDict = msgspec.field(default_factory=dict)
    
    # Enhanced AI features
    viral_predictor: AIViralPredictor = msgspec.field(default_factory=AIViralPredictor)
    multimodal_analysis: MultimodalAnalysis = msgspec.field(default_factory=MultimodalAnalysis)
    content_optimizer: ContentOptimizer = msgspec.field(default_factory=ContentOptimizer)
    enhanced_analytics: EnhancedAnalytics = msgspec.field(default_factory=EnhancedAnalytics)
    ai_generator: AIContentGenerator = msgspec.field(default_factory=AIContentGenerator)
    
    # Platform optimizations
    platform_optimizations: Dict[PlatformOptimization, Dict] = msgspec.field(default_factory=dict)
    
    # Quality metrics
    content_quality: ContentQuality = ContentQuality.AVERAGE
    quality_score: float = 5.0
    engagement_metrics: Dict[EngagementMetric, float] = msgspec.field(default_factory=dict)
    
    # Advanced features
    ab_test_results: Dict[str, Any] = msgspec.field(default_factory=dict)
    competitor_analysis: Dict[str, Any] = msgspec.field(default_factory=dict)
    trend_alignment: Dict[str, float] = msgspec.field(default_factory=dict)
    
    # Processing metadata
    processing_pipeline: List[str] = msgspec.field(default_factory=list)
    model_versions: Dict[str, str] = msgspec.field(default_factory=dict)
    performance_metrics: Dict[str, float] = msgspec.field(default_factory=dict)
    
    def calculate_viral_score(self) -> float:
        """Calculate comprehensive viral score using AI predictions."""
        base_score = self.viral_predictor.viral_score
        
        # Weight different factors
        weights = {
            'hook_score': 0.25,
            'retention_score': 0.20,
            'share_score': 0.20,
            'emotional_score': 0.15,
            'comment_score': 0.10,
            'platform_avg': 0.10
        }
        
        # Calculate weighted score
        weighted_score = (
            self.viral_predictor.hook_score * weights['hook_score'] +
            self.viral_predictor.retention_score * weights['retention_score'] +
            self.viral_predictor.share_score * weights['share_score'] +
            self.viral_predictor.emotional_score * weights['emotional_score'] +
            self.viral_predictor.comment_score * weights['comment_score'] +
            self._get_average_platform_score() * weights['platform_avg']
        )
        
        return min(max(weighted_score, 0.0), 10.0)
    
    def _get_average_platform_score(self) -> float:
        """Get average score across all platforms."""
        if not self.viral_predictor.platform_scores:
            return 5.0
        return sum(self.viral_predictor.platform_scores.values()) / len(self.viral_predictor.platform_scores)
    
    def get_optimization_recommendations(self, platform: PlatformOptimization) -> Dict[str, Any]:
        """Get AI-powered optimization recommendations for specific platform."""
        recommendations = {
            'title_optimizations': self.content_optimizer.optimized_titles[:3],
            'duration_recommendation': self.content_optimizer.optimal_video_length.get(platform.value, self.duration),
            'hashtag_suggestions': self.content_optimizer.hashtag_suggestions[:10],
            'posting_time_recommendations': self.content_optimizer.optimal_posting_times.get(platform.value, []),
            'visual_improvements': self._get_visual_recommendations(),
            'audio_improvements': self._get_audio_recommendations(),
            'engagement_tactics': self._get_engagement_recommendations(platform)
        }
        
        return recommendations
    
    def _get_visual_recommendations(self) -> List[str]:
        """Get visual improvement recommendations."""
        recommendations = []
        
        if self.multimodal_analysis.composition_score < 7.0:
            recommendations.append("Improve visual composition and framing")
        
        if 'face_detection' in self.multimodal_analysis.visual_features and \
           self.multimodal_analysis.visual_features['face_detection'] < 0.5:
            recommendations.append("Include more human faces for better engagement")
        
        if self.multimodal_analysis.color_analysis.get('vibrancy', 0) < 0.6:
            recommendations.append("Use more vibrant and contrasting colors")
        
        return recommendations
    
    def _get_audio_recommendations(self) -> List[str]:
        """Get audio improvement recommendations."""
        recommendations = []
        
        if self.multimodal_analysis.audio_quality_score < 7.0:
            recommendations.append("Improve audio quality and clarity")
        
        if 'music_presence' in self.multimodal_analysis.audio_features and \
           self.multimodal_analysis.audio_features['music_presence'] < 0.3:
            recommendations.append("Add background music to increase engagement")
        
        return recommendations
    
    def _get_engagement_recommendations(self, platform: PlatformOptimization) -> List[str]:
        """Get platform-specific engagement recommendations."""
        recommendations = []
        
        if platform == PlatformOptimization.TIKTOK:
            recommendations.extend([
                "Use trending sounds and music",
                "Include text overlays for key points",
                "Create a strong hook in first 3 seconds",
                "Use vertical (9:16) format"
            ])
        elif platform == PlatformOptimization.YOUTUBE_SHORTS:
            recommendations.extend([
                "Optimize for discovery with relevant keywords",
                "Create compelling thumbnails",
                "Use clear calls-to-action",
                "Keep content under 60 seconds"
            ])
        elif platform == PlatformOptimization.INSTAGRAM_REELS:
            recommendations.extend([
                "Use Instagram's native editing tools",
                "Include relevant hashtags",
                "Create shareable moments",
                "Leverage Instagram trends"
            ])
        
        return recommendations
    
    def predict_performance(self, platform: PlatformOptimization, timeframe: str = "24h") -> Dict[str, float]:
        """Predict video performance using AI models."""
        base_prediction = {
            'views': self.enhanced_analytics.predicted_views_24h,
            'engagement_rate': self.enhanced_analytics.predicted_engagement_rate,
            'viral_probability': self.enhanced_analytics.viral_probability
        }
        
        # Platform-specific adjustments
        platform_multiplier = {
            PlatformOptimization.TIKTOK: 1.5,
            PlatformOptimization.YOUTUBE_SHORTS: 1.2,
            PlatformOptimization.INSTAGRAM_REELS: 1.1,
            PlatformOptimization.SNAPCHAT_SPOTLIGHT: 0.9,
            PlatformOptimization.TWITTER_VIDEO: 0.8
        }
        
        multiplier = platform_multiplier.get(platform, 1.0)
        
        return {
            'predicted_views': base_prediction['views'] * multiplier,
            'predicted_engagement_rate': base_prediction['engagement_rate'] * multiplier,
            'viral_probability': min(base_prediction['viral_probability'] * multiplier, 1.0),
            'confidence': self.viral_predictor.confidence
        }
    
    def generate_ab_test_variants(self, n_variants: int = 3) -> List[Dict[str, Any]]:
        """Generate A/B test variants for optimization."""
        variants = []
        
        for i in range(n_variants):
            variant = {
                'variant_id': f"variant_{i+1}",
                'title': self.content_optimizer.optimized_titles[i] if i < len(self.content_optimizer.optimized_titles) else self.title,
                'description': self.content_optimizer.optimized_descriptions[i] if i < len(self.content_optimizer.optimized_descriptions) else self.description,
                'hashtags': self.content_optimizer.hashtag_suggestions[i*3:(i+1)*3],
                'thumbnail_concept': self.ai_generator.thumbnail_concepts[i] if i < len(self.ai_generator.thumbnail_concepts) else {},
                'expected_performance': self._calculate_variant_performance(i)
            }
            variants.append(variant)
        
        return variants
    
    def _calculate_variant_performance(self, variant_index: int) -> Dict[str, float]:
        """Calculate expected performance for a variant."""
        base_score = self.calculate_viral_score()
        variance = 0.1 * variant_index  # Each variant has slight performance variance
        
        return {
            'viral_score': max(0, min(10, base_score + variance)),
            'engagement_rate': self.enhanced_analytics.predicted_engagement_rate * (1 + variance/10),
            'view_prediction': self.enhanced_analytics.predicted_views_24h * (1 + variance/5)
        }
    
    def export_for_production(self) -> Dict[str, Any]:
        """Export optimized data for video production."""
        return {
            'video_specifications': {
                'duration': self.duration,
                'resolution': self.resolution,
                'aspect_ratio': '9:16',  # Optimized for mobile
                'frame_rate': 30
            },
            'content_elements': {
                'script': self.ai_generator.script_variants[0] if self.ai_generator.script_variants else "",
                'visual_style': self.ai_generator.visual_style_suggestions[0] if self.ai_generator.visual_style_suggestions else "",
                'color_palette': self.ai_generator.color_palette_suggestions[0] if self.ai_generator.color_palette_suggestions else {},
                'music_recommendation': self.ai_generator.background_music_suggestions[0] if self.ai_generator.background_music_suggestions else {}
            },
            'optimization_data': {
                'best_title': self.content_optimizer.optimized_titles[0] if self.content_optimizer.optimized_titles else self.title,
                'best_description': self.content_optimizer.optimized_descriptions[0] if self.content_optimizer.optimized_descriptions else self.description,
                'recommended_hashtags': self.content_optimizer.hashtag_suggestions[:5],
                'optimal_posting_time': self.content_optimizer.optimal_posting_times
            },
            'performance_predictions': {
                'viral_score': self.calculate_viral_score(),
                'expected_engagement': self.enhanced_analytics.predicted_engagement_rate,
                'platform_recommendations': {platform.value: self.get_optimization_recommendations(platform) for platform in PlatformOptimization}
            }
        }
    
    @classmethod
    def create_from_basic_video(cls, basic_video: AIVideo, **enhancements) -> 'EnhancedAIVideo':
        """Create enhanced video from basic AIVideo instance."""
        return cls(
            title=basic_video.title,
            description=basic_video.description,
            prompts=basic_video.prompts,
            ai_model=VideoAIModel.GPT4_VISION,
            duration=basic_video.duration,
            resolution=basic_video.resolution,
            status=basic_video.status,
            created_at=basic_video.created_at,
            updated_at=basic_video.updated_at,
            metadata=basic_video.metadata,
            **enhancements
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, datetime):
                data[field_name] = field_value.isoformat()
            elif isinstance(field_value, Enum):
                data[field_name] = field_value.value
            elif hasattr(field_value, '__dict__'):
                data[field_name] = field_value.__dict__
            else:
                data[field_name] = field_value
        return data

# =============================================================================
# AI PROCESSING PIPELINE
# =============================================================================

class AIVideoProcessor:
    """Enhanced AI video processing pipeline."""
    
    def __init__(self) -> Any:
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self) -> Any:
        """Initialize AI models for processing."""
        if TORCH_AVAILABLE:
            try:
                # Vision models
                self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.models['clip_processor'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                # Text generation
                self.models['text_classifier'] = pipeline("text-classification", 
                                                        model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                
                # Multimodal models
                self.models['blip'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.models['blip_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                
            except Exception as e:
                print(f"Warning: Could not initialize some AI models: {e}")
    
    async def process_video(self, video: EnhancedAIVideo, video_path: Optional[str] = None) -> EnhancedAIVideo:
        """Process video with AI enhancement pipeline."""
        tasks = []
        
        # Visual analysis
        if video_path and COMPUTER_VISION_AVAILABLE:
            tasks.append(self._analyze_visual_content(video_path))
        
        # Audio analysis
        if video_path and AUDIO_PROCESSING_AVAILABLE:
            tasks.append(self._analyze_audio_content(video_path))
        
        # Text analysis
        tasks.append(self._analyze_text_content(video))
        
        # Viral prediction
        tasks.append(self._predict_viral_performance(video))
        
        # Execute all analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        enhanced_video = self._combine_analysis_results(video, results)
        
        return enhanced_video
    
    async def _analyze_visual_content(self, video_path: str) -> MultimodalAnalysis:
        """Analyze visual content of video."""
        # Placeholder for visual analysis
        return MultimodalAnalysis(
            composition_score=7.5,
            visual_features={'brightness': 0.7, 'contrast': 0.8, 'face_detection': 0.6}
        )
    
    async def _analyze_audio_content(self, video_path: str) -> Dict[str, Any]:
        """Analyze audio content of video."""
        # Placeholder for audio analysis
        return {
            'audio_quality_score': 8.0,
            'music_presence': 0.7,
            'speech_clarity': 0.9
        }
    
    async def _analyze_text_content(self, video: EnhancedAIVideo) -> Dict[str, Any]:
        """Analyze text content for sentiment and engagement potential."""
        if 'text_classifier' not in self.models:
            return {}
        
        try:
            # Analyze title and description
            text_to_analyze = f"{video.title} {video.description}"
            sentiment_result = self.models['text_classifier'](text_to_analyze)
            
            return {
                'sentiment': sentiment_result[0]['label'],
                'sentiment_score': sentiment_result[0]['score'],
                'readability_score': len(text_to_analyze.split()) / 20.0  # Simple readability
            }
        except Exception:
            return {}
    
    async def _predict_viral_performance(self, video: EnhancedAIVideo) -> AIViralPredictor:
        """Predict viral performance using AI models."""
        # Simplified viral prediction algorithm
        base_score = 5.0
        
        # Title length optimization
        title_length = len(video.title)
        if 30 <= title_length <= 60:
            base_score += 1.0
        
        # Description quality
        if len(video.description) > 100:
            base_score += 0.5
        
        # Duration optimization (shorter is better for viral content)
        if video.duration <= 30:
            base_score += 1.5
        elif video.duration <= 60:
            base_score += 1.0
        
        return AIViralPredictor(
            viral_score=min(base_score, 10.0),
            confidence=0.75,
            hook_score=base_score * 0.9,
            retention_score=base_score * 0.8,
            share_score=base_score * 0.7,
            comment_score=base_score * 0.6,
            emotional_score=base_score * 0.8
        )
    
    def _combine_analysis_results(self, video: EnhancedAIVideo, results: List[Any]) -> EnhancedAIVideo:
        """Combine all analysis results into enhanced video."""
        # Extract results (handling exceptions)
        visual_analysis = results[0] if not isinstance(results[0], Exception) else MultimodalAnalysis()
        audio_analysis = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}
        text_analysis = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}
        viral_prediction = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else AIViralPredictor()
        
        # Update video with analysis results
        return video.update(
            multimodal_analysis=visual_analysis,
            viral_predictor=viral_prediction,
            enhanced_analytics=EnhancedAnalytics(
                predicted_engagement_rate=viral_prediction.viral_score / 10.0,
                viral_probability=viral_prediction.confidence
            )
        )

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enhanced_video(
    title: str,
    description: str,
    prompts: List[str],
    duration: float = 30.0,
    resolution: str = "1920x1080",
    ai_model: VideoAIModel = VideoAIModel.GPT4_VISION,
    **kwargs
) -> EnhancedAIVideo:
    """Create enhanced AI video with default optimizations."""
    return EnhancedAIVideo(
        title=title,
        description=description,
        prompts=prompts,
        duration=duration,
        resolution=resolution,
        ai_model=ai_model,
        **kwargs
    )

def create_viral_optimized_video(
    title: str,
    description: str,
    target_platform: PlatformOptimization = PlatformOptimization.TIKTOK,
    **kwargs
) -> EnhancedAIVideo:
    """Create video optimized for viral performance on specific platform."""
    # Platform-specific optimizations
    duration = 15.0 if target_platform == PlatformOptimization.TIKTOK else 30.0
    resolution = "1080x1920"  # Vertical format for mobile
    
    video = create_enhanced_video(
        title=title,
        description=description,
        prompts=[f"Create viral content for {target_platform.value}: {title}"],
        duration=duration,
        resolution=resolution,
        **kwargs
    )
    
    # Set platform optimizations
    video.platform_optimizations[target_platform] = {
        'optimized': True,
        'vertical_format': True,
        'duration_optimized': True
    }
    
    return video

async def process_video_with_ai(video: EnhancedAIVideo, video_path: Optional[str] = None) -> EnhancedAIVideo:
    """Process video through AI enhancement pipeline."""
    processor = AIVideoProcessor()
    return await processor.process_video(video, video_path) 