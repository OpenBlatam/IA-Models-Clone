from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from ..models.facebook_models import (
from ..services.langchain_service import FacebookLangChainService
            from ..models.facebook_models import (
            from ..models.facebook_models import ContentMetrics, EngagementPrediction, QualityAssessment, QualityTier
        import re
        from ..models.facebook_models import QualityTier
from typing import Any, List, Dict, Optional
"""
🎯 Facebook Posts Engine - Core
===============================

Motor principal para generación y análisis de Facebook posts.
Integración completa con Onyx y LangChain.
"""


    FacebookPostEntity, FacebookPostRequest, FacebookPostResponse,
    FacebookPostAnalysis, ContentIdentifier, PostSpecification,
    GenerationConfig, FacebookPostContent, FacebookPostFactory,
    ContentStatus, PostType, ContentTone, TargetAudience
)


class FacebookPostEngine:
    """
    Motor principal para Facebook posts integrado con Onyx.
    Maneja generación, análisis y optimización de contenido.
    """
    
    def __init__(self, langchain_service: FacebookLangChainService):
        
    """__init__ function."""
self.langchain_service = langchain_service
        self.logger = logging.getLogger(__name__)
        
        # Analytics y métricas
        self.analytics = {
            'posts_generated': 0,
            'posts_analyzed': 0,
            'average_quality_score': 0.0,
            'high_quality_posts': 0,
            'total_processing_time': 0.0
        }
        
        # Cache para optimización
        self._content_cache: Dict[str, FacebookPostEntity] = {}
        self._analysis_cache: Dict[str, FacebookPostAnalysis] = {}
    
    async def generate_post(self, request: FacebookPostRequest) -> FacebookPostResponse:
        """
        Generar Facebook post completo con análisis.
        
        Args:
            request: Configuración de generación
            
        Returns:
            Respuesta completa con post, análisis y recomendaciones
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Generating Facebook post for topic: {request.topic}")
            
            # Step 1: Generate content using LangChain
            content_text = await self._generate_content_text(request)
            
            # Step 2: Generate hashtags if enabled
            hashtags = []
            if request.include_hashtags:
                hashtags = await self._generate_hashtags(request.topic, request.keywords)
            
            # Step 3: Create post entity
            post = FacebookPostFactory.create_from_specification(
                specification=PostSpecification(
                    topic=request.topic,
                    post_type=request.post_type,
                    tone=request.tone,
                    target_audience=request.target_audience,
                    keywords=request.keywords,
                    target_engagement=request.target_engagement
                ),
                generation_config=GenerationConfig(
                    max_length=request.max_length,
                    include_hashtags=request.include_hashtags,
                    include_emojis=request.include_emojis,
                    include_call_to_action=request.include_call_to_action,
                    brand_voice=request.brand_voice,
                    campaign_context=request.campaign_context,
                    custom_instructions=request.custom_instructions
                ),
                content_text=content_text,
                hashtags=hashtags,
                workspace_id=request.workspace_id,
                user_id=request.user_id,
                project_id=request.project_id
            )
            
            # Step 4: Analyze the generated post
            analysis = await self.analyze_post(post)
            post.set_analysis(analysis)
            
            # Step 5: Generate variations if needed
            variations = await self._generate_variations(post, request)
            
            # Step 6: Get recommendations
            recommendations = self._get_optimization_recommendations(post, analysis)
            
            # Step 7: Update analytics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_analytics(post, analysis, processing_time)
            
            # Step 8: Cache results
            self._cache_results(post, analysis)
            
            return FacebookPostResponse(
                success=True,
                post=post,
                variations=variations,
                analysis=analysis,
                recommendations=recommendations,
                processing_time_ms=processing_time,
                langchain_session_id=post.langchain_session_id
            )
            
        except Exception as e:
            self.logger.error(f"Error generating Facebook post: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FacebookPostResponse(
                success=False,
                post=None,
                variations=[],
                analysis=None,
                recommendations=[],
                processing_time_ms=processing_time,
                error_message=str(e)
            )
    
    async def analyze_post(self, post: FacebookPostEntity) -> FacebookPostAnalysis:
        """
        Analizar un Facebook post existente.
        
        Args:
            post: Post a analizar
            
        Returns:
            Análisis completo del post
        """
        try:
            # Check cache first
            cache_key = f"{post.identifier.content_hash}_{post.version}"
            if cache_key in self._analysis_cache:
                self.logger.debug(f"Analysis cache hit for post {post.identifier.content_id}")
                return self._analysis_cache[cache_key]
            
            self.logger.info(f"Analyzing Facebook post: {post.identifier.content_id}")
            
            # Use LangChain for comprehensive analysis
            analysis_result = await self.langchain_service.analyze_facebook_post(
                post.content.text,
                {
                    'post_type': post.specification.post_type.value,
                    'tone': post.specification.tone.value,
                    'audience': post.specification.target_audience.value,
                    'hashtags': post.content.hashtags,
                    'has_media': bool(post.content.media_urls),
                    'call_to_action': post.content.call_to_action,
                    'topic': post.specification.topic,
                    'keywords': post.specification.keywords
                }
            )
            
            # Create comprehensive analysis
                ContentMetrics, EngagementPrediction, QualityAssessment, QualityTier
            )
            
            # Content metrics
            content_metrics = ContentMetrics(
                character_count=post.content.get_character_count(),
                word_count=post.content.get_word_count(),
                hashtag_count=len(post.content.hashtags),
                mention_count=len(post.content.mentions),
                emoji_count=self._count_emojis(post.content.text),
                readability_score=analysis_result.get('readability_score', 0.7),
                sentiment_score=analysis_result.get('sentiment_score', 0.5)
            )
            
            # Engagement prediction
            engagement_prediction = EngagementPrediction(
                engagement_rate=analysis_result.get('engagement_prediction', 0.6),
                virality_score=analysis_result.get('virality_score', 0.4),
                predicted_likes=analysis_result.get('predicted_likes', 100),
                predicted_shares=analysis_result.get('predicted_shares', 20),
                predicted_comments=analysis_result.get('predicted_comments', 15),
                predicted_reach=analysis_result.get('predicted_reach', 1000),
                confidence_level=analysis_result.get('confidence_level', 0.8)
            )
            
            # Quality assessment
            overall_score = analysis_result.get('overall_score', 0.6)
            quality_tier = self._determine_quality_tier(overall_score)
            
            quality_assessment = QualityAssessment(
                overall_score=overall_score,
                quality_tier=quality_tier,
                brand_alignment=analysis_result.get('brand_alignment', 0.6),
                audience_relevance=analysis_result.get('audience_relevance', 0.7),
                trend_alignment=analysis_result.get('trend_alignment', 0.5),
                clarity_score=analysis_result.get('clarity_score', 0.7),
                strengths=analysis_result.get('strengths', []),
                weaknesses=analysis_result.get('weaknesses', []),
                improvement_suggestions=analysis_result.get('improvements', [])
            )
            
            # Set optimal posting time
            optimal_time = await self._calculate_optimal_posting_time(post)
            
            # Create final analysis
            analysis = FacebookPostAnalysis(
                content_metrics=content_metrics,
                engagement_prediction=engagement_prediction,
                quality_assessment=quality_assessment,
                processing_time_ms=analysis_result.get('processing_time_ms', 0),
                analysis_models_used=analysis_result.get('models_used', ['langchain']),
                onyx_model_id=analysis_result.get('onyx_model_id'),
                optimal_posting_time=optimal_time,
                hashtag_suggestions=analysis_result.get('hashtag_suggestions', []),
                similar_successful_posts=analysis_result.get('similar_posts', []),
                competitive_analysis=analysis_result.get('competitive_analysis', {})
            )
            
            # Cache analysis
            self._analysis_cache[cache_key] = analysis
            
            self.analytics['posts_analyzed'] += 1
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing Facebook post: {e}")
            
            # Return basic analysis on error
            
            return FacebookPostAnalysis(
                content_metrics=ContentMetrics(
                    character_count=post.content.get_character_count(),
                    word_count=post.content.get_word_count(),
                    hashtag_count=len(post.content.hashtags),
                    mention_count=len(post.content.mentions),
                    emoji_count=0,
                    readability_score=0.5,
                    sentiment_score=0.5
                ),
                engagement_prediction=EngagementPrediction(
                    engagement_rate=0.5,
                    virality_score=0.3,
                    predicted_likes=50,
                    predicted_shares=10,
                    predicted_comments=8,
                    predicted_reach=500
                ),
                quality_assessment=QualityAssessment(
                    overall_score=0.5,
                    quality_tier=QualityTier.FAIR,
                    brand_alignment=0.5,
                    audience_relevance=0.5,
                    trend_alignment=0.5,
                    clarity_score=0.5,
                    strengths=["Post created successfully"],
                    weaknesses=["Analysis failed - using defaults"],
                    improvement_suggestions=["Retry analysis with different parameters"]
                )
            )
    
    # ===== PRIVATE METHODS =====
    
    async def _generate_content_text(self, request: FacebookPostRequest) -> str:
        """Generar texto del contenido usando LangChain."""
        try:
            content = await self.langchain_service.generate_facebook_post(
                topic=request.topic,
                tone=request.tone.value,
                audience=request.target_audience.value,
                max_length=request.max_length,
                include_emojis=request.include_emojis,
                include_call_to_action=request.include_call_to_action,
                brand_voice=request.brand_voice,
                campaign_context=request.campaign_context,
                custom_instructions=request.custom_instructions
            )
            return content
            
        except Exception as e:
            self.logger.warning(f"LangChain generation failed, using fallback: {e}")
            
            # Fallback content generation
            emoji = "✨" if request.include_emojis else ""
            cta = "\n\nWhat do you think? Share your thoughts below! 👇" if request.include_call_to_action else ""
            
            return f"{emoji} Discover amazing insights about {request.topic}! Transform your approach today.{cta}"
    
    async def _generate_hashtags(self, topic: str, keywords: List[str]) -> List[str]:
        """Generar hashtags relevantes."""
        try:
            hashtags = await self.langchain_service.generate_hashtags(topic, keywords)
            return hashtags[:10]  # Limit to 10 hashtags
            
        except Exception as e:
            self.logger.warning(f"Hashtag generation failed, using fallback: {e}")
            
            # Fallback hashtag generation
            base_hashtags = [
                topic.lower().replace(' ', '').replace('-', ''),
                'success',
                'growth',
                'tips'
            ]
            
            # Add keyword-based hashtags
            keyword_hashtags = [kw.lower().replace(' ', '') for kw in keywords[:3]]
            
            return list(set(base_hashtags + keyword_hashtags))[:8]
    
    async def _generate_variations(self, base_post: FacebookPostEntity, request: FacebookPostRequest) -> List[FacebookPostEntity]:
        """Generar variaciones del post base."""
        variations = []
        
        try:
            # Generate 2-3 variations with different tones
            variation_tones = [ContentTone.PROFESSIONAL, ContentTone.HUMOROUS, ContentTone.INSPIRING]
            
            for tone in variation_tones[:2]:  # Limit to 2 variations
                if tone == request.tone:
                    continue
                
                variation_request = FacebookPostRequest(
                    topic=request.topic,
                    tone=tone,
                    target_audience=request.target_audience,
                    max_length=request.max_length,
                    keywords=request.keywords,
                    include_hashtags=request.include_hashtags,
                    include_emojis=request.include_emojis,
                    include_call_to_action=request.include_call_to_action
                )
                
                variation_content = await self._generate_content_text(variation_request)
                
                variation = FacebookPostFactory.create_from_specification(
                    specification=PostSpecification(
                        topic=request.topic,
                        post_type=request.post_type,
                        tone=tone,
                        target_audience=request.target_audience,
                        keywords=request.keywords,
                        target_engagement=request.target_engagement
                    ),
                    generation_config=base_post.generation_config,
                    content_text=variation_content,
                    hashtags=base_post.content.hashtags.copy()
                )
                
                # Analyze variation
                variation_analysis = await self.analyze_post(variation)
                variation.set_analysis(variation_analysis)
                
                variations.append(variation)
                
        except Exception as e:
            self.logger.warning(f"Variation generation failed: {e}")
        
        return variations
    
    def _get_optimization_recommendations(self, post: FacebookPostEntity, analysis: FacebookPostAnalysis) -> List[str]:
        """Obtener recomendaciones de optimización."""
        recommendations = []
        
        # Quality-based recommendations
        if analysis.quality_assessment.overall_score < 0.7:
            recommendations.extend(analysis.quality_assessment.improvement_suggestions)
        
        # Engagement-based recommendations
        if analysis.engagement_prediction.engagement_rate < 0.6:
            recommendations.append("Consider adding more engaging elements like questions or polls")
        
        # Content-based recommendations
        if len(post.content.hashtags) < 3:
            recommendations.append("Add more relevant hashtags to increase discoverability")
        
        if not post.content.call_to_action:
            recommendations.append("Add a clear call-to-action to encourage engagement")
        
        # Hashtag recommendations
        if analysis.hashtag_suggestions:
            recommendations.append(f"Consider trending hashtags: {', '.join(analysis.hashtag_suggestions[:3])}")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _count_emojis(self, text: str) -> int:
        """Contar emojis en el texto."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def _determine_quality_tier(self, score: float) -> 'QualityTier':
        """Determinar tier de calidad basado en score."""
        
        if score >= 0.9:
            return QualityTier.PREMIUM
        elif score >= 0.8:
            return QualityTier.EXCELLENT
        elif score >= 0.7:
            return QualityTier.GOOD
        elif score >= 0.6:
            return QualityTier.FAIR
        else:
            return QualityTier.POOR
    
    async def _calculate_optimal_posting_time(self, post: FacebookPostEntity) -> Optional[datetime]:
        """Calcular tiempo óptimo de publicación."""
        try:
            # Simple heuristic based on audience
            base_hour = 12  # Default noon
            
            if post.specification.target_audience == TargetAudience.PROFESSIONALS:
                base_hour = 9  # 9 AM for professionals
            elif post.specification.target_audience == TargetAudience.YOUNG_ADULTS:
                base_hour = 19  # 7 PM for young adults
            elif post.specification.target_audience == TargetAudience.PARENTS:
                base_hour = 20  # 8 PM for parents
            
            optimal_time = datetime.now().replace(hour=base_hour, minute=0, second=0, microsecond=0)
            return optimal_time
            
        except Exception:
            return None
    
    def _update_analytics(self, post: FacebookPostEntity, analysis: FacebookPostAnalysis, processing_time: float) -> None:
        """Actualizar métricas de analytics."""
        self.analytics['posts_generated'] += 1
        self.analytics['total_processing_time'] += processing_time
        
        if analysis:
            score = analysis.get_overall_score()
            
            # Update average quality score
            current_avg = self.analytics['average_quality_score']
            total_posts = self.analytics['posts_generated']
            self.analytics['average_quality_score'] = (current_avg * (total_posts - 1) + score) / total_posts
            
            # Count high quality posts
            if score >= 0.8:
                self.analytics['high_quality_posts'] += 1
    
    def _cache_results(self, post: FacebookPostEntity, analysis: FacebookPostAnalysis) -> None:
        """Cachear resultados para optimización."""
        cache_key = f"{post.identifier.content_hash}_{post.version}"
        
        self._content_cache[cache_key] = post
        self._analysis_cache[cache_key] = analysis
        
        # Maintain cache size (keep last 100 entries)
        if len(self._content_cache) > 100:
            oldest_key = next(iter(self._content_cache))
            del self._content_cache[oldest_key]
            
        if len(self._analysis_cache) > 100:
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
    
    # ===== PUBLIC UTILITY METHODS =====
    
    def get_analytics(self) -> Dict[str, Any]:
        """Obtener métricas de analytics."""
        return self.analytics.copy()
    
    def clear_cache(self) -> None:
        """Limpiar cache."""
        self._content_cache.clear()
        self._analysis_cache.clear()
        self.logger.info("Engine cache cleared")
    
    async def batch_generate_posts(self, requests: List[FacebookPostRequest]) -> List[FacebookPostResponse]:
        """Generar múltiples posts en batch."""
        self.logger.info(f"Batch generating {len(requests)} Facebook posts")
        
        tasks = [self.generate_post(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch generation failed for request {i}: {response}")
                valid_responses.append(FacebookPostResponse(
                    success=False,
                    post=None,
                    variations=[],
                    analysis=None,
                    recommendations=[],
                    processing_time_ms=0,
                    error_message=str(response)
                ))
            else:
                valid_responses.append(response)
        
        return valid_responses 