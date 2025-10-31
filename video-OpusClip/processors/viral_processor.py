"""
Viral Video Processor

Advanced processor for generating viral video variants with enhanced editing capabilities.
Enhanced with LangChain integration for intelligent content analysis and optimization.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any, Tuple
import asyncio
import time
import uuid
import structlog
from dataclasses import dataclass

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..models.viral_models import (
    ViralVideoVariant,
    ViralVideoBatchResponse,
    ViralCaptionConfig,
    CaptionSegment,
    ScreenDivision,
    Transition,
    VideoEffect,
    LangChainAnalysis,
    ContentOptimization,
    ShortVideoOptimization,
    ContentType,
    EngagementType,
    TransitionType,
    ScreenDivisionType,
    CaptionStyle,
    VideoEffect as VideoEffectEnum,
    create_default_caption_config,
    create_split_screen_layout,
    create_viral_transition,
    serializer,
    batch_serializer
)
from ..utils.parallel_utils import (
    HybridParallelProcessor,
    BackendType,
    ParallelConfig,
    parallel_map
)
from .langchain_processor import LangChainVideoProcessor, LangChainConfig

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(slots=True)
class ViralProcessorConfig:
    """Configuration for viral video processing."""
    # Processing settings
    max_variants: int = 10
    min_viral_score: float = 0.3
    max_processing_time: float = 300.0  # 5 minutes
    
    # Quality settings
    min_caption_length: int = 10
    max_caption_length: int = 200
    min_caption_duration: float = 2.0
    max_caption_duration: float = 8.0
    
    # Editing settings
    enable_screen_division: bool = True
    enable_transitions: bool = True
    enable_effects: bool = True
    enable_animations: bool = True
    
    # LangChain integration
    enable_langchain: bool = True
    langchain_api_key: Optional[str] = None
    langchain_model: str = "gpt-4"
    
    # Performance settings
    batch_size: int = 5
    max_workers: int = 4
    timeout: float = 60.0
    
    # Advanced settings
    enable_audit_logging: bool = True
    enable_performance_tracking: bool = True
    enable_error_recovery: bool = True

# =============================================================================
# VIRAL PROCESSOR
# =============================================================================

class ViralVideoProcessor:
    """Advanced processor for generating viral video variants with LangChain optimization."""
    
    def __init__(self, config: Optional[ViralProcessorConfig] = None):
        self.config = config or ViralProcessorConfig()
        
        # Initialize parallel processor
        self.parallel_processor = HybridParallelProcessor(
            ParallelConfig(
                max_workers=self.config.max_workers,
                chunk_size=self.config.batch_size,
                timeout=self.config.timeout
            )
        )
        
        # Initialize LangChain processor if enabled
        self.langchain_processor = None
        if self.config.enable_langchain:
            try:
                langchain_config = LangChainConfig(
                    openai_api_key=self.config.langchain_api_key,
                    model_name=self.config.langchain_model,
                    batch_size=self.config.batch_size,
                    enable_content_analysis=True,
                    enable_engagement_analysis=True,
                    enable_viral_analysis=True,
                    enable_title_optimization=True,
                    enable_caption_optimization=True,
                    enable_timing_optimization=True
                )
                self.langchain_processor = LangChainVideoProcessor(langchain_config)
                logger.info("LangChain processor initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize LangChain processor", error=str(e))
                self.langchain_processor = None
    
    def process_viral_variants(
        self,
        request: VideoClipRequest,
        n_variants: Optional[int] = None,
        audience_profile: Optional[Dict] = None,
        use_langchain: Optional[bool] = None
    ) -> ViralVideoBatchResponse:
        """Process video to generate viral variants with LangChain optimization."""
        start_time = time.perf_counter()
        
        try:
            # Determine if LangChain should be used
            use_langchain = use_langchain if use_langchain is not None else self.config.enable_langchain
            
            # Use LangChain if available and enabled
            if use_langchain and self.langchain_processor:
                logger.info("Processing with LangChain optimization")
                return self._process_with_langchain(request, n_variants, audience_profile)
            else:
                logger.info("Processing with standard optimization")
                return self._process_standard(request, n_variants, audience_profile)
                
        except Exception as e:
            logger.error("Viral processing failed", error=str(e))
            return ViralVideoBatchResponse(
                success=False,
                original_clip_id=request.youtube_url,
                errors=[str(e)]
            )
    
    def _process_with_langchain(
        self,
        request: VideoClipRequest,
        n_variants: Optional[int],
        audience_profile: Optional[Dict]
    ) -> ViralVideoBatchResponse:
        """Process with LangChain optimization."""
        try:
            n_variants = n_variants or self.config.max_variants
            
            # Use LangChain processor
            response = self.langchain_processor.process_video_with_langchain(
                request=request,
                n_variants=n_variants,
                audience_profile=audience_profile
            )
            
            # Enhance with additional viral features
            enhanced_variants = self._enhance_variants_with_viral_features(
                response.variants, request, audience_profile
            )
            
            # Update response with enhanced variants
            response.variants = enhanced_variants
            response.successful_variants = len(enhanced_variants)
            response.average_viral_score = sum(v.viral_score for v in enhanced_variants) / len(enhanced_variants)
            response.best_viral_score = max(v.viral_score for v in enhanced_variants)
            
            return response
            
        except Exception as e:
            logger.error("LangChain processing failed, falling back to standard", error=str(e))
            return self._process_standard(request, n_variants, audience_profile)
    
    def _process_standard(
        self,
        request: VideoClipRequest,
        n_variants: Optional[int],
        audience_profile: Optional[Dict]
    ) -> ViralVideoBatchResponse:
        """Process with standard optimization."""
        try:
            n_variants = n_variants or self.config.max_variants
            
            # Generate variants in parallel
            variant_tasks = []
            for i in range(n_variants):
                task = self._generate_viral_variant_async(
                    request, i, audience_profile
                )
                variant_tasks.append(task)
            
            # Execute tasks
            variants = asyncio.run(self._execute_variant_tasks(variant_tasks))
            
            # Filter and optimize variants
            filtered_variants = self._filter_and_optimize_variants(variants)
            
            processing_time = time.perf_counter() - start_time
            
            return ViralVideoBatchResponse(
                success=True,
                original_clip_id=request.youtube_url,
                variants=filtered_variants,
                processing_time=processing_time,
                total_variants_generated=len(variants),
                successful_variants=len(filtered_variants),
                average_viral_score=sum(v.viral_score for v in filtered_variants) / len(filtered_variants) if filtered_variants else 0.0,
                best_viral_score=max(v.viral_score for v in filtered_variants) if filtered_variants else 0.0
            )
            
        except Exception as e:
            logger.error("Standard processing failed", error=str(e))
            return ViralVideoBatchResponse(
                success=False,
                original_clip_id=request.youtube_url,
                errors=[str(e)]
            )
    
    async def _execute_variant_tasks(self, tasks: List) -> List[ViralVideoVariant]:
        """Execute variant generation tasks."""
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            variants = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Variant generation failed", error=str(result))
                elif result is not None:
                    variants.append(result)
            
            return variants
            
        except Exception as e:
            logger.error("Task execution failed", error=str(e))
            return []
    
    async def _generate_viral_variant_async(
        self,
        request: VideoClipRequest,
        variant_index: int,
        audience_profile: Optional[Dict]
    ) -> Optional[ViralVideoVariant]:
        """Generate a viral variant asynchronously."""
        try:
            # Create base configuration
            config = create_default_caption_config()
            
            # Generate captions
            captions = await self._generate_viral_captions_async(
                request, config, variant_index, audience_profile
            )
            
            # Generate screen division
            screen_division = self._generate_screen_division(variant_index)
            
            # Generate transitions
            transitions = self._generate_transitions(variant_index)
            
            # Generate effects
            effects = self._generate_effects(variant_index)
            
            # Calculate viral score
            viral_score = self._calculate_viral_score(
                captions, screen_division, transitions, effects, audience_profile
            )
            
            # Create variant
            variant = ViralVideoVariant(
                variant_id=f"viral_variant_{request.youtube_url}_{variant_index}_{uuid.uuid4().hex[:8]}",
                title=self._generate_viral_title(request, variant_index),
                description=self._generate_viral_description(request, variant_index),
                viral_score=viral_score,
                engagement_prediction=viral_score * 0.9,
                captions=captions,
                screen_division=screen_division,
                transitions=transitions,
                effects=effects,
                total_duration=request.max_clip_length,
                estimated_views=int(viral_score * 10000),
                estimated_likes=int(viral_score * 5000),
                estimated_shares=int(viral_score * 2000),
                estimated_comments=int(viral_score * 1000),
                tags=self._generate_viral_tags(request, variant_index),
                hashtags=self._generate_viral_hashtags(request, variant_index),
                target_audience=self._get_target_audience(audience_profile),
                created_at=time.time(),
                generation_time=time.perf_counter(),
                model_version="v3.0"
            )
            
            return variant
            
        except Exception as e:
            logger.error(f"Variant {variant_index} generation failed", error=str(e))
            return None
    
    async def _generate_viral_captions_async(
        self,
        request: VideoClipRequest,
        config: ViralCaptionConfig,
        variant_index: int,
        audience_profile: Optional[Dict]
    ) -> List[CaptionSegment]:
        """Generate viral captions asynchronously."""
        try:
            captions = []
            
            # Generate hook caption
            hook_caption = CaptionSegment(
                text=self._generate_hook_text(variant_index),
                start_time=0.5,
                end_time=3.5,
                font_size=28,
                font_color="#FF6B6B",
                styles=[CaptionStyle.BOLD, CaptionStyle.SHADOW],
                animation="fade_in",
                engagement_score=0.9,
                viral_potential=0.8,
                audience_relevance=0.85
            )
            captions.append(hook_caption)
            
            # Generate main captions
            main_captions = self._generate_main_captions(variant_index, config)
            captions.extend(main_captions)
            
            # Generate CTA caption
            cta_caption = CaptionSegment(
                text=self._generate_cta_text(variant_index),
                start_time=request.max_clip_length - 3.0,
                end_time=request.max_clip_length - 0.5,
                font_size=24,
                font_color="#4ECDC4",
                styles=[CaptionStyle.BOLD],
                animation="slide_up",
                engagement_score=0.7,
                viral_potential=0.6,
                audience_relevance=0.75
            )
            captions.append(cta_caption)
            
            return captions
            
        except Exception as e:
            logger.error("Caption generation failed", error=str(e))
            return self._create_fallback_captions(variant_index)
    
    def _generate_hook_text(self, variant_index: int) -> str:
        """Generate hook text for captions."""
        hooks = [
            "🔥 This is going VIRAL! 🔥",
            "🚨 You won't BELIEVE this! 🚨",
            "💥 This just BROKE the internet! 💥",
            "👀 Wait for it... 👀",
            "🎯 This is INSANE! 🎯",
            "⚡ This is the FUTURE! ⚡",
            "🌟 This will BLOW your mind! 🌟",
            "🔥 The internet is OBSESSED! 🔥",
            "🚀 This is NEXT LEVEL! 🚀",
            "💯 This is PURE GOLD! 💯"
        ]
        return hooks[variant_index % len(hooks)]
    
    def _generate_main_captions(self, variant_index: int, config: ViralCaptionConfig) -> List[CaptionSegment]:
        """Generate main captions."""
        captions = []
        
        main_texts = [
            "This content is absolutely AMAZING! 🤯",
            "You need to see this RIGHT NOW! 👀",
            "This is what everyone is talking about! 🗣️",
            "This will change everything! 🔄",
            "The best thing you'll see today! 🏆",
            "This is pure GENIUS! 🧠",
            "You won't regret watching this! ✅",
            "This is the content we all need! ❤️",
            "This is absolutely INCREDIBLE! 😱",
            "This is what viral looks like! 📈"
        ]
        
        for i, text in enumerate(main_texts[:3]):  # Limit to 3 main captions
            caption = CaptionSegment(
                text=text,
                start_time=4.0 + (i * 3.0),
                end_time=7.0 + (i * 3.0),
                font_size=24,
                font_color="#FFFFFF",
                styles=[CaptionStyle.ITALIC],
                animation="slide_in",
                engagement_score=0.7 + (i * 0.1),
                viral_potential=0.6 + (i * 0.1),
                audience_relevance=0.75 + (i * 0.05)
            )
            captions.append(caption)
        
        return captions
    
    def _generate_cta_text(self, variant_index: int) -> str:
        """Generate call-to-action text."""
        ctas = [
            "🔥 FOLLOW for more! 🔥",
            "💯 LIKE & SHARE! 💯",
            "🚀 SUBSCRIBE NOW! 🚀",
            "👀 COMMENT below! 👀",
            "🔥 SAVE this video! 🔥",
            "💯 TAG your friends! 💯",
            "🚀 TURN ON notifications! 🚀",
            "👀 SHARE with everyone! 👀",
            "🔥 FOLLOW for daily content! 🔥",
            "💯 LIKE if you agree! 💯"
        ]
        return ctas[variant_index % len(ctas)]
    
    def _generate_screen_division(self, variant_index: int) -> Optional[ScreenDivision]:
        """Generate screen division layout."""
        if not self.config.enable_screen_division:
            return None
        
        division_types = [
            ScreenDivisionType.SPLIT_HORIZONTAL,
            ScreenDivisionType.SPLIT_VERTICAL,
            ScreenDivisionType.GRID_2X2,
            ScreenDivisionType.PIP
        ]
        
        division_type = division_types[variant_index % len(division_types)]
        return create_split_screen_layout(division_type)
    
    def _generate_transitions(self, variant_index: int) -> List[Transition]:
        """Generate transitions."""
        if not self.config.enable_transitions:
            return []
        
        transition_types = [
            TransitionType.FADE,
            TransitionType.SLIDE,
            TransitionType.ZOOM,
            TransitionType.FLIP,
            TransitionType.GLITCH
        ]
        
        transitions = []
        for i in range(2):  # Generate 2 transitions
            transition_type = transition_types[(variant_index + i) % len(transition_types)]
            transition = create_viral_transition(transition_type)
            transitions.append(transition)
        
        return transitions
    
    def _generate_effects(self, variant_index: int) -> List[VideoEffect]:
        """Generate video effects."""
        if not self.config.enable_effects:
            return []
        
        effect_types = [
            VideoEffectEnum.NEON,
            VideoEffectEnum.GLITCH,
            VideoEffectEnum.SLOW_MOTION,
            VideoEffectEnum.MIRROR,
            VideoEffectEnum.SEPIA
        ]
        
        effects = []
        for i in range(2):  # Generate 2 effects
            effect_type = effect_types[(variant_index + i) % len(effect_types)]
            effect = VideoEffect(
                effect_type=effect_type,
                intensity=0.7,
                duration=3.0,
                viral_impact=0.8,
                audience_appeal=0.7
            )
            effects.append(effect)
        
        return effects
    
    def _calculate_viral_score(
        self,
        captions: List[CaptionSegment],
        screen_division: Optional[ScreenDivision],
        transitions: List[Transition],
        effects: List[VideoEffect],
        audience_profile: Optional[Dict]
    ) -> float:
        """Calculate viral score for the variant."""
        base_score = 0.5
        
        # Caption scoring
        if captions:
            caption_scores = [c.engagement_score for c in captions]
            base_score += sum(caption_scores) / len(caption_scores) * 0.2
        
        # Screen division scoring
        if screen_division and screen_division.engagement_optimized:
            base_score += 0.1
        
        # Transition scoring
        if transitions:
            transition_scores = [t.engagement_impact for t in transitions]
            base_score += sum(transition_scores) / len(transition_scores) * 0.1
        
        # Effect scoring
        if effects:
            effect_scores = [e.viral_impact for e in effects]
            base_score += sum(effect_scores) / len(effect_scores) * 0.1
        
        # Audience fit scoring
        if audience_profile:
            base_score += 0.05
        
        return min(base_score, 1.0)
    
    def _generate_viral_title(self, request: VideoClipRequest, variant_index: int) -> str:
        """Generate viral title."""
        titles = [
            "🔥 This is GOING VIRAL! 🔥",
            "🚨 You won't BELIEVE this! 🚨",
            "💥 This just BROKE the internet! 💥",
            "👀 Wait for it... 👀",
            "🎯 This is INSANE! 🎯",
            "⚡ This is the FUTURE! ⚡",
            "🌟 This will BLOW your mind! 🌟",
            "🔥 The internet is OBSESSED! 🔥",
            "🚀 This is NEXT LEVEL! 🚀",
            "💯 This is PURE GOLD! 💯"
        ]
        return titles[variant_index % len(titles)]
    
    def _generate_viral_description(self, request: VideoClipRequest, variant_index: int) -> str:
        """Generate viral description."""
        descriptions = [
            "🔥 This content is absolutely AMAZING! Don't miss this viral moment! 🔥",
            "🚨 You need to see this RIGHT NOW! This is going viral! 🚨",
            "💥 This just BROKE the internet! Share with everyone! 💥",
            "👀 Wait for it... This is INSANE! 👀",
            "🎯 This is what everyone is talking about! Pure GOLD! 🎯",
            "⚡ This is the FUTURE! You won't believe your eyes! ⚡",
            "🌟 This will BLOW your mind! Must watch! 🌟",
            "🔥 The internet is OBSESSED! This is NEXT LEVEL! 🔥",
            "🚀 This is what viral looks like! Absolutely INCREDIBLE! 🚀",
            "💯 This is PURE GOLD! The best content you'll see today! 💯"
        ]
        return descriptions[variant_index % len(descriptions)]
    
    def _generate_viral_tags(self, request: VideoClipRequest, variant_index: int) -> List[str]:
        """Generate viral tags."""
        base_tags = ["viral", "trending", "amazing", "incredible", "mindblowing"]
        variant_tags = [
            ["fire", "hot", "lit"],
            ["crazy", "insane", "wild"],
            ["epic", "legendary", "iconic"],
            ["mindblowing", "jawdropping", "stunning"],
            ["viral", "trending", "popular"],
            ["amazing", "incredible", "fantastic"],
            ["awesome", "brilliant", "genius"],
            ["perfect", "flawless", "excellent"],
            ["outstanding", "remarkable", "extraordinary"],
            ["phenomenal", "spectacular", "magnificent"]
        ]
        
        tags = base_tags + variant_tags[variant_index % len(variant_tags)]
        return tags[:10]  # Limit to 10 tags
    
    def _generate_viral_hashtags(self, request: VideoClipRequest, variant_index: int) -> List[str]:
        """Generate viral hashtags."""
        base_hashtags = ["#viral", "#trending", "#amazing"]
        variant_hashtags = [
            ["#fire", "#hot", "#lit"],
            ["#crazy", "#insane", "#wild"],
            ["#epic", "#legendary", "#iconic"],
            ["#mindblowing", "#jawdropping", "#stunning"],
            ["#viral", "#trending", "#popular"],
            ["#amazing", "#incredible", "#fantastic"],
            ["#awesome", "#brilliant", "#genius"],
            ["#perfect", "#flawless", "#excellent"],
            ["#outstanding", "#remarkable", "#extraordinary"],
            ["#phenomenal", "#spectacular", "#magnificent"]
        ]
        
        hashtags = base_hashtags + variant_hashtags[variant_index % len(variant_hashtags)]
        return hashtags[:15]  # Limit to 15 hashtags
    
    def _get_target_audience(self, audience_profile: Optional[Dict]) -> List[str]:
        """Get target audience from profile."""
        if audience_profile:
            return audience_profile.get("interests", ["general"])
        return ["general", "social_media_users", "young_adults"]
    
    def _enhance_variants_with_viral_features(
        self,
        variants: List[ViralVideoVariant],
        request: VideoClipRequest,
        audience_profile: Optional[Dict]
    ) -> List[ViralVideoVariant]:
        """Enhance variants with additional viral features."""
        try:
            enhanced_variants = []
            
            for variant in variants:
                # Enhance viral score
                variant.viral_score = min(variant.viral_score * 1.1, 1.0)
                
                # Add viral elements
                variant.ai_viral_elements.extend([
                    "emotional_triggers",
                    "trending_topics",
                    "shareable_content",
                    "engagement_hooks"
                ])
                
                # Enhance captions with viral elements
                for caption in variant.captions:
                    if caption.text and "🔥" in caption.text:
                        caption.viral_potential = min(caption.viral_potential * 1.2, 1.0)
                
                enhanced_variants.append(variant)
            
            return enhanced_variants
            
        except Exception as e:
            logger.error("Variant enhancement failed", error=str(e))
            return variants
    
    def _filter_and_optimize_variants(self, variants: List[ViralVideoVariant]) -> List[ViralVideoVariant]:
        """Filter and optimize variants."""
        try:
            # Filter by minimum viral score
            filtered_variants = [
                v for v in variants 
                if v.viral_score >= self.config.min_viral_score
            ]
            
            # Sort by viral score
            filtered_variants.sort(key=lambda v: v.viral_score, reverse=True)
            
            # Limit to max variants
            if len(filtered_variants) > self.config.max_variants:
                filtered_variants = filtered_variants[:self.config.max_variants]
            
            return filtered_variants
            
        except Exception as e:
            logger.error("Variant filtering failed", error=str(e))
            return variants
    
    def _create_fallback_captions(self, variant_index: int) -> List[CaptionSegment]:
        """Create fallback captions."""
        return [
            CaptionSegment(
                text=f"🔥 Viral Content {variant_index + 1}! 🔥",
                start_time=1.0,
                end_time=4.0,
                font_size=24,
                styles=[CaptionStyle.BOLD],
                engagement_score=0.6,
                viral_potential=0.5,
                audience_relevance=0.6
            )
        ]
    
    # =============================================================================
    # BATCH PROCESSING
    # =============================================================================
    
    def process_batch(
        self,
        requests: List[VideoClipRequest],
        n_variants_per_request: int = 5,
        audience_profiles: Optional[List[Dict]] = None
    ) -> List[ViralVideoBatchResponse]:
        """Process multiple requests in batch."""
        try:
            # Prepare batch tasks
            batch_tasks = []
            for i, request in enumerate(requests):
                audience_profile = audience_profiles[i] if audience_profiles and i < len(audience_profiles) else None
                task = self.process_viral_variants(
                    request=request,
                    n_variants=n_variants_per_request,
                    audience_profile=audience_profile
                )
                batch_tasks.append(task)
            
            # Execute batch processing
            results = self.parallel_processor.process_batch(
                batch_tasks,
                backend=BackendType.ASYNCIO
            )
            
            return results
            
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            return [
                ViralVideoBatchResponse(
                    success=False,
                    original_clip_id=req.youtube_url,
                    errors=[str(e)]
                )
                for req in requests
            ]

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_viral_processor(
    enable_langchain: bool = True,
    api_key: Optional[str] = None,
    max_variants: int = 10
) -> ViralVideoProcessor:
    """Create a viral video processor."""
    config = ViralProcessorConfig(
        enable_langchain=enable_langchain,
        langchain_api_key=api_key,
        max_variants=max_variants,
        enable_screen_division=True,
        enable_transitions=True,
        enable_effects=True,
        enable_animations=True
    )
    
    return ViralVideoProcessor(config)

def create_optimized_viral_processor(
    api_key: Optional[str] = None,
    batch_size: int = 5,
    max_workers: int = 4
) -> ViralVideoProcessor:
    """Create an optimized viral processor for production use."""
    config = ViralProcessorConfig(
        enable_langchain=True,
        langchain_api_key=api_key,
        max_variants=10,
        batch_size=batch_size,
        max_workers=max_workers,
        enable_audit_logging=True,
        enable_performance_tracking=True,
        enable_error_recovery=True
    )
    
    return ViralVideoProcessor(config)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ViralVideoProcessor',
    'ViralProcessorConfig',
    'create_viral_processor',
    'create_optimized_viral_processor'
] 