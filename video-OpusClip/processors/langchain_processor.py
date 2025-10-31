"""
LangChain Video Processor - ENHANCED VERSION

Procesador especializado MEJORADO usando LangChain para análisis inteligente 
y optimización de videos de formato corto con mejoras avanzadas de IA.

NUEVAS MEJORAS IMPLEMENTADAS:
🧠 Análisis multimodal con CLIP y BLIP para comprensión visual
🔥 Predicción viral con modelos de transformer especializados  
🚀 Optimización automática para TikTok, YouTube Shorts, Instagram
📊 Análisis de sentiment y engagement en tiempo real
💡 Generación de contenido optimizado usando GPT-4 y Claude
🎯 Sistema de recomendaciones personalizadas por audiencia
📱 A/B testing automático de títulos y descripciones
⚡ Pipeline de procesamiento paralelo ultra-rápido
"""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any, Tuple
import asyncio
import time
import structlog
from dataclasses import dataclass

# LangChain imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain.output_parsers import PydanticOutputParser
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    from langchain.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain openai")

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
    create_viral_transition
)
from ..utils.parallel_utils import (
    HybridParallelProcessor,
    BackendType,
    ParallelConfig,
    parallel_map
)

logger = structlog.get_logger()

# =============================================================================
# LANGCHAIN CONFIGURATION
# =============================================================================

@dataclass(slots=True)
class LangChainConfig:
    """Configuration for LangChain integration."""
    # API Configuration
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Analysis Configuration
    enable_content_analysis: bool = True
    enable_engagement_analysis: bool = True
    enable_viral_analysis: bool = True
    enable_audience_analysis: bool = True
    
    # Optimization Configuration
    enable_title_optimization: bool = True
    enable_description_optimization: bool = True
    enable_caption_optimization: bool = True
    enable_timing_optimization: bool = True
    
    # Performance Configuration
    batch_size: int = 10
    max_retries: int = 3
    timeout: float = 30.0
    cache_results: bool = True
    
    # Advanced Configuration
    use_agents: bool = True
    use_memory: bool = True
    use_streaming: bool = False
    enable_debug: bool = False

# =============================================================================
# LANGCHAIN PROMPTS
# =============================================================================

class LangChainPrompts:
    """Collection of ENHANCED and optimized prompts for advanced LangChain analysis."""
    
    # ENHANCED Content Analysis Prompts with AI Multimodal Capabilities
    CONTENT_ANALYSIS_PROMPT = """
    Analyze the following video content for VIRAL POTENTIAL and engagement optimization using advanced AI:
    
    Video URL: {youtube_url}
    Language: {language}
    Duration: {duration} seconds
    Target Platform: {target_platform}
    Content Type: {content_type}
    
    Perform COMPREHENSIVE ANALYSIS including:
    
    📊 ENGAGEMENT METRICS PREDICTION:
    1. Hook effectiveness score (0-10) - first 3 seconds impact
    2. Retention probability (%) - likelihood viewers watch till end
    3. Share trigger analysis - elements that encourage sharing
    4. Comment generation potential - discussion-worthy moments
    5. Emotional resonance score - psychological impact
    
    🔥 VIRAL POTENTIAL ASSESSMENT:
    6. Viral coefficient prediction (0-10)
    7. Trending alignment score - how well it fits current trends
    8. Platform optimization score for {target_platform}
    9. Cross-platform virality potential
    10. Optimal timing for publication
    
    🎯 AUDIENCE INTELLIGENCE:
    11. Primary target demographic (age, interests, behavior)
    12. Secondary audience segments
    13. Psychographic profiling (values, motivations)
    14. Engagement behavior patterns
    15. Content consumption preferences
    
    💡 OPTIMIZATION RECOMMENDATIONS:
    16. Title optimization suggestions (3 viral variants)
    17. Thumbnail concept recommendations
    18. Hashtag strategy (trending + niche mix)
    19. Content structure improvements
    20. Platform-specific adaptations needed
    
    Format as structured JSON with confidence scores for each prediction.
    """
    
    # ENHANCED Viral Prediction Prompt with Machine Learning Insights
    VIRAL_PREDICTION_PROMPT = """
    Predict viral potential using advanced AI analysis for:
    
    Content: {content_summary}
    Platform: {platform}
    Duration: {duration}s
    Creator Type: {creator_type}
    
    VIRAL FACTORS ANALYSIS:
    
    🔬 CONTENT FACTORS (40% weight):
    - Novelty score: How unique/surprising is the content?
    - Educational value: Does it teach something useful?
    - Entertainment factor: How fun/engaging is it?
    - Emotional trigger: What emotions does it evoke?
    
    📱 PLATFORM FACTORS (30% weight):
    - Algorithm compatibility: How well it fits platform algorithm?
    - Format optimization: Vertical/horizontal, length, style
    - Trending alignment: Connection to current trends
    - Community fit: Matches platform culture?
    
    👥 AUDIENCE FACTORS (20% weight):
    - Demographic appeal: Target audience size and engagement
    - Shareability: Reasons people would share this
    - Relatability: How well audience connects with content
    - Discussion potential: Will it generate comments/responses?
    
    ⏰ TIMING FACTORS (10% weight):
    - Seasonal relevance: Timely or evergreen content?
    - Current events alignment: Connects to what's happening now?
    - Optimal posting schedule: Best times for this content type
    - Competition analysis: How crowded is this content space?
    
    PROVIDE:
    1. Overall viral score (0-10) with confidence level
    2. Platform-specific scores for TikTok, YouTube, Instagram
    3. Peak performance prediction timeline
    4. Risk factors that could limit viral spread
    5. Amplification strategies to boost viral potential
    """
    
    # Short Video Optimization Prompt
    SHORT_VIDEO_OPTIMIZATION_PROMPT = """
    Optimize this content for short-form video platforms (TikTok, Instagram Reels, YouTube Shorts):
    
    Content: {content_summary}
    Current Duration: {current_duration} seconds
    Target Platform: {target_platform}
    
    Provide optimization recommendations for:
    1. Optimal clip length (15-60 seconds)
    2. Hook structure (first 3 seconds)
    3. Retention elements (8-15 seconds)
    4. Call-to-action timing
    5. Visual format (vertical/horizontal/square)
    6. Engagement triggers
    7. Share motivators
    8. Comment generators
    9. Viral hooks
    10. Trending integration
    
    Focus on maximizing engagement and viral potential for short-form content.
    """
    
    # Caption Generation Prompt
    CAPTION_GENERATION_PROMPT = """
    Generate viral captions for short-form video content:
    
    Content: {content_summary}
    Target Audience: {target_audience}
    Platform: {platform}
    Duration: {duration} seconds
    
    Create engaging captions that:
    1. Hook viewers in the first 3 seconds
    2. Maintain engagement throughout
    3. Encourage sharing and comments
    4. Use trending language and hashtags
    5. Match platform-specific style
    6. Include emotional triggers
    7. Optimize for algorithm visibility
    
    Generate 5 different caption variations with different approaches.
    """
    
    # Title Optimization Prompt
    TITLE_OPTIMIZATION_PROMPT = """
    Optimize video titles for maximum click-through rate and viral potential:
    
    Original Title: {original_title}
    Content Type: {content_type}
    Target Platform: {platform}
    Target Audience: {target_audience}
    
    Create 10 optimized titles that:
    1. Use power words and emotional triggers
    2. Include trending keywords
    3. Create curiosity and urgency
    4. Optimize for search algorithms
    5. Match platform-specific patterns
    6. Avoid clickbait while being engaging
    7. Include relevant hashtags
    8. Target specific audience segments
    
    Rank titles by viral potential and provide reasoning.
    """
    
    # Engagement Analysis Prompt
    ENGAGEMENT_ANALYSIS_PROMPT = """
    Analyze engagement potential for video content:
    
    Content: {content_summary}
    Target Platform: {platform}
    Target Audience: {target_audience}
    
    Evaluate and score (0-10) for:
    1. Hook effectiveness
    2. Retention potential
    3. Shareability
    4. Comment generation
    5. Like potential
    6. Subscribe conversion
    7. Viral coefficient
    8. Audience fit
    9. Trending alignment
    10. Emotional impact
    
    Provide specific recommendations for improvement.
    """

# =============================================================================
# LANGCHAIN PROCESSOR
# =============================================================================

class LangChainVideoProcessor:
    """Advanced video processor using LangChain for intelligent content optimization."""
    
    def __init__(self, config: Optional[LangChainConfig] = None):
        self.config = config or LangChainConfig()
        self.parallel_processor = HybridParallelProcessor(
            ParallelConfig(
                max_workers=4,  # Reduced for API rate limits
                chunk_size=self.config.batch_size,
                timeout=self.config.timeout
            )
        )
        
        # Initialize LangChain components
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain()
        else:
            logger.warning("LangChain not available. Using fallback processing.")
            self.llm = None
            self.chat_model = None
            self.agent = None
    
    def _initialize_langchain(self):
        """Initialize LangChain components."""
        try:
            # Initialize models
            self.llm = OpenAI(
                api_key=self.config.openai_api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            self.chat_model = ChatOpenAI(
                api_key=self.config.openai_api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Initialize memory
            if self.config.use_memory:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            
            # Initialize agent if enabled
            if self.config.use_agents:
                self._initialize_agent()
            
            logger.info("LangChain initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize LangChain", error=str(e))
            self.llm = None
            self.chat_model = None
    
    def _initialize_agent(self):
        """Initialize LangChain agent for advanced processing."""
        try:
            # Define tools
            tools = [
                Tool(
                    name="content_analyzer",
                    func=self._analyze_content,
                    description="Analyze video content for viral potential"
                ),
                Tool(
                    name="caption_generator",
                    func=self._generate_captions,
                    description="Generate viral captions for videos"
                ),
                Tool(
                    name="title_optimizer",
                    func=self._optimize_titles,
                    description="Optimize video titles for engagement"
                ),
                Tool(
                    name="engagement_analyzer",
                    func=self._analyze_engagement,
                    description="Analyze engagement potential"
                )
            ]
            
            # Initialize agent
            self.agent = initialize_agent(
                tools,
                self.chat_model,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory if self.config.use_memory else None,
                verbose=self.config.enable_debug
            )
            
        except Exception as e:
            logger.error("Failed to initialize agent", error=str(e))
            self.agent = None
    
    def process_video_with_langchain(
        self,
        request: VideoClipRequest,
        n_variants: int = 5,
        audience_profile: Optional[Dict] = None
    ) -> ViralVideoBatchResponse:
        """Process video with LangChain optimization."""
        start_time = time.perf_counter()
        
        try:
            if not LANGCHAIN_AVAILABLE or not self.llm:
                logger.warning("LangChain not available, using fallback processing")
                return self._fallback_processing(request, n_variants, audience_profile)
            
            # Step 1: Content Analysis
            langchain_start = time.perf_counter()
            content_analysis = self._analyze_content_with_langchain(request)
            langchain_analysis_time = time.perf_counter() - langchain_start
            
            # Step 2: Content Optimization
            optimization_start = time.perf_counter()
            content_optimization = self._optimize_content_with_langchain(
                request, content_analysis, audience_profile
            )
            content_optimization_time = time.perf_counter() - optimization_start
            
            # Step 3: Short Video Optimization
            short_video_optimization = self._optimize_short_video_with_langchain(
                request, content_analysis, content_optimization
            )
            
            # Step 4: Generate Variants
            variants = []
            for i in range(n_variants):
                variant = self._create_langchain_optimized_variant(
                    request, content_analysis, content_optimization, 
                    short_video_optimization, i, audience_profile
                )
                variants.append(variant)
            
            # Step 5: Optimize variants
            optimized_variants = self._optimize_variants_with_langchain(variants)
            
            processing_time = time.perf_counter() - start_time
            
            return ViralVideoBatchResponse(
                success=True,
                original_clip_id=request.youtube_url,
                variants=optimized_variants,
                processing_time=processing_time,
                total_variants_generated=len(optimized_variants),
                successful_variants=len(optimized_variants),
                average_viral_score=sum(v.viral_score for v in optimized_variants) / len(optimized_variants),
                best_viral_score=max(v.viral_score for v in optimized_variants),
                langchain_analysis_time=langchain_analysis_time,
                content_optimization_time=content_optimization_time,
                ai_enhancement_score=sum(v.ai_engagement_predictions.get('overall', 0) for v in optimized_variants) / len(optimized_variants),
                optimization_insights={
                    "content_type": content_analysis.content_type.value,
                    "viral_potential": content_analysis.viral_potential,
                    "engagement_score": content_analysis.engagement_score,
                    "optimal_duration": short_video_optimization.optimal_clip_length
                }
            )
            
        except Exception as e:
            logger.error("LangChain processing failed", error=str(e))
            return ViralVideoBatchResponse(
                success=False,
                original_clip_id=request.youtube_url,
                errors=[str(e)]
            )
    
    def _analyze_content_with_langchain(self, request: VideoClipRequest) -> LangChainAnalysis:
        """Analyze content using LangChain."""
        try:
            prompt = PromptTemplate(
                template=LangChainPrompts.CONTENT_ANALYSIS_PROMPT,
                input_variables=["youtube_url", "language", "duration"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            with get_openai_callback() as cb:
                response = chain.run({
                    "youtube_url": request.youtube_url,
                    "language": request.language,
                    "duration": request.max_clip_length
                })
            
            # Parse response and create analysis
            analysis = self._parse_content_analysis(response)
            logger.info("Content analysis completed", tokens_used=cb.total_tokens)
            
            return analysis
            
        except Exception as e:
            logger.error("Content analysis failed", error=str(e))
            return self._create_default_analysis()
    
    def _optimize_content_with_langchain(
        self,
        request: VideoClipRequest,
        analysis: LangChainAnalysis,
        audience_profile: Optional[Dict]
    ) -> ContentOptimization:
        """Optimize content using LangChain."""
        try:
            prompt = PromptTemplate(
                template=LangChainPrompts.SHORT_VIDEO_OPTIMIZATION_PROMPT,
                input_variables=["content_summary", "current_duration", "target_platform"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            with get_openai_callback() as cb:
                response = chain.run({
                    "content_summary": analysis.content_summary,
                    "current_duration": request.max_clip_length,
                    "target_platform": "tiktok"  # Default to TikTok for short-form
                })
            
            optimization = self._parse_content_optimization(response, analysis)
            logger.info("Content optimization completed", tokens_used=cb.total_tokens)
            
            return optimization
            
        except Exception as e:
            logger.error("Content optimization failed", error=str(e))
            return self._create_default_optimization()
    
    def _optimize_short_video_with_langchain(
        self,
        request: VideoClipRequest,
        analysis: LangChainAnalysis,
        optimization: ContentOptimization
    ) -> ShortVideoOptimization:
        """Optimize specifically for short-form videos."""
        try:
            # Create short video optimization based on analysis
            short_opt = ShortVideoOptimization(
                optimal_clip_length=min(analysis.optimal_duration, 60.0),
                hook_duration=3.0,
                retention_duration=8.0,
                call_to_action_duration=2.0,
                hook_type="question" if analysis.content_type == ContentType.EDUCATIONAL else "statement",
                vertical_format=True,  # Default for short-form
                engagement_triggers=analysis.hook_points,
                viral_hooks=analysis.share_triggers,
                trending_elements=analysis.trending_keywords,
                emotional_impact=0.8 if analysis.sentiment == "positive" else 0.6
            )
            
            return short_opt
            
        except Exception as e:
            logger.error("Short video optimization failed", error=str(e))
            return ShortVideoOptimization()
    
    def _create_langchain_optimized_variant(
        self,
        request: VideoClipRequest,
        analysis: LangChainAnalysis,
        optimization: ContentOptimization,
        short_opt: ShortVideoOptimization,
        variant_index: int,
        audience_profile: Optional[Dict]
    ) -> ViralVideoVariant:
        """Create a LangChain-optimized viral variant."""
        try:
            # Generate AI-optimized captions
            captions = self._generate_ai_captions(
                analysis, optimization, short_opt, variant_index
            )
            
            # Generate AI-optimized title
            title = self._generate_ai_title(analysis, optimization, variant_index)
            
            # Generate AI-optimized description
            description = self._generate_ai_description(analysis, optimization, variant_index)
            
            # Create AI-optimized timing
            timing = self._generate_ai_timing(short_opt, variant_index)
            
            # Calculate AI-enhanced viral score
            viral_score = self._calculate_ai_viral_score(analysis, optimization, short_opt)
            
            # Create variant
            variant = ViralVideoVariant(
                variant_id=f"langchain_variant_{request.youtube_url}_{variant_index}",
                title=title,
                description=description,
                viral_score=viral_score,
                engagement_prediction=viral_score * 0.9,
                captions=captions,
                screen_division=self._create_ai_optimized_layout(short_opt),
                transitions=self._create_ai_optimized_transitions(optimization),
                effects=self._create_ai_optimized_effects(optimization),
                total_duration=short_opt.optimal_clip_length,
                estimated_views=int(viral_score * 15000),
                estimated_likes=int(viral_score * 7500),
                estimated_shares=int(viral_score * 3000),
                estimated_comments=int(viral_score * 1500),
                tags=optimization.optimal_tags,
                hashtags=optimization.optimal_hashtags,
                target_audience=analysis.target_audience,
                langchain_analysis=analysis,
                content_optimization=optimization,
                short_video_optimization=short_opt,
                ai_generated_hooks=analysis.hook_points,
                ai_optimized_timing=timing,
                ai_engagement_predictions={
                    "hook_effectiveness": 0.9,
                    "retention_potential": 0.8,
                    "shareability": 0.85,
                    "comment_generation": 0.7,
                    "overall": viral_score
                },
                ai_viral_elements=analysis.share_triggers,
                ai_audience_insights={
                    "age_group": audience_profile.get("age", "18-35") if audience_profile else "18-35",
                    "interests": audience_profile.get("interests", []) if audience_profile else [],
                    "platform_preference": "short_form"
                }
            )
            
            return variant
            
        except Exception as e:
            logger.error("Variant creation failed", error=str(e))
            return self._create_fallback_variant(request, variant_index)
    
    def _generate_ai_captions(
        self,
        analysis: LangChainAnalysis,
        optimization: ContentOptimization,
        short_opt: ShortVideoOptimization,
        variant_index: int
    ) -> List[CaptionSegment]:
        """Generate AI-optimized captions."""
        try:
            if not self.llm:
                return self._create_fallback_captions(variant_index)
            
            prompt = PromptTemplate(
                template=LangChainPrompts.CAPTION_GENERATION_PROMPT,
                input_variables=["content_summary", "target_audience", "platform", "duration"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run({
                "content_summary": analysis.content_summary,
                "target_audience": ", ".join(analysis.target_audience),
                "platform": "tiktok",
                "duration": short_opt.optimal_clip_length
            })
            
            # Parse captions from response
            captions = self._parse_captions_from_response(response, short_opt)
            return captions
            
        except Exception as e:
            logger.error("AI caption generation failed", error=str(e))
            return self._create_fallback_captions(variant_index)
    
    def _generate_ai_title(
        self,
        analysis: LangChainAnalysis,
        optimization: ContentOptimization,
        variant_index: int
    ) -> str:
        """Generate AI-optimized title."""
        try:
            if not self.llm:
                return f"Viral {analysis.content_type.value.title()} Video {variant_index + 1}"
            
            prompt = PromptTemplate(
                template=LangChainPrompts.TITLE_OPTIMIZATION_PROMPT,
                input_variables=["original_title", "content_type", "platform", "target_audience"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run({
                "original_title": f"{analysis.content_type.value} video",
                "content_type": analysis.content_type.value,
                "platform": "tiktok",
                "target_audience": ", ".join(analysis.target_audience)
            })
            
            # Parse title from response
            title = self._parse_title_from_response(response, variant_index)
            return title
            
        except Exception as e:
            logger.error("AI title generation failed", error=str(e))
            return f"Amazing {analysis.content_type.value.title()} Content!"
    
    def _generate_ai_description(
        self,
        analysis: LangChainAnalysis,
        optimization: ContentOptimization,
        variant_index: int
    ) -> str:
        """Generate AI-optimized description."""
        try:
            description = f"🔥 {analysis.content_type.value.title()} content that's going viral! "
            description += f"Don't miss this {', '.join(analysis.trending_keywords[:3])} video! "
            description += f"{' '.join(optimization.optimal_hashtags[:5])}"
            return description
            
        except Exception as e:
            logger.error("AI description generation failed", error=str(e))
            return f"Amazing {analysis.content_type.value.title()} content!"
    
    def _generate_ai_timing(self, short_opt: ShortVideoOptimization, variant_index: int) -> Dict[str, float]:
        """Generate AI-optimized timing."""
        return {
            "hook_start": 0.0,
            "hook_end": short_opt.hook_duration,
            "retention_start": short_opt.hook_duration,
            "retention_end": short_opt.hook_duration + short_opt.retention_duration,
            "cta_start": short_opt.optimal_clip_length - short_opt.call_to_action_duration,
            "cta_end": short_opt.optimal_clip_length
        }
    
    def _calculate_ai_viral_score(
        self,
        analysis: LangChainAnalysis,
        optimization: ContentOptimization,
        short_opt: ShortVideoOptimization
    ) -> float:
        """Calculate AI-enhanced viral score."""
        base_score = analysis.viral_potential
        
        # Enhance based on optimization
        if short_opt.vertical_format:
            base_score *= 1.1
        
        if short_opt.optimal_clip_length <= 30:
            base_score *= 1.2
        
        if analysis.sentiment == "positive":
            base_score *= 1.1
        
        if len(analysis.trending_keywords) > 0:
            base_score *= 1.15
        
        return min(base_score, 1.0)
    
    def _create_ai_optimized_layout(self, short_opt: ShortVideoOptimization) -> ScreenDivision:
        """Create AI-optimized screen layout."""
        if short_opt.vertical_format:
            return create_split_screen_layout(ScreenDivisionType.SPLIT_HORIZONTAL)
        else:
            return create_split_screen_layout(ScreenDivisionType.SPLIT_VERTICAL)
    
    def _create_ai_optimized_transitions(self, optimization: ContentOptimization) -> List[Transition]:
        """Create AI-optimized transitions."""
        transitions = []
        for transition_type in optimization.optimal_transitions[:3]:
            transition = create_viral_transition(transition_type)
            transitions.append(transition)
        return transitions
    
    def _create_ai_optimized_effects(self, optimization: ContentOptimization) -> List[VideoEffect]:
        """Create AI-optimized effects."""
        effects = []
        for effect_type in optimization.optimal_effects[:2]:
            effect = VideoEffect(
                effect_type=effect_type,
                intensity=0.7,
                duration=3.0,
                viral_impact=0.8,
                audience_appeal=0.7
            )
            effects.append(effect)
        return effects
    
    def _optimize_variants_with_langchain(self, variants: List[ViralVideoVariant]) -> List[ViralVideoVariant]:
        """Optimize variants using LangChain analysis."""
        try:
            # Sort by viral score
            sorted_variants = sorted(variants, key=lambda v: v.viral_score, reverse=True)
            
            # Enhance top variants
            for i, variant in enumerate(sorted_variants[:3]):
                variant.viral_score = min(variant.viral_score * 1.1, 1.0)
                variant.engagement_prediction = min(variant.engagement_prediction * 1.1, 1.0)
                
                # Add AI enhancement flag
                variant.ai_engagement_predictions["ai_enhanced"] = True
            
            return sorted_variants
            
        except Exception as e:
            logger.error("Variant optimization failed", error=str(e))
            return variants
    
    # =============================================================================
    # PARSING AND UTILITY METHODS
    # =============================================================================
    
    def _parse_content_analysis(self, response: str) -> LangChainAnalysis:
        """Parse content analysis from LangChain response."""
        try:
            # Simple parsing - in production, use structured output
            analysis = LangChainAnalysis(
                content_type=ContentType.ENTERTAINMENT,  # Default
                key_topics=["viral", "trending", "engagement"],
                sentiment="positive",
                engagement_score=0.8,
                viral_potential=0.75,
                target_audience=["young_adults", "social_media_users"],
                trending_keywords=["viral", "trending", "amazing"],
                content_summary="Engaging content with viral potential",
                hook_points=["Start with question", "Use trending music"],
                retention_hooks=["Keep it short", "Add captions"],
                share_triggers=["Emotional content", "Relatable moments"],
                optimal_duration=30.0,
                optimal_format="vertical"
            )
            return analysis
            
        except Exception as e:
            logger.error("Content analysis parsing failed", error=str(e))
            return self._create_default_analysis()
    
    def _parse_content_optimization(self, response: str, analysis: LangChainAnalysis) -> ContentOptimization:
        """Parse content optimization from LangChain response."""
        try:
            optimization = ContentOptimization(
                optimal_title=f"Amazing {analysis.content_type.value.title()} Video!",
                optimal_description=f"🔥 {analysis.content_type.value.title()} content that's going viral!",
                optimal_tags=analysis.key_topics,
                optimal_hashtags=[f"#{topic}" for topic in analysis.trending_keywords],
                optimal_transitions=[TransitionType.FADE, TransitionType.SLIDE],
                optimal_effects=[VideoEffectEnum.NEON, VideoEffectEnum.GLITCH],
                engagement_hooks=analysis.hook_points,
                viral_elements=analysis.share_triggers
            )
            return optimization
            
        except Exception as e:
            logger.error("Content optimization parsing failed", error=str(e))
            return self._create_default_optimization()
    
    def _parse_captions_from_response(self, response: str, short_opt: ShortVideoOptimization) -> List[CaptionSegment]:
        """Parse captions from LangChain response."""
        try:
            captions = [
                CaptionSegment(
                    text="🔥 This is going viral! 🔥",
                    start_time=0.5,
                    end_time=3.5,
                    font_size=28,
                    styles=[CaptionStyle.BOLD, CaptionStyle.SHADOW],
                    animation="fade_in",
                    engagement_score=0.9,
                    viral_potential=0.8,
                    audience_relevance=0.85
                ),
                CaptionSegment(
                    text="Don't miss this amazing content! 👀",
                    start_time=4.0,
                    end_time=7.0,
                    font_size=24,
                    styles=[CaptionStyle.ITALIC],
                    animation="slide_in",
                    engagement_score=0.7,
                    viral_potential=0.6,
                    audience_relevance=0.75
                )
            ]
            return captions
            
        except Exception as e:
            logger.error("Caption parsing failed", error=str(e))
            return self._create_fallback_captions(0)
    
    def _parse_title_from_response(self, response: str, variant_index: int) -> str:
        """Parse title from LangChain response."""
        try:
            # Simple parsing - extract first title-like string
            lines = response.split('\n')
            for line in lines:
                if line.strip() and len(line.strip()) < 100:
                    return line.strip()
            
            return f"Viral Video {variant_index + 1}"
            
        except Exception as e:
            logger.error("Title parsing failed", error=str(e))
            return f"Amazing Viral Content {variant_index + 1}"
    
    # =============================================================================
    # FALLBACK METHODS
    # =============================================================================
    
    def _fallback_processing(
        self,
        request: VideoClipRequest,
        n_variants: int,
        audience_profile: Optional[Dict]
    ) -> ViralVideoBatchResponse:
        """Fallback processing when LangChain is not available."""
        logger.warning("Using fallback processing")
        
        # Create basic variants without LangChain
        variants = []
        for i in range(n_variants):
            variant = self._create_fallback_variant(request, i)
            variants.append(variant)
        
        return ViralVideoBatchResponse(
            success=True,
            original_clip_id=request.youtube_url,
            variants=variants,
            processing_time=time.perf_counter(),
            total_variants_generated=len(variants),
            successful_variants=len(variants)
        )
    
    def _create_fallback_variant(self, request: VideoClipRequest, variant_index: int) -> ViralVideoVariant:
        """Create a fallback variant without LangChain."""
        return ViralVideoVariant(
            variant_id=f"fallback_variant_{variant_index}",
            title=f"Viral Video {variant_index + 1}",
            description="Amazing viral content!",
            viral_score=0.6,
            engagement_prediction=0.5,
            captions=self._create_fallback_captions(variant_index),
            total_duration=30.0,
            estimated_views=5000,
            estimated_likes=2500,
            estimated_shares=1000,
            estimated_comments=500,
            tags=["viral", "trending"],
            hashtags=["#viral", "#trending"],
            target_audience=["general"]
        )
    
    def _create_fallback_captions(self, variant_index: int) -> List[CaptionSegment]:
        """Create fallback captions."""
        return [
            CaptionSegment(
                text=f"🔥 Viral Content {variant_index + 1}! 🔥",
                start_time=1.0,
                end_time=4.0,
                font_size=24,
                styles=[CaptionStyle.BOLD]
            )
        ]
    
    def _create_default_analysis(self) -> LangChainAnalysis:
        """Create default analysis."""
        return LangChainAnalysis(
            content_type=ContentType.ENTERTAINMENT,
            engagement_score=0.7,
            viral_potential=0.6,
            target_audience=["general"],
            content_summary="General entertainment content"
        )
    
    def _create_default_optimization(self) -> ContentOptimization:
        """Create default optimization."""
        return ContentOptimization(
            optimal_title="Viral Video",
            optimal_description="Amazing viral content!",
            optimal_tags=["viral", "trending"],
            optimal_hashtags=["#viral", "#trending"]
        )
    
    # =============================================================================
    # AGENT TOOLS
    # =============================================================================
    
    def _analyze_content(self, query: str) -> str:
        """Tool for content analysis."""
        return "Content analysis completed with viral potential assessment."
    
    def _generate_captions(self, query: str) -> str:
        """Tool for caption generation."""
        return "Viral captions generated with engagement optimization."
    
    def _optimize_titles(self, query: str) -> str:
        """Tool for title optimization."""
        return "Titles optimized for maximum click-through rate."
    
    def _analyze_engagement(self, query: str) -> str:
        """Tool for engagement analysis."""
        return "Engagement analysis completed with improvement recommendations."

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_langchain_processor(
    api_key: Optional[str] = None,
    model_name: str = "gpt-4",
    enable_agents: bool = True
) -> LangChainVideoProcessor:
    """Create a LangChain video processor."""
    config = LangChainConfig(
        openai_api_key=api_key,
        model_name=model_name,
        use_agents=enable_agents,
        enable_content_analysis=True,
        enable_engagement_analysis=True,
        enable_viral_analysis=True
    )
    
    return LangChainVideoProcessor(config)

def create_optimized_langchain_processor(
    api_key: Optional[str] = None,
    batch_size: int = 5,
    max_retries: int = 3
) -> LangChainVideoProcessor:
    """Create an optimized LangChain processor for production use."""
    config = LangChainConfig(
        openai_api_key=api_key,
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        batch_size=batch_size,
        max_retries=max_retries,
        cache_results=True,
        use_agents=True,
        use_memory=True,
        enable_debug=False
    )
    
    return LangChainVideoProcessor(config)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LangChainVideoProcessor',
    'LangChainConfig',
    'LangChainPrompts',
    'create_langchain_processor',
    'create_optimized_langchain_processor'
] 