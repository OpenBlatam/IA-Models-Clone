"""
Advanced AI Service for Facebook Posts API
AI-powered content generation, optimization, and analysis
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..core.config import get_settings
from ..core.models import PostRequest, FacebookPost, ContentType, AudienceType, OptimizationLevel
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)


@dataclass
class AIGenerationResult:
    """Result of AI content generation"""
    content: str
    confidence_score: float
    processing_time: float
    model_used: str
    tokens_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentAnalysis:
    """Content analysis result"""
    sentiment_score: float  # -1 to 1
    engagement_score: float  # 0 to 1
    readability_score: float  # 0 to 1
    creativity_score: float  # 0 to 1
    relevance_score: float  # 0 to 1
    quality_score: float  # 0 to 1
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class OptimizationSuggestion:
    """Content optimization suggestion"""
    type: str  # "engagement", "readability", "sentiment", "creativity"
    priority: str  # "high", "medium", "low"
    suggestion: str
    expected_improvement: float
    implementation: str


class OpenAIService:
    """OpenAI API service for content generation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.settings = get_settings()
    
    async def generate_content(self, request: PostRequest) -> AIGenerationResult:
        """Generate content using OpenAI"""
        start_time = time.time()
        
        try:
            # Build prompt based on request
            prompt = self._build_prompt(request)
            
            # Generate content
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert social media content creator specializing in Facebook posts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.settings.ai_max_tokens,
                temperature=self.settings.ai_temperature,
                timeout=self.settings.ai_timeout
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            processing_time = time.time() - start_time
            
            return AIGenerationResult(
                content=content,
                confidence_score=0.9,  # OpenAI doesn't provide confidence scores
                processing_time=processing_time,
                model_used=self.model,
                tokens_used=tokens_used,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error("OpenAI content generation failed", error=str(e))
            raise
    
    def _build_prompt(self, request: PostRequest) -> str:
        """Build prompt for content generation"""
        prompt_parts = [
            f"Create a Facebook post about: {request.topic}",
            f"Target audience: {request.audience_type.value}",
            f"Content type: {request.content_type.value}",
            f"Tone: {request.tone}",
            f"Optimization level: {request.optimization_level.value}"
        ]
        
        if request.include_hashtags:
            prompt_parts.append("Include relevant hashtags")
        
        if request.tags:
            prompt_parts.append(f"Use these tags: {', '.join(request.tags)}")
        
        if request.length:
            prompt_parts.append(f"Target length: {request.length} characters")
        
        prompt_parts.extend([
            "Make it engaging and professional.",
            "Ensure it follows Facebook best practices.",
            "Include a clear call-to-action if appropriate."
        ])
        
        return "\n".join(prompt_parts)


class LocalAIService:
    """Local AI service using transformers"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package not available")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the local AI model"""
        if self._initialized:
            return
        
        try:
            logger.info("Loading local AI model", model=self.model_name)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._initialized = True
            logger.info("Local AI model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load local AI model", error=str(e))
            raise
    
    async def generate_content(self, request: PostRequest) -> AIGenerationResult:
        """Generate content using local AI model"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_prompt(request)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate content
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,  # Generate up to 200 more tokens
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated content
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            content = generated_text[len(prompt):].strip()
            
            processing_time = time.time() - start_time
            tokens_used = outputs.shape[1]
            
            return AIGenerationResult(
                content=content,
                confidence_score=0.7,  # Local models have lower confidence
                processing_time=processing_time,
                model_used=self.model_name,
                tokens_used=tokens_used,
                metadata={
                    "input_length": inputs.shape[1],
                    "output_length": outputs.shape[1],
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            )
            
        except Exception as e:
            logger.error("Local AI content generation failed", error=str(e))
            raise
    
    def _build_prompt(self, request: PostRequest) -> str:
        """Build prompt for local AI model"""
        return f"Topic: {request.topic}\nAudience: {request.audience_type.value}\nType: {request.content_type.value}\nTone: {request.tone}\nPost:"


class ContentAnalyzer:
    """Content analysis service"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize content analysis models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using mock analysis")
            return
        
        try:
            # Load sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            self._initialized = True
            logger.info("Content analyzer initialized successfully")
            
        except Exception as e:
            logger.warning("Failed to initialize content analyzer", error=str(e))
    
    @timed("content_analysis")
    async def analyze_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> ContentAnalysis:
        """Analyze content for various metrics"""
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Mock analysis if models not available
            if not self._initialized or not self.sentiment_analyzer:
                return self._mock_analysis(content, context)
            
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer(content)
            sentiment_score = self._calculate_sentiment_score(sentiment_result)
            
            # Calculate other scores
            engagement_score = self._calculate_engagement_score(content)
            readability_score = self._calculate_readability_score(content)
            creativity_score = self._calculate_creativity_score(content)
            relevance_score = self._calculate_relevance_score(content, context)
            quality_score = self._calculate_quality_score(content)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                content, sentiment_score, engagement_score, readability_score
            )
            
            processing_time = time.time() - start_time
            
            return ContentAnalysis(
                sentiment_score=sentiment_score,
                engagement_score=engagement_score,
                readability_score=readability_score,
                creativity_score=creativity_score,
                relevance_score=relevance_score,
                quality_score=quality_score,
                recommendations=recommendations,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error("Content analysis failed", error=str(e))
            return self._mock_analysis(content, context)
    
    def _calculate_sentiment_score(self, sentiment_result: List[Dict]) -> float:
        """Calculate sentiment score from analysis result"""
        if not sentiment_result:
            return 0.0
        
        # Convert sentiment labels to scores
        label_scores = {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}  # Negative, Neutral, Positive
        
        total_score = 0.0
        total_confidence = 0.0
        
        for result in sentiment_result[0]:
            label = result["label"]
            confidence = result["score"]
            score = label_scores.get(label, 0.0)
            
            total_score += score * confidence
            total_confidence += confidence
        
        return total_score / total_confidence if total_confidence > 0 else 0.0
    
    def _calculate_engagement_score(self, content: str) -> float:
        """Calculate engagement score based on content features"""
        score = 0.0
        
        # Length factor (optimal length around 100-200 characters)
        length = len(content)
        if 100 <= length <= 200:
            score += 0.3
        elif 50 <= length <= 300:
            score += 0.2
        
        # Question marks (encourage interaction)
        if "?" in content:
            score += 0.2
        
        # Exclamation marks (show enthusiasm)
        if "!" in content:
            score += 0.1
        
        # Hashtags (increase discoverability)
        hashtag_count = content.count("#")
        score += min(hashtag_count * 0.1, 0.3)
        
        # Mentions (encourage engagement)
        mention_count = content.count("@")
        score += min(mention_count * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score"""
        words = content.split()
        sentences = content.split(".")
        
        if len(words) == 0 or len(sentences) == 0:
            return 0.0
        
        # Simple readability calculation
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Optimal ranges
        if 10 <= avg_words_per_sentence <= 20 and 4 <= avg_chars_per_word <= 6:
            return 1.0
        elif 8 <= avg_words_per_sentence <= 25 and 3 <= avg_chars_per_word <= 8:
            return 0.8
        else:
            return 0.6
    
    def _calculate_creativity_score(self, content: str) -> float:
        """Calculate creativity score"""
        score = 0.0
        
        # Unique words ratio
        words = content.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness_ratio = len(unique_words) / len(words)
            score += uniqueness_ratio * 0.4
        
        # Emojis (show creativity)
        emoji_count = sum(1 for char in content if ord(char) > 127)
        score += min(emoji_count * 0.1, 0.3)
        
        # Creative punctuation
        if "..." in content or "—" in content:
            score += 0.1
        
        # Alliteration (simple check)
        words = content.lower().split()
        if len(words) >= 3:
            first_letters = [word[0] for word in words if word]
            if len(set(first_letters)) < len(first_letters) * 0.7:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_relevance_score(self, content: str, context: Optional[Dict[str, Any]]) -> float:
        """Calculate relevance score"""
        if not context:
            return 0.8  # Default score
        
        score = 0.8
        
        # Check topic relevance
        topic = context.get("topic", "")
        if topic and topic.lower() in content.lower():
            score += 0.1
        
        # Check audience relevance
        audience = context.get("audience_type", "")
        if audience:
            audience_keywords = {
                "professionals": ["business", "professional", "career", "industry"],
                "general": ["everyone", "people", "community"],
                "students": ["learn", "study", "education", "student"]
            }
            
            keywords = audience_keywords.get(audience.lower(), [])
            for keyword in keywords:
                if keyword in content.lower():
                    score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate overall quality score"""
        # Combine other scores with weights
        engagement = self._calculate_engagement_score(content)
        readability = self._calculate_readability_score(content)
        creativity = self._calculate_creativity_score(content)
        
        return (engagement * 0.4 + readability * 0.4 + creativity * 0.2)
    
    def _generate_recommendations(self, content: str, sentiment: float, engagement: float, readability: float) -> List[str]:
        """Generate content improvement recommendations"""
        recommendations = []
        
        if sentiment < -0.3:
            recommendations.append("Consider making the tone more positive to improve engagement")
        
        if engagement < 0.5:
            recommendations.append("Add a question or call-to-action to increase engagement")
        
        if readability < 0.6:
            recommendations.append("Simplify sentence structure for better readability")
        
        if len(content) < 50:
            recommendations.append("Consider adding more detail to make the post more informative")
        elif len(content) > 300:
            recommendations.append("Consider shortening the post for better engagement")
        
        if not any(char in content for char in ["?", "!"]):
            recommendations.append("Add punctuation to make the post more engaging")
        
        return recommendations
    
    def _mock_analysis(self, content: str, context: Optional[Dict[str, Any]]) -> ContentAnalysis:
        """Mock analysis when AI models are not available"""
        return ContentAnalysis(
            sentiment_score=0.2,
            engagement_score=0.7,
            readability_score=0.8,
            creativity_score=0.6,
            relevance_score=0.8,
            quality_score=0.7,
            recommendations=["Consider adding more engaging content"],
            processing_time=0.1
        )


class ContentOptimizer:
    """Content optimization service"""
    
    def __init__(self, ai_service: Union[OpenAIService, LocalAIService]):
        self.ai_service = ai_service
        self.analyzer = ContentAnalyzer()
    
    async def optimize_content(self, content: str, target_audience: AudienceType, optimization_goals: List[str]) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions for content"""
        try:
            # Analyze current content
            analysis = await self.analyzer.analyze_content(content)
            
            suggestions = []
            
            # Generate suggestions based on goals
            for goal in optimization_goals:
                if goal == "engagement" and analysis.engagement_score < 0.7:
                    suggestions.append(OptimizationSuggestion(
                        type="engagement",
                        priority="high",
                        suggestion="Add a question or call-to-action to increase engagement",
                        expected_improvement=0.2,
                        implementation="Add 'What do you think?' or 'Share your experience'"
                    ))
                
                elif goal == "readability" and analysis.readability_score < 0.7:
                    suggestions.append(OptimizationSuggestion(
                        type="readability",
                        priority="medium",
                        suggestion="Simplify sentence structure and use shorter sentences",
                        expected_improvement=0.15,
                        implementation="Break long sentences into shorter ones"
                    ))
                
                elif goal == "sentiment" and analysis.sentiment_score < 0.3:
                    suggestions.append(OptimizationSuggestion(
                        type="sentiment",
                        priority="high",
                        suggestion="Make the tone more positive and encouraging",
                        expected_improvement=0.3,
                        implementation="Replace negative words with positive alternatives"
                    ))
                
                elif goal == "creativity" and analysis.creativity_score < 0.6:
                    suggestions.append(OptimizationSuggestion(
                        type="creativity",
                        priority="low",
                        suggestion="Add creative elements like emojis or wordplay",
                        expected_improvement=0.1,
                        implementation="Add relevant emojis or use creative language"
                    ))
            
            return suggestions
            
        except Exception as e:
            logger.error("Content optimization failed", error=str(e))
            return []


class AIService:
    """Main AI service orchestrator"""
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_service = None
        self.local_ai_service = None
        self.analyzer = ContentAnalyzer()
        self.optimizer = None
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        
        # Initialize services based on configuration
        if self.settings.ai_api_key and OPENAI_AVAILABLE:
            self.openai_service = OpenAIService(self.settings.ai_api_key, self.settings.ai_model)
            self.optimizer = ContentOptimizer(self.openai_service)
        elif TRANSFORMERS_AVAILABLE:
            self.local_ai_service = LocalAIService()
            self.optimizer = ContentOptimizer(self.local_ai_service)
    
    async def initialize(self):
        """Initialize AI services"""
        try:
            if self.local_ai_service:
                await self.local_ai_service.initialize()
            
            await self.analyzer.initialize()
            
            logger.info("AI service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize AI service", error=str(e))
            raise
    
    @timed("ai_content_generation")
    async def generate_content(self, request: PostRequest) -> AIGenerationResult:
        """Generate content using available AI service"""
        try:
            # Check cache first
            cache_key = f"ai_generation:{hash(str(request.dict()))}"
            cached_result = await self.cache_manager.cache.get(cache_key)
            
            if cached_result:
                logger.debug("AI generation cache hit", cache_key=cache_key)
                return AIGenerationResult(**cached_result)
            
            # Generate content
            if self.openai_service:
                result = await self.openai_service.generate_content(request)
            elif self.local_ai_service:
                result = await self.local_ai_service.generate_content(request)
            else:
                # Fallback to mock generation
                result = self._mock_generation(request)
            
            # Cache result
            await self.cache_manager.cache.set(cache_key, result.__dict__, ttl=3600)
            
            # Record metrics
            self.monitor.record_api_request("POST", "/ai/generate", 200, result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error("AI content generation failed", error=str(e))
            self.monitor.record_api_request("POST", "/ai/generate", 500, 0.0)
            raise
    
    @timed("ai_content_analysis")
    async def analyze_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> ContentAnalysis:
        """Analyze content using AI"""
        try:
            # Check cache first
            cache_key = f"ai_analysis:{hash(content)}"
            cached_result = await self.cache_manager.cache.get(cache_key)
            
            if cached_result:
                logger.debug("AI analysis cache hit", cache_key=cache_key)
                return ContentAnalysis(**cached_result)
            
            # Analyze content
            result = await self.analyzer.analyze_content(content, context)
            
            # Cache result
            await self.cache_manager.cache.set(cache_key, result.__dict__, ttl=1800)
            
            return result
            
        except Exception as e:
            logger.error("AI content analysis failed", error=str(e))
            raise
    
    @timed("ai_content_optimization")
    async def optimize_content(self, content: str, target_audience: AudienceType, optimization_goals: List[str]) -> List[OptimizationSuggestion]:
        """Optimize content using AI"""
        try:
            if not self.optimizer:
                return []
            
            # Check cache first
            cache_key = f"ai_optimization:{hash(content)}:{target_audience.value}:{':'.join(optimization_goals)}"
            cached_result = await self.cache_manager.cache.get(cache_key)
            
            if cached_result:
                logger.debug("AI optimization cache hit", cache_key=cache_key)
                return [OptimizationSuggestion(**s) for s in cached_result]
            
            # Optimize content
            result = await self.optimizer.optimize_content(content, target_audience, optimization_goals)
            
            # Cache result
            await self.cache_manager.cache.set(cache_key, [s.__dict__ for s in result], ttl=1800)
            
            return result
            
        except Exception as e:
            logger.error("AI content optimization failed", error=str(e))
            return []
    
    def _mock_generation(self, request: PostRequest) -> AIGenerationResult:
        """Mock content generation when AI services are not available"""
        content = f"🚀 Exciting content about: {request.topic}\n\n"
        content += f"This post is tailored for {request.audience_type.value} audience "
        content += f"with {request.content_type.value} content type.\n\n"
        content += f"Tone: {request.tone}\n"
        
        if request.include_hashtags:
            content += f"\n#Innovation #Tech #Future"
        
        return AIGenerationResult(
            content=content,
            confidence_score=0.5,
            processing_time=0.1,
            model_used="mock",
            tokens_used=50,
            metadata={"mock": True}
        )


# Global AI service instance
_ai_service: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Get global AI service instance"""
    global _ai_service
    
    if _ai_service is None:
        _ai_service = AIService()
    
    return _ai_service


async def initialize_ai_service():
    """Initialize global AI service"""
    ai_service = get_ai_service()
    await ai_service.initialize()


# Export all classes and functions
__all__ = [
    # Data classes
    'AIGenerationResult',
    'ContentAnalysis',
    'OptimizationSuggestion',
    
    # AI services
    'OpenAIService',
    'LocalAIService',
    'ContentAnalyzer',
    'ContentOptimizer',
    'AIService',
    
    # Utility functions
    'get_ai_service',
    'initialize_ai_service',
]






























