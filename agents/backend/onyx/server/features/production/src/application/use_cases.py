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
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from src.domain.entities import (
from src.domain.interfaces import (
from src.core.exceptions import (
from typing import Any, List, Dict, Optional
"""
🎯 Application Use Cases
=======================

Production-ready business logic implementing clean architecture
with proper error handling, validation, and performance optimization.
"""


    User, ContentRequest, GeneratedContent, ContentTemplate, UsageMetrics,
    Status, ContentType, Language, Tone
)
    UserRepository, ContentRepository, TemplateRepository, MetricsRepository,
    AIService, EventPublisher, CacheService, RateLimiter
)
    BusinessException, ValidationException, NotFoundException,
    UnauthorizedException, InsufficientCreditsException, AIServiceUnavailableException
)


class GenerateContentUseCase:
    """Use case for generating content with AI"""
    
    def __init__(
        self,
        user_repo: UserRepository,
        content_repo: ContentRepository,
        ai_service: AIService,
        event_publisher: EventPublisher,
        cache_service: CacheService,
        rate_limiter: RateLimiter
    ):
        
    """__init__ function."""
self.user_repo = user_repo
        self.content_repo = content_repo
        self.ai_service = ai_service
        self.event_publisher = event_publisher
        self.cache_service = cache_service
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: UUID, request_data: Dict[str, Any]) -> GeneratedContent:
        """Execute content generation use case"""
        
        try:
            # 1. Validate and create content request
            request = await self._create_content_request(user_id, request_data)
            
            # 2. Check rate limits
            await self._check_rate_limits(user_id)
            
            # 3. Validate user and credits
            user = await self._validate_user_and_credits(user_id, request)
            
            # 4. Check cache for similar requests
            cached_content = await self._check_cache(request)
            if cached_content:
                return cached_content
            
            # 5. Create request record
            request = await self.content_repo.create_request(request)
            
            # 6. Generate content
            content = await self._generate_content(request)
            
            # 7. Save generated content
            content = await self.content_repo.create_content(content)
            
            # 8. Update user credits
            await self._update_user_credits(user, request)
            
            # 9. Publish events
            await self._publish_events(request, content)
            
            # 10. Cache result
            await self._cache_result(request, content)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}", exc_info=True)
            await self._handle_generation_error(request, str(e))
            raise
    
    async async def _create_content_request(self, user_id: UUID, request_data: Dict[str, Any]) -> ContentRequest:
        """Create and validate content request"""
        
        # Validate required fields
        required_fields = ['content_type', 'prompt']
        for field in required_fields:
            if field not in request_data:
                raise ValidationException(f"Missing required field: {field}")
        
        # Create request object
        request = ContentRequest(
            user_id=user_id,
            content_type=ContentType(request_data['content_type']),
            prompt=request_data['prompt'],
            title=request_data.get('title'),
            keywords=request_data.get('keywords', []),
            tone=Tone(request_data.get('tone', 'professional')),
            language=Language(request_data.get('language', 'en')),
            word_count=request_data.get('word_count'),
            target_audience=request_data.get('target_audience'),
            brand_voice=request_data.get('brand_voice'),
            call_to_action=request_data.get('call_to_action'),
            seo_optimized=request_data.get('seo_optimized', True),
            tags=request_data.get('tags', []),
            metadata=request_data.get('metadata', {})
        )
        
        return request
    
    async def _check_rate_limits(self, user_id: UUID) -> None:
        """Check rate limits for user"""
        
        rate_limit_key = f"rate_limit:{user_id}"
        limit = 100  # requests per hour
        window = 3600  # 1 hour
        
        if not await self.rate_limiter.is_allowed(rate_limit_key, limit, window):
            raise BusinessException(
                "Rate limit exceeded. Please try again later.",
                retry_after=3600
            )
        
        await self.rate_limiter.increment(rate_limit_key, window)
    
    async def _validate_user_and_credits(self, user_id: UUID, request: ContentRequest) -> User:
        """Validate user exists and has sufficient credits"""
        
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundException("User not found")
        
        if not user.is_active:
            raise UnauthorizedException("User account is inactive")
        
        required_credits = request.estimate_credits()
        if not user.has_sufficient_credits(required_credits):
            raise InsufficientCreditsException(required_credits, user.credits)
        
        return user
    
    async def _check_cache(self, request: ContentRequest) -> Optional[GeneratedContent]:
        """Check cache for similar requests"""
        
        cache_key = self._generate_cache_key(request)
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            self.logger.info(f"Cache hit for request: {request.id}")
            return GeneratedContent(**cached_data)
        
        return None
    
    async def _generate_content(self, request: ContentRequest) -> GeneratedContent:
        """Generate content using AI service"""
        
        start_time = datetime.utcnow()
        
        try:
            # Prepare prompt with context
            enhanced_prompt = await self._enhance_prompt(request)
            
            # Generate content
            raw_content = await self.ai_service.generate_content(
                prompt=enhanced_prompt,
                max_tokens=request.word_count * 2 if request.word_count else 1000,
                temperature=0.7,
                model="gpt-4"
            )
            
            # Analyze and optimize content
            analysis = await self.ai_service.analyze_content(raw_content)
            
            # SEO optimization if requested
            if request.seo_optimized and request.keywords:
                raw_content = await self.ai_service.optimize_seo(raw_content, request.keywords)
            
            # Calculate generation time
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create content object
            content = GeneratedContent(
                request_id=request.id,
                user_id=request.user_id,
                title=request.title or f"Generated {request.content_type.value}",
                content=raw_content,
                summary=analysis.get('summary'),
                word_count=len(raw_content.split()),
                reading_time=analysis.get('reading_time'),
                seo_score=analysis.get('seo_score'),
                readability_score=analysis.get('readability_score'),
                model_used="gpt-4",
                generation_time=generation_time,
                tokens_used=analysis.get('tokens_used'),
                confidence_score=analysis.get('confidence_score'),
                plagiarism_score=analysis.get('plagiarism_score'),
                status=Status.COMPLETED,
                metadata={
                    'original_prompt': request.prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'analysis': analysis
                }
            )
            
            return content
            
        except Exception as e:
            self.logger.error(f"AI generation failed: {e}")
            raise AIServiceUnavailableException("OpenAI", str(e))
    
    async def _enhance_prompt(self, request: ContentRequest) -> str:
        """Enhance prompt with context and instructions"""
        
        base_prompt = request.prompt
        
        # Add content type instructions
        type_instructions = {
            ContentType.BLOG_POST: "Write a comprehensive blog post with engaging introduction, detailed body, and strong conclusion.",
            ContentType.SOCIAL_MEDIA: "Create engaging social media content that encourages interaction and sharing.",
            ContentType.EMAIL: "Write a professional email that is clear, concise, and action-oriented.",
            ContentType.AD_COPY: "Create compelling ad copy that drives action and conversion.",
            ContentType.PRODUCT_DESCRIPTION: "Write a detailed product description that highlights benefits and features.",
            ContentType.LANDING_PAGE: "Create persuasive landing page content that converts visitors.",
            ContentType.VIDEO_SCRIPT: "Write a video script that is engaging and easy to follow.",
            ContentType.PODCAST_SCRIPT: "Create a podcast script that flows naturally and maintains listener interest.",
            ContentType.NEWS_RELEASE: "Write a professional news release following standard format.",
            ContentType.WHITEPAPER: "Create a comprehensive whitepaper with detailed analysis and insights."
        }
        
        enhanced_prompt = f"{type_instructions.get(request.content_type, '')}\n\n{base_prompt}"
        
        # Add tone instructions
        tone_instructions = {
            Tone.PROFESSIONAL: "Use a professional and authoritative tone.",
            Tone.CASUAL: "Use a casual and friendly tone.",
            Tone.FRIENDLY: "Use a warm and approachable tone.",
            Tone.AUTHORITATIVE: "Use a confident and expert tone.",
            Tone.HUMOROUS: "Include appropriate humor and wit.",
            Tone.INSPIRATIONAL: "Use motivational and uplifting language.",
            Tone.CONVERSATIONAL: "Write as if having a natural conversation.",
            Tone.FORMAL: "Use formal and structured language."
        }
        
        enhanced_prompt += f"\n\n{tone_instructions.get(request.tone, '')}"
        
        # Add audience context
        if request.target_audience:
            enhanced_prompt += f"\n\nTarget audience: {request.target_audience}"
        
        # Add brand voice
        if request.brand_voice:
            enhanced_prompt += f"\n\nBrand voice: {request.brand_voice}"
        
        # Add call to action
        if request.call_to_action:
            enhanced_prompt += f"\n\nInclude call to action: {request.call_to_action}"
        
        # Add word count requirement
        if request.word_count:
            enhanced_prompt += f"\n\nTarget word count: approximately {request.word_count} words"
        
        return enhanced_prompt
    
    async def _update_user_credits(self, user: User, request: ContentRequest) -> None:
        """Update user credits after successful generation"""
        
        credits_used = request.estimate_credits()
        user.deduct_credits(credits_used)
        await self.user_repo.update(user)
        
        self.logger.info(f"User {user.id} used {credits_used} credits")
    
    async def _publish_events(self, request: ContentRequest, content: GeneratedContent) -> None:
        """Publish events for monitoring and analytics"""
        
        events = [
            {
                "type": "content_generated",
                "data": {
                    "user_id": str(request.user_id),
                    "content_type": request.content_type.value,
                    "content_id": str(content.id),
                    "word_count": content.word_count,
                    "generation_time": content.generation_time
                }
            },
            {
                "type": "credits_consumed",
                "data": {
                    "user_id": str(request.user_id),
                    "credits_used": request.estimate_credits(),
                    "remaining_credits": 0  # Will be updated by user service
                }
            }
        ]
        
        for event in events:
            await self.event_publisher.publish(event["type"], event["data"])
    
    async def _cache_result(self, request: ContentRequest, content: GeneratedContent) -> None:
        """Cache the generated content for future similar requests"""
        
        cache_key = self._generate_cache_key(request)
        cache_data = content.dict()
        
        # Cache for 1 hour
        await self.cache_service.set(cache_key, cache_data, ttl=3600)
    
    def _generate_cache_key(self, request: ContentRequest) -> str:
        """Generate cache key for request"""
        
        key_parts = [
            "content",
            request.content_type.value,
            request.tone.value,
            request.language.value,
            str(hash(request.prompt))[:8]
        ]
        
        return ":".join(key_parts)
    
    async def _handle_generation_error(self, request: ContentRequest, error_message: str) -> None:
        """Handle generation errors"""
        
        # Update request status
        request.status = Status.FAILED
        await self.content_repo.update_request(request)
        
        # Publish error event
        await self.event_publisher.publish("content_generation_failed", {
            "user_id": str(request.user_id),
            "request_id": str(request.id),
            "error": error_message
        })


class GetUserContentUseCase:
    """Use case for retrieving user content"""
    
    def __init__(self, content_repo: ContentRepository, cache_service: CacheService):
        
    """__init__ function."""
self.content_repo = content_repo
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[GeneratedContent]:
        """Get user's generated content"""
        
        cache_key = f"user_content:{user_id}:{skip}:{limit}"
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            return [GeneratedContent(**item) for item in cached_data]
        
        content_list = await self.content_repo.list_user_content(user_id, skip, limit)
        
        # Cache for 5 minutes
        cache_data = [content.dict() for content in content_list]
        await self.cache_service.set(cache_key, cache_data, ttl=300)
        
        return content_list


class SearchContentUseCase:
    """Use case for searching content"""
    
    def __init__(self, content_repo: ContentRepository, cache_service: CacheService):
        
    """__init__ function."""
self.content_repo = content_repo
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)
    
    async def execute(
        self,
        user_id: UUID,
        query: str,
        content_type: Optional[ContentType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[GeneratedContent]:
        """Search user's content"""
        
        cache_key = f"search:{user_id}:{hash(query)}:{content_type}:{skip}:{limit}"
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            return [GeneratedContent(**item) for item in cached_data]
        
        results = await self.content_repo.search_content(
            user_id, query, content_type, skip, limit
        )
        
        # Cache for 10 minutes
        cache_data = [content.dict() for content in results]
        await self.cache_service.set(cache_key, cache_data, ttl=600)
        
        return results


class CreateTemplateUseCase:
    """Use case for creating content templates"""
    
    def __init__(self, template_repo: TemplateRepository, cache_service: CacheService):
        
    """__init__ function."""
self.template_repo = template_repo
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: UUID, template_data: Dict[str, Any]) -> ContentTemplate:
        """Create a new content template"""
        
        # Validate required fields
        required_fields = ['name', 'prompt_template', 'content_type']
        for field in required_fields:
            if field not in template_data:
                raise ValidationException(f"Missing required field: {field}")
        
        # Create template
        template = ContentTemplate(
            user_id=user_id,
            name=template_data['name'],
            description=template_data.get('description'),
            prompt_template=template_data['prompt_template'],
            content_type=ContentType(template_data['content_type']),
            default_tone=Tone(template_data.get('default_tone', 'professional')),
            default_language=Language(template_data.get('default_language', 'en')),
            parameters=template_data.get('parameters', []),
            default_values=template_data.get('default_values', {}),
            tags=template_data.get('tags', []),
            is_public=template_data.get('is_public', False)
        )
        
        # Save template
        template = await self.template_repo.create_template(template)
        
        # Clear user templates cache
        await self.cache_service.clear_pattern(f"user_templates:{user_id}:*")
        
        return template


class GetUserMetricsUseCase:
    """Use case for retrieving user metrics"""
    
    def __init__(self, metrics_repo: MetricsRepository, cache_service: CacheService):
        
    """__init__ function."""
self.metrics_repo = metrics_repo
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)
    
    async def execute(
        self,
        user_id: UUID,
        start_date: str,
        end_date: str
    ) -> List[UsageMetrics]:
        """Get user usage metrics for date range"""
        
        cache_key = f"user_metrics:{user_id}:{start_date}:{end_date}"
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            return [UsageMetrics(**item) for item in cached_data]
        
        metrics = await self.metrics_repo.get_user_metrics_range(user_id, start_date, end_date)
        
        # Cache for 1 hour
        cache_data = [metric.dict() for metric in metrics]
        await self.cache_service.set(cache_key, cache_data, ttl=3600)
        
        return metrics


class UpdateUserCreditsUseCase:
    """Use case for updating user credits"""
    
    def __init__(self, user_repo: UserRepository, event_publisher: EventPublisher):
        
    """__init__ function."""
self.user_repo = user_repo
        self.event_publisher = event_publisher
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: UUID, credits: int, reason: str) -> User:
        """Update user credits"""
        
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundException("User not found")
        
        if credits > 0:
            user.add_credits(credits)
        else:
            if not user.has_sufficient_credits(abs(credits)):
                raise InsufficientCreditsException(abs(credits), user.credits)
            user.deduct_credits(abs(credits))
        
        user = await self.user_repo.update(user)
        
        # Publish event
        await self.event_publisher.publish("credits_updated", {
            "user_id": str(user_id),
            "credits_change": credits,
            "new_balance": user.credits,
            "reason": reason
        })
        
        return user 