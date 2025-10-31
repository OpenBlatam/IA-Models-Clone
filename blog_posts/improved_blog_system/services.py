"""
Advanced Blog Posts Services
===========================

Comprehensive business logic and services for blog posts system.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from .schemas import (
    BlogPost, BlogPostRequest, BlogPostResponse, BlogPostListResponse,
    ContentAnalysisRequest, ContentAnalysisResponse, ContentGenerationRequest,
    ContentGenerationResponse, SEOOptimizationRequest, SEOOptimizationResponse,
    BlogPostSearchRequest, BlogPostAnalytics, BlogPostPerformance,
    BlogPostTemplate, BlogPostWorkflow, BlogPostCollaboration,
    BlogPostComment, BlogPostCategory, BlogPostTag, BlogPostAuthor,
    BlogPostSettings, BlogPostSystemStatus, BlogPostSystemConfig
)
from .exceptions import (
    PostNotFoundError, PostAlreadyExistsError, PostValidationError,
    PostPermissionDeniedError, PostContentError, PostSEOError,
    PostAnalyticsError, PostCollaborationError, PostWorkflowError,
    PostTemplateError, PostCategoryError, PostTagError, PostAuthorError,
    PostCommentError, PostMediaError, PostPublishingError, PostSchedulingError,
    PostArchivingError, PostDeletionError, PostSystemError,
    create_blog_error, log_blog_error, handle_blog_error
)
from .config import get_settings

logger = logging.getLogger(__name__)


class BlogPostService:
    """Main blog post service"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
    
    async def create_post(self, request: BlogPostRequest, author_id: str) -> BlogPostResponse:
        """Create a new blog post"""
        try:
            # Generate slug from title
            slug = self._generate_slug(request.title)
            
            # Check if slug already exists
            existing_post = await self._get_post_by_slug(slug)
            if existing_post:
                raise PostAlreadyExistsError(slug, f"Post with slug '{slug}' already exists")
            
            # Create blog post
            post = BlogPost(
                title=request.title,
                slug=slug,
                content=request.content,
                excerpt=request.excerpt,
                author_id=author_id,
                status=request.status,
                content_type=request.content_type,
                content_format=request.content_format,
                meta_title=request.meta_title,
                meta_description=request.meta_description,
                meta_keywords=request.meta_keywords,
                featured_image=request.featured_image,
                images=request.images,
                videos=request.videos,
                categories=request.categories,
                tags=request.tags,
                scheduled_at=request.scheduled_at,
                metadata=request.metadata
            )
            
            # Calculate content metrics
            await self._calculate_content_metrics(post)
            
            # Save to database
            self.db.add(post)
            await self.db.commit()
            await self.db.refresh(post)
            
            # Cache the post
            await self._cache_post(post)
            
            # Log creation
            logger.info(f"Blog post created: {post.post_id} by {author_id}")
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post created successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            await self.db.rollback()
            error = handle_blog_error(e, author_id=author_id)
            log_blog_error(error)
            raise error
    
    async def get_post(self, post_id: str) -> BlogPostResponse:
        """Get blog post by ID"""
        try:
            # Try cache first
            cached_post = await self._get_cached_post(post_id)
            if cached_post:
                return BlogPostResponse(
                    success=True,
                    data=cached_post,
                    message="Blog post retrieved from cache",
                    processing_time=0.0
                )
            
            # Get from database
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Cache the post
            await self._cache_post(post)
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post retrieved successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            error = handle_blog_error(e, post_id=post_id)
            log_blog_error(error)
            raise error
    
    async def update_post(self, post_id: str, request: BlogPostRequest, user_id: str) -> BlogPostResponse:
        """Update blog post"""
        try:
            # Get existing post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Check permissions
            if post.author_id != user_id:
                raise PostPermissionDeniedError(
                    post_id, user_id, "edit",
                    "You don't have permission to edit this post"
                )
            
            # Update fields
            post.title = request.title
            post.slug = self._generate_slug(request.title)
            post.content = request.content
            post.excerpt = request.excerpt
            post.content_type = request.content_type
            post.content_format = request.content_format
            post.meta_title = request.meta_title
            post.meta_description = request.meta_description
            post.meta_keywords = request.meta_keywords
            post.featured_image = request.featured_image
            post.images = request.images
            post.videos = request.videos
            post.categories = request.categories
            post.tags = request.tags
            post.scheduled_at = request.scheduled_at
            post.metadata = request.metadata
            post.updated_at = datetime.utcnow()
            
            # Recalculate content metrics
            await self._calculate_content_metrics(post)
            
            # Save changes
            await self.db.commit()
            await self.db.refresh(post)
            
            # Update cache
            await self._cache_post(post)
            
            # Log update
            logger.info(f"Blog post updated: {post_id} by {user_id}")
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post updated successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            await self.db.rollback()
            error = handle_blog_error(e, post_id=post_id, user_id=user_id)
            log_blog_error(error)
            raise error
    
    async def delete_post(self, post_id: str, user_id: str) -> BlogPostResponse:
        """Delete blog post"""
        try:
            # Get existing post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Check permissions
            if post.author_id != user_id:
                raise PostPermissionDeniedError(
                    post_id, user_id, "delete",
                    "You don't have permission to delete this post"
                )
            
            # Soft delete - mark as deleted
            post.status = "deleted"
            post.updated_at = datetime.utcnow()
            
            # Save changes
            await self.db.commit()
            
            # Remove from cache
            await self._remove_cached_post(post_id)
            
            # Log deletion
            logger.info(f"Blog post deleted: {post_id} by {user_id}")
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post deleted successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            await self.db.rollback()
            error = handle_blog_error(e, post_id=post_id, user_id=user_id)
            log_blog_error(error)
            raise error
    
    async def publish_post(self, post_id: str, user_id: str) -> BlogPostResponse:
        """Publish blog post"""
        try:
            # Get existing post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Check permissions
            if post.author_id != user_id:
                raise PostPermissionDeniedError(
                    post_id, user_id, "publish",
                    "You don't have permission to publish this post"
                )
            
            # Check if post is ready for publishing
            if not self._is_post_ready_for_publishing(post):
                raise PostValidationError(
                    post_id, "content",
                    "Post is not ready for publishing. Check required fields."
                )
            
            # Update status and publish time
            post.status = "published"
            post.published_at = datetime.utcnow()
            post.updated_at = datetime.utcnow()
            
            # Save changes
            await self.db.commit()
            await self.db.refresh(post)
            
            # Update cache
            await self._cache_post(post)
            
            # Trigger publishing workflow
            await self._trigger_publishing_workflow(post)
            
            # Log publishing
            logger.info(f"Blog post published: {post_id} by {user_id}")
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post published successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            await self.db.rollback()
            error = handle_blog_error(e, post_id=post_id, user_id=user_id)
            log_blog_error(error)
            raise error
    
    async def schedule_post(self, post_id: str, scheduled_at: datetime, user_id: str) -> BlogPostResponse:
        """Schedule blog post for future publishing"""
        try:
            # Get existing post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Check permissions
            if post.author_id != user_id:
                raise PostPermissionDeniedError(
                    post_id, user_id, "schedule",
                    "You don't have permission to schedule this post"
                )
            
            # Validate scheduled time
            if scheduled_at <= datetime.utcnow():
                raise PostValidationError(
                    post_id, "scheduled_at",
                    "Scheduled time must be in the future"
                )
            
            # Update status and scheduled time
            post.status = "scheduled"
            post.scheduled_at = scheduled_at
            post.updated_at = datetime.utcnow()
            
            # Save changes
            await self.db.commit()
            await self.db.refresh(post)
            
            # Update cache
            await self._cache_post(post)
            
            # Schedule publishing task
            await self._schedule_publishing_task(post)
            
            # Log scheduling
            logger.info(f"Blog post scheduled: {post_id} for {scheduled_at} by {user_id}")
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post scheduled successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            await self.db.rollback()
            error = handle_blog_error(e, post_id=post_id, user_id=user_id)
            log_blog_error(error)
            raise error
    
    async def archive_post(self, post_id: str, user_id: str) -> BlogPostResponse:
        """Archive blog post"""
        try:
            # Get existing post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Check permissions
            if post.author_id != user_id:
                raise PostPermissionDeniedError(
                    post_id, user_id, "archive",
                    "You don't have permission to archive this post"
                )
            
            # Update status
            post.status = "archived"
            post.updated_at = datetime.utcnow()
            
            # Save changes
            await self.db.commit()
            await self.db.refresh(post)
            
            # Update cache
            await self._cache_post(post)
            
            # Log archiving
            logger.info(f"Blog post archived: {post_id} by {user_id}")
            
            return BlogPostResponse(
                success=True,
                data=post,
                message="Blog post archived successfully",
                processing_time=0.0
            )
            
        except Exception as e:
            await self.db.rollback()
            error = handle_blog_error(e, post_id=post_id, user_id=user_id)
            log_blog_error(error)
            raise error
    
    async def search_posts(self, request: BlogPostSearchRequest) -> BlogPostListResponse:
        """Search blog posts with filters"""
        try:
            # Build query
            query = select(BlogPost)
            
            # Apply filters
            if request.query:
                query = query.where(
                    or_(
                        BlogPost.title.ilike(f"%{request.query}%"),
                        BlogPost.content.ilike(f"%{request.query}%"),
                        BlogPost.excerpt.ilike(f"%{request.query}%")
                    )
                )
            
            if request.categories:
                query = query.where(BlogPost.categories.any(func.any(request.categories)))
            
            if request.tags:
                query = query.where(BlogPost.tags.any(func.any(request.tags)))
            
            if request.content_type:
                query = query.where(BlogPost.content_type == request.content_type)
            
            if request.status:
                query = query.where(BlogPost.status == request.status)
            
            if request.author_id:
                query = query.where(BlogPost.author_id == request.author_id)
            
            if request.date_from:
                query = query.where(BlogPost.created_at >= request.date_from)
            
            if request.date_to:
                query = query.where(BlogPost.created_at <= request.date_to)
            
            # Apply sorting
            if request.sort_by == "created_at":
                if request.sort_order == "desc":
                    query = query.order_by(BlogPost.created_at.desc())
                else:
                    query = query.order_by(BlogPost.created_at.asc())
            elif request.sort_by == "updated_at":
                if request.sort_order == "desc":
                    query = query.order_by(BlogPost.updated_at.desc())
                else:
                    query = query.order_by(BlogPost.updated_at.asc())
            elif request.sort_by == "views":
                if request.sort_order == "desc":
                    query = query.order_by(BlogPost.views.desc())
                else:
                    query = query.order_by(BlogPost.views.asc())
            elif request.sort_by == "title":
                if request.sort_order == "desc":
                    query = query.order_by(BlogPost.title.desc())
                else:
                    query = query.order_by(BlogPost.title.asc())
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await self.db.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination
            offset = (request.page - 1) * request.per_page
            query = query.offset(offset).limit(request.per_page)
            
            # Execute query
            result = await self.db.execute(query)
            posts = result.scalars().all()
            
            # Calculate pagination info
            total_pages = (total + request.per_page - 1) // request.per_page
            
            return BlogPostListResponse(
                success=True,
                data=list(posts),
                total=total,
                page=request.page,
                per_page=request.per_page,
                total_pages=total_pages,
                message=f"Found {total} blog posts",
                processing_time=0.0
            )
            
        except Exception as e:
            error = handle_blog_error(e)
            log_blog_error(error)
            raise error
    
    async def get_post_analytics(self, post_id: str, time_period: str = "30d") -> BlogPostAnalytics:
        """Get blog post analytics"""
        try:
            # Get post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            elif time_period == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Get analytics data (simplified - would integrate with analytics service)
            analytics = BlogPostAnalytics(
                post_id=post_id,
                time_period=time_period,
                views=post.views,
                unique_views=post.views,  # Would be calculated separately
                likes=post.likes,
                shares=post.shares,
                comments=post.comments,
                engagement_rate=self._calculate_engagement_rate(post),
                bounce_rate=0.0,  # Would be calculated from analytics
                avg_reading_time=post.reading_time,
                traffic_sources={},  # Would be populated from analytics
                top_keywords=post.meta_keywords,
                social_shares={},  # Would be populated from social media APIs
                generated_at=datetime.utcnow()
            )
            
            return analytics
            
        except Exception as e:
            error = handle_blog_error(e, post_id=post_id)
            log_blog_error(error)
            raise error
    
    async def get_post_performance(self, post_id: str) -> BlogPostPerformance:
        """Get blog post performance metrics"""
        try:
            # Get post
            post = await self._get_post_by_id(post_id)
            if not post:
                raise PostNotFoundError(post_id, f"Blog post {post_id} not found")
            
            # Calculate performance metrics
            performance = BlogPostPerformance(
                post_id=post_id,
                performance_score=self._calculate_performance_score(post),
                seo_performance=post.seo_score,
                engagement_performance=post.engagement_score,
                content_quality=post.quality_score,
                viral_potential=self._calculate_viral_potential(post),
                recommendations=self._generate_performance_recommendations(post),
                benchmark_comparison=self._get_benchmark_comparison(post),
                generated_at=datetime.utcnow()
            )
            
            return performance
            
        except Exception as e:
            error = handle_blog_error(e, post_id=post_id)
            log_blog_error(error)
            raise error
    
    # Helper methods
    def _generate_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title"""
        import re
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    async def _get_post_by_id(self, post_id: str) -> Optional[BlogPost]:
        """Get post by ID from database"""
        result = await self.db.execute(
            select(BlogPost).where(BlogPost.post_id == post_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_post_by_slug(self, slug: str) -> Optional[BlogPost]:
        """Get post by slug from database"""
        result = await self.db.execute(
            select(BlogPost).where(BlogPost.slug == slug)
        )
        return result.scalar_one_or_none()
    
    async def _cache_post(self, post: BlogPost) -> None:
        """Cache blog post in Redis"""
        try:
            cache_key = f"blog_post:{post.post_id}"
            post_data = post.dict()
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(post_data, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache post {post.post_id}: {e}")
    
    async def _get_cached_post(self, post_id: str) -> Optional[BlogPost]:
        """Get cached blog post from Redis"""
        try:
            cache_key = f"blog_post:{post_id}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                post_dict = json.loads(cached_data)
                return BlogPost(**post_dict)
        except Exception as e:
            logger.warning(f"Failed to get cached post {post_id}: {e}")
        return None
    
    async def _remove_cached_post(self, post_id: str) -> None:
        """Remove cached blog post from Redis"""
        try:
            cache_key = f"blog_post:{post_id}"
            await self.redis.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to remove cached post {post_id}: {e}")
    
    async def _calculate_content_metrics(self, post: BlogPost) -> None:
        """Calculate content metrics for blog post"""
        try:
            # Word count
            post.word_count = len(post.content.split())
            
            # Reading time (average 200 words per minute)
            post.reading_time = max(1, post.word_count // 200)
            
            # Basic SEO score calculation
            post.seo_score = self._calculate_basic_seo_score(post)
            
            # Basic readability score
            post.readability_score = self._calculate_basic_readability_score(post)
            
            # Basic engagement score
            post.engagement_score = self._calculate_basic_engagement_score(post)
            
            # Basic quality score
            post.quality_score = (
                post.seo_score + post.readability_score + post.engagement_score
            ) / 3
            
        except Exception as e:
            logger.warning(f"Failed to calculate content metrics for post {post.post_id}: {e}")
    
    def _calculate_basic_seo_score(self, post: BlogPost) -> float:
        """Calculate basic SEO score"""
        score = 0.0
        
        # Title length (optimal: 50-60 characters)
        if post.meta_title:
            title_len = len(post.meta_title)
            if 50 <= title_len <= 60:
                score += 0.3
            elif 40 <= title_len <= 70:
                score += 0.2
        
        # Meta description length (optimal: 150-160 characters)
        if post.meta_description:
            desc_len = len(post.meta_description)
            if 150 <= desc_len <= 160:
                score += 0.3
            elif 120 <= desc_len <= 180:
                score += 0.2
        
        # Content length (optimal: 300-2000 words)
        if 300 <= post.word_count <= 2000:
            score += 0.2
        elif 200 <= post.word_count <= 3000:
            score += 0.1
        
        # Keywords presence
        if post.meta_keywords:
            score += 0.1
        
        # Featured image
        if post.featured_image:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_basic_readability_score(self, post: BlogPost) -> float:
        """Calculate basic readability score"""
        try:
            # Simple readability calculation
            sentences = post.content.split('.')
            words = post.content.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.5
            
            avg_sentence_length = len(words) / len(sentences)
            
            # Optimal sentence length: 15-20 words
            if 15 <= avg_sentence_length <= 20:
                return 0.8
            elif 10 <= avg_sentence_length <= 25:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    def _calculate_basic_engagement_score(self, post: BlogPost) -> float:
        """Calculate basic engagement score"""
        score = 0.0
        
        # Check for engagement elements
        content_lower = post.content.lower()
        
        # Questions
        if '?' in post.content:
            score += 0.2
        
        # Call-to-action words
        cta_words = ['click', 'subscribe', 'follow', 'share', 'comment', 'like']
        cta_count = sum(1 for word in cta_words if word in content_lower)
        score += min(0.3, cta_count * 0.1)
        
        # Emotional words
        emotional_words = ['amazing', 'incredible', 'fantastic', 'wonderful', 'terrible', 'awful']
        emotional_count = sum(1 for word in emotional_words if word in content_lower)
        score += min(0.3, emotional_count * 0.1)
        
        # Storytelling elements
        story_words = ['story', 'experience', 'journey', 'adventure', 'tale']
        story_count = sum(1 for word in story_words if word in content_lower)
        score += min(0.2, story_count * 0.1)
        
        return min(1.0, score)
    
    def _is_post_ready_for_publishing(self, post: BlogPost) -> bool:
        """Check if post is ready for publishing"""
        # Check required fields
        if not post.title or not post.content:
            return False
        
        # Check content length
        if post.word_count < 100:
            return False
        
        # Check SEO elements
        if not post.meta_title or not post.meta_description:
            return False
        
        return True
    
    def _calculate_engagement_rate(self, post: BlogPost) -> float:
        """Calculate engagement rate"""
        if post.views == 0:
            return 0.0
        
        total_engagement = post.likes + post.shares + post.comments
        return total_engagement / post.views
    
    def _calculate_performance_score(self, post: BlogPost) -> float:
        """Calculate overall performance score"""
        # Weighted average of different metrics
        weights = {
            'seo': 0.3,
            'engagement': 0.3,
            'quality': 0.2,
            'views': 0.2
        }
        
        # Normalize views (assuming max 10000 views)
        normalized_views = min(1.0, post.views / 10000)
        
        score = (
            weights['seo'] * post.seo_score +
            weights['engagement'] * post.engagement_score +
            weights['quality'] * post.quality_score +
            weights['views'] * normalized_views
        )
        
        return min(100.0, score * 100)
    
    def _calculate_viral_potential(self, post: BlogPost) -> float:
        """Calculate viral potential"""
        # Based on engagement rate and content quality
        engagement_rate = self._calculate_engagement_rate(post)
        viral_potential = (engagement_rate * 0.6 + post.quality_score * 0.4) * 100
        
        return min(100.0, viral_potential)
    
    def _generate_performance_recommendations(self, post: BlogPost) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if post.seo_score < 0.6:
            recommendations.append("Improve SEO by optimizing meta tags and content structure")
        
        if post.engagement_score < 0.6:
            recommendations.append("Increase engagement by adding questions and call-to-actions")
        
        if post.quality_score < 0.6:
            recommendations.append("Improve content quality by enhancing readability and structure")
        
        if post.views < 100:
            recommendations.append("Increase visibility through better promotion and SEO")
        
        return recommendations
    
    def _get_benchmark_comparison(self, post: BlogPost) -> Dict[str, Any]:
        """Get benchmark comparison data"""
        # This would typically compare against similar posts or industry benchmarks
        return {
            "industry_average": {
                "seo_score": 0.65,
                "engagement_score": 0.45,
                "quality_score": 0.70
            },
            "similar_posts_average": {
                "seo_score": 0.60,
                "engagement_score": 0.50,
                "quality_score": 0.65
            },
            "performance_vs_benchmark": {
                "seo_score": post.seo_score - 0.65,
                "engagement_score": post.engagement_score - 0.45,
                "quality_score": post.quality_score - 0.70
            }
        }
    
    async def _trigger_publishing_workflow(self, post: BlogPost) -> None:
        """Trigger publishing workflow"""
        try:
            # This would integrate with workflow engine
            logger.info(f"Triggering publishing workflow for post {post.post_id}")
        except Exception as e:
            logger.warning(f"Failed to trigger publishing workflow for post {post.post_id}: {e}")
    
    async def _schedule_publishing_task(self, post: BlogPost) -> None:
        """Schedule publishing task"""
        try:
            # This would integrate with task scheduler
            logger.info(f"Scheduling publishing task for post {post.post_id} at {post.scheduled_at}")
        except Exception as e:
            logger.warning(f"Failed to schedule publishing task for post {post.post_id}: {e}")


class ContentAnalysisService:
    """Content analysis service"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
    
    async def analyze_content(self, request: ContentAnalysisRequest) -> ContentAnalysisResponse:
        """Analyze content using ML pipeline"""
        try:
            # Generate content hash for caching
            content_hash = hashlib.md5(request.content.encode()).hexdigest()
            
            # Check cache first
            cached_analysis = await self._get_cached_analysis(content_hash)
            if cached_analysis:
                return ContentAnalysisResponse(
                    analysis_id=str(uuid4()),
                    content_hash=hash(request.content),
                    analysis_results=cached_analysis,
                    processing_time=0.0,
                    confidence_score=0.9,
                    recommendations=cached_analysis.get("recommendations", []),
                    generated_at=datetime.utcnow()
                )
            
            # Perform analysis (would integrate with ML pipeline)
            analysis_results = await self._perform_content_analysis(request)
            
            # Cache results
            await self._cache_analysis(content_hash, analysis_results)
            
            return ContentAnalysisResponse(
                analysis_id=str(uuid4()),
                content_hash=hash(request.content),
                analysis_results=analysis_results,
                processing_time=0.0,
                confidence_score=analysis_results.get("overall_quality", {}).get("overall_score", 0.5),
                recommendations=analysis_results.get("overall_quality", {}).get("recommendations", []),
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            error = handle_blog_error(e)
            log_blog_error(error)
            raise error
    
    async def _perform_content_analysis(self, request: ContentAnalysisRequest) -> Dict[str, Any]:
        """Perform content analysis using ML models"""
        # This would integrate with the ML pipeline service
        # For now, return mock analysis results
        
        return {
            "sentiment_score": 0.7,
            "readability_score": 0.8,
            "seo_score": 0.6,
            "engagement_score": 0.7,
            "viral_potential": 0.5,
            "topic_relevance": 0.8,
            "keyword_analysis": {
                "top_keywords": ["blog", "content", "analysis"],
                "keyword_density": {"blog": 2.5, "content": 1.8, "analysis": 1.2},
                "total_keywords": 15,
                "unique_keywords": 12
            },
            "structure_score": 0.7,
            "originality_score": 0.8,
            "overall_quality": {
                "overall_score": 0.7,
                "quality_level": "good",
                "individual_scores": {
                    "sentiment": 0.7,
                    "readability": 0.8,
                    "seo": 0.6,
                    "engagement": 0.7,
                    "viral_potential": 0.5,
                    "topic_relevance": 0.8,
                    "structure": 0.7,
                    "originality": 0.8
                },
                "recommendations": [
                    "Improve SEO by adding more relevant keywords",
                    "Consider adding more engaging elements to increase viral potential"
                ]
            }
        }
    
    async def _get_cached_analysis(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results"""
        try:
            cache_key = f"content_analysis:{content_hash}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Failed to get cached analysis {content_hash}: {e}")
        return None
    
    async def _cache_analysis(self, content_hash: str, analysis_results: Dict[str, Any]) -> None:
        """Cache analysis results"""
        try:
            cache_key = f"content_analysis:{content_hash}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(analysis_results, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache analysis {content_hash}: {e}")


class MLPipelineService:
    """ML Pipeline service for blog posts"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
    
    async def process_content_analysis(self, request: ContentAnalysisRequest) -> ContentAnalysisResponse:
        """Process content analysis request"""
        try:
            # This would integrate with the ML pipeline from the ml_pipeline.py
            # For now, return mock response
            
            analysis_results = {
                "sentiment_score": 0.7,
                "readability_score": 0.8,
                "seo_score": 0.6,
                "engagement_score": 0.7,
                "viral_potential": 0.5,
                "topic_relevance": 0.8,
                "overall_quality": {
                    "overall_score": 0.7,
                    "quality_level": "good",
                    "recommendations": [
                        "Improve SEO by adding more relevant keywords",
                        "Consider adding more engaging elements"
                    ]
                }
            }
            
            return ContentAnalysisResponse(
                analysis_id=str(uuid4()),
                content_hash=hash(request.content),
                analysis_results=analysis_results,
                processing_time=0.0,
                confidence_score=0.7,
                recommendations=analysis_results["overall_quality"]["recommendations"],
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            error = handle_blog_error(e)
            log_blog_error(error)
            raise error
    
    async def process_content_generation(self, request: ContentGenerationRequest) -> ContentGenerationResponse:
        """Process content generation request"""
        try:
            # This would integrate with the ML pipeline from the ml_pipeline.py
            # For now, return mock response
            
            generated_content = f"""
# {request.topic}

This is a generated blog post about {request.topic} for {request.target_audience}.

## Introduction

{request.topic} is an important topic that affects many people today. In this article, we'll explore the key aspects and provide valuable insights.

## Key Points

1. **Understanding {request.topic}**: This is crucial for success
2. **Best Practices**: Here are some recommendations
3. **Common Mistakes**: Things to avoid
4. **Future Trends**: What to expect

## Conclusion

In conclusion, {request.topic} is a complex topic that requires careful consideration. By following the guidelines in this article, you can improve your understanding and implementation.

## Call to Action

What are your thoughts on {request.topic}? Share your experiences in the comments below!
"""
            
            quality_metrics = {
                "quality_score": 0.7,
                "readability": 0.8,
                "seo_score": 0.6,
                "engagement": 0.7,
                "recommendations": [
                    "Add more specific examples",
                    "Include relevant statistics",
                    "Optimize for target keywords"
                ]
            }
            
            return ContentGenerationResponse(
                generation_id=str(uuid4()),
                generated_content=generated_content,
                word_count=len(generated_content.split()),
                quality_metrics=quality_metrics,
                generation_metadata={
                    "model_used": "gpt-4",
                    "prompt_tokens": 100,
                    "generation_time": datetime.utcnow().isoformat()
                },
                processing_time=0.0,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            error = handle_blog_error(e)
            log_blog_error(error)
            raise error
    
    async def process_seo_optimization(self, request: SEOOptimizationRequest) -> SEOOptimizationResponse:
        """Process SEO optimization request"""
        try:
            # This would integrate with the ML pipeline from the ml_pipeline.py
            # For now, return mock response
            
            # Analyze current SEO
            current_seo_score = 0.6
            
            # Generate optimized content
            optimized_content = request.content
            
            # Add keywords if missing
            content_lower = request.content.lower()
            for keyword in request.target_keywords:
                if keyword.lower() not in content_lower:
                    optimized_content += f"\n\nThis content covers important aspects of {keyword}."
            
            # Generate recommendations
            recommendations = [
                f"Add target keyword '{keyword}' to the content" for keyword in request.target_keywords
            ]
            
            if len(request.content.split()) < 300:
                recommendations.append("Increase content length to at least 300 words for better SEO")
            
            # Analyze keyword usage
            keyword_analysis = {}
            for keyword in request.target_keywords:
                keyword_lower = keyword.lower()
                count = optimized_content.lower().count(keyword_lower)
                density = (count / len(optimized_content.split())) * 100 if optimized_content.split() else 0
                
                keyword_analysis[keyword] = {
                    "count": count,
                    "density": round(density, 2),
                    "first_occurrence": optimized_content.lower().find(keyword_lower),
                    "in_title": keyword_lower in optimized_content.lower().split('\n')[0].lower()
                }
            
            return SEOOptimizationResponse(
                optimization_id=str(uuid4()),
                original_content=request.content,
                optimized_content=optimized_content,
                seo_score_before=current_seo_score,
                seo_score_after=0.8,
                recommendations=recommendations,
                keyword_analysis=keyword_analysis,
                processing_time=0.0,
                optimized_at=datetime.utcnow()
            )
            
        except Exception as e:
            error = handle_blog_error(e)
            log_blog_error(error)
            raise error





























