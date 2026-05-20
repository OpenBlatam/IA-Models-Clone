from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4
import logging
from ..entities.linkedin_post import LinkedInPost, PostStatus, PostType, PostTone
from ..entities.user import User
from ..entities.template import Template
from ..repositories.post_repository import PostRepository
from ..repositories.user_repository import UserRepository
from ..repositories.template_repository import TemplateRepository
            from ..entities.linkedin_post import PostContent
from typing import Any, List, Dict, Optional
"""
Post service for LinkedIn Posts business logic.
"""




logger = logging.getLogger(__name__)


class PostService:
    """
    Post service for LinkedIn Posts business logic.
    
    Features:
    - Post creation and management
    - Content optimization
    - Scheduling and publishing
    - Analytics integration
    - AI enhancement
    """
    
    def __init__(
        self,
        post_repository: PostRepository,
        user_repository: UserRepository,
        template_repository: TemplateRepository,
        ai_service: Optional['AIService'] = None,
        analytics_service: Optional['AnalyticsService'] = None
    ):
        
    """__init__ function."""
self.post_repository = post_repository
        self.user_repository = user_repository
        self.template_repository = template_repository
        self.ai_service = ai_service
        self.analytics_service = analytics_service
    
    async def create_post(
        self,
        user_id: UUID,
        title: str,
        content: str,
        post_type: PostType = PostType.TEXT,
        tone: PostTone = PostTone.PROFESSIONAL,
        hashtags: Optional[List[str]] = None,
        mentions: Optional[List[str]] = None,
        links: Optional[List[str]] = None,
        media_urls: Optional[List[str]] = None,
        call_to_action: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
        enable_ai_optimization: bool = True
    ) -> LinkedInPost:
        """Create a new LinkedIn post."""
        try:
            # Validate user exists
            user = await self.user_repository.get_by_id(user_id)
            if not user:
                raise ValueError(f"User with id {user_id} not found")
            
            # Create post content
            post_content = PostContent(
                text=content,
                hashtags=hashtags or [],
                mentions=mentions or [],
                links=links or [],
                media_urls=media_urls or [],
                call_to_action=call_to_action
            )
            
            # Create post
            post = LinkedInPost(
                user_id=user_id,
                title=title,
                content=post_content,
                post_type=post_type,
                tone=tone,
                scheduled_at=scheduled_at
            )
            
            # AI optimization if enabled
            if enable_ai_optimization and self.ai_service:
                await self._optimize_post_with_ai(post)
            
            # Save post
            saved_post = await self.post_repository.create(post)
            
            logger.info(f"Created post {saved_post.id} for user {user_id}")
            return saved_post
            
        except Exception as e:
            logger.error(f"Error creating post for user {user_id}: {e}")
            raise
    
    async def create_post_from_template(
        self,
        user_id: UUID,
        template_id: UUID,
        variables: Dict[str, str],
        enable_ai_optimization: bool = True
    ) -> LinkedInPost:
        """Create a post from a template."""
        try:
            # Get template
            template = await self.template_repository.get_by_id(template_id)
            if not template:
                raise ValueError(f"Template with id {template_id} not found")
            
            # Validate variables
            missing_vars = template.validate_variables(variables)
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")
            
            # Render template
            content = template.render(variables)
            
            # Create post
            post = await self.create_post(
                user_id=user_id,
                title=template.name,
                content=content,
                post_type=template.template_type,
                enable_ai_optimization=enable_ai_optimization
            )
            
            # Update template usage
            template.increment_usage()
            await self.template_repository.update(template)
            
            return post
            
        except Exception as e:
            logger.error(f"Error creating post from template {template_id}: {e}")
            raise
    
    async def get_user_posts(
        self,
        user_id: UUID,
        status: Optional[PostStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[LinkedInPost]:
        """Get posts for a user with optional filtering."""
        try:
            posts = await self.post_repository.get_by_user_id(
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            return posts
        except Exception as e:
            logger.error(f"Error getting posts for user {user_id}: {e}")
            raise
    
    async def get_post_by_id(self, post_id: UUID) -> Optional[LinkedInPost]:
        """Get a post by ID."""
        try:
            return await self.post_repository.get_by_id(post_id)
        except Exception as e:
            logger.error(f"Error getting post {post_id}: {e}")
            raise
    
    async def update_post(
        self,
        post_id: UUID,
        user_id: UUID,
        **updates
    ) -> Optional[LinkedInPost]:
        """Update a post."""
        try:
            # Get post and verify ownership
            post = await self.post_repository.get_by_id(post_id)
            if not post:
                raise ValueError(f"Post with id {post_id} not found")
            
            if post.user_id != user_id:
                raise ValueError("User not authorized to update this post")
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(post, key):
                    setattr(post, key, value)
            
            post.updated_at = datetime.utcnow()
            
            # Save updated post
            updated_post = await self.post_repository.update(post)
            
            logger.info(f"Updated post {post_id}")
            return updated_post
            
        except Exception as e:
            logger.error(f"Error updating post {post_id}: {e}")
            raise
    
    async def delete_post(self, post_id: UUID, user_id: UUID) -> bool:
        """Delete a post."""
        try:
            # Get post and verify ownership
            post = await self.post_repository.get_by_id(post_id)
            if not post:
                raise ValueError(f"Post with id {post_id} not found")
            
            if post.user_id != user_id:
                raise ValueError("User not authorized to delete this post")
            
            # Delete post
            success = await self.post_repository.delete(post_id)
            
            if success:
                logger.info(f"Deleted post {post_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting post {post_id}: {e}")
            raise
    
    async def schedule_post(
        self,
        post_id: UUID,
        user_id: UUID,
        scheduled_time: datetime
    ) -> Optional[LinkedInPost]:
        """Schedule a post for later publication."""
        try:
            post = await self.update_post(
                post_id=post_id,
                user_id=user_id,
                scheduled_at=scheduled_time
            )
            
            if post:
                post.schedule(scheduled_time)
                return await self.post_repository.update(post)
            
            return None
            
        except Exception as e:
            logger.error(f"Error scheduling post {post_id}: {e}")
            raise
    
    async def publish_post(self, post_id: UUID, user_id: UUID) -> Optional[LinkedInPost]:
        """Publish a post immediately."""
        try:
            post = await self.get_post_by_id(post_id)
            if not post:
                raise ValueError(f"Post with id {post_id} not found")
            
            if post.user_id != user_id:
                raise ValueError("User not authorized to publish this post")
            
            if not post.can_publish:
                raise ValueError("Post cannot be published")
            
            # Publish post
            post.publish()
            published_post = await self.post_repository.update(post)
            
            # Update analytics
            if self.analytics_service:
                await self.analytics_service.record_post_published(published_post)
            
            logger.info(f"Published post {post_id}")
            return published_post
            
        except Exception as e:
            logger.error(f"Error publishing post {post_id}: {e}")
            raise
    
    async def get_scheduled_posts(self, user_id: UUID) -> List[LinkedInPost]:
        """Get all scheduled posts for a user."""
        try:
            return await self.get_user_posts(
                user_id=user_id,
                status=PostStatus.SCHEDULED
            )
        except Exception as e:
            logger.error(f"Error getting scheduled posts for user {user_id}: {e}")
            raise
    
    async def get_draft_posts(self, user_id: UUID) -> List[LinkedInPost]:
        """Get all draft posts for a user."""
        try:
            return await self.get_user_posts(
                user_id=user_id,
                status=PostStatus.DRAFT
            )
        except Exception as e:
            logger.error(f"Error getting draft posts for user {user_id}: {e}")
            raise
    
    async def get_published_posts(self, user_id: UUID) -> List[LinkedInPost]:
        """Get all published posts for a user."""
        try:
            return await self.get_user_posts(
                user_id=user_id,
                status=PostStatus.PUBLISHED
            )
        except Exception as e:
            logger.error(f"Error getting published posts for user {user_id}: {e}")
            raise
    
    async def search_posts(
        self,
        user_id: UUID,
        query: str,
        limit: int = 20
    ) -> List[LinkedInPost]:
        """Search posts by content."""
        try:
            return await self.post_repository.search(
                user_id=user_id,
                query=query,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error searching posts for user {user_id}: {e}")
            raise
    
    async def get_post_analytics(self, post_id: UUID, user_id: UUID) -> Dict[str, Any]:
        """Get analytics for a specific post."""
        try:
            post = await self.get_post_by_id(post_id)
            if not post or post.user_id != user_id:
                raise ValueError("Post not found or access denied")
            
            if self.analytics_service:
                return await self.analytics_service.get_post_analytics(post_id)
            
            # Return basic analytics
            return {
                "post_id": str(post_id),
                "engagement": post.engagement.to_dict(),
                "engagement_rate": post.engagement_rate,
                "total_engagement": post.total_engagement,
                "ai_score": post.ai_score,
                "performance_score": post.performance_score
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics for post {post_id}: {e}")
            raise
    
    async def _optimize_post_with_ai(self, post: LinkedInPost) -> None:
        """Optimize post content using AI."""
        try:
            if not self.ai_service:
                return
            
            # Get optimization suggestions
            optimization_result = await self.ai_service.optimize_post_content(post)
            
            # Apply optimizations
            post.optimize_for_ai(
                score=optimization_result.get('score', 0.0),
                suggestions=optimization_result.get('suggestions', [])
            )
            
            # Update keywords
            if 'keywords' in optimization_result:
                post.keywords = optimization_result['keywords']
            
            logger.info(f"Optimized post {post.id} with AI")
            
        except Exception as e:
            logger.error(f"Error optimizing post {post.id} with AI: {e}")
            # Don't fail the post creation if AI optimization fails
    
    async def batch_create_posts(
        self,
        user_id: UUID,
        posts_data: List[Dict[str, Any]]
    ) -> List[LinkedInPost]:
        """Create multiple posts in batch."""
        try:
            created_posts = []
            
            # Create posts concurrently
            tasks = [
                self.create_post(
                    user_id=user_id,
                    title=post_data.get('title', ''),
                    content=post_data.get('content', ''),
                    post_type=PostType(post_data.get('post_type', PostType.TEXT)),
                    tone=PostTone(post_data.get('tone', PostTone.PROFESSIONAL)),
                    hashtags=post_data.get('hashtags'),
                    mentions=post_data.get('mentions'),
                    links=post_data.get('links'),
                    media_urls=post_data.get('media_urls'),
                    call_to_action=post_data.get('call_to_action'),
                    scheduled_at=post_data.get('scheduled_at'),
                    enable_ai_optimization=post_data.get('enable_ai_optimization', True)
                )
                for post_data in posts_data
            ]
            
            created_posts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_posts = []
            for post in created_posts:
                if isinstance(post, Exception):
                    logger.error(f"Error in batch post creation: {post}")
                else:
                    valid_posts.append(post)
            
            logger.info(f"Created {len(valid_posts)} posts in batch for user {user_id}")
            return valid_posts
            
        except Exception as e:
            logger.error(f"Error in batch post creation for user {user_id}: {e}")
            raise 