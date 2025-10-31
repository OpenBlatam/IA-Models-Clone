from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from ...core.domain.entities.linkedin_post import LinkedInPost, PostStatus, PostType, PostTone
from ...core.domain.repositories.linkedin_post_repository import LinkedInPostRepository
from ...infrastructure.nlp import get_pipeline
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
 AND WORKING WITH DIFFUSION MODELS.
 - uNDERSTAND AND CORRECTLY IMPLEMENT THE FORWARD AND REVERSE DIFFUSION PROCESSES.
 - uTILIZE APPROPRIATE NOISE SCHEDULERS AND SAMPLING METHODS.
 - uNDERSTAND AND CORRECTLY IMPLEMENT THE DIFFERENT PIPELINE, E.G., sTABLEdIFFUSIONpIPELINE AND sTABLEdIFFUSIONxlpIPELINE, ETC.
 
 mODEL t"""
LinkedIn Post Use Cases
======================

Application layer use cases for LinkedIn post management with fast NLP integration.
"""



logger = get_logger(__name__)


class LinkedInPostUseCases:
    """
    Use cases for LinkedIn post management with fast NLP integration.
    """
    
    def __init__(self, repository: LinkedInPostRepository):
        """Initialize use cases with repository and fast NLP enhancer."""
        self.repository = repository
        self.pipeline = get_pipeline()
    
    async def generate_post(
        self,
        content: str,
        post_type: PostType = PostType.ARTICLE,
        tone: PostTone = PostTone.PROFESSIONAL,
        target_audience: str = "",
        industry: str = "",
        use_fast_nlp: bool = True,
        use_async_nlp: bool = False
    ) -> LinkedInPost:
        """
        Generate a LinkedIn post with optional fast NLP enhancement.
        
        Args:
            content: Initial post content
            post_type: Type of post
            tone: Tone of the post
            target_audience: Target audience
            industry: Industry context
            use_fast_nlp: Enable fast NLP enhancement
            use_async_nlp: Use async NLP processor for maximum speed
        
        Returns:
            Generated LinkedIn post
        """
        try:
            logger.info(f"Generating LinkedIn post with fast NLP: {use_fast_nlp}, async: {use_async_nlp}")
            
            # Generate post ID
            post_id = str(uuid.uuid4())
            
            # Create initial post
            post = LinkedInPost(
                id=post_id,
                content=content,
                post_type=post_type,
                tone=tone,
                target_audience=target_audience,
                industry=industry,
                status=PostStatus.DRAFT,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Apply NLP enhancement if enabled
            if use_fast_nlp or use_async_nlp:
                enhanced_content = await self._enhance_content(content)
                if enhanced_content:
                    post.content = enhanced_content
                    post.nlp_enhanced = True
                    post.nlp_processing_time = enhanced_content.get("processing_time", 0)
            
            # Save to repository
            await self.repository.save(post)
            
            logger.info(f"LinkedIn post generated successfully: {post_id}")
            return post
            
        except Exception as e:
            logger.error(f"Error generating LinkedIn post: {e}")
            raise
    
    async def _enhance_content(self, content: str) -> str:
        """Enhance content using unified NLP pipeline."""
        try:
            result = await self.pipeline.enhance(content)
            return result.get("enhanced", {}).get("rewritten", content)
        except Exception as e:
            logger.error(f"Pipeline enhancement error: {e}")
            return content
    
    async def update_post(
        self,
        post_id: str,
        content: str = None,
        post_type: PostType = None,
        tone: PostTone = None,
        target_audience: str = None,
        industry: str = None,
        use_fast_nlp: bool = False,
        use_async_nlp: bool = False
    ) -> Optional[LinkedInPost]:
        """
        Update a LinkedIn post with optional NLP enhancement.
        
        Args:
            post_id: Post ID to update
            content: New content
            post_type: New post type
            tone: New tone
            target_audience: New target audience
            industry: New industry
            use_fast_nlp: Enable fast NLP enhancement
            use_async_nlp: Use async NLP processor
        
        Returns:
            Updated LinkedIn post
        """
        try:
            # Get existing post
            post = await self.repository.get_by_id(post_id)
            if not post:
                logger.warning(f"Post not found: {post_id}")
                return None
            
            # Update fields
            if content is not None:
                # Apply NLP enhancement if enabled
                if use_fast_nlp or use_async_nlp:
                    enhanced_content = await self._enhance_content(content)
                    if enhanced_content:
                        post.content = enhanced_content
                        post.nlp_enhanced = True
                        post.nlp_processing_time = enhanced_content.get("processing_time", 0)
                else:
                    post.content = content
            
            if post_type is not None:
                post.post_type = post_type
            if tone is not None:
                post.tone = tone
            if target_audience is not None:
                post.target_audience = target_audience
            if industry is not None:
                post.industry = industry
            
            post.updated_at = datetime.utcnow()
            
            # Save to repository
            await self.repository.save(post)
            
            logger.info(f"LinkedIn post updated successfully: {post_id}")
            return post
            
        except Exception as e:
            logger.error(f"Error updating LinkedIn post: {e}")
            raise
    
    async def list_posts(
        self,
        user_id: str = None,
        status: PostStatus = None,
        post_type: PostType = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[LinkedInPost]:
        """
        List LinkedIn posts with filtering.
        
        Args:
            user_id: Filter by user ID
            status: Filter by status
            post_type: Filter by post type
            limit: Maximum number of posts
            offset: Offset for pagination
        
        Returns:
            List of LinkedIn posts
        """
        try:
            posts = await self.repository.list_posts(
                user_id=user_id,
                status=status,
                post_type=post_type,
                limit=limit,
                offset=offset
            )
            
            logger.info(f"Retrieved {len(posts)} LinkedIn posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error listing LinkedIn posts: {e}")
            raise
    
    async def delete_post(self, post_id: str) -> bool:
        """
        Delete a LinkedIn post.
        
        Args:
            post_id: Post ID to delete
        
        Returns:
            True if deleted successfully
        """
        try:
            success = await self.repository.delete(post_id)
            if success:
                logger.info(f"LinkedIn post deleted successfully: {post_id}")
            else:
                logger.warning(f"Post not found for deletion: {post_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting LinkedIn post: {e}")
            raise
    
    async def optimize_post(
        self,
        post_id: str,
        use_async_nlp: bool = True
    ) -> Optional[LinkedInPost]:
        """
        Optimize a LinkedIn post using fast NLP.
        
        Args:
            post_id: Post ID to optimize
            use_async_nlp: Use async NLP processor for maximum speed
        
        Returns:
            Optimized LinkedIn post
        """
        try:
            # Get existing post
            post = await self.repository.get_by_id(post_id)
            if not post:
                logger.warning(f"Post not found for optimization: {post_id}")
                return None
            
            # Optimize content using fast NLP
            optimized_content = await self._enhance_content(post.content)
            
            if optimized_content and optimized_content != post.content:
                post.content = optimized_content
                post.nlp_enhanced = True
                post.nlp_processing_time = optimized_content.get("processing_time", 0)
                post.updated_at = datetime.utcnow()
                
                # Save optimized post
                await self.repository.save(post)
                
                logger.info(f"LinkedIn post optimized successfully: {post_id}")
                return post
            else:
                logger.info(f"No optimization needed for post: {post_id}")
                return post
                
        except Exception as e:
            logger.error(f"Error optimizing LinkedIn post: {e}")
            raise
    
    async def analyze_post_engagement(
        self,
        post_id: str,
        use_async_nlp: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze post engagement potential using fast NLP.
        
        Args:
            post_id: Post ID to analyze
            use_async_nlp: Use async NLP processor
        
        Returns:
            Engagement analysis results
        """
        try:
            # Get existing post
            post = await self.repository.get_by_id(post_id)
            if not post:
                logger.warning(f"Post not found for analysis: {post_id}")
                return {}
            
            # Analyze content using fast NLP
            if use_async_nlp:
                result = await self.pipeline.enhance(post.content)
            else:
                result = await self.pipeline.enhance(post.content)
            
            enhanced = result.get("enhanced", {})
            
            # Extract engagement metrics
            analysis = {
                "post_id": post_id,
                "sentiment_score": enhanced.get("sentiment", {}).get("compound", 0),
                "readability_score": enhanced.get("readability", {}).get("flesch_reading_ease", 0),
                "keywords": enhanced.get("keywords", []),
                "entities": enhanced.get("entities", []),
                "processing_time": result.get("processing_time", 0),
                "cached": result.get("cached", False),
                "async_optimized": result.get("async_optimized", False),
            }
            
            logger.info(f"Post engagement analysis completed: {post_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing post engagement: {e}")
            raise
    
    async def batch_optimize_posts(
        self,
        post_ids: List[str],
        use_async_nlp: bool = True
    ) -> List[LinkedInPost]:
        """
        Optimize multiple posts using batch processing.
        
        Args:
            post_ids: List of post IDs to optimize
            use_async_nlp: Use async NLP processor
        
        Returns:
            List of optimized posts
        """
        try:
            logger.info(f"Starting batch optimization of {len(post_ids)} posts")
            
            # Get all posts
            posts = []
            for post_id in post_ids:
                post = await self.repository.get_by_id(post_id)
                if post:
                    posts.append(post)
            
            if not posts:
                logger.warning("No posts found for batch optimization")
                return []
            
            # Extract content for batch processing
            contents = [post.content for post in posts]
            
            # Batch enhance content
            if use_async_nlp:
                results = await self.pipeline.enhance_multiple(contents)
            else:
                results = await self.pipeline.enhance_multiple(contents)
            
            # Update posts with optimized content
            optimized_posts = []
            for i, (post, result) in enumerate(zip(posts, results)):
                if result and not result.get("enhanced", {}).get("error"):
                    enhanced_content = result.get("enhanced", {}).get("rewritten")
                    if enhanced_content and enhanced_content != post.content:
                        post.content = enhanced_content
                        post.nlp_enhanced = True
                        post.nlp_processing_time = result.get("processing_time", 0)
                        post.updated_at = datetime.utcnow()
                        
                        # Save optimized post
                        await self.repository.save(post)
                        optimized_posts.append(post)
            
            logger.info(f"Batch optimization completed: {len(optimized_posts)} posts optimized")
            return optimized_posts
            
        except Exception as e:
            logger.error(f"Error in batch optimization: {e}")
            raise
    
    async def get_nlp_performance_metrics(self) -> Dict[str, Any]:
        """
        Get NLP performance metrics.
        
        Returns:
            Performance metrics for both fast and async NLP processors
        """
        try:
            metrics = self.pipeline.get_performance_metrics()
            
            return {
                "pipeline": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error getting NLP performance metrics: {e}")
            raise
    
    async def clear_nlp_cache(self) -> bool:
        """
        Clear NLP cache for both processors.
        
        Returns:
            True if cache cleared successfully
        """
        try:
            await self.pipeline.clear_cache()
            
            logger.info("NLP cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing NLP cache: {e}")
            raise 