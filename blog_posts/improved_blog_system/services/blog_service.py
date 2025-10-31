"""
Blog service for business logic and data operations
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..models.database import BlogPost, User, Comment, Like, Category, Tag
from ..models.schemas import (
    BlogPostCreate, BlogPostUpdate, BlogPostResponse, BlogPostListResponse,
    PaginationParams, SearchParams, PostStatus, PostCategory
)
from ..core.exceptions import NotFoundError, ConflictError, DatabaseError
from ..utils.text_processing import generate_slug, calculate_reading_time, extract_keywords


class BlogService:
    """Service for blog post operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_post(
        self,
        post_data: BlogPostCreate,
        author_id: str
    ) -> BlogPostResponse:
        """Create a new blog post."""
        try:
            # Generate slug from title
            slug = await self._generate_unique_slug(post_data.title)
            
            # Calculate content metrics
            word_count = len(post_data.content.split())
            reading_time = calculate_reading_time(post_data.content)
            
            # Create post
            db_post = BlogPost(
                title=post_data.title,
                slug=slug,
                content=post_data.content,
                excerpt=post_data.excerpt or self._generate_excerpt(post_data.content),
                author_id=author_id,
                category=post_data.category.value,
                tags=post_data.tags,
                seo_title=post_data.seo_title,
                seo_description=post_data.seo_description,
                seo_keywords=post_data.seo_keywords,
                featured_image_url=post_data.featured_image_url,
                scheduled_at=post_data.scheduled_at,
                word_count=word_count,
                reading_time_minutes=reading_time,
                status=PostStatus.DRAFT.value
            )
            
            self.session.add(db_post)
            await self.session.commit()
            await self.session.refresh(db_post)
            
            # Update tag usage counts
            await self._update_tag_usage_counts(post_data.tags)
            
            return BlogPostResponse.model_validate(db_post)
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create blog post: {str(e)}")
    
    async def get_post(self, post_id: int) -> BlogPostResponse:
        """Get a blog post by ID."""
        query = select(BlogPost).where(BlogPost.id == post_id)
        result = await self.session.execute(query)
        post = result.scalar_one_or_none()
        
        if not post:
            raise NotFoundError("Blog post", post_id)
        
        return BlogPostResponse.model_validate(post)
    
    async def get_post_by_slug(self, slug: str) -> BlogPostResponse:
        """Get a blog post by slug."""
        query = select(BlogPost).where(BlogPost.slug == slug)
        result = await self.session.execute(query)
        post = result.scalar_one_or_none()
        
        if not post:
            raise NotFoundError("Blog post", slug)
        
        return BlogPostResponse.model_validate(post)
    
    async def update_post(
        self,
        post_id: int,
        post_data: BlogPostUpdate,
        user_id: str
    ) -> BlogPostResponse:
        """Update an existing blog post."""
        try:
            # Get existing post
            query = select(BlogPost).where(BlogPost.id == post_id)
            result = await self.session.execute(query)
            post = result.scalar_one_or_none()
            
            if not post:
                raise NotFoundError("Blog post", post_id)
            
            # Check authorization (author or admin)
            if post.author_id != user_id:
                # In a real implementation, you would check user roles here
                pass
            
            # Update fields
            update_data = post_data.model_dump(exclude_unset=True)
            
            # Handle slug update if title changed
            if 'title' in update_data and update_data['title'] != post.title:
                update_data['slug'] = await self._generate_unique_slug(
                    update_data['title'], exclude_id=post_id
                )
            
            # Update content metrics if content changed
            if 'content' in update_data:
                update_data['word_count'] = len(update_data['content'].split())
                update_data['reading_time_minutes'] = calculate_reading_time(update_data['content'])
                if not update_data.get('excerpt'):
                    update_data['excerpt'] = self._generate_excerpt(update_data['content'])
            
            # Update post
            for field, value in update_data.items():
                setattr(post, field, value)
            
            await self.session.commit()
            await self.session.refresh(post)
            
            # Update tag usage counts if tags changed
            if 'tags' in update_data:
                await self._update_tag_usage_counts(update_data['tags'])
            
            return BlogPostResponse.model_validate(post)
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update blog post: {str(e)}")
    
    async def delete_post(self, post_id: int, user_id: str) -> bool:
        """Delete a blog post."""
        try:
            query = select(BlogPost).where(BlogPost.id == post_id)
            result = await self.session.execute(query)
            post = result.scalar_one_or_none()
            
            if not post:
                raise NotFoundError("Blog post", post_id)
            
            # Check authorization
            if post.author_id != user_id:
                # In a real implementation, you would check user roles here
                pass
            
            await self.session.delete(post)
            await self.session.commit()
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to delete blog post: {str(e)}")
    
    async def list_posts(
        self,
        pagination: PaginationParams,
        status: Optional[PostStatus] = None,
        category: Optional[PostCategory] = None,
        author_id: Optional[str] = None,
        include_drafts: bool = False
    ) -> Tuple[List[BlogPostListResponse], int]:
        """List blog posts with pagination and filtering."""
        try:
            # Build query
            query = select(BlogPost)
            count_query = select(func.count(BlogPost.id))
            
            # Apply filters
            filters = []
            
            if status:
                filters.append(BlogPost.status == status.value)
            elif not include_drafts:
                filters.append(BlogPost.status == PostStatus.PUBLISHED.value)
            
            if category:
                filters.append(BlogPost.category == category.value)
            
            if author_id:
                filters.append(BlogPost.author_id == author_id)
            
            if filters:
                query = query.where(and_(*filters))
                count_query = count_query.where(and_(*filters))
            
            # Get total count
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            query = query.order_by(desc(BlogPost.created_at))
            query = query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await self.session.execute(query)
            posts = result.scalars().all()
            
            return [BlogPostListResponse.model_validate(post) for post in posts], total
            
        except Exception as e:
            raise DatabaseError(f"Failed to list blog posts: {str(e)}")
    
    async def search_posts(
        self,
        search_params: SearchParams,
        pagination: PaginationParams
    ) -> Tuple[List[BlogPostListResponse], int]:
        """Search blog posts with advanced filtering."""
        try:
            # Build base query
            query = select(BlogPost)
            count_query = select(func.count(BlogPost.id))
            
            # Apply search filters
            filters = []
            
            # Text search
            if search_params.query:
                search_term = f"%{search_params.query}%"
                filters.append(
                    or_(
                        BlogPost.title.ilike(search_term),
                        BlogPost.content.ilike(search_term),
                        BlogPost.excerpt.ilike(search_term)
                    )
                )
            
            # Category filter
            if search_params.category:
                filters.append(BlogPost.category == search_params.category.value)
            
            # Tags filter
            if search_params.tags:
                for tag in search_params.tags:
                    filters.append(BlogPost.tags.contains([tag]))
            
            # Author filter
            if search_params.author_id:
                filters.append(BlogPost.author_id == search_params.author_id)
            
            # Status filter
            if search_params.status:
                filters.append(BlogPost.status == search_params.status.value)
            
            # Date filters
            if search_params.date_from:
                filters.append(BlogPost.created_at >= search_params.date_from)
            
            if search_params.date_to:
                filters.append(BlogPost.created_at <= search_params.date_to)
            
            # Apply filters
            if filters:
                query = query.where(and_(*filters))
                count_query = count_query.where(and_(*filters))
            
            # Get total count
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Apply sorting
            sort_field = getattr(BlogPost, search_params.sort_by, BlogPost.created_at)
            if search_params.sort_order == "asc":
                query = query.order_by(asc(sort_field))
            else:
                query = query.order_by(desc(sort_field))
            
            # Apply pagination
            query = query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await self.session.execute(query)
            posts = result.scalars().all()
            
            return [BlogPostListResponse.model_validate(post) for post in posts], total
            
        except Exception as e:
            raise DatabaseError(f"Failed to search blog posts: {str(e)}")
    
    async def increment_view_count(self, post_id: int) -> None:
        """Increment view count for a blog post."""
        try:
            query = select(BlogPost).where(BlogPost.id == post_id)
            result = await self.session.execute(query)
            post = result.scalar_one_or_none()
            
            if post:
                post.view_count += 1
                await self.session.commit()
                
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to increment view count: {str(e)}")
    
    async def _generate_unique_slug(self, title: str, exclude_id: Optional[int] = None) -> str:
        """Generate a unique slug from title."""
        base_slug = generate_slug(title)
        slug = base_slug
        counter = 1
        
        while True:
            query = select(BlogPost).where(BlogPost.slug == slug)
            if exclude_id:
                query = query.where(BlogPost.id != exclude_id)
            
            result = await self.session.execute(query)
            existing_post = result.scalar_one_or_none()
            
            if not existing_post:
                break
            
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug
    
    def _generate_excerpt(self, content: str, max_length: int = 200) -> str:
        """Generate excerpt from content."""
        # Remove HTML tags and get plain text
        import re
        plain_text = re.sub(r'<[^>]+>', '', content)
        
        if len(plain_text) <= max_length:
            return plain_text
        
        # Find the last complete word within the limit
        truncated = plain_text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can find a good break point
            return truncated[:last_space] + "..."
        
        return truncated + "..."
    
    async def _update_tag_usage_counts(self, tags: List[str]) -> None:
        """Update tag usage counts."""
        try:
            for tag_name in tags:
                # Get or create tag
                query = select(Tag).where(Tag.name == tag_name)
                result = await self.session.execute(query)
                tag = result.scalar_one_or_none()
                
                if not tag:
                    tag = Tag(
                        name=tag_name,
                        slug=generate_slug(tag_name)
                    )
                    self.session.add(tag)
                
                # Count posts with this tag
                count_query = select(func.count(BlogPost.id)).where(
                    BlogPost.tags.contains([tag_name])
                )
                count_result = await self.session.execute(count_query)
                tag.usage_count = count_result.scalar()
            
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update tag usage counts: {str(e)}")






























