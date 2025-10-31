"""
Comment service for comment management operations
"""

from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from sqlalchemy.orm import selectinload

from ..models.database import Comment, BlogPost, User
from ..models.schemas import CommentCreate, CommentResponse, PaginationParams
from ..core.exceptions import NotFoundError, DatabaseError
from ..utils.pagination import create_paginated_response


class CommentService:
    """Service for comment operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_comment(
        self,
        comment_data: CommentCreate,
        post_id: int,
        author_id: str
    ) -> CommentResponse:
        """Create a new comment."""
        try:
            # Verify post exists
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise NotFoundError("Blog post", post_id)
            
            # Verify parent comment exists if specified
            if comment_data.parent_id:
                parent_query = select(Comment).where(Comment.id == comment_data.parent_id)
                parent_result = await self.session.execute(parent_query)
                parent = parent_result.scalar_one_or_none()
                
                if not parent:
                    raise NotFoundError("Parent comment", comment_data.parent_id)
            
            # Create comment
            db_comment = Comment(
                post_id=post_id,
                author_id=author_id,
                parent_id=comment_data.parent_id,
                content=comment_data.content,
                is_approved=False  # Comments need approval by default
            )
            
            self.session.add(db_comment)
            await self.session.commit()
            await self.session.refresh(db_comment)
            
            # Update post comment count
            post.comment_count += 1
            await self.session.commit()
            
            return CommentResponse.model_validate(db_comment)
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to create comment: {str(e)}")
    
    async def get_comment(self, comment_id: int) -> CommentResponse:
        """Get comment by ID."""
        query = select(Comment).where(Comment.id == comment_id)
        result = await self.session.execute(query)
        comment = result.scalar_one_or_none()
        
        if not comment:
            raise NotFoundError("Comment", comment_id)
        
        return CommentResponse.model_validate(comment)
    
    async def list_comments(
        self,
        post_id: int,
        pagination: PaginationParams,
        approved_only: bool = True
    ) -> Tuple[List[CommentResponse], int]:
        """List comments for a blog post."""
        try:
            # Build query
            query = select(Comment).where(Comment.post_id == post_id)
            count_query = select(func.count(Comment.id)).where(Comment.post_id == post_id)
            
            if approved_only:
                query = query.where(Comment.is_approved == True)
                count_query = count_query.where(Comment.is_approved == True)
            
            # Get total count
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            query = query.order_by(desc(Comment.created_at))
            query = query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await self.session.execute(query)
            comments = result.scalars().all()
            
            return [CommentResponse.model_validate(comment) for comment in comments], total
            
        except Exception as e:
            raise DatabaseError(f"Failed to list comments: {str(e)}")
    
    async def approve_comment(self, comment_id: int) -> CommentResponse:
        """Approve a comment."""
        try:
            query = select(Comment).where(Comment.id == comment_id)
            result = await self.session.execute(query)
            comment = result.scalar_one_or_none()
            
            if not comment:
                raise NotFoundError("Comment", comment_id)
            
            comment.is_approved = True
            await self.session.commit()
            await self.session.refresh(comment)
            
            return CommentResponse.model_validate(comment)
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to approve comment: {str(e)}")
    
    async def reject_comment(self, comment_id: int) -> CommentResponse:
        """Reject a comment."""
        try:
            query = select(Comment).where(Comment.id == comment_id)
            result = await self.session.execute(query)
            comment = result.scalar_one_or_none()
            
            if not comment:
                raise NotFoundError("Comment", comment_id)
            
            comment.is_approved = False
            await self.session.commit()
            await self.session.refresh(comment)
            
            return CommentResponse.model_validate(comment)
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to reject comment: {str(e)}")
    
    async def mark_as_spam(self, comment_id: int) -> CommentResponse:
        """Mark a comment as spam."""
        try:
            query = select(Comment).where(Comment.id == comment_id)
            result = await self.session.execute(query)
            comment = result.scalar_one_or_none()
            
            if not comment:
                raise NotFoundError("Comment", comment_id)
            
            comment.is_spam = True
            comment.is_approved = False
            await self.session.commit()
            await self.session.refresh(comment)
            
            return CommentResponse.model_validate(comment)
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to mark comment as spam: {str(e)}")
    
    async def delete_comment(self, comment_id: int, user_id: str) -> bool:
        """Delete a comment."""
        try:
            query = select(Comment).where(Comment.id == comment_id)
            result = await self.session.execute(query)
            comment = result.scalar_one_or_none()
            
            if not comment:
                raise NotFoundError("Comment", comment_id)
            
            # Check authorization (author or admin)
            if comment.author_id != user_id:
                # In a real implementation, you would check user roles here
                pass
            
            # Update post comment count
            post_query = select(BlogPost).where(BlogPost.id == comment.post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if post:
                post.comment_count = max(0, post.comment_count - 1)
            
            await self.session.delete(comment)
            await self.session.commit()
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to delete comment: {str(e)}")
    
    async def get_pending_comments(
        self,
        pagination: PaginationParams
    ) -> Tuple[List[CommentResponse], int]:
        """Get pending comments for moderation."""
        try:
            # Build query
            query = select(Comment).where(
                and_(
                    Comment.is_approved == False,
                    Comment.is_spam == False
                )
            )
            count_query = select(func.count(Comment.id)).where(
                and_(
                    Comment.is_approved == False,
                    Comment.is_spam == False
                )
            )
            
            # Get total count
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            query = query.order_by(desc(Comment.created_at))
            query = query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await self.session.execute(query)
            comments = result.scalars().all()
            
            return [CommentResponse.model_validate(comment) for comment in comments], total
            
        except Exception as e:
            raise DatabaseError(f"Failed to get pending comments: {str(e)}")






























