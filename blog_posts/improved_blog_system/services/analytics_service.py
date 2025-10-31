"""
Analytics service for blog statistics and insights
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text

from ..models.database import BlogPost, User, Comment, Like, View
from ..core.exceptions import DatabaseError


class AnalyticsService:
    """Service for analytics and statistics."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_blog_overview_stats(self) -> Dict[str, Any]:
        """Get overall blog statistics."""
        try:
            # Total posts
            total_posts_query = select(func.count(BlogPost.id))
            total_posts_result = await self.session.execute(total_posts_query)
            total_posts = total_posts_result.scalar()
            
            # Published posts
            published_posts_query = select(func.count(BlogPost.id)).where(
                BlogPost.status == "published"
            )
            published_posts_result = await self.session.execute(published_posts_query)
            published_posts = published_posts_result.scalar()
            
            # Total users
            total_users_query = select(func.count(User.id))
            total_users_result = await self.session.execute(total_users_query)
            total_users = total_users_result.scalar()
            
            # Active users (users with posts)
            active_users_query = select(func.count(func.distinct(BlogPost.author_id)))
            active_users_result = await self.session.execute(active_users_query)
            active_users = active_users_result.scalar()
            
            # Total comments
            total_comments_query = select(func.count(Comment.id))
            total_comments_result = await self.session.execute(total_comments_query)
            total_comments = total_comments_result.scalar()
            
            # Total likes
            total_likes_query = select(func.count(Like.id))
            total_likes_result = await self.session.execute(total_likes_query)
            total_likes = total_likes_result.scalar()
            
            # Total views
            total_views_query = select(func.sum(BlogPost.view_count))
            total_views_result = await self.session.execute(total_views_query)
            total_views = total_views_result.scalar() or 0
            
            return {
                "total_posts": total_posts,
                "published_posts": published_posts,
                "draft_posts": total_posts - published_posts,
                "total_users": total_users,
                "active_users": active_users,
                "total_comments": total_comments,
                "total_likes": total_likes,
                "total_views": total_views,
                "average_views_per_post": total_views / published_posts if published_posts > 0 else 0
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get blog overview stats: {str(e)}")
    
    async def get_popular_posts(
        self,
        limit: int = 10,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get most popular posts."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            query = select(BlogPost).where(
                and_(
                    BlogPost.status == "published",
                    BlogPost.published_at >= since_date
                )
            ).order_by(desc(BlogPost.view_count)).limit(limit)
            
            result = await self.session.execute(query)
            posts = result.scalars().all()
            
            return [
                {
                    "id": post.id,
                    "title": post.title,
                    "slug": post.slug,
                    "author_id": str(post.author_id),
                    "view_count": post.view_count,
                    "like_count": post.like_count,
                    "comment_count": post.comment_count,
                    "published_at": post.published_at,
                    "category": post.category
                }
                for post in posts
            ]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get popular posts: {str(e)}")
    
    async def get_author_stats(self, author_id: str) -> Dict[str, Any]:
        """Get statistics for a specific author."""
        try:
            # Author's posts
            posts_query = select(BlogPost).where(BlogPost.author_id == author_id)
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            if not posts:
                return {
                    "total_posts": 0,
                    "published_posts": 0,
                    "total_views": 0,
                    "total_likes": 0,
                    "total_comments": 0,
                    "average_views_per_post": 0
                }
            
            published_posts = [p for p in posts if p.status == "published"]
            total_views = sum(p.view_count for p in posts)
            total_likes = sum(p.like_count for p in posts)
            total_comments = sum(p.comment_count for p in posts)
            
            return {
                "total_posts": len(posts),
                "published_posts": len(published_posts),
                "draft_posts": len(posts) - len(published_posts),
                "total_views": total_views,
                "total_likes": total_likes,
                "total_comments": total_comments,
                "average_views_per_post": total_views / len(published_posts) if published_posts else 0
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get author stats: {str(e)}")
    
    async def get_category_stats(self) -> List[Dict[str, Any]]:
        """Get statistics by category."""
        try:
            query = select(
                BlogPost.category,
                func.count(BlogPost.id).label('post_count'),
                func.sum(BlogPost.view_count).label('total_views'),
                func.sum(BlogPost.like_count).label('total_likes'),
                func.sum(BlogPost.comment_count).label('total_comments')
            ).where(
                BlogPost.status == "published"
            ).group_by(BlogPost.category).order_by(desc('post_count'))
            
            result = await self.session.execute(query)
            categories = result.all()
            
            return [
                {
                    "category": cat.category,
                    "post_count": cat.post_count,
                    "total_views": cat.total_views or 0,
                    "total_likes": cat.total_likes or 0,
                    "total_comments": cat.total_comments or 0,
                    "average_views": (cat.total_views or 0) / cat.post_count if cat.post_count > 0 else 0
                }
                for cat in categories
            ]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get category stats: {str(e)}")
    
    async def get_trending_tags(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get trending tags based on usage."""
        try:
            # This is a simplified implementation
            # In a real system, you might want to track tag usage over time
            query = select(BlogPost.tags).where(BlogPost.status == "published")
            result = await self.session.execute(query)
            all_tags = result.scalars().all()
            
            # Count tag usage
            tag_counts = {}
            for tags in all_tags:
                if tags:
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Sort by usage count
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            
            return [
                {"tag": tag, "usage_count": count}
                for tag, count in sorted_tags[:limit]
            ]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get trending tags: {str(e)}")
    
    async def get_engagement_metrics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get engagement metrics for the specified period."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Posts published in period
            posts_query = select(func.count(BlogPost.id)).where(
                and_(
                    BlogPost.status == "published",
                    BlogPost.published_at >= since_date
                )
            )
            posts_result = await self.session.execute(posts_query)
            posts_published = posts_result.scalar()
            
            # Comments in period
            comments_query = select(func.count(Comment.id)).where(
                Comment.created_at >= since_date
            )
            comments_result = await self.session.execute(comments_query)
            comments_count = comments_result.scalar()
            
            # Likes in period
            likes_query = select(func.count(Like.id)).where(
                Like.created_at >= since_date
            )
            likes_result = await self.session.execute(likes_query)
            likes_count = likes_result.scalar()
            
            # Views in period (approximate based on posts published)
            views_query = select(func.sum(BlogPost.view_count)).where(
                and_(
                    BlogPost.status == "published",
                    BlogPost.published_at >= since_date
                )
            )
            views_result = await self.session.execute(views_query)
            views_count = views_result.scalar() or 0
            
            return {
                "period_days": days,
                "posts_published": posts_published,
                "total_comments": comments_count,
                "total_likes": likes_count,
                "total_views": views_count,
                "average_comments_per_post": comments_count / posts_published if posts_published > 0 else 0,
                "average_likes_per_post": likes_count / posts_published if posts_published > 0 else 0,
                "average_views_per_post": views_count / posts_published if posts_published > 0 else 0
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get engagement metrics: {str(e)}")
    
    async def get_user_activity_timeline(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get user activity timeline."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Get user's posts
            posts_query = select(BlogPost).where(
                and_(
                    BlogPost.author_id == user_id,
                    BlogPost.created_at >= since_date
                )
            ).order_by(desc(BlogPost.created_at))
            
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            # Get user's comments
            comments_query = select(Comment).where(
                and_(
                    Comment.author_id == user_id,
                    Comment.created_at >= since_date
                )
            ).order_by(desc(Comment.created_at))
            
            comments_result = await self.session.execute(comments_query)
            comments = comments_result.scalars().all()
            
            # Combine and sort activities
            activities = []
            
            for post in posts:
                activities.append({
                    "type": "post_created",
                    "title": post.title,
                    "timestamp": post.created_at,
                    "data": {
                        "post_id": post.id,
                        "status": post.status
                    }
                })
            
            for comment in comments:
                activities.append({
                    "type": "comment_created",
                    "title": f"Commented on post",
                    "timestamp": comment.created_at,
                    "data": {
                        "comment_id": comment.id,
                        "post_id": comment.post_id
                    }
                })
            
            # Sort by timestamp
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return activities
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user activity timeline: {str(e)}")






























