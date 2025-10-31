"""
Social features service for user interactions, follows, and social engagement
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.orm import selectinload

from ..models.database import User, BlogPost, Like, Comment, Follow, View
from ..models.schemas import PostStatus
from ..core.exceptions import DatabaseError, ValidationError


class SocialService:
    """Service for social features and user interactions."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def like_post(
        self,
        post_id: int,
        user_id: str
    ) -> Dict[str, Any]:
        """Like a blog post."""
        try:
            # Check if post exists
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Check if already liked
            existing_like_query = select(Like).where(
                and_(Like.post_id == post_id, Like.user_id == user_id)
            )
            existing_like_result = await self.session.execute(existing_like_query)
            existing_like = existing_like_result.scalar_one_or_none()
            
            if existing_like:
                # Unlike the post
                await self.session.delete(existing_like)
                post.like_count = max(0, post.like_count - 1)
                action = "unliked"
            else:
                # Like the post
                new_like = Like(post_id=post_id, user_id=user_id)
                self.session.add(new_like)
                post.like_count += 1
                action = "liked"
            
            await self.session.commit()
            
            return {
                "action": action,
                "post_id": post_id,
                "like_count": post.like_count,
                "user_liked": action == "liked"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to like/unlike post: {str(e)}")
    
    async def follow_user(
        self,
        target_user_id: str,
        follower_user_id: str
    ) -> Dict[str, Any]:
        """Follow or unfollow a user."""
        try:
            # Check if target user exists
            target_user_query = select(User).where(User.id == target_user_id)
            target_user_result = await self.session.execute(target_user_query)
            target_user = target_user_result.scalar_one_or_none()
            
            if not target_user:
                raise ValidationError("Target user not found")
            
            if target_user_id == follower_user_id:
                raise ValidationError("Cannot follow yourself")
            
            # Check if already following
            existing_follow_query = select(Follow).where(
                and_(
                    Follow.follower_id == follower_user_id,
                    Follow.following_id == target_user_id
                )
            )
            existing_follow_result = await self.session.execute(existing_follow_query)
            existing_follow = existing_follow_result.scalar_one_or_none()
            
            if existing_follow:
                # Unfollow
                await self.session.delete(existing_follow)
                action = "unfollowed"
            else:
                # Follow
                new_follow = Follow(follower_id=follower_user_id, following_id=target_user_id)
                self.session.add(new_follow)
                action = "followed"
            
            await self.session.commit()
            
            return {
                "action": action,
                "target_user_id": target_user_id,
                "follower_user_id": follower_user_id
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to follow/unfollow user: {str(e)}")
    
    async def get_user_followers(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get followers of a user."""
        try:
            # Get followers
            followers_query = select(User).join(Follow, User.id == Follow.follower_id).where(
                Follow.following_id == user_id
            ).offset(offset).limit(limit)
            
            followers_result = await self.session.execute(followers_query)
            followers = followers_result.scalars().all()
            
            # Get total count
            count_query = select(func.count(Follow.id)).where(Follow.following_id == user_id)
            count_result = await self.session.execute(count_query)
            total = count_result.scalar()
            
            # Format followers
            followers_list = []
            for follower in followers:
                followers_list.append({
                    "id": str(follower.id),
                    "username": follower.username,
                    "full_name": follower.full_name,
                    "bio": follower.bio,
                    "avatar_url": follower.avatar_url,
                    "is_active": follower.is_active,
                    "created_at": follower.created_at
                })
            
            return followers_list, total
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user followers: {str(e)}")
    
    async def get_user_following(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get users that a user is following."""
        try:
            # Get following
            following_query = select(User).join(Follow, User.id == Follow.following_id).where(
                Follow.follower_id == user_id
            ).offset(offset).limit(limit)
            
            following_result = await self.session.execute(following_query)
            following = following_result.scalars().all()
            
            # Get total count
            count_query = select(func.count(Follow.id)).where(Follow.follower_id == user_id)
            count_result = await self.session.execute(count_query)
            total = count_result.scalar()
            
            # Format following
            following_list = []
            for user in following:
                following_list.append({
                    "id": str(user.id),
                    "username": user.username,
                    "full_name": user.full_name,
                    "bio": user.bio,
                    "avatar_url": user.avatar_url,
                    "is_active": user.is_active,
                    "created_at": user.created_at
                })
            
            return following_list, total
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user following: {str(e)}")
    
    async def get_user_feed(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get personalized feed for a user based on following."""
        try:
            # Get posts from users that the current user follows
            feed_query = select(BlogPost).join(Follow, BlogPost.author_id == Follow.following_id).where(
                and_(
                    Follow.follower_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            ).order_by(desc(BlogPost.published_at)).offset(offset).limit(limit)
            
            feed_result = await self.session.execute(feed_query)
            feed_posts = feed_result.scalars().all()
            
            # Get total count
            count_query = select(func.count(BlogPost.id)).join(Follow, BlogPost.author_id == Follow.following_id).where(
                and_(
                    Follow.follower_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            count_result = await self.session.execute(count_query)
            total = count_result.scalar()
            
            # Format feed posts
            feed_list = []
            for post in feed_posts:
                # Check if user liked this post
                like_query = select(Like).where(
                    and_(Like.post_id == post.id, Like.user_id == user_id)
                )
                like_result = await self.session.execute(like_query)
                user_liked = like_result.scalar_one_or_none() is not None
                
                feed_list.append({
                    "id": post.id,
                    "uuid": str(post.uuid),
                    "title": post.title,
                    "slug": post.slug,
                    "excerpt": post.excerpt,
                    "author_id": str(post.author_id),
                    "category": post.category,
                    "tags": post.tags,
                    "view_count": post.view_count,
                    "like_count": post.like_count,
                    "comment_count": post.comment_count,
                    "reading_time_minutes": post.reading_time_minutes,
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "featured_image_url": post.featured_image_url,
                    "user_liked": user_liked
                })
            
            return feed_list, total
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user feed: {str(e)}")
    
    async def get_user_activity(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's recent activity."""
        try:
            activities = []
            
            # Get recent posts
            posts_query = select(BlogPost).where(
                and_(
                    BlogPost.author_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            ).order_by(desc(BlogPost.published_at)).limit(5)
            
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            for post in posts:
                activities.append({
                    "type": "post_published",
                    "timestamp": post.published_at,
                    "data": {
                        "post_id": post.id,
                        "post_title": post.title,
                        "post_slug": post.slug
                    }
                })
            
            # Get recent likes
            likes_query = select(Like).join(BlogPost, Like.post_id == BlogPost.id).where(
                Like.user_id == user_id
            ).order_by(desc(Like.created_at)).limit(10)
            
            likes_result = await self.session.execute(likes_query)
            likes = likes_result.scalars().all()
            
            for like in likes:
                activities.append({
                    "type": "post_liked",
                    "timestamp": like.created_at,
                    "data": {
                        "post_id": like.post_id,
                        "post_title": like.post.title if like.post else "Unknown Post"
                    }
                })
            
            # Get recent comments
            comments_query = select(Comment).join(BlogPost, Comment.post_id == BlogPost.id).where(
                Comment.author_id == user_id
            ).order_by(desc(Comment.created_at)).limit(10)
            
            comments_result = await self.session.execute(comments_query)
            comments = comments_result.scalars().all()
            
            for comment in comments:
                activities.append({
                    "type": "comment_created",
                    "timestamp": comment.created_at,
                    "data": {
                        "comment_id": comment.id,
                        "post_id": comment.post_id,
                        "post_title": comment.post.title if comment.post else "Unknown Post",
                        "comment_content": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content
                    }
                })
            
            # Sort activities by timestamp
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return activities[offset:offset + limit]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user activity: {str(e)}")
    
    async def get_user_stats(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        try:
            # Get follower count
            followers_count_query = select(func.count(Follow.id)).where(Follow.following_id == user_id)
            followers_count_result = await self.session.execute(followers_count_query)
            followers_count = followers_count_result.scalar()
            
            # Get following count
            following_count_query = select(func.count(Follow.id)).where(Follow.follower_id == user_id)
            following_count_result = await self.session.execute(following_count_query)
            following_count = following_count_result.scalar()
            
            # Get posts count
            posts_count_query = select(func.count(BlogPost.id)).where(
                and_(
                    BlogPost.author_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            posts_count_result = await self.session.execute(posts_count_query)
            posts_count = posts_count_result.scalar()
            
            # Get total likes received
            likes_received_query = select(func.sum(BlogPost.like_count)).where(
                and_(
                    BlogPost.author_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            likes_received_result = await self.session.execute(likes_received_query)
            likes_received = likes_received_result.scalar() or 0
            
            # Get total views received
            views_received_query = select(func.sum(BlogPost.view_count)).where(
                and_(
                    BlogPost.author_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            views_received_result = await self.session.execute(views_received_query)
            views_received = views_received_result.scalar() or 0
            
            # Get comments count
            comments_count_query = select(func.count(Comment.id)).where(Comment.author_id == user_id)
            comments_count_result = await self.session.execute(comments_count_query)
            comments_count = comments_count_result.scalar()
            
            # Get likes given count
            likes_given_query = select(func.count(Like.id)).where(Like.user_id == user_id)
            likes_given_result = await self.session.execute(likes_given_query)
            likes_given = likes_given_result.scalar()
            
            return {
                "followers_count": followers_count,
                "following_count": following_count,
                "posts_count": posts_count,
                "likes_received": likes_received,
                "views_received": views_received,
                "comments_count": comments_count,
                "likes_given": likes_given,
                "engagement_rate": (likes_received + comments_count) / max(views_received, 1) * 100
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user stats: {str(e)}")
    
    async def get_trending_users(
        self,
        limit: int = 10,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get trending users based on recent activity."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Get users with most recent activity
            trending_query = select(
                User,
                func.count(BlogPost.id).label('recent_posts'),
                func.sum(BlogPost.like_count).label('recent_likes'),
                func.sum(BlogPost.view_count).label('recent_views')
            ).join(BlogPost, User.id == BlogPost.author_id).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).group_by(User.id).order_by(
                desc('recent_posts'),
                desc('recent_likes'),
                desc('recent_views')
            ).limit(limit)
            
            trending_result = await self.session.execute(trending_query)
            trending_users = trending_result.all()
            
            trending_list = []
            for user, recent_posts, recent_likes, recent_views in trending_users:
                trending_list.append({
                    "id": str(user.id),
                    "username": user.username,
                    "full_name": user.full_name,
                    "bio": user.bio,
                    "avatar_url": user.avatar_url,
                    "recent_posts": recent_posts,
                    "recent_likes": recent_likes or 0,
                    "recent_views": recent_views or 0,
                    "trending_score": recent_posts + (recent_likes or 0) * 0.5 + (recent_views or 0) * 0.1
                })
            
            return trending_list
            
        except Exception as e:
            raise DatabaseError(f"Failed to get trending users: {str(e)}")
    
    async def check_follow_status(
        self,
        follower_id: str,
        following_id: str
    ) -> Dict[str, Any]:
        """Check if one user follows another."""
        try:
            follow_query = select(Follow).where(
                and_(
                    Follow.follower_id == follower_id,
                    Follow.following_id == following_id
                )
            )
            follow_result = await self.session.execute(follow_query)
            follow = follow_result.scalar_one_or_none()
            
            return {
                "is_following": follow is not None,
                "follower_id": follower_id,
                "following_id": following_id
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to check follow status: {str(e)}")
    
    async def get_mutual_follows(
        self,
        user1_id: str,
        user2_id: str
    ) -> List[Dict[str, Any]]:
        """Get mutual follows between two users."""
        try:
            # Get users that both users follow
            mutual_query = select(User).join(Follow, User.id == Follow.following_id).where(
                and_(
                    Follow.follower_id == user1_id,
                    Follow.following_id.in_(
                        select(Follow.following_id).where(Follow.follower_id == user2_id)
                    )
                )
            )
            
            mutual_result = await self.session.execute(mutual_query)
            mutual_users = mutual_result.scalars().all()
            
            mutual_list = []
            for user in mutual_users:
                mutual_list.append({
                    "id": str(user.id),
                    "username": user.username,
                    "full_name": user.full_name,
                    "avatar_url": user.avatar_url
                })
            
            return mutual_list
            
        except Exception as e:
            raise DatabaseError(f"Failed to get mutual follows: {str(e)}")






























