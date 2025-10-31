"""
Advanced Analytics Service for comprehensive data analysis and insights
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text, extract
from collections import Counter, defaultdict
import json

from ..models.database import BlogPost, User, Comment, Like, View, Follow
from ..models.schemas import PostStatus
from ..core.exceptions import DatabaseError


class AdvancedAnalyticsService:
    """Service for advanced analytics and data insights."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_comprehensive_analytics(
        self,
        days: int = 30,
        granularity: str = "daily"
    ) -> Dict[str, Any]:
        """Get comprehensive analytics overview."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Get all analytics data
            content_analytics = await self._get_content_analytics(since_date, granularity)
            user_analytics = await self._get_user_analytics(since_date, granularity)
            engagement_analytics = await self._get_engagement_analytics(since_date, granularity)
            performance_analytics = await self._get_performance_analytics(since_date, granularity)
            social_analytics = await self._get_social_analytics(since_date, granularity)
            
            return {
                "period": {
                    "days": days,
                    "granularity": granularity,
                    "start_date": since_date.isoformat(),
                    "end_date": datetime.now().isoformat()
                },
                "content": content_analytics,
                "users": user_analytics,
                "engagement": engagement_analytics,
                "performance": performance_analytics,
                "social": social_analytics,
                "summary": await self._generate_analytics_summary(
                    content_analytics, user_analytics, engagement_analytics
                )
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get comprehensive analytics: {str(e)}")
    
    async def get_content_performance_analysis(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze content performance in detail."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Get top performing content
            top_posts_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).order_by(desc(BlogPost.view_count + BlogPost.like_count * 2 + BlogPost.comment_count * 3))
            
            top_posts_result = await self.session.execute(top_posts_query)
            top_posts = top_posts_result.scalars().all()
            
            # Analyze content metrics
            content_metrics = await self._analyze_content_metrics(top_posts)
            
            # Category performance
            category_performance = await self._analyze_category_performance(since_date)
            
            # Tag analysis
            tag_analysis = await self._analyze_tag_performance(since_date)
            
            # Content trends
            content_trends = await self._analyze_content_trends(since_date)
            
            return {
                "top_performing_posts": [
                    {
                        "id": post.id,
                        "title": post.title,
                        "slug": post.slug,
                        "view_count": post.view_count,
                        "like_count": post.like_count,
                        "comment_count": post.comment_count,
                        "engagement_score": post.view_count + post.like_count * 2 + post.comment_count * 3,
                        "published_at": post.published_at,
                        "category": post.category
                    }
                    for post in top_posts[:20]
                ],
                "content_metrics": content_metrics,
                "category_performance": category_performance,
                "tag_analysis": tag_analysis,
                "content_trends": content_trends
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to analyze content performance: {str(e)}")
    
    async def get_user_behavior_analysis(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # User activity patterns
            activity_patterns = await self._analyze_user_activity_patterns(since_date)
            
            # User engagement analysis
            engagement_analysis = await self._analyze_user_engagement(since_date)
            
            # User retention analysis
            retention_analysis = await self._analyze_user_retention(since_date)
            
            # User segmentation
            user_segments = await self._analyze_user_segments(since_date)
            
            return {
                "activity_patterns": activity_patterns,
                "engagement_analysis": engagement_analysis,
                "retention_analysis": retention_analysis,
                "user_segments": user_segments
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to analyze user behavior: {str(e)}")
    
    async def get_engagement_insights(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get detailed engagement insights."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Engagement metrics over time
            engagement_timeline = await self._get_engagement_timeline(since_date)
            
            # Comment analysis
            comment_analysis = await self._analyze_comments(since_date)
            
            # Like patterns
            like_patterns = await self._analyze_like_patterns(since_date)
            
            # View patterns
            view_patterns = await self._analyze_view_patterns(since_date)
            
            # Engagement correlation
            engagement_correlation = await self._analyze_engagement_correlation(since_date)
            
            return {
                "engagement_timeline": engagement_timeline,
                "comment_analysis": comment_analysis,
                "like_patterns": like_patterns,
                "view_patterns": view_patterns,
                "engagement_correlation": engagement_correlation
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get engagement insights: {str(e)}")
    
    async def get_predictive_analytics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get predictive analytics and forecasting."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Content performance prediction
            content_predictions = await self._predict_content_performance(since_date)
            
            # User growth prediction
            user_growth_prediction = await self._predict_user_growth(since_date)
            
            # Engagement trend prediction
            engagement_prediction = await self._predict_engagement_trends(since_date)
            
            # Seasonal analysis
            seasonal_analysis = await self._analyze_seasonal_patterns(since_date)
            
            return {
                "content_predictions": content_predictions,
                "user_growth_prediction": user_growth_prediction,
                "engagement_prediction": engagement_prediction,
                "seasonal_analysis": seasonal_analysis
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get predictive analytics: {str(e)}")
    
    async def get_advanced_metrics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get advanced metrics and KPIs."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Calculate advanced metrics
            metrics = await self._calculate_advanced_metrics(since_date)
            
            # Cohort analysis
            cohort_analysis = await self._perform_cohort_analysis(since_date)
            
            # Funnel analysis
            funnel_analysis = await self._perform_funnel_analysis(since_date)
            
            # A/B test results (mock)
            ab_test_results = await self._get_ab_test_results(since_date)
            
            return {
                "advanced_metrics": metrics,
                "cohort_analysis": cohort_analysis,
                "funnel_analysis": funnel_analysis,
                "ab_test_results": ab_test_results
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get advanced metrics: {str(e)}")
    
    async def _get_content_analytics(self, since_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get content analytics data."""
        try:
            # Posts published over time
            if granularity == "daily":
                posts_timeline_query = select(
                    func.date(BlogPost.published_at).label('date'),
                    func.count(BlogPost.id).label('count')
                ).where(
                    and_(
                        BlogPost.status == PostStatus.PUBLISHED.value,
                        BlogPost.published_at >= since_date
                    )
                ).group_by(func.date(BlogPost.published_at)).order_by('date')
            else:  # weekly
                posts_timeline_query = select(
                    func.date_trunc('week', BlogPost.published_at).label('week'),
                    func.count(BlogPost.id).label('count')
                ).where(
                    and_(
                        BlogPost.status == PostStatus.PUBLISHED.value,
                        BlogPost.published_at >= since_date
                    )
                ).group_by(func.date_trunc('week', BlogPost.published_at)).order_by('week')
            
            posts_timeline_result = await self.session.execute(posts_timeline_query)
            posts_timeline = posts_timeline_result.all()
            
            # Category distribution
            category_query = select(
                BlogPost.category,
                func.count(BlogPost.id).label('count')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).group_by(BlogPost.category).order_by(desc('count'))
            
            category_result = await self.session.execute(category_query)
            categories = category_result.all()
            
            return {
                "posts_timeline": [{"date": str(row.date), "count": row.count} for row in posts_timeline],
                "category_distribution": [{"category": row.category, "count": row.count} for row in categories],
                "total_posts": sum(row.count for row in posts_timeline)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_user_analytics(self, since_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get user analytics data."""
        try:
            # User registrations over time
            if granularity == "daily":
                users_timeline_query = select(
                    func.date(User.created_at).label('date'),
                    func.count(User.id).label('count')
                ).where(User.created_at >= since_date).group_by(func.date(User.created_at)).order_by('date')
            else:  # weekly
                users_timeline_query = select(
                    func.date_trunc('week', User.created_at).label('week'),
                    func.count(User.id).label('count')
                ).where(User.created_at >= since_date).group_by(func.date_trunc('week', User.created_at)).order_by('week')
            
            users_timeline_result = await self.session.execute(users_timeline_query)
            users_timeline = users_timeline_result.all()
            
            # Active users
            active_users_query = select(func.count(func.distinct(Like.user_id))).where(
                Like.created_at >= since_date
            )
            active_users_result = await self.session.execute(active_users_query)
            active_users = active_users_result.scalar()
            
            return {
                "users_timeline": [{"date": str(row.date), "count": row.count} for row in users_timeline],
                "active_users": active_users,
                "total_new_users": sum(row.count for row in users_timeline)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_engagement_analytics(self, since_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get engagement analytics data."""
        try:
            # Engagement over time
            if granularity == "daily":
                engagement_query = select(
                    func.date(Like.created_at).label('date'),
                    func.count(Like.id).label('likes'),
                    func.count(Comment.id).label('comments')
                ).outerjoin(Comment, func.date(Like.created_at) == func.date(Comment.created_at)).where(
                    Like.created_at >= since_date
                ).group_by(func.date(Like.created_at)).order_by('date')
            else:  # weekly
                engagement_query = select(
                    func.date_trunc('week', Like.created_at).label('week'),
                    func.count(Like.id).label('likes'),
                    func.count(Comment.id).label('comments')
                ).outerjoin(Comment, func.date_trunc('week', Like.created_at) == func.date_trunc('week', Comment.created_at)).where(
                    Like.created_at >= since_date
                ).group_by(func.date_trunc('week', Like.created_at)).order_by('week')
            
            engagement_result = await self.session.execute(engagement_query)
            engagement_timeline = engagement_result.all()
            
            return {
                "engagement_timeline": [
                    {
                        "date": str(row.date),
                        "likes": row.likes,
                        "comments": row.comments
                    }
                    for row in engagement_timeline
                ],
                "total_likes": sum(row.likes for row in engagement_timeline),
                "total_comments": sum(row.comments for row in engagement_timeline)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_performance_analytics(self, since_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get performance analytics data."""
        try:
            # View counts over time
            views_query = select(
                func.sum(BlogPost.view_count).label('total_views'),
                func.avg(BlogPost.view_count).label('avg_views'),
                func.max(BlogPost.view_count).label('max_views')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            )
            
            views_result = await self.session.execute(views_query)
            views_stats = views_result.first()
            
            return {
                "total_views": views_stats.total_views or 0,
                "average_views": float(views_stats.avg_views or 0),
                "max_views": views_stats.max_views or 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_social_analytics(self, since_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get social analytics data."""
        try:
            # Follow relationships
            follows_query = select(func.count(Follow.id)).where(Follow.created_at >= since_date)
            follows_result = await self.session.execute(follows_query)
            new_follows = follows_result.scalar()
            
            # Total follows
            total_follows_query = select(func.count(Follow.id))
            total_follows_result = await self.session.execute(total_follows_query)
            total_follows = total_follows_result.scalar()
            
            return {
                "new_follows": new_follows,
                "total_follows": total_follows
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _generate_analytics_summary(
        self,
        content_analytics: Dict[str, Any],
        user_analytics: Dict[str, Any],
        engagement_analytics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analytics summary."""
        try:
            return {
                "total_posts": content_analytics.get("total_posts", 0),
                "total_users": user_analytics.get("total_new_users", 0),
                "total_engagement": (
                    engagement_analytics.get("total_likes", 0) +
                    engagement_analytics.get("total_comments", 0)
                ),
                "engagement_rate": self._calculate_engagement_rate(
                    content_analytics.get("total_posts", 0),
                    engagement_analytics.get("total_likes", 0),
                    engagement_analytics.get("total_comments", 0)
                )
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_engagement_rate(self, posts: int, likes: int, comments: int) -> float:
        """Calculate engagement rate."""
        if posts == 0:
            return 0.0
        return (likes + comments) / posts
    
    async def _analyze_content_metrics(self, posts: List[BlogPost]) -> Dict[str, Any]:
        """Analyze content metrics."""
        try:
            if not posts:
                return {}
            
            view_counts = [post.view_count for post in posts]
            like_counts = [post.like_count for post in posts]
            comment_counts = [post.comment_count for post in posts]
            
            return {
                "views": {
                    "total": sum(view_counts),
                    "average": np.mean(view_counts),
                    "median": np.median(view_counts),
                    "max": max(view_counts)
                },
                "likes": {
                    "total": sum(like_counts),
                    "average": np.mean(like_counts),
                    "median": np.median(like_counts),
                    "max": max(like_counts)
                },
                "comments": {
                    "total": sum(comment_counts),
                    "average": np.mean(comment_counts),
                    "median": np.median(comment_counts),
                    "max": max(comment_counts)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_category_performance(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze category performance."""
        try:
            category_query = select(
                BlogPost.category,
                func.count(BlogPost.id).label('post_count'),
                func.sum(BlogPost.view_count).label('total_views'),
                func.sum(BlogPost.like_count).label('total_likes'),
                func.sum(BlogPost.comment_count).label('total_comments')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).group_by(BlogPost.category).order_by(desc('total_views'))
            
            category_result = await self.session.execute(category_query)
            categories = category_result.all()
            
            return [
                {
                    "category": row.category,
                    "post_count": row.post_count,
                    "total_views": row.total_views or 0,
                    "total_likes": row.total_likes or 0,
                    "total_comments": row.total_comments or 0,
                    "avg_views": (row.total_views or 0) / row.post_count if row.post_count > 0 else 0
                }
                for row in categories
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_tag_performance(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze tag performance."""
        try:
            # Get all posts with tags
            posts_query = select(BlogPost.tags, BlogPost.view_count, BlogPost.like_count, BlogPost.comment_count).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date,
                    BlogPost.tags.isnot(None)
                )
            )
            
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.all()
            
            # Analyze tag performance
            tag_stats = defaultdict(lambda: {"count": 0, "views": 0, "likes": 0, "comments": 0})
            
            for post in posts:
                if post.tags:
                    for tag in post.tags:
                        tag_stats[tag]["count"] += 1
                        tag_stats[tag]["views"] += post.view_count
                        tag_stats[tag]["likes"] += post.like_count
                        tag_stats[tag]["comments"] += post.comment_count
            
            # Convert to list and sort by views
            tag_performance = []
            for tag, stats in tag_stats.items():
                tag_performance.append({
                    "tag": tag,
                    "post_count": stats["count"],
                    "total_views": stats["views"],
                    "total_likes": stats["likes"],
                    "total_comments": stats["comments"],
                    "avg_views": stats["views"] / stats["count"] if stats["count"] > 0 else 0
                })
            
            tag_performance.sort(key=lambda x: x["total_views"], reverse=True)
            
            return tag_performance[:20]  # Top 20 tags
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_content_trends(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze content trends."""
        try:
            # Get posts by week
            weekly_query = select(
                func.date_trunc('week', BlogPost.published_at).label('week'),
                func.count(BlogPost.id).label('post_count'),
                func.avg(BlogPost.view_count).label('avg_views'),
                func.avg(BlogPost.like_count).label('avg_likes'),
                func.avg(BlogPost.comment_count).label('avg_comments')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).group_by(func.date_trunc('week', BlogPost.published_at)).order_by('week')
            
            weekly_result = await self.session.execute(weekly_query)
            weekly_data = weekly_result.all()
            
            return [
                {
                    "week": str(row.week),
                    "post_count": row.post_count,
                    "avg_views": float(row.avg_views or 0),
                    "avg_likes": float(row.avg_likes or 0),
                    "avg_comments": float(row.avg_comments or 0)
                }
                for row in weekly_data
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_user_activity_patterns(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        try:
            # Activity by hour of day
            hourly_query = select(
                extract('hour', Like.created_at).label('hour'),
                func.count(Like.id).label('activity_count')
            ).where(Like.created_at >= since_date).group_by(
                extract('hour', Like.created_at)
            ).order_by('hour')
            
            hourly_result = await self.session.execute(hourly_query)
            hourly_activity = hourly_result.all()
            
            # Activity by day of week
            daily_query = select(
                extract('dow', Like.created_at).label('day_of_week'),
                func.count(Like.id).label('activity_count')
            ).where(Like.created_at >= since_date).group_by(
                extract('dow', Like.created_at)
            ).order_by('day_of_week')
            
            daily_result = await self.session.execute(daily_query)
            daily_activity = daily_result.all()
            
            return {
                "hourly_activity": [{"hour": row.hour, "count": row.activity_count} for row in hourly_activity],
                "daily_activity": [{"day": row.day_of_week, "count": row.activity_count} for row in daily_activity]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_user_engagement(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze user engagement patterns."""
        try:
            # Most active users
            active_users_query = select(
                User.id,
                User.username,
                func.count(Like.id).label('likes_given'),
                func.count(Comment.id).label('comments_made')
            ).outerjoin(Like, User.id == Like.user_id).outerjoin(
                Comment, User.id == Comment.author_id
            ).where(
                and_(
                    Like.created_at >= since_date,
                    Comment.created_at >= since_date
                )
            ).group_by(User.id, User.username).order_by(
                desc('likes_given' + 'comments_made')
            ).limit(20)
            
            active_users_result = await self.session.execute(active_users_query)
            active_users = active_users_result.all()
            
            return [
                {
                    "user_id": str(row.id),
                    "username": row.username,
                    "likes_given": row.likes_given or 0,
                    "comments_made": row.comments_made or 0,
                    "total_activity": (row.likes_given or 0) + (row.comments_made or 0)
                }
                for row in active_users
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_user_retention(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze user retention."""
        try:
            # This is a simplified retention analysis
            # In a real implementation, you would track user cohorts
            
            # New users
            new_users_query = select(func.count(User.id)).where(User.created_at >= since_date)
            new_users_result = await self.session.execute(new_users_query)
            new_users = new_users_result.scalar()
            
            # Active users (users who have liked or commented)
            active_users_query = select(func.count(func.distinct(Like.user_id))).where(
                Like.created_at >= since_date
            )
            active_users_result = await self.session.execute(active_users_query)
            active_users = active_users_result.scalar()
            
            retention_rate = (active_users / new_users * 100) if new_users > 0 else 0
            
            return {
                "new_users": new_users,
                "active_users": active_users,
                "retention_rate": retention_rate
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_user_segments(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze user segments."""
        try:
            # Define user segments based on activity
            segments = {
                "highly_active": 0,
                "moderately_active": 0,
                "low_activity": 0,
                "inactive": 0
            }
            
            # Get user activity counts
            user_activity_query = select(
                User.id,
                func.count(Like.id).label('likes'),
                func.count(Comment.id).label('comments')
            ).outerjoin(Like, User.id == Like.user_id).outerjoin(
                Comment, User.id == Comment.author_id
            ).where(
                and_(
                    Like.created_at >= since_date,
                    Comment.created_at >= since_date
                )
            ).group_by(User.id)
            
            user_activity_result = await self.session.execute(user_activity_query)
            user_activities = user_activity_result.all()
            
            for user in user_activities:
                total_activity = (user.likes or 0) + (user.comments or 0)
                
                if total_activity >= 20:
                    segments["highly_active"] += 1
                elif total_activity >= 5:
                    segments["moderately_active"] += 1
                elif total_activity >= 1:
                    segments["low_activity"] += 1
                else:
                    segments["inactive"] += 1
            
            return segments
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_engagement_timeline(self, since_date: datetime) -> Dict[str, Any]:
        """Get engagement timeline."""
        try:
            # Daily engagement
            daily_query = select(
                func.date(Like.created_at).label('date'),
                func.count(Like.id).label('likes'),
                func.count(Comment.id).label('comments')
            ).outerjoin(Comment, func.date(Like.created_at) == func.date(Comment.created_at)).where(
                Like.created_at >= since_date
            ).group_by(func.date(Like.created_at)).order_by('date')
            
            daily_result = await self.session.execute(daily_query)
            daily_engagement = daily_result.all()
            
            return [
                {
                    "date": str(row.date),
                    "likes": row.likes,
                    "comments": row.comments,
                    "total_engagement": row.likes + row.comments
                }
                for row in daily_engagement
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_comments(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze comment patterns."""
        try:
            # Comment length analysis
            comment_length_query = select(
                func.length(Comment.content).label('length'),
                func.count(Comment.id).label('count')
            ).where(Comment.created_at >= since_date).group_by(
                func.length(Comment.content)
            ).order_by('length')
            
            comment_length_result = await self.session.execute(comment_length_query)
            comment_lengths = comment_length_result.all()
            
            # Most commented posts
            top_commented_query = select(
                BlogPost.id,
                BlogPost.title,
                func.count(Comment.id).label('comment_count')
            ).join(Comment, BlogPost.id == Comment.post_id).where(
                Comment.created_at >= since_date
            ).group_by(BlogPost.id, BlogPost.title).order_by(
                desc('comment_count')
            ).limit(10)
            
            top_commented_result = await self.session.execute(top_commented_query)
            top_commented = top_commented_result.all()
            
            return {
                "comment_length_distribution": [
                    {"length": row.length, "count": row.count}
                    for row in comment_lengths
                ],
                "most_commented_posts": [
                    {
                        "post_id": row.id,
                        "title": row.title,
                        "comment_count": row.comment_count
                    }
                    for row in top_commented
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_like_patterns(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze like patterns."""
        try:
            # Most liked posts
            top_liked_query = select(
                BlogPost.id,
                BlogPost.title,
                func.count(Like.id).label('like_count')
            ).join(Like, BlogPost.id == Like.post_id).where(
                Like.created_at >= since_date
            ).group_by(BlogPost.id, BlogPost.title).order_by(
                desc('like_count')
            ).limit(10)
            
            top_liked_result = await self.session.execute(top_liked_query)
            top_liked = top_liked_result.all()
            
            return [
                {
                    "post_id": row.id,
                    "title": row.title,
                    "like_count": row.like_count
                }
                for row in top_liked
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_view_patterns(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze view patterns."""
        try:
            # Most viewed posts
            top_viewed_query = select(
                BlogPost.id,
                BlogPost.title,
                BlogPost.view_count
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).order_by(desc(BlogPost.view_count)).limit(10)
            
            top_viewed_result = await self.session.execute(top_viewed_query)
            top_viewed = top_viewed_result.all()
            
            return [
                {
                    "post_id": row.id,
                    "title": row.title,
                    "view_count": row.view_count
                }
                for row in top_viewed
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_engagement_correlation(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze engagement correlation."""
        try:
            # Get posts with engagement data
            posts_query = select(
                BlogPost.view_count,
                BlogPost.like_count,
                BlogPost.comment_count
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            )
            
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.all()
            
            if not posts:
                return {"error": "No data available"}
            
            # Calculate correlations
            views = [post.view_count for post in posts]
            likes = [post.like_count for post in posts]
            comments = [post.comment_count for post in posts]
            
            # Simple correlation calculation
            views_likes_corr = np.corrcoef(views, likes)[0, 1] if len(views) > 1 else 0
            views_comments_corr = np.corrcoef(views, comments)[0, 1] if len(views) > 1 else 0
            likes_comments_corr = np.corrcoef(likes, comments)[0, 1] if len(likes) > 1 else 0
            
            return {
                "views_likes_correlation": float(views_likes_corr),
                "views_comments_correlation": float(views_comments_corr),
                "likes_comments_correlation": float(likes_comments_corr)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _predict_content_performance(self, since_date: datetime) -> Dict[str, Any]:
        """Predict content performance (mock implementation)."""
        try:
            # This would typically use machine learning models
            # For now, we'll return mock predictions
            
            return {
                "predicted_views": 1000,
                "predicted_likes": 50,
                "predicted_comments": 10,
                "confidence": 0.75,
                "model_used": "linear_regression"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _predict_user_growth(self, since_date: datetime) -> Dict[str, Any]:
        """Predict user growth (mock implementation)."""
        try:
            # This would typically use time series forecasting
            # For now, we'll return mock predictions
            
            return {
                "predicted_new_users_30_days": 150,
                "predicted_new_users_90_days": 450,
                "confidence": 0.8,
                "model_used": "arima"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _predict_engagement_trends(self, since_date: datetime) -> Dict[str, Any]:
        """Predict engagement trends (mock implementation)."""
        try:
            # This would typically use time series forecasting
            # For now, we'll return mock predictions
            
            return {
                "predicted_engagement_30_days": 5000,
                "predicted_engagement_90_days": 15000,
                "trend": "increasing",
                "confidence": 0.7,
                "model_used": "exponential_smoothing"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_seasonal_patterns(self, since_date: datetime) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        try:
            # Analyze by month
            monthly_query = select(
                extract('month', BlogPost.published_at).label('month'),
                func.count(BlogPost.id).label('post_count'),
                func.avg(BlogPost.view_count).label('avg_views')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).group_by(extract('month', BlogPost.published_at)).order_by('month')
            
            monthly_result = await self.session.execute(monthly_query)
            monthly_data = monthly_result.all()
            
            return [
                {
                    "month": int(row.month),
                    "post_count": row.post_count,
                    "avg_views": float(row.avg_views or 0)
                }
                for row in monthly_data
            ]
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _calculate_advanced_metrics(self, since_date: datetime) -> Dict[str, Any]:
        """Calculate advanced metrics."""
        try:
            # Get basic counts
            posts_count_query = select(func.count(BlogPost.id)).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            )
            posts_count_result = await self.session.execute(posts_count_query)
            posts_count = posts_count_result.scalar()
            
            users_count_query = select(func.count(User.id)).where(User.created_at >= since_date)
            users_count_result = await self.session.execute(users_count_query)
            users_count = users_count_result.scalar()
            
            likes_count_query = select(func.count(Like.id)).where(Like.created_at >= since_date)
            likes_count_result = await self.session.execute(likes_count_query)
            likes_count = likes_count_result.scalar()
            
            comments_count_query = select(func.count(Comment.id)).where(Comment.created_at >= since_date)
            comments_count_result = await self.session.execute(comments_count_query)
            comments_count = comments_count_result.scalar()
            
            # Calculate advanced metrics
            engagement_rate = (likes_count + comments_count) / posts_count if posts_count > 0 else 0
            comments_per_post = comments_count / posts_count if posts_count > 0 else 0
            likes_per_post = likes_count / posts_count if posts_count > 0 else 0
            
            return {
                "engagement_rate": engagement_rate,
                "comments_per_post": comments_per_post,
                "likes_per_post": likes_per_post,
                "posts_per_user": posts_count / users_count if users_count > 0 else 0,
                "total_engagement": likes_count + comments_count
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _perform_cohort_analysis(self, since_date: datetime) -> Dict[str, Any]:
        """Perform cohort analysis (mock implementation)."""
        try:
            # This would typically analyze user cohorts over time
            # For now, we'll return mock data
            
            return {
                "cohorts": [
                    {"month": "2024-01", "users": 100, "retention": 0.8},
                    {"month": "2024-02", "users": 120, "retention": 0.75},
                    {"month": "2024-03", "users": 150, "retention": 0.85}
                ],
                "average_retention": 0.8
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _perform_funnel_analysis(self, since_date: datetime) -> Dict[str, Any]:
        """Perform funnel analysis (mock implementation)."""
        try:
            # This would typically analyze user conversion funnels
            # For now, we'll return mock data
            
            return {
                "funnel_steps": [
                    {"step": "visitors", "count": 1000, "conversion_rate": 1.0},
                    {"step": "registered", "count": 200, "conversion_rate": 0.2},
                    {"step": "active_users", "count": 100, "conversion_rate": 0.5},
                    {"step": "content_creators", "count": 50, "conversion_rate": 0.5}
                ],
                "overall_conversion": 0.05
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_ab_test_results(self, since_date: datetime) -> Dict[str, Any]:
        """Get A/B test results (mock implementation)."""
        try:
            # This would typically return real A/B test results
            # For now, we'll return mock data
            
            return {
                "active_tests": 2,
                "completed_tests": 5,
                "test_results": [
                    {
                        "test_name": "Homepage Layout",
                        "variant_a": {"conversion_rate": 0.12, "sample_size": 1000},
                        "variant_b": {"conversion_rate": 0.15, "sample_size": 1000},
                        "winner": "variant_b",
                        "confidence": 0.95
                    }
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}






























