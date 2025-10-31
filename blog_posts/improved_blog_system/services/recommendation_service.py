"""
Content recommendation service using collaborative filtering and content-based algorithms
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from ..models.database import BlogPost, User, Like, Comment, View
from ..models.schemas import PostStatus
from ..core.exceptions import DatabaseError


class RecommendationService:
    """Service for content recommendations using multiple algorithms."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.content_similarity_cache = {}
        self.user_preferences_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        algorithm: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Get personalized content recommendations for a user."""
        try:
            if algorithm == "collaborative":
                recommendations = await self._collaborative_filtering(user_id, limit)
            elif algorithm == "content_based":
                recommendations = await self._content_based_filtering(user_id, limit)
            elif algorithm == "hybrid":
                recommendations = await self._hybrid_recommendations(user_id, limit)
            else:
                recommendations = await self._popularity_based_recommendations(limit)
            
            return recommendations
            
        except Exception as e:
            raise DatabaseError(f"Failed to get personalized recommendations: {str(e)}")
    
    async def get_similar_posts(
        self,
        post_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get posts similar to a given post."""
        try:
            # Get the target post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            target_post = post_result.scalar_one_or_none()
            
            if not target_post:
                return []
            
            # Get all published posts except the target
            all_posts_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.id != post_id
                )
            )
            all_posts_result = await self.session.execute(all_posts_query)
            all_posts = all_posts_result.scalars().all()
            
            if not all_posts:
                return []
            
            # Calculate similarities
            similarities = []
            target_content = f"{target_post.title} {target_post.content} {' '.join(target_post.tags or [])}"
            
            for post in all_posts:
                post_content = f"{post.title} {post.content} {' '.join(post.tags or [])}"
                similarity = await self._calculate_content_similarity(target_content, post_content)
                
                similarities.append({
                    "post": post,
                    "similarity": similarity
                })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            similar_posts = []
            for item in similarities[:limit]:
                post = item["post"]
                similar_posts.append({
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
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "similarity_score": item["similarity"]
                })
            
            return similar_posts
            
        except Exception as e:
            raise DatabaseError(f"Failed to get similar posts: {str(e)}")
    
    async def get_trending_posts(
        self,
        limit: int = 10,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get trending posts based on recent activity."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Calculate trending score based on views, likes, and comments
            trending_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.published_at >= since_date
                )
            ).order_by(desc(BlogPost.view_count + BlogPost.like_count * 2 + BlogPost.comment_count * 3))
            
            trending_result = await self.session.execute(trending_query)
            trending_posts = trending_result.scalars().all()
            
            trending_list = []
            for post in trending_posts[:limit]:
                trending_score = post.view_count + post.like_count * 2 + post.comment_count * 3
                
                trending_list.append({
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
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "trending_score": trending_score
                })
            
            return trending_list
            
        except Exception as e:
            raise DatabaseError(f"Failed to get trending posts: {str(e)}")
    
    async def get_author_recommendations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get author recommendations based on user preferences."""
        try:
            # Get user's liked posts and their authors
            liked_posts_query = select(BlogPost).join(Like).where(
                and_(
                    Like.user_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            liked_posts_result = await self.session.execute(liked_posts_query)
            liked_posts = liked_posts_result.scalars().all()
            
            if not liked_posts:
                # If no liked posts, return popular authors
                return await self._get_popular_authors(limit)
            
            # Get authors of liked posts
            liked_authors = [post.author_id for post in liked_posts]
            
            # Find similar authors based on content similarity
            author_similarities = {}
            
            for author_id in liked_authors:
                # Get posts by this author
                author_posts_query = select(BlogPost).where(
                    and_(
                        BlogPost.author_id == author_id,
                        BlogPost.status == PostStatus.PUBLISHED.value
                    )
                )
                author_posts_result = await self.session.execute(author_posts_query)
                author_posts = author_posts_result.scalars().all()
                
                if not author_posts:
                    continue
                
                # Find other authors with similar content
                all_authors_query = select(func.distinct(BlogPost.author_id)).where(
                    and_(
                        BlogPost.status == PostStatus.PUBLISHED.value,
                        BlogPost.author_id != author_id,
                        BlogPost.author_id.notin_(liked_authors)
                    )
                )
                all_authors_result = await self.session.execute(all_authors_query)
                all_authors = all_authors_result.scalars().all()
                
                for other_author_id in all_authors:
                    if other_author_id not in author_similarities:
                        author_similarities[other_author_id] = 0
                    
                    # Calculate similarity between authors' content
                    similarity = await self._calculate_author_similarity(author_id, other_author_id)
                    author_similarities[other_author_id] += similarity
            
            # Sort authors by similarity
            sorted_authors = sorted(author_similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Get author details
            recommended_authors = []
            for author_id, similarity in sorted_authors[:limit]:
                author_query = select(User).where(User.id == author_id)
                author_result = await self.session.execute(author_query)
                author = author_result.scalar_one_or_none()
                
                if author:
                    # Get author's post count
                    post_count_query = select(func.count(BlogPost.id)).where(
                        and_(
                            BlogPost.author_id == author_id,
                            BlogPost.status == PostStatus.PUBLISHED.value
                        )
                    )
                    post_count_result = await self.session.execute(post_count_query)
                    post_count = post_count_result.scalar()
                    
                    recommended_authors.append({
                        "id": str(author.id),
                        "username": author.username,
                        "full_name": author.full_name,
                        "bio": author.bio,
                        "avatar_url": author.avatar_url,
                        "post_count": post_count,
                        "similarity_score": similarity
                    })
            
            return recommended_authors
            
        except Exception as e:
            raise DatabaseError(f"Failed to get author recommendations: {str(e)}")
    
    async def get_category_recommendations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get category recommendations based on user preferences."""
        try:
            # Get user's interaction history
            user_interactions = await self._get_user_interactions(user_id)
            
            if not user_interactions:
                # If no interactions, return popular categories
                return await self._get_popular_categories(limit)
            
            # Calculate category preferences
            category_scores = {}
            for interaction in user_interactions:
                category = interaction.get("category")
                if category:
                    if category not in category_scores:
                        category_scores[category] = 0
                    
                    # Weight different types of interactions
                    weight = 1
                    if interaction["type"] == "like":
                        weight = 2
                    elif interaction["type"] == "comment":
                        weight = 3
                    elif interaction["type"] == "view":
                        weight = 1
                    
                    category_scores[category] += weight
            
            # Get all categories with post counts
            all_categories_query = select(
                BlogPost.category,
                func.count(BlogPost.id).label('post_count')
            ).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            ).group_by(BlogPost.category).order_by(desc('post_count'))
            
            all_categories_result = await self.session.execute(all_categories_query)
            all_categories = all_categories_result.all()
            
            # Calculate recommendation scores
            recommendations = []
            for category, post_count in all_categories:
                user_score = category_scores.get(category, 0)
                popularity_score = post_count
                
                # Combine user preference with popularity
                recommendation_score = user_score * 0.7 + popularity_score * 0.3
                
                recommendations.append({
                    "category": category,
                    "post_count": post_count,
                    "user_score": user_score,
                    "recommendation_score": recommendation_score
                })
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
            
            return recommendations[:limit]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get category recommendations: {str(e)}")
    
    async def _collaborative_filtering(
        self,
        user_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Collaborative filtering based on user behavior similarity."""
        try:
            # Get users with similar preferences
            similar_users = await self._find_similar_users(user_id)
            
            if not similar_users:
                return await self._popularity_based_recommendations(limit)
            
            # Get posts liked by similar users but not by current user
            similar_user_ids = [user["user_id"] for user in similar_users]
            
            recommendations_query = select(BlogPost).join(Like).where(
                and_(
                    Like.user_id.in_(similar_user_ids),
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.id.notin_(
                        select(Like.post_id).where(Like.user_id == user_id)
                    )
                )
            ).order_by(desc(BlogPost.created_at))
            
            recommendations_result = await self.session.execute(recommendations_query)
            recommended_posts = recommendations_result.scalars().all()
            
            # Format recommendations
            recommendations = []
            for post in recommended_posts[:limit]:
                recommendations.append({
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
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "recommendation_type": "collaborative"
                })
            
            return recommendations
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform collaborative filtering: {str(e)}")
    
    async def _content_based_filtering(
        self,
        user_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Content-based filtering based on user's content preferences."""
        try:
            # Get user's liked posts
            liked_posts_query = select(BlogPost).join(Like).where(
                and_(
                    Like.user_id == user_id,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            liked_posts_result = await self.session.execute(liked_posts_query)
            liked_posts = liked_posts_result.scalars().all()
            
            if not liked_posts:
                return await self._popularity_based_recommendations(limit)
            
            # Create user profile from liked posts
            user_profile = await self._create_user_profile(liked_posts)
            
            # Get all published posts not liked by user
            all_posts_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.id.notin_(
                        select(Like.post_id).where(Like.user_id == user_id)
                    )
                )
            )
            all_posts_result = await self.session.execute(all_posts_query)
            all_posts = all_posts_result.scalars().all()
            
            # Calculate content similarity
            similarities = []
            for post in all_posts:
                similarity = await self._calculate_user_post_similarity(user_profile, post)
                similarities.append({
                    "post": post,
                    "similarity": similarity
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Format recommendations
            recommendations = []
            for item in similarities[:limit]:
                post = item["post"]
                recommendations.append({
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
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "similarity_score": item["similarity"],
                    "recommendation_type": "content_based"
                })
            
            return recommendations
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform content-based filtering: {str(e)}")
    
    async def _hybrid_recommendations(
        self,
        user_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Hybrid recommendations combining collaborative and content-based filtering."""
        try:
            # Get recommendations from both algorithms
            collaborative_recs = await self._collaborative_filtering(user_id, limit * 2)
            content_based_recs = await self._content_based_filtering(user_id, limit * 2)
            
            # Combine and score recommendations
            combined_recs = {}
            
            # Add collaborative recommendations
            for rec in collaborative_recs:
                post_id = rec["id"]
                if post_id not in combined_recs:
                    combined_recs[post_id] = rec.copy()
                    combined_recs[post_id]["hybrid_score"] = 0
                combined_recs[post_id]["hybrid_score"] += 0.6  # Collaborative weight
            
            # Add content-based recommendations
            for rec in content_based_recs:
                post_id = rec["id"]
                if post_id not in combined_recs:
                    combined_recs[post_id] = rec.copy()
                    combined_recs[post_id]["hybrid_score"] = 0
                combined_recs[post_id]["hybrid_score"] += 0.4  # Content-based weight
            
            # Sort by hybrid score
            sorted_recs = sorted(combined_recs.values(), key=lambda x: x["hybrid_score"], reverse=True)
            
            return sorted_recs[:limit]
            
        except Exception as e:
            raise DatabaseError(f"Failed to generate hybrid recommendations: {str(e)}")
    
    async def _popularity_based_recommendations(self, limit: int) -> List[Dict[str, Any]]:
        """Fallback popularity-based recommendations."""
        try:
            popular_posts_query = select(BlogPost).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            ).order_by(desc(BlogPost.view_count + BlogPost.like_count * 2))
            
            popular_posts_result = await self.session.execute(popular_posts_query)
            popular_posts = popular_posts_result.scalars().all()
            
            recommendations = []
            for post in popular_posts[:limit]:
                recommendations.append({
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
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "recommendation_type": "popularity"
                })
            
            return recommendations
            
        except Exception as e:
            raise DatabaseError(f"Failed to get popularity-based recommendations: {str(e)}")
    
    async def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content pieces."""
        try:
            # Use TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([content1, content2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception:
            return 0.0
    
    async def _get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's interaction history."""
        try:
            interactions = []
            
            # Get liked posts
            liked_query = select(BlogPost).join(Like).where(Like.user_id == user_id)
            liked_result = await self.session.execute(liked_query)
            liked_posts = liked_result.scalars().all()
            
            for post in liked_posts:
                interactions.append({
                    "type": "like",
                    "post_id": post.id,
                    "category": post.category,
                    "tags": post.tags
                })
            
            # Get commented posts
            commented_query = select(BlogPost).join(Comment).where(Comment.author_id == user_id)
            commented_result = await self.session.execute(commented_query)
            commented_posts = commented_result.scalars().all()
            
            for post in commented_posts:
                interactions.append({
                    "type": "comment",
                    "post_id": post.id,
                    "category": post.category,
                    "tags": post.tags
                })
            
            return interactions
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user interactions: {str(e)}")
    
    async def _create_user_profile(self, liked_posts: List[BlogPost]) -> Dict[str, Any]:
        """Create user profile from liked posts."""
        try:
            profile = {
                "categories": {},
                "tags": {},
                "content_keywords": []
            }
            
            for post in liked_posts:
                # Count categories
                category = post.category
                if category:
                    profile["categories"][category] = profile["categories"].get(category, 0) + 1
                
                # Count tags
                if post.tags:
                    for tag in post.tags:
                        profile["tags"][tag] = profile["tags"].get(tag, 0) + 1
                
                # Extract keywords from content
                content_words = post.title.lower().split() + post.content.lower().split()
                profile["content_keywords"].extend(content_words)
            
            return profile
            
        except Exception as e:
            raise DatabaseError(f"Failed to create user profile: {str(e)}")
    
    async def _calculate_user_post_similarity(
        self,
        user_profile: Dict[str, Any],
        post: BlogPost
    ) -> float:
        """Calculate similarity between user profile and post."""
        try:
            similarity = 0.0
            
            # Category similarity
            if post.category in user_profile["categories"]:
                similarity += 0.4
            
            # Tag similarity
            if post.tags:
                tag_matches = sum(1 for tag in post.tags if tag in user_profile["tags"])
                if tag_matches > 0:
                    similarity += 0.3 * (tag_matches / len(post.tags))
            
            # Content similarity (simplified)
            post_content = f"{post.title} {post.content}".lower()
            user_keywords = user_profile["content_keywords"]
            keyword_matches = sum(1 for keyword in user_keywords if keyword in post_content)
            if keyword_matches > 0:
                similarity += 0.3 * min(keyword_matches / 100, 1.0)
            
            return similarity
            
        except Exception:
            return 0.0






























