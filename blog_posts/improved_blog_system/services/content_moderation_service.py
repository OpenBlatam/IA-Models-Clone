"""
Content moderation service for automated content filtering and safety
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
import openai
from transformers import pipeline
import torch

from ..models.database import BlogPost, Comment, User
from ..models.schemas import PostStatus
from ..core.exceptions import DatabaseError, ValidationError


class ContentModerationService:
    """Service for automated content moderation and safety."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.toxicity_classifier = None
        self.spam_detector = None
        self.openai_client = None
        
        # Initialize AI models
        self._initialize_models()
        
        # Content moderation rules
        self.spam_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'\b(?:buy|sell|cheap|discount|offer|deal|promo|free|win|prize|lottery)\b',
            r'\b(?:viagra|casino|poker|gambling|loan|credit|debt)\b',
            r'(.)\1{4,}',  # Repeated characters
            r'[A-Z]{5,}',  # Excessive caps
            r'[!]{3,}',    # Excessive exclamation marks
        ]
        
        self.profanity_words = [
            'spam', 'scam', 'fake', 'fraud', 'hate', 'abuse', 'harassment'
        ]
    
    def _initialize_models(self):
        """Initialize AI models for content moderation."""
        try:
            # Initialize toxicity classifier
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize spam detector
            self.spam_detector = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            print(f"Warning: Could not initialize some moderation models: {e}")
    
    async def moderate_content(
        self,
        content: str,
        content_type: str = "post",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Moderate content for safety and quality."""
        try:
            moderation_result = {
                "is_approved": True,
                "confidence": 1.0,
                "flags": [],
                "suggestions": [],
                "risk_score": 0.0,
                "moderation_details": {}
            }
            
            # Basic content checks
            basic_checks = await self._perform_basic_checks(content)
            moderation_result["moderation_details"]["basic_checks"] = basic_checks
            
            if not basic_checks["passed"]:
                moderation_result["is_approved"] = False
                moderation_result["flags"].extend(basic_checks["flags"])
                moderation_result["risk_score"] += 0.3
            
            # Spam detection
            spam_check = await self._detect_spam(content)
            moderation_result["moderation_details"]["spam_check"] = spam_check
            
            if spam_check["is_spam"]:
                moderation_result["is_approved"] = False
                moderation_result["flags"].append("spam")
                moderation_result["risk_score"] += 0.4
            
            # Toxicity detection
            toxicity_check = await self._detect_toxicity(content)
            moderation_result["moderation_details"]["toxicity_check"] = toxicity_check
            
            if toxicity_check["is_toxic"]:
                moderation_result["is_approved"] = False
                moderation_result["flags"].append("toxic")
                moderation_result["risk_score"] += 0.5
            
            # Quality assessment
            quality_check = await self._assess_content_quality(content, content_type)
            moderation_result["moderation_details"]["quality_check"] = quality_check
            
            if quality_check["quality_score"] < 0.5:
                moderation_result["flags"].append("low_quality")
                moderation_result["suggestions"].extend(quality_check["suggestions"])
                moderation_result["risk_score"] += 0.2
            
            # User reputation check
            if user_id:
                user_reputation = await self._check_user_reputation(user_id)
                moderation_result["moderation_details"]["user_reputation"] = user_reputation
                
                if user_reputation["reputation_score"] < 0.3:
                    moderation_result["risk_score"] += 0.2
                    moderation_result["flags"].append("low_reputation")
            
            # Final decision
            if moderation_result["risk_score"] > 0.7:
                moderation_result["is_approved"] = False
            elif moderation_result["risk_score"] > 0.4:
                moderation_result["is_approved"] = False
                moderation_result["flags"].append("requires_review")
            
            moderation_result["confidence"] = 1.0 - moderation_result["risk_score"]
            
            return moderation_result
            
        except Exception as e:
            raise DatabaseError(f"Failed to moderate content: {str(e)}")
    
    async def moderate_comment(
        self,
        comment: Comment,
        post: BlogPost
    ) -> Dict[str, Any]:
        """Moderate a comment with context."""
        try:
            # Get moderation result
            moderation_result = await self.moderate_content(
                comment.content,
                "comment",
                str(comment.author_id)
            )
            
            # Add comment-specific checks
            comment_checks = await self._perform_comment_checks(comment, post)
            moderation_result["moderation_details"]["comment_checks"] = comment_checks
            
            if not comment_checks["passed"]:
                moderation_result["is_approved"] = False
                moderation_result["flags"].extend(comment_checks["flags"])
                moderation_result["risk_score"] += 0.2
            
            # Update comment status
            comment.is_approved = moderation_result["is_approved"]
            comment.is_spam = "spam" in moderation_result["flags"]
            
            await self.session.commit()
            
            return moderation_result
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to moderate comment: {str(e)}")
    
    async def moderate_post(
        self,
        post: BlogPost
    ) -> Dict[str, Any]:
        """Moderate a blog post."""
        try:
            # Get moderation result
            moderation_result = await self.moderate_content(
                post.content,
                "post",
                str(post.author_id)
            )
            
            # Add post-specific checks
            post_checks = await self._perform_post_checks(post)
            moderation_result["moderation_details"]["post_checks"] = post_checks
            
            if not post_checks["passed"]:
                moderation_result["is_approved"] = False
                moderation_result["flags"].extend(post_checks["flags"])
                moderation_result["risk_score"] += 0.2
            
            # Update post status if needed
            if not moderation_result["is_approved"]:
                post.status = PostStatus.DRAFT.value
            
            await self.session.commit()
            
            return moderation_result
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to moderate post: {str(e)}")
    
    async def get_moderation_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get content moderation statistics."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Get total content
            total_posts_query = select(func.count(BlogPost.id)).where(
                BlogPost.created_at >= since_date
            )
            total_posts_result = await self.session.execute(total_posts_query)
            total_posts = total_posts_result.scalar()
            
            total_comments_query = select(func.count(Comment.id)).where(
                Comment.created_at >= since_date
            )
            total_comments_result = await self.session.execute(total_comments_query)
            total_comments = total_comments_result.scalar()
            
            # Get approved content
            approved_posts_query = select(func.count(BlogPost.id)).where(
                and_(
                    BlogPost.created_at >= since_date,
                    BlogPost.status == PostStatus.PUBLISHED.value
                )
            )
            approved_posts_result = await self.session.execute(approved_posts_query)
            approved_posts = approved_posts_result.scalar()
            
            approved_comments_query = select(func.count(Comment.id)).where(
                and_(
                    Comment.created_at >= since_date,
                    Comment.is_approved == True
                )
            )
            approved_comments_result = await self.session.execute(approved_comments_query)
            approved_comments = approved_comments_result.scalar()
            
            # Get flagged content
            spam_comments_query = select(func.count(Comment.id)).where(
                and_(
                    Comment.created_at >= since_date,
                    Comment.is_spam == True
                )
            )
            spam_comments_result = await self.session.execute(spam_comments_query)
            spam_comments = spam_comments_result.scalar()
            
            return {
                "period_days": days,
                "total_posts": total_posts,
                "total_comments": total_comments,
                "approved_posts": approved_posts,
                "approved_comments": approved_comments,
                "spam_comments": spam_comments,
                "approval_rate_posts": approved_posts / max(total_posts, 1) * 100,
                "approval_rate_comments": approved_comments / max(total_comments, 1) * 100,
                "spam_rate": spam_comments / max(total_comments, 1) * 100
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get moderation stats: {str(e)}")
    
    async def _perform_basic_checks(self, content: str) -> Dict[str, Any]:
        """Perform basic content checks."""
        try:
            checks = {
                "passed": True,
                "flags": [],
                "details": {}
            }
            
            # Length check
            if len(content) < 10:
                checks["passed"] = False
                checks["flags"].append("too_short")
                checks["details"]["length"] = len(content)
            
            if len(content) > 10000:
                checks["flags"].append("too_long")
                checks["details"]["length"] = len(content)
            
            # Character diversity check
            unique_chars = len(set(content.lower()))
            if unique_chars < 5:
                checks["passed"] = False
                checks["flags"].append("low_diversity")
                checks["details"]["unique_chars"] = unique_chars
            
            # Excessive repetition check
            if re.search(r'(.)\1{4,}', content):
                checks["passed"] = False
                checks["flags"].append("excessive_repetition")
            
            # Excessive caps check
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > 0.7:
                checks["passed"] = False
                checks["flags"].append("excessive_caps")
                checks["details"]["caps_ratio"] = caps_ratio
            
            return checks
            
        except Exception as e:
            return {"passed": False, "flags": ["check_error"], "details": {"error": str(e)}}
    
    async def _detect_spam(self, content: str) -> Dict[str, Any]:
        """Detect spam content."""
        try:
            spam_score = 0.0
            detected_patterns = []
            
            # Check against spam patterns
            for pattern in self.spam_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    spam_score += 0.2
                    detected_patterns.append(pattern)
            
            # Check for profanity
            content_lower = content.lower()
            for word in self.profanity_words:
                if word in content_lower:
                    spam_score += 0.3
                    detected_patterns.append(f"profanity: {word}")
            
            # Use AI model if available
            if self.spam_detector:
                try:
                    result = await asyncio.to_thread(self.spam_detector, content)
                    if result and len(result) > 0:
                        spam_score += result[0]["score"] * 0.5
                except Exception:
                    pass  # Fallback to rule-based detection
            
            return {
                "is_spam": spam_score > 0.5,
                "spam_score": spam_score,
                "detected_patterns": detected_patterns
            }
            
        except Exception as e:
            return {"is_spam": False, "spam_score": 0.0, "detected_patterns": [], "error": str(e)}
    
    async def _detect_toxicity(self, content: str) -> Dict[str, Any]:
        """Detect toxic content."""
        try:
            if not self.toxicity_classifier:
                return {"is_toxic": False, "toxicity_score": 0.0, "categories": []}
            
            # Truncate content if too long
            max_length = 512
            if len(content) > max_length:
                content = content[:max_length]
            
            result = await asyncio.to_thread(self.toxicity_classifier, content)
            
            if result and len(result) > 0:
                toxicity_score = result[0]["score"]
                is_toxic = toxicity_score > 0.5
                
                return {
                    "is_toxic": is_toxic,
                    "toxicity_score": toxicity_score,
                    "categories": [result[0]["label"]] if is_toxic else []
                }
            
            return {"is_toxic": False, "toxicity_score": 0.0, "categories": []}
            
        except Exception as e:
            return {"is_toxic": False, "toxicity_score": 0.0, "categories": [], "error": str(e)}
    
    async def _assess_content_quality(self, content: str, content_type: str) -> Dict[str, Any]:
        """Assess content quality."""
        try:
            quality_score = 1.0
            suggestions = []
            
            # Length appropriateness
            if content_type == "post":
                if len(content) < 300:
                    quality_score -= 0.2
                    suggestions.append("Consider adding more content for better engagement")
                elif len(content) > 5000:
                    quality_score -= 0.1
                    suggestions.append("Consider breaking into multiple posts")
            elif content_type == "comment":
                if len(content) < 10:
                    quality_score -= 0.3
                    suggestions.append("Please provide a more detailed comment")
                elif len(content) > 1000:
                    quality_score -= 0.1
                    suggestions.append("Consider shortening your comment")
            
            # Readability check
            sentences = content.split('.')
            if len(sentences) > 1:
                avg_sentence_length = len(content) / len(sentences)
                if avg_sentence_length > 50:
                    quality_score -= 0.1
                    suggestions.append("Consider using shorter sentences for better readability")
            
            # Structure check
            if content_type == "post":
                if not any(char in content for char in ['\n', '.', '!', '?']):
                    quality_score -= 0.2
                    suggestions.append("Consider adding proper paragraph breaks")
            
            return {
                "quality_score": max(0.0, quality_score),
                "suggestions": suggestions
            }
            
        except Exception as e:
            return {"quality_score": 0.5, "suggestions": [], "error": str(e)}
    
    async def _check_user_reputation(self, user_id: str) -> Dict[str, Any]:
        """Check user reputation for moderation context."""
        try:
            # Get user's recent activity
            since_date = datetime.now() - timedelta(days=30)
            
            # Get posts count
            posts_query = select(func.count(BlogPost.id)).where(
                and_(
                    BlogPost.author_id == user_id,
                    BlogPost.created_at >= since_date
                )
            )
            posts_result = await self.session.execute(posts_query)
            posts_count = posts_result.scalar()
            
            # Get comments count
            comments_query = select(func.count(Comment.id)).where(
                and_(
                    Comment.author_id == user_id,
                    Comment.created_at >= since_date
                )
            )
            comments_result = await self.session.execute(comments_query)
            comments_count = comments_result.scalar()
            
            # Get approved comments ratio
            approved_comments_query = select(func.count(Comment.id)).where(
                and_(
                    Comment.author_id == user_id,
                    Comment.created_at >= since_date,
                    Comment.is_approved == True
                )
            )
            approved_comments_result = await self.session.execute(approved_comments_query)
            approved_comments = approved_comments_result.scalar()
            
            # Get spam comments count
            spam_comments_query = select(func.count(Comment.id)).where(
                and_(
                    Comment.author_id == user_id,
                    Comment.created_at >= since_date,
                    Comment.is_spam == True
                )
            )
            spam_comments_result = await self.session.execute(spam_comments_query)
            spam_comments = spam_comments_result.scalar()
            
            # Calculate reputation score
            total_activity = posts_count + comments_count
            if total_activity == 0:
                reputation_score = 0.5  # Neutral for new users
            else:
                approval_rate = approved_comments / max(comments_count, 1)
                spam_rate = spam_comments / max(comments_count, 1)
                reputation_score = approval_rate - spam_rate
            
            return {
                "reputation_score": max(0.0, min(1.0, reputation_score)),
                "posts_count": posts_count,
                "comments_count": comments_count,
                "approved_comments": approved_comments,
                "spam_comments": spam_comments,
                "approval_rate": approval_rate if comments_count > 0 else 1.0
            }
            
        except Exception as e:
            return {"reputation_score": 0.5, "error": str(e)}
    
    async def _perform_comment_checks(self, comment: Comment, post: BlogPost) -> Dict[str, Any]:
        """Perform comment-specific checks."""
        try:
            checks = {
                "passed": True,
                "flags": [],
                "details": {}
            }
            
            # Check if comment is too similar to post content
            if comment.content.lower() in post.content.lower():
                checks["passed"] = False
                checks["flags"].append("duplicate_content")
            
            # Check comment length vs post length
            if len(comment.content) > len(post.content) * 0.5:
                checks["flags"].append("comment_too_long")
                checks["details"]["comment_length"] = len(comment.content)
                checks["details"]["post_length"] = len(post.content)
            
            return checks
            
        except Exception as e:
            return {"passed": False, "flags": ["check_error"], "details": {"error": str(e)}}
    
    async def _perform_post_checks(self, post: BlogPost) -> Dict[str, Any]:
        """Perform post-specific checks."""
        try:
            checks = {
                "passed": True,
                "flags": [],
                "details": {}
            }
            
            # Check for duplicate titles
            duplicate_title_query = select(BlogPost.id).where(
                and_(
                    BlogPost.title == post.title,
                    BlogPost.id != post.id
                )
            )
            duplicate_title_result = await self.session.execute(duplicate_title_query)
            if duplicate_title_result.scalar_one_or_none():
                checks["flags"].append("duplicate_title")
                checks["details"]["duplicate_title"] = post.title
            
            # Check for proper structure
            if not post.excerpt:
                checks["flags"].append("missing_excerpt")
            
            if not post.tags or len(post.tags) == 0:
                checks["flags"].append("missing_tags")
            
            return checks
            
        except Exception as e:
            return {"passed": False, "flags": ["check_error"], "details": {"error": str(e)}}






























