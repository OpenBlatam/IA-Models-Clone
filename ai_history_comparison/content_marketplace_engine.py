"""
Content Marketplace Engine - Advanced Content Commerce and Distribution
================================================================

This module provides comprehensive content marketplace capabilities including:
- Content marketplace and e-commerce
- Content licensing and rights management
- Multi-platform content distribution
- Content monetization and revenue tracking
- Content subscription management
- Content recommendation and discovery
- Content creator management
- Content analytics and reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
from decimal import Decimal
import stripe
from paypalrestsdk import Payment
import boto3
from google.cloud import storage
import requests
from fastapi import HTTPException
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentLicense(Enum):
    """Content license enumeration"""
    FREE = "free"
    PREMIUM = "premium"
    SUBSCRIPTION = "subscription"
    ONE_TIME_PURCHASE = "one_time_purchase"
    CUSTOM = "custom"

class ContentStatus(Enum):
    """Content status enumeration"""
    DRAFT = "draft"
    PUBLISHED = "published"
    FEATURED = "featured"
    ARCHIVED = "archived"
    DELETED = "deleted"

class PaymentStatus(Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

class DistributionChannel(Enum):
    """Distribution channel enumeration"""
    WEBSITE = "website"
    MOBILE_APP = "mobile_app"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    API = "api"
    THIRD_PARTY = "third_party"

@dataclass
class ContentItem:
    """Content item data structure"""
    content_id: str
    title: str
    description: str
    content: str
    author_id: str
    category: str
    tags: List[str] = field(default_factory=list)
    license_type: ContentLicense = ContentLicense.FREE
    price: Decimal = Decimal('0.00')
    currency: str = "USD"
    status: ContentStatus = ContentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    download_count: int = 0
    view_count: int = 0
    rating: float = 0.0
    review_count: int = 0

@dataclass
class ContentPurchase:
    """Content purchase data structure"""
    purchase_id: str
    content_id: str
    buyer_id: str
    seller_id: str
    amount: Decimal
    currency: str
    payment_status: PaymentStatus
    purchase_date: datetime = field(default_factory=datetime.utcnow)
    license_details: Dict[str, Any] = field(default_factory=dict)
    download_url: str = ""
    expires_at: Optional[datetime] = None

@dataclass
class ContentSubscription:
    """Content subscription data structure"""
    subscription_id: str
    user_id: str
    content_id: str
    plan_type: str
    amount: Decimal
    currency: str
    billing_cycle: str  # monthly, yearly
    status: str  # active, cancelled, expired
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    auto_renew: bool = True

@dataclass
class ContentCreator:
    """Content creator data structure"""
    creator_id: str
    username: str
    email: str
    display_name: str
    bio: str
    avatar_url: str = ""
    social_links: Dict[str, str] = field(default_factory=dict)
    total_content: int = 0
    total_sales: Decimal = Decimal('0.00')
    total_earnings: Decimal = Decimal('0.00')
    rating: float = 0.0
    follower_count: int = 0
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentMarketplaceEngine:
    """
    Advanced Content Marketplace Engine
    
    Provides comprehensive content commerce and distribution capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Marketplace Engine"""
        self.config = config
        self.content_items = {}
        self.content_purchases = {}
        self.content_subscriptions = {}
        self.content_creators = {}
        self.recommendation_engine = None
        self.payment_processors = {}
        self.distribution_channels = {}
        
        # Initialize payment processors
        self._initialize_payment_processors()
        
        # Initialize distribution channels
        self._initialize_distribution_channels()
        
        # Initialize recommendation engine
        self._initialize_recommendation_engine()
        
        logger.info("Content Marketplace Engine initialized successfully")
    
    def _initialize_payment_processors(self):
        """Initialize payment processors"""
        try:
            # Stripe
            if self.config.get("stripe_secret_key"):
                stripe.api_key = self.config["stripe_secret_key"]
                self.payment_processors["stripe"] = stripe
            
            # PayPal
            if self.config.get("paypal_client_id"):
                self.payment_processors["paypal"] = Payment
            
            logger.info("Payment processors initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing payment processors: {e}")
    
    def _initialize_distribution_channels(self):
        """Initialize distribution channels"""
        try:
            # AWS S3
            if self.config.get("aws_access_key_id"):
                self.distribution_channels["s3"] = boto3.client(
                    's3',
                    aws_access_key_id=self.config["aws_access_key_id"],
                    aws_secret_access_key=self.config["aws_secret_access_key"],
                    region_name=self.config.get("aws_region", "us-east-1")
                )
            
            # Google Cloud Storage
            if self.config.get("google_cloud_credentials"):
                self.distribution_channels["gcs"] = storage.Client()
            
            logger.info("Distribution channels initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing distribution channels: {e}")
    
    def _initialize_recommendation_engine(self):
        """Initialize content recommendation engine"""
        try:
            self.recommendation_engine = {
                "vectorizer": TfidfVectorizer(max_features=1000, stop_words='english'),
                "content_vectors": {},
                "user_preferences": {},
                "collaborative_filter": {}
            }
            
            logger.info("Recommendation engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommendation engine: {e}")
    
    async def create_content_item(self, content_data: Dict[str, Any]) -> ContentItem:
        """Create a new content item in the marketplace"""
        try:
            content_id = str(uuid.uuid4())
            
            content_item = ContentItem(
                content_id=content_id,
                title=content_data["title"],
                description=content_data["description"],
                content=content_data["content"],
                author_id=content_data["author_id"],
                category=content_data["category"],
                tags=content_data.get("tags", []),
                license_type=ContentLicense(content_data.get("license_type", "free")),
                price=Decimal(str(content_data.get("price", 0))),
                currency=content_data.get("currency", "USD"),
                status=ContentStatus(content_data.get("status", "draft")),
                metadata=content_data.get("metadata", {})
            )
            
            # Store content item
            self.content_items[content_id] = content_item
            
            # Update creator stats
            await self._update_creator_stats(content_item.author_id)
            
            # Update recommendation engine
            await self._update_recommendation_engine(content_item)
            
            logger.info(f"Content item {content_id} created successfully")
            
            return content_item
            
        except Exception as e:
            logger.error(f"Error creating content item: {e}")
            raise
    
    async def _update_creator_stats(self, creator_id: str):
        """Update creator statistics"""
        try:
            if creator_id in self.content_creators:
                creator = self.content_creators[creator_id]
                creator.total_content = len([
                    item for item in self.content_items.values() 
                    if item.author_id == creator_id
                ])
                
        except Exception as e:
            logger.error(f"Error updating creator stats: {e}")
    
    async def _update_recommendation_engine(self, content_item: ContentItem):
        """Update recommendation engine with new content"""
        try:
            # Create content vector
            content_text = f"{content_item.title} {content_item.description} {' '.join(content_item.tags)}"
            vector = self.recommendation_engine["vectorizer"].fit_transform([content_text])
            self.recommendation_engine["content_vectors"][content_item.content_id] = vector
            
        except Exception as e:
            logger.error(f"Error updating recommendation engine: {e}")
    
    async def purchase_content(self, content_id: str, buyer_id: str, 
                             payment_method: str = "stripe") -> ContentPurchase:
        """Purchase content item"""
        try:
            if content_id not in self.content_items:
                raise ValueError(f"Content item {content_id} not found")
            
            content_item = self.content_items[content_id]
            
            if content_item.license_type == ContentLicense.FREE:
                # Free content - no payment required
                purchase = ContentPurchase(
                    purchase_id=str(uuid.uuid4()),
                    content_id=content_id,
                    buyer_id=buyer_id,
                    seller_id=content_item.author_id,
                    amount=Decimal('0.00'),
                    currency=content_item.currency,
                    payment_status=PaymentStatus.COMPLETED,
                    download_url=await self._generate_download_url(content_id)
                )
            else:
                # Paid content - process payment
                purchase = await self._process_payment(
                    content_id, buyer_id, content_item, payment_method
                )
            
            # Store purchase
            self.content_purchases[purchase.purchase_id] = purchase
            
            # Update content stats
            content_item.download_count += 1
            
            # Update creator earnings
            await self._update_creator_earnings(content_item.author_id, purchase.amount)
            
            logger.info(f"Content {content_id} purchased by {buyer_id}")
            
            return purchase
            
        except Exception as e:
            logger.error(f"Error purchasing content: {e}")
            raise
    
    async def _process_payment(self, content_id: str, buyer_id: str, 
                             content_item: ContentItem, payment_method: str) -> ContentPurchase:
        """Process payment for content purchase"""
        try:
            purchase_id = str(uuid.uuid4())
            
            if payment_method == "stripe" and "stripe" in self.payment_processors:
                # Process Stripe payment
                payment_intent = self.payment_processors["stripe"].PaymentIntent.create(
                    amount=int(content_item.price * 100),  # Convert to cents
                    currency=content_item.currency.lower(),
                    metadata={
                        "content_id": content_id,
                        "buyer_id": buyer_id,
                        "purchase_id": purchase_id
                    }
                )
                
                # For demo purposes, assume payment is successful
                payment_status = PaymentStatus.COMPLETED
                
            elif payment_method == "paypal" and "paypal" in self.payment_processors:
                # Process PayPal payment
                payment_status = PaymentStatus.COMPLETED  # Simplified for demo
                
            else:
                raise ValueError(f"Unsupported payment method: {payment_method}")
            
            return ContentPurchase(
                purchase_id=purchase_id,
                content_id=content_id,
                buyer_id=buyer_id,
                seller_id=content_item.author_id,
                amount=content_item.price,
                currency=content_item.currency,
                payment_status=payment_status,
                download_url=await self._generate_download_url(content_id)
            )
            
        except Exception as e:
            logger.error(f"Error processing payment: {e}")
            raise
    
    async def _generate_download_url(self, content_id: str) -> str:
        """Generate secure download URL for content"""
        try:
            # Generate secure download URL
            download_token = hashlib.sha256(f"{content_id}_{datetime.utcnow()}".encode()).hexdigest()
            download_url = f"/api/v1/content/{content_id}/download?token={download_token}"
            
            return download_url
            
        except Exception as e:
            logger.error(f"Error generating download URL: {e}")
            return ""
    
    async def _update_creator_earnings(self, creator_id: str, amount: Decimal):
        """Update creator earnings"""
        try:
            if creator_id in self.content_creators:
                creator = self.content_creators[creator_id]
                creator.total_earnings += amount
                creator.total_sales += amount
                
        except Exception as e:
            logger.error(f"Error updating creator earnings: {e}")
    
    async def create_subscription(self, user_id: str, content_id: str, 
                                plan_type: str, billing_cycle: str) -> ContentSubscription:
        """Create content subscription"""
        try:
            if content_id not in self.content_items:
                raise ValueError(f"Content item {content_id} not found")
            
            content_item = self.content_items[content_id]
            
            # Calculate subscription amount
            if plan_type == "basic":
                amount = Decimal('9.99')
            elif plan_type == "premium":
                amount = Decimal('19.99')
            elif plan_type == "enterprise":
                amount = Decimal('49.99')
            else:
                amount = content_item.price
            
            # Calculate end date
            if billing_cycle == "monthly":
                end_date = datetime.utcnow() + timedelta(days=30)
            elif billing_cycle == "yearly":
                end_date = datetime.utcnow() + timedelta(days=365)
            else:
                end_date = None
            
            subscription = ContentSubscription(
                subscription_id=str(uuid.uuid4()),
                user_id=user_id,
                content_id=content_id,
                plan_type=plan_type,
                amount=amount,
                currency=content_item.currency,
                billing_cycle=billing_cycle,
                status="active",
                end_date=end_date
            )
            
            # Store subscription
            self.content_subscriptions[subscription.subscription_id] = subscription
            
            logger.info(f"Subscription created for user {user_id} and content {content_id}")
            
            return subscription
            
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            raise
    
    async def get_content_recommendations(self, user_id: str, 
                                        limit: int = 10) -> List[ContentItem]:
        """Get personalized content recommendations for user"""
        try:
            recommendations = []
            
            # Get user preferences
            user_preferences = self.recommendation_engine["user_preferences"].get(user_id, {})
            
            # Get user's purchase history
            user_purchases = [
                purchase for purchase in self.content_purchases.values()
                if purchase.buyer_id == user_id
            ]
            
            # Get user's viewed content
            user_viewed = user_preferences.get("viewed_content", [])
            
            # Content-based filtering
            if user_viewed:
                # Find similar content based on viewed items
                similar_content = await self._find_similar_content(user_viewed, limit)
                recommendations.extend(similar_content)
            
            # Collaborative filtering
            collaborative_recs = await self._get_collaborative_recommendations(user_id, limit)
            recommendations.extend(collaborative_recs)
            
            # Popular content fallback
            if len(recommendations) < limit:
                popular_content = await self._get_popular_content(limit - len(recommendations))
                recommendations.extend(popular_content)
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_recommendations = []
            for item in recommendations:
                if item.content_id not in seen_ids:
                    seen_ids.add(item.content_id)
                    unique_recommendations.append(item)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            return []
    
    async def _find_similar_content(self, viewed_content_ids: List[str], 
                                  limit: int) -> List[ContentItem]:
        """Find content similar to viewed items"""
        try:
            similar_content = []
            
            for viewed_id in viewed_content_ids:
                if viewed_id in self.recommendation_engine["content_vectors"]:
                    viewed_vector = self.recommendation_engine["content_vectors"][viewed_id]
                    
                    # Calculate similarity with all other content
                    similarities = {}
                    for content_id, vector in self.recommendation_engine["content_vectors"].items():
                        if content_id != viewed_id:
                            similarity = cosine_similarity(viewed_vector, vector)[0][0]
                            similarities[content_id] = similarity
                    
                    # Get top similar content
                    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    for content_id, similarity in sorted_similarities[:limit]:
                        if content_id in self.content_items:
                            similar_content.append(self.content_items[content_id])
            
            return similar_content[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    async def _get_collaborative_recommendations(self, user_id: str, 
                                               limit: int) -> List[ContentItem]:
        """Get collaborative filtering recommendations"""
        try:
            # Find users with similar preferences
            user_purchases = [
                purchase.content_id for purchase in self.content_purchases.values()
                if purchase.buyer_id == user_id
            ]
            
            # Find other users who purchased similar content
            similar_users = []
            for purchase in self.content_purchases.values():
                if purchase.buyer_id != user_id and purchase.content_id in user_purchases:
                    similar_users.append(purchase.buyer_id)
            
            # Get content purchased by similar users
            recommendations = []
            for similar_user in similar_users:
                similar_user_purchases = [
                    purchase.content_id for purchase in self.content_purchases.values()
                    if purchase.buyer_id == similar_user
                ]
                
                for content_id in similar_user_purchases:
                    if content_id not in user_purchases and content_id in self.content_items:
                        recommendations.append(self.content_items[content_id])
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []
    
    async def _get_popular_content(self, limit: int) -> List[ContentItem]:
        """Get popular content based on downloads and ratings"""
        try:
            # Sort content by popularity metrics
            popular_content = sorted(
                self.content_items.values(),
                key=lambda x: (x.download_count * 0.7 + x.rating * 0.3),
                reverse=True
            )
            
            return popular_content[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular content: {e}")
            return []
    
    async def search_content(self, query: str, filters: Dict[str, Any] = None, 
                           limit: int = 20) -> List[ContentItem]:
        """Search content with filters"""
        try:
            results = []
            query_lower = query.lower()
            
            for content_item in self.content_items.values():
                # Text search
                if (query_lower in content_item.title.lower() or 
                    query_lower in content_item.description.lower() or
                    any(query_lower in tag.lower() for tag in content_item.tags)):
                    
                    # Apply filters
                    if filters:
                        if "category" in filters and content_item.category != filters["category"]:
                            continue
                        if "license_type" in filters and content_item.license_type.value != filters["license_type"]:
                            continue
                        if "min_price" in filters and content_item.price < Decimal(str(filters["min_price"])):
                            continue
                        if "max_price" in filters and content_item.price > Decimal(str(filters["max_price"])):
                            continue
                        if "min_rating" in filters and content_item.rating < filters["min_rating"]:
                            continue
                    
                    results.append(content_item)
            
            # Sort by relevance (simplified)
            results.sort(key=lambda x: x.download_count, reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    async def get_marketplace_analytics(self, time_period: str = "30d") -> Dict[str, Any]:
        """Get marketplace analytics and insights"""
        try:
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Filter data by time period
            recent_purchases = [
                purchase for purchase in self.content_purchases.values()
                if start_date <= purchase.purchase_date <= end_date
            ]
            
            # Calculate metrics
            total_revenue = sum(purchase.amount for purchase in recent_purchases)
            total_purchases = len(recent_purchases)
            unique_buyers = len(set(purchase.buyer_id for purchase in recent_purchases))
            unique_sellers = len(set(purchase.seller_id for purchase in recent_purchases))
            
            # Top selling content
            content_sales = defaultdict(int)
            for purchase in recent_purchases:
                content_sales[purchase.content_id] += 1
            
            top_selling_content = sorted(content_sales.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Top earning creators
            creator_earnings = defaultdict(Decimal)
            for purchase in recent_purchases:
                creator_earnings[purchase.seller_id] += purchase.amount
            
            top_earning_creators = sorted(creator_earnings.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Category breakdown
            category_stats = defaultdict(lambda: {"count": 0, "revenue": Decimal('0.00')})
            for content_item in self.content_items.values():
                category_stats[content_item.category]["count"] += 1
                for purchase in recent_purchases:
                    if purchase.content_id == content_item.content_id:
                        category_stats[content_item.category]["revenue"] += purchase.amount
            
            return {
                "time_period": time_period,
                "total_revenue": float(total_revenue),
                "total_purchases": total_purchases,
                "unique_buyers": unique_buyers,
                "unique_sellers": unique_sellers,
                "average_purchase_value": float(total_revenue / total_purchases) if total_purchases > 0 else 0,
                "top_selling_content": [
                    {
                        "content_id": content_id,
                        "title": self.content_items[content_id].title,
                        "sales_count": sales_count
                    }
                    for content_id, sales_count in top_selling_content
                ],
                "top_earning_creators": [
                    {
                        "creator_id": creator_id,
                        "username": self.content_creators.get(creator_id, {}).get("username", "Unknown"),
                        "earnings": float(earnings)
                    }
                    for creator_id, earnings in top_earning_creators
                ],
                "category_breakdown": {
                    category: {
                        "content_count": stats["count"],
                        "revenue": float(stats["revenue"])
                    }
                    for category, stats in category_stats.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting marketplace analytics: {e}")
            return {"error": str(e)}
    
    async def register_creator(self, creator_data: Dict[str, Any]) -> ContentCreator:
        """Register a new content creator"""
        try:
            creator_id = str(uuid.uuid4())
            
            creator = ContentCreator(
                creator_id=creator_id,
                username=creator_data["username"],
                email=creator_data["email"],
                display_name=creator_data["display_name"],
                bio=creator_data.get("bio", ""),
                avatar_url=creator_data.get("avatar_url", ""),
                social_links=creator_data.get("social_links", {}),
                verified=creator_data.get("verified", False)
            )
            
            # Store creator
            self.content_creators[creator_id] = creator
            
            logger.info(f"Creator {creator_id} registered successfully")
            
            return creator
            
        except Exception as e:
            logger.error(f"Error registering creator: {e}")
            raise
    
    async def distribute_content(self, content_id: str, 
                               channels: List[DistributionChannel]) -> Dict[str, str]:
        """Distribute content to multiple channels"""
        try:
            if content_id not in self.content_items:
                raise ValueError(f"Content item {content_id} not found")
            
            content_item = self.content_items[content_id]
            distribution_results = {}
            
            for channel in channels:
                try:
                    if channel == DistributionChannel.WEBSITE:
                        # Website distribution
                        distribution_results["website"] = f"/content/{content_id}"
                    
                    elif channel == DistributionChannel.MOBILE_APP:
                        # Mobile app distribution
                        distribution_results["mobile_app"] = f"app://content/{content_id}"
                    
                    elif channel == DistributionChannel.SOCIAL_MEDIA:
                        # Social media distribution
                        distribution_results["social_media"] = await self._distribute_to_social_media(content_item)
                    
                    elif channel == DistributionChannel.EMAIL:
                        # Email distribution
                        distribution_results["email"] = await self._distribute_to_email(content_item)
                    
                    elif channel == DistributionChannel.API:
                        # API distribution
                        distribution_results["api"] = f"/api/v1/content/{content_id}"
                    
                    elif channel == DistributionChannel.THIRD_PARTY:
                        # Third-party distribution
                        distribution_results["third_party"] = await self._distribute_to_third_party(content_item)
                    
                except Exception as e:
                    logger.error(f"Error distributing to {channel.value}: {e}")
                    distribution_results[channel.value] = f"Error: {str(e)}"
            
            return distribution_results
            
        except Exception as e:
            logger.error(f"Error distributing content: {e}")
            raise
    
    async def _distribute_to_social_media(self, content_item: ContentItem) -> str:
        """Distribute content to social media platforms"""
        try:
            # Simplified social media distribution
            return f"Posted to social media: {content_item.title}"
            
        except Exception as e:
            logger.error(f"Error distributing to social media: {e}")
            return f"Error: {str(e)}"
    
    async def _distribute_to_email(self, content_item: ContentItem) -> str:
        """Distribute content via email"""
        try:
            # Simplified email distribution
            return f"Email sent: {content_item.title}"
            
        except Exception as e:
            logger.error(f"Error distributing via email: {e}")
            return f"Error: {str(e)}"
    
    async def _distribute_to_third_party(self, content_item: ContentItem) -> str:
        """Distribute content to third-party platforms"""
        try:
            # Simplified third-party distribution
            return f"Distributed to third-party: {content_item.title}"
            
        except Exception as e:
            logger.error(f"Error distributing to third-party: {e}")
            return f"Error: {str(e)}"

# Example usage and testing
async def main():
    """Example usage of the Content Marketplace Engine"""
    try:
        # Initialize engine
        config = {
            "stripe_secret_key": "sk_test_your_stripe_key",
            "paypal_client_id": "your_paypal_client_id",
            "aws_access_key_id": "your_aws_access_key",
            "aws_secret_access_key": "your_aws_secret_key"
        }
        
        engine = ContentMarketplaceEngine(config)
        
        # Register a creator
        print("Registering content creator...")
        creator = await engine.register_creator({
            "username": "ai_writer",
            "email": "writer@example.com",
            "display_name": "AI Content Writer",
            "bio": "Professional AI content creator",
            "verified": True
        })
        
        # Create content item
        print("Creating content item...")
        content_item = await engine.create_content_item({
            "title": "Advanced AI Techniques Guide",
            "description": "Comprehensive guide to advanced AI techniques",
            "content": "This is a comprehensive guide covering...",
            "author_id": creator.creator_id,
            "category": "Technology",
            "tags": ["AI", "Machine Learning", "Guide"],
            "license_type": "premium",
            "price": 29.99
        })
        
        # Purchase content
        print("Purchasing content...")
        purchase = await engine.purchase_content(content_item.content_id, "buyer_123")
        print(f"Purchase completed: {purchase.purchase_id}")
        
        # Get recommendations
        print("Getting content recommendations...")
        recommendations = await engine.get_content_recommendations("buyer_123", 5)
        print(f"Found {len(recommendations)} recommendations")
        
        # Search content
        print("Searching content...")
        search_results = await engine.search_content("AI", {"category": "Technology"}, 10)
        print(f"Found {len(search_results)} search results")
        
        # Get marketplace analytics
        print("Getting marketplace analytics...")
        analytics = await engine.get_marketplace_analytics("30d")
        print(f"Total revenue: ${analytics['total_revenue']:.2f}")
        print(f"Total purchases: {analytics['total_purchases']}")
        
        # Distribute content
        print("Distributing content...")
        distribution = await engine.distribute_content(
            content_item.content_id, 
            [DistributionChannel.WEBSITE, DistributionChannel.SOCIAL_MEDIA]
        )
        print(f"Distribution results: {distribution}")
        
        print("\nContent Marketplace Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























