from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import copy
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Engagement Metrics Value Object
==============================

Value object for tracking post engagement metrics and analytics.
"""



@dataclass
class EngagementMetrics:
    """
    Engagement metrics value object for tracking post performance.
    
    This value object encapsulates all engagement-related metrics
    and provides methods for calculating engagement scores and trends.
    """
    
    # Core metrics
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    clicks: int = 0
    
    # Advanced metrics
    reactions: Dict[str, int] = field(default_factory=dict)  # Like, Love, Celebrate, etc.
    comment_replies: int = 0
    profile_visits: int = 0
    follows_generated: int = 0
    
    # Timing
    first_engagement_at: Optional[datetime] = None
    peak_engagement_at: Optional[datetime] = None
    engagement_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    engagement_rate: float = 0.0
    viral_coefficient: float = 0.0
    reach_ratio: float = 0.0
    
    def update(self, likes: int = 0, comments: int = 0, shares: int = 0, 
               saves: int = 0, clicks: int = 0) -> None:
        """Update basic engagement metrics."""
        self.likes += likes
        self.comments += comments
        self.shares += shares
        self.saves += saves
        self.clicks += clicks
        
        # Update first engagement timestamp
        if not self.first_engagement_at and (likes > 0 or comments > 0 or shares > 0):
            self.first_engagement_at = datetime.utcnow()
        
        # Record engagement history
        self._record_engagement_history()
    
    def add_reaction(self, reaction_type: str, count: int = 1) -> None:
        """Add reaction count."""
        if reaction_type not in self.reactions:
            self.reactions[reaction_type] = 0
        self.reactions[reaction_type] += count
    
    def increment_profile_visits(self, count: int = 1) -> None:
        """Increment profile visits."""
        self.profile_visits += count
    
    def increment_follows_generated(self, count: int = 1) -> None:
        """Increment follows generated."""
        self.follows_generated += count
    
    def increment_comment_replies(self, count: int = 1) -> None:
        """Increment comment replies."""
        self.comment_replies += count
    
    def calculate_engagement_rate(self, impressions: int) -> float:
        """Calculate engagement rate based on impressions."""
        if impressions > 0:
            total_engagement = self.likes + self.comments + self.shares + self.saves
            self.engagement_rate = (total_engagement / impressions) * 100
        return self.engagement_rate
    
    def calculate_viral_coefficient(self, followers: int) -> float:
        """Calculate viral coefficient."""
        if followers > 0:
            self.viral_coefficient = self.shares / followers
        return self.viral_coefficient
    
    def calculate_reach_ratio(self, reach: int, impressions: int) -> float:
        """Calculate reach ratio."""
        if impressions > 0:
            self.reach_ratio = (reach / impressions) * 100
        return self.reach_ratio
    
    def get_total_engagement(self) -> int:
        """Get total engagement count."""
        return self.likes + self.comments + self.shares + self.saves
    
    def get_engagement_score(self) -> float:
        """Calculate overall engagement score."""
        score = 0.0
        
        # Weighted engagement calculation
        score += self.likes * 1.0
        score += self.comments * 3.0  # Comments are worth more
        score += self.shares * 5.0    # Shares are worth the most
        score += self.saves * 2.0     # Saves indicate high value
        score += self.clicks * 0.5    # Clicks are worth less
        
        return score
    
    def get_engagement_trend(self, hours: int = 24) -> Dict[str, float]:
        """Calculate engagement trend over time."""
        if not self.engagement_history:
            return {"trend": "no_data", "growth_rate": 0.0}
        
        # Filter history for the specified time period
        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        recent_engagement = [
            entry for entry in self.engagement_history 
            if entry["timestamp"] > cutoff_time
        ]
        
        if len(recent_engagement) < 2:
            return {"trend": "insufficient_data", "growth_rate": 0.0}
        
        # Calculate growth rate
        first_total = recent_engagement[0]["total_engagement"]
        last_total = recent_engagement[-1]["total_engagement"]
        
        match first_total:
    case 0:
            growth_rate = 100.0 if last_total > 0 else 0.0
        else:
            growth_rate = ((last_total - first_total) / first_total) * 100
        
        # Determine trend
        if growth_rate > 10:
            trend = "growing"
        elif growth_rate < -10:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "growth_rate": growth_rate,
            "recent_engagement": len(recent_engagement)
        }
    
    def get_best_performing_hour(self) -> Optional[int]:
        """Get the hour with the highest engagement."""
        if not self.engagement_history:
            return None
        
        hourly_engagement = {}
        for entry in self.engagement_history:
            hour = entry["timestamp"].hour
            if hour not in hourly_engagement:
                hourly_engagement[hour] = 0
            hourly_engagement[hour] += entry["total_engagement"]
        
        if not hourly_engagement:
            return None
        
        return max(hourly_engagement, key=hourly_engagement.get)
    
    def get_engagement_velocity(self) -> float:
        """Calculate engagement velocity (engagement per hour)."""
        if not self.first_engagement_at:
            return 0.0
        
        hours_since_first = (datetime.utcnow() - self.first_engagement_at).total_seconds() / 3600
        if hours_since_first <= 0:
            return 0.0
        
        return self.get_total_engagement() / hours_since_first
    
    def _record_engagement_history(self) -> None:
        """Record current engagement state in history."""
        entry = {
            "timestamp": datetime.utcnow(),
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "saves": self.saves,
            "clicks": self.clicks,
            "total_engagement": self.get_total_engagement(),
            "engagement_score": self.get_engagement_score()
        }
        
        self.engagement_history.append(entry)
        
        # Keep only last 100 entries to prevent memory bloat
        if len(self.engagement_history) > 100:
            self.engagement_history = self.engagement_history[-100:]
    
    def copy(self) -> 'EngagementMetrics':
        """Create a copy of the engagement metrics."""
        return copy(self)
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.likes = 0
        self.comments = 0
        self.shares = 0
        self.saves = 0
        self.clicks = 0
        self.reactions.clear()
        self.comment_replies = 0
        self.profile_visits = 0
        self.follows_generated = 0
        self.first_engagement_at = None
        self.peak_engagement_at = None
        self.engagement_history.clear()
        self.engagement_rate = 0.0
        self.viral_coefficient = 0.0
        self.reach_ratio = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "saves": self.saves,
            "clicks": self.clicks,
            "reactions": self.reactions,
            "comment_replies": self.comment_replies,
            "profile_visits": self.profile_visits,
            "follows_generated": self.follows_generated,
            "first_engagement_at": self.first_engagement_at.isoformat() if self.first_engagement_at else None,
            "peak_engagement_at": self.peak_engagement_at.isoformat() if self.peak_engagement_at else None,
            "engagement_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "likes": entry["likes"],
                    "comments": entry["comments"],
                    "shares": entry["shares"],
                    "saves": entry["saves"],
                    "clicks": entry["clicks"],
                    "total_engagement": entry["total_engagement"],
                    "engagement_score": entry["engagement_score"]
                }
                for entry in self.engagement_history
            ],
            "engagement_rate": self.engagement_rate,
            "viral_coefficient": self.viral_coefficient,
            "reach_ratio": self.reach_ratio,
            "total_engagement": self.get_total_engagement(),
            "engagement_score": self.get_engagement_score(),
            "engagement_velocity": self.get_engagement_velocity(),
            "best_performing_hour": self.get_best_performing_hour(),
            "engagement_trend": self.get_engagement_trend()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EngagementMetrics':
        """Create from dictionary."""
        metrics = cls(
            likes=data.get("likes", 0),
            comments=data.get("comments", 0),
            shares=data.get("shares", 0),
            saves=data.get("saves", 0),
            clicks=data.get("clicks", 0),
            reactions=data.get("reactions", {}),
            comment_replies=data.get("comment_replies", 0),
            profile_visits=data.get("profile_visits", 0),
            follows_generated=data.get("follows_generated", 0),
            engagement_rate=data.get("engagement_rate", 0.0),
            viral_coefficient=data.get("viral_coefficient", 0.0),
            reach_ratio=data.get("reach_ratio", 0.0)
        )
        
        # Parse timestamps
        if data.get("first_engagement_at"):
            metrics.first_engagement_at = datetime.fromisoformat(data["first_engagement_at"])
        
        if data.get("peak_engagement_at"):
            metrics.peak_engagement_at = datetime.fromisoformat(data["peak_engagement_at"])
        
        # Parse engagement history
        for entry_data in data.get("engagement_history", []):
            entry = {
                "timestamp": datetime.fromisoformat(entry_data["timestamp"]),
                "likes": entry_data["likes"],
                "comments": entry_data["comments"],
                "shares": entry_data["shares"],
                "saves": entry_data["saves"],
                "clicks": entry_data["clicks"],
                "total_engagement": entry_data["total_engagement"],
                "engagement_score": entry_data["engagement_score"]
            }
            metrics.engagement_history.append(entry)
        
        return metrics
    
    def __str__(self) -> str:
        return f"EngagementMetrics(likes={self.likes}, comments={self.comments}, shares={self.shares}, score={self.get_engagement_score():.1f})"
    
    def __repr__(self) -> str:
        return f"EngagementMetrics(likes={self.likes}, comments={self.comments}, shares={self.shares})" 