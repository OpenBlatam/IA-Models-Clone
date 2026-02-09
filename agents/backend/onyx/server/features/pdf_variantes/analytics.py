"""
PDF Variantes Analytics
====================

Analytics and reporting features.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Usage statistics."""
    total_uploads: int = 0
    total_downloads: int = 0
    total_variants_generated: int = 0
    total_brainstorms: int = 0
    total_topics_extracted: int = 0
    average_processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_uploads": self.total_uploads,
            "total_downloads": self.total_downloads,
            "total_variants_generated": self.total_variants_generated,
            "total_brainstorms": self.total_brainstorms,
            "total_topics_extracted": self.total_topics_extracted,
            "average_processing_time_ms": self.average_processing_time_ms
        }


@dataclass
class UserActivity:
    """User activity tracking."""
    user_id: str
    file_id: str
    action: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "file_id": self.file_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }


class AnalyticsEngine:
    """Analytics and reporting engine."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.analytics_dir = self.upload_dir / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = UsageStats()
        self.user_activities: Dict[str, List[UserActivity]] = defaultdict(list)
        
        logger.info("Initialized Analytics Engine")
    
    def track_event(
        self,
        user_id: str,
        action: str,
        file_id: Optional[str] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track user event."""
        activity = UserActivity(
            user_id=user_id,
            file_id=file_id or "unknown",
            action=action,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        self.user_activities[user_id].append(activity)
        
        # Update global stats
        if action == "upload":
            self.stats.total_uploads += 1
        elif action == "download":
            self.stats.total_downloads += 1
        elif action == "generate_variant":
            self.stats.total_variants_generated += 1
        elif action == "brainstorm":
            self.stats.total_brainstorms += 1
        elif action == "extract_topics":
            self.stats.total_topics_extracted += 1
        
        logger.info(f"Tracked event: {action} by {user_id}")
    
    def get_stats(self) -> UsageStats:
        """Get usage statistics."""
        return self.stats
    
    def get_user_activity(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[UserActivity]:
        """Get user activity."""
        activities = self.user_activities.get(user_id, [])
        return activities[-limit:]
    
    def get_popular_features(self) -> Dict[str, int]:
        """Get popular features by usage."""
        feature_counts = defaultdict(int)
        
        for activities in self.user_activities.values():
            for activity in activities:
                feature_counts[activity.action] += 1
        
        return dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))
    
    async def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate analytics report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "usage_stats": self.stats.to_dict(),
            "popular_features": self.get_popular_features(),
            "total_users": len(self.user_activities),
            "user_breakdown": {}
        }
        
        logger.info("Generated analytics report")
        
        return report
    
    async def export_data(self, output_file: Optional[Path] = None) -> bool:
        """Export analytics data."""
        output_file = output_file or self.analytics_dir / "analytics_export.json"
        
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "stats": self.stats.to_dict(),
            "user_activities": {
                user_id: [activity.to_dict() for activity in activities[-100:]]
                for user_id, activities in self.user_activities.items()
            }
        }
        
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported analytics to {output_file}")
        
        return True







