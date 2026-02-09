"""
Feedback System
===============

User feedback collection and continuous improvement system.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserFeedback:
    """User feedback record."""
    operation_id: str
    timestamp: float
    satisfaction: float  # 0.0-1.0
    quality_rating: float  # 0.0-1.0
    speed_rating: float  # 0.0-1.0
    comments: Optional[str] = None
    issues: List[str] = None
    improvements: List[str] = None


class FeedbackSystem:
    """
    Feedback collection and analysis system.
    
    Features:
    - User feedback collection
    - Satisfaction tracking
    - Issue identification
    - Improvement suggestions
    - Continuous learning
    """
    
    def __init__(self, feedback_file: Optional[str] = None):
        """
        Initialize feedback system.
        
        Args:
            feedback_file: File to store feedback
        """
        self.feedback_file = feedback_file or "./feedback.json"
        self.feedback_path = Path(self.feedback_file)
        
        # Feedback storage
        self.feedback_records: List[UserFeedback] = []
        
        # Aggregated data
        self.satisfaction_by_method: Dict[str, List[float]] = defaultdict(list)
        self.issues_by_type: Dict[str, int] = defaultdict(int)
        
        # Load existing feedback
        self._load_feedback()
        
        logger.info(f"FeedbackSystem initialized with {len(self.feedback_records)} records")
    
    def submit_feedback(
        self,
        operation_id: str,
        satisfaction: float,
        quality_rating: float,
        speed_rating: float,
        comments: Optional[str] = None,
        issues: Optional[List[str]] = None,
        improvements: Optional[List[str]] = None
    ) -> None:
        """
        Submit user feedback.
        
        Args:
            operation_id: Operation ID
            satisfaction: Overall satisfaction (0.0-1.0)
            quality_rating: Quality rating (0.0-1.0)
            speed_rating: Speed rating (0.0-1.0)
            comments: Optional comments
            issues: List of issues encountered
            improvements: Suggested improvements
        """
        feedback = UserFeedback(
            operation_id=operation_id,
            timestamp=time.time(),
            satisfaction=satisfaction,
            quality_rating=quality_rating,
            speed_rating=speed_rating,
            comments=comments,
            issues=issues or [],
            improvements=improvements or []
        )
        
        self.feedback_records.append(feedback)
        
        # Update aggregates
        # Note: Would need operation details to track by method
        # This would be integrated with metrics system
        
        # Track issues
        for issue in feedback.issues:
            self.issues_by_type[issue] += 1
        
        # Save periodically
        if len(self.feedback_records) % 10 == 0:
            self._save_feedback()
        
        logger.info(f"Feedback submitted for operation {operation_id}")
    
    def get_satisfaction_stats(self) -> Dict[str, Any]:
        """Get satisfaction statistics."""
        if not self.feedback_records:
            return {
                "total_feedback": 0,
                "avg_satisfaction": 0.0,
                "avg_quality": 0.0,
                "avg_speed": 0.0,
            }
        
        return {
            "total_feedback": len(self.feedback_records),
            "avg_satisfaction": sum(f.satisfaction for f in self.feedback_records) / len(self.feedback_records),
            "avg_quality": sum(f.quality_rating for f in self.feedback_records) / len(self.feedback_records),
            "avg_speed": sum(f.speed_rating for f in self.feedback_records) / len(self.feedback_records),
            "satisfaction_distribution": self._calculate_distribution([f.satisfaction for f in self.feedback_records]),
        }
    
    def get_common_issues(self, top_n: int = 5) -> List[tuple]:
        """Get most common issues."""
        sorted_issues = sorted(
            self.issues_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_issues[:top_n]
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get improvement suggestions based on feedback."""
        suggestions = []
        
        stats = self.get_satisfaction_stats()
        
        if stats["avg_quality"] < 0.7:
            suggestions.append("Improve upscaling quality - consider using higher quality methods")
        
        if stats["avg_speed"] < 0.6:
            suggestions.append("Optimize processing speed - consider caching or faster methods")
        
        if stats["avg_satisfaction"] < 0.7:
            suggestions.append("Overall satisfaction is low - review common issues")
        
        # Check common issues
        common_issues = self.get_common_issues(3)
        for issue, count in common_issues:
            suggestions.append(f"Address frequent issue: {issue} (reported {count} times)")
        
        return suggestions
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate value distribution."""
        distribution = {
            "excellent": 0,  # >= 0.9
            "good": 0,      # 0.7-0.9
            "fair": 0,      # 0.5-0.7
            "poor": 0       # < 0.5
        }
        
        for value in values:
            if value >= 0.9:
                distribution["excellent"] += 1
            elif value >= 0.7:
                distribution["good"] += 1
            elif value >= 0.5:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _load_feedback(self) -> None:
        """Load feedback from file."""
        if not self.feedback_path.exists():
            return
        
        try:
            with open(self.feedback_path, 'r') as f:
                data = json.load(f)
            
            self.feedback_records = [
                UserFeedback(**feedback_data)
                for feedback_data in data.get("feedback", [])
            ]
            
            # Rebuild aggregates
            for feedback in self.feedback_records:
                for issue in feedback.issues:
                    self.issues_by_type[issue] += 1
            
            logger.info(f"Loaded {len(self.feedback_records)} feedback records")
            
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
    
    def _save_feedback(self) -> None:
        """Save feedback to file."""
        try:
            data = {
                "feedback": [asdict(f) for f in self.feedback_records],
                "last_updated": time.time()
            }
            
            with open(self.feedback_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.feedback_records)} feedback records")
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def export_feedback(self, file_path: str) -> None:
        """Export feedback to file."""
        try:
            data = {
                "feedback": [asdict(f) for f in self.feedback_records],
                "statistics": self.get_satisfaction_stats(),
                "common_issues": self.get_common_issues(),
                "suggestions": self.get_improvement_suggestions(),
                "export_timestamp": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Feedback exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")


