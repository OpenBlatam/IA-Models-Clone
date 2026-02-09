from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO Score Value Object
Domain-Driven Design with grading and analysis logic
"""



class ScoreGrade(Enum):
    """SEO Score Grades"""
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D_PLUS = "D+"
    D = "D"
    D_MINUS = "D-"
    F = "F"


@dataclass(frozen=True)
class SEOScore:
    """
    SEO Score value object with grading and analysis logic
    
    This value object encapsulates SEO score calculation, grading,
    and performance analysis.
    """
    
    value: int
    
    def __post_init__(self) -> Any:
        """Validate SEO score after initialization"""
        if not isinstance(self.value, int):
            raise ValueError("Score must be an integer")
        
        if self.value < 0 or self.value > 100:
            raise ValueError("Score must be between 0 and 100")
    
    def get_grade(self) -> ScoreGrade:
        """
        Get letter grade for SEO score
        
        Returns:
            ScoreGrade: Letter grade
        """
        if self.value >= 95:
            return ScoreGrade.A_PLUS
        elif self.value >= 90:
            return ScoreGrade.A
        elif self.value >= 85:
            return ScoreGrade.A_MINUS
        elif self.value >= 80:
            return ScoreGrade.B_PLUS
        elif self.value >= 75:
            return ScoreGrade.B
        elif self.value >= 70:
            return ScoreGrade.B_MINUS
        elif self.value >= 65:
            return ScoreGrade.C_PLUS
        elif self.value >= 60:
            return ScoreGrade.C
        elif self.value >= 55:
            return ScoreGrade.C_MINUS
        elif self.value >= 50:
            return ScoreGrade.D_PLUS
        elif self.value >= 45:
            return ScoreGrade.D
        elif self.value >= 40:
            return ScoreGrade.D_MINUS
        else:
            return ScoreGrade.F
    
    def get_percentage(self) -> float:
        """
        Get score as percentage
        
        Returns:
            float: Score percentage
        """
        return float(self.value)
    
    def is_excellent(self) -> bool:
        """
        Check if score is excellent (A range)
        
        Returns:
            bool: True if excellent
        """
        return self.value >= 85
    
    def is_good(self) -> bool:
        """
        Check if score is good (B range)
        
        Returns:
            bool: True if good
        """
        return 70 <= self.value < 85
    
    def is_fair(self) -> bool:
        """
        Check if score is fair (C range)
        
        Returns:
            bool: True if fair
        """
        return 55 <= self.value < 70
    
    def is_poor(self) -> bool:
        """
        Check if score is poor (D range)
        
        Returns:
            bool: True if poor
        """
        return 40 <= self.value < 55
    
    def is_failing(self) -> bool:
        """
        Check if score is failing (F range)
        
        Returns:
            bool: True if failing
        """
        return self.value < 40
    
    def get_level(self) -> str:
        """
        Get performance level description
        
        Returns:
            str: Performance level
        """
        if self.is_excellent():
            return "Excellent"
        elif self.is_good():
            return "Good"
        elif self.is_fair():
            return "Fair"
        elif self.is_poor():
            return "Poor"
        else:
            return "Failing"
    
    def get_color(self) -> str:
        """
        Get color for score display
        
        Returns:
            str: Color code
        """
        if self.is_excellent():
            return "#28a745"  # Green
        elif self.is_good():
            return "#17a2b8"  # Blue
        elif self.is_fair():
            return "#ffc107"  # Yellow
        elif self.is_poor():
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
    
    def get_priority(self) -> str:
        """
        Get improvement priority
        
        Returns:
            str: Priority level
        """
        if self.is_excellent():
            return "Low"
        elif self.is_good():
            return "Medium"
        elif self.is_fair():
            return "High"
        elif self.is_poor():
            return "Critical"
        else:
            return "Emergency"
    
    def get_improvement_potential(self) -> int:
        """
        Get improvement potential (points to gain)
        
        Returns:
            int: Points that can be gained
        """
        return 100 - self.value
    
    def get_improvement_percentage(self) -> float:
        """
        Get improvement potential as percentage
        
        Returns:
            float: Improvement percentage
        """
        return (100 - self.value) / 100.0 * 100
    
    def get_benchmark_comparison(self, benchmark_score: int) -> Dict[str, any]:
        """
        Compare score with benchmark
        
        Args:
            benchmark_score: Benchmark score to compare against
            
        Returns:
            Dict[str, any]: Comparison results
        """
        difference = self.value - benchmark_score
        percentage_diff = (difference / benchmark_score) * 100 if benchmark_score > 0 else 0
        
        return {
            "benchmark_score": benchmark_score,
            "current_score": self.value,
            "difference": difference,
            "percentage_difference": percentage_diff,
            "is_better": difference > 0,
            "is_worse": difference < 0,
            "is_equal": difference == 0
        }
    
    def get_competitor_analysis(self, competitor_scores: List[int]) -> Dict[str, any]:
        """
        Analyze score against competitors
        
        Args:
            competitor_scores: List of competitor scores
            
        Returns:
            Dict[str, any]: Competitor analysis
        """
        if not competitor_scores:
            return {
                "rank": 1,
                "percentile": 100,
                "above_average": True,
                "competitor_count": 0
            }
        
        sorted_scores = sorted(competitor_scores + [self.value], reverse=True)
        rank = sorted_scores.index(self.value) + 1
        percentile = (len(sorted_scores) - rank + 1) / len(sorted_scores) * 100
        
        avg_competitor_score = sum(competitor_scores) / len(competitor_scores)
        above_average = self.value > avg_competitor_score
        
        return {
            "rank": rank,
            "percentile": percentile,
            "above_average": above_average,
            "competitor_count": len(competitor_scores),
            "average_competitor_score": avg_competitor_score,
            "score_difference_from_average": self.value - avg_competitor_score
        }
    
    def get_trend_analysis(self, historical_scores: List[int]) -> Dict[str, any]:
        """
        Analyze score trends over time
        
        Args:
            historical_scores: List of historical scores
            
        Returns:
            Dict[str, any]: Trend analysis
        """
        if len(historical_scores) < 2:
            return {
                "trend": "insufficient_data",
                "change": 0,
                "change_percentage": 0,
                "direction": "stable"
            }
        
        # Calculate trend
        recent_score = historical_scores[-1]
        previous_score = historical_scores[-2]
        change = recent_score - previous_score
        change_percentage = (change / previous_score) * 100 if previous_score > 0 else 0
        
        # Determine direction
        if change > 0:
            direction = "improving"
        elif change < 0:
            direction = "declining"
        else:
            direction = "stable"
        
        # Calculate average change
        changes = []
        for i in range(1, len(historical_scores)):
            changes.append(historical_scores[i] - historical_scores[i-1])
        
        avg_change = sum(changes) / len(changes) if changes else 0
        
        return {
            "trend": "sufficient_data",
            "change": change,
            "change_percentage": change_percentage,
            "direction": direction,
            "average_change": avg_change,
            "historical_scores": historical_scores,
            "score_history": len(historical_scores)
        }
    
    def get_recommendations_by_score(self) -> List[str]:
        """
        Get recommendations based on score level
        
        Returns:
            List[str]: Score-specific recommendations
        """
        if self.is_excellent():
            return [
                "Maintain current SEO practices",
                "Focus on content quality and user experience",
                "Monitor for any potential issues",
                "Consider advanced optimization techniques"
            ]
        elif self.is_good():
            return [
                "Address minor SEO issues",
                "Improve meta descriptions and titles",
                "Add more relevant internal links",
                "Optimize content for target keywords"
            ]
        elif self.is_fair():
            return [
                "Fix moderate SEO issues",
                "Improve page load speed",
                "Add missing meta tags",
                "Create more comprehensive content",
                "Fix broken links and redirects"
            ]
        elif self.is_poor():
            return [
                "Address critical SEO issues immediately",
                "Fix technical SEO problems",
                "Improve content quality and relevance",
                "Add essential meta tags",
                "Fix mobile responsiveness issues",
                "Improve site structure and navigation"
            ]
        else:
            return [
                "Emergency SEO overhaul required",
                "Fix all technical SEO issues",
                "Completely rewrite content",
                "Add all essential meta tags",
                "Improve site architecture",
                "Fix all broken links and redirects",
                "Optimize for mobile devices",
                "Improve page load speed significantly"
            ]
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, any]: Score as dictionary
        """
        return {
            "value": self.value,
            "grade": self.get_grade().value,
            "percentage": self.get_percentage(),
            "level": self.get_level(),
            "color": self.get_color(),
            "priority": self.get_priority(),
            "improvement_potential": self.get_improvement_potential(),
            "improvement_percentage": self.get_improvement_percentage(),
            "recommendations": self.get_recommendations_by_score()
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.value}/100 ({self.get_grade().value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"SEOScore(value={self.value}, grade={self.get_grade().value}, level='{self.get_level()}')"
    
    def __eq__(self, other) -> bool:
        """Compare scores"""
        if not isinstance(other, SEOScore):
            return False
        return self.value == other.value
    
    def __lt__(self, other) -> bool:
        """Compare scores for sorting"""
        if not isinstance(other, SEOScore):
            return NotImplemented
        return self.value < other.value
    
    def __le__(self, other) -> bool:
        """Compare scores for sorting"""
        if not isinstance(other, SEOScore):
            return NotImplemented
        return self.value <= other.value
    
    def __gt__(self, other) -> bool:
        """Compare scores for sorting"""
        if not isinstance(other, SEOScore):
            return NotImplemented
        return self.value > other.value
    
    def __ge__(self, other) -> bool:
        """Compare scores for sorting"""
        if not isinstance(other, SEOScore):
            return NotImplemented
        return self.value >= other.value
    
    def __hash__(self) -> int:
        """Hash score"""
        return hash(self.value)
    
    def __add__(self, other) -> "SEOScore":
        """Add scores"""
        if isinstance(other, SEOScore):
            return SEOScore(min(self.value + other.value, 100))
        elif isinstance(other, int):
            return SEOScore(min(self.value + other, 100))
        else:
            return NotImplemented
    
    def __sub__(self, other) -> "SEOScore":
        """Subtract scores"""
        if isinstance(other, SEOScore):
            return SEOScore(max(self.value - other.value, 0))
        elif isinstance(other, int):
            return SEOScore(max(self.value - other, 0))
        else:
            return NotImplemented
    
    @classmethod
    def create(cls, value: int) -> "SEOScore":
        """
        Factory method to create SEO score
        
        Args:
            value: Score value (0-100)
            
        Returns:
            SEOScore: New SEO score instance
            
        Raises:
            ValueError: If score is invalid
        """
        return cls(value)
    
    @classmethod
    def create_perfect(cls) -> "SEOScore":
        """
        Create perfect score
        
        Returns:
            SEOScore: Perfect score (100)
        """
        return cls(100)
    
    @classmethod
    def create_zero(cls) -> "SEOScore":
        """
        Create zero score
        
        Returns:
            SEOScore: Zero score (0)
        """
        return cls(0)
    
    @classmethod
    def create_average(cls) -> "SEOScore":
        """
        Create average score
        
        Returns:
            SEOScore: Average score (50)
        """
        return cls(50)
    
    @classmethod
    def create_from_percentage(cls, percentage: float) -> "SEOScore":
        """
        Create score from percentage
        
        Args:
            percentage: Percentage (0.0-100.0)
            
        Returns:
            SEOScore: New SEO score instance
        """
        value = int(round(percentage))
        return cls(max(0, min(value, 100))) 