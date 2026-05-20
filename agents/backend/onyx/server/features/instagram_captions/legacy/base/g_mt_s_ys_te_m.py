from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
import logging
from models import CaptionStyle, InstagramTarget
from typing import Any, List, Dict, Optional
"""
GMT System for Instagram Captions.

Simplified and consolidated GMT system for optimal timing,
cultural adaptation, and global campaign coordination.
"""



logger = logging.getLogger(__name__)

class EngagementLevel(str, Enum):
    """Engagement levels for different time windows."""
    PEAK = "peak"           # 85-95% engagement
    HIGH = "high"           # 70-84% engagement  
    GOOD = "good"           # 55-69% engagement
    MODERATE = "moderate"   # 40-54% engagement
    LOW = "low"            # <40% engagement

@dataclass
class EngagementWindow:
    """Engagement window with timing and performance data."""
    start_time: str
    end_time: str
    level: EngagementLevel
    confidence: float
    audience_multiplier: float

@dataclass
class TimezoneInsights:
    """Comprehensive timezone insights."""
    timezone: str
    peak_windows: List[EngagementWindow]
    optimal_posting_times: List[str]
    cultural_context: Dict[str, Any]
    best_days: List[str]
    audience_activity: Dict[str, float]

class CulturalAdapter:
    """Handle cultural adaptation for different regions."""
    
    def __init__(self) -> Any:
        self.cultural_profiles = {
            "US/Eastern": {
                "greeting_style": "Hey there! / What's up?",
                "communication": "direct, friendly, casual",
                "values": ["efficiency", "innovation", "success"],
                "content_preferences": ["behind-the-scenes", "success stories", "tips"],
                "avoid": ["overly formal", "too much small talk"]
            },
            "US/Pacific": {
                "greeting_style": "Hey! / Ready for this?",
                "communication": "relaxed, creative, authentic",
                "values": ["creativity", "wellness", "sustainability"],
                "content_preferences": ["lifestyle", "wellness", "creative process"],
                "avoid": ["aggressive sales", "corporate speak"]
            },
            "Europe/London": {
                "greeting_style": "Hello! / Right then,",
                "communication": "polite, thoughtful, subtle humor",
                "values": ["tradition", "quality", "understatement"],
                "content_preferences": ["educational", "thoughtful insights", "dry humor"],
                "avoid": ["overly enthusiastic", "hard sell"]
            },
            "Asia/Tokyo": {
                "greeting_style": "こんにちは! / Hello everyone!",
                "communication": "respectful, detailed, group-focused",
                "values": ["respect", "quality", "community"],
                "content_preferences": ["detailed explanations", "group benefits", "quality focus"],
                "avoid": ["individual promotion", "rushed content"]
            },
            "Australia/Sydney": {
                "greeting_style": "G'day! / Hey mate!",
                "communication": "casual, humorous, straightforward",
                "values": ["authenticity", "humor", "work-life balance"],
                "content_preferences": ["authentic moments", "humor", "outdoor/lifestyle"],
                "avoid": ["pretentious", "overly serious"]
            }
        }
    
    def get_cultural_adaptation(self, timezone: str, content: str, 
                              style: CaptionStyle, audience: InstagramTarget) -> str:
        """Adapt content for cultural context."""
        
        profile = self.cultural_profiles.get(timezone)
        if not profile:
            return content  # No adaptation needed
        
        # Apply cultural greeting if appropriate
        if style == CaptionStyle.CASUAL and not any(greeting in content.lower() 
                                                  for greeting in ['hey', 'hello', 'hi']):
            greeting = profile["greeting_style"].split(" / ")[0]
            content = f"{greeting} {content}"
        
        # Add cultural context hints (subtle)
        cultural_hint = self._get_cultural_hint(timezone, audience)
        if cultural_hint and len(content) < 200:  # Only for shorter content
            content += f"\n\n{cultural_hint}"
        
        return content
    
    def _get_cultural_hint(self, timezone: str, audience: InstagramTarget) -> Optional[str]:
        """Get subtle cultural hint based on timezone and audience."""
        
        hints = {
            ("US/Eastern", InstagramTarget.BUSINESS): "Perfect for your morning coffee read ☕",
            ("US/Pacific", InstagramTarget.LIFESTYLE): "Great vibes for your mindful morning 🌅",
            ("Europe/London", InstagramTarget.PROFESSIONALS): "Brilliant for your afternoon tea break 🫖",
            ("Asia/Tokyo", InstagramTarget.BUSINESS): "Excellent for your productive planning session 📋",
            ("Australia/Sydney", InstagramTarget.LIFESTYLE): "Perfect for your arvo scroll 🌴"
        }
        
        return hints.get((timezone, audience))

class GMTEngagementCalculator:
    """Calculate optimal engagement windows for different timezones."""
    
    def __init__(self) -> Any:
        # Base engagement patterns (24-hour format, local time)
        self.base_patterns = {
            "weekday": {
                "06:00-09:00": EngagementLevel.HIGH,     # Morning commute
                "12:00-13:00": EngagementLevel.PEAK,     # Lunch break
                "17:00-19:00": EngagementLevel.PEAK,     # Evening commute
                "20:00-22:00": EngagementLevel.HIGH,     # Evening leisure
                "22:00-23:00": EngagementLevel.GOOD      # Late evening
            },
            "weekend": {
                "09:00-11:00": EngagementLevel.HIGH,     # Weekend morning
                "13:00-15:00": EngagementLevel.PEAK,     # Weekend afternoon
                "19:00-21:00": EngagementLevel.PEAK,     # Weekend evening
                "21:00-23:00": EngagementLevel.GOOD      # Weekend night
            }
        }
        
        # Audience multipliers
        self.audience_multipliers = {
            InstagramTarget.GEN_Z: {
                "morning": 0.8, "afternoon": 1.2, "evening": 1.3, "night": 1.1
            },
            InstagramTarget.MILLENNIALS: {
                "morning": 1.1, "afternoon": 1.0, "evening": 1.2, "night": 0.9
            },
            InstagramTarget.BUSINESS: {
                "morning": 1.3, "afternoon": 0.8, "evening": 1.0, "night": 0.6
            },
            InstagramTarget.LIFESTYLE: {
                "morning": 1.0, "afternoon": 1.2, "evening": 1.1, "night": 0.8
            }
        }
    
    def get_engagement_windows(self, timezone: str, audience: InstagramTarget) -> List[EngagementWindow]:
        """Get optimal engagement windows for timezone and audience."""
        
        windows = []
        day_type = "weekday"  # Simplified - could be enhanced with actual day detection
        
        pattern = self.base_patterns[day_type]
        multipliers = self.audience_multipliers.get(audience, {
            "morning": 1.0, "afternoon": 1.0, "evening": 1.0, "night": 1.0
        })
        
        for time_range, base_level in pattern.items():
            start_time, end_time = time_range.split('-')
            
            # Determine time period
            hour = int(start_time.split(':')[0])
            if 6 <= hour < 12:
                period = "morning"
            elif 12 <= hour < 17:
                period = "afternoon"
            elif 17 <= hour < 21:
                period = "evening"
            else:
                period = "night"
            
            # Apply audience multiplier
            multiplier = multipliers.get(period, 1.0)
            
            # Calculate confidence based on base level and multiplier
            confidence_map = {
                EngagementLevel.PEAK: 0.9,
                EngagementLevel.HIGH: 0.8,
                EngagementLevel.GOOD: 0.7,
                EngagementLevel.MODERATE: 0.6,
                EngagementLevel.LOW: 0.5
            }
            
            base_confidence = confidence_map.get(base_level, 0.7)
            adjusted_confidence = min(0.95, base_confidence * multiplier)
            
            # Adjust engagement level based on multiplier
            if multiplier >= 1.2:
                if base_level == EngagementLevel.HIGH:
                    adjusted_level = EngagementLevel.PEAK
                elif base_level == EngagementLevel.GOOD:
                    adjusted_level = EngagementLevel.HIGH
                else:
                    adjusted_level = base_level
            elif multiplier <= 0.8:
                if base_level == EngagementLevel.PEAK:
                    adjusted_level = EngagementLevel.HIGH
                elif base_level == EngagementLevel.HIGH:
                    adjusted_level = EngagementLevel.GOOD
                else:
                    adjusted_level = base_level
            else:
                adjusted_level = base_level
            
            windows.append(EngagementWindow(
                start_time=start_time,
                end_time=end_time,
                level=adjusted_level,
                confidence=adjusted_confidence,
                audience_multiplier=multiplier
            ))
        
        # Sort by engagement level and confidence
        return sorted(windows, key=lambda w: (w.level.value, w.confidence), reverse=True)
    
    def get_optimal_posting_times(self, windows: List[EngagementWindow]) -> List[str]:
        """Extract optimal posting times from engagement windows."""
        
        optimal_times = []
        
        for window in windows:
            if window.level in [EngagementLevel.PEAK, EngagementLevel.HIGH]:
                # Use the start time of high-engagement windows
                optimal_times.append(window.start_time)
        
        return optimal_times[:3]  # Return top 3 times

class SimplifiedGMTSystem:
    """Simplified GMT system integrating timing and cultural adaptation."""
    
    def __init__(self) -> Any:
        self.cultural_adapter = CulturalAdapter()
        self.engagement_calculator = GMTEngagementCalculator()
    
    def get_timezone_insights(self, timezone: str, audience: InstagramTarget = None) -> TimezoneInsights:
        """Get comprehensive timezone insights."""
        
        if audience is None:
            audience = InstagramTarget.GENERAL
        
        # Get engagement windows
        windows = self.engagement_calculator.get_engagement_windows(timezone, audience)
        
        # Get optimal posting times
        optimal_times = self.engagement_calculator.get_optimal_posting_times(windows)
        
        # Get cultural context
        cultural_context = self.cultural_adapter.cultural_profiles.get(timezone, {
            "greeting_style": "Hello!",
            "communication": "friendly, engaging",
            "values": ["authenticity", "value"],
            "content_preferences": ["helpful", "engaging"],
            "avoid": ["spam", "overly promotional"]
        })
        
        # Best days (simplified)
        best_days = ["Tuesday", "Wednesday", "Thursday"]  # Generally best for engagement
        
        # Audience activity (simplified)
        audience_activity = {
            "morning": 0.7,
            "afternoon": 0.8,
            "evening": 0.9,
            "night": 0.6
        }
        
        return TimezoneInsights(
            timezone=timezone,
            peak_windows=[w for w in windows if w.level == EngagementLevel.PEAK],
            optimal_posting_times=optimal_times,
            cultural_context=cultural_context,
            best_days=best_days,
            audience_activity=audience_activity
        )
    
    def adapt_content_culturally(self, content: str, timezone: str, 
                               style: CaptionStyle, audience: InstagramTarget) -> str:
        """Adapt content for cultural context."""
        return self.cultural_adapter.get_cultural_adaptation(timezone, content, style, audience)
    
    def get_engagement_recommendations(self, timezone: str, style: CaptionStyle, 
                                     audience: InstagramTarget) -> Dict[str, Any]:
        """Get engagement recommendations for timezone and context."""
        
        insights = self.get_timezone_insights(timezone, audience)
        
        # Style-specific recommendations
        style_recommendations = {
            CaptionStyle.CASUAL: [
                "Ask relatable questions about daily experiences",
                "Share behind-the-scenes moments",
                "Use conversational tone"
            ],
            CaptionStyle.PROFESSIONAL: [
                "Share industry insights and tips",
                "Ask for professional experiences",
                "Provide actionable business advice"
            ],
            CaptionStyle.INSPIRATIONAL: [
                "Ask followers to share their goals",
                "Create motivational discussions",
                "Encourage positive mindset sharing"
            ]
        }
        
        recommendations = {
            "optimal_posting_times": insights.optimal_posting_times,
            "best_days": insights.best_days,
            "engagement_strategy": style_recommendations.get(style, [
                "Create engaging, valuable content",
                "Ask questions to encourage interaction",
                "Share authentic experiences"
            ]),
            "cultural_tips": [
                f"Communication style: {insights.cultural_context.get('communication', 'engaging')}",
                f"Preferred content: {', '.join(insights.cultural_context.get('content_preferences', ['helpful']))}",
                f"Avoid: {', '.join(insights.cultural_context.get('avoid', ['spam']))}"
            ],
            "timing_confidence": max((w.confidence for w in insights.peak_windows), default=0.8)
        }
        
        return recommendations
    
    def calculate_posting_score(self, timezone: str, posting_time: str, 
                              audience: InstagramTarget) -> float:
        """Calculate score for posting at specific time."""
        
        windows = self.engagement_calculator.get_engagement_windows(timezone, audience)
        
        # Convert posting time to comparable format
        posting_hour = int(posting_time.split(':')[0])
        posting_minute = int(posting_time.split(':')[1])
        posting_total_minutes = posting_hour * 60 + posting_minute
        
        best_score = 0.0
        
        for window in windows:
            start_hour = int(window.start_time.split(':')[0])
            start_minute = int(window.start_time.split(':')[1])
            start_total_minutes = start_hour * 60 + start_minute
            
            end_hour = int(window.end_time.split(':')[0])
            end_minute = int(window.end_time.split(':')[1])
            end_total_minutes = end_hour * 60 + end_minute
            
            # Check if posting time falls within window
            if start_total_minutes <= posting_total_minutes <= end_total_minutes:
                # Calculate score based on engagement level and confidence
                level_scores = {
                    EngagementLevel.PEAK: 1.0,
                    EngagementLevel.HIGH: 0.8,
                    EngagementLevel.GOOD: 0.6,
                    EngagementLevel.MODERATE: 0.4,
                    EngagementLevel.LOW: 0.2
                }
                
                window_score = level_scores.get(window.level, 0.5) * window.confidence
                best_score = max(best_score, window_score)
        
        return best_score 