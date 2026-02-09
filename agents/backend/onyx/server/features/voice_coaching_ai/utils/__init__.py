"""
🔧 VOICE COACHING AI - UTILS MODULE
===================================

Utility functions and helpers for voice coaching system including:
- Audio processing and analysis utilities
- Performance monitoring and analytics
- Caching and optimization
- Validation and error handling
- Data transformation and formatting
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import numpy as np

from ..core import (
    VoiceAnalysis, VoiceProfile, CoachingSession, VoiceExercise,
    VoiceToneType, ConfidenceLevel, CoachingFocus, VoiceAnalysisMetrics
)

logger = logging.getLogger(__name__)

# =============================================================================
# 🎵 AUDIO PROCESSING UTILITIES
# =============================================================================

class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"

@dataclass
class AudioMetadata:
    """Audio file metadata"""
    format: AudioFormat
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    file_size: int
    encoding: str

class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def validate_audio_data(audio_data: bytes) -> bool:
        """Validate audio data format and integrity"""
        try:
            if len(audio_data) < 1024:  # Minimum size check
                return False
            
            # Basic format detection (simplified)
            if audio_data.startswith(b'RIFF'):  # WAV
                return True
            elif audio_data.startswith(b'ID3') or audio_data[:2] == b'\xff\xfb':  # MP3
                return True
            elif audio_data.startswith(b'ftyp'):  # M4A
                return True
            else:
                # Assume valid if not empty
                return len(audio_data) > 0
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False
    
    @staticmethod
    def encode_audio_base64(audio_data: bytes) -> str:
        """Encode audio data to base64 string"""
        try:
            return base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Audio encoding error: {e}")
            return ""
    
    @staticmethod
    def decode_audio_base64(base64_string: str) -> bytes:
        """Decode base64 string to audio data"""
        try:
            return base64.b64decode(base64_string.encode('utf-8'))
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return b""
    
    @staticmethod
    def extract_audio_features(audio_data: bytes) -> Dict[str, Any]:
        """Extract basic audio features (placeholder for actual implementation)"""
        try:
            # This would integrate with actual audio processing libraries
            # For now, return mock features
            return {
                "duration": len(audio_data) / 1000,  # Mock duration
                "sample_rate": 44100,
                "channels": 1,
                "format": "wav",
                "size_bytes": len(audio_data),
                "bit_rate": 128000
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}

# =============================================================================
# 📊 ANALYTICS AND METRICS UTILITIES
# =============================================================================

@dataclass
class AnalyticsEvent:
    """Analytics event data"""
    event_type: str
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    session_id: Optional[str] = None

class AnalyticsTracker:
    """Track and analyze voice coaching events"""
    
    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.metrics: Dict[str, Any] = {}
    
    def track_event(self, event_type: str, user_id: str, data: Dict[str, Any], session_id: Optional[str] = None):
        """Track an analytics event"""
        event = AnalyticsEvent(
            event_type=event_type,
            user_id=user_id,
            timestamp=datetime.now(),
            data=data,
            session_id=session_id
        )
        self.events.append(event)
        logger.info(f"Tracked event: {event_type} for user {user_id}")
    
    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        cutoff_date = datetime.now() - timedelta(days=days)
        user_events = [
            event for event in self.events 
            if event.user_id == user_id and event.timestamp >= cutoff_date
        ]
        
        return {
            "total_sessions": len([e for e in user_events if e.event_type == "session_start"]),
            "total_analyses": len([e for e in user_events if e.event_type == "voice_analysis"]),
            "average_confidence": self._calculate_average_confidence(user_events),
            "most_common_tone": self._get_most_common_tone(user_events),
            "improvement_trend": self._calculate_improvement_trend(user_events)
        }
    
    def _calculate_average_confidence(self, events: List[AnalyticsEvent]) -> float:
        """Calculate average confidence from events"""
        confidence_values = [
            event.data.get("confidence", 0) 
            for event in events 
            if "confidence" in event.data
        ]
        return sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    
    def _get_most_common_tone(self, events: List[AnalyticsEvent]) -> str:
        """Get most common voice tone from events"""
        tones = [
            event.data.get("tone", "unknown") 
            for event in events 
            if "tone" in event.data
        ]
        if tones:
            return max(set(tones), key=tones.count)
        return "unknown"
    
    def _calculate_improvement_trend(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Calculate improvement trend over time"""
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        
        if len(sorted_events) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        # Extract confidence values over time
        confidence_data = [
            (event.timestamp, event.data.get("confidence", 0))
            for event in sorted_events
            if "confidence" in event.data
        ]
        
        if len(confidence_data) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        # Calculate simple linear trend
        x_values = [(event[0] - confidence_data[0][0]).total_seconds() for event in confidence_data]
        y_values = [event[1] for event in confidence_data]
        
        # Simple linear regression
        n = len(x_values)
        if n > 1:
            slope = (n * sum(x * y for x, y in zip(x_values, y_values)) - 
                    sum(x_values) * sum(y_values)) / (n * sum(x * x for x in x_values) - sum(x_values) ** 2)
        else:
            slope = 0.0
        
        return {
            "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
            "slope": slope,
            "data_points": len(confidence_data)
        }

# =============================================================================
# 💾 CACHING AND OPTIMIZATION UTILITIES
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl: int
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def access(self):
        """Mark entry as accessed"""
        self.access_count += 1

class VoiceCoachingCache:
    """Caching system for voice coaching data"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_log: List[Tuple[str, datetime]] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        entry = self.cache.get(key)
        if entry and not entry.is_expired():
            entry.access()
            return entry.value
        elif entry and entry.is_expired():
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl
        )
        
        self.cache[key] = entry
        return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def _evict_oldest(self):
        """Evict oldest cache entry"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k].timestamp)
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "max_size": self.max_size,
            "utilization": (total_entries / self.max_size) * 100 if self.max_size > 0 else 0
        }

# =============================================================================
# ✅ VALIDATION AND ERROR HANDLING UTILITIES
# =============================================================================

class ValidationError(Exception):
    """Custom validation error"""
    pass

class VoiceCoachingValidator:
    """Validation utilities for voice coaching data"""
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format"""
        if not user_id or not isinstance(user_id, str):
            return False
        if len(user_id) < 3 or len(user_id) > 50:
            return False
        return True
    
    @staticmethod
    def validate_audio_data(audio_data: bytes) -> bool:
        """Validate audio data"""
        if not audio_data or not isinstance(audio_data, bytes):
            return False
        if len(audio_data) < 1024:  # Minimum size
            return False
        return True
    
    @staticmethod
    def validate_voice_analysis(analysis: VoiceAnalysis) -> bool:
        """Validate voice analysis data"""
        if not analysis:
            return False
        
        required_fields = ['user_id', 'tone', 'confidence_level', 'metrics']
        for field in required_fields:
            if not hasattr(analysis, field):
                return False
        
        return True
    
    @staticmethod
    def validate_coaching_session(session: CoachingSession) -> bool:
        """Validate coaching session data"""
        if not session:
            return False
        
        required_fields = ['session_id', 'user_id', 'focus_area', 'start_time']
        for field in required_fields:
            if not hasattr(session, field):
                return False
        
        return True

class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    def handle_api_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle API-related errors"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"API Error in {context}: {error}")
        return error_info
    
    @staticmethod
    def handle_validation_error(error: ValidationError, context: str = "") -> Dict[str, Any]:
        """Handle validation errors"""
        error_info = {
            "error_type": "ValidationError",
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"Validation Error in {context}: {error}")
        return error_info
    
    @staticmethod
    def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
        """Retry function with exponential backoff"""
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        logger.warning(f"Retry attempt {attempt + 1} for {func.__name__}: {e}")
            
            raise last_exception
        
        return wrapper

# =============================================================================
# 🔄 DATA TRANSFORMATION UTILITIES
# =============================================================================

class DataTransformer:
    """Data transformation utilities"""
    
    @staticmethod
    def voice_analysis_to_dict(analysis: VoiceAnalysis) -> Dict[str, Any]:
        """Convert VoiceAnalysis to dictionary"""
        if not analysis:
            return {}
        
        return {
            "user_id": analysis.user_id,
            "tone": analysis.tone.value if analysis.tone else None,
            "confidence_level": analysis.confidence_level.value if analysis.confidence_level else None,
            "confidence_score": analysis.confidence_score,
            "metrics": asdict(analysis.metrics) if analysis.metrics else {},
            "feedback": analysis.feedback,
            "suggestions": analysis.suggestions,
            "timestamp": analysis.timestamp.isoformat() if analysis.timestamp else None
        }
    
    @staticmethod
    def dict_to_voice_analysis(data: Dict[str, Any]) -> VoiceAnalysis:
        """Convert dictionary to VoiceAnalysis"""
        try:
            return VoiceAnalysis(
                user_id=data.get("user_id", ""),
                tone=VoiceToneType(data.get("tone", "unknown")),
                confidence_level=ConfidenceLevel(data.get("confidence_level", "low")),
                confidence_score=data.get("confidence_score", 0.0),
                metrics=VoiceAnalysisMetrics(**data.get("metrics", {})),
                feedback=data.get("feedback", ""),
                suggestions=data.get("suggestions", []),
                timestamp=datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else datetime.now()
            )
        except Exception as e:
            logger.error(f"Error converting dict to VoiceAnalysis: {e}")
            return None
    
    @staticmethod
    def format_progress_report(user_id: str, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Format progress report for user"""
        return {
            "user_id": user_id,
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_sessions": analytics.get("total_sessions", 0),
                "total_analyses": analytics.get("total_analyses", 0),
                "average_confidence": round(analytics.get("average_confidence", 0.0), 2),
                "most_common_tone": analytics.get("most_common_tone", "unknown"),
                "improvement_trend": analytics.get("improvement_trend", {}).get("trend", "unknown")
            },
            "recommendations": DataTransformer._generate_recommendations(analytics)
        }
    
    @staticmethod
    def _generate_recommendations(analytics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analytics"""
        recommendations = []
        
        avg_confidence = analytics.get("average_confidence", 0.0)
        trend = analytics.get("improvement_trend", {}).get("trend", "unknown")
        total_sessions = analytics.get("total_sessions", 0)
        
        if avg_confidence < 0.5:
            recommendations.append("Focus on confidence-building exercises")
        
        if trend == "declining":
            recommendations.append("Consider reviewing recent practice sessions")
        
        if total_sessions < 5:
            recommendations.append("Increase practice frequency for better results")
        
        if not recommendations:
            recommendations.append("Continue with current practice routine")
        
        return recommendations

# =============================================================================
# 🚀 PERFORMANCE MONITORING UTILITIES
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for voice coaching operations"""
    operation: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PerformanceMonitor:
    """Monitor performance of voice coaching operations"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, float] = {}
    
    def start_operation(self, operation: str) -> str:
        """Start timing an operation"""
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        self.active_operations[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: str = None):
        """End timing an operation"""
        if operation_id in self.active_operations:
            start_time = self.active_operations.pop(operation_id)
            duration = time.time() - start_time
            
            metric = PerformanceMetrics(
                operation=operation_id.split('_')[0],
                duration=duration,
                success=success,
                error_message=error_message
            )
            
            self.metrics.append(metric)
    
    def get_performance_summary(self, operation: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if operation:
            relevant_metrics = [
                m for m in self.metrics 
                if m.operation == operation and m.timestamp >= cutoff_time
            ]
        else:
            relevant_metrics = [
                m for m in self.metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not relevant_metrics:
            return {"error": "No metrics available"}
        
        durations = [m.duration for m in relevant_metrics]
        success_count = sum(1 for m in relevant_metrics if m.success)
        
        return {
            "total_operations": len(relevant_metrics),
            "success_rate": (success_count / len(relevant_metrics)) * 100,
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "operation": operation or "all"
        }

# =============================================================================
# 🎯 QUICK UTILITY FUNCTIONS
# =============================================================================

def create_audio_processor() -> AudioProcessor:
    """Create audio processor instance"""
    return AudioProcessor()

def create_analytics_tracker() -> AnalyticsTracker:
    """Create analytics tracker instance"""
    return AnalyticsTracker()

def create_cache(max_size: int = 1000) -> VoiceCoachingCache:
    """Create cache instance"""
    return VoiceCoachingCache(max_size=max_size)

def create_validator() -> VoiceCoachingValidator:
    """Create validator instance"""
    return VoiceCoachingValidator()

def create_performance_monitor() -> PerformanceMonitor:
    """Create performance monitor instance"""
    return PerformanceMonitor()

def create_data_transformer() -> DataTransformer:
    """Create data transformer instance"""
    return DataTransformer()

# =============================================================================
# 🔧 UTILITY HELPERS
# =============================================================================

def generate_session_id(user_id: str) -> str:
    """Generate unique session ID"""
    timestamp = int(time.time() * 1000)
    hash_input = f"{user_id}_{timestamp}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def calculate_progress_score(initial_confidence: float, current_confidence: float) -> float:
    """Calculate progress score between two confidence levels"""
    if initial_confidence == 0:
        return 0.0
    return ((current_confidence - initial_confidence) / initial_confidence) * 100

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace unsafe characters
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe_filename[:255]  # Limit length 