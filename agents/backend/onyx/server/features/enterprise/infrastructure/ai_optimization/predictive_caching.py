from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import pickle
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
from typing import Any, List, Dict, Optional
"""
Predictive Caching with Machine Learning
========================================

AI-powered caching system that predicts what data to cache before it's requested:
- ML models to predict cache hits
- User behavior analysis
- Pattern recognition
- Proactive cache warming
- Intelligent cache eviction
"""


logger = logging.getLogger(__name__)

@dataclass
class CacheRequest:
    """Cache request metadata."""
    key: str
    user_id: Optional[str]
    timestamp: datetime
    endpoint: str
    method: str
    ip_address: str
    user_agent: str
    success: bool = True
    response_time: float = 0.0
    cache_hit: bool = False


@dataclass
class UserPattern:
    """User behavior pattern."""
    user_id: str
    frequent_keys: List[str]
    access_times: List[datetime]
    session_duration: float
    request_frequency: float
    pattern_score: float = 0.0


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns for predictive caching."""
    
    def __init__(self, history_window: int = 1000):
        
    """__init__ function."""
self.history_window = history_window
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.patterns: Dict[str, UserPattern] = {}
        self.global_patterns: List[str] = []
        
    def record_request(self, request: CacheRequest):
        """Record a cache request for analysis."""
        if request.user_id:
            self.user_requests[request.user_id].append(request)
            
        # Update patterns periodically
        if len(self.user_requests[request.user_id]) % 100 == 0:
            asyncio.create_task(self.analyze_user_pattern(request.user_id))
    
    async def analyze_user_pattern(self, user_id: str) -> UserPattern:
        """Analyze patterns for a specific user."""
        requests = list(self.user_requests[user_id])
        if len(requests) < 10:
            return None
        
        # Extract frequent keys
        key_counts = defaultdict(int)
        access_times = []
        
        for req in requests:
            key_counts[req.key] += 1
            access_times.append(req.timestamp)
        
        # Find most frequent keys
        frequent_keys = sorted(key_counts.keys(), key=lambda k: key_counts[k], reverse=True)[:10]
        
        # Calculate session metrics
        if len(access_times) >= 2:
            session_duration = (access_times[-1] - access_times[0]).total_seconds()
            request_frequency = len(requests) / max(session_duration, 1)
        else:
            session_duration = 0
            request_frequency = 0
        
        # Calculate pattern score (how predictable the user is)
        pattern_score = self._calculate_pattern_score(requests)
        
        pattern = UserPattern(
            user_id=user_id,
            frequent_keys=frequent_keys,
            access_times=access_times,
            session_duration=session_duration,
            request_frequency=request_frequency,
            pattern_score=pattern_score
        )
        
        self.patterns[user_id] = pattern
        return pattern
    
    def _calculate_pattern_score(self, requests: List[CacheRequest]) -> float:
        """Calculate how predictable a user's behavior is."""
        if len(requests) < 5:
            return 0.0
        
        # Analyze key repetition patterns
        key_sequence = [req.key for req in requests[-20:]]  # Last 20 requests
        unique_keys = len(set(key_sequence))
        repetition_score = 1.0 - (unique_keys / len(key_sequence))
        
        # Analyze timing patterns
        times = [req.timestamp for req in requests[-10:]]
        if len(times) >= 2:
            intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
            if intervals:
                avg_interval = np.mean(intervals)
                interval_variance = np.var(intervals)
                timing_score = 1.0 / (1.0 + interval_variance / max(avg_interval, 1))
            else:
                timing_score = 0.0
        else:
            timing_score = 0.0
        
        return (repetition_score * 0.7) + (timing_score * 0.3)
    
    def predict_next_keys(self, user_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """Predict next keys a user is likely to request."""
        if user_id not in self.patterns:
            return []
        
        pattern = self.patterns[user_id]
        if not pattern.frequent_keys:
            return []
        
        # Simple prediction based on frequency and recency
        recent_requests = list(self.user_requests[user_id])[-10:]
        recent_keys = [req.key for req in recent_requests]
        
        predictions = []
        for key in pattern.frequent_keys[:n]:
            # Calculate prediction confidence
            frequency_score = pattern.frequent_keys.index(key) / len(pattern.frequent_keys)
            recency_score = 1.0 if key in recent_keys else 0.5
            confidence = (frequency_score * 0.6) + (recency_score * 0.4) * pattern.pattern_score
            
            predictions.append((key, confidence))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)


class MLCachePredictor:
    """Machine Learning model for cache hit prediction."""
    
    def __init__(self) -> Any:
        self.model = None
        self.feature_history: List[Dict] = []
        self.training_data: List[Tuple[List[float], bool]] = []
        self.is_trained = False
        
    def extract_features(self, request: CacheRequest, user_pattern: Optional[UserPattern] = None) -> List[float]:
        """Extract features for ML prediction."""
        features = []
        
        # Time-based features
        hour = request.timestamp.hour
        day_of_week = request.timestamp.weekday()
        features.extend([
            hour / 24.0,  # Normalized hour
            day_of_week / 7.0,  # Normalized day
            (time.time() % 3600) / 3600.0,  # Current hour progress
        ])
        
        # Request features
        key_hash = int(hashlib.md5(request.key.encode()).hexdigest()[:8], 16) % 1000
        endpoint_hash = int(hashlib.md5(request.endpoint.encode()).hexdigest()[:8], 16) % 100
        features.extend([
            key_hash / 1000.0,  # Normalized key hash
            endpoint_hash / 100.0,  # Normalized endpoint hash
            len(request.key) / 100.0,  # Key length (normalized)
        ])
        
        # User pattern features
        if user_pattern:
            features.extend([
                user_pattern.request_frequency / 10.0,  # Normalized frequency
                user_pattern.pattern_score,
                len(user_pattern.frequent_keys) / 20.0,  # Normalized frequent keys count
            ])
        else:
            features.extend([0.0, 0.0, 0.0])  # Default values
        
        # Historical features (simplified)
        features.extend([
            len(self.feature_history) / 1000.0,  # Total requests (normalized)
            0.5,  # Placeholder for cache hit rate
        ])
        
        return features
    
    def record_training_data(self, request: CacheRequest, user_pattern: Optional[UserPattern] = None):
        """Record data for model training."""
        features = self.extract_features(request, user_pattern)
        label = request.cache_hit
        
        self.training_data.append((features, label))
        self.feature_history.append({
            'timestamp': request.timestamp,
            'key': request.key,
            'cache_hit': request.cache_hit
        })
        
        # Maintain limited history
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-5000:]  # Keep last 5000
        
        # Retrain periodically
        if len(self.training_data) % 500 == 0 and len(self.training_data) >= 100:
            asyncio.create_task(self.train_model())
    
    async def train_model(self) -> Any:
        """Train the ML model for cache prediction."""
        try:
            # Use simple logistic regression (can be upgraded to more complex models)
            
            if len(self.training_data) < 50:
                return
            
            # Prepare training data
            X = np.array([features for features, _ in self.training_data])
            y = np.array([label for _, label in self.training_data])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.is_trained = True
            
            logger.info(f"ML Cache model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
        except ImportError:
            logger.warning("scikit-learn not available for ML cache prediction")
        except Exception as e:
            logger.error(f"Failed to train cache prediction model: {e}")
    
    def predict_cache_hit_probability(self, request: CacheRequest, user_pattern: Optional[UserPattern] = None) -> float:
        """Predict probability of cache hit."""
        if not self.is_trained or not self.model:
            return 0.5  # Default probability
        
        try:
            features = self.extract_features(request, user_pattern)
            probability = self.model.predict_proba([features])[0][1]  # Probability of True (cache hit)
            return float(probability)
        except Exception as e:
            logger.error(f"Cache hit prediction failed: {e}")
            return 0.5


class PredictiveCacheManager:
    """Intelligent cache manager with ML-powered predictions."""
    
    def __init__(self, cache_backend, preload_threshold: float = 0.7):
        
    """__init__ function."""
self.cache_backend = cache_backend
        self.preload_threshold = preload_threshold
        
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.ml_predictor = MLCachePredictor()
        
        self.preloaded_keys: set = set()
        self.stats = {
            'predictions_made': 0,
            'successful_predictions': 0,
            'preloads_performed': 0,
            'cache_hits_improved': 0
        }
        
        # Start background tasks
        asyncio.create_task(self._prediction_loop())
    
    async def get(self, key: str, user_id: str = None, endpoint: str = "", method: str = "GET", ip_address: str = "") -> Optional[Dict[str, Any]]:
        """Get value from cache with prediction recording."""
        start_time = time.time()
        
        # Try to get from cache
        value = await self.cache_backend.get(key)
        cache_hit = value is not None
        
        # Record the request
        request = CacheRequest(
            key=key,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            response_time=time.time() - start_time,
            cache_hit=cache_hit
        )
        
        # Record for analysis
        self.behavior_analyzer.record_request(request)
        
        # Get user pattern for ML prediction
        user_pattern = self.behavior_analyzer.patterns.get(user_id) if user_id else None
        self.ml_predictor.record_training_data(request, user_pattern)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        return await self.cache_backend.set(key, value, ttl)
    
    async def _prediction_loop(self) -> Any:
        """Background loop for predictive cache operations."""
        while True:
            try:
                await self._perform_predictive_operations()
                await asyncio.sleep(30)  # Run every 30 seconds
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_predictive_operations(self) -> Any:
        """Perform predictive caching operations."""
        # Predict and preload for active users
        for user_id, pattern in self.behavior_analyzer.patterns.items():
            if pattern.pattern_score > 0.5:  # Only for predictable users
                await self._preload_for_user(user_id, pattern)
        
        # Clean up old preloaded keys
        await self._cleanup_preloaded_keys()
    
    async def _preload_for_user(self, user_id: str, pattern: UserPattern):
        """Preload cache for a specific user based on predictions."""
        predictions = self.behavior_analyzer.predict_next_keys(user_id, 5)
        
        for key, confidence in predictions:
            if confidence > self.preload_threshold and key not in self.preloaded_keys:
                # Create a mock request for ML prediction
                mock_request = CacheRequest(
                    key=key,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    endpoint="predicted",
                    method="GET",
                    ip_address="",
                    cache_hit=False
                )
                
                # Get ML prediction
                ml_probability = self.ml_predictor.predict_cache_hit_probability(mock_request, pattern)
                
                # Combine behavioral and ML predictions
                combined_confidence = (confidence * 0.6) + (ml_probability * 0.4)
                
                if combined_confidence > self.preload_threshold:
                    await self._preload_key(key, user_id)
                    self.stats['preloads_performed'] += 1
    
    async def _preload_key(self, key: str, user_id: str):
        """Preload a specific key into cache."""
        try:
            # Check if already cached
            existing = await self.cache_backend.get(key)
            if existing is not None:
                return
            
            # Generate or fetch data for the key
            # This would typically call the original data source
            # For now, we'll mark it as preloaded
            self.preloaded_keys.add(key)
            
            logger.info(f"Preloaded cache key '{key}' for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to preload key '{key}': {e}")
    
    async def _cleanup_preloaded_keys(self) -> Any:
        """Clean up old preloaded keys."""
        # Remove preloaded keys older than 1 hour
        # In a real implementation, you'd track timestamps
        if len(self.preloaded_keys) > 1000:
            # Keep only recent keys (simplified)
            keys_to_remove = list(self.preloaded_keys)[:500]
            for key in keys_to_remove:
                self.preloaded_keys.discard(key)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get predictive caching statistics."""
        total_users = len(self.behavior_analyzer.patterns)
        predictable_users = sum(1 for p in self.behavior_analyzer.patterns.values() if p.pattern_score > 0.5)
        
        return {
            'total_users_analyzed': total_users,
            'predictable_users': predictable_users,
            'predictable_ratio': predictable_users / max(total_users, 1),
            'ml_model_trained': self.ml_predictor.is_trained,
            'training_data_size': len(self.ml_predictor.training_data),
            'preloaded_keys_count': len(self.preloaded_keys),
            'prediction_stats': self.stats.copy()
        } 