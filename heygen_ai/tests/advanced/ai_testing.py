"""
AI-Powered Testing Framework for HeyGen AI system.
Implements intelligent test generation, anomaly detection, and predictive testing.
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

@dataclass
class TestPattern:
    """Represents a learned test pattern."""
    pattern_id: str
    test_name: str
    features: List[float]
    success_rate: float
    avg_duration: float
    frequency: int
    last_seen: datetime
    cluster_id: Optional[int] = None
    anomaly_score: float = 0.0

@dataclass
class AITestResult:
    """Result of AI-powered test analysis."""
    test_name: str
    prediction: str  # success, failure, anomaly
    confidence: float
    features: List[float]
    anomaly_score: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class TestDataCollector:
    """Collects and processes test execution data for AI analysis."""
    
    def __init__(self, data_file: str = "test_ai_data.json"):
        self.data_file = Path(data_file)
        self.test_data: List[Dict[str, Any]] = []
        self.load_data()
    
    def load_data(self):
        """Load existing test data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    self.test_data = json.load(f)
            except Exception as e:
                logging.error(f"Error loading test data: {e}")
                self.test_data = []
    
    def save_data(self):
        """Save test data to file."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.test_data, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Error saving test data: {e}")
    
    def add_test_execution(self, test_name: str, duration: float, success: bool, 
                          features: Dict[str, Any] = None):
        """Add a test execution record."""
        record = {
            "test_name": test_name,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "features": features or {},
            "day_of_week": datetime.now().weekday(),
            "hour_of_day": datetime.now().hour,
            "memory_usage": features.get("memory_usage", 0) if features else 0,
            "cpu_usage": features.get("cpu_usage", 0) if features else 0
        }
        
        self.test_data.append(record)
        
        # Keep only last 10000 records
        if len(self.test_data) > 10000:
            self.test_data = self.test_data[-10000:]
        
        self.save_data()
    
    def get_features_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Extract features matrix for ML analysis."""
        if not self.test_data:
            return np.array([]), []
        
        df = pd.DataFrame(self.test_data)
        
        # Feature engineering
        features = []
        feature_names = []
        
        # Basic features
        features.append(df['duration'].values)
        feature_names.append('duration')
        
        features.append(df['memory_usage'].values)
        feature_names.append('memory_usage')
        
        features.append(df['cpu_usage'].values)
        feature_names.append('cpu_usage')
        
        features.append(df['day_of_week'].values)
        feature_names.append('day_of_week')
        
        features.append(df['hour_of_day'].values)
        feature_names.append('hour_of_day')
        
        # Success rate over time
        success_rates = []
        for i in range(len(df)):
            window_data = df[max(0, i-10):i+1]
            success_rate = window_data['success'].mean() if len(window_data) > 0 else 0.5
            success_rates.append(success_rate)
        
        features.append(np.array(success_rates))
        feature_names.append('success_rate_window')
        
        # Duration trend
        duration_trends = []
        for i in range(len(df)):
            window_data = df[max(0, i-5):i+1]
            if len(window_data) > 1:
                trend = np.polyfit(range(len(window_data)), window_data['duration'], 1)[0]
            else:
                trend = 0
            duration_trends.append(trend)
        
        features.append(np.array(duration_trends))
        feature_names.append('duration_trend')
        
        # Combine features
        X = np.column_stack(features)
        
        return X, feature_names

class AnomalyDetector:
    """Detects anomalies in test execution patterns."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit the anomaly detection model."""
        if len(X) == 0:
            return
        
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in test data."""
        if not self.is_fitted or len(X) == 0:
            return np.array([]), np.array([])
        
        X_scaled = self.scaler.transform(X)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)
        
        return predictions, anomaly_scores

class TestPatternLearner:
    """Learns patterns from test execution data."""
    
    def __init__(self):
        self.patterns: Dict[str, TestPattern] = {}
        self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        self.scaler = StandardScaler()
    
    def learn_patterns(self, test_data: List[Dict[str, Any]]):
        """Learn patterns from test execution data."""
        if not test_data:
            return
        
        df = pd.DataFrame(test_data)
        
        # Group by test name
        for test_name in df['test_name'].unique():
            test_records = df[df['test_name'] == test_name]
            
            if len(test_records) < 3:  # Need at least 3 records
                continue
            
            # Calculate features for this test
            features = self._extract_test_features(test_records)
            
            # Create pattern
            pattern = TestPattern(
                pattern_id=f"{test_name}_{int(time.time())}",
                test_name=test_name,
                features=features,
                success_rate=test_records['success'].mean(),
                avg_duration=test_records['duration'].mean(),
                frequency=len(test_records),
                last_seen=datetime.now()
            )
            
            self.patterns[pattern.pattern_id] = pattern
        
        # Cluster similar patterns
        self._cluster_patterns()
    
    def _extract_test_features(self, test_records: pd.DataFrame) -> List[float]:
        """Extract features from test records."""
        features = []
        
        # Basic statistics
        features.append(test_records['duration'].mean())
        features.append(test_records['duration'].std())
        features.append(test_records['success'].mean())
        features.append(test_records['memory_usage'].mean())
        features.append(test_records['cpu_usage'].mean())
        
        # Temporal features
        features.append(test_records['day_of_week'].mode().iloc[0] if len(test_records['day_of_week'].mode()) > 0 else 0)
        features.append(test_records['hour_of_day'].mode().iloc[0] if len(test_records['hour_of_day'].mode()) > 0 else 0)
        
        # Trend features
        if len(test_records) > 1:
            duration_trend = np.polyfit(range(len(test_records)), test_records['duration'], 1)[0]
            success_trend = np.polyfit(range(len(test_records)), test_records['success'].astype(int), 1)[0]
        else:
            duration_trend = 0
            success_trend = 0
        
        features.append(duration_trend)
        features.append(success_trend)
        
        return features
    
    def _cluster_patterns(self):
        """Cluster similar test patterns."""
        if len(self.patterns) < 2:
            return
        
        # Extract features matrix
        features_matrix = np.array([pattern.features for pattern in self.patterns.values()])
        
        if len(features_matrix) < 2:
            return
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_matrix)
        
        # Cluster
        cluster_labels = self.clusterer.fit_predict(features_scaled)
        
        # Assign cluster IDs to patterns
        pattern_ids = list(self.patterns.keys())
        for i, pattern_id in enumerate(pattern_ids):
            self.patterns[pattern_id].cluster_id = int(cluster_labels[i])
    
    def find_similar_patterns(self, test_name: str, features: List[float]) -> List[TestPattern]:
        """Find patterns similar to given test."""
        if not self.patterns:
            return []
        
        # Find patterns for the same test
        same_test_patterns = [
            p for p in self.patterns.values() 
            if p.test_name == test_name
        ]
        
        if not same_test_patterns:
            return []
        
        # Calculate similarity (simple Euclidean distance)
        similarities = []
        for pattern in same_test_patterns:
            distance = np.linalg.norm(np.array(features) - np.array(pattern.features))
            similarities.append((pattern, distance))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1])
        
        return [pattern for pattern, _ in similarities[:3]]  # Top 3 similar

class PredictiveTester:
    """Predicts test outcomes and generates intelligent recommendations."""
    
    def __init__(self, data_collector: TestDataCollector):
        self.data_collector = data_collector
        self.anomaly_detector = AnomalyDetector()
        self.pattern_learner = TestPatternLearner()
        self.model_file = Path("ai_test_model.joblib")
        self.load_model()
    
    def load_model(self):
        """Load trained model if available."""
        if self.model_file.exists():
            try:
                model_data = joblib.load(self.model_file)
                self.anomaly_detector = model_data.get('anomaly_detector', self.anomaly_detector)
                self.pattern_learner = model_data.get('pattern_learner', self.pattern_learner)
            except Exception as e:
                logging.error(f"Error loading AI model: {e}")
    
    def save_model(self):
        """Save trained model."""
        try:
            model_data = {
                'anomaly_detector': self.anomaly_detector,
                'pattern_learner': self.pattern_learner,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_file)
        except Exception as e:
            logging.error(f"Error saving AI model: {e}")
    
    def train_model(self):
        """Train the AI model on collected data."""
        print("🤖 Training AI Testing Model...")
        
        # Get features matrix
        X, feature_names = self.data_collector.get_features_matrix()
        
        if len(X) == 0:
            print("⚠️  No data available for training")
            return
        
        # Train anomaly detector
        self.anomaly_detector.fit(X)
        
        # Learn patterns
        self.pattern_learner.learn_patterns(self.data_collector.test_data)
        
        # Save model
        self.save_model()
        
        print(f"✅ AI model trained on {len(X)} samples")
        print(f"   Features: {', '.join(feature_names)}")
        print(f"   Patterns learned: {len(self.pattern_learner.patterns)}")
    
    def predict_test_outcome(self, test_name: str, features: Dict[str, Any] = None) -> AITestResult:
        """Predict the outcome of a test execution."""
        # Extract features
        feature_vector = self._extract_prediction_features(test_name, features)
        
        # Get features matrix for anomaly detection
        X, _ = self.data_collector.get_features_matrix()
        
        if len(X) == 0:
            return AITestResult(
                test_name=test_name,
                prediction="unknown",
                confidence=0.0,
                features=feature_vector,
                anomaly_score=0.0,
                recommendations=["Insufficient data for prediction"]
            )
        
        # Predict anomaly
        predictions, anomaly_scores = self.anomaly_detector.predict_anomaly(X)
        
        # Find similar patterns
        similar_patterns = self.pattern_learner.find_similar_patterns(test_name, feature_vector)
        
        # Make prediction
        if similar_patterns:
            avg_success_rate = np.mean([p.success_rate for p in similar_patterns])
            avg_duration = np.mean([p.avg_duration for p in similar_patterns])
            
            if avg_success_rate > 0.8:
                prediction = "success"
                confidence = avg_success_rate
            elif avg_success_rate < 0.3:
                prediction = "failure"
                confidence = 1 - avg_success_rate
            else:
                prediction = "uncertain"
                confidence = 0.5
        else:
            prediction = "unknown"
            confidence = 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_name, feature_vector, similar_patterns, anomaly_scores
        )
        
        return AITestResult(
            test_name=test_name,
            prediction=prediction,
            confidence=confidence,
            features=feature_vector,
            anomaly_score=float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0,
            recommendations=recommendations
        )
    
    def _extract_prediction_features(self, test_name: str, features: Dict[str, Any] = None) -> List[float]:
        """Extract features for prediction."""
        if features is None:
            features = {}
        
        # Basic features
        feature_vector = [
            features.get('duration', 0.0),
            features.get('memory_usage', 0.0),
            features.get('cpu_usage', 0.0),
            datetime.now().weekday(),
            datetime.now().hour
        ]
        
        # Historical success rate
        test_data = [d for d in self.data_collector.test_data if d['test_name'] == test_name]
        if test_data:
            recent_data = test_data[-10:]  # Last 10 executions
            success_rate = np.mean([d['success'] for d in recent_data])
            avg_duration = np.mean([d['duration'] for d in recent_data])
        else:
            success_rate = 0.5
            avg_duration = 1.0
        
        feature_vector.extend([success_rate, avg_duration])
        
        return feature_vector
    
    def _generate_recommendations(self, test_name: str, features: List[float], 
                                similar_patterns: List[TestPattern], 
                                anomaly_scores: np.ndarray) -> List[str]:
        """Generate intelligent recommendations."""
        recommendations = []
        
        # Anomaly-based recommendations
        if len(anomaly_scores) > 0 and np.mean(anomaly_scores) < -0.5:
            recommendations.append("⚠️  Test execution shows anomalous patterns - investigate potential issues")
        
        # Pattern-based recommendations
        if similar_patterns:
            avg_success_rate = np.mean([p.success_rate for p in similar_patterns])
            avg_duration = np.mean([p.avg_duration for p in similar_patterns])
            
            if avg_success_rate < 0.5:
                recommendations.append("🔧 Test has low success rate - consider debugging or refactoring")
            
            if avg_duration > 10.0:
                recommendations.append("⚡ Test execution is slow - consider optimization")
            
            if len(similar_patterns) > 1:
                recommendations.append(f"📊 Found {len(similar_patterns)} similar test patterns - good consistency")
        
        # Feature-based recommendations
        if features[0] > 5.0:  # Duration
            recommendations.append("⏱️  Test duration is high - consider breaking into smaller tests")
        
        if features[1] > 100.0:  # Memory usage
            recommendations.append("🧠 High memory usage detected - check for memory leaks")
        
        if features[2] > 80.0:  # CPU usage
            recommendations.append("💻 High CPU usage detected - consider performance optimization")
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            recommendations.append("🌙 Test running outside business hours - consider scheduling optimization")
        
        if not recommendations:
            recommendations.append("✅ Test execution looks normal - no specific recommendations")
        
        return recommendations

class AITestGenerator:
    """Generates intelligent test cases using AI."""
    
    def __init__(self, predictive_tester: PredictiveTester):
        self.predictive_tester = predictive_tester
    
    def generate_test_variations(self, base_test: Callable, test_name: str) -> List[Callable]:
        """Generate variations of a test based on learned patterns."""
        variations = []
        
        # Get similar patterns
        features = self.predictive_tester._extract_prediction_features(test_name)
        similar_patterns = self.predictive_tester.pattern_learner.find_similar_patterns(
            test_name, features
        )
        
        if not similar_patterns:
            return [base_test]
        
        # Generate variations based on patterns
        for i, pattern in enumerate(similar_patterns[:3]):
            def create_variation(original_test, pattern_id, variation_id):
                def variation_test():
                    # Add pattern-specific setup
                    print(f"Running test variation {variation_id} based on pattern {pattern_id}")
                    return original_test()
                return variation_test
            
            variation = create_variation(base_test, pattern.pattern_id, i)
            variation.__name__ = f"{test_name}_variation_{i}"
            variations.append(variation)
        
        return variations
    
    def suggest_test_improvements(self, test_name: str) -> List[str]:
        """Suggest improvements for a test based on AI analysis."""
        suggestions = []
        
        # Get test data
        test_data = [d for d in self.predictive_tester.data_collector.test_data 
                    if d['test_name'] == test_name]
        
        if not test_data:
            return ["No data available for analysis"]
        
        # Analyze patterns
        success_rate = np.mean([d['success'] for d in test_data])
        avg_duration = np.mean([d['duration'] for d in test_data])
        duration_std = np.std([d['duration'] for d in test_data])
        
        # Generate suggestions
        if success_rate < 0.7:
            suggestions.append("🔧 Improve test reliability - current success rate is low")
        
        if avg_duration > 5.0:
            suggestions.append("⚡ Optimize test performance - execution time is high")
        
        if duration_std > avg_duration * 0.5:
            suggestions.append("📊 Reduce test variability - execution time is inconsistent")
        
        # Check for time patterns
        recent_data = test_data[-10:]
        if len(recent_data) > 5:
            recent_success_rate = np.mean([d['success'] for d in recent_data])
            if recent_success_rate < success_rate * 0.8:
                suggestions.append("📉 Test quality is declining - investigate recent changes")
        
        return suggestions

class AITestingFramework:
    """Main AI testing framework integrating all AI capabilities."""
    
    def __init__(self):
        self.data_collector = TestDataCollector()
        self.predictive_tester = PredictiveTester(self.data_collector)
        self.test_generator = AITestGenerator(self.predictive_tester)
        self.is_trained = False
    
    def train(self):
        """Train the AI framework on existing data."""
        print("🤖 Training AI Testing Framework...")
        self.predictive_tester.train_model()
        self.is_trained = True
        print("✅ AI Testing Framework trained successfully")
    
    def analyze_test(self, test_name: str, duration: float, success: bool, 
                    features: Dict[str, Any] = None) -> AITestResult:
        """Analyze a test execution using AI."""
        # Add data point
        self.data_collector.add_test_execution(test_name, duration, success, features)
        
        # Predict outcome
        result = self.predictive_tester.predict_test_outcome(test_name, features)
        
        return result
    
    def generate_intelligent_tests(self, base_test: Callable, test_name: str) -> List[Callable]:
        """Generate intelligent test variations."""
        return self.test_generator.generate_test_variations(base_test, test_name)
    
    def get_test_insights(self, test_name: str) -> Dict[str, Any]:
        """Get comprehensive insights for a test."""
        # Get test data
        test_data = [d for d in self.data_collector.test_data 
                    if d['test_name'] == test_name]
        
        if not test_data:
            return {"error": "No data available for test"}
        
        # Calculate statistics
        df = pd.DataFrame(test_data)
        
        insights = {
            "test_name": test_name,
            "total_executions": len(test_data),
            "success_rate": df['success'].mean(),
            "avg_duration": df['duration'].mean(),
            "duration_std": df['duration'].std(),
            "avg_memory_usage": df['memory_usage'].mean(),
            "avg_cpu_usage": df['cpu_usage'].mean(),
            "last_execution": max(test_data, key=lambda x: x['timestamp'])['timestamp'],
            "trends": {
                "success_trend": self._calculate_trend(df['success'].astype(int)),
                "duration_trend": self._calculate_trend(df['duration'])
            },
            "recommendations": self.test_generator.suggest_test_improvements(test_name)
        }
        
        return insights
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status."""
        return {
            "is_trained": self.is_trained,
            "total_data_points": len(self.data_collector.test_data),
            "learned_patterns": len(self.predictive_tester.pattern_learner.patterns),
            "model_file_exists": self.predictive_tester.model_file.exists(),
            "last_training": self._get_last_training_time()
        }
    
    def _get_last_training_time(self) -> Optional[str]:
        """Get last training time."""
        if self.predictive_tester.model_file.exists():
            try:
                model_data = joblib.load(self.predictive_tester.model_file)
                return model_data.get('timestamp')
            except:
                return None
        return None

# Example usage and demo
def demo_ai_testing():
    """Demonstrate AI testing capabilities."""
    print("🤖 AI Testing Framework Demo")
    print("=" * 40)
    
    # Create AI framework
    ai_framework = AITestingFramework()
    
    # Simulate some test data
    test_names = ["test_basic", "test_performance", "test_integration"]
    
    print("📊 Generating simulated test data...")
    for i in range(50):
        test_name = test_names[i % len(test_names)]
        duration = np.random.normal(1.0, 0.3)
        success = np.random.random() > 0.1  # 90% success rate
        features = {
            "memory_usage": np.random.normal(50, 10),
            "cpu_usage": np.random.normal(30, 5)
        }
        
        ai_framework.data_collector.add_test_execution(
            test_name, duration, success, features
        )
    
    # Train the framework
    ai_framework.train()
    
    # Analyze a test
    print("\n🔍 Analyzing test execution...")
    result = ai_framework.analyze_test(
        "test_basic", 1.2, True, 
        {"memory_usage": 45, "cpu_usage": 25}
    )
    
    print(f"Test: {result.test_name}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Anomaly Score: {result.anomaly_score:.2f}")
    print("Recommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    # Get test insights
    print(f"\n📈 Test Insights for 'test_basic':")
    insights = ai_framework.get_test_insights("test_basic")
    for key, value in insights.items():
        if key != "recommendations":
            print(f"  {key}: {value}")
    
    print("\n🎯 Framework Status:")
    status = ai_framework.get_framework_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    # Run demo
    demo_ai_testing()
