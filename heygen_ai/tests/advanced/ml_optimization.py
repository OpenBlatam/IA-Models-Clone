"""
Machine Learning Optimization Framework for HeyGen AI Testing System.
Advanced ML-based test optimization, intelligent test selection, and predictive analytics.
"""

import numpy as np
import pandas as pd
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import threading
import concurrent.futures

@dataclass
class TestExecutionData:
    """Represents test execution data for ML analysis."""
    test_id: str
    test_name: str
    execution_time: float
    success: bool
    duration: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    features: Dict[str, Any] = field(default_factory=dict)
    environment: str = "default"
    test_category: str = "unknown"

@dataclass
class MLPrediction:
    """Represents ML prediction results."""
    test_id: str
    predicted_duration: float
    predicted_success_probability: float
    confidence: float
    risk_score: float
    optimization_suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TestOptimization:
    """Represents test optimization recommendations."""
    test_id: str
    current_score: float
    optimized_score: float
    improvement_percentage: float
    recommendations: List[str] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high, critical

class TestDataCollector:
    """Collects and manages test execution data for ML training."""
    
    def __init__(self, data_file: str = "ml_test_data.json"):
        self.data_file = Path(data_file)
        self.test_data: List[TestExecutionData] = []
        self.load_data()
        self.lock = threading.Lock()
    
    def load_data(self):
        """Load existing test data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.test_data = [
                        TestExecutionData(
                            test_id=item['test_id'],
                            test_name=item['test_name'],
                            execution_time=item['execution_time'],
                            success=item['success'],
                            duration=item['duration'],
                            memory_usage=item['memory_usage'],
                            cpu_usage=item['cpu_usage'],
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            features=item.get('features', {}),
                            environment=item.get('environment', 'default'),
                            test_category=item.get('test_category', 'unknown')
                        )
                        for item in data
                    ]
            except Exception as e:
                logging.error(f"Error loading ML test data: {e}")
                self.test_data = []
    
    def save_data(self):
        """Save test data to file."""
        with self.lock:
            try:
                data = [
                    {
                        'test_id': item.test_id,
                        'test_name': item.test_name,
                        'execution_time': item.execution_time,
                        'success': item.success,
                        'duration': item.duration,
                        'memory_usage': item.memory_usage,
                        'cpu_usage': item.cpu_usage,
                        'timestamp': item.timestamp.isoformat(),
                        'features': item.features,
                        'environment': item.environment,
                        'test_category': item.test_category
                    }
                    for item in self.test_data
                ]
                with open(self.data_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                logging.error(f"Error saving ML test data: {e}")
    
    def add_test_execution(self, test_execution: TestExecutionData):
        """Add a test execution record."""
        with self.lock:
            self.test_data.append(test_execution)
            # Keep only last 50000 records
            if len(self.test_data) > 50000:
                self.test_data = self.test_data[-50000:]
            self.save_data()
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get training data for ML models."""
        if not self.test_data:
            return np.array([]), np.array([]), []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([
            {
                'test_name': item.test_name,
                'execution_time': item.execution_time,
                'success': item.success,
                'duration': item.duration,
                'memory_usage': item.memory_usage,
                'cpu_usage': item.cpu_usage,
                'timestamp': item.timestamp,
                'environment': item.environment,
                'test_category': item.test_category,
                **item.features
            }
            for item in self.test_data
        ])
        
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
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        features.append(df['hour'].values)
        feature_names.append('hour')
        
        features.append(df['day_of_week'].values)
        feature_names.append('day_of_week')
        
        features.append(df['month'].values)
        feature_names.append('month')
        
        # Categorical features (encoded)
        le_test_name = LabelEncoder()
        le_environment = LabelEncoder()
        le_category = LabelEncoder()
        
        features.append(le_test_name.fit_transform(df['test_name']))
        feature_names.append('test_name_encoded')
        
        features.append(le_environment.fit_transform(df['environment']))
        feature_names.append('environment_encoded')
        
        features.append(le_category.fit_transform(df['test_category']))
        feature_names.append('test_category_encoded')
        
        # Historical features
        df_sorted = df.sort_values('timestamp')
        df_sorted['prev_duration'] = df_sorted.groupby('test_name')['duration'].shift(1)
        df_sorted['prev_success'] = df_sorted.groupby('test_name')['success'].shift(1)
        df_sorted['duration_trend'] = df_sorted.groupby('test_name')['duration'].rolling(5).mean().values
        df_sorted['success_rate'] = df_sorted.groupby('test_name')['success'].rolling(10).mean().values
        
        features.append(df_sorted['prev_duration'].fillna(0).values)
        feature_names.append('prev_duration')
        
        features.append(df_sorted['prev_success'].fillna(0).astype(int).values)
        feature_names.append('prev_success')
        
        features.append(df_sorted['duration_trend'].fillna(0).values)
        feature_names.append('duration_trend')
        
        features.append(df_sorted['success_rate'].fillna(0.5).values)
        feature_names.append('success_rate')
        
        # Combine features
        X = np.column_stack(features)
        
        # Target variables
        y_duration = df['duration'].values
        y_success = df['success'].astype(int).values
        
        return X, y_duration, y_success, feature_names

class DurationPredictor:
    """ML model for predicting test execution duration."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the duration prediction model."""
        if len(X) == 0:
            return
        
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate feature importance
        feature_importance = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, feature_importance))
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict test duration and confidence."""
        if not self.is_trained or len(X) == 0:
            return np.array([]), np.array([])
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence based on prediction variance
        # (simplified - in practice, you'd use prediction intervals)
        confidence = np.ones_like(predictions) * 0.8  # Placeholder
        
        return predictions, confidence
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return getattr(self, 'feature_importance', {})

class SuccessPredictor:
    """ML model for predicting test success probability."""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the success prediction model."""
        if len(X) == 0:
            return
        
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict test success probability and confidence."""
        if not self.is_trained or len(X) == 0:
            return np.array([]), np.array([])
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Ensure predictions are in [0, 1] range
        predictions = np.clip(predictions, 0, 1)
        
        # Calculate confidence
        confidence = np.ones_like(predictions) * 0.8  # Placeholder
        
        return predictions, confidence

class TestClustering:
    """Clusters tests based on execution patterns."""
    
    def __init__(self):
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cluster_labels = []
    
    def train(self, X: np.ndarray):
        """Train clustering models."""
        if len(X) == 0:
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-means
        self.kmeans.fit(X_scaled)
        
        # Train DBSCAN
        self.dbscan.fit(X_scaled)
        
        self.is_trained = True
        self.cluster_labels = self.kmeans.labels_
    
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster for new data."""
        if not self.is_trained or len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.is_trained:
            return np.array([])
        return self.kmeans.cluster_centers_

class TestOptimizer:
    """Optimizes test execution based on ML predictions."""
    
    def __init__(self, data_collector: TestDataCollector):
        self.data_collector = data_collector
        self.duration_predictor = DurationPredictor()
        self.success_predictor = SuccessPredictor()
        self.clustering = TestClustering()
        self.is_trained = False
    
    def train_models(self):
        """Train all ML models."""
        print("🤖 Training ML Optimization Models...")
        
        # Get training data
        X, y_duration, y_success, feature_names = self.data_collector.get_training_data()
        
        if len(X) == 0:
            print("⚠️  No training data available")
            return
        
        # Train duration predictor
        self.duration_predictor.train(X, y_duration, feature_names)
        
        # Train success predictor
        self.success_predictor.train(X, y_success, feature_names)
        
        # Train clustering
        self.clustering.train(X)
        
        self.is_trained = True
        
        print(f"✅ Models trained on {len(X)} samples")
        print(f"   Features: {len(feature_names)}")
        print(f"   Duration predictor R²: {self._calculate_r2_score(X, y_duration, self.duration_predictor):.3f}")
        print(f"   Success predictor accuracy: {self._calculate_accuracy(X, y_success, self.success_predictor):.3f}")
    
    def _calculate_r2_score(self, X: np.ndarray, y: np.ndarray, model) -> float:
        """Calculate R² score for regression model."""
        if not model.is_trained or len(X) == 0:
            return 0.0
        
        X_scaled = model.scaler.transform(X)
        predictions = model.model.predict(X_scaled)
        return r2_score(y, predictions)
    
    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray, model) -> float:
        """Calculate accuracy for classification model."""
        if not model.is_trained or len(X) == 0:
            return 0.0
        
        X_scaled = model.scaler.transform(X)
        predictions = model.model.predict(X_scaled)
        predictions_binary = (predictions > 0.5).astype(int)
        return accuracy_score(y, predictions_binary)
    
    def predict_test_performance(self, test_name: str, features: Dict[str, Any] = None) -> MLPrediction:
        """Predict test performance."""
        if not self.is_trained:
            return MLPrediction(
                test_id=test_name,
                predicted_duration=0.0,
                predicted_success_probability=0.5,
                confidence=0.0,
                risk_score=0.5
            )
        
        # Prepare features
        feature_vector = self._prepare_features(test_name, features)
        
        if len(feature_vector) == 0:
            return MLPrediction(
                test_id=test_name,
                predicted_duration=0.0,
                predicted_success_probability=0.5,
                confidence=0.0,
                risk_score=0.5
            )
        
        # Predict duration
        duration_pred, duration_conf = self.duration_predictor.predict(feature_vector.reshape(1, -1))
        
        # Predict success probability
        success_pred, success_conf = self.success_predictor.predict(feature_vector.reshape(1, -1))
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(duration_pred[0], success_pred[0])
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(
            test_name, duration_pred[0], success_pred[0], risk_score
        )
        
        return MLPrediction(
            test_id=test_name,
            predicted_duration=float(duration_pred[0]),
            predicted_success_probability=float(success_pred[0]),
            confidence=float((duration_conf[0] + success_conf[0]) / 2),
            risk_score=float(risk_score),
            optimization_suggestions=suggestions
        )
    
    def _prepare_features(self, test_name: str, features: Dict[str, Any] = None) -> np.ndarray:
        """Prepare features for prediction."""
        if features is None:
            features = {}
        
        # Get historical data for this test
        test_data = [d for d in self.data_collector.test_data if d.test_name == test_name]
        
        # Basic features
        feature_vector = [
            features.get('duration', 1.0),
            features.get('memory_usage', 50.0),
            features.get('cpu_usage', 30.0),
            datetime.now().hour,
            datetime.now().weekday(),
            datetime.now().month
        ]
        
        # Historical features
        if test_data:
            recent_data = test_data[-10:]  # Last 10 executions
            avg_duration = np.mean([d.duration for d in recent_data])
            success_rate = np.mean([d.success for d in recent_data])
            duration_trend = np.polyfit(range(len(recent_data)), [d.duration for d in recent_data], 1)[0] if len(recent_data) > 1 else 0
        else:
            avg_duration = 1.0
            success_rate = 0.5
            duration_trend = 0
        
        feature_vector.extend([avg_duration, success_rate, duration_trend])
        
        # Categorical features (simplified encoding)
        feature_vector.extend([
            hash(test_name) % 1000,  # Simple encoding
            hash(features.get('environment', 'default')) % 100,
            hash(features.get('test_category', 'unknown')) % 100
        ])
        
        return np.array(feature_vector)
    
    def _calculate_risk_score(self, duration: float, success_prob: float) -> float:
        """Calculate risk score for a test."""
        # Higher duration and lower success probability = higher risk
        duration_risk = min(duration / 10.0, 1.0)  # Normalize to [0, 1]
        success_risk = 1.0 - success_prob
        
        return (duration_risk + success_risk) / 2
    
    def _generate_optimization_suggestions(self, test_name: str, duration: float, 
                                         success_prob: float, risk_score: float) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        if duration > 5.0:
            suggestions.append("⚡ Consider optimizing test performance - predicted duration is high")
        
        if success_prob < 0.7:
            suggestions.append("🔧 Improve test reliability - low success probability predicted")
        
        if risk_score > 0.7:
            suggestions.append("⚠️  High risk test - consider refactoring or additional debugging")
        
        if duration < 0.1 and success_prob > 0.9:
            suggestions.append("✅ Test looks well-optimized")
        
        return suggestions
    
    def optimize_test_execution_order(self, test_list: List[str]) -> List[str]:
        """Optimize the order of test execution based on ML predictions."""
        if not self.is_trained or not test_list:
            return test_list
        
        # Get predictions for all tests
        predictions = []
        for test_name in test_list:
            pred = self.predict_test_performance(test_name)
            predictions.append((test_name, pred))
        
        # Sort by risk score (lowest risk first) and success probability (highest first)
        predictions.sort(key=lambda x: (x[1].risk_score, -x[1].predicted_success_probability))
        
        return [pred[0] for pred in predictions]
    
    def get_test_insights(self, test_name: str) -> Dict[str, Any]:
        """Get comprehensive insights for a test."""
        # Get historical data
        test_data = [d for d in self.data_collector.test_data if d.test_name == test_name]
        
        if not test_data:
            return {"error": "No data available for test"}
        
        # Calculate statistics
        durations = [d.duration for d in test_data]
        successes = [d.success for d in test_data]
        
        insights = {
            "test_name": test_name,
            "total_executions": len(test_data),
            "success_rate": np.mean(successes),
            "avg_duration": np.mean(durations),
            "duration_std": np.std(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "last_execution": max(test_data, key=lambda x: x.timestamp).timestamp.isoformat()
        }
        
        # Add ML predictions
        if self.is_trained:
            prediction = self.predict_test_performance(test_name)
            insights.update({
                "predicted_duration": prediction.predicted_duration,
                "predicted_success_probability": prediction.predicted_success_probability,
                "risk_score": prediction.risk_score,
                "optimization_suggestions": prediction.optimization_suggestions
            })
        
        return insights

class MLOptimizationFramework:
    """Main ML optimization framework."""
    
    def __init__(self):
        self.data_collector = TestDataCollector()
        self.optimizer = TestOptimizer(self.data_collector)
        self.model_file = Path("ml_optimization_model.joblib")
        self.load_models()
    
    def load_models(self):
        """Load trained models if available."""
        if self.model_file.exists():
            try:
                model_data = joblib.load(self.model_file)
                self.optimizer.duration_predictor = model_data.get('duration_predictor', self.optimizer.duration_predictor)
                self.optimizer.success_predictor = model_data.get('success_predictor', self.optimizer.success_predictor)
                self.optimizer.clustering = model_data.get('clustering', self.optimizer.clustering)
                self.optimizer.is_trained = model_data.get('is_trained', False)
            except Exception as e:
                logging.error(f"Error loading ML models: {e}")
    
    def save_models(self):
        """Save trained models."""
        try:
            model_data = {
                'duration_predictor': self.optimizer.duration_predictor,
                'success_predictor': self.optimizer.success_predictor,
                'clustering': self.optimizer.clustering,
                'is_trained': self.optimizer.is_trained,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_file)
        except Exception as e:
            logging.error(f"Error saving ML models: {e}")
    
    def train_models(self):
        """Train all ML models."""
        self.optimizer.train_models()
        self.save_models()
    
    def add_test_execution(self, test_name: str, duration: float, success: bool, 
                          memory_usage: float = 0.0, cpu_usage: float = 0.0,
                          features: Dict[str, Any] = None):
        """Add a test execution record."""
        test_execution = TestExecutionData(
            test_id=f"{test_name}_{int(time.time())}",
            test_name=test_name,
            execution_time=time.time(),
            success=success,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=datetime.now(),
            features=features or {}
        )
        
        self.data_collector.add_test_execution(test_execution)
    
    def predict_test_performance(self, test_name: str, features: Dict[str, Any] = None) -> MLPrediction:
        """Predict test performance."""
        return self.optimizer.predict_test_performance(test_name, features)
    
    def optimize_test_order(self, test_list: List[str]) -> List[str]:
        """Optimize test execution order."""
        return self.optimizer.optimize_test_execution_order(test_list)
    
    def get_test_insights(self, test_name: str) -> Dict[str, Any]:
        """Get test insights."""
        return self.optimizer.get_test_insights(test_name)
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.optimizer.is_trained:
            return {"error": "Models not trained"}
        
        # Get all test names
        test_names = list(set(d.test_name for d in self.data_collector.test_data))
        
        # Get predictions for all tests
        predictions = []
        for test_name in test_names:
            pred = self.predict_test_performance(test_name)
            predictions.append(pred)
        
        # Calculate statistics
        avg_duration = np.mean([p.predicted_duration for p in predictions])
        avg_success_prob = np.mean([p.predicted_success_probability for p in predictions])
        avg_risk_score = np.mean([p.risk_score for p in predictions])
        
        # Find high-risk tests
        high_risk_tests = [p for p in predictions if p.risk_score > 0.7]
        
        # Find slow tests
        slow_tests = [p for p in predictions if p.predicted_duration > 5.0]
        
        # Find unreliable tests
        unreliable_tests = [p for p in predictions if p.predicted_success_probability < 0.7]
        
        return {
            "summary": {
                "total_tests": len(test_names),
                "avg_predicted_duration": avg_duration,
                "avg_success_probability": avg_success_prob,
                "avg_risk_score": avg_risk_score,
                "high_risk_tests": len(high_risk_tests),
                "slow_tests": len(slow_tests),
                "unreliable_tests": len(unreliable_tests)
            },
            "high_risk_tests": [
                {
                    "test_id": p.test_id,
                    "predicted_duration": p.predicted_duration,
                    "predicted_success_probability": p.predicted_success_probability,
                    "risk_score": p.risk_score,
                    "suggestions": p.optimization_suggestions
                }
                for p in high_risk_tests
            ],
            "optimization_recommendations": [
                "Focus on high-risk tests first",
                "Optimize slow tests for better performance",
                "Improve reliability of unreliable tests",
                "Consider test execution order optimization"
            ]
        }

# Example usage and demo
def demo_ml_optimization():
    """Demonstrate ML optimization capabilities."""
    print("🤖 ML Optimization Framework Demo")
    print("=" * 40)
    
    # Create ML framework
    ml_framework = MLOptimizationFramework()
    
    # Simulate some test data
    test_names = ["test_basic", "test_performance", "test_integration", "test_security"]
    
    print("📊 Generating simulated test data...")
    for i in range(100):
        test_name = test_names[i % len(test_names)]
        duration = np.random.normal(2.0, 1.0)
        success = np.random.random() > 0.1  # 90% success rate
        memory_usage = np.random.normal(50, 20)
        cpu_usage = np.random.normal(30, 10)
        
        ml_framework.add_test_execution(
            test_name, duration, success, memory_usage, cpu_usage
        )
    
    # Train models
    print("\n🤖 Training ML models...")
    ml_framework.train_models()
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    for test_name in test_names:
        prediction = ml_framework.predict_test_performance(test_name)
        print(f"   {test_name}:")
        print(f"     Predicted duration: {prediction.predicted_duration:.2f}s")
        print(f"     Success probability: {prediction.predicted_success_probability:.2f}")
        print(f"     Risk score: {prediction.risk_score:.2f}")
        print(f"     Suggestions: {len(prediction.optimization_suggestions)}")
    
    # Test optimization
    print("\n⚡ Testing execution order optimization...")
    optimized_order = ml_framework.optimize_test_order(test_names)
    print(f"   Original order: {test_names}")
    print(f"   Optimized order: {optimized_order}")
    
    # Generate report
    print("\n📊 Generating optimization report...")
    report = ml_framework.generate_optimization_report()
    print(f"   Total tests: {report['summary']['total_tests']}")
    print(f"   High risk tests: {report['summary']['high_risk_tests']}")
    print(f"   Slow tests: {report['summary']['slow_tests']}")
    print(f"   Unreliable tests: {report['summary']['unreliable_tests']}")

if __name__ == "__main__":
    # Run demo
    demo_ml_optimization()
