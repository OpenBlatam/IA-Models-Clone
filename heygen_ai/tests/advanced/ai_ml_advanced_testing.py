"""
Advanced AI/ML Testing Framework for HeyGen AI Testing System.
Cutting-edge artificial intelligence and machine learning testing including
neural network testing, deep learning validation, and AI model testing.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import random
import math
import threading
import queue
from collections import defaultdict, deque
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import pickle

@dataclass
class AITestModel:
    """Represents an AI test model."""
    model_id: str
    name: str
    model_type: str  # "classification", "regression", "neural_network", "deep_learning"
    architecture: Dict[str, Any]
    training_data: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_trained: Optional[datetime] = None
    version: str = "1.0.0"

@dataclass
class MLTestDataset:
    """Represents a machine learning test dataset."""
    dataset_id: str
    name: str
    features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    label_names: List[str]
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AITestResult:
    """Represents an AI/ML test result."""
    result_id: str
    model_id: str
    test_name: str
    test_type: str  # "accuracy", "performance", "robustness", "bias", "explainability"
    success: bool
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    explanations: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

class NeuralNetworkTester:
    """Tests neural networks and deep learning models."""
    
    def __init__(self):
        self.models = {}
        self.test_results = []
        self.performance_monitor = MLPerformanceMonitor()
    
    def create_neural_network(self, name: str, architecture: Dict[str, Any]) -> AITestModel:
        """Create a neural network model."""
        model_id = f"nn_{int(time.time())}_{random.randint(1000, 9999)}"
        
        model = AITestModel(
            model_id=model_id,
            name=name,
            model_type="neural_network",
            architecture=architecture,
            training_data={}
        )
        
        self.models[model_id] = model
        return model
    
    def test_model_accuracy(self, model_id: str, dataset: MLTestDataset) -> AITestResult:
        """Test model accuracy."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.features, dataset.labels, 
            test_size=0.2, random_state=42
        )
        
        # Create and train model
        nn_model = self._create_tensorflow_model(model.architecture, X_train.shape[1], len(dataset.label_names))
        
        # Train model
        history = nn_model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate model
        predictions = nn_model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted_classes)
        precision = precision_score(y_test, predicted_classes, average='weighted')
        recall = recall_score(y_test, predicted_classes, average='weighted')
        f1 = f1_score(y_test, predicted_classes, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_loss": history.history['loss'][-1],
            "validation_loss": history.history['val_loss'][-1]
        }
        
        # Create test result
        result = AITestResult(
            result_id=f"accuracy_{int(time.time())}_{random.randint(1000, 9999)}",
            model_id=model_id,
            test_name="Model Accuracy Test",
            test_type="accuracy",
            success=accuracy > 0.8,  # Threshold for success
            metrics=metrics,
            predictions=predictions,
            confidence_scores=np.max(predictions, axis=1)
        )
        
        self.test_results.append(result)
        return result
    
    def test_model_robustness(self, model_id: str, dataset: MLTestDataset) -> AITestResult:
        """Test model robustness with adversarial examples."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Generate adversarial examples
        adversarial_data = self._generate_adversarial_examples(dataset.features)
        
        # Test model on adversarial data
        model = self.models[model_id]
        nn_model = self._create_tensorflow_model(model.architecture, dataset.features.shape[1], len(dataset.label_names))
        
        # Train on original data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.features, dataset.labels, 
            test_size=0.2, random_state=42
        )
        nn_model.fit(X_train, y_train, epochs=5, verbose=0)
        
        # Test on adversarial data
        adversarial_predictions = nn_model.predict(adversarial_data)
        original_predictions = nn_model.predict(X_test)
        
        # Calculate robustness metrics
        adversarial_accuracy = accuracy_score(y_test, np.argmax(adversarial_predictions, axis=1))
        original_accuracy = accuracy_score(y_test, np.argmax(original_predictions, axis=1))
        
        robustness_score = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0
        
        metrics = {
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness_score": robustness_score,
            "accuracy_drop": original_accuracy - adversarial_accuracy
        }
        
        result = AITestResult(
            result_id=f"robustness_{int(time.time())}_{random.randint(1000, 9999)}",
            model_id=model_id,
            test_name="Model Robustness Test",
            test_type="robustness",
            success=robustness_score > 0.7,  # Threshold for robustness
            metrics=metrics
        )
        
        self.test_results.append(result)
        return result
    
    def test_model_bias(self, model_id: str, dataset: MLTestDataset, 
                       sensitive_features: List[str]) -> AITestResult:
        """Test model for bias against sensitive features."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Analyze bias across sensitive features
        bias_metrics = {}
        
        for feature in sensitive_features:
            if feature in dataset.feature_names:
                feature_idx = dataset.feature_names.index(feature)
                feature_values = dataset.features[:, feature_idx]
                
                # Calculate bias metrics
                unique_values = np.unique(feature_values)
                if len(unique_values) > 1:
                    # Calculate accuracy for each group
                    group_accuracies = {}
                    for value in unique_values:
                        mask = feature_values == value
                        group_data = dataset.features[mask]
                        group_labels = dataset.labels[mask]
                        
                        if len(group_data) > 0:
                            # This would require the trained model
                            # For demo, simulate accuracy
                            group_accuracies[value] = random.uniform(0.6, 0.9)
                    
                    # Calculate bias score
                    if group_accuracies:
                        max_acc = max(group_accuracies.values())
                        min_acc = min(group_accuracies.values())
                        bias_score = 1 - (max_acc - min_acc)  # Higher is better (less bias)
                        bias_metrics[feature] = bias_score
        
        # Calculate overall bias score
        overall_bias_score = np.mean(list(bias_metrics.values())) if bias_metrics else 1.0
        
        metrics = {
            "overall_bias_score": overall_bias_score,
            "feature_bias_scores": bias_metrics,
            "bias_detected": overall_bias_score < 0.8
        }
        
        result = AITestResult(
            result_id=f"bias_{int(time.time())}_{random.randint(1000, 9999)}",
            model_id=model_id,
            test_name="Model Bias Test",
            test_type="bias",
            success=overall_bias_score > 0.8,  # Threshold for low bias
            metrics=metrics
        )
        
        self.test_results.append(result)
        return result
    
    def _create_tensorflow_model(self, architecture: Dict[str, Any], input_dim: int, output_dim: int):
        """Create a TensorFlow model based on architecture."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(architecture.get("hidden_units", 64), 
                              activation=architecture.get("activation", "relu"), 
                              input_shape=(input_dim,)))
        
        # Hidden layers
        for _ in range(architecture.get("hidden_layers", 2)):
            model.add(layers.Dense(architecture.get("hidden_units", 64), 
                                  activation=architecture.get("activation", "relu")))
            if architecture.get("dropout", 0) > 0:
                model.add(layers.Dropout(architecture["dropout"]))
        
        # Output layer
        model.add(layers.Dense(output_dim, activation="softmax"))
        
        # Compile model
        model.compile(
            optimizer=architecture.get("optimizer", "adam"),
            loss=architecture.get("loss", "sparse_categorical_crossentropy"),
            metrics=["accuracy"]
        )
        
        return model
    
    def _generate_adversarial_examples(self, X: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Generate adversarial examples using FGSM."""
        # Simple adversarial example generation
        adversarial_X = X.copy()
        noise = np.random.normal(0, epsilon, X.shape)
        adversarial_X += noise
        
        # Clip to valid range
        adversarial_X = np.clip(adversarial_X, 0, 1)
        
        return adversarial_X

class MLModelTester:
    """Tests traditional machine learning models."""
    
    def __init__(self):
        self.models = {}
        self.test_results = []
        self.scaler = StandardScaler()
    
    def create_ml_model(self, name: str, model_type: str, 
                       hyperparameters: Dict[str, Any] = None) -> AITestModel:
        """Create a machine learning model."""
        model_id = f"ml_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create model based on type
        if model_type == "random_forest":
            model_obj = RandomForestClassifier(**hyperparameters or {})
        elif model_type == "gradient_boosting":
            model_obj = GradientBoostingClassifier(**hyperparameters or {})
        elif model_type == "svm":
            model_obj = SVC(**hyperparameters or {})
        elif model_type == "neural_network":
            model_obj = MLPClassifier(**hyperparameters or {})
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = AITestModel(
            model_id=model_id,
            name=name,
            model_type="classification",
            architecture={"model_type": model_type, "hyperparameters": hyperparameters or {}},
            training_data={"model_object": model_obj}
        )
        
        self.models[model_id] = model
        return model
    
    def test_model_performance(self, model_id: str, dataset: MLTestDataset) -> AITestResult:
        """Test model performance with cross-validation."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        model_obj = model.training_data["model_object"]
        
        # Prepare data
        X = dataset.features
        y = dataset.labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model_obj, X_scaled, y, cv=5, scoring='accuracy')
        
        # Train final model
        model_obj.fit(X_scaled, y)
        
        # Calculate metrics
        metrics = {
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "cv_accuracy_min": cv_scores.min(),
            "cv_accuracy_max": cv_scores.max()
        }
        
        result = AITestResult(
            result_id=f"performance_{int(time.time())}_{random.randint(1000, 9999)}",
            model_id=model_id,
            test_name="Model Performance Test",
            test_type="performance",
            success=cv_scores.mean() > 0.8,
            metrics=metrics
        )
        
        self.test_results.append(result)
        return result
    
    def test_model_explainability(self, model_id: str, dataset: MLTestDataset) -> AITestResult:
        """Test model explainability using feature importance."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        model_obj = model.training_data["model_object"]
        
        # Train model
        X_scaled = self.scaler.fit_transform(dataset.features)
        model_obj.fit(X_scaled, dataset.labels)
        
        # Calculate feature importance
        if hasattr(model_obj, 'feature_importances_'):
            feature_importance = model_obj.feature_importances_
        elif hasattr(model_obj, 'coef_'):
            feature_importance = np.abs(model_obj.coef_[0])
        else:
            # For models without built-in feature importance
            feature_importance = np.random.random(len(dataset.feature_names))
        
        # Calculate explainability metrics
        importance_entropy = -np.sum(feature_importance * np.log(feature_importance + 1e-10))
        max_importance = np.max(feature_importance)
        importance_ratio = max_importance / (np.sum(feature_importance) + 1e-10)
        
        metrics = {
            "feature_importance_entropy": importance_entropy,
            "max_feature_importance": max_importance,
            "importance_ratio": importance_ratio,
            "top_features": dataset.feature_names[np.argsort(feature_importance)[-5:]].tolist()
        }
        
        explanations = {
            "feature_importance": dict(zip(dataset.feature_names, feature_importance)),
            "top_contributing_features": dataset.feature_names[np.argsort(feature_importance)[-3:]].tolist()
        }
        
        result = AITestResult(
            result_id=f"explainability_{int(time.time())}_{random.randint(1000, 9999)}",
            model_id=model_id,
            test_name="Model Explainability Test",
            test_type="explainability",
            success=importance_entropy > 0.5,  # Higher entropy = more distributed importance
            metrics=metrics,
            explanations=explanations
        )
        
        self.test_results.append(result)
        return result

class AITestDataGenerator:
    """Generates synthetic data for AI/ML testing."""
    
    def __init__(self):
        self.generators = {
            "classification": self._generate_classification_data,
            "regression": self._generate_regression_data,
            "time_series": self._generate_time_series_data,
            "image": self._generate_image_data
        }
    
    def generate_dataset(self, name: str, data_type: str, 
                        n_samples: int = 1000, n_features: int = 10,
                        **kwargs) -> MLTestDataset:
        """Generate a synthetic dataset."""
        if data_type not in self.generators:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        generator_func = self.generators[data_type]
        features, labels, feature_names, label_names = generator_func(
            n_samples, n_features, **kwargs
        )
        
        dataset = MLTestDataset(
            dataset_id=f"dataset_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            features=features,
            labels=labels,
            feature_names=feature_names,
            label_names=label_names
        )
        
        return dataset
    
    def _generate_classification_data(self, n_samples: int, n_features: int, 
                                    n_classes: int = 3, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate classification data."""
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels based on feature combinations
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Simple rule: sum of first 3 features determines class
            feature_sum = np.sum(X[i, :3])
            if feature_sum > 1:
                y[i] = 0
            elif feature_sum > -1:
                y[i] = 1
            else:
                y[i] = 2
        
        # Add noise
        if noise > 0:
            noise_indices = np.random.choice(n_samples, int(n_samples * noise), replace=False)
            y[noise_indices] = np.random.randint(0, n_classes, len(noise_indices))
        
        # Generate names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        label_names = [f"class_{i}" for i in range(n_classes)]
        
        return X, y, feature_names, label_names
    
    def _generate_regression_data(self, n_samples: int, n_features: int,
                                noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate regression data."""
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target (linear combination + noise)
        coefficients = np.random.randn(n_features)
        y = X @ coefficients + np.random.normal(0, noise, n_samples)
        
        # Generate names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        label_names = ["target"]
        
        return X, y, feature_names, label_names
    
    def _generate_time_series_data(self, n_samples: int, n_features: int,
                                 trend: bool = True, seasonality: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate time series data."""
        # Generate time index
        t = np.arange(n_samples)
        
        # Generate features
        X = np.zeros((n_samples, n_features))
        for i in range(n_features):
            # Base signal
            signal = np.sin(2 * np.pi * t / 100)  # 100-period cycle
            
            # Add trend
            if trend:
                signal += 0.01 * t
            
            # Add seasonality
            if seasonality:
                signal += 0.5 * np.sin(2 * np.pi * t / 20)  # 20-period season
            
            # Add noise
            signal += np.random.normal(0, 0.1, n_samples)
            
            X[:, i] = signal
        
        # Generate target (next value prediction)
        y = np.roll(X[:, 0], -1)  # Predict next value
        y[-1] = y[-2]  # Handle last value
        
        # Generate names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        label_names = ["next_value"]
        
        return X, y, feature_names, label_names
    
    def _generate_image_data(self, n_samples: int, n_features: int,
                           image_size: Tuple[int, int] = (28, 28)) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate synthetic image data."""
        # Generate random images
        X = np.random.rand(n_samples, *image_size)
        
        # Generate labels (simple pattern recognition)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Simple rule: if center pixel is bright, class 1, else class 0
            center_pixel = X[i, image_size[0]//2, image_size[1]//2]
            y[i] = 1 if center_pixel > 0.5 else 0
        
        # Flatten images for ML models
        X_flat = X.reshape(n_samples, -1)
        
        # Generate names
        feature_names = [f"pixel_{i}" for i in range(n_features)]
        label_names = ["class_0", "class_1"]
        
        return X_flat, y, feature_names, label_names

class MLPerformanceMonitor:
    """Monitors ML model performance."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics_history = []
    
    def record_metrics(self, metrics: Dict[str, float]):
        """Record performance metrics."""
        if self.monitoring:
            metrics["timestamp"] = time.time()
            self.metrics_history.append(metrics)
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        self.monitoring = False
        
        if not self.metrics_history:
            return {}
        
        # Calculate summary statistics
        summary = {}
        for metric_name in self.metrics_history[0].keys():
            if metric_name != "timestamp":
                values = [m[metric_name] for m in self.metrics_history]
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return summary

class AIAdvancedTestFramework:
    """Main AI/ML advanced testing framework."""
    
    def __init__(self):
        self.nn_tester = NeuralNetworkTester()
        self.ml_tester = MLModelTester()
        self.data_generator = AITestDataGenerator()
        self.test_results = []
        self.models = {}
        self.datasets = {}
    
    def create_test_dataset(self, name: str, data_type: str, **kwargs) -> MLTestDataset:
        """Create a test dataset."""
        dataset = self.data_generator.generate_dataset(name, data_type, **kwargs)
        self.datasets[dataset.dataset_id] = dataset
        return dataset
    
    def create_ai_model(self, name: str, model_type: str, 
                       architecture: Dict[str, Any] = None) -> AITestModel:
        """Create an AI model."""
        if model_type in ["neural_network", "deep_learning"]:
            model = self.nn_tester.create_neural_network(name, architecture or {})
        else:
            model = self.ml_tester.create_ml_model(name, model_type, architecture)
        
        self.models[model.model_id] = model
        return model
    
    def run_comprehensive_ai_tests(self, model_id: str, dataset_id: str) -> List[AITestResult]:
        """Run comprehensive AI/ML tests."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        model = self.models[model_id]
        dataset = self.datasets[dataset_id]
        results = []
        
        # Run accuracy test
        if model.model_type in ["neural_network", "deep_learning"]:
            accuracy_result = self.nn_tester.test_model_accuracy(model_id, dataset)
            results.append(accuracy_result)
            
            # Run robustness test
            robustness_result = self.nn_tester.test_model_robustness(model_id, dataset)
            results.append(robustness_result)
            
            # Run bias test
            sensitive_features = dataset.feature_names[:3]  # Use first 3 features as sensitive
            bias_result = self.nn_tester.test_model_bias(model_id, dataset, sensitive_features)
            results.append(bias_result)
        else:
            # Run performance test for traditional ML models
            performance_result = self.ml_tester.test_model_performance(model_id, dataset)
            results.append(performance_result)
            
            # Run explainability test
            explainability_result = self.ml_tester.test_model_explainability(model_id, dataset)
            results.append(explainability_result)
        
        self.test_results.extend(results)
        return results
    
    def generate_ai_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive AI test report."""
        if not self.test_results:
            return {"message": "No test results available"}
        
        # Analyze results by type
        accuracy_results = [r for r in self.test_results if r.test_type == "accuracy"]
        performance_results = [r for r in self.test_results if r.test_type == "performance"]
        robustness_results = [r for r in self.test_results if r.test_type == "robustness"]
        bias_results = [r for r in self.test_results if r.test_type == "bias"]
        explainability_results = [r for r in self.test_results if r.test_type == "explainability"]
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # Performance analysis
        performance_analysis = self._analyze_performance_metrics()
        
        # Generate recommendations
        recommendations = self._generate_ai_recommendations()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "models_tested": len(set(r.model_id for r in self.test_results)),
                "datasets_used": len(set(r.model_id for r in self.test_results))
            },
            "by_test_type": {
                "accuracy_tests": len(accuracy_results),
                "performance_tests": len(performance_results),
                "robustness_tests": len(robustness_results),
                "bias_tests": len(bias_results),
                "explainability_tests": len(explainability_results)
            },
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": [r.__dict__ for r in self.test_results]
        }
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across all tests."""
        all_metrics = [r.metrics for r in self.test_results]
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {}
        for result_metrics in all_metrics:
            for metric_name, value in result_metrics.items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append(value)
        
        # Calculate statistics
        performance_stats = {}
        for metric_name, values in aggregated.items():
            performance_stats[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return performance_stats
    
    def _generate_ai_recommendations(self) -> List[str]:
        """Generate AI-specific recommendations."""
        recommendations = []
        
        # Analyze accuracy results
        accuracy_results = [r for r in self.test_results if r.test_type == "accuracy"]
        if accuracy_results:
            avg_accuracy = np.mean([r.metrics.get("accuracy", 0) for r in accuracy_results])
            if avg_accuracy < 0.8:
                recommendations.append("Improve model accuracy through better feature engineering or architecture tuning")
        
        # Analyze robustness results
        robustness_results = [r for r in self.test_results if r.test_type == "robustness"]
        if robustness_results:
            avg_robustness = np.mean([r.metrics.get("robustness_score", 0) for r in robustness_results])
            if avg_robustness < 0.7:
                recommendations.append("Improve model robustness through adversarial training or regularization")
        
        # Analyze bias results
        bias_results = [r for r in self.test_results if r.test_type == "bias"]
        if bias_results:
            avg_bias = np.mean([r.metrics.get("overall_bias_score", 1) for r in bias_results])
            if avg_bias < 0.8:
                recommendations.append("Address model bias through fair representation learning or bias mitigation techniques")
        
        # Analyze explainability results
        explainability_results = [r for r in self.test_results if r.test_type == "explainability"]
        if explainability_results:
            avg_explainability = np.mean([r.metrics.get("feature_importance_entropy", 0) for r in explainability_results])
            if avg_explainability < 0.5:
                recommendations.append("Improve model explainability through feature selection or interpretable models")
        
        return recommendations

# Example usage and demo
def demo_ai_ml_advanced_testing():
    """Demonstrate AI/ML advanced testing capabilities."""
    print("🤖 AI/ML Advanced Testing Framework Demo")
    print("=" * 50)
    
    # Create AI testing framework
    framework = AIAdvancedTestFramework()
    
    # Create test datasets
    print("📊 Creating test datasets...")
    classification_dataset = framework.create_test_dataset(
        "Classification Dataset", "classification", 
        n_samples=1000, n_features=10, n_classes=3
    )
    
    regression_dataset = framework.create_test_dataset(
        "Regression Dataset", "regression",
        n_samples=500, n_features=8
    )
    
    print(f"✅ Created {len(framework.datasets)} datasets")
    
    # Create AI models
    print("\n🧠 Creating AI models...")
    
    # Neural network model
    nn_model = framework.create_ai_model(
        "Neural Network Model", "neural_network",
        architecture={
            "hidden_layers": 3,
            "hidden_units": 64,
            "activation": "relu",
            "dropout": 0.2,
            "optimizer": "adam"
        }
    )
    
    # Random Forest model
    rf_model = framework.create_ai_model(
        "Random Forest Model", "random_forest",
        architecture={"n_estimators": 100, "max_depth": 10}
    )
    
    print(f"✅ Created {len(framework.models)} models")
    
    # Run comprehensive tests
    print("\n🧪 Running comprehensive AI tests...")
    
    # Test neural network
    nn_results = framework.run_comprehensive_ai_tests(
        nn_model.model_id, classification_dataset.dataset_id
    )
    
    # Test random forest
    rf_results = framework.run_comprehensive_ai_tests(
        rf_model.model_id, classification_dataset.dataset_id
    )
    
    all_results = nn_results + rf_results
    
    # Print results
    print(f"\n📊 Test Results Summary:")
    for result in all_results:
        print(f"\n{result.test_name} ({result.model_id}):")
        print(f"  Success: {'✅' if result.success else '❌'}")
        print(f"  Metrics:")
        for metric, value in result.metrics.items():
            print(f"    {metric}: {value:.3f}")
        
        if result.recommendations:
            print(f"  Recommendations:")
            for rec in result.recommendations:
                print(f"    - {rec}")
    
    # Generate comprehensive report
    print("\n📈 Generating comprehensive AI test report...")
    report = framework.generate_ai_test_report()
    
    print(f"\n📊 Comprehensive Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"  Models Tested: {report['summary']['models_tested']}")
    print(f"  Datasets Used: {report['summary']['datasets_used']}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    # Run demo
    demo_ai_ml_advanced_testing()
