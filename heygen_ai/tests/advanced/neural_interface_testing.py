"""
Neural Interface Testing Framework for HeyGen AI Testing System.
Advanced brain-computer interface testing including neural signal processing,
cognitive load testing, and direct neural feedback validation.
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
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class NeuralSignal:
    """Represents a neural signal."""
    signal_id: str
    timestamp: datetime
    raw_data: np.ndarray
    sampling_rate: float
    channels: List[str]
    frequency_bands: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    quality_score: float = 0.0

@dataclass
class CognitiveState:
    """Represents a cognitive state."""
    state_id: str
    timestamp: datetime
    attention_level: float
    mental_workload: float
    stress_level: float
    fatigue_level: float
    emotional_state: str
    confidence: float

@dataclass
class NeuralTestResult:
    """Represents a neural interface test result."""
    result_id: str
    test_name: str
    test_type: str
    success: bool
    neural_metrics: Dict[str, float]
    cognitive_metrics: Dict[str, float]
    signal_quality: float
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class NeuralSignalProcessor:
    """Processes neural signals for testing."""
    
    def __init__(self):
        self.sampling_rate = 1000.0  # Hz
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def process_signal(self, raw_data: np.ndarray, channels: List[str]) -> NeuralSignal:
        """Process raw neural signal data."""
        # Apply preprocessing
        filtered_data = self._apply_filters(raw_data)
        
        # Extract frequency bands
        frequency_bands = self._extract_frequency_bands(filtered_data)
        
        # Detect artifacts
        artifacts = self._detect_artifacts(filtered_data)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(filtered_data, artifacts)
        
        signal = NeuralSignal(
            signal_id=f"signal_{int(time.time())}_{random.randint(1000, 9999)}",
            timestamp=datetime.now(),
            raw_data=filtered_data,
            sampling_rate=self.sampling_rate,
            channels=channels,
            frequency_bands=frequency_bands,
            artifacts=artifacts,
            quality_score=quality_score
        )
        
        return signal
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply signal processing filters."""
        # High-pass filter (0.5 Hz)
        b, a = signal.butter(4, 0.5, btype='high', fs=self.sampling_rate)
        filtered = signal.filtfilt(b, a, data, axis=0)
        
        # Low-pass filter (100 Hz)
        b, a = signal.butter(4, 100, btype='low', fs=self.sampling_rate)
        filtered = signal.filtfilt(b, a, filtered, axis=0)
        
        # Notch filter (50 Hz power line)
        b, a = signal.iirnotch(50, 30, fs=self.sampling_rate)
        filtered = signal.filtfilt(b, a, filtered, axis=0)
        
        return filtered
    
    def _extract_frequency_bands(self, data: np.ndarray) -> Dict[str, float]:
        """Extract power in different frequency bands."""
        bands = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Calculate power spectral density
            freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=1024)
            
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Calculate band power
            band_power = np.mean(psd[freq_mask])
            bands[band_name] = band_power
        
        return bands
    
    def _detect_artifacts(self, data: np.ndarray) -> List[str]:
        """Detect signal artifacts."""
        artifacts = []
        
        # Check for high amplitude artifacts
        threshold = 3 * np.std(data)
        if np.max(np.abs(data)) > threshold:
            artifacts.append("high_amplitude")
        
        # Check for flat signals
        if np.std(data) < 0.1:
            artifacts.append("flat_signal")
        
        # Check for high frequency noise
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=1024)
        high_freq_power = np.mean(psd[freqs > 50])
        if high_freq_power > 0.1:
            artifacts.append("high_frequency_noise")
        
        return artifacts
    
    def _calculate_quality_score(self, data: np.ndarray, artifacts: List[str]) -> float:
        """Calculate signal quality score."""
        base_score = 1.0
        
        # Penalize artifacts
        for artifact in artifacts:
            if artifact == "high_amplitude":
                base_score -= 0.3
            elif artifact == "flat_signal":
                base_score -= 0.5
            elif artifact == "high_frequency_noise":
                base_score -= 0.2
        
        # Reward good signal characteristics
        snr = np.std(data) / (np.std(data) + 1e-10)
        base_score += min(0.2, snr * 0.1)
        
        return max(0.0, min(1.0, base_score))

class CognitiveStateAnalyzer:
    """Analyzes cognitive states from neural signals."""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize cognitive state models."""
        # Simple models for demonstration
        self.models = {
            'attention': self._attention_model,
            'workload': self._workload_model,
            'stress': self._stress_model,
            'fatigue': self._fatigue_model,
            'emotion': self._emotion_model
        }
    
    def analyze_cognitive_state(self, neural_signal: NeuralSignal) -> CognitiveState:
        """Analyze cognitive state from neural signal."""
        # Extract features
        features = self._extract_cognitive_features(neural_signal)
        
        # Predict cognitive states
        attention = self.models['attention'](features)
        workload = self.models['workload'](features)
        stress = self.models['stress'](features)
        fatigue = self.models['fatigue'](features)
        emotion = self.models['emotion'](features)
        
        # Calculate confidence
        confidence = self._calculate_confidence(features)
        
        state = CognitiveState(
            state_id=f"state_{int(time.time())}_{random.randint(1000, 9999)}",
            timestamp=datetime.now(),
            attention_level=attention,
            mental_workload=workload,
            stress_level=stress,
            fatigue_level=fatigue,
            emotional_state=emotion,
            confidence=confidence
        )
        
        return state
    
    def _extract_cognitive_features(self, neural_signal: NeuralSignal) -> Dict[str, float]:
        """Extract features for cognitive state analysis."""
        features = {}
        
        # Alpha/beta ratio (attention indicator)
        alpha_power = neural_signal.frequency_bands.get('alpha', 0)
        beta_power = neural_signal.frequency_bands.get('beta', 0)
        features['alpha_beta_ratio'] = alpha_power / (beta_power + 1e-10)
        
        # Theta power (workload indicator)
        features['theta_power'] = neural_signal.frequency_bands.get('theta', 0)
        
        # Gamma power (stress indicator)
        features['gamma_power'] = neural_signal.frequency_bands.get('gamma', 0)
        
        # Signal variability (fatigue indicator)
        features['signal_variability'] = np.std(neural_signal.raw_data)
        
        # Signal quality
        features['signal_quality'] = neural_signal.quality_score
        
        return features
    
    def _attention_model(self, features: Dict[str, float]) -> float:
        """Model for attention level."""
        alpha_beta_ratio = features.get('alpha_beta_ratio', 0)
        signal_quality = features.get('signal_quality', 0)
        
        # Higher alpha/beta ratio and quality = higher attention
        attention = min(1.0, alpha_beta_ratio * 0.5 + signal_quality * 0.5)
        return attention
    
    def _workload_model(self, features: Dict[str, float]) -> float:
        """Model for mental workload."""
        theta_power = features.get('theta_power', 0)
        signal_quality = features.get('signal_quality', 0)
        
        # Higher theta power = higher workload
        workload = min(1.0, theta_power * 0.3 + (1 - signal_quality) * 0.2)
        return workload
    
    def _stress_model(self, features: Dict[str, float]) -> float:
        """Model for stress level."""
        gamma_power = features.get('gamma_power', 0)
        signal_variability = features.get('signal_variability', 0)
        
        # Higher gamma and variability = higher stress
        stress = min(1.0, gamma_power * 0.4 + signal_variability * 0.1)
        return stress
    
    def _fatigue_model(self, features: Dict[str, float]) -> float:
        """Model for fatigue level."""
        signal_variability = features.get('signal_variability', 0)
        signal_quality = features.get('signal_quality', 0)
        
        # Lower variability and quality = higher fatigue
        fatigue = min(1.0, (1 - signal_variability) * 0.5 + (1 - signal_quality) * 0.3)
        return fatigue
    
    def _emotion_model(self, features: Dict[str, float]) -> str:
        """Model for emotional state."""
        alpha_power = features.get('alpha_power', 0)
        beta_power = features.get('beta_power', 0)
        gamma_power = features.get('gamma_power', 0)
        
        # Simple emotion classification
        if alpha_power > beta_power and alpha_power > gamma_power:
            return "calm"
        elif beta_power > alpha_power and beta_power > gamma_power:
            return "focused"
        elif gamma_power > alpha_power and gamma_power > beta_power:
            return "excited"
        else:
            return "neutral"
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in cognitive state analysis."""
        signal_quality = features.get('signal_quality', 0)
        feature_completeness = len([f for f in features.values() if f > 0]) / len(features)
        
        confidence = (signal_quality + feature_completeness) / 2
        return min(1.0, max(0.0, confidence))

class NeuralInterfaceTester:
    """Tests neural interface systems."""
    
    def __init__(self):
        self.signal_processor = NeuralSignalProcessor()
        self.cognitive_analyzer = CognitiveStateAnalyzer()
        self.test_results = []
    
    def test_signal_quality(self, duration: float = 30.0) -> NeuralTestResult:
        """Test neural signal quality."""
        # Simulate neural signal acquisition
        signals = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Generate synthetic neural data
            raw_data = self._generate_synthetic_neural_data()
            channels = ['Fz', 'Cz', 'Pz', 'Oz']
            
            # Process signal
            signal = self.signal_processor.process_signal(raw_data, channels)
            signals.append(signal)
            
            time.sleep(0.1)  # 10 Hz sampling
        
        # Analyze signal quality
        quality_scores = [s.quality_score for s in signals]
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        
        # Count artifacts
        total_artifacts = sum(len(s.artifacts) for s in signals)
        artifact_rate = total_artifacts / len(signals) if signals else 0
        
        metrics = {
            'average_quality': avg_quality,
            'quality_std': quality_std,
            'artifact_rate': artifact_rate,
            'total_signals': len(signals),
            'duration': duration
        }
        
        result = NeuralTestResult(
            result_id=f"signal_quality_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Neural Signal Quality Test",
            test_type="signal_quality",
            success=avg_quality > 0.7,
            neural_metrics=metrics,
            cognitive_metrics={},
            signal_quality=avg_quality
        )
        
        self.test_results.append(result)
        return result
    
    def test_cognitive_load(self, task_difficulty: str = "medium") -> NeuralTestResult:
        """Test cognitive load during tasks."""
        # Simulate different task difficulties
        difficulty_levels = {
            'easy': {'workload': 0.3, 'attention': 0.7},
            'medium': {'workload': 0.6, 'attention': 0.5},
            'hard': {'workload': 0.8, 'attention': 0.3}
        }
        
        target_levels = difficulty_levels.get(task_difficulty, difficulty_levels['medium'])
        
        # Generate signals with different cognitive loads
        signals = []
        for i in range(50):  # 50 samples
            raw_data = self._generate_synthetic_neural_data(cognitive_load=target_levels['workload'])
            channels = ['Fz', 'Cz', 'Pz', 'Oz']
            
            signal = self.signal_processor.process_signal(raw_data, channels)
            signals.append(signal)
        
        # Analyze cognitive states
        cognitive_states = []
        for signal in signals:
            state = self.cognitive_analyzer.analyze_cognitive_state(signal)
            cognitive_states.append(state)
        
        # Calculate metrics
        attention_levels = [s.attention_level for s in cognitive_states]
        workload_levels = [s.mental_workload for s in cognitive_states]
        stress_levels = [s.stress_level for s in cognitive_states]
        
        avg_attention = np.mean(attention_levels)
        avg_workload = np.mean(workload_levels)
        avg_stress = np.mean(stress_levels)
        
        # Calculate cognitive load accuracy
        workload_accuracy = 1 - abs(avg_workload - target_levels['workload'])
        attention_accuracy = 1 - abs(avg_attention - target_levels['attention'])
        
        metrics = {
            'average_attention': avg_attention,
            'average_workload': avg_workload,
            'average_stress': avg_stress,
            'workload_accuracy': workload_accuracy,
            'attention_accuracy': attention_accuracy,
            'task_difficulty': task_difficulty
        }
        
        result = NeuralTestResult(
            result_id=f"cognitive_load_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Cognitive Load Test",
            test_type="cognitive_load",
            success=workload_accuracy > 0.7 and attention_accuracy > 0.7,
            neural_metrics={},
            cognitive_metrics=metrics,
            signal_quality=np.mean([s.quality_score for s in signals])
        )
        
        self.test_results.append(result)
        return result
    
    def test_neural_feedback(self, feedback_type: str = "visual") -> NeuralTestResult:
        """Test neural feedback systems."""
        # Simulate neural feedback loop
        feedback_accuracy = []
        response_times = []
        
        for trial in range(20):  # 20 trials
            # Generate initial neural state
            raw_data = self._generate_synthetic_neural_data()
            channels = ['Fz', 'Cz', 'Pz', 'Oz']
            signal = self.signal_processor.process_signal(raw_data, channels)
            state = self.cognitive_analyzer.analyze_cognitive_state(signal)
            
            # Simulate feedback presentation
            start_time = time.time()
            
            # Simulate neural response to feedback
            response_data = self._generate_synthetic_neural_data(
                cognitive_load=state.mental_workload + 0.1
            )
            response_signal = self.signal_processor.process_signal(response_data, channels)
            response_state = self.cognitive_analyzer.analyze_cognitive_state(response_signal)
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Calculate feedback accuracy
            state_change = abs(response_state.attention_level - state.attention_level)
            feedback_accuracy.append(state_change)
        
        # Calculate metrics
        avg_accuracy = np.mean(feedback_accuracy)
        avg_response_time = np.mean(response_times)
        response_consistency = 1 - np.std(response_times) / np.mean(response_times)
        
        metrics = {
            'feedback_accuracy': avg_accuracy,
            'average_response_time': avg_response_time,
            'response_consistency': response_consistency,
            'feedback_type': feedback_type,
            'total_trials': 20
        }
        
        result = NeuralTestResult(
            result_id=f"neural_feedback_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Neural Feedback Test",
            test_type="neural_feedback",
            success=avg_accuracy > 0.5 and avg_response_time < 2.0,
            neural_metrics={},
            cognitive_metrics=metrics,
            signal_quality=0.8  # Simulated
        )
        
        self.test_results.append(result)
        return result
    
    def _generate_synthetic_neural_data(self, cognitive_load: float = 0.5) -> np.ndarray:
        """Generate synthetic neural data for testing."""
        duration = 1.0  # 1 second
        samples = int(self.signal_processor.sampling_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Generate base signal
        signal = np.zeros(samples)
        
        # Add frequency components based on cognitive load
        if cognitive_load < 0.3:  # Low load - more alpha
            signal += 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha
        elif cognitive_load < 0.7:  # Medium load - more beta
            signal += 0.3 * np.sin(2 * np.pi * 10 * t)  # Alpha
            signal += 0.4 * np.sin(2 * np.pi * 20 * t)  # Beta
        else:  # High load - more gamma
            signal += 0.2 * np.sin(2 * np.pi * 10 * t)  # Alpha
            signal += 0.3 * np.sin(2 * np.pi * 20 * t)  # Beta
            signal += 0.5 * np.sin(2 * np.pi * 40 * t)  # Gamma
        
        # Add noise
        noise = np.random.normal(0, 0.1, samples)
        signal += noise
        
        # Add artifacts occasionally
        if random.random() < 0.1:  # 10% chance of artifact
            artifact_start = random.randint(0, samples // 2)
            artifact_end = artifact_start + random.randint(10, 50)
            signal[artifact_start:artifact_end] += np.random.normal(0, 2, artifact_end - artifact_start)
        
        return signal
    
    def generate_neural_report(self) -> Dict[str, Any]:
        """Generate comprehensive neural interface test report."""
        if not self.test_results:
            return {"message": "No test results available"}
        
        # Analyze results by type
        test_types = {}
        for result in self.test_results:
            if result.test_type not in test_types:
                test_types[result.test_type] = []
            test_types[result.test_type].append(result)
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        avg_signal_quality = np.mean([r.signal_quality for r in self.test_results])
        
        # Performance analysis
        performance_analysis = self._analyze_neural_performance()
        
        # Generate recommendations
        recommendations = self._generate_neural_recommendations()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_signal_quality": avg_signal_quality
            },
            "by_test_type": {test_type: len(results) for test_type, results in test_types.items()},
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": [r.__dict__ for r in self.test_results]
        }
    
    def _analyze_neural_performance(self) -> Dict[str, Any]:
        """Analyze neural interface performance."""
        all_neural_metrics = [r.neural_metrics for r in self.test_results if r.neural_metrics]
        all_cognitive_metrics = [r.cognitive_metrics for r in self.test_results if r.cognitive_metrics]
        
        analysis = {}
        
        if all_neural_metrics:
            # Aggregate neural metrics
            neural_aggregated = {}
            for metrics in all_neural_metrics:
                for metric_name, value in metrics.items():
                    if metric_name not in neural_aggregated:
                        neural_aggregated[metric_name] = []
                    neural_aggregated[metric_name].append(value)
            
            analysis["neural_metrics"] = {
                metric_name: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric_name, values in neural_aggregated.items()
            }
        
        if all_cognitive_metrics:
            # Aggregate cognitive metrics
            cognitive_aggregated = {}
            for metrics in all_cognitive_metrics:
                for metric_name, value in metrics.items():
                    if metric_name not in cognitive_aggregated:
                        cognitive_aggregated[metric_name] = []
                    cognitive_aggregated[metric_name].append(value)
            
            analysis["cognitive_metrics"] = {
                metric_name: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric_name, values in cognitive_aggregated.items()
            }
        
        return analysis
    
    def _generate_neural_recommendations(self) -> List[str]:
        """Generate neural interface specific recommendations."""
        recommendations = []
        
        # Analyze signal quality
        signal_quality_results = [r for r in self.test_results if r.test_type == "signal_quality"]
        if signal_quality_results:
            avg_quality = np.mean([r.signal_quality for r in signal_quality_results])
            if avg_quality < 0.8:
                recommendations.append("Improve signal acquisition quality through better electrode placement and noise reduction")
        
        # Analyze cognitive load results
        cognitive_load_results = [r for r in self.test_results if r.test_type == "cognitive_load"]
        if cognitive_load_results:
            avg_accuracy = np.mean([r.cognitive_metrics.get('workload_accuracy', 0) for r in cognitive_load_results])
            if avg_accuracy < 0.8:
                recommendations.append("Enhance cognitive state detection algorithms for better accuracy")
        
        # Analyze neural feedback results
        feedback_results = [r for r in self.test_results if r.test_type == "neural_feedback"]
        if feedback_results:
            avg_response_time = np.mean([r.cognitive_metrics.get('average_response_time', 0) for r in feedback_results])
            if avg_response_time > 1.5:
                recommendations.append("Optimize neural feedback response time for better user experience")
        
        return recommendations

# Example usage and demo
def demo_neural_interface_testing():
    """Demonstrate neural interface testing capabilities."""
    print("🧠 Neural Interface Testing Framework Demo")
    print("=" * 50)
    
    # Create neural interface tester
    tester = NeuralInterfaceTester()
    
    # Run comprehensive tests
    print("🧪 Running neural interface tests...")
    
    # Test signal quality
    print("\n📡 Testing signal quality...")
    signal_result = tester.test_signal_quality(duration=10.0)
    print(f"Signal Quality Test: {'✅' if signal_result.success else '❌'}")
    print(f"  Average Quality: {signal_result.signal_quality:.3f}")
    print(f"  Artifact Rate: {signal_result.neural_metrics.get('artifact_rate', 0):.3f}")
    
    # Test cognitive load
    print("\n🧠 Testing cognitive load...")
    for difficulty in ['easy', 'medium', 'hard']:
        cognitive_result = tester.test_cognitive_load(difficulty)
        print(f"Cognitive Load Test ({difficulty}): {'✅' if cognitive_result.success else '❌'}")
        print(f"  Workload Accuracy: {cognitive_result.cognitive_metrics.get('workload_accuracy', 0):.3f}")
        print(f"  Attention Accuracy: {cognitive_result.cognitive_metrics.get('attention_accuracy', 0):.3f}")
    
    # Test neural feedback
    print("\n🔄 Testing neural feedback...")
    feedback_result = tester.test_neural_feedback("visual")
    print(f"Neural Feedback Test: {'✅' if feedback_result.success else '❌'}")
    print(f"  Feedback Accuracy: {feedback_result.cognitive_metrics.get('feedback_accuracy', 0):.3f}")
    print(f"  Response Time: {feedback_result.cognitive_metrics.get('average_response_time', 0):.3f}s")
    
    # Generate comprehensive report
    print("\n📈 Generating neural interface report...")
    report = tester.generate_neural_report()
    
    print(f"\n📊 Neural Interface Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"  Average Signal Quality: {report['summary']['average_signal_quality']:.3f}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    # Run demo
    demo_neural_interface_testing()
