"""
VR/AR Testing Framework for HeyGen AI Testing System.
Advanced virtual and augmented reality testing including 3D environment testing,
haptic feedback validation, and immersive user experience testing.
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
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class VRTestEnvironment:
    """Represents a VR test environment."""
    environment_id: str
    name: str
    dimensions: Tuple[float, float, float]  # width, height, depth
    objects: List[Dict[str, Any]] = field(default_factory=list)
    lighting: Dict[str, Any] = field(default_factory=dict)
    physics: Dict[str, Any] = field(default_factory=dict)
    audio: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ARTestMarker:
    """Represents an AR test marker."""
    marker_id: str
    name: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    marker_type: str  # "image", "qr", "fiducial"
    content: Dict[str, Any] = field(default_factory=dict)
    tracking_accuracy: float = 1.0

@dataclass
class HapticFeedback:
    """Represents haptic feedback data."""
    feedback_id: str
    intensity: float  # 0.0 to 1.0
    duration: float  # seconds
    pattern: str  # "constant", "pulse", "wave", "random"
    frequency: float  # Hz
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class VRTestResult:
    """Represents a VR/AR test result."""
    result_id: str
    test_name: str
    environment_id: str
    test_type: str  # "vr", "ar", "haptic", "interaction"
    success: bool
    performance_metrics: Dict[str, float]
    user_experience_score: float
    technical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class VRTestEngine:
    """VR testing engine for immersive testing."""
    
    def __init__(self):
        self.environments = {}
        self.active_tests = {}
        self.performance_monitor = VRPerformanceMonitor()
        self.interaction_tracker = InteractionTracker()
    
    def create_environment(self, name: str, dimensions: Tuple[float, float, float]) -> VRTestEnvironment:
        """Create a VR test environment."""
        environment = VRTestEnvironment(
            environment_id=f"env_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            dimensions=dimensions,
            lighting={
                "ambient": 0.3,
                "directional": 0.7,
                "color": (1.0, 1.0, 1.0)
            },
            physics={
                "gravity": 9.81,
                "friction": 0.5,
                "bounce": 0.3
            },
            audio={
                "spatial": True,
                "reverb": 0.2,
                "volume": 0.8
            }
        )
        
        self.environments[environment.environment_id] = environment
        return environment
    
    def add_object_to_environment(self, environment_id: str, obj_type: str, 
                                position: Tuple[float, float, float],
                                properties: Dict[str, Any] = None) -> str:
        """Add an object to a VR environment."""
        if environment_id not in self.environments:
            raise ValueError(f"Environment {environment_id} not found")
        
        obj_id = f"obj_{int(time.time())}_{random.randint(1000, 9999)}"
        obj = {
            "id": obj_id,
            "type": obj_type,
            "position": position,
            "properties": properties or {}
        }
        
        self.environments[environment_id].objects.append(obj)
        return obj_id
    
    def run_vr_test(self, test_name: str, environment_id: str, 
                   test_func: Callable, duration: float = 10.0) -> VRTestResult:
        """Run a VR test."""
        if environment_id not in self.environments:
            raise ValueError(f"Environment {environment_id} not found")
        
        environment = self.environments[environment_id]
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Start interaction tracking
        self.interaction_tracker.start_tracking()
        
        # Run test
        start_time = time.time()
        try:
            result = test_func(environment)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
        
        test_duration = time.time() - start_time
        
        # Stop monitoring
        performance_metrics = self.performance_monitor.stop_monitoring()
        interaction_data = self.interaction_tracker.stop_tracking()
        
        # Calculate user experience score
        ux_score = self._calculate_ux_score(performance_metrics, interaction_data, success)
        
        # Generate recommendations
        recommendations = self._generate_vr_recommendations(performance_metrics, interaction_data)
        
        # Create test result
        test_result = VRTestResult(
            result_id=f"vr_result_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name=test_name,
            environment_id=environment_id,
            test_type="vr",
            success=success,
            performance_metrics=performance_metrics,
            user_experience_score=ux_score,
            technical_issues=[error_message] if error_message else [],
            recommendations=recommendations
        )
        
        return test_result
    
    def _calculate_ux_score(self, performance_metrics: Dict[str, float], 
                          interaction_data: Dict[str, Any], success: bool) -> float:
        """Calculate user experience score."""
        base_score = 1.0 if success else 0.0
        
        # Performance factors
        fps = performance_metrics.get("fps", 60)
        latency = performance_metrics.get("latency", 0.0)
        frame_drops = performance_metrics.get("frame_drops", 0)
        
        # FPS score (target: 90 FPS)
        fps_score = min(1.0, fps / 90.0)
        
        # Latency score (target: <20ms)
        latency_score = max(0.0, 1.0 - latency / 0.02)
        
        # Frame drops score
        drops_score = max(0.0, 1.0 - frame_drops / 100.0)
        
        # Interaction score
        interaction_score = interaction_data.get("interaction_success_rate", 1.0)
        
        # Calculate final score
        final_score = base_score * 0.3 + fps_score * 0.25 + latency_score * 0.25 + drops_score * 0.1 + interaction_score * 0.1
        
        return min(1.0, max(0.0, final_score))
    
    def _generate_vr_recommendations(self, performance_metrics: Dict[str, float], 
                                   interaction_data: Dict[str, Any]) -> List[str]:
        """Generate VR-specific recommendations."""
        recommendations = []
        
        fps = performance_metrics.get("fps", 60)
        if fps < 90:
            recommendations.append("Optimize rendering to achieve 90+ FPS for smooth VR experience")
        
        latency = performance_metrics.get("latency", 0.0)
        if latency > 0.02:
            recommendations.append("Reduce input latency to below 20ms for better responsiveness")
        
        frame_drops = performance_metrics.get("frame_drops", 0)
        if frame_drops > 10:
            recommendations.append("Investigate and fix frame drops for stable performance")
        
        interaction_rate = interaction_data.get("interaction_success_rate", 1.0)
        if interaction_rate < 0.9:
            recommendations.append("Improve interaction detection and response accuracy")
        
        return recommendations

class ARTestEngine:
    """AR testing engine for augmented reality testing."""
    
    def __init__(self):
        self.markers = {}
        self.tracking_system = ARTrackingSystem()
        self.overlay_manager = OverlayManager()
        self.camera_simulator = CameraSimulator()
    
    def create_marker(self, name: str, marker_type: str, 
                     position: Tuple[float, float, float],
                     content: Dict[str, Any] = None) -> ARTestMarker:
        """Create an AR test marker."""
        marker = ARTestMarker(
            marker_id=f"marker_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            position=position,
            rotation=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            marker_type=marker_type,
            content=content or {},
            tracking_accuracy=random.uniform(0.8, 1.0)
        )
        
        self.markers[marker.marker_id] = marker
        return marker
    
    def run_ar_test(self, test_name: str, marker_id: str, 
                   test_func: Callable) -> VRTestResult:
        """Run an AR test."""
        if marker_id not in self.markers:
            raise ValueError(f"Marker {marker_id} not found")
        
        marker = self.markers[marker_id]
        
        # Start tracking
        self.tracking_system.start_tracking(marker)
        
        # Start camera simulation
        self.camera_simulator.start_simulation()
        
        # Run test
        start_time = time.time()
        try:
            result = test_func(marker)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
        
        test_duration = time.time() - start_time
        
        # Stop tracking and simulation
        tracking_metrics = self.tracking_system.stop_tracking()
        camera_metrics = self.camera_simulator.stop_simulation()
        
        # Calculate performance metrics
        performance_metrics = {
            "tracking_accuracy": tracking_metrics.get("accuracy", 0.0),
            "tracking_latency": tracking_metrics.get("latency", 0.0),
            "camera_fps": camera_metrics.get("fps", 30),
            "overlay_rendering_time": camera_metrics.get("overlay_time", 0.0)
        }
        
        # Calculate user experience score
        ux_score = self._calculate_ar_ux_score(performance_metrics, success)
        
        # Generate recommendations
        recommendations = self._generate_ar_recommendations(performance_metrics)
        
        # Create test result
        test_result = VRTestResult(
            result_id=f"ar_result_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name=test_name,
            environment_id=marker_id,
            test_type="ar",
            success=success,
            performance_metrics=performance_metrics,
            user_experience_score=ux_score,
            technical_issues=[error_message] if error_message else [],
            recommendations=recommendations
        )
        
        return test_result
    
    def _calculate_ar_ux_score(self, performance_metrics: Dict[str, float], success: bool) -> float:
        """Calculate AR user experience score."""
        base_score = 1.0 if success else 0.0
        
        # Tracking accuracy score
        tracking_accuracy = performance_metrics.get("tracking_accuracy", 0.0)
        accuracy_score = tracking_accuracy
        
        # Latency score
        tracking_latency = performance_metrics.get("tracking_latency", 0.0)
        latency_score = max(0.0, 1.0 - tracking_latency / 0.1)  # Target: <100ms
        
        # Camera FPS score
        camera_fps = performance_metrics.get("camera_fps", 30)
        fps_score = min(1.0, camera_fps / 60.0)  # Target: 60 FPS
        
        # Calculate final score
        final_score = base_score * 0.4 + accuracy_score * 0.3 + latency_score * 0.2 + fps_score * 0.1
        
        return min(1.0, max(0.0, final_score))
    
    def _generate_ar_recommendations(self, performance_metrics: Dict[str, float]) -> List[str]:
        """Generate AR-specific recommendations."""
        recommendations = []
        
        tracking_accuracy = performance_metrics.get("tracking_accuracy", 0.0)
        if tracking_accuracy < 0.9:
            recommendations.append("Improve marker tracking accuracy for better AR experience")
        
        tracking_latency = performance_metrics.get("tracking_latency", 0.0)
        if tracking_latency > 0.1:
            recommendations.append("Reduce tracking latency to below 100ms for real-time AR")
        
        camera_fps = performance_metrics.get("camera_fps", 30)
        if camera_fps < 60:
            recommendations.append("Optimize camera processing to achieve 60+ FPS")
        
        return recommendations

class HapticTestEngine:
    """Haptic testing engine for tactile feedback testing."""
    
    def __init__(self):
        self.haptic_devices = {}
        self.feedback_patterns = {}
        self.sensitivity_analyzer = SensitivityAnalyzer()
    
    def register_haptic_device(self, device_name: str, capabilities: Dict[str, Any]) -> str:
        """Register a haptic device."""
        device_id = f"haptic_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.haptic_devices[device_id] = {
            "name": device_name,
            "capabilities": capabilities,
            "active": True,
            "feedback_count": 0
        }
        
        return device_id
    
    def create_feedback_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]) -> str:
        """Create a haptic feedback pattern."""
        pattern_id = f"pattern_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.feedback_patterns[pattern_id] = {
            "name": pattern_name,
            "data": pattern_data,
            "created_at": datetime.now()
        }
        
        return pattern_id
    
    def run_haptic_test(self, test_name: str, device_id: str, pattern_id: str,
                       test_func: Callable) -> VRTestResult:
        """Run a haptic test."""
        if device_id not in self.haptic_devices:
            raise ValueError(f"Device {device_id} not found")
        
        if pattern_id not in self.feedback_patterns:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        device = self.haptic_devices[device_id]
        pattern = self.feedback_patterns[pattern_id]
        
        # Start sensitivity analysis
        self.sensitivity_analyzer.start_analysis()
        
        # Run test
        start_time = time.time()
        try:
            result = test_func(device, pattern)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
        
        test_duration = time.time() - start_time
        
        # Stop analysis
        sensitivity_metrics = self.sensitivity_analyzer.stop_analysis()
        
        # Calculate performance metrics
        performance_metrics = {
            "feedback_intensity": pattern["data"].get("intensity", 0.5),
            "feedback_duration": pattern["data"].get("duration", 1.0),
            "sensitivity_score": sensitivity_metrics.get("sensitivity_score", 0.5),
            "response_time": sensitivity_metrics.get("response_time", 0.1)
        }
        
        # Calculate user experience score
        ux_score = self._calculate_haptic_ux_score(performance_metrics, success)
        
        # Generate recommendations
        recommendations = self._generate_haptic_recommendations(performance_metrics)
        
        # Create test result
        test_result = VRTestResult(
            result_id=f"haptic_result_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name=test_name,
            environment_id=device_id,
            test_type="haptic",
            success=success,
            performance_metrics=performance_metrics,
            user_experience_score=ux_score,
            technical_issues=[error_message] if error_message else [],
            recommendations=recommendations
        )
        
        return test_result
    
    def _calculate_haptic_ux_score(self, performance_metrics: Dict[str, float], success: bool) -> float:
        """Calculate haptic user experience score."""
        base_score = 1.0 if success else 0.0
        
        # Intensity score (optimal range: 0.3-0.7)
        intensity = performance_metrics.get("feedback_intensity", 0.5)
        intensity_score = 1.0 - abs(intensity - 0.5) * 2  # Peak at 0.5
        
        # Duration score (optimal range: 0.1-2.0 seconds)
        duration = performance_metrics.get("feedback_duration", 1.0)
        duration_score = 1.0 - max(0, abs(duration - 1.0) - 1.0)  # Peak at 1.0
        
        # Sensitivity score
        sensitivity = performance_metrics.get("sensitivity_score", 0.5)
        sensitivity_score = sensitivity
        
        # Response time score (target: <100ms)
        response_time = performance_metrics.get("response_time", 0.1)
        response_score = max(0.0, 1.0 - response_time / 0.1)
        
        # Calculate final score
        final_score = base_score * 0.3 + intensity_score * 0.25 + duration_score * 0.2 + sensitivity_score * 0.15 + response_score * 0.1
        
        return min(1.0, max(0.0, final_score))
    
    def _generate_haptic_recommendations(self, performance_metrics: Dict[str, float]) -> List[str]:
        """Generate haptic-specific recommendations."""
        recommendations = []
        
        intensity = performance_metrics.get("feedback_intensity", 0.5)
        if intensity < 0.3:
            recommendations.append("Increase haptic feedback intensity for better user perception")
        elif intensity > 0.7:
            recommendations.append("Reduce haptic feedback intensity to avoid discomfort")
        
        duration = performance_metrics.get("feedback_duration", 1.0)
        if duration < 0.1:
            recommendations.append("Increase feedback duration for better user recognition")
        elif duration > 2.0:
            recommendations.append("Reduce feedback duration to avoid user fatigue")
        
        response_time = performance_metrics.get("response_time", 0.1)
        if response_time > 0.1:
            recommendations.append("Improve haptic response time for better user experience")
        
        return recommendations

# Supporting classes
class VRPerformanceMonitor:
    """Monitors VR performance metrics."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = {
            "fps": 0,
            "latency": 0.0,
            "frame_drops": 0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0
        }
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        
        # Simulate performance metrics
        self.metrics = {
            "fps": random.uniform(60, 120),
            "latency": random.uniform(0.01, 0.05),
            "frame_drops": random.randint(0, 20),
            "memory_usage": random.uniform(0.5, 2.0),
            "cpu_usage": random.uniform(0.3, 0.8)
        }
        
        return self.metrics.copy()

class InteractionTracker:
    """Tracks user interactions in VR/AR."""
    
    def __init__(self):
        self.tracking = False
        self.interactions = []
        self.start_time = None
    
    def start_tracking(self):
        """Start interaction tracking."""
        self.tracking = True
        self.start_time = time.time()
        self.interactions = []
    
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking and return data."""
        self.tracking = False
        
        # Simulate interaction data
        total_interactions = random.randint(5, 20)
        successful_interactions = int(total_interactions * random.uniform(0.7, 1.0))
        
        return {
            "total_interactions": total_interactions,
            "successful_interactions": successful_interactions,
            "interaction_success_rate": successful_interactions / total_interactions,
            "average_interaction_time": random.uniform(0.5, 2.0)
        }

class ARTrackingSystem:
    """AR marker tracking system."""
    
    def __init__(self):
        self.tracking = False
        self.marker = None
        self.start_time = None
    
    def start_tracking(self, marker: ARTestMarker):
        """Start tracking a marker."""
        self.tracking = True
        self.marker = marker
        self.start_time = time.time()
    
    def stop_tracking(self) -> Dict[str, float]:
        """Stop tracking and return metrics."""
        self.tracking = False
        
        # Simulate tracking metrics
        return {
            "accuracy": random.uniform(0.8, 1.0),
            "latency": random.uniform(0.01, 0.1),
            "stability": random.uniform(0.7, 1.0)
        }

class OverlayManager:
    """Manages AR overlays."""
    
    def __init__(self):
        self.overlays = {}
    
    def create_overlay(self, content: Dict[str, Any]) -> str:
        """Create an AR overlay."""
        overlay_id = f"overlay_{int(time.time())}_{random.randint(1000, 9999)}"
        self.overlays[overlay_id] = content
        return overlay_id

class CameraSimulator:
    """Simulates camera for AR testing."""
    
    def __init__(self):
        self.simulating = False
        self.start_time = None
    
    def start_simulation(self):
        """Start camera simulation."""
        self.simulating = True
        self.start_time = time.time()
    
    def stop_simulation(self) -> Dict[str, float]:
        """Stop simulation and return metrics."""
        self.simulating = False
        
        # Simulate camera metrics
        return {
            "fps": random.uniform(30, 60),
            "overlay_time": random.uniform(0.01, 0.05),
            "processing_time": random.uniform(0.02, 0.08)
        }

class SensitivityAnalyzer:
    """Analyzes haptic sensitivity."""
    
    def __init__(self):
        self.analyzing = False
        self.start_time = None
    
    def start_analysis(self):
        """Start sensitivity analysis."""
        self.analyzing = True
        self.start_time = time.time()
    
    def stop_analysis(self) -> Dict[str, float]:
        """Stop analysis and return metrics."""
        self.analyzing = False
        
        # Simulate sensitivity metrics
        return {
            "sensitivity_score": random.uniform(0.6, 1.0),
            "response_time": random.uniform(0.05, 0.15),
            "accuracy": random.uniform(0.8, 1.0)
        }

class VRARTestFramework:
    """Main VR/AR testing framework."""
    
    def __init__(self):
        self.vr_engine = VRTestEngine()
        self.ar_engine = ARTestEngine()
        self.haptic_engine = HapticTestEngine()
        self.test_results = []
    
    def run_comprehensive_test(self, test_name: str, test_config: Dict[str, Any]) -> List[VRTestResult]:
        """Run comprehensive VR/AR test suite."""
        results = []
        
        # VR tests
        if "vr" in test_config:
            vr_config = test_config["vr"]
            environment = self.vr_engine.create_environment(
                vr_config["environment_name"],
                vr_config["dimensions"]
            )
            
            # Add objects to environment
            for obj in vr_config.get("objects", []):
                self.vr_engine.add_object_to_environment(
                    environment.environment_id,
                    obj["type"],
                    obj["position"],
                    obj.get("properties", {})
                )
            
            # Run VR test
            vr_result = self.vr_engine.run_vr_test(
                f"{test_name}_VR",
                environment.environment_id,
                vr_config["test_function"],
                vr_config.get("duration", 10.0)
            )
            results.append(vr_result)
        
        # AR tests
        if "ar" in test_config:
            ar_config = test_config["ar"]
            marker = self.ar_engine.create_marker(
                ar_config["marker_name"],
                ar_config["marker_type"],
                ar_config["position"],
                ar_config.get("content", {})
            )
            
            # Run AR test
            ar_result = self.ar_engine.run_ar_test(
                f"{test_name}_AR",
                marker.marker_id,
                ar_config["test_function"]
            )
            results.append(ar_result)
        
        # Haptic tests
        if "haptic" in test_config:
            haptic_config = test_config["haptic"]
            
            # Register device
            device_id = self.haptic_engine.register_haptic_device(
                haptic_config["device_name"],
                haptic_config.get("capabilities", {})
            )
            
            # Create pattern
            pattern_id = self.haptic_engine.create_feedback_pattern(
                haptic_config["pattern_name"],
                haptic_config["pattern_data"]
            )
            
            # Run haptic test
            haptic_result = self.haptic_engine.run_haptic_test(
                f"{test_name}_Haptic",
                device_id,
                pattern_id,
                haptic_config["test_function"]
            )
            results.append(haptic_result)
        
        self.test_results.extend(results)
        return results
    
    def generate_vr_ar_report(self) -> Dict[str, Any]:
        """Generate comprehensive VR/AR test report."""
        if not self.test_results:
            return {"message": "No test results available"}
        
        # Analyze results by type
        vr_results = [r for r in self.test_results if r.test_type == "vr"]
        ar_results = [r for r in self.test_results if r.test_type == "ar"]
        haptic_results = [r for r in self.test_results if r.test_type == "haptic"]
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        avg_ux_score = np.mean([r.user_experience_score for r in self.test_results])
        
        # Performance analysis
        performance_analysis = self._analyze_performance_metrics()
        
        # Generate recommendations
        recommendations = self._generate_overall_recommendations()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_ux_score": avg_ux_score
            },
            "by_type": {
                "vr_tests": len(vr_results),
                "ar_tests": len(ar_results),
                "haptic_tests": len(haptic_results)
            },
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": [r.__dict__ for r in self.test_results]
        }
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across all tests."""
        all_metrics = [r.performance_metrics for r in self.test_results]
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {}
        for metric_name in all_metrics[0].keys():
            values = [m.get(metric_name, 0) for m in all_metrics]
            aggregated[metric_name] = {
                "average": np.mean(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values)
            }
        
        return aggregated
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []
        
        # Analyze success rate
        success_rate = sum(1 for r in self.test_results if r.success) / len(self.test_results) if self.test_results else 0
        if success_rate < 0.9:
            recommendations.append("Improve overall test success rate for better reliability")
        
        # Analyze UX scores
        avg_ux = np.mean([r.user_experience_score for r in self.test_results]) if self.test_results else 0
        if avg_ux < 0.8:
            recommendations.append("Focus on improving user experience scores across all test types")
        
        # Analyze performance
        all_metrics = [r.performance_metrics for r in self.test_results]
        if all_metrics:
            avg_fps = np.mean([m.get("fps", 0) for m in all_metrics])
            if avg_fps < 60:
                recommendations.append("Optimize rendering performance to achieve 60+ FPS")
        
        return recommendations

# Example usage and demo
def demo_vr_ar_testing():
    """Demonstrate VR/AR testing capabilities."""
    print("🥽 VR/AR Testing Framework Demo")
    print("=" * 50)
    
    # Create VR/AR testing framework
    framework = VRARTestFramework()
    
    # Define test functions
    def vr_interaction_test(environment):
        """VR interaction test."""
        # Simulate VR interaction
        time.sleep(0.1)
        return True
    
    def ar_overlay_test(marker):
        """AR overlay test."""
        # Simulate AR overlay
        time.sleep(0.05)
        return True
    
    def haptic_feedback_test(device, pattern):
        """Haptic feedback test."""
        # Simulate haptic feedback
        time.sleep(0.02)
        return True
    
    # Create comprehensive test configuration
    test_config = {
        "vr": {
            "environment_name": "Demo VR Environment",
            "dimensions": (10.0, 5.0, 10.0),
            "objects": [
                {"type": "cube", "position": (0, 0, 0), "properties": {"color": "red"}},
                {"type": "sphere", "position": (2, 1, 2), "properties": {"color": "blue"}}
            ],
            "test_function": vr_interaction_test,
            "duration": 5.0
        },
        "ar": {
            "marker_name": "Demo AR Marker",
            "marker_type": "image",
            "position": (0, 0, 0),
            "content": {"overlay": "3D model", "interactive": True},
            "test_function": ar_overlay_test
        },
        "haptic": {
            "device_name": "Demo Haptic Device",
            "capabilities": {"intensity_range": (0, 1), "frequency_range": (1, 1000)},
            "pattern_name": "Demo Pattern",
            "pattern_data": {"intensity": 0.5, "duration": 1.0, "pattern": "pulse"},
            "test_function": haptic_feedback_test
        }
    }
    
    # Run comprehensive test
    print("🚀 Running comprehensive VR/AR test suite...")
    results = framework.run_comprehensive_test("Demo_Test", test_config)
    
    # Print results
    print(f"\n📊 Test Results:")
    for result in results:
        print(f"\n{result.test_type.upper()} Test: {result.test_name}")
        print(f"  Success: {'✅' if result.success else '❌'}")
        print(f"  UX Score: {result.user_experience_score:.2f}")
        print(f"  Performance Metrics:")
        for metric, value in result.performance_metrics.items():
            print(f"    {metric}: {value:.3f}")
        
        if result.recommendations:
            print(f"  Recommendations:")
            for rec in result.recommendations:
                print(f"    - {rec}")
    
    # Generate comprehensive report
    print("\n📈 Generating comprehensive report...")
    report = framework.generate_vr_ar_report()
    
    print(f"\n📊 Comprehensive Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"  Average UX Score: {report['summary']['average_ux_score']:.2f}")
    print(f"  VR Tests: {report['by_type']['vr_tests']}")
    print(f"  AR Tests: {report['by_type']['ar_tests']}")
    print(f"  Haptic Tests: {report['by_type']['haptic_tests']}")
    
    print(f"\n💡 Overall Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    # Run demo
    demo_vr_ar_testing()
