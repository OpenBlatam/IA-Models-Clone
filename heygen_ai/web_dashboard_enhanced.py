#!/usr/bin/env python3
"""
Enhanced Web Dashboard for Advanced Distributed AI System
Advanced monitoring, optimization control, and intelligent system management
"""

import logging
import time
import json
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import numpy as np
from collections import deque, defaultdict
import psutil
import os

# ===== ENHANCED CONFIGURATION =====

@dataclass
class DashboardConfig:
    """Enhanced configuration for the web dashboard."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    secret_key: str = "enhanced_ai_dashboard_secret_key_2024"
    
    # Dashboard settings
    refresh_interval: float = 1.0  # seconds
    max_data_points: int = 2000
    enable_real_time_updates: bool = True
    enable_websockets: bool = True
    enable_auto_scaling: bool = True
    
    # System integration
    auto_start_monitoring: bool = True
    auto_start_optimization: bool = True
    enable_auto_optimization: bool = True
    enable_predictive_analytics: bool = True
    
    # Security
    enable_authentication: bool = False
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100

# ===== ENHANCED DATA MODELS =====

@dataclass
class SystemMetrics:
    """Enhanced system metrics."""
    timestamp: float
    datetime: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    quantum_advantage: float
    learning_efficiency: float
    overall_score: float
    system_load: float
    gpu_usage: Optional[float] = None
    temperature: Optional[float] = None

@dataclass
class OptimizationMetrics:
    """Enhanced optimization metrics."""
    timestamp: float
    datetime: str
    optimization_type: str
    success: bool
    score: float
    execution_time: float
    resource_usage: Dict[str, float]
    improvement_factor: float
    convergence_rate: float

# ===== ENHANCED DASHBOARD SYSTEM =====

class EnhancedDashboardSystem:
    """Enhanced dashboard system with advanced features."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedDashboard")
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.secret_key
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Enhanced system components
        self.performance_monitor = None
        self.optimization_engine = None
        
        # Enhanced dashboard state
        self.system_status = "initializing"
        self.last_update = time.time()
        self.monitoring_active = False
        self.optimization_active = False
        self.auto_scaling_active = False
        
        # Enhanced data storage
        self.performance_data = deque(maxlen=config.max_data_points)
        self.optimization_data = deque(maxlen=config.max_data_points)
        self.alerts = deque(maxlen=200)
        self.system_metrics = {}
        self.predictive_data = {}
        
        # Performance tracking
        self.performance_history = []
        self.optimization_history = []
        self.system_health_score = 1.0
        
        # Setup enhanced routes and events
        self._setup_enhanced_routes()
        self._setup_enhanced_socketio_events()
        self._setup_middleware()
        
        self.logger.info("Enhanced dashboard system initialized")
    
    def _setup_middleware(self):
        """Setup middleware for enhanced functionality."""
        
        @self.app.before_request
        def before_request():
            """Pre-request processing."""
            # Rate limiting
            if self.config.enable_rate_limiting:
                if not self._check_rate_limit(request):
                    return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Request logging
            self.logger.info(f"Request: {request.method} {request.path}")
    
    def _setup_enhanced_routes(self):
        """Setup enhanced Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('enhanced_dashboard.html')
        
        @self.app.route('/api/v2/status')
        def get_enhanced_status():
            """Get enhanced system status."""
            return jsonify(self._get_enhanced_system_status())
        
        @self.app.route('/api/v2/performance')
        def get_enhanced_performance():
            """Get enhanced performance data."""
            return jsonify(self._get_enhanced_performance_data())
        
        @self.app.route('/api/v2/optimization')
        def get_enhanced_optimization():
            """Get enhanced optimization data."""
            return jsonify(self._get_enhanced_optimization_data())
        
        @self.app.route('/api/v2/alerts')
        def get_enhanced_alerts():
            """Get enhanced system alerts."""
            return jsonify(self._get_enhanced_alerts())
        
        @self.app.route('/api/v2/predictions')
        def get_predictions():
            """Get predictive analytics data."""
            return jsonify(self._get_predictive_data())
        
        @self.app.route('/api/v2/system_health')
        def get_system_health():
            """Get comprehensive system health."""
            return jsonify(self._get_system_health())
        
        @self.app.route('/api/v2/auto_scaling', methods=['POST'])
        def toggle_auto_scaling():
            """Toggle auto-scaling functionality."""
            data = request.get_json()
            enable = data.get('enable', False)
            success = self._toggle_auto_scaling(enable)
            return jsonify({"success": success, "auto_scaling": self.auto_scaling_active})
        
        @self.app.route('/api/v2/performance_analysis', methods=['POST'])
        def analyze_performance():
            """Analyze performance patterns."""
            data = request.get_json()
            analysis = self._analyze_performance_patterns(data)
            return jsonify(analysis)
    
    def _setup_enhanced_socketio_events(self):
        """Setup enhanced SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_enhanced_connect():
            """Handle enhanced client connection."""
            self.logger.info("Enhanced client connected to dashboard")
            emit('enhanced_system_status', self._get_enhanced_system_status())
            emit('system_health_update', self._get_system_health())
        
        @self.socketio.on('disconnect')
        def handle_enhanced_disconnect():
            """Handle enhanced client disconnection."""
            self.logger.info("Enhanced client disconnected from dashboard")
        
        @self.socketio.on('request_enhanced_update')
        def handle_enhanced_update_request():
            """Handle enhanced update request from client."""
            self._send_enhanced_real_time_update()
        
        @self.socketio.on('performance_analysis_request')
        def handle_performance_analysis_request(data):
            """Handle performance analysis request."""
            analysis = self._analyze_performance_patterns(data)
            emit('performance_analysis_result', analysis)
    
    def _check_rate_limit(self, request) -> bool:
        """Check rate limiting for requests."""
        # Simple rate limiting implementation
        client_ip = request.remote_addr
        current_time = time.time()
        
        if not hasattr(self, '_rate_limit_data'):
            self._rate_limit_data = defaultdict(list)
        
        # Clean old requests
        self._rate_limit_data[client_ip] = [
            req_time for req_time in self._rate_limit_data[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check limit
        if len(self._rate_limit_data[client_ip]) >= self.config.max_requests_per_minute:
            return False
        
        # Add current request
        self._rate_limit_data[client_ip].append(current_time)
        return True
    
    def _get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status."""
        return {
            "status": self.system_status,
            "monitoring_active": self.monitoring_active,
            "optimization_active": self.optimization_active,
            "auto_scaling_active": self.auto_scaling_active,
            "last_update": self.last_update,
            "uptime": time.time() - self.last_update if self.last_update > 0 else 0,
            "performance_data_points": len(self.performance_data),
            "optimization_data_points": len(self.optimization_data),
            "active_alerts": len([a for a in self.alerts if not a.get("acknowledged", False)]),
            "system_health_score": self.system_health_score,
            "auto_scaling_enabled": self.config.enable_auto_scaling,
            "predictive_analytics_enabled": self.config.enable_predictive_analytics
        }
    
    def _get_enhanced_performance_data(self) -> Dict[str, Any]:
        """Get enhanced performance data."""
        if not self.performance_data:
            return {"data": [], "summary": {}, "trends": {}}
        
        # Get recent data
        recent_data = list(self.performance_data)[-100:] if len(self.performance_data) > 100 else list(self.performance_data)
        
        # Calculate enhanced summary
        summary = self._calculate_enhanced_performance_summary()
        
        # Calculate trends
        trends = self._calculate_performance_trends()
        
        return {
            "data": recent_data,
            "summary": summary,
            "trends": trends,
            "total_points": len(self.performance_data),
            "data_quality": self._assess_data_quality()
        }
    
    def _calculate_enhanced_performance_summary(self) -> Dict[str, Any]:
        """Calculate enhanced performance summary."""
        if not self.performance_data:
            return {"error": "No performance data available"}
        
        try:
            data_list = list(self.performance_data)
            
            # Basic metrics
            scores = [m.overall_score for m in data_list]
            cpu_usage = [m.cpu_usage for m in data_list]
            memory_usage = [m.memory_usage for m in data_list]
            quantum_advantage = [m.quantum_advantage for m in data_list]
            learning_efficiency = [m.learning_efficiency for m in data_list]
            
            # Enhanced metrics
            system_load = [m.system_load for m in data_list if m.system_load is not None]
            gpu_usage = [m.gpu_usage for m in data_list if m.gpu_usage is not None]
            temperature = [m.temperature for m in data_list if m.temperature is not None]
            
            return {
                "total_measurements": len(data_list),
                "overall_score": {
                    "current": scores[-1] if scores else 0.0,
                    "average": np.mean(scores),
                    "best": max(scores),
                    "worst": min(scores),
                    "trend": self._calculate_trend(scores),
                    "volatility": np.std(scores)
                },
                "system_performance": {
                    "cpu_usage": {
                        "current": cpu_usage[-1] if cpu_usage else 0.0,
                        "average": np.mean(cpu_usage),
                        "peak": max(cpu_usage),
                        "trend": self._calculate_trend(cpu_usage)
                    },
                    "memory_usage": {
                        "current": memory_usage[-1] if memory_usage else 0.0,
                        "average": np.mean(memory_usage),
                        "peak": max(memory_usage),
                        "trend": self._calculate_trend(memory_usage)
                    },
                    "system_load": {
                        "current": system_load[-1] if system_load else 0.0,
                        "average": np.mean(system_load) if system_load else 0.0,
                        "peak": max(system_load) if system_load else 0.0
                    }
                },
                "ai_performance": {
                    "quantum_advantage": {
                        "current": quantum_advantage[-1] if quantum_advantage else 1.0,
                        "average": np.mean(quantum_advantage),
                        "best": max(quantum_advantage),
                        "trend": self._calculate_trend(quantum_advantage)
                    },
                    "learning_efficiency": {
                        "current": learning_efficiency[-1] if learning_efficiency else 0.0,
                        "average": np.mean(learning_efficiency),
                        "best": max(learning_efficiency),
                        "trend": self._calculate_trend(learning_efficiency)
                    }
                },
                "hardware_metrics": {
                    "gpu_usage": {
                        "current": gpu_usage[-1] if gpu_usage else 0.0,
                        "average": np.mean(gpu_usage) if gpu_usage else 0.0
                    },
                    "temperature": {
                        "current": temperature[-1] if temperature else 0.0,
                        "average": np.mean(temperature) if temperature else 0.0
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced performance summary: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends and patterns."""
        if not self.performance_data:
            return {"error": "No performance data available"}
        
        try:
            data_list = list(self.performance_data)
            
            # Time-based analysis
            timestamps = [m.timestamp for m in data_list]
            scores = [m.overall_score for m in data_list]
            
            # Trend analysis
            short_term_trend = self._calculate_trend(scores[-20:]) if len(scores) >= 20 else "insufficient_data"
            medium_term_trend = self._calculate_trend(scores[-50:]) if len(scores) >= 50 else "insufficient_data"
            long_term_trend = self._calculate_trend(scores[-100:]) if len(scores) >= 100 else "insufficient_data"
            
            # Pattern detection
            patterns = self._detect_performance_patterns(scores)
            
            # Anomaly detection
            anomalies = self._detect_performance_anomalies(scores)
            
            return {
                "trends": {
                    "short_term": short_term_trend,
                    "medium_term": medium_term_trend,
                    "long_term": long_term_trend
                },
                "patterns": patterns,
                "anomalies": anomalies,
                "stability_score": self._calculate_stability_score(scores)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance trends: {e}")
            return {"error": str(e)}
    
    def _detect_performance_patterns(self, scores: List[float]) -> Dict[str, Any]:
        """Detect performance patterns."""
        if len(scores) < 10:
            return {"error": "Insufficient data for pattern detection"}
        
        try:
            # Cyclic patterns
            fft = np.fft.fft(scores)
            frequencies = np.fft.fftfreq(len(scores))
            
            # Find dominant frequencies
            dominant_freq_idx = np.argsort(np.abs(fft))[-3:]
            dominant_frequencies = frequencies[dominant_freq_idx]
            
            # Trend consistency
            trend_consistency = self._calculate_trend_consistency(scores)
            
            return {
                "cyclic_patterns": {
                    "dominant_frequencies": dominant_frequencies.tolist(),
                    "strength": np.abs(fft[dominant_freq_idx]).tolist()
                },
                "trend_consistency": trend_consistency,
                "volatility_pattern": self._analyze_volatility_pattern(scores)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect performance patterns: {e}")
            return {"error": str(e)}
    
    def _detect_performance_anomalies(self, scores: List[float]) -> Dict[str, Any]:
        """Detect performance anomalies."""
        if len(scores) < 5:
            return {"error": "Insufficient data for anomaly detection"}
        
        try:
            # Statistical anomaly detection
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Z-score based detection
            z_scores = [(score - mean_score) / std_score for score in scores]
            anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 2.0]
            
            # Change point detection
            change_points = self._detect_change_points(scores)
            
            return {
                "statistical_anomalies": {
                    "count": len(anomalies),
                    "indices": anomalies,
                    "z_scores": z_scores
                },
                "change_points": change_points,
                "anomaly_score": len(anomalies) / len(scores)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect performance anomalies: {e}")
            return {"error": str(e)}
    
    def _calculate_stability_score(self, scores: List[float]) -> float:
        """Calculate system stability score."""
        if len(scores) < 2:
            return 0.0
        
        try:
            # Calculate variance and trend consistency
            variance = np.var(scores)
            trend_consistency = self._calculate_trend_consistency(scores)
            
            # Normalize to 0-1 scale
            stability_score = max(0.0, 1.0 - variance - (1.0 - trend_consistency))
            return min(1.0, stability_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate stability score: {e}")
            return 0.0
    
    def _calculate_trend_consistency(self, scores: List[float]) -> float:
        """Calculate trend consistency score."""
        if len(scores) < 3:
            return 0.0
        
        try:
            # Calculate multiple trend lines
            windows = [5, 10, 20]
            trends = []
            
            for window in windows:
                if len(scores) >= window:
                    recent_scores = scores[-window:]
                    trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                    trends.append(trend)
            
            if not trends:
                return 0.0
            
            # Calculate consistency (lower variance = higher consistency)
            trend_variance = np.var(trends)
            consistency = max(0.0, 1.0 - trend_variance)
            
            return min(1.0, consistency)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trend consistency: {e}")
            return 0.0
    
    def _detect_change_points(self, scores: List[float]) -> List[int]:
        """Detect change points in performance data."""
        if len(scores) < 10:
            return []
        
        try:
            change_points = []
            
            # Simple change point detection using rolling statistics
            window_size = min(10, len(scores) // 2)
            
            for i in range(window_size, len(scores) - window_size):
                before_mean = np.mean(scores[i-window_size:i])
                after_mean = np.mean(scores[i:i+window_size])
                
                # Detect significant change
                change_magnitude = abs(after_mean - before_mean) / (before_mean + 1e-8)
                
                if change_magnitude > 0.2:  # 20% change threshold
                    change_points.append(i)
            
            return change_points
            
        except Exception as e:
            self.logger.error(f"Failed to detect change points: {e}")
            return []
    
    def _analyze_volatility_pattern(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze volatility patterns in performance data."""
        if len(scores) < 10:
            return {"error": "Insufficient data for volatility analysis"}
        
        try:
            # Calculate rolling volatility
            window_size = min(10, len(scores) // 2)
            rolling_volatility = []
            
            for i in range(window_size, len(scores)):
                window_scores = scores[i-window_size:i]
                volatility = np.std(window_scores)
                rolling_volatility.append(volatility)
            
            # Analyze volatility trends
            volatility_trend = self._calculate_trend(rolling_volatility)
            volatility_consistency = self._calculate_trend_consistency(rolling_volatility)
            
            return {
                "current_volatility": rolling_volatility[-1] if rolling_volatility else 0.0,
                "average_volatility": np.mean(rolling_volatility) if rolling_volatility else 0.0,
                "volatility_trend": volatility_trend,
                "volatility_consistency": volatility_consistency,
                "volatility_pattern": "stable" if volatility_consistency > 0.7 else "unstable"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze volatility pattern: {e}")
            return {"error": str(e)}
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of collected performance data."""
        if not self.performance_data:
            return {"error": "No data available"}
        
        try:
            data_list = list(self.performance_data)
            
            # Completeness
            total_expected = int((time.time() - self.last_update) / self.config.refresh_interval)
            completeness = len(data_list) / max(total_expected, 1)
            
            # Consistency
            timestamps = [m.timestamp for m in data_list]
            time_gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            consistency = 1.0 - (np.std(time_gaps) / np.mean(time_gaps)) if time_gaps else 1.0
            
            # Validity
            valid_entries = sum(1 for m in data_list if 0.0 <= m.overall_score <= 1.0)
            validity = valid_entries / len(data_list) if data_list else 0.0
            
            return {
                "completeness": min(1.0, completeness),
                "consistency": max(0.0, min(1.0, consistency)),
                "validity": validity,
                "overall_quality": (completeness + consistency + validity) / 3,
                "data_points": len(data_list),
                "expected_points": total_expected
            }
            
        except Exception as e:
            self.logger.error(f"Failed to assess data quality: {e}")
            return {"error": str(e)}
    
    def _toggle_auto_scaling(self, enable: bool) -> bool:
        """Toggle auto-scaling functionality."""
        try:
            self.auto_scaling_active = enable
            
            if enable:
                # Start auto-scaling thread
                scaling_thread = threading.Thread(target=self._auto_scaling_loop, daemon=True)
                scaling_thread.start()
                self.logger.info("Auto-scaling enabled")
            else:
                self.logger.info("Auto-scaling disabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to toggle auto-scaling: {e}")
            return False
    
    def _auto_scaling_loop(self):
        """Auto-scaling loop for system optimization."""
        while self.auto_scaling_active:
            try:
                # Analyze current performance
                if self.performance_data:
                    current_metrics = list(self.performance_data)[-1]
                    
                    # Check if scaling is needed
                    if current_metrics.overall_score < 0.6:
                        self._trigger_auto_scaling("performance_degradation")
                    elif current_metrics.cpu_usage > 0.8:
                        self._trigger_auto_scaling("high_cpu_usage")
                    elif current_metrics.memory_usage > 0.8:
                        self._trigger_auto_scaling("high_memory_usage")
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(30)
    
    def _trigger_auto_scaling(self, reason: str):
        """Trigger automatic system scaling."""
        try:
            self.logger.info(f"Auto-scaling triggered: {reason}")
            
            # Create scaling optimization problem
            scaling_problem = {
                "id": f"auto_scaling_{int(time.time())}",
                "type": "auto_scaling",
                "reason": reason,
                "target": "improve_system_performance",
                "constraints": {
                    "max_iterations": 100,
                    "timeout_seconds": 60
                }
            }
            
            # Execute scaling optimization
            if self.optimization_engine and self.optimization_active:
                result = self.optimization_engine.optimize(scaling_problem, "hybrid")
                
                # Log scaling result
                self.logger.info(f"Auto-scaling completed: {result.get('success', False)}")
                
                # Send real-time update
                if self.config.enable_websockets:
                    self.socketio.emit('auto_scaling_completed', {
                        "reason": reason,
                        "result": result,
                        "timestamp": time.time()
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to trigger auto-scaling: {e}")
    
    def _analyze_performance_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns based on request data."""
        try:
            analysis_type = data.get('type', 'general')
            
            if analysis_type == 'trend':
                return self._analyze_trend_patterns()
            elif analysis_type == 'anomaly':
                return self._analyze_anomaly_patterns()
            elif analysis_type == 'prediction':
                return self._generate_performance_predictions()
            else:
                return self._generate_comprehensive_analysis()
                
        except Exception as e:
            self.logger.error(f"Failed to analyze performance patterns: {e}")
            return {"error": str(e)}
    
    def _generate_performance_predictions(self) -> Dict[str, Any]:
        """Generate performance predictions using historical data."""
        if not self.performance_data:
            return {"error": "No data available for predictions"}
        
        try:
            data_list = list(self.performance_data)
            scores = [m.overall_score for m in data_list]
            
            if len(scores) < 20:
                return {"error": "Insufficient data for predictions"}
            
            # Simple linear prediction
            x = np.arange(len(scores))
            coeffs = np.polyfit(x, scores, 2)  # Quadratic fit
            
            # Predict next 10 points
            future_x = np.arange(len(scores), len(scores) + 10)
            predictions = np.polyval(coeffs, future_x)
            
            # Confidence intervals (simplified)
            confidence = 0.8
            std_error = np.std(scores)
            
            return {
                "predictions": predictions.tolist(),
                "confidence": confidence,
                "prediction_horizon": 10,
                "method": "quadratic_regression",
                "accuracy_estimate": 1.0 - std_error
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance predictions: {e}")
            return {"error": str(e)}
    
    def _send_enhanced_real_time_update(self):
        """Send enhanced real-time update to connected clients."""
        try:
            if self.config.enable_websockets:
                update_data = {
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "system_status": self._get_enhanced_system_status(),
                    "performance_summary": self._calculate_enhanced_performance_summary(),
                    "optimization_summary": self._get_enhanced_optimization_data(),
                    "system_health": self._get_system_health(),
                    "predictions": self._generate_performance_predictions()
                }
                
                self.socketio.emit('enhanced_system_update', update_data)
                
        except Exception as e:
            self.logger.error(f"Failed to send enhanced real-time update: {e}")
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health assessment."""
        try:
            if not self.performance_data:
                return {"error": "No performance data available"}
            
            data_list = list(self.performance_data)
            recent_data = data_list[-20:] if len(data_list) >= 20 else data_list
            
            # Calculate health metrics
            scores = [m.overall_score for m in recent_data]
            cpu_usage = [m.cpu_usage for m in recent_data]
            memory_usage = [m.memory_usage for m in recent_data]
            
            # Health indicators
            performance_health = np.mean(scores)
            resource_health = 1.0 - (np.mean(cpu_usage) + np.mean(memory_usage)) / 2
            stability_health = self._calculate_stability_score(scores)
            
            # Overall health score
            overall_health = (performance_health + resource_health + stability_health) / 3
            self.system_health_score = overall_health
            
            # Health status
            if overall_health >= 0.8:
                health_status = "excellent"
            elif overall_health >= 0.6:
                health_status = "good"
            elif overall_health >= 0.4:
                health_status = "fair"
            else:
                health_status = "poor"
            
            return {
                "overall_health": overall_health,
                "health_status": health_status,
                "performance_health": performance_health,
                "resource_health": resource_health,
                "stability_health": stability_health,
                "trend": self._calculate_trend(scores),
                "recommendations": self._generate_health_recommendations(overall_health)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {"error": str(e)}
    
    def _generate_health_recommendations(self, health_score: float) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if health_score < 0.6:
            recommendations.append("Consider system optimization to improve performance")
            recommendations.append("Check for resource bottlenecks")
            recommendations.append("Review system configuration")
        
        if health_score < 0.8:
            recommendations.append("Monitor system trends for potential issues")
            recommendations.append("Consider preventive maintenance")
        
        if health_score >= 0.8:
            recommendations.append("System is performing well - maintain current configuration")
        
        return recommendations

# ===== MAIN EXECUTION =====

def main():
    """Main enhanced dashboard execution."""
    print("🚀 Enhanced Advanced AI System Web Dashboard")
    print("="*60)
    
    # Create enhanced configuration
    config = DashboardConfig()
    
    # Setup enhanced logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create enhanced dashboard system
    dashboard = EnhancedDashboardSystem(config)
    
    try:
        # Start enhanced dashboard
        dashboard.start()
        
    except KeyboardInterrupt:
        print("\n⏹️ Enhanced dashboard interrupted by user")
    except Exception as e:
        print(f"❌ Enhanced dashboard failed: {e}")
        raise
    finally:
        # Stop enhanced dashboard
        dashboard.stop()

if __name__ == "__main__":
    main()
