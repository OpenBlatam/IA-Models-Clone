#!/usr/bin/env python3
"""
HeyGen AI - Intelligent Performance Analyzer

This module provides intelligent performance analysis, pattern recognition,
and predictive optimization recommendations for the HeyGen AI system.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformancePattern:
    """Performance pattern definition."""
    name: str
    pattern_type: str  # 'trend', 'cycle', 'anomaly', 'correlation'
    confidence: float
    description: str
    metrics: List[str]
    parameters: Dict[str, Any]
    detected_at: float = field(default_factory=time.time)

@dataclass
class AnalysisResult:
    """Result of performance analysis."""
    timestamp: float
    analysis_type: str
    patterns_found: List[PerformancePattern]
    recommendations: List[str]
    confidence_score: float
    execution_time: float

class IntelligentAnalyzer:
    """Intelligent performance analyzer for HeyGen AI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.analysis_history: List[AnalysisResult] = []
        self.detected_patterns: List[PerformancePattern] = []
        self.analysis_active = False
        self.analyzer_thread: Optional[threading.Thread] = None
        self.analysis_interval = self.config.get('analysis_interval', 300.0)  # 5 minutes
        
        # Data storage for analysis
        self.metrics_history: Dict[str, List[Tuple[float, float]]] = {}  # metric_name -> [(timestamp, value)]
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, Tuple[float, float]] = {}
        
        # Analysis capabilities
        self.analysis_capabilities = {
            'trend_analysis': True,
            'cycle_detection': True,
            'anomaly_detection': True,
            'correlation_analysis': True,
            'predictive_analysis': True
        }
        
        # Initialize analyzer
        self._setup_analyzer()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default analysis configuration."""
        return {
            'analysis_interval': 300.0,  # seconds
            'max_history_size': 10000,
            'min_data_points': 50,
            'confidence_threshold': 0.7,
            'anomaly_sensitivity': 2.0,  # Standard deviations for anomaly detection
            'trend_window_size': 100,
            'cycle_detection_window': 200,
            'correlation_threshold': 0.6,
            'prediction_horizon': 24,  # hours
            'analysis_methods': {
                'statistical': True,
                'machine_learning': False,  # Could be enabled with ML libraries
                'heuristic': True
            }
        }
    
    def _setup_analyzer(self):
        """Setup analysis infrastructure."""
        try:
            # Initialize metric tracking
            self._initialize_metric_tracking()
            
            # Load historical data if available
            self._load_historical_data()
            
            logger.info("Intelligent analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup analyzer: {e}")
    
    def _initialize_metric_tracking(self):
        """Initialize metric tracking structures."""
        # Initialize for common metrics
        common_metrics = [
            'cpu_percent', 'memory_percent', 'disk_usage_percent',
            'active_processes', 'active_threads', 'network_io_bytes'
        ]
        
        for metric in common_metrics:
            self.metrics_history[metric] = []
            self.anomaly_thresholds[metric] = (0.0, 100.0)  # Default thresholds
        
        # Initialize correlation matrix
        for metric1 in common_metrics:
            self.correlation_matrix[metric1] = {}
            for metric2 in common_metrics:
                self.correlation_matrix[metric1][metric2] = 0.0
    
    def _load_historical_data(self):
        """Load historical performance data if available."""
        # This would typically load from a database or file
        # For now, we'll start with empty data
        logger.info("Starting with empty historical data")
    
    def start_analysis(self):
        """Start continuous performance analysis."""
        if self.analysis_active:
            logger.warning("Analysis is already active")
            return
        
        self.analysis_active = True
        self.analyzer_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analyzer_thread.start()
        logger.info("Intelligent performance analysis started")
    
    def stop_analysis(self):
        """Stop continuous performance analysis."""
        self.analysis_active = False
        if self.analyzer_thread:
            self.analyzer_thread.join(timeout=5.0)
        logger.info("Intelligent performance analysis stopped")
    
    def _analysis_loop(self):
        """Main analysis loop."""
        while self.analysis_active:
            try:
                # Perform comprehensive analysis
                self._perform_comprehensive_analysis()
                
                # Wait for next analysis cycle
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(self.analysis_interval)
    
    def _perform_comprehensive_analysis(self):
        """Perform comprehensive performance analysis."""
        start_time = time.time()
        
        try:
            logger.info("Starting comprehensive performance analysis")
            
            # Collect current metrics
            current_metrics = self._collect_current_metrics()
            
            # Update metrics history
            self._update_metrics_history(current_metrics)
            
            # Perform various analyses
            patterns_found = []
            recommendations = []
            
            # Trend analysis
            if self.analysis_capabilities['trend_analysis']:
                trends = self._analyze_trends()
                patterns_found.extend(trends)
            
            # Cycle detection
            if self.analysis_capabilities['cycle_detection']:
                cycles = self._detect_cycles()
                patterns_found.extend(cycles)
            
            # Anomaly detection
            if self.analysis_capabilities['anomaly_detection']:
                anomalies = self._detect_anomalies()
                patterns_found.extend(anomalies)
            
            # Correlation analysis
            if self.analysis_capabilities['correlation_analysis']:
                correlations = self._analyze_correlations()
                patterns_found.extend(correlations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(patterns_found, current_metrics)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(patterns_found)
            
            # Create analysis result
            result = AnalysisResult(
                timestamp=time.time(),
                analysis_type="comprehensive",
                patterns_found=patterns_found,
                recommendations=recommendations,
                confidence_score=confidence_score,
                execution_time=time.time() - start_time
            )
            
            # Store result
            self.analysis_history.append(result)
            self.detected_patterns.extend(patterns_found)
            
            # Maintain history size
            self._maintain_history_size()
            
            logger.info(f"Analysis completed: {len(patterns_found)} patterns found, confidence: {confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            import psutil
            
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
            
            # Process metrics
            metrics['active_processes'] = len(psutil.pids())
            metrics['active_threads'] = threading.active_count()
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics['network_io_bytes'] = network.bytes_sent + network.bytes_recv
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _update_metrics_history(self, current_metrics: Dict[str, float]):
        """Update metrics history with current values."""
        current_time = time.time()
        
        for metric_name, value in current_metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append((current_time, value))
                
                # Maintain history size
                max_history = self.config.get('max_history_size', 10000)
                if len(self.metrics_history[metric_name]) > max_history:
                    self.metrics_history[metric_name] = self.metrics_history[metric_name][-max_history:]
    
    def _analyze_trends(self) -> List[PerformancePattern]:
        """Analyze performance trends."""
        patterns = []
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < self.config.get('min_data_points', 50):
                continue
            
            # Get recent data for trend analysis
            window_size = self.config.get('trend_window_size', 100)
            recent_data = history[-window_size:]
            
            if len(recent_data) < 10:  # Need minimum data points
                continue
            
            # Calculate trend using linear regression
            timestamps = [t for t, v in recent_data]
            values = [v for t, v in recent_data]
            
            trend_slope, trend_intercept = self._linear_regression(timestamps, values)
            
            # Determine trend direction and confidence
            if abs(trend_slope) > 0.01:  # Significant trend
                trend_direction = "increasing" if trend_slope > 0 else "decreasing"
                confidence = min(0.95, abs(trend_slope) * 100)  # Scale confidence
                
                pattern = PerformancePattern(
                    name=f"{metric_name}_trend",
                    pattern_type="trend",
                    confidence=confidence,
                    description=f"{metric_name} shows {trend_direction} trend (slope: {trend_slope:.4f})",
                    metrics=[metric_name],
                    parameters={
                        'slope': trend_slope,
                        'intercept': trend_intercept,
                        'direction': trend_direction,
                        'window_size': window_size
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cycles(self) -> List[PerformancePattern]:
        """Detect cyclical patterns in performance."""
        patterns = []
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < self.config.get('cycle_detection_window', 200):
                continue
            
            # Get data for cycle detection
            cycle_data = history[-self.config.get('cycle_detection_window', 200):]
            
            if len(cycle_data) < 50:  # Need sufficient data
                continue
            
            # Simple cycle detection using autocorrelation
            values = [v for t, v in cycle_data]
            cycle_period = self._detect_cycle_period(values)
            
            if cycle_period and cycle_period > 5:  # Valid cycle
                confidence = self._calculate_cycle_confidence(values, cycle_period)
                
                if confidence > self.config.get('confidence_threshold', 0.7):
                    pattern = PerformancePattern(
                        name=f"{metric_name}_cycle",
                        pattern_type="cycle",
                        confidence=confidence,
                        description=f"{metric_name} shows cyclical pattern with period {cycle_period}",
                        metrics=[metric_name],
                        parameters={
                            'period': cycle_period,
                            'confidence': confidence
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_anomalies(self) -> List[PerformancePattern]:
        """Detect performance anomalies."""
        patterns = []
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 20:  # Need minimum data for baseline
                continue
            
            # Get recent values
            recent_values = [v for t, v in history[-50:]]
            current_value = recent_values[-1] if recent_values else 0
            
            # Calculate baseline statistics
            mean_val = statistics.mean(recent_values[:-1]) if len(recent_values) > 1 else 0
            std_val = statistics.stdev(recent_values[:-1]) if len(recent_values) > 1 else 0
            
            if std_val > 0:  # Avoid division by zero
                # Calculate z-score for current value
                z_score = abs((current_value - mean_val) / std_val)
                sensitivity = self.config.get('anomaly_sensitivity', 2.0)
                
                if z_score > sensitivity:
                    # Anomaly detected
                    confidence = min(0.95, z_score / (sensitivity * 2))
                    
                    pattern = PerformancePattern(
                        name=f"{metric_name}_anomaly",
                        pattern_type="anomaly",
                        confidence=confidence,
                        description=f"{metric_name} shows anomalous value: {current_value:.2f} (z-score: {z_score:.2f})",
                        metrics=[metric_name],
                        parameters={
                            'current_value': current_value,
                            'baseline_mean': mean_val,
                            'baseline_std': std_val,
                            'z_score': z_score
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_correlations(self) -> List[PerformancePattern]:
        """Analyze correlations between different metrics."""
        patterns = []
        correlation_threshold = self.config.get('correlation_threshold', 0.6)
        
        metrics_list = list(self.metrics_history.keys())
        
        for i, metric1 in enumerate(metrics_list):
            for j, metric2 in enumerate(metrics_list[i+1:], i+1):
                correlation = self._calculate_correlation(metric1, metric2)
                
                if abs(correlation) > correlation_threshold:
                    # Significant correlation found
                    confidence = min(0.95, abs(correlation))
                    correlation_type = "positive" if correlation > 0 else "negative"
                    
                    pattern = PerformancePattern(
                        name=f"{metric1}_{metric2}_correlation",
                        pattern_type="correlation",
                        confidence=confidence,
                        description=f"{metric1} and {metric2} show {correlation_type} correlation ({correlation:.3f})",
                        metrics=[metric1, metric2],
                        parameters={
                            'correlation_coefficient': correlation,
                            'correlation_type': correlation_type
                        }
                    )
                    patterns.append(pattern)
                    
                    # Update correlation matrix
                    self.correlation_matrix[metric1][metric2] = correlation
                    self.correlation_matrix[metric2][metric1] = correlation
        
        return patterns
    
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Perform linear regression on data points."""
        try:
            n = len(x)
            if n < 2:
                return 0.0, 0.0
            
            # Calculate means
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)
            
            # Calculate slope and intercept
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0, y_mean
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            return slope, intercept
            
        except Exception as e:
            logger.error(f"Error in linear regression: {e}")
            return 0.0, 0.0
    
    def _detect_cycle_period(self, values: List[float]) -> Optional[int]:
        """Detect cycle period using autocorrelation."""
        try:
            if len(values) < 20:
                return None
            
            # Simple autocorrelation-based cycle detection
            max_lag = min(len(values) // 2, 50)
            autocorr_values = []
            
            for lag in range(1, max_lag + 1):
                if lag >= len(values):
                    break
                
                # Calculate autocorrelation for this lag
                numerator = 0
                denominator = 0
                
                for i in range(len(values) - lag):
                    numerator += (values[i] - statistics.mean(values)) * (values[i + lag] - statistics.mean(values))
                    denominator += (values[i] - statistics.mean(values)) ** 2
                
                if denominator > 0:
                    autocorr = numerator / denominator
                    autocorr_values.append((lag, autocorr))
            
            # Find lag with highest autocorrelation
            if autocorr_values:
                best_lag = max(autocorr_values, key=lambda x: x[1])
                if best_lag[1] > 0.3:  # Minimum autocorrelation threshold
                    return best_lag[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error in cycle detection: {e}")
            return None
    
    def _calculate_cycle_confidence(self, values: List[float], period: int) -> float:
        """Calculate confidence in detected cycle."""
        try:
            if len(values) < period * 2:
                return 0.0
            
            # Calculate how well the data fits the detected cycle
            cycle_values = []
            for i in range(0, len(values) - period, period):
                cycle_values.append(values[i:i+period])
            
            if len(cycle_values) < 2:
                return 0.0
            
            # Calculate consistency between cycles
            consistency_scores = []
            for i in range(len(cycle_values) - 1):
                cycle1 = cycle_values[i]
                cycle2 = cycle_values[i + 1]
                
                if len(cycle1) == len(cycle2):
                    # Calculate correlation between cycles
                    correlation = self._calculate_correlation_lists(cycle1, cycle2)
                    consistency_scores.append(abs(correlation))
            
            if consistency_scores:
                return statistics.mean(consistency_scores)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cycle confidence: {e}")
            return 0.0
    
    def _calculate_correlation(self, metric1: str, metric2: str) -> float:
        """Calculate correlation between two metrics."""
        try:
            if metric1 not in self.metrics_history or metric2 not in self.metrics_history:
                return 0.0
            
            # Get common time range
            history1 = self.metrics_history[metric1]
            history2 = self.metrics_history[metric2]
            
            if not history1 or not history2:
                return 0.0
            
            # Find common timestamps
            timestamps1 = {t for t, v in history1}
            timestamps2 = {t for t, v in history2}
            common_timestamps = timestamps1.intersection(timestamps2)
            
            if len(common_timestamps) < 10:
                return 0.0
            
            # Get values for common timestamps
            values1 = []
            values2 = []
            
            for t in sorted(common_timestamps):
                # Find closest values for each timestamp
                val1 = self._get_closest_value(history1, t)
                val2 = self._get_closest_value(history2, t)
                
                if val1 is not None and val2 is not None:
                    values1.append(val1)
                    values2.append(val2)
            
            if len(values1) < 10:
                return 0.0
            
            # Calculate correlation
            return self._calculate_correlation_lists(values1, values2)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _get_closest_value(self, history: List[Tuple[float, float]], timestamp: float) -> Optional[float]:
        """Get the closest value to a given timestamp."""
        if not history:
            return None
        
        # Find closest timestamp
        closest_time = min(history, key=lambda x: abs(x[0] - timestamp))
        
        # Return value if within reasonable time window (5 minutes)
        if abs(closest_time[0] - timestamp) < 300:
            return closest_time[1]
        
        return None
    
    def _calculate_correlation_lists(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two lists."""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            n = len(x)
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator_x = sum((x[i] - x_mean) ** 2 for i in range(n))
            denominator_y = sum((y[i] - y_mean) ** 2 for i in range(n))
            
            if denominator_x == 0 or denominator_y == 0:
                return 0.0
            
            correlation = numerator / math.sqrt(denominator_x * denominator_y)
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _generate_recommendations(self, patterns: List[PerformancePattern], current_metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on detected patterns."""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type == "trend":
                if "increasing" in pattern.description:
                    if "cpu_percent" in pattern.metrics:
                        recommendations.append("CPU usage is trending upward. Consider process optimization or scaling.")
                    elif "memory_percent" in pattern.metrics:
                        recommendations.append("Memory usage is trending upward. Consider memory cleanup or optimization.")
                    elif "disk_usage_percent" in pattern.metrics:
                        recommendations.append("Disk usage is trending upward. Consider cleanup or storage expansion.")
            
            elif pattern.pattern_type == "cycle":
                recommendations.append(f"Cyclical pattern detected in {pattern.metrics[0]}. Plan resource allocation accordingly.")
            
            elif pattern.pattern_type == "anomaly":
                metric_name = pattern.metrics[0]
                current_value = current_metrics.get(metric_name, 0)
                recommendations.append(f"Anomaly detected in {metric_name}: {current_value:.2f}. Investigate root cause.")
            
            elif pattern.pattern_type == "correlation":
                if pattern.parameters.get('correlation_type') == "positive":
                    recommendations.append(f"Strong positive correlation between {pattern.metrics[0]} and {pattern.metrics[1]}. Optimize both together.")
                else:
                    recommendations.append(f"Strong negative correlation between {pattern.metrics[0]} and {pattern.metrics[1]}. Balance optimization efforts.")
        
        # Add general recommendations based on current metrics
        if current_metrics.get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected. Consider process optimization.")
        
        if current_metrics.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected. Consider memory cleanup.")
        
        if current_metrics.get('disk_usage_percent', 0) > 85:
            recommendations.append("High disk usage detected. Consider cleanup or expansion.")
        
        if not recommendations:
            recommendations.append("System performance is within normal ranges.")
        
        return recommendations
    
    def _calculate_confidence_score(self, patterns: List[PerformancePattern]) -> float:
        """Calculate overall confidence score for the analysis."""
        if not patterns:
            return 0.0
        
        # Calculate weighted average confidence
        total_confidence = sum(pattern.confidence for pattern in patterns)
        return total_confidence / len(patterns)
    
    def _maintain_history_size(self):
        """Maintain analysis history size within limits."""
        max_history = self.config.get('max_history_size', 10000)
        
        if len(self.analysis_history) > max_history:
            self.analysis_history = self.analysis_history[-max_history:]
        
        if len(self.detected_patterns) > max_history:
            self.detected_patterns = self.detected_patterns[-max_history:]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        if not self.analysis_history:
            return {"error": "No analysis data available"}
        
        recent_analysis = self.analysis_history[-1] if self.analysis_history else None
        
        summary = {
            "analysis_status": {
                "active": self.analysis_active,
                "last_analysis": datetime.fromtimestamp(recent_analysis.timestamp).isoformat() if recent_analysis else None,
                "total_analyses": len(self.analysis_history),
                "patterns_detected": len(self.detected_patterns)
            },
            "recent_patterns": [
                {
                    "name": p.name,
                    "type": p.pattern_type,
                    "confidence": p.confidence,
                    "description": p.description,
                    "detected_at": datetime.fromtimestamp(p.detected_at).isoformat()
                }
                for p in self.detected_patterns[-10:]  # Last 10 patterns
            ],
            "correlation_matrix": self.correlation_matrix,
            "metrics_coverage": {
                metric: len(history) for metric, history in self.metrics_history.items()
            }
        }
        
        if recent_analysis:
            summary["last_analysis_result"] = {
                "patterns_found": len(recent_analysis.patterns_found),
                "recommendations": recent_analysis.recommendations,
                "confidence_score": recent_analysis.confidence_score,
                "execution_time": recent_analysis.execution_time
            }
        
        return summary
    
    def get_predictions(self, metric_name: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get performance predictions for a specific metric."""
        try:
            if metric_name not in self.metrics_history:
                return {"error": f"Metric {metric_name} not found"}
            
            history = self.metrics_history[metric_name]
            if len(history) < 20:
                return {"error": "Insufficient data for prediction"}
            
            # Get recent data for prediction
            recent_data = history[-100:]  # Last 100 data points
            timestamps = [t for t, v in recent_data]
            values = [v for t, v in recent_data]
            
            # Perform linear regression for trend prediction
            slope, intercept = self._linear_regression(timestamps, values)
            
            # Calculate prediction
            current_time = time.time()
            future_time = current_time + (hours_ahead * 3600)
            
            predicted_value = slope * future_time + intercept
            
            # Calculate confidence interval (simplified)
            if len(values) > 1:
                std_error = statistics.stdev(values)
                confidence_interval = 1.96 * std_error  # 95% confidence
            else:
                confidence_interval = 0
            
            return {
                "metric": metric_name,
                "prediction_horizon_hours": hours_ahead,
                "predicted_value": predicted_value,
                "confidence_interval": confidence_interval,
                "trend_slope": slope,
                "prediction_confidence": min(0.95, abs(slope) * 1000),  # Scale confidence
                "current_value": values[-1] if values else 0,
                "prediction_timestamp": datetime.fromtimestamp(future_time).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return {"error": f"Prediction failed: {e}"}
    
    def export_analysis_report(self, format: str = "json", filepath: Optional[str] = None) -> str:
        """Export analysis report to various formats."""
        if format not in ['json', 'html']:
            raise ValueError(f"Unsupported format: {format}")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"heygen_ai_analysis_{timestamp}.{format}"
        
        if format == "json":
            return self._export_json_report(filepath)
        elif format == "html":
            return self._export_html_report(filepath)
        
        raise ValueError(f"Export format {format} not implemented")
    
    def _export_json_report(self, filepath: str) -> str:
        """Export analysis report to JSON format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "analysis_summary": self.get_analysis_summary(),
            "detected_patterns": [
                {
                    "name": p.name,
                    "type": p.pattern_type,
                    "confidence": p.confidence,
                    "description": p.description,
                    "detected_at": datetime.fromtimestamp(p.detected_at).isoformat(),
                    "parameters": p.parameters
                }
                for p in self.detected_patterns
            ],
            "analysis_history": [
                {
                    "timestamp": datetime.fromtimestamp(r.timestamp).isoformat(),
                    "analysis_type": r.analysis_type,
                    "patterns_found": len(r.patterns_found),
                    "recommendations": r.recommendations,
                    "confidence_score": r.confidence_score,
                    "execution_time": r.execution_time
                }
                for r in self.analysis_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def _export_html_report(self, filepath: str) -> str:
        """Export analysis report to HTML format."""
        summary = self.get_analysis_summary()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HeyGen AI Performance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .pattern {{ margin: 10px 0; padding: 10px; border-left: 4px solid #4ecdc4; background: #f0fffd; }}
                .high-confidence {{ border-left-color: #2ecc71; }}
                .medium-confidence {{ border-left-color: #f39c12; }}
                .low-confidence {{ border-left-color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 HeyGen AI Performance Analysis Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>📊 Analysis Status</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Active</td><td>{summary.get('analysis_status', {}).get('active', False)}</td></tr>
                    <tr><td>Total Analyses</td><td>{summary.get('analysis_status', {}).get('total_analyses', 0)}</td></tr>
                    <tr><td>Patterns Detected</td><td>{summary.get('analysis_status', {}).get('patterns_detected', 0)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Recent Patterns</h2>
        """
        
        recent_patterns = summary.get('recent_patterns', [])
        for pattern in recent_patterns:
            confidence_class = "high-confidence" if pattern['confidence'] > 0.8 else "medium-confidence" if pattern['confidence'] > 0.6 else "low-confidence"
            html_content += f"""
                <div class="pattern {confidence_class}">
                    <h3>{pattern['name']}</h3>
                    <p><strong>Type:</strong> {pattern['type']}</p>
                    <p><strong>Confidence:</strong> {pattern['confidence']:.2f}</p>
                    <p><strong>Description:</strong> {pattern['description']}</p>
                    <p><strong>Detected:</strong> {pattern['detected_at']}</p>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath

def main():
    """Demo function for the intelligent analyzer."""
    print("🧠 HeyGen AI Intelligent Analyzer Demo")
    print("=" * 50)
    
    # Create analyzer instance
    analyzer = IntelligentAnalyzer()
    
    try:
        # Show initial status
        print("\n📊 Initial Analysis Status:")
        summary = analyzer.get_analysis_summary()
        print(json.dumps(summary, indent=2, default=str))
        
        # Start analysis
        print("\n🚀 Starting intelligent analysis...")
        analyzer.start_analysis()
        
        # Let it run for a bit
        print("Running analysis for 120 seconds...")
        time.sleep(120)
        
        # Show final status
        print("\n📊 Final Analysis Status:")
        summary = analyzer.get_analysis_summary()
        print(json.dumps(summary, indent=2, default=str))
        
        # Get predictions
        print("\n🔮 Performance Predictions:")
        for metric in ['cpu_percent', 'memory_percent', 'disk_usage_percent']:
            prediction = analyzer.get_predictions(metric, hours_ahead=24)
            print(f"{metric}: {json.dumps(prediction, indent=2, default=str)}")
        
        # Export report
        print("\n📁 Exporting analysis report...")
        json_file = analyzer.export_analysis_report("json")
        print(f"JSON report: {json_file}")
        
        html_file = analyzer.export_analysis_report("html")
        print(f"HTML report: {html_file}")
        
        # Stop analysis
        analyzer.stop_analysis()
        print("\n✅ Intelligent analyzer demo completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        analyzer.stop_analysis()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        analyzer.stop_analysis()

if __name__ == "__main__":
    main()
