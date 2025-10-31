#!/usr/bin/env python3
"""
Predictive Analytics System for Enhanced Unified AI Interface v3.5
Machine learning-powered performance prediction and proactive optimization
"""
import time
import threading
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalyticsSystem:
    """Advanced predictive analytics system with machine learning capabilities"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.prediction_models = {}
        self.historical_data = []
        self.predictions_history = []
        self.is_analyzing = False
        self.analysis_thread = None
        self.performance_monitor = None
        self.optimization_engine = None
        self.prediction_callbacks = []
        
        # Initialize prediction state
        self.current_predictions = {}
        self.prediction_confidence = 0.0
        self.last_analysis_time = None
        
        # Load prediction models
        self._initialize_prediction_models()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'analysis_interval': 60.0,  # seconds between analyses
            'enable_predictive_optimization': True,
            'enable_anomaly_detection': True,
            'enable_trend_analysis': True,
            'prediction_horizon': 300,  # seconds to predict ahead
            'confidence_threshold': 0.75,  # minimum confidence for predictions
            'anomaly_threshold': 2.0,  # standard deviations for anomaly detection
            'trend_window': 300,  # seconds for trend analysis
            'learning_parameters': {
                'learning_rate': 0.001,
                'memory_size': 2000,
                'update_frequency': 50,
                'feature_importance_threshold': 0.1
            },
            'prediction_types': {
                'performance_degradation': True,
                'resource_exhaustion': True,
                'system_failure': True,
                'optimization_opportunity': True,
                'capacity_planning': True
            }
        }
    
    def _initialize_prediction_models(self):
        """Initialize prediction models"""
        try:
            # Performance degradation prediction model
            self.prediction_models['performance_degradation'] = {
                'name': 'Performance Degradation Predictor',
                'description': 'Predicts when system performance will degrade',
                'features': ['cpu_usage', 'memory_usage', 'disk_usage', 'gpu_usage', 'load_avg'],
                'target': 'performance_score',
                'model_type': 'regression',
                'prediction_window': 60,  # seconds
                'confidence': 0.0,
                'last_trained': None,
                'training_data_size': 0
            }
            
            # Resource exhaustion prediction model
            self.prediction_models['resource_exhaustion'] = {
                'name': 'Resource Exhaustion Predictor',
                'description': 'Predicts when system resources will be exhausted',
                'features': ['cpu_usage', 'memory_usage', 'disk_usage', 'gpu_usage'],
                'target': 'resource_availability',
                'model_type': 'classification',
                'prediction_window': 120,  # seconds
                'confidence': 0.0,
                'last_trained': None,
                'training_data_size': 0
            }
            
            # System failure prediction model
            self.prediction_models['system_failure'] = {
                'name': 'System Failure Predictor',
                'description': 'Predicts potential system failures',
                'features': ['error_rate', 'response_time', 'system_load', 'memory_pressure'],
                'target': 'failure_probability',
                'model_type': 'classification',
                'prediction_window': 300,  # seconds
                'confidence': 0.0,
                'last_trained': None,
                'training_data_size': 0
            }
            
            # Optimization opportunity prediction model
            self.prediction_models['optimization_opportunity'] = {
                'name': 'Optimization Opportunity Predictor',
                'description': 'Identifies when optimization would be most beneficial',
                'features': ['performance_score', 'resource_utilization', 'workload_intensity'],
                'target': 'optimization_benefit',
                'model_type': 'regression',
                'prediction_window': 180,  # seconds
                'confidence': 0.0,
                'last_trained': None,
                'training_data_size': 0
            }
            
            # Capacity planning prediction model
            self.prediction_models['capacity_planning'] = {
                'name': 'Capacity Planning Predictor',
                'description': 'Predicts future resource requirements',
                'features': ['historical_usage', 'growth_rate', 'seasonal_patterns'],
                'target': 'resource_requirement',
                'model_type': 'regression',
                'prediction_window': 3600,  # 1 hour
                'confidence': 0.0,
                'last_trained': None,
                'training_data_size': 0
            }
            
            print("🧠 Prediction models initialized")
            
        except Exception as e:
            print(f"❌ Error initializing prediction models: {e}")
    
    def set_performance_monitor(self, monitor):
        """Set the performance monitor instance"""
        self.performance_monitor = monitor
        print("🔗 Performance monitor connected to predictive analytics")
    
    def set_optimization_engine(self, engine):
        """Set the optimization engine instance"""
        self.optimization_engine = engine
        print("🔗 Optimization engine connected to predictive analytics")
    
    def start_analysis(self):
        """Start predictive analysis"""
        if self.is_analyzing:
            return False
        
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("🚀 Predictive analytics started")
        return True
    
    def stop_analysis(self):
        """Stop predictive analysis"""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1.0)
        print("🛑 Predictive analytics stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_analyzing:
            try:
                # Collect current data
                current_data = self._collect_current_data()
                
                # Store historical data
                self._store_historical_data(current_data)
                
                # Perform predictions
                predictions = self._perform_predictions(current_data)
                
                # Analyze trends
                trends = self._analyze_trends()
                
                # Detect anomalies
                anomalies = self._detect_anomalies(current_data)
                
                # Generate insights
                insights = self._generate_insights(predictions, trends, anomalies)
                
                # Update current predictions
                self.current_predictions = insights
                
                # Trigger callbacks
                self._trigger_prediction_callbacks(insights)
                
                # Wait for next analysis cycle
                time.sleep(self.config['analysis_interval'])
                
            except Exception as e:
                print(f"❌ Predictive analysis error: {e}")
                time.sleep(10.0)  # Wait longer on error
    
    def _collect_current_data(self) -> Dict:
        """Collect current system data for analysis"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'derived_features': {}
            }
            
            # Get performance metrics
            if self.performance_monitor:
                metrics = self.performance_monitor.get_current_metrics()
                data['metrics'] = metrics
                
                # Extract key metrics
                cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
                memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
                disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
                load_avg = metrics.get('cpu', {}).get('load_avg_1m', 0)
                
                # Calculate derived features
                data['derived_features'] = {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'load_avg': load_avg,
                    'gpu_usage': self._extract_gpu_usage(metrics),
                    'performance_score': self.performance_monitor.get_performance_score(),
                    'resource_utilization': (cpu_usage + memory_usage + disk_usage) / 3,
                    'system_pressure': (cpu_usage * 0.4 + memory_usage * 0.4 + load_avg * 20) / 100,
                    'workload_intensity': load_avg / max(metrics.get('cpu', {}).get('count', 1), 1)
                }
            else:
                # Use simulated data for testing
                data['derived_features'] = self._get_simulated_features()
            
            return data
            
        except Exception as e:
            print(f"❌ Error collecting data: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def _extract_gpu_usage(self, metrics: Dict) -> float:
        """Extract GPU usage from metrics"""
        try:
            if 'gpu' in metrics and 'gpu_0' in metrics['gpu']:
                return metrics['gpu']['gpu_0'].get('memory_usage_percent', 0)
            return 0.0
        except:
            return 0.0
    
    def _get_simulated_features(self) -> Dict:
        """Get simulated features for testing"""
        return {
            'cpu_usage': np.random.randint(30, 90),
            'memory_usage': np.random.randint(40, 95),
            'disk_usage': np.random.randint(20, 85),
            'gpu_usage': np.random.randint(10, 80),
            'load_avg': np.random.uniform(0.5, 3.0),
            'performance_score': np.random.uniform(60, 95),
            'resource_utilization': np.random.uniform(0.3, 0.9),
            'system_pressure': np.random.uniform(0.2, 0.8),
            'workload_intensity': np.random.uniform(0.1, 0.7)
        }
    
    def _store_historical_data(self, data: Dict):
        """Store data in historical database"""
        try:
            self.historical_data.append(data)
            
            # Limit historical data size
            max_size = self.config['learning_parameters']['memory_size']
            if len(self.historical_data) > max_size:
                self.historical_data.pop(0)
                
        except Exception as e:
            print(f"❌ Error storing historical data: {e}")
    
    def _perform_predictions(self, current_data: Dict) -> Dict:
        """Perform predictions using all models"""
        try:
            predictions = {}
            
            for model_name, model in self.prediction_models.items():
                if not self.config['prediction_types'].get(model_name, True):
                    continue
                
                try:
                    prediction = self._predict_with_model(model, current_data)
                    predictions[model_name] = prediction
                except Exception as e:
                    predictions[model_name] = {
                        'error': str(e),
                        'confidence': 0.0,
                        'prediction': None
                    }
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error performing predictions: {e}")
            return {}
    
    def _predict_with_model(self, model: Dict, current_data: Dict) -> Dict:
        """Make prediction using specific model"""
        try:
            # Extract features
            features = self._extract_model_features(model, current_data)
            
            if not features:
                return {
                    'prediction': None,
                    'confidence': 0.0,
                    'error': 'Insufficient features'
                }
            
            # Make prediction based on model type
            if model['model_type'] == 'regression':
                prediction = self._regression_prediction(model, features)
            elif model['model_type'] == 'classification':
                prediction = self._classification_prediction(model, features)
            else:
                prediction = self._baseline_prediction(model, features)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(model, features)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'features_used': list(features.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _extract_model_features(self, model: Dict, current_data: Dict) -> Dict:
        """Extract features required by the model"""
        try:
            features = {}
            derived_features = current_data.get('derived_features', {})
            
            for feature_name in model.get('features', []):
                if feature_name in derived_features:
                    features[feature_name] = derived_features[feature_name]
                else:
                    # Try to find in metrics
                    value = self._find_feature_in_metrics(feature_name, current_data.get('metrics', {}))
                    if value is not None:
                        features[feature_name] = value
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return {}
    
    def _find_feature_in_metrics(self, feature_name: str, metrics: Dict) -> Optional[float]:
        """Find feature value in metrics dictionary"""
        try:
            if 'cpu' in feature_name:
                return metrics.get('cpu', {}).get('usage_percent', 0)
            elif 'memory' in feature_name:
                return metrics.get('memory', {}).get('usage_percent', 0)
            elif 'disk' in feature_name:
                return metrics.get('disk', {}).get('usage_percent', 0)
            elif 'gpu' in feature_name:
                if 'gpu_0' in metrics.get('gpu', {}):
                    return metrics['gpu']['gpu_0'].get('memory_usage_percent', 0)
            elif 'load' in feature_name:
                return metrics.get('cpu', {}).get('load_avg_1m', 0)
            elif 'response_time' in feature_name:
                # Simulate response time
                return np.random.uniform(50, 200)
            elif 'error_rate' in feature_name:
                # Simulate error rate
                return np.random.uniform(0, 0.05)
            
            return None
            
        except Exception as e:
            print(f"❌ Error finding feature {feature_name}: {e}")
            return None
    
    def _regression_prediction(self, model: Dict, features: Dict) -> float:
        """Make regression prediction"""
        try:
            # Simple linear regression model
            if model['name'] == 'Performance Degradation Predictor':
                # Predict performance score degradation
                cpu_weight = -0.3
                memory_weight = -0.3
                disk_weight = -0.2
                gpu_weight = -0.2
                
                prediction = 100.0
                prediction += features.get('cpu_usage', 0) * cpu_weight
                prediction += features.get('memory_usage', 0) * memory_weight
                prediction += features.get('disk_usage', 0) * disk_weight
                prediction += features.get('gpu_usage', 0) * gpu_weight
                
                return max(0.0, min(100.0, prediction))
                
            elif model['name'] == 'Capacity Planning Predictor':
                # Predict resource requirements
                current_usage = features.get('resource_utilization', 0.5)
                growth_rate = 0.1  # 10% growth per hour
                prediction_horizon = model['prediction_window'] / 3600  # hours
                
                prediction = current_usage * (1 + growth_rate) ** prediction_horizon
                return min(1.0, prediction)
                
            else:
                # Default regression prediction
                return np.mean(list(features.values()))
                
        except Exception as e:
            print(f"❌ Error in regression prediction: {e}")
            return 0.0
    
    def _classification_prediction(self, model: Dict, features: Dict) -> str:
        """Make classification prediction"""
        try:
            if model['name'] == 'Resource Exhaustion Predictor':
                # Predict resource exhaustion
                cpu_usage = features.get('cpu_usage', 0)
                memory_usage = features.get('memory_usage', 0)
                disk_usage = features.get('disk_usage', 0)
                
                if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
                    return 'high_risk'
                elif cpu_usage > 80 or memory_usage > 85 or disk_usage > 90:
                    return 'medium_risk'
                else:
                    return 'low_risk'
                    
            elif model['name'] == 'System Failure Predictor':
                # Predict system failure
                system_pressure = features.get('system_pressure', 0)
                error_rate = features.get('error_rate', 0)
                
                if system_pressure > 0.8 or error_rate > 0.03:
                    return 'high_risk'
                elif system_pressure > 0.6 or error_rate > 0.01:
                    return 'medium_risk'
                else:
                    return 'low_risk'
                    
            else:
                # Default classification
                return 'unknown'
                
        except Exception as e:
            print(f"❌ Error in classification prediction: {e}")
            return 'error'
    
    def _baseline_prediction(self, model: Dict, features: Dict) -> Any:
        """Make baseline prediction"""
        try:
            # Simple baseline prediction
            if model['name'] == 'Optimization Opportunity Predictor':
                performance_score = features.get('performance_score', 75)
                resource_utilization = features.get('resource_utilization', 0.5)
                
                if performance_score < 70 and resource_utilization > 0.7:
                    return 'high_opportunity'
                elif performance_score < 80 and resource_utilization > 0.6:
                    return 'medium_opportunity'
                else:
                    return 'low_opportunity'
            else:
                return 'baseline_prediction'
                
        except Exception as e:
            print(f"❌ Error in baseline prediction: {e}")
            return 'error'
    
    def _calculate_prediction_confidence(self, model: Dict, features: Dict) -> float:
        """Calculate confidence in prediction"""
        try:
            # Base confidence on feature availability and model training
            feature_coverage = len(features) / len(model.get('features', []))
            training_confidence = min(model.get('training_data_size', 0) / 100, 1.0)
            
            # Feature quality confidence
            feature_confidence = 0.0
            for feature_name, value in features.items():
                if isinstance(value, (int, float)) and value >= 0:
                    feature_confidence += 1.0
            
            feature_confidence /= len(features) if features else 1
            
            # Overall confidence
            confidence = (feature_coverage * 0.4 + training_confidence * 0.4 + feature_confidence * 0.2)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            print(f"❌ Error calculating confidence: {e}")
            return 0.0
    
    def _analyze_trends(self) -> Dict:
        """Analyze trends in historical data"""
        try:
            if len(self.historical_data) < 10:
                return {'error': 'Insufficient data for trend analysis'}
            
            trends = {}
            
            # Extract time series data
            timestamps = [pd.to_datetime(d['timestamp']) for d in self.historical_data]
            cpu_usage = [d.get('derived_features', {}).get('cpu_usage', 0) for d in self.historical_data]
            memory_usage = [d.get('derived_features', {}).get('memory_usage', 0) for d in self.historical_data]
            performance_scores = [d.get('derived_features', {}).get('performance_score', 75) for d in self.historical_data]
            
            # Calculate trends
            if len(cpu_usage) >= 2:
                cpu_trend = (cpu_usage[-1] - cpu_usage[0]) / len(cpu_usage)
                memory_trend = (memory_usage[-1] - memory_usage[0]) / len(memory_usage)
                performance_trend = (performance_scores[-1] - performance_scores[0]) / len(performance_scores)
                
                trends = {
                    'cpu_trend': cpu_trend,
                    'memory_trend': memory_trend,
                    'performance_trend': performance_trend,
                    'trend_direction': {
                        'cpu': 'increasing' if cpu_trend > 0 else 'decreasing',
                        'memory': 'increasing' if memory_trend > 0 else 'decreasing',
                        'performance': 'improving' if performance_trend > 0 else 'degrading'
                    },
                    'trend_strength': {
                        'cpu': abs(cpu_trend),
                        'memory': abs(memory_trend),
                        'performance': abs(performance_trend)
                    }
                }
            
            return trends
            
        except Exception as e:
            return {'error': f'Error analyzing trends: {str(e)}'}
    
    def _detect_anomalies(self, current_data: Dict) -> Dict:
        """Detect anomalies in current data"""
        try:
            if len(self.historical_data) < 5:
                return {'error': 'Insufficient data for anomaly detection'}
            
            anomalies = {}
            threshold = self.config['anomaly_threshold']
            
            # Extract current values
            current_features = current_data.get('derived_features', {})
            
            # Check each feature for anomalies
            for feature_name, current_value in current_features.items():
                if isinstance(current_value, (int, float)):
                    # Get historical values for this feature
                    historical_values = []
                    for data_point in self.historical_data[-20:]:  # Last 20 data points
                        value = data_point.get('derived_features', {}).get(feature_name)
                        if value is not None:
                            historical_values.append(value)
                    
                    if len(historical_values) >= 3:
                        # Calculate statistics
                        mean_val = np.mean(historical_values)
                        std_val = np.std(historical_values)
                        
                        if std_val > 0:
                            # Calculate z-score
                            z_score = abs(current_value - mean_val) / std_val
                            
                            if z_score > threshold:
                                anomalies[feature_name] = {
                                    'current_value': current_value,
                                    'expected_range': [mean_val - threshold * std_val, mean_val + threshold * std_val],
                                    'z_score': z_score,
                                    'severity': 'high' if z_score > threshold * 2 else 'medium'
                                }
            
            return anomalies
            
        except Exception as e:
            return {'error': f'Error detecting anomalies: {str(e)}'}
    
    def _generate_insights(self, predictions: Dict, trends: Dict, anomalies: Dict) -> Dict:
        """Generate comprehensive insights from all analyses"""
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'trends': trends,
                'anomalies': anomalies,
                'recommendations': [],
                'risk_assessment': 'low',
                'overall_confidence': 0.0
            }
            
            # Generate recommendations
            recommendations = []
            
            # Performance degradation recommendations
            if 'performance_degradation' in predictions:
                pred = predictions['performance_degradation']
                if pred.get('prediction', 100) < 70 and pred.get('confidence', 0) > 0.7:
                    recommendations.append({
                        'type': 'performance',
                        'priority': 'high',
                        'action': 'Immediate performance optimization recommended',
                        'reason': f"Performance predicted to degrade to {pred['prediction']:.1f}",
                        'confidence': pred.get('confidence', 0)
                    })
            
            # Resource exhaustion recommendations
            if 'resource_exhaustion' in predictions:
                pred = predictions['resource_exhaustion']
                if pred.get('prediction') == 'high_risk' and pred.get('confidence', 0) > 0.7:
                    recommendations.append({
                        'type': 'resource',
                        'priority': 'critical',
                        'action': 'Resource exhaustion imminent - immediate action required',
                        'reason': 'High risk of resource exhaustion detected',
                        'confidence': pred.get('confidence', 0)
                    })
            
            # Anomaly-based recommendations
            for feature_name, anomaly in anomalies.items():
                if isinstance(anomaly, dict) and anomaly.get('severity') == 'high':
                    recommendations.append({
                        'type': 'anomaly',
                        'priority': 'high',
                        'action': f'Investigate {feature_name} anomaly',
                        'reason': f"Unusual {feature_name} value: {anomaly['current_value']:.1f}",
                        'confidence': 0.9
                    })
            
            # Trend-based recommendations
            if 'trends' in trends and not isinstance(trends, dict):
                if trends.get('performance_trend', 0) < -0.5:
                    recommendations.append({
                        'type': 'trend',
                        'priority': 'medium',
                        'action': 'Performance trending downward - consider optimization',
                        'reason': 'Performance degradation trend detected',
                        'confidence': 0.7
                    })
            
            insights['recommendations'] = recommendations
            
            # Risk assessment
            risk_factors = []
            if any(r['priority'] == 'critical' for r in recommendations):
                risk_factors.append('critical')
            if any(r['priority'] == 'high' for r in recommendations):
                risk_factors.append('high')
            if any(r['priority'] == 'medium' for r in recommendations):
                risk_factors.append('medium')
            
            if 'critical' in risk_factors:
                insights['risk_assessment'] = 'critical'
            elif 'high' in risk_factors:
                insights['risk_assessment'] = 'high'
            elif 'medium' in risk_factors:
                insights['risk_assessment'] = 'medium'
            else:
                insights['risk_assessment'] = 'low'
            
            # Overall confidence
            confidences = [pred.get('confidence', 0) for pred in predictions.values() if isinstance(pred, dict)]
            insights['overall_confidence'] = np.mean(confidences) if confidences else 0.0
            
            return insights
            
        except Exception as e:
            return {'error': f'Error generating insights: {str(e)}'}
    
    def _trigger_prediction_callbacks(self, insights: Dict):
        """Trigger prediction callbacks"""
        for callback in self.prediction_callbacks:
            try:
                callback(insights)
            except Exception as e:
                print(f"❌ Error in prediction callback: {e}")
    
    def add_prediction_callback(self, callback: Callable):
        """Add prediction callback function"""
        self.prediction_callbacks.append(callback)
    
    def get_current_predictions(self) -> Dict:
        """Get current predictions"""
        return self.current_predictions.copy()
    
    def get_prediction_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get prediction history"""
        if limit is None:
            return self.predictions_history.copy()
        else:
            return self.predictions_history[-limit:].copy()
    
    def get_historical_data(self, limit: Optional[int] = None) -> List[Dict]:
        """Get historical data"""
        if limit is None:
            return self.historical_data.copy()
        else:
            return self.historical_data[-limit:].copy()
    
    def get_analytics_summary(self) -> Dict:
        """Get analytics summary"""
        try:
            if not self.predictions_history:
                return {'error': 'No prediction history available'}
            
            # Calculate statistics
            total_predictions = len(self.predictions_history)
            high_confidence_predictions = len([
                p for p in self.predictions_history 
                if p.get('overall_confidence', 0) > 0.8
            ])
            
            # Risk assessment distribution
            risk_distribution = {}
            for prediction in self.predictions_history:
                risk = prediction.get('risk_assessment', 'unknown')
                risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
            
            # Recommendation types
            recommendation_types = {}
            for prediction in self.predictions_history:
                for rec in prediction.get('recommendations', []):
                    rec_type = rec.get('type', 'unknown')
                    recommendation_types[rec_type] = recommendation_types.get(rec_type, 0) + 1
            
            return {
                'total_predictions': total_predictions,
                'high_confidence_predictions': high_confidence_predictions,
                'confidence_rate': high_confidence_predictions / total_predictions if total_predictions > 0 else 0,
                'risk_distribution': risk_distribution,
                'recommendation_types': recommendation_types,
                'historical_data_size': len(self.historical_data),
                'last_prediction': self.predictions_history[-1] if self.predictions_history else None
            }
            
        except Exception as e:
            return {'error': f'Error generating summary: {str(e)}'}
    
    def export_analytics_data(self, format: str = 'json') -> str:
        """Export analytics data"""
        try:
            if format.lower() == 'json':
                return json.dumps({
                    'predictions_history': self.predictions_history,
                    'historical_data': self.historical_data[-100:],  # Last 100 data points
                    'summary': self.get_analytics_summary()
                }, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to DataFrame and export
                df = pd.DataFrame(self.predictions_history)
                return df.to_csv(index=False)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            return f"Error exporting data: {str(e)}"
    
    def clear_history(self):
        """Clear prediction history and historical data"""
        self.predictions_history.clear()
        self.historical_data.clear()
        print("🗑️ Analytics history cleared")
    
    def update_config(self, new_config: Dict):
        """Update analytics configuration"""
        self.config.update(new_config)
        print("⚙️ Analytics configuration updated")

# Example usage and testing
if __name__ == "__main__":
    # Create predictive analytics system
    analytics = PredictiveAnalyticsSystem()
    
    # Add prediction callback
    def prediction_handler(insights):
        print(f"🔮 New insights generated:")
        print(f"   Risk assessment: {insights.get('risk_assessment', 'unknown')}")
        print(f"   Confidence: {insights.get('overall_confidence', 0):.2f}")
        print(f"   Recommendations: {len(insights.get('recommendations', []))}")
    
    analytics.add_prediction_callback(prediction_handler)
    
    # Start analysis
    analytics.start_analysis()
    
    try:
        # Run for 120 seconds
        time.sleep(120)
        
        # Print summary
        summary = analytics.get_analytics_summary()
        print("\n📊 Analytics Summary:")
        print(f"Total predictions: {summary['total_predictions']}")
        print(f"Confidence rate: {summary['confidence_rate']:.2%}")
        print(f"Risk distribution: {summary['risk_distribution']}")
        
    finally:
        # Stop analysis
        analytics.stop_analysis()
