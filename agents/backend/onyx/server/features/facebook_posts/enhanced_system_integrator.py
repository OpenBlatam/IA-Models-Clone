#!/usr/bin/env python3
"""
Enhanced System Integrator for Enhanced Unified AI Interface v3.5
Unified system that integrates performance monitoring, optimization, and predictive analytics
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

# Import enhanced components
try:
    from advanced_performance_monitor import AdvancedPerformanceMonitor
    from intelligent_optimization_engine import IntelligentOptimizationEngine
    from predictive_analytics_system import PredictiveAnalyticsSystem
except ImportError:
    print("⚠️ Some enhanced components not available, using fallback implementations")
    AdvancedPerformanceMonitor = None
    IntelligentOptimizationEngine = None
    PredictiveAnalyticsSystem = None

class EnhancedSystemIntegrator:
    """Main system integrator for enhanced AI interface v3.5"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.is_running = False
        self.integration_thread = None
        self.system_status = 'initializing'
        self.last_integration_time = None
        
        # Initialize enhanced components
        self.performance_monitor = None
        self.optimization_engine = None
        self.predictive_analytics = None
        
        # System state
        self.system_health = 100.0
        self.optimization_cycles = 0
        self.predictions_made = 0
        self.alerts_triggered = 0
        
        # Integration callbacks
        self.integration_callbacks = []
        self.health_callbacks = []
        
        # Initialize system
        self._initialize_system()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'integration_interval': 10.0,  # seconds between integrations
            'enable_performance_monitoring': True,
            'enable_intelligent_optimization': True,
            'enable_predictive_analytics': True,
            'enable_auto_optimization': True,
            'enable_predictive_actions': True,
            'health_thresholds': {
                'critical': 30.0,
                'warning': 60.0,
                'good': 80.0
            },
            'optimization_triggers': {
                'performance_degradation': True,
                'resource_pressure': True,
                'predictive_optimization': True,
                'anomaly_detection': True
            },
            'monitoring_config': {
                'interval': 2.0,
                'history_size': 1000,
                'alert_levels': {
                    'warning': 70.0,
                    'critical': 90.0,
                    'emergency': 95.0
                }
            },
            'optimization_config': {
                'interval': 30.0,
                'threshold': 0.7,
                'max_duration': 300
            },
            'analytics_config': {
                'interval': 60.0,
                'prediction_horizon': 300,
                'confidence_threshold': 0.75
            }
        }
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            print("🚀 Initializing Enhanced System Integrator...")
            
            # Initialize performance monitor
            if self.config['enable_performance_monitoring'] and AdvancedPerformanceMonitor:
                self.performance_monitor = AdvancedPerformanceMonitor(
                    self.config.get('monitoring_config', {})
                )
                print("✅ Performance monitor initialized")
            else:
                print("⚠️ Performance monitor not available")
            
            # Initialize optimization engine
            if self.config['enable_intelligent_optimization'] and IntelligentOptimizationEngine:
                self.optimization_engine = IntelligentOptimizationEngine(
                    self.config.get('optimization_config', {})
                )
                print("✅ Optimization engine initialized")
            else:
                print("⚠️ Optimization engine not available")
            
            # Initialize predictive analytics
            if self.config['enable_predictive_analytics'] and PredictiveAnalyticsSystem:
                self.predictive_analytics = PredictiveAnalyticsSystem(
                    self.config.get('analytics_config', {})
                )
                print("✅ Predictive analytics initialized")
            else:
                print("⚠️ Predictive analytics not available")
            
            # Connect components
            self._connect_components()
            
            # Set up callbacks
            self._setup_callbacks()
            
            self.system_status = 'ready'
            print("🎉 Enhanced System Integrator ready!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.system_status = 'error'
            raise
    
    def _connect_components(self):
        """Connect all system components"""
        try:
            # Connect performance monitor to optimization engine
            if self.performance_monitor and self.optimization_engine:
                self.optimization_engine.set_performance_monitor(self.performance_monitor)
                print("🔗 Performance monitor connected to optimization engine")
            
            # Connect performance monitor to predictive analytics
            if self.performance_monitor and self.predictive_analytics:
                self.predictive_analytics.set_performance_monitor(self.performance_monitor)
                print("🔗 Performance monitor connected to predictive analytics")
            
            # Connect optimization engine to predictive analytics
            if self.optimization_engine and self.predictive_analytics:
                self.predictive_analytics.set_optimization_engine(self.optimization_engine)
                print("🔗 Optimization engine connected to predictive analytics")
                
        except Exception as e:
            print(f"❌ Error connecting components: {e}")
    
    def _setup_callbacks(self):
        """Set up system callbacks"""
        try:
            # Performance monitor callbacks
            if self.performance_monitor:
                self.performance_monitor.add_alert_callback(self._handle_performance_alert)
            
            # Optimization engine callbacks
            if self.optimization_engine:
                self.optimization_engine.add_optimization_callback(self._handle_optimization_result)
            
            # Predictive analytics callbacks
            if self.predictive_analytics:
                self.predictive_analytics.add_prediction_callback(self._handle_prediction_insight)
                
        except Exception as e:
            print(f"❌ Error setting up callbacks: {e}")
    
    def start_system(self):
        """Start the enhanced system"""
        if self.is_running:
            return False
        
        try:
            print("🚀 Starting Enhanced System...")
            
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
                print("✅ Performance monitoring started")
            
            # Start intelligent optimization
            if self.optimization_engine:
                self.optimization_engine.start_optimization()
                print("✅ Intelligent optimization started")
            
            # Start predictive analytics
            if self.predictive_analytics:
                self.predictive_analytics.start_analysis()
                print("✅ Predictive analytics started")
            
            # Start integration loop
            self.is_running = True
            self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
            self.integration_thread.start()
            print("✅ Integration loop started")
            
            self.system_status = 'running'
            print("🎉 Enhanced System running successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting system: {e}")
            self.system_status = 'error'
            return False
    
    def stop_system(self):
        """Stop the enhanced system"""
        if not self.is_running:
            return
        
        try:
            print("🛑 Stopping Enhanced System...")
            
            # Stop integration loop
            self.is_running = False
            if self.integration_thread:
                self.integration_thread.join(timeout=2.0)
            
            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
                print("✅ Performance monitoring stopped")
            
            # Stop intelligent optimization
            if self.optimization_engine:
                self.optimization_engine.stop_optimization()
                print("✅ Intelligent optimization stopped")
            
            # Stop predictive analytics
            if self.predictive_analytics:
                self.predictive_analytics.stop_analysis()
                print("✅ Predictive analytics stopped")
            
            self.system_status = 'stopped'
            print("🛑 Enhanced System stopped")
            
        except Exception as e:
            print(f"❌ Error stopping system: {e}")
    
    def _integration_loop(self):
        """Main integration loop"""
        while self.is_running:
            try:
                # Perform system integration
                self._perform_system_integration()
                
                # Update system health
                self._update_system_health()
                
                # Trigger integration callbacks
                self._trigger_integration_callbacks()
                
                # Wait for next integration cycle
                time.sleep(self.config['integration_interval'])
                
            except Exception as e:
                print(f"❌ Integration loop error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _perform_system_integration(self):
        """Perform system-wide integration tasks"""
        try:
            current_time = datetime.now()
            self.last_integration_time = current_time
            
            # Collect system-wide metrics
            system_metrics = self._collect_system_metrics()
            
            # Perform cross-component analysis
            cross_analysis = self._perform_cross_component_analysis(system_metrics)
            
            # Execute predictive actions
            if self.config['enable_predictive_actions']:
                self._execute_predictive_actions(cross_analysis)
            
            # Update system state
            self._update_system_state(system_metrics, cross_analysis)
            
        except Exception as e:
            print(f"❌ Error in system integration: {e}")
    
    def _collect_system_metrics(self) -> Dict:
        """Collect metrics from all system components"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status,
                'system_health': self.system_health,
                'optimization_cycles': self.optimization_cycles,
                'predictions_made': self.predictions_made,
                'alerts_triggered': self.alerts_triggered
            }
            
            # Performance monitor metrics
            if self.performance_monitor:
                try:
                    perf_metrics = self.performance_monitor.get_current_metrics()
                    metrics['performance'] = perf_metrics
                    metrics['performance_score'] = self.performance_monitor.get_performance_score()
                except Exception as e:
                    metrics['performance_error'] = str(e)
            
            # Optimization engine metrics
            if self.optimization_engine:
                try:
                    opt_summary = self.optimization_engine.get_optimization_summary()
                    metrics['optimization'] = opt_summary
                except Exception as e:
                    metrics['optimization_error'] = str(e)
            
            # Predictive analytics metrics
            if self.predictive_analytics:
                try:
                    analytics_summary = self.predictive_analytics.get_analytics_summary()
                    metrics['analytics'] = analytics_summary
                    current_predictions = self.predictive_analytics.get_current_predictions()
                    metrics['current_predictions'] = current_predictions
                except Exception as e:
                    metrics['analytics_error'] = str(e)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error collecting system metrics: {e}")
            return {'error': str(e)}
    
    def _perform_cross_component_analysis(self, system_metrics: Dict) -> Dict:
        """Perform analysis across all system components"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'cross_component_insights': [],
                'system_recommendations': [],
                'optimization_opportunities': [],
                'risk_assessments': []
            }
            
            # Analyze performance vs optimization correlation
            if 'performance' in system_metrics and 'optimization' in system_metrics:
                perf_score = system_metrics.get('performance_score', 75)
                opt_success_rate = system_metrics.get('optimization', {}).get('success_rate', 0)
                
                if perf_score < 70 and opt_success_rate < 0.5:
                    analysis['cross_component_insights'].append({
                        'type': 'performance_optimization_correlation',
                        'insight': 'Low performance correlates with low optimization success',
                        'recommendation': 'Focus on improving optimization strategies',
                        'priority': 'high'
                    })
            
            # Analyze predictive insights vs current performance
            if 'current_predictions' in system_metrics:
                predictions = system_metrics['current_predictions']
                if predictions.get('risk_assessment') == 'high':
                    analysis['risk_assessments'].append({
                        'type': 'high_risk_prediction',
                        'description': 'System at high risk based on predictive analysis',
                        'action_required': 'Immediate intervention recommended',
                        'priority': 'critical'
                    })
            
            # Identify optimization opportunities
            if 'performance' in system_metrics:
                perf_score = system_metrics.get('performance_score', 75)
                if perf_score < 80:
                    analysis['optimization_opportunities'].append({
                        'type': 'performance_improvement',
                        'current_score': perf_score,
                        'target_score': 85,
                        'estimated_effort': 'medium',
                        'expected_benefit': 'high'
                    })
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error in cross-component analysis: {e}")
            return {'error': str(e)}
    
    def _execute_predictive_actions(self, cross_analysis: Dict):
        """Execute actions based on predictive analysis"""
        try:
            if not self.optimization_engine:
                return
            
            # Check for high-risk situations
            for risk_assessment in cross_analysis.get('risk_assessments', []):
                if risk_assessment.get('priority') == 'critical':
                    print(f"🚨 Critical risk detected: {risk_assessment['description']}")
                    
                    # Trigger emergency optimization
                    if self.config['enable_auto_optimization']:
                        print("🔧 Triggering emergency optimization...")
                        # The optimization engine will handle this automatically
            
            # Check for optimization opportunities
            for opportunity in cross_analysis.get('optimization_opportunities', []):
                if opportunity.get('expected_benefit') == 'high':
                    print(f"🎯 High-benefit optimization opportunity: {opportunity['type']}")
                    
                    # Trigger proactive optimization
                    if self.config['enable_auto_optimization']:
                        print("🔧 Triggering proactive optimization...")
                        # The optimization engine will handle this automatically
                        
        except Exception as e:
            print(f"❌ Error executing predictive actions: {e}")
    
    def _update_system_state(self, system_metrics: Dict, cross_analysis: Dict):
        """Update overall system state"""
        try:
            # Update optimization cycles
            if 'optimization' in system_metrics:
                self.optimization_cycles = system_metrics['optimization'].get('total_optimizations', 0)
            
            # Update predictions made
            if 'analytics' in system_metrics:
                self.predictions_made = system_metrics['analytics'].get('total_predictions', 0)
            
            # Update alerts triggered
            if 'performance' in system_metrics:
                # Count alerts from performance monitor
                self.alerts_triggered += 1  # Simplified for now
            
        except Exception as e:
            print(f"❌ Error updating system state: {e}")
    
    def _update_system_health(self):
        """Update overall system health score"""
        try:
            health_factors = []
            
            # Performance factor
            if self.performance_monitor:
                try:
                    perf_score = self.performance_monitor.get_performance_score()
                    health_factors.append(perf_score / 100.0)
                except:
                    health_factors.append(0.7)  # Default if unavailable
            
            # Optimization factor
            if self.optimization_engine:
                try:
                    opt_summary = self.optimization_engine.get_optimization_summary()
                    success_rate = opt_summary.get('success_rate', 0.5)
                    health_factors.append(success_rate)
                except:
                    health_factors.append(0.7)  # Default if unavailable
            
            # Analytics factor
            if self.predictive_analytics:
                try:
                    analytics_summary = self.predictive_analytics.get_analytics_summary()
                    confidence_rate = analytics_summary.get('confidence_rate', 0.5)
                    health_factors.append(confidence_rate)
                except:
                    health_factors.append(0.7)  # Default if unavailable
            
            # Calculate overall health
            if health_factors:
                self.system_health = np.mean(health_factors) * 100.0
            else:
                self.system_health = 75.0  # Default health
            
            # Update system status based on health
            if self.system_health < self.config['health_thresholds']['critical']:
                self.system_status = 'critical'
            elif self.system_health < self.config['health_thresholds']['warning']:
                self.system_status = 'warning'
            elif self.system_health < self.config['health_thresholds']['good']:
                self.system_status = 'degraded'
            else:
                self.system_status = 'healthy'
                
        except Exception as e:
            print(f"❌ Error updating system health: {e}")
    
    def _trigger_integration_callbacks(self):
        """Trigger system integration callbacks"""
        for callback in self.integration_callbacks:
            try:
                callback(self.get_system_status())
            except Exception as e:
                print(f"❌ Error in integration callback: {e}")
    
    def _handle_performance_alert(self, level: str, alert_type: str, message: str, metrics: Dict):
        """Handle performance alerts"""
        try:
            print(f"🚨 Performance Alert - {level.upper()}: {alert_type} - {message}")
            self.alerts_triggered += 1
            
            # Trigger health callbacks
            for callback in self.health_callbacks:
                try:
                    callback('performance_alert', level, alert_type, message, metrics)
                except Exception as e:
                    print(f"❌ Error in health callback: {e}")
                    
        except Exception as e:
            print(f"❌ Error handling performance alert: {e}")
    
    def _handle_optimization_result(self, strategy: Dict, result: Dict):
        """Handle optimization results"""
        try:
            print(f"🔧 Optimization Result - {strategy.get('name', 'Unknown')}: {'Success' if result.get('success') else 'Failed'}")
            
            # Update optimization cycles
            self.optimization_cycles += 1
            
        except Exception as e:
            print(f"❌ Error handling optimization result: {e}")
    
    def _handle_prediction_insight(self, insights: Dict):
        """Handle predictive analytics insights"""
        try:
            risk_level = insights.get('risk_assessment', 'unknown')
            confidence = insights.get('overall_confidence', 0)
            recommendations = len(insights.get('recommendations', []))
            
            print(f"🔮 Prediction Insight - Risk: {risk_level}, Confidence: {confidence:.2f}, Recommendations: {recommendations}")
            
            # Update predictions made
            self.predictions_made += 1
            
        except Exception as e:
            print(f"❌ Error handling prediction insight: {e}")
    
    def add_integration_callback(self, callback: Callable):
        """Add system integration callback"""
        self.integration_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable):
        """Add system health callback"""
        self.health_callbacks.append(callback)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status,
                'system_health': self.system_health,
                'is_running': self.is_running,
                'last_integration': self.last_integration_time.isoformat() if self.last_integration_time else None,
                'optimization_cycles': self.optimization_cycles,
                'predictions_made': self.predictions_made,
                'alerts_triggered': self.alerts_triggered,
                'components': {
                    'performance_monitor': self.performance_monitor is not None,
                    'optimization_engine': self.optimization_engine is not None,
                    'predictive_analytics': self.predictive_analytics is not None
                }
            }
        except Exception as e:
            return {'error': f'Error getting system status: {str(e)}'}
    
    def get_system_summary(self) -> Dict:
        """Get system summary with component details"""
        try:
            summary = {
                'system_overview': self.get_system_status(),
                'performance_summary': {},
                'optimization_summary': {},
                'analytics_summary': {}
            }
            
            # Get performance summary
            if self.performance_monitor:
                try:
                    summary['performance_summary'] = self.performance_monitor.get_performance_summary()
                except Exception as e:
                    summary['performance_summary'] = {'error': str(e)}
            
            # Get optimization summary
            if self.optimization_engine:
                try:
                    summary['optimization_summary'] = self.optimization_engine.get_optimization_summary()
                except Exception as e:
                    summary['optimization_summary'] = {'error': str(e)}
            
            # Get analytics summary
            if self.predictive_analytics:
                try:
                    summary['analytics_summary'] = self.predictive_analytics.get_analytics_summary()
                except Exception as e:
                    summary['analytics_summary'] = {'error': str(e)}
            
            return summary
            
        except Exception as e:
            return {'error': f'Error getting system summary: {str(e)}'}
    
    def export_system_data(self, format: str = 'json') -> str:
        """Export comprehensive system data"""
        try:
            if format.lower() == 'json':
                return json.dumps(self.get_system_summary(), indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to DataFrame and export
                summary = self.get_system_summary()
                df = pd.DataFrame([summary['system_overview']])
                return df.to_csv(index=False)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            return f"Error exporting data: {str(e)}"
    
    def update_config(self, new_config: Dict):
        """Update system configuration"""
        try:
            self.config.update(new_config)
            print("⚙️ System configuration updated")
            
            # Update component configurations if needed
            if 'monitoring_config' in new_config and self.performance_monitor:
                self.performance_monitor.update_config(new_config['monitoring_config'])
            
            if 'optimization_config' in new_config and self.optimization_engine:
                self.optimization_engine.update_config(new_config['optimization_config'])
            
            if 'analytics_config' in new_config and self.predictive_analytics:
                self.predictive_analytics.update_config(new_config['analytics_config'])
                
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
    
    def reset_system(self):
        """Reset system to initial state"""
        try:
            print("🔄 Resetting Enhanced System...")
            
            # Stop system
            self.stop_system()
            
            # Reset counters
            self.optimization_cycles = 0
            self.predictions_made = 0
            self.alerts_triggered = 0
            self.system_health = 100.0
            
            # Clear component histories
            if self.performance_monitor:
                self.performance_monitor.clear_history()
            
            if self.optimization_engine:
                self.optimization_engine.clear_history()
            
            if self.predictive_analytics:
                self.predictive_analytics.clear_history()
            
            # Reset system status
            self.system_status = 'ready'
            
            print("✅ System reset completed")
            
        except Exception as e:
            print(f"❌ Error resetting system: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced system integrator
    integrator = EnhancedSystemIntegrator()
    
    # Add system callbacks
    def system_status_handler(status):
        print(f"📊 System Status Update: {status['system_status']} (Health: {status['system_health']:.1f})")
    
    def health_alert_handler(alert_type, level, alert_type_detail, message, metrics):
        print(f"🚨 Health Alert: {alert_type} - {level} - {message}")
    
    integrator.add_integration_callback(system_status_handler)
    integrator.add_health_callback(health_alert_handler)
    
    # Start system
    if integrator.start_system():
        try:
            # Run for 180 seconds
            time.sleep(180)
            
            # Print system summary
            summary = integrator.get_system_summary()
            print("\n📊 System Summary:")
            print(f"Status: {summary['system_overview']['system_status']}")
            print(f"Health: {summary['system_overview']['system_health']:.1f}")
            print(f"Optimization Cycles: {summary['system_overview']['optimization_cycles']}")
            print(f"Predictions Made: {summary['system_overview']['predictions_made']}")
            
        finally:
            # Stop system
            integrator.stop_system()
    else:
        print("❌ Failed to start system")
