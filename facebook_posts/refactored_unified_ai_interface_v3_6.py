#!/usr/bin/env python3
"""
Refactored Unified AI Interface v3.6
Enhanced, optimized, and refactored version with improved architecture
"""
import time
import threading
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Enhanced system components
try:
    from enhanced_system_integrator import EnhancedSystemIntegrator
    from advanced_performance_monitor import AdvancedPerformanceMonitor
    from intelligent_optimization_engine import IntelligentOptimizationEngine
    from predictive_analytics_system import PredictiveAnalyticsSystem
except ImportError as e:
    print(f"⚠️ Enhanced components not available: {e}")
    EnhancedSystemIntegrator = None
    AdvancedPerformanceMonitor = None
    IntelligentOptimizationEngine = None
    PredictiveAnalyticsSystem = None

class RefactoredUnifiedAIInterface:
    """
    Refactored and optimized unified AI interface with enhanced architecture
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the refactored AI interface"""
        self.config = config or self._get_default_config()
        self.system_integrator = None
        self.is_initialized = False
        self.interface_state = 'initializing'
        
        # Performance tracking
        self.startup_time = None
        self.operation_count = 0
        self.error_count = 0
        
        # Initialize system
        self._initialize_system()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration with improved structure"""
        return {
            'system': {
                'auto_start': True,
                'health_check_interval': 30.0,
                'max_retries': 3,
                'timeout': 60.0
            },
            'monitoring': {
                'enabled': True,
                'interval': 2.0,
                'history_size': 1000,
                'alert_thresholds': {
                    'cpu': 80.0,
                    'memory': 85.0,
                    'disk': 90.0,
                    'gpu': 95.0
                }
            },
            'optimization': {
                'enabled': True,
                'auto_optimize': True,
                'interval': 30.0,
                'threshold': 0.7
            },
            'analytics': {
                'enabled': True,
                'prediction_horizon': 300,
                'confidence_threshold': 0.75
            },
            'ui': {
                'theme': 'dark',
                'auto_refresh': True,
                'refresh_interval': 5.0
            }
        }
    
    def _initialize_system(self):
        """Initialize the enhanced system integrator"""
        try:
            print("🚀 Initializing Refactored Unified AI Interface...")
            
            if EnhancedSystemIntegrator:
                # Create system integrator with refactored config
                integrator_config = {
                    'enable_performance_monitoring': self.config['monitoring']['enabled'],
                    'enable_intelligent_optimization': self.config['optimization']['enabled'],
                    'enable_predictive_analytics': self.config['analytics']['enabled'],
                    'enable_auto_optimization': self.config['optimization']['auto_optimize'],
                    'monitoring_config': self.config['monitoring'],
                    'optimization_config': self.config['optimization'],
                    'analytics_config': self.config['analytics']
                }
                
                self.system_integrator = EnhancedSystemIntegrator(integrator_config)
                print("✅ Enhanced system integrator initialized")
                
                # Set up system callbacks
                self._setup_system_callbacks()
                
                # Auto-start if configured
                if self.config['system']['auto_start']:
                    self.start_system()
                
                self.is_initialized = True
                self.interface_state = 'ready'
                print("🎉 Refactored AI Interface ready!")
                
            else:
                print("⚠️ Enhanced system integrator not available")
                self.interface_state = 'limited'
                
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.interface_state = 'error'
            self.error_count += 1
            raise
    
    def _setup_system_callbacks(self):
        """Set up system-wide callbacks for monitoring and alerts"""
        try:
            if not self.system_integrator:
                return
            
            # System health callback
            self.system_integrator.add_health_callback(self._handle_system_health)
            
            # Integration callback
            self.system_integrator.add_integration_callback(self._handle_system_integration)
            
            print("✅ System callbacks configured")
            
        except Exception as e:
            print(f"❌ Error setting up callbacks: {e}")
    
    def start_system(self) -> bool:
        """Start the enhanced system"""
        try:
            if not self.system_integrator:
                print("❌ System integrator not available")
                return False
            
            if self.system_integrator.start_system():
                self.interface_state = 'running'
                self.startup_time = datetime.now()
                print("✅ Enhanced system started successfully")
                return True
            else:
                print("❌ Failed to start enhanced system")
                return False
                
        except Exception as e:
            print(f"❌ Error starting system: {e}")
            self.error_count += 1
            return False
    
    def stop_system(self):
        """Stop the enhanced system"""
        try:
            if self.system_integrator:
                self.system_integrator.stop_system()
                self.interface_state = 'stopped'
                print("✅ Enhanced system stopped")
            
        except Exception as e:
            print(f"❌ Error stopping system: {e}")
            self.error_count += 1
    
    def _handle_system_health(self, alert_type: str, level: str, alert_category: str, message: str, metrics: Dict):
        """Handle system health alerts"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"🚨 [{timestamp}] {level.upper()} - {alert_category}: {message}")
            
            # Log critical alerts
            if level in ['critical', 'emergency']:
                self._log_critical_alert(alert_type, level, alert_category, message, metrics)
                
        except Exception as e:
            print(f"❌ Error handling health alert: {e}")
    
    def _handle_system_integration(self, system_status: Dict):
        """Handle system integration updates"""
        try:
            # Update interface state based on system status
            if 'system_status' in system_status:
                self.interface_state = system_status['system_status']
                
        except Exception as e:
            print(f"❌ Error handling system integration: {e}")
    
    def _log_critical_alert(self, alert_type: str, level: str, category: str, message: str, metrics: Dict):
        """Log critical alerts for analysis"""
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'level': level,
                'category': category,
                'message': message,
                'metrics': metrics
            }
            
            # In a production system, this would be logged to a file or database
            print(f"📝 CRITICAL ALERT LOGGED: {json.dumps(alert_data, indent=2)}")
            
        except Exception as e:
            print(f"❌ Error logging critical alert: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                'interface': {
                    'state': self.interface_state,
                    'initialized': self.is_initialized,
                    'startup_time': self.startup_time.isoformat() if self.startup_time else None,
                    'operation_count': self.operation_count,
                    'error_count': self.error_count
                },
                'config': self.config
            }
            
            # Add system integrator status if available
            if self.system_integrator:
                status['system'] = self.system_integrator.get_system_status()
            
            return status
            
        except Exception as e:
            return {'error': f'Error getting status: {str(e)}'}
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            if not self.system_integrator:
                return {'error': 'System integrator not available'}
            
            # Get performance summary from system integrator
            system_summary = self.system_integrator.get_system_summary()
            
            if 'performance_summary' in system_summary:
                return system_summary['performance_summary']
            else:
                return {'error': 'Performance data not available'}
                
        except Exception as e:
            return {'error': f'Error getting performance metrics: {str(e)}'}
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        try:
            if not self.system_integrator:
                return {'error': 'System integrator not available'}
            
            system_summary = self.system_integrator.get_system_summary()
            
            if 'optimization_summary' in system_summary:
                return system_summary['optimization_summary']
            else:
                return {'error': 'Optimization data not available'}
                
        except Exception as e:
            return {'error': f'Error getting optimization status: {str(e)}'}
    
    def get_analytics_insights(self) -> Dict:
        """Get current analytics insights"""
        try:
            if not self.system_integrator:
                return {'error': 'System integrator not available'}
            
            system_summary = self.system_integrator.get_system_summary()
            
            if 'analytics_summary' in system_summary:
                return system_summary['analytics_summary']
            else:
                return {'error': 'Analytics data not available'}
                
        except Exception as e:
            return {'error': f'Error getting analytics insights: {str(e)}'}
    
    def trigger_optimization(self, strategy: Optional[str] = None) -> bool:
        """Trigger manual optimization"""
        try:
            if not self.system_integrator or not self.system_integrator.optimization_engine:
                print("❌ Optimization engine not available")
                return False
            
            if strategy:
                print(f"🔧 Triggering manual optimization: {strategy}")
                return self.system_integrator.optimization_engine.start_optimization(strategy)
            else:
                print("🔧 Triggering automatic optimization")
                return self.system_integrator.optimization_engine.start_optimization()
                
        except Exception as e:
            print(f"❌ Error triggering optimization: {e}")
            self.error_count += 1
            return False
    
    def export_system_data(self, format: str = 'json') -> str:
        """Export comprehensive system data"""
        try:
            if not self.system_integrator:
                return "System integrator not available"
            
            return self.system_integrator.export_system_data(format)
            
        except Exception as e:
            return f"Error exporting data: {str(e)}"
    
    def update_config(self, new_config: Dict):
        """Update system configuration"""
        try:
            # Update local config
            self._deep_update_config(self.config, new_config)
            
            # Update system integrator config if available
            if self.system_integrator:
                self.system_integrator.update_config(new_config)
            
            print("⚙️ Configuration updated successfully")
            
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
            self.error_count += 1
    
    def _deep_update_config(self, base_config: Dict, updates: Dict):
        """Deep update configuration dictionary"""
        for key, value in updates.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._deep_update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        try:
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'interface_health': {
                    'state': self.interface_state,
                    'initialized': self.is_initialized,
                    'errors': self.error_count,
                    'operations': self.operation_count
                },
                'system_health': {},
                'recommendations': []
            }
            
            # Add system health if available
            if self.system_integrator:
                system_status = self.system_integrator.get_system_status()
                health_report['system_health'] = {
                    'status': system_status.get('system_status', 'unknown'),
                    'health_score': system_status.get('system_health', 0),
                    'components': system_status.get('components', {})
                }
                
                # Generate recommendations
                health_report['recommendations'] = self._generate_health_recommendations(system_status)
            
            return health_report
            
        except Exception as e:
            return {'error': f'Error generating health report: {str(e)}'}
    
    def _generate_health_recommendations(self, system_status: Dict) -> List[Dict]:
        """Generate health recommendations based on system status"""
        recommendations = []
        
        try:
            # Check system health
            health_score = system_status.get('system_health', 100)
            if health_score < 50:
                recommendations.append({
                    'priority': 'critical',
                    'action': 'Immediate system intervention required',
                    'reason': f'System health score is critically low: {health_score:.1f}'
                })
            elif health_score < 70:
                recommendations.append({
                    'priority': 'high',
                    'action': 'System optimization recommended',
                    'reason': f'System health score is below optimal: {health_score:.1f}'
                })
            
            # Check component status
            components = system_status.get('components', {})
            if not components.get('performance_monitor', False):
                recommendations.append({
                    'priority': 'medium',
                    'action': 'Enable performance monitoring',
                    'reason': 'Performance monitoring is disabled'
                })
            
            if not components.get('optimization_engine', False):
                recommendations.append({
                    'priority': 'medium',
                    'action': 'Enable optimization engine',
                    'reason': 'Optimization engine is disabled'
                })
            
        except Exception as e:
            recommendations.append({
                'priority': 'low',
                'action': 'Check system configuration',
                'reason': f'Error analyzing system status: {str(e)}'
            })
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            print("🧹 Cleaning up resources...")
            
            # Stop system if running
            if self.interface_state == 'running':
                self.stop_system()
            
            # Clear any cached data
            self.operation_count = 0
            self.error_count = 0
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Example usage and testing
if __name__ == "__main__":
    try:
        # Create refactored interface
        ai_interface = RefactoredUnifiedAIInterface()
        
        # Get initial status
        status = ai_interface.get_system_status()
        print(f"Initial status: {status['interface']['state']}")
        
        # Run for a while to see the system in action
        print("🔄 Running system for 60 seconds...")
        time.sleep(60)
        
        # Get health report
        health_report = ai_interface.get_health_report()
        print(f"Health report: {json.dumps(health_report, indent=2)}")
        
        # Export system data
        export_data = ai_interface.export_system_data('json')
        print(f"System data exported: {len(export_data)} characters")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if 'ai_interface' in locals():
            ai_interface.cleanup()
