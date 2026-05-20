#!/usr/bin/env python3
"""
Advanced Performance Monitor for Enhanced Unified AI Interface v3.5
Real-time performance tracking, predictive analytics, and intelligent optimization
"""
import time
import threading
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class AdvancedPerformanceMonitor:
    """Advanced performance monitoring system with predictive analytics"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.metrics_history = []
        self.alert_callbacks = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.performance_thresholds = self.config.get('performance_thresholds', {})
        self.optimization_rules = self.config.get('optimization_rules', {})
        
        # Initialize performance tracking
        self.current_metrics = {}
        self.performance_score = 100.0
        self.optimization_recommendations = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'monitoring_interval': 2.0,  # seconds
            'history_size': 1000,        # max metrics to store
            'performance_thresholds': {
                'cpu_usage': 80.0,       # % CPU usage threshold
                'memory_usage': 85.0,    # % memory usage threshold
                'disk_usage': 90.0,      # % disk usage threshold
                'response_time': 200.0,  # ms response time threshold
                'gpu_utilization': 95.0  # % GPU utilization threshold
            },
            'optimization_rules': {
                'enable_auto_optimization': True,
                'enable_predictive_scaling': True,
                'enable_resource_balancing': True,
                'enable_performance_tuning': True
            },
            'alert_levels': {
                'warning': 70.0,         # % threshold for warnings
                'critical': 90.0,        # % threshold for critical alerts
                'emergency': 95.0        # % threshold for emergency alerts
            }
        }
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("🚀 Advanced Performance Monitor started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🛑 Advanced Performance Monitor stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                self._collect_metrics()
                
                # Analyze performance
                self._analyze_performance()
                
                # Generate optimization recommendations
                self._generate_recommendations()
                
                # Check for alerts
                self._check_alerts()
                
                # Store metrics in history
                self._store_metrics()
                
                # Wait for next monitoring cycle
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                print(f"❌ Performance monitoring error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _collect_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # System load
            try:
                load_avg = psutil.getloadavg()
            except:
                load_avg = (0.0, 0.0, 0.0)
            
            # Timestamp
            timestamp = datetime.now()
            
            # Store current metrics
            self.current_metrics = {
                'timestamp': timestamp,
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                    'load_avg_1m': load_avg[0],
                    'load_avg_5m': load_avg[1],
                    'load_avg_15m': load_avg[2]
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'usage_percent': memory.percent,
                    'swap_total_gb': swap.total / (1024**3),
                    'swap_used_gb': swap.used / (1024**3)
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'usage_percent': (disk.used / disk.total) * 100,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'gpu': gpu_metrics,
                'system': {
                    'process_count': process_count,
                    'uptime_seconds': time.time() - psutil.boot_time()
                }
            }
            
        except Exception as e:
            print(f"❌ Error collecting metrics: {e}")
    
    def _get_gpu_metrics(self) -> Dict:
        """Get GPU metrics if available"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_metrics = {}
                
                for i in range(gpu_count):
                    try:
                        # Get GPU memory info
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        
                        gpu_metrics[f'gpu_{i}'] = {
                            'name': torch.cuda.get_device_name(i),
                            'memory_allocated_gb': memory_allocated,
                            'memory_reserved_gb': memory_reserved,
                            'memory_total_gb': memory_total,
                            'memory_usage_percent': (memory_allocated / memory_total) * 100
                        }
                    except Exception as e:
                        gpu_metrics[f'gpu_{i}'] = {'error': str(e)}
                
                return gpu_metrics
            else:
                return {'status': 'CUDA not available'}
                
        except ImportError:
            return {'status': 'PyTorch not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_performance(self):
        """Analyze current performance and calculate score"""
        try:
            score = 100.0
            issues = []
            
            # CPU analysis
            cpu_usage = self.current_metrics['cpu']['usage_percent']
            if cpu_usage > self.performance_thresholds['cpu_usage']:
                score -= (cpu_usage - self.performance_thresholds['cpu_usage']) * 0.5
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            # Memory analysis
            memory_usage = self.current_metrics['memory']['usage_percent']
            if memory_usage > self.performance_thresholds['memory_usage']:
                score -= (memory_usage - self.performance_thresholds['memory_usage']) * 0.5
                issues.append(f"High memory usage: {memory_usage:.1f}%")
            
            # Disk analysis
            disk_usage = self.current_metrics['disk']['usage_percent']
            if disk_usage > self.performance_thresholds['disk_usage']:
                score -= (disk_usage - self.performance_thresholds['disk_usage']) * 0.3
                issues.append(f"High disk usage: {disk_usage:.1f}%")
            
            # GPU analysis
            if 'gpu_0' in self.current_metrics['gpu']:
                gpu_memory = self.current_metrics['gpu']['gpu_0']['memory_usage_percent']
                if gpu_memory > self.performance_thresholds['gpu_utilization']:
                    score -= (gpu_memory - self.performance_thresholds['gpu_utilization']) * 0.4
                    issues.append(f"High GPU memory usage: {gpu_memory:.1f}%")
            
            # Load average analysis
            load_avg = self.current_metrics['cpu']['load_avg_1m']
            cpu_count = self.current_metrics['cpu']['count']
            if load_avg > cpu_count * 0.8:
                score -= 10.0
                issues.append(f"High system load: {load_avg:.2f}")
            
            # Ensure score doesn't go below 0
            score = max(0.0, score)
            
            # Update performance score
            self.performance_score = score
            
            # Store analysis results
            self.current_metrics['analysis'] = {
                'performance_score': score,
                'issues': issues,
                'status': self._get_performance_status(score)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing performance: {e}")
    
    def _get_performance_status(self, score: float) -> str:
        """Get performance status based on score"""
        if score >= 90:
            return "🟢 Excellent"
        elif score >= 75:
            return "🟡 Good"
        elif score >= 60:
            return "🟠 Fair"
        elif score >= 40:
            return "🔴 Poor"
        else:
            return "💀 Critical"
    
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # CPU recommendations
            cpu_usage = self.current_metrics['cpu']['usage_percent']
            if cpu_usage > 80:
                recommendations.append({
                    'type': 'cpu',
                    'priority': 'high',
                    'action': 'Consider reducing concurrent processes or optimizing algorithms',
                    'metric': f"CPU usage: {cpu_usage:.1f}%"
                })
            
            # Memory recommendations
            memory_usage = self.current_metrics['memory']['usage_percent']
            if memory_usage > 85:
                recommendations.append({
                    'type': 'memory',
                    'priority': 'high',
                    'action': 'Consider memory cleanup, garbage collection, or reducing batch sizes',
                    'metric': f"Memory usage: {memory_usage:.1f}%"
                })
            
            # Disk recommendations
            disk_usage = self.current_metrics['disk']['usage_percent']
            if disk_usage > 90:
                recommendations.append({
                    'type': 'disk',
                    'priority': 'critical',
                    'action': 'Immediate disk cleanup required. Consider archiving old data.',
                    'metric': f"Disk usage: {disk_usage:.1f}%"
                })
            
            # GPU recommendations
            if 'gpu_0' in self.current_metrics['gpu']:
                gpu_memory = self.current_metrics['gpu']['gpu_0']['memory_usage_percent']
                if gpu_memory > 90:
                    recommendations.append({
                        'type': 'gpu',
                        'priority': 'high',
                        'action': 'Consider reducing model size, batch size, or using gradient checkpointing',
                        'metric': f"GPU memory usage: {gpu_memory:.1f}%"
                    })
            
            # Load balancing recommendations
            load_avg = self.current_metrics['cpu']['load_avg_1m']
            cpu_count = self.current_metrics['cpu']['count']
            if load_avg > cpu_count * 0.8:
                recommendations.append({
                    'type': 'system',
                    'priority': 'medium',
                    'action': 'Consider distributing workload across more cores or reducing concurrent tasks',
                    'metric': f"System load: {load_avg:.2f}"
                })
            
            # Update recommendations
            self.optimization_recommendations = recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
    
    def _check_alerts(self):
        """Check for performance alerts and trigger callbacks"""
        try:
            alerts = []
            
            # Check CPU alerts
            cpu_usage = self.current_metrics['cpu']['usage_percent']
            if cpu_usage > self.config['alert_levels']['emergency']:
                alerts.append(('emergency', 'cpu', f"Critical CPU usage: {cpu_usage:.1f}%"))
            elif cpu_usage > self.config['alert_levels']['critical']:
                alerts.append(('critical', 'cpu', f"High CPU usage: {cpu_usage:.1f}%"))
            elif cpu_usage > self.config['alert_levels']['warning']:
                alerts.append(('warning', 'cpu', f"Elevated CPU usage: {cpu_usage:.1f}%"))
            
            # Check memory alerts
            memory_usage = self.current_metrics['memory']['usage_percent']
            if memory_usage > self.config['alert_levels']['emergency']:
                alerts.append(('emergency', 'memory', f"Critical memory usage: {memory_usage:.1f}%"))
            elif memory_usage > self.config['alert_levels']['critical']:
                alerts.append(('critical', 'memory', f"High memory usage: {memory_usage:.1f}%"))
            elif memory_usage > self.config['alert_levels']['warning']:
                alerts.append(('warning', 'memory', f"Elevated memory usage: {memory_usage:.1f}%"))
            
            # Check disk alerts
            disk_usage = self.current_metrics['disk']['usage_percent']
            if disk_usage > self.config['alert_levels']['emergency']:
                alerts.append(('emergency', 'disk', f"Critical disk usage: {disk_usage:.1f}%"))
            elif disk_usage > self.config['alert_levels']['critical']:
                alerts.append(('critical', 'disk', f"High disk usage: {disk_usage:.1f}%"))
            elif disk_usage > self.config['alert_levels']['warning']:
                alerts.append(('warning', 'disk', f"Elevated disk usage: {disk_usage:.1f}%"))
            
            # Trigger alert callbacks
            for alert_level, alert_type, message in alerts:
                self._trigger_alert(alert_level, alert_type, message)
            
        except Exception as e:
            print(f"❌ Error checking alerts: {e}")
    
    def _trigger_alert(self, level: str, alert_type: str, message: str):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(level, alert_type, message, self.current_metrics)
            except Exception as e:
                print(f"❌ Error in alert callback: {e}")
    
    def _store_metrics(self):
        """Store metrics in history"""
        try:
            # Add timestamp for storage
            metrics_copy = self.current_metrics.copy()
            metrics_copy['timestamp'] = metrics_copy['timestamp'].isoformat()
            
            self.metrics_history.append(metrics_copy)
            
            # Limit history size
            if len(self.metrics_history) > self.config['history_size']:
                self.metrics_history.pop(0)
                
        except Exception as e:
            print(f"❌ Error storing metrics: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.current_metrics.copy()
    
    def get_performance_score(self) -> float:
        """Get current performance score"""
        return self.performance_score
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get current optimization recommendations"""
        return self.optimization_recommendations.copy()
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get metrics history"""
        if limit is None:
            return self.metrics_history.copy()
        else:
            return self.metrics_history[-limit:].copy()
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            if not self.metrics_history:
                return {'error': 'No metrics available'}
            
            # Calculate averages
            cpu_usage_avg = np.mean([m['cpu']['usage_percent'] for m in self.metrics_history])
            memory_usage_avg = np.mean([m['memory']['usage_percent'] for m in self.metrics_history])
            disk_usage_avg = np.mean([m['disk']['usage_percent'] for m in self.metrics_history])
            
            # Calculate trends
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
            if len(recent_metrics) >= 2:
                cpu_trend = recent_metrics[-1]['cpu']['usage_percent'] - recent_metrics[0]['cpu']['usage_percent']
                memory_trend = recent_metrics[-1]['memory']['usage_percent'] - recent_metrics[0]['memory']['usage_percent']
            else:
                cpu_trend = memory_trend = 0.0
            
            return {
                'current_score': self.performance_score,
                'current_status': self.current_metrics.get('analysis', {}).get('status', 'Unknown'),
                'averages': {
                    'cpu_usage': cpu_usage_avg,
                    'memory_usage': memory_usage_avg,
                    'disk_usage': disk_usage_avg
                },
                'trends': {
                    'cpu_trend': cpu_trend,
                    'memory_trend': memory_trend
                },
                'recommendations_count': len(self.optimization_recommendations),
                'metrics_count': len(self.metrics_history),
                'monitoring_duration': len(self.metrics_history) * self.config['monitoring_interval']
            }
            
        except Exception as e:
            return {'error': f'Error generating summary: {str(e)}'}
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        try:
            if format.lower() == 'json':
                import json
                return json.dumps(self.metrics_history, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to DataFrame and export as CSV
                df = pd.DataFrame(self.metrics_history)
                return df.to_csv(index=False)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            return f"Error exporting metrics: {str(e)}"
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        print("🗑️ Metrics history cleared")
    
    def update_config(self, new_config: Dict):
        """Update monitoring configuration"""
        self.config.update(new_config)
        print("⚙️ Configuration updated")
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        try:
            return {
                'platform': {
                    'system': psutil.sys.platform,
                    'release': psutil.sys.platform,
                    'version': psutil.sys.version
                },
                'cpu': {
                    'count': psutil.cpu_count(),
                    'count_logical': psutil.cpu_count(logical=True),
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'total_gb': psutil.virtual_memory().total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3)
                },
                'disk': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'free_gb': psutil.disk_usage('/').free / (1024**3)
                }
            }
        except Exception as e:
            return {'error': f'Error getting system info: {str(e)}'}

# Example usage and testing
if __name__ == "__main__":
    # Create monitor instance
    monitor = AdvancedPerformanceMonitor()
    
    # Add alert callback
    def alert_handler(level, alert_type, message, metrics):
        print(f"🚨 {level.upper()} ALERT - {alert_type}: {message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Run for 30 seconds
        time.sleep(30)
        
        # Print summary
        summary = monitor.get_performance_summary()
        print("\n📊 Performance Summary:")
        print(f"Score: {summary['current_score']:.1f}")
        print(f"Status: {summary['current_status']}")
        print(f"Recommendations: {summary['recommendations_count']}")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
