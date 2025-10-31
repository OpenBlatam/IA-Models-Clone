#!/usr/bin/env python3
"""
HeyGen AI - Smart Manager

This module provides a unified management interface for all HeyGen AI systems,
including performance monitoring, auto-optimization, and intelligent analysis.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """System status information."""
    timestamp: float = field(default_factory=time.time)
    overall_health: str = "unknown"  # excellent, good, fair, poor, critical
    performance_score: float = 0.0
    active_systems: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class SmartManager:
    """Unified smart management system for HeyGen AI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.management_active = False
        self.manager_thread: Optional[threading.Thread] = None
        self.management_interval = self.config.get('management_interval', 30.0)
        
        # System components
        self.performance_monitor = None
        self.auto_optimizer = None
        self.intelligent_analyzer = None
        
        # Status tracking
        self.system_status_history: List[SystemStatus] = []
        self.management_stats = {
            'total_optimizations': 0,
            'total_alerts': 0,
            'performance_improvements': 0.0,
            'last_optimization': None
        }
        
        # Initialize manager
        self._setup_manager()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default management configuration."""
        return {
            'management_interval': 30.0,  # seconds
            'auto_optimization': True,
            'performance_threshold': 0.7,  # Minimum performance score
            'alert_threshold': 0.5,  # Performance score threshold for alerts
            'optimization_cooldown': 300.0,  # seconds between optimizations
            'systems': {
                'performance_monitoring': True,
                'auto_optimization': True,
                'intelligent_analysis': True
            },
            'notifications': {
                'email': False,
                'slack': False,
                'webhook': False
            }
        }
    
    def _setup_manager(self):
        """Setup management infrastructure."""
        try:
            # Initialize system components
            self._initialize_system_components()
            
            logger.info("Smart manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup manager: {e}")
    
    def _initialize_system_components(self):
        """Initialize all system components."""
        try:
            # Performance Monitor
            if self.config['systems']['performance_monitoring']:
                from performance_monitor import PerformanceMonitor
                self.performance_monitor = PerformanceMonitor()
                logger.info("Performance monitor initialized")
            
            # Auto-Optimizer
            if self.config['systems']['auto_optimization']:
                from auto_optimizer import AutoOptimizer
                self.auto_optimizer = AutoOptimizer()
                logger.info("Auto-optimizer initialized")
            
            # Intelligent Analyzer
            if self.config['systems']['intelligent_analysis']:
                from intelligent_analyzer import IntelligentAnalyzer
                self.intelligent_analyzer = IntelligentAnalyzer()
                logger.info("Intelligent analyzer initialized")
                
        except ImportError as e:
            logger.warning(f"Some system components could not be initialized: {e}")
        except Exception as e:
            logger.error(f"Error initializing system components: {e}")
    
    def start_management(self):
        """Start unified system management."""
        if self.management_active:
            logger.warning("Management is already active")
            return
        
        self.management_active = True
        
        # Start all system components
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
        
        if self.auto_optimizer:
            self.auto_optimizer.start_optimization()
        
        if self.intelligent_analyzer:
            self.intelligent_analyzer.start_analysis()
        
        # Start management thread
        self.manager_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.manager_thread.start()
        
        logger.info("Unified smart management started")
    
    def stop_management(self):
        """Stop unified system management."""
        self.management_active = False
        
        # Stop all system components
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        if self.auto_optimizer:
            self.auto_optimizer.stop_optimization()
        
        if self.intelligent_analyzer:
            self.intelligent_analyzer.stop_analysis()
        
        # Stop management thread
        if self.manager_thread:
            self.manager_thread.join(timeout=5.0)
        
        logger.info("Unified smart management stopped")
    
    def _management_loop(self):
        """Main management loop."""
        while self.management_active:
            try:
                # Assess overall system health
                system_status = self._assess_system_health()
                self.system_status_history.append(system_status)
                
                # Take action based on status
                self._take_management_action(system_status)
                
                # Maintain history size
                self._maintain_history_size()
                
                # Wait for next management cycle
                time.sleep(self.management_interval)
                
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                time.sleep(self.management_interval)
    
    def _assess_system_health(self) -> SystemStatus:
        """Assess overall system health."""
        status = SystemStatus()
        
        try:
            # Collect performance metrics
            performance_metrics = {}
            if self.performance_monitor:
                try:
                    current_metrics = self.performance_monitor.get_current_metrics()
                    performance_metrics = current_metrics.__dict__
                except:
                    pass
            
            # Collect optimization status
            optimization_status = {}
            if self.auto_optimizer:
                try:
                    optimization_status = self.auto_optimizer.get_optimization_status()
                except:
                    pass
            
            # Collect analysis results
            analysis_summary = {}
            if self.intelligent_analyzer:
                try:
                    analysis_summary = self.intelligent_analyzer.get_analysis_summary()
                except:
                    pass
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                performance_metrics, optimization_status, analysis_summary
            )
            status.performance_score = performance_score
            
            # Determine overall health
            status.overall_health = self._determine_health_level(performance_score)
            
            # Track active systems
            status.active_systems = []
            if self.performance_monitor:
                status.active_systems.append("Performance Monitor")
            if self.auto_optimizer:
                status.active_systems.append("Auto-Optimizer")
            if self.intelligent_analyzer:
                status.active_systems.append("Intelligent Analyzer")
            
            # Generate alerts and recommendations
            status.alerts = self._generate_alerts(performance_metrics, optimization_status, analysis_summary)
            status.recommendations = self._generate_recommendations(performance_score, status.alerts)
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            status.overall_health = "critical"
            status.alerts.append(f"System health assessment failed: {e}")
        
        return status
    
    def _calculate_performance_score(self, performance_metrics: Dict[str, Any], 
                                   optimization_status: Dict[str, Any], 
                                   analysis_summary: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        try:
            score = 0.0
            factors = 0
            
            # Performance metrics factor (40% weight)
            if performance_metrics:
                cpu_score = 1.0 - (performance_metrics.get('cpu_percent', 0) / 100.0)
                memory_score = 1.0 - (performance_metrics.get('memory_percent', 0) / 100.0)
                disk_score = 1.0 - (performance_metrics.get('disk_usage_percent', 0) / 100.0)
                
                performance_factor = (cpu_score + memory_score + disk_score) / 3.0
                score += performance_factor * 0.4
                factors += 1
            
            # Optimization status factor (30% weight)
            if optimization_status:
                if optimization_status.get('active', False):
                    optimization_factor = 0.8  # Active optimization is good
                else:
                    optimization_factor = 0.5  # No optimization might be needed
                
                score += optimization_factor * 0.3
                factors += 1
            
            # Analysis confidence factor (30% weight)
            if analysis_summary:
                confidence = analysis_summary.get('last_analysis_result', {}).get('confidence_score', 0.5)
                score += confidence * 0.3
                factors += 1
            
            # Normalize score
            if factors > 0:
                score = score / factors
            else:
                score = 0.5  # Default score
            
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def _determine_health_level(self, performance_score: float) -> str:
        """Determine system health level based on performance score."""
        if performance_score >= 0.9:
            return "excellent"
        elif performance_score >= 0.8:
            return "good"
        elif performance_score >= 0.7:
            return "fair"
        elif performance_score >= 0.6:
            return "poor"
        else:
            return "critical"
    
    def _generate_alerts(self, performance_metrics: Dict[str, Any], 
                        optimization_status: Dict[str, Any], 
                        analysis_summary: Dict[str, Any]) -> List[str]:
        """Generate system alerts."""
        alerts = []
        
        # Performance alerts
        if performance_metrics:
            cpu_percent = performance_metrics.get('cpu_percent', 0)
            memory_percent = performance_metrics.get('memory_percent', 0)
            disk_percent = performance_metrics.get('disk_usage_percent', 0)
            
            if cpu_percent > 90:
                alerts.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 90:
                alerts.append(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                alerts.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                alerts.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                alerts.append(f"High disk usage: {disk_percent:.1f}%")
        
        # Optimization alerts
        if optimization_status:
            if not optimization_status.get('active', False):
                alerts.append("Auto-optimization is not active")
            
            total_optimizations = optimization_status.get('stats', {}).get('total_optimizations', 0)
            if total_optimizations == 0:
                alerts.append("No optimizations have been performed")
        
        # Analysis alerts
        if analysis_summary:
            if not analysis_summary.get('analysis_status', {}).get('active', False):
                alerts.append("Intelligent analysis is not active")
        
        return alerts
    
    def _generate_recommendations(self, performance_score: float, alerts: List[str]) -> List[str]:
        """Generate system recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if performance_score < 0.7:
            recommendations.append("System performance is below optimal levels. Consider optimization.")
        
        if performance_score < 0.6:
            recommendations.append("Critical performance issues detected. Immediate action required.")
        
        # Alert-based recommendations
        for alert in alerts:
            if "CPU usage" in alert:
                recommendations.append("Consider CPU optimization or process management")
            elif "memory usage" in alert:
                recommendations.append("Consider memory cleanup or optimization")
            elif "disk usage" in alert:
                recommendations.append("Consider disk cleanup or storage expansion")
            elif "optimization" in alert:
                recommendations.append("Review and enable auto-optimization settings")
            elif "analysis" in alert:
                recommendations.append("Enable intelligent analysis for better insights")
        
        # General recommendations
        if not recommendations:
            if performance_score >= 0.9:
                recommendations.append("System is performing excellently. Maintain current configuration.")
            else:
                recommendations.append("Monitor system performance and apply optimizations as needed.")
        
        return recommendations
    
    def _take_management_action(self, system_status: SystemStatus):
        """Take action based on system status."""
        try:
            # Check if optimization is needed
            if (system_status.overall_health in ['poor', 'critical'] and 
                self.config['auto_optimization'] and 
                self.auto_optimizer):
                
                # Check cooldown
                last_opt = self.management_stats['last_optimization']
                cooldown = self.config['optimization_cooldown']
                
                if not last_opt or (time.time() - last_opt) > cooldown:
                    logger.info("Triggering auto-optimization due to poor system health")
                    
                    # This would trigger optimization in a real system
                    # For now, we just log the action
                    self.management_stats['last_optimization'] = time.time()
                    self.management_stats['total_optimizations'] += 1
            
            # Update management statistics
            if system_status.alerts:
                self.management_stats['total_alerts'] += len(system_status.alerts)
            
            # Log status
            logger.info(f"System health: {system_status.overall_health} (score: {system_status.performance_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error taking management action: {e}")
    
    def _maintain_history_size(self):
        """Maintain system status history size."""
        max_history = 1000  # Keep last 1000 status entries
        
        if len(self.system_status_history) > max_history:
            self.system_status_history = self.system_status_history[-max_history:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.system_status_history:
            return {"error": "No system status available"}
        
        current_status = self.system_status_history[-1]
        
        return {
            "current_status": {
                "overall_health": current_status.overall_health,
                "performance_score": current_status.performance_score,
                "active_systems": current_status.active_systems,
                "alerts": current_status.alerts,
                "recommendations": current_status.recommendations,
                "timestamp": datetime.fromtimestamp(current_status.timestamp).isoformat()
            },
            "management_stats": self.management_stats.copy(),
            "system_components": {
                "performance_monitor": self.performance_monitor is not None,
                "auto_optimizer": self.auto_optimizer is not None,
                "intelligent_analyzer": self.intelligent_analyzer is not None
            },
            "configuration": {
                "auto_optimization": self.config['auto_optimization'],
                "management_interval": self.config['management_interval'],
                "performance_threshold": self.config['performance_threshold']
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        # Performance monitor summary
        if self.performance_monitor:
            try:
                summary['performance_monitor'] = self.performance_monitor.get_performance_summary()
            except Exception as e:
                summary['performance_monitor'] = {"error": str(e)}
        
        # Auto-optimizer summary
        if self.auto_optimizer:
            try:
                summary['auto_optimizer'] = self.auto_optimizer.get_optimization_status()
            except Exception as e:
                summary['auto_optimizer'] = {"error": str(e)}
        
        # Intelligent analyzer summary
        if self.intelligent_analyzer:
            try:
                summary['intelligent_analyzer'] = self.intelligent_analyzer.get_analysis_summary()
            except Exception as e:
                summary['intelligent_analyzer'] = {"error": str(e)}
        
        # Overall system status
        summary['system_status'] = self.get_system_status()
        
        return summary
    
    def export_system_report(self, format: str = "json", filepath: Optional[str] = None) -> str:
        """Export comprehensive system report."""
        if format not in ['json', 'html']:
            raise ValueError(f"Unsupported format: {format}")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"heygen_ai_system_report_{timestamp}.{format}"
        
        if format == "json":
            return self._export_json_report(filepath)
        elif format == "html":
            return self._export_html_report(filepath)
        
        raise ValueError(f"Export format {format} not implemented")
    
    def _export_json_report(self, filepath: str) -> str:
        """Export system report to JSON format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "performance_summary": self.get_performance_summary(),
            "status_history": [
                {
                    "timestamp": datetime.fromtimestamp(s.timestamp).isoformat(),
                    "overall_health": s.overall_health,
                    "performance_score": s.performance_score,
                    "alerts": s.alerts,
                    "recommendations": s.recommendations
                }
                for s in self.system_status_history[-100:]  # Last 100 entries
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def _export_html_report(self, filepath: str) -> str:
        """Export system report to HTML format."""
        system_status = self.get_system_status()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HeyGen AI System Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .status {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .excellent {{ background: #d4edda; border-left: 4px solid #28a745; }}
                .good {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
                .fair {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .poor {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
                .critical {{ background: #f5c6cb; border-left: 4px solid #721c24; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 HeyGen AI System Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>📊 System Status</h2>
                <div class="status {system_status.get('current_status', {}).get('overall_health', 'unknown')}">
                    <h3>Overall Health: {system_status.get('current_status', {}).get('overall_health', 'unknown').upper()}</h3>
                    <p><strong>Performance Score:</strong> {system_status.get('current_status', {}).get('performance_score', 0):.2f}</p>
                    <p><strong>Active Systems:</strong> {', '.join(system_status.get('current_status', {}).get('active_systems', []))}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>⚠️ Current Alerts</h2>
                <ul>
        """
        
        alerts = system_status.get('current_status', {}).get('alerts', [])
        for alert in alerts:
            html_content += f"<li>{alert}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <ul>
        """
        
        recommendations = system_status.get('current_status', {}).get('recommendations', [])
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>📈 Management Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        stats = system_status.get('management_stats', {})
        for key, value in stats.items():
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            html_content += f"<tr><td>{key}</td><td>{display_value}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath

def main():
    """Main function for the smart manager."""
    parser = argparse.ArgumentParser(description="HeyGen AI Smart Manager")
    parser.add_argument("--start", action="store_true", help="Start smart management")
    parser.add_argument("--stop", action="store_true", help="Stop smart management")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--summary", action="store_true", help="Show performance summary")
    parser.add_argument("--export", choices=["json", "html"], help="Export system report")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    
    args = parser.parse_args()
    
    # Create manager instance
    manager = SmartManager()
    
    try:
        if args.start:
            print("🚀 Starting HeyGen AI Smart Manager...")
            manager.start_management()
            print("✅ Smart management started")
            
        elif args.stop:
            print("⏹️ Stopping HeyGen AI Smart Manager...")
            manager.stop_management()
            print("✅ Smart management stopped")
            
        elif args.status:
            print("📊 System Status:")
            status = manager.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.summary:
            print("📈 Performance Summary:")
            summary = manager.get_performance_summary()
            print(json.dumps(summary, indent=2, default=str))
            
        elif args.export:
            print(f"📁 Exporting system report to {args.export} format...")
            filepath = manager.export_system_report(args.export)
            print(f"✅ Report exported to: {filepath}")
            
        elif args.demo:
            print("🎯 HeyGen AI Smart Manager Demo")
            print("=" * 50)
            
            # Start management
            print("Starting smart management...")
            manager.start_management()
            
            # Let it run for a bit
            print("Running for 60 seconds...")
            time.sleep(60)
            
            # Show status
            print("\n📊 Final System Status:")
            status = manager.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            
            # Export report
            print("\n📁 Exporting system report...")
            json_file = manager.export_system_report("json")
            print(f"JSON report: {json_file}")
            
            html_file = manager.export_system_report("html")
            print(f"HTML report: {html_file}")
            
            # Stop management
            manager.stop_management()
            print("\n✅ Smart manager demo completed!")
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation interrupted by user")
        manager.stop_management()
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        manager.stop_management()

if __name__ == "__main__":
    main()
