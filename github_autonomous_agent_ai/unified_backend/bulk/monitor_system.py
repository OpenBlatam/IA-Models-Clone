"""
BUL System Monitor
=================

Real-time monitoring and health checking for the BUL system.
"""

import asyncio
import logging
import sys
import time
import psutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.api_handler import APIHandler
from modules.document_processor import DocumentProcessor
from modules.query_analyzer import QueryAnalyzer
from modules.business_agents import BusinessAgentManager
from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Real-time system monitoring."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.start_time = datetime.now()
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'documents_generated': 0,
            'average_response_time': 0,
            'uptime': 0
        }
        self.health_checks = []
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # CPU and Memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': current_time.isoformat(),
            'uptime_seconds': uptime,
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'cpu_percent': cpu_percent,
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'metrics': self.metrics.copy()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # CPU check
        if cpu_percent > 90:
            health_status['checks'].append({
                'component': 'cpu',
                'status': 'warning',
                'message': f'High CPU usage: {cpu_percent}%'
            })
        else:
            health_status['checks'].append({
                'component': 'cpu',
                'status': 'healthy',
                'message': f'CPU usage: {cpu_percent}%'
            })
        
        # Memory check
        if memory.percent > 90:
            health_status['checks'].append({
                'component': 'memory',
                'status': 'warning',
                'message': f'High memory usage: {memory.percent}%'
            })
        else:
            health_status['checks'].append({
                'component': 'memory',
                'status': 'healthy',
                'message': f'Memory usage: {memory.percent}%'
            })
        
        # Check output directory
        output_dir = Path(self.config.output_directory)
        if output_dir.exists():
            health_status['checks'].append({
                'component': 'output_directory',
                'status': 'healthy',
                'message': f'Output directory accessible: {output_dir}'
            })
        else:
            health_status['checks'].append({
                'component': 'output_directory',
                'status': 'error',
                'message': f'Output directory not accessible: {output_dir}'
            })
        
        # Determine overall status
        error_checks = [c for c in health_status['checks'] if c['status'] == 'error']
        warning_checks = [c for c in health_status['checks'] if c['status'] == 'warning']
        
        if error_checks:
            health_status['status'] = 'unhealthy'
        elif warning_checks:
            health_status['status'] = 'degraded'
        
        return health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'requests_per_minute': self.metrics['requests_total'] / max(1, self.metrics['uptime'] / 60),
            'success_rate': (self.metrics['requests_success'] / max(1, self.metrics['requests_total'])) * 100,
            'average_response_time': self.metrics['average_response_time'],
            'documents_per_hour': self.metrics['documents_generated'] / max(1, self.metrics['uptime'] / 3600),
            'uptime_hours': self.metrics['uptime'] / 3600
        }
    
    def update_metrics(self, request_type: str, success: bool, response_time: float):
        """Update system metrics."""
        self.metrics['requests_total'] += 1
        if success:
            self.metrics['requests_success'] += 1
        else:
            self.metrics['requests_failed'] += 1
        
        if request_type == 'document_generation':
            self.metrics['documents_generated'] += 1
        
        # Update average response time
        total_requests = self.metrics['requests_total']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        
        # Update uptime
        self.metrics['uptime'] = (datetime.now() - self.start_time).total_seconds()
    
    async def check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoint availability."""
        base_url = f"http://{self.config.api_host}:{self.config.api_port}"
        endpoints = [
            '/',
            '/health',
            '/agents',
            '/tasks'
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                results[endpoint] = {
                    'status': 'healthy' if response.status_code == 200 else 'error',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
            except Exception as e:
                results[endpoint] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive system report."""
        metrics = self.get_system_metrics()
        health = self.get_health_status()
        performance = self.get_performance_stats()
        
        report = f"""
BUL System Monitor Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM STATUS
-------------
Overall Status: {health['status'].upper()}
Uptime: {metrics['uptime_formatted']}
CPU Usage: {metrics['cpu_percent']}%
Memory Usage: {metrics['memory']['percent']}%
Disk Usage: {metrics['disk']['percent']:.1f}%

PERFORMANCE METRICS
------------------
Total Requests: {metrics['metrics']['requests_total']}
Success Rate: {performance['success_rate']:.1f}%
Average Response Time: {performance['average_response_time']:.3f}s
Documents Generated: {metrics['metrics']['documents_generated']}
Requests/Minute: {performance['requests_per_minute']:.1f}

HEALTH CHECKS
-------------
"""
        
        for check in health['checks']:
            status_icon = "✅" if check['status'] == 'healthy' else "⚠️" if check['status'] == 'warning' else "❌"
            report += f"{status_icon} {check['component']}: {check['message']}\n"
        
        return report

class BULMonitor:
    """Main monitoring application."""
    
    def __init__(self):
        self.config = BULConfig()
        self.monitor = SystemMonitor(self.config)
        self.running = False
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring."""
        self.running = True
        print("🔍 Starting BUL System Monitor")
        print("=" * 50)
        
        while self.running:
            try:
                # Get current status
                health = self.monitor.get_health_status()
                metrics = self.monitor.get_system_metrics()
                
                # Display status
                print(f"\n📊 System Status - {datetime.now().strftime('%H:%M:%S')}")
                print(f"Status: {health['status'].upper()}")
                print(f"Uptime: {metrics['uptime_formatted']}")
                print(f"CPU: {metrics['cpu_percent']}% | Memory: {metrics['memory']['percent']}%")
                print(f"Requests: {metrics['metrics']['requests_total']} | Success Rate: {(metrics['metrics']['requests_success']/max(1, metrics['metrics']['requests_total']))*100:.1f}%")
                
                # Check for issues
                if health['status'] != 'healthy':
                    print("⚠️  System issues detected:")
                    for check in health['checks']:
                        if check['status'] != 'healthy':
                            print(f"   - {check['component']}: {check['message']}")
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n⏹️  Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(interval)
        
        self.running = False
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
    
    def generate_report(self):
        """Generate and display system report."""
        report = self.monitor.generate_report()
        print(report)
        
        # Save report to file
        report_file = f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")

def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL System Monitor")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--report", action="store_true", help="Generate system report")
    parser.add_argument("--once", action="store_true", help="Run monitoring once and exit")
    
    args = parser.parse_args()
    
    monitor = BULMonitor()
    
    if args.report:
        monitor.generate_report()
        return
    
    if args.once:
        health = monitor.monitor.get_health_status()
        metrics = monitor.monitor.get_system_metrics()
        print("📊 BUL System Status (One-time check)")
        print("=" * 40)
        print(f"Status: {health['status'].upper()}")
        print(f"Uptime: {metrics['uptime_formatted']}")
        print(f"CPU: {metrics['cpu_percent']}% | Memory: {metrics['memory']['percent']}%")
        return
    
    # Start continuous monitoring
    try:
        asyncio.run(monitor.start_monitoring(args.interval))
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()
