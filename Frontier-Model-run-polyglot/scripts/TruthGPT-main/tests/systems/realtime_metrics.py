"""
Real-time Metrics
Real-time metrics tracking and monitoring
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class RealTimeMetrics:
    """Real-time metrics tracking"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.metrics_file = project_root / "realtime_metrics.json"
    
    def get_realtime_metrics(self, window_minutes: int = 60) -> Dict:
        """Get real-time metrics"""
        history = self._load_history()
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent = [
            r for r in history
            if datetime.fromisoformat(r.get('timestamp', '2000-01-01').replace('Z', '+00:00').split('.')[0]) >= cutoff_time
        ]
        
        if not recent:
            return {'error': 'No data in the specified window'}
        
        # Calculate real-time metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        total_tests_sum = sum(total_tests)
        total_failures = sum(failures)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'window_minutes': window_minutes,
            'total_runs': len(recent),
            'current_metrics': {
                'success_rate': round(mean(success_rates), 2) if success_rates else 0,
                'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
                'total_tests': total_tests_sum,
                'total_failures': total_failures,
                'failure_rate': round((total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0, 2),
                'tests_per_minute': round((total_tests_sum / window_minutes) if window_minutes > 0 else 0, 2)
            },
            'status': self._determine_status(mean(success_rates) if success_rates else 0, total_failures, total_tests_sum),
            'alerts': self._check_alerts(success_rates, execution_times, failures)
        }
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _determine_status(self, success_rate: float, failures: int, total_tests: int) -> Dict:
        """Determine current status"""
        failure_rate = (failures / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate >= 95 and failure_rate < 5:
            status = 'healthy'
            emoji = '🟢'
        elif success_rate >= 90 and failure_rate < 10:
            status = 'warning'
            emoji = '🟡'
        else:
            status = 'critical'
            emoji = '🔴'
        
        return {
            'status': status,
            'emoji': emoji,
            'success_rate': round(success_rate, 1),
            'failure_rate': round(failure_rate, 1)
        }
    
    def _check_alerts(self, success_rates: List[float], execution_times: List[float], failures: List[int]) -> List[Dict]:
        """Check for alerts"""
        alerts = []
        
        if success_rates:
            avg_success = mean(success_rates)
            if avg_success < 90:
                alerts.append({
                    'type': 'low_success_rate',
                    'severity': 'high' if avg_success < 80 else 'medium',
                    'message': f'Success rate is {avg_success:.1f}%',
                    'timestamp': datetime.now().isoformat()
                })
        
        if execution_times:
            avg_time = mean(execution_times)
            if avg_time > 300:
                alerts.append({
                    'type': 'slow_execution',
                    'severity': 'medium',
                    'message': f'Average execution time is {avg_time:.0f}s',
                    'timestamp': datetime.now().isoformat()
                })
        
        if failures:
            total_failures = sum(failures)
            if total_failures > 10:
                alerts.append({
                    'type': 'high_failures',
                    'severity': 'high',
                    'message': f'Total failures: {total_failures}',
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _save_metrics(self, metrics: Dict):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass
    
    def generate_realtime_report(self, metrics: Dict) -> str:
        """Generate real-time report"""
        lines = []
        lines.append("=" * 80)
        lines.append("REAL-TIME METRICS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in metrics:
            lines.append(f"❌ {metrics['error']}")
            return "\n".join(lines)
        
        lines.append(f"Timestamp: {metrics['timestamp'][:19]}")
        lines.append(f"Window: Last {metrics['window_minutes']} minutes")
        lines.append(f"Total Runs: {metrics['total_runs']}")
        lines.append("")
        
        status = metrics['status']
        lines.append(f"{status['emoji']} STATUS: {status['status'].upper()}")
        lines.append(f"Success Rate: {status['success_rate']}%")
        lines.append(f"Failure Rate: {status['failure_rate']}%")
        lines.append("")
        
        lines.append("📊 CURRENT METRICS")
        lines.append("-" * 80)
        current = metrics['current_metrics']
        lines.append(f"Success Rate: {current['success_rate']}%")
        lines.append(f"Average Execution Time: {current['avg_execution_time']}s")
        lines.append(f"Total Tests: {current['total_tests']:,}")
        lines.append(f"Total Failures: {current['total_failures']}")
        lines.append(f"Failure Rate: {current['failure_rate']}%")
        lines.append(f"Tests per Minute: {current['tests_per_minute']}")
        lines.append("")
        
        if metrics['alerts']:
            lines.append("🚨 ALERTS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for alert in metrics['alerts']:
                emoji = severity_emoji.get(alert['severity'], '⚪')
                lines.append(f"{emoji} [{alert['severity'].upper()}] {alert['type'].replace('_', ' ').title()}")
                lines.append(f"   {alert['message']}")
                lines.append(f"   Time: {alert['timestamp'][:19]}")
            lines.append("")
        else:
            lines.append("✅ No alerts - system is healthy")
            lines.append("")
        
        return "\n".join(lines)
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

def main():
    """Main function"""
    from pathlib import Path
    import time
    
    project_root = Path(__file__).parent.parent
    
    metrics = RealTimeMetrics(project_root)
    
    print("Real-time Metrics Monitor (Press Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        while True:
            realtime_data = metrics.get_realtime_metrics(window_minutes=60)
            report = metrics.generate_realtime_report(realtime_data)
            print("\n" + report)
            print("\n" + "=" * 80)
            print("Refreshing in 30 seconds...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()







