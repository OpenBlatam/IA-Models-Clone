"""
Intelligent Alerts System
ML-based intelligent alerting system
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class IntelligentAlerts:
    """Intelligent alerting system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.alerts_file = project_root / "intelligent_alerts.json"
    
    def generate_intelligent_alerts(self, lookback_days: int = 30) -> Dict:
        """Generate intelligent alerts"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze patterns
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Detect anomalies
        anomalies = self._detect_anomalies(success_rates, execution_times, failures)
        
        # Generate intelligent alerts
        alerts = []
        
        # Success rate anomaly
        if anomalies['success_rate_anomaly']:
            alerts.append({
                'type': 'success_rate_anomaly',
                'severity': self._calculate_severity(success_rates),
                'message': f"Unusual success rate pattern detected: {anomalies['success_rate_anomaly']['description']}",
                'confidence': anomalies['success_rate_anomaly']['confidence'],
                'recommendation': 'Investigate recent changes that may have affected test success rate'
            })
        
        # Execution time anomaly
        if anomalies['execution_time_anomaly']:
            alerts.append({
                'type': 'execution_time_anomaly',
                'severity': 'medium',
                'message': f"Execution time anomaly detected: {anomalies['execution_time_anomaly']['description']}",
                'confidence': anomalies['execution_time_anomaly']['confidence'],
                'recommendation': 'Review slow tests and optimize execution time'
            })
        
        # Failure pattern anomaly
        if anomalies['failure_pattern_anomaly']:
            alerts.append({
                'type': 'failure_pattern_anomaly',
                'severity': 'high',
                'message': f"Failure pattern anomaly detected: {anomalies['failure_pattern_anomaly']['description']}",
                'confidence': anomalies['failure_pattern_anomaly']['confidence'],
                'recommendation': 'Investigate root causes of increased failures'
            })
        
        # Trend-based alerts
        trend_alerts = self._generate_trend_alerts(success_rates, execution_times)
        alerts.extend(trend_alerts)
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'total_alerts': len(alerts),
            'alerts': alerts,
            'anomalies_detected': sum(1 for a in anomalies.values() if a),
            'summary': self._generate_alert_summary(alerts)
        }
    
    def _detect_anomalies(self, success_rates: List[float], execution_times: List[float], failures: List[int]) -> Dict:
        """Detect anomalies using statistical methods"""
        anomalies = {
            'success_rate_anomaly': None,
            'execution_time_anomaly': None,
            'failure_pattern_anomaly': None
        }
        
        if len(success_rates) < 3:
            return anomalies
        
        # Success rate anomaly detection
        mean_sr = mean(success_rates)
        std_sr = stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Check for significant deviation
        if std_sr > 10:  # High variance
            anomalies['success_rate_anomaly'] = {
                'description': f'High variance detected (std: {std_sr:.2f}%)',
                'confidence': min(100, std_sr * 5)
            }
        
        # Execution time anomaly
        mean_et = mean(execution_times)
        std_et = stdev(execution_times) if len(execution_times) > 1 else 0
        
        if std_et > mean_et * 0.5:  # High relative variance
            anomalies['execution_time_anomaly'] = {
                'description': f'High execution time variance (std: {std_et:.2f}s)',
                'confidence': min(100, (std_et / mean_et * 100) if mean_et > 0 else 0)
            }
        
        # Failure pattern anomaly
        if len(failures) > 0:
            mean_failures = mean(failures)
            recent_failures = failures[-3:] if len(failures) >= 3 else failures
            
            if any(f > mean_failures * 2 for f in recent_failures):
                anomalies['failure_pattern_anomaly'] = {
                    'description': 'Recent failures significantly above average',
                    'confidence': 75
                }
        
        return anomalies
    
    def _calculate_severity(self, values: List[float]) -> str:
        """Calculate severity based on values"""
        if not values:
            return 'low'
        
        mean_val = mean(values)
        
        if mean_val < 80:
            return 'high'
        elif mean_val < 90:
            return 'medium'
        else:
            return 'low'
    
    def _generate_trend_alerts(self, success_rates: List[float], execution_times: List[float]) -> List[Dict]:
        """Generate alerts based on trends"""
        alerts = []
        
        if len(success_rates) < 4:
            return alerts
        
        # Check for declining trend
        first_half = success_rates[:len(success_rates)//2]
        second_half = success_rates[len(success_rates)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        if second_avg < first_avg - 5:  # Significant decline
            alerts.append({
                'type': 'declining_trend',
                'severity': 'high',
                'message': f'Success rate declining: {first_avg:.1f}% → {second_avg:.1f}%',
                'confidence': 85,
                'recommendation': 'Immediate action required to reverse declining trend'
            })
        
        return alerts
    
    def _generate_alert_summary(self, alerts: List[Dict]) -> Dict:
        """Generate alert summary"""
        if not alerts:
            return {'status': 'healthy', 'message': 'No alerts generated'}
        
        high_severity = sum(1 for a in alerts if a['severity'] == 'high')
        medium_severity = sum(1 for a in alerts if a['severity'] == 'medium')
        
        if high_severity > 0:
            status = 'critical'
            message = f'{high_severity} high-severity alerts require immediate attention'
        elif medium_severity > 0:
            status = 'warning'
            message = f'{medium_severity} medium-severity alerts need review'
        else:
            status = 'info'
            message = f'{len(alerts)} informational alerts'
        
        return {
            'status': status,
            'message': message,
            'high_severity_count': high_severity,
            'medium_severity_count': medium_severity
        }
    
    def generate_alerts_report(self, alerts_data: Dict) -> str:
        """Generate alerts report"""
        lines = []
        lines.append("=" * 80)
        lines.append("INTELLIGENT ALERTS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in alerts_data:
            lines.append(f"❌ {alerts_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {alerts_data['period']}")
        lines.append(f"Total Runs: {alerts_data['total_runs']}")
        lines.append(f"Total Alerts: {alerts_data['total_alerts']}")
        lines.append(f"Anomalies Detected: {alerts_data['anomalies_detected']}")
        lines.append("")
        
        summary = alerts_data['summary']
        status_emoji = {'critical': '🔴', 'warning': '🟡', 'info': '🟢', 'healthy': '✅'}
        emoji = status_emoji.get(summary['status'], '⚪')
        lines.append(f"{emoji} Status: {summary['status'].upper()}")
        lines.append(f"   {summary['message']}")
        lines.append("")
        
        if alerts_data['alerts']:
            lines.append("🚨 ALERTS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            
            for i, alert in enumerate(alerts_data['alerts'], 1):
                emoji = severity_emoji.get(alert['severity'], '⚪')
                lines.append(f"\n{i}. {emoji} [{alert['severity'].upper()}] {alert['type'].replace('_', ' ').title()}")
                lines.append(f"   Message: {alert['message']}")
                lines.append(f"   Confidence: {alert['confidence']}%")
                lines.append(f"   Recommendation: {alert['recommendation']}")
            lines.append("")
        else:
            lines.append("✅ No alerts generated - system is healthy")
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
    project_root = Path(__file__).parent.parent
    
    alerts = IntelligentAlerts(project_root)
    alerts_data = alerts.generate_intelligent_alerts(lookback_days=30)
    
    report = alerts.generate_alerts_report(alerts_data)
    print(report)
    
    # Save report
    report_file = project_root / "intelligent_alerts_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Intelligent alerts report saved to: {report_file}")

if __name__ == "__main__":
    main()







