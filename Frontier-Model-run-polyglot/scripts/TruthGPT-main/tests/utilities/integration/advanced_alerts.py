"""
Advanced Alerts System
Advanced alerting with multiple channels and rules
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class AdvancedAlerts:
    """Advanced alerting system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.alerts_config_file = project_root / "alerts_config.json"
    
    def check_alerts(self, lookback_days: int = 7) -> Dict:
        """Check all configured alerts"""
        history = self._load_history()
        config = self._load_alerts_config()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        triggered_alerts = []
        
        # Check each alert rule
        for rule in config.get('rules', []):
            if self._evaluate_rule(rule, recent):
                alert = {
                    'rule_name': rule['name'],
                    'severity': rule.get('severity', 'medium'),
                    'message': rule.get('message', ''),
                    'threshold': rule.get('threshold'),
                    'actual_value': self._get_metric_value(rule['metric'], recent),
                    'timestamp': datetime.now().isoformat()
                }
                triggered_alerts.append(alert)
        
        return {
            'total_alerts': len(triggered_alerts),
            'alerts': triggered_alerts,
            'period': f'Last {lookback_days} days'
        }
    
    def _evaluate_rule(self, rule: Dict, recent: List[Dict]) -> bool:
        """Evaluate if alert rule is triggered"""
        metric = rule['metric']
        operator = rule.get('operator', '>')
        threshold = rule.get('threshold', 0)
        
        value = self._get_metric_value(metric, recent)
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.01
        
        return False
    
    def _get_metric_value(self, metric: str, recent: List[Dict]) -> float:
        """Get metric value from recent runs"""
        if metric == 'success_rate':
            success_rates = [r.get('success_rate', 0) for r in recent]
            return mean(success_rates) if success_rates else 0
        elif metric == 'execution_time':
            execution_times = [r.get('execution_time', 0) for r in recent]
            return mean(execution_times) if execution_times else 0
        elif metric == 'failure_count':
            return sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        else:
            return 0.0
    
    def _load_alerts_config(self) -> Dict:
        """Load alerts configuration"""
        if self.alerts_config_file.exists():
            try:
                with open(self.alerts_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default configuration
        default_config = {
            'rules': [
                {
                    'name': 'Low Success Rate',
                    'metric': 'success_rate',
                    'operator': '<',
                    'threshold': 90,
                    'severity': 'high',
                    'message': 'Success rate is below 90%'
                },
                {
                    'name': 'High Execution Time',
                    'metric': 'execution_time',
                    'operator': '>',
                    'threshold': 300,
                    'severity': 'medium',
                    'message': 'Execution time exceeds 300 seconds'
                },
                {
                    'name': 'High Failure Count',
                    'metric': 'failure_count',
                    'operator': '>',
                    'threshold': 10,
                    'severity': 'high',
                    'message': 'Failure count exceeds 10'
                }
            ]
        }
        
        # Save default config
        with open(self.alerts_config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def generate_alerts_report(self, alerts: Dict) -> str:
        """Generate alerts report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED ALERTS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in alerts:
            lines.append(f"❌ {alerts['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {alerts['period']}")
        lines.append(f"Total Alerts: {alerts['total_alerts']}")
        lines.append("")
        
        if not alerts['alerts']:
            lines.append("✅ No alerts triggered")
            return "\n".join(lines)
        
        severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        
        lines.append("🚨 TRIGGERED ALERTS")
        lines.append("-" * 80)
        
        for alert in alerts['alerts']:
            emoji = severity_emoji.get(alert['severity'], '⚪')
            lines.append(f"\n{emoji} [{alert['severity'].upper()}] {alert['rule_name']}")
            lines.append(f"   Message: {alert['message']}")
            lines.append(f"   Threshold: {alert['threshold']}")
            lines.append(f"   Actual Value: {alert['actual_value']}")
            lines.append(f"   Triggered: {alert['timestamp'][:19]}")
        
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
    
    alerts = AdvancedAlerts(project_root)
    alert_results = alerts.check_alerts(lookback_days=7)
    
    report = alerts.generate_alerts_report(alert_results)
    print(report)
    
    # Save report
    report_file = project_root / "alerts_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Alerts report saved to: {report_file}")

if __name__ == "__main__":
    main()







