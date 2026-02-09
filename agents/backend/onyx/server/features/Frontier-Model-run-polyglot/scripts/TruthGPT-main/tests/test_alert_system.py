"""
Intelligent Test Alert System
Configurable alerts based on test results, trends, and thresholds
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class TestAlertSystem:
    """Intelligent alert system for test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_file = project_root / "config" / "alerts.json"
        self.config = self._load_config()
        self.history_file = project_root / "test_alert_history.json"
        self.alert_history = self._load_history()
    
    def _load_config(self) -> Dict:
        """Load alert configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default alert configuration"""
        return {
            'alerts': [
                {
                    'name': 'high_failure_rate',
                    'enabled': True,
                    'condition': 'failure_rate > 0.2',
                    'severity': 'high',
                    'message': 'Failure rate exceeds 20%'
                },
                {
                    'name': 'critical_test_failed',
                    'enabled': True,
                    'condition': 'test_name in critical_tests and status == "failed"',
                    'severity': 'critical',
                    'message': 'Critical test failed',
                    'critical_tests': []
                },
                {
                    'name': 'performance_regression',
                    'enabled': True,
                    'condition': 'duration_increase > 0.5',
                    'severity': 'medium',
                    'message': 'Test duration increased by more than 50%'
                },
                {
                    'name': 'new_failures',
                    'enabled': True,
                    'condition': 'status == "failed" and previous_status == "passed"',
                    'severity': 'high',
                    'message': 'Previously passing test now failing'
                }
            ],
            'notification_channels': {
                'email': {'enabled': False},
                'slack': {'enabled': False},
                'webhook': {'enabled': False}
            }
        }
    
    def _load_history(self) -> List[Dict]:
        """Load alert history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save alert history"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.alert_history, f, indent=2)
    
    def evaluate_alerts(
        self,
        test_results: Dict,
        previous_results: Dict = None
    ) -> List[Dict]:
        """Evaluate all configured alerts"""
        alerts = []
        
        for alert_config in self.config.get('alerts', []):
            if not alert_config.get('enabled', True):
                continue
            
            alert = self._evaluate_alert(alert_config, test_results, previous_results)
            if alert:
                alerts.append(alert)
                self._record_alert(alert)
        
        return alerts
    
    def _evaluate_alert(
        self,
        alert_config: Dict,
        current: Dict,
        previous: Dict = None
    ) -> Optional[Dict]:
        """Evaluate a single alert condition"""
        condition = alert_config.get('condition', '')
        
        # Build evaluation context
        context = {
            'test_name': current.get('test_name', ''),
            'status': current.get('status', ''),
            'duration': current.get('duration', 0),
            'error_message': current.get('error_message', ''),
            'failure_rate': self._calculate_failure_rate(current),
            'previous_status': previous.get('status') if previous else None,
            'duration_increase': self._calculate_duration_increase(current, previous),
            'critical_tests': alert_config.get('critical_tests', [])
        }
        
        # Simple condition evaluation (for production, use a proper expression evaluator)
        try:
            if self._evaluate_condition(condition, context):
                return {
                    'name': alert_config.get('name', 'unknown'),
                    'severity': alert_config.get('severity', 'medium'),
                    'message': alert_config.get('message', 'Alert triggered'),
                    'test_name': context['test_name'],
                    'timestamp': datetime.now().isoformat(),
                    'context': context
                }
        except Exception as e:
            print(f"Error evaluating alert {alert_config.get('name')}: {e}")
        
        return None
    
    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate condition string (simplified)"""
        # Replace context variables
        for key, value in context.items():
            if isinstance(value, str):
                condition = condition.replace(f'{key}', f'"{value}"')
            else:
                condition = condition.replace(f'{key}', str(value))
        
        # Simple evaluation (in production, use ast.literal_eval or similar)
        try:
            return eval(condition)
        except:
            return False
    
    def _calculate_failure_rate(self, result: Dict) -> float:
        """Calculate failure rate (simplified)"""
        # In real implementation, would look at history
        return 0.0
    
    def _calculate_duration_increase(
        self,
        current: Dict,
        previous: Dict = None
    ) -> float:
        """Calculate duration increase"""
        if not previous:
            return 0.0
        
        current_dur = current.get('duration', 0)
        previous_dur = previous.get('duration', 0)
        
        if previous_dur == 0:
            return 0.0
        
        return (current_dur - previous_dur) / previous_dur
    
    def _record_alert(self, alert: Dict):
        """Record alert in history"""
        self.alert_history.append(alert)
        
        # Keep last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        self._save_history()
    
    def get_recent_alerts(
        self,
        hours: int = 24,
        severity: str = None
    ) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]
        
        if severity:
            recent = [a for a in recent if a.get('severity') == severity]
        
        return sorted(recent, key=lambda x: x['timestamp'], reverse=True)
    
    def generate_alert_summary(
        self,
        hours: int = 24
    ) -> Dict:
        """Generate alert summary"""
        recent = self.get_recent_alerts(hours)
        
        by_severity = defaultdict(int)
        by_alert_type = defaultdict(int)
        
        for alert in recent:
            by_severity[alert.get('severity', 'unknown')] += 1
            by_alert_type[alert.get('name', 'unknown')] += 1
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent),
            'by_severity': dict(by_severity),
            'by_type': dict(by_alert_type),
            'recent_alerts': recent[:20]
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Alert System')
    parser.add_argument('--recent', type=int, default=24, help='Get recent alerts (hours)')
    parser.add_argument('--severity', type=str, help='Filter by severity')
    parser.add_argument('--summary', action='store_true', help='Generate alert summary')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    alert_system = TestAlertSystem(project_root)
    
    if args.summary:
        print(f"📊 Generating alert summary (last {args.recent} hours)...")
        summary = alert_system.generate_alert_summary(args.recent)
        print(f"\n  Total Alerts: {summary['total_alerts']}")
        print(f"  By Severity: {summary['by_severity']}")
        print(f"  By Type: {summary['by_type']}")
    else:
        print(f"🔔 Recent alerts (last {args.recent} hours):")
        alerts = alert_system.get_recent_alerts(args.recent, args.severity)
        for alert in alerts[:10]:
            print(f"  [{alert['severity']}] {alert['name']}: {alert['message']}")


if __name__ == '__main__':
    main()

