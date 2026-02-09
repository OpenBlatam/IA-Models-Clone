"""
Test Alerts and Thresholds
Configure alerts and thresholds for test metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

class AlertThreshold:
    """Alert threshold configuration"""
    
    def __init__(
        self,
        name: str,
        metric: str,
        operator: str,
        value: float,
        severity: str = "warning"
    ):
        self.name = name
        self.metric = metric  # success_rate, execution_time, failure_count, etc.
        self.operator = operator  # <, >, <=, >=, ==
        self.value = value
        self.severity = severity  # info, warning, error, critical
    
    def check(self, current_value: float) -> bool:
        """Check if threshold is breached"""
        if self.operator == '<':
            return current_value < self.value
        elif self.operator == '>':
            return current_value > self.value
        elif self.operator == '<=':
            return current_value <= self.value
        elif self.operator == '>=':
            return current_value >= self.value
        elif self.operator == '==':
            return abs(current_value - self.value) < 0.01
        return False

class TestAlertSystem:
    """Alert system for test metrics"""
    
    def __init__(self, project_root: Path, config_file: str = "alert_config.json"):
        self.project_root = project_root
        self.config_file = project_root / config_file
        self.thresholds: List[AlertThreshold] = []
        self.alert_history: List[Dict] = []
        self._load_config()
    
    def _load_config(self):
        """Load alert configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    for threshold_config in config.get('thresholds', []):
                        threshold = AlertThreshold(**threshold_config)
                        self.thresholds.append(threshold)
            except Exception as e:
                print(f"⚠️  Error loading alert config: {e}")
    
    def _save_config(self):
        """Save alert configuration"""
        config = {
            'thresholds': [
                {
                    'name': t.name,
                    'metric': t.metric,
                    'operator': t.operator,
                    'value': t.value,
                    'severity': t.severity
                }
                for t in self.thresholds
            ]
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def add_threshold(
        self,
        name: str,
        metric: str,
        operator: str,
        value: float,
        severity: str = "warning"
    ):
        """Add alert threshold"""
        threshold = AlertThreshold(name, metric, operator, value, severity)
        self.thresholds.append(threshold)
        self._save_config()
    
    def check_thresholds(self, test_results: Dict) -> List[Dict]:
        """Check all thresholds against test results"""
        alerts = []
        
        metrics = {
            'success_rate': test_results.get('success_rate', 0),
            'execution_time': test_results.get('execution_time', 0),
            'failure_count': test_results.get('failures', 0) + test_results.get('errors', 0),
            'total_tests': test_results.get('total_tests', 0),
            'skipped_count': test_results.get('skipped', 0)
        }
        
        for threshold in self.thresholds:
            if threshold.metric in metrics:
                current_value = metrics[threshold.metric]
                if threshold.check(current_value):
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'threshold': threshold.name,
                        'metric': threshold.metric,
                        'current_value': current_value,
                        'threshold_value': threshold.value,
                        'operator': threshold.operator,
                        'severity': threshold.severity,
                        'message': self._generate_alert_message(threshold, current_value)
                    }
                    alerts.append(alert)
                    self.alert_history.append(alert)
        
        return alerts
    
    def _generate_alert_message(self, threshold: AlertThreshold, current_value: float) -> str:
        """Generate alert message"""
        return f"{threshold.name}: {threshold.metric} ({current_value}) {threshold.operator} {threshold.value}"
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return sorted(
            self.alert_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict]:
        """Get alerts by severity"""
        return [a for a in self.alert_history if a['severity'] == severity]

def create_default_alerts(alert_system: TestAlertSystem):
    """Create default alert thresholds"""
    # Success rate alerts
    alert_system.add_threshold(
        "Low Success Rate",
        "success_rate",
        "<",
        90.0,
        "error"
    )
    
    alert_system.add_threshold(
        "Warning Success Rate",
        "success_rate",
        "<",
        95.0,
        "warning"
    )
    
    # Execution time alerts
    alert_system.add_threshold(
        "Slow Execution",
        "execution_time",
        ">",
        300.0,  # 5 minutes
        "warning"
    )
    
    # Failure count alerts
    alert_system.add_threshold(
        "High Failure Count",
        "failure_count",
        ">",
        10,
        "error"
    )

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    alert_system = TestAlertSystem(project_root)
    
    # Create default alerts
    create_default_alerts(alert_system)
    
    # Check test results
    test_results = {
        'total_tests': 204,
        'passed': 180,
        'failures': 20,
        'errors': 4,
        'skipped': 0,
        'success_rate': 88.2,  # Below 90% threshold
        'execution_time': 350.0  # Above 300s threshold
    }
    
    alerts = alert_system.check_thresholds(test_results)
    
    print(f"🔔 Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  [{alert['severity'].upper()}] {alert['message']}")

if __name__ == "__main__":
    main()







