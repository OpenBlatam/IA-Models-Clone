"""
Reliability Monitor
Monitor test reliability over time
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class ReliabilityMonitor:
    """Monitor test reliability"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def monitor_reliability(self, lookback_days: int = 30) -> Dict:
        """Monitor test reliability"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate reliability metrics
        reliability_metrics = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'overall_reliability': 0.0,
            'success_rate_reliability': self._calculate_success_rate_reliability(recent),
            'execution_reliability': self._calculate_execution_reliability(recent),
            'failure_reliability': self._calculate_failure_reliability(recent),
            'reliability_trend': self._analyze_reliability_trend(recent),
            'reliability_issues': []
        }
        
        # Calculate overall reliability
        reliability_scores = [
            reliability_metrics['success_rate_reliability']['score'],
            reliability_metrics['execution_reliability']['score'],
            reliability_metrics['failure_reliability']['score']
        ]
        reliability_metrics['overall_reliability'] = round(mean(reliability_scores), 1)
        
        # Identify reliability issues
        reliability_metrics['reliability_issues'] = self._identify_reliability_issues(reliability_metrics)
        
        return reliability_metrics
    
    def _calculate_success_rate_reliability(self, recent: List[Dict]) -> Dict:
        """Calculate success rate reliability"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not success_rates:
            return {'score': 0, 'reliable': False}
        
        avg_success = mean(success_rates)
        
        # Reliability score based on average success rate
        score = avg_success
        reliable = avg_success >= 95
        
        return {
            'score': round(score, 1),
            'reliable': reliable,
            'average': round(avg_success, 2),
            'min': round(min(success_rates), 2),
            'max': round(max(success_rates), 2)
        }
    
    def _calculate_execution_reliability(self, recent: List[Dict]) -> Dict:
        """Calculate execution reliability"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        if not execution_times:
            return {'score': 0, 'reliable': False}
        
        # Check if tests complete reliably (no timeouts, no crashes)
        completed = sum(1 for r in recent if r.get('execution_time', 0) > 0)
        completion_rate = (completed / len(recent)) * 100
        
        # Reliability score based on completion rate
        score = completion_rate
        reliable = completion_rate >= 98
        
        return {
            'score': round(score, 1),
            'reliable': reliable,
            'completion_rate': round(completion_rate, 2),
            'total_runs': len(recent),
            'completed_runs': completed
        }
    
    def _calculate_failure_reliability(self, recent: List[Dict]) -> Dict:
        """Calculate failure reliability"""
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not total_tests or sum(total_tests) == 0:
            return {'score': 0, 'reliable': False}
        
        total_failures = sum(failures)
        total_tests_sum = sum(total_tests)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        # Reliability score: 100 - failure_rate
        score = max(0, 100 - failure_rate)
        reliable = failure_rate < 5
        
        return {
            'score': round(score, 1),
            'reliable': reliable,
            'failure_rate': round(failure_rate, 2),
            'total_failures': total_failures,
            'total_tests': total_tests_sum
        }
    
    def _analyze_reliability_trend(self, recent: List[Dict]) -> Dict:
        """Analyze reliability trend"""
        if len(recent) < 4:
            return {}
        
        # Split into two halves
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_sr = mean([r.get('success_rate', 0) for r in first_half])
        second_sr = mean([r.get('success_rate', 0) for r in second_half])
        
        trend = second_sr - first_sr
        
        return {
            'trend': round(trend, 1),
            'direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
            'first_half_reliability': round(first_sr, 1),
            'second_half_reliability': round(second_sr, 1)
        }
    
    def _identify_reliability_issues(self, metrics: Dict) -> List[Dict]:
        """Identify reliability issues"""
        issues = []
        
        if not metrics['success_rate_reliability']['reliable']:
            issues.append({
                'type': 'success_rate',
                'severity': 'high',
                'description': 'Success rate reliability is below acceptable threshold',
                'current': f"{metrics['success_rate_reliability']['average']}%",
                'target': '95%+'
            })
        
        if not metrics['execution_reliability']['reliable']:
            issues.append({
                'type': 'execution',
                'severity': 'high',
                'description': 'Test execution completion rate is below threshold',
                'current': f"{metrics['execution_reliability']['completion_rate']}%",
                'target': '98%+'
            })
        
        if not metrics['failure_reliability']['reliable']:
            issues.append({
                'type': 'failure_rate',
                'severity': 'medium',
                'description': 'Failure rate is above acceptable threshold',
                'current': f"{metrics['failure_reliability']['failure_rate']}%",
                'target': '<5%'
            })
        
        return issues
    
    def generate_reliability_report(self, metrics: Dict) -> str:
        """Generate reliability report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RELIABILITY MONITOR REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in metrics:
            lines.append(f"❌ {metrics['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {metrics['period']}")
        lines.append(f"Total Runs: {metrics['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if metrics['overall_reliability'] >= 95 else "🟡" if metrics['overall_reliability'] >= 85 else "🔴"
        lines.append(f"{score_emoji} Overall Reliability: {metrics['overall_reliability']}/100")
        lines.append("")
        
        lines.append("✅ SUCCESS RATE RELIABILITY")
        lines.append("-" * 80)
        sr_rel = metrics['success_rate_reliability']
        status = "✅ Reliable" if sr_rel['reliable'] else "⚠️ Needs Improvement"
        lines.append(f"{status} (Score: {sr_rel['score']}/100)")
        lines.append(f"Average: {sr_rel['average']}%")
        lines.append(f"Range: {sr_rel['min']}% - {sr_rel['max']}%")
        lines.append("")
        
        lines.append("⏱️ EXECUTION RELIABILITY")
        lines.append("-" * 80)
        ex_rel = metrics['execution_reliability']
        status = "✅ Reliable" if ex_rel['reliable'] else "⚠️ Needs Improvement"
        lines.append(f"{status} (Score: {ex_rel['score']}/100)")
        lines.append(f"Completion Rate: {ex_rel['completion_rate']}%")
        lines.append(f"Completed Runs: {ex_rel['completed_runs']}/{ex_rel['total_runs']}")
        lines.append("")
        
        lines.append("❌ FAILURE RELIABILITY")
        lines.append("-" * 80)
        fail_rel = metrics['failure_reliability']
        status = "✅ Reliable" if fail_rel['reliable'] else "⚠️ Needs Improvement"
        lines.append(f"{status} (Score: {fail_rel['score']}/100)")
        lines.append(f"Failure Rate: {fail_rel['failure_rate']}%")
        lines.append(f"Total Failures: {fail_rel['total_failures']}/{fail_rel['total_tests']}")
        lines.append("")
        
        if 'reliability_trend' in metrics and metrics['reliability_trend']:
            trend = metrics['reliability_trend']
            trend_emoji = "📈" if trend['direction'] == 'improving' else "📉" if trend['direction'] == 'declining' else "➡️"
            lines.append(f"{trend_emoji} RELIABILITY TREND")
            lines.append("-" * 80)
            lines.append(f"Direction: {trend['direction'].title()}")
            lines.append(f"Trend: {trend['trend']:+.1f}%")
            lines.append(f"First Half: {trend['first_half_reliability']:.1f}%")
            lines.append(f"Second Half: {trend['second_half_reliability']:.1f}%")
            lines.append("")
        
        if metrics['reliability_issues']:
            lines.append("⚠️ RELIABILITY ISSUES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for issue in metrics['reliability_issues']:
                emoji = severity_emoji.get(issue['severity'], '⚪')
                lines.append(f"{emoji} [{issue['severity'].upper()}] {issue['type'].replace('_', ' ').title()}")
                lines.append(f"   {issue['description']}")
                lines.append(f"   Current: {issue['current']} | Target: {issue['target']}")
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
    
    monitor = ReliabilityMonitor(project_root)
    metrics = monitor.monitor_reliability(lookback_days=30)
    
    report = monitor.generate_reliability_report(metrics)
    print(report)
    
    # Save report
    report_file = project_root / "reliability_monitor_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Reliability monitor report saved to: {report_file}")

if __name__ == "__main__":
    main()







