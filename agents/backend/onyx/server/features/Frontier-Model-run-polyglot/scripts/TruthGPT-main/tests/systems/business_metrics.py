"""
Business Metrics
Business-focused metrics and KPIs
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class BusinessMetrics:
    """Business-focused metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def calculate_business_metrics(self, lookback_days: int = 30) -> Dict:
        """Calculate business metrics"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate business KPIs
        total_tests = [r.get('total_tests', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        total_tests_sum = sum(total_tests)
        total_time = sum(execution_times)
        avg_success = mean(success_rates) if success_rates else 0
        total_failures = sum(failures)
        
        # Business metrics
        metrics = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'kpis': {
                'test_velocity': round(total_tests_sum / lookback_days, 1),  # Tests per day
                'success_rate': round(avg_success, 2),
                'failure_rate': round((total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0, 2),
                'avg_execution_time': round(mean(execution_times), 2) if execution_times else 0,
                'total_execution_time': round(total_time, 2),
                'tests_per_hour': round((total_tests_sum / (total_time / 3600)) if total_time > 0 else 0, 1)
            },
            'business_value': {
                'quality_score': round(avg_success, 1),
                'efficiency_score': self._calculate_efficiency_score(total_tests_sum, total_time),
                'reliability_score': self._calculate_reliability_score(avg_success, total_failures, total_tests_sum),
                'overall_score': 0
            },
            'trends': self._calculate_business_trends(recent)
        }
        
        # Calculate overall business score
        business_value = metrics['business_value']
        metrics['business_value']['overall_score'] = round(
            (business_value['quality_score'] * 0.4) +
            (business_value['efficiency_score'] * 0.3) +
            (business_value['reliability_score'] * 0.3),
            1
        )
        
        return metrics
    
    def _calculate_efficiency_score(self, total_tests: int, total_time: float) -> float:
        """Calculate efficiency score"""
        if total_time == 0:
            return 0
        
        tests_per_hour = total_tests / (total_time / 3600)
        # Normalize: 100 tests/hour = 100 points
        score = min(100, (tests_per_hour / 100) * 100)
        return round(score, 1)
    
    def _calculate_reliability_score(self, success_rate: float, failures: int, total_tests: int) -> float:
        """Calculate reliability score"""
        failure_rate = (failures / total_tests * 100) if total_tests > 0 else 0
        # Combine success rate and low failure rate
        score = (success_rate * 0.7) + ((100 - failure_rate) * 0.3)
        return round(score, 1)
    
    def _calculate_business_trends(self, recent: List[Dict]) -> Dict:
        """Calculate business trends"""
        if len(recent) < 4:
            return {}
        
        # Split into two halves
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_sr = mean([r.get('success_rate', 0) for r in first_half])
        second_sr = mean([r.get('success_rate', 0) for r in second_half])
        
        first_tests = sum([r.get('total_tests', 0) for r in first_half])
        second_tests = sum([r.get('total_tests', 0) for r in second_half])
        
        return {
            'success_rate_trend': round(second_sr - first_sr, 2),
            'test_velocity_trend': round(second_tests - first_tests, 0),
            'direction': 'improving' if second_sr > first_sr else 'declining' if second_sr < first_sr else 'stable'
        }
    
    def generate_business_report(self, metrics: Dict) -> str:
        """Generate business report"""
        lines = []
        lines.append("=" * 80)
        lines.append("BUSINESS METRICS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in metrics:
            lines.append(f"❌ {metrics['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {metrics['period']}")
        lines.append(f"Total Runs: {metrics['total_runs']}")
        lines.append("")
        
        overall_score = metrics['business_value']['overall_score']
        score_emoji = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Business Score: {overall_score}/100")
        lines.append("")
        
        lines.append("📊 KEY PERFORMANCE INDICATORS (KPIs)")
        lines.append("-" * 80)
        kpis = metrics['kpis']
        lines.append(f"Test Velocity: {kpis['test_velocity']} tests/day")
        lines.append(f"Success Rate: {kpis['success_rate']}%")
        lines.append(f"Failure Rate: {kpis['failure_rate']}%")
        lines.append(f"Average Execution Time: {kpis['avg_execution_time']}s")
        lines.append(f"Total Execution Time: {kpis['total_execution_time']}s")
        lines.append(f"Tests per Hour: {kpis['tests_per_hour']}")
        lines.append("")
        
        lines.append("💼 BUSINESS VALUE SCORES")
        lines.append("-" * 80)
        bv = metrics['business_value']
        lines.append(f"Quality Score: {bv['quality_score']}/100")
        lines.append(f"Efficiency Score: {bv['efficiency_score']}/100")
        lines.append(f"Reliability Score: {bv['reliability_score']}/100")
        lines.append(f"Overall Score: {bv['overall_score']}/100")
        lines.append("")
        
        if 'trends' in metrics and metrics['trends']:
            trends = metrics['trends']
            trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends.get('direction', 'stable'), '➡️')
            lines.append(f"{emoji} BUSINESS TRENDS")
            lines.append("-" * 80)
            lines.append(f"Success Rate Trend: {trends.get('success_rate_trend', 0):+.2f}%")
            lines.append(f"Test Velocity Trend: {trends.get('test_velocity_trend', 0):+.0f} tests")
            lines.append(f"Direction: {trends.get('direction', 'stable').title()}")
        
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
    
    metrics = BusinessMetrics(project_root)
    business_metrics = metrics.calculate_business_metrics(lookback_days=30)
    
    report = metrics.generate_business_report(business_metrics)
    print(report)
    
    # Save report
    report_file = project_root / "business_metrics_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Business metrics report saved to: {report_file}")

if __name__ == "__main__":
    main()







