"""
Enhanced Business Metrics
Enhanced business metrics with advanced KPIs
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class EnhancedBusinessMetrics:
    """Enhanced business metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def calculate_business_metrics(self, lookback_days: int = 30, cost_per_test: float = 0.10) -> Dict:
        """Calculate enhanced business metrics"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate business metrics
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        metrics = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'cost_metrics': self._calculate_cost_metrics(recent, cost_per_test),
            'efficiency_metrics': self._calculate_efficiency_metrics(recent, total_tests, total_time),
            'quality_metrics': self._calculate_quality_metrics(recent, success_rates, total_failures),
            'value_metrics': self._calculate_value_metrics(recent, total_failures),
            'trend_metrics': self._analyze_business_trends(recent),
            'kpis': self._calculate_kpis(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        metrics['recommendations'] = self._generate_business_recommendations(metrics)
        
        return metrics
    
    def _calculate_cost_metrics(self, recent: List[Dict], cost_per_test: float) -> Dict:
        """Calculate cost metrics"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        
        # Estimate costs
        test_execution_cost = total_tests * cost_per_test
        time_cost = (total_time / 3600) * 50  # $50/hour for compute time
        
        total_cost = test_execution_cost + time_cost
        
        # Cost per successful test
        successful_tests = sum(
            r.get('total_tests', 0) * (r.get('success_rate', 0) / 100)
            for r in recent
        )
        cost_per_successful_test = (total_cost / successful_tests) if successful_tests > 0 else 0
        
        return {
            'total_cost': round(total_cost, 2),
            'test_execution_cost': round(test_execution_cost, 2),
            'time_cost': round(time_cost, 2),
            'cost_per_test': round(cost_per_test, 2),
            'cost_per_successful_test': round(cost_per_successful_test, 4),
            'total_tests': total_tests,
            'total_time_hours': round(total_time / 3600, 2)
        }
    
    def _calculate_efficiency_metrics(self, recent: List[Dict], total_tests: int, total_time: float) -> Dict:
        """Calculate efficiency metrics"""
        if total_time == 0:
            return {}
        
        tests_per_hour = (total_tests / total_time) * 3600
        avg_time_per_test = total_time / total_tests if total_tests > 0 else 0
        
        # Efficiency score (normalized)
        efficiency_score = min(100, (tests_per_hour / 100) * 100)  # Normalize to 100
        
        return {
            'tests_per_hour': round(tests_per_hour, 2),
            'avg_time_per_test': round(avg_time_per_test, 3),
            'efficiency_score': round(efficiency_score, 1),
            'total_tests': total_tests,
            'total_time_hours': round(total_time / 3600, 2)
        }
    
    def _calculate_quality_metrics(self, recent: List[Dict], success_rates: List[float], total_failures: int) -> Dict:
        """Calculate quality metrics"""
        avg_success = mean(success_rates) if success_rates else 0
        success_std = stdev(success_rates) if len(success_rates) > 1 else 0
        
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        failure_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
        
        # Quality score
        quality_score = avg_success - (success_std * 0.5)  # Penalize variance
        
        return {
            'avg_success_rate': round(avg_success, 1),
            'success_rate_std': round(success_std, 2),
            'consistency': round(100 - (success_std * 2), 1),
            'total_failures': total_failures,
            'failure_rate': round(failure_rate, 2),
            'quality_score': round(max(0, quality_score), 1)
        }
    
    def _calculate_value_metrics(self, recent: List[Dict], total_failures: int) -> Dict:
        """Calculate value metrics"""
        # Estimate value of catching failures
        value_per_failure_caught = 100  # $100 value per failure caught
        
        total_value = total_failures * value_per_failure_caught
        
        # ROI estimate (simplified)
        total_cost = sum(
            r.get('total_tests', 0) * 0.10 + (r.get('execution_time', 0) / 3600) * 50
            for r in recent
        )
        roi = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_value_generated': round(total_value, 2),
            'value_per_failure': value_per_failure_caught,
            'total_failures_caught': total_failures,
            'estimated_roi': round(roi, 1)
        }
    
    def _analyze_business_trends(self, recent: List[Dict]) -> Dict:
        """Analyze business trends"""
        if len(recent) < 4:
            return {}
        
        # Split into halves
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        # Cost trends
        first_cost = sum(
            r.get('total_tests', 0) * 0.10 + (r.get('execution_time', 0) / 3600) * 50
            for r in first_half
        )
        second_cost = sum(
            r.get('total_tests', 0) * 0.10 + (r.get('execution_time', 0) / 3600) * 50
            for r in second_half
        )
        cost_trend = second_cost - first_cost
        
        # Efficiency trends
        first_tests = sum(r.get('total_tests', 0) for r in first_half)
        first_time = sum(r.get('execution_time', 0) for r in first_half)
        first_efficiency = (first_tests / first_time * 3600) if first_time > 0 else 0
        
        second_tests = sum(r.get('total_tests', 0) for r in second_half)
        second_time = sum(r.get('execution_time', 0) for r in second_half)
        second_efficiency = (second_tests / second_time * 3600) if second_time > 0 else 0
        
        efficiency_trend = second_efficiency - first_efficiency
        
        return {
            'cost_trend': {
                'change': round(cost_trend, 2),
                'direction': 'increasing' if cost_trend > 0 else 'decreasing' if cost_trend < 0 else 'stable',
                'percent_change': round((cost_trend / first_cost * 100) if first_cost > 0 else 0, 1)
            },
            'efficiency_trend': {
                'change': round(efficiency_trend, 2),
                'direction': 'improving' if efficiency_trend > 0 else 'degrading' if efficiency_trend < 0 else 'stable',
                'percent_change': round((efficiency_trend / first_efficiency * 100) if first_efficiency > 0 else 0, 1)
            }
        }
    
    def _calculate_kpis(self, recent: List[Dict]) -> Dict:
        """Calculate key performance indicators"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        success_rates = [r.get('success_rate', 0) for r in recent]
        total_time = sum(r.get('execution_time', 0) for r in recent)
        
        avg_success = mean(success_rates) if success_rates else 0
        failure_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
        tests_per_hour = (total_tests / total_time * 3600) if total_time > 0 else 0
        
        return {
            'test_coverage_rate': round(avg_success, 1),
            'defect_detection_rate': round(failure_rate, 2),
            'test_execution_velocity': round(tests_per_hour, 2),
            'overall_health_score': round((avg_success * 0.6 + (100 - failure_rate) * 0.4), 1)
        }
    
    def _generate_business_recommendations(self, metrics: Dict) -> List[str]:
        """Generate business recommendations"""
        recommendations = []
        
        cost = metrics['cost_metrics']
        if cost.get('cost_per_successful_test', 0) > 0.15:
            recommendations.append(f"Reduce cost per successful test from ${cost['cost_per_successful_test']:.4f} to <$0.10")
        
        efficiency = metrics['efficiency_metrics']
        if efficiency.get('efficiency_score', 0) < 70:
            recommendations.append(f"Improve efficiency score from {efficiency['efficiency_score']:.1f} to 80+")
        
        quality = metrics['quality_metrics']
        if quality.get('quality_score', 0) < 85:
            recommendations.append(f"Improve quality score from {quality['quality_score']:.1f} to 90+")
        
        value = metrics['value_metrics']
        if value.get('estimated_roi', 0) < 100:
            recommendations.append(f"Increase ROI from {value['estimated_roi']:.1f}% to 200%+")
        
        trends = metrics.get('trend_metrics', {})
        if trends.get('cost_trend', {}).get('direction') == 'increasing':
            recommendations.append("Cost trend is increasing - optimize test execution")
        
        if not recommendations:
            recommendations.append("✅ Business metrics are optimal - maintain current practices")
        
        return recommendations
    
    def generate_business_report(self, metrics: Dict) -> str:
        """Generate business report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED BUSINESS METRICS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in metrics:
            lines.append(f"❌ {metrics['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {metrics['period']}")
        lines.append(f"Total Runs: {metrics['total_runs']}")
        lines.append("")
        
        lines.append("💰 COST METRICS")
        lines.append("-" * 80)
        cost = metrics['cost_metrics']
        lines.append(f"Total Cost: ${cost['total_cost']:.2f}")
        lines.append(f"Test Execution Cost: ${cost['test_execution_cost']:.2f}")
        lines.append(f"Time Cost: ${cost['time_cost']:.2f}")
        lines.append(f"Cost per Test: ${cost['cost_per_test']:.2f}")
        lines.append(f"Cost per Successful Test: ${cost['cost_per_successful_test']:.4f}")
        lines.append(f"Total Tests: {cost['total_tests']:,}")
        lines.append(f"Total Time: {cost['total_time_hours']:.2f} hours")
        lines.append("")
        
        lines.append("⚡ EFFICIENCY METRICS")
        lines.append("-" * 80)
        eff = metrics['efficiency_metrics']
        lines.append(f"Tests per Hour: {eff['tests_per_hour']:.2f}")
        lines.append(f"Average Time per Test: {eff['avg_time_per_test']:.3f}s")
        lines.append(f"Efficiency Score: {eff['efficiency_score']}/100")
        lines.append("")
        
        lines.append("✅ QUALITY METRICS")
        lines.append("-" * 80)
        qual = metrics['quality_metrics']
        lines.append(f"Average Success Rate: {qual['avg_success_rate']}%")
        lines.append(f"Success Rate Std Dev: {qual['success_rate_std']}%")
        lines.append(f"Consistency: {qual['consistency']}%")
        lines.append(f"Total Failures: {qual['total_failures']}")
        lines.append(f"Failure Rate: {qual['failure_rate']}%")
        lines.append(f"Quality Score: {qual['quality_score']}/100")
        lines.append("")
        
        lines.append("💎 VALUE METRICS")
        lines.append("-" * 80)
        value = metrics['value_metrics']
        lines.append(f"Total Value Generated: ${value['total_value_generated']:.2f}")
        lines.append(f"Value per Failure: ${value['value_per_failure']}")
        lines.append(f"Failures Caught: {value['total_failures_caught']}")
        lines.append(f"Estimated ROI: {value['estimated_roi']:.1f}%")
        lines.append("")
        
        lines.append("📊 KEY PERFORMANCE INDICATORS")
        lines.append("-" * 80)
        kpis = metrics['kpis']
        lines.append(f"Test Coverage Rate: {kpis['test_coverage_rate']}%")
        lines.append(f"Defect Detection Rate: {kpis['defect_detection_rate']}%")
        lines.append(f"Test Execution Velocity: {kpis['test_execution_velocity']:.2f} tests/hour")
        lines.append(f"Overall Health Score: {kpis['overall_health_score']}/100")
        lines.append("")
        
        if metrics.get('trend_metrics'):
            trends = metrics['trend_metrics']
            lines.append("📈 BUSINESS TRENDS")
            lines.append("-" * 80)
            cost_trend = trends['cost_trend']
            trend_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(cost_trend['direction'], '➡️')
            lines.append(f"{emoji} Cost Trend: {cost_trend['direction'].title()} (${cost_trend['change']:+.2f}, {cost_trend['percent_change']:+.1f}%)")
            
            eff_trend = trends['efficiency_trend']
            emoji = trend_emoji.get(eff_trend['direction'], '➡️')
            lines.append(f"{emoji} Efficiency Trend: {eff_trend['direction'].title()} ({eff_trend['change']:+.2f}, {eff_trend['percent_change']:+.1f}%)")
            lines.append("")
        
        if metrics['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in metrics['recommendations']:
                lines.append(f"• {rec}")
        
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
    
    metrics = EnhancedBusinessMetrics(project_root)
    business_data = metrics.calculate_business_metrics(lookback_days=30, cost_per_test=0.10)
    
    report = metrics.generate_business_report(business_data)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_business_metrics_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced business metrics report saved to: {report_file}")

if __name__ == "__main__":
    main()






