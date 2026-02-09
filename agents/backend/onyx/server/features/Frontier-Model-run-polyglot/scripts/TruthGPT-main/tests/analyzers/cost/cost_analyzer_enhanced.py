"""
Enhanced Cost Analyzer
Enhanced cost analysis with comprehensive breakdown
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class EnhancedCostAnalyzer:
    """Enhanced cost analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_costs(self, lookback_days: int = 30, cost_per_test: float = 0.10, hourly_rate: float = 50.0) -> Dict:
        """Analyze costs comprehensively"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Comprehensive cost analysis
        cost_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'cost_parameters': {
                'cost_per_test': cost_per_test,
                'hourly_rate': hourly_rate
            },
            'total_costs': self._calculate_total_costs(recent, cost_per_test, hourly_rate),
            'cost_breakdown': self._breakdown_costs(recent, cost_per_test, hourly_rate),
            'cost_efficiency': self._analyze_cost_efficiency(recent, cost_per_test, hourly_rate),
            'cost_trends': self._analyze_cost_trends(recent, cost_per_test, hourly_rate),
            'cost_optimization': self._identify_cost_optimization(recent, cost_per_test, hourly_rate),
            'roi_analysis': self._analyze_roi(recent, cost_per_test, hourly_rate),
            'recommendations': []
        }
        
        # Generate recommendations
        cost_analysis['recommendations'] = self._generate_cost_recommendations(cost_analysis)
        
        return cost_analysis
    
    def _calculate_total_costs(self, recent: List[Dict], cost_per_test: float, hourly_rate: float) -> Dict:
        """Calculate total costs"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        
        test_execution_cost = total_tests * cost_per_test
        time_cost = (total_time / 3600) * hourly_rate
        total_cost = test_execution_cost + time_cost
        
        return {
            'test_execution_cost': round(test_execution_cost, 2),
            'time_cost': round(time_cost, 2),
            'total_cost': round(total_cost, 2),
            'total_tests': total_tests,
            'total_time_hours': round(total_time / 3600, 2)
        }
    
    def _breakdown_costs(self, recent: List[Dict], cost_per_test: float, hourly_rate: float) -> Dict:
        """Breakdown costs by category"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        
        # Cost breakdown
        successful_tests = sum(
            r.get('total_tests', 0) * (r.get('success_rate', 0) / 100)
            for r in recent
        )
        failed_tests = total_tests - successful_tests
        
        successful_test_cost = successful_tests * cost_per_test
        failed_test_cost = failed_tests * cost_per_test
        time_cost = (total_time / 3600) * hourly_rate
        
        return {
            'successful_tests_cost': round(successful_test_cost, 2),
            'failed_tests_cost': round(failed_test_cost, 2),
            'time_cost': round(time_cost, 2),
            'successful_tests': round(successful_tests, 0),
            'failed_tests': failed_tests,
            'cost_per_successful_test': round((successful_test_cost / successful_tests) if successful_tests > 0 else 0, 4),
            'cost_per_failed_test': round((failed_test_cost / failed_tests) if failed_tests > 0 else 0, 4)
        }
    
    def _analyze_cost_efficiency(self, recent: List[Dict], cost_per_test: float, hourly_rate: float) -> Dict:
        """Analyze cost efficiency"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        total_cost = (total_tests * cost_per_test) + ((total_time / 3600) * hourly_rate)
        
        if total_cost == 0:
            return {}
        
        # Efficiency metrics
        tests_per_dollar = total_tests / total_cost if total_cost > 0 else 0
        tests_per_hour = (total_tests / total_time * 3600) if total_time > 0 else 0
        
        # Cost per test
        cost_per_test_actual = total_cost / total_tests if total_tests > 0 else 0
        
        return {
            'tests_per_dollar': round(tests_per_dollar, 2),
            'tests_per_hour': round(tests_per_hour, 2),
            'cost_per_test': round(cost_per_test_actual, 4),
            'efficiency_score': round(min(100, tests_per_dollar * 10), 1)
        }
    
    def _analyze_cost_trends(self, recent: List[Dict], cost_per_test: float, hourly_rate: float) -> Dict:
        """Analyze cost trends"""
        if len(recent) < 4:
            return {}
        
        # Split into halves
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        first_cost = sum(
            (r.get('total_tests', 0) * cost_per_test) + ((r.get('execution_time', 0) / 3600) * hourly_rate)
            for r in first_half
        )
        second_cost = sum(
            (r.get('total_tests', 0) * cost_per_test) + ((r.get('execution_time', 0) / 3600) * hourly_rate)
            for r in second_half
        )
        
        cost_change = second_cost - first_cost
        percent_change = (cost_change / first_cost * 100) if first_cost > 0 else 0
        
        return {
            'first_half_cost': round(first_cost, 2),
            'second_half_cost': round(second_cost, 2),
            'cost_change': round(cost_change, 2),
            'percent_change': round(percent_change, 2),
            'direction': 'increasing' if cost_change > 0 else 'decreasing' if cost_change < 0 else 'stable',
            'trend': 'concerning' if percent_change > 10 else 'acceptable' if percent_change > -10 else 'improving'
        }
    
    def _identify_cost_optimization(self, recent: List[Dict], cost_per_test: float, hourly_rate: float) -> Dict:
        """Identify cost optimization opportunities"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        total_cost = (total_tests * cost_per_test) + ((total_time / 3600) * hourly_rate)
        
        # Optimization potential
        avg_time = mean([r.get('execution_time', 0) for r in recent]) if recent else 0
        target_time = 120  # seconds
        
        time_reduction_potential = max(0, avg_time - target_time)
        cost_savings_potential = (time_reduction_potential * len(recent) / 3600) * hourly_rate
        
        # Failure reduction potential
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        failure_cost = total_failures * cost_per_test
        failure_reduction_potential = failure_cost * 0.5  # Assume 50% reduction possible
        
        total_savings_potential = cost_savings_potential + failure_reduction_potential
        savings_percent = (total_savings_potential / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'time_reduction_potential': round(time_reduction_potential, 2),
            'cost_savings_potential': round(cost_savings_potential, 2),
            'failure_reduction_potential': round(failure_reduction_potential, 2),
            'total_savings_potential': round(total_savings_potential, 2),
            'savings_percent': round(savings_percent, 1),
            'optimization_score': round(min(100, savings_percent * 2), 1)
        }
    
    def _analyze_roi(self, recent: List[Dict], cost_per_test: float, hourly_rate: float) -> Dict:
        """Analyze ROI"""
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        total_time = sum(r.get('execution_time', 0) for r in recent)
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        
        # Costs
        total_cost = (total_tests * cost_per_test) + ((total_time / 3600) * hourly_rate)
        
        # Value (estimated value per failure caught)
        value_per_failure = 100  # $100 value per failure caught
        total_value = total_failures * value_per_failure
        
        # ROI
        roi = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_cost': round(total_cost, 2),
            'total_value': round(total_value, 2),
            'net_value': round(total_value - total_cost, 2),
            'roi_percent': round(roi, 1),
            'value_per_failure': value_per_failure,
            'failures_caught': total_failures,
            'roi_status': 'excellent' if roi > 200 else 'good' if roi > 100 else 'acceptable' if roi > 0 else 'negative'
        }
    
    def _generate_cost_recommendations(self, analysis: Dict) -> List[str]:
        """Generate cost recommendations"""
        recommendations = []
        
        total_costs = analysis['total_costs']
        if total_costs.get('total_cost', 0) > 1000:
            recommendations.append(f"Total cost is ${total_costs['total_cost']:.2f} - optimize to reduce costs")
        
        breakdown = analysis['cost_breakdown']
        if breakdown.get('cost_per_failed_test', 0) > breakdown.get('cost_per_successful_test', 0) * 2:
            recommendations.append("Failed tests cost significantly more - improve test reliability")
        
        efficiency = analysis.get('cost_efficiency', {})
        if efficiency.get('efficiency_score', 0) < 70:
            recommendations.append(f"Improve cost efficiency from {efficiency.get('efficiency_score', 0):.1f} to 80+")
        
        trends = analysis.get('cost_trends', {})
        if trends.get('trend') == 'concerning':
            recommendations.append(f"Cost trend is concerning ({trends['percent_change']:+.1f}%) - take action to reduce costs")
        
        optimization = analysis.get('cost_optimization', {})
        if optimization.get('savings_potential', 0) > 100:
            recommendations.append(f"Potential cost savings: ${optimization['savings_potential']:.2f} ({optimization['savings_percent']:.1f}%)")
        
        roi = analysis.get('roi_analysis', {})
        if roi.get('roi_status') == 'negative':
            recommendations.append(f"🚨 Negative ROI ({roi['roi_percent']:.1f}%) - improve test value or reduce costs")
        
        if not recommendations:
            recommendations.append("✅ Costs are optimized - maintain current practices")
        
        return recommendations
    
    def generate_cost_report(self, analysis: Dict) -> str:
        """Generate cost report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED COST ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Cost per Test: ${analysis['cost_parameters']['cost_per_test']:.2f}")
        lines.append(f"Hourly Rate: ${analysis['cost_parameters']['hourly_rate']:.2f}")
        lines.append("")
        
        lines.append("💰 TOTAL COSTS")
        lines.append("-" * 80)
        costs = analysis['total_costs']
        lines.append(f"Test Execution Cost: ${costs['test_execution_cost']:.2f}")
        lines.append(f"Time Cost: ${costs['time_cost']:.2f}")
        lines.append(f"Total Cost: ${costs['total_cost']:.2f}")
        lines.append(f"Total Tests: {costs['total_tests']:,}")
        lines.append(f"Total Time: {costs['total_time_hours']:.2f} hours")
        lines.append("")
        
        lines.append("📊 COST BREAKDOWN")
        lines.append("-" * 80)
        breakdown = analysis['cost_breakdown']
        lines.append(f"Successful Tests Cost: ${breakdown['successful_tests_cost']:.2f}")
        lines.append(f"Failed Tests Cost: ${breakdown['failed_tests_cost']:.2f}")
        lines.append(f"Time Cost: ${breakdown['time_cost']:.2f}")
        lines.append(f"Cost per Successful Test: ${breakdown['cost_per_successful_test']:.4f}")
        lines.append(f"Cost per Failed Test: ${breakdown['cost_per_failed_test']:.4f}")
        lines.append("")
        
        lines.append("⚡ COST EFFICIENCY")
        lines.append("-" * 80)
        efficiency = analysis['cost_efficiency']
        lines.append(f"Tests per Dollar: {efficiency['tests_per_dollar']:.2f}")
        lines.append(f"Tests per Hour: {efficiency['tests_per_hour']:.2f}")
        lines.append(f"Cost per Test: ${efficiency['cost_per_test']:.4f}")
        lines.append(f"Efficiency Score: {efficiency['efficiency_score']}/100")
        lines.append("")
        
        if analysis.get('cost_trends'):
            trends = analysis['cost_trends']
            lines.append("📈 COST TRENDS")
            lines.append("-" * 80)
            trend_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends['direction'], '➡️')
            lines.append(f"{emoji} Direction: {trends['direction'].title()}")
            lines.append(f"Cost Change: ${trends['cost_change']:+.2f} ({trends['percent_change']:+.2f}%)")
            lines.append(f"Trend: {trends['trend'].upper()}")
            lines.append("")
        
        if analysis.get('cost_optimization'):
            opt = analysis['cost_optimization']
            lines.append("🎯 COST OPTIMIZATION")
            lines.append("-" * 80)
            lines.append(f"Time Reduction Potential: {opt['time_reduction_potential']:.2f}s")
            lines.append(f"Cost Savings Potential: ${opt['cost_savings_potential']:.2f}")
            lines.append(f"Failure Reduction Potential: ${opt['failure_reduction_potential']:.2f}")
            lines.append(f"Total Savings Potential: ${opt['total_savings_potential']:.2f} ({opt['savings_percent']:.1f}%)")
            lines.append(f"Optimization Score: {opt['optimization_score']}/100")
            lines.append("")
        
        if analysis.get('roi_analysis'):
            roi = analysis['roi_analysis']
            status_emoji = {'excellent': '🟢', 'good': '🟡', 'acceptable': '🟠', 'negative': '🔴'}
            emoji = status_emoji.get(roi['roi_status'], '⚪')
            lines.append(f"{emoji} ROI ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Total Cost: ${roi['total_cost']:.2f}")
            lines.append(f"Total Value: ${roi['total_value']:.2f}")
            lines.append(f"Net Value: ${roi['net_value']:.2f}")
            lines.append(f"ROI: {roi['roi_percent']:.1f}%")
            lines.append(f"ROI Status: {roi['roi_status'].upper()}")
            lines.append("")
        
        if analysis['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in analysis['recommendations']:
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
    
    analyzer = EnhancedCostAnalyzer(project_root)
    analysis = analyzer.analyze_costs(lookback_days=30, cost_per_test=0.10, hourly_rate=50.0)
    
    report = analyzer.generate_cost_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_cost_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced cost analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






