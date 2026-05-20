"""
Resource Efficiency Analyzer
Analyze and optimize resource efficiency
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class ResourceEfficiencyAnalyzer:
    """Resource efficiency analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_efficiency(self, lookback_days: int = 30) -> Dict:
        """Analyze resource efficiency"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract resource metrics
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        # Calculate efficiency metrics
        efficiency_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'time_efficiency': self._analyze_time_efficiency(execution_times, total_tests),
            'test_efficiency': self._analyze_test_efficiency(total_tests, execution_times, success_rates),
            'resource_utilization': self._analyze_resource_utilization(execution_times, total_tests),
            'efficiency_trends': self._analyze_efficiency_trends(recent),
            'optimization_opportunities': self._identify_optimization_opportunities(recent),
            'recommendations': []
        }
        
        # Calculate overall efficiency score
        efficiency_analysis['overall_efficiency_score'] = self._calculate_overall_efficiency_score(efficiency_analysis)
        
        # Generate recommendations
        efficiency_analysis['recommendations'] = self._generate_efficiency_recommendations(efficiency_analysis)
        
        return efficiency_analysis
    
    def _analyze_time_efficiency(self, execution_times: List[float], total_tests: List[int]) -> Dict:
        """Analyze time efficiency"""
        if not execution_times or not total_tests:
            return {}
        
        # Calculate time per test
        time_per_test = []
        for time, tests in zip(execution_times, total_tests):
            if tests > 0:
                time_per_test.append(time / tests)
        
        if not time_per_test:
            return {}
        
        avg_time_per_test = mean(time_per_test)
        total_time = sum(execution_times)
        total_tests_sum = sum(total_tests)
        avg_time_per_test_overall = total_time / total_tests_sum if total_tests_sum > 0 else 0
        
        return {
            'avg_time_per_test': round(avg_time_per_test, 3),
            'overall_time_per_test': round(avg_time_per_test_overall, 3),
            'total_execution_time': round(total_time, 2),
            'total_tests': total_tests_sum,
            'efficiency_score': round(max(0, 100 - (avg_time_per_test_overall * 10)), 1)  # Normalize
        }
    
    def _analyze_test_efficiency(self, total_tests: List[int], execution_times: List[float], success_rates: List[float]) -> Dict:
        """Analyze test efficiency"""
        if not total_tests or not execution_times or not success_rates:
            return {}
        
        # Calculate tests per second
        tests_per_second = []
        for tests, time in zip(total_tests, execution_times):
            if time > 0:
                tests_per_second.append(tests / time)
        
        if not tests_per_second:
            return {}
        
        # Calculate success-weighted efficiency
        weighted_efficiency = []
        for tests, time, success in zip(total_tests, execution_times, success_rates):
            if time > 0:
                weighted_efficiency.append((tests / time) * (success / 100))
        
        return {
            'avg_tests_per_second': round(mean(tests_per_second), 2),
            'max_tests_per_second': round(max(tests_per_second), 2),
            'min_tests_per_second': round(min(tests_per_second), 2),
            'weighted_efficiency': round(mean(weighted_efficiency), 2) if weighted_efficiency else 0,
            'efficiency_score': round(min(100, mean(tests_per_second) * 10), 1)  # Normalize
        }
    
    def _analyze_resource_utilization(self, execution_times: List[float], total_tests: List[int]) -> Dict:
        """Analyze resource utilization"""
        if not execution_times or not total_tests:
            return {}
        
        # Identify underutilized and overutilized runs
        avg_time = mean(execution_times)
        avg_tests = mean(total_tests)
        
        utilization_scores = []
        for time, tests in zip(execution_times, total_tests):
            if avg_time > 0 and avg_tests > 0:
                # Normalize and calculate utilization
                time_ratio = time / avg_time
                tests_ratio = tests / avg_tests
                utilization = (tests_ratio / time_ratio) * 100 if time_ratio > 0 else 0
                utilization_scores.append(utilization)
        
        if not utilization_scores:
            return {}
        
        return {
            'avg_utilization': round(mean(utilization_scores), 1),
            'utilization_std': round(stdev(utilization_scores), 1) if len(utilization_scores) > 1 else 0,
            'min_utilization': round(min(utilization_scores), 1),
            'max_utilization': round(max(utilization_scores), 1),
            'consistency': round(100 - (stdev(utilization_scores) * 2) if len(utilization_scores) > 1 else 100, 1)
        }
    
    def _analyze_efficiency_trends(self, recent: List[Dict]) -> Dict:
        """Analyze efficiency trends"""
        if len(recent) < 4:
            return {}
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        # Calculate efficiency over time
        efficiencies = []
        for time, tests in zip(execution_times, total_tests):
            if time > 0:
                efficiencies.append(tests / time)
        
        if len(efficiencies) < 4:
            return {}
        
        first_half = efficiencies[:len(efficiencies)//2]
        second_half = efficiencies[len(efficiencies)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        trend = second_avg - first_avg
        percent_change = (trend / first_avg * 100) if first_avg > 0 else 0
        
        return {
            'trend': round(trend, 3),
            'percent_change': round(percent_change, 2),
            'direction': 'improving' if trend > 0 else 'degrading' if trend < 0 else 'stable',
            'first_half_avg': round(first_avg, 3),
            'second_half_avg': round(second_avg, 3)
        }
    
    def _identify_optimization_opportunities(self, recent: List[Dict]) -> List[Dict]:
        """Identify optimization opportunities"""
        opportunities = []
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not execution_times or not total_tests:
            return opportunities
        
        avg_time = mean(execution_times)
        avg_tests = mean(total_tests)
        
        # Find inefficient runs
        for i, (time, tests) in enumerate(zip(execution_times, total_tests)):
            if time > avg_time * 1.5 and tests < avg_tests * 0.8:
                opportunities.append({
                    'run_index': i,
                    'type': 'slow_with_few_tests',
                    'execution_time': round(time, 2),
                    'tests_count': tests,
                    'time_per_test': round(time / tests, 3) if tests > 0 else 0,
                    'severity': 'high' if time > avg_time * 2 else 'medium',
                    'recommendation': 'Optimize test execution or increase test coverage'
                })
        
        return opportunities[:5]  # Top 5
    
    def _calculate_overall_efficiency_score(self, analysis: Dict) -> float:
        """Calculate overall efficiency score"""
        time_eff = analysis['time_efficiency']
        test_eff = analysis['test_efficiency']
        resource_util = analysis['resource_utilization']
        
        scores = []
        
        if time_eff.get('efficiency_score'):
            scores.append(time_eff['efficiency_score'])
        
        if test_eff.get('efficiency_score'):
            scores.append(test_eff['efficiency_score'])
        
        if resource_util.get('avg_utilization'):
            scores.append(min(100, resource_util['avg_utilization']))
        
        if not scores:
            return 0.0
        
        return round(mean(scores), 1)
    
    def _generate_efficiency_recommendations(self, analysis: Dict) -> List[str]:
        """Generate efficiency recommendations"""
        recommendations = []
        
        time_eff = analysis['time_efficiency']
        if time_eff.get('avg_time_per_test', 0) > 0.5:
            recommendations.append(f"Reduce average time per test from {time_eff['avg_time_per_test']:.3f}s to <0.3s")
        
        test_eff = analysis['test_efficiency']
        if test_eff.get('avg_tests_per_second', 0) < 5:
            recommendations.append(f"Increase throughput from {test_eff['avg_tests_per_second']:.1f} tests/s to 10+ tests/s")
        
        if analysis['optimization_opportunities']:
            recommendations.append(f"Found {len(analysis['optimization_opportunities'])} optimization opportunities - review slow runs")
        
        trends = analysis.get('efficiency_trends', {})
        if trends.get('direction') == 'degrading':
            recommendations.append("Efficiency is degrading - investigate recent changes")
        
        if not recommendations:
            recommendations.append("Resource efficiency is optimal - maintain current practices")
        
        return recommendations
    
    def generate_efficiency_report(self, analysis: Dict) -> str:
        """Generate efficiency report"""
        lines = []
        lines.append("=" * 80)
        lines.append("RESOURCE EFFICIENCY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['overall_efficiency_score'] >= 80 else "🟡" if analysis['overall_efficiency_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Efficiency Score: {analysis['overall_efficiency_score']}/100")
        lines.append("")
        
        lines.append("⏱️ TIME EFFICIENCY")
        lines.append("-" * 80)
        te = analysis['time_efficiency']
        lines.append(f"Average Time per Test: {te['avg_time_per_test']}s")
        lines.append(f"Overall Time per Test: {te['overall_time_per_test']}s")
        lines.append(f"Total Execution Time: {te['total_execution_time']}s")
        lines.append(f"Total Tests: {te['total_tests']:,}")
        lines.append(f"Efficiency Score: {te['efficiency_score']}/100")
        lines.append("")
        
        lines.append("⚡ TEST EFFICIENCY")
        lines.append("-" * 80)
        test_eff = analysis['test_efficiency']
        lines.append(f"Average Tests/Second: {test_eff['avg_tests_per_second']}")
        lines.append(f"Max Tests/Second: {test_eff['max_tests_per_second']}")
        lines.append(f"Min Tests/Second: {test_eff['min_tests_per_second']}")
        lines.append(f"Weighted Efficiency: {test_eff['weighted_efficiency']}")
        lines.append(f"Efficiency Score: {test_eff['efficiency_score']}/100")
        lines.append("")
        
        lines.append("📊 RESOURCE UTILIZATION")
        lines.append("-" * 80)
        util = analysis['resource_utilization']
        lines.append(f"Average Utilization: {util['avg_utilization']}%")
        lines.append(f"Utilization Std Dev: {util['utilization_std']}%")
        lines.append(f"Range: {util['min_utilization']}% - {util['max_utilization']}%")
        lines.append(f"Consistency: {util['consistency']}%")
        lines.append("")
        
        if analysis.get('efficiency_trends'):
            trends = analysis['efficiency_trends']
            trend_emoji = {'improving': '📈', 'degrading': '📉', 'stable': '➡️'}
            emoji = trend_emoji.get(trends['direction'], '➡️')
            lines.append(f"{emoji} EFFICIENCY TRENDS")
            lines.append("-" * 80)
            lines.append(f"Direction: {trends['direction'].title()}")
            lines.append(f"Trend: {trends['trend']:+.3f}")
            lines.append(f"Percent Change: {trends['percent_change']:+.2f}%")
            lines.append("")
        
        if analysis['optimization_opportunities']:
            lines.append("🔧 OPTIMIZATION OPPORTUNITIES")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for opp in analysis['optimization_opportunities']:
                emoji = severity_emoji.get(opp['severity'], '⚪')
                lines.append(f"{emoji} Run #{opp['run_index']} - {opp['type'].replace('_', ' ').title()}")
                lines.append(f"   Execution Time: {opp['execution_time']}s")
                lines.append(f"   Tests: {opp['tests_count']}")
                lines.append(f"   Time per Test: {opp['time_per_test']}s")
                lines.append(f"   {opp['recommendation']}")
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
    
    analyzer = ResourceEfficiencyAnalyzer(project_root)
    analysis = analyzer.analyze_efficiency(lookback_days=30)
    
    report = analyzer.generate_efficiency_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "resource_efficiency_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Resource efficiency analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






