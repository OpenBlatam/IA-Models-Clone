"""
Enhanced Optimization Analyzer
Enhanced optimization analysis with comprehensive recommendations
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, stdev

class EnhancedOptimizationAnalyzer:
    """Enhanced optimization analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_optimization(self, lookback_days: int = 30) -> Dict:
        """Analyze optimization opportunities"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Comprehensive optimization analysis
        optimization_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'performance_optimization': self._analyze_performance_optimization(recent),
            'resource_optimization': self._analyze_resource_optimization(recent),
            'test_optimization': self._analyze_test_optimization(recent),
            'efficiency_optimization': self._analyze_efficiency_optimization(recent),
            'optimization_opportunities': self._identify_optimization_opportunities(recent),
            'optimization_impact': self._estimate_optimization_impact(recent),
            'recommendations': []
        }
        
        # Calculate overall optimization score
        optimization_analysis['overall_optimization_score'] = self._calculate_optimization_score(optimization_analysis)
        
        # Generate recommendations
        optimization_analysis['recommendations'] = self._generate_optimization_recommendations(optimization_analysis)
        
        return optimization_analysis
    
    def _analyze_performance_optimization(self, recent: List[Dict]) -> Dict:
        """Analyze performance optimization"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not execution_times or not total_tests:
            return {}
        
        avg_time = mean(execution_times)
        avg_tests = mean(total_tests)
        
        # Optimization potential
        target_time = 120  # seconds
        time_reduction_potential = max(0, avg_time - target_time)
        time_reduction_percent = (time_reduction_potential / avg_time * 100) if avg_time > 0 else 0
        
        return {
            'current_avg_time': round(avg_time, 2),
            'target_time': target_time,
            'reduction_potential': round(time_reduction_potential, 2),
            'reduction_percent': round(time_reduction_percent, 1),
            'optimization_score': round(min(100, (target_time / avg_time * 100) if avg_time > 0 else 0), 1)
        }
    
    def _analyze_resource_optimization(self, recent: List[Dict]) -> Dict:
        """Analyze resource optimization"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        if not execution_times or not total_tests:
            return {}
        
        # Resource utilization
        total_time = sum(execution_times)
        total_tests_sum = sum(total_tests)
        
        # Efficiency metrics
        tests_per_hour = (total_tests_sum / total_time * 3600) if total_time > 0 else 0
        target_tests_per_hour = 1000
        
        efficiency_gap = max(0, target_tests_per_hour - tests_per_hour)
        efficiency_improvement_potential = (efficiency_gap / target_tests_per_hour * 100) if target_tests_per_hour > 0 else 0
        
        return {
            'current_tests_per_hour': round(tests_per_hour, 2),
            'target_tests_per_hour': target_tests_per_hour,
            'efficiency_gap': round(efficiency_gap, 2),
            'improvement_potential': round(efficiency_improvement_potential, 1),
            'optimization_score': round(min(100, (tests_per_hour / target_tests_per_hour * 100) if target_tests_per_hour > 0 else 0), 1)
        }
    
    def _analyze_test_optimization(self, recent: List[Dict]) -> Dict:
        """Analyze test optimization"""
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not total_tests:
            return {}
        
        total_tests_sum = sum(total_tests)
        total_failures = sum(failures)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        # Optimization potential
        target_failure_rate = 2  # percent
        failure_reduction_potential = max(0, failure_rate - target_failure_rate)
        
        avg_success = mean(success_rates) if success_rates else 0
        success_improvement_potential = max(0, 95 - avg_success)
        
        return {
            'current_failure_rate': round(failure_rate, 2),
            'target_failure_rate': target_failure_rate,
            'failure_reduction_potential': round(failure_reduction_potential, 2),
            'current_success_rate': round(avg_success, 1),
            'target_success_rate': 95,
            'success_improvement_potential': round(success_improvement_potential, 1),
            'optimization_score': round(min(100, (avg_success / 95 * 100) if 95 > 0 else 0), 1)
        }
    
    def _analyze_efficiency_optimization(self, recent: List[Dict]) -> Dict:
        """Analyze efficiency optimization"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not execution_times or not total_tests:
            return {}
        
        # Calculate efficiency
        efficiencies = []
        for time, tests, success in zip(execution_times, total_tests, success_rates if success_rates else [100]*len(execution_times)):
            if time > 0:
                efficiencies.append((tests / time) * (success / 100))
        
        if not efficiencies:
            return {}
        
        avg_efficiency = mean(efficiencies)
        target_efficiency = 10  # normalized target
        
        efficiency_gap = max(0, target_efficiency - avg_efficiency)
        improvement_potential = (efficiency_gap / target_efficiency * 100) if target_efficiency > 0 else 0
        
        return {
            'current_efficiency': round(avg_efficiency, 3),
            'target_efficiency': target_efficiency,
            'efficiency_gap': round(efficiency_gap, 3),
            'improvement_potential': round(improvement_potential, 1),
            'optimization_score': round(min(100, (avg_efficiency / target_efficiency * 100) if target_efficiency > 0 else 0), 1)
        }
    
    def _identify_optimization_opportunities(self, recent: List[Dict]) -> List[Dict]:
        """Identify optimization opportunities"""
        opportunities = []
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not execution_times or not total_tests:
            return opportunities
        
        avg_time = mean(execution_times)
        avg_tests = mean(total_tests)
        avg_success = mean(success_rates) if success_rates else 100
        
        # Slow runs
        for i, (time, tests, success) in enumerate(zip(execution_times, total_tests, success_rates if success_rates else [100]*len(execution_times))):
            if time > avg_time * 1.5:
                opportunities.append({
                    'run_index': i,
                    'type': 'slow_execution',
                    'current_value': round(time, 2),
                    'target_value': round(avg_time * 0.8, 2),
                    'improvement_potential': round(time - avg_time * 0.8, 2),
                    'priority': 'high' if time > avg_time * 2 else 'medium',
                    'impact': 'Reduce execution time by optimizing slow tests'
                })
        
        # Low success runs
        for i, (time, tests, success) in enumerate(zip(execution_times, total_tests, success_rates if success_rates else [100]*len(execution_times))):
            if success < avg_success * 0.9:
                opportunities.append({
                    'run_index': i,
                    'type': 'low_success',
                    'current_value': round(success, 2),
                    'target_value': round(avg_success * 1.05, 2),
                    'improvement_potential': round(avg_success * 1.05 - success, 2),
                    'priority': 'high' if success < 80 else 'medium',
                    'impact': 'Improve test reliability and stability'
                })
        
        return sorted(opportunities, key=lambda x: x['improvement_potential'], reverse=True)[:10]  # Top 10
    
    def _estimate_optimization_impact(self, recent: List[Dict]) -> Dict:
        """Estimate optimization impact"""
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        success_rates = [r.get('success_rate', 0) for r in recent]
        
        if not execution_times or not total_tests:
            return {}
        
        avg_time = mean(execution_times)
        total_time = sum(execution_times)
        avg_success = mean(success_rates) if success_rates else 100
        
        # Estimated impact
        time_savings = total_time * 0.2  # 20% reduction
        success_improvement = max(0, 95 - avg_success)
        
        return {
            'estimated_time_savings': round(time_savings, 2),
            'estimated_time_savings_percent': 20,
            'estimated_success_improvement': round(success_improvement, 1),
            'estimated_efficiency_gain': round(15, 1),  # percent
            'overall_impact': 'high' if time_savings > 1000 or success_improvement > 5 else 'medium' if time_savings > 500 or success_improvement > 2 else 'low'
        }
    
    def _calculate_optimization_score(self, analysis: Dict) -> float:
        """Calculate overall optimization score"""
        scores = []
        
        perf = analysis.get('performance_optimization', {})
        if perf.get('optimization_score'):
            scores.append(perf['optimization_score'])
        
        res = analysis.get('resource_optimization', {})
        if res.get('optimization_score'):
            scores.append(res['optimization_score'])
        
        test = analysis.get('test_optimization', {})
        if test.get('optimization_score'):
            scores.append(test['optimization_score'])
        
        eff = analysis.get('efficiency_optimization', {})
        if eff.get('optimization_score'):
            scores.append(eff['optimization_score'])
        
        if not scores:
            return 0.0
        
        return round(mean(scores), 1)
    
    def _generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        perf = analysis.get('performance_optimization', {})
        if perf.get('reduction_potential', 0) > 30:
            recommendations.append(f"Reduce execution time by {perf['reduction_potential']:.0f}s ({perf['reduction_percent']:.1f}%) - optimize slow tests")
        
        res = analysis.get('resource_optimization', {})
        if res.get('improvement_potential', 0) > 20:
            recommendations.append(f"Increase throughput by {res['improvement_potential']:.1f}% - optimize resource usage")
        
        test = analysis.get('test_optimization', {})
        if test.get('failure_reduction_potential', 0) > 2:
            recommendations.append(f"Reduce failure rate by {test['failure_reduction_potential']:.1f}% - improve test stability")
        
        if test.get('success_improvement_potential', 0) > 5:
            recommendations.append(f"Increase success rate by {test['success_improvement_potential']:.1f}% - fix failing tests")
        
        if analysis['optimization_opportunities']:
            high_priority = sum(1 for o in analysis['optimization_opportunities'] if o['priority'] == 'high')
            if high_priority > 0:
                recommendations.append(f"🚨 {high_priority} high-priority optimization opportunity/opportunities identified")
        
        impact = analysis.get('optimization_impact', {})
        if impact.get('overall_impact') == 'high':
            recommendations.append(f"High optimization impact potential - estimated {impact.get('estimated_time_savings_percent', 0)}% time savings")
        
        if not recommendations:
            recommendations.append("✅ Optimization opportunities are limited - maintain current practices")
        
        return recommendations
    
    def generate_optimization_report(self, analysis: Dict) -> str:
        """Generate optimization report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED OPTIMIZATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        score_emoji = "🟢" if analysis['overall_optimization_score'] >= 80 else "🟡" if analysis['overall_optimization_score'] >= 60 else "🔴"
        lines.append(f"{score_emoji} Overall Optimization Score: {analysis['overall_optimization_score']}/100")
        lines.append("")
        
        lines.append("⚡ PERFORMANCE OPTIMIZATION")
        lines.append("-" * 80)
        perf = analysis['performance_optimization']
        lines.append(f"Current Avg Time: {perf['current_avg_time']}s")
        lines.append(f"Target Time: {perf['target_time']}s")
        lines.append(f"Reduction Potential: {perf['reduction_potential']}s ({perf['reduction_percent']:.1f}%)")
        lines.append(f"Optimization Score: {perf['optimization_score']}/100")
        lines.append("")
        
        lines.append("📊 RESOURCE OPTIMIZATION")
        lines.append("-" * 80)
        res = analysis['resource_optimization']
        lines.append(f"Current Tests/Hour: {res['current_tests_per_hour']:.2f}")
        lines.append(f"Target Tests/Hour: {res['target_tests_per_hour']}")
        lines.append(f"Efficiency Gap: {res['efficiency_gap']:.2f}")
        lines.append(f"Improvement Potential: {res['improvement_potential']:.1f}%")
        lines.append(f"Optimization Score: {res['optimization_score']}/100")
        lines.append("")
        
        lines.append("🧪 TEST OPTIMIZATION")
        lines.append("-" * 80)
        test = analysis['test_optimization']
        lines.append(f"Current Failure Rate: {test['current_failure_rate']}%")
        lines.append(f"Target Failure Rate: {test['target_failure_rate']}%")
        lines.append(f"Failure Reduction Potential: {test['failure_reduction_potential']:.2f}%")
        lines.append(f"Current Success Rate: {test['current_success_rate']}%")
        lines.append(f"Success Improvement Potential: {test['success_improvement_potential']:.1f}%")
        lines.append(f"Optimization Score: {test['optimization_score']}/100")
        lines.append("")
        
        lines.append("📈 EFFICIENCY OPTIMIZATION")
        lines.append("-" * 80)
        eff = analysis['efficiency_optimization']
        lines.append(f"Current Efficiency: {eff['current_efficiency']:.3f}")
        lines.append(f"Target Efficiency: {eff['target_efficiency']}")
        lines.append(f"Efficiency Gap: {eff['efficiency_gap']:.3f}")
        lines.append(f"Improvement Potential: {eff['improvement_potential']:.1f}%")
        lines.append(f"Optimization Score: {eff['optimization_score']}/100")
        lines.append("")
        
        if analysis['optimization_opportunities']:
            lines.append("🎯 OPTIMIZATION OPPORTUNITIES")
            lines.append("-" * 80)
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for opp in analysis['optimization_opportunities'][:5]:  # Top 5
                emoji = priority_emoji.get(opp['priority'], '⚪')
                lines.append(f"{emoji} Run #{opp['run_index']} - {opp['type'].replace('_', ' ').title()}")
                lines.append(f"   Current: {opp['current_value']}")
                lines.append(f"   Target: {opp['target_value']}")
                lines.append(f"   Improvement Potential: {opp['improvement_potential']}")
                lines.append(f"   {opp['impact']}")
            lines.append("")
        
        if analysis.get('optimization_impact'):
            impact = analysis['optimization_impact']
            impact_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            emoji = impact_emoji.get(impact['overall_impact'], '⚪')
            lines.append(f"{emoji} OPTIMIZATION IMPACT")
            lines.append("-" * 80)
            lines.append(f"Estimated Time Savings: {impact['estimated_time_savings']:.2f}s ({impact['estimated_time_savings_percent']}%)")
            lines.append(f"Estimated Success Improvement: {impact['estimated_success_improvement']:.1f}%")
            lines.append(f"Estimated Efficiency Gain: {impact['estimated_efficiency_gain']:.1f}%")
            lines.append(f"Overall Impact: {impact['overall_impact'].upper()}")
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
    
    analyzer = EnhancedOptimizationAnalyzer(project_root)
    analysis = analyzer.analyze_optimization(lookback_days=30)
    
    report = analyzer.generate_optimization_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_optimization_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced optimization analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()






