"""
Advanced Cost Analyzer
Advanced cost analysis for testing
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class AdvancedCostAnalyzer:
    """Advanced cost analysis"""
    
    DEFAULT_COSTS = {
        'compute_per_hour': 10.0,  # $ per hour
        'developer_per_hour': 50.0,  # $ per hour
        'maintenance_per_test': 0.5,  # $ per test
        'infrastructure_per_month': 100.0  # $ per month
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.costs_config_file = project_root / "costs_config.json"
    
    def analyze_costs(self, lookback_days: int = 30) -> Dict:
        """Analyze testing costs"""
        history = self._load_history()
        costs_config = self._load_costs_config()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate costs
        total_execution_time = sum([r.get('execution_time', 0) for r in recent])
        total_tests = sum([r.get('total_tests', 0) for r in recent])
        total_runs = len(recent)
        
        execution_hours = total_execution_time / 3600
        
        # Cost breakdown
        compute_cost = execution_hours * costs_config['compute_per_hour']
        maintenance_cost = total_tests * costs_config['maintenance_per_test']
        infrastructure_cost = (lookback_days / 30) * costs_config['infrastructure_per_month']
        
        # Estimate developer time (simplified: 10% of execution time for maintenance)
        developer_hours = execution_hours * 0.1
        developer_cost = developer_hours * costs_config['developer_per_hour']
        
        total_cost = compute_cost + maintenance_cost + infrastructure_cost + developer_cost
        
        # Cost per test
        cost_per_test = total_cost / total_tests if total_tests > 0 else 0
        cost_per_run = total_cost / total_runs if total_runs > 0 else 0
        
        # Cost efficiency metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        # Cost per successful test
        total_successful = int(total_tests * (avg_success / 100))
        cost_per_successful_test = total_cost / total_successful if total_successful > 0 else 0
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': total_runs,
            'total_tests': total_tests,
            'total_execution_time_hours': round(execution_hours, 2),
            'cost_breakdown': {
                'compute': round(compute_cost, 2),
                'maintenance': round(maintenance_cost, 2),
                'infrastructure': round(infrastructure_cost, 2),
                'developer': round(developer_cost, 2),
                'total': round(total_cost, 2)
            },
            'cost_metrics': {
                'cost_per_test': round(cost_per_test, 4),
                'cost_per_run': round(cost_per_run, 2),
                'cost_per_successful_test': round(cost_per_successful_test, 4),
                'cost_per_hour': round(total_cost / execution_hours if execution_hours > 0 else 0, 2)
            },
            'efficiency': {
                'tests_per_dollar': round(total_tests / total_cost if total_cost > 0 else 0, 2),
                'successful_tests_per_dollar': round(total_successful / total_cost if total_cost > 0 else 0, 2),
                'cost_efficiency_score': self._calculate_cost_efficiency_score(total_cost, total_tests, avg_success)
            },
            'recommendations': self._generate_cost_recommendations(total_cost, cost_per_test, execution_hours)
        }
    
    def _calculate_cost_efficiency_score(self, total_cost: float, total_tests: int, success_rate: float) -> float:
        """Calculate cost efficiency score"""
        if total_cost == 0 or total_tests == 0:
            return 0
        
        # Normalize: lower cost per test and higher success rate = better
        cost_per_test = total_cost / total_tests
        # Ideal: $0.01 per test with 95% success = 100 points
        ideal_cost = 0.01
        ideal_success = 95
        
        cost_score = max(0, 100 - ((cost_per_test / ideal_cost) * 50))
        success_score = (success_rate / ideal_success) * 50
        
        return round(cost_score + success_score, 1)
    
    def _generate_cost_recommendations(self, total_cost: float, cost_per_test: float, execution_hours: float) -> List[str]:
        """Generate cost recommendations"""
        recommendations = []
        
        if cost_per_test > 0.05:
            recommendations.append(f"Cost per test is ${cost_per_test:.4f} - consider optimization")
            recommendations.append("Enable parallel execution to reduce compute time")
        
        if execution_hours > 100:
            recommendations.append(f"Total execution time is {execution_hours:.1f} hours - optimize slow tests")
        
        if total_cost > 1000:
            recommendations.append(f"Total cost is ${total_cost:.2f} - review infrastructure costs")
        
        if not recommendations:
            recommendations.append("Costs are within acceptable range - maintain current practices")
        
        return recommendations
    
    def generate_cost_report(self, analysis: Dict) -> str:
        """Generate cost report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED COST ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append(f"Total Tests: {analysis['total_tests']:,}")
        lines.append(f"Total Execution Time: {analysis['total_execution_time_hours']:.2f} hours")
        lines.append("")
        
        lines.append("💰 COST BREAKDOWN")
        lines.append("-" * 80)
        breakdown = analysis['cost_breakdown']
        lines.append(f"Compute Cost: ${breakdown['compute']:.2f}")
        lines.append(f"Maintenance Cost: ${breakdown['maintenance']:.2f}")
        lines.append(f"Infrastructure Cost: ${breakdown['infrastructure']:.2f}")
        lines.append(f"Developer Cost: ${breakdown['developer']:.2f}")
        lines.append(f"Total Cost: ${breakdown['total']:.2f}")
        lines.append("")
        
        lines.append("📊 COST METRICS")
        lines.append("-" * 80)
        metrics = analysis['cost_metrics']
        lines.append(f"Cost per Test: ${metrics['cost_per_test']:.4f}")
        lines.append(f"Cost per Run: ${metrics['cost_per_run']:.2f}")
        lines.append(f"Cost per Successful Test: ${metrics['cost_per_successful_test']:.4f}")
        lines.append(f"Cost per Hour: ${metrics['cost_per_hour']:.2f}")
        lines.append("")
        
        lines.append("⚡ COST EFFICIENCY")
        lines.append("-" * 80)
        efficiency = analysis['efficiency']
        lines.append(f"Tests per Dollar: {efficiency['tests_per_dollar']:.2f}")
        lines.append(f"Successful Tests per Dollar: {efficiency['successful_tests_per_dollar']:.2f}")
        lines.append(f"Cost Efficiency Score: {efficiency['cost_efficiency_score']}/100")
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
    
    def _load_costs_config(self) -> Dict:
        """Load costs configuration"""
        if self.costs_config_file.exists():
            try:
                with open(self.costs_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Save default config
        with open(self.costs_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.DEFAULT_COSTS, f, indent=2)
        
        return self.DEFAULT_COSTS

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    analyzer = AdvancedCostAnalyzer(project_root)
    analysis = analyzer.analyze_costs(lookback_days=30)
    
    report = analyzer.generate_cost_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_cost_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced cost analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







