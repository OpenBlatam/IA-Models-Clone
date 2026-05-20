"""
Cost Calculator
Calculate costs associated with test execution
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class CostCalculator:
    """Calculate test execution costs"""
    
    def __init__(
        self,
        project_root: Path,
        hourly_rate: float = 50.0,  # Developer hourly rate
        infrastructure_cost_per_hour: float = 0.10  # Infrastructure cost
    ):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.hourly_rate = hourly_rate
        self.infrastructure_cost_per_hour = infrastructure_cost_per_hour
    
    def calculate_costs(self, lookback_days: int = 30) -> Dict:
        """Calculate costs for test execution"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        total_runs = len(recent)
        total_execution_time = sum(r.get('execution_time', 0) for r in recent)
        avg_execution_time = mean([r.get('execution_time', 0) for r in recent])
        
        # Calculate costs
        total_hours = total_execution_time / 3600  # Convert seconds to hours
        
        # Developer time cost (assuming 10% overhead per run)
        developer_time_per_run = avg_execution_time * 1.1 / 3600  # 10% overhead
        total_developer_hours = developer_time_per_run * total_runs
        developer_cost = total_developer_hours * self.hourly_rate
        
        # Infrastructure cost
        infrastructure_cost = total_hours * self.infrastructure_cost_per_hour
        
        # Failure cost (time to investigate and fix)
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        failure_investigation_hours = total_failures * 0.5  # 30 min per failure
        failure_cost = failure_investigation_hours * self.hourly_rate
        
        # Total cost
        total_cost = developer_cost + infrastructure_cost + failure_cost
        
        # Cost per test
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        cost_per_test = total_cost / total_tests if total_tests > 0 else 0
        
        # Cost per run
        cost_per_run = total_cost / total_runs if total_runs > 0 else 0
        
        return {
            'period_days': lookback_days,
            'total_runs': total_runs,
            'total_tests': total_tests,
            'total_execution_time_hours': round(total_hours, 2),
            'costs': {
                'developer': round(developer_cost, 2),
                'infrastructure': round(infrastructure_cost, 2),
                'failure_investigation': round(failure_cost, 2),
                'total': round(total_cost, 2)
            },
            'cost_per_test': round(cost_per_test, 4),
            'cost_per_run': round(cost_per_run, 2),
            'total_failures': total_failures,
            'failure_rate': round((total_failures / total_tests * 100) if total_tests > 0 else 0, 2)
        }
    
    def generate_cost_report(self, cost_data: Dict) -> str:
        """Generate cost report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST COST ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in cost_data:
            lines.append(f"❌ {cost_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: Last {cost_data['period_days']} days")
        lines.append(f"Total Runs: {cost_data['total_runs']}")
        lines.append(f"Total Tests: {cost_data['total_tests']}")
        lines.append(f"Total Execution Time: {cost_data['total_execution_time_hours']} hours")
        lines.append("")
        
        lines.append("💰 COST BREAKDOWN")
        lines.append("-" * 80)
        costs = cost_data['costs']
        lines.append(f"Developer Time:     ${costs['developer']:,.2f}")
        lines.append(f"Infrastructure:     ${costs['infrastructure']:,.2f}")
        lines.append(f"Failure Investigation: ${costs['failure_investigation']:,.2f}")
        lines.append(f"Total Cost:         ${costs['total']:,.2f}")
        lines.append("")
        
        lines.append("📊 COST METRICS")
        lines.append("-" * 80)
        lines.append(f"Cost per Test:      ${cost_data['cost_per_test']:.4f}")
        lines.append(f"Cost per Run:       ${cost_data['cost_per_run']:.2f}")
        lines.append("")
        
        lines.append("⚠️  FAILURE IMPACT")
        lines.append("-" * 80)
        lines.append(f"Total Failures:     {cost_data['total_failures']}")
        lines.append(f"Failure Rate:       {cost_data['failure_rate']}%")
        lines.append(f"Failure Cost:       ${costs['failure_investigation']:,.2f}")
        lines.append(f"Failure Cost %:     {(costs['failure_investigation'] / costs['total'] * 100) if costs['total'] > 0 else 0:.1f}%")
        lines.append("")
        
        lines.append("💡 COST OPTIMIZATION OPPORTUNITIES")
        lines.append("-" * 80)
        
        if cost_data['failure_rate'] > 5:
            lines.append("• Reduce failure rate to lower investigation costs")
        
        if cost_data['cost_per_test'] > 0.10:
            lines.append("• Optimize test execution to reduce per-test cost")
        
        if costs['failure_investigation'] / costs['total'] > 0.3:
            lines.append("• Focus on test stability to reduce failure investigation time")
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Cost Calculator')
    parser.add_argument('--hourly-rate', type=float, default=50.0, help='Developer hourly rate')
    parser.add_argument('--infra-cost', type=float, default=0.10, help='Infrastructure cost per hour')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    calculator = CostCalculator(project_root, args.hourly_rate, args.infra_cost)
    
    cost_data = calculator.calculate_costs(lookback_days=30)
    report = calculator.generate_cost_report(cost_data)
    
    print(report)
    
    # Save report
    report_file = project_root / "cost_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Cost analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







