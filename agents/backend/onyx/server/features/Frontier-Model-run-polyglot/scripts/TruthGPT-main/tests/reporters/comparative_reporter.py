"""
Comparative Reporter
Generates comparative reports between different time periods or configurations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean

class ComparativeReporter:
    """Generate comparative reports"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def compare_periods(
        self,
        period1_days: int = 7,
        period2_days: int = 7
    ) -> Dict:
        """Compare two time periods"""
        history = self._load_history()
        
        if len(history) < period1_days + period2_days:
            return {'error': 'Insufficient data for comparison'}
        
        # Get recent periods
        cutoff1 = (datetime.now() - timedelta(days=period1_days + period2_days)).isoformat()
        cutoff2 = (datetime.now() - timedelta(days=period2_days)).isoformat()
        
        period1 = [r for r in history if cutoff1 <= r.get('timestamp', '') < cutoff2]
        period2 = [r for r in history if r.get('timestamp', '') >= cutoff2]
        
        if not period1 or not period2:
            return {'error': 'Insufficient data in periods'}
        
        # Calculate metrics for each period
        def calculate_metrics(runs):
            return {
                'runs': len(runs),
                'avg_success_rate': mean([r.get('success_rate', 0) for r in runs]),
                'avg_execution_time': mean([r.get('execution_time', 0) for r in runs]),
                'total_tests': sum([r.get('total_tests', 0) for r in runs]),
                'total_failures': sum([r.get('failed', 0) + r.get('errors', 0) for r in runs])
            }
        
        metrics1 = calculate_metrics(period1)
        metrics2 = calculate_metrics(period2)
        
        # Calculate changes
        changes = {
            'success_rate_change': metrics2['avg_success_rate'] - metrics1['avg_success_rate'],
            'execution_time_change': metrics2['avg_execution_time'] - metrics1['avg_execution_time'],
            'total_tests_change': metrics2['total_tests'] - metrics1['total_tests'],
            'failure_rate_change': (
                (metrics2['total_failures'] / metrics2['total_tests'] * 100) if metrics2['total_tests'] > 0 else 0
            ) - (
                (metrics1['total_failures'] / metrics1['total_tests'] * 100) if metrics1['total_tests'] > 0 else 0
            )
        }
        
        return {
            'period1': {
                'label': f'Last {period1_days + period2_days}-{period2_days} days',
                'metrics': metrics1
            },
            'period2': {
                'label': f'Last {period2_days} days',
                'metrics': metrics2
            },
            'changes': changes
        }
    
    def generate_comparative_report(self, comparison: Dict) -> str:
        """Generate comparative report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPARATIVE TEST REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in comparison:
            lines.append(f"❌ {comparison['error']}")
            return "\n".join(lines)
        
        p1 = comparison['period1']
        p2 = comparison['period2']
        changes = comparison['changes']
        
        lines.append(f"Period 1: {p1['label']}")
        lines.append(f"Period 2: {p2['label']}")
        lines.append("")
        
        lines.append("📊 METRICS COMPARISON")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<30} {'Period 1':<20} {'Period 2':<20} {'Change':<15}")
        lines.append("-" * 80)
        
        # Success rate
        lines.append(f"{'Success Rate':<30} "
                    f"{p1['metrics']['avg_success_rate']:>6.1f}%{'':<13} "
                    f"{p2['metrics']['avg_success_rate']:>6.1f}%{'':<13} "
                    f"{changes['success_rate_change']:>+6.1f}%")
        
        # Execution time
        lines.append(f"{'Execution Time':<30} "
                    f"{p1['metrics']['avg_execution_time']:>6.2f}s{'':<13} "
                    f"{p2['metrics']['avg_execution_time']:>6.2f}s{'':<13} "
                    f"{changes['execution_time_change']:>+6.2f}s")
        
        # Total tests
        lines.append(f"{'Total Tests':<30} "
                    f"{p1['metrics']['total_tests']:>6}{'':<14} "
                    f"{p2['metrics']['total_tests']:>6}{'':<14} "
                    f"{changes['total_tests_change']:>+6}")
        
        # Failure rate
        p1_failure_rate = (p1['metrics']['total_failures'] / p1['metrics']['total_tests'] * 100) if p1['metrics']['total_tests'] > 0 else 0
        p2_failure_rate = (p2['metrics']['total_failures'] / p2['metrics']['total_tests'] * 100) if p2['metrics']['total_tests'] > 0 else 0
        
        lines.append(f"{'Failure Rate':<30} "
                    f"{p1_failure_rate:>6.1f}%{'':<13} "
                    f"{p2_failure_rate:>6.1f}%{'':<13} "
                    f"{changes['failure_rate_change']:>+6.1f}%")
        
        lines.append("")
        
        # Analysis
        lines.append("📈 ANALYSIS")
        lines.append("-" * 80)
        
        if changes['success_rate_change'] > 0:
            lines.append("✅ Success rate improved")
        elif changes['success_rate_change'] < 0:
            lines.append("❌ Success rate declined")
        else:
            lines.append("➡️  Success rate stable")
        
        if changes['execution_time_change'] < 0:
            lines.append("✅ Execution time improved (faster)")
        elif changes['execution_time_change'] > 0:
            lines.append("⚠️  Execution time increased (slower)")
        else:
            lines.append("➡️  Execution time stable")
        
        if changes['failure_rate_change'] < 0:
            lines.append("✅ Failure rate decreased")
        elif changes['failure_rate_change'] > 0:
            lines.append("❌ Failure rate increased")
        else:
            lines.append("➡️  Failure rate stable")
        
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
    
    reporter = ComparativeReporter(project_root)
    comparison = reporter.compare_periods(period1_days=14, period2_days=7)
    
    report = reporter.generate_comparative_report(comparison)
    print(report)
    
    # Save report
    report_file = project_root / "comparative_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Comparative report saved to: {report_file}")

if __name__ == "__main__":
    main()







