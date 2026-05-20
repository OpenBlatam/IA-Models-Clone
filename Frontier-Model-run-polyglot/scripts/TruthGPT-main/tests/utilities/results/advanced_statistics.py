"""
Advanced Statistics Calculator
Calculate advanced statistics for test results
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev, mode
from collections import Counter

class AdvancedStatistics:
    """Calculate advanced statistics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def calculate_statistics(self, lookback_days: int = 30) -> Dict:
        """Calculate advanced statistics"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Calculate statistics
        stats = {
            'success_rate': {
                'mean': mean(success_rates) if success_rates else 0,
                'median': median(success_rates) if len(success_rates) > 1 else (success_rates[0] if success_rates else 0),
                'stdev': stdev(success_rates) if len(success_rates) > 1 else 0,
                'min': min(success_rates) if success_rates else 0,
                'max': max(success_rates) if success_rates else 0
            },
            'execution_time': {
                'mean': mean(execution_times) if execution_times else 0,
                'median': median(execution_times) if len(execution_times) > 1 else (execution_times[0] if execution_times else 0),
                'stdev': stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times) if execution_times else 0,
                'max': max(execution_times) if execution_times else 0
            },
            'total_tests': {
                'mean': mean(total_tests) if total_tests else 0,
                'median': median(total_tests) if len(total_tests) > 1 else (total_tests[0] if total_tests else 0),
                'stdev': stdev(total_tests) if len(total_tests) > 1 else 0,
                'min': min(total_tests) if total_tests else 0,
                'max': max(total_tests) if total_tests else 0
            },
            'failures': {
                'mean': mean(failures) if failures else 0,
                'median': median(failures) if len(failures) > 1 else (failures[0] if failures else 0),
                'stdev': stdev(failures) if len(failures) > 1 else 0,
                'min': min(failures) if failures else 0,
                'max': max(failures) if failures else 0
            },
            'runs_analyzed': len(recent),
            'period_days': lookback_days
        }
        
        return stats
    
    def generate_statistics_report(self, stats: Dict) -> str:
        """Generate statistics report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED STATISTICS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in stats:
            lines.append(f"❌ {stats['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: Last {stats['period_days']} days")
        lines.append(f"Runs Analyzed: {stats['runs_analyzed']}")
        lines.append("")
        
        def format_stat(metric_name, metric_data):
            lines.append(f"📊 {metric_name.upper().replace('_', ' ')}")
            lines.append("-" * 80)
            lines.append(f"Mean:    {metric_data['mean']:.2f}")
            lines.append(f"Median:  {metric_data['median']:.2f}")
            lines.append(f"Std Dev: {metric_data['stdev']:.2f}")
            lines.append(f"Min:     {metric_data['min']:.2f}")
            lines.append(f"Max:     {metric_data['max']:.2f}")
            lines.append("")
        
        format_stat("Success Rate", stats['success_rate'])
        format_stat("Execution Time", stats['execution_time'])
        format_stat("Total Tests", stats['total_tests'])
        format_stat("Failures", stats['failures'])
        
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
    
    stats_calc = AdvancedStatistics(project_root)
    stats = stats_calc.calculate_statistics(lookback_days=30)
    
    report = stats_calc.generate_statistics_report(stats)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_statistics_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Statistics report saved to: {report_file}")

if __name__ == "__main__":
    main()







