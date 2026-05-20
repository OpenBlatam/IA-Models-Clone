"""
Comprehensive Reporter
Generate comprehensive reports with all metrics
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev

class ComprehensiveReporter:
    """Generate comprehensive reports"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def generate_comprehensive_report(
        self,
        period_days: int = 30,
        include_all_metrics: bool = True
    ) -> Dict:
        """Generate comprehensive report"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=period_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Extract all metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        # Comprehensive metrics
        report = {
            'period': f'Last {period_days} days',
            'total_runs': len(recent),
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': sum(total_tests),
                'total_passed': sum(r.get('passed', 0) for r in recent),
                'total_failures': sum(failures),
                'average_success_rate': round(mean(success_rates), 2) if success_rates else 0,
                'average_execution_time': round(mean(execution_times), 2) if execution_times else 0
            },
            'statistics': {
                'success_rate': {
                    'mean': round(mean(success_rates), 2) if success_rates else 0,
                    'median': round(median(success_rates), 2) if len(success_rates) > 1 else (round(success_rates[0], 2) if success_rates else 0),
                    'stdev': round(stdev(success_rates), 2) if len(success_rates) > 1 else 0,
                    'min': round(min(success_rates), 2) if success_rates else 0,
                    'max': round(max(success_rates), 2) if success_rates else 0
                },
                'execution_time': {
                    'mean': round(mean(execution_times), 2) if execution_times else 0,
                    'median': round(median(execution_times), 2) if len(execution_times) > 1 else (round(execution_times[0], 2) if execution_times else 0),
                    'stdev': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0,
                    'min': round(min(execution_times), 2) if execution_times else 0,
                    'max': round(max(execution_times), 2) if execution_times else 0
                }
            },
            'trends': self._calculate_trends(recent),
            'insights': self._generate_insights(recent)
        }
        
        return report
    
    def _calculate_trends(self, recent: List[Dict]) -> Dict:
        """Calculate trends"""
        if len(recent) < 2:
            return {}
        
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_success = mean([r.get('success_rate', 0) for r in first_half])
        second_success = mean([r.get('success_rate', 0) for r in second_half])
        
        first_time = mean([r.get('execution_time', 0) for r in first_half])
        second_time = mean([r.get('execution_time', 0) for r in second_half])
        
        return {
            'success_rate_trend': round(second_success - first_success, 2),
            'execution_time_trend': round(second_time - first_time, 2),
            'direction': 'improving' if second_success > first_success else 'declining' if second_success < first_success else 'stable'
        }
    
    def _generate_insights(self, recent: List[Dict]) -> List[str]:
        """Generate insights"""
        insights = []
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success >= 95:
            insights.append("Excellent test success rate - maintain current quality")
        elif avg_success >= 90:
            insights.append("Good test success rate - room for minor improvements")
        else:
            insights.append("Test success rate needs improvement - focus on fixing failures")
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time < 60:
            insights.append("Fast test execution - excellent performance")
        elif avg_time < 300:
            insights.append("Acceptable test execution time")
        else:
            insights.append("Slow test execution - consider optimization")
        
        return insights
    
    def generate_report_text(self, report: Dict) -> str:
        """Generate text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE TEST REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in report:
            lines.append(f"❌ {report['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {report['period']}")
        lines.append(f"Total Runs: {report['total_runs']}")
        lines.append(f"Generated: {report['timestamp'][:19]}")
        lines.append("")
        
        lines.append("📊 SUMMARY")
        lines.append("-" * 80)
        summary = report['summary']
        lines.append(f"Total Tests:          {summary['total_tests']}")
        lines.append(f"Total Passed:         {summary['total_passed']}")
        lines.append(f"Total Failures:       {summary['total_failures']}")
        lines.append(f"Average Success Rate: {summary['average_success_rate']:.1f}%")
        lines.append(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
        lines.append("")
        
        lines.append("📈 STATISTICS")
        lines.append("-" * 80)
        stats = report['statistics']
        lines.append("Success Rate:")
        lines.append(f"  Mean: {stats['success_rate']['mean']:.2f}%")
        lines.append(f"  Median: {stats['success_rate']['median']:.2f}%")
        lines.append(f"  Std Dev: {stats['success_rate']['stdev']:.2f}%")
        lines.append("")
        
        if 'trends' in report and report['trends']:
            lines.append("📊 TRENDS")
            lines.append("-" * 80)
            trends = report['trends']
            lines.append(f"Success Rate Trend: {trends['success_rate_trend']:+.2f}%")
            lines.append(f"Direction: {trends['direction'].upper()}")
            lines.append("")
        
        if 'insights' in report:
            lines.append("💡 INSIGHTS")
            lines.append("-" * 80)
            for insight in report['insights']:
                lines.append(f"• {insight}")
        
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
    
    reporter = ComprehensiveReporter(project_root)
    report = reporter.generate_comprehensive_report(period_days=30)
    
    text_report = reporter.generate_report_text(report)
    print(text_report)
    
    # Save report
    report_file = project_root / "comprehensive_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Comprehensive report saved to: {report_file}")

if __name__ == "__main__":
    main()







