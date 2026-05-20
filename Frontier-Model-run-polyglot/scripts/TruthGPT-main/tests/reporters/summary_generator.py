"""
Summary Generator
Generate comprehensive summaries of test results
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class SummaryGenerator:
    """Generate test result summaries"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def generate_summary(
        self,
        period_days: int = 7,
        include_trends: bool = True
    ) -> Dict:
        """Generate comprehensive summary"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=period_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate summary metrics
        total_runs = len(recent)
        total_tests = sum(r.get('total_tests', 0) for r in recent)
        avg_success_rate = mean([r.get('success_rate', 0) for r in recent])
        avg_execution_time = mean([r.get('execution_time', 0) for r in recent])
        total_failures = sum(r.get('failures', 0) + r.get('errors', 0) for r in recent)
        
        summary = {
            'period': f'Last {period_days} days',
            'total_runs': total_runs,
            'total_tests': total_tests,
            'average_success_rate': round(avg_success_rate, 1),
            'average_execution_time': round(avg_execution_time, 2),
            'total_failures': total_failures,
            'failure_rate': round((total_failures / total_tests * 100) if total_tests > 0 else 0, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add trends if requested
        if include_trends and len(recent) >= 2:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            
            first_avg = mean([r.get('success_rate', 0) for r in first_half])
            second_avg = mean([r.get('success_rate', 0) for r in second_half])
            
            summary['trend'] = {
                'success_rate_change': round(second_avg - first_avg, 1),
                'direction': 'improving' if second_avg > first_avg else 'declining' if second_avg < first_avg else 'stable'
            }
        
        return summary
    
    def generate_summary_report(self, summary: Dict) -> str:
        """Generate summary report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULTS SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in summary:
            lines.append(f"❌ {summary['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {summary['period']}")
        lines.append(f"Generated: {summary['timestamp'][:19]}")
        lines.append("")
        
        lines.append("📊 KEY METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Runs:           {summary['total_runs']}")
        lines.append(f"Total Tests:          {summary['total_tests']}")
        lines.append(f"Average Success Rate: {summary['average_success_rate']:.1f}%")
        lines.append(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
        lines.append(f"Total Failures:       {summary['total_failures']}")
        lines.append(f"Failure Rate:         {summary['failure_rate']:.2f}%")
        lines.append("")
        
        if 'trend' in summary:
            trend = summary['trend']
            trend_emoji = {
                'improving': '📈',
                'declining': '📉',
                'stable': '➡️'
            }.get(trend['direction'], '➡️')
            
            lines.append("📈 TREND")
            lines.append("-" * 80)
            lines.append(f"{trend_emoji} Direction: {trend['direction'].upper()}")
            lines.append(f"   Change: {trend['success_rate_change']:+.1f}%")
        
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
    
    generator = SummaryGenerator(project_root)
    summary = generator.generate_summary(period_days=7, include_trends=True)
    
    report = generator.generate_summary_report(summary)
    print(report)
    
    # Save summary
    summary_file = project_root / "test_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()







