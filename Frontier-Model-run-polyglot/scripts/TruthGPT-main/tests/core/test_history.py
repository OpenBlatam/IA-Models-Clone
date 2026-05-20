"""
Test Result History
Tracks test results over time
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

class TestHistory:
    """Tracks test execution history"""
    
    def __init__(self, history_file: str = "test_history.json"):
        self.project_root = Path(__file__).parent.parent
        self.history_file = self.project_root / history_file
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load test history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save test history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def record_test_run(
        self,
        total_tests: int,
        passed: int,
        failed: int,
        errors: int,
        skipped: int,
        execution_time: float,
        test_category: Optional[str] = None
    ):
        """Record a test run"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'execution_time': execution_time,
            'test_category': test_category,
            'success_rate': ((passed / total_tests * 100) if total_tests > 0 else 0)
        }
        
        self.history.append(record)
        self._save_history()
        
        return record
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent test runs"""
        return sorted(
            self.history,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]
    
    def get_statistics(self) -> Dict:
        """Get statistics from history"""
        if not self.history:
            return {
                'total_runs': 0,
                'average_success_rate': 0,
                'average_execution_time': 0,
                'total_tests_run': 0
            }
        
        total_runs = len(self.history)
        success_rates = [r['success_rate'] for r in self.history]
        execution_times = [r['execution_time'] for r in self.history]
        total_tests = sum(r['total_tests'] for r in self.history)
        
        return {
            'total_runs': total_runs,
            'average_success_rate': sum(success_rates) / len(success_rates),
            'min_success_rate': min(success_rates),
            'max_success_rate': max(success_rates),
            'average_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'total_tests_run': total_tests,
            'average_tests_per_run': total_tests / total_runs if total_runs > 0 else 0
        }
    
    def get_trends(self) -> Dict:
        """Get trends over time"""
        if len(self.history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Compare last 2 runs
        recent = sorted(self.history, key=lambda x: x['timestamp'], reverse=True)[:2]
        
        if len(recent) < 2:
            return {'trend': 'insufficient_data'}
        
        latest = recent[0]
        previous = recent[1]
        
        success_rate_change = latest['success_rate'] - previous['success_rate']
        execution_time_change = latest['execution_time'] - previous['execution_time']
        
        return {
            'success_rate_change': success_rate_change,
            'execution_time_change': execution_time_change,
            'trend': 'improving' if success_rate_change > 0 else 'declining' if success_rate_change < 0 else 'stable'
        }
    
    def generate_report(self) -> str:
        """Generate history report"""
        stats = self.get_statistics()
        trends = self.get_trends()
        recent = self.get_recent_runs(5)
        
        lines = []
        lines.append("=" * 80)
        lines.append("TEST HISTORY REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Runs:              {stats['total_runs']}")
        lines.append(f"Average Success Rate:    {stats['average_success_rate']:.1f}%")
        lines.append(f"Success Rate Range:      {stats['min_success_rate']:.1f}% - {stats['max_success_rate']:.1f}%")
        lines.append(f"Average Execution Time:  {stats['average_execution_time']:.2f}s")
        lines.append(f"Execution Time Range:    {stats['min_execution_time']:.2f}s - {stats['max_execution_time']:.2f}s")
        lines.append(f"Total Tests Run:         {stats['total_tests_run']}")
        lines.append(f"Average Tests per Run:    {stats['average_tests_per_run']:.1f}")
        lines.append("")
        
        if trends.get('trend') != 'insufficient_data':
            lines.append("TRENDS")
            lines.append("-" * 80)
            trend_emoji = "📈" if trends['trend'] == 'improving' else "📉" if trends['trend'] == 'declining' else "➡️"
            lines.append(f"Trend:                   {trend_emoji} {trends['trend'].upper()}")
            lines.append(f"Success Rate Change:     {trends['success_rate_change']:+.1f}%")
            lines.append(f"Execution Time Change:   {trends['execution_time_change']:+.2f}s")
            lines.append("")
        
        lines.append("RECENT RUNS")
        lines.append("-" * 80)
        for run in recent:
            timestamp = datetime.fromisoformat(run['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{timestamp}")
            lines.append(f"  Tests: {run['total_tests']} | Passed: {run['passed']} | Failed: {run['failed']} | Errors: {run['errors']} | Skipped: {run['skipped']}")
            lines.append(f"  Success Rate: {run['success_rate']:.1f}% | Time: {run['execution_time']:.2f}s")
            if run.get('test_category'):
                lines.append(f"  Category: {run['test_category']}")
            lines.append("")
        
        return "\n".join(lines)

def main():
    """Main function"""
    history = TestHistory()
    report = history.generate_report()
    print(report)
    
    # Save report to file
    report_file = history.project_root / "test_history_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ History report saved to: {report_file}")

if __name__ == "__main__":
    main()







