"""
Performance Regression Detector
Detects performance regressions in test execution
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from statistics import mean, stdev

class PerformanceRegressionDetector:
    """Detect performance regressions from historical data"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
        self.regression_threshold = 1.2  # 20% slower is considered regression
    
    def detect_regressions(self, lookback: int = 10) -> Dict:
        """Detect performance regressions"""
        # Load history
        history = self._load_history()
        
        if len(history) < lookback:
            return {
                'error': f'Need at least {lookback} test runs, found {len(history)}',
                'regressions': []
            }
        
        # Get recent runs
        recent_runs = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:lookback]
        recent_runs.reverse()  # Oldest first
        
        # Calculate baseline (first half)
        baseline_runs = recent_runs[:lookback // 2]
        current_runs = recent_runs[lookback // 2:]
        
        baseline_time = mean([r.get('execution_time', 0) for r in baseline_runs])
        current_time = mean([r.get('execution_time', 0) for r in current_runs])
        
        # Check for overall regression
        overall_regression = None
        if current_time > baseline_time * self.regression_threshold:
            regression_percent = ((current_time - baseline_time) / baseline_time) * 100
            overall_regression = {
                'type': 'overall',
                'baseline_time': baseline_time,
                'current_time': current_time,
                'regression_percent': regression_percent,
                'severity': 'high' if regression_percent > 50 else 'medium' if regression_percent > 20 else 'low'
            }
        
        # Analyze per-test performance (if available)
        test_regressions = self._analyze_test_performance(baseline_runs, current_runs)
        
        return {
            'overall_regression': overall_regression,
            'test_regressions': test_regressions,
            'baseline_period': {
                'runs': len(baseline_runs),
                'avg_time': baseline_time
            },
            'current_period': {
                'runs': len(current_runs),
                'avg_time': current_time
            }
        }
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _analyze_test_performance(self, baseline_runs: List[Dict], current_runs: List[Dict]) -> List[Dict]:
        """Analyze individual test performance"""
        # This would require detailed per-test timing data
        # For now, return empty list
        return []
    
    def generate_regression_report(self, analysis: Dict) -> str:
        """Generate performance regression report"""
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE REGRESSION ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        # Baseline vs Current
        lines.append("📊 PERFORMANCE COMPARISON")
        lines.append("-" * 80)
        baseline = analysis['baseline_period']
        current = analysis['current_period']
        
        lines.append(f"Baseline Period: {baseline['runs']} runs, avg {baseline['avg_time']:.2f}s")
        lines.append(f"Current Period:  {current['runs']} runs, avg {current['avg_time']:.2f}s")
        
        change = current['avg_time'] - baseline['avg_time']
        change_percent = (change / baseline['avg_time'] * 100) if baseline['avg_time'] > 0 else 0
        
        if change > 0:
            lines.append(f"Change: +{change:.2f}s ({change_percent:+.1f}%) - SLOWER")
        elif change < 0:
            lines.append(f"Change: {change:.2f}s ({change_percent:+.1f}%) - FASTER")
        else:
            lines.append("Change: No change")
        lines.append("")
        
        # Overall regression
        if analysis['overall_regression']:
            reg = analysis['overall_regression']
            lines.append(f"🔴 OVERALL PERFORMANCE REGRESSION DETECTED")
            lines.append("-" * 80)
            lines.append(f"Severity: {reg['severity'].upper()}")
            lines.append(f"Regression: {reg['regression_percent']:.1f}% slower")
            lines.append(f"Baseline: {reg['baseline_time']:.2f}s")
            lines.append(f"Current:  {reg['current_time']:.2f}s")
            lines.append("")
            
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            lines.append("1. Review recent code changes that might affect performance")
            lines.append("2. Check for new tests that are particularly slow")
            lines.append("3. Verify system resources (CPU, memory, disk)")
            lines.append("4. Consider optimizing slow tests")
            lines.append("5. Check for test dependencies that might cause sequential execution")
        else:
            lines.append("✅ NO OVERALL REGRESSION DETECTED")
            lines.append("Test execution time is within acceptable range.")
            lines.append("")
        
        # Test-specific regressions
        if analysis['test_regressions']:
            lines.append(f"🔴 TEST-SPECIFIC REGRESSIONS ({len(analysis['test_regressions'])})")
            lines.append("-" * 80)
            for reg in analysis['test_regressions'][:10]:  # Top 10
                lines.append(f"  • {reg['test']}: {reg['regression_percent']:.1f}% slower")
            lines.append("")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    detector = PerformanceRegressionDetector(project_root)
    analysis = detector.detect_regressions(lookback=10)
    
    report = detector.generate_regression_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "performance_regression_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Performance regression report saved to: {report_file}")

if __name__ == "__main__":
    main()







