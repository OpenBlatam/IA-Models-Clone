"""
Performance Regression Detector
Detects performance regressions in test execution
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class PerformanceRegressionDetector:
    """Detect performance regressions in tests"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_performance_history.json"
        self.history = self._load_history()
        self.regression_threshold = 1.5  # 50% slower is regression
        self.alert_threshold = 2.0  # 100% slower is alert
    
    def _load_history(self) -> List[Dict]:
        """Load performance history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save performance history"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
    def record_performance(
        self,
        test_name: str,
        duration: float,
        timestamp: datetime = None
    ):
        """Record test performance"""
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'test_name': test_name,
            'duration': duration,
            'timestamp': timestamp.isoformat()
        }
        
        self.history.append(record)
        
        # Keep last 1000 records
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self._save_history()
    
    def detect_regressions(
        self,
        baseline_days: int = 7,
        comparison_days: int = 1
    ) -> List[Dict]:
        """Detect performance regressions"""
        cutoff_baseline = datetime.now() - timedelta(days=baseline_days + comparison_days)
        cutoff_comparison = datetime.now() - timedelta(days=comparison_days)
        
        # Group by test
        by_test = defaultdict(lambda: {'baseline': [], 'comparison': []})
        
        for record in self.history:
            timestamp = datetime.fromisoformat(record['timestamp'])
            test_name = record['test_name']
            duration = record['duration']
            
            if cutoff_baseline <= timestamp < cutoff_baseline + timedelta(days=baseline_days):
                by_test[test_name]['baseline'].append(duration)
            elif cutoff_comparison <= timestamp:
                by_test[test_name]['comparison'].append(duration)
        
        regressions = []
        
        for test_name, data in by_test.items():
            baseline = data['baseline']
            comparison = data['comparison']
            
            if len(baseline) < 3 or len(comparison) < 1:
                continue
            
            baseline_avg = statistics.mean(baseline)
            comparison_avg = statistics.mean(comparison)
            
            if baseline_avg == 0:
                continue
            
            slowdown_ratio = comparison_avg / baseline_avg
            
            if slowdown_ratio >= self.regression_threshold:
                severity = 'critical' if slowdown_ratio >= self.alert_threshold else 'high'
                
                regressions.append({
                    'test_name': test_name,
                    'baseline_avg': round(baseline_avg, 3),
                    'comparison_avg': round(comparison_avg, 3),
                    'slowdown_ratio': round(slowdown_ratio, 2),
                    'slowdown_percent': round((slowdown_ratio - 1) * 100, 1),
                    'severity': severity,
                    'baseline_samples': len(baseline),
                    'comparison_samples': len(comparison)
                })
        
        return sorted(regressions, key=lambda x: x['slowdown_ratio'], reverse=True)
    
    def detect_improvements(
        self,
        baseline_days: int = 7,
        comparison_days: int = 1
    ) -> List[Dict]:
        """Detect performance improvements"""
        cutoff_baseline = datetime.now() - timedelta(days=baseline_days + comparison_days)
        cutoff_comparison = datetime.now() - timedelta(days=comparison_days)
        
        by_test = defaultdict(lambda: {'baseline': [], 'comparison': []})
        
        for record in self.history:
            timestamp = datetime.fromisoformat(record['timestamp'])
            test_name = record['test_name']
            duration = record['duration']
            
            if cutoff_baseline <= timestamp < cutoff_baseline + timedelta(days=baseline_days):
                by_test[test_name]['baseline'].append(duration)
            elif cutoff_comparison <= timestamp:
                by_test[test_name]['comparison'].append(duration)
        
        improvements = []
        
        for test_name, data in by_test.items():
            baseline = data['baseline']
            comparison = data['comparison']
            
            if len(baseline) < 3 or len(comparison) < 1:
                continue
            
            baseline_avg = statistics.mean(baseline)
            comparison_avg = statistics.mean(comparison)
            
            if baseline_avg == 0:
                continue
            
            speedup_ratio = baseline_avg / comparison_avg
            
            if speedup_ratio >= 1.2:  # 20% faster
                improvements.append({
                    'test_name': test_name,
                    'baseline_avg': round(baseline_avg, 3),
                    'comparison_avg': round(comparison_avg, 3),
                    'speedup_ratio': round(speedup_ratio, 2),
                    'speedup_percent': round((speedup_ratio - 1) * 100, 1),
                    'baseline_samples': len(baseline),
                    'comparison_samples': len(comparison)
                })
        
        return sorted(improvements, key=lambda x: x['speedup_ratio'], reverse=True)
    
    def generate_regression_report(
        self,
        output_file: Path = None
    ) -> str:
        """Generate regression report"""
        regressions = self.detect_regressions()
        improvements = self.detect_improvements()
        
        lines = []
        lines.append("⚡ PERFORMANCE REGRESSION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        if regressions:
            lines.append("🔴 PERFORMANCE REGRESSIONS")
            lines.append("-" * 80)
            for reg in regressions[:20]:
                lines.append(f"\n  [{reg['severity'].upper()}] {reg['test_name']}")
                lines.append(f"    Baseline: {reg['baseline_avg']}s")
                lines.append(f"    Current: {reg['comparison_avg']}s")
                lines.append(f"    Slowdown: {reg['slowdown_percent']}% ({reg['slowdown_ratio']}x)")
            lines.append("")
        
        if improvements:
            lines.append("🟢 PERFORMANCE IMPROVEMENTS")
            lines.append("-" * 80)
            for imp in improvements[:10]:
                lines.append(f"\n  {imp['test_name']}")
                lines.append(f"    Baseline: {imp['baseline_avg']}s")
                lines.append(f"    Current: {imp['comparison_avg']}s")
                lines.append(f"    Speedup: {imp['speedup_percent']}% ({imp['speedup_ratio']}x)")
            lines.append("")
        
        # Summary
        lines.append("📊 SUMMARY")
        lines.append("-" * 80)
        lines.append(f"  Regressions: {len(regressions)}")
        lines.append(f"  Improvements: {len(improvements)}")
        lines.append(f"  Critical Regressions: {sum(1 for r in regressions if r['severity'] == 'critical')}")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to {output_file}")
        
        return report


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Regression Detector')
    parser.add_argument('--detect', action='store_true', help='Detect regressions')
    parser.add_argument('--improvements', action='store_true', help='Detect improvements')
    parser.add_argument('--report', type=str, help='Generate report')
    parser.add_argument('--baseline-days', type=int, default=7, help='Baseline period in days')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    detector = PerformanceRegressionDetector(project_root)
    
    if args.report:
        print("⚡ Generating performance regression report...")
        detector.generate_regression_report(Path(args.report))
    elif args.improvements:
        print("⚡ Detecting performance improvements...")
        improvements = detector.detect_improvements(args.baseline_days)
        print(f"\n🟢 Found {len(improvements)} improvements:")
        for imp in improvements[:10]:
            print(f"  {imp['test_name']}: {imp['speedup_percent']}% faster")
    elif args.detect:
        print("⚡ Detecting performance regressions...")
        regressions = detector.detect_regressions(args.baseline_days)
        print(f"\n🔴 Found {len(regressions)} regressions:")
        for reg in regressions[:10]:
            print(f"  {reg['test_name']}: {reg['slowdown_percent']}% slower ({reg['severity']})")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

