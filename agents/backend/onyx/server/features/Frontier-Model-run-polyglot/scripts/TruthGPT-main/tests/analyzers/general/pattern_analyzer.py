"""
Pattern Analyzer
Analyze patterns in test results
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

class PatternAnalyzer:
    """Analyze patterns in test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.history_file = project_root / "test_history.json"
    
    def analyze_patterns(self, lookback_days: int = 30) -> Dict:
        """Analyze patterns in test results"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze time patterns
        time_patterns = defaultdict(int)
        for run in recent:
            try:
                timestamp = run.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    time_patterns[hour] += 1
            except Exception:
                pass
        
        # Analyze failure patterns
        failure_patterns = defaultdict(int)
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                test_details = data.get('test_details', {})
                for failure in test_details.get('failures', []):
                    test_name = str(failure.get('test', ''))
                    # Extract pattern
                    pattern = self._extract_test_pattern(test_name)
                    failure_patterns[pattern] += 1
            except Exception:
                continue
        
        # Analyze error type patterns
        error_patterns = defaultdict(int)
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                test_details = data.get('test_details', {})
                for failure in test_details.get('failures', []) + test_details.get('errors', []):
                    traceback = str(failure.get('traceback', ''))
                    error_type = self._extract_error_type(traceback)
                    error_patterns[error_type] += 1
            except Exception:
                continue
        
        return {
            'time_patterns': dict(sorted(time_patterns.items())),
            'failure_patterns': dict(sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
            'error_patterns': dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
            'period_days': lookback_days
        }
    
    def _extract_test_pattern(self, test_name: str) -> str:
        """Extract pattern from test name"""
        # Extract category (e.g., test_inference_*)
        if 'test_' in test_name:
            parts = test_name.split('test_')
            if len(parts) > 1:
                category = parts[1].split('_')[0]
                return f"test_{category}_*"
        return "other"
    
    def _extract_error_type(self, traceback: str) -> str:
        """Extract error type from traceback"""
        common_errors = [
            'AssertionError',
            'ValueError',
            'TypeError',
            'AttributeError',
            'KeyError',
            'ImportError',
            'RuntimeError',
            'TimeoutError'
        ]
        
        for error in common_errors:
            if error in traceback:
                return error
        
        return 'OtherError'
    
    def generate_pattern_report(self, patterns: Dict) -> str:
        """Generate pattern analysis report"""
        lines = []
        lines.append("=" * 80)
        lines.append("PATTERN ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in patterns:
            lines.append(f"❌ {patterns['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: Last {patterns['period_days']} days")
        lines.append("")
        
        lines.append("🕐 TIME PATTERNS")
        lines.append("-" * 80)
        for hour, count in sorted(patterns['time_patterns'].items()):
            lines.append(f"Hour {hour:02d}:00 - {count} runs")
        lines.append("")
        
        lines.append("🔴 FAILURE PATTERNS")
        lines.append("-" * 80)
        for pattern, count in list(patterns['failure_patterns'].items())[:10]:
            lines.append(f"{pattern}: {count} failures")
        lines.append("")
        
        lines.append("⚠️  ERROR TYPE PATTERNS")
        lines.append("-" * 80)
        for error_type, count in list(patterns['error_patterns'].items())[:10]:
            lines.append(f"{error_type}: {count} occurrences")
        
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
    
    analyzer = PatternAnalyzer(project_root)
    patterns = analyzer.analyze_patterns(lookback_days=30)
    
    report = analyzer.generate_pattern_report(patterns)
    print(report)
    
    # Save report
    report_file = project_root / "pattern_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Pattern analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







