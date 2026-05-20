"""
Test Optimization Suggestions
Analyzes test suite and provides optimization suggestions
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import statistics


class TestOptimizationAnalyzer:
    """Analyze tests and provide optimization suggestions"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_result_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load test result history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def analyze_test_suite(self) -> Dict:
        """Analyze entire test suite and provide suggestions"""
        suggestions = []
        
        # Analyze slow tests
        slow_tests = self._analyze_slow_tests()
        if slow_tests:
            suggestions.extend(slow_tests)
        
        # Analyze flaky tests
        flaky_tests = self._analyze_flaky_tests()
        if flaky_tests:
            suggestions.extend(flaky_tests)
        
        # Analyze duplicate tests
        duplicate_tests = self._analyze_duplicates()
        if duplicate_tests:
            suggestions.extend(duplicate_tests)
        
        # Analyze test organization
        org_suggestions = self._analyze_organization()
        if org_suggestions:
            suggestions.extend(org_suggestions)
        
        # Analyze test coverage gaps
        coverage_gaps = self._analyze_coverage_gaps()
        if coverage_gaps:
            suggestions.extend(coverage_gaps)
        
        return {
            'total_suggestions': len(suggestions),
            'by_priority': self._group_by_priority(suggestions),
            'suggestions': suggestions
        }
    
    def _analyze_slow_tests(self) -> List[Dict]:
        """Find slow tests that could be optimized"""
        # Group by test name
        by_test = defaultdict(list)
        for record in self.history:
            test = record.get('test_name', 'unknown')
            duration = record.get('duration', 0)
            if duration > 0:
                by_test[test].append(duration)
        
        suggestions = []
        
        for test, durations in by_test.items():
            if len(durations) < 3:
                continue
            
            avg_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)
            
            # Flag if average > 5 seconds
            if avg_duration > 5.0:
                suggestions.append({
                    'type': 'slow_test',
                    'test_name': test,
                    'priority': 'high' if avg_duration > 10 else 'medium',
                    'current_avg': round(avg_duration, 2),
                    'median': round(median_duration, 2),
                    'suggestion': f"Test takes {avg_duration:.1f}s on average. Consider:",
                    'recommendations': [
                        "Use mocking to avoid slow I/O operations",
                        "Split into smaller, focused tests",
                        "Use fixtures to share expensive setup",
                        "Consider parallel execution"
                    ]
                })
        
        return suggestions
    
    def _analyze_flaky_tests(self) -> List[Dict]:
        """Find flaky tests"""
        by_test = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total': 0})
        
        for record in self.history:
            test = record.get('test_name', 'unknown')
            status = record.get('status', 'unknown')
            
            by_test[test]['total'] += 1
            if status == 'passed':
                by_test[test]['passed'] += 1
            elif status in ('failed', 'error'):
                by_test[test]['failed'] += 1
        
        suggestions = []
        
        for test, stats in by_test.items():
            if stats['total'] < 5:
                continue
            
            failure_rate = stats['failed'] / stats['total']
            success_rate = stats['passed'] / stats['total']
            
            # Flag if inconsistent (both failures and successes)
            if failure_rate > 0.1 and success_rate > 0.1:
                suggestions.append({
                    'type': 'flaky_test',
                    'test_name': test,
                    'priority': 'high' if failure_rate > 0.3 else 'medium',
                    'failure_rate': round(failure_rate * 100, 1),
                    'success_rate': round(success_rate * 100, 1),
                    'total_runs': stats['total'],
                    'suggestion': f"Test is flaky: {failure_rate*100:.1f}% failure rate",
                    'recommendations': [
                        "Fix timing issues (add proper waits/sleeps)",
                        "Remove non-deterministic behavior",
                        "Use fixed seeds for random operations",
                        "Isolate test from external dependencies",
                        "Add retry logic as temporary fix"
                    ]
                })
        
        return suggestions
    
    def _analyze_duplicates(self) -> List[Dict]:
        """Find duplicate or similar tests"""
        # This is a simplified check - in reality would need AST comparison
        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(list(self.project_root.rglob("*_test.py")))
        
        # Group tests by name patterns
        name_patterns = defaultdict(list)
        for test_file in test_files:
            name = test_file.stem
            # Extract base pattern (remove numbers, etc.)
            base = ''.join(c for c in name if c.isalpha() or c == '_')
            name_patterns[base].append(test_file)
        
        suggestions = []
        
        for pattern, files in name_patterns.items():
            if len(files) > 1:
                # Check if they might be duplicates
                file_sizes = [f.stat().st_size for f in files]
                if len(set(file_sizes)) == 1:  # Same size - might be duplicates
                    suggestions.append({
                        'type': 'possible_duplicate',
                        'pattern': pattern,
                        'files': [str(f.relative_to(self.project_root)) for f in files],
                        'priority': 'low',
                        'suggestion': f"Found {len(files)} tests with similar names",
                        'recommendations': [
                            "Review if tests are actually duplicates",
                            "Merge if testing same functionality",
                            "Rename if they test different things"
                        ]
                    })
        
        return suggestions
    
    def _analyze_organization(self) -> List[Dict]:
        """Analyze test organization"""
        suggestions = []
        
        # Check for tests in wrong directories
        test_files = list(self.project_root.rglob("test_*.py"))
        
        # Count tests in root vs subdirectories
        root_tests = [f for f in test_files if f.parent == self.project_root]
        subdir_tests = [f for f in test_files if f.parent != self.project_root]
        
        if len(root_tests) > 10:
            suggestions.append({
                'type': 'organization',
                'priority': 'medium',
                'suggestion': f"{len(root_tests)} tests in root directory",
                'recommendations': [
                    "Organize tests into subdirectories by feature/module",
                    "Use core/unit/ and core/integration/ structure",
                    "Group related tests together"
                ]
            })
        
        return suggestions
    
    def _analyze_coverage_gaps(self) -> List[Dict]:
        """Analyze potential coverage gaps"""
        # This would integrate with coverage tools
        # For now, provide general suggestions
        suggestions = []
        
        # Check if coverage data exists
        coverage_file = self.project_root / ".coverage"
        if not coverage_file.exists():
            suggestions.append({
                'type': 'coverage',
                'priority': 'medium',
                'suggestion': "No coverage data found",
                'recommendations': [
                    "Run tests with coverage: pytest --cov",
                    "Set coverage thresholds in pytest.ini",
                    "Review coverage reports regularly"
                ]
            })
        
        return suggestions
    
    def _group_by_priority(self, suggestions: List[Dict]) -> Dict:
        """Group suggestions by priority"""
        grouped = defaultdict(list)
        for suggestion in suggestions:
            priority = suggestion.get('priority', 'low')
            grouped[priority].append(suggestion)
        
        return {
            'high': grouped['high'],
            'medium': grouped['medium'],
            'low': grouped['low']
        }
    
    def generate_report(self, output_file: Path = None) -> str:
        """Generate optimization report"""
        analysis = self.analyze_test_suite()
        
        lines = []
        lines.append("💡 TEST OPTIMIZATION SUGGESTIONS")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Suggestions: {analysis['total_suggestions']}")
        lines.append("")
        
        # High priority
        if analysis['by_priority']['high']:
            lines.append("🔴 HIGH PRIORITY")
            lines.append("-" * 80)
            for suggestion in analysis['by_priority']['high']:
                lines.append(f"\n  [{suggestion['type'].upper()}] {suggestion.get('test_name', suggestion.get('suggestion', ''))}")
                lines.append(f"    {suggestion.get('suggestion', '')}")
                if 'recommendations' in suggestion:
                    for rec in suggestion['recommendations']:
                        lines.append(f"    • {rec}")
            lines.append("")
        
        # Medium priority
        if analysis['by_priority']['medium']:
            lines.append("🟡 MEDIUM PRIORITY")
            lines.append("-" * 80)
            for suggestion in analysis['by_priority']['medium'][:10]:
                lines.append(f"\n  [{suggestion['type'].upper()}] {suggestion.get('test_name', suggestion.get('suggestion', ''))}")
                lines.append(f"    {suggestion.get('suggestion', '')}")
            lines.append("")
        
        # Low priority
        if analysis['by_priority']['low']:
            lines.append("🟢 LOW PRIORITY")
            lines.append("-" * 80)
            for suggestion in analysis['by_priority']['low'][:5]:
                lines.append(f"\n  [{suggestion['type'].upper()}] {suggestion.get('suggestion', '')}")
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to {output_file}")
        
        return report


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Optimization Suggestions')
    parser.add_argument('--analyze', action='store_true', help='Analyze test suite')
    parser.add_argument('--report', type=str, help='Generate optimization report')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    analyzer = TestOptimizationAnalyzer(project_root)
    
    if args.report:
        print("💡 Generating optimization report...")
        analyzer.generate_report(Path(args.report))
    elif args.analyze:
        print("💡 Analyzing test suite...")
        analysis = analyzer.analyze_test_suite()
        print(f"\n📊 Found {analysis['total_suggestions']} suggestions:")
        print(f"  High Priority: {len(analysis['by_priority']['high'])}")
        print(f"  Medium Priority: {len(analysis['by_priority']['medium'])}")
        print(f"  Low Priority: {len(analysis['by_priority']['low'])}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

