"""
Impact Analyzer
Analyzes impact of test failures
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from datetime import datetime, timedelta

class ImpactAnalyzer:
    """Analyze impact of test failures"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.history_file = project_root / "test_history.json"
    
    def analyze_impact(self, lookback_days: int = 30) -> Dict:
        """Analyze impact of test failures"""
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        # Collect all failures
        failing_tests = defaultdict(int)
        error_types = defaultdict(int)
        test_categories = defaultdict(int)
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                timestamp = data.get('timestamp', '')
                if timestamp < cutoff_date:
                    continue
                
                test_details = data.get('test_details', {})
                
                # Count failures
                for failure in test_details.get('failures', []):
                    test_name = str(failure.get('test', ''))
                    failing_tests[test_name] += 1
                    
                    # Categorize
                    category = self._categorize_test(test_name)
                    test_categories[category] += 1
                    
                    # Error type
                    traceback = failure.get('traceback', '')
                    error_type = self._extract_error_type(traceback)
                    error_types[error_type] += 1
                
                # Count errors
                for error in test_details.get('errors', []):
                    test_name = str(error.get('test', ''))
                    failing_tests[test_name] += 1
                    
                    category = self._categorize_test(test_name)
                    test_categories[category] += 1
                    
                    traceback = error.get('traceback', '')
                    error_type = self._extract_error_type(traceback)
                    error_types[error_type] += 1
            except Exception:
                continue
        
        # Calculate impact scores
        total_failures = sum(failing_tests.values())
        
        # Most impactful tests (fail most often)
        most_impactful = sorted(
            failing_tests.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Impact by category
        category_impact = dict(sorted(
            test_categories.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Impact by error type
        error_impact = dict(sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return {
            'total_failures': total_failures,
            'unique_failing_tests': len(failing_tests),
            'most_impactful_tests': [
                {'test': test, 'failures': count, 'impact_score': round((count / total_failures * 100), 1)}
                for test, count in most_impactful
            ],
            'category_impact': category_impact,
            'error_type_impact': error_impact,
            'period_days': lookback_days
        }
    
    def _categorize_test(self, test_name: str) -> str:
        """Categorize test by name"""
        test_lower = test_name.lower()
        
        if 'integration' in test_lower:
            return 'Integration'
        elif 'unit' in test_lower:
            return 'Unit'
        elif 'performance' in test_lower or 'benchmark' in test_lower:
            return 'Performance'
        elif 'security' in test_lower:
            return 'Security'
        elif 'edge' in test_lower:
            return 'Edge Case'
        elif 'regression' in test_lower:
            return 'Regression'
        elif 'core' in test_lower:
            return 'Core'
        else:
            return 'Other'
    
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
    
    def generate_impact_report(self, analysis: Dict) -> str:
        """Generate impact analysis report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST FAILURE IMPACT ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Period: Last {analysis['period_days']} days")
        lines.append(f"Total Failures: {analysis['total_failures']}")
        lines.append(f"Unique Failing Tests: {analysis['unique_failing_tests']}")
        lines.append("")
        
        lines.append("🔴 MOST IMPACTFUL TESTS")
        lines.append("-" * 80)
        for item in analysis['most_impactful_tests']:
            lines.append(f"{item['test']}")
            lines.append(f"  Failures: {item['failures']}")
            lines.append(f"  Impact Score: {item['impact_score']}%")
            lines.append("")
        
        lines.append("📊 IMPACT BY CATEGORY")
        lines.append("-" * 80)
        for category, count in list(analysis['category_impact'].items())[:10]:
            percentage = (count / analysis['total_failures'] * 100) if analysis['total_failures'] > 0 else 0
            lines.append(f"{category}: {count} ({percentage:.1f}%)")
        
        lines.append("")
        lines.append("⚠️  IMPACT BY ERROR TYPE")
        lines.append("-" * 80)
        for error_type, count in list(analysis['error_type_impact'].items())[:10]:
            percentage = (count / analysis['total_failures'] * 100) if analysis['total_failures'] > 0 else 0
            lines.append(f"{error_type}: {count} ({percentage:.1f}%)")
        
        lines.append("")
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("1. Focus on fixing most impactful tests first")
        lines.append("2. Address common error types systematically")
        lines.append("3. Prioritize tests in high-impact categories")
        lines.append("4. Monitor impact trends over time")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    analyzer = ImpactAnalyzer(project_root)
    analysis = analyzer.analyze_impact(lookback_days=30)
    
    report = analyzer.generate_impact_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "impact_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Impact analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







