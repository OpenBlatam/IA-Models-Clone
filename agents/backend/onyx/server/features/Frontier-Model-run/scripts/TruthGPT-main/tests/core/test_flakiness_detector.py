"""
Test Flakiness Detector
Detects tests that fail intermittently (flaky tests)
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from datetime import datetime

class FlakinessDetector:
    """Detect flaky tests from historical data"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.history_file = project_root / "test_history.json"
    
    def analyze_flakiness(self, min_runs: int = 5) -> Dict:
        """Analyze test flakiness from history"""
        # Load test history
        history = self._load_history()
        
        if len(history) < min_runs:
            return {
                'error': f'Need at least {min_runs} test runs, found {len(history)}',
                'flaky_tests': []
            }
        
        # Track test outcomes
        test_outcomes = defaultdict(list)
        
        # Load detailed results
        for result_file in sorted(self.results_dir.glob("*.json"), reverse=True)[:50]:  # Last 50 runs
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Track failures
                    for failure in data.get('test_details', {}).get('failures', []):
                        test_name = failure['test']
                        test_outcomes[test_name].append('failed')
                    
                    # Track errors
                    for error in data.get('test_details', {}).get('errors', []):
                        test_name = error['test']
                        test_outcomes[test_name].append('error')
            except Exception:
                continue
        
        # Analyze flakiness
        flaky_tests = []
        
        for test_name, outcomes in test_outcomes.items():
            if len(outcomes) < min_runs:
                continue
            
            failure_rate = outcomes.count('failed') / len(outcomes)
            error_rate = outcomes.count('error') / len(outcomes)
            total_failure_rate = failure_rate + error_rate
            
            # A test is flaky if it fails sometimes but not always
            if 0.1 < total_failure_rate < 0.9:  # Fails 10-90% of the time
                flaky_tests.append({
                    'test': test_name,
                    'total_runs': len(outcomes),
                    'failures': outcomes.count('failed'),
                    'errors': outcomes.count('error'),
                    'failure_rate': total_failure_rate * 100,
                    'flakiness_score': abs(total_failure_rate - 0.5) * 2  # 0 = very flaky, 1 = consistent
                })
        
        # Sort by flakiness (most flaky first)
        flaky_tests.sort(key=lambda x: x['flakiness_score'])
        
        return {
            'total_tests_analyzed': len(test_outcomes),
            'flaky_tests': flaky_tests,
            'flakiness_threshold': 0.1
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
    
    def generate_flakiness_report(self, analysis: Dict) -> str:
        """Generate flakiness report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST FLAKINESS ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Total Tests Analyzed: {analysis['total_tests_analyzed']}")
        lines.append(f"Flaky Tests Found: {len(analysis['flaky_tests'])}")
        lines.append("")
        
        if not analysis['flaky_tests']:
            lines.append("✅ No flaky tests detected!")
            lines.append("All tests are either consistently passing or consistently failing.")
            return "\n".join(lines)
        
        lines.append("🔴 FLAKY TESTS (Most flaky first)")
        lines.append("-" * 80)
        
        for i, test in enumerate(analysis['flaky_tests'][:20], 1):  # Top 20
            lines.append(f"{i}. {test['test']}")
            lines.append(f"   Runs: {test['total_runs']} | "
                        f"Failures: {test['failures']} | "
                        f"Errors: {test['errors']} | "
                        f"Failure Rate: {test['failure_rate']:.1f}%")
            lines.append(f"   Flakiness Score: {test['flakiness_score']:.2f} "
                        f"({'Very Flaky' if test['flakiness_score'] < 0.3 else 'Moderately Flaky' if test['flakiness_score'] < 0.6 else 'Slightly Flaky'})")
            lines.append("")
        
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("1. Investigate flaky tests - they indicate unstable test conditions")
        lines.append("2. Add retry logic for network-dependent tests")
        lines.append("3. Ensure test isolation - tests should not depend on each other")
        lines.append("4. Use deterministic test data")
        lines.append("5. Check for race conditions in concurrent tests")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    detector = FlakinessDetector(project_root)
    analysis = detector.analyze_flakiness(min_runs=5)
    
    report = detector.generate_flakiness_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "flakiness_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Flakiness report saved to: {report_file}")

if __name__ == "__main__":
    main()







