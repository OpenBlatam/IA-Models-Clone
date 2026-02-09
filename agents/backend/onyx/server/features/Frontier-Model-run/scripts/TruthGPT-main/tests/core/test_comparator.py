"""
Test Result Comparator
Compares test results between different runs to identify changes
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

class TestResultComparator:
    """Compare test results between runs"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def save_result(self, test_results: Dict, run_name: str = None) -> Path:
        """Save test results for later comparison"""
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_file = self.results_dir / f"{run_name}.json"
        
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'run_name': run_name,
            'summary': {
                'total_tests': test_results.get('total_tests', 0),
                'passed': test_results.get('total_tests', 0) - test_results.get('failures', 0) - test_results.get('errors', 0) - test_results.get('skipped', 0),
                'failed': test_results.get('failures', 0),
                'errors': test_results.get('errors', 0),
                'skipped': test_results.get('skipped', 0),
                'success_rate': test_results.get('success_rate', 0),
                'execution_time': test_results.get('execution_time', 0)
            },
            'test_details': self._extract_test_details(test_results)
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        return result_file
    
    def _extract_test_details(self, test_results: Dict) -> Dict:
        """Extract detailed test information"""
        details = {
            'failures': [],
            'errors': [],
            'skipped': []
        }
        
        if 'failures' in test_results:
            for test, traceback in test_results.get('failures', []):
                details['failures'].append({
                    'test': str(test),
                    'traceback': traceback[:500]  # First 500 chars
                })
        
        if 'errors' in test_results:
            for test, traceback in test_results.get('errors', []):
                details['errors'].append({
                    'test': str(test),
                    'traceback': traceback[:500]
                })
        
        if 'skipped' in test_results:
            for test, reason in test_results.get('skipped', []):
                details['skipped'].append({
                    'test': str(test),
                    'reason': str(reason)
                })
        
        return details
    
    def load_result(self, run_name: str) -> Optional[Dict]:
        """Load saved test result"""
        result_file = self.results_dir / f"{run_name}.json"
        if not result_file.exists():
            return None
        
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_results(self) -> List[Dict]:
        """List all saved test results"""
        results = []
        for result_file in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['file'] = result_file.name
                    results.append(data)
            except Exception:
                continue
        return results
    
    def compare(self, run1_name: str, run2_name: str) -> Dict:
        """Compare two test runs"""
        result1 = self.load_result(run1_name)
        result2 = self.load_result(run2_name)
        
        if not result1 or not result2:
            return {'error': 'One or both runs not found'}
        
        summary1 = result1['summary']
        summary2 = result2['summary']
        
        comparison = {
            'run1': {
                'name': run1_name,
                'timestamp': result1['timestamp'],
                'summary': summary1
            },
            'run2': {
                'name': run2_name,
                'timestamp': result2['timestamp'],
                'summary': summary2
            },
            'differences': {
                'total_tests': summary2['total_tests'] - summary1['total_tests'],
                'passed': summary2['passed'] - summary1['passed'],
                'failed': summary2['failed'] - summary1['failed'],
                'errors': summary2['errors'] - summary1['errors'],
                'skipped': summary2['skipped'] - summary1['skipped'],
                'success_rate': summary2['success_rate'] - summary1['success_rate'],
                'execution_time': summary2['execution_time'] - summary1['execution_time']
            },
            'new_failures': [],
            'fixed_tests': [],
            'new_errors': [],
            'fixed_errors': []
        }
        
        # Compare test details
        failures1 = {f['test'] for f in result1['test_details']['failures']}
        failures2 = {f['test'] for f in result2['test_details']['failures']}
        
        comparison['new_failures'] = list(failures2 - failures1)
        comparison['fixed_tests'] = list(failures1 - failures2)
        
        errors1 = {e['test'] for e in result1['test_details']['errors']}
        errors2 = {e['test'] for e in result2['test_details']['errors']}
        
        comparison['new_errors'] = list(errors2 - errors1)
        comparison['fixed_errors'] = list(errors1 - errors2)
        
        return comparison
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate human-readable comparison report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULT COMPARISON")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Run 1: {comparison['run1']['name']} ({comparison['run1']['timestamp']})")
        lines.append(f"Run 2: {comparison['run2']['name']} ({comparison['run2']['timestamp']})")
        lines.append("")
        
        lines.append("SUMMARY DIFFERENCES")
        lines.append("-" * 80)
        diff = comparison['differences']
        
        for key, value in diff.items():
            if isinstance(value, float):
                sign = "+" if value >= 0 else ""
                lines.append(f"  {key:20s}: {sign}{value:+.2f}")
            else:
                sign = "+" if value >= 0 else ""
                lines.append(f"  {key:20s}: {sign}{value:+d}")
        
        lines.append("")
        
        # New failures
        if comparison['new_failures']:
            lines.append(f"❌ NEW FAILURES ({len(comparison['new_failures'])})")
            lines.append("-" * 80)
            for test in comparison['new_failures']:
                lines.append(f"  • {test}")
            lines.append("")
        
        # Fixed tests
        if comparison['fixed_tests']:
            lines.append(f"✅ FIXED TESTS ({len(comparison['fixed_tests'])})")
            lines.append("-" * 80)
            for test in comparison['fixed_tests']:
                lines.append(f"  • {test}")
            lines.append("")
        
        # New errors
        if comparison['new_errors']:
            lines.append(f"💥 NEW ERRORS ({len(comparison['new_errors'])})")
            lines.append("-" * 80)
            for test in comparison['new_errors']:
                lines.append(f"  • {test}")
            lines.append("")
        
        # Fixed errors
        if comparison['fixed_errors']:
            lines.append(f"✅ FIXED ERRORS ({len(comparison['fixed_errors'])})")
            lines.append("-" * 80)
            for test in comparison['fixed_errors']:
                lines.append(f"  • {test}")
            lines.append("")
        
        # Overall status
        if diff['success_rate'] > 0:
            lines.append("📈 IMPROVEMENT: Success rate increased!")
        elif diff['success_rate'] < 0:
            lines.append("📉 REGRESSION: Success rate decreased!")
        else:
            lines.append("➡️  STABLE: No change in success rate")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    comparator = TestResultComparator(project_root)
    
    # List all results
    results = comparator.list_results()
    
    if len(results) < 2:
        print("Need at least 2 test runs to compare")
        print(f"Found {len(results)} saved results")
        return
    
    print("Available test runs:")
    for i, result in enumerate(results[:10], 1):
        print(f"  {i}. {result['run_name']} - {result['timestamp']}")
    
    if len(results) >= 2:
        # Compare last two
        run1 = results[1]['run_name']
        run2 = results[0]['run_name']
        
        print(f"\nComparing {run1} vs {run2}...")
        comparison = comparator.compare(run1, run2)
        report = comparator.generate_comparison_report(comparison)
        print(report)
        
        # Save report
        report_file = project_root / f"comparison_{run1}_vs_{run2}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Comparison report saved to: {report_file}")

if __name__ == "__main__":
    main()







