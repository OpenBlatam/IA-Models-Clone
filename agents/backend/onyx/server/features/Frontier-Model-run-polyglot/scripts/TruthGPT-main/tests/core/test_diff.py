"""
Test Result Diff
Compare test results between two runs
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

class TestResultDiff:
    """Compare test results between runs"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def diff_results(
        self,
        run1_file: str,
        run2_file: str
    ) -> Dict:
        """Compare two test result files"""
        run1_path = self.results_dir / run1_file
        run2_path = self.results_dir / run2_file
        
        if not run1_path.exists() or not run2_path.exists():
            return {'error': 'One or both result files not found'}
        
        # Load results
        with open(run1_path, 'r', encoding='utf-8') as f:
            run1 = json.load(f)
        
        with open(run2_path, 'r', encoding='utf-8') as f:
            run2 = json.load(f)
        
        # Extract test names
        def get_test_names(results: Dict) -> Set[str]:
            names = set()
            test_details = results.get('test_details', {})
            
            for test_list in [
                test_details.get('failures', []),
                test_details.get('errors', []),
                test_details.get('skipped', [])
            ]:
                for test in test_list:
                    names.add(str(test.get('test', '')))
            
            return names
        
        run1_tests = get_test_names(run1)
        run2_tests = get_test_names(run2)
        
        # Calculate differences
        only_in_run1 = run1_tests - run2_tests
        only_in_run2 = run2_tests - run1_tests
        in_both = run1_tests & run2_tests
        
        # Compare metrics
        metrics_diff = {
            'total_tests': run2.get('total_tests', 0) - run1.get('total_tests', 0),
            'success_rate': run2.get('success_rate', 0) - run1.get('success_rate', 0),
            'execution_time': run2.get('execution_time', 0) - run1.get('execution_time', 0),
            'failures': run2.get('failures', 0) - run1.get('failures', 0),
            'errors': run2.get('errors', 0) - run1.get('errors', 0)
        }
        
        return {
            'run1': {
                'file': run1_file,
                'timestamp': run1.get('timestamp', ''),
                'total_tests': run1.get('total_tests', 0),
                'success_rate': run1.get('success_rate', 0)
            },
            'run2': {
                'file': run2_file,
                'timestamp': run2.get('timestamp', ''),
                'total_tests': run2.get('total_tests', 0),
                'success_rate': run2.get('success_rate', 0)
            },
            'test_differences': {
                'only_in_run1': list(only_in_run1),
                'only_in_run2': list(only_in_run2),
                'in_both': list(in_both)
            },
            'metrics_diff': metrics_diff
        }
    
    def generate_diff_report(self, diff_data: Dict) -> str:
        """Generate diff report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULT DIFF")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in diff_data:
            lines.append(f"❌ {diff_data['error']}")
            return "\n".join(lines)
        
        lines.append(f"Run 1: {diff_data['run1']['file']}")
        lines.append(f"  Timestamp: {diff_data['run1']['timestamp']}")
        lines.append(f"  Tests: {diff_data['run1']['total_tests']} | Success: {diff_data['run1']['success_rate']:.1f}%")
        lines.append("")
        
        lines.append(f"Run 2: {diff_data['run2']['file']}")
        lines.append(f"  Timestamp: {diff_data['run2']['timestamp']}")
        lines.append(f"  Tests: {diff_data['run2']['total_tests']} | Success: {diff_data['run2']['success_rate']:.1f}%")
        lines.append("")
        
        lines.append("📊 METRICS DIFFERENCES")
        lines.append("-" * 80)
        metrics = diff_data['metrics_diff']
        lines.append(f"Total Tests:     {metrics['total_tests']:+.0f}")
        lines.append(f"Success Rate:    {metrics['success_rate']:+.1f}%")
        lines.append(f"Execution Time:  {metrics['execution_time']:+.2f}s")
        lines.append(f"Failures:        {metrics['failures']:+.0f}")
        lines.append(f"Errors:          {metrics['errors']:+.0f}")
        lines.append("")
        
        lines.append("🔍 TEST DIFFERENCES")
        lines.append("-" * 80)
        test_diff = diff_data['test_differences']
        
        if test_diff['only_in_run1']:
            lines.append(f"Only in Run 1 ({len(test_diff['only_in_run1'])}):")
            for test in test_diff['only_in_run1'][:10]:
                lines.append(f"  • {test}")
            if len(test_diff['only_in_run1']) > 10:
                lines.append(f"  ... and {len(test_diff['only_in_run1']) - 10} more")
            lines.append("")
        
        if test_diff['only_in_run2']:
            lines.append(f"Only in Run 2 ({len(test_diff['only_in_run2'])}):")
            for test in test_diff['only_in_run2'][:10]:
                lines.append(f"  • {test}")
            if len(test_diff['only_in_run2']) > 10:
                lines.append(f"  ... and {len(test_diff['only_in_run2']) - 10} more")
            lines.append("")
        
        lines.append(f"In Both Runs: {len(test_diff['in_both'])}")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python test_diff.py <run1_file> <run2_file>")
        return
    
    project_root = Path(__file__).parent.parent
    diff_tool = TestResultDiff(project_root)
    
    diff_data = diff_tool.diff_results(sys.argv[1], sys.argv[2])
    report = diff_tool.generate_diff_report(diff_data)
    
    print(report)
    
    # Save report
    report_file = project_root / "test_diff_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Diff report saved to: {report_file}")

if __name__ == "__main__":
    main()







