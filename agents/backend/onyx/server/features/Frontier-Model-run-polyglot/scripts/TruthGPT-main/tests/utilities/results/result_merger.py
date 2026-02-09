"""
Result Merger
Merge test results from multiple runs
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

class ResultMerger:
    """Merge test results from multiple runs"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def merge_results(
        self,
        result_files: List[str],
        output_file: str = "merged_results.json"
    ) -> Dict:
        """Merge multiple test result files into one"""
        merged = {
            'total_tests': 0,
            'passed': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'execution_time': 0.0,
            'test_details': {
                'failures': [],
                'errors': [],
                'skipped': []
            },
            'runs_merged': [],
            'timestamp': datetime.now().isoformat()
        }
        
        all_tests = set()
        
        for result_file in result_files:
            result_path = self.results_dir / result_file
            if not result_path.exists():
                continue
            
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                merged['total_tests'] = max(merged['total_tests'], data.get('total_tests', 0))
                merged['passed'] = max(merged['passed'], data.get('passed', 0))
                merged['failures'] += data.get('failures', 0)
                merged['errors'] += data.get('errors', 0)
                merged['skipped'] += data.get('skipped', 0)
                merged['execution_time'] += data.get('execution_time', 0.0)
                
                # Merge test details
                test_details = data.get('test_details', {})
                
                for failure in test_details.get('failures', []):
                    test_name = str(failure.get('test', ''))
                    if test_name and test_name not in all_tests:
                        merged['test_details']['failures'].append(failure)
                        all_tests.add(test_name)
                
                for error in test_details.get('errors', []):
                    test_name = str(error.get('test', ''))
                    if test_name and test_name not in all_tests:
                        merged['test_details']['errors'].append(error)
                        all_tests.add(test_name)
                
                for skipped in test_details.get('skipped', []):
                    test_name = str(skipped.get('test', ''))
                    if test_name and test_name not in all_tests:
                        merged['test_details']['skipped'].append(skipped)
                        all_tests.add(test_name)
                
                merged['runs_merged'].append({
                    'file': result_file,
                    'timestamp': data.get('timestamp', '')
                })
            except Exception as e:
                print(f"⚠️  Error processing {result_file}: {e}")
                continue
        
        # Calculate merged metrics
        total = merged['total_tests']
        if total > 0:
            merged['success_rate'] = ((merged['passed'] / total) * 100)
        else:
            merged['success_rate'] = 0
        
        # Save merged results
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2)
        
        return merged
    
    def generate_merge_report(self, merged: Dict) -> str:
        """Generate merge report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULT MERGE")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Runs Merged: {len(merged['runs_merged'])}")
        lines.append(f"Total Tests: {merged['total_tests']}")
        lines.append(f"Success Rate: {merged['success_rate']:.1f}%")
        lines.append(f"Total Execution Time: {merged['execution_time']:.2f}s")
        lines.append("")
        
        lines.append("📊 MERGED METRICS")
        lines.append("-" * 80)
        lines.append(f"Passed:    {merged['passed']}")
        lines.append(f"Failures:  {merged['failures']}")
        lines.append(f"Errors:    {merged['errors']}")
        lines.append(f"Skipped:   {merged['skipped']}")
        lines.append("")
        
        lines.append("📋 MERGED RUNS")
        lines.append("-" * 80)
        for run in merged['runs_merged']:
            lines.append(f"  • {run['file']} ({run['timestamp']})")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python result_merger.py <file1.json> <file2.json> ...")
        return
    
    project_root = Path(__file__).parent.parent
    merger = ResultMerger(project_root)
    
    result_files = sys.argv[1:]
    merged = merger.merge_results(result_files)
    
    report = merger.generate_merge_report(merged)
    print(report)
    
    print(f"\n✅ Merged results saved to: test_results/merged_results.json")

if __name__ == "__main__":
    main()







