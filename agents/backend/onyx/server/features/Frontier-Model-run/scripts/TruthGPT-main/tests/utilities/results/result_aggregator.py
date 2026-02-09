"""
Result Aggregator
Aggregate test results from multiple sources
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

class ResultAggregator:
    """Aggregate test results from multiple sources"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def aggregate_results(
        self,
        result_files: List[str],
        output_file: str = "aggregated_results.json"
    ) -> Dict:
        """Aggregate multiple test result files"""
        aggregated = {
            'total_runs': 0,
            'total_tests': 0,
            'total_passed': 0,
            'total_failures': 0,
            'total_errors': 0,
            'total_skipped': 0,
            'total_execution_time': 0.0,
            'runs': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for result_file in result_files:
            result_path = self.results_dir / result_file
            if not result_path.exists():
                continue
            
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                aggregated['total_runs'] += 1
                aggregated['total_tests'] += data.get('total_tests', 0)
                aggregated['total_passed'] += data.get('passed', 0)
                aggregated['total_failures'] += data.get('failures', 0)
                aggregated['total_errors'] += data.get('errors', 0)
                aggregated['total_skipped'] += data.get('skipped', 0)
                aggregated['total_execution_time'] += data.get('execution_time', 0.0)
                
                aggregated['runs'].append({
                    'file': result_file,
                    'timestamp': data.get('timestamp', ''),
                    'summary': {
                        'total_tests': data.get('total_tests', 0),
                        'passed': data.get('passed', 0),
                        'failures': data.get('failures', 0),
                        'errors': data.get('errors', 0),
                        'success_rate': data.get('success_rate', 0)
                    }
                })
            except Exception as e:
                print(f"⚠️  Error processing {result_file}: {e}")
                continue
        
        # Calculate aggregated metrics
        if aggregated['total_tests'] > 0:
            aggregated['overall_success_rate'] = (
                (aggregated['total_passed'] / aggregated['total_tests']) * 100
            )
            aggregated['average_execution_time'] = (
                aggregated['total_execution_time'] / aggregated['total_runs']
                if aggregated['total_runs'] > 0 else 0
            )
        else:
            aggregated['overall_success_rate'] = 0
            aggregated['average_execution_time'] = 0
        
        # Save aggregated results
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        
        return aggregated
    
    def generate_aggregation_report(self, aggregated: Dict) -> str:
        """Generate aggregation report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULT AGGREGATION")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Total Runs Aggregated: {aggregated['total_runs']}")
        lines.append(f"Total Tests: {aggregated['total_tests']}")
        lines.append(f"Overall Success Rate: {aggregated.get('overall_success_rate', 0):.1f}%")
        lines.append(f"Average Execution Time: {aggregated.get('average_execution_time', 0):.2f}s")
        lines.append("")
        
        lines.append("📊 BREAKDOWN")
        lines.append("-" * 80)
        lines.append(f"Passed:    {aggregated['total_passed']}")
        lines.append(f"Failures:  {aggregated['total_failures']}")
        lines.append(f"Errors:    {aggregated['total_errors']}")
        lines.append(f"Skipped:   {aggregated['total_skipped']}")
        lines.append("")
        
        lines.append("📋 INDIVIDUAL RUNS")
        lines.append("-" * 80)
        for run in aggregated['runs']:
            lines.append(f"{run['file']}")
            lines.append(f"  Tests: {run['summary']['total_tests']} | "
                        f"Success: {run['summary']['success_rate']:.1f}%")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python result_aggregator.py <file1.json> <file2.json> ...")
        return
    
    project_root = Path(__file__).parent.parent
    aggregator = ResultAggregator(project_root)
    
    result_files = sys.argv[1:]
    aggregated = aggregator.aggregate_results(result_files)
    
    report = aggregator.generate_aggregation_report(aggregated)
    print(report)
    
    print(f"\n✅ Aggregated results saved to: test_results/aggregated_results.json")

if __name__ == "__main__":
    main()







