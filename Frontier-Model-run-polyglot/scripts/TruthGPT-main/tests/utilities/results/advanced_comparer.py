"""
Advanced Comparer
Compare multiple test runs with detailed analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

class AdvancedComparer:
    """Compare multiple test runs"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def compare_multiple(
        self,
        result_files: List[str]
    ) -> Dict:
        """Compare multiple test result files"""
        results = []
        
        for result_file in result_files:
            result_path = self.results_dir / result_file
            if not result_path.exists():
                continue
            
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_filename'] = result_file
                    results.append(data)
            except Exception:
                continue
        
        if len(results) < 2:
            return {'error': 'Need at least 2 result files to compare'}
        
        # Compare metrics
        comparison = {
            'files_compared': [r['_filename'] for r in results],
            'total_files': len(results),
            'metrics': {}
        }
        
        # Compare each metric
        metrics_to_compare = [
            'total_tests', 'passed', 'failures', 'errors', 
            'skipped', 'success_rate', 'execution_time'
        ]
        
        for metric in metrics_to_compare:
            values = [r.get(metric, 0) for r in results]
            comparison['metrics'][metric] = {
                'values': values,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'range': max(values) - min(values) if values and len(values) > 1 else 0,
                'variance': self._calculate_variance(values)
            }
        
        # Compare test details
        all_tests = set()
        for result in results:
            test_details = result.get('test_details', {})
            for test_list in [
                test_details.get('failures', []),
                test_details.get('errors', []),
                test_details.get('skipped', [])
            ]:
                for test in test_list:
                    all_tests.add(str(test.get('test', '')))
        
        comparison['unique_tests'] = len(all_tests)
        comparison['common_tests'] = self._find_common_tests(results)
        
        return comparison
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance"""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance
    
    def _find_common_tests(self, results: List[Dict]) -> int:
        """Find tests common to all results"""
        if not results:
            return 0
        
        # Get tests from first result
        first_result = results[0]
        test_details = first_result.get('test_details', {})
        first_tests = set()
        
        for test_list in [
            test_details.get('failures', []),
            test_details.get('errors', []),
            test_details.get('skipped', [])
        ]:
            for test in test_list:
                first_tests.add(str(test.get('test', '')))
        
        # Find common tests
        for result in results[1:]:
            test_details = result.get('test_details', {})
            result_tests = set()
            
            for test_list in [
                test_details.get('failures', []),
                test_details.get('errors', []),
                test_details.get('skipped', [])
            ]:
                for test in test_list:
                    result_tests.add(str(test.get('test', '')))
            
            first_tests &= result_tests
        
        return len(first_tests)
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate comparison report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in comparison:
            lines.append(f"❌ {comparison['error']}")
            return "\n".join(lines)
        
        lines.append(f"Files Compared: {comparison['total_files']}")
        for filename in comparison['files_compared']:
            lines.append(f"  • {filename}")
        lines.append("")
        
        lines.append("📊 METRICS COMPARISON")
        lines.append("-" * 80)
        
        for metric, data in comparison['metrics'].items():
            lines.append(f"\n{metric.replace('_', ' ').title()}")
            lines.append(f"  Values: {data['values']}")
            lines.append(f"  Min: {data['min']}")
            lines.append(f"  Max: {data['max']}")
            lines.append(f"  Range: {data['range']}")
            lines.append(f"  Variance: {data['variance']:.2f}")
        
        lines.append("")
        lines.append("🔍 TEST ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Unique Tests: {comparison['unique_tests']}")
        lines.append(f"Common Tests: {comparison['common_tests']}")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python advanced_comparer.py <file1.json> <file2.json> [file3.json] ...")
        return
    
    project_root = Path(__file__).parent.parent
    comparer = AdvancedComparer(project_root)
    
    result_files = sys.argv[1:]
    comparison = comparer.compare_multiple(result_files)
    
    report = comparer.generate_comparison_report(comparison)
    print(report)

if __name__ == "__main__":
    main()







