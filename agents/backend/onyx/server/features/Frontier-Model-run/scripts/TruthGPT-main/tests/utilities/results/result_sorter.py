"""
Result Sorter
Sort test results by various criteria
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class ResultSorter:
    """Sort test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def sort_results(
        self,
        result_files: List[str],
        sort_by: str = 'timestamp',
        reverse: bool = False,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """Sort multiple result files"""
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
        
        # Sort by criteria
        if sort_by == 'timestamp':
            results.sort(key=lambda x: x.get('timestamp', ''), reverse=reverse)
        elif sort_by == 'success_rate':
            results.sort(key=lambda x: x.get('success_rate', 0), reverse=reverse)
        elif sort_by == 'execution_time':
            results.sort(key=lambda x: x.get('execution_time', 0), reverse=reverse)
        elif sort_by == 'total_tests':
            results.sort(key=lambda x: x.get('total_tests', 0), reverse=reverse)
        elif sort_by == 'failures':
            results.sort(key=lambda x: x.get('failures', 0) + x.get('errors', 0), reverse=reverse)
        else:
            print(f"Unknown sort criteria: {sort_by}")
            return []
        
        # Save sorted results
        if output_file:
            output_path = self.results_dir / output_file
            sorted_data = {
                'sorted_by': sort_by,
                'reverse': reverse,
                'total_results': len(results),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, indent=2)
        
        return results
    
    def generate_sort_report(self, sorted_results: List[Dict], sort_by: str) -> str:
        """Generate sort report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"TEST RESULTS SORTED BY {sort_by.upper()}")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Total Results: {len(sorted_results)}")
        lines.append("")
        
        lines.append("📊 SORTED RESULTS")
        lines.append("-" * 80)
        
        for i, result in enumerate(sorted_results[:20], 1):  # Show first 20
            filename = result.get('_filename', 'unknown')
            if sort_by == 'timestamp':
                value = result.get('timestamp', '')
            elif sort_by == 'success_rate':
                value = f"{result.get('success_rate', 0):.1f}%"
            elif sort_by == 'execution_time':
                value = f"{result.get('execution_time', 0):.2f}s"
            elif sort_by == 'total_tests':
                value = result.get('total_tests', 0)
            elif sort_by == 'failures':
                value = result.get('failures', 0) + result.get('errors', 0)
            else:
                value = 'N/A'
            
            lines.append(f"{i:3d}. {filename} - {sort_by}: {value}")
        
        if len(sorted_results) > 20:
            lines.append(f"\n... and {len(sorted_results) - 20} more results")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    import glob
    
    if len(sys.argv) < 2:
        print("Usage: python result_sorter.py <pattern> [sort_by] [reverse]")
        print("Sort by: timestamp, success_rate, execution_time, total_tests, failures")
        print("Example: python result_sorter.py '*.json' success_rate")
        return
    
    project_root = Path(__file__).parent.parent
    sorter = ResultSorter(project_root)
    
    pattern = sys.argv[1]
    sort_by = sys.argv[2] if len(sys.argv) > 2 else 'timestamp'
    reverse = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    
    # Find matching files
    results_dir = project_root / "test_results"
    result_files = [f.name for f in results_dir.glob(pattern)]
    
    if not result_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    sorted_results = sorter.sort_results(result_files, sort_by, reverse)
    report = sorter.generate_sort_report(sorted_results, sort_by)
    
    print(report)
    
    # Save sorted results
    output_file = f"sorted_by_{sort_by}.json"
    sorter.sort_results(result_files, sort_by, reverse, output_file)
    print(f"\n✅ Sorted results saved to: test_results/{output_file}")

if __name__ == "__main__":
    main()







