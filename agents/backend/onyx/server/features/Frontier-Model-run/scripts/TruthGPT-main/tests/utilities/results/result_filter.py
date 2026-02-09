"""
Result Filter
Filter test results by various criteria
"""

import json
from pathlib import Path
from typing import Dict, List, Callable, Optional
from datetime import datetime

class ResultFilter:
    """Filter test results by criteria"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def filter_by_success_rate(
        self,
        result_file: str,
        min_rate: float = 0.0,
        max_rate: float = 100.0,
        output_file: Optional[str] = None
    ) -> Dict:
        """Filter results by success rate range"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        success_rate = data.get('success_rate', 0)
        
        if min_rate <= success_rate <= max_rate:
            if output_file:
                output_path = self.results_dir / output_file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            return data
        else:
            return {'filtered_out': True, 'reason': f'Success rate {success_rate:.1f}% outside range [{min_rate}, {max_rate}]'}
    
    def filter_by_execution_time(
        self,
        result_file: str,
        max_time: float = float('inf'),
        output_file: Optional[str] = None
    ) -> Dict:
        """Filter results by maximum execution time"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        execution_time = data.get('execution_time', 0)
        
        if execution_time <= max_time:
            if output_file:
                output_path = self.results_dir / output_file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            return data
        else:
            return {'filtered_out': True, 'reason': f'Execution time {execution_time:.2f}s exceeds {max_time}s'}
    
    def filter_by_failures(
        self,
        result_file: str,
        max_failures: int = 0,
        output_file: Optional[str] = None
    ) -> Dict:
        """Filter results by maximum failure count"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        failures = data.get('failures', 0) + data.get('errors', 0)
        
        if failures <= max_failures:
            if output_file:
                output_path = self.results_dir / output_file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            return data
        else:
            return {'filtered_out': True, 'reason': f'Failure count {failures} exceeds {max_failures}'}
    
    def filter_custom(
        self,
        result_file: str,
        filter_func: Callable[[Dict], bool],
        output_file: Optional[str] = None
    ) -> Dict:
        """Filter using custom function"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        if filter_func(data):
            if output_file:
                output_path = self.results_dir / output_file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            return data
        else:
            return {'filtered_out': True, 'reason': 'Custom filter condition not met'}

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python result_filter.py <file.json> <filter_type> [params...]")
        print("Filter types: success_rate, execution_time, failures, custom")
        return
    
    project_root = Path(__file__).parent.parent
    filter_tool = ResultFilter(project_root)
    
    result_file = sys.argv[1]
    filter_type = sys.argv[2]
    
    if filter_type == 'success_rate':
        min_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        max_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
        result = filter_tool.filter_by_success_rate(result_file, min_rate, max_rate)
    elif filter_type == 'execution_time':
        max_time = float(sys.argv[3]) if len(sys.argv) > 3 else float('inf')
        result = filter_tool.filter_by_execution_time(result_file, max_time)
    elif filter_type == 'failures':
        max_failures = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        result = filter_tool.filter_by_failures(result_file, max_failures)
    else:
        print(f"Unknown filter type: {filter_type}")
        return
    
    if 'error' in result:
        print(f"❌ {result['error']}")
    elif 'filtered_out' in result:
        print(f"⏭️  {result['reason']}")
    else:
        print(f"✅ Result passed filter")

if __name__ == "__main__":
    main()







