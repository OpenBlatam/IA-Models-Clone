"""
Result Transformer
Transform test results between different formats and structures
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ResultTransformer:
    """Transform test results between formats"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def transform_to_simple(
        self,
        result_file: str,
        output_file: str = "simple_results.json"
    ) -> Dict:
        """Transform to simple format (minimal fields)"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        simple = {
            'tests': data.get('total_tests', 0),
            'passed': data.get('passed', 0),
            'failed': data.get('failures', 0) + data.get('errors', 0),
            'rate': data.get('success_rate', 0),
            'time': data.get('execution_time', 0)
        }
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simple, f, indent=2)
        
        return simple
    
    def transform_to_detailed(
        self,
        result_file: str,
        output_file: str = "detailed_results.json"
    ) -> Dict:
        """Transform to detailed format (all fields expanded)"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        detailed = {
            'metadata': {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'version': data.get('version', '1.0'),
                'environment': data.get('environment', 'unknown')
            },
            'summary': {
                'total_tests': data.get('total_tests', 0),
                'passed': data.get('passed', 0),
                'failures': data.get('failures', 0),
                'errors': data.get('errors', 0),
                'skipped': data.get('skipped', 0),
                'success_rate': data.get('success_rate', 0),
                'execution_time': data.get('execution_time', 0)
            },
            'test_details': data.get('test_details', {}),
            'performance': {
                'average_test_time': (
                    data.get('execution_time', 0) / data.get('total_tests', 1)
                    if data.get('total_tests', 0) > 0 else 0
                ),
                'total_execution_time': data.get('execution_time', 0)
            }
        }
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2)
        
        return detailed
    
    def transform_to_ci_format(
        self,
        result_file: str,
        output_file: str = "ci_results.json"
    ) -> Dict:
        """Transform to CI/CD format"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        ci_format = {
            'status': 'passed' if data.get('success_rate', 0) >= 95 else 'failed',
            'tests': {
                'total': data.get('total_tests', 0),
                'passed': data.get('passed', 0),
                'failed': data.get('failures', 0) + data.get('errors', 0),
                'skipped': data.get('skipped', 0)
            },
            'duration': data.get('execution_time', 0),
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'success_rate': data.get('success_rate', 0)
        }
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ci_format, f, indent=2)
        
        return ci_format
    
    def transform(
        self,
        result_file: str,
        target_format: str,
        output_file: Optional[str] = None
    ) -> Dict:
        """Transform result to target format"""
        if output_file is None:
            output_file = f"{target_format}_results.json"
        
        if target_format == 'simple':
            return self.transform_to_simple(result_file, output_file)
        elif target_format == 'detailed':
            return self.transform_to_detailed(result_file, output_file)
        elif target_format == 'ci':
            return self.transform_to_ci_format(result_file, output_file)
        else:
            return {'error': f'Unknown format: {target_format}'}

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python result_transformer.py <file.json> <format> [output.json]")
        print("Formats: simple, detailed, ci")
        return
    
    project_root = Path(__file__).parent.parent
    transformer = ResultTransformer(project_root)
    
    result_file = sys.argv[1]
    target_format = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = transformer.transform(result_file, target_format, output_file)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
    else:
        print(f"✅ Transformed to {target_format} format")
        if output_file:
            print(f"   Saved to: test_results/{output_file}")

if __name__ == "__main__":
    main()







