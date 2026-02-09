"""
Result Normalizer
Normalize test result formats to standard structure
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ResultNormalizer:
    """Normalize test result formats"""
    
    STANDARD_FORMAT = {
        'total_tests': 0,
        'passed': 0,
        'failures': 0,
        'errors': 0,
        'skipped': 0,
        'success_rate': 0.0,
        'execution_time': 0.0,
        'timestamp': '',
        'test_details': {
            'failures': [],
            'errors': [],
            'skipped': []
        }
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def normalize(
        self,
        result_file: str,
        output_file: Optional[str] = None
    ) -> Dict:
        """Normalize test result to standard format"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        # Create normalized structure
        normalized = self.STANDARD_FORMAT.copy()
        
        # Map common field names
        field_mapping = {
            'total': 'total_tests',
            'totalTests': 'total_tests',
            'test_count': 'total_tests',
            'success': 'passed',
            'successful': 'passed',
            'failed': 'failures',
            'failure': 'failures',
            'error': 'errors',
            'skip': 'skipped',
            'skipped_tests': 'skipped',
            'time': 'execution_time',
            'duration': 'execution_time',
            'exec_time': 'execution_time',
            'rate': 'success_rate',
            'successRate': 'success_rate',
            'pass_rate': 'success_rate'
        }
        
        # Normalize fields
        for key, value in data.items():
            normalized_key = field_mapping.get(key.lower(), key.lower())
            if normalized_key in normalized:
                if isinstance(normalized[normalized_key], type(value)):
                    normalized[normalized_key] = value
                else:
                    # Type conversion
                    try:
                        normalized[normalized_key] = type(normalized[normalized_key])(value)
                    except (ValueError, TypeError):
                        pass
        
        # Ensure timestamp
        if not normalized['timestamp']:
            normalized['timestamp'] = datetime.now().isoformat()
        
        # Calculate success rate if missing
        if normalized['success_rate'] == 0 and normalized['total_tests'] > 0:
            normalized['success_rate'] = (
                (normalized['passed'] / normalized['total_tests']) * 100
            )
        
        # Normalize test details
        if 'test_details' not in data or not isinstance(data['test_details'], dict):
            normalized['test_details'] = {
                'failures': [],
                'errors': [],
                'skipped': []
            }
        else:
            # Ensure all required keys exist
            for key in ['failures', 'errors', 'skipped']:
                if key not in normalized['test_details']:
                    normalized['test_details'][key] = []
        
        # Save normalized result
        if output_file is None:
            output_file = f"normalized_{result_file}"
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, indent=2)
        
        return normalized
    
    def generate_normalization_report(self, original: Dict, normalized: Dict) -> str:
        """Generate normalization report"""
        lines = []
        lines.append("=" * 80)
        lines.append("RESULT NORMALIZATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("✅ Normalization completed successfully")
        lines.append("")
        
        lines.append("📊 NORMALIZED METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Tests: {normalized['total_tests']}")
        lines.append(f"Passed: {normalized['passed']}")
        lines.append(f"Failures: {normalized['failures']}")
        lines.append(f"Errors: {normalized['errors']}")
        lines.append(f"Skipped: {normalized['skipped']}")
        lines.append(f"Success Rate: {normalized['success_rate']:.1f}%")
        lines.append(f"Execution Time: {normalized['execution_time']:.2f}s")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python result_normalizer.py <file.json> [output.json]")
        return
    
    project_root = Path(__file__).parent.parent
    normalizer = ResultNormalizer(project_root)
    
    result_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    normalized = normalizer.normalize(result_file, output_file)
    
    if 'error' in normalized:
        print(f"❌ {normalized['error']}")
    else:
        print("✅ Result normalized successfully")
        if output_file:
            print(f"   Saved to: test_results/{output_file}")

if __name__ == "__main__":
    main()







