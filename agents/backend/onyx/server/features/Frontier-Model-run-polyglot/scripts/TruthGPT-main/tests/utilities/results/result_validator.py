"""
Result Validator
Validate test result structure and data
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class ResultValidator:
    """Validate test result files"""
    
    REQUIRED_FIELDS = [
        'total_tests',
        'timestamp'
    ]
    
    OPTIONAL_FIELDS = [
        'passed',
        'failures',
        'errors',
        'skipped',
        'execution_time',
        'success_rate',
        'test_details'
    ]
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def validate_result(self, result_file: str) -> Tuple[bool, List[str]]:
        """Validate a test result file"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return False, [f"File not found: {result_file}"]
        
        errors = []
        warnings = []
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate field types and values
        if 'total_tests' in data:
            if not isinstance(data['total_tests'], int) or data['total_tests'] < 0:
                errors.append("total_tests must be a non-negative integer")
        
        if 'passed' in data:
            if not isinstance(data['passed'], int) or data['passed'] < 0:
                errors.append("passed must be a non-negative integer")
        
        if 'failures' in data:
            if not isinstance(data['failures'], int) or data['failures'] < 0:
                errors.append("failures must be a non-negative integer")
        
        if 'execution_time' in data:
            if not isinstance(data['execution_time'], (int, float)) or data['execution_time'] < 0:
                errors.append("execution_time must be a non-negative number")
        
        # Validate consistency
        total = data.get('total_tests', 0)
        passed = data.get('passed', 0)
        failures = data.get('failures', 0)
        errors_count = data.get('errors', 0)
        skipped = data.get('skipped', 0)
        
        calculated_total = passed + failures + errors_count + skipped
        
        if total > 0 and abs(total - calculated_total) > 0:
            warnings.append(
                f"Total mismatch: declared {total}, calculated {calculated_total}"
            )
        
        # Validate success rate
        if 'success_rate' in data and total > 0:
            expected_rate = (passed / total) * 100
            actual_rate = data['success_rate']
            if abs(expected_rate - actual_rate) > 0.1:
                warnings.append(
                    f"Success rate mismatch: declared {actual_rate:.1f}%, "
                    f"calculated {expected_rate:.1f}%"
                )
        
        # Validate timestamp
        if 'timestamp' in data:
            try:
                datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except Exception:
                errors.append("Invalid timestamp format")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors + warnings
    
    def validate_all_results(self) -> Dict:
        """Validate all result files"""
        results = {
            'valid': [],
            'invalid': [],
            'total': 0
        }
        
        for result_file in self.results_dir.glob("*.json"):
            is_valid, issues = self.validate_result(result_file.name)
            
            if is_valid:
                results['valid'].append({
                    'file': result_file.name,
                    'warnings': [i for i in issues if 'mismatch' in i.lower() or 'warning' in i.lower()]
                })
            else:
                results['invalid'].append({
                    'file': result_file.name,
                    'errors': issues
                })
            
            results['total'] += 1
        
        return results
    
    def generate_validation_report(self, validation: Dict) -> str:
        """Generate validation report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST RESULT VALIDATION")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Total Files: {validation['total']}")
        lines.append(f"Valid: {len(validation['valid'])}")
        lines.append(f"Invalid: {len(validation['invalid'])}")
        lines.append("")
        
        if validation['invalid']:
            lines.append("❌ INVALID FILES")
            lines.append("-" * 80)
            for item in validation['invalid']:
                lines.append(f"\n{item['file']}")
                for error in item['errors']:
                    lines.append(f"  • {error}")
        
        if validation['valid']:
            lines.append("\n✅ VALID FILES")
            lines.append("-" * 80)
            for item in validation['valid']:
                if item['warnings']:
                    lines.append(f"\n{item['file']} (with warnings)")
                    for warning in item['warnings']:
                        lines.append(f"  ⚠️  {warning}")
                else:
                    lines.append(f"✅ {item['file']}")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    import sys
    
    project_root = Path(__file__).parent.parent
    validator = ResultValidator(project_root)
    
    if len(sys.argv) > 1:
        # Validate specific file
        result_file = sys.argv[1]
        is_valid, issues = validator.validate_result(result_file)
        
        if is_valid:
            print(f"✅ {result_file} is valid")
            if issues:
                print("Warnings:")
                for issue in issues:
                    print(f"  ⚠️  {issue}")
        else:
            print(f"❌ {result_file} is invalid")
            for issue in issues:
                print(f"  • {issue}")
    else:
        # Validate all files
        validation = validator.validate_all_results()
        report = validator.generate_validation_report(validation)
        print(report)

if __name__ == "__main__":
    main()







