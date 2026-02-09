"""
Enhanced Validator
Enhanced validation with detailed checks and suggestions
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class EnhancedValidator:
    """Enhanced validation with detailed checks"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def validate_enhanced(self, result_file: str) -> Dict:
        """Enhanced validation with detailed checks"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'valid': False, 'errors': [f'File not found: {result_file}']}
        
        errors = []
        warnings = []
        suggestions = []
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {'valid': False, 'errors': [f'Invalid JSON: {e}']}
        except Exception as e:
            return {'valid': False, 'errors': [f'Error reading file: {e}']}
        
        # Basic structure validation
        required_fields = ['total_tests', 'timestamp']
        for field in required_fields:
            if field not in data:
                errors.append(f'Missing required field: {field}')
        
        # Type validation
        type_checks = {
            'total_tests': (int, lambda x: x >= 0),
            'passed': (int, lambda x: x >= 0),
            'failures': (int, lambda x: x >= 0),
            'errors': (int, lambda x: x >= 0),
            'skipped': (int, lambda x: x >= 0),
            'execution_time': ((int, float), lambda x: x >= 0),
            'success_rate': ((int, float), lambda x: 0 <= x <= 100)
        }
        
        for field, (expected_type, validator) in type_checks.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(f'{field} must be {expected_type.__name__ if hasattr(expected_type, "__name__") else str(expected_type)}')
                elif not validator(data[field]):
                    errors.append(f'{field} has invalid value: {data[field]}')
        
        # Consistency checks
        total = data.get('total_tests', 0)
        passed = data.get('passed', 0)
        failures = data.get('failures', 0)
        errors_count = data.get('errors', 0)
        skipped = data.get('skipped', 0)
        
        calculated_total = passed + failures + errors_count + skipped
        
        if total > 0 and abs(total - calculated_total) > 0:
            warnings.append(f'Total mismatch: declared {total}, calculated {calculated_total}')
            suggestions.append('Recalculate total_tests based on passed + failures + errors + skipped')
        
        # Success rate validation
        if 'success_rate' in data and total > 0:
            expected_rate = (passed / total) * 100
            actual_rate = data['success_rate']
            if abs(expected_rate - actual_rate) > 0.1:
                warnings.append(f'Success rate mismatch: declared {actual_rate:.1f}%, calculated {expected_rate:.1f}%')
                suggestions.append(f'Update success_rate to {expected_rate:.1f}%')
        
        # Timestamp validation
        if 'timestamp' in data:
            try:
                datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except Exception:
                errors.append('Invalid timestamp format')
                suggestions.append('Use ISO 8601 format: YYYY-MM-DDTHH:MM:SS')
        
        # Test details validation
        if 'test_details' in data:
            test_details = data['test_details']
            if not isinstance(test_details, dict):
                errors.append('test_details must be a dictionary')
            else:
                for category in ['failures', 'errors', 'skipped']:
                    if category in test_details:
                        if not isinstance(test_details[category], list):
                            errors.append(f'test_details.{category} must be a list')
        
        # Quality suggestions
        if total > 0:
            success_rate = (passed / total) * 100
            if success_rate < 90:
                suggestions.append('Success rate is below 90% - consider improving test quality')
            
            if failures + errors_count > total * 0.1:
                suggestions.append('High failure rate - investigate and fix failing tests')
        
        is_valid = len(errors) == 0
        
        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'score': self._calculate_validation_score(errors, warnings)
        }
    
    def _calculate_validation_score(self, errors: List[str], warnings: List[str]) -> float:
        """Calculate validation score"""
        base_score = 100.0
        base_score -= len(errors) * 20  # -20 per error
        base_score -= len(warnings) * 5  # -5 per warning
        return max(0, min(100, base_score))
    
    def generate_validation_report(self, validation: Dict) -> str:
        """Generate enhanced validation report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        status_emoji = "✅" if validation['valid'] else "❌"
        lines.append(f"{status_emoji} Validation Status: {'VALID' if validation['valid'] else 'INVALID'}")
        lines.append(f"   Score: {validation['score']}/100")
        lines.append("")
        
        if validation['errors']:
            lines.append("❌ ERRORS")
            lines.append("-" * 80)
            for error in validation['errors']:
                lines.append(f"  • {error}")
            lines.append("")
        
        if validation['warnings']:
            lines.append("⚠️  WARNINGS")
            lines.append("-" * 80)
            for warning in validation['warnings']:
                lines.append(f"  • {warning}")
            lines.append("")
        
        if validation['suggestions']:
            lines.append("💡 SUGGESTIONS")
            lines.append("-" * 80)
            for suggestion in validation['suggestions']:
                lines.append(f"  • {suggestion}")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_validator.py <file.json>")
        return
    
    project_root = Path(__file__).parent.parent
    validator = EnhancedValidator(project_root)
    
    result_file = sys.argv[1]
    validation = validator.validate_enhanced(result_file)
    
    report = validator.generate_validation_report(validation)
    print(report)

if __name__ == "__main__":
    main()







