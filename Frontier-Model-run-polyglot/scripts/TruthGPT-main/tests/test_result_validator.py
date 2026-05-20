"""
Test Result Validator
Validates test result files for correctness and completeness
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import jsonschema


class TestResultValidator:
    """Validate test result files"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.schema = self._get_schema()
    
    def _get_schema(self) -> Dict:
        """Get JSON schema for test results"""
        return {
            "type": "object",
            "required": ["timestamp", "run_name"],
            "properties": {
                "timestamp": {
                    "type": "string",
                    "format": "date-time"
                },
                "run_name": {
                    "type": "string"
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "total_tests": {"type": "integer", "minimum": 0},
                        "passed": {"type": "integer", "minimum": 0},
                        "failed": {"type": "integer", "minimum": 0},
                        "errors": {"type": "integer", "minimum": 0},
                        "skipped": {"type": "integer", "minimum": 0},
                        "success_rate": {"type": "number", "minimum": 0, "maximum": 100},
                        "execution_time": {"type": "number", "minimum": 0}
                    }
                },
                "test_details": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["passed", "failed", "error", "skipped", "unknown"]
                            },
                            "duration": {"type": "number", "minimum": 0},
                            "error_message": {"type": "string"},
                            "test_file": {"type": "string"},
                            "test_class": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    def validate_structure(self, results: Dict) -> Tuple[bool, List[str]]:
        """Validate result structure"""
        errors = []
        
        # Check required fields
        if 'timestamp' not in results:
            errors.append("Missing required field: timestamp")
        if 'run_name' not in results:
            errors.append("Missing required field: run_name")
        
        # Validate timestamp format
        if 'timestamp' in results:
            try:
                datetime.fromisoformat(results['timestamp'])
            except Exception:
                errors.append("Invalid timestamp format")
        
        # Validate summary
        if 'summary' in results:
            summary = results['summary']
            required_summary_fields = ['total_tests', 'passed', 'failed']
            for field in required_summary_fields:
                if field not in summary:
                    errors.append(f"Missing summary field: {field}")
        
        # Validate test details
        if 'test_details' in results:
            test_details = results['test_details']
            if not isinstance(test_details, dict):
                errors.append("test_details must be a dictionary")
            else:
                for test_name, test_data in test_details.items():
                    if not isinstance(test_data, dict):
                        errors.append(f"Test data for '{test_name}' must be a dictionary")
                    else:
                        if 'status' not in test_data:
                            errors.append(f"Test '{test_name}' missing status")
                        elif test_data['status'] not in ['passed', 'failed', 'error', 'skipped', 'unknown']:
                            errors.append(f"Test '{test_name}' has invalid status: {test_data['status']}")
        
        return len(errors) == 0, errors
    
    def validate_consistency(self, results: Dict) -> Tuple[bool, List[str]]:
        """Validate consistency between summary and details"""
        errors = []
        
        summary = results.get('summary', {})
        test_details = results.get('test_details', {})
        
        if not summary or not test_details:
            return True, []  # Can't validate if missing
        
        # Count tests
        total_tests = len(test_details)
        if summary.get('total_tests', 0) != total_tests:
            errors.append(
                f"Summary total_tests ({summary.get('total_tests')}) "
                f"doesn't match test_details count ({total_tests})"
            )
        
        # Count by status
        status_counts = {'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0}
        for test_data in test_details.values():
            status = test_data.get('status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1
        
        if summary.get('passed', 0) != status_counts['passed']:
            errors.append(
                f"Summary passed ({summary.get('passed')}) "
                f"doesn't match actual passed count ({status_counts['passed']})"
            )
        
        if summary.get('failed', 0) != status_counts['failed']:
            errors.append(
                f"Summary failed ({summary.get('failed')}) "
                f"doesn't match actual failed count ({status_counts['failed']})"
            )
        
        # Validate success rate
        if total_tests > 0:
            calculated_rate = (status_counts['passed'] / total_tests) * 100
            reported_rate = summary.get('success_rate', 0)
            if abs(calculated_rate - reported_rate) > 0.1:  # Allow small rounding differences
                errors.append(
                    f"Success rate mismatch: calculated {calculated_rate:.1f}%, "
                    f"reported {reported_rate:.1f}%"
                )
        
        return len(errors) == 0, errors
    
    def validate_with_schema(self, results: Dict) -> Tuple[bool, List[str]]:
        """Validate using JSON schema"""
        try:
            jsonschema.validate(instance=results, schema=self.schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [f"Schema validation error: {e.message}"]
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def validate_file(self, result_file: Path) -> Dict:
        """Validate a result file"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'errors': [f"Invalid JSON: {str(e)}"],
                'warnings': []
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Error reading file: {str(e)}"],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        # Structure validation
        struct_valid, struct_errors = self.validate_structure(results)
        errors.extend(struct_errors)
        
        # Consistency validation
        if struct_valid:
            cons_valid, cons_errors = self.validate_consistency(results)
            errors.extend(cons_errors)
        
        # Schema validation
        schema_valid, schema_errors = self.validate_with_schema(results)
        if not schema_valid:
            warnings.extend(schema_errors)  # Schema errors as warnings
        
        # Additional checks
        if 'test_details' in results and len(results['test_details']) == 0:
            warnings.append("No test details found")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'file': str(result_file)
        }
    
    def validate_directory(self, directory: Path) -> Dict:
        """Validate all result files in a directory"""
        result_files = list(directory.glob("*.json"))
        
        results = {
            'total_files': len(result_files),
            'valid_files': 0,
            'invalid_files': 0,
            'files': []
        }
        
        for result_file in result_files:
            validation = self.validate_file(result_file)
            results['files'].append(validation)
            
            if validation['valid']:
                results['valid_files'] += 1
            else:
                results['invalid_files'] += 1
        
        return results


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Validator')
    parser.add_argument('--file', type=str, help='Validate a single file')
    parser.add_argument('--directory', type=str, help='Validate all files in directory')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    validator = TestResultValidator(project_root)
    
    if args.file:
        print(f"🔍 Validating file: {args.file}")
        result = validator.validate_file(Path(args.file))
        
        if result['valid']:
            print("✅ File is valid")
            if result['warnings']:
                print("⚠️ Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
        else:
            print("❌ File is invalid")
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
    
    elif args.directory:
        print(f"🔍 Validating directory: {args.directory}")
        results = validator.validate_directory(Path(args.directory))
        
        print(f"\n📊 Validation Results:")
        print(f"  Total Files: {results['total_files']}")
        print(f"  Valid: {results['valid_files']}")
        print(f"  Invalid: {results['invalid_files']}")
        
        if results['invalid_files'] > 0:
            print("\n❌ Invalid Files:")
            for file_result in results['files']:
                if not file_result['valid']:
                    print(f"  {file_result['file']}:")
                    for error in file_result['errors']:
                        print(f"    - {error}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

