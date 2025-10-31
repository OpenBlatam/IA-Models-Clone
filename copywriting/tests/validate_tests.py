#!/usr/bin/env python3
"""
Comprehensive test validation script for copywriting service.
This script validates all tests without requiring pytest installation.
"""
import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback


class TestValidator:
    """Validates test files and structure."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.success_count = 0
        
    def validate_all_tests(self) -> Dict[str, Any]:
        """Validate all test files."""
        print("🔍 Validating copywriting service tests...")
        print("=" * 60)
        
        # Find all test files
        test_files = self._find_test_files()
        
        if not test_files:
            print("❌ No test files found!")
            return {'success': False, 'errors': ['No test files found']}
        
        print(f"📁 Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  - {test_file.relative_to(self.test_dir)}")
        
        print("\n🔧 Validating test files...")
        print("-" * 40)
        
        # Validate each test file
        for test_file in test_files:
            self._validate_test_file(test_file)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Successful validations: {self.success_count}")
        print(f"❌ Errors: {len(self.errors)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        success = len(self.errors) == 0
        print(f"\n🎯 Overall Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return {
            'success': success,
            'success_count': self.success_count,
            'errors': self.errors,
            'warnings': self.warnings,
            'total_files': len(test_files)
        }
    
    def _find_test_files(self) -> List[Path]:
        """Find all test files."""
        test_files = []
        
        # Look for test files in all subdirectories
        for pattern in ["test_*.py", "**/test_*.py"]:
            test_files.extend(self.test_dir.glob(pattern))
        
        # Remove duplicates and sort
        test_files = sorted(list(set(test_files)))
        
        return test_files
    
    def _validate_test_file(self, test_file: Path):
        """Validate a single test file."""
        print(f"\n🔍 Validating: {test_file.relative_to(self.test_dir)}")
        
        try:
            # Check file exists and is readable
            if not test_file.exists():
                self.errors.append(f"{test_file}: File does not exist")
                return
            
            if not test_file.is_file():
                self.errors.append(f"{test_file}: Not a file")
                return
            
            # Read file content
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                self.warnings.append(f"{test_file}: File is empty")
                return
            
            # Parse Python syntax
            try:
                tree = ast.parse(content, filename=str(test_file))
            except SyntaxError as e:
                self.errors.append(f"{test_file}: Syntax error - {e}")
                return
            
            # Validate test structure
            self._validate_test_structure(test_file, tree)
            
            # Validate imports
            self._validate_imports(test_file, tree)
            
            # Validate test functions
            self._validate_test_functions(test_file, tree)
            
            # Try to import the module
            self._validate_import(test_file)
            
            self.success_count += 1
            print(f"  ✅ Valid")
            
        except Exception as e:
            self.errors.append(f"{test_file}: Validation error - {e}")
            print(f"  ❌ Error: {e}")
    
    def _validate_test_structure(self, test_file: Path, tree: ast.AST):
        """Validate test file structure."""
        # Check for test classes and functions
        has_test_class = False
        has_test_function = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    has_test_class = True
            elif isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    has_test_function = True
        
        if not has_test_class and not has_test_function:
            self.warnings.append(f"{test_file}: No test classes or functions found")
    
    def _validate_imports(self, test_file: Path, tree: ast.AST):
        """Validate imports in test file."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        # Check for common test imports
        required_imports = ['pytest']
        missing_imports = []
        
        for required in required_imports:
            if not any(required in imp for imp in imports):
                missing_imports.append(required)
        
        if missing_imports:
            self.warnings.append(f"{test_file}: Missing imports: {', '.join(missing_imports)}")
    
    def _validate_test_functions(self, test_file: Path, tree: ast.AST):
        """Validate test functions."""
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_functions.append(node.name)
                
                # Check function has proper signature
                if not node.args.args:
                    self.warnings.append(f"{test_file}: Test function '{node.name}' has no parameters")
        
        if not test_functions:
            self.warnings.append(f"{test_file}: No test functions found")
        else:
            print(f"    📝 Found {len(test_functions)} test functions: {', '.join(test_functions)}")
    
    def _validate_import(self, test_file: Path):
        """Try to import the test module."""
        try:
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            # Get module name
            relative_path = test_file.relative_to(self.test_dir)
            module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
            
            # Try to import
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"    ✅ Import successful")
            else:
                self.warnings.append(f"{test_file}: Could not create module spec")
                
        except Exception as e:
            self.warnings.append(f"{test_file}: Import error - {e}")
    
    def validate_test_structure(self) -> Dict[str, Any]:
        """Validate overall test structure."""
        print("\n🏗️  Validating test structure...")
        print("-" * 40)
        
        structure_issues = []
        
        # Check for required directories
        required_dirs = ['unit', 'integration']
        for dir_name in required_dirs:
            dir_path = self.test_dir / dir_name
            if not dir_path.exists():
                structure_issues.append(f"Missing directory: {dir_name}")
            elif not any(dir_path.iterdir()):
                structure_issues.append(f"Empty directory: {dir_name}")
        
        # Check for conftest.py
        conftest_path = self.test_dir / 'conftest.py'
        if not conftest_path.exists():
            structure_issues.append("Missing conftest.py")
        
        # Check for pytest.ini
        pytest_ini_path = self.test_dir.parent / 'pytest.ini'
        if not pytest_ini_path.exists():
            structure_issues.append("Missing pytest.ini")
        
        if structure_issues:
            print("❌ Structure issues found:")
            for issue in structure_issues:
                print(f"  - {issue}")
        else:
            print("✅ Test structure is valid")
        
        return {
            'valid': len(structure_issues) == 0,
            'issues': structure_issues
        }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a test validation report."""
        report = f"""
# Test Validation Report

## Summary
- **Status**: {'✅ PASSED' if results['success'] else '❌ FAILED'}
- **Files Validated**: {results['total_files']}
- **Successful Validations**: {results['success_count']}
- **Errors**: {len(results['errors'])}
- **Warnings**: {len(results['warnings'])}

## Errors
"""
        
        if results['errors']:
            for error in results['errors']:
                report += f"- {error}\n"
        else:
            report += "- None\n"
        
        report += "\n## Warnings\n"
        
        if results['warnings']:
            for warning in results['warnings']:
                report += f"- {warning}\n"
        else:
            report += "- None\n"
        
        return report


def main():
    """Main entry point."""
    validator = TestValidator()
    
    # Validate test structure
    structure_results = validator.validate_test_structure()
    
    # Validate all test files
    validation_results = validator.validate_all_tests()
    
    # Generate report
    report = validator.generate_test_report(validation_results)
    
    # Save report
    report_file = validator.test_dir / 'validation_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Validation report saved to: {report_file}")
    
    # Exit with error code if validation failed
    if not validation_results['success']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()


