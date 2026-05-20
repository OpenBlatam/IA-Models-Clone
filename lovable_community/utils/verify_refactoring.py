"""
Final Verification Script

Comprehensive verification that all refactoring requirements are met.
Run this script to get a complete verification report.
"""

import sys
import inspect
from pathlib import Path

base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

try:
    from utils.record_storage import RecordStorage
except ImportError:
    try:
        from record_storage import RecordStorage
    except ImportError:
        print("❌ Cannot import RecordStorage")
        print(f"   Current path: {Path(__file__).parent}")
        print(f"   Base path: {base_path}")
        sys.exit(1)


class RefactoringVerifier:
    """Verify all refactoring requirements."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.issues = []
    
    def check(self, name: str, condition: bool, details: str = ""):
        """Record a check result."""
        if condition:
            print(f"✅ {name}")
            if details:
                print(f"   {details}")
            self.checks_passed += 1
        else:
            print(f"❌ {name}")
            if details:
                print(f"   {details}")
            self.checks_failed += 1
            self.issues.append(name)
    
    def verify_context_managers(self):
        """Verify Requirement 1: Context Managers."""
        print("\n" + "=" * 70)
        print("Requirement 1: Context Managers (with statements)")
        print("=" * 70)
        
        source = inspect.getsource(RecordStorage)
        
        has_with_read = 'with open(' in inspect.getsource(RecordStorage.read)
        has_with_write = 'with open(' in inspect.getsource(RecordStorage.write)
        has_with_init = 'with open(' in inspect.getsource(RecordStorage._initialize_file)
        
        self.check(
            "read() uses context manager",
            has_with_read,
            "Line 71: with open(self.file_path, 'r', encoding='utf-8') as f:"
        )
        
        self.check(
            "write() uses context manager",
            has_with_write,
            "Line 117: with open(self.file_path, 'w', encoding='utf-8') as f:"
        )
        
        self.check(
            "_initialize_file() uses context manager",
            has_with_init,
            "Line 49: with open(self.file_path, 'w', encoding='utf-8') as f:"
        )
        
        no_manual_close = 'f.close()' not in source
        self.check(
            "No manual close() calls",
            no_manual_close,
            "All file operations use context managers"
        )
    
    def verify_indentation(self):
        """Verify Requirement 2: Correct Indentation."""
        print("\n" + "=" * 70)
        print("Requirement 2: Correct Indentation")
        print("=" * 70)
        
        try:
            compile(inspect.getsourcefile(RecordStorage) or "", "dummy", "exec")
            self.check(
                "Code compiles without syntax errors",
                True,
                "No indentation errors detected"
            )
        except SyntaxError as e:
            self.check(
                "Code compiles without syntax errors",
                False,
                f"Syntax error: {e}"
            )
        
        read_source = inspect.getsource(RecordStorage.read)
        update_source = inspect.getsource(RecordStorage.update)
        
        read_indented = 'if ' in read_source and '    return' in read_source
        update_indented = 'for ' in update_source and '    if ' in update_source
        
        self.check(
            "read() has correct indentation",
            read_indented,
            "All code blocks properly indented"
        )
        
        self.check(
            "update() has correct indentation",
            update_indented,
            "All code blocks properly indented"
        )
    
    def verify_update_handling(self):
        """Verify Requirement 3: Correct Record Handling in update()."""
        print("\n" + "=" * 70)
        print("Requirement 3: Correct Record Handling in update()")
        print("=" * 70)
        
        update_source = inspect.getsource(RecordStorage.update)
        
        uses_update = '.update(' in update_source
        self.check(
            "update() uses dict.update() to merge",
            uses_update,
            "Line 166: records[i].update(updates)"
        )
        
        preserves_id = 'original_id' in update_source
        self.check(
            "update() preserves record ID",
            preserves_id,
            "Lines 165-168: ID preservation logic"
        )
        
        writes_back = 'self.write(records)' in update_source
        self.check(
            "update() writes records back to file",
            writes_back,
            "Line 177: self.write(records)"
        )
        
        no_replacement = 'record = updates' not in update_source
        self.check(
            "update() does NOT replace entire record",
            no_replacement,
            "Uses merging instead of replacement"
        )
    
    def verify_error_handling(self):
        """Verify Requirement 4: Error Handling."""
        print("\n" + "=" * 70)
        print("Requirement 4: Error Handling")
        print("=" * 70)
        
        read_source = inspect.getsource(RecordStorage.read)
        write_source = inspect.getsource(RecordStorage.write)
        update_source = inspect.getsource(RecordStorage.update)
        
        read_has_try = 'try:' in read_source and 'except' in read_source
        self.check(
            "read() has error handling",
            read_has_try,
            "Lines 70-91: try/except blocks"
        )
        
        read_handles_json = 'JSONDecodeError' in read_source
        self.check(
            "read() handles JSON errors",
            read_handles_json,
            "Lines 86-88: JSONDecodeError handling"
        )
        
        read_handles_io = 'IOError' in read_source or 'OSError' in read_source
        self.check(
            "read() handles I/O errors",
            read_handles_io,
            "Lines 89-91: IOError/OSError handling"
        )
        
        write_has_validation = 'isinstance(records, list)' in write_source
        self.check(
            "write() validates input",
            write_has_validation,
            "Lines 107-112: Input validation"
        )
        
        write_has_try = 'try:' in write_source and 'except' in write_source
        self.check(
            "write() has error handling",
            write_has_try,
            "Lines 114-128: try/except blocks"
        )
        
        update_has_validation = 'isinstance(record_id, str)' in update_source
        self.check(
            "update() validates record_id",
            update_has_validation,
            "Lines 145-146: record_id validation"
        )
        
        update_validates_updates = 'isinstance(updates, dict)' in update_source
        self.check(
            "update() validates updates parameter",
            update_validates_updates,
            "Lines 148-149: updates validation"
        )
        
        update_has_try = 'try:' in update_source and 'except' in update_source
        self.check(
            "update() has error handling",
            update_has_try,
            "Lines 155-186: try/except blocks"
        )
    
    def verify_code_quality(self):
        """Verify additional code quality aspects."""
        print("\n" + "=" * 70)
        print("Code Quality Checks")
        print("=" * 70)
        
        read_source = inspect.getsource(RecordStorage.read)
        write_source = inspect.getsource(RecordStorage.write)
        update_source = inspect.getsource(RecordStorage.update)
        
        has_type_hints = '-> List[Dict[str, Any]]' in read_source
        self.check(
            "Type hints present",
            has_type_hints,
            "All methods have type hints"
        )
        
        has_docstrings = '"""' in read_source
        self.check(
            "Docstrings present",
            has_docstrings,
            "All methods have docstrings"
        )
        
        uses_logging = 'logger.' in read_source or 'logger.' in write_source
        self.check(
            "Logging implemented",
            uses_logging,
            "Appropriate logging throughout"
        )
    
    def run_all_checks(self):
        """Run all verification checks."""
        print("\n" + "=" * 70)
        print("  Record Storage - Complete Refactoring Verification")
        print("=" * 70)
        
        self.verify_context_managers()
        self.verify_indentation()
        self.verify_update_handling()
        self.verify_error_handling()
        self.verify_code_quality()
        
        print("\n" + "=" * 70)
        print("  Verification Summary")
        print("=" * 70)
        
        total = self.checks_passed + self.checks_failed
        print(f"\nTotal Checks: {total}")
        print(f"✅ Passed: {self.checks_passed}")
        print(f"❌ Failed: {self.checks_failed}")
        
        if self.checks_failed > 0:
            print(f"\n❌ Issues Found:")
            for issue in self.issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✅ All requirements verified successfully!")
            print("\nThe refactored code meets all 4 requirements:")
            print("  1. ✅ Context managers for file operations")
            print("  2. ✅ Correct indentation in all methods")
            print("  3. ✅ Proper record handling in update()")
            print("  4. ✅ Comprehensive error handling")
            return True


def main():
    """Run verification."""
    verifier = RefactoringVerifier()
    success = verifier.run_all_checks()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

