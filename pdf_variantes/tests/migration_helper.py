"""
Migration Helper
================
Helper script to migrate existing tests to use new refactored base classes.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


class TestMigrator:
    """Helper to migrate tests to new base classes."""
    
    # Mapping of old patterns to new patterns
    MIGRATION_PATTERNS = [
        # Import statements
        (
            r"from\s+playwright_base\s+import\s+BasePlaywrightTest",
            "from playwright_base_unified import BasePlaywrightTest"
        ),
        (
            r"from\s+base_playwright_test\s+import\s+BasePlaywrightTest",
            "from playwright_base_unified import BasePlaywrightTest"
        ),
        (
            r"from\s+base_playwright_test\s+import\s+BaseAPITest",
            "from playwright_base_unified import BaseAPITest"
        ),
        # Method calls
        (
            r"self\.make_request\(page,\s*['\"](GET|POST|PUT|DELETE)['\"],\s*([^,]+),\s*\*\*kwargs\)",
            r"self.make_request_simple(page, '\1', \2, self.api_base_url, self.auth_headers)"
        ),
        # Upload patterns
        (
            r"files\s*=\s*\{[^}]+\}\s*response\s*=\s*page\.request\.post\([^)]+upload[^)]+\)",
            "result = self.upload_pdf(page, api_base_url, filename, content, auth_headers)"
        ),
    ]
    
    def __init__(self, test_file: Path):
        self.test_file = test_file
        self.content = test_file.read_text()
        self.changes: List[str] = []
    
    def analyze(self) -> Dict[str, any]:
        """Analyze test file for migration opportunities."""
        issues = []
        suggestions = []
        
        # Check for old imports
        if "from playwright_base import" in self.content:
            issues.append("Uses old playwright_base import")
            suggestions.append("Migrate to playwright_base_unified")
        
        if "from base_playwright_test import" in self.content:
            issues.append("Uses old base_playwright_test import")
            suggestions.append("Migrate to playwright_base_unified")
        
        # Check for manual request construction
        if re.search(r"page\.request\.(get|post|put|delete)\([^)]+api_base_url", self.content):
            issues.append("Uses manual request construction")
            suggestions.append("Use make_request_simple() from base class")
        
        # Check for manual upload
        if re.search(r"multipart\s*=\s*\{[^}]+\}", self.content):
            issues.append("Uses manual file upload")
            suggestions.append("Use upload_pdf() from BaseAPITest")
        
        # Check for manual assertions
        if re.search(r"assert\s+response\.status\s*==\s*\d+", self.content):
            issues.append("Uses manual status assertions")
            suggestions.append("Use assert_success() from base class")
        
        return {
            "file": str(self.test_file),
            "issues": issues,
            "suggestions": suggestions,
            "can_migrate": len(issues) > 0
        }
    
    def migrate(self, dry_run: bool = True) -> List[str]:
        """Migrate test file to use new base classes."""
        new_content = self.content
        changes = []
        
        # Migrate imports
        for old_pattern, new_pattern in self.MIGRATION_PATTERNS:
            if re.search(old_pattern, new_content):
                new_content = re.sub(old_pattern, new_pattern, new_content)
                changes.append(f"Updated import: {old_pattern} -> {new_pattern}")
        
        # Suggest class changes
        if "class Test" in new_content and "BasePlaywrightTest" not in new_content:
            # Try to detect what base class should be used
            if "upload" in new_content.lower() or "variant" in new_content.lower():
                changes.append("SUGGESTION: Consider using BaseAPITest instead of BasePlaywrightTest")
            if "performance" in new_content.lower() or "response_time" in new_content.lower():
                changes.append("SUGGESTION: Consider using BasePerformanceTest")
            if "workflow" in new_content.lower():
                changes.append("SUGGESTION: Consider using BaseWorkflowTest")
        
        if not dry_run:
            self.test_file.write_text(new_content)
            self.changes = changes
        
        return changes
    
    def generate_migration_report(self) -> str:
        """Generate migration report."""
        analysis = self.analyze()
        
        report = f"""
Migration Report for {self.test_file.name}
{'=' * 60}

Issues Found: {len(analysis['issues'])}
"""
        
        if analysis['issues']:
            report += "\nIssues:\n"
            for issue in analysis['issues']:
                report += f"  - {issue}\n"
        
        if analysis['suggestions']:
            report += "\nSuggestions:\n"
            for suggestion in analysis['suggestions']:
                report += f"  - {suggestion}\n"
        
        return report


def find_test_files(directory: Path) -> List[Path]:
    """Find all test files in directory."""
    return list(directory.glob("test_playwright*.py"))


def migrate_directory(directory: Path, dry_run: bool = True) -> Dict[str, any]:
    """Migrate all test files in directory."""
    test_files = find_test_files(directory)
    results = {
        "total_files": len(test_files),
        "migrated": 0,
        "needs_migration": 0,
        "reports": []
    }
    
    for test_file in test_files:
        migrator = TestMigrator(test_file)
        analysis = migrator.analyze()
        
        if analysis['can_migrate']:
            results['needs_migration'] += 1
            changes = migrator.migrate(dry_run=dry_run)
            
            if changes:
                results['migrated'] += 1
                results['reports'].append({
                    "file": str(test_file),
                    "changes": changes,
                    "analysis": analysis
                })
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Playwright tests to new base classes")
    parser.add_argument("--directory", default=".", help="Directory containing test files")
    parser.add_argument("--file", help="Specific file to migrate")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run (don't modify files)")
    parser.add_argument("--execute", action="store_true", help="Actually migrate files")
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    if args.file:
        # Migrate single file
        test_file = Path(args.file)
        migrator = TestMigrator(test_file)
        
        print(migrator.generate_migration_report())
        
        if args.execute:
            changes = migrator.migrate(dry_run=False)
            print(f"\nMigrated {test_file.name}")
            for change in changes:
                print(f"  - {change}")
    else:
        # Migrate directory
        results = migrate_directory(directory, dry_run=not args.execute)
        
        print(f"\nMigration Summary")
        print(f"{'=' * 60}")
        print(f"Total files: {results['total_files']}")
        print(f"Need migration: {results['needs_migration']}")
        print(f"Migrated: {results['migrated']}")
        
        if results['reports']:
            print(f"\nDetailed Reports:")
            for report in results['reports']:
                print(f"\n{report['file']}:")
                for change in report['changes']:
                    print(f"  - {change}")



