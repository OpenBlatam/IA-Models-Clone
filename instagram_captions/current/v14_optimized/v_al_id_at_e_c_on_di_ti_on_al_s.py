from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Conditional Statement Validator - Instagram Captions API v14.0
Validates that all conditional statements follow Python best practices
"""


class ConditionalValidator:
    """Validates conditional statements in Python code"""
    
    def __init__(self, root_dir: str):
        
    """__init__ function."""
self.root_dir = Path(root_dir)
        self.issues = []
        self.stats = {
            "files_checked": 0,
            "total_conditionals": 0,
            "issues_found": 0
        }
    
    def validate_file(self, file_path: Path) -> List[Dict]:
        """Validate a single Python file"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Check for curly braces (should not exist in Python)
            curly_brace_pattern = r'if\s*\([^)]*\)\s*\{'
            curly_matches = re.finditer(curly_brace_pattern, content)
            for match in curly_matches:
                issues.append({
                    "type": "curly_braces",
                    "line": content[:match.start()].count('\n') + 1,
                    "message": "Found curly braces in conditional statement - use Python syntax",
                    "code": match.group()
                })
            
            # Check for unnecessary parentheses around simple conditions
            unnecessary_parens_pattern = r'if\s*\(([a-zA-Z_][a-zA-Z0-9_]*)\):'
            paren_matches = re.finditer(unnecessary_parens_pattern, content)
            for match in paren_matches:
                issues.append({
                    "type": "unnecessary_parentheses",
                    "line": content[:match.start()].count('\n') + 1,
                    "message": f"Unnecessary parentheses around simple condition: {match.group(1)}",
                    "code": match.group()
                })
            
            # Parse AST for more detailed analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path, issues)
            except SyntaxError as e:
                issues.append({
                    "type": "syntax_error",
                    "line": e.lineno,
                    "message": f"Syntax error: {e.msg}",
                    "code": "N/A"
                })
            
        except Exception as e:
            issues.append({
                "type": "file_error",
                "line": 0,
                "message": f"Error reading file: {e}",
                "code": "N/A"
            })
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, issues: List[Dict]):
        """Analyze AST for conditional statement issues"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                self._check_if_statement(node, file_path, issues)
            elif isinstance(node, ast.Compare):
                self._check_comparison(node, file_path, issues)
    
    def _check_if_statement(self, node: ast.If, file_path: Path, issues: List[Dict]):
        """Check individual if statement"""
        # Check for unnecessary parentheses around simple conditions
        if isinstance(node.test, ast.Name):
            issues.append({
                "type": "simple_condition",
                "line": node.lineno,
                "message": "Simple condition - ensure no unnecessary parentheses",
                "code": f"if {ast.unparse(node.test)}:"
            })
        
        # Check for complex conditions that might need parentheses
        if isinstance(node.test, ast.BoolOp):
            # Check if parentheses are needed for operator precedence
            if isinstance(node.test.op, ast.Or) and any(isinstance(val, ast.And) for val in node.test.values):
                issues.append({
                    "type": "complex_condition",
                    "line": node.lineno,
                    "message": "Complex boolean condition - consider adding parentheses for clarity",
                    "code": f"if {ast.unparse(node.test)}:"
                })
    
    def _check_comparison(self, node: ast.Compare, file_path: Path, issues: List[Dict]):
        """Check comparison operations"""
        # Check for chained comparisons that might be unclear
        if len(node.ops) > 2:
            issues.append({
                "type": "chained_comparison",
                "line": node.lineno,
                "message": "Chained comparison - consider breaking into multiple conditions for clarity",
                "code": ast.unparse(node)
            })
    
    def validate_all(self) -> Dict:
        """Validate all Python files in the directory"""
        python_files = list(self.root_dir.rglob("*.py"))
        
        for file_path in python_files:
            if "venv" in str(file_path) or "__pycache__" in str(file_path):
                continue
            
            self.stats["files_checked"] += 1
            file_issues = self.validate_file(file_path)
            
            if file_issues:
                self.issues.append({
                    "file": str(file_path.relative_to(self.root_dir)),
                    "issues": file_issues
                })
                self.stats["issues_found"] += len(file_issues)
        
        return {
            "stats": self.stats,
            "issues": self.issues
        }
    
    def print_report(self, results: Dict):
        """Print validation report"""
        print("=" * 80)
        print("CONDITIONAL STATEMENT VALIDATION REPORT")
        print("=" * 80)
        
        stats = results["stats"]
        print(f"\n📊 STATISTICS:")
        print(f"   Files checked: {stats['files_checked']}")
        print(f"   Issues found: {stats['issues_found']}")
        
        if results["issues"]:
            print(f"\n❌ ISSUES FOUND:")
            for file_result in results["issues"]:
                print(f"\n📁 {file_result['file']}:")
                for issue in file_result["issues"]:
                    print(f"   Line {issue['line']}: {issue['type']}")
                    print(f"   {issue['message']}")
                    print(f"   Code: {issue['code']}")
                    print()
        else:
            print(f"\n✅ NO ISSUES FOUND!")
            print("All conditional statements follow Python best practices.")
        
        print("=" * 80)

def main():
    """Main validation function"""
    # Get the current directory (v14_optimized)
    current_dir = Path(__file__).parent
    
    print("🔍 Validating conditional statements in Instagram Captions API v14.0...")
    
    validator = ConditionalValidator(current_dir)
    results = validator.validate_all()
    validator.print_report(results)
    
    # Exit with error code if issues found
    if results["stats"]["issues_found"] > 0:
        print(f"\n⚠️  {results['stats']['issues_found']} issues found. Please review and fix.")
        exit(1)
    else:
        print(f"\n🎉 All conditional statements are properly formatted!")
        exit(0)

match __name__:
    case "__main__":
    main() 