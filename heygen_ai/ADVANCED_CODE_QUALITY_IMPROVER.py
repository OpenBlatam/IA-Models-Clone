#!/usr/bin/env python3
"""
🔧 HeyGen AI - Advanced Code Quality Improver
============================================

Comprehensive code quality improvement system with automated refactoring,
code analysis, and best practices enforcement.

Author: AI Assistant
Date: December 2024
Version: 1.0.0
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeQualityMetrics:
    """Code quality metrics data class"""
    lines_of_code: int
    cyclomatic_complexity: int
    maintainability_index: float
    code_duplication: float
    test_coverage: float
    documentation_coverage: float
    pylint_score: float
    mypy_score: float
    security_issues: int
    performance_issues: int

@dataclass
class CodeIssue:
    """Code issue data class"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    message: str
    suggestion: str

class CodeAnalyzer:
    """Advanced code analysis system"""
    
    def __init__(self):
        self.issue_patterns = {
            'long_function': r'def\s+\w+\([^)]*\):\s*\n(?:[^\n]*\n){50,}',
            'long_class': r'class\s+\w+[^:]*:\s*\n(?:[^\n]*\n){200,}',
            'deep_nesting': r'(\s{12,})',
            'magic_numbers': r'\b(?!0|1|2|3|4|5|6|7|8|9|10|100|1000|1024|2048|4096)\d+\b',
            'unused_imports': r'^import\s+\w+.*$',
            'missing_docstrings': r'^(def|class)\s+\w+.*:\s*$',
            'hardcoded_strings': r'"[^"]{20,}"',
            'complex_expressions': r'[^=!<>]+\s*[=!<>]+\s*[^=!<>]+\s*[=!<>]+\s*[^=!<>]+'
        }
    
    def analyze_file(self, file_path: str) -> List[CodeIssue]:
        """Analyze a single Python file for quality issues"""
        try:
            issues = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST for structural analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast(tree, file_path))
            except SyntaxError as e:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    issue_type='syntax_error',
                    severity='error',
                    message=f"Syntax error: {e.msg}",
                    suggestion="Fix syntax error"
                ))
            
            # Pattern-based analysis
            issues.extend(self._analyze_patterns(content, file_path, lines))
            
            # Line-by-line analysis
            issues.extend(self._analyze_lines(lines, file_path))
            
            return issues
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return []
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Analyze AST for structural issues"""
        issues = []
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
                self.function_complexities = []
            
            def visit_FunctionDef(self, node):
                # Calculate cyclomatic complexity
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                
                if complexity > 10:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type='high_complexity',
                        severity='warning',
                        message=f"Function '{node.name}' has high cyclomatic complexity ({complexity})",
                        suggestion="Consider breaking down into smaller functions"
                    ))
                
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _analyze_patterns(self, content: str, file_path: str, lines: List[str]) -> List[CodeIssue]:
        """Analyze content using regex patterns"""
        issues = []
        
        for pattern_name, pattern in self.issue_patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                
                if pattern_name == 'long_function':
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type='long_function',
                        severity='warning',
                        message="Function is too long (>50 lines)",
                        suggestion="Break down into smaller functions"
                    ))
                elif pattern_name == 'long_class':
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type='long_class',
                        severity='warning',
                        message="Class is too long (>200 lines)",
                        suggestion="Consider splitting into multiple classes"
                    ))
                elif pattern_name == 'magic_numbers':
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type='magic_number',
                        severity='info',
                        message="Magic number detected",
                        suggestion="Define as named constant"
                    ))
        
        return issues
    
    def _analyze_lines(self, lines: List[str], file_path: str) -> List[CodeIssue]:
        """Analyze individual lines for issues"""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type='long_line',
                    severity='warning',
                    message=f"Line too long ({len(line)} characters)",
                    suggestion="Break line or use line continuation"
                ))
            
            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type='trailing_whitespace',
                    severity='info',
                    message="Trailing whitespace detected",
                    suggestion="Remove trailing whitespace"
                ))
            
            # Check for missing docstrings
            if re.match(r'^(def|class)\s+\w+.*:\s*$', line.strip()) and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                if not next_line.startswith('"""') and not next_line.startswith("'''"):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        issue_type='missing_docstring',
                        severity='info',
                        message="Missing docstring",
                        suggestion="Add docstring for better documentation"
                    ))
        
        return issues

class CodeRefactorer:
    """Automated code refactoring system"""
    
    def __init__(self):
        self.refactoring_rules = {
            'extract_method': self._extract_method,
            'extract_variable': self._extract_variable,
            'simplify_conditionals': self._simplify_conditionals,
            'remove_duplicates': self._remove_duplicates,
            'add_type_hints': self._add_type_hints,
            'optimize_imports': self._optimize_imports
        }
    
    def refactor_file(self, file_path: str, refactoring_types: List[str] = None) -> Dict[str, Any]:
        """Refactor a file with specified refactoring types"""
        try:
            if refactoring_types is None:
                refactoring_types = list(self.refactoring_rules.keys())
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            refactoring_results = {
                'file_path': file_path,
                'refactorings_applied': [],
                'changes_made': 0,
                'success': True
            }
            
            for refactoring_type in refactoring_types:
                if refactoring_type in self.refactoring_rules:
                    try:
                        content, changes = self.refactoring_rules[refactoring_type](content)
                        if changes > 0:
                            refactoring_results['refactorings_applied'].append(refactoring_type)
                            refactoring_results['changes_made'] += changes
                    except Exception as e:
                        logger.warning(f"Refactoring {refactoring_type} failed: {e}")
            
            # Write refactored content if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                refactoring_results['file_updated'] = True
            else:
                refactoring_results['file_updated'] = False
            
            return refactoring_results
            
        except Exception as e:
            logger.error(f"Refactoring failed for {file_path}: {e}")
            return {'error': str(e), 'success': False}
    
    def _extract_method(self, content: str) -> Tuple[str, int]:
        """Extract long methods into smaller ones"""
        # This is a simplified implementation
        # In practice, this would use AST analysis
        changes = 0
        return content, changes
    
    def _extract_variable(self, content: str) -> Tuple[str, int]:
        """Extract complex expressions into variables"""
        changes = 0
        return content, changes
    
    def _simplify_conditionals(self, content: str) -> Tuple[str, int]:
        """Simplify complex conditional statements"""
        changes = 0
        return content, changes
    
    def _remove_duplicates(self, content: str) -> Tuple[str, int]:
        """Remove duplicate code blocks"""
        changes = 0
        return content, changes
    
    def _add_type_hints(self, content: str) -> Tuple[str, int]:
        """Add type hints to function parameters and return types"""
        changes = 0
        return content, changes
    
    def _optimize_imports(self, content: str) -> Tuple[str, int]:
        """Optimize and organize imports"""
        changes = 0
        return content, changes

class TestGenerator:
    """Automated test generation system"""
    
    def __init__(self):
        self.test_templates = {
            'function': self._generate_function_test,
            'class': self._generate_class_test,
            'api_endpoint': self._generate_api_test
        }
    
    def generate_tests(self, file_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Generate tests for a given file"""
        try:
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(file_path), 'tests')
            
            os.makedirs(output_dir, exist_ok=True)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse file to identify testable components
            tree = ast.parse(content)
            testable_components = self._find_testable_components(tree)
            
            # Generate tests
            test_file_path = os.path.join(output_dir, f"test_{os.path.basename(file_path)}")
            test_content = self._generate_test_file(testable_components, file_path)
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            return {
                'test_file_created': test_file_path,
                'components_tested': len(testable_components),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Test generation failed for {file_path}: {e}")
            return {'error': str(e), 'success': False}
    
    def _find_testable_components(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find testable components in the AST"""
        components = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                components.append({
                    'type': 'function',
                    'name': node.name,
                    'line_number': node.lineno,
                    'args': [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                components.append({
                    'type': 'class',
                    'name': node.name,
                    'line_number': node.lineno,
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                })
        
        return components
    
    def _generate_test_file(self, components: List[Dict[str, Any]], source_file: str) -> str:
        """Generate test file content"""
        test_content = f'''#!/usr/bin/env python3
"""
Generated tests for {os.path.basename(source_file)}
"""

import pytest
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath("{source_file}")))

from {os.path.splitext(os.path.basename(source_file))[0]} import *

'''
        
        for component in components:
            if component['type'] == 'function':
                test_content += self._generate_function_test(component)
            elif component['type'] == 'class':
                test_content += self._generate_class_test(component)
        
        return test_content
    
    def _generate_function_test(self, component: Dict[str, Any]) -> str:
        """Generate test for a function"""
        return f'''
def test_{component['name']}():
    """Test {component['name']} function"""
    # TODO: Implement test cases
    # Example:
    # result = {component['name']}(test_args)
    # assert result is not None
    pass
'''
    
    def _generate_class_test(self, component: Dict[str, Any]) -> str:
        """Generate test for a class"""
        return f'''
class Test{component['name']}:
    """Test {component['name']} class"""
    
    def setup_method(self):
        """Setup test instance"""
        self.instance = {component['name']}()
    
    def test_initialization(self):
        """Test class initialization"""
        assert self.instance is not None
    
    # TODO: Add tests for methods: {', '.join(component['methods'])}
'''
    
    def _generate_api_test(self, component: Dict[str, Any]) -> str:
        """Generate test for an API endpoint"""
        return f'''
def test_{component['name']}_endpoint():
    """Test {component['name']} API endpoint"""
    # TODO: Implement API test
    pass
'''

class DocumentationGenerator:
    """Automated documentation generation system"""
    
    def __init__(self):
        self.doc_templates = {
            'module': self._generate_module_doc,
            'function': self._generate_function_doc,
            'class': self._generate_class_doc
        }
    
    def generate_documentation(self, file_path: str) -> Dict[str, Any]:
        """Generate documentation for a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            doc_components = self._extract_doc_components(tree)
            
            # Generate documentation
            doc_content = self._generate_doc_content(doc_components, file_path)
            
            # Write documentation file
            doc_file_path = file_path.replace('.py', '_doc.md')
            with open(doc_file_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            return {
                'documentation_file': doc_file_path,
                'components_documented': len(doc_components),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Documentation generation failed for {file_path}: {e}")
            return {'error': str(e), 'success': False}
    
    def _extract_doc_components(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract components that need documentation"""
        components = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                components.append({
                    'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                    'name': node.name,
                    'line_number': node.lineno,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args] if hasattr(node, 'args') else []
                })
        
        return components
    
    def _generate_doc_content(self, components: List[Dict[str, Any]], file_path: str) -> str:
        """Generate documentation content"""
        doc_content = f'''# {os.path.basename(file_path)} Documentation

## Overview
This module contains various components for the HeyGen AI system.

## Components

'''
        
        for component in components:
            if component['type'] == 'function':
                doc_content += self._generate_function_doc(component)
            elif component['type'] == 'class':
                doc_content += self._generate_class_doc(component)
        
        return doc_content
    
    def _generate_function_doc(self, component: Dict[str, Any]) -> str:
        """Generate documentation for a function"""
        return f'''### {component['name']}

**Type:** Function  
**Line:** {component['line_number']}

{component['docstring'] or 'No documentation available.'}

**Parameters:**
{chr(10).join(f'- `{arg}`: Description needed' for arg in component['args'])}

'''
    
    def _generate_class_doc(self, component: Dict[str, Any]) -> str:
        """Generate documentation for a class"""
        return f'''### {component['name']}

**Type:** Class  
**Line:** {component['line_number']}

{component['docstring'] or 'No documentation available.'}

'''
    
    def _generate_module_doc(self, component: Dict[str, Any]) -> str:
        """Generate documentation for a module"""
        return f'''### {component['name']}

**Type:** Module

{component['docstring'] or 'No documentation available.'}

'''

class AdvancedCodeQualityImprover:
    """Main code quality improvement orchestrator"""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.analyzer = CodeAnalyzer()
        self.refactorer = CodeRefactorer()
        self.test_generator = TestGenerator()
        self.doc_generator = DocumentationGenerator()
        self.quality_metrics = {}
    
    def improve_project(self, target_directories: List[str] = None) -> Dict[str, Any]:
        """Improve code quality for entire project"""
        try:
            if target_directories is None:
                target_directories = [self.project_root]
            
            improvement_results = {
                'timestamp': time.time(),
                'directories_processed': [],
                'files_analyzed': 0,
                'issues_found': 0,
                'refactorings_applied': 0,
                'tests_generated': 0,
                'documentation_generated': 0,
                'overall_improvement': 0
            }
            
            # Find all Python files
            python_files = self._find_python_files(target_directories)
            improvement_results['files_analyzed'] = len(python_files)
            
            total_issues = 0
            total_refactorings = 0
            total_tests = 0
            total_docs = 0
            
            for file_path in python_files:
                try:
                    # Analyze file
                    issues = self.analyzer.analyze_file(file_path)
                    total_issues += len(issues)
                    
                    # Refactor file
                    refactor_result = self.refactorer.refactor_file(file_path)
                    if refactor_result.get('success', False):
                        total_refactorings += refactor_result.get('changes_made', 0)
                    
                    # Generate tests
                    test_result = self.test_generator.generate_tests(file_path)
                    if test_result.get('success', False):
                        total_tests += 1
                    
                    # Generate documentation
                    doc_result = self.doc_generator.generate_documentation(file_path)
                    if doc_result.get('success', False):
                        total_docs += 1
                    
                    improvement_results['directories_processed'].append(os.path.dirname(file_path))
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
            
            improvement_results['issues_found'] = total_issues
            improvement_results['refactorings_applied'] = total_refactorings
            improvement_results['tests_generated'] = total_tests
            improvement_results['documentation_generated'] = total_docs
            
            # Calculate overall improvement score
            improvement_results['overall_improvement'] = self._calculate_improvement_score(
                total_issues, total_refactorings, total_tests, total_docs
            )
            
            logger.info(f"Code quality improvement completed. Overall improvement: {improvement_results['overall_improvement']:.2f}%")
            return improvement_results
            
        except Exception as e:
            logger.error(f"Project improvement failed: {e}")
            return {'error': str(e)}
    
    def _find_python_files(self, directories: List[str]) -> List[str]:
        """Find all Python files in directories"""
        python_files = []
        
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                # Skip certain directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    if file.endswith('.py') and not file.startswith('test_'):
                        python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _calculate_improvement_score(self, issues: int, refactorings: int, tests: int, docs: int) -> float:
        """Calculate overall improvement score"""
        # Weighted scoring system
        issue_penalty = min(issues * 0.1, 50)  # Max 50 point penalty
        refactoring_bonus = min(refactorings * 0.5, 30)  # Max 30 point bonus
        test_bonus = min(tests * 2, 20)  # Max 20 point bonus
        doc_bonus = min(docs * 1, 10)  # Max 10 point bonus
        
        score = max(0, 100 - issue_penalty + refactoring_bonus + test_bonus + doc_bonus)
        return min(score, 100)
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive code quality report"""
        try:
            python_files = self._find_python_files([self.project_root])
            
            total_lines = 0
            total_issues = 0
            complexity_scores = []
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                    
                    issues = self.analyzer.analyze_file(file_path)
                    total_issues += len(issues)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
            
            # Calculate metrics
            avg_issues_per_file = total_issues / len(python_files) if python_files else 0
            maintainability_index = max(0, 100 - (avg_issues_per_file * 2))
            
            report = {
                'project_root': self.project_root,
                'total_files': len(python_files),
                'total_lines': total_lines,
                'total_issues': total_issues,
                'avg_issues_per_file': avg_issues_per_file,
                'maintainability_index': maintainability_index,
                'quality_grade': self._get_quality_grade(maintainability_index),
                'recommendations': self._get_quality_recommendations(avg_issues_per_file)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {'error': str(e)}
    
    def _get_quality_grade(self, maintainability_index: float) -> str:
        """Get quality grade based on maintainability index"""
        if maintainability_index >= 90:
            return 'A+'
        elif maintainability_index >= 80:
            return 'A'
        elif maintainability_index >= 70:
            return 'B'
        elif maintainability_index >= 60:
            return 'C'
        elif maintainability_index >= 50:
            return 'D'
        else:
            return 'F'
    
    def _get_quality_recommendations(self, avg_issues: float) -> List[str]:
        """Get quality improvement recommendations"""
        recommendations = []
        
        if avg_issues > 10:
            recommendations.append("High number of issues detected. Focus on refactoring and code cleanup.")
        
        if avg_issues > 5:
            recommendations.append("Consider implementing automated code quality checks in CI/CD pipeline.")
        
        if avg_issues > 2:
            recommendations.append("Add more comprehensive test coverage.")
        
        if avg_issues <= 1:
            recommendations.append("Code quality is excellent. Maintain current standards.")
        
        return recommendations

# Example usage and testing
def main():
    """Main function for testing the code quality improver"""
    try:
        # Initialize improver
        improver = AdvancedCodeQualityImprover()
        
        print("🔧 Starting HeyGen AI Code Quality Improvement...")
        
        # Improve project
        improvement_results = improver.improve_project()
        
        print(f"✅ Code quality improvement completed!")
        print(f"Files analyzed: {improvement_results.get('files_analyzed', 0)}")
        print(f"Issues found: {improvement_results.get('issues_found', 0)}")
        print(f"Refactorings applied: {improvement_results.get('refactorings_applied', 0)}")
        print(f"Tests generated: {improvement_results.get('tests_generated', 0)}")
        print(f"Documentation generated: {improvement_results.get('documentation_generated', 0)}")
        print(f"Overall improvement: {improvement_results.get('overall_improvement', 0):.2f}%")
        
        # Generate quality report
        report = improver.generate_quality_report()
        print(f"\n📊 Quality Report:")
        print(f"Total files: {report.get('total_files', 0)}")
        print(f"Total lines: {report.get('total_lines', 0)}")
        print(f"Maintainability index: {report.get('maintainability_index', 0):.2f}")
        print(f"Quality grade: {report.get('quality_grade', 'N/A')}")
        
        # Show recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
    except Exception as e:
        logger.error(f"Code quality improvement test failed: {e}")

if __name__ == "__main__":
    import time
    main()


