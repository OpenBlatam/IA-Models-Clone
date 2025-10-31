"""
Gamma App - Real Improvement Analyzer
Analyzes and recommends real improvements that actually work
"""

import asyncio
import logging
import time
import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ast
import re
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Analysis types"""
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    DOCUMENTATION = "documentation"

class Severity(Enum):
    """Issue severity"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CodeIssue:
    """Code issue"""
    issue_id: str
    file_path: str
    line_number: int
    issue_type: str
    severity: Severity
    description: str
    suggestion: str
    code_snippet: str
    impact_score: int  # 1-10
    effort_to_fix: int  # 1-10
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ImprovementRecommendation:
    """Improvement recommendation"""
    recommendation_id: str
    title: str
    description: str
    analysis_type: AnalysisType
    severity: Severity
    impact_score: int  # 1-10
    effort_score: int  # 1-10
    priority_score: float  # calculated
    issues: List[CodeIssue]
    implementation_steps: List[str]
    code_examples: List[str]
    testing_notes: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Calculate priority score
        self.priority_score = (self.impact_score * 0.7) + (self.severity.value == "critical" and 3 or 0) + (self.severity.value == "high" and 2 or 0) + (self.severity.value == "medium" and 1 or 0)

class RealImprovementAnalyzer:
    """
    Analyzes code and recommends real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement analyzer"""
        self.project_root = Path(project_root)
        self.issues: Dict[str, CodeIssue] = {}
        self.recommendations: Dict[str, ImprovementRecommendation] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info(f"Real Improvement Analyzer initialized for {self.project_root}")
    
    async def analyze_project(self) -> Dict[str, Any]:
        """Analyze entire project"""
        try:
            logger.info("Starting project analysis...")
            
            # Run different types of analysis
            code_quality_results = await self._analyze_code_quality()
            performance_results = await self._analyze_performance()
            security_results = await self._analyze_security()
            maintainability_results = await self._analyze_maintainability()
            testing_results = await self._analyze_testing()
            documentation_results = await self._analyze_documentation()
            
            # Generate recommendations
            recommendations = await self._generate_recommendations()
            
            # Calculate overall score
            overall_score = self._calculate_overall_score()
            
            self.analysis_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_score": overall_score,
                "code_quality": code_quality_results,
                "performance": performance_results,
                "security": security_results,
                "maintainability": maintainability_results,
                "testing": testing_results,
                "documentation": documentation_results,
                "recommendations": recommendations,
                "total_issues": len(self.issues),
                "total_recommendations": len(self.recommendations)
            }
            
            logger.info(f"Project analysis completed. Overall score: {overall_score}")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze project: {e}")
            return {"error": str(e)}
    
    async def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality"""
        try:
            issues = []
            
            # Find Python files
            python_files = list(self.project_root.rglob("*.py"))
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Analyze file
                    file_issues = await self._analyze_python_file(file_path, content)
                    issues.extend(file_issues)
                    
                except Exception as e:
                    logger.warning(f"Could not analyze {file_path}: {e}")
            
            return {
                "total_files_analyzed": len(python_files),
                "issues_found": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze code quality: {e}")
            return {"error": str(e)}
    
    async def _analyze_python_file(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Analyze individual Python file"""
        issues = []
        lines = content.split('\n')
        
        try:
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze AST
            ast_issues = self._analyze_ast(tree, file_path, lines)
            issues.extend(ast_issues)
            
            # Analyze patterns
            pattern_issues = self._analyze_patterns(content, file_path, lines)
            issues.extend(pattern_issues)
            
            # Analyze complexity
            complexity_issues = self._analyze_complexity(content, file_path, lines)
            issues.extend(complexity_issues)
            
        except SyntaxError as e:
            issue = CodeIssue(
                issue_id=f"syntax_{int(time.time() * 1000)}",
                file_path=str(file_path),
                line_number=e.lineno or 0,
                issue_type="syntax_error",
                severity=Severity.CRITICAL,
                description=f"Syntax error: {e.msg}",
                suggestion="Fix syntax error",
                code_snippet=lines[e.lineno - 1] if e.lineno else "",
                impact_score=10,
                effort_to_fix=3
            )
            issues.append(issue)
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[CodeIssue]:
        """Analyze AST for issues"""
        issues = []
        
        class ASTAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.current_line = 0
            
            def visit_FunctionDef(self, node):
                # Check function length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    function_length = node.end_lineno - node.lineno
                    if function_length > 50:
                        issue = CodeIssue(
                            issue_id=f"long_function_{int(time.time() * 1000)}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="long_function",
                            severity=Severity.MEDIUM,
                            description=f"Function '{node.name}' is too long ({function_length} lines)",
                            suggestion="Break down into smaller functions",
                            code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            impact_score=6,
                            effort_to_fix=5
                        )
                        self.issues.append(issue)
                
                # Check for too many parameters
                if len(node.args.args) > 5:
                    issue = CodeIssue(
                        issue_id=f"too_many_params_{int(time.time() * 1000)}",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type="too_many_parameters",
                        severity=Severity.MEDIUM,
                        description=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        suggestion="Use a data class or dictionary for parameters",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                        impact_score=5,
                        effort_to_fix=4
                    )
                    self.issues.append(issue)
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check class length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    class_length = node.end_lineno - node.lineno
                    if class_length > 200:
                        issue = CodeIssue(
                            issue_id=f"long_class_{int(time.time() * 1000)}",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="long_class",
                            severity=Severity.MEDIUM,
                            description=f"Class '{node.name}' is too long ({class_length} lines)",
                            suggestion="Break down into smaller classes",
                            code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            impact_score=7,
                            effort_to_fix=6
                        )
                        self.issues.append(issue)
                
                self.generic_visit(node)
        
        analyzer = ASTAnalyzer()
        analyzer.visit(tree)
        return analyzer.issues
    
    def _analyze_patterns(self, content: str, file_path: Path, lines: List[str]) -> List[CodeIssue]:
        """Analyze code patterns"""
        issues = []
        
        # Check for common anti-patterns
        patterns = [
            {
                "pattern": r"except\s*:",
                "issue_type": "bare_except",
                "severity": Severity.HIGH,
                "description": "Bare except clause",
                "suggestion": "Specify exception types"
            },
            {
                "pattern": r"print\s*\(",
                "issue_type": "print_statement",
                "severity": Severity.LOW,
                "description": "Print statement found",
                "suggestion": "Use logging instead of print"
            },
            {
                "pattern": r"TODO|FIXME|HACK",
                "issue_type": "todo_comment",
                "severity": Severity.LOW,
                "description": "TODO/FIXME comment found",
                "suggestion": "Address TODO/FIXME items"
            },
            {
                "pattern": r"import \*",
                "issue_type": "wildcard_import",
                "severity": Severity.MEDIUM,
                "description": "Wildcard import found",
                "suggestion": "Import specific modules"
            }
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], line):
                    issue = CodeIssue(
                        issue_id=f"{pattern_info['issue_type']}_{int(time.time() * 1000)}",
                        file_path=str(file_path),
                        line_number=i,
                        issue_type=pattern_info["issue_type"],
                        severity=pattern_info["severity"],
                        description=pattern_info["description"],
                        suggestion=pattern_info["suggestion"],
                        code_snippet=line,
                        impact_score=5,
                        effort_to_fix=2
                    )
                    issues.append(issue)
        
        return issues
    
    def _analyze_complexity(self, content: str, file_path: Path, lines: List[str]) -> List[CodeIssue]:
        """Analyze code complexity"""
        issues = []
        
        # Simple complexity analysis
        complexity_indicators = [
            ("if", "conditional"),
            ("for", "loop"),
            ("while", "loop"),
            ("try", "exception_handling"),
            ("except", "exception_handling")
        ]
        
        for i, line in enumerate(lines, 1):
            complexity_count = sum(line.count(indicator) for indicator, _ in complexity_indicators)
            
            if complexity_count > 3:
                issue = CodeIssue(
                    issue_id=f"high_complexity_{int(time.time() * 1000)}",
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="high_complexity",
                    severity=Severity.MEDIUM,
                    description=f"High complexity line ({complexity_count} indicators)",
                    suggestion="Simplify the logic",
                    code_snippet=line,
                    impact_score=6,
                    effort_to_fix=4
                )
                issues.append(issue)
        
        return issues
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance issues"""
        try:
            issues = []
            
            # Check for common performance issues
            performance_patterns = [
                {
                    "pattern": r"for.*in.*range\(len\(",
                    "issue_type": "inefficient_loop",
                    "description": "Inefficient loop with range(len())",
                    "suggestion": "Use enumerate() or direct iteration"
                },
                {
                    "pattern": r"\.append\(.*\) in loop",
                    "issue_type": "list_append_in_loop",
                    "description": "List append in loop",
                    "suggestion": "Use list comprehension"
                },
                {
                    "pattern": r"SELECT \* FROM",
                    "issue_type": "select_all",
                    "description": "SELECT * query found",
                    "suggestion": "Select specific columns"
                }
            ]
            
            python_files = list(self.project_root.rglob("*.py"))
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        for pattern_info in performance_patterns:
                            if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                                issue = CodeIssue(
                                    issue_id=f"{pattern_info['issue_type']}_{int(time.time() * 1000)}",
                                    file_path=str(file_path),
                                    line_number=i,
                                    issue_type=pattern_info["issue_type"],
                                    severity=Severity.MEDIUM,
                                    description=pattern_info["description"],
                                    suggestion=pattern_info["suggestion"],
                                    code_snippet=line,
                                    impact_score=7,
                                    effort_to_fix=3
                                )
                                issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"Could not analyze {file_path}: {e}")
            
            return {
                "performance_issues": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return {"error": str(e)}
    
    async def _analyze_security(self) -> Dict[str, Any]:
        """Analyze security issues"""
        try:
            issues = []
            
            # Check for security vulnerabilities
            security_patterns = [
                {
                    "pattern": r"password\s*=\s*['\"].*['\"]",
                    "issue_type": "hardcoded_password",
                    "severity": Severity.CRITICAL,
                    "description": "Hardcoded password found",
                    "suggestion": "Use environment variables or secure storage"
                },
                {
                    "pattern": r"sql.*\+.*input",
                    "issue_type": "sql_injection",
                    "severity": Severity.CRITICAL,
                    "description": "Potential SQL injection",
                    "suggestion": "Use parameterized queries"
                },
                {
                    "pattern": r"eval\(",
                    "issue_type": "eval_usage",
                    "severity": Severity.HIGH,
                    "description": "eval() function used",
                    "suggestion": "Avoid eval() for security"
                },
                {
                    "pattern": r"exec\(",
                    "issue_type": "exec_usage",
                    "severity": Severity.HIGH,
                    "description": "exec() function used",
                    "suggestion": "Avoid exec() for security"
                }
            ]
            
            python_files = list(self.project_root.rglob("*.py"))
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        for pattern_info in security_patterns:
                            if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                                issue = CodeIssue(
                                    issue_id=f"{pattern_info['issue_type']}_{int(time.time() * 1000)}",
                                    file_path=str(file_path),
                                    line_number=i,
                                    issue_type=pattern_info["issue_type"],
                                    severity=pattern_info["severity"],
                                    description=pattern_info["description"],
                                    suggestion=pattern_info["suggestion"],
                                    code_snippet=line,
                                    impact_score=9,
                                    effort_to_fix=4
                                )
                                issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"Could not analyze {file_path}: {e}")
            
            return {
                "security_issues": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze security: {e}")
            return {"error": str(e)}
    
    async def _analyze_maintainability(self) -> Dict[str, Any]:
        """Analyze maintainability issues"""
        try:
            issues = []
            
            # Check for maintainability issues
            maintainability_patterns = [
                {
                    "pattern": r"# TODO|# FIXME|# HACK",
                    "issue_type": "todo_comment",
                    "severity": Severity.LOW,
                    "description": "TODO/FIXME comment found",
                    "suggestion": "Address TODO/FIXME items"
                },
                {
                    "pattern": r"def.*\(\):",
                    "issue_type": "empty_function",
                    "severity": Severity.LOW,
                    "description": "Empty function found",
                    "suggestion": "Implement function or remove"
                },
                {
                    "pattern": r"class.*:.*pass",
                    "issue_type": "empty_class",
                    "severity": Severity.LOW,
                    "description": "Empty class found",
                    "suggestion": "Implement class or remove"
                }
            ]
            
            python_files = list(self.project_root.rglob("*.py"))
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        for pattern_info in maintainability_patterns:
                            if re.search(pattern_info["pattern"], line):
                                issue = CodeIssue(
                                    issue_id=f"{pattern_info['issue_type']}_{int(time.time() * 1000)}",
                                    file_path=str(file_path),
                                    line_number=i,
                                    issue_type=pattern_info["issue_type"],
                                    severity=pattern_info["severity"],
                                    description=pattern_info["description"],
                                    suggestion=pattern_info["suggestion"],
                                    code_snippet=line,
                                    impact_score=3,
                                    effort_to_fix=2
                                )
                                issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"Could not analyze {file_path}: {e}")
            
            return {
                "maintainability_issues": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze maintainability: {e}")
            return {"error": str(e)}
    
    async def _analyze_testing(self) -> Dict[str, Any]:
        """Analyze testing coverage and quality"""
        try:
            issues = []
            
            # Check for test files
            test_files = list(self.project_root.rglob("*test*.py"))
            test_files.extend(list(self.project_root.rglob("*_test.py")))
            
            if not test_files:
                issue = CodeIssue(
                    issue_id=f"no_tests_{int(time.time() * 1000)}",
                    file_path=str(self.project_root),
                    line_number=0,
                    issue_type="no_tests",
                    severity=Severity.HIGH,
                    description="No test files found",
                    suggestion="Add test files for your code",
                    code_snippet="",
                    impact_score=8,
                    effort_to_fix=6
                )
                issues.append(issue)
            
            # Check for test coverage
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", "--cov", ".", "--cov-report=json"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Parse coverage report
                    coverage_data = json.loads(result.stdout)
                    coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0)
                    
                    if coverage_percentage < 80:
                        issue = CodeIssue(
                            issue_id=f"low_coverage_{int(time.time() * 1000)}",
                            file_path=str(self.project_root),
                            line_number=0,
                            issue_type="low_test_coverage",
                            severity=Severity.MEDIUM,
                            description=f"Low test coverage ({coverage_percentage}%)",
                            suggestion="Increase test coverage to at least 80%",
                            code_snippet="",
                            impact_score=6,
                            effort_to_fix=5
                        )
                        issues.append(issue)
                
            except Exception as e:
                logger.warning(f"Could not run test coverage: {e}")
            
            return {
                "test_files_found": len(test_files),
                "testing_issues": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze testing: {e}")
            return {"error": str(e)}
    
    async def _analyze_documentation(self) -> Dict[str, Any]:
        """Analyze documentation quality"""
        try:
            issues = []
            
            # Check for docstrings
            python_files = list(self.project_root.rglob("*.py"))
            files_without_docstrings = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if file has docstring
                    if not re.search(r'""".*"""', content, re.DOTALL):
                        files_without_docstrings += 1
                        
                        issue = CodeIssue(
                            issue_id=f"no_docstring_{int(time.time() * 1000)}",
                            file_path=str(file_path),
                            line_number=1,
                            issue_type="no_docstring",
                            severity=Severity.LOW,
                            description="File missing docstring",
                            suggestion="Add module docstring",
                            code_snippet="",
                            impact_score=3,
                            effort_to_fix=2
                        )
                        issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"Could not analyze {file_path}: {e}")
            
            return {
                "total_files": len(python_files),
                "files_without_docstrings": files_without_docstrings,
                "documentation_issues": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze documentation: {e}")
            return {"error": str(e)}
    
    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate improvement recommendations"""
        try:
            recommendations = []
            
            # Group issues by type
            issues_by_type = {}
            for issue in self.issues.values():
                if issue.issue_type not in issues_by_type:
                    issues_by_type[issue.issue_type] = []
                issues_by_type[issue.issue_type].append(issue)
            
            # Generate recommendations for each issue type
            for issue_type, issues in issues_by_type.items():
                if len(issues) > 0:
                    recommendation = await self._create_recommendation(issue_type, issues)
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def _create_recommendation(self, issue_type: str, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Create recommendation for issue type"""
        try:
            # Calculate impact and effort
            total_impact = sum(issue.impact_score for issue in issues)
            total_effort = sum(issue.effort_to_fix for issue in issues)
            avg_impact = total_impact / len(issues)
            avg_effort = total_effort / len(issues)
            
            # Determine severity
            severities = [issue.severity for issue in issues]
            critical_count = severities.count(Severity.CRITICAL)
            high_count = severities.count(Severity.HIGH)
            
            if critical_count > 0:
                severity = Severity.CRITICAL
            elif high_count > 0:
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM
            
            # Generate recommendation
            recommendation_id = f"rec_{int(time.time() * 1000)}"
            
            recommendation = {
                "recommendation_id": recommendation_id,
                "title": f"Fix {issue_type.replace('_', ' ').title()} Issues",
                "description": f"Address {len(issues)} {issue_type} issues found in the codebase",
                "issue_type": issue_type,
                "severity": severity.value,
                "impact_score": int(avg_impact),
                "effort_score": int(avg_effort),
                "priority_score": (avg_impact * 0.7) + (severity.value == "critical" and 3 or 0) + (severity.value == "high" and 2 or 0),
                "issue_count": len(issues),
                "issues": [issue.__dict__ for issue in issues],
                "implementation_steps": self._get_implementation_steps(issue_type),
                "code_examples": self._get_code_examples(issue_type),
                "testing_notes": self._get_testing_notes(issue_type)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to create recommendation: {e}")
            return {}
    
    def _get_implementation_steps(self, issue_type: str) -> List[str]:
        """Get implementation steps for issue type"""
        steps_map = {
            "syntax_error": [
                "1. Fix syntax errors in the code",
                "2. Run syntax check with python -m py_compile",
                "3. Test the fixed code"
            ],
            "long_function": [
                "1. Identify logical sections in the function",
                "2. Extract each section into separate functions",
                "3. Update function calls",
                "4. Test the refactored code"
            ],
            "bare_except": [
                "1. Identify bare except clauses",
                "2. Replace with specific exception types",
                "3. Add proper error handling",
                "4. Test exception scenarios"
            ],
            "hardcoded_password": [
                "1. Remove hardcoded passwords",
                "2. Use environment variables",
                "3. Implement secure password storage",
                "4. Update configuration"
            ],
            "sql_injection": [
                "1. Identify SQL string concatenation",
                "2. Replace with parameterized queries",
                "3. Use ORM or query builders",
                "4. Test with malicious inputs"
            ]
        }
        
        return steps_map.get(issue_type, [
            "1. Analyze the issue",
            "2. Implement the fix",
            "3. Test the solution",
            "4. Document the changes"
        ])
    
    def _get_code_examples(self, issue_type: str) -> List[str]:
        """Get code examples for issue type"""
        examples_map = {
            "syntax_error": [
                "# Fix syntax error\nif condition:\n    print('Fixed')",
                "# Use proper indentation\nfor item in items:\n    process(item)"
            ],
            "long_function": [
                "# Break down long function\ndef process_data(data):\n    cleaned_data = clean_data(data)\n    processed_data = transform_data(cleaned_data)\n    return processed_data",
                "# Extract helper functions\ndef clean_data(data):\n    return [item.strip() for item in data if item]"
            ],
            "bare_except": [
                "# Replace bare except\ntry:\n    risky_operation()\nexcept SpecificException as e:\n    handle_error(e)",
                "# Add specific exception handling\ntry:\n    file_operation()\nexcept FileNotFoundError:\n    create_file()\nexcept PermissionError:\n    request_permission()"
            ],
            "hardcoded_password": [
                "# Use environment variables\nimport os\npassword = os.getenv('DB_PASSWORD')",
                "# Use secure configuration\nfrom config import settings\npassword = settings.database.password"
            ],
            "sql_injection": [
                "# Use parameterized queries\ncursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                "# Use ORM\nUser.query.filter_by(id=user_id).first()"
            ]
        }
        
        return examples_map.get(issue_type, [
            "# Example fix\n# TODO: Add specific example"
        ])
    
    def _get_testing_notes(self, issue_type: str) -> List[str]:
        """Get testing notes for issue type"""
        notes_map = {
            "syntax_error": "Test that code compiles without syntax errors",
            "long_function": "Test that refactored functions work correctly",
            "bare_except": "Test exception handling with different error types",
            "hardcoded_password": "Test with different environment configurations",
            "sql_injection": "Test with malicious input to ensure security"
        }
        
        return notes_map.get(issue_type, "Test the fix thoroughly")
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall project score"""
        try:
            if not self.issues:
                return 100.0
            
            # Calculate score based on issues
            total_issues = len(self.issues)
            critical_issues = len([i for i in self.issues.values() if i.severity == Severity.CRITICAL])
            high_issues = len([i for i in self.issues.values() if i.severity == Severity.HIGH])
            medium_issues = len([i for i in self.issues.values() if i.severity == Severity.MEDIUM])
            low_issues = len([i for i in self.issues.values() if i.severity == Severity.LOW])
            
            # Calculate penalty
            penalty = (critical_issues * 20) + (high_issues * 10) + (medium_issues * 5) + (low_issues * 2)
            
            # Calculate score
            score = max(0, 100 - penalty)
            
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate overall score: {e}")
            return 0.0
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary"""
        try:
            total_issues = len(self.issues)
            total_recommendations = len(self.recommendations)
            
            issues_by_severity = {}
            for severity in Severity:
                issues_by_severity[severity.value] = len([
                    issue for issue in self.issues.values()
                    if issue.severity == severity
                ])
            
            issues_by_type = {}
            for issue in self.issues.values():
                if issue.issue_type not in issues_by_type:
                    issues_by_type[issue.issue_type] = 0
                issues_by_type[issue.issue_type] += 1
            
            return {
                "total_issues": total_issues,
                "total_recommendations": total_recommendations,
                "overall_score": self._calculate_overall_score(),
                "issues_by_severity": issues_by_severity,
                "issues_by_type": issues_by_type,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis summary: {e}")
            return {"error": str(e)}

# Global analyzer instance
improvement_analyzer = None

def get_improvement_analyzer() -> RealImprovementAnalyzer:
    """Get improvement analyzer instance"""
    global improvement_analyzer
    if not improvement_analyzer:
        improvement_analyzer = RealImprovementAnalyzer()
    return improvement_analyzer













