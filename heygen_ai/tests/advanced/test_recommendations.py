"""
Intelligent Test Recommendations Framework for HeyGen AI Testing System.
Advanced recommendation engine including test optimization, coverage analysis,
and intelligent suggestions for test improvement.
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from collections import defaultdict, Counter
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class TestRecommendation:
    """Represents a test recommendation."""
    recommendation_id: str
    type: str  # optimization, coverage, performance, maintenance, security
    priority: str  # low, medium, high, critical
    title: str
    description: str
    rationale: str
    impact: str  # low, medium, high
    effort: str  # low, medium, high
    confidence: float
    affected_tests: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    estimated_benefit: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TestPattern:
    """Represents a pattern found in tests."""
    pattern_id: str
    pattern_type: str  # naming, structure, assertion, setup
    description: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

@dataclass
class CoverageGap:
    """Represents a coverage gap in testing."""
    gap_id: str
    component: str
    coverage_type: str  # line, branch, function, integration
    current_coverage: float
    target_coverage: float
    missing_lines: List[int] = field(default_factory=list)
    criticality: str = "medium"
    suggestions: List[str] = field(default_factory=list)

class TestAnalyzer:
    """Analyzes test code and execution patterns."""
    
    def __init__(self):
        self.test_files = []
        self.execution_data = []
        self.patterns = []
        self.coverage_data = {}
    
    def analyze_test_files(self, test_directory: str) -> List[TestPattern]:
        """Analyze test files for patterns and issues."""
        test_dir = Path(test_directory)
        patterns = []
        
        # Find test files
        test_files = list(test_dir.rglob("test_*.py")) + list(test_dir.rglob("*_test.py"))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                
                # Analyze patterns
                patterns.extend(self._analyze_naming_patterns(test_file, tree))
                patterns.extend(self._analyze_structure_patterns(test_file, tree))
                patterns.extend(self._analyze_assertion_patterns(test_file, tree))
                patterns.extend(self._analyze_setup_patterns(test_file, tree))
                
            except Exception as e:
                logging.error(f"Error analyzing {test_file}: {e}")
        
        self.patterns = patterns
        return patterns
    
    def _analyze_naming_patterns(self, file_path: Path, tree: ast.AST) -> List[TestPattern]:
        """Analyze naming patterns in tests."""
        patterns = []
        
        # Find test functions
        test_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_functions.append(node.name)
        
        # Analyze naming conventions
        naming_issues = []
        for func_name in test_functions:
            if not re.match(r'^test_[a-z0-9_]+$', func_name):
                naming_issues.append(func_name)
        
        if naming_issues:
            patterns.append(TestPattern(
                pattern_id=f"naming_{file_path.stem}",
                pattern_type="naming",
                description="Non-standard test function naming",
                frequency=len(naming_issues),
                examples=naming_issues,
                quality_score=0.3,
                recommendations=[
                    "Use snake_case for test function names",
                    "Prefix test functions with 'test_'",
                    "Use descriptive names that explain what is being tested"
                ]
            ))
        
        return patterns
    
    def _analyze_structure_patterns(self, file_path: Path, tree: ast.AST) -> List[TestPattern]:
        """Analyze structural patterns in tests."""
        patterns = []
        
        # Find test classes
        test_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and (node.name.startswith('Test') or 'Test' in node.name):
                test_classes.append(node.name)
        
        # Analyze class structure
        if test_classes:
            patterns.append(TestPattern(
                pattern_id=f"structure_{file_path.stem}",
                pattern_type="structure",
                description="Test classes found",
                frequency=len(test_classes),
                examples=test_classes,
                quality_score=0.8,
                recommendations=[
                    "Consider using pytest fixtures for setup/teardown",
                    "Group related tests in classes",
                    "Use descriptive class names"
                ]
            ))
        
        return patterns
    
    def _analyze_assertion_patterns(self, file_path: Path, tree: ast.AST) -> List[TestPattern]:
        """Analyze assertion patterns in tests."""
        patterns = []
        
        # Find assertion statements
        assertions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertions.append(ast.unparse(node))
            elif isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                if node.func.attr in ['assertEqual', 'assertTrue', 'assertFalse', 'assertIn', 'assertNotIn']:
                    assertions.append(ast.unparse(node))
        
        # Analyze assertion quality
        if assertions:
            # Check for basic assertions
            basic_assertions = [a for a in assertions if 'assert ' in a and '==' in a]
            
            if len(basic_assertions) > len(assertions) * 0.5:
                patterns.append(TestPattern(
                    pattern_id=f"assertion_{file_path.stem}",
                    pattern_type="assertion",
                    description="Basic assertions used instead of specific assertion methods",
                    frequency=len(basic_assertions),
                    examples=basic_assertions[:3],
                    quality_score=0.4,
                    recommendations=[
                        "Use specific assertion methods (assertEqual, assertTrue, etc.)",
                        "Provide descriptive error messages",
                        "Use pytest's built-in assertion introspection"
                    ]
                ))
        
        return patterns
    
    def _analyze_setup_patterns(self, file_path: Path, tree: ast.AST) -> List[TestPattern]:
        """Analyze setup and teardown patterns."""
        patterns = []
        
        # Find setup/teardown methods
        setup_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass']:
                    setup_methods.append(node.name)
        
        if setup_methods:
            patterns.append(TestPattern(
                pattern_id=f"setup_{file_path.stem}",
                pattern_type="setup",
                description="Setup/teardown methods found",
                frequency=len(setup_methods),
                examples=setup_methods,
                quality_score=0.7,
                recommendations=[
                    "Consider using pytest fixtures for better setup/teardown",
                    "Use context managers for resource cleanup",
                    "Keep setup/teardown simple and focused"
                ]
            ))
        
        return patterns
    
    def analyze_execution_data(self, executions: List[Dict[str, Any]]) -> List[TestPattern]:
        """Analyze test execution data for patterns."""
        patterns = []
        
        if not executions:
            return patterns
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(executions)
        
        # Analyze performance patterns
        slow_tests = df[df['duration'] > df['duration'].quantile(0.9)]
        if len(slow_tests) > 0:
            patterns.append(TestPattern(
                pattern_id="performance_slow",
                pattern_type="performance",
                description="Slow tests identified",
                frequency=len(slow_tests),
                examples=slow_tests['test_name'].tolist(),
                quality_score=0.3,
                recommendations=[
                    "Optimize slow tests for better performance",
                    "Consider parallel execution for independent tests",
                    "Profile tests to identify bottlenecks"
                ]
            ))
        
        # Analyze reliability patterns
        flaky_tests = df.groupby('test_name')['success'].apply(lambda x: x.mean() < 0.9)
        flaky_test_names = flaky_tests[flaky_tests].index.tolist()
        
        if flaky_test_names:
            patterns.append(TestPattern(
                pattern_id="reliability_flaky",
                pattern_type="reliability",
                description="Flaky tests identified",
                frequency=len(flaky_test_names),
                examples=flaky_test_names,
                quality_score=0.2,
                recommendations=[
                    "Investigate and fix flaky tests",
                    "Add retry mechanisms for transient failures",
                    "Improve test isolation and determinism"
                ]
            ))
        
        return patterns

class CoverageAnalyzer:
    """Analyzes test coverage and identifies gaps."""
    
    def __init__(self):
        self.coverage_data = {}
        self.target_coverage = {
            'line': 80.0,
            'branch': 70.0,
            'function': 85.0,
            'integration': 60.0
        }
    
    def analyze_coverage(self, coverage_file: str) -> List[CoverageGap]:
        """Analyze coverage data and identify gaps."""
        gaps = []
        
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            # Analyze line coverage
            line_gaps = self._analyze_line_coverage(coverage_data)
            gaps.extend(line_gaps)
            
            # Analyze branch coverage
            branch_gaps = self._analyze_branch_coverage(coverage_data)
            gaps.extend(branch_gaps)
            
            # Analyze function coverage
            function_gaps = self._analyze_function_coverage(coverage_data)
            gaps.extend(function_gaps)
            
        except Exception as e:
            logging.error(f"Error analyzing coverage: {e}")
        
        return gaps
    
    def _analyze_line_coverage(self, coverage_data: Dict[str, Any]) -> List[CoverageGap]:
        """Analyze line coverage gaps."""
        gaps = []
        
        if 'files' not in coverage_data:
            return gaps
        
        for file_path, file_data in coverage_data['files'].items():
            if 'summary' not in file_data:
                continue
            
            summary = file_data['summary']
            line_coverage = summary.get('percent_covered', 0.0)
            
            if line_coverage < self.target_coverage['line']:
                # Find missing lines
                missing_lines = []
                if 'missing_lines' in file_data:
                    missing_lines = file_data['missing_lines']
                
                gap = CoverageGap(
                    gap_id=f"line_{file_path.replace('/', '_')}",
                    component=file_path,
                    coverage_type="line",
                    current_coverage=line_coverage,
                    target_coverage=self.target_coverage['line'],
                    missing_lines=missing_lines,
                    criticality="high" if line_coverage < 50 else "medium",
                    suggestions=[
                        "Add tests for uncovered lines",
                        "Focus on critical business logic",
                        "Consider refactoring complex functions"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _analyze_branch_coverage(self, coverage_data: Dict[str, Any]) -> List[CoverageGap]:
        """Analyze branch coverage gaps."""
        gaps = []
        
        if 'files' not in coverage_data:
            return gaps
        
        for file_path, file_data in coverage_data['files'].items():
            if 'summary' not in file_data:
                continue
            
            summary = file_data['summary']
            branch_coverage = summary.get('percent_covered_display', 0.0)
            
            if branch_coverage < self.target_coverage['branch']:
                gap = CoverageGap(
                    gap_id=f"branch_{file_path.replace('/', '_')}",
                    component=file_path,
                    coverage_type="branch",
                    current_coverage=branch_coverage,
                    target_coverage=self.target_coverage['branch'],
                    criticality="high" if branch_coverage < 40 else "medium",
                    suggestions=[
                        "Add tests for uncovered branches",
                        "Test edge cases and error conditions",
                        "Ensure all conditional paths are tested"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _analyze_function_coverage(self, coverage_data: Dict[str, Any]) -> List[CoverageGap]:
        """Analyze function coverage gaps."""
        gaps = []
        
        if 'files' not in coverage_data:
            return gaps
        
        for file_path, file_data in coverage_data['files'].items():
            if 'summary' not in file_data:
                continue
            
            summary = file_data['summary']
            function_coverage = summary.get('percent_covered', 0.0)
            
            if function_coverage < self.target_coverage['function']:
                gap = CoverageGap(
                    gap_id=f"function_{file_path.replace('/', '_')}",
                    component=file_path,
                    coverage_type="function",
                    current_coverage=function_coverage,
                    target_coverage=self.target_coverage['function'],
                    criticality="high" if function_coverage < 60 else "medium",
                    suggestions=[
                        "Add tests for uncovered functions",
                        "Test all public methods",
                        "Consider testing private methods if critical"
                    ]
                )
                gaps.append(gap)
        
        return gaps

class RecommendationEngine:
    """Generates intelligent test recommendations."""
    
    def __init__(self):
        self.analyzer = TestAnalyzer()
        self.coverage_analyzer = CoverageAnalyzer()
        self.recommendations = []
        self.priority_weights = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
    
    def generate_recommendations(self, test_directory: str, 
                               execution_data: List[Dict[str, Any]] = None,
                               coverage_file: str = None) -> List[TestRecommendation]:
        """Generate comprehensive test recommendations."""
        recommendations = []
        
        # Analyze test patterns
        patterns = self.analyzer.analyze_test_files(test_directory)
        recommendations.extend(self._generate_pattern_recommendations(patterns))
        
        # Analyze execution data
        if execution_data:
            exec_patterns = self.analyzer.analyze_execution_data(execution_data)
            recommendations.extend(self._generate_execution_recommendations(exec_patterns))
        
        # Analyze coverage gaps
        if coverage_file and Path(coverage_file).exists():
            coverage_gaps = self.coverage_analyzer.analyze_coverage(coverage_file)
            recommendations.extend(self._generate_coverage_recommendations(coverage_gaps))
        
        # Generate optimization recommendations
        recommendations.extend(self._generate_optimization_recommendations())
        
        # Generate maintenance recommendations
        recommendations.extend(self._generate_maintenance_recommendations())
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (
            self.priority_weights.get(x.priority, 0),
            x.confidence
        ), reverse=True)
        
        self.recommendations = recommendations
        return recommendations
    
    def _generate_pattern_recommendations(self, patterns: List[TestPattern]) -> List[TestRecommendation]:
        """Generate recommendations based on test patterns."""
        recommendations = []
        
        for pattern in patterns:
            if pattern.quality_score < 0.5:  # Low quality patterns
                recommendation = TestRecommendation(
                    recommendation_id=f"pattern_{pattern.pattern_id}",
                    type="maintenance",
                    priority="high" if pattern.quality_score < 0.3 else "medium",
                    title=f"Improve {pattern.pattern_type} patterns",
                    description=pattern.description,
                    rationale=f"Quality score: {pattern.quality_score:.2f}",
                    impact="medium",
                    effort="medium",
                    confidence=0.8,
                    affected_tests=pattern.examples,
                    suggested_actions=pattern.recommendations,
                    estimated_benefit="Improved test quality and maintainability"
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_execution_recommendations(self, patterns: List[TestPattern]) -> List[TestRecommendation]:
        """Generate recommendations based on execution patterns."""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type == "performance":
                recommendation = TestRecommendation(
                    recommendation_id=f"perf_{pattern.pattern_id}",
                    type="optimization",
                    priority="high",
                    title="Optimize slow tests",
                    description=f"Found {pattern.frequency} slow tests",
                    rationale="Performance optimization needed",
                    impact="high",
                    effort="medium",
                    confidence=0.9,
                    affected_tests=pattern.examples,
                    suggested_actions=pattern.recommendations,
                    estimated_benefit="Faster test execution and better CI/CD performance"
                )
                recommendations.append(recommendation)
            
            elif pattern.pattern_type == "reliability":
                recommendation = TestRecommendation(
                    recommendation_id=f"reliability_{pattern.pattern_id}",
                    type="maintenance",
                    priority="critical",
                    title="Fix flaky tests",
                    description=f"Found {pattern.frequency} flaky tests",
                    rationale="Reliability issues affect test confidence",
                    impact="high",
                    effort="high",
                    confidence=0.95,
                    affected_tests=pattern.examples,
                    suggested_actions=pattern.recommendations,
                    estimated_benefit="Improved test reliability and reduced false failures"
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_coverage_recommendations(self, gaps: List[CoverageGap]) -> List[TestRecommendation]:
        """Generate recommendations based on coverage gaps."""
        recommendations = []
        
        for gap in gaps:
            priority = "critical" if gap.criticality == "high" else "high"
            
            recommendation = TestRecommendation(
                recommendation_id=f"coverage_{gap.gap_id}",
                type="coverage",
                priority=priority,
                title=f"Improve {gap.coverage_type} coverage",
                description=f"Coverage: {gap.current_coverage:.1f}% (target: {gap.target_coverage:.1f}%)",
                rationale=f"Low {gap.coverage_type} coverage in {gap.component}",
                impact="high",
                effort="medium",
                confidence=0.9,
                affected_tests=[gap.component],
                suggested_actions=gap.suggestions,
                estimated_benefit="Better test coverage and reduced risk of bugs"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_optimization_recommendations(self) -> List[TestRecommendation]:
        """Generate general optimization recommendations."""
        recommendations = []
        
        # Parallel execution recommendation
        recommendations.append(TestRecommendation(
            recommendation_id="opt_parallel",
            type="optimization",
            priority="medium",
            title="Implement parallel test execution",
            description="Run independent tests in parallel to reduce execution time",
            rationale="Parallel execution can significantly reduce test suite runtime",
            impact="high",
            effort="medium",
            confidence=0.8,
            suggested_actions=[
                "Use pytest-xdist for parallel execution",
                "Identify independent tests",
                "Configure appropriate number of workers",
                "Handle shared resources properly"
            ],
            estimated_benefit="50-70% reduction in test execution time"
        ))
        
        # Test data optimization
        recommendations.append(TestRecommendation(
            recommendation_id="opt_data",
            type="optimization",
            priority="medium",
            title="Optimize test data management",
            description="Use efficient test data generation and cleanup",
            rationale="Better test data management improves performance and reliability",
            impact="medium",
            effort="low",
            confidence=0.7,
            suggested_actions=[
                "Use factories for test data generation",
                "Implement proper test data cleanup",
                "Use database transactions for test isolation",
                "Consider using test data builders"
            ],
            estimated_benefit="Improved test performance and reliability"
        ))
        
        return recommendations
    
    def _generate_maintenance_recommendations(self) -> List[TestRecommendation]:
        """Generate maintenance recommendations."""
        recommendations = []
        
        # Test organization
        recommendations.append(TestRecommendation(
            recommendation_id="maint_organization",
            type="maintenance",
            priority="medium",
            title="Improve test organization",
            description="Organize tests by feature and type for better maintainability",
            rationale="Well-organized tests are easier to maintain and understand",
            impact="medium",
            effort="low",
            confidence=0.8,
            suggested_actions=[
                "Group tests by feature or module",
                "Use consistent naming conventions",
                "Create test utilities and helpers",
                "Document test structure and conventions"
            ],
            estimated_benefit="Improved test maintainability and developer experience"
        ))
        
        # Test documentation
        recommendations.append(TestRecommendation(
            recommendation_id="maint_documentation",
            type="maintenance",
            priority="low",
            title="Improve test documentation",
            description="Add documentation for complex tests and test utilities",
            rationale="Good documentation helps maintain and understand tests",
            impact="low",
            effort="low",
            confidence=0.6,
            suggested_actions=[
                "Add docstrings to test functions",
                "Document test utilities and helpers",
                "Create testing guidelines",
                "Add comments for complex test logic"
            ],
            estimated_benefit="Better test understanding and maintenance"
        ))
        
        return recommendations
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get summary of recommendations."""
        if not self.recommendations:
            return {}
        
        # Count by type and priority
        type_counts = Counter(rec.type for rec in self.recommendations)
        priority_counts = Counter(rec.priority for rec in self.recommendations)
        
        # Calculate average confidence
        avg_confidence = np.mean([rec.confidence for rec in self.recommendations])
        
        return {
            "total_recommendations": len(self.recommendations),
            "by_type": dict(type_counts),
            "by_priority": dict(priority_counts),
            "average_confidence": avg_confidence,
            "critical_recommendations": len([r for r in self.recommendations if r.priority == "critical"]),
            "high_priority_recommendations": len([r for r in self.recommendations if r.priority in ["critical", "high"]])
        }
    
    def export_recommendations(self, output_file: str = "test_recommendations.json"):
        """Export recommendations to JSON file."""
        recommendations_data = []
        
        for rec in self.recommendations:
            rec_data = {
                "recommendation_id": rec.recommendation_id,
                "type": rec.type,
                "priority": rec.priority,
                "title": rec.title,
                "description": rec.description,
                "rationale": rec.rationale,
                "impact": rec.impact,
                "effort": rec.effort,
                "confidence": rec.confidence,
                "affected_tests": rec.affected_tests,
                "suggested_actions": rec.suggested_actions,
                "estimated_benefit": rec.estimated_benefit,
                "generated_at": rec.generated_at.isoformat()
            }
            recommendations_data.append(rec_data)
        
        with open(output_file, 'w') as f:
            json.dump(recommendations_data, f, indent=2)
        
        print(f"Recommendations exported to: {output_file}")

# Example usage and demo
def demo_test_recommendations():
    """Demonstrate test recommendations capabilities."""
    print("💡 Test Recommendations Framework Demo")
    print("=" * 50)
    
    # Create recommendation engine
    engine = RecommendationEngine()
    
    # Generate sample execution data
    execution_data = [
        {
            "test_name": "test_user_login",
            "duration": 2.5,
            "success": True,
            "timestamp": datetime.now()
        },
        {
            "test_name": "test_payment_processing",
            "duration": 8.2,
            "success": False,
            "timestamp": datetime.now()
        },
        {
            "test_name": "test_api_endpoints",
            "duration": 1.1,
            "success": True,
            "timestamp": datetime.now()
        },
        {
            "test_name": "test_database_operations",
            "duration": 12.3,
            "success": True,
            "timestamp": datetime.now()
        }
    ]
    
    # Generate recommendations
    print("🔍 Analyzing tests and generating recommendations...")
    recommendations = engine.generate_recommendations(
        test_directory="tests",
        execution_data=execution_data
    )
    
    # Print recommendations
    print(f"\n📊 Generated {len(recommendations)} recommendations:")
    
    for i, rec in enumerate(recommendations[:10], 1):  # Show first 10
        priority_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(rec.priority, "⚪")
        print(f"\n{i}. {priority_icon} {rec.title}")
        print(f"   Type: {rec.type} | Priority: {rec.priority} | Confidence: {rec.confidence:.1%}")
        print(f"   Description: {rec.description}")
        print(f"   Impact: {rec.impact} | Effort: {rec.effort}")
        print(f"   Estimated Benefit: {rec.estimated_benefit}")
        
        if rec.suggested_actions:
            print(f"   Suggested Actions:")
            for action in rec.suggested_actions[:3]:  # Show first 3
                print(f"     - {action}")
    
    # Print summary
    print("\n📈 Recommendation Summary:")
    summary = engine.get_recommendation_summary()
    print(f"   Total Recommendations: {summary['total_recommendations']}")
    print(f"   Critical: {summary['critical_recommendations']}")
    print(f"   High Priority: {summary['high_priority_recommendations']}")
    print(f"   Average Confidence: {summary['average_confidence']:.1%}")
    
    # Export recommendations
    engine.export_recommendations("demo_recommendations.json")

if __name__ == "__main__":
    # Run demo
    demo_test_recommendations()
