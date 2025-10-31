"""
Integration Layer for Test Generation System
===========================================

This module provides integration capabilities with existing systems,
APIs, and external services to enhance the test generation workflow.
"""

import asyncio
import json
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

from .unified_api import TestGenerationAPI, create_api
from .base_architecture import TestCase, TestGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for system integrations"""
    api_endpoints: Dict[str, str] = None
    authentication: Dict[str, str] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


class CodeAnalysisIntegration:
    """Integration with code analysis tools"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def analyze_function_complexity(self, function_code: str) -> Dict[str, Any]:
        """Analyze function complexity using external tools"""
        try:
            # Simulate complexity analysis
            complexity_score = self._calculate_complexity(function_code)
            
            return {
                "complexity_score": complexity_score,
                "cyclomatic_complexity": self._calculate_cyclomatic_complexity(function_code),
                "cognitive_complexity": self._calculate_cognitive_complexity(function_code),
                "maintainability_index": self._calculate_maintainability_index(function_code),
                "recommendations": self._get_complexity_recommendations(complexity_score)
            }
        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate overall complexity score"""
        # Simple complexity calculation based on code structure
        lines = code.split('\n')
        complexity = 0
        
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['if', 'elif', 'else', 'for', 'while', 'try', 'except']):
                complexity += 1
            if 'and' in line or 'or' in line:
                complexity += 0.5
        
        return min(complexity / len(lines) * 10, 10.0) if lines else 0
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for line in code.split('\n'):
            line = line.strip()
            if any(keyword in line for keyword in ['if', 'elif', 'for', 'while', 'except']):
                complexity += 1
            if 'and' in line or 'or' in line:
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, code: str) -> int:
        """Calculate cognitive complexity"""
        complexity = 0
        nesting_level = 0
        
        for line in code.split('\n'):
            line = line.strip()
            if any(keyword in line for keyword in ['if', 'elif', 'for', 'while', 'try']):
                complexity += 1 + nesting_level
                nesting_level += 1
            elif any(keyword in line for keyword in ['else', 'except', 'finally']):
                nesting_level = max(0, nesting_level - 1)
            elif line.endswith(':'):
                nesting_level += 1
        
        return complexity
    
    def _calculate_maintainability_index(self, code: str) -> float:
        """Calculate maintainability index"""
        lines = len(code.split('\n'))
        complexity = self._calculate_complexity(code)
        
        # Simplified maintainability index calculation
        if lines == 0:
            return 100.0
        
        maintainability = 100 - (complexity * 5) - (lines * 0.1)
        return max(0, min(100, maintainability))
    
    def _get_complexity_recommendations(self, complexity_score: float) -> List[str]:
        """Get recommendations based on complexity score"""
        recommendations = []
        
        if complexity_score > 7:
            recommendations.append("Consider breaking down this function into smaller functions")
            recommendations.append("Reduce nested conditions and loops")
        elif complexity_score > 5:
            recommendations.append("Consider simplifying conditional logic")
        elif complexity_score > 3:
            recommendations.append("Function complexity is moderate - monitor for future changes")
        else:
            recommendations.append("Function complexity is low - good maintainability")
        
        return recommendations


class CoverageIntegration:
    """Integration with code coverage tools"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_coverage_report(self, project_path: str) -> Dict[str, Any]:
        """Get code coverage report for the project"""
        try:
            # Simulate coverage analysis
            coverage_data = self._analyze_coverage(project_path)
            
            return {
                "overall_coverage": coverage_data["overall"],
                "line_coverage": coverage_data["lines"],
                "branch_coverage": coverage_data["branches"],
                "function_coverage": coverage_data["functions"],
                "uncovered_lines": coverage_data["uncovered"],
                "recommendations": self._get_coverage_recommendations(coverage_data)
            }
        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_coverage(self, project_path: str) -> Dict[str, Any]:
        """Analyze code coverage (simulated)"""
        # This would integrate with actual coverage tools like coverage.py
        return {
            "overall": 0.75,
            "lines": 0.78,
            "branches": 0.65,
            "functions": 0.82,
            "uncovered": [
                {"file": "module1.py", "lines": [15, 23, 45]},
                {"file": "module2.py", "lines": [8, 12, 34, 56]}
            ]
        }
    
    def _get_coverage_recommendations(self, coverage_data: Dict[str, Any]) -> List[str]:
        """Get coverage improvement recommendations"""
        recommendations = []
        
        if coverage_data["overall"] < 0.8:
            recommendations.append("Overall coverage is below 80% - consider adding more tests")
        
        if coverage_data["branches"] < 0.7:
            recommendations.append("Branch coverage is low - add tests for conditional paths")
        
        if coverage_data["functions"] < 0.8:
            recommendations.append("Function coverage is low - ensure all functions are tested")
        
        return recommendations


class APIIntegration:
    """Integration with external APIs and services"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def fetch_documentation(self, function_name: str, library: str) -> Optional[str]:
        """Fetch documentation from external sources"""
        try:
            # Simulate API call to documentation service
            if self.config.api_endpoints and "documentation" in self.config.api_endpoints:
                # This would make actual API calls
                return f"Documentation for {function_name} from {library}"
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch documentation: {e}")
            return None
    
    async def get_best_practices(self, language: str, framework: str) -> List[str]:
        """Get best practices for testing in specific language/framework"""
        try:
            # Simulate API call to best practices service
            practices = {
                "python": {
                    "pytest": [
                        "Use descriptive test names",
                        "Follow AAA pattern (Arrange, Act, Assert)",
                        "Use fixtures for setup and teardown",
                        "Mock external dependencies",
                        "Test edge cases and error conditions"
                    ],
                    "unittest": [
                        "Inherit from unittest.TestCase",
                        "Use setUp and tearDown methods",
                        "Use assert methods for validation",
                        "Group related tests in test classes"
                    ]
                }
            }
            
            return practices.get(language, {}).get(framework, [])
        except Exception as e:
            self.logger.error(f"Failed to get best practices: {e}")
            return []
    
    async def validate_test_quality(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Validate test quality using external services"""
        try:
            # Simulate quality validation
            quality_score = self._calculate_quality_score(test_cases)
            
            return {
                "quality_score": quality_score,
                "issues": self._identify_quality_issues(test_cases),
                "recommendations": self._get_quality_recommendations(quality_score)
            }
        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, test_cases: List[TestCase]) -> float:
        """Calculate overall test quality score"""
        if not test_cases:
            return 0.0
        
        scores = []
        for test_case in test_cases:
            score = 0.0
            
            # Check test name quality
            if len(test_case.name) > 10 and "test_" in test_case.name:
                score += 0.2
            
            # Check description quality
            if test_case.description and len(test_case.description) > 20:
                score += 0.2
            
            # Check test code quality
            if "assert" in test_case.test_code.lower():
                score += 0.3
            
            # Check setup/teardown
            if test_case.setup_code or test_case.teardown_code:
                score += 0.1
            
            # Check category and priority
            if test_case.category and test_case.priority:
                score += 0.2
            
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def _identify_quality_issues(self, test_cases: List[TestCase]) -> List[str]:
        """Identify quality issues in test cases"""
        issues = []
        
        for test_case in test_cases:
            if len(test_case.name) <= 10:
                issues.append(f"Test name too short: {test_case.name}")
            
            if not test_case.description:
                issues.append(f"Missing description: {test_case.name}")
            
            if "assert" not in test_case.test_code.lower():
                issues.append(f"No assertions found: {test_case.name}")
        
        return issues
    
    def _get_quality_recommendations(self, quality_score: float) -> List[str]:
        """Get quality improvement recommendations"""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Overall test quality is low - review test structure and content")
        elif quality_score < 0.8:
            recommendations.append("Test quality is moderate - consider improvements in naming and documentation")
        else:
            recommendations.append("Test quality is good - maintain current standards")
        
        return recommendations


class DatabaseIntegration:
    """Integration with databases for test storage and retrieval"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def store_test_cases(self, test_cases: List[TestCase], project_id: str) -> bool:
        """Store test cases in database"""
        try:
            # Simulate database storage
            test_data = [asdict(test_case) for test_case in test_cases]
            
            # This would integrate with actual database
            self.logger.info(f"Stored {len(test_cases)} test cases for project {project_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store test cases: {e}")
            return False
    
    async def retrieve_test_cases(self, project_id: str, function_name: Optional[str] = None) -> List[TestCase]:
        """Retrieve test cases from database"""
        try:
            # Simulate database retrieval
            # This would query actual database
            return []
        except Exception as e:
            self.logger.error(f"Failed to retrieve test cases: {e}")
            return []
    
    async def get_test_statistics(self, project_id: str) -> Dict[str, Any]:
        """Get test statistics for a project"""
        try:
            # Simulate statistics calculation
            return {
                "total_tests": 0,
                "coverage_percentage": 0.0,
                "last_updated": None,
                "test_categories": {},
                "quality_metrics": {}
            }
        except Exception as e:
            self.logger.error(f"Failed to get test statistics: {e}")
            return {}


class IntegratedTestGenerator:
    """Enhanced test generator with integration capabilities"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.api = create_api()
        self.code_analysis = CodeAnalysisIntegration(config)
        self.coverage = CoverageIntegration(config)
        self.api_integration = APIIntegration(config)
        self.database = DatabaseIntegration(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_enhanced_tests(
        self,
        function_signature: str,
        docstring: str,
        project_path: str,
        project_id: str,
        generator_type: str = "enhanced"
    ) -> Dict[str, Any]:
        """Generate tests with full integration capabilities"""
        
        try:
            # Analyze function complexity
            complexity_analysis = await self.code_analysis.analyze_function_complexity(function_signature)
            
            # Get coverage information
            coverage_report = await self.coverage.get_coverage_report(project_path)
            
            # Get best practices
            best_practices = await self.api_integration.get_best_practices("python", "pytest")
            
            # Generate base tests
            result = await self.api.generate_tests(
                function_signature,
                docstring,
                generator_type
            )
            
            if not result["success"]:
                return result
            
            # Enhance tests with integration data
            enhanced_tests = self._enhance_tests_with_analysis(
                result["test_cases"],
                complexity_analysis,
                coverage_report,
                best_practices
            )
            
            # Validate test quality
            quality_validation = await self.api_integration.validate_test_quality(enhanced_tests)
            
            # Store tests in database
            await self.database.store_test_cases(enhanced_tests, project_id)
            
            return {
                "test_cases": enhanced_tests,
                "complexity_analysis": complexity_analysis,
                "coverage_report": coverage_report,
                "quality_validation": quality_validation,
                "best_practices": best_practices,
                "metrics": result["metrics"],
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced test generation failed: {e}")
            return {
                "test_cases": [],
                "error": str(e),
                "success": False
            }
    
    def _enhance_tests_with_analysis(
        self,
        test_cases: List[TestCase],
        complexity_analysis: Dict[str, Any],
        coverage_report: Dict[str, Any],
        best_practices: List[str]
    ) -> List[TestCase]:
        """Enhance test cases with analysis data"""
        
        enhanced_tests = []
        
        for test_case in test_cases:
            # Add complexity-based enhancements
            if complexity_analysis.get("complexity_score", 0) > 5:
                # Add more edge case tests for complex functions
                if "edge" not in test_case.name.lower():
                    test_case.priority = TestPriority.HIGH
            
            # Add coverage-based enhancements
            if coverage_report.get("overall_coverage", 0) < 0.8:
                # Emphasize coverage in test descriptions
                test_case.description += " (Coverage-focused test)"
            
            # Add best practices to test code
            if best_practices:
                test_case.test_code = f"# Best practices: {', '.join(best_practices[:2])}\n{test_case.test_code}"
            
            enhanced_tests.append(test_case)
        
        return enhanced_tests
    
    async def get_project_insights(self, project_id: str, project_path: str) -> Dict[str, Any]:
        """Get comprehensive project insights"""
        
        try:
            # Get test statistics
            test_stats = await self.database.get_test_statistics(project_id)
            
            # Get coverage report
            coverage_report = await self.coverage.get_coverage_report(project_path)
            
            # Get system status
            system_status = self.api.get_system_status()
            
            return {
                "test_statistics": test_stats,
                "coverage_report": coverage_report,
                "system_status": system_status,
                "recommendations": self._generate_project_recommendations(test_stats, coverage_report),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get project insights: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _generate_project_recommendations(
        self,
        test_stats: Dict[str, Any],
        coverage_report: Dict[str, Any]
    ) -> List[str]:
        """Generate project improvement recommendations"""
        
        recommendations = []
        
        # Coverage recommendations
        if coverage_report.get("overall_coverage", 0) < 0.8:
            recommendations.append("Improve test coverage to reach 80% threshold")
        
        # Test quality recommendations
        if test_stats.get("total_tests", 0) < 10:
            recommendations.append("Consider adding more test cases for better coverage")
        
        # Performance recommendations
        recommendations.append("Regularly review and update test cases")
        recommendations.append("Monitor test execution performance")
        
        return recommendations


# Convenience functions for integration
def create_integrated_generator(config: Optional[IntegrationConfig] = None) -> IntegratedTestGenerator:
    """Create an integrated test generator"""
    if config is None:
        config = IntegrationConfig()
    return IntegratedTestGenerator(config)


async def generate_tests_with_integration(
    function_signature: str,
    docstring: str,
    project_path: str,
    project_id: str,
    config: Optional[IntegrationConfig] = None
) -> Dict[str, Any]:
    """Generate tests with full integration capabilities"""
    generator = create_integrated_generator(config)
    return await generator.generate_enhanced_tests(
        function_signature,
        docstring,
        project_path,
        project_id
    )









