"""
Quality Analyzer and Reporter
=============================

Comprehensive quality analysis and reporting system for test case generation
that provides detailed insights into the quality, performance, and effectiveness
of generated test cases.

This analyzer focuses on:
- Comprehensive quality analysis
- Performance metrics and reporting
- Quality trend analysis
- Optimization recommendations
- Detailed reporting and visualization
"""

import ast
import inspect
import re
import random
import string
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for test cases"""
    uniqueness: float
    diversity: float
    intuition: float
    creativity: float
    coverage: float
    overall_quality: float
    # Additional metrics
    complexity_score: float
    maintainability_score: float
    readability_score: float
    performance_score: float
    reliability_score: float


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    total_tests: int
    average_quality: float
    quality_distribution: Dict[str, int]
    quality_breakdown: Dict[str, float]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    trends: Dict[str, List[float]]
    generated_at: datetime


@dataclass
class TestCaseAnalysis:
    """Individual test case analysis"""
    test_case_id: str
    name: str
    function_name: str
    test_type: str
    quality_metrics: QualityMetrics
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    optimization_potential: float


class QualityAnalyzer:
    """Comprehensive quality analyzer for test case generation"""
    
    def __init__(self):
        self.quality_calculators = self._setup_quality_calculators()
        self.performance_analyzers = self._setup_performance_analyzers()
        self.recommendation_engines = self._setup_recommendation_engines()
        self.trend_analyzers = self._setup_trend_analyzers()
        
    def _setup_quality_calculators(self) -> Dict[str, Callable]:
        """Setup quality calculation functions"""
        return {
            "uniqueness": self._calculate_uniqueness_metrics,
            "diversity": self._calculate_diversity_metrics,
            "intuition": self._calculate_intuition_metrics,
            "creativity": self._calculate_creativity_metrics,
            "coverage": self._calculate_coverage_metrics,
            "complexity": self._calculate_complexity_metrics,
            "maintainability": self._calculate_maintainability_metrics,
            "readability": self._calculate_readability_metrics,
            "performance": self._calculate_performance_metrics,
            "reliability": self._calculate_reliability_metrics
        }
    
    def _setup_performance_analyzers(self) -> Dict[str, Callable]:
        """Setup performance analysis functions"""
        return {
            "generation_speed": self._analyze_generation_speed,
            "memory_usage": self._analyze_memory_usage,
            "execution_time": self._analyze_execution_time,
            "scalability": self._analyze_scalability
        }
    
    def _setup_recommendation_engines(self) -> Dict[str, Callable]:
        """Setup recommendation engines"""
        return {
            "quality_improvements": self._recommend_quality_improvements,
            "performance_optimizations": self._recommend_performance_optimizations,
            "best_practices": self._recommend_best_practices,
            "optimization_strategies": self._recommend_optimization_strategies
        }
    
    def _setup_trend_analyzers(self) -> Dict[str, Callable]:
        """Setup trend analysis functions"""
        return {
            "quality_trends": self._analyze_quality_trends,
            "performance_trends": self._analyze_performance_trends,
            "improvement_trends": self._analyze_improvement_trends
        }
    
    def analyze_test_cases(self, test_cases: List[Any]) -> QualityReport:
        """Analyze test cases and generate comprehensive quality report"""
        # Calculate individual test case analyses
        individual_analyses = []
        for i, test_case in enumerate(test_cases):
            analysis = self._analyze_individual_test_case(test_case, f"test_{i}")
            individual_analyses.append(analysis)
        
        # Calculate aggregate metrics
        total_tests = len(test_cases)
        average_quality = statistics.mean([a.quality_metrics.overall_quality for a in individual_analyses])
        
        # Calculate quality distribution
        quality_distribution = self._calculate_quality_distribution(individual_analyses)
        
        # Calculate quality breakdown
        quality_breakdown = self._calculate_quality_breakdown(individual_analyses)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(individual_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(individual_analyses)
        
        # Analyze trends
        trends = self._analyze_trends(individual_analyses)
        
        return QualityReport(
            total_tests=total_tests,
            average_quality=average_quality,
            quality_distribution=quality_distribution,
            quality_breakdown=quality_breakdown,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            trends=trends,
            generated_at=datetime.now()
        )
    
    def _analyze_individual_test_case(self, test_case: Any, test_case_id: str) -> TestCaseAnalysis:
        """Analyze individual test case"""
        # Calculate quality metrics
        quality_metrics = QualityMetrics(
            uniqueness=self._calculate_uniqueness_metrics(test_case),
            diversity=self._calculate_diversity_metrics(test_case),
            intuition=self._calculate_intuition_metrics(test_case),
            creativity=self._calculate_creativity_metrics(test_case),
            coverage=self._calculate_coverage_metrics(test_case),
            overall_quality=self._calculate_overall_quality(test_case),
            complexity_score=self._calculate_complexity_metrics(test_case),
            maintainability_score=self._calculate_maintainability_metrics(test_case),
            readability_score=self._calculate_readability_metrics(test_case),
            performance_score=self._calculate_performance_metrics(test_case),
            reliability_score=self._calculate_reliability_metrics(test_case)
        )
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(test_case, quality_metrics)
        weaknesses = self._identify_weaknesses(test_case, quality_metrics)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(test_case, quality_metrics)
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(quality_metrics)
        
        return TestCaseAnalysis(
            test_case_id=test_case_id,
            name=getattr(test_case, 'name', 'Unknown'),
            function_name=getattr(test_case, 'function_name', 'Unknown'),
            test_type=getattr(test_case, 'test_type', 'Unknown'),
            quality_metrics=quality_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvement_suggestions,
            optimization_potential=optimization_potential
        )
    
    def _calculate_uniqueness_metrics(self, test_case: Any) -> float:
        """Calculate uniqueness metrics"""
        score = 0.0
        
        # Name uniqueness
        name = getattr(test_case, 'name', '')
        if 'unique' in name.lower() or 'creative' in name.lower():
            score += 0.3
        
        # Parameter uniqueness
        parameters = getattr(test_case, 'parameters', {})
        if len(parameters) > 3:
            score += 0.2
        
        # Assertion uniqueness
        assertions = getattr(test_case, 'assertions', [])
        if len(assertions) > 4:
            score += 0.2
        
        # Scenario uniqueness
        scenario = getattr(test_case, 'scenario', '')
        if 'unique' in scenario.lower() or 'creative' in scenario.lower():
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_diversity_metrics(self, test_case: Any) -> float:
        """Calculate diversity metrics"""
        score = 0.0
        
        # Parameter diversity
        parameters = getattr(test_case, 'parameters', {})
        param_types = set(type(v).__name__ for v in parameters.values())
        score += len(param_types) * 0.2
        
        # Test type diversity
        test_type = getattr(test_case, 'test_type', '')
        if test_type in ['unique', 'diverse', 'intuitive', 'creative']:
            score += 0.2
        
        # Scenario diversity
        scenario = getattr(test_case, 'scenario', '')
        if 'diverse' in scenario.lower() or 'comprehensive' in scenario.lower():
            score += 0.3
        
        # Assertion diversity
        assertions = getattr(test_case, 'assertions', [])
        if len(assertions) > 3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_intuition_metrics(self, test_case: Any) -> float:
        """Calculate intuition metrics"""
        score = 0.0
        
        # Name clarity
        name = getattr(test_case, 'name', '')
        if 'should' in name.lower():
            score += 0.3
        
        # Description clarity
        description = getattr(test_case, 'description', '')
        if 'verify' in description.lower() or 'test' in description.lower():
            score += 0.3
        
        # Assertion clarity
        assertions = getattr(test_case, 'assertions', [])
        clear_assertions = sum(1 for a in assertions if 'assert' in a.lower())
        score += (clear_assertions / max(len(assertions), 1)) * 0.4
        
        return min(score, 1.0)
    
    def _calculate_creativity_metrics(self, test_case: Any) -> float:
        """Calculate creativity metrics"""
        score = 0.0
        
        # Name creativity
        name = getattr(test_case, 'name', '')
        if 'creative' in name.lower() or 'innovative' in name.lower():
            score += 0.4
        
        # Parameter creativity
        parameters = getattr(test_case, 'parameters', {})
        creative_params = sum(1 for k, v in parameters.items() if 'creative' in k.lower() or 'unique' in k.lower())
        score += (creative_params / max(len(parameters), 1)) * 0.3
        
        # Assertion creativity
        assertions = getattr(test_case, 'assertions', [])
        creative_assertions = sum(1 for a in assertions if 'creative' in a.lower() or 'innovative' in a.lower())
        score += (creative_assertions / max(len(assertions), 1)) * 0.3
        
        return min(score, 1.0)
    
    def _calculate_coverage_metrics(self, test_case: Any) -> float:
        """Calculate coverage metrics"""
        score = 0.0
        
        # Assertion coverage
        assertions = getattr(test_case, 'assertions', [])
        if len(assertions) > 5:
            score += 0.4
        
        # Scenario coverage
        scenario = getattr(test_case, 'scenario', '')
        if 'comprehensive' in scenario.lower() or 'coverage' in scenario.lower():
            score += 0.3
        
        # Edge case coverage
        if 'edge' in scenario.lower() or 'boundary' in scenario.lower():
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_complexity_metrics(self, test_case: Any) -> float:
        """Calculate complexity metrics"""
        score = 0.0
        
        # Parameter complexity
        parameters = getattr(test_case, 'parameters', {})
        score += len(parameters) * 0.1
        
        # Assertion complexity
        assertions = getattr(test_case, 'assertions', [])
        score += len(assertions) * 0.05
        
        # Setup/teardown complexity
        setup_code = getattr(test_case, 'setup_code', '')
        teardown_code = getattr(test_case, 'teardown_code', '')
        score += (len(setup_code) + len(teardown_code)) * 0.001
        
        return min(score, 1.0)
    
    def _calculate_maintainability_metrics(self, test_case: Any) -> float:
        """Calculate maintainability metrics"""
        score = 0.0
        
        # Name clarity
        name = getattr(test_case, 'name', '')
        if len(name) > 10 and 'test' in name.lower():
            score += 0.3
        
        # Description clarity
        description = getattr(test_case, 'description', '')
        if len(description) > 20:
            score += 0.3
        
        # Code organization
        assertions = getattr(test_case, 'assertions', [])
        if len(assertions) > 2:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_readability_metrics(self, test_case: Any) -> float:
        """Calculate readability metrics"""
        score = 0.0
        
        # Name readability
        name = getattr(test_case, 'name', '')
        if '_' in name or 'should' in name.lower():
            score += 0.4
        
        # Description readability
        description = getattr(test_case, 'description', '')
        if 'verify' in description.lower() or 'test' in description.lower():
            score += 0.3
        
        # Assertion readability
        assertions = getattr(test_case, 'assertions', [])
        readable_assertions = sum(1 for a in assertions if 'assert' in a.lower())
        score += (readable_assertions / max(len(assertions), 1)) * 0.3
        
        return min(score, 1.0)
    
    def _calculate_performance_metrics(self, test_case: Any) -> float:
        """Calculate performance metrics"""
        score = 0.0
        
        # Parameter efficiency
        parameters = getattr(test_case, 'parameters', {})
        if len(parameters) <= 5:
            score += 0.4
        
        # Assertion efficiency
        assertions = getattr(test_case, 'assertions', [])
        if len(assertions) <= 8:
            score += 0.3
        
        # Code efficiency
        setup_code = getattr(test_case, 'setup_code', '')
        teardown_code = getattr(test_case, 'teardown_code', '')
        if len(setup_code) + len(teardown_code) <= 200:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_reliability_metrics(self, test_case: Any) -> float:
        """Calculate reliability metrics"""
        score = 0.0
        
        # Assertion reliability
        assertions = getattr(test_case, 'assertions', [])
        reliable_assertions = sum(1 for a in assertions if 'assert' in a.lower() and 'not None' in a.lower())
        score += (reliable_assertions / max(len(assertions), 1)) * 0.5
        
        # Error handling
        if 'error' in str(test_case).lower() or 'exception' in str(test_case).lower():
            score += 0.3
        
        # Validation
        if 'valid' in str(test_case).lower() or 'validate' in str(test_case).lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_overall_quality(self, test_case: Any) -> float:
        """Calculate overall quality score"""
        uniqueness = self._calculate_uniqueness_metrics(test_case)
        diversity = self._calculate_diversity_metrics(test_case)
        intuition = self._calculate_intuition_metrics(test_case)
        creativity = self._calculate_creativity_metrics(test_case)
        coverage = self._calculate_coverage_metrics(test_case)
        
        return (
            uniqueness * 0.25 +
            diversity * 0.25 +
            intuition * 0.25 +
            creativity * 0.15 +
            coverage * 0.10
        )
    
    def _calculate_quality_distribution(self, analyses: List[TestCaseAnalysis]) -> Dict[str, int]:
        """Calculate quality distribution"""
        distribution = {
            "excellent": 0,  # > 0.9
            "good": 0,       # 0.7-0.9
            "fair": 0,       # 0.5-0.7
            "poor": 0        # < 0.5
        }
        
        for analysis in analyses:
            quality = analysis.quality_metrics.overall_quality
            if quality > 0.9:
                distribution["excellent"] += 1
            elif quality > 0.7:
                distribution["good"] += 1
            elif quality > 0.5:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _calculate_quality_breakdown(self, analyses: List[TestCaseAnalysis]) -> Dict[str, float]:
        """Calculate quality breakdown"""
        if not analyses:
            return {}
        
        breakdown = {
            "uniqueness": statistics.mean([a.quality_metrics.uniqueness for a in analyses]),
            "diversity": statistics.mean([a.quality_metrics.diversity for a in analyses]),
            "intuition": statistics.mean([a.quality_metrics.intuition for a in analyses]),
            "creativity": statistics.mean([a.quality_metrics.creativity for a in analyses]),
            "coverage": statistics.mean([a.quality_metrics.coverage for a in analyses]),
            "complexity": statistics.mean([a.quality_metrics.complexity_score for a in analyses]),
            "maintainability": statistics.mean([a.quality_metrics.maintainability_score for a in analyses]),
            "readability": statistics.mean([a.quality_metrics.readability_score for a in analyses]),
            "performance": statistics.mean([a.quality_metrics.performance_score for a in analyses]),
            "reliability": statistics.mean([a.quality_metrics.reliability_score for a in analyses])
        }
        
        return breakdown
    
    def _calculate_performance_metrics(self, analyses: List[TestCaseAnalysis]) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not analyses:
            return {}
        
        return {
            "average_parameter_count": statistics.mean([len(getattr(a, 'parameters', {})) for a in analyses]),
            "average_assertion_count": statistics.mean([len(getattr(a, 'assertions', [])) for a in analyses]),
            "average_name_length": statistics.mean([len(getattr(a, 'name', '')) for a in analyses]),
            "average_description_length": statistics.mean([len(getattr(a, 'description', '')) for a in analyses])
        }
    
    def _generate_recommendations(self, analyses: List[TestCaseAnalysis]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze overall quality
        avg_quality = statistics.mean([a.quality_metrics.overall_quality for a in analyses])
        if avg_quality < 0.6:
            recommendations.append("Overall quality is below target. Consider implementing quality optimization strategies.")
        
        # Analyze specific metrics
        avg_uniqueness = statistics.mean([a.quality_metrics.uniqueness for a in analyses])
        if avg_uniqueness < 0.6:
            recommendations.append("Uniqueness scores are low. Consider adding more creative and distinctive test cases.")
        
        avg_diversity = statistics.mean([a.quality_metrics.diversity for a in analyses])
        if avg_diversity < 0.6:
            recommendations.append("Diversity scores are low. Consider adding more varied test scenarios and parameter types.")
        
        avg_intuition = statistics.mean([a.quality_metrics.intuition for a in analyses])
        if avg_intuition < 0.6:
            recommendations.append("Intuition scores are low. Consider improving test naming and descriptions for clarity.")
        
        avg_creativity = statistics.mean([a.quality_metrics.creativity for a in analyses])
        if avg_creativity < 0.6:
            recommendations.append("Creativity scores are low. Consider adding more innovative and creative test approaches.")
        
        avg_coverage = statistics.mean([a.quality_metrics.coverage for a in analyses])
        if avg_coverage < 0.6:
            recommendations.append("Coverage scores are low. Consider adding more comprehensive assertions and edge cases.")
        
        return recommendations
    
    def _analyze_trends(self, analyses: List[TestCaseAnalysis]) -> Dict[str, List[float]]:
        """Analyze trends in test case quality"""
        if len(analyses) < 2:
            return {}
        
        # Sort by test case ID to ensure consistent ordering
        sorted_analyses = sorted(analyses, key=lambda x: x.test_case_id)
        
        trends = {
            "uniqueness": [a.quality_metrics.uniqueness for a in sorted_analyses],
            "diversity": [a.quality_metrics.diversity for a in sorted_analyses],
            "intuition": [a.quality_metrics.intuition for a in sorted_analyses],
            "creativity": [a.quality_metrics.creativity for a in sorted_analyses],
            "coverage": [a.quality_metrics.coverage for a in sorted_analyses],
            "overall_quality": [a.quality_metrics.overall_quality for a in sorted_analyses]
        }
        
        return trends
    
    def _identify_strengths(self, test_case: Any, quality_metrics: QualityMetrics) -> List[str]:
        """Identify strengths in test case"""
        strengths = []
        
        if quality_metrics.uniqueness > 0.8:
            strengths.append("High uniqueness score")
        if quality_metrics.diversity > 0.8:
            strengths.append("High diversity score")
        if quality_metrics.intuition > 0.8:
            strengths.append("High intuition score")
        if quality_metrics.creativity > 0.8:
            strengths.append("High creativity score")
        if quality_metrics.coverage > 0.8:
            strengths.append("High coverage score")
        if quality_metrics.maintainability_score > 0.8:
            strengths.append("High maintainability")
        if quality_metrics.readability_score > 0.8:
            strengths.append("High readability")
        
        return strengths
    
    def _identify_weaknesses(self, test_case: Any, quality_metrics: QualityMetrics) -> List[str]:
        """Identify weaknesses in test case"""
        weaknesses = []
        
        if quality_metrics.uniqueness < 0.5:
            weaknesses.append("Low uniqueness score")
        if quality_metrics.diversity < 0.5:
            weaknesses.append("Low diversity score")
        if quality_metrics.intuition < 0.5:
            weaknesses.append("Low intuition score")
        if quality_metrics.creativity < 0.5:
            weaknesses.append("Low creativity score")
        if quality_metrics.coverage < 0.5:
            weaknesses.append("Low coverage score")
        if quality_metrics.maintainability_score < 0.5:
            weaknesses.append("Low maintainability")
        if quality_metrics.readability_score < 0.5:
            weaknesses.append("Low readability")
        
        return weaknesses
    
    def _generate_improvement_suggestions(self, test_case: Any, quality_metrics: QualityMetrics) -> List[str]:
        """Generate improvement suggestions for test case"""
        suggestions = []
        
        if quality_metrics.uniqueness < 0.6:
            suggestions.append("Add more unique and creative elements to test case")
        if quality_metrics.diversity < 0.6:
            suggestions.append("Increase parameter and scenario diversity")
        if quality_metrics.intuition < 0.6:
            suggestions.append("Improve test naming and description clarity")
        if quality_metrics.creativity < 0.6:
            suggestions.append("Add more creative and innovative approaches")
        if quality_metrics.coverage < 0.6:
            suggestions.append("Add more comprehensive assertions and edge cases")
        
        return suggestions
    
    def _calculate_optimization_potential(self, quality_metrics: QualityMetrics) -> float:
        """Calculate optimization potential for test case"""
        # Calculate how much improvement is possible
        max_possible = 1.0
        current_quality = quality_metrics.overall_quality
        optimization_potential = max_possible - current_quality
        
        return min(optimization_potential, 1.0)
    
    def generate_report(self, quality_report: QualityReport, format: str = "text") -> str:
        """Generate quality report in specified format"""
        if format == "json":
            return self._generate_json_report(quality_report)
        elif format == "html":
            return self._generate_html_report(quality_report)
        else:
            return self._generate_text_report(quality_report)
    
    def _generate_text_report(self, quality_report: QualityReport) -> str:
        """Generate text format quality report"""
        report = []
        report.append("=" * 80)
        report.append("QUALITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {quality_report.generated_at}")
        report.append(f"Total tests analyzed: {quality_report.total_tests}")
        report.append(f"Average quality: {quality_report.average_quality:.3f}")
        report.append("")
        
        # Quality distribution
        report.append("QUALITY DISTRIBUTION:")
        report.append("-" * 40)
        for category, count in quality_report.quality_distribution.items():
            percentage = (count / quality_report.total_tests) * 100
            report.append(f"{category.capitalize()}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Quality breakdown
        report.append("QUALITY BREAKDOWN:")
        report.append("-" * 40)
        for metric, score in quality_report.quality_breakdown.items():
            report.append(f"{metric.capitalize()}: {score:.3f}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 40)
        for metric, value in quality_report.performance_metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        for i, recommendation in enumerate(quality_report.recommendations, 1):
            report.append(f"{i}. {recommendation}")
        report.append("")
        
        return "\n".join(report)
    
    def _generate_json_report(self, quality_report: QualityReport) -> str:
        """Generate JSON format quality report"""
        report_data = {
            "generated_at": quality_report.generated_at.isoformat(),
            "total_tests": quality_report.total_tests,
            "average_quality": quality_report.average_quality,
            "quality_distribution": quality_report.quality_distribution,
            "quality_breakdown": quality_report.quality_breakdown,
            "performance_metrics": quality_report.performance_metrics,
            "recommendations": quality_report.recommendations,
            "trends": quality_report.trends
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self, quality_report: QualityReport) -> str:
        """Generate HTML format quality report"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head><title>Quality Analysis Report</title></head><body>")
        html.append("<h1>Quality Analysis Report</h1>")
        html.append(f"<p>Generated at: {quality_report.generated_at}</p>")
        html.append(f"<p>Total tests analyzed: {quality_report.total_tests}</p>")
        html.append(f"<p>Average quality: {quality_report.average_quality:.3f}</p>")
        
        # Quality distribution
        html.append("<h2>Quality Distribution</h2>")
        html.append("<ul>")
        for category, count in quality_report.quality_distribution.items():
            percentage = (count / quality_report.total_tests) * 100
            html.append(f"<li>{category.capitalize()}: {count} ({percentage:.1f}%)</li>")
        html.append("</ul>")
        
        # Quality breakdown
        html.append("<h2>Quality Breakdown</h2>")
        html.append("<ul>")
        for metric, score in quality_report.quality_breakdown.items():
            html.append(f"<li>{metric.capitalize()}: {score:.3f}</li>")
        html.append("</ul>")
        
        # Recommendations
        html.append("<h2>Recommendations</h2>")
        html.append("<ol>")
        for recommendation in quality_report.recommendations:
            html.append(f"<li>{recommendation}</li>")
        html.append("</ol>")
        
        html.append("</body></html>")
        return "\n".join(html)


def demonstrate_quality_analyzer():
    """Demonstrate the quality analyzer"""
    
    # Create sample test cases
    test_cases = [
        type('TestCase', (), {
            'name': 'test_validate_user_unique',
            'description': 'Test user validation with unique approach',
            'function_name': 'validate_user',
            'test_type': 'unique',
            'scenario': 'unique_validation',
            'parameters': {'user_data': {'name': 'John', 'email': 'john@example.com'}, 'unique_param': 'creative_value'},
            'assertions': ['assert result is not None', 'assert result is unique and distinctive'],
            'setup_code': 'setup unique test',
            'teardown_code': 'cleanup unique test'
        })(),
        type('TestCase', (), {
            'name': 'test_transform_data_diverse',
            'description': 'Test data transformation with diverse scenarios',
            'function_name': 'transform_data',
            'test_type': 'diverse',
            'scenario': 'diverse_transform',
            'parameters': {'data': [1, 2, 3], 'format': 'json', 'diverse_param': 'varied_value'},
            'assertions': ['assert result is not None', 'assert isinstance(result, dict)', 'assert result covers multiple scenarios'],
            'setup_code': 'setup diverse test',
            'teardown_code': 'cleanup diverse test'
        })()
    ]
    
    # Create analyzer
    analyzer = QualityAnalyzer()
    
    # Analyze test cases
    quality_report = analyzer.analyze_test_cases(test_cases)
    
    # Generate report
    report_text = analyzer.generate_report(quality_report, format="text")
    
    print("QUALITY ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print(report_text)


if __name__ == "__main__":
    demonstrate_quality_analyzer()
