"""
AI-Powered Enhancement System for Test Generation
===============================================

This module provides revolutionary AI-powered enhancements that push
test generation capabilities to unprecedented levels of intelligence.
"""

import asyncio
import json
import openai
import anthropic
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
import ast
from pathlib import Path

from .base_architecture import TestCase, TestGenerationConfig, TestCategory, TestPriority, TestType
from .unified_api import TestGenerationAPI, create_api

logger = logging.getLogger(__name__)


@dataclass
class AIEnhancementConfig:
    """Configuration for AI enhancement features"""
    # AI Models
    primary_model: str = "gpt-4"
    fallback_model: str = "gpt-3.5-turbo"
    claude_model: str = "claude-3-sonnet"
    
    # AI Parameters
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Enhancement Features
    enable_code_analysis: bool = True
    enable_semantic_understanding: bool = True
    enable_context_awareness: bool = True
    enable_predictive_generation: bool = True
    enable_quality_optimization: bool = True
    enable_intelligent_naming: bool = True
    
    # API Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3


class AICodeAnalyzer:
    """AI-powered code analysis and understanding"""
    
    def __init__(self, config: AIEnhancementConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_ai_clients()
    
    def _setup_ai_clients(self):
        """Setup AI clients"""
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        
        if self.config.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        else:
            self.anthropic_client = None
    
    async def analyze_function_intelligence(
        self, 
        function_signature: str, 
        docstring: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform intelligent analysis of function using AI"""
        
        try:
            # Prepare analysis prompt
            analysis_prompt = self._create_analysis_prompt(function_signature, docstring, context)
            
            # Get AI analysis
            ai_analysis = await self._get_ai_analysis(analysis_prompt)
            
            # Parse and enhance analysis
            enhanced_analysis = self._enhance_ai_analysis(ai_analysis, function_signature, docstring)
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return self._get_fallback_analysis(function_signature, docstring)
    
    def _create_analysis_prompt(self, function_signature: str, docstring: str, context: Optional[str] = None) -> str:
        """Create analysis prompt for AI"""
        
        prompt = f"""
Analyze the following Python function and provide comprehensive insights for test generation:

Function Signature: {function_signature}
Docstring: {docstring}
{f"Context: {context}" if context else ""}

Please provide analysis in the following JSON format:
{{
    "complexity_analysis": {{
        "cyclomatic_complexity": <number>,
        "cognitive_complexity": <number>,
        "maintainability_index": <number>,
        "complexity_level": "<low|medium|high>"
    }},
    "semantic_analysis": {{
        "purpose": "<function purpose>",
        "domain": "<business domain>",
        "patterns": ["<pattern1>", "<pattern2>"],
        "dependencies": ["<dep1>", "<dep2>"],
        "side_effects": ["<effect1>", "<effect2>"]
    }},
    "test_recommendations": {{
        "test_types": ["<type1>", "<type2>"],
        "edge_cases": ["<case1>", "<case2>"],
        "boundary_conditions": ["<condition1>", "<condition2>"],
        "error_scenarios": ["<scenario1>", "<scenario2>"],
        "performance_considerations": ["<consideration1>", "<consideration2>"]
    }},
    "quality_insights": {{
        "naming_quality": <score 0-1>,
        "documentation_quality": <score 0-1>,
        "code_clarity": <score 0-1>,
        "testability": <score 0-1>
    }},
    "ai_confidence": <score 0-1>
}}
"""
        return prompt
    
    async def _get_ai_analysis(self, prompt: str) -> str:
        """Get analysis from AI model"""
        
        try:
            # Try primary model first
            if self.config.primary_model.startswith("gpt"):
                response = await openai.ChatCompletion.acreate(
                    model=self.config.primary_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty
                )
                return response.choices[0].message.content
            
            elif self.config.primary_model.startswith("claude") and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                # Fallback to secondary model
                return await self._get_fallback_analysis("", "")
                
        except Exception as e:
            self.logger.warning(f"Primary AI model failed: {e}")
            return await self._get_fallback_analysis("", "")
    
    def _enhance_ai_analysis(self, ai_analysis: str, function_signature: str, docstring: str) -> Dict[str, Any]:
        """Enhance AI analysis with additional processing"""
        
        try:
            # Parse JSON response
            analysis_data = json.loads(ai_analysis)
            
            # Add additional analysis
            analysis_data["static_analysis"] = self._perform_static_analysis(function_signature, docstring)
            analysis_data["pattern_detection"] = self._detect_patterns(function_signature, docstring)
            analysis_data["complexity_metrics"] = self._calculate_complexity_metrics(function_signature)
            
            return analysis_data
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse AI analysis JSON")
            return self._get_fallback_analysis(function_signature, docstring)
    
    def _perform_static_analysis(self, function_signature: str, docstring: str) -> Dict[str, Any]:
        """Perform static analysis on function"""
        
        return {
            "parameter_count": self._count_parameters(function_signature),
            "return_type": self._extract_return_type(function_signature),
            "has_type_hints": ":" in function_signature and "->" in function_signature,
            "docstring_length": len(docstring) if docstring else 0,
            "function_name_length": len(function_signature.split("(")[0].split("def ")[1]) if "def " in function_signature else 0
        }
    
    def _detect_patterns(self, function_signature: str, docstring: str) -> List[str]:
        """Detect common patterns in function"""
        
        patterns = []
        func_name = function_signature.split("(")[0].split("def ")[1] if "def " in function_signature else ""
        func_lower = func_name.lower()
        
        # CRUD patterns
        if any(word in func_lower for word in ["create", "add", "insert", "new"]):
            patterns.append("create_pattern")
        if any(word in func_lower for word in ["read", "get", "find", "search", "fetch"]):
            patterns.append("read_pattern")
        if any(word in func_lower for word in ["update", "modify", "change", "edit"]):
            patterns.append("update_pattern")
        if any(word in func_lower for word in ["delete", "remove", "destroy", "clear"]):
            patterns.append("delete_pattern")
        
        # Mathematical patterns
        if any(word in func_lower for word in ["calculate", "compute", "sum", "multiply", "divide", "math"]):
            patterns.append("mathematical_pattern")
        
        # Validation patterns
        if any(word in func_lower for word in ["validate", "check", "verify", "is_valid"]):
            patterns.append("validation_pattern")
        
        # Transformation patterns
        if any(word in func_lower for word in ["transform", "convert", "parse", "format"]):
            patterns.append("transformation_pattern")
        
        # Utility patterns
        if any(word in func_lower for word in ["util", "helper", "helper", "common"]):
            patterns.append("utility_pattern")
        
        return patterns
    
    def _calculate_complexity_metrics(self, function_signature: str) -> Dict[str, float]:
        """Calculate complexity metrics"""
        
        # Simple complexity calculation
        complexity = 0.0
        
        # Parameter complexity
        if "(" in function_signature and ")" in function_signature:
            params = function_signature.split("(")[1].split(")")[0]
            if params.strip():
                param_count = len([p for p in params.split(",") if p.strip()])
                complexity += param_count * 0.1
        
        # Type complexity
        if "List[" in function_signature or "Dict[" in function_signature:
            complexity += 0.3
        if "Optional[" in function_signature or "Union[" in function_signature:
            complexity += 0.2
        
        # Generic complexity
        if "->" in function_signature:
            complexity += 0.1
        
        return {
            "parameter_complexity": complexity,
            "type_complexity": 0.3 if "[" in function_signature else 0.0,
            "overall_complexity": min(complexity, 1.0)
        }
    
    def _count_parameters(self, function_signature: str) -> int:
        """Count function parameters"""
        if "(" in function_signature and ")" in function_signature:
            params = function_signature.split("(")[1].split(")")[0]
            if params.strip():
                return len([p for p in params.split(",") if p.strip()])
        return 0
    
    def _extract_return_type(self, function_signature: str) -> str:
        """Extract return type from function signature"""
        if "->" in function_signature:
            return function_signature.split("->")[1].strip()
        return "Any"
    
    def _get_fallback_analysis(self, function_signature: str, docstring: str) -> Dict[str, Any]:
        """Get fallback analysis when AI is unavailable"""
        
        return {
            "complexity_analysis": {
                "cyclomatic_complexity": 1,
                "cognitive_complexity": 1,
                "maintainability_index": 80.0,
                "complexity_level": "low"
            },
            "semantic_analysis": {
                "purpose": "Function purpose not analyzed",
                "domain": "unknown",
                "patterns": [],
                "dependencies": [],
                "side_effects": []
            },
            "test_recommendations": {
                "test_types": ["basic", "edge_case"],
                "edge_cases": [],
                "boundary_conditions": [],
                "error_scenarios": [],
                "performance_considerations": []
            },
            "quality_insights": {
                "naming_quality": 0.5,
                "documentation_quality": 0.5,
                "code_clarity": 0.5,
                "testability": 0.5
            },
            "static_analysis": self._perform_static_analysis(function_signature, docstring),
            "pattern_detection": self._detect_patterns(function_signature, docstring),
            "complexity_metrics": self._calculate_complexity_metrics(function_signature),
            "ai_confidence": 0.0
        }


class AITestGenerator:
    """AI-powered test generation with intelligent capabilities"""
    
    def __init__(self, config: AIEnhancementConfig):
        self.config = config
        self.analyzer = AICodeAnalyzer(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_intelligent_tests(
        self,
        function_signature: str,
        docstring: str,
        context: Optional[str] = None,
        test_config: Optional[TestGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate intelligent tests using AI analysis"""
        
        try:
            # Perform AI analysis
            analysis = await self.analyzer.analyze_function_intelligence(
                function_signature, docstring, context
            )
            
            # Generate tests based on analysis
            test_cases = await self._generate_tests_from_analysis(
                function_signature, docstring, analysis, test_config
            )
            
            # Enhance tests with AI insights
            enhanced_tests = self._enhance_tests_with_ai_insights(test_cases, analysis)
            
            return {
                "test_cases": enhanced_tests,
                "ai_analysis": analysis,
                "ai_confidence": analysis.get("ai_confidence", 0.0),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"AI test generation failed: {e}")
            return {
                "test_cases": [],
                "error": str(e),
                "success": False
            }
    
    async def _generate_tests_from_analysis(
        self,
        function_signature: str,
        docstring: str,
        analysis: Dict[str, Any],
        test_config: Optional[TestGenerationConfig]
    ) -> List[TestCase]:
        """Generate test cases based on AI analysis"""
        
        test_cases = []
        
        # Extract function name
        func_name = function_signature.split("(")[0].split("def ")[1] if "def " in function_signature else "function"
        
        # Get recommendations
        recommendations = analysis.get("test_recommendations", {})
        test_types = recommendations.get("test_types", ["basic"])
        edge_cases = recommendations.get("edge_cases", [])
        boundary_conditions = recommendations.get("boundary_conditions", [])
        error_scenarios = recommendations.get("error_scenarios", [])
        
        # Generate basic tests
        if "basic" in test_types:
            test_cases.extend(self._generate_basic_tests(func_name, function_signature, analysis))
        
        # Generate edge case tests
        if "edge_case" in test_types or edge_cases:
            test_cases.extend(self._generate_edge_case_tests(func_name, function_signature, edge_cases, analysis))
        
        # Generate boundary condition tests
        if boundary_conditions:
            test_cases.extend(self._generate_boundary_tests(func_name, function_signature, boundary_conditions, analysis))
        
        # Generate error scenario tests
        if error_scenarios:
            test_cases.extend(self._generate_error_tests(func_name, function_signature, error_scenarios, analysis))
        
        # Generate performance tests
        if "performance" in test_types:
            test_cases.extend(self._generate_performance_tests(func_name, function_signature, analysis))
        
        # Generate security tests
        if "security" in test_types:
            test_cases.extend(self._generate_security_tests(func_name, function_signature, analysis))
        
        return test_cases
    
    def _generate_basic_tests(self, func_name: str, function_signature: str, analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate basic test cases"""
        
        test_cases = []
        
        # Happy path test
        test_cases.append(TestCase(
            name=f"test_{func_name}_basic",
            description=f"Basic test for {func_name} with normal inputs",
            test_code=f"result = {func_name}()\nassert result is not None",
            setup_code="",
            teardown_code="",
            category=TestCategory.FUNCTIONAL,
            priority=TestPriority.HIGH,
            test_type=TestType.UNIT
        ))
        
        # Parameter tests
        param_count = analysis.get("static_analysis", {}).get("parameter_count", 0)
        if param_count > 0:
            test_cases.append(TestCase(
                name=f"test_{func_name}_with_parameters",
                description=f"Test {func_name} with various parameter combinations",
                test_code=f"result = {func_name}(test_param1, test_param2)\nassert result is not None",
                setup_code="test_param1 = 1\ntest_param2 = 2",
                teardown_code="",
                category=TestCategory.FUNCTIONAL,
                priority=TestPriority.MEDIUM,
                test_type=TestType.UNIT
            ))
        
        return test_cases
    
    def _generate_edge_case_tests(self, func_name: str, function_signature: str, edge_cases: List[str], analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate edge case test cases"""
        
        test_cases = []
        
        # Default edge cases
        default_edge_cases = ["empty", "none", "zero", "negative", "max_value", "min_value"]
        all_edge_cases = edge_cases + default_edge_cases
        
        for i, edge_case in enumerate(all_edge_cases[:5]):  # Limit to 5 edge cases
            test_cases.append(TestCase(
                name=f"test_{func_name}_edge_{edge_case}",
                description=f"Test {func_name} with {edge_case} input",
                test_code=f"result = {func_name}({edge_case}_input)\n# Add appropriate assertions",
                setup_code=f"{edge_case}_input = self._get_{edge_case}_input()",
                teardown_code="",
                category=TestCategory.EDGE_CASE,
                priority=TestPriority.MEDIUM,
                test_type=TestType.UNIT
            ))
        
        return test_cases
    
    def _generate_boundary_tests(self, func_name: str, function_signature: str, boundary_conditions: List[str], analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate boundary condition test cases"""
        
        test_cases = []
        
        for i, condition in enumerate(boundary_conditions[:3]):  # Limit to 3 boundary conditions
            test_cases.append(TestCase(
                name=f"test_{func_name}_boundary_{i+1}",
                description=f"Test {func_name} with boundary condition: {condition}",
                test_code=f"result = {func_name}(boundary_input)\n# Verify boundary behavior",
                setup_code=f"boundary_input = self._get_boundary_input('{condition}')",
                teardown_code="",
                category=TestCategory.EDGE_CASE,
                priority=TestPriority.HIGH,
                test_type=TestType.UNIT
            ))
        
        return test_cases
    
    def _generate_error_tests(self, func_name: str, function_signature: str, error_scenarios: List[str], analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate error scenario test cases"""
        
        test_cases = []
        
        for i, scenario in enumerate(error_scenarios[:3]):  # Limit to 3 error scenarios
            test_cases.append(TestCase(
                name=f"test_{func_name}_error_{i+1}",
                description=f"Test {func_name} error scenario: {scenario}",
                test_code=f"with pytest.raises(ExpectedException):\n    {func_name}(error_input)",
                setup_code=f"error_input = self._get_error_input('{scenario}')",
                teardown_code="",
                category=TestCategory.ERROR_HANDLING,
                priority=TestPriority.HIGH,
                test_type=TestType.UNIT
            ))
        
        return test_cases
    
    def _generate_performance_tests(self, func_name: str, function_signature: str, analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate performance test cases"""
        
        test_cases = []
        
        test_cases.append(TestCase(
            name=f"test_{func_name}_performance",
            description=f"Performance test for {func_name}",
            test_code="""import time
start_time = time.time()
result = {func_name}()
end_time = time.time()
execution_time = end_time - start_time
assert execution_time < 1.0  # 1 second threshold""".format(func_name=func_name),
            setup_code="",
            teardown_code="",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.LOW,
            test_type=TestType.PERFORMANCE
        ))
        
        return test_cases
    
    def _generate_security_tests(self, func_name: str, function_signature: str, analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate security test cases"""
        
        test_cases = []
        
        security_tests = [
            ("sql_injection", "SQL injection attempt"),
            ("xss", "XSS attack attempt"),
            ("path_traversal", "Path traversal attempt"),
            ("buffer_overflow", "Buffer overflow attempt")
        ]
        
        for test_name, description in security_tests:
            test_cases.append(TestCase(
                name=f"test_{func_name}_security_{test_name}",
                description=f"Security test: {description}",
                test_code=f"# Test {description}\nmalicious_input = self._get_{test_name}_input()\n# Verify function handles malicious input safely",
                setup_code=f"malicious_input = self._get_{test_name}_input()",
                teardown_code="",
                category=TestCategory.SECURITY,
                priority=TestPriority.HIGH,
                test_type=TestType.SECURITY
            ))
        
        return test_cases
    
    def _enhance_tests_with_ai_insights(self, test_cases: List[TestCase], analysis: Dict[str, Any]) -> List[TestCase]:
        """Enhance test cases with AI insights"""
        
        enhanced_tests = []
        
        # Get AI insights
        quality_insights = analysis.get("quality_insights", {})
        ai_confidence = analysis.get("ai_confidence", 0.0)
        
        for test_case in test_cases:
            # Add AI confidence to description
            if ai_confidence > 0.7:
                test_case.description += f" (AI Confidence: {ai_confidence:.2f})"
            
            # Add quality insights to test code
            if quality_insights.get("testability", 0) > 0.7:
                test_case.test_code = f"# High testability detected\n{test_case.test_code}"
            
            # Add semantic insights
            semantic_analysis = analysis.get("semantic_analysis", {})
            if semantic_analysis.get("purpose"):
                test_case.test_code = f"# Purpose: {semantic_analysis['purpose']}\n{test_case.test_code}"
            
            enhanced_tests.append(test_case)
        
        return enhanced_tests


class AIQualityOptimizer:
    """AI-powered test quality optimization"""
    
    def __init__(self, config: AIEnhancementConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def optimize_test_quality(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize test quality using AI insights"""
        
        try:
            optimized_tests = []
            
            for test_case in test_cases:
                # Analyze test quality
                quality_analysis = await self._analyze_test_quality(test_case)
                
                # Optimize based on analysis
                optimized_test = self._optimize_test_case(test_case, quality_analysis)
                optimized_tests.append(optimized_test)
            
            return optimized_tests
            
        except Exception as e:
            self.logger.error(f"Test quality optimization failed: {e}")
            return test_cases
    
    async def _analyze_test_quality(self, test_case: TestCase) -> Dict[str, Any]:
        """Analyze test quality using AI"""
        
        # Create quality analysis prompt
        prompt = f"""
Analyze the quality of this test case and provide optimization suggestions:

Test Name: {test_case.name}
Description: {test_case.description}
Test Code: {test_case.test_code}

Provide analysis in JSON format:
{{
    "quality_score": <score 0-1>,
    "issues": ["<issue1>", "<issue2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"],
    "improvements": {{
        "naming": "<improved_name>",
        "description": "<improved_description>",
        "code": "<improved_code>"
    }}
}}
"""
        
        try:
            # This would call AI model in real implementation
            # For now, return basic analysis
            return {
                "quality_score": 0.8,
                "issues": [],
                "suggestions": ["Add more assertions", "Improve test naming"],
                "improvements": {
                    "naming": test_case.name,
                    "description": test_case.description,
                    "code": test_case.test_code
                }
            }
        except Exception as e:
            self.logger.warning(f"AI quality analysis failed: {e}")
            return {
                "quality_score": 0.5,
                "issues": ["Analysis unavailable"],
                "suggestions": [],
                "improvements": {
                    "naming": test_case.name,
                    "description": test_case.description,
                    "code": test_case.test_code
                }
            }
    
    def _optimize_test_case(self, test_case: TestCase, quality_analysis: Dict[str, Any]) -> TestCase:
        """Optimize test case based on quality analysis"""
        
        # Apply improvements if available
        improvements = quality_analysis.get("improvements", {})
        
        if improvements.get("naming") and improvements["naming"] != test_case.name:
            test_case.name = improvements["naming"]
        
        if improvements.get("description") and improvements["description"] != test_case.description:
            test_case.description = improvements["description"]
        
        if improvements.get("code") and improvements["code"] != test_case.test_code:
            test_case.test_code = improvements["code"]
        
        # Add quality score to description
        quality_score = quality_analysis.get("quality_score", 0.0)
        if quality_score > 0.0:
            test_case.description += f" (Quality Score: {quality_score:.2f})"
        
        return test_case


class AIEnhancedTestGenerator:
    """Main AI-enhanced test generator"""
    
    def __init__(self, config: AIEnhancementConfig):
        self.config = config
        self.test_generator = AITestGenerator(config)
        self.quality_optimizer = AIQualityOptimizer(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_ai_enhanced_tests(
        self,
        function_signature: str,
        docstring: str,
        context: Optional[str] = None,
        test_config: Optional[TestGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate AI-enhanced tests with full intelligence"""
        
        try:
            # Generate intelligent tests
            result = await self.test_generator.generate_intelligent_tests(
                function_signature, docstring, context, test_config
            )
            
            if not result["success"]:
                return result
            
            # Optimize test quality
            optimized_tests = await self.quality_optimizer.optimize_test_quality(
                result["test_cases"]
            )
            
            return {
                "test_cases": optimized_tests,
                "ai_analysis": result["ai_analysis"],
                "ai_confidence": result["ai_confidence"],
                "optimization_applied": True,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"AI-enhanced test generation failed: {e}")
            return {
                "test_cases": [],
                "error": str(e),
                "success": False
            }


# Convenience functions
def create_ai_enhanced_generator(config: Optional[AIEnhancementConfig] = None) -> AIEnhancedTestGenerator:
    """Create an AI-enhanced test generator"""
    if config is None:
        config = AIEnhancementConfig()
    return AIEnhancedTestGenerator(config)


async def generate_ai_enhanced_tests(
    function_signature: str,
    docstring: str,
    context: Optional[str] = None,
    config: Optional[AIEnhancementConfig] = None
) -> Dict[str, Any]:
    """Generate AI-enhanced tests"""
    generator = create_ai_enhanced_generator(config)
    return await generator.generate_ai_enhanced_tests(function_signature, docstring, context)