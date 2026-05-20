"""
Testing Module

Testing utilities and fixtures for generator testing.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import time

logger = logging.getLogger(__name__)


class GeneratorTester:
    """
    Testing utilities for generators.
    """
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
    
    def test_config_validation(
        self,
        config: Dict[str, Any],
        should_pass: bool = True
    ) -> Dict[str, Any]:
        """
        Test configuration validation.
        
        Args:
            config: Configuration to test
            should_pass: Whether validation should pass
            
        Returns:
            Test result dictionary
        """
        from .validators import validate_generator_config
        from .constants import SUPPORTED_FRAMEWORKS, SUPPORTED_MODEL_TYPES
        
        start_time = time.time()
        is_valid, error = validate_generator_config(
            config,
            SUPPORTED_FRAMEWORKS,
            SUPPORTED_MODEL_TYPES
        )
        elapsed = time.time() - start_time
        
        result = {
            "test": "config_validation",
            "config": config,
            "expected_pass": should_pass,
            "actual_pass": is_valid,
            "error": error,
            "passed": (is_valid == should_pass),
            "elapsed_time": elapsed
        }
        
        self.test_results.append(result)
        return result
    
    def test_generator_creation(
        self,
        framework: str,
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        should_succeed: bool = True
    ) -> Dict[str, Any]:
        """
        Test generator creation.
        
        Args:
            framework: Framework to test
            model_type: Model type to test
            config: Additional configuration
            should_succeed: Whether creation should succeed
            
        Returns:
            Test result dictionary
        """
        from .factory import get_factory
        
        start_time = time.time()
        factory = get_factory()
        
        if not factory.is_available:
            result = {
                "test": "generator_creation",
                "framework": framework,
                "model_type": model_type,
                "expected_success": should_succeed,
                "actual_success": False,
                "error": "Generator not available",
                "passed": not should_succeed,
                "elapsed_time": time.time() - start_time
            }
            self.test_results.append(result)
            return result
        
        try:
            generator = factory.create(
                framework=framework,
                model_type=model_type,
                config=config or {}
            )
            success = generator is not None
            error = None
        except Exception as e:
            success = False
            error = str(e)
        
        elapsed = time.time() - start_time
        
        result = {
            "test": "generator_creation",
            "framework": framework,
            "model_type": model_type,
            "expected_success": should_succeed,
            "actual_success": success,
            "error": error,
            "passed": (success == should_succeed),
            "elapsed_time": elapsed
        }
        
        self.test_results.append(result)
        return result
    
    def test_preset_loading(
        self,
        preset_name: str,
        should_exist: bool = True
    ) -> Dict[str, Any]:
        """
        Test preset loading.
        
        Args:
            preset_name: Name of preset to test
            should_exist: Whether preset should exist
            
        Returns:
            Test result dictionary
        """
        from .presets import get_preset, list_presets
        
        start_time = time.time()
        
        try:
            preset = get_preset(preset_name)
            exists = True
            error = None
        except ValueError as e:
            exists = False
            preset = None
            error = str(e)
        
        elapsed = time.time() - start_time
        
        result = {
            "test": "preset_loading",
            "preset_name": preset_name,
            "expected_exists": should_exist,
            "actual_exists": exists,
            "preset": preset,
            "error": error,
            "passed": (exists == should_exist),
            "elapsed_time": elapsed,
            "available_presets": list_presets()
        }
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all available tests.
        
        Returns:
            Summary of all test results
        """
        from .constants import SUPPORTED_FRAMEWORKS, SUPPORTED_MODEL_TYPES
        from .presets import list_presets
        
        logger.info("Running all generator tests...")
        
        # Test all frameworks
        for framework in SUPPORTED_FRAMEWORKS:
            self.test_generator_creation(framework)
        
        # Test all model types
        for model_type in SUPPORTED_MODEL_TYPES[:3]:  # Limit to first 3 for speed
            self.test_generator_creation("pytorch", model_type)
        
        # Test all presets
        for preset_name in list_presets():
            self.test_preset_loading(preset_name)
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "results": self.test_results
        }
    
    def reset(self) -> None:
        """Reset test results."""
        self.test_results.clear()


def create_tester() -> GeneratorTester:
    """Create a new generator tester."""
    return GeneratorTester()















