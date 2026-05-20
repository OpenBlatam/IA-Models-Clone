#!/usr/bin/env python3
"""
Test Validation Script for Brand Voice AI System
===============================================

This script provides quick validation of the test suite setup and
basic functionality tests for the Brand Voice AI system.
"""

import asyncio
import logging
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Import test utilities
from test_utils import (
    create_mock_config, create_test_data, create_test_images,
    create_test_audio, setup_test_environment, teardown_test_environment
)

# Import test configuration
from test_config import TestConfig, get_quick_test_config

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestValidator:
    """Test validation utility"""
    
    def __init__(self):
        self.config = get_quick_test_config()
        self.temp_dirs = []
        self.validation_results = {}
    
    async def validate_test_setup(self) -> bool:
        """Validate test setup and dependencies"""
        logger.info("Validating test setup...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                logger.error("Python 3.8+ is required")
                return False
            logger.info(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check required modules
            required_modules = [
                'pytest', 'unittest', 'asyncio', 'numpy', 'pandas',
                'PIL', 'cv2', 'librosa', 'torch', 'transformers'
            ]
            
            missing_modules = []
            for module in required_modules:
                try:
                    __import__(module)
                    logger.info(f"✓ Module {module} is available")
                except ImportError:
                    missing_modules.append(module)
                    logger.warning(f"✗ Module {module} is missing")
            
            if missing_modules:
                logger.warning(f"Missing modules: {missing_modules}")
                logger.warning("Some tests may not run properly")
            
            # Check test directories
            self.temp_dirs = setup_test_environment(self.config)
            logger.info("✓ Test environment setup successful")
            
            # Validate test data generation
            await self._validate_test_data_generation()
            
            # Validate mock configurations
            await self._validate_mock_configurations()
            
            return True
            
        except Exception as e:
            logger.error(f"Test setup validation failed: {e}")
            return False
    
    async def _validate_test_data_generation(self):
        """Validate test data generation"""
        logger.info("Validating test data generation...")
        
        try:
            # Test text data generation
            text_data = create_test_data("text", 5)
            assert len(text_data) == 5
            assert all(isinstance(item, str) for item in text_data)
            logger.info("✓ Text data generation works")
            
            # Test brand data generation
            brand_data = create_test_data("brand_data", 2)
            assert len(brand_data) == 2
            assert all("brand_name" in item for item in brand_data)
            logger.info("✓ Brand data generation works")
            
            # Test image generation
            test_images = create_test_images(2)
            assert len(test_images) == 2
            assert all(os.path.exists(img) for img in test_images)
            logger.info("✓ Image generation works")
            
            # Test audio generation
            test_audio = create_test_audio(1.0)
            assert os.path.exists(test_audio)
            logger.info("✓ Audio generation works")
            
            # Cleanup test files
            for img in test_images:
                if os.path.exists(img):
                    os.remove(img)
            if os.path.exists(test_audio):
                os.remove(test_audio)
            
        except Exception as e:
            logger.error(f"Test data generation validation failed: {e}")
            raise
    
    async def _validate_mock_configurations(self):
        """Validate mock configurations"""
        logger.info("Validating mock configurations...")
        
        try:
            # Test mock config generation
            config = create_mock_config()
            assert isinstance(config, dict)
            assert "device" in config
            assert "batch_size" in config
            logger.info("✓ Mock configuration generation works")
            
            # Test different config types
            config_types = ["transformer", "training", "serving", "deployment", "monitoring"]
            for config_type in config_types:
                config = create_mock_config(config_type)
                assert isinstance(config, dict)
                assert "device" in config
            logger.info("✓ Multiple config types work")
            
        except Exception as e:
            logger.error(f"Mock configuration validation failed: {e}")
            raise
    
    async def validate_basic_functionality(self) -> bool:
        """Validate basic functionality of core modules"""
        logger.info("Validating basic functionality...")
        
        try:
            # Test configuration loading
            config = TestConfig()
            assert config.test_mode == True
            assert config.parallel_execution == False
            logger.info("✓ Test configuration works")
            
            # Test environment variables
            assert os.environ.get('BRAND_AI_TEST_MODE') == 'true'
            logger.info("✓ Environment variables set correctly")
            
            # Test file operations
            test_file = os.path.join(self.config.test_data_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            assert os.path.exists(test_file)
            os.remove(test_file)
            logger.info("✓ File operations work")
            
            return True
            
        except Exception as e:
            logger.error(f"Basic functionality validation failed: {e}")
            return False
    
    async def validate_test_structure(self) -> bool:
        """Validate test file structure"""
        logger.info("Validating test file structure...")
        
        try:
            # Check if test files exist
            test_files = [
                "test_brand_ai_comprehensive.py",
                "test_brand_ai_performance.py",
                "test_utils.py",
                "test_config.py",
                "run_tests.py",
                "requirements-test.txt"
            ]
            
            current_dir = Path(__file__).parent
            missing_files = []
            
            for test_file in test_files:
                file_path = current_dir / test_file
                if file_path.exists():
                    logger.info(f"✓ {test_file} exists")
                else:
                    missing_files.append(test_file)
                    logger.warning(f"✗ {test_file} is missing")
            
            if missing_files:
                logger.warning(f"Missing test files: {missing_files}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Test structure validation failed: {e}")
            return False
    
    async def run_quick_tests(self) -> bool:
        """Run a few quick tests to validate functionality"""
        logger.info("Running quick validation tests...")
        
        try:
            # Test 1: Configuration validation
            config = get_quick_test_config()
            assert config.test_mode == True
            assert config.test_data_size == "small"
            logger.info("✓ Quick test config validation passed")
            
            # Test 2: Test data generation
            test_data = create_test_data("text", 3)
            assert len(test_data) == 3
            logger.info("✓ Test data generation validation passed")
            
            # Test 3: Mock configuration
            mock_config = create_mock_config("transformer")
            assert "transformer_models" in mock_config
            logger.info("✓ Mock configuration validation passed")
            
            # Test 4: Environment setup
            temp_dirs = setup_test_environment(self.config)
            assert len(temp_dirs) >= 3
            assert all(os.path.exists(d) for d in temp_dirs)
            logger.info("✓ Environment setup validation passed")
            
            return True
            
        except Exception as e:
            logger.error(f"Quick tests validation failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up test environment"""
        try:
            teardown_test_environment(self.config)
            logger.info("✓ Test environment cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run full validation suite"""
        logger.info("Starting full test validation...")
        
        start_time = time.time()
        results = {
            'test_setup': False,
            'basic_functionality': False,
            'test_structure': False,
            'quick_tests': False,
            'validation_time': 0,
            'errors': []
        }
        
        try:
            # Validate test setup
            results['test_setup'] = await self.validate_test_setup()
            
            # Validate basic functionality
            results['basic_functionality'] = await self.validate_basic_functionality()
            
            # Validate test structure
            results['test_structure'] = await self.validate_test_structure()
            
            # Run quick tests
            results['quick_tests'] = await self.run_quick_tests()
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            results['errors'].append(str(e))
        
        finally:
            # Cleanup
            self.cleanup()
        
        end_time = time.time()
        results['validation_time'] = end_time - start_time
        
        return results

async def main():
    """Main validation function"""
    print("Brand Voice AI System - Test Validation")
    print("=" * 50)
    
    validator = TestValidator()
    
    try:
        results = await validator.run_full_validation()
        
        print("\nValidation Results:")
        print("-" * 30)
        print(f"Test Setup: {'✓ PASS' if results['test_setup'] else '✗ FAIL'}")
        print(f"Basic Functionality: {'✓ PASS' if results['basic_functionality'] else '✗ FAIL'}")
        print(f"Test Structure: {'✓ PASS' if results['test_structure'] else '✗ FAIL'}")
        print(f"Quick Tests: {'✓ PASS' if results['quick_tests'] else '✗ FAIL'}")
        print(f"Validation Time: {results['validation_time']:.2f} seconds")
        
        if results['errors']:
            print(f"\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Overall result
        all_passed = all([
            results['test_setup'],
            results['basic_functionality'],
            results['test_structure'],
            results['quick_tests']
        ])
        
        if all_passed:
            print("\n🎉 All validations passed! Test suite is ready to use.")
            return 0
        else:
            print("\n❌ Some validations failed. Please check the errors above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
