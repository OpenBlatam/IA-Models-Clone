#!/usr/bin/env python3
"""
🧪 IMPROVED CONTENT MODULES SYSTEM - TEST SUITE
===============================================

Comprehensive test suite for the improved content modules system.
Tests all features, error handling, and performance monitoring.
"""

import unittest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import the improved system
from __init__ import (
    ContentModuleManager, ModuleRegistry, ModuleInfo, ModuleStatus, ModuleCategory,
    get_content_manager, list_all_modules, find_module, get_category_modules,
    search_modules, get_featured_modules, get_statistics, get_usage_examples
)

class TestImprovedContentModulesSystem(unittest.TestCase):
    """Test suite for the improved content modules system."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = get_content_manager()
        self.registry = self.manager.registry
    
    def test_module_registry_initialization(self):
        """Test that the module registry initializes correctly."""
        self.assertIsNotNone(self.registry)
        self.assertIsInstance(self.registry, ModuleRegistry)
        
        # Check that modules are registered
        all_modules = self.registry.get_all_modules()
        self.assertGreater(len(all_modules), 0)
        
        # Check that categories are initialized
        for category in ModuleCategory:
            modules = self.registry.get_modules_by_category(category)
            self.assertIsInstance(modules, dict)
    
    def test_module_info_structure(self):
        """Test that ModuleInfo dataclass works correctly."""
        module_info = ModuleInfo(
            name="test_module",
            path="../test_module",
            description="Test module for testing",
            status=ModuleStatus.AVAILABLE,
            category=ModuleCategory.TECHNICAL,
            features=["test_feature"],
            dependencies=["test_dependency"],
            performance_score=8.5
        )
        
        self.assertEqual(module_info.name, "test_module")
        self.assertEqual(module_info.status, ModuleStatus.AVAILABLE)
        self.assertEqual(module_info.category, ModuleCategory.TECHNICAL)
        self.assertEqual(module_info.performance_score, 8.5)
        
        # Test to_dict method
        module_dict = module_info.to_dict()
        self.assertIsInstance(module_dict, dict)
        self.assertEqual(module_dict['name'], "test_module")
        self.assertEqual(module_dict['status'], "available")
    
    def test_enum_values(self):
        """Test that enums have correct values."""
        # Test ModuleStatus
        self.assertEqual(ModuleStatus.AVAILABLE.value, "available")
        self.assertEqual(ModuleStatus.UNAVAILABLE.value, "unavailable")
        self.assertEqual(ModuleStatus.DEPRECATED.value, "deprecated")
        self.assertEqual(ModuleStatus.BETA.value, "beta")
        self.assertEqual(ModuleStatus.EXPERIMENTAL.value, "experimental")
        
        # Test ModuleCategory
        self.assertEqual(ModuleCategory.SOCIAL_MEDIA.value, "social_media")
        self.assertEqual(ModuleCategory.EDITORIAL.value, "editorial")
        self.assertEqual(ModuleCategory.MARKETING.value, "marketing")
        self.assertEqual(ModuleCategory.ECOMMERCE.value, "ecommerce")
        self.assertEqual(ModuleCategory.MULTIMEDIA.value, "multimedia")
        self.assertEqual(ModuleCategory.TECHNICAL.value, "technical")
        self.assertEqual(ModuleCategory.ENTERPRISE.value, "enterprise")
        self.assertEqual(ModuleCategory.AI_MODELS.value, "ai_models")
    
    def test_list_all_modules(self):
        """Test list_all_modules function."""
        modules = list_all_modules()
        
        self.assertIsInstance(modules, dict)
        self.assertGreater(len(modules), 0)
        
        # Check that all categories are present
        expected_categories = [cat.value for cat in ModuleCategory]
        for category in expected_categories:
            self.assertIn(category, modules)
    
    def test_find_module(self):
        """Test find_module function."""
        # Test finding an existing module
        module_info = find_module('product_descriptions')
        self.assertIsNotNone(module_info)
        self.assertIn('category', module_info)
        self.assertIn('module_info', module_info)
        self.assertEqual(module_info['module_info']['name'], 'product_descriptions')
        
        # Test finding a non-existent module
        non_existent = find_module('non_existent_module')
        self.assertIsNone(non_existent)
    
    def test_get_category_modules(self):
        """Test get_category_modules function."""
        # Test valid category
        social_modules = get_category_modules('social_media')
        self.assertIsInstance(social_modules, dict)
        self.assertGreater(len(social_modules), 0)
        
        # Test invalid category
        invalid_modules = get_category_modules('invalid_category')
        self.assertEqual(invalid_modules, {})
    
    def test_search_modules(self):
        """Test search_modules function."""
        # Test search for 'ai'
        ai_results = search_modules('ai')
        self.assertIsInstance(ai_results, list)
        
        # Test search for 'optimization'
        opt_results = search_modules('optimization')
        self.assertIsInstance(opt_results, list)
        
        # Test search with no results
        no_results = search_modules('xyz123nonexistent')
        self.assertEqual(no_results, [])
    
    def test_get_featured_modules(self):
        """Test get_featured_modules function."""
        featured = get_featured_modules()
        
        self.assertIsInstance(featured, dict)
        self.assertIn('ai_powered', featured)
        self.assertIn('enterprise', featured)
        self.assertIn('social_media', featured)
        
        # Check that each category has modules
        for category, modules in featured.items():
            self.assertIsInstance(modules, dict)
    
    def test_get_statistics(self):
        """Test get_statistics function."""
        stats = get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_modules', stats)
        self.assertIn('categories', stats)
        self.assertIn('average_performance_score', stats)
        self.assertIn('top_performing_modules', stats)
        self.assertIn('status_distribution', stats)
        self.assertIn('most_used_features', stats)
        
        # Check data types
        self.assertIsInstance(stats['total_modules'], int)
        self.assertIsInstance(stats['categories'], dict)
        self.assertIsInstance(stats['average_performance_score'], (int, float))
        self.assertIsInstance(stats['top_performing_modules'], list)
    
    def test_manager_methods(self):
        """Test ContentModuleManager methods."""
        # Test get_all_modules
        all_modules = self.manager.get_all_modules()
        self.assertIsInstance(all_modules, dict)
        self.assertGreater(len(all_modules), 0)
        
        # Test get_module_by_name
        module_info = self.manager.get_module_by_name('product_descriptions')
        self.assertIsNotNone(module_info)
        self.assertIn('category', module_info)
        self.assertIn('module_info', module_info)
        
        # Test get_modules_by_category
        social_modules = self.manager.get_modules_by_category('social_media')
        self.assertIsInstance(social_modules, dict)
        
        # Test search_modules
        search_results = self.manager.search_modules('ai')
        self.assertIsInstance(search_results, list)
        
        # Test get_featured_modules
        featured = self.manager.get_featured_modules()
        self.assertIsInstance(featured, dict)
        
        # Test get_statistics
        stats = self.manager.get_statistics()
        self.assertIsInstance(stats, dict)
    
    @unittest.skip("Async test requires proper async environment")
    async def test_async_performance_monitoring(self):
        """Test async performance monitoring."""
        # Test get_module_performance
        performance = await self.manager.get_module_performance('product_descriptions')
        self.assertIsInstance(performance, dict)
        
        if 'error' not in performance:
            self.assertIn('module_name', performance)
            self.assertIn('performance_score', performance)
            self.assertIn('usage_count', performance)
            self.assertIn('historical_metrics', performance)
            self.assertIn('average_performance', performance)
    
    def test_registry_search_functionality(self):
        """Test registry search functionality."""
        # Test search by name
        results = self.registry.search_modules('product')
        self.assertIsInstance(results, list)
        
        # Test search by description
        results = self.registry.search_modules('generation')
        self.assertIsInstance(results, list)
        
        # Test search by features
        results = self.registry.search_modules('ai_generation')
        self.assertIsInstance(results, list)
    
    def test_registry_statistics(self):
        """Test registry statistics generation."""
        stats = self.registry.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_modules', stats)
        self.assertIn('categories', stats)
        self.assertIn('status_distribution', stats)
        self.assertIn('average_performance_score', stats)
        self.assertIn('top_performing_modules', stats)
        self.assertIn('most_used_features', stats)
        
        # Check that statistics are reasonable
        self.assertGreater(stats['total_modules'], 0)
        self.assertGreater(stats['average_performance_score'], 0)
        self.assertLessEqual(stats['average_performance_score'], 10)
    
    def test_top_performing_modules(self):
        """Test top performing modules functionality."""
        top_modules = self.registry.get_top_performing_modules(5)
        
        self.assertIsInstance(top_modules, list)
        self.assertLessEqual(len(top_modules), 5)
        
        # Check that modules are sorted by performance score
        if len(top_modules) > 1:
            for i in range(len(top_modules) - 1):
                self.assertGreaterEqual(
                    top_modules[i].performance_score,
                    top_modules[i + 1].performance_score
                )
    
    def test_usage_examples(self):
        """Test usage examples function."""
        examples = get_usage_examples()
        
        self.assertIsInstance(examples, dict)
        self.assertIn('basic_usage', examples)
        self.assertIn('advanced_usage', examples)
        self.assertIn('manager_usage', examples)
        self.assertIn('error_handling', examples)
        
        # Check that examples are strings
        for category, example in examples.items():
            self.assertIsInstance(example, str)
            self.assertGreater(len(example), 0)
    
    def test_error_handling(self):
        """Test error handling functionality."""
        # Test invalid module name
        invalid_module = self.manager.get_module_by_name('invalid_module_name')
        self.assertIsNone(invalid_module)
        
        # Test invalid category
        invalid_category = self.manager.get_modules_by_category('invalid_category')
        self.assertEqual(invalid_category, {})
        
        # Test search with no results
        no_results = self.manager.search_modules('xyz123nonexistent')
        self.assertEqual(no_results, [])
    
    def test_module_categories(self):
        """Test that all expected modules are in their correct categories."""
        expected_modules = {
            'social_media': ['instagram_captions', 'facebook_posts', 'linkedin_posts'],
            'editorial': ['blog_posts', 'copywriting'],
            'marketing': ['ads', 'key_messages', 'email_sequence'],
            'ecommerce': ['product_descriptions'],
            'multimedia': ['ai_video', 'image_process'],
            'technical': ['seo'],
            'enterprise': ['enterprise', 'ultra_extreme_v18'],
            'ai_models': ['advanced_ai_models']
        }
        
        for category, expected_module_names in expected_modules.items():
            modules = get_category_modules(category)
            actual_module_names = list(modules.keys())
            
            for expected_name in expected_module_names:
                self.assertIn(expected_name, actual_module_names)
    
    def test_performance_scores(self):
        """Test that performance scores are within valid range."""
        all_modules = self.registry.get_all_modules()
        
        for module in all_modules.values():
            self.assertGreaterEqual(module.performance_score, 0.0)
            self.assertLessEqual(module.performance_score, 10.0)
    
    def test_module_features(self):
        """Test that modules have appropriate features."""
        # Test that product_descriptions has AI features
        product_module = self.registry.get_module('product_descriptions')
        self.assertIsNotNone(product_module)
        self.assertIn('seo_optimization', product_module.features)
        self.assertIn('conversion_focused', product_module.features)
        
        # Test that enterprise module has enterprise features
        enterprise_module = self.registry.get_module('enterprise')
        self.assertIsNotNone(enterprise_module)
        self.assertIn('scalability', enterprise_module.features)
        self.assertIn('security', enterprise_module.features)

class TestImprovedSystemIntegration(unittest.TestCase):
    """Integration tests for the improved system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.manager = get_content_manager()
    
    def test_full_workflow(self):
        """Test a complete workflow using the improved system."""
        # 1. List all modules
        all_modules = list_all_modules()
        self.assertGreater(len(all_modules), 0)
        
        # 2. Find a specific module
        module_info = find_module('product_descriptions')
        self.assertIsNotNone(module_info)
        
        # 3. Get modules by category
        ecommerce_modules = get_category_modules('ecommerce')
        self.assertIn('product_descriptions', ecommerce_modules)
        
        # 4. Search for modules
        search_results = search_modules('ai')
        self.assertIsInstance(search_results, list)
        
        # 5. Get featured modules
        featured = get_featured_modules()
        self.assertIsInstance(featured, dict)
        
        # 6. Get statistics
        stats = get_statistics()
        self.assertIsInstance(stats, dict)
        
        # 7. Use manager methods
        manager_modules = self.manager.get_all_modules()
        self.assertIsInstance(manager_modules, dict)
    
    def test_error_recovery(self):
        """Test that the system recovers gracefully from errors."""
        # Test with invalid inputs
        invalid_results = [
            find_module(''),
            find_module(None),
            get_category_modules(''),
            get_category_modules(None),
            search_modules(''),
            search_modules(None)
        ]
        
        # All should handle gracefully without raising exceptions
        for result in invalid_results:
            self.assertIsNotNone(result)  # Should return empty dict/list, not None

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestImprovedContentModulesSystem))
    test_suite.addTest(unittest.makeSuite(TestImprovedSystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("🧪 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print(f"\n✅ ALL TESTS PASSED!")
        print("🎉 The improved content modules system is working correctly!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)





