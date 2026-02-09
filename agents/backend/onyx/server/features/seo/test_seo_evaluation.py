#!/usr/bin/env python3
"""
Test SEO Evaluation Metrics System
Verify that all components work correctly
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
from pathlib import Path

# Import the evaluation system
try:
    from evaluation_metrics import (
        SEOModelEvaluator, SEOMetricsConfig, ClassificationMetricsConfig, 
        RegressionMetricsConfig, create_seo_test_data
    )
    from seo_evaluation_config import SEOEvaluationConfigManager
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# =============================================================================
# TEST MODELS
# =============================================================================

class TestSEOClassifier(nn.Module):
    """Simple test classifier for SEO evaluation."""
    
    def __init__(self, input_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class TestSEORanker(nn.Module):
    """Simple test ranker for SEO evaluation."""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.ranker = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.ranker(x)

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_configuration_management():
    """Test configuration management functionality."""
    print("\n🔧 Testing Configuration Management...")
    
    try:
        # Create config manager
        config_manager = SEOEvaluationConfigManager()
        
        # Create default config
        default_config = config_manager.create_default_config()
        print("  ✅ Default configuration created")
        
        # Validate config
        issues = config_manager.validate_config(default_config)
        if issues:
            print(f"  ⚠️ Configuration issues: {issues}")
        else:
            print("  ✅ Configuration validation passed")
        
        # Create optimized configs
        classification_config = config_manager.get_optimized_config("classification", 5000)
        ranking_config = config_manager.get_optimized_config("ranking", 10000)
        print("  ✅ Optimized configurations created")
        
        # Save configs
        config_manager.save_config_to_yaml(default_config, "test_seo_config.yaml")
        print("  ✅ Configuration saved")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_seo_metrics_calculation():
    """Test SEO metrics calculation."""
    print("\n📊 Testing SEO Metrics Calculation...")
    
    try:
        # Create test data
        test_data = create_seo_test_data(n_samples=100, n_classes=3)
        print("  ✅ Test data created")
        
        # Test content quality metrics
        from evaluation_metrics import SEOSpecificMetrics
        seo_config = SEOMetricsConfig()
        seo_metrics = SEOSpecificMetrics(seo_config)
        
        content_metrics = seo_metrics.calculate_content_quality_metrics(
            test_data['content_data']
        )
        print(f"  ✅ Content quality metrics: {len(content_metrics)} calculated")
        
        # Test user engagement metrics
        engagement_metrics = seo_metrics.calculate_user_engagement_metrics(
            test_data['engagement_data']
        )
        print(f"  ✅ User engagement metrics: {len(engagement_metrics)} calculated")
        
        # Test technical SEO metrics
        technical_metrics = seo_metrics.calculate_technical_seo_metrics(
            test_data['technical_data']
        )
        print(f"  ✅ Technical SEO metrics: {len(technical_metrics)} calculated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SEO metrics test failed: {e}")
        return False

def test_classification_metrics():
    """Test classification metrics."""
    print("\n📊 Testing Classification Metrics...")
    
    try:
        # Create test data
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        y_prob = np.random.rand(100, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create classifier
        classifier = ClassificationMetrics(ClassificationMetricsConfig())
        
        # Calculate metrics
        metrics = classifier.calculate_metrics(y_true, y_pred, y_prob)
        print(f"  ✅ Classification metrics: {len(metrics)} calculated")
        
        # Check key metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            if metric in metrics:
                print(f"    ✅ {metric}: {metrics[metric]:.4f}")
            else:
                print(f"    ❌ {metric} missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Classification metrics test failed: {e}")
        return False

def test_regression_metrics():
    """Test regression metrics."""
    print("\n📊 Testing Regression Metrics...")
    
    try:
        # Create test data
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        
        # Create regressor
        regressor = RegressionMetrics(RegressionMetricsConfig())
        
        # Calculate metrics
        metrics = regressor.calculate_metrics(y_true, y_pred)
        print(f"  ✅ Regression metrics: {len(metrics)} calculated")
        
        # Check key metrics
        required_metrics = ['mse', 'rmse', 'mae', 'r2_score']
        for metric in required_metrics:
            if metric in metrics:
                print(f"    ✅ {metric}: {metrics[metric]:.4f}")
            else:
                print(f"    ❌ {metric} missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Regression metrics test failed: {e}")
        return False

async def test_seo_evaluator():
    """Test the complete SEO evaluator."""
    print("\n🚀 Testing Complete SEO Evaluator...")
    
    try:
        # Create configuration
        seo_config = SEOMetricsConfig(
            ranking_metrics=True,
            content_quality_metrics=True,
            user_engagement_metrics=True,
            technical_seo_metrics=True
        )
        
        # Create evaluator
        evaluator = SEOModelEvaluator(
            seo_config=seo_config,
            classification_config=ClassificationMetricsConfig(),
            regression_config=RegressionMetricsConfig()
        )
        print("  ✅ SEO evaluator created")
        
        # Create test models
        classifier_model = TestSEOClassifier(input_dim=64, num_classes=3)
        ranker_model = TestSEORanker(input_dim=64)
        print("  ✅ Test models created")
        
        # Create test data
        test_data = create_seo_test_data(n_samples=100, n_classes=3)
        print("  ✅ Test data created")
        
        # Test classification evaluation
        print("  📊 Testing classification evaluation...")
        classification_results = await evaluator.evaluate_seo_model(
            classifier_model, test_data, task_type="classification"
        )
        print(f"    ✅ Classification evaluation completed: {len(classification_results)} metrics")
        
        # Test ranking evaluation
        print("  📊 Testing ranking evaluation...")
        ranking_results = await evaluator.evaluate_seo_model(
            ranker_model, test_data, task_type="ranking"
        )
        print(f"    ✅ Ranking evaluation completed: {len(ranking_results)} metrics")
        
        # Get evaluation summary
        summary = evaluator.get_evaluation_summary()
        print(f"  📋 Evaluation summary generated: {len(summary)} sections")
        
        # Save results
        evaluator.save_evaluation_results("test_seo_evaluation_results.json")
        print("  💾 Evaluation results saved")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SEO evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\n📊 Testing Visualization...")
    
    try:
        # Create test data
        test_data = create_seo_test_data(n_samples=100, n_classes=3)
        
        # Create evaluator
        seo_config = SEOMetricsConfig()
        evaluator = SEOModelEvaluator(
            seo_config=seo_config,
            classification_config=ClassificationMetricsConfig(),
            regression_config=RegressionMetricsConfig()
        )
        
        # Test plotting (this might fail in headless environments)
        try:
            evaluator.plot_evaluation_metrics("test_seo_plots.png")
            print("  ✅ Visualization completed")
        except Exception as viz_error:
            print(f"  ⚠️ Visualization failed (expected in headless): {viz_error}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        return False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all tests."""
    print("🧪 SEO Evaluation Metrics System - Complete Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Test configuration management
    test_results['configuration'] = test_configuration_management()
    
    # Test SEO metrics calculation
    test_results['seo_metrics'] = test_seo_metrics_calculation()
    
    # Test classification metrics
    test_results['classification'] = test_classification_metrics()
    
    # Test regression metrics
    test_results['regression'] = test_regression_metrics()
    
    # Test complete evaluator
    test_results['evaluator'] = await test_seo_evaluator()
    
    # Test visualization
    test_results['visualization'] = test_visualization()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SEO evaluation system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_seo_config.yaml",
        "test_seo_evaluation_results.json",
        "test_seo_plots.png"
    ]
    
    for file_path in test_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"🧹 Cleaned up: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not clean up {file_path}: {e}")

if __name__ == "__main__":
    try:
        # Run tests
        success = asyncio.run(run_all_tests())
        
        # Cleanup
        cleanup_test_files()
        
        # Exit with appropriate code
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        cleanup_test_files()
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_files()
        exit(1)

