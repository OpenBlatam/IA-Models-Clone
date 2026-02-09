#!/usr/bin/env python3
"""
Test script for Evaluation Metrics in the Advanced LLM SEO Engine
Tests comprehensive evaluation metrics for classification, regression, ranking, and content quality tasks
"""

import torch
import sys
import os
import time
import numpy as np
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_llm_seo_engine import (
    SEOConfig, AdvancedLLMSEOEngine
)

def test_classification_evaluation():
    """Test SEO classification evaluation metrics."""
    print("Testing SEO Classification Evaluation...")
    
    # Create engine instance
    config = SEOConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=2,
        device="cpu"
    )
    
    engine = AdvancedLLMSEOEngine(config)
    
    # Test data
    y_true = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    
    # Test different task types
    task_types = ["seo_optimization", "content_quality", "keyword_relevance"]
    
    for task_type in task_types:
        print(f"\nTesting {task_type}...")
        metrics = engine.evaluate_seo_classification(y_true, y_pred, task_type)
        
        # Verify basic metrics
        assert 'accuracy' in metrics
        assert 'precision_weighted' in metrics
        assert 'recall_weighted' in metrics
        assert 'f1_weighted' in metrics
        assert 'cohen_kappa' in metrics
        
        # Verify task-specific metrics
        if task_type == "seo_optimization":
            assert 'seo_precision' in metrics
            assert 'seo_recall' in metrics
            assert 'seo_f1' in metrics
            assert 'seo_optimization_accuracy' in metrics
        
        print(f"✓ {task_type} metrics: {metrics}")
    
    print("✓ Classification evaluation tests passed!")

def test_regression_evaluation():
    """Test SEO regression evaluation metrics."""
    print("\nTesting SEO Regression Evaluation...")
    
    config = SEOConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=2,
        device="cpu"
    )
    
    engine = AdvancedLLMSEOEngine(config)
    
    # Test data
    y_true = [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 0.8]
    y_pred = [0.75, 0.65, 0.85, 0.75, 0.55, 0.8, 0.7, 0.85, 0.75, 0.8]
    
    # Test different task types
    task_types = ["seo_score", "readability_score", "content_quality_score"]
    
    for task_type in task_types:
        print(f"\nTesting {task_type}...")
        metrics = engine.evaluate_seo_regression(y_true, y_pred, task_type)
        
        # Verify basic metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        assert 'max_error' in metrics
        
        # Verify task-specific metrics
        if task_type == "seo_score":
            assert 'seo_accuracy_within_threshold' in metrics
            assert 'high_quality_detection_accuracy' in metrics
            assert 'seo_score_correlation' in metrics
        
        print(f"✓ {task_type} metrics: {metrics}")
    
    print("✓ Regression evaluation tests passed!")

def test_ranking_evaluation():
    """Test SEO ranking evaluation metrics."""
    print("\nTesting SEO Ranking Evaluation...")
    
    config = SEOConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=2,
        device="cpu"
    )
    
    engine = AdvancedLLMSEOEngine(config)
    
    # Test data
    y_true = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    y_pred = [0.85, 0.75, 0.8, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    
    # Test different task types
    task_types = ["content_ranking", "keyword_ranking", "seo_ranking"]
    
    for task_type in task_types:
        print(f"\nTesting {task_type}...")
        metrics = engine.evaluate_seo_ranking(y_true, y_pred, task_type=task_type)
        
        # Verify basic metrics
        assert 'ndcg_5' in metrics
        assert 'ndcg_10' in metrics
        assert 'ndcg_20' in metrics
        
        # Verify task-specific metrics
        if task_type == "content_ranking":
            assert 'top_1_content_relevance' in metrics
            assert 'top_5_content_precision' in metrics
        
        print(f"✓ {task_type} metrics: {metrics}")
    
    print("✓ Ranking evaluation tests passed!")

def test_content_quality_evaluation():
    """Test SEO content quality evaluation metrics."""
    print("\nTesting SEO Content Quality Evaluation...")
    
    config = SEOConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=2,
        device="cpu"
    )
    
    engine = AdvancedLLMSEOEngine(config)
    
    # Test data
    texts = [
        "This is a sample SEO content with good structure and readability.",
        "Another example of content that demonstrates proper keyword usage and formatting.",
        "A third piece of content to test the evaluation system thoroughly."
    ]
    
    html_contents = [
        "<h1>Title</h1><p>Content</p>",
        "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>",
        "<h1>Title</h1><p>Content</p><ul><li>List item</li></ul>"
    ]
    
    # Test content quality evaluation
    metrics = engine.evaluate_seo_content_quality(texts, html_contents)
    
    # Verify metrics structure
    assert 'readability_scores' in metrics
    assert 'keyword_density_scores' in metrics
    assert 'technical_seo_scores' in metrics
    assert 'content_structure_scores' in metrics
    assert 'engagement_potential_scores' in metrics
    
    # Verify aggregate metrics
    assert 'avg_readability' in metrics
    assert 'avg_keyword_density' in metrics
    assert 'avg_technical_seo' in metrics
    assert 'avg_content_structure' in metrics
    assert 'avg_engagement_potential' in metrics
    assert 'content_consistency' in metrics
    assert 'quality_distribution' in metrics
    
    print(f"✓ Content quality metrics: {metrics}")
    print("✓ Content quality evaluation tests passed!")

def test_evaluation_summaries():
    """Test evaluation summary generation."""
    print("\nTesting Evaluation Summary Generation...")
    
    config = SEOConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=2,
        device="cpu"
    )
    
    engine = AdvancedLLMSEOEngine(config)
    
    # Test classification summary
    classification_metrics = {
        'accuracy': 0.8,
        'precision_weighted': 0.75,
        'recall_weighted': 0.8,
        'f1_weighted': 0.77,
        'cohen_kappa': 0.6,
        'seo_f1': 0.85
    }
    
    summary = engine.get_evaluation_summary("classification", classification_metrics)
    assert "Classification Performance Summary" in summary
    assert "Accuracy: 0.800" in summary
    assert "SEO F1 Score: 0.850" in summary
    
    # Test regression summary
    regression_metrics = {
        'r2': 0.85,
        'rmse': 0.12,
        'mae': 0.08,
        'mape': 0.15,
        'seo_accuracy_within_threshold': 0.9
    }
    
    summary = engine.get_evaluation_summary("regression", regression_metrics)
    assert "Regression Performance Summary" in summary
    assert "R² Score: 0.850" in summary
    assert "SEO Accuracy within Threshold: 0.900" in summary
    
    # Test ranking summary
    ranking_metrics = {
        'ndcg_5': 0.85,
        'ndcg_10': 0.78,
        'ndcg_20': 0.72,
        'top_5_content_relevance': 0.88
    }
    
    summary = engine.get_evaluation_summary("ranking", ranking_metrics)
    assert "Ranking Performance Summary" in summary
    assert "NDCG@5: 0.850" in summary
    assert "Top 5 Content Relevance: 0.880" in summary
    
    # Test content quality summary
    content_quality_metrics = {
        'avg_readability': 0.75,
        'avg_keyword_density': 0.65,
        'avg_technical_seo': 0.82,
        'avg_content_structure': 0.78,
        'content_consistency': 0.85,
        'quality_distribution': {
            'high_quality': 2,
            'medium_quality': 1,
            'low_quality': 0
        }
    }
    
    summary = engine.get_evaluation_summary("content_quality", content_quality_metrics)
    assert "Content Quality Summary" in summary
    assert "Average Readability: 0.750" in summary
    assert "High(2), Medium(1), Low(0)" in summary
    
    print("✓ Evaluation summary tests passed!")

def test_integration_with_engine():
    """Test integration of evaluation metrics with the main engine."""
    print("\nTesting Integration with Main Engine...")
    
    config = SEOConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=2,
        device="cpu"
    )
    
    engine = AdvancedLLMSEOEngine(config)
    
    # Test that all evaluation methods are available
    assert hasattr(engine, 'evaluate_seo_classification')
    assert hasattr(engine, 'evaluate_seo_regression')
    assert hasattr(engine, 'evaluate_seo_ranking')
    assert hasattr(engine, 'evaluate_seo_content_quality')
    assert hasattr(engine, 'get_evaluation_summary')
    
    # Test that helper methods are available
    assert hasattr(engine, '_calculate_seo_optimization_metrics')
    assert hasattr(engine, '_calculate_content_quality_classification_metrics')
    assert hasattr(engine, '_calculate_keyword_relevance_metrics')
    assert hasattr(engine, '_calculate_seo_score_regression_metrics')
    assert hasattr(engine, '_calculate_readability_regression_metrics')
    assert hasattr(engine, '_calculate_content_quality_regression_metrics')
    assert hasattr(engine, '_calculate_content_ranking_metrics')
    assert hasattr(engine, '_calculate_keyword_ranking_metrics')
    assert hasattr(engine, '_calculate_seo_ranking_metrics')
    assert hasattr(engine, '_calculate_single_content_metrics')
    assert hasattr(engine, '_calculate_content_structure_score')
    assert hasattr(engine, '_calculate_engagement_potential')
    
    print("✓ Integration tests passed!")

def main():
    """Run all evaluation metrics tests."""
    print("🚀 Starting Evaluation Metrics Tests...")
    print("=" * 50)
    
    try:
        test_classification_evaluation()
        test_regression_evaluation()
        test_ranking_evaluation()
        test_content_quality_evaluation()
        test_evaluation_summaries()
        test_integration_with_engine()
        
        print("\n" + "=" * 50)
        print("✅ All Evaluation Metrics Tests Passed!")
        print("🎯 The SEO engine now has comprehensive evaluation metrics for:")
        print("   - Classification tasks (SEO optimization, content quality, keyword relevance)")
        print("   - Regression tasks (SEO scores, readability, content quality)")
        print("   - Ranking tasks (content ranking, keyword ranking, SEO ranking)")
        print("   - Content quality evaluation (readability, structure, engagement)")
        print("   - Human-readable evaluation summaries")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())






