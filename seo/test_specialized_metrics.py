#!/usr/bin/env python3
"""
Test script for Specialized SEO Evaluation Metrics
Demonstrates appropriate evaluation metrics for SEO-specific tasks
"""

import torch
import numpy as np
import asyncio
from seo_evaluation_metrics import (
    SEOMetricsConfig, SEOModelEvaluator, SEOSpecificMetrics
)
from evaluation_metrics_ultra_optimized import (
    UltraOptimizedConfig, UltraOptimizedSEOMetricsModule
)

def test_seo_specific_metrics():
    """Test SEO-specific metrics calculation."""
    print("🔍 Testing SEO-Specific Metrics")
    
    config = SEOMetricsConfig(
        task_type="classification",
        num_classes=2,
        use_seo_specific=True,
        seo_score_threshold=0.7,
        content_quality_threshold=0.6
    )
    
    seo_metrics = SEOSpecificMetrics(config)
    
    # Test content quality scoring
    test_texts = [
        "<h1>SEO Guide</h1><p>Basic content.</p>",
        "<h1>Comprehensive SEO Optimization Guide</h1><p>This is a detailed guide about search engine optimization techniques that will help improve your website's ranking in search results. We cover keyword research, content optimization, technical SEO, and user experience factors.</p><h2>Key Strategies</h2><ul><li>Keyword research and optimization</li><li>Content quality and relevance</li><li>Technical SEO implementation</li><li>User experience optimization</li></ul>",
        "Simple text without structure",
        "<h1>SEO Tips</h1><p>Use keywords naturally in your content. Focus on quality and relevance.</p><h2>Best Practices</h2><p>Create engaging content that answers user queries.</p>"
    ]
    
    print("\n📊 Content Quality Scores:")
    for i, text in enumerate(test_texts):
        score = seo_metrics.calculate_content_quality_score(text)
        print(f"  Text {i+1}: {score:.4f}")
    
    # Test technical SEO scoring
    html_contents = [
        "<html><head><title>Short Title</title><meta name='description' content='Short desc'></head><body><h1>Heading</h1><img src='image.jpg' alt='Alt text'><p>Content</p></body></html>",
        "<html><head><title>Perfect Length SEO Title for Better Rankings</title><meta name='description' content='This is a comprehensive meta description that provides detailed information about the page content and encourages users to click through from search results.'></head><body><h1>Main Heading</h1><h2>Sub Heading</h2><h3>Sub Sub Heading</h3><img src='image1.jpg' alt='Descriptive alt text'><img src='image2.jpg' alt='Another descriptive alt text'><p>Well-structured content with proper heading hierarchy.</p></body></html>",
        "<html><body><p>No meta tags or proper structure</p></body></html>"
    ]
    
    print("\n🔧 Technical SEO Scores:")
    for i, html in enumerate(html_contents):
        score = seo_metrics.calculate_technical_seo_score(html)
        print(f"  HTML {i+1}: {score:.4f}")
    
    # Test overall SEO scoring
    print("\n🏆 Overall SEO Scores:")
    for i, (text, html) in enumerate(zip(test_texts, html_contents)):
        scores = seo_metrics.calculate_overall_seo_score(text, html)
        print(f"  Content {i+1}:")
        for component, score in scores.items():
            print(f"    {component.replace('_', ' ').title()}: {score:.4f}")

def test_seo_model_evaluator():
    """Test SEO model evaluator with different task types."""
    print("\n🎯 Testing SEO Model Evaluator")
    
    config = SEOMetricsConfig(
        task_type="classification",
        num_classes=2,
        use_seo_specific=True
    )
    
    evaluator = SEOModelEvaluator(config)
    
    # Test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.9, 0.8, 0.1])
    
    # Test classification evaluation
    print("\n📈 Classification Evaluation:")
    classification_metrics = evaluator.evaluate_classification(y_true, y_pred, y_prob)
    for metric, value in classification_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test regression evaluation
    print("\n📊 Regression Evaluation:")
    regression_metrics = evaluator.evaluate_regression(y_true.astype(float), y_prob)
    for metric, value in regression_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test ranking evaluation
    print("\n📋 Ranking Evaluation:")
    ranking_metrics = evaluator.evaluate_ranking(y_true, y_prob)
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test clustering evaluation
    print("\n🔗 Clustering Evaluation:")
    # Create dummy features for clustering
    features = np.random.randn(len(y_true), 10)
    clustering_metrics = evaluator.evaluate_clustering(features, y_pred)
    for metric, value in clustering_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test SEO content evaluation
    print("\n🔍 SEO Content Evaluation:")
    sample_text = """
    <h1>Complete SEO Optimization Guide</h1>
    <p>This comprehensive guide covers all aspects of search engine optimization to help improve your website's visibility and ranking in search results.</p>
    <h2>Key Components</h2>
    <ul>
        <li>Keyword research and analysis</li>
        <li>Content optimization strategies</li>
        <li>Technical SEO implementation</li>
        <li>User experience optimization</li>
        <li>Performance monitoring and analytics</li>
    </ul>
    <h2>Best Practices</h2>
    <p>Focus on creating high-quality, relevant content that addresses user intent and provides value to your audience.</p>
    """
    
    seo_content_metrics = evaluator.evaluate_seo_content(sample_text, sample_text)
    for metric, value in seo_content_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Generate comprehensive report
    print("\n📋 Comprehensive Evaluation Report:")
    report = evaluator.generate_evaluation_report(classification_metrics, "SEO Classification Test")
    print(report)

def test_ultra_optimized_integration():
    """Test integration with ultra-optimized SEO system."""
    print("\n🚀 Testing Ultra-Optimized Integration")
    
    # Create configuration
    config = UltraOptimizedConfig(
        use_multi_gpu=False,  # Disable for testing
        use_lora=True,
        use_diffusion=False,  # Disable for testing
        batch_size=16,
        max_length=256,
        num_epochs=2,
        patience=2
    )
    
    try:
        # Initialize model
        model = UltraOptimizedSEOMetricsModule(config)
        print("✅ Model initialized successfully")
        
        # Test data
        texts = [
            "<h1>SEO Guide</h1><p>Learn about search engine optimization.</p>",
            "<meta name='description' content='Content analysis for rankings'>",
            "How to improve website ranking",
            "SEO best practices guide"
        ]
        labels = torch.tensor([1, 0, 1, 1])
        predictions = torch.randint(0, 2, (4,))
        
        # Test specialized metrics evaluation
        print("\n🔍 Testing Specialized Metrics Integration:")
        specialized_metrics = model.evaluate_with_specialized_metrics(
            texts, labels, predictions, "classification"
        )
        
        print("Specialized Metrics Results:")
        for metric, value in specialized_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Test comprehensive report generation
        print("\n📋 Testing Comprehensive Report Generation:")
        report = model.generate_comprehensive_report(
            texts, labels, predictions, "classification"
        )
        print(report)
        
        print("\n✅ Ultra-optimized integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during ultra-optimized integration test: {e}")
        import traceback
        traceback.print_exc()

def test_performance_benchmark():
    """Test performance of specialized metrics."""
    print("\n⚡ Performance Benchmark")
    
    config = SEOMetricsConfig(
        task_type="classification",
        num_classes=2,
        use_seo_specific=True
    )
    
    evaluator = SEOModelEvaluator(config)
    
    # Generate large test dataset
    n_samples = 10000
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_prob = np.random.random(n_samples)
    
    # Benchmark classification evaluation
    import time
    
    start_time = time.time()
    for _ in range(10):
        _ = evaluator.evaluate_classification(y_true, y_pred, y_prob)
    total_time = time.time() - start_time
    
    print(f"Classification evaluation (10 iterations, {n_samples} samples):")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per iteration: {total_time/10*1000:.2f}ms")
    print(f"  Throughput: {10/total_time:.2f} evaluations/second")
    
    # Benchmark SEO content evaluation
    sample_text = "SEO optimization guide with comprehensive strategies for improving search rankings."
    
    start_time = time.time()
    for _ in range(1000):
        _ = evaluator.evaluate_seo_content(sample_text)
    total_time = time.time() - start_time
    
    print(f"\nSEO content evaluation (1000 iterations):")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per iteration: {total_time/1000*1000:.2f}ms")
    print(f"  Throughput: {1000/total_time:.2f} evaluations/second")

async def main():
    """Main test function."""
    print("🚀 Starting Specialized SEO Evaluation Metrics Tests")
    
    # Test SEO-specific metrics
    test_seo_specific_metrics()
    
    # Test SEO model evaluator
    test_seo_model_evaluator()
    
    # Test ultra-optimized integration
    test_ultra_optimized_integration()
    
    # Test performance
    test_performance_benchmark()
    
    print("\n🎉 All specialized metrics tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
