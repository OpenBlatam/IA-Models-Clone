"""
Example usage of Blaze AI evaluation metrics.

This file demonstrates how to use the comprehensive evaluation metrics
for different AI tasks including classification, text generation, image generation,
SEO optimization, and brand voice analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

# Import evaluation functions
from .metrics_registry import get_evaluation_registry, evaluate_model
from .text_generation import evaluate_text_generation_batch
from .image_generation import evaluate_image_generation_batch
from .seo_optimization import evaluate_seo_optimization
from .brand_voice import evaluate_brand_voice


def example_classification_evaluation():
    """Example of evaluating a classification model."""
    print("=== Classification Model Evaluation ===")
    
    # Create a simple classification model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size=10, num_classes=3):
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleClassifier()
    
    # Create dummy data
    batch_size = 32
    input_size = 10
    num_classes = 3
    
    # Generate random input data
    inputs = torch.randn(batch_size, input_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create data loader
    dataset = TensorDataset(inputs, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Evaluate using the registry
    registry = get_evaluation_registry()
    result = registry.evaluate(
        task_type="classification",
        model=model,
        data_loader=data_loader,
        device="cpu",
        model_name="simple_classifier"
    )
    
    print(f"Model: {result.model_name}")
    print(f"Task: {result.task_type}")
    print(f"Metrics: {result.metrics}")
    print(f"Timestamp: {result.timestamp}")
    print()


def example_text_generation_evaluation():
    """Example of evaluating text generation models."""
    print("=== Text Generation Model Evaluation ===")
    
    # Sample generated and reference texts
    generated_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data."
    ]
    
    reference_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models need substantial training data."
    ]
    
    # Evaluate text generation quality
    metrics = evaluate_text_generation_batch(
        generated_texts=generated_texts,
        reference_texts=reference_texts
    )
    
    print("Text Generation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()
    
    # You can also evaluate with a model if you have one
    # result = evaluate_model(
    #     task_type="text_generation",
    #     model=your_model,
    #     references=reference_texts,
    #     candidates=generated_texts,
    #     model_name="text_generation_model"
    # )


def example_image_generation_evaluation():
    """Example of evaluating image generation models."""
    print("=== Image Generation Model Evaluation ===")
    
    # Create dummy image data (batch_size, channels, height, width)
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    
    # Generate random images (normalized to [0, 1])
    real_images = torch.rand(batch_size, channels, height, width)
    generated_images = torch.rand(batch_size, channels, height, width)
    
    # Sample text prompts
    text_prompts = [
        "A beautiful sunset over mountains",
        "A cute cat playing with yarn",
        "A modern city skyline at night",
        "A peaceful forest scene"
    ]
    
    # Evaluate image generation quality
    metrics = evaluate_image_generation_batch(
        real_images=real_images,
        generated_images=generated_images,
        text_prompts=text_prompts,
        device="cpu"  # Use CPU for dummy data
    )
    
    print("Image Generation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()
    
    # Note: For real evaluation, you would use actual images and GPU
    # result = evaluate_model(
    #     task_type="image_generation",
    #     model=your_diffusion_model,
    #     real_images=real_images,
    #     generated_images=generated_images,
    #     text_prompts=text_prompts,
    #     model_name="diffusion_model"
    # )


def example_seo_optimization_evaluation():
    """Example of evaluating SEO optimization."""
    print("=== SEO Optimization Evaluation ===")
    
    # Sample content to evaluate
    content = """
    # Artificial Intelligence in Modern Business
    
    Artificial intelligence (AI) has revolutionized the way businesses operate in the digital age. 
    From automated customer service to predictive analytics, AI technologies are transforming 
    industries across the globe.
    
    ## Key Benefits of AI
    
    - **Increased Efficiency**: AI automates repetitive tasks, saving time and resources
    - **Better Decision Making**: Data-driven insights improve business strategies
    - **Enhanced Customer Experience**: Personalized interactions through AI-powered systems
    
    ## Implementation Strategies
    
    Companies looking to implement AI should start with a clear strategy and identify 
    specific use cases that align with their business objectives. It's essential to 
    invest in proper training and ensure data quality for optimal results.
    
    The future of business lies in embracing artificial intelligence and leveraging 
    its capabilities to drive innovation and growth.
    """
    
    # Target keywords for SEO
    target_keywords = ["artificial intelligence", "AI", "business", "digital transformation"]
    
    # Evaluate SEO optimization
    evaluation = evaluate_seo_optimization(
        text=content,
        target_keywords=target_keywords
    )
    
    print("SEO Evaluation Results:")
    print(f"Overall Score: {evaluation['quality_score']['overall_score']}/100")
    print(f"Grade: {evaluation['quality_score']['grade']}")
    print()
    
    print("Component Scores:")
    print(f"  Readability: {evaluation['quality_score']['readability_score']}/25")
    print(f"  Content Length: {evaluation['quality_score']['length_score']}/20")
    print(f"  Keyword Optimization: {evaluation['quality_score']['keyword_score']}/25")
    print(f"  Content Structure: {evaluation['quality_score']['structure_score']}/20")
    print(f"  Technical SEO: {evaluation['quality_score']['technical_score']}/10")
    print()
    
    print("Keyword Density:")
    for keyword, density in evaluation['keyword_density'].items():
        print(f"  {keyword}: {density}%")
    print()
    
    print("Recommendations:")
    for rec in evaluation['recommendations']:
        print(f"  - {rec}")
    print()


def example_brand_voice_evaluation():
    """Example of evaluating brand voice consistency."""
    print("=== Brand Voice Evaluation ===")
    
    # Sample brand guidelines
    brand_guidelines = {
        "tone": {
            "formality": 0.7,      # 0-1 scale, 0.7 = moderately formal
            "friendliness": 0.6,   # 0-1 scale, 0.6 = moderately friendly
            "professionalism": 0.8  # 0-1 scale, 0.8 = high professionalism
        },
        "vocabulary": {
            "brand_terms": ["innovation", "quality", "excellence", "trust"],
            "preferred_terms": {
                "customer": ["client", "user"],
                "product": ["solution", "service"]
            },
            "avoided_terms": ["cheap", "basic", "simple"]
        },
        "content_structure": {
            "paragraph_length": 150,
            "heading_structure": "hierarchical"
        }
    }
    
    # Sample content pieces to evaluate
    content_pieces = [
        """
        Our innovative solutions deliver exceptional quality and build lasting trust with clients. 
        We believe in excellence in everything we do, providing cutting-edge technology that 
        transforms businesses and drives success.
        """,
        
        """
        Innovation is at the heart of our approach. We maintain the highest standards of quality 
        while ensuring our clients receive the best possible service. Trust and excellence are 
        our core values, reflected in every solution we deliver.
        """,
        
        """
        We specialize in innovative technology solutions that meet the highest quality standards. 
        Our commitment to excellence and trust has made us a leader in the industry, serving 
        clients with cutting-edge products and services.
        """
    ]
    
    # Evaluate brand voice consistency
    evaluation = evaluate_brand_voice(
        texts=content_pieces,
        brand_guidelines=brand_guidelines
    )
    
    print("Brand Voice Evaluation Results:")
    print(f"Overall Brand Alignment: {evaluation['brand_alignment']['overall_brand_alignment']:.3f}")
    print(f"Grade: {evaluation['brand_alignment']['brand_alignment_grade']}")
    print()
    
    print("Component Scores:")
    print(f"  Tone Consistency: {evaluation['tone_consistency']['overall_tone_consistency']:.3f}")
    print(f"  Vocabulary Consistency: {evaluation['vocabulary_consistency']['brand_term_consistency']:.3f}")
    print(f"  Sentiment Consistency: {evaluation['sentiment_consistency']['sentiment_consistency']:.3f}")
    print()
    
    print("Detailed Tone Analysis:")
    print(f"  Formality: {evaluation['tone_consistency']['avg_formality']:.3f}")
    print(f"  Friendliness: {evaluation['tone_consistency']['avg_friendliness']:.3f}")
    print(f"  Professionalism: {evaluation['tone_consistency']['avg_professionalism']:.3f}")
    print()
    
    print("Vocabulary Analysis:")
    print(f"  Brand Term Density: {evaluation['vocabulary_consistency']['brand_term_density']:.4f}")
    print(f"  Vocabulary Diversity: {evaluation['vocabulary_consistency']['vocabulary_diversity']:.3f}")
    print()
    
    print("Recommendations:")
    for rec in evaluation['recommendations']:
        print(f"  - {rec}")
    print()


def example_comprehensive_evaluation():
    """Example of comprehensive evaluation using the registry."""
    print("=== Comprehensive Evaluation Using Registry ===")
    
    # Get the evaluation registry
    registry = get_evaluation_registry(storage_dir="evaluation_results")
    
    # Example 1: SEO Evaluation
    seo_content = "This is a sample content about artificial intelligence and machine learning."
    seo_keywords = ["artificial intelligence", "machine learning"]
    
    seo_result = registry.evaluate(
        task_type="seo_optimization",
        model=None,  # Not needed for SEO evaluation
        text=seo_content,
        target_keywords=seo_keywords,
        model_name="seo_analysis"
    )
    
    print(f"SEO Evaluation - Score: {seo_result.metrics['overall_score']}/100")
    print(f"Grade: {seo_result.metrics['grade']}")
    print()
    
    # Example 2: Brand Voice Evaluation
    brand_texts = [
        "Our innovative solutions deliver quality and trust.",
        "We provide excellent service with innovative technology."
    ]
    
    brand_guidelines = {
        "tone": {"formality": 0.7, "friendliness": 0.6, "professionalism": 0.8},
        "vocabulary": {"brand_terms": ["innovation", "quality", "trust"]}
    }
    
    brand_result = registry.evaluate(
        task_type="brand_voice",
        model=None,  # Not needed for brand voice evaluation
        texts=brand_texts,
        brand_guidelines=brand_guidelines,
        model_name="brand_voice_analysis"
    )
    
    print(f"Brand Voice Evaluation - Alignment: {brand_result.metrics['overall_brand_alignment']:.3f}")
    print(f"Grade: {brand_result.metrics['brand_alignment_grade']}")
    print()
    
    # Get statistics
    stats = registry.get_statistics()
    print("Registry Statistics:")
    print(f"Total Results: {stats['total_results']}")
    print(f"Task Distribution: {stats['task_type_distribution']}")
    print(f"Model Distribution: {stats['model_distribution']}")
    print()
    
    # Export results
    registry.export_results("comprehensive_evaluation_export.json")
    print("Results exported to comprehensive_evaluation_export.json")


def main():
    """Run all evaluation examples."""
    print("Blaze AI Evaluation Metrics Examples")
    print("=" * 50)
    print()
    
    try:
        example_classification_evaluation()
        example_text_generation_evaluation()
        example_image_generation_evaluation()
        example_seo_optimization_evaluation()
        example_brand_voice_evaluation()
        example_comprehensive_evaluation()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
