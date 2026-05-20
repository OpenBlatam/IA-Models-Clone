from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from core.evaluation_metrics import (
from models.sequence import EmailSequence, SequenceStep
from models.subscriber import Subscriber
from models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Evaluation Metrics Example

Demonstrates how to use the comprehensive evaluation metrics system
for email sequence models with various evaluation scenarios.
"""


# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

    MetricsConfig,
    ContentQualityMetrics,
    EngagementMetrics,
    BusinessImpactMetrics,
    TechnicalMetrics,
    EmailSequenceEvaluator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEmailModel(nn.Module):
    """Simple email sequence model for demonstration"""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 64, output_size: int = 1):
        
    """__init__ function."""
super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x) -> Any:
        return self.layers(x)


def create_sample_data():
    """Create sample data for evaluation"""
    
    # Create sample subscribers
    subscribers = [
        Subscriber(
            id="sub_1",
            email="john@example.com",
            name="John Doe",
            company="Tech Corp",
            interests=["technology", "AI", "machine learning"],
            industry="Technology"
        ),
        Subscriber(
            id="sub_2",
            email="jane@example.com",
            name="Jane Smith",
            company="Marketing Inc",
            interests=["marketing", "social media", "content creation"],
            industry="Marketing"
        ),
        Subscriber(
            id="sub_3",
            email="bob@example.com",
            name="Bob Johnson",
            company="Finance Ltd",
            interests=["finance", "investment", "business"],
            industry="Finance"
        )
    ]
    
    # Create sample templates
    templates = [
        EmailTemplate(
            id="template_1",
            name="Welcome Series",
            subject_template="Welcome to {company}!",
            content_template="Hi {name}, welcome to our platform. We're excited to help you succeed with {interest}."
        ),
        EmailTemplate(
            id="template_2",
            name="Feature Introduction",
            subject_template="Discover our {feature} feature",
            content_template="Hello {name}, we think you'll love our new {feature} feature. It's perfect for {industry} professionals."
        ),
        EmailTemplate(
            id="template_3",
            name="Conversion",
            subject_template="Ready to get started?",
            content_template="Hi {name}, based on your interest in {interest}, we think you're ready to take the next step. Click here to get started!"
        )
    ]
    
    # Create sample email sequence
    sequence = EmailSequence(
        id="seq_1",
        name="Onboarding Sequence",
        description="Welcome and onboarding sequence for new subscribers",
        steps=[
            SequenceStep(
                order=1,
                subscriber_id="sub_1",
                template_id="template_1",
                content="Hi John, welcome to our platform! We're excited to help you succeed with AI and machine learning. Our platform offers cutting-edge tools that will revolutionize your workflow.",
                delay_hours=0
            ),
            SequenceStep(
                order=2,
                subscriber_id="sub_1",
                template_id="template_2",
                content="Hello John, we think you'll love our new AI-powered analytics feature. It's perfect for technology professionals like you. Discover insights that were previously hidden in your data.",
                delay_hours=24
            ),
            SequenceStep(
                order=3,
                subscriber_id="sub_1",
                template_id="template_3",
                content="Hi John, based on your interest in machine learning, we think you're ready to take the next step. Click here to start your free trial and see the power of our AI platform in action!",
                delay_hours=48
            )
        ]
    )
    
    return subscribers, templates, sequence


async def demonstrate_content_quality_metrics():
    """Demonstrate content quality metrics"""
    
    logger.info("=== Content Quality Metrics Demo ===")
    
    config = MetricsConfig(
        enable_content_quality=True,
        enable_readability=True,
        enable_sentiment=True,
        enable_grammar=True
    )
    
    content_metrics = ContentQualityMetrics(config)
    
    # Sample content with different characteristics
    sample_contents = [
        "This is a simple, clear message that everyone can understand easily.",
        "The implementation of sophisticated machine learning algorithms necessitates comprehensive understanding of complex mathematical frameworks and advanced computational methodologies.",
        "Hi there! We're super excited to share this AMAZING opportunity with you!!! Don't miss out on this INCREDIBLE deal that will change your life forever!!!",
        "Welcome to our platform. We offer various features that can help improve your workflow and increase productivity."
    ]
    
    for i, content in enumerate(sample_contents, 1):
        logger.info(f"\nContent {i}: {content[:50]}...")
        metrics = content_metrics.evaluate_content_quality(content)
        
        logger.info(f"Content Quality Score: {metrics['content_quality_score']:.3f}")
        logger.info(f"Word Count: {metrics.get('word_count', 0)}")
        logger.info(f"Readability Score: {metrics.get('readability_score', 0):.3f}")
        logger.info(f"Sentiment Score: {metrics.get('sentiment_score', 0):.3f}")
        logger.info(f"Grammar Score: {metrics.get('grammar_score', 0):.3f}")


async def demonstrate_engagement_metrics():
    """Demonstrate engagement metrics"""
    
    logger.info("\n=== Engagement Metrics Demo ===")
    
    config = MetricsConfig(
        enable_engagement=True,
        enable_cta_analysis=True,
        enable_urgency_detection=True
    )
    
    engagement_metrics = EngagementMetrics(config)
    
    # Sample content with different engagement characteristics
    sample_content_pairs = [
        (
            "Learn more about our features",
            "Discover our amazing features that will transform your business. Click here to learn more and get started today!"
        ),
        (
            "Limited time offer",
            "Act now! This exclusive offer expires in 24 hours. Don't miss this incredible opportunity to save 50% on our premium package."
        ),
        (
            "Personalized recommendation",
            "Hi John, based on your interest in AI, we think you'll love our new machine learning course. It's designed specifically for technology professionals like you."
        )
    ]
    
    for i, (subject, content) in enumerate(sample_content_pairs, 1):
        logger.info(f"\nEmail {i}:")
        logger.info(f"Subject: {subject}")
        logger.info(f"Content: {content[:50]}...")
        
        metrics = engagement_metrics.evaluate_engagement(content, subject)
        
        logger.info(f"Engagement Score: {metrics['engagement_score']:.3f}")
        logger.info(f"CTA Effectiveness: {metrics.get('cta_effectiveness', 0):.3f}")
        logger.info(f"Urgency Score: {metrics.get('urgency_score', 0):.3f}")
        logger.info(f"Keyword Diversity: {metrics.get('keyword_diversity', 0):.3f}")


async def demonstrate_business_impact_metrics():
    """Demonstrate business impact metrics"""
    
    logger.info("\n=== Business Impact Metrics Demo ===")
    
    config = MetricsConfig(
        enable_business_impact=True,
        enable_conversion=True,
        enable_revenue=True
    )
    
    business_metrics = BusinessImpactMetrics(config)
    
    # Sample content with different business characteristics
    sample_contents = [
        "Learn about our new features and how they can improve your workflow.",
        "Buy now and save 50% on our premium package. Limited time offer!",
        "Our customers have seen a 300% increase in productivity. Join them today!",
        "Download our free guide to learn more about industry best practices."
    ]
    
    for i, content in enumerate(sample_contents, 1):
        logger.info(f"\nContent {i}: {content}")
        
        metrics = business_metrics.evaluate_business_impact(content)
        
        logger.info(f"Business Impact Score: {metrics['business_impact_score']:.3f}")
        logger.info(f"Conversion Potential: {metrics.get('conversion_potential', 0):.3f}")
        logger.info(f"Revenue Potential: {metrics.get('revenue_potential', 0):.3f}")
        logger.info(f"ROI Score: {metrics.get('roi_score', 0):.3f}")


async def demonstrate_technical_metrics():
    """Demonstrate technical metrics"""
    
    logger.info("\n=== Technical Metrics Demo ===")
    
    config = MetricsConfig(enable_technical=True)
    technical_metrics = TechnicalMetrics(config)
    
    # Create sample model
    model = SimpleEmailModel()
    
    # Generate sample predictions and targets
    batch_size = 100
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    
    # Add some correlation to make it more realistic
    targets = 0.7 * predictions + 0.3 * torch.randn(batch_size, 1)
    
    logger.info("Evaluating regression metrics...")
    regression_metrics = technical_metrics.evaluate_technical_metrics(
        predictions, targets, model
    )
    
    logger.info(f"MSE: {regression_metrics['mse']:.4f}")
    logger.info(f"RMSE: {regression_metrics['rmse']:.4f}")
    logger.info(f"MAE: {regression_metrics['mae']:.4f}")
    logger.info(f"R² Score: {regression_metrics['r2_score']:.4f}")
    logger.info(f"Model Size: {regression_metrics['model_size_mb']:.2f} MB")
    logger.info(f"Total Parameters: {regression_metrics['total_parameters']:,}")
    
    # Test classification metrics
    logger.info("\nEvaluating classification metrics...")
    classification_predictions = torch.softmax(torch.randn(batch_size, 3), dim=1)
    classification_targets = torch.randint(0, 3, (batch_size,))
    classification_targets_onehot = torch.zeros(batch_size, 3)
    classification_targets_onehot.scatter_(1, classification_targets.unsqueeze(1), 1)
    
    classification_metrics = technical_metrics.evaluate_technical_metrics(
        classification_predictions, classification_targets_onehot, model
    )
    
    logger.info(f"Accuracy: {classification_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {classification_metrics['precision']:.4f}")
    logger.info(f"Recall: {classification_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {classification_metrics['f1_score']:.4f}")


async def demonstrate_complete_evaluation():
    """Demonstrate complete email sequence evaluation"""
    
    logger.info("\n=== Complete Email Sequence Evaluation Demo ===")
    
    # Create configuration
    config = MetricsConfig(
        content_weight=0.3,
        engagement_weight=0.3,
        personalization_weight=0.2,
        business_weight=0.2
    )
    
    # Create evaluator
    evaluator = EmailSequenceEvaluator(config)
    
    # Create sample data
    subscribers, templates, sequence = create_sample_data()
    
    # Create sample predictions and targets for demonstration
    num_steps = len(sequence.steps)
    predictions = torch.randn(num_steps, 1)
    targets = torch.randn(num_steps, 1)
    
    # Create simple model
    model = SimpleEmailModel()
    
    logger.info(f"Evaluating sequence: {sequence.name}")
    logger.info(f"Number of steps: {len(sequence.steps)}")
    
    # Perform evaluation
    evaluation_results = await evaluator.evaluate_sequence(
        sequence=sequence,
        subscribers=subscribers,
        templates=templates,
        predictions=predictions,
        targets=targets,
        model=model
    )
    
    # Display results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Sequence ID: {evaluation_results['sequence_id']}")
    logger.info(f"Sequence Name: {evaluation_results['sequence_name']}")
    
    overall_metrics = evaluation_results['overall_metrics']
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Content Quality Score: {overall_metrics['content_quality_score']:.3f}")
    logger.info(f"  Engagement Score: {overall_metrics['engagement_score']:.3f}")
    logger.info(f"  Business Impact Score: {overall_metrics['business_impact_score']:.3f}")
    logger.info(f"  Sequence Coherence: {overall_metrics['sequence_coherence']:.3f}")
    logger.info(f"  Progression Effectiveness: {overall_metrics['progression_effectiveness']:.3f}")
    logger.info(f"  Overall Score: {overall_metrics['overall_score']:.3f}")
    
    # Display step-by-step results
    logger.info(f"\nStep-by-Step Results:")
    for i, step_eval in enumerate(evaluation_results['step_evaluations'], 1):
        logger.info(f"\nStep {i}:")
        if 'content_metrics' in step_eval:
            content_score = step_eval['content_metrics'].get('content_quality_score', 0)
            logger.info(f"  Content Quality: {content_score:.3f}")
        
        if 'engagement_metrics' in step_eval:
            engagement_score = step_eval['engagement_metrics'].get('engagement_score', 0)
            logger.info(f"  Engagement: {engagement_score:.3f}")
        
        if 'business_metrics' in step_eval:
            business_score = step_eval['business_metrics'].get('business_impact_score', 0)
            logger.info(f"  Business Impact: {business_score:.3f}")
    
    # Display technical metrics
    if 'technical_metrics' in evaluation_results:
        tech_metrics = evaluation_results['technical_metrics']
        logger.info(f"\nTechnical Metrics:")
        logger.info(f"  MSE: {tech_metrics.get('mse', 0):.4f}")
        logger.info(f"  R² Score: {tech_metrics.get('r2_score', 0):.4f}")
        logger.info(f"  Model Size: {tech_metrics.get('model_size_mb', 0):.2f} MB")
    
    # Generate evaluation report
    report = evaluator.get_evaluation_report()
    logger.info(f"\n=== Evaluation Report ===")
    logger.info(f"Total Evaluations: {report['total_evaluations']}")
    logger.info(f"Average Overall Score: {report['average_overall_score']:.3f}")
    logger.info(f"Score Distribution:")
    logger.info(f"  Min: {report['score_distribution']['min']:.3f}")
    logger.info(f"  Max: {report['score_distribution']['max']:.3f}")
    logger.info(f"  Mean: {report['score_distribution']['std']:.3f}")
    logger.info(f"  Median: {report['score_distribution']['median']:.3f}")
    
    # Save results to file
    output_file = Path("evaluation_results.json")
    with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return evaluation_results


async def demonstrate_custom_metrics():
    """Demonstrate custom metrics configuration"""
    
    logger.info("\n=== Custom Metrics Configuration Demo ===")
    
    # Create custom configuration focusing on specific aspects
    custom_config = MetricsConfig(
        # Focus on content quality
        enable_content_quality=True,
        enable_readability=True,
        enable_sentiment=True,
        enable_grammar=True,
        
        # Disable engagement metrics
        enable_engagement=False,
        enable_cta_analysis=False,
        enable_urgency_detection=False,
        
        # Focus on business impact
        enable_business_impact=True,
        enable_conversion=True,
        enable_revenue=True,
        
        # Custom weights
        content_weight=0.5,
        engagement_weight=0.0,
        personalization_weight=0.2,
        business_weight=0.3,
        
        # Custom thresholds
        min_content_length=100,
        max_content_length=1500,
        min_readability_score=40.0,
        max_readability_score=70.0
    )
    
    evaluator = EmailSequenceEvaluator(custom_config)
    
    # Create sample data
    subscribers, templates, sequence = create_sample_data()
    
    logger.info("Evaluating with custom configuration...")
    evaluation_results = await evaluator.evaluate_sequence(
        sequence=sequence,
        subscribers=subscribers,
        templates=templates
    )
    
    overall_metrics = evaluation_results['overall_metrics']
    logger.info(f"Custom Overall Score: {overall_metrics['overall_score']:.3f}")
    logger.info(f"Content Quality Score: {overall_metrics['content_quality_score']:.3f}")
    logger.info(f"Business Impact Score: {overall_metrics['business_impact_score']:.3f}")


async def main():
    """Run all demonstration functions"""
    
    logger.info("Starting Email Sequence Evaluation Metrics Demo")
    logger.info("=" * 60)
    
    try:
        # Run individual metric demonstrations
        await demonstrate_content_quality_metrics()
        await demonstrate_engagement_metrics()
        await demonstrate_business_impact_metrics()
        await demonstrate_technical_metrics()
        
        # Run complete evaluation
        await demonstrate_complete_evaluation()
        
        # Run custom metrics demonstration
        await demonstrate_custom_metrics()
        
        logger.info("\n" + "=" * 60)
        logger.info("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


match __name__:
    case "__main__":
    asyncio.run(main()) 