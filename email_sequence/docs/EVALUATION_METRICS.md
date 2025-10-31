# Email Sequence Evaluation Metrics

A comprehensive evaluation system for email sequence models that provides task-specific metrics for content quality, engagement prediction, personalization effectiveness, and business impact.

## Overview

The evaluation metrics system is designed to assess email sequences from multiple perspectives:

- **Content Quality**: Readability, sentiment, grammar, and style analysis
- **Engagement**: CTA effectiveness, urgency detection, and engagement keywords
- **Business Impact**: Conversion potential, revenue indicators, and ROI analysis
- **Technical Performance**: Model accuracy, performance metrics, and efficiency

## Features

### 🎯 Task-Specific Metrics

- **Content Quality Metrics**: Flesch Reading Ease, sentiment analysis, grammar checking
- **Engagement Metrics**: CTA analysis, urgency detection, personalization effectiveness
- **Business Impact Metrics**: Conversion funnel analysis, revenue potential, ROI indicators
- **Technical Metrics**: Regression and classification metrics, model performance analysis

### 📊 Comprehensive Evaluation

- **Multi-dimensional scoring**: Combines multiple metric types into composite scores
- **Sequence coherence**: Analyzes consistency across email sequence steps
- **Progression effectiveness**: Evaluates how engagement builds through the sequence
- **Customizable weights**: Adjust importance of different metric categories

### 🔧 Flexible Configuration

- **Modular design**: Enable/disable specific metric categories
- **Customizable thresholds**: Set minimum/maximum acceptable values
- **Weighted scoring**: Configure importance of different metric types
- **Extensible architecture**: Easy to add new metric types

## Quick Start

### Installation

```bash
pip install -r requirements/evaluation_requirements.txt
```

### Basic Usage

```python
from core.evaluation_metrics import MetricsConfig, EmailSequenceEvaluator
from models.sequence import EmailSequence

# Create configuration
config = MetricsConfig(
    content_weight=0.3,
    engagement_weight=0.3,
    personalization_weight=0.2,
    business_weight=0.2
)

# Create evaluator
evaluator = EmailSequenceEvaluator(config)

# Evaluate email sequence
results = await evaluator.evaluate_sequence(
    sequence=email_sequence,
    subscribers=subscribers,
    templates=templates
)

# Get overall score
overall_score = results['overall_metrics']['overall_score']
print(f"Sequence Score: {overall_score:.3f}")
```

## Metric Categories

### Content Quality Metrics

Evaluates the quality and effectiveness of email content.

#### Features:
- **Readability Analysis**: Flesch Reading Ease, Gunning Fog, SMOG Index
- **Sentiment Analysis**: VADER and TextBlob sentiment scoring
- **Grammar & Style**: Basic grammar checks and style analysis
- **Content Length**: Optimal length scoring based on audience

#### Example:
```python
from core.evaluation_metrics import ContentQualityMetrics, MetricsConfig

config = MetricsConfig(enable_readability=True, enable_sentiment=True)
content_metrics = ContentQualityMetrics(config)

metrics = content_metrics.evaluate_content_quality(
    content="Your email content here...",
    target_audience=subscriber
)

print(f"Content Quality Score: {metrics['content_quality_score']:.3f}")
print(f"Readability Score: {metrics['readability_score']:.3f}")
print(f"Sentiment Score: {metrics['sentiment_score']:.3f}")
```

### Engagement Metrics

Analyzes engagement potential and call-to-action effectiveness.

#### Features:
- **CTA Analysis**: Call-to-action placement and effectiveness
- **Urgency Detection**: Time-sensitive language and urgency indicators
- **Engagement Keywords**: Analysis of engagement-driving vocabulary
- **Personalization**: Audience relevance and customization level

#### Example:
```python
from core.evaluation_metrics import EngagementMetrics

engagement_metrics = EngagementMetrics(config)

metrics = engagement_metrics.evaluate_engagement(
    content="Your email content...",
    subject_line="Your subject line...",
    target_audience=subscriber
)

print(f"Engagement Score: {metrics['engagement_score']:.3f}")
print(f"CTA Effectiveness: {metrics['cta_effectiveness']:.3f}")
print(f"Urgency Score: {metrics['urgency_score']:.3f}")
```

### Business Impact Metrics

Assesses potential business outcomes and conversion effectiveness.

#### Features:
- **Conversion Analysis**: Conversion funnel stage analysis
- **Revenue Potential**: Pricing mentions and value proposition
- **ROI Indicators**: Return on investment and cost-benefit analysis
- **Funnel Completeness**: Coverage of awareness-to-action stages

#### Example:
```python
from core.evaluation_metrics import BusinessImpactMetrics

business_metrics = BusinessImpactMetrics(config)

metrics = business_metrics.evaluate_business_impact(
    content="Your email content...",
    historical_data=historical_performance_data
)

print(f"Business Impact Score: {metrics['business_impact_score']:.3f}")
print(f"Conversion Potential: {metrics['conversion_potential']:.3f}")
print(f"Revenue Potential: {metrics['revenue_potential']:.3f}")
```

### Technical Metrics

Evaluates model performance and technical efficiency.

#### Features:
- **Regression Metrics**: MSE, RMSE, MAE, R² Score
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Model Performance**: Parameter count, model size, efficiency
- **Advanced Metrics**: ROC-AUC, Cohen's Kappa, Matthews Correlation

#### Example:
```python
from core.evaluation_metrics import TechnicalMetrics

technical_metrics = TechnicalMetrics(config)

metrics = technical_metrics.evaluate_technical_metrics(
    predictions=model_predictions,
    targets=ground_truth,
    model=your_model
)

print(f"MSE: {metrics['mse']:.4f}")
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
```

## Configuration Options

### MetricsConfig

```python
@dataclass
class MetricsConfig:
    # Content quality metrics
    enable_content_quality: bool = True
    enable_readability: bool = True
    enable_sentiment: bool = True
    enable_grammar: bool = True
    
    # Engagement metrics
    enable_engagement: bool = True
    enable_cta_analysis: bool = True
    enable_urgency_detection: bool = True
    
    # Business metrics
    enable_business_impact: bool = True
    enable_conversion: bool = True
    enable_revenue: bool = True
    
    # Technical metrics
    enable_technical: bool = True
    enable_performance: bool = True
    
    # Thresholds
    min_content_length: int = 50
    max_content_length: int = 2000
    min_readability_score: float = 30.0
    max_readability_score: float = 80.0
    
    # Weights for composite scores
    content_weight: float = 0.3
    engagement_weight: float = 0.3
    personalization_weight: float = 0.2
    business_weight: float = 0.2
```

### Custom Configuration Example

```python
# Focus on content quality and business impact
custom_config = MetricsConfig(
    enable_engagement=False,  # Disable engagement metrics
    content_weight=0.5,       # Increase content importance
    business_weight=0.5,      # Increase business importance
    min_content_length=100,   # Require longer content
    max_content_length=1500   # Limit content length
)
```

## Advanced Usage

### Complete Sequence Evaluation

```python
async def evaluate_complete_sequence():
    # Create evaluator with custom configuration
    config = MetricsConfig(
        content_weight=0.3,
        engagement_weight=0.3,
        personalization_weight=0.2,
        business_weight=0.2
    )
    
    evaluator = EmailSequenceEvaluator(config)
    
    # Evaluate sequence with all components
    results = await evaluator.evaluate_sequence(
        sequence=email_sequence,
        subscribers=subscribers,
        templates=templates,
        predictions=model_predictions,
        targets=ground_truth,
        model=your_model
    )
    
    # Access comprehensive results
    overall_metrics = results['overall_metrics']
    step_evaluations = results['step_evaluations']
    technical_metrics = results.get('technical_metrics', {})
    
    print(f"Overall Score: {overall_metrics['overall_score']:.3f}")
    print(f"Content Quality: {overall_metrics['content_quality_score']:.3f}")
    print(f"Engagement: {overall_metrics['engagement_score']:.3f}")
    print(f"Business Impact: {overall_metrics['business_impact_score']:.3f}")
    print(f"Sequence Coherence: {overall_metrics['sequence_coherence']:.3f}")
    
    return results
```

### Evaluation Reports

```python
# Generate comprehensive evaluation report
report = evaluator.get_evaluation_report()

print(f"Total Evaluations: {report['total_evaluations']}")
print(f"Average Score: {report['average_overall_score']:.3f}")
print(f"Score Range: {report['score_distribution']['min']:.3f} - {report['score_distribution']['max']:.3f}")

# Plot evaluation results
evaluator.plot_evaluation_results(save_path="evaluation_results.png")
```

### Custom Metrics Integration

```python
class CustomMetrics:
    """Example of custom metrics integration"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
    
    def evaluate_custom_metrics(self, content: str) -> Dict[str, float]:
        """Evaluate custom metrics"""
        # Implement your custom metrics here
        custom_score = self._calculate_custom_score(content)
        
        return {
            "custom_score": custom_score,
            "custom_metric_1": value_1,
            "custom_metric_2": value_2
        }
    
    def _calculate_custom_score(self, content: str) -> float:
        """Calculate custom score"""
        # Your custom scoring logic
        return score

# Integrate with main evaluator
class ExtendedEmailSequenceEvaluator(EmailSequenceEvaluator):
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
        self.custom_metrics = CustomMetrics(config)
    
    async def evaluate_sequence(self, *args, **kwargs):
        results = await super().evaluate_sequence(*args, **kwargs)
        
        # Add custom metrics
        custom_results = self.custom_metrics.evaluate_custom_metrics(content)
        results['custom_metrics'] = custom_results
        
        return results
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Process multiple sequences together for better efficiency
2. **Caching**: Cache expensive computations like sentiment analysis
3. **Async Processing**: Use async evaluation for I/O-bound operations
4. **Memory Management**: Monitor memory usage for large datasets

### Memory Usage

```python
# Monitor memory usage during evaluation
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Use in evaluation
monitor_memory()
results = await evaluator.evaluate_sequence(...)
monitor_memory()
```

## Testing

### Unit Tests

```python
import pytest
from core.evaluation_metrics import ContentQualityMetrics, MetricsConfig

def test_content_quality_metrics():
    config = MetricsConfig()
    metrics = ContentQualityMetrics(config)
    
    # Test with simple content
    content = "This is a simple test message."
    results = metrics.evaluate_content_quality(content)
    
    assert 'content_quality_score' in results
    assert 0.0 <= results['content_quality_score'] <= 1.0
    assert results['word_count'] == 7

def test_empty_content():
    config = MetricsConfig()
    metrics = ContentQualityMetrics(config)
    
    results = metrics.evaluate_content_quality("")
    assert results['content_quality_score'] == 0.0
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_complete_evaluation():
    config = MetricsConfig()
    evaluator = EmailSequenceEvaluator(config)
    
    # Create test data
    sequence = create_test_sequence()
    subscribers = create_test_subscribers()
    templates = create_test_templates()
    
    # Perform evaluation
    results = await evaluator.evaluate_sequence(
        sequence=sequence,
        subscribers=subscribers,
        templates=templates
    )
    
    # Verify results structure
    assert 'overall_metrics' in results
    assert 'step_evaluations' in results
    assert 'overall_score' in results['overall_metrics']
```

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Download required NLTK data
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

2. **Memory Issues**: Reduce batch size or use memory-efficient processing
3. **Performance Issues**: Enable caching and use async processing
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create evaluator with debug info
config = MetricsConfig()
evaluator = EmailSequenceEvaluator(config)

# Evaluation will now show detailed debug information
results = await evaluator.evaluate_sequence(...)
```

## Contributing

### Adding New Metrics

1. Create a new metrics class following the existing pattern
2. Implement the evaluation method
3. Add configuration options to `MetricsConfig`
4. Integrate with `EmailSequenceEvaluator`
5. Add tests and documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include unit tests for new features

## License

This evaluation metrics system is part of the Email Sequence AI project and follows the same licensing terms.

## Support

For questions and support:
- Check the documentation
- Review the example files
- Open an issue in the project repository
- Contact the development team 