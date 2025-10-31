# SEO Evaluation Metrics System

## Overview

The SEO Evaluation Metrics System provides comprehensive evaluation capabilities for SEO deep learning models, including classification, ranking, regression, and multitask evaluation with SEO-specific metrics.

## 🚀 Features

### Core Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC AUC, Cohen's Kappa
- **Regression Metrics**: MSE, RMSE, MAE, R², MAPE, SMAPE, Huber Loss
- **Ranking Metrics**: NDCG@K, MAP@K, MRR, Precision@K, Recall@K
- **SEO-Specific Metrics**: Content quality, User engagement, Technical SEO

### SEO-Specific Capabilities
- **Content Quality Evaluation**: Length, keyword density, readability scores
- **User Engagement Metrics**: Time on page, CTR, bounce rate, scroll depth
- **Technical SEO Metrics**: Core Web Vitals, mobile friendliness, page load speed
- **Ranking Performance**: NDCG, MAP, MRR with bias correction

### Advanced Features
- **Cross-Validation**: Configurable fold counts and bootstrap sampling
- **Statistical Analysis**: Confidence intervals and significance testing
- **Visualization**: Comprehensive plotting and reporting
- **Configuration Management**: YAML-based configuration with validation

## 📁 File Structure

```
seo/
├── evaluation_metrics.py          # Core evaluation system
├── example_seo_evaluation.py      # Usage examples
├── seo_evaluation_config.py       # Configuration management
└── README_SEO_EVALUATION_METRICS.md  # This documentation
```

## 🛠️ Installation

### Dependencies
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn pyyaml
```

### Import
```python
from evaluation_metrics import (
    SEOModelEvaluator, SEOMetricsConfig, 
    ClassificationMetricsConfig, RegressionMetricsConfig
)
```

## 📊 Usage Examples

### Basic SEO Model Evaluation

```python
import asyncio
from evaluation_metrics import SEOModelEvaluator, SEOMetricsConfig

async def evaluate_seo_model():
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
    
    # Evaluate model
    results = await evaluator.evaluate_seo_model(
        model, test_data, task_type="ranking"
    )
    
    return results

# Run evaluation
results = asyncio.run(evaluate_seo_model())
```

### Classification Task Evaluation

```python
# Evaluate keyword classification
classification_results = await evaluator.evaluate_seo_model(
    keyword_model, test_data, task_type="classification"
)

print(f"Accuracy: {classification_results['accuracy']:.4f}")
print(f"F1 Score: {classification_results['f1_score']:.4f}")
print(f"Precision: {classification_results['precision']:.4f}")
print(f"Recall: {classification_results['recall']:.4f}")
```

### Ranking Task Evaluation

```python
# Evaluate content ranking
ranking_results = await evaluator.evaluate_seo_model(
    ranking_model, test_data, task_type="ranking"
)

print(f"NDCG@5: {ranking_results['ndcg_at_5']:.4f}")
print(f"MAP@5: {ranking_results['map_at_5']:.4f}")
print(f"MRR: {ranking_results['mrr']:.4f}")
print(f"Content Quality: {ranking_results['overall_content_quality']:.4f}")
```

### Technical SEO Evaluation

```python
# Evaluate technical SEO metrics
technical_data = test_data.get('technical_data', {})
technical_metrics = evaluator.seo_metrics.calculate_technical_seo_metrics(
    technical_data
)

print(f"Page Load Speed: {technical_metrics['load_speed_score']:.4f}")
print(f"Mobile Friendliness: {technical_metrics['mobile_score_normalized']:.4f}")
print(f"LCP Score: {technical_metrics['lcp_score']:.4f}")
print(f"Overall Technical Score: {technical_metrics['overall_technical_score']:.4f}")
```

## ⚙️ Configuration

### SEO Metrics Configuration

```python
seo_config = SEOMetricsConfig(
    # Ranking evaluation
    ranking=SEORankingConfig(
        ndcg_k_values=[1, 3, 5, 10, 20],
        map_k_values=[1, 3, 5, 10, 20],
        apply_bias_correction=True
    ),
    
    # Content quality
    content_quality=SEOContentQualityConfig(
        min_content_length=300,
        optimal_content_length=1500,
        max_keyword_density=0.03
    ),
    
    # User engagement
    user_engagement=SEOUserEngagementConfig(
        min_time_on_page=30.0,
        max_bounce_rate=0.7
    ),
    
    # Technical SEO
    technical=SEOTechnicalConfig(
        max_load_time=3.0,
        max_lcp=2.5,
        max_fid=100.0,
        max_cls=0.1
    )
)
```

### YAML Configuration

```yaml
seo_metrics:
  task_types: ["classification", "ranking"]
  enable_cross_validation: true
  cv_folds: 5
  
  ranking:
    ndcg_k_values: [1, 3, 5, 10]
    apply_bias_correction: true
  
  content_quality:
    min_content_length: 300
    max_keyword_density: 0.03
  
  user_engagement:
    min_time_on_page: 30.0
    max_bounce_rate: 0.7
  
  technical:
    max_load_time: 3.0
    max_lcp: 2.5

classification:
  default_average: "weighted"
  enable_roc_auc: true

visualization:
  create_visualizations: true
  save_format: "png"
```

## 📈 Metrics Explained

### Ranking Metrics

#### NDCG (Normalized Discounted Cumulative Gain)
- **Purpose**: Measures ranking quality considering position and relevance
- **Range**: 0.0 to 1.0 (higher is better)
- **Formula**: DCG / IDCG where DCG = Σ(relevance_i / log2(i+1))

#### MAP (Mean Average Precision)
- **Purpose**: Measures precision across different recall levels
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Case**: Content ranking, search result evaluation

#### MRR (Mean Reciprocal Rank)
- **Purpose**: Measures the rank of the first relevant item
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Case**: First-click optimization, top-result evaluation

### Content Quality Metrics

#### Content Length Score
- **Calculation**: min(1.0, actual_length / min_required_length)
- **Thresholds**: 300 words minimum, 1500 words optimal

#### Keyword Density Score
- **Calculation**: 1.0 - min(1.0, density / max_allowed_density)
- **Thresholds**: 0.5% minimum, 1.5% optimal, 3% maximum

#### Readability Score
- **Calculation**: actual_score / 100.0
- **Thresholds**: 60 minimum, 80 optimal

### Technical SEO Metrics

#### Core Web Vitals
- **LCP (Largest Contentful Paint)**: ≤2.5s optimal
- **FID (First Input Delay)**: ≤100ms optimal
- **CLS (Cumulative Layout Shift)**: ≤0.1 optimal

#### Page Load Speed
- **Score**: 1.0 - (actual_time / max_allowed_time)
- **Threshold**: 3.0s maximum, 1.5s optimal

## 🔧 Advanced Usage

### Custom Metric Calculation

```python
class CustomSEOMetrics(SEOSpecificMetrics):
    def calculate_custom_metric(self, data):
        """Calculate custom SEO metric."""
        # Your custom logic here
        return custom_score

# Use custom metrics
evaluator.seo_metrics = CustomSEOMetrics(seo_config)
```

### Cross-Validation Integration

```python
from sklearn.model_selection import KFold

def cross_validate_seo_model(model, data, config):
    kfold = KFold(n_splits=config.seo_metrics.cv_folds, shuffle=True)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        # Train and evaluate
        fold_results = await evaluator.evaluate_seo_model(
            model, data[val_idx], task_type="ranking"
        )
        cv_scores.append(fold_results['ndcg_at_5'])
    
    return np.mean(cv_scores), np.std(cv_scores)
```

### Bootstrap Confidence Intervals

```python
def calculate_bootstrap_confidence_intervals(results, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for metrics."""
    bootstrap_samples = []
    
    for _ in range(1000):
        # Bootstrap sample
        sample_indices = np.random.choice(
            len(results), size=len(results), replace=True
        )
        sample_results = results[sample_indices]
        bootstrap_samples.append(np.mean(sample_results))
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_samples, lower_percentile)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile)
    
    return lower_bound, upper_bound
```

## 📊 Visualization

### Automatic Plot Generation

```python
# Generate evaluation plots
evaluator.plot_evaluation_metrics("seo_evaluation_plots.png")
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_seo_metrics_comparison(results_dict):
    """Create custom SEO metrics comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Task performance comparison
    tasks = list(results_dict.keys())
    overall_scores = [results.get('overall_score', 0) for results in results_dict.values()]
    
    axes[0, 0].bar(tasks, overall_scores)
    axes[0, 0].set_title('Overall Performance by Task')
    axes[0, 0].set_ylabel('Score')
    
    # Content quality breakdown
    if 'content_ranking' in results_dict:
        content_metrics = results_dict['content_ranking']
        quality_scores = [
            content_metrics.get('overall_content_quality', 0),
            content_metrics.get('overall_engagement_score', 0),
            content_metrics.get('overall_technical_score', 0)
        ]
        quality_labels = ['Content Quality', 'User Engagement', 'Technical SEO']
        
        axes[0, 1].pie(quality_scores, labels=quality_labels, autopct='%1.1f%%')
        axes[0, 1].set_title('SEO Quality Distribution')
    
    plt.tight_layout()
    plt.show()
```

## 🚀 Performance Optimization

### GPU Acceleration

```python
# Enable GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    evaluator.enable_gpu = True
```

### Batch Processing

```python
# Configure batch processing
config.batch_size = 64
config.num_workers = 8
config.enable_async_evaluation = True
```

### Memory Management

```python
# Optimize memory usage
import gc

def memory_efficient_evaluation(evaluator, model, data, batch_size=32):
    """Memory-efficient evaluation with garbage collection."""
    results = []
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_results = await evaluator.evaluate_seo_model(
            model, batch_data, task_type="ranking"
        )
        results.append(batch_results)
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return aggregate_results(results)
```

## 📋 Best Practices

### 1. Configuration Management
- Use YAML files for reproducible configurations
- Validate configurations before use
- Create task-specific optimized configurations

### 2. Evaluation Strategy
- Start with basic metrics, then add advanced ones
- Use cross-validation for robust evaluation
- Implement bootstrap confidence intervals

### 3. Performance Monitoring
- Monitor evaluation time and memory usage
- Use async evaluation for large datasets
- Implement progress bars for long evaluations

### 4. Result Interpretation
- Focus on business-relevant metrics
- Consider confidence intervals
- Track metrics over time for trends

### 5. SEO-Specific Considerations
- Align metrics with business goals
- Consider user experience metrics
- Monitor technical SEO performance

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Install missing dependencies
pip install pyyaml matplotlib seaborn
```

#### Memory Issues
```python
# Reduce batch size
config.batch_size = 16

# Enable garbage collection
import gc
gc.collect()
```

#### Performance Issues
```python
# Use async evaluation
config.enable_async_evaluation = True

# Optimize worker count
config.num_workers = min(8, os.cpu_count())
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable PyTorch anomaly detection
torch.autograd.set_detect_anomaly(True)
```

## 📚 References

### Academic Papers
- [Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/publication/learning-to-rank-for-information-retrieval/)
- [A Survey of Learning to Rank for Information Retrieval](https://link.springer.com/article/10.1007/s10791-009-9125-9)

### Industry Standards
- [Google Core Web Vitals](https://web.dev/vitals/)
- [SEO Best Practices](https://developers.google.com/search/docs/advanced/guidelines/quality-guidelines)

### Libraries
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [PyTorch Evaluation](https://pytorch.org/docs/stable/torch.html)

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies
3. Run tests: `python -m pytest tests/`
4. Follow PEP 8 style guidelines

### Adding New Metrics
1. Extend the appropriate metrics class
2. Add configuration options
3. Include tests and documentation
4. Update this README

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the examples and documentation

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Compatibility**: Python 3.8+, PyTorch 1.8+

