# 🔍 Specialized SEO Evaluation Metrics

## Overview

This module provides **appropriate evaluation metrics for SEO-specific tasks**, ensuring that model performance is measured using metrics that are relevant and meaningful for search engine optimization applications.

## 🎯 Key Features

### 1. **Task-Specific Metrics**
- **Classification**: Accuracy, precision, recall, F1-score, ROC-AUC, Cohen's Kappa
- **Regression**: MSE, RMSE, MAE, R², MAPE, SMAPE
- **Ranking**: NDCG, MAP, MRR, Top-K relevance
- **Clustering**: Silhouette score, Calinski-Harabasz index, content diversity

### 2. **SEO-Specific Metrics**
- **Content Quality**: Length, keyword density, readability, structure
- **Technical SEO**: Meta tags, headings, image alt text, HTML structure
- **User Experience**: Engagement, mobile-friendliness, content structure
- **Overall SEO Score**: Weighted combination of all components

### 3. **Comprehensive Evaluation**
- Automatic metric selection based on task type
- SEO content quality assessment
- Detailed performance reports
- Metric history tracking

## 🏗️ Architecture

```
SEOModelEvaluator
├── SEOSpecificMetrics
│   ├── Content Quality Scoring
│   ├── Technical SEO Analysis
│   ├── User Experience Evaluation
│   └── Overall SEO Scoring
├── Task-Specific Evaluators
│   ├── Classification Metrics
│   ├── Regression Metrics
│   ├── Ranking Metrics
│   └── Clustering Metrics
└── Report Generation
    ├── Metric Categorization
    ├── Performance Analysis
    └── SEO Insights
```

## 📊 Available Metrics

### **Classification Metrics**
```python
# Basic metrics
accuracy, precision, recall, f1_score

# Advanced metrics
roc_auc, cohen_kappa

# SEO-specific metrics
seo_precision, seo_recall, seo_f1
```

### **Regression Metrics**
```python
# Basic metrics
mse, rmse, mae, r2

# Additional metrics
mape, smape

# SEO-specific metrics
seo_accuracy_within_threshold, high_quality_detection_accuracy
```

### **Ranking Metrics**
```python
# Standard ranking metrics
ndcg, map, mrr

# SEO-specific ranking metrics
top_1_seo_relevance, top_3_seo_relevance, top_5_seo_relevance, top_10_seo_relevance
```

### **Clustering Metrics**
```python
# Standard clustering metrics
silhouette, calinski_harabasz

# SEO-specific clustering metrics
content_diversity
```

### **SEO Content Metrics**
```python
# Component scores
content_quality, keyword_optimization, technical_seo, readability, user_experience

# Overall score
overall_seo_score
```

## 🔧 Usage Examples

### **1. Basic Classification Evaluation**
```python
from seo_evaluation_metrics import SEOMetricsConfig, SEOModelEvaluator

# Configuration
config = SEOMetricsConfig(
    task_type="classification",
    num_classes=2,
    use_seo_specific=True
)

# Initialize evaluator
evaluator = SEOModelEvaluator(config)

# Evaluate model
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]
y_prob = [0.9, 0.1, 0.8, 0.4, 0.2]

metrics = evaluator.evaluate_classification(y_true, y_pred, y_prob)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"SEO F1: {metrics['seo_f1']:.4f}")
```

### **2. SEO Content Evaluation**
```python
from seo_evaluation_metrics import SEOSpecificMetrics

# Initialize SEO metrics
seo_metrics = SEOSpecificMetrics(config)

# Evaluate content
text = "<h1>SEO Guide</h1><p>Learn about optimization techniques.</p>"
html_content = "<html><head><title>SEO Guide</title></head><body>...</body></html>"

scores = seo_metrics.calculate_overall_seo_score(text, html_content)
print(f"Overall SEO Score: {scores['overall_seo_score']:.4f}")
print(f"Content Quality: {scores['content_quality']:.4f}")
```

### **3. Comprehensive Model Evaluation**
```python
# Evaluate with specialized metrics
specialized_metrics = model.evaluate_with_specialized_metrics(
    input_texts, y_true, y_pred, "classification"
)

# Generate comprehensive report
report = model.generate_comprehensive_report(
    input_texts, y_true, y_pred, "classification"
)
print(report)
```

## ⚙️ Configuration

### **SEOMetricsConfig Options**
```python
@dataclass
class SEOMetricsConfig:
    # Task settings
    task_type: str = "classification"  # classification, regression, clustering, ranking
    num_classes: int = 2
    average: str = "weighted"  # micro, macro, weighted, binary
    
    # SEO thresholds
    seo_score_threshold: float = 0.7
    content_quality_threshold: float = 0.6
    keyword_density_threshold: float = 0.02
    readability_threshold: float = 0.5
    
    # Evaluation options
    use_custom_metrics: bool = True
    use_seo_specific: bool = True
    normalize_scores: bool = True
```

### **SEO Scoring Weights**
```python
seo_weights = {
    'content_quality': 0.25,      # Content length, structure, quality
    'keyword_optimization': 0.20,  # Keyword density and relevance
    'technical_seo': 0.20,        # HTML structure, meta tags
    'readability': 0.15,          # Flesch Reading Ease score
    'user_experience': 0.20       # Engagement, mobile-friendliness
}
```

## 📈 Performance Features

### **1. Efficient Calculation**
- Vectorized operations using NumPy
- Optimized algorithms for large datasets
- Caching for repeated calculations

### **2. Memory Optimization**
- Minimal memory footprint
- Efficient data structures
- Batch processing support

### **3. Scalability**
- Handles datasets of any size
- Parallel processing capabilities
- GPU acceleration support (when available)

## 🧪 Testing

### **Run Specialized Metrics Tests**
```bash
# Test all specialized metrics
python test_specialized_metrics.py

# Test specific components
python -c "
from seo_evaluation_metrics import SEOSpecificMetrics, SEOMetricsConfig
config = SEOMetricsConfig(use_seo_specific=True)
seo_metrics = SEOSpecificMetrics(config)
score = seo_metrics.calculate_content_quality_score('SEO optimization guide')
print(f'Content Quality Score: {score:.4f}')
"
```

### **Test Integration with Ultra-Optimized System**
```bash
# Test complete integration
python test_specialized_metrics.py --integration

# Test performance benchmarks
python test_specialized_metrics.py --benchmark
```

## 📊 Metric Interpretations

### **Classification Metrics**
- **Accuracy**: Overall correctness of predictions
- **Precision**: How many predicted positive cases are actually positive
- **Recall**: How many actual positive cases were detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes
- **Cohen's Kappa**: Agreement between predictions and true labels

### **Regression Metrics**
- **MSE/RMSE**: Average squared/poot mean squared error
- **MAE**: Average absolute error
- **R²**: Proportion of variance explained by the model
- **MAPE**: Mean absolute percentage error
- **SMAPE**: Symmetric mean absolute percentage error

### **Ranking Metrics**
- **NDCG**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank

### **SEO Content Metrics**
- **Content Quality**: Based on length, structure, and readability
- **Keyword Optimization**: Keyword density and relevance
- **Technical SEO**: HTML structure and meta tags
- **Readability**: Flesch Reading Ease score
- **User Experience**: Engagement and mobile-friendliness

## 🔍 Advanced Features

### **1. Custom Metric Calculation**
```python
class CustomSEOMetrics(SEOSpecificMetrics):
    def calculate_custom_score(self, text: str) -> float:
        # Implement custom scoring logic
        return custom_score
    
    def calculate_overall_seo_score(self, text: str, html_content: str = "") -> Dict[str, float]:
        scores = super().calculate_overall_seo_score(text, html_content)
        scores['custom_score'] = self.calculate_custom_score(text)
        return scores
```

### **2. Metric Customization**
```python
# Custom configuration
config = SEOMetricsConfig(
    task_type="classification",
    num_classes=3,  # Multi-class classification
    average="macro",  # Macro averaging for multi-class
    seo_score_threshold=0.8,  # Higher threshold
    use_seo_specific=True
)
```

### **3. Report Customization**
```python
# Generate custom report
report = evaluator.generate_evaluation_report(
    metrics, 
    task_name="Custom SEO Analysis"
)

# Save metrics to file
evaluator.save_metrics(metrics, "custom_seo_metrics.json")

# Load metrics from file
loaded_metrics = evaluator.load_metrics("custom_seo_metrics.json")
```

## 🚨 Error Handling

### **Robust Metric Calculation**
- Handles edge cases (empty data, single class)
- Graceful degradation for unsupported metrics
- Comprehensive error messages

### **Input Validation**
- Type checking for inputs
- Range validation for parameters
- Automatic data conversion when possible

### **Performance Monitoring**
- Execution time tracking
- Memory usage monitoring
- Error rate tracking

## 📚 Dependencies

### **Core Dependencies**
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning metrics
- **Torch**: PyTorch tensor support

### **Optional Dependencies**
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualization

## 🎯 Best Practices

### **1. Metric Selection**
- Choose metrics appropriate for your task type
- Consider business requirements and SEO goals
- Use multiple metrics for comprehensive evaluation

### **2. Threshold Configuration**
- Set appropriate thresholds for your use case
- Consider industry standards and best practices
- Regularly review and adjust thresholds

### **3. Performance Monitoring**
- Track metrics over time
- Monitor for degradation or improvement
- Use metrics for model selection and tuning

## 🔮 Future Enhancements

- **Custom Metric Support**: User-defined evaluation functions
- **Real-time Evaluation**: Streaming metric calculation
- **Advanced Visualizations**: Interactive metric dashboards
- **Metric Comparison**: A/B testing and model comparison
- **Automated Threshold Tuning**: ML-based threshold optimization

## 📄 License

This module is part of the Blatam Academy SEO evaluation system.

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive tests for new metrics
3. Document metric calculations and interpretations
4. Ensure backward compatibility
5. Optimize for performance

---

**🔍 Ready for production use with comprehensive SEO evaluation capabilities!**
