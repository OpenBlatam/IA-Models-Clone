# Blaze AI Evaluation Metrics

A comprehensive evaluation framework for AI models, providing appropriate metrics for different tasks including classification, text generation, image generation, SEO optimization, and brand voice analysis.

## 🎯 Overview

This evaluation module provides industry-standard and custom metrics for assessing AI model performance across various domains. Each task type has been carefully designed with appropriate evaluation metrics that align with best practices in the field.

## 🚀 Features

### Task-Specific Evaluation Metrics

- **Classification**: Accuracy, precision, recall, F1-score, confusion matrix
- **Text Generation**: BLEU, ROUGE, BERTScore, perplexity, content quality metrics
- **Image Generation**: FID, Inception Score, LPIPS, CLIP score, image quality metrics
- **SEO Optimization**: Keyword density, readability, content structure, technical SEO
- **Brand Voice**: Tone consistency, vocabulary analysis, sentiment analysis, brand alignment

### Key Benefits

- **Unified Interface**: Single registry for all evaluation types
- **Automatic Recommendations**: AI-powered suggestions for improvement
- **Result Persistence**: Automatic saving and export capabilities
- **Model Comparison**: Built-in tools for comparing multiple models
- **Extensible**: Easy to add new metrics and evaluation types

## 📦 Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Optional Dependencies
Some evaluation metrics require additional libraries:
```bash
# For enhanced text evaluation
pip install bert-score transformers

# For image evaluation
pip install torchmetrics
pip install git+https://github.com/openai/CLIP.git

# For SEO evaluation
pip install beautifulsoup4 requests

# For brand voice analysis
pip install nltk textblob
```

## 🎯 Usage Examples

### Quick Start

```python
from blaze_ai.evaluation import get_evaluation_registry

# Get the evaluation registry
registry = get_evaluation_registry()

# Evaluate a classification model
result = registry.evaluate(
    task_type="classification",
    model=your_model,
    data_loader=test_loader,
    device="cuda",
    model_name="my_classifier"
)

print(f"Accuracy: {result.metrics['accuracy']}")
print(f"Recommendations: {result.recommendations}")
```

### Text Generation Evaluation

```python
from blaze_ai.evaluation import evaluate_text_generation_batch

# Evaluate generated text against references
metrics = evaluate_text_generation_batch(
    generated_texts=["AI is transforming business."],
    reference_texts=["Artificial intelligence is changing business."]
)

print(f"BLEU Score: {metrics['bleu']}")
print(f"BERTScore: {metrics['bert_score']}")
```

### Image Generation Evaluation

```python
from blaze_ai.evaluation import evaluate_image_generation_batch

# Evaluate generated images
metrics = evaluate_image_generation_batch(
    real_images=real_image_tensor,
    generated_images=generated_image_tensor,
    text_prompts=["A beautiful landscape"],
    device="cuda"
)

print(f"FID Score: {metrics['fid_score']}")
print(f"Inception Score: {metrics['inception_score']}")
```

### SEO Optimization Evaluation

```python
from blaze_ai.evaluation import evaluate_seo_optimization

# Evaluate content for SEO
evaluation = evaluate_seo_optimization(
    text="Your content here...",
    target_keywords=["artificial intelligence", "AI"]
)

print(f"Overall Score: {evaluation['quality_score']['overall_score']}/100")
print(f"Grade: {evaluation['quality_score']['grade']}")
```

### Brand Voice Evaluation

```python
from blaze_ai.evaluation import evaluate_brand_voice

# Define brand guidelines
brand_guidelines = {
    "tone": {"formality": 0.7, "friendliness": 0.6},
    "vocabulary": {"brand_terms": ["innovation", "quality"]}
}

# Evaluate brand voice consistency
evaluation = evaluate_brand_voice(
    texts=["Content piece 1...", "Content piece 2..."],
    brand_guidelines=brand_guidelines
)

print(f"Brand Alignment: {evaluation['brand_alignment']['overall_brand_alignment']}")
```

## 🔧 Advanced Usage

### Custom Evaluation Registry

```python
from blaze_ai.evaluation import EvaluationMetricsRegistry

# Create custom registry with specific storage
registry = EvaluationMetricsRegistry(storage_dir="my_evaluation_results")

# Evaluate multiple models
model_names = ["model_v1", "model_v2", "model_v3"]
comparison = registry.compare_models("classification", model_names)

print("Best performing model:", comparison["summary"]["accuracy"]["best_model"])
```

### Batch Evaluation

```python
# Evaluate multiple models in batch
models = [model1, model2, model3]
results = []

for i, model in enumerate(models):
    result = registry.evaluate(
        task_type="text_generation",
        model=model,
        references=reference_texts,
        candidates=generated_texts[i],
        model_name=f"model_{i+1}"
    )
    results.append(result)

# Export all results
registry.export_results("batch_evaluation_results.json")
```

### Custom Metrics Integration

```python
# The registry automatically handles different task types
# You can extend it by adding new evaluators

class CustomEvaluator:
    def evaluate(self, model, **kwargs):
        # Your custom evaluation logic
        return {"custom_metric": 0.95}

# Register custom evaluator
registry.task_evaluators["custom_task"] = CustomEvaluator().evaluate
```

## 📊 Available Metrics

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Text Generation Metrics
- **BLEU**: N-gram overlap with references
- **ROUGE**: Recall-oriented evaluation
- **BERTScore**: Semantic similarity using BERT
- **Perplexity**: Language model performance
- **Content Quality**: Readability, repetition, vocabulary diversity

### Image Generation Metrics
- **FID**: Fréchet Inception Distance
- **Inception Score**: Image quality and diversity
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **CLIP Score**: Text-image alignment
- **Image Quality**: Brightness, contrast, sharpness, color diversity

### SEO Optimization Metrics
- **Keyword Density**: Target keyword usage
- **Readability**: Flesch Reading Ease, grade level
- **Content Structure**: Headings, paragraphs, lists
- **Technical SEO**: Meta tags, H1 tags, alt text
- **Overall Score**: Comprehensive quality assessment

### Brand Voice Metrics
- **Tone Consistency**: Formality, friendliness, professionalism
- **Vocabulary Consistency**: Brand term usage, preferred terms
- **Sentiment Consistency**: Emotional tone stability
- **Brand Alignment**: Overall consistency score
- **Content Structure**: Alignment with brand guidelines

## 🎨 Customization

### Adding New Metrics

```python
def custom_metric_function(predictions, targets):
    # Your custom metric calculation
    return custom_score

# Add to existing evaluation
def evaluate_with_custom_metric(model, data_loader, device):
    # Standard evaluation
    standard_metrics = evaluate_classification(model, data_loader, device)
    
    # Add custom metric
    custom_score = custom_metric_function(predictions, targets)
    standard_metrics["custom_metric"] = custom_score
    
    return standard_metrics
```

### Custom Recommendations

```python
def generate_custom_recommendations(metrics):
    recommendations = []
    
    if metrics["custom_metric"] < 0.8:
        recommendations.append("Custom metric below threshold - consider optimization")
    
    return recommendations
```

## 📈 Performance Considerations

### GPU Acceleration
- Most metrics support GPU acceleration via PyTorch
- Set `device="cuda"` for optimal performance
- Large image batches benefit significantly from GPU

### Memory Management
- Evaluation results are automatically saved to disk
- Use `registry.clear_history()` to free memory
- Export results periodically to maintain performance

### Batch Processing
- Process large datasets in batches
- Use `DataLoader` for efficient data iteration
- Consider parallel evaluation for multiple models

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Run Examples
```bash
python -m blaze_ai.evaluation.example_usage
```

### Coverage Report
```bash
pytest --cov=blaze_ai.evaluation tests/ --cov-report=html
```

## 📚 API Reference

### Core Functions

#### `get_evaluation_registry(storage_dir=None)`
Get the global evaluation metrics registry.

#### `evaluate_model(task_type, model, **kwargs)`
Convenience function to evaluate a model using the global registry.

#### `EvaluationMetricsRegistry`
Main registry class for managing evaluation metrics.

### Task Types

- `"classification"`: Classification model evaluation
- `"text_generation"`: Text generation model evaluation
- `"image_generation"`: Image generation model evaluation
- `"seo_optimization"`: SEO content evaluation
- `"brand_voice"`: Brand voice consistency evaluation

## 🤝 Contributing

### Adding New Metrics
1. Create new evaluation module in appropriate directory
2. Implement evaluation function with standard interface
3. Add to registry task evaluators
4. Include comprehensive tests
5. Update documentation

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Include docstrings for all functions
- Add unit tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Check the examples in `example_usage.py`
- Review the test files for usage patterns
- Open an issue for bugs or feature requests
- Consult the main Blaze AI documentation

## 🔮 Future Enhancements

- **Real-time Evaluation**: Live monitoring during training
- **Automated Optimization**: AI-powered metric improvement suggestions
- **Integration APIs**: Connect with external evaluation services
- **Visualization Tools**: Interactive charts and dashboards
- **Multi-modal Evaluation**: Combined text, image, and audio metrics
