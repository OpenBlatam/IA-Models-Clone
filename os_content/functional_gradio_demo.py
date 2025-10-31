from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import logging
from dataclasses import dataclass, asdict
import yaml
from pathlib import Path
import time
from functools import partial, reduce
import operator
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional
import asyncio
"""
Functional Gradio Inference Demo - No Classes, Pure Functions
Follows functional programming principles with declarative style.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InferenceConfig:
    """Immutable configuration for model inference."""
    model_path: str = "models/best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_samples: int = 1000
    confidence_threshold: float = 0.5
    max_sequence_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

# Pure Functions for Model Operations
def create_simple_transformer_model(
    vocab_size: int = 1000,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    max_seq_length: int = 512
) -> nn.Module:
    """Create a simple transformer model using pure function approach."""
    
    model = nn.Module()
    model.d_model = d_model
    model.max_seq_length = max_seq_length
    
    # Embeddings
    model.token_embedding = nn.Embedding(vocab_size, d_model)
    model.positional_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
    
    # Transformer layers
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4,
        dropout=0.1,
        batch_first=True
    )
    model.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    # Output layers
    model.classifier = nn.Linear(d_model, 2)
    model.dropout = nn.Dropout(0.1)
    
    return model

def forward_pass(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pure function for model forward pass."""
    batch_size, seq_length = x.shape
    
    # Token embeddings
    token_embeddings = model.token_embedding(x)
    
    # Add positional encoding
    positional_embeddings = model.positional_encoding[:seq_length].unsqueeze(0)
    embeddings = token_embeddings + positional_embeddings
    
    # Apply transformer
    transformer_output = model.transformer(embeddings)
    
    # Global average pooling
    pooled_output = torch.mean(transformer_output, dim=1)
    
    # Classification
    output = model.dropout(pooled_output)
    logits = model.classifier(output)
    
    return logits

# Data Processing Functions
def generate_synthetic_data(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data using pure function."""
    sequences = np.random.randint(0, 1000, size=(num_samples, 50))
    labels = (np.sum(sequences, axis=1) > 25000).astype(np.int64)
    return sequences, labels

def preprocess_text(text: str, max_length: int) -> torch.Tensor:
    """Preprocess text input using pure function."""
    tokens = [ord(c) % 1000 for c in text[:max_length]]
    
    # Pad or truncate
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
    
    return torch.tensor([tokens], dtype=torch.long)

def load_model_weights(model: nn.Module, model_path: str, device: str) -> nn.Module:
    """Load model weights using pure function."""
    try:
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using random weights")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Prediction Functions
def predict_single(
    model: nn.Module,
    text: str,
    config: InferenceConfig,
    device: torch.device
) -> Dict[str, Any]:
    """Single prediction using pure function."""
    try:
        with torch.no_grad():
            input_tensor = preprocess_text(text, config.max_sequence_length)
            input_tensor = input_tensor.to(device)
            
            start_time = time.time()
            logits = forward_pass(model, input_tensor)
            inference_time = time.time() - start_time
            
            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities).item()
            prediction = torch.argmax(logits, dim=1).item()
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().tolist(),
                'inference_time': inference_time,
                'text_length': len(text)
            }
            
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            'prediction': -1,
            'confidence': 0.0,
            'probabilities': [0.0, 0.0],
            'inference_time': 0.0,
            'error': str(e)
        }

def batch_predict(
    model: nn.Module,
    texts: List[str],
    config: InferenceConfig,
    device: torch.device
) -> List[Dict[str, Any]]:
    """Batch prediction using pure function and functional programming."""
    predict_fn = partial(predict_single, model, config=config, device=device)
    return list(map(predict_fn, texts))

# Analysis Functions
def calculate_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Calculate performance metrics using pure function."""
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    return {'accuracy': accuracy}

def extract_prediction_data(results: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[float], List[int]]:
    """Extract data from prediction results using pure function."""
    predictions = [r['prediction'] for r in results]
    confidences = [r['confidence'] for r in results]
    inference_times = [r['inference_time'] for r in results]
    text_lengths = [r['text_length'] for r in results]
    return predictions, confidences, inference_times, text_lengths

def analyze_model_performance(
    model: nn.Module,
    config: InferenceConfig,
    device: torch.device,
    num_samples: int = 100
) -> Dict[str, Any]:
    """Analyze model performance using pure functions."""
    try:
        # Generate data
        sequences, labels = generate_synthetic_data(num_samples)
        texts = [''.join([chr(t % 26 + 97) for t in seq]) for seq in sequences]
        
        # Get predictions
        results = batch_predict(model, texts, config, device)
        
        # Extract metrics
        predictions, confidences, inference_times, text_lengths = extract_prediction_data(results)
        
        # Calculate statistics
        metrics = calculate_metrics(predictions, labels)
        avg_confidence = np.mean(confidences)
        avg_inference_time = np.mean(inference_times)
        
        return {
            'accuracy': metrics['accuracy'],
            'avg_confidence': avg_confidence,
            'avg_inference_time': avg_inference_time,
            'total_samples': num_samples,
            'predictions': predictions,
            'labels': labels,
            'confidences': confidences,
            'inference_times': inference_times,
            'text_lengths': text_lengths
        }
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {e}")
        return {'error': str(e)}

# Visualization Functions
def create_confusion_matrix(predictions: List[int], labels: List[int]) -> Any:
    """Create confusion matrix using pure function."""
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig

def create_confidence_distribution(confidences: List[float]) -> Any:
    """Create confidence distribution plot using pure function."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Prediction Confidence Distribution')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    ax.axvline(mean_conf, color='red', linestyle='--', 
              label=f'Mean: {mean_conf:.3f}')
    ax.axvline(mean_conf + std_conf, color='orange', linestyle='--', 
              label=f'+1 Std: {mean_conf + std_conf:.3f}')
    ax.axvline(mean_conf - std_conf, color='orange', linestyle='--', 
              label=f'-1 Std: {mean_conf - std_conf:.3f}')
    ax.legend()
    
    return fig

def create_inference_time_analysis(inference_times: List[float], text_lengths: List[int]) -> Any:
    """Create inference time analysis plot using pure function."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Inference time distribution
    ax1.hist(inference_times, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.set_title('Inference Time Distribution')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Inference time vs text length
    ax2.scatter(text_lengths, inference_times, alpha=0.6, color='purple')
    ax2.set_title('Inference Time vs Text Length')
    ax2.set_xlabel('Text Length')
    ax2.set_ylabel('Inference Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    if len(text_lengths) > 1:
        z = np.polyfit(text_lengths, inference_times, 1)
        p = np.poly1d(z)
        ax2.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)
    
    plt.tight_layout()
    return fig

# Gradio Interface Functions
def create_single_prediction_interface(
    model: nn.Module,
    config: InferenceConfig,
    device: torch.device
) -> gr.Blocks:
    """Create single prediction interface using pure function."""
    
    def predict_wrapper(text: str) -> Tuple[Dict[str, Any], float]:
        result = predict_single(model, text, config, device)
        return result, result.get('confidence', 0.0)
    
    with gr.Blocks() as interface:
        gr.Markdown("## Single Text Prediction")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text for classification...",
                    lines=3,
                    max_lines=10
                )
                predict_btn = gr.Button("Predict", variant="primary")
            
            with gr.Column():
                prediction_output = gr.JSON(label="Prediction Results")
                confidence_gauge = gr.Gauge(label="Confidence", minimum=0, maximum=1)
        
        predict_btn.click(
            fn=predict_wrapper,
            inputs=text_input,
            outputs=[prediction_output, confidence_gauge]
        )
    
    return interface

def create_batch_analysis_interface(
    model: nn.Module,
    config: InferenceConfig,
    device: torch.device
) -> gr.Blocks:
    """Create batch analysis interface using pure function."""
    
    def analyze_wrapper(num_samples: int) -> Tuple[Dict[str, Any], Any, Any, Any]:
        results = analyze_model_performance(model, config, device, num_samples)
        
        if 'error' in results:
            return results, None, None, None
        
        confusion_fig = create_confusion_matrix(results['predictions'], results['labels'])
        confidence_fig = create_confidence_distribution(results['confidences'])
        time_fig = create_inference_time_analysis(results['inference_times'], results['text_lengths'])
        
        metrics = {
            'accuracy': f"{results['accuracy']:.3f}",
            'avg_confidence': f"{results['avg_confidence']:.3f}",
            'avg_inference_time': f"{results['avg_inference_time']:.4f}s",
            'total_samples': results['total_samples']
        }
        
        return metrics, confusion_fig, confidence_fig, time_fig
    
    with gr.Blocks() as interface:
        gr.Markdown("## Batch Performance Analysis")
        
        with gr.Row():
            num_samples_input = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="Number of Samples"
            )
            analyze_btn = gr.Button("Analyze Performance", variant="primary")
        
        with gr.Row():
            with gr.Column():
                metrics_output = gr.JSON(label="Performance Metrics")
            
            with gr.Column():
                confusion_plot = gr.Plot(label="Confusion Matrix")
        
        with gr.Row():
            confidence_plot = gr.Plot(label="Confidence Distribution")
            time_plot = gr.Plot(label="Inference Time Analysis")
        
        analyze_btn.click(
            fn=analyze_wrapper,
            inputs=num_samples_input,
            outputs=[metrics_output, confusion_plot, confidence_plot, time_plot]
        )
    
    return interface

def create_model_info_interface(config: InferenceConfig) -> gr.Blocks:
    """Create model information interface using pure function."""
    
    with gr.Blocks() as interface:
        gr.Markdown("## Model and System Information")
        
        model_info = gr.JSON(label="Model Configuration", value=asdict(config))
        
        device_info = gr.Textbox(
            label="Device Information",
            value=f"Device: {config.device}\nCUDA Available: {torch.cuda.is_available()}\nGPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
            lines=3
        )
        
        if torch.cuda.is_available():
            gpu_info = gr.Textbox(
                label="GPU Information",
                value=f"GPU Name: {torch.cuda.get_device_name()}\nGPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                lines=2
            )
    
    return interface

def create_configuration_interface(config: InferenceConfig) -> gr.Blocks:
    """Create configuration interface using pure function."""
    
    def save_config(conf_thresh: float, temp: float, max_seq: int, batch_size: int) -> Dict[str, Any]:
        new_config = InferenceConfig(
            confidence_threshold=conf_thresh,
            temperature=temp,
            max_sequence_length=int(max_seq),
            batch_size=int(batch_size)
        )
        return asdict(new_config)
    
    with gr.Blocks() as interface:
        gr.Markdown("## System Configuration")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Inference Settings")
                confidence_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=config.confidence_threshold,
                    step=0.05,
                    label="Confidence Threshold"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=config.temperature,
                    step=0.1,
                    label="Temperature"
                )
            
            with gr.Column():
                gr.Markdown("### Model Settings")
                max_seq_length = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=config.max_sequence_length,
                    step=64,
                    label="Max Sequence Length"
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=config.batch_size,
                    step=1,
                    label="Batch Size"
                )
        
        save_config_btn = gr.Button("Save Configuration", variant="secondary")
        config_output = gr.JSON(label="Current Configuration")
        
        save_config_btn.click(
            fn=save_config,
            inputs=[confidence_threshold, temperature, max_seq_length, batch_size],
            outputs=config_output
        )
    
    return interface

# Main Interface Creation Function
def create_functional_gradio_interface() -> gr.Blocks:
    """Create the complete functional Gradio interface."""
    
    # Initialize configuration and model
    config = InferenceConfig()
    device = torch.device(config.device)
    model = create_simple_transformer_model()
    model = load_model_weights(model, config.model_path, device)
    
    # Create interface components
    with gr.Blocks(title="Functional Training System - Inference Demo", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("# 🤖 Functional Training System - Model Inference Demo")
        gr.Markdown("This demo showcases functional programming principles with pure functions and no classes.")
        
        with gr.Tab("Single Prediction"):
            single_interface = create_single_prediction_interface(model, config, device)
        
        with gr.Tab("Batch Analysis"):
            batch_interface = create_batch_analysis_interface(model, config, device)
        
        with gr.Tab("Model Information"):
            info_interface = create_model_info_interface(config)
        
        with gr.Tab("Configuration"):
            config_interface = create_configuration_interface(config)
    
    return interface

# Application Entry Point
def launch_functional_demo() -> None:
    """Launch the functional Gradio demo."""
    interface = create_functional_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

match __name__:
    case "__main__":
    launch_functional_demo() 