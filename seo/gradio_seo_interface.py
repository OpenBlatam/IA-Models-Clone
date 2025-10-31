#!/usr/bin/env python3
"""
Gradio Interface for Ultra-Optimized SEO Evaluation System
With Gradient Clipping and NaN/Inf Handling
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import logging
from typing import Dict, List, Tuple, Any
import asyncio
from datetime import datetime

# Add the current directory to the path to import the SEO modules
sys.path.append(str(Path(__file__).parent))

from evaluation_metrics_ultra_optimized import (
    UltraOptimizedConfig, 
    UltraOptimizedSEOMetricsModule, 
    UltraOptimizedSEOTrainer,
    SEOTokenizer,
    SEODataset
)
from seo_evaluation_metrics import SEOModelEvaluator, SEOMetricsConfig
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradioSEOInterface:
    """Gradio interface for the ultra-optimized SEO evaluation system."""
    
    def __init__(self):
        self.model = None
        self.trainer = None
        self.config = None
        self.training_history = []
        self.current_health_status = {}
        
        # Initialize default configuration
        self.default_config = UltraOptimizedConfig(
            use_multi_gpu=False,  # Disable for web interface
            use_amp=True,
            use_lora=True,
            use_diffusion=False,  # Disable for web interface
            batch_size=8,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            patience=5,
            num_epochs=10
        )
    
    def initialize_model(self, config_dict: Dict[str, Any]) -> str:
        """Initialize the SEO model with custom configuration."""
        try:
            # Update configuration with user inputs
            for key, value in config_dict.items():
                if hasattr(self.default_config, key):
                    setattr(self.default_config, key, value)
            
            self.config = self.default_config
            
            # Initialize model and trainer
            self.model = UltraOptimizedSEOMetricsModule(self.config)
            self.trainer = UltraOptimizedSEOTrainer(self.model, self.config)
            
            # Get model statistics
            stats = self.trainer.get_training_stats()
            
            return f"✅ Model initialized successfully!\n\n" \
                   f"📊 Model Statistics:\n" \
                   f"• Total Parameters: {stats['total_parameters']:,}\n" \
                   f"• Trainable Parameters: {stats['trainable_parameters']:,}\n" \
                   f"• Device: {stats['device']}\n" \
                   f"• AMP Enabled: {stats['amp_enabled']}\n" \
                   f"• Learning Rate: {stats['current_learning_rate']:.2e}\n" \
                   f"• Weight Decay: {stats['current_weight_decay']:.2e}"
        
        except Exception as e:
            return f"❌ Error initializing model: {str(e)}"
    
    def create_sample_data(self, num_samples: int, task_type: str) -> Tuple[List[str], List[int]]:
        """Create sample SEO data for demonstration."""
        sample_texts = [
            "SEO optimization techniques for better search engine rankings",
            "Content marketing strategies for improved organic traffic",
            "Technical SEO best practices for website optimization",
            "Keyword research and analysis for content creation",
            "On-page SEO elements and their importance",
            "Link building strategies for domain authority",
            "Mobile-first indexing and responsive design",
            "Local SEO optimization for small businesses",
            "E-commerce SEO strategies for online stores",
            "Voice search optimization techniques"
        ]
        
        # Extend sample texts
        extended_texts = []
        for i in range(num_samples):
            text = sample_texts[i % len(sample_texts)]
            extended_texts.append(f"{text} - Sample {i+1}")
        
        # Create labels based on task type
        if task_type == "classification":
            labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]  # Binary classification
        elif task_type == "regression":
            labels = [0.1 + (i * 0.1) for i in range(num_samples)]  # Regression scores
        else:
            labels = [i % 3 for i in range(num_samples)]  # Multi-class
        
        return extended_texts, labels
    
    def train_model(self, training_texts: str, training_labels: str, 
                   validation_texts: str, validation_labels: str,
                   epochs: int, batch_size: int) -> Tuple[str, Dict[str, Any]]:
        """Train the SEO model with provided data."""
        try:
            if self.model is None or self.trainer is None:
                return "❌ Model not initialized. Please initialize the model first.", {}
            
            # Parse input data
            train_texts = [text.strip() for text in training_texts.split('\n') if text.strip()]
            train_labels = [int(label.strip()) for label in training_labels.split('\n') if label.strip()]
            
            val_texts = [text.strip() for text in validation_texts.split('\n') if text.strip()]
            val_labels = [int(label.strip()) for label in validation_labels.split('\n') if label.strip()]
            
            if len(train_texts) != len(train_labels) or len(val_texts) != len(val_labels):
                return "❌ Number of texts and labels must match.", {}
            
            # Update configuration
            self.config.num_epochs = epochs
            self.config.batch_size = batch_size
            
            # Create datasets
            train_dataset = SEODataset(train_texts, torch.tensor(train_labels), self.model.seo_tokenizer)
            val_dataset = SEODataset(val_texts, torch.tensor(val_labels), self.model.seo_tokenizer)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training loop
            training_log = []
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                train_accuracy = 0.0
                
                for batch in train_loader:
                    # Move batch to device
                    batch = {k: v.to(self.trainer.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    # Training step
                    metrics = self.trainer.train_step(batch)
                    train_loss += metrics['loss']
                    train_accuracy += metrics['accuracy']
                
                avg_train_loss = train_loss / len(train_loader)
                avg_train_accuracy = train_accuracy / len(train_loader)
                
                # Validation
                val_metrics = self.trainer.validate(val_loader)
                
                # Check training health
                health_status = self.trainer.monitor_training_health()
                self.current_health_status = health_status
                
                # Log progress
                log_entry = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_accuracy': avg_train_accuracy,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'health_status': health_status
                }
                training_log.append(log_entry)
                
                # Early stopping check
                if not health_status['overall_healthy']:
                    training_log.append({
                        'epoch': epoch + 1,
                        'message': 'Training stopped due to health issues'
                    })
                    break
            
            # Generate training report
            report = self._generate_training_report(training_log)
            
            # Create training plots
            plots = self._create_training_plots(training_log)
            
            return report, plots
        
        except Exception as e:
            return f"❌ Training error: {str(e)}", {}
    
    def _generate_training_report(self, training_log: List[Dict]) -> str:
        """Generate a comprehensive training report."""
        if not training_log:
            return "No training data available."
        
        report = "📊 Training Report\n\n"
        
        # Final metrics
        final_log = training_log[-1]
        report += f"🎯 Final Metrics:\n"
        report += f"• Training Loss: {final_log['train_loss']:.4f}\n"
        report += f"• Training Accuracy: {final_log['train_accuracy']:.4f}\n"
        report += f"• Validation Loss: {final_log['val_loss']:.4f}\n"
        report += f"• Validation Accuracy: {final_log['val_accuracy']:.4f}\n\n"
        
        # Health status
        health_status = final_log.get('health_status', {})
        report += f"🏥 Training Health:\n"
        report += f"• Model Healthy: {'✅' if health_status.get('model_healthy', False) else '❌'}\n"
        report += f"• Gradients Healthy: {'✅' if health_status.get('gradients_healthy', False) else '❌'}\n"
        report += f"• Losses Healthy: {'✅' if health_status.get('losses_healthy', False) else '❌'}\n"
        report += f"• Parameters Healthy: {'✅' if health_status.get('parameters_healthy', False) else '❌'}\n"
        report += f"• Overall Healthy: {'✅' if health_status.get('overall_healthy', False) else '❌'}\n\n"
        
        # Training statistics
        if self.trainer:
            stats = self.trainer.get_training_stats()
            report += f"📈 Training Statistics:\n"
            report += f"• Current Learning Rate: {stats['current_learning_rate']:.2e}\n"
            report += f"• Current Weight Decay: {stats['current_weight_decay']:.2e}\n"
            if stats.get('current_gradient_norm'):
                report += f"• Current Gradient Norm: {stats['current_gradient_norm']:.6f}\n"
        
        return report
    
    def _create_training_plots(self, training_log: List[Dict]) -> Dict[str, Any]:
        """Create training visualization plots."""
        if not training_log:
            return {}
        
        # Extract data for plotting
        epochs = [log['epoch'] for log in training_log if 'epoch' in log]
        train_losses = [log['train_loss'] for log in training_log if 'train_loss' in log]
        val_losses = [log['val_loss'] for log in training_log if 'val_loss' in log]
        train_accuracies = [log['train_accuracy'] for log in training_log if 'train_accuracy' in log]
        val_accuracies = [log['val_accuracy'] for log in training_log if 'val_accuracy' in log]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Health status plot
        health_data = []
        for log in training_log:
            if 'health_status' in log:
                health = log['health_status']
                health_data.append([
                    health.get('model_healthy', False),
                    health.get('gradients_healthy', False),
                    health.get('losses_healthy', False),
                    health.get('parameters_healthy', False)
                ])
        
        if health_data:
            health_array = np.array(health_data)
            health_labels = ['Model', 'Gradients', 'Losses', 'Parameters']
            ax3.imshow(health_array.T, cmap='RdYlGn', aspect='auto')
            ax3.set_title('Training Health Status Over Time')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Health Component')
            ax3.set_yticks(range(len(health_labels)))
            ax3.set_yticklabels(health_labels)
            ax3.set_xticks(range(0, len(epochs), max(1, len(epochs)//5)))
            ax3.set_xticklabels([epochs[i] for i in range(0, len(epochs), max(1, len(epochs)//5))])
        
        # Gradient norm plot (if available)
        if self.trainer:
            try:
                grad_norm = self.trainer._get_gradient_norm()
                ax4.bar(['Current'], [grad_norm], color='skyblue', alpha=0.7)
                ax4.axhline(y=self.config.max_grad_norm, color='red', linestyle='--', 
                           label=f'Clipping Threshold ({self.config.max_grad_norm})')
                ax4.set_title('Current Gradient Norm')
                ax4.set_ylabel('Gradient Norm')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            except:
                ax4.text(0.5, 0.5, 'Gradient norm\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Convert to base64 for Gradio
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        return {"image": f"data:image/png;base64,{img_str}"}
    
    def evaluate_text(self, input_text: str, task_type: str) -> str:
        """Evaluate a single text using the trained model."""
        try:
            if self.model is None:
                return "❌ Model not initialized. Please initialize and train the model first."
            
            # Create dummy labels for evaluation
            dummy_labels = torch.tensor([1])  # Assuming positive class
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model([input_text], dummy_labels, dummy_labels)
            
            # Extract features and predictions
            seo_features = outputs['seo_features']
            predictions = torch.softmax(seo_features, dim=1)
            
            # Evaluate with specialized metrics
            specialized_metrics = self.model.evaluate_with_specialized_metrics(
                [input_text], dummy_labels, torch.argmax(predictions, dim=1), task_type
            )
            
            # Generate comprehensive report
            report = self.model.generate_comprehensive_report(
                [input_text], dummy_labels, torch.argmax(predictions, dim=1), task_type
            )
            
            return f"📝 Text Evaluation Report\n\n{report}"
        
        except Exception as e:
            return f"❌ Evaluation error: {str(e)}"
    
    def batch_evaluate(self, texts: str, task_type: str) -> Tuple[str, Dict[str, Any]]:
        """Evaluate multiple texts in batch."""
        try:
            if self.model is None:
                return "❌ Model not initialized. Please initialize and train the model first.", {}
            
            # Parse input texts
            text_list = [text.strip() for text in texts.split('\n') if text.strip()]
            
            if not text_list:
                return "❌ No valid texts provided.", {}
            
            # Create dummy labels
            dummy_labels = torch.tensor([1] * len(text_list))
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(text_list, dummy_labels, dummy_labels)
            
            # Extract features and predictions
            seo_features = outputs['seo_features']
            predictions = torch.softmax(seo_features, dim=1)
            predicted_labels = torch.argmax(predictions, dim=1)
            
            # Evaluate with specialized metrics
            specialized_metrics = self.model.evaluate_with_specialized_metrics(
                text_list, dummy_labels, predicted_labels, task_type
            )
            
            # Create results DataFrame
            results_data = []
            for i, text in enumerate(text_list):
                results_data.append({
                    'Text': text[:100] + '...' if len(text) > 100 else text,
                    'Predicted_Label': predicted_labels[i].item(),
                    'Confidence': predictions[i].max().item(),
                    'SEO_Score': specialized_metrics.get('seo_score', 0.0)
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Generate summary
            summary = f"📊 Batch Evaluation Summary\n\n"
            summary += f"• Total Texts: {len(text_list)}\n"
            summary += f"• Average Confidence: {results_df['Confidence'].mean():.4f}\n"
            summary += f"• Average SEO Score: {results_df['SEO_Score'].mean():.4f}\n"
            summary += f"• Task Type: {task_type}\n\n"
            
            # Add specialized metrics
            for metric_name, value in specialized_metrics.items():
                if isinstance(value, (int, float)):
                    summary += f"• {metric_name.replace('_', ' ').title()}: {value:.4f}\n"
            
            return summary, {"dataframe": results_df}
        
        except Exception as e:
            return f"❌ Batch evaluation error: {str(e)}", {}
    
    def save_model(self, save_path: str) -> str:
        """Save the trained model."""
        try:
            if self.trainer is None:
                return "❌ No model to save. Please initialize and train the model first."
            
            self.trainer.save_checkpoint(save_path)
            return f"✅ Model saved successfully to {save_path}"
        
        except Exception as e:
            return f"❌ Error saving model: {str(e)}"
    
    def load_model(self, load_path: str) -> str:
        """Load a trained model."""
        try:
            if self.trainer is None:
                return "❌ Trainer not initialized. Please initialize the model first."
            
            success = self.trainer.load_checkpoint(load_path)
            if success:
                return f"✅ Model loaded successfully from {load_path}"
            else:
                return f"❌ Failed to load model from {load_path}"
        
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface."""
    interface = GradioSEOInterface()
    
    with gr.Blocks(title="Ultra-Optimized SEO Evaluation System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Ultra-Optimized SEO Evaluation System")
        gr.Markdown("### With Gradient Clipping and NaN/Inf Handling")
        
        with gr.Tabs():
            # Model Configuration Tab
            with gr.Tab("🔧 Model Configuration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Configuration")
                        
                        learning_rate = gr.Slider(
                            minimum=1e-6, maximum=1e-2, value=1e-3, step=1e-6,
                            label="Learning Rate", info="Initial learning rate for training"
                        )
                        
                        batch_size = gr.Slider(
                            minimum=1, maximum=32, value=8, step=1,
                            label="Batch Size", info="Training batch size"
                        )
                        
                        max_grad_norm = gr.Slider(
                            minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                            label="Max Gradient Norm", info="Gradient clipping threshold"
                        )
                        
                        patience = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Patience", info="Early stopping patience"
                        )
                        
                        use_amp = gr.Checkbox(
                            value=True, label="Use AMP", 
                            info="Enable Automatic Mixed Precision"
                        )
                        
                        use_lora = gr.Checkbox(
                            value=True, label="Use LoRA", 
                            info="Enable LoRA fine-tuning"
                        )
                        
                        init_button = gr.Button("🚀 Initialize Model", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Model Status")
                        model_status = gr.Textbox(
                            label="Initialization Status", 
                            placeholder="Click 'Initialize Model' to start...",
                            lines=10
                        )
                
                init_button.click(
                    fn=interface.initialize_model,
                    inputs=[gr.State({
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'max_grad_norm': max_grad_norm,
                        'patience': patience,
                        'use_amp': use_amp,
                        'use_lora': use_lora
                    })],
                    outputs=model_status
                )
            
            # Training Tab
            with gr.Tab("🎯 Model Training"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Data")
                        
                        task_type = gr.Dropdown(
                            choices=["classification", "regression", "ranking", "clustering"],
                            value="classification", label="Task Type"
                        )
                        
                        num_samples = gr.Slider(
                            minimum=10, maximum=100, value=20, step=5,
                            label="Number of Samples", info="For sample data generation"
                        )
                        
                        generate_sample_btn = gr.Button("📊 Generate Sample Data")
                        
                        training_texts = gr.Textbox(
                            label="Training Texts (one per line)",
                            placeholder="Enter training texts here...",
                            lines=10
                        )
                        
                        training_labels = gr.Textbox(
                            label="Training Labels (one per line)",
                            placeholder="Enter training labels here...",
                            lines=5
                        )
                        
                        validation_texts = gr.Textbox(
                            label="Validation Texts (one per line)",
                            placeholder="Enter validation texts here...",
                            lines=10
                        )
                        
                        validation_labels = gr.Textbox(
                            label="Validation Labels (one per line)",
                            placeholder="Enter validation labels here...",
                            lines=5
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Training Parameters")
                        
                        epochs = gr.Slider(
                            minimum=1, maximum=50, value=10, step=1,
                            label="Number of Epochs"
                        )
                        
                        train_batch_size = gr.Slider(
                            minimum=1, maximum=32, value=8, step=1,
                            label="Training Batch Size"
                        )
                        
                        train_button = gr.Button("🏋️ Start Training", variant="primary")
                        
                        gr.Markdown("### Training Results")
                        training_report = gr.Textbox(
                            label="Training Report",
                            placeholder="Training results will appear here...",
                            lines=15
                        )
                        
                        training_plots = gr.Image(
                            label="Training Visualizations",
                            type="pil"
                        )
                
                def generate_sample_data(task_type, num_samples):
                    texts, labels = interface.create_sample_data(num_samples, task_type)
                    return '\n'.join(texts), '\n'.join(map(str, labels)), '\n'.join(texts), '\n'.join(map(str, labels))
                
                generate_sample_btn.click(
                    fn=generate_sample_data,
                    inputs=[task_type, num_samples],
                    outputs=[training_texts, training_labels, validation_texts, validation_labels]
                )
                
                train_button.click(
                    fn=interface.train_model,
                    inputs=[training_texts, training_labels, validation_texts, validation_labels, epochs, train_batch_size],
                    outputs=[training_report, training_plots]
                )
            
            # Evaluation Tab
            with gr.Tab("📊 Text Evaluation"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Single Text Evaluation")
                        
                        eval_task_type = gr.Dropdown(
                            choices=["classification", "regression", "ranking", "clustering"],
                            value="classification", label="Evaluation Task Type"
                        )
                        
                        single_text = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to evaluate...",
                            lines=5
                        )
                        
                        eval_button = gr.Button("🔍 Evaluate Text", variant="primary")
                        
                        single_eval_result = gr.Textbox(
                            label="Evaluation Result",
                            placeholder="Evaluation results will appear here...",
                            lines=15
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Batch Evaluation")
                        
                        batch_task_type = gr.Dropdown(
                            choices=["classification", "regression", "ranking", "clustering"],
                            value="classification", label="Batch Task Type"
                        )
                        
                        batch_texts = gr.Textbox(
                            label="Input Texts (one per line)",
                            placeholder="Enter multiple texts to evaluate...",
                            lines=10
                        )
                        
                        batch_eval_button = gr.Button("📋 Batch Evaluate", variant="primary")
                        
                        batch_eval_result = gr.Textbox(
                            label="Batch Evaluation Summary",
                            placeholder="Batch evaluation results will appear here...",
                            lines=10
                        )
                        
                        batch_results_table = gr.Dataframe(
                            label="Detailed Results",
                            headers=["Text", "Predicted_Label", "Confidence", "SEO_Score"]
                        )
                
                eval_button.click(
                    fn=interface.evaluate_text,
                    inputs=[single_text, eval_task_type],
                    outputs=single_eval_result
                )
                
                batch_eval_button.click(
                    fn=interface.batch_evaluate,
                    inputs=[batch_texts, batch_task_type],
                    outputs=[batch_eval_result, batch_results_table]
                )
            
            # Model Management Tab
            with gr.Tab("💾 Model Management"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Save Model")
                        
                        save_path = gr.Textbox(
                            label="Save Path",
                            placeholder="./models/seo_model.pth",
                            value="./models/seo_model.pth"
                        )
                        
                        save_button = gr.Button("💾 Save Model", variant="primary")
                        
                        save_status = gr.Textbox(
                            label="Save Status",
                            placeholder="Save status will appear here...",
                            lines=3
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Load Model")
                        
                        load_path = gr.Textbox(
                            label="Load Path",
                            placeholder="./models/seo_model.pth",
                            value="./models/seo_model.pth"
                        )
                        
                        load_button = gr.Button("📂 Load Model", variant="primary")
                        
                        load_status = gr.Textbox(
                            label="Load Status",
                            placeholder="Load status will appear here...",
                            lines=3
                        )
                
                save_button.click(
                    fn=interface.save_model,
                    inputs=[save_path],
                    outputs=save_status
                )
                
                load_button.click(
                    fn=interface.load_model,
                    inputs=[load_path],
                    outputs=load_status
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        ### 🛡️ Features Included:
        - **Gradient Clipping**: Prevents exploding gradients
        - **NaN/Inf Handling**: Automatic detection and recovery
        - **Training Stability Monitoring**: Real-time health checks
        - **Automatic Corrective Actions**: Adaptive learning rate and weight decay
        - **Comprehensive Logging**: TensorBoard integration
        - **Checkpoint Safety**: Safe model saving/loading
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
