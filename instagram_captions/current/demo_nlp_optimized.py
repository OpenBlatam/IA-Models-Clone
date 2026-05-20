"""
Optimized NLP System Demo v15.0 - Comprehensive Demonstration
Advanced deep learning with PyTorch, Transformers, and optimization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    pipeline, TrainingArguments, Trainer
)
import numpy as np
import gradio as gr
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Import our optimized NLP system
from nlp_system_optimized import (
    OptimizedNLPSystem, NLPSystemConfig, 
    NLPAnalyzer, AdvancedNLPTrainer, CustomNLPDataset
)

# GPU optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPDemoSystem:
    def __init__(self):
        self.nlp_system = None
        self.analyzer = None
        self.trainer = None
        self.setup_system()
        
    def setup_system(self):
        """Initialize the NLP system with optimized configuration"""
        try:
            config = NLPSystemConfig(
                model_name="gpt2",
                max_length=512,
                batch_size=8,
                learning_rate=2e-5,
                num_epochs=3,
                warmup_steps=500,
                weight_decay=0.01,
                gradient_accumulation_steps=4,
                fp16=True,
                mixed_precision=True
            )
            
            self.nlp_system = OptimizedNLPSystem(config)
            self.nlp_system.load_model()
            self.nlp_system.setup_training()
            
            # Setup analyzer
            self.analyzer = NLPAnalyzer(self.nlp_system)
            self.analyzer.setup_analyzers()
            
            # Setup trainer
            self.trainer = AdvancedNLPTrainer(self.nlp_system)
            
            logger.info("NLP Demo System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up NLP system: {e}")
            raise
    
    def generate_text_demo(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> Dict:
        """Generate text with performance metrics"""
        try:
            start_time = time.time()
            
            # Update generation parameters
            self.nlp_system.model.config.temperature = temperature
            
            generated_text = self.nlp_system.generate_text(prompt, max_length)
            
            generation_time = time.time() - start_time
            
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "generation_time": generation_time,
                "max_length": max_length,
                "temperature": temperature,
                "device": str(self.nlp_system.device)
            }
            
        except Exception as e:
            logger.error(f"Error in text generation demo: {e}")
            return {"error": str(e)}
    
    def batch_generate_demo(self, prompts: List[str], max_length: int = 100) -> Dict:
        """Batch text generation demo"""
        try:
            start_time = time.time()
            
            generated_texts = self.nlp_system.batch_generate(prompts, max_length)
            
            generation_time = time.time() - start_time
            
            return {
                "prompts": prompts,
                "generated_texts": generated_texts,
                "generation_time": generation_time,
                "avg_time_per_prompt": generation_time / len(prompts),
                "device": str(self.nlp_system.device)
            }
            
        except Exception as e:
            logger.error(f"Error in batch generation demo: {e}")
            return {"error": str(e)}
    
    def sentiment_analysis_demo(self, text: str) -> Dict:
        """Sentiment analysis demo"""
        try:
            if not self.analyzer or not self.analyzer.sentiment_analyzer:
                return {"error": "Sentiment analyzer not available"}
            
            result = self.analyzer.analyze_sentiment(text)
            
            if "error" in result:
                return result
            
            # Add visualization data
            sentiment_scores = {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0
            }
            
            sentiment_scores[result["sentiment"].lower()] = result["confidence"]
            
            return {
                "text": text,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "sentiment_scores": sentiment_scores,
                "visualization_data": sentiment_scores
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis demo: {e}")
            return {"error": str(e)}
    
    def text_classification_demo(self, text: str, candidate_labels: List[str]) -> Dict:
        """Text classification demo"""
        try:
            if not self.analyzer or not self.analyzer.text_classifier:
                return {"error": "Text classifier not available"}
            
            result = self.analyzer.classify_text(text, candidate_labels)
            
            if "error" in result:
                return result
            
            return {
                "text": text,
                "predicted_label": result["label"],
                "confidence": result["score"],
                "candidate_labels": candidate_labels,
                "all_scores": result.get("scores", [])
            }
            
        except Exception as e:
            logger.error(f"Error in text classification demo: {e}")
            return {"error": str(e)}
    
    def training_demo(self, train_texts: List[str], val_texts: List[str] = None, epochs: int = 2) -> Dict:
        """Training demo with progress tracking"""
        try:
            if not self.trainer:
                return {"error": "Trainer not available"}
            
            # Update training config
            self.nlp_system.config.num_epochs = epochs
            
            # Start training
            start_time = time.time()
            
            self.trainer.train(train_texts, val_texts)
            
            training_time = time.time() - start_time
            
            return {
                "status": "Training completed successfully",
                "epochs": epochs,
                "training_time": training_time,
                "train_samples": len(train_texts),
                "val_samples": len(val_texts) if val_texts else 0,
                "device": str(self.nlp_system.device)
            }
            
        except Exception as e:
            logger.error(f"Error in training demo: {e}")
            return {"error": str(e)}
    
    def performance_benchmark(self) -> Dict:
        """Performance benchmarking"""
        try:
            # Test prompts
            test_prompts = [
                "The future of artificial intelligence is",
                "Machine learning algorithms can",
                "Deep learning models are",
                "Natural language processing enables",
                "Computer vision systems can"
            ]
            
            # Single generation benchmark
            single_times = []
            for prompt in test_prompts[:3]:
                start_time = time.time()
                self.nlp_system.generate_text(prompt, max_length=50)
                single_times.append(time.time() - start_time)
            
            # Batch generation benchmark
            start_time = time.time()
            self.nlp_system.batch_generate(test_prompts, max_length=50)
            batch_time = time.time() - start_time
            
            return {
                "single_generation_avg_time": np.mean(single_times),
                "single_generation_std_time": np.std(single_times),
                "batch_generation_time": batch_time,
                "batch_efficiency_gain": (sum(single_times) - batch_time) / sum(single_times) * 100,
                "device": str(self.nlp_system.device),
                "model_name": self.nlp_system.config.model_name,
                "fp16_enabled": self.nlp_system.config.fp16,
                "mixed_precision": self.nlp_system.config.mixed_precision
            }
            
        except Exception as e:
            logger.error(f"Error in performance benchmark: {e}")
            return {"error": str(e)}

# Create demo system instance
demo_system = NLPDemoSystem()

# Gradio Interface
def create_gradio_interface():
    """Create comprehensive Gradio interface"""
    
    with gr.Blocks(title="Optimized NLP System Demo v15.0", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Optimized NLP System Demo v15.0")
        gr.Markdown("Advanced deep learning with PyTorch, Transformers, and optimization techniques")
        
        with gr.Tabs():
            # Text Generation Tab
            with gr.Tab("📝 Text Generation"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Input Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3
                        )
                        max_length_slider = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Max Length"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        generate_btn = gr.Button("Generate Text", variant="primary")
                    
                    with gr.Column():
                        generation_output = gr.JSON(label="Generation Results")
                        generation_metrics = gr.JSON(label="Performance Metrics")
                
                generate_btn.click(
                    fn=lambda p, ml, t: demo_system.generate_text_demo(p, ml, t),
                    inputs=[prompt_input, max_length_slider, temperature_slider],
                    outputs=[generation_output, generation_metrics]
                )
            
            # Batch Generation Tab
            with gr.Tab("🔄 Batch Generation"):
                with gr.Row():
                    with gr.Column():
                        batch_prompts = gr.Textbox(
                            label="Batch Prompts (one per line)",
                            placeholder="Enter multiple prompts, one per line...",
                            lines=5
                        )
                        batch_max_length = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Max Length"
                        )
                        batch_generate_btn = gr.Button("Batch Generate", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.JSON(label="Batch Results")
                        batch_metrics = gr.JSON(label="Batch Performance")
                
                batch_generate_btn.click(
                    fn=lambda p, ml: demo_system.batch_generate_demo(p.split('\n'), ml),
                    inputs=[batch_prompts, batch_max_length],
                    outputs=[batch_output, batch_metrics]
                )
            
            # Sentiment Analysis Tab
            with gr.Tab("😊 Sentiment Analysis"):
                with gr.Row():
                    with gr.Column():
                        sentiment_text = gr.Textbox(
                            label="Text for Sentiment Analysis",
                            placeholder="Enter text to analyze sentiment...",
                            lines=3
                        )
                        sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
                    
                    with gr.Column():
                        sentiment_output = gr.JSON(label="Sentiment Results")
                        sentiment_chart = gr.Plot(label="Sentiment Visualization")
                
                sentiment_btn.click(
                    fn=demo_system.sentiment_analysis_demo,
                    inputs=[sentiment_text],
                    outputs=[sentiment_output, sentiment_chart]
                )
            
            # Text Classification Tab
            with gr.Tab("🏷️ Text Classification"):
                with gr.Row():
                    with gr.Column():
                        classification_text = gr.Textbox(
                            label="Text to Classify",
                            placeholder="Enter text to classify...",
                            lines=3
                        )
                        candidate_labels = gr.Textbox(
                            label="Candidate Labels (comma-separated)",
                            placeholder="technology, sports, politics, entertainment",
                            lines=1
                        )
                        classify_btn = gr.Button("Classify Text", variant="primary")
                    
                    with gr.Column():
                        classification_output = gr.JSON(label="Classification Results")
                
                classify_btn.click(
                    fn=lambda t, l: demo_system.text_classification_demo(t, [x.strip() for x in l.split(',')]),
                    inputs=[classification_text, candidate_labels],
                    outputs=[classification_output]
                )
            
            # Training Tab
            with gr.Tab("🎓 Model Training"):
                with gr.Row():
                    with gr.Column():
                        train_texts = gr.Textbox(
                            label="Training Texts (one per line)",
                            placeholder="Enter training texts, one per line...",
                            lines=5
                        )
                        val_texts = gr.Textbox(
                            label="Validation Texts (one per line, optional)",
                            placeholder="Enter validation texts, one per line...",
                            lines=3
                        )
                        epochs_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=2,
                            step=1,
                            label="Training Epochs"
                        )
                        train_btn = gr.Button("Start Training", variant="primary")
                    
                    with gr.Column():
                        training_output = gr.JSON(label="Training Results")
                        training_progress = gr.Textbox(label="Training Progress", lines=3)
                
                train_btn.click(
                    fn=lambda t, v, e: demo_system.training_demo(t.split('\n'), v.split('\n') if v else None, e),
                    inputs=[train_texts, val_texts, epochs_slider],
                    outputs=[training_output, training_progress]
                )
            
            # Performance Benchmark Tab
            with gr.Tab("⚡ Performance Benchmark"):
                benchmark_btn = gr.Button("Run Benchmark", variant="primary")
                benchmark_output = gr.JSON(label="Benchmark Results")
                benchmark_chart = gr.Plot(label="Performance Comparison")
                
                benchmark_btn.click(
                    fn=demo_system.performance_benchmark,
                    inputs=[],
                    outputs=[benchmark_output, benchmark_chart]
                )
            
            # System Info Tab
            with gr.Tab("ℹ️ System Information"):
                system_info = gr.JSON(label="System Configuration")
                
                def get_system_info():
                    return {
                        "device": str(demo_system.nlp_system.device),
                        "model_name": demo_system.nlp_system.config.model_name,
                        "max_length": demo_system.nlp_system.config.max_length,
                        "batch_size": demo_system.nlp_system.config.batch_size,
                        "learning_rate": demo_system.nlp_system.config.learning_rate,
                        "fp16_enabled": demo_system.nlp_system.config.fp16,
                        "mixed_precision": demo_system.nlp_system.config.mixed_precision,
                        "gradient_accumulation_steps": demo_system.nlp_system.config.gradient_accumulation_steps,
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                    }
                
                gr.Button("Refresh System Info").click(
                    fn=get_system_info,
                    inputs=[],
                    outputs=[system_info]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("**Optimized NLP System v15.0** - Built with PyTorch, Transformers, and advanced optimization techniques")
    
    return demo

# Main execution
if __name__ == "__main__":
    try:
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=8151,
            share=False,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Error launching demo: {e}")
        raise





