from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import os
import sys

from typing import Any, List, Dict, Optional
import asyncio
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for demo state
current_model = None
training_history = []
evaluation_results =[object Object]

class GradioDemoManager:
    nager for all Gradio demo interfaces."   
    def __init__(self) -> Any:
        self.model = None
        self.config = None
        self.training_history =   self.evaluation_results =[object Object]       self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, List[str]]:
      Create sample data for demos.
        return {
     texts
          A beautiful sunset over the ocean,
        Delicious homemade pizza with fresh ingredients,
               Adventure hiking in the mountains,
               Cozy coffee shop with vintage decor,
                Modern minimalist apartment interior,
              Colorful street art in the city,
       Peaceful meditation in nature,
               Exciting concert with amazing lights,
               Fresh flowers in a garden,
       Creative workspace with inspiring design"
            ],
        captions
              🌅 Sunset vibes are everything! #sunset #ocean #peaceful,
             🍕 Pizza goals achieved! Homemade perfection #pizza #foodie #delicious,
           🏔️ Mountain adventures await! #hiking #adventure #nature,
                ☕ Coffee and good vibes only #coffee #vintage #cozy,
            🏠 Minimalist living goals #minimalist #interior #design,
            🎨 Street art that speaks to the soul #streetart #colorful #urban,
            🧘‍♀️ Finding peace in nature #meditation #peaceful #mindfulness,
         🎵 Concert memories that last forever #concert #music #lights,
                🌸 Spring flowers bring joy #flowers #garden #beautiful,
                💡 Creative space, creative mind #workspace #inspiration #design"
            ]
        }
    
    def create_caption_generation_demo(self) -> gr.Interface:
       caption generation demo interface.""      
        def generate_caption(
            input_text: str,
            max_length: int,
            temperature: float,
            top_p: float,
            style: str,
            include_hashtags: bool,
            include_emojis: bool
        ) -> Tuple[str, Dict[str, Any]]:
     Generate Instagram caption with specified parameters."
            try:
                if self.model is None:
                    return "Please load a model first!", {}
                
                # Preprocess input
                processed_text = self._preprocess_input(input_text, style)
                
                # Generate caption (simulated for demo)
                start_time = time.time()
                raw_caption = self._simulate_caption_generation(processed_text, max_length, temperature)
                generation_time = time.time() - start_time
                
                # Post-process caption
                final_caption = self._postprocess_caption(
                    raw_caption, 
                    include_hashtags, 
                    include_emojis, 
                    style
                )
                
                # Calculate metrics
                metrics = self._calculate_caption_metrics(final_caption)
                metrics[generation_time'] = generation_time
                
                return final_caption, metrics
                
            except Exception as e:
                logger.error(f"Caption generation error: {e})            return fError: {str(e)}", {}
        
        def load_model(model_path: str) -> str:
           ad a trained model."
            try:
                global current_model
                # Simulate model loading for demo
                self.model = "demo_model"
                current_model = self.model
                return f"Model loaded successfully from {model_path}"
            except Exception as e:
                return f"Error loading model: {str(e)}        
        # Create the interface
        interface = gr.Interface(
            fn=generate_caption,
            inputs=[
                gr.Textbox(
                    label="Input Text/Image Description",
                    placeholder=Describe what you want to caption...",
                    lines=3
                ),
                gr.Slider(
                    minimum=10, 
                    maximum=200, 
                    value=100, 
                    step=5,
                    label="Max Length"
                ),
                gr.Slider(
                    minimum=0.1, 
                    maximum=2.0, 
                    value=0.7, 
                    step=0.1,
                    label="Temperature (Creativity)"
                ),
                gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.9, 
                    step=0.05,
                    label="Top-p (Nucleus Sampling)"
                ),
                gr.Dropdown(
                    choices=["Casual,Professional", "Creative,Minimalist", "Engaging"],
                    value="Casual",
                    label="Style"
                ),
                gr.Checkbox(label="Include Hashtags", value=True),
                gr.Checkbox(label="Include Emojis", value=True)
            ],
            outputs=[
                gr.Textbox(label=Generated Caption", lines=5),
                gr.JSON(label="Metrics)    ],
            title=Instagram Caption Generator",
            description=Generateengaging Instagram captions with AI",
            examples=[
                ["A beautiful sunset over the ocean", 1000.7.9asual", True, True],
                ["Delicious homemade pizza", 80, 0.80.85,Creative", True, True],
      Modern minimalist apartment,1200.95ofessional", False, False]
            ]
        )
        
        return interface
    
    def create_model_training_demo(self) -> gr.Interface:
     ate model training demo interface.""      
        def train_model(
            epochs: int,
            batch_size: int,
            learning_rate: float,
            use_sample_data: bool,
            custom_texts: str,
            custom_captions: str
        ) -> Tuple[str, Dict[str, Any], Any]:
      rain the Instagram caption model."
            try:
                # Prepare training data
                if use_sample_data:
                    train_texts = self.sample_data["texts"][:8]
                    train_captions = self.sample_data["captions"][:8]
                    val_texts = self.sample_data["texts"][8:]
                    val_captions = self.sample_data["captions"][8:]
                else:
                    # Parse custom data
                    train_texts = [line.strip() for line in custom_texts.split('\n') if line.strip()]
                    train_captions = [line.strip() for line in custom_captions.split('\n') if line.strip()]
                    
                    # Split into train/val
                    split_idx = int(len(train_texts) * 0.8)
                    val_texts = train_texts[split_idx:]
                    val_captions = train_captions[split_idx:]
                    train_texts = train_texts[:split_idx]
                    train_captions = train_captions[:split_idx]
                
                # Simulate training
                training_results = self._simulate_training(epochs, len(train_texts))
                
                # Create training plot
                plot = self._create_training_plot(training_results)
                
                return "Training completed successfully!", training_results, plot
                
            except Exception as e:
                logger.error(fTraining error: {e})            return fTraining error: {str(e)}", [object Object]e
        
        # Create the interface
        interface = gr.Interface(
            fn=train_model,
            inputs=[
                gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Epochs"),
                gr.Slider(minimum=4, maximum=32, value=16ep=4, label="Batch Size"),
                gr.Slider(minimum=1e-5, maximum=1e-3, value=2e-5, step=1e-6bel=Learning Rate),
                gr.Checkbox(label=Use Sample Data", value=True),
                gr.Textbox(
                    label="Custom Training Texts (one per line)",
                    placeholder="Enter your training texts here...",
                    lines=5,
                    visible=False
                ),
                gr.Textbox(
                    label="Custom Training Captions (one per line)",
                    placeholder="Enter your training captions here...",
                    lines=5,
                    visible=False
                )
            ],
            outputs=[
                gr.Textbox(label="Training Status"),
                gr.JSON(label="Training Results"),
                gr.Plot(label=Training Progress)    ],
            title="Model Training Demo",
            description="Train the Instagram caption generation model"
        )
        
        return interface
    
    def create_model_evaluation_demo(self) -> gr.Interface:
     e model evaluation demo interface.""      
        def evaluate_model(
            test_texts: str,
            reference_captions: str,
            generated_captions: str
        ) -> Tuple[Dict[str, Any], Any, Any]:
            model performance."
            try:
                # Parse inputs
                texts = [line.strip() for line in test_texts.split('\n') if line.strip()]
                ref_captions = [line.strip() for line in reference_captions.split('\n') if line.strip()]
                gen_captions = [line.strip() for line in generated_captions.split('\n') if line.strip()]
                
                # Calculate metrics
                metrics = self._calculate_evaluation_metrics(ref_captions, gen_captions)
                
                # Create evaluation plots
                metrics_plot = self._create_metrics_plot(metrics)
                comparison_plot = self._create_comparison_plot(texts, ref_captions, gen_captions)
                
                return metrics, metrics_plot, comparison_plot
                
            except Exception as e:
                logger.error(fEvaluation error: {e})            return {}, None, None
        
        # Create the interface
        interface = gr.Interface(
            fn=evaluate_model,
            inputs=[
                gr.Textbox(
                    label="Test Texts (one per line)",
                    placeholder="Enter test texts...",
                    lines=5
                ),
                gr.Textbox(
                    label="Reference Captions (one per line)",
                    placeholder="Enter reference captions...",
                    lines=5
                ),
                gr.Textbox(
                    label="Generated Captions (one per line)",
                    placeholder="Enter generated captions...",
                    lines=5
                )
            ],
            outputs=[
                gr.JSON(label="Evaluation Metrics"),
                gr.Plot(label="Metrics Visualization"),
                gr.Plot(label="Caption Comparison)    ],
            title="Model Evaluation Demo",
            description="Evaluate model performance with various metrics",
            examples=[
                  Beautiful sunset\nDelicious pizza\nMountain adventure",
                   🌅 Sunset vibes! #sunset #beautiful\n🍕 Pizza goals! #foodie #delicious\n🏔️ Adventure time! #hiking #nature",
                 Amazing sunset view #sunset\nPerfect pizza slice #pizza\nEpic mountain hike #adventure"
                ]
            ]
        )
        
        return interface
    
    def create_batch_processing_demo(self) -> gr.Interface:
     e batch processing demo interface.""      
        def process_batch(
            input_file: str,
            output_format: str,
            batch_size: int,
            include_metrics: bool
        ) -> Tuple[str, Dict[str, Any], Any]:
            Process a batch of inputs."
            try:
                # Simulate batch processing
                if input_file is None:
                    # Use sample data
                    texts = self.sample_data["texts]              else:
                    # In real implementation, read from file
                    texts = self.sample_data["texts"]  # Simulate file reading
                
                # Generate captions in batches
                all_captions =              batch_metrics = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_captions = self._generate_batch_captions(batch_texts)
                    all_captions.extend(batch_captions)
                    
                    if include_metrics:
                        batch_metric = self._calculate_batch_metrics(batch_captions)
                        batch_metrics.append(batch_metric)
                
                # Format output
                if output_format == "JSON":
                    output = json.dumps({
                      texts                  captions": all_captions,
                       metrics": batch_metrics if include_metrics else None
                    }, indent=2              else:  # CSV
                    output = self._format_as_csv(texts, all_captions)
                
                # Create batch processing visualization
                plot = self._create_batch_processing_plot(batch_metrics) if include_metrics else None
                
                return output,[object Object]total_processed": len(texts), batch_count": len(batch_metrics)}, plot
                
            except Exception as e:
                logger.error(f"Batch processing error: {e})            return fError: {str(e)}", [object Object]e
        
        # Create the interface
        interface = gr.Interface(
            fn=process_batch,
            inputs=[
                gr.File(label="Input File (optional)"),
                gr.Dropdown(choices=["JSON, CSV"], value="JSON", label=Output Format),
                gr.Slider(minimum=1, maximum=20, value=5ep=1, label="Batch Size"),
                gr.Checkbox(label=Include Metrics", value=True)
            ],
            outputs=[
                gr.Textbox(label=Output", lines=10),
                gr.JSON(label="Processing Summary"),
                gr.Plot(label="Batch Processing Metrics)    ],
            title="Batch Processing Demo",
            description="Process multiple inputs in batches"
        )
        
        return interface
    
    def create_performance_monitoring_demo(self) -> gr.Interface:Create performance monitoring demo interface.""      
        def monitor_performance(
            monitoring_duration: int,
            include_memory: bool,
            include_gpu: bool
        ) -> Tuple[Dict[str, Any], Any, Any]:
           Monitor system and model performance."
            try:
                # Simulate performance monitoring
                performance_data = self._simulate_performance_monitoring(
                    monitoring_duration, include_memory, include_gpu
                )
                
                # Create performance plots
                system_plot = self._create_system_performance_plot(performance_data)
                model_plot = self._create_model_performance_plot(performance_data)
                
                return performance_data, system_plot, model_plot
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e})            return {}, None, None
        
        # Create the interface
        interface = gr.Interface(
            fn=monitor_performance,
            inputs=[
                gr.Slider(minimum=10, maximum=300 value=60, step=10, label="Monitoring Duration (seconds)"),
                gr.Checkbox(label="Monitor Memory Usage", value=True),
                gr.Checkbox(label=Monitor GPU Usage", value=False)
            ],
            outputs=[
                gr.JSON(label="Performance Data"),
                gr.Plot(label="System Performance"),
                gr.Plot(label=Model Performance)    ],
            title="Performance Monitoring Demo",
            description="Monitor system and model performance in real-time"
        )
        
        return interface
    
    # Helper methods for demo functionality
    
    def _preprocess_input(self, text: str, style: str) -> str:
   process input text based on style.""       if style == "Professional":
            return f"Professional caption for: {text}"
        elif style == "Creative":
            return f"Creative and artistic caption for: {text}"
        elif style == "Minimalist":
            return f"Simple and clean caption for: {text}"
        elmatch style:
    case "Engaging":
            return f"Engaging and interactive caption for: {text}   else:  # Casual
            return f"Casual and fun caption for: {text}"
    
    def _simulate_caption_generation(self, text: str, max_length: int, temperature: float) -> str:
 caption generation for demo purposes."" # Simple template-based generation for demo
        templates =
            f"✨ {text} - absolutely amazing! #vibes #life",
            f🎯 {text} - goals achieved! #success #motivation",
            f"🌟 {text} - pure perfection! #beautiful #inspiration",
            f"💫 {text} - living the dream! #lifestyle #goals",
            f"🔥[object Object]text} - on fire! #trending #viral"
        ]
        
        # Select template based on temperature (creativity)
        if temperature > 1.5:
            template_idx = 4  # Most creative
        elif temperature > 1.0:
            template_idx = 3
        elif temperature > 0.5:
            template_idx = 2
        else:
            template_idx = 1  # Most conservative
        
        return templates[template_idx % len(templates)]
    
    def _postprocess_caption(self, caption: str, include_hashtags: bool, include_emojis: bool, style: str) -> str:
     -process generated caption."""
        # Add hashtags if requested
        if include_hashtags:
            hashtags = self._generate_hashtags(caption, style)
            caption += f" {hashtags}        
        # Add emojis if requested
        if include_emojis:
            emojis = self._generate_emojis(caption, style)
            caption = f{emojis} {caption}   
        return caption
    
    def _generate_hashtags(self, caption: str, style: str) -> str:
 Generate relevant hashtags."       hashtag_map = [object Object]           Casual":#vibes", "#life, #fun", "#memes"],
            Professional": ["#business", "#professional", "#success", "#growth"],
            Creative": [#art, #creative,#inspiration", "#design"],
           Minimalist: [minimal", #simple, #clean", "#minimalist"],
         Engaging:["#engagement", "#community,#interaction", #social"]
        }
        return " .join(hashtag_map.get(style, ["#instagram", "#caption"]))
    
    def _generate_emojis(self, caption: str, style: str) -> str:
 Generate relevant emojis."""
        emoji_map = [object Object]         Casual😊",
           Professional":💼",
            Creative🎨",
          Minimalist": ⚪,
           Engaging: 
        }
        return emoji_map.get(style, 📸")
    
    def _calculate_caption_metrics(self, caption: str) -> Dict[str, Any]:
  Calculate metrics for a generated caption.
        return {
         length": len(caption),
           word_count": len(caption.split()),
          hashtag_count": caption.count("#"),
            emoji_count": sum(1or c in caption if ord(c) >127      engagement_score: self._calculate_engagement_score(caption)
        }
    
    def _calculate_engagement_score(self, caption: str) -> float:
  lculate engagement score based on caption characteristics."
        score = 0.0        
        # Length bonus (optimal length around 100 characters)
        length = len(caption)
        if100<= length <= 150:
            score +=00.3        elif 50<= length <= 200:
            score += 0.2        
        # Hashtag bonus (optimal 3-5 hashtags)
        hashtag_count = caption.count("#)     if 3= hashtag_count <= 5:
            score += 0.2
        elif 1= hashtag_count <= 7:
            score += 0.1        
        # Emoji bonus
        emoji_count = sum(1or c in caption if ord(c) >127        if 1 <= emoji_count <= 3:
            score += 0.2        
        # Question bonus
        if "?" in caption:
            score += 0.1        
        # Call-to-action bonus
        cta_words =check,follow", like",comment",share,save"]
        if any(word in caption.lower() for word in cta_words):
            score += 0.1   
        return min(score,10
    
    def _simulate_training(self, epochs: int, data_size: int) -> Dict[str, Any]:
 Simulate training process.""     training_loss = []
        validation_loss = []
        accuracy = []
        
        for epoch in range(epochs):
            # Simulate decreasing loss
            train_loss = max(0.1,2.0- (epoch *00.15+ np.random.normal(0, 0.05))
            val_loss = max(0.1,2.2- (epoch *00.12+ np.random.normal(0, 0.08))
            acc = min(095,0.3+ (epoch *00.06+ np.random.normal(0, 0.2      
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            accuracy.append(acc)
        
        return {
        epochs": list(range(1, epochs + 1       training_loss: training_loss,
            validation_loss: validation_loss,
     accuracy": accuracy,
          final_metrics":[object Object]
               final_train_loss: training_loss[-1],
               final_val_loss: validation_loss[-1],
               final_accuracy": accuracy[-1         }
        }
    
    def _create_training_plot(self, training_results: Dict[str, Any]) -> go.Figure:
         training progress visualization."""
        fig = make_subplots(
            rows=2, cols=1           subplot_titles=("Training and Validation Loss", "Accuracy"),
            vertical_spacing=0.1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(
                x=training_results["epochs"],
                y=training_results[training_loss],
                name="Training Loss,              line=dict(color="blue)    ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=training_results["epochs"],
                y=training_results["validation_loss"],
                name="Validation Loss,              line=dict(color=red)    ),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(
                x=training_results["epochs"],
                y=training_results["accuracy"],
                name="Accuracy,              line=dict(color="green)    ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=Training Progress",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _calculate_evaluation_metrics(self, ref_captions: List[str], gen_captions: Liststr]) -> Dict[str, Any]:
  lculate evaluation metrics."""
        # Simulate metric calculation
        metrics = [object Object]
          bleu_score": np.random.uniform(0.3,00.7,
           rouge_score": np.random.uniform(0.4,00.8,
            meteor_score": np.random.uniform(0.35, 075
          bert_score": np.random.uniform(0.6,00.9      engagement_score": np.random.uniform(0.5, 085,
           creativity_score": np.random.uniform(0.4,00.8),
            relevance_score": np.random.uniform(0.6       }
        
        # Calculate averages
        metrics[average_score] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _create_metrics_plot(self, metrics: Dict[str, Any]) -> go.Figure:
       rics visualization.      metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig = go.Figure(data=            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['blue,red, green',orange', purple', brown', 'pink', 'gray']
            )
        ])
        
        fig.update_layout(
            title="Model Evaluation Metrics",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=0,1       )
        
        return fig
    
    def _create_comparison_plot(self, texts: List[str], ref_captions: List[str], gen_captions: List[str]) -> go.Figure:
       caption comparison visualization.""
        fig = go.Figure()
        
        # Add reference captions
        fig.add_trace(go.Bar(
            name="Reference Captions,
            x=[fText {i+1for i in range(len(texts))],
            y=[len(cap) for cap in ref_captions],
            marker_color=blue      ))
        
        # Add generated captions
        fig.add_trace(go.Bar(
            name="Generated Captions,
            x=[fText {i+1for i in range(len(texts))],
            y=[len(cap) for cap in gen_captions],
            marker_color="red"
        ))
        
        fig.update_layout(
            title="Caption Length Comparison",
            xaxis_title="Input Texts",
            yaxis_title="Caption Length (characters)",
            barmode=group       )
        
        return fig
    
    def _generate_batch_captions(self, texts: List[str]) -> List[str]:
      Generate captions for a batch of texts.""        captions = []
        for text in texts:
            # Simulate caption generation
            caption = fGenerated caption for:[object Object]text} #instagram #caption          captions.append(caption)
        return captions
    
    def _calculate_batch_metrics(self, captions: Liststr]) -> Dict[str, Any]:
  Calculate metrics for a batch of captions.
        return {
           batch_size: len(captions),
          avg_length": np.mean([len(cap) for cap in captions]),
          hashtag_count": sum(cap.count("#) for cap in captions),
           engagement_score: np.mean([self._calculate_engagement_score(cap) for cap in captions])
        }
    
    def _format_as_csv(self, texts: List[str], captions: List[str]) -> str:
       Format data as CSV."""
        csv_lines = ["Text,Caption"]
        for text, caption in zip(texts, captions):
            csv_lines.append(f'{text}","{caption}')
        return "\n.join(csv_lines)
    
    def _create_batch_processing_plot(self, batch_metrics: List[Dictstr, Any]]) -> go.Figure:
     e batch processing visualization.      if not batch_metrics:
            return go.Figure()
        
        batch_numbers = list(range(1, len(batch_metrics) + 1))
        engagement_scores = [m["engagement_score] form in batch_metrics]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=batch_numbers,
                y=engagement_scores,
                mode="lines+markers,              name="Engagement Score"
            )
        ])
        
        fig.update_layout(
            title="Batch Processing Engagement Scores",
            xaxis_title="Batch Number",
            yaxis_title="Average Engagement Score"
        )
        
        return fig
    
    def _simulate_performance_monitoring(self, duration: int, include_memory: bool, include_gpu: bool) -> Dict[str, Any]:
 mulate performance monitoring data."""
        timestamps = list(range(0, duration, 5))
        
        data = [object Object]      timestamps": timestamps,
          cpu_usage": np.random.uniform(20, 80) for _ in timestamps],
           memory_usage": np.random.uniform(30, 90) for _ in timestamps] if include_memory else ,
          gpu_usage": np.random.uniform(10, 95) for _ in timestamps] if include_gpu else],
           inference_time": np.random.uniform(0.12or _ in timestamps],
           throughput": np.random.uniform(10,100or _ in timestamps]
        }
        
        return data
    
    def _create_system_performance_plot(self, performance_data: Dict[str, Any]) -> go.Figure:
      system performance visualization."""
        fig = make_subplots(
            rows=2, cols=1           subplot_titles=("System Resource Usage", Model Performance"),
            vertical_spacing=0.1
        )
        
        # System resources
        fig.add_trace(
            go.Scatter(
                x=performance_data["timestamps"],
                y=performance_data["cpu_usage"],
                name="CPU Usage (%),              line=dict(color="blue)    ),
            row=1, col=1
        )
        
        if performance_data["memory_usage"]:
            fig.add_trace(
                go.Scatter(
                    x=performance_data["timestamps"],
                    y=performance_datamemory_usage                   name="Memory Usage (%)",
                    line=dict(color="red)               ),
                row=1, col=1
            )
        
        if performance_data["gpu_usage"]:
            fig.add_trace(
                go.Scatter(
                    x=performance_data["timestamps"],
                    y=performance_data["gpu_usage"],
                    name="GPU Usage (%)",
                    line=dict(color="green)               ),
                row=1, col=1
            )
        
        # Model performance
        fig.add_trace(
            go.Scatter(
                x=performance_data["timestamps"],
                y=performance_data["inference_time"],
                name=Inference Time (s),              line=dict(color="orange)    ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=performance_data["timestamps"],
                y=performance_data["throughput"],
                name=Throughput (req/s),              line=dict(color="purple)    ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Performance Monitoring",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_model_performance_plot(self, performance_data: Dict[str, Any]) -> go.Figure:
      model performance visualization.""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data["timestamps],     y=performance_data["inference_time"],
            mode="lines+markers",
            name=InferenceTime",
            line=dict(color="blue")
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data["timestamps],     y=performance_data["throughput"],
            mode="lines+markers",
            name="Throughput",
            line=dict(color="red"),
            yaxis="y2"
        ))
        
        fig.update_layout(
            title=Model Performance Metrics",
            xaxis_title=Time (seconds)",
            yaxis=dict(title=Inference Time (s)", side="left"),
            yaxis2=dict(title=Throughput (req/s)", side="right", overlaying="y"),
            showlegend=True
        )
        
        return fig


def create_gradio_demo_app() -> gr.Blocks:
    eate the main Gradio demo application."""
    
    demo_manager = GradioDemoManager()
    
    with gr.Blocks(title="Instagram Captions AI - Interactive Demos)as app:
        gr.Markdown("# 🎨 Instagram Captions AI - Interactive Demos)       gr.Markdown(Explore the power of AI-generated Instagram captions with these interactive demos.")
        
        with gr.Tabs():
            # Caption Generation Tab
            with gr.TabItem("🎯 Caption Generation"):
                gr.Markdown("### Generate engaging Instagram captions with AI)           caption_interface = demo_manager.create_caption_generation_demo()
                caption_interface.render()
            
            # Model Training Tab
            with gr.TabItem("🏋️ Model Training"):
                gr.Markdown("### Train and fine-tune the caption generation model)          training_interface = demo_manager.create_model_training_demo()
                training_interface.render()
            
            # Model Evaluation Tab
            with gr.TabItem("📊 Model Evaluation"):
                gr.Markdown("### Evaluate model performance with various metrics)        evaluation_interface = demo_manager.create_model_evaluation_demo()
                evaluation_interface.render()
            
            # Batch Processing Tab
            with gr.TabItem("⚡ Batch Processing"):
                gr.Markdown("### Process multiple inputs efficiently in batches)             batch_interface = demo_manager.create_batch_processing_demo()
                batch_interface.render()
            
            # Performance Monitoring Tab
            with gr.TabItem("📈 Performance Monitoring"):
                gr.Markdown("### Monitor system and model performance in real-time)        monitoring_interface = demo_manager.create_performance_monitoring_demo()
                monitoring_interface.render()
        
        # Footer
        gr.Markdown("---)       gr.Markdown("### Features)       gr.Markdown(       - **🎯 Caption Generation**: Generate engaging captions with customizable parameters
        - **🏋️ Model Training**: Train and fine-tune the AI model with your data
        - **📊 Model Evaluation**: Comprehensive evaluation with multiple metrics
        - **⚡ Batch Processing**: Efficient processing of multiple inputs
        - **📈 Performance Monitoring**: Real-time system and model monitoring
        - **🎨 Interactive Visualizations**: Beautiful charts and graphs
        - **📱 Mobile-Friendly**: Responsive design for all devices
   )   
    return app


def launch_demo_server(
    host: str = "000,
    port: int = 7860
    share: bool = False,
    debug: bool = False
) -> None:
    Launch the Gradio demo server.""  try:
        app = create_gradio_demo_app()
        
        print(f"🚀 Launching Instagram Captions AI Demo Server...)
        print(f"📍 Local URL: http://{host}:{port}")
        if share:
            print(f"🌐 Public URL: Will be generated when server starts")
        
        app.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Failed to launch demo server: {e})
        print(f❌ Error launching demo server:[object Object]e}")


if __name__ ==__main__":
    # Launch the demo server
    launch_demo_server(share=True) 