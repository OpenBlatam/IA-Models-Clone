from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Any, List, Dict, Optional
import logging
import asyncio
class InstagramCaptionDemo:
    def __init__(self) -> Any:
        self.sample_data = self._create_sample_data()
    
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
    
    def generate_caption(self, input_text: str, max_length: int, temperature: float, 
                        style: str, include_hashtags: bool, include_emojis: bool) -> Tuple[str, Dict[str, Any]]:
 Generate Instagram caption with specified parameters."""
        try:
            # Simulate processing time
            time.sleep(0.5      
            # Generate caption based on style and temperature
            templates =[object Object]
          Casual                   f"✨ {input_text} - absolutely amazing! #vibes #life",
                    f🎯 {input_text} - goals achieved! #success #motivation",
                    f🌟 {input_text} - pure perfection! #beautiful #inspiration"
                ],
                Professional                   f💼 {input_text} - professional excellence #business #success",
                    f📈 {input_text} - achieving milestones #growth #professional",
                    f🎯 {input_text} - strategic success #leadership #excellence"
                ],
            Creative                   f🎨 {input_text} - artistic masterpiece #art #creative",
                    f"✨ {input_text} - creative genius #inspiration #design",
                    f🌟 {input_text} - pure artistry #creative #beautiful"
                ],
              Minimalist                   f"⚪ {input_text} #minimalist #simple",
                    f▫️ {input_text} #clean #minimal",
                    f"○ {input_text} #minimalist #design"
                ],
            Engaging                   f🔥 {input_text} - what do you think? #engagement #community",
                    f💬 {input_text} - share your thoughts! #interaction #social",
                    f🤔 {input_text} - opinions welcome! #discussion #community"
                ]
            }
            
            # Select template based on temperature
            style_templates = templates.get(style, templates["Casual"])
            template_idx = min(int(temperature * 2), len(style_templates) - 1)
            caption = style_templates[template_idx]
            
            # Add hashtags if requested
            if include_hashtags:
                hashtags = self._generate_hashtags(style)
                caption += f" {hashtags}"
            
            # Add emojis if requested
            if include_emojis:
                emojis = self._generate_emojis(style)
                caption = f{emojis} {caption}"
            
            # Calculate metrics
            metrics = self._calculate_caption_metrics(caption)
            
            return caption, metrics
            
        except Exception as e:
            return fError: {str(e)},[object Object]
    def _generate_hashtags(self, style: str) -> str:
 Generate relevant hashtags."       hashtag_map = [object Object]          Casual:#vibes #life #fun #memes",
            Professional": "#business #professional #success #growth",
           Creative:#art #creative #inspiration #design",
          Minimalist": "#minimalist #simple #clean #minimal",
        Engaging": "#engagement #community #interaction #social"
        }
        return hashtag_map.get(style, "#instagram #caption")
    
    def _generate_emojis(self, style: str) -> str:
 Generate relevant emojis."""
        emoji_map = [object Object]         Casual😊",
           Professional:💼          Creative🎨",
          Minimalist": ⚪,
           Engaging: 
        }
        return emoji_map.get(style, 📸")
    
    def _calculate_caption_metrics(self, caption: str) -> Dict[str, Any]:
  Calculate metrics for a generated caption.
        return[object Object]
         length": len(caption),
           word_count": len(caption.split()),
          hashtag_count": caption.count("#),
            emoji_count": sum(1or c in caption if ord(c) >127  engagement_score: self._calculate_engagement_score(caption)
        }
    
    def _calculate_engagement_score(self, caption: str) -> float:
  lculate engagement score based on caption characteristics.
        score = 00        
        # Length bonus (optimal length around 100 characters)
        length = len(caption)
        if10<= length <= 150:
            score +=003  elif 50<= length <= 200:
            score += 00.2        
        # Hashtag bonus (optimal 3-5 hashtags)
        hashtag_count = caption.count("#)     if 3= hashtag_count <= 5:
            score += 00.2     elif 1= hashtag_count <= 7:
            score += 00.1        
        # Emoji bonus
        emoji_count = sum(1or c in caption if ord(c) >127        if 1 <= emoji_count <= 3:
            score += 00.2        
        # Question bonus
        if "?" in caption:
            score += 00.1        
        # Call-to-action bonus
        cta_words =check,follow, like,comment",share,save"]
        if any(word in caption.lower() for word in cta_words):
            score += 0.1   
        return min(score,1 
    def simulate_training(self, epochs: int, batch_size: int, learning_rate: float) -> Tuple[str, Dict[str, Any], go.Figure]:
       ate model training."""
        try:
            # Simulate training process
            training_loss = []
            validation_loss = []
            accuracy = []
            
            for epoch in range(epochs):
                # Simulate decreasing loss
                train_loss = max(0.120 (epoch *0.15+ np.random.normal(0, 0.05))
                val_loss = max(0.120.2 (epoch *0.12+ np.random.normal(0, 0.08))
                acc = min(09500.3 (epoch *0.06+ np.random.normal(0, 0.02))
                
                training_loss.append(train_loss)
                validation_loss.append(val_loss)
                accuracy.append(acc)
            
            training_results =[object Object]
            epochs": list(range(1, epochs + 1          training_loss: training_loss,
                validation_loss: validation_loss,
         accuracy": accuracy,
              final_metrics": {
                   final_train_loss: training_loss[-1],
                   final_val_loss: validation_loss[-1],
                   final_accuracy": accuracy[-1                }
            }
            
            # Create training plot
            plot = self._create_training_plot(training_results)
            
            return "Training completed successfully!", training_results, plot
            
        except Exception as e:
            return fTraining error: {str(e)}", {}, go.Figure()
    
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
    
    def evaluate_model(self, test_texts: str, reference_captions: str, 
                      generated_captions: str) -> Tuple[Dict[str, Any], go.Figure, go.Figure]:
        model performance."""
        try:
            # Parse inputs
            texts = [line.strip() for line in test_texts.split('\n') if line.strip()]
            ref_captions = [line.strip() for line in reference_captions.split('\n') if line.strip()]
            gen_captions = [line.strip() for line in generated_captions.split('\n') if line.strip()]
            
            # Calculate metrics (simulated)
            metrics =[object Object]
              bleu_score": np.random.uniform(0.3, 0.7             rouge_score": np.random.uniform(0.4, 0.8),
                meteor_score": np.random.uniform(0.35, 0.75              bert_score": np.random.uniform(0.6, 0.9        engagement_score": np.random.uniform(0.5, 0.85        creativity_score": np.random.uniform(0.4, 0.8),
                relevance_score": np.random.uniform(0.6, 00.9   }
            
            metrics[average_score] = np.mean(list(metrics.values()))
            
            # Create evaluation plots
            metrics_plot = self._create_metrics_plot(metrics)
            comparison_plot = self._create_comparison_plot(texts, ref_captions, gen_captions)
            
            return metrics, metrics_plot, comparison_plot
            
        except Exception as e:
            return {}, go.Figure(), go.Figure()
    
    def _create_metrics_plot(self, metrics: Dict[str, Any]) -> go.Figure:
       rics visualization.      metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig = go.Figure(data=            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=[blue,red, green',orange', purple, brown', 'pink', 'gray']
            )
        ])
        
        fig.update_layout(
            title="Model Evaluation Metrics",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=0,1       )
        
        return fig
    
    def _create_comparison_plot(self, texts: List[str], ref_captions: List[str], 
                               gen_captions: List[str]) -> go.Figure:
       caption comparison visualization."
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
    
    def batch_process(self, batch_size: int, include_metrics: bool) -> Tuple[str, Dict[str, Any], go.Figure]:
        Process a batch of inputs."""
        try:
            texts = self.sample_data["texts]
            all_captions = []
            batch_metrics = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_captions = [fGenerated caption for:[object Object]text} #instagram #caption" for text in batch_texts]
                all_captions.extend(batch_captions)
                
                if include_metrics:
                    batch_metric = {
                       batch_size": len(batch_captions),
                      avg_length": np.mean([len(cap) for cap in batch_captions]),
                      hashtag_count": sum(cap.count("#") for cap in batch_captions),
                       engagement_score: np.mean([self._calculate_engagement_score(cap) for cap in batch_captions])
                    }
                    batch_metrics.append(batch_metric)
            
            # Format output
            output = json.dumps({
              textss,
               captions": all_captions,
               metrics": batch_metrics if include_metrics else None
            }, indent=2)
            
            # Create batch processing visualization
            plot = self._create_batch_processing_plot(batch_metrics) if include_metrics else go.Figure()
            
            return output,[object Object]total_processed": len(texts), batch_count": len(batch_metrics)}, plot
            
        except Exception as e:
            return fError: {str(e)}", {}, go.Figure()
    
    def _create_batch_processing_plot(self, batch_metrics: List[Dictstr, Any]]) -> go.Figure:
     e batch processing visualization.      if not batch_metrics:
            return go.Figure()
        
        batch_numbers = list(range(1, len(batch_metrics) + 1))
        engagement_scores = [mengagement_score] form in batch_metrics]
        
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


def create_demo_app():
    
    """create_demo_app function."""
eate the main Gradio demo application."""
    demo = InstagramCaptionDemo()
    
    with gr.Blocks(title="Instagram Captions AI - Interactive Demo)as app:
        gr.Markdown("# 🎨 Instagram Captions AI - Interactive Demo)       gr.Markdown(Explore the power of AI-generated Instagram captions with this interactive demo.")
        
        with gr.Tabs():
            # Caption Generation Tab
            with gr.TabItem("🎯 Caption Generation"):
                gr.Markdown("### Generate engaging Instagram captions with AI")
                
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Input Text/Image Description",
                            placeholder=Describe what you want to caption...",
                            lines=3
                        )
                        max_length = gr.Slider(
                            minimum=10, maximum=200, value=100, step=5,
                            label="Max Length"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature (Creativity)"
                        )
                        style = gr.Dropdown(
                            choices=["Casual,Professional",Creative,Minimalist", "Engaging"],
                            value="Casual",
                            label="Style"
                        )
                        include_hashtags = gr.Checkbox(label="Include Hashtags", value=True)
                        include_emojis = gr.Checkbox(label="Include Emojis", value=True)
                        generate_btn = gr.Button("Generate Caption", variant="primary")
                    
                    with gr.Column():
                        output_caption = gr.Textbox(
                            label=Generated Caption",
                            lines=5,
                            interactive=False
                        )
                        output_metrics = gr.JSON(label="Metrics")
                
                generate_btn.click(
                    fn=demo.generate_caption,
                    inputs=[input_text, max_length, temperature, style, include_hashtags, include_emojis],
                    outputs=[output_caption, output_metrics]
                )
            
            # Model Training Tab
            with gr.TabItem("🏋️ Model Training"):
                gr.Markdown("### Train and fine-tune the caption generation model")
                
                with gr.Row():
                    with gr.Column():
                        epochs = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Epochs")
                        batch_size = gr.Slider(minimum=4, maximum=32, value=16ep=4, label="Batch Size")
                        learning_rate = gr.Slider(minimum=1e-5, maximum=1e-3, value=2-5=1e-6bel="Learning Rate")
                        train_btn = gr.Button("Start Training", variant="primary")
                    
                    with gr.Column():
                        training_status = gr.Textbox(label=Training Status", interactive=False)
                        training_results = gr.JSON(label="Training Results")
                
                training_plot = gr.Plot(label=Training Progress")
                
                train_btn.click(
                    fn=demo.simulate_training,
                    inputs=[epochs, batch_size, learning_rate],
                    outputs=[training_status, training_results, training_plot]
                )
            
            # Model Evaluation Tab
            with gr.TabItem("📊 Model Evaluation"):
                gr.Markdown("### Evaluate model performance with various metrics")
                
                with gr.Row():
                    with gr.Column():
                        test_texts = gr.Textbox(
                            label="Test Texts (one per line)",
                            placeholder="Enter test texts...",
                            lines=5
                        )
                        reference_captions = gr.Textbox(
                            label="Reference Captions (one per line)",
                            placeholder="Enter reference captions...",
                            lines=5
                        )
                        generated_captions = gr.Textbox(
                            label="Generated Captions (one per line)",
                            placeholder="Enter generated captions...",
                            lines=5
                        )
                        evaluate_btn = gr.Button("Evaluate Model", variant="primary")
                    
                    with gr.Column():
                        evaluation_metrics = gr.JSON(label="Evaluation Metrics")
                
                with gr.Row():
                    metrics_plot = gr.Plot(label="Metrics Visualization")
                    comparison_plot = gr.Plot(label="Caption Comparison")
                
                evaluate_btn.click(
                    fn=demo.evaluate_model,
                    inputs=[test_texts, reference_captions, generated_captions],
                    outputs=[evaluation_metrics, metrics_plot, comparison_plot]
                )
            
            # Batch Processing Tab
            with gr.TabItem("⚡ Batch Processing"):
                gr.Markdown("### Process multiple inputs efficiently in batches")
                
                with gr.Row():
                    with gr.Column():
                        batch_size_input = gr.Slider(minimum=1, maximum=20, value=5ep=1, label="Batch Size")
                        include_metrics_check = gr.Checkbox(label=Include Metrics", value=True)
                        process_btn = gr.Button("Process Batch", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.Textbox(label=Output", lines=10, interactive=False)
                        batch_summary = gr.JSON(label="Processing Summary")
                
                batch_plot = gr.Plot(label="Batch Processing Metrics")
                
                process_btn.click(
                    fn=demo.batch_process,
                    inputs=[batch_size_input, include_metrics_check],
                    outputs=[batch_output, batch_summary, batch_plot]
                )
        
        # Footer
        gr.Markdown("---)       gr.Markdown("### Features)       gr.Markdown(       - **🎯 Caption Generation**: Generate engaging captions with customizable parameters
        - **🏋️ Model Training**: Train and fine-tune the AI model with your data
        - **📊 Model Evaluation**: Comprehensive evaluation with multiple metrics
        - **⚡ Batch Processing**: Efficient processing of multiple inputs
        - **🎨 Interactive Visualizations**: Beautiful charts and graphs
        - **📱 Mobile-Friendly**: Responsive design for all devices
   )   
    return app


def launch_demo_server(host: str = 00.0ort: int =7860hare: bool = False):
    
    """launch_demo_server function."""
Launch the Gradio demo server.""  try:
        app = create_demo_app()
        
        print(f"🚀 Launching Instagram Captions AI Demo Server...)
        print(f"📍 Local URL: http://{host}:{port}")
        if share:
            print(f"🌐 Public URL: Will be generated when server starts")
        
        app.launch(
            server_name=host,
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f❌ Error launching demo server:[object Object]e}")


if __name__ ==__main__":
    # Launch the demo server
    launch_demo_server(share=True) 