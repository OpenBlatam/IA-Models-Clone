from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import gradio as gr
import asyncio
import aiohttp
import json
import time
import torch
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
"""
Instagram Captions API v8.0 - Interactive Gradio Demo

Interactive demo showcasing deep learning and transformer capabilities
with real-time AI generation and analysis.
"""



class GradioAIDemo:
    """Interactive Gradio demo for AI-powered caption generation."""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        
    """__init__ function."""
self.api_url = api_url
        self.session_stats = {
            "total_generations": 0,
            "avg_quality": 0,
            "avg_processing_time": 0,
            "model_usage": {}
        }
    
    async def generate_ai_caption(self, content_description: str, style: str, 
                                 hashtag_count: int, model_size: str,
                                 analyze_semantics: bool, predict_engagement: bool) -> Tuple[str, str, Dict, str]:
        """
        Generate AI caption with comprehensive analysis.
        
        Returns:
            Tuple of (caption, hashtags, analysis, debug_info)
        """
        if not content_description.strip():
            return "❌ Please provide a content description", "", {}, "No input provided"
        
        # Prepare request payload
        payload = {
            "content_description": content_description,
            "style": style.lower(),
            "hashtag_count": hashtag_count,
            "model_size": model_size.lower(),
            "analyze_semantics": analyze_semantics,
            "predict_engagement": predict_engagement,
            "client_id": f"gradio-demo-{int(time.time())}"
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/v8/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update session stats
                        self._update_stats(result, time.time() - start_time, model_size)
                        
                        # Format response
                        caption = result["caption"]
                        hashtags = " ".join(result["hashtags"])
                        
                        # Create analysis summary
                        analysis = self._format_analysis(result)
                        
                        # Debug information
                        debug_info = self._format_debug_info(result, time.time() - start_time)
                        
                        return caption, hashtags, analysis, debug_info
                    
                    else:
                        error_text = await response.text()
                        return f"❌ API Error ({response.status})", "", {}, f"Error: {error_text}"
        
        except asyncio.TimeoutError:
            return "❌ Request timeout", "", {}, "Request took too long (>60s)"
        except Exception as e:
            return f"❌ Connection error: {str(e)}", "", {}, f"Exception: {type(e).__name__}: {str(e)}"
    
    def _update_stats(self, result: Dict, processing_time: float, model_size: str):
        """Update session statistics."""
        self.session_stats["total_generations"] += 1
        
        # Update averages
        total = self.session_stats["total_generations"]
        self.session_stats["avg_quality"] = (
            (self.session_stats["avg_quality"] * (total - 1) + result["quality_score"]) / total
        )
        self.session_stats["avg_processing_time"] = (
            (self.session_stats["avg_processing_time"] * (total - 1) + processing_time) / total
        )
        
        # Track model usage
        if model_size not in self.session_stats["model_usage"]:
            self.session_stats["model_usage"][model_size] = 0
        self.session_stats["model_usage"][model_size] += 1
    
    def _format_analysis(self, result: Dict) -> Dict[str, Any]:
        """Format analysis results for display."""
        analysis = {
            "🎯 Quality Score": f"{result['quality_score']:.1f}/100",
            "🧠 Model": result["model_metadata"]["model_size"].title(),
            "⚡ Processing Time": f"{result['processing_time_seconds']:.3f}s"
        }
        
        # Add semantic analysis if available
        semantic = result.get("semantic_analysis", {})
        if "content_similarity" in semantic and semantic["content_similarity"] is not None:
            analysis["🔗 Content Similarity"] = f"{semantic['content_similarity']:.3f}"
        
        # Add engagement analysis
        if "engagement_analysis" in semantic and semantic["engagement_analysis"]:
            engagement = semantic["engagement_analysis"]
            analysis["📈 Engagement Potential"] = f"{engagement.get('overall_engagement', 0):.3f}"
            
            # Top engagement factors
            factors = {k: v for k, v in engagement.items() if k != 'overall_engagement'}
            if factors:
                top_factor = max(factors.items(), key=lambda x: x[1])
                analysis["🏆 Top Factor"] = f"{top_factor[0].replace('_', ' ').title()}: {top_factor[1]:.3f}"
        
        # Add GPU info if available
        if result.get("gpu_memory_used_mb"):
            analysis["🖥️ GPU Memory"] = f"{result['gpu_memory_used_mb']:.1f} MB"
        
        return analysis
    
    def _format_debug_info(self, result: Dict, total_time: float) -> str:
        """Format debug information."""
        debug_lines = [
            f"🔍 Debug Information:",
            f"├── Request ID: {result['request_id']}",
            f"├── API Version: {result['api_version']}",
            f"├── Total Time: {total_time:.3f}s",
            f"├── AI Processing: {result['processing_time_seconds']:.3f}s",
            f"└── Model Metadata: {result['model_metadata']}"
        ]
        
        if result.get("semantic_analysis"):
            debug_lines.extend([
                f"",
                f"🧠 Semantic Analysis:",
                f"└── {result['semantic_analysis']}"
            ])
        
        return "\n".join(debug_lines)
    
    async async def get_api_health(self) -> str:
        """Get API health status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/ai/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        health = await response.json()
                        
                        status = "🟢 Healthy" if health["status"] == "healthy" else "🟡 Degraded"
                        gpu_status = "🖥️ GPU Available" if health.get("gpu_info") else "💻 CPU Only"
                        models = len(health["ai_services"]["available_models"])
                        
                        return f"{status} | {gpu_status} | {models} Models Loaded"
                    else:
                        return f"🔴 API Error ({response.status})"
        except Exception as e:
            return f"🔴 Connection Failed: {str(e)}"
    
    def get_session_stats(self) -> str:
        """Get current session statistics."""
        if self.session_stats["total_generations"] == 0:
            return "📊 No generations yet in this session"
        
        stats_lines = [
            f"📊 Session Statistics:",
            f"├── Total Generations: {self.session_stats['total_generations']}",
            f"├── Avg Quality: {self.session_stats['avg_quality']:.1f}/100",
            f"├── Avg Processing: {self.session_stats['avg_processing_time']:.3f}s",
            f"└── Model Usage: {dict(self.session_stats['model_usage'])}"
        ]
        
        return "\n".join(stats_lines)
    
    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-markdown {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        .metric-box {
            background: linear-gradient(45deg, #1e3a8a, #3b82f6);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        """
        
        # Wrapper function for async call
        def generate_wrapper(content, style, hashtag_count, model_size, semantics, engagement) -> Any:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.generate_ai_caption(content, style, hashtag_count, model_size, semantics, engagement)
                )
                return result
            finally:
                loop.close()
        
        def health_wrapper():
            
    """health_wrapper function."""
loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.get_api_health())
                return result
            finally:
                loop.close()
        
        # Create interface
        with gr.Blocks(
            title="Instagram Captions AI v8.0",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as demo:
            
            gr.Markdown("""
            # 🧠 Instagram Captions API v8.0 - Deep Learning Demo
            
            Experience the power of **real transformer models** and **advanced AI** for Instagram caption generation.
            This demo showcases state-of-the-art deep learning techniques including:
            - 🤖 **Real transformer models** (GPT-2, DialoGPT)
            - 🧠 **Semantic analysis** with sentence transformers
            - 📊 **Quality prediction** using neural networks
            - 📈 **Engagement analysis** with deep learning
            - ⚡ **GPU acceleration** for faster processing
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 📝 Input Configuration")
                    
                    content_input = gr.Textbox(
                        label="Content Description",
                        placeholder="Describe your content in detail (e.g., 'Beautiful sunset at the beach with golden reflections on the water')",
                        lines=3,
                        max_lines=5
                    )
                    
                    with gr.Row():
                        style_input = gr.Dropdown(
                            choices=["Casual", "Professional", "Playful", "Inspirational", "Educational", "Promotional"],
                            value="Casual",
                            label="Style"
                        )
                        
                        model_size_input = gr.Dropdown(
                            choices=["Tiny", "Small", "Medium"],
                            value="Small",
                            label="Model Size"
                        )
                    
                    with gr.Row():
                        hashtag_count = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=15,
                            step=1,
                            label="Hashtag Count"
                        )
                    
                    with gr.Row():
                        analyze_semantics = gr.Checkbox(
                            value=True,
                            label="🧠 Semantic Analysis"
                        )
                        
                        predict_engagement = gr.Checkbox(
                            value=True,
                            label="📈 Engagement Prediction"
                        )
                    
                    generate_btn = gr.Button(
                        "🚀 Generate AI Caption",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=3):
                    gr.Markdown("## 🎯 AI Generation Results")
                    
                    caption_output = gr.Textbox(
                        label="Generated Caption",
                        lines=4,
                        max_lines=8
                    )
                    
                    hashtags_output = gr.Textbox(
                        label="AI-Generated Hashtags",
                        lines=2
                    )
                    
                    analysis_output = gr.JSON(
                        label="📊 AI Analysis & Metrics"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 🔍 Debug Information")
                    debug_output = gr.Code(
                        label="Technical Details",
                        language="yaml"
                    )
                
                with gr.Column():
                    gr.Markdown("## 📊 Session & Health")
                    
                    health_btn = gr.Button("🏥 Check API Health")
                    health_output = gr.Textbox(label="API Status")
                    
                    stats_btn = gr.Button("📊 Session Stats")
                    stats_output = gr.Code(label="Statistics", language="yaml")
            
            # Event handlers
            generate_btn.click(
                fn=generate_wrapper,
                inputs=[content_input, style_input, hashtag_count, model_size_input, analyze_semantics, predict_engagement],
                outputs=[caption_output, hashtags_output, analysis_output, debug_output]
            )
            
            health_btn.click(
                fn=health_wrapper,
                outputs=health_output
            )
            
            stats_btn.click(
                fn=self.get_session_stats,
                outputs=stats_output
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["Beautiful sunset at the beach with golden reflections on the water", "Inspirational", 15, "Small", True, True],
                    ["Professional headshot for LinkedIn profile", "Professional", 10, "Small", True, False],
                    ["Delicious homemade pasta with fresh herbs", "Casual", 20, "Small", True, True],
                    ["Team building event at the office", "Professional", 12, "Medium", True, True],
                    ["Cute puppy playing in the park", "Playful", 18, "Small", True, True]
                ],
                inputs=[content_input, style_input, hashtag_count, model_size_input, analyze_semantics, predict_engagement]
            )
            
            gr.Markdown("""
            ---
            ### 🔬 Technical Notes:
            - **Model Sizes**: Tiny (fastest), Small (balanced), Medium (highest quality)
            - **Semantic Analysis**: Uses sentence transformers for content similarity
            - **Engagement Prediction**: Deep learning analysis of engagement potential
            - **GPU Acceleration**: Automatically used when available
            - **Processing Time**: Includes model inference and analysis
            """)
        
        return demo


def main():
    """Launch the Gradio demo."""
    print("🚀 Starting Instagram Captions AI v8.0 - Gradio Demo")
    print("="*60)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"🖥️ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 Using CPU (GPU not available)")
    
    print("="*60)
    
    # Create demo
    demo_app = GradioAIDemo()
    interface = demo_app.create_interface()
    
    # Launch interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        inbrowser=True,
        show_error=True
    )


match __name__:
    case "__main__":
    main() 