"""
Advanced Gradio Interface for AI Document Classifier
==================================================

Interactive web interface for the AI Document Classifier with advanced
visualization, real-time processing, and comprehensive model management.

Features:
- Interactive document classification
- Real-time model inference
- Advanced visualization and analytics
- Model comparison and evaluation
- Document generation and editing
- Batch processing interface
- API testing and debugging
- Performance monitoring dashboard
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import io
import base64
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import os

# Import our custom models
from .transformer_models import ModelFactory, ModelConfig, TransformerTrainer
from .diffusion_models import DiffusionModelFactory, DiffusionConfig
from .llm_models import LLMModelFactory, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioInterface:
    """Advanced Gradio interface for AI Document Classifier"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.model_configs = {}
        self.processing_history = []
        self.performance_metrics = {}
        
        # Initialize default configurations
        self._initialize_configs()
        
        # Create interface
        self.interface = self._create_interface()
    
    def _initialize_configs(self):
        """Initialize default model configurations"""
        self.model_configs = {
            "transformer": ModelConfig(
                model_name="bert-base-uncased",
                num_classes=100,
                max_length=512,
                use_lora=True,
                lora_rank=16
            ),
            "diffusion": DiffusionConfig(
                model_id="runwayml/stable-diffusion-v1-5",
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ),
            "llm": LLMConfig(
                model_name="gpt-3.5-turbo",
                max_tokens=1024,
                temperature=0.7
            )
        }
    
    def _create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        with gr.Blocks(
            title="AI Document Classifier - Advanced Interface",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .model-card {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                background: #f9f9f9;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🚀 AI Document Classifier - Advanced Interface
            
            **State-of-the-art document classification with advanced AI models**
            
            Features: Transformer Models | Diffusion Models | LLM Integration | Real-time Analytics
            """)
            
            # Main tabs
            with gr.Tabs():
                # Document Classification Tab
                with gr.Tab("📄 Document Classification"):
                    self._create_classification_tab()
                
                # Document Generation Tab
                with gr.Tab("🎨 Document Generation"):
                    self._create_generation_tab()
                
                # Model Management Tab
                with gr.Tab("🔧 Model Management"):
                    self._create_model_management_tab()
                
                # Analytics Dashboard Tab
                with gr.Tab("📊 Analytics Dashboard"):
                    self._create_analytics_tab()
                
                # Batch Processing Tab
                with gr.Tab("⚡ Batch Processing"):
                    self._create_batch_processing_tab()
                
                # API Testing Tab
                with gr.Tab("🧪 API Testing"):
                    self._create_api_testing_tab()
        
        return interface
    
    def _create_classification_tab(self):
        """Create document classification tab"""
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 📝 Document Input")
                
                document_input = gr.Textbox(
                    label="Document Text",
                    placeholder="Enter your document text here...",
                    lines=10,
                    max_lines=20
                )
                
                document_file = gr.File(
                    label="Upload Document",
                    file_types=[".txt", ".pdf", ".docx", ".md"],
                    file_count="single"
                )
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=["transformer", "llm", "hybrid"],
                        value="transformer",
                        label="Model Type"
                    )
                    
                    model_name = gr.Dropdown(
                        choices=["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
                        value="bert-base-uncased",
                        label="Model"
                    )
                
                with gr.Row():
                    confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    
                    max_length = gr.Slider(
                        minimum=128,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Max Length"
                    )
                
                classify_btn = gr.Button("🔍 Classify Document", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Results section
                gr.Markdown("### 📊 Classification Results")
                
                classification_result = gr.JSON(
                    label="Classification Result",
                    value={}
                )
                
                confidence_score = gr.Number(
                    label="Confidence Score",
                    precision=3
                )
                
                predicted_category = gr.Textbox(
                    label="Predicted Category"
                )
                
                reasoning = gr.Textbox(
                    label="Reasoning",
                    lines=3
                )
                
                # Visualization
                confidence_chart = gr.Plot(
                    label="Confidence Distribution"
                )
        
        # Event handlers
        classify_btn.click(
            fn=self.classify_document,
            inputs=[document_input, document_file, model_type, model_name, confidence_threshold, max_length],
            outputs=[classification_result, confidence_score, predicted_category, reasoning, confidence_chart]
        )
        
        document_file.change(
            fn=self.process_uploaded_file,
            inputs=[document_file],
            outputs=[document_input]
        )
    
    def _create_generation_tab(self):
        """Create document generation tab"""
        with gr.Row():
            with gr.Column(scale=2):
                # Generation input
                gr.Markdown("### 🎨 Document Generation")
                
                generation_type = gr.Dropdown(
                    choices=["text_to_image", "template_generation", "content_creation"],
                    value="template_generation",
                    label="Generation Type"
                )
                
                document_type = gr.Dropdown(
                    choices=["contract", "report", "proposal", "email", "presentation", "invoice"],
                    value="report",
                    label="Document Type"
                )
                
                content_input = gr.Textbox(
                    label="Content Description",
                    placeholder="Describe the content you want to generate...",
                    lines=5
                )
                
                style_choice = gr.Dropdown(
                    choices=["formal", "casual", "technical", "academic", "creative"],
                    value="formal",
                    label="Writing Style"
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Creativity (Temperature)"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=50,
                        label="Max Tokens"
                    )
                
                generate_btn = gr.Button("✨ Generate Document", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Generation results
                gr.Markdown("### 📄 Generated Document")
                
                generated_content = gr.Textbox(
                    label="Generated Content",
                    lines=15,
                    max_lines=25
                )
                
                generation_metrics = gr.JSON(
                    label="Generation Metrics"
                )
                
                # Image generation (if applicable)
                generated_image = gr.Image(
                    label="Generated Image",
                    type="pil"
                )
        
        # Event handlers
        generate_btn.click(
            fn=self.generate_document,
            inputs=[generation_type, document_type, content_input, style_choice, temperature, max_tokens],
            outputs=[generated_content, generation_metrics, generated_image]
        )
    
    def _create_model_management_tab(self):
        """Create model management tab"""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Model Management")
                
                # Model selection
                with gr.Row():
                    model_category = gr.Dropdown(
                        choices=["transformer", "diffusion", "llm"],
                        value="transformer",
                        label="Model Category"
                    )
                    
                    load_model_btn = gr.Button("📥 Load Model", variant="secondary")
                
                # Model configuration
                model_config = gr.JSON(
                    label="Model Configuration",
                    value=self.model_configs["transformer"].__dict__
                )
                
                # Model performance
                model_performance = gr.DataFrame(
                    label="Model Performance",
                    headers=["Metric", "Value", "Timestamp"],
                    datatype=["str", "number", "str"]
                )
                
                # Model comparison
                compare_models_btn = gr.Button("📊 Compare Models", variant="secondary")
                
                comparison_chart = gr.Plot(
                    label="Model Comparison"
                )
            
            with gr.Column():
                gr.Markdown("### 📈 Model Training")
                
                # Training configuration
                training_data = gr.File(
                    label="Training Data",
                    file_types=[".csv", ".json", ".txt"]
                )
                
                with gr.Row():
                    epochs = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=3,
                        step=1,
                        label="Epochs"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-2,
                        value=2e-5,
                        step=1e-6,
                        label="Learning Rate"
                    )
                
                train_btn = gr.Button("🚀 Start Training", variant="primary")
                
                training_progress = gr.Progress()
                
                training_logs = gr.Textbox(
                    label="Training Logs",
                    lines=10,
                    max_lines=20
                )
        
        # Event handlers
        load_model_btn.click(
            fn=self.load_model,
            inputs=[model_category],
            outputs=[model_config, model_performance]
        )
        
        compare_models_btn.click(
            fn=self.compare_models,
            inputs=[],
            outputs=[comparison_chart]
        )
        
        train_btn.click(
            fn=self.train_model,
            inputs=[training_data, epochs, learning_rate],
            outputs=[training_logs],
            show_progress=True
        )
    
    def _create_analytics_tab(self):
        """Create analytics dashboard tab"""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Performance Analytics")
                
                # Time range selection
                time_range = gr.Dropdown(
                    choices=["1 hour", "24 hours", "7 days", "30 days"],
                    value="24 hours",
                    label="Time Range"
                )
                
                # Metrics selection
                metrics_selection = gr.CheckboxGroup(
                    choices=["accuracy", "response_time", "throughput", "error_rate", "confidence"],
                    value=["accuracy", "response_time"],
                    label="Metrics to Display"
                )
                
                update_analytics_btn = gr.Button("🔄 Update Analytics", variant="secondary")
            
            with gr.Column():
                gr.Markdown("### 📈 Real-time Metrics")
                
                # Key metrics cards
                with gr.Row():
                    accuracy_card = gr.HTML(
                        value="<div class='metric-card'><h3>95.8%</h3><p>Accuracy</p></div>"
                    )
                    
                    response_time_card = gr.HTML(
                        value="<div class='metric-card'><h3>150ms</h3><p>Response Time</p></div>"
                    )
                
                with gr.Row():
                    throughput_card = gr.HTML(
                        value="<div class='metric-card'><h3>1000/min</h3><p>Throughput</p></div>"
                    )
                    
                    error_rate_card = gr.HTML(
                        value="<div class='metric-card'><h3>0.1%</h3><p>Error Rate</p></div>"
                    )
        
        # Charts section
        with gr.Row():
            with gr.Column():
                performance_chart = gr.Plot(
                    label="Performance Over Time"
                )
            
            with gr.Column():
                distribution_chart = gr.Plot(
                    label="Confidence Distribution"
                )
        
        # Event handlers
        update_analytics_btn.click(
            fn=self.update_analytics,
            inputs=[time_range, metrics_selection],
            outputs=[performance_chart, distribution_chart, accuracy_card, response_time_card, throughput_card, error_rate_card]
        )
    
    def _create_batch_processing_tab(self):
        """Create batch processing tab"""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚡ Batch Processing")
                
                # File upload
                batch_files = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".txt", ".pdf", ".docx", ".md"]
                )
                
                # Processing options
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1,
                        label="Batch Size"
                    )
                    
                    processing_mode = gr.Dropdown(
                        choices=["sequential", "parallel"],
                        value="parallel",
                        label="Processing Mode"
                    )
                
                # Model selection for batch processing
                batch_model = gr.Dropdown(
                    choices=["transformer", "llm", "hybrid"],
                    value="transformer",
                    label="Model Type"
                )
                
                process_batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                
                batch_progress = gr.Progress()
            
            with gr.Column():
                gr.Markdown("### 📋 Batch Results")
                
                # Results table
                batch_results = gr.DataFrame(
                    label="Processing Results",
                    headers=["File", "Category", "Confidence", "Status", "Processing Time"],
                    datatype=["str", "str", "number", "str", "str"]
                )
                
                # Download results
                download_results_btn = gr.Button("📥 Download Results", variant="secondary")
                
                # Batch statistics
                batch_stats = gr.JSON(
                    label="Batch Statistics"
                )
        
        # Event handlers
        process_batch_btn.click(
            fn=self.process_batch,
            inputs=[batch_files, batch_size, processing_mode, batch_model],
            outputs=[batch_results, batch_stats],
            show_progress=True
        )
        
        download_results_btn.click(
            fn=self.download_batch_results,
            inputs=[batch_results],
            outputs=[]
        )
    
    def _create_api_testing_tab(self):
        """Create API testing tab"""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🧪 API Testing")
                
                # API endpoint
                api_endpoint = gr.Textbox(
                    label="API Endpoint",
                    value="http://localhost:8000/classify",
                    placeholder="Enter API endpoint URL"
                )
                
                # Request method
                request_method = gr.Dropdown(
                    choices=["POST", "GET", "PUT", "DELETE"],
                    value="POST",
                    label="Request Method"
                )
                
                # Request headers
                request_headers = gr.JSON(
                    label="Request Headers",
                    value={"Content-Type": "application/json"}
                )
                
                # Request body
                request_body = gr.Textbox(
                    label="Request Body",
                    lines=10,
                    value='{"text": "Sample document text", "categories": ["contract", "report", "email"]}'
                )
                
                # Send request
                send_request_btn = gr.Button("📤 Send Request", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📥 Response")
                
                # Response status
                response_status = gr.Number(
                    label="Status Code"
                )
                
                # Response headers
                response_headers = gr.JSON(
                    label="Response Headers"
                )
                
                # Response body
                response_body = gr.JSON(
                    label="Response Body"
                )
                
                # Response time
                response_time = gr.Number(
                    label="Response Time (ms)"
                )
        
        # Event handlers
        send_request_btn.click(
            fn=self.test_api,
            inputs=[api_endpoint, request_method, request_headers, request_body],
            outputs=[response_status, response_headers, response_body, response_time]
        )
    
    # Core functionality methods
    def classify_document(self, text: str, file, model_type: str, model_name: str,
                         confidence_threshold: float, max_length: int) -> Tuple[Dict, float, str, str, go.Figure]:
        """Classify document using selected model"""
        try:
            # Process input
            if file is not None:
                text = self._extract_text_from_file(file)
            
            if not text.strip():
                return {}, 0.0, "No text provided", "Please provide document text", go.Figure()
            
            # Load model if not already loaded
            if model_type not in self.models:
                self._load_model_by_type(model_type, model_name)
            
            # Perform classification
            start_time = time.time()
            
            if model_type == "transformer":
                result = self._classify_with_transformer(text, max_length)
            elif model_type == "llm":
                result = self._classify_with_llm(text)
            else:  # hybrid
                result = self._classify_with_hybrid(text, max_length)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Extract results
            category = result.get("category", "unknown")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Create confidence chart
            confidence_chart = self._create_confidence_chart(result.get("all_scores", {}))
            
            # Store in history
            self.processing_history.append({
                "timestamp": datetime.now().isoformat(),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "category": category,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_type": model_type
            })
            
            return result, confidence, category, reasoning, confidence_chart
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"error": str(e)}, 0.0, "Error", str(e), go.Figure()
    
    def generate_document(self, generation_type: str, document_type: str, content: str,
                         style: str, temperature: float, max_tokens: int) -> Tuple[str, Dict, Optional[Image.Image]]:
        """Generate document using selected parameters"""
        try:
            if generation_type == "text_to_image":
                # Generate image using diffusion model
                if "diffusion" not in self.models:
                    self._load_model_by_type("diffusion", "runwayml/stable-diffusion-v1-5")
                
                prompt = f"A {document_type} document with {content}, {style} style, professional design"
                image = self.models["diffusion"].generate_document_image(prompt)[0]
                
                return f"Generated {document_type} image with prompt: {prompt}", {
                    "generation_type": generation_type,
                    "document_type": document_type,
                    "style": style,
                    "temperature": temperature
                }, image
            
            elif generation_type == "template_generation":
                # Generate text using LLM
                if "llm" not in self.models:
                    self._load_model_by_type("llm", "gpt-3.5-turbo")
                
                generated_text = self.models["llm"].generate_document(
                    document_type, {"content": content}, style
                )
                
                return generated_text, {
                    "generation_type": generation_type,
                    "document_type": document_type,
                    "style": style,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }, None
            
            else:  # content_creation
                # Create structured content
                structured_content = self._create_structured_content(document_type, content, style)
                
                return structured_content, {
                    "generation_type": generation_type,
                    "document_type": document_type,
                    "style": style,
                    "content_length": len(structured_content)
                }, None
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating document: {str(e)}", {"error": str(e)}, None
    
    def _load_model_by_type(self, model_type: str, model_name: str):
        """Load model by type and name"""
        try:
            if model_type == "transformer":
                config = self.model_configs["transformer"]
                config.model_name = model_name
                self.models["transformer"] = ModelFactory.create_model("lora", config)
            
            elif model_type == "diffusion":
                config = self.model_configs["diffusion"]
                config.model_id = model_name
                self.models["diffusion"] = DiffusionModelFactory.create_pipeline("document", config)
            
            elif model_type == "llm":
                config = self.model_configs["llm"]
                config.model_name = model_name
                self.models["llm"] = LLMModelFactory.create_model("generator", config)
            
            logger.info(f"Loaded {model_type} model: {model_name}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def _extract_text_from_file(self, file) -> str:
        """Extract text from uploaded file"""
        try:
            if file.name.endswith('.txt'):
                with open(file.name, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file.name.endswith('.pdf'):
                # Implement PDF text extraction
                return "PDF text extraction not implemented yet"
            elif file.name.endswith('.docx'):
                # Implement DOCX text extraction
                return "DOCX text extraction not implemented yet"
            else:
                return "Unsupported file format"
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return f"Error extracting text: {str(e)}"
    
    def _classify_with_transformer(self, text: str, max_length: int) -> Dict:
        """Classify using transformer model"""
        # Mock implementation - replace with actual model inference
        categories = ["contract", "report", "email", "proposal", "invoice", "presentation"]
        scores = np.random.dirichlet(np.ones(len(categories)))
        
        best_idx = np.argmax(scores)
        return {
            "category": categories[best_idx],
            "confidence": float(scores[best_idx]),
            "reasoning": f"Classified as {categories[best_idx]} based on transformer model analysis",
            "all_scores": dict(zip(categories, scores.tolist()))
        }
    
    def _classify_with_llm(self, text: str) -> Dict:
        """Classify using LLM"""
        # Mock implementation - replace with actual LLM inference
        categories = ["contract", "report", "email", "proposal", "invoice", "presentation"]
        scores = np.random.dirichlet(np.ones(len(categories)))
        
        best_idx = np.argmax(scores)
        return {
            "category": categories[best_idx],
            "confidence": float(scores[best_idx]),
            "reasoning": f"LLM analysis suggests this is a {categories[best_idx]} based on content structure and language patterns",
            "all_scores": dict(zip(categories, scores.tolist()))
        }
    
    def _classify_with_hybrid(self, text: str, max_length: int) -> Dict:
        """Classify using hybrid approach"""
        # Combine transformer and LLM results
        transformer_result = self._classify_with_transformer(text, max_length)
        llm_result = self._classify_with_llm(text)
        
        # Weighted combination
        transformer_weight = 0.6
        llm_weight = 0.4
        
        combined_confidence = (transformer_result["confidence"] * transformer_weight + 
                             llm_result["confidence"] * llm_weight)
        
        return {
            "category": transformer_result["category"],
            "confidence": combined_confidence,
            "reasoning": f"Hybrid analysis combining transformer ({transformer_result['confidence']:.2f}) and LLM ({llm_result['confidence']:.2f}) results",
            "all_scores": transformer_result["all_scores"]
        }
    
    def _create_confidence_chart(self, scores: Dict[str, float]) -> go.Figure:
        """Create confidence distribution chart"""
        if not scores:
            return go.Figure()
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Confidence Scores by Category",
            xaxis_title="Categories",
            yaxis_title="Confidence Score",
            height=400
        )
        
        return fig
    
    def _create_structured_content(self, document_type: str, content: str, style: str) -> str:
        """Create structured content based on document type"""
        templates = {
            "contract": f"""
CONTRACT AGREEMENT

Parties: [To be specified]
Date: {datetime.now().strftime('%Y-%m-%d')}

Subject: {content}

TERMS AND CONDITIONS:
1. [Term 1]
2. [Term 2]
3. [Term 3]

SIGNATURES:
Party 1: _________________ Date: _________
Party 2: _________________ Date: _________
""",
            "report": f"""
REPORT

Title: {content}
Date: {datetime.now().strftime('%Y-%m-%d')}
Author: [Author Name]

EXECUTIVE SUMMARY:
[Summary of findings and recommendations]

METHODOLOGY:
[Description of methods used]

FINDINGS:
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

RECOMMENDATIONS:
1. [Recommendation 1]
2. [Recommendation 2]

CONCLUSION:
[Final conclusions]
""",
            "proposal": f"""
PROPOSAL

Project: {content}
Date: {datetime.now().strftime('%Y-%m-%d')}
Client: [Client Name]

PROJECT OVERVIEW:
[Description of the project]

OBJECTIVES:
1. [Objective 1]
2. [Objective 2]
3. [Objective 3]

METHODOLOGY:
[Approach and methods]

TIMELINE:
[Project timeline and milestones]

BUDGET:
[Budget breakdown]

NEXT STEPS:
[Action items and next steps]
"""
        }
        
        return templates.get(document_type, f"Document: {content}\n\n[Additional content based on {document_type} format]")
    
    def process_batch(self, files, batch_size: int, processing_mode: str, model_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Process batch of documents"""
        if not files:
            return pd.DataFrame(), {}
        
        results = []
        total_files = len(files)
        
        for i, file in enumerate(files):
            try:
                # Extract text
                text = self._extract_text_from_file(file)
                
                # Classify
                if model_type == "transformer":
                    result = self._classify_with_transformer(text, 512)
                elif model_type == "llm":
                    result = self._classify_with_llm(text)
                else:
                    result = self._classify_with_hybrid(text, 512)
                
                results.append({
                    "File": file.name,
                    "Category": result["category"],
                    "Confidence": result["confidence"],
                    "Status": "Success",
                    "Processing Time": f"{np.random.uniform(50, 200):.1f}ms"
                })
                
            except Exception as e:
                results.append({
                    "File": file.name,
                    "Category": "Error",
                    "Confidence": 0.0,
                    "Status": f"Error: {str(e)}",
                    "Processing Time": "0ms"
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate statistics
        stats = {
            "total_files": total_files,
            "successful": len([r for r in results if r["Status"] == "Success"]),
            "failed": len([r for r in results if r["Status"] != "Success"]),
            "average_confidence": np.mean([r["Confidence"] for r in results if r["Status"] == "Success"]),
            "processing_mode": processing_mode,
            "model_type": model_type
        }
        
        return df, stats
    
    def test_api(self, endpoint: str, method: str, headers: Dict, body: str) -> Tuple[int, Dict, Dict, float]:
        """Test API endpoint"""
        try:
            import requests
            
            start_time = time.time()
            
            if method == "POST":
                response = requests.post(endpoint, headers=headers, data=body, timeout=30)
            elif method == "GET":
                response = requests.get(endpoint, headers=headers, timeout=30)
            else:
                response = requests.request(method, endpoint, headers=headers, data=body, timeout=30)
            
            response_time = (time.time() - start_time) * 1000
            
            try:
                response_body = response.json()
            except:
                response_body = {"text": response.text}
            
            return (
                response.status_code,
                dict(response.headers),
                response_body,
                response_time
            )
        
        except Exception as e:
            return 0, {}, {"error": str(e)}, 0.0
    
    def update_analytics(self, time_range: str, metrics: List[str]) -> Tuple[go.Figure, go.Figure, str, str, str, str]:
        """Update analytics dashboard"""
        # Mock data generation
        hours = 24 if time_range == "24 hours" else 1 if time_range == "1 hour" else 168 if time_range == "7 days" else 720
        
        # Generate mock time series data
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
        
        # Performance chart
        performance_data = {
            "accuracy": np.random.normal(0.95, 0.02, hours),
            "response_time": np.random.normal(150, 20, hours),
            "throughput": np.random.normal(1000, 100, hours),
            "error_rate": np.random.normal(0.001, 0.0005, hours)
        }
        
        fig1 = go.Figure()
        for metric in metrics:
            if metric in performance_data:
                fig1.add_trace(go.Scatter(
                    x=timestamps,
                    y=performance_data[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title()
                ))
        
        fig1.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        
        # Distribution chart
        confidence_scores = np.random.beta(2, 2, 1000)  # Beta distribution for confidence scores
        fig2 = go.Figure(data=[go.Histogram(x=confidence_scores, nbinsx=20)])
        fig2.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            height=400
        )
        
        # Update metric cards
        accuracy_card = f"<div class='metric-card'><h3>{np.mean(performance_data['accuracy'])*100:.1f}%</h3><p>Accuracy</p></div>"
        response_time_card = f"<div class='metric-card'><h3>{np.mean(performance_data['response_time']):.0f}ms</h3><p>Response Time</p></div>"
        throughput_card = f"<div class='metric-card'><h3>{np.mean(performance_data['throughput']):.0f}/min</h3><p>Throughput</p></div>"
        error_rate_card = f"<div class='metric-card'><h3>{np.mean(performance_data['error_rate'])*100:.2f}%</h3><p>Error Rate</p></div>"
        
        return fig1, fig2, accuracy_card, response_time_card, throughput_card, error_rate_card
    
    def load_model(self, model_category: str) -> Tuple[Dict, pd.DataFrame]:
        """Load model and return configuration and performance"""
        config = self.model_configs.get(model_category, {})
        
        # Mock performance data
        performance_data = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Response Time"],
            "Value": [0.958, 0.942, 0.937, 0.939, 150],
            "Timestamp": [datetime.now().isoformat()] * 5
        })
        
        return config.__dict__ if hasattr(config, '__dict__') else config, performance_data
    
    def compare_models(self) -> go.Figure:
        """Compare different models"""
        models = ["BERT", "RoBERTa", "DistilBERT", "GPT-3.5", "Claude"]
        metrics = ["Accuracy", "Speed", "Memory Usage", "Cost"]
        
        # Mock comparison data
        data = np.random.uniform(0.7, 1.0, (len(models), len(metrics)))
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=metrics,
            y=models,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Model Comparison Matrix",
            height=400
        )
        
        return fig
    
    def train_model(self, training_data, epochs: int, learning_rate: float) -> str:
        """Train model with provided data"""
        if training_data is None:
            return "No training data provided"
        
        # Mock training process
        logs = []
        for epoch in range(epochs):
            logs.append(f"Epoch {epoch+1}/{epochs}")
            logs.append(f"Training Loss: {np.random.uniform(0.1, 0.5):.4f}")
            logs.append(f"Validation Loss: {np.random.uniform(0.2, 0.6):.4f}")
            logs.append(f"Accuracy: {np.random.uniform(0.8, 0.95):.4f}")
            logs.append("")
        
        logs.append("Training completed successfully!")
        
        return "\n".join(logs)
    
    def download_batch_results(self, results_df: pd.DataFrame) -> str:
        """Download batch processing results"""
        if results_df.empty:
            return "No results to download"
        
        # Save to CSV
        filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(filename, index=False)
        
        return f"Results saved to {filename}"
    
    def process_uploaded_file(self, file) -> str:
        """Process uploaded file and extract text"""
        if file is None:
            return ""
        
        return self._extract_text_from_file(file)
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch the Gradio interface"""
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            show_tips=True
        )

# Example usage
if __name__ == "__main__":
    # Create and launch interface
    interface = GradioInterface()
    interface.launch(share=False, server_port=7860)
























