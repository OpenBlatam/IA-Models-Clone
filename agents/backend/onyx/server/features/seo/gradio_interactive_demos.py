#!/usr/bin/env python3
"""
Interactive Gradio Demos for SEO Model Inference and Visualization
Ultra-Optimized SEO Evaluation System with Advanced Visualization
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import sys
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import asyncio
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

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

class InteractiveSEODemos:
    """Interactive demos for SEO model inference and visualization."""
    
    def __init__(self):
        self.model = None
        self.trainer = None
        self.config = None
        self.demo_data = self._load_demo_data()
        self.inference_queue = queue.Queue()
        self.visualization_cache = {}
        
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load demo data for interactive demonstrations."""
        return {
            'sample_texts': [
                "SEO optimization techniques for better search engine rankings and improved organic traffic",
                "Content marketing strategies that drive engagement and boost conversion rates",
                "Technical SEO best practices including site speed optimization and mobile-first indexing",
                "Keyword research and analysis for content creation and audience targeting",
                "On-page SEO elements like meta titles, descriptions, and header optimization",
                "Link building strategies for domain authority and referral traffic growth",
                "Local SEO optimization for small businesses and location-based searches",
                "E-commerce SEO strategies for online stores and product page optimization",
                "Voice search optimization techniques for smart speakers and mobile devices",
                "Core Web Vitals and user experience metrics for better search rankings"
            ],
            'sample_keywords': [
                "seo optimization", "content marketing", "technical seo", "keyword research",
                "on-page seo", "link building", "local seo", "e-commerce seo", "voice search", "core web vitals"
            ],
            'performance_metrics': {
                'page_speed': [85, 92, 78, 88, 95, 82, 90, 87, 79, 93],
                'mobile_score': [88, 95, 82, 90, 97, 85, 92, 89, 81, 94],
                'accessibility': [92, 88, 95, 90, 87, 93, 89, 91, 94, 86],
                'best_practices': [90, 93, 87, 91, 94, 88, 92, 89, 86, 95]
            }
        }
    
    def initialize_demo_model(self, model_type: str = "bert-base") -> str:
        """Initialize a demo model for interactive demonstrations."""
        try:
            # Create lightweight configuration for demos
            self.config = UltraOptimizedConfig(
                use_multi_gpu=False,
                use_amp=True,
                use_lora=True,
                use_diffusion=False,
                batch_size=4,
                learning_rate=1e-4,
                max_grad_norm=1.0,
                patience=3,
                num_epochs=5,
                model_name=model_type
            )
            
            # Initialize model and trainer
            self.model = UltraOptimizedSEOMetricsModule(self.config)
            self.trainer = UltraOptimizedSEOTrainer(self.model, self.config)
            
            return f"✅ Demo model initialized successfully!\n\n" \
                   f"📊 Model: {model_type}\n" \
                   f"🔧 Configuration: LoRA + AMP enabled\n" \
                   f"📱 Device: {self.config.device}\n" \
                   f"⚡ Ready for interactive demos!"
        
        except Exception as e:
            return f"❌ Error initializing demo model: {str(e)}"
    
    def real_time_seo_analysis(self, input_text: str, analysis_type: str) -> Tuple[str, Dict[str, Any]]:
        """Real-time SEO analysis with interactive visualization."""
        try:
            if self.model is None:
                return "❌ Model not initialized. Please initialize the demo model first.", {}
            
            # Perform real-time analysis
            start_time = time.time()
            
            # Create dummy labels for analysis
            dummy_labels = torch.tensor([1])
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model([input_text], dummy_labels, dummy_labels)
            
            # Extract features and predictions
            seo_features = outputs['seo_features']
            predictions = torch.softmax(seo_features, dim=1)
            
            # Calculate analysis time
            analysis_time = time.time() - start_time
            
            # Generate comprehensive analysis
            analysis_result = self._generate_seo_analysis(input_text, predictions, analysis_time)
            
            # Create interactive visualizations
            visualizations = self._create_interactive_visualizations(input_text, predictions, analysis_type)
            
            return analysis_result, visualizations
        
        except Exception as e:
            return f"❌ Analysis error: {str(e)}", {}
    
    def _generate_seo_analysis(self, text: str, predictions: torch.Tensor, analysis_time: float) -> str:
        """Generate comprehensive SEO analysis report."""
        # Calculate text metrics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Calculate readability (simplified Flesch Reading Ease)
        syllables = sum(1 for char in text.lower() if char in 'aeiou')
        readability_score = max(0, min(100, 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count)))
        
        # Extract SEO features
        seo_keywords = [kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()]
        keyword_density = len(seo_keywords) / max(1, word_count)
        
        # Generate analysis report
        report = f"📊 Real-Time SEO Analysis Report\n\n"
        report += f"⏱️ Analysis Time: {analysis_time:.3f} seconds\n\n"
        
        report += f"📝 Content Analysis:\n"
        report += f"• Word Count: {word_count}\n"
        report += f"• Character Count: {char_count}\n"
        report += f"• Sentence Count: {sentence_count}\n"
        report += f"• Readability Score: {readability_score:.1f}/100\n"
        report += f"• SEO Keywords Found: {len(seo_keywords)}\n"
        report += f"• Keyword Density: {keyword_density:.3f}\n\n"
        
        report += f"🎯 SEO Score: {predictions.max().item():.3f}\n"
        report += f"📈 Confidence: {predictions.max().item() * 100:.1f}%\n\n"
        
        report += f"🔍 Keywords Detected:\n"
        for keyword in seo_keywords:
            report += f"• {keyword}\n"
        
        # Add recommendations
        report += f"\n💡 Recommendations:\n"
        if word_count < 300:
            report += f"• Increase content length (aim for 300+ words)\n"
        if readability_score < 60:
            report += f"• Improve readability (aim for 60+)\n"
        if keyword_density < 0.01:
            report += f"• Add more relevant SEO keywords\n"
        if keyword_density > 0.05:
            report += f"• Reduce keyword density to avoid stuffing\n"
        
        return report
    
    def _create_interactive_visualizations(self, text: str, predictions: torch.Tensor, analysis_type: str) -> Dict[str, Any]:
        """Create interactive visualizations for SEO analysis."""
        visualizations = {}
        
        # 1. SEO Score Radar Chart
        if analysis_type == "comprehensive":
            fig_radar = go.Figure()
            
            # Calculate various SEO metrics
            word_count = len(text.split())
            char_count = len(text)
            readability = max(0, min(100, 206.835 - 1.015 * (word_count / max(1, len([s for s in text.split('.') if s.strip()]))) - 84.6 * (sum(1 for char in text.lower() if char in 'aeiou') / max(1, word_count))))
            
            categories = ['Content Length', 'Readability', 'SEO Score', 'Keyword Density', 'Technical SEO']
            values = [
                min(100, (word_count / 500) * 100),  # Content length score
                readability,  # Readability score
                predictions.max().item() * 100,  # SEO score
                min(100, (len([kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()]) / max(1, word_count)) * 1000),  # Keyword density score
                85  # Technical SEO score (placeholder)
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Score',
                line_color='rgb(32, 201, 151)',
                fillcolor='rgba(32, 201, 151, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="SEO Performance Radar Chart"
            )
            
            visualizations['radar_chart'] = fig_radar
        
        # 2. Text Analysis Bar Chart
        fig_bar = go.Figure()
        
        metrics = ['Words', 'Characters', 'Sentences', 'Keywords']
        values = [
            len(text.split()),
            len(text),
            len([s for s in text.split('.') if s.strip()]),
            len([kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()])
        ]
        
        fig_bar.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            text=values,
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            title="Text Analysis Metrics",
            xaxis_title="Metrics",
            yaxis_title="Count",
            showlegend=False
        )
        
        visualizations['bar_chart'] = fig_bar
        
        # 3. SEO Score Distribution
        fig_dist = go.Figure()
        
        # Create a distribution around the predicted score
        predicted_score = predictions.max().item()
        scores = np.random.normal(predicted_score, 0.1, 1000)
        scores = np.clip(scores, 0, 1)
        
        fig_dist.add_trace(go.Histogram(
            x=scores,
            nbinsx=30,
            marker_color='rgba(32, 201, 151, 0.7)',
            name='Score Distribution'
        ))
        
        fig_dist.add_vline(
            x=predicted_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Predicted: {predicted_score:.3f}"
        )
        
        fig_dist.update_layout(
            title="SEO Score Distribution",
            xaxis_title="SEO Score",
            yaxis_title="Frequency",
            showlegend=True
        )
        
        visualizations['distribution_chart'] = fig_dist
        
        # 4. Keyword Analysis Heatmap
        if analysis_type == "comprehensive":
            keywords = self.demo_data['sample_keywords']
            keyword_presence = [1 if kw.lower() in text.lower() else 0 for kw in keywords]
            
            # Create heatmap data
            heatmap_data = np.array(keyword_presence).reshape(1, -1)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=keywords,
                y=['Keyword Presence'],
                colorscale='RdYlGn',
                showscale=True
            ))
            
            fig_heatmap.update_layout(
                title="Keyword Presence Heatmap",
                xaxis_title="SEO Keywords",
                yaxis_title="",
                height=200
            )
            
            visualizations['heatmap'] = fig_heatmap
        
        return visualizations
    
    def batch_seo_analysis(self, texts: str, analysis_mode: str) -> Tuple[str, Dict[str, Any]]:
        """Batch SEO analysis with comparative visualizations."""
        try:
            if self.model is None:
                return "❌ Model not initialized. Please initialize the demo model first.", {}
            
            # Parse input texts
            text_list = [text.strip() for text in texts.split('\n') if text.strip()]
            
            if not text_list:
                return "❌ No valid texts provided.", {}
            
            # Perform batch analysis
            results = []
            for text in text_list:
                # Create dummy labels
                dummy_labels = torch.tensor([1])
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model([text], dummy_labels, dummy_labels)
                
                seo_features = outputs['seo_features']
                predictions = torch.softmax(seo_features, dim=1)
                
                # Calculate metrics
                word_count = len(text.split())
                char_count = len(text)
                sentence_count = len([s for s in text.split('.') if s.strip()])
                readability = max(0, min(100, 206.835 - 1.015 * (word_count / max(1, sentence_count)) - 84.6 * (sum(1 for char in text.lower() if char in 'aeiou') / max(1, word_count))))
                keyword_count = len([kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()])
                
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'seo_score': predictions.max().item(),
                    'confidence': predictions.max().item(),
                    'word_count': word_count,
                    'char_count': char_count,
                    'sentence_count': sentence_count,
                    'readability': readability,
                    'keyword_count': keyword_count
                })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Generate summary
            summary = f"📊 Batch SEO Analysis Summary\n\n"
            summary += f"• Total Texts: {len(text_list)}\n"
            summary += f"• Average SEO Score: {results_df['seo_score'].mean():.4f}\n"
            summary += f"• Average Word Count: {results_df['word_count'].mean():.1f}\n"
            summary += f"• Average Readability: {results_df['readability'].mean():.1f}\n"
            summary += f"• Analysis Mode: {analysis_mode}\n\n"
            
            # Add top performers
            top_texts = results_df.nlargest(3, 'seo_score')
            summary += f"🏆 Top Performers:\n"
            for i, (_, row) in enumerate(top_texts.iterrows(), 1):
                summary += f"{i}. Score: {row['seo_score']:.3f} - {row['text'][:50]}...\n"
            
            # Create comparative visualizations
            visualizations = self._create_comparative_visualizations(results_df, analysis_mode)
            
            return summary, {"dataframe": results_df, "visualizations": visualizations}
        
        except Exception as e:
            return f"❌ Batch analysis error: {str(e)}", {}
    
    def _create_comparative_visualizations(self, results_df: pd.DataFrame, analysis_mode: str) -> Dict[str, Any]:
        """Create comparative visualizations for batch analysis."""
        visualizations = {}
        
        # 1. SEO Score Comparison
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            x=[f"Text {i+1}" for i in range(len(results_df))],
            y=results_df['seo_score'],
            marker_color='rgba(32, 201, 151, 0.8)',
            name='SEO Score',
            text=[f"{score:.3f}" for score in results_df['seo_score']],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title="SEO Score Comparison Across Texts",
            xaxis_title="Texts",
            yaxis_title="SEO Score",
            showlegend=True
        )
        
        visualizations['comparison_chart'] = fig_comparison
        
        # 2. Multi-metric Radar Chart
        if analysis_mode == "comprehensive":
            fig_radar = go.Figure()
            
            # Normalize metrics for radar chart
            metrics = ['SEO Score', 'Word Count', 'Readability', 'Keyword Count']
            normalized_values = []
            
            for i in range(len(results_df)):
                values = [
                    results_df.iloc[i]['seo_score'] * 100,  # SEO Score (0-100)
                    min(100, (results_df.iloc[i]['word_count'] / 500) * 100),  # Word Count (0-100)
                    results_df.iloc[i]['readability'],  # Readability (0-100)
                    min(100, (results_df.iloc[i]['keyword_count'] / 5) * 100)  # Keyword Count (0-100)
                ]
                normalized_values.append(values)
            
            colors = ['rgb(32, 201, 151)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']
            
            for i, values in enumerate(normalized_values):
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=f'Text {i+1}',
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.3)')
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Multi-Metric Comparison Radar Chart"
            )
            
            visualizations['radar_comparison'] = fig_radar
        
        # 3. Correlation Heatmap
        if analysis_mode == "comprehensive":
            # Calculate correlations
            numeric_cols = ['seo_score', 'word_count', 'char_count', 'sentence_count', 'readability', 'keyword_count']
            corr_matrix = results_df[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig_corr.update_layout(
                title="Metric Correlation Heatmap",
                xaxis_title="Metrics",
                yaxis_title="Metrics"
            )
            
            visualizations['correlation_heatmap'] = fig_corr
        
        # 4. Performance Distribution
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=results_df['seo_score'],
            nbinsx=10,
            marker_color='rgba(32, 201, 151, 0.7)',
            name='SEO Score Distribution'
        ))
        
        fig_dist.add_vline(
            x=results_df['seo_score'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {results_df['seo_score'].mean():.3f}"
        )
        
        fig_dist.update_layout(
            title="SEO Score Distribution Across Texts",
            xaxis_title="SEO Score",
            yaxis_title="Frequency",
            showlegend=True
        )
        
        visualizations['distribution_comparison'] = fig_dist
        
        return visualizations
    
    def interactive_training_demo(self, training_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Interactive training demonstration with real-time progress."""
        try:
            if self.model is None:
                return "❌ Model not initialized. Please initialize the demo model first.", {}
            
            # Update configuration
            for key, value in training_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Create demo training data
            demo_texts = self.demo_data['sample_texts'][:8]  # Use 8 texts for demo
            demo_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # Binary classification
            
            # Create dataset and dataloader
            demo_dataset = SEODataset(demo_texts, torch.tensor(demo_labels), self.model.seo_tokenizer)
            demo_loader = DataLoader(demo_dataset, batch_size=2, shuffle=True)
            
            # Training loop with progress tracking
            training_log = []
            total_steps = len(demo_loader) * self.config.num_epochs
            
            for epoch in range(self.config.num_epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                
                for batch_idx, batch in enumerate(demo_loader):
                    # Move batch to device
                    batch = {k: v.to(self.trainer.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    # Training step
                    metrics = self.trainer.train_step(batch)
                    
                    epoch_loss += metrics['loss']
                    epoch_accuracy += metrics['accuracy']
                    
                    # Log progress
                    current_step = epoch * len(demo_loader) + batch_idx + 1
                    progress = current_step / total_steps
                    
                    training_log.append({
                        'step': current_step,
                        'epoch': epoch + 1,
                        'loss': metrics['loss'],
                        'accuracy': metrics['accuracy'],
                        'progress': progress
                    })
                
                # Epoch summary
                avg_loss = epoch_loss / len(demo_loader)
                avg_accuracy = epoch_accuracy / len(demo_loader)
                
                training_log.append({
                    'step': (epoch + 1) * len(demo_loader),
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': avg_accuracy,
                    'progress': (epoch + 1) / self.config.num_epochs
                })
            
            # Generate training report
            report = self._generate_training_demo_report(training_log)
            
            # Create training visualizations
            visualizations = self._create_training_demo_visualizations(training_log)
            
            return report, visualizations
        
        except Exception as e:
            return f"❌ Training demo error: {str(e)}", {}
    
    def _generate_training_demo_report(self, training_log: List[Dict]) -> str:
        """Generate training demonstration report."""
        if not training_log:
            return "No training data available."
        
        report = "🎯 Interactive Training Demo Report\n\n"
        
        # Final metrics
        final_log = training_log[-1]
        report += f"🏁 Final Results:\n"
        report += f"• Total Steps: {final_log['step']}\n"
        report += f"• Final Loss: {final_log['loss']:.4f}\n"
        report += f"• Final Accuracy: {final_log['accuracy']:.4f}\n"
        report += f"• Training Progress: {final_log['progress']*100:.1f}%\n\n"
        
        # Training statistics
        losses = [log['loss'] for log in training_log if 'loss' in log]
        accuracies = [log['accuracy'] for log in training_log if 'accuracy' in log]
        
        report += f"📈 Training Statistics:\n"
        report += f"• Loss Improvement: {losses[0] - losses[-1]:.4f}\n"
        report += f"• Accuracy Improvement: {accuracies[-1] - accuracies[0]:.4f}\n"
        report += f"• Best Loss: {min(losses):.4f}\n"
        report += f"• Best Accuracy: {max(accuracies):.4f}\n"
        
        return report
    
    def _create_training_demo_visualizations(self, training_log: List[Dict]) -> Dict[str, Any]:
        """Create training demonstration visualizations."""
        visualizations = {}
        
        # Extract data
        steps = [log['step'] for log in training_log if 'step' in log]
        losses = [log['loss'] for log in training_log if 'loss' in log]
        accuracies = [log['accuracy'] for log in training_log if 'accuracy' in log]
        progress = [log['progress'] for log in training_log if 'progress' in log]
        
        # 1. Training Progress
        fig_progress = go.Figure()
        
        fig_progress.add_trace(go.Scatter(
            x=steps,
            y=progress,
            mode='lines+markers',
            name='Training Progress',
            line=dict(color='rgb(32, 201, 151)', width=3),
            marker=dict(size=8)
        ))
        
        fig_progress.update_layout(
            title="Training Progress Over Time",
            xaxis_title="Training Steps",
            yaxis_title="Progress (%)",
            yaxis=dict(tickformat='.0%'),
            showlegend=True
        )
        
        visualizations['progress_chart'] = fig_progress
        
        # 2. Loss and Accuracy
        fig_metrics = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Loss', 'Training Accuracy'),
            vertical_spacing=0.1
        )
        
        fig_metrics.add_trace(
            go.Scatter(x=steps, y=losses, mode='lines+markers', name='Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        fig_metrics.add_trace(
            go.Scatter(x=steps, y=accuracies, mode='lines+markers', name='Accuracy', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig_metrics.update_layout(
            title="Training Metrics Over Time",
            height=600,
            showlegend=True
        )
        
        visualizations['metrics_chart'] = fig_metrics
        
        # 3. 3D Training Surface
        if len(steps) > 5:
            fig_3d = go.Figure(data=[go.Surface(
                x=np.array(steps).reshape(-1, 1),
                y=np.array(progress).reshape(-1, 1),
                z=np.array(losses).reshape(-1, 1),
                colorscale='Viridis',
                name='Training Surface'
            )])
            
            fig_3d.update_layout(
                title="3D Training Surface (Steps vs Progress vs Loss)",
                scene=dict(
                    xaxis_title="Steps",
                    yaxis_title="Progress",
                    zaxis_title="Loss"
                ),
                height=500
            )
            
            visualizations['3d_surface'] = fig_3d
        
        return visualizations

def create_interactive_demos():
    """Create the interactive demos interface."""
    demos = InteractiveSEODemos()
    
    with gr.Blocks(title="Interactive SEO Model Demos", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Interactive SEO Model Demos")
        gr.Markdown("### Real-time Inference and Advanced Visualization")
        
        with gr.Tabs():
            # Real-time Analysis Tab
            with gr.Tab("🔍 Real-time SEO Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Real-time SEO Analysis")
                        
                        demo_model_type = gr.Dropdown(
                            choices=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
                            value="bert-base-uncased",
                            label="Demo Model Type"
                        )
                        
                        init_demo_btn = gr.Button("🚀 Initialize Demo Model", variant="primary")
                        
                        demo_status = gr.Textbox(
                            label="Model Status",
                            placeholder="Click 'Initialize Demo Model' to start...",
                            lines=3
                        )
                        
                        gr.Markdown("### Analysis Input")
                        
                        input_text = gr.Textbox(
                            label="Input Text for SEO Analysis",
                            placeholder="Enter text to analyze...",
                            lines=5
                        )
                        
                        analysis_type = gr.Dropdown(
                            choices=["basic", "comprehensive"],
                            value="comprehensive",
                            label="Analysis Type"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Analysis Results")
                        analysis_result = gr.Textbox(
                            label="SEO Analysis Report",
                            placeholder="Analysis results will appear here...",
                            lines=15
                        )
                        
                        gr.Markdown("### Interactive Visualizations")
                        
                        # Visualization components
                        radar_chart = gr.Plot(label="SEO Performance Radar")
                        bar_chart = gr.Plot(label="Text Analysis Metrics")
                        distribution_chart = gr.Plot(label="SEO Score Distribution")
                        heatmap = gr.Plot(label="Keyword Analysis Heatmap")
                
                # Event handlers
                init_demo_btn.click(
                    fn=demos.initialize_demo_model,
                    inputs=[demo_model_type],
                    outputs=demo_status
                )
                
                analyze_btn.click(
                    fn=demos.real_time_seo_analysis,
                    inputs=[input_text, analysis_type],
                    outputs=[analysis_result, gr.State({})]
                ).then(
                    fn=lambda result, viz: viz.get('radar_chart', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=radar_chart
                ).then(
                    fn=lambda result, viz: viz.get('bar_chart', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=bar_chart
                ).then(
                    fn=lambda result, viz: viz.get('distribution_chart', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=distribution_chart
                ).then(
                    fn=lambda result, viz: viz.get('heatmap', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=heatmap
                )
            
            # Batch Analysis Tab
            with gr.Tab("📊 Batch SEO Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Batch Analysis")
                        
                        batch_texts = gr.Textbox(
                            label="Input Texts (one per line)",
                            placeholder="Enter multiple texts to analyze...",
                            lines=10
                        )
                        
                        batch_analysis_mode = gr.Dropdown(
                            choices=["basic", "comprehensive"],
                            value="comprehensive",
                            label="Analysis Mode"
                        )
                        
                        batch_analyze_btn = gr.Button("📊 Analyze Batch", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Batch Results")
                        batch_result = gr.Textbox(
                            label="Batch Analysis Summary",
                            placeholder="Batch analysis results will appear here...",
                            lines=10
                        )
                        
                        batch_results_table = gr.Dataframe(
                            label="Detailed Results",
                            headers=["Text", "SEO_Score", "Word_Count", "Readability", "Keywords"]
                        )
                
                # Event handler
                batch_analyze_btn.click(
                    fn=demos.batch_seo_analysis,
                    inputs=[batch_texts, batch_analysis_mode],
                    outputs=[batch_result, gr.State({})]
                ).then(
                    fn=lambda result, viz: viz.get('dataframe', pd.DataFrame()),
                    inputs=[batch_result, gr.State({})],
                    outputs=batch_results_table
                )
            
            # Interactive Training Tab
            with gr.Tab("🏋️ Interactive Training Demo"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Configuration")
                        
                        demo_epochs = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Number of Epochs"
                        )
                        
                        demo_batch_size = gr.Slider(
                            minimum=1, maximum=8, value=2, step=1,
                            label="Batch Size"
                        )
                        
                        demo_learning_rate = gr.Slider(
                            minimum=1e-5, maximum=1e-3, value=1e-4, step=1e-5,
                            label="Learning Rate"
                        )
                        
                        demo_patience = gr.Slider(
                            minimum=1, maximum=5, value=2, step=1,
                            label="Patience"
                        )
                        
                        start_training_btn = gr.Button("🏋️ Start Training Demo", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Training Results")
                        training_result = gr.Textbox(
                            label="Training Demo Report",
                            placeholder="Training results will appear here...",
                            lines=10
                        )
                        
                        gr.Markdown("### Training Visualizations")
                        
                        # Training visualization components
                        progress_chart = gr.Plot(label="Training Progress")
                        metrics_chart = gr.Plot(label="Training Metrics")
                        surface_3d = gr.Plot(label="3D Training Surface")
                
                # Event handler
                start_training_btn.click(
                    fn=demos.interactive_training_demo,
                    inputs=[gr.State({
                        'num_epochs': demo_epochs,
                        'batch_size': demo_batch_size,
                        'learning_rate': demo_learning_rate,
                        'patience': demo_patience
                    })],
                    outputs=[training_result, gr.State({})]
                ).then(
                    fn=lambda result, viz: viz.get('progress_chart', None),
                    inputs=[training_result, gr.State({})],
                    outputs=progress_chart
                ).then(
                    fn=lambda result, viz: viz.get('metrics_chart', None),
                    inputs=[training_result, gr.State({})],
                    outputs=metrics_chart
                ).then(
                    fn=lambda result, viz: viz.get('3d_surface', None),
                    inputs=[training_result, gr.State({})],
                    outputs=surface_3d
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        ### 🎯 Demo Features:
        - **Real-time Analysis**: Instant SEO evaluation with live metrics
        - **Interactive Visualizations**: Plotly charts with zoom, pan, and hover
        - **Batch Processing**: Compare multiple texts simultaneously
        - **Training Demonstrations**: Watch model learning in real-time
        - **Advanced Metrics**: Comprehensive SEO scoring and recommendations
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interactive demos
    demo = create_interactive_demos()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
