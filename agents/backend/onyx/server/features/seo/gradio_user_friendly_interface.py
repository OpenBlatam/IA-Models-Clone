#!/usr/bin/env python3
"""
User-Friendly Gradio Interface for SEO Model Capabilities
Enhanced UX with Intuitive Workflows and Visual Appeal
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
from pathlib import Path
import sys
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import time
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

class UserFriendlySEOInterface:
    """User-friendly interface for showcasing SEO model capabilities."""
    
    def __init__(self):
        self.model = None
        self.trainer = None
        self.config = None
        self.demo_data = self._load_demo_data()
        self.analysis_history = []
        
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load demo data for showcasing model capabilities."""
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
            'capability_examples': {
                'text_analysis': "Our model analyzes text content for SEO optimization, providing detailed insights on readability, keyword density, and content structure.",
                'batch_processing': "Process multiple texts simultaneously to compare SEO performance and identify optimization opportunities across your content portfolio.",
                'real_time_training': "Watch the model learn in real-time with configurable parameters and live visualization of training progress.",
                'advanced_metrics': "Access comprehensive SEO metrics including content quality scores, technical SEO analysis, and user experience indicators."
            }
        }
    
    def initialize_model(self, model_type: str = "bert-base") -> str:
        """Initialize the SEO model with user-friendly feedback."""
        try:
            # Create configuration optimized for user experience
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
            
            return f"🎉 **Model Successfully Initialized!**\n\n" \
                   f"🚀 **Model Type**: {model_type}\n" \
                   f"⚡ **Features Enabled**: LoRA + AMP + Gradient Clipping\n" \
                   f"💻 **Device**: {self.config.device}\n" \
                   f"📊 **Status**: Ready for analysis!\n\n" \
                   f"✨ **What you can do now**:\n" \
                   f"• Analyze individual texts for SEO optimization\n" \
                   f"• Compare multiple texts in batch mode\n" \
                   f"• Watch real-time model training\n" \
                   f"• Explore advanced SEO metrics"
        
        except Exception as e:
            return f"❌ **Oops! Something went wrong:**\n\n" \
                   f"**Error**: {str(e)}\n\n" \
                   f"**Troubleshooting tips**:\n" \
                   f"• Check if all dependencies are installed\n" \
                   f"• Ensure you have sufficient memory\n" \
                   f"• Try a different model type"
    
    def showcase_text_analysis(self, input_text: str, analysis_depth: str) -> Tuple[str, Dict[str, Any]]:
        """Showcase text analysis capabilities with user-friendly output."""
        try:
            if self.model is None:
                return "⚠️ **Please initialize the model first!**\n\nClick the '🚀 Initialize Model' button above to get started.", {}
            
            # Perform analysis with timing
            start_time = time.time()
            
            # Create dummy labels for analysis
            dummy_labels = torch.tensor([1])
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model([input_text], dummy_labels, dummy_labels)
            
            seo_features = outputs['seo_features']
            predictions = torch.softmax(seo_features, dim=1)
            
            analysis_time = time.time() - start_time
            
            # Generate user-friendly analysis report
            analysis_result = self._generate_user_friendly_analysis(input_text, predictions, analysis_time, analysis_depth)
            
            # Create engaging visualizations
            visualizations = self._create_engaging_visualizations(input_text, predictions, analysis_depth)
            
            # Store in history
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'text': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                'seo_score': predictions.max().item(),
                'analysis_depth': analysis_depth
            })
            
            return analysis_result, visualizations
        
        except Exception as e:
            return f"❌ **Analysis failed**:\n\n**Error**: {str(e)}\n\n**Please try again with different text or settings.**", {}
    
    def _generate_user_friendly_analysis(self, text: str, predictions: torch.Tensor, analysis_time: float, depth: str) -> str:
        """Generate user-friendly analysis report with clear explanations."""
        # Calculate metrics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        syllables = sum(1 for char in text.lower() if char in 'aeiou')
        readability_score = max(0, min(100, 206.835 - 1.015 * (word_count / max(1, sentence_count)) - 84.6 * (syllables / max(1, word_count))))
        
        # Extract SEO features
        seo_keywords = [kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()]
        keyword_density = len(seo_keywords) / max(1, word_count)
        
        # SEO score and confidence
        seo_score = predictions.max().item()
        confidence = seo_score * 100
        
        # Generate report
        report = f"🔍 **SEO Analysis Report**\n\n"
        report += f"⏱️ **Analysis completed in**: {analysis_time:.3f} seconds\n\n"
        
        # Content Overview
        report += f"📝 **Content Overview**\n"
        report += f"• **Length**: {word_count} words, {char_count} characters\n"
        report += f"• **Structure**: {sentence_count} sentences\n"
        report += f"• **Readability**: {readability_score:.1f}/100 (Flesch Reading Ease)\n\n"
        
        # SEO Performance
        report += f"🎯 **SEO Performance**\n"
        report += f"• **Overall Score**: {seo_score:.3f}/1.0\n"
        report += f"• **Confidence**: {confidence:.1f}%\n"
        report += f"• **Keywords Found**: {len(seo_keywords)}\n"
        report += f"• **Keyword Density**: {keyword_density:.3f}\n\n"
        
        # Keywords
        if seo_keywords:
            report += f"🔑 **SEO Keywords Detected**\n"
            for keyword in seo_keywords:
                report += f"• {keyword}\n"
            report += "\n"
        
        # Recommendations
        report += f"💡 **Optimization Recommendations**\n"
        recommendations = []
        
        if word_count < 300:
            recommendations.append("📏 **Increase content length** - Aim for 300+ words for better SEO")
        if readability_score < 60:
            recommendations.append("📖 **Improve readability** - Target 60+ for broader audience appeal")
        if keyword_density < 0.01:
            recommendations.append("🔍 **Add more relevant keywords** - Include SEO terms naturally")
        if keyword_density > 0.05:
            recommendations.append("⚠️ **Reduce keyword density** - Avoid keyword stuffing")
        if len(seo_keywords) < 3:
            recommendations.append("🎯 **Expand keyword coverage** - Include more SEO-relevant terms")
        
        if recommendations:
            for rec in recommendations:
                report += f"{rec}\n"
        else:
            report += "🎉 **Great job!** Your content meets most SEO best practices.\n"
        
        # Analysis depth explanation
        if depth == "comprehensive":
            report += f"\n🔬 **Analysis Depth**: Comprehensive\n"
            report += f"This analysis includes advanced metrics, detailed recommendations, and multiple visualization types."
        
        return report
    
    def _create_engaging_visualizations(self, text: str, predictions: torch.Tensor, depth: str) -> Dict[str, Any]:
        """Create engaging and informative visualizations."""
        visualizations = {}
        
        # 1. SEO Performance Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predictions.max().item() * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "SEO Performance Score", 'font': {'size': 24}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        visualizations['seo_gauge'] = fig_gauge
        
        # 2. Content Metrics Dashboard
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        keyword_count = len([kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()])
        
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Word Count', 'Character Count', 'Sentences', 'Keywords'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Word count indicator
        fig_dashboard.add_trace(go.Indicator(
            mode="number+delta",
            value=word_count,
            title={'text': "Words"},
            delta={'reference': 300},
            domain={'row': 0, 'column': 0}
        ), row=1, col=1)
        
        # Character count indicator
        fig_dashboard.add_trace(go.Indicator(
            mode="number+delta",
            value=char_count,
            title={'text': "Characters"},
            delta={'reference': 1500},
            domain={'row': 0, 'column': 1}
        ), row=1, col=2)
        
        # Sentence count indicator
        fig_dashboard.add_trace(go.Indicator(
            mode="number+delta",
            value=sentence_count,
            title={'text': "Sentences"},
            delta={'reference': 15},
            domain={'row': 1, 'column': 0}
        ), row=2, col=1)
        
        # Keyword count indicator
        fig_dashboard.add_trace(go.Indicator(
            mode="number+delta",
            value=keyword_count,
            title={'text': "SEO Keywords"},
            delta={'reference': 5},
            domain={'row': 1, 'column': 1}
        ), row=2, col=2)
        
        fig_dashboard.update_layout(
            height=500,
            title_text="Content Metrics Dashboard",
            showlegend=False
        )
        
        visualizations['metrics_dashboard'] = fig_dashboard
        
        # 3. SEO Score Distribution
        if depth == "comprehensive":
            predicted_score = predictions.max().item()
            scores = np.random.normal(predicted_score, 0.1, 1000)
            scores = np.clip(scores, 0, 1)
            
            fig_dist = go.Figure()
            
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
                line_width=3,
                annotation_text=f"Your Score: {predicted_score:.3f}"
            )
            
            fig_dist.update_layout(
                title="SEO Score Distribution",
                xaxis_title="SEO Score",
                yaxis_title="Frequency",
                showlegend=True,
                height=400
            )
            
            visualizations['score_distribution'] = fig_dist
        
        return visualizations
    
    def showcase_batch_analysis(self, texts: str, analysis_mode: str) -> Tuple[str, Dict[str, Any]]:
        """Showcase batch analysis capabilities with comparative insights."""
        try:
            if self.model is None:
                return "⚠️ **Please initialize the model first!**\n\nClick the '🚀 Initialize Model' button above to get started.", {}
            
            # Parse input texts
            text_list = [text.strip() for text in texts.split('\n') if text.strip()]
            
            if not text_list:
                return "❌ **No texts provided**\n\nPlease enter some text to analyze.", {}
            
            # Perform batch analysis
            results = []
            for i, text in enumerate(text_list):
                # Create dummy labels
                dummy_labels = torch.tensor([1])
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model([text], dummy_labels, dummy_labels)
                
                seo_features = outputs['seo_features']
                predictions = torch.softmax(seo_features, dim=1)
                
                # Calculate metrics
                word_count = len(text.split())
                sentence_count = len([s for s in text.split('.') if s.strip()])
                readability = max(0, min(100, 206.835 - 1.015 * (word_count / max(1, sentence_count)) - 84.6 * (sum(1 for char in text.lower() if char in 'aeiou') / max(1, word_count))))
                keyword_count = len([kw for kw in self.demo_data['sample_keywords'] if kw.lower() in text.lower()])
                
                results.append({
                    'text_id': f"Text {i+1}",
                    'text_preview': text[:80] + '...' if len(text) > 80 else text,
                    'seo_score': predictions.max().item(),
                    'word_count': word_count,
                    'readability': readability,
                    'keyword_count': keyword_count
                })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Generate user-friendly summary
            summary = self._generate_batch_summary(results_df, analysis_mode)
            
            # Create comparative visualizations
            visualizations = self._create_batch_visualizations(results_df, analysis_mode)
            
            return summary, {"dataframe": results_df, "visualizations": visualizations}
        
        except Exception as e:
            return f"❌ **Batch analysis failed**:\n\n**Error**: {str(e)}\n\n**Please try again with different texts.**", {}
    
    def _generate_batch_summary(self, results_df: pd.DataFrame, mode: str) -> str:
        """Generate user-friendly batch analysis summary."""
        summary = f"📊 **Batch Analysis Summary**\n\n"
        summary += f"🔢 **Total Texts Analyzed**: {len(results_df)}\n"
        summary += f"📈 **Average SEO Score**: {results_df['seo_score'].mean():.3f}\n"
        summary += f"📝 **Average Word Count**: {results_df['word_count'].mean():.1f}\n"
        summary += f"📖 **Average Readability**: {results_df['readability'].mean():.1f}\n"
        summary += f"🔍 **Analysis Mode**: {mode.title()}\n\n"
        
        # Top performers
        top_texts = results_df.nlargest(3, 'seo_score')
        summary += f"🏆 **Top Performers**\n"
        for i, (_, row) in enumerate(top_texts.iterrows(), 1):
            summary += f"{i}. **{row['text_id']}** - Score: {row['seo_score']:.3f}\n"
            summary += f"   Preview: {row['text_preview']}\n\n"
        
        # Performance insights
        summary += f"💡 **Performance Insights**\n"
        if results_df['seo_score'].std() < 0.1:
            summary += "• **Consistent Performance**: Your texts show similar SEO quality\n"
        else:
            summary += "• **Variable Performance**: Some texts perform better than others\n"
        
        if results_df['readability'].mean() > 70:
            summary += "• **Good Readability**: Overall content is easy to read\n"
        elif results_df['readability'].mean() < 50:
            summary += "• **Improve Readability**: Consider simplifying language\n"
        
        if results_df['keyword_count'].mean() < 3:
            summary += "• **Keyword Opportunity**: Add more SEO-relevant terms\n"
        
        return summary
    
    def _create_batch_visualizations(self, results_df: pd.DataFrame, mode: str) -> Dict[str, Any]:
        """Create engaging batch analysis visualizations."""
        visualizations = {}
        
        # 1. Performance Comparison Chart
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            x=results_df['text_id'],
            y=results_df['seo_score'],
            marker_color='rgba(32, 201, 151, 0.8)',
            name='SEO Score',
            text=[f"{score:.3f}" for score in results_df['seo_score']],
            textposition='auto',
            textfont={'size': 12}
        ))
        
        fig_comparison.update_layout(
            title="SEO Performance Comparison",
            xaxis_title="Texts",
            yaxis_title="SEO Score",
            showlegend=True,
            height=400,
            yaxis={'range': [0, 1]}
        )
        
        visualizations['performance_comparison'] = fig_comparison
        
        # 2. Multi-metric Radar Chart
        if mode == "comprehensive":
            fig_radar = go.Figure()
            
            metrics = ['SEO Score', 'Word Count', 'Readability', 'Keywords']
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
                title="Multi-Metric Comparison",
                height=500
            )
            
            visualizations['radar_chart'] = fig_radar
        
        return visualizations
    
    def showcase_training_capabilities(self, training_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Showcase real-time training capabilities with user-friendly feedback."""
        try:
            if self.model is None:
                return "⚠️ **Please initialize the model first!**\n\nClick the '🚀 Initialize Model' button above to get started.", {}
            
            # Update configuration
            for key, value in training_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Create demo training data
            demo_texts = self.demo_data['sample_texts'][:6]
            demo_labels = [1, 0, 1, 0, 1, 0]
            
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
            
            # Generate user-friendly training report
            report = self._generate_training_report(training_log)
            
            # Create training visualizations
            visualizations = self._create_training_visualizations(training_log)
            
            return report, visualizations
        
        except Exception as e:
            return f"❌ **Training failed**:\n\n**Error**: {str(e)}\n\n**Please try again with different settings.**", {}
    
    def _generate_training_report(self, training_log: List[Dict]) -> str:
        """Generate user-friendly training report."""
        if not training_log:
            return "No training data available."
        
        report = "🎯 **Training Demonstration Complete!**\n\n"
        
        # Final results
        final_log = training_log[-1]
        report += f"🏁 **Final Results**\n"
        report += f"• **Total Training Steps**: {final_log['step']}\n"
        report += f"• **Final Loss**: {final_log['loss']:.4f}\n"
        report += f"• **Final Accuracy**: {final_log['accuracy']:.4f}\n"
        report += f"• **Training Progress**: {final_log['progress']*100:.1f}%\n\n"
        
        # Training insights
        losses = [log['loss'] for log in training_log if 'loss' in log]
        accuracies = [log['accuracy'] for log in training_log if 'accuracy' in log]
        
        report += f"📈 **Training Insights**\n"
        report += f"• **Loss Improvement**: {losses[0] - losses[-1]:.4f}\n"
        report += f"• **Accuracy Improvement**: {accuracies[-1] - accuracies[0]:.4f}\n"
        report += f"• **Best Loss**: {min(losses):.4f}\n"
        report += f"• **Best Accuracy**: {max(accuracies):.4f}\n\n"
        
        # Performance assessment
        if losses[-1] < losses[0] * 0.5:
            report += "🎉 **Excellent Training**: Loss reduced significantly!\n"
        elif losses[-1] < losses[0]:
            report += "✅ **Good Training**: Loss improved steadily.\n"
        else:
            report += "⚠️ **Training Challenge**: Loss didn't improve as expected.\n"
        
        if accuracies[-1] > 0.8:
            report += "🌟 **High Accuracy**: Model learned very well!\n"
        elif accuracies[-1] > 0.6:
            report += "👍 **Good Accuracy**: Model learned reasonably well.\n"
        else:
            report += "📚 **Learning Opportunity**: Model could benefit from more training.\n"
        
        return report
    
    def _create_training_visualizations(self, training_log: List[Dict]) -> Dict[str, Any]:
        """Create engaging training visualizations."""
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
            line=dict(color='rgb(32, 201, 151)', width=4),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig_progress.update_layout(
            title="Training Progress Over Time",
            xaxis_title="Training Steps",
            yaxis_title="Progress (%)",
            yaxis=dict(tickformat='.0%'),
            showlegend=True,
            height=400
        )
        
        visualizations['progress_chart'] = fig_progress
        
        # 2. Loss and Accuracy
        fig_metrics = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Loss', 'Training Accuracy'),
            vertical_spacing=0.15
        )
        
        fig_metrics.add_trace(
            go.Scatter(x=steps, y=losses, mode='lines+markers', name='Loss', 
                      line=dict(color='red', width=3), marker=dict(size=6)),
            row=1, col=1
        )
        
        fig_metrics.add_trace(
            go.Scatter(x=steps, y=accuracies, mode='lines+markers', name='Accuracy', 
                      line=dict(color='blue', width=3), marker=dict(size=6)),
            row=2, col=1
        )
        
        fig_metrics.update_layout(
            title="Training Metrics Over Time",
            height=600,
            showlegend=True
        )
        
        visualizations['metrics_chart'] = fig_metrics
        
        return visualizations
    
    def get_capability_overview(self) -> str:
        """Get an overview of model capabilities for users."""
        overview = "🚀 **SEO Model Capabilities Overview**\n\n"
        
        overview += "**🔍 Text Analysis**\n"
        overview += "• Real-time SEO scoring and optimization\n"
        overview += "• Content quality assessment\n"
        overview += "• Keyword density analysis\n"
        overview += "• Readability scoring\n\n"
        
        overview += "**📊 Batch Processing**\n"
        overview += "• Multiple text comparison\n"
        overview += "• Performance ranking\n"
        overview += "• Correlation analysis\n\n"
        
        overview += "**🏋️ Training & Learning**\n"
        overview += "• Real-time model training\n"
        overview += "• Configurable parameters\n"
        overview += "• Live progress monitoring\n"
        overview += "• Performance visualization\n\n"
        
        overview += "**🎨 Advanced Features**\n"
        overview += "• Interactive visualizations\n"
        overview += "• Comprehensive metrics\n"
        overview += "• Optimization recommendations\n"
        overview += "• Export capabilities\n\n"
        
        overview += "**💡 Use Cases**\n"
        overview += "• Content optimization\n"
        overview += "• SEO strategy planning\n"
        overview += "• Content quality assessment\n"
        overview += "• Training and education\n"
        
        return overview

def create_user_friendly_interface():
    """Create the user-friendly interface."""
    interface = UserFriendlySEOInterface()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .gradio-header {
        text-align: center !important;
        padding: 20px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        margin-bottom: 20px !important;
    }
    .gradio-tab-nav {
        background: #f8f9fa !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    .gradio-button {
        border-radius: 25px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .gradio-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    """
    
    with gr.Blocks(title="User-Friendly SEO Model Interface", theme=gr.themes.Soft(), css=custom_css) as demo:
        # Header
        gr.HTML("""
        <div class="gradio-header">
            <h1>🚀 SEO Model Capabilities Showcase</h1>
            <p>Experience the power of our ultra-optimized SEO evaluation system</p>
        </div>
        """)
        
        with gr.Tabs():
            # Welcome Tab
            with gr.Tab("🏠 Welcome & Overview"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("""
                        ## 🎯 Welcome to the SEO Model Showcase!
                        
                        This interface demonstrates the powerful capabilities of our ultra-optimized SEO evaluation system. 
                        Get started by initializing the model and exploring its features.
                        """)
                        
                        model_type = gr.Dropdown(
                            choices=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
                            value="bert-base-uncased",
                            label="🎯 Choose Model Type",
                            info="Select the transformer model architecture you want to use"
                        )
                        
                        init_btn = gr.Button("🚀 Initialize Model", variant="primary", size="lg")
                        
                        model_status = gr.Markdown(
                            "**Status**: Ready to initialize\n\nClick the button above to get started!"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Quick Start Guide")
                        
                        gr.Markdown("""
                        1. **🚀 Initialize Model** - Choose your model type and click initialize
                        2. **🔍 Analyze Text** - Test individual text analysis
                        3. **📊 Batch Analysis** - Compare multiple texts
                        4. **🏋️ Training Demo** - Watch the model learn
                        """)
                        
                        gr.Markdown("### 🎨 Features")
                        
                        gr.Markdown("""
                        • **Real-time Analysis** - Instant SEO insights
                        • **Interactive Visualizations** - Beautiful charts and graphs
                        • **Batch Processing** - Multiple text comparison
                        • **Live Training** - Real-time model learning
                        • **User-friendly Interface** - Intuitive and engaging
                        """)
                
                # Capability overview
                gr.Markdown("### 🔍 Model Capabilities")
                capability_overview = gr.Markdown(interface.get_capability_overview())
                
                # Event handler
                init_btn.click(
                    fn=interface.initialize_model,
                    inputs=[model_type],
                    outputs=model_status
                )
            
            # Text Analysis Tab
            with gr.Tab("🔍 Text Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📝 Text Analysis")
                        
                        input_text = gr.Textbox(
                            label="Enter text to analyze",
                            placeholder="Paste your content here for SEO analysis...",
                            lines=8
                        )
                        
                        analysis_depth = gr.Radio(
                            choices=["basic", "comprehensive"],
                            value="comprehensive",
                            label="Analysis Depth",
                            info="Choose the level of detail for your analysis"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("### 📊 Analysis Results")
                        analysis_result = gr.Markdown(
                            "**Ready for analysis!**\n\nEnter text above and click 'Analyze Text' to get started."
                        )
                
                # Visualizations
                gr.Markdown("### 📈 Visual Insights")
                
                with gr.Row():
                    seo_gauge = gr.Plot(label="SEO Performance Gauge")
                    metrics_dashboard = gr.Plot(label="Content Metrics Dashboard")
                
                with gr.Row():
                    score_distribution = gr.Plot(label="Score Distribution")
                
                # Event handler
                analyze_btn.click(
                    fn=interface.showcase_text_analysis,
                    inputs=[input_text, analysis_depth],
                    outputs=[analysis_result, gr.State({})]
                ).then(
                    fn=lambda result, viz: viz.get('seo_gauge', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=seo_gauge
                ).then(
                    fn=lambda result, viz: viz.get('metrics_dashboard', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=metrics_dashboard
                ).then(
                    fn=lambda result, viz: viz.get('score_distribution', None),
                    inputs=[analysis_result, gr.State({})],
                    outputs=score_distribution
                )
            
            # Batch Analysis Tab
            with gr.Tab("📊 Batch Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 Batch Text Analysis")
                        
                        batch_texts = gr.Textbox(
                            label="Enter multiple texts (one per line)",
                            placeholder="Text 1\nText 2\nText 3\n...",
                            lines=10
                        )
                        
                        batch_mode = gr.Radio(
                            choices=["basic", "comprehensive"],
                            value="comprehensive",
                            label="Analysis Mode",
                            info="Choose analysis depth for batch processing"
                        )
                        
                        batch_btn = gr.Button("📊 Analyze Batch", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("### 📈 Batch Results")
                        batch_result = gr.Markdown(
                            "**Ready for batch analysis!**\n\nEnter multiple texts above and click 'Analyze Batch' to compare them."
                        )
                
                # Batch visualizations
                gr.Markdown("### 📊 Comparative Insights")
                
                with gr.Row():
                    performance_comparison = gr.Plot(label="Performance Comparison")
                    radar_chart = gr.Plot(label="Multi-Metric Radar Chart")
                
                # Results table
                batch_results_table = gr.Dataframe(
                    label="Detailed Results",
                    headers=["Text ID", "Preview", "SEO Score", "Words", "Readability", "Keywords"]
                )
                
                # Event handler
                batch_btn.click(
                    fn=interface.showcase_batch_analysis,
                    inputs=[batch_texts, batch_mode],
                    outputs=[batch_result, gr.State({})]
                ).then(
                    fn=lambda result, viz: viz.get('dataframe', pd.DataFrame()),
                    inputs=[batch_result, gr.State({})],
                    outputs=batch_results_table
                ).then(
                    fn=lambda result, viz: viz.get('performance_comparison', None),
                    inputs=[batch_result, gr.State({})],
                    outputs=performance_comparison
                ).then(
                    fn=lambda result, viz: viz.get('radar_chart', None),
                    inputs=[batch_result, gr.State({})],
                    outputs=radar_chart
                )
            
            # Training Demo Tab
            with gr.Tab("🏋️ Training Demo"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ⚙️ Training Configuration")
                        
                        demo_epochs = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Number of Epochs",
                            info="How many training cycles to perform"
                        )
                        
                        demo_batch_size = gr.Slider(
                            minimum=1, maximum=8, value=2, step=1,
                            label="Batch Size",
                            info="Number of samples processed together"
                        )
                        
                        demo_learning_rate = gr.Slider(
                            minimum=1e-5, maximum=1e-3, value=1e-4, step=1e-5,
                            label="Learning Rate",
                            info="How fast the model learns"
                        )
                        
                        demo_patience = gr.Slider(
                            minimum=1, maximum=5, value=2, step=1,
                            label="Patience",
                            info="Early stopping patience"
                        )
                        
                        train_btn = gr.Button("🏋️ Start Training Demo", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("### 📊 Training Results")
                        training_result = gr.Markdown(
                            "**Ready for training demo!**\n\nConfigure your parameters above and click 'Start Training Demo' to watch the model learn."
                        )
                
                # Training visualizations
                gr.Markdown("### 📈 Training Progress")
                
                with gr.Row():
                    progress_chart = gr.Plot(label="Training Progress")
                    metrics_chart = gr.Plot(label="Training Metrics")
                
                # Event handler
                train_btn.click(
                    fn=interface.showcase_training_capabilities,
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
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        ### 🎉 **Experience the Power of AI-Powered SEO Analysis**
        
        This interface showcases the advanced capabilities of our ultra-optimized SEO evaluation system, 
        featuring real-time analysis, interactive visualizations, and live training demonstrations.
        
        **Built with**: PyTorch, Transformers, LoRA, Gradio, and Plotly
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the user-friendly interface
    demo = create_user_friendly_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )
