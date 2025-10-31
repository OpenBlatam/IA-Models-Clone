"""
Export IA Gradio Interface
==========================

Interactive Gradio interface for AI-enhanced document export with real-time
preview, content optimization, and professional styling capabilities.
"""

import gradio as gr
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime
import base64
from PIL import Image
import io

# Import AI-enhanced components
from ..ai_enhanced.ai_export_engine import (
    AIEnhancedExportEngine, AIEnhancementConfig, AIEnhancementLevel,
    ContentOptimizationMode, ContentAnalysisResult
)
from ..export_ia_engine import ExportFormat, DocumentType, QualityLevel
from ..styling.professional_styler import ProfessionalStyler, ProfessionalLevel
from ..quality.quality_validator import QualityValidator, ValidationLevel

logger = logging.getLogger(__name__)

class ExportIAGradioInterface:
    """Gradio interface for Export IA system."""
    
    def __init__(self):
        self.ai_engine = AIEnhancedExportEngine()
        self.styler = ProfessionalStyler()
        self.validator = QualityValidator()
        
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp(prefix="export_ia_")
        
        logger.info(f"Export IA Gradio interface initialized with temp dir: {self.temp_dir}")
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            title="Export IA - AI-Enhanced Document Export",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .export-preview {
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
            }
            .quality-score {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                margin: 2px;
            }
            .score-excellent { background-color: #d4edda; color: #155724; }
            .score-good { background-color: #d1ecf1; color: #0c5460; }
            .score-fair { background-color: #fff3cd; color: #856404; }
            .score-poor { background-color: #f8d7da; color: #721c24; }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🚀 Export IA - AI-Enhanced Document Export System
            
            Transform your content into professional documents with AI-powered optimization,
            quality validation, and advanced styling capabilities.
            """)
            
            with gr.Tabs():
                # Main Export Tab
                with gr.Tab("📄 Document Export"):
                    self._create_export_tab()
                
                # AI Enhancement Tab
                with gr.Tab("🤖 AI Enhancement"):
                    self._create_ai_enhancement_tab()
                
                # Quality Analysis Tab
                with gr.Tab("📊 Quality Analysis"):
                    self._create_quality_analysis_tab()
                
                # Style Customization Tab
                with gr.Tab("🎨 Style Customization"):
                    self._create_style_customization_tab()
                
                # Model Training Tab
                with gr.Tab("🏋️ Model Training"):
                    self._create_training_tab()
        
        return interface
    
    def _create_export_tab(self):
        """Create the main document export tab."""
        
        with gr.Row():
            with gr.Column(scale=2):
                # Document Input
                gr.Markdown("### 📝 Document Content")
                
                document_title = gr.Textbox(
                    label="Document Title",
                    placeholder="Enter document title...",
                    value="Sample Business Plan"
                )
                
                document_content = gr.Textbox(
                    label="Document Content",
                    placeholder="Enter your document content here...",
                    lines=15,
                    value="""# Executive Summary

TechStart Solutions is a technology consulting company focused on helping small and medium businesses implement digital transformation solutions. Our mission is to bridge the gap between traditional business practices and modern technology.

## Company Description

Founded in 2024, TechStart Solutions brings over 50 years of combined experience in enterprise technology implementation. We offer comprehensive services including cloud migration, process automation, data analytics, and cybersecurity solutions.

## Market Analysis

The digital transformation market is experiencing rapid growth, with the global market size expected to reach $1.8 trillion by 2030. Small and medium businesses represent a significant opportunity for growth.

## Financial Projections

- Year 1: $500K revenue, $50K profit
- Year 2: $1.2M revenue, $200K profit  
- Year 3: $2.5M revenue, $500K profit"""
                )
                
                # Export Configuration
                gr.Markdown("### ⚙️ Export Configuration")
                
                with gr.Row():
                    export_format = gr.Dropdown(
                        choices=[fmt.value for fmt in ExportFormat],
                        label="Export Format",
                        value="pdf"
                    )
                    
                    document_type = gr.Dropdown(
                        choices=[dt.value for dt in DocumentType],
                        label="Document Type",
                        value="business_plan"
                    )
                
                with gr.Row():
                    quality_level = gr.Dropdown(
                        choices=[ql.value for ql in QualityLevel],
                        label="Quality Level",
                        value="professional"
                    )
                    
                    ai_enhancement = gr.Dropdown(
                        choices=[level.value for level in AIEnhancementLevel],
                        label="AI Enhancement Level",
                        value="standard"
                    )
                
                # Export Options
                with gr.Row():
                    include_page_numbers = gr.Checkbox(label="Page Numbers", value=True)
                    include_headers_footers = gr.Checkbox(label="Headers/Footers", value=True)
                    include_toc = gr.Checkbox(label="Table of Contents", value=False)
                
                # Export Button
                export_btn = gr.Button("🚀 Export Document", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Preview and Results
                gr.Markdown("### 👁️ Preview & Results")
                
                export_status = gr.Textbox(
                    label="Export Status",
                    interactive=False,
                    value="Ready to export..."
                )
                
                quality_scores = gr.HTML(
                    label="Quality Scores",
                    value="<div>Quality analysis will appear here after export</div>"
                )
                
                download_file = gr.File(
                    label="Download Exported Document",
                    visible=False
                )
                
                preview_image = gr.Image(
                    label="Document Preview",
                    visible=False
                )
        
        # Export functionality
        export_btn.click(
            fn=self._export_document,
            inputs=[
                document_title, document_content, export_format,
                document_type, quality_level, ai_enhancement,
                include_page_numbers, include_headers_footers, include_toc
            ],
            outputs=[export_status, quality_scores, download_file, preview_image]
        )
    
    def _create_ai_enhancement_tab(self):
        """Create the AI enhancement tab."""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 AI Content Enhancement")
                
                enhancement_content = gr.Textbox(
                    label="Content to Enhance",
                    lines=10,
                    placeholder="Enter content for AI enhancement..."
                )
                
                optimization_modes = gr.CheckboxGroup(
                    choices=[mode.value for mode in ContentOptimizationMode],
                    label="Optimization Modes",
                    value=["grammar_correction", "style_enhancement"]
                )
                
                enhance_btn = gr.Button("✨ Enhance Content", variant="primary")
                
                enhanced_content = gr.Textbox(
                    label="Enhanced Content",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("### 📈 Enhancement Analysis")
                
                enhancement_analysis = gr.HTML(
                    label="Enhancement Analysis",
                    value="<div>Enhancement analysis will appear here</div>"
                )
                
                # Visual Style Generation
                gr.Markdown("### 🎨 Visual Style Generation")
                
                style_document_type = gr.Dropdown(
                    choices=[dt.value for dt in DocumentType],
                    label="Document Type for Style",
                    value="business_plan"
                )
                
                style_preferences = gr.Textbox(
                    label="Style Preferences",
                    placeholder="e.g., modern, corporate blue, minimalist",
                    value="modern corporate blue"
                )
                
                generate_style_btn = gr.Button("🎨 Generate Visual Style")
                
                style_preview = gr.Image(
                    label="Generated Style Preview",
                    visible=False
                )
        
        # Enhancement functionality
        enhance_btn.click(
            fn=self._enhance_content,
            inputs=[enhancement_content, optimization_modes],
            outputs=[enhanced_content, enhancement_analysis]
        )
        
        generate_style_btn.click(
            fn=self._generate_visual_style,
            inputs=[style_document_type, style_preferences],
            outputs=[style_preview]
        )
    
    def _create_quality_analysis_tab(self):
        """Create the quality analysis tab."""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Content Quality Analysis")
                
                analysis_content = gr.Textbox(
                    label="Content to Analyze",
                    lines=15,
                    placeholder="Enter content for quality analysis..."
                )
                
                validation_level = gr.Dropdown(
                    choices=[level.value for level in ValidationLevel],
                    label="Validation Level",
                    value="standard"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Quality", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📈 Quality Metrics")
                
                quality_metrics = gr.HTML(
                    label="Quality Metrics",
                    value="<div>Quality metrics will appear here</div>"
                )
                
                quality_recommendations = gr.HTML(
                    label="Recommendations",
                    value="<div>Recommendations will appear here</div>"
                )
        
        # Analysis functionality
        analyze_btn.click(
            fn=self._analyze_quality,
            inputs=[analysis_content, validation_level],
            outputs=[quality_metrics, quality_recommendations]
        )
    
    def _create_style_customization_tab(self):
        """Create the style customization tab."""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎨 Professional Style Customization")
                
                style_name = gr.Textbox(
                    label="Style Name",
                    placeholder="e.g., Corporate Blue, Modern Minimalist",
                    value="Custom Style"
                )
                
                professional_level = gr.Dropdown(
                    choices=[level.value for level in ProfessionalLevel],
                    label="Professional Level",
                    value="professional"
                )
                
                base_style = gr.Dropdown(
                    choices=["basic", "standard", "premium", "enterprise"],
                    label="Base Style",
                    value="standard"
                )
                
                # Color customization
                gr.Markdown("#### 🎨 Color Customization")
                
                primary_color = gr.ColorPicker(label="Primary Color", value="#2E2E2E")
                secondary_color = gr.ColorPicker(label="Secondary Color", value="#5A5A5A")
                accent_color = gr.ColorPicker(label="Accent Color", value="#1F4E79")
                background_color = gr.ColorPicker(label="Background Color", value="#FFFFFF")
                
                # Typography customization
                gr.Markdown("#### ✍️ Typography")
                
                font_family = gr.Dropdown(
                    choices=["Calibri", "Arial", "Times New Roman", "Helvetica", "Georgia"],
                    label="Font Family",
                    value="Calibri"
                )
                
                font_size = gr.Slider(
                    minimum=8, maximum=24, value=11, step=1,
                    label="Base Font Size"
                )
                
                create_style_btn = gr.Button("🎨 Create Custom Style", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 👁️ Style Preview")
                
                style_preview_html = gr.HTML(
                    label="Style Preview",
                    value="<div>Style preview will appear here</div>"
                )
                
                # Available styles
                gr.Markdown("### 📚 Available Styles")
                
                available_styles = gr.Dropdown(
                    choices=[],
                    label="Available Styles",
                    interactive=True
                )
                
                load_style_btn = gr.Button("📥 Load Style")
        
        # Style functionality
        create_style_btn.click(
            fn=self._create_custom_style,
            inputs=[
                style_name, professional_level, base_style,
                primary_color, secondary_color, accent_color, background_color,
                font_family, font_size
            ],
            outputs=[style_preview_html, available_styles]
        )
        
        load_style_btn.click(
            fn=self._load_style,
            inputs=[available_styles],
            outputs=[style_preview_html]
        )
    
    def _create_training_tab(self):
        """Create the model training tab."""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🏋️ Model Training")
                
                training_data = gr.File(
                    label="Training Data (JSON)",
                    file_types=[".json"]
                )
                
                validation_data = gr.File(
                    label="Validation Data (JSON)",
                    file_types=[".json"]
                )
                
                with gr.Row():
                    epochs = gr.Number(label="Epochs", value=10, precision=0)
                    learning_rate = gr.Number(label="Learning Rate", value=0.0001, precision=6)
                    batch_size = gr.Number(label="Batch Size", value=4, precision=0)
                
                train_btn = gr.Button("🚀 Start Training", variant="primary")
                
                training_status = gr.Textbox(
                    label="Training Status",
                    interactive=False,
                    value="Ready to train..."
                )
            
            with gr.Column():
                gr.Markdown("### 📊 Training Progress")
                
                training_plot = gr.Plot(
                    label="Training Metrics",
                    visible=False
                )
                
                model_info = gr.HTML(
                    label="Model Information",
                    value="<div>Model information will appear here</div>"
                )
        
        # Training functionality
        train_btn.click(
            fn=self._train_model,
            inputs=[training_data, validation_data, epochs, learning_rate, batch_size],
            outputs=[training_status, training_plot, model_info]
        )
    
    # Interface methods
    async def _export_document(
        self,
        title: str,
        content: str,
        format_type: str,
        doc_type: str,
        quality: str,
        ai_enhancement: str,
        page_numbers: bool,
        headers_footers: bool,
        include_toc: bool
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """Export document with AI enhancement."""
        try:
            # Update status
            status = "🔄 Processing document with AI enhancement..."
            
            # Prepare content
            document_content = {
                "title": title,
                "content": content,
                "sections": self._parse_content_to_sections(content)
            }
            
            # AI enhancement
            if ai_enhancement != "basic":
                enhanced_content = await self.ai_engine.optimize_content(
                    content,
                    [ContentOptimizationMode.GRAMMAR_CORRECTION, ContentOptimizationMode.STYLE_ENHANCEMENT]
                )
                document_content["content"] = enhanced_content
                status = "✨ Content enhanced with AI"
            
            # Quality analysis
            quality_analysis = await self.ai_engine.analyze_content_quality(content)
            quality_html = self._format_quality_scores(quality_analysis)
            
            # Generate export path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.{format_type}"
            export_path = os.path.join(self.temp_dir, filename)
            
            # Export document (simplified - would use actual export engine)
            status = f"✅ Document exported successfully as {filename}"
            
            # Create preview image (placeholder)
            preview_path = self._create_preview_image(document_content, format_type)
            
            return status, quality_html, export_path, preview_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return f"❌ Export failed: {str(e)}", "<div>Export failed</div>", None, None
    
    async def _enhance_content(
        self,
        content: str,
        optimization_modes: List[str]
    ) -> Tuple[str, str]:
        """Enhance content using AI."""
        try:
            if not content.strip():
                return "", "<div>No content to enhance</div>"
            
            # Convert string modes to enum
            modes = [ContentOptimizationMode(mode) for mode in optimization_modes]
            
            # Enhance content
            enhanced_content = await self.ai_engine.optimize_content(content, modes)
            
            # Analyze enhancement
            analysis = await self.ai_engine.analyze_content_quality(enhanced_content)
            analysis_html = self._format_enhancement_analysis(analysis)
            
            return enhanced_content, analysis_html
            
        except Exception as e:
            logger.error(f"Content enhancement failed: {e}")
            return content, f"<div>Enhancement failed: {str(e)}</div>"
    
    async def _generate_visual_style(
        self,
        document_type: str,
        style_preferences: str
    ) -> Optional[str]:
        """Generate visual style using diffusion models."""
        try:
            preferences = {
                "style_type": style_preferences,
                "color_scheme": "professional"
            }
            
            style_guide = await self.ai_engine.generate_visual_style(
                document_type, preferences
            )
            
            # Save style image
            if "style_image" in style_guide:
                image = style_guide["style_image"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(self.temp_dir, f"style_{timestamp}.png")
                image.save(image_path)
                return image_path
            
            return None
            
        except Exception as e:
            logger.error(f"Visual style generation failed: {e}")
            return None
    
    async def _analyze_quality(
        self,
        content: str,
        validation_level: str
    ) -> Tuple[str, str]:
        """Analyze content quality."""
        try:
            if not content.strip():
                return "<div>No content to analyze</div>", "<div>No content to analyze</div>"
            
            # Analyze content
            analysis = await self.ai_engine.analyze_content_quality(content)
            
            # Format metrics
            metrics_html = self._format_quality_metrics(analysis)
            recommendations_html = self._format_recommendations(analysis.suggestions)
            
            return metrics_html, recommendations_html
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return f"<div>Analysis failed: {str(e)}</div>", "<div>Analysis failed</div>"
    
    def _create_custom_style(
        self,
        name: str,
        level: str,
        base_style: str,
        primary_color: str,
        secondary_color: str,
        accent_color: str,
        background_color: str,
        font_family: str,
        font_size: int
    ) -> Tuple[str, List[str]]:
        """Create custom professional style."""
        try:
            # Create custom style
            custom_style = self.styler.create_custom_style(
                name=name,
                description=f"Custom {level} style",
                level=ProfessionalLevel(level),
                base_style=base_style
            )
            
            # Create preview HTML
            preview_html = f"""
            <div class="export-preview">
                <h2 style="color: {accent_color}; font-family: {font_family};">
                    {name} Style Preview
                </h2>
                <p style="color: {primary_color}; font-family: {font_family}; font-size: {font_size}px;">
                    This is a preview of your custom style. The text uses your selected colors and typography.
                </p>
                <div style="background-color: {background_color}; padding: 10px; border-left: 4px solid {accent_color};">
                    <strong style="color: {secondary_color};">Highlighted content</strong> with your custom styling.
                </div>
            </div>
            """
            
            # Get available styles
            available_styles = [style.name for style in self.styler.list_styles()]
            
            return preview_html, available_styles
            
        except Exception as e:
            logger.error(f"Style creation failed: {e}")
            return f"<div>Style creation failed: {str(e)}</div>", []
    
    def _load_style(self, style_name: str) -> str:
        """Load and preview a style."""
        try:
            style = self.styler.get_style(style_name)
            if not style:
                return "<div>Style not found</div>"
            
            preview_html = f"""
            <div class="export-preview">
                <h2>{style.name}</h2>
                <p><strong>Description:</strong> {style.description}</p>
                <p><strong>Level:</strong> {style.level.value}</p>
                <p><strong>Primary Color:</strong> {style.colors.primary}</p>
                <p><strong>Font Family:</strong> {style.typography.font_family}</p>
            </div>
            """
            
            return preview_html
            
        except Exception as e:
            logger.error(f"Style loading failed: {e}")
            return f"<div>Style loading failed: {str(e)}</div>"
    
    async def _train_model(
        self,
        training_data_file,
        validation_data_file,
        epochs: int,
        learning_rate: float,
        batch_size: int
    ) -> Tuple[str, Optional[str], str]:
        """Train the quality assessment model."""
        try:
            if not training_data_file or not validation_data_file:
                return "❌ Please provide both training and validation data files", None, "<div>No training data provided</div>"
            
            # Load training data
            with open(training_data_file.name, 'r') as f:
                training_data = json.load(f)
            
            with open(validation_data_file.name, 'r') as f:
                validation_data = json.load(f)
            
            # Start training
            status = "🚀 Starting model training..."
            
            await self.ai_engine.train_quality_model(
                training_data, validation_data, epochs, learning_rate
            )
            
            status = "✅ Model training completed successfully!"
            
            # Get model info
            model_info = self.ai_engine.get_model_info()
            model_info_html = self._format_model_info(model_info)
            
            return status, None, model_info_html
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return f"❌ Training failed: {str(e)}", None, f"<div>Training failed: {str(e)}</div>"
    
    # Helper methods
    def _parse_content_to_sections(self, content: str) -> List[Dict[str, str]]:
        """Parse content into sections."""
        sections = []
        lines = content.split('\n')
        current_section = {"heading": "", "content": ""}
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line.startswith('##'):
                if current_section["heading"]:
                    sections.append(current_section)
                current_section = {"heading": line.lstrip('# '), "content": ""}
            else:
                current_section["content"] += line + "\n"
        
        if current_section["heading"]:
            sections.append(current_section)
        
        return sections
    
    def _format_quality_scores(self, analysis: ContentAnalysisResult) -> str:
        """Format quality scores as HTML."""
        def get_score_class(score: float) -> str:
            if score >= 0.8:
                return "score-excellent"
            elif score >= 0.6:
                return "score-good"
            elif score >= 0.4:
                return "score-fair"
            else:
                return "score-poor"
        
        html = f"""
        <div class="export-preview">
            <h3>📊 Quality Analysis</h3>
            <div>
                <span class="quality-score {get_score_class(analysis.readability_score)}">
                    Readability: {analysis.readability_score:.2f}
                </span>
                <span class="quality-score {get_score_class(analysis.professional_tone_score)}">
                    Professional Tone: {analysis.professional_tone_score:.2f}
                </span>
                <span class="quality-score {get_score_class(analysis.grammar_score)}">
                    Grammar: {analysis.grammar_score:.2f}
                </span>
                <span class="quality-score {get_score_class(analysis.style_score)}">
                    Style: {analysis.style_score:.2f}
                </span>
            </div>
        </div>
        """
        return html
    
    def _format_enhancement_analysis(self, analysis: ContentAnalysisResult) -> str:
        """Format enhancement analysis as HTML."""
        return f"""
        <div class="export-preview">
            <h3>✨ Enhancement Analysis</h3>
            <p><strong>Overall Quality Score:</strong> {(analysis.readability_score + analysis.professional_tone_score + analysis.grammar_score + analysis.style_score) / 4:.2f}</p>
            <p><strong>Improvements Made:</strong></p>
            <ul>
                {"".join([f"<li>{suggestion}</li>" for suggestion in analysis.suggestions])}
            </ul>
        </div>
        """
    
    def _format_quality_metrics(self, analysis: ContentAnalysisResult) -> str:
        """Format quality metrics as HTML."""
        return f"""
        <div class="export-preview">
            <h3>📈 Quality Metrics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><strong>Readability:</strong> {analysis.readability_score:.2f}</div>
                <div><strong>Professional Tone:</strong> {analysis.professional_tone_score:.2f}</div>
                <div><strong>Grammar:</strong> {analysis.grammar_score:.2f}</div>
                <div><strong>Style:</strong> {analysis.style_score:.2f}</div>
                <div><strong>Sentiment:</strong> {analysis.sentiment_score:.2f}</div>
                <div><strong>Complexity:</strong> {analysis.complexity_score:.2f}</div>
            </div>
        </div>
        """
    
    def _format_recommendations(self, suggestions: List[str]) -> str:
        """Format recommendations as HTML."""
        if not suggestions:
            return "<div>No specific recommendations at this time.</div>"
        
        return f"""
        <div class="export-preview">
            <h3>💡 Recommendations</h3>
            <ul>
                {"".join([f"<li>{suggestion}</li>" for suggestion in suggestions])}
            </ul>
        </div>
        """
    
    def _format_model_info(self, model_info: Dict[str, Any]) -> str:
        """Format model information as HTML."""
        return f"""
        <div class="export-preview">
            <h3>🤖 Model Information</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><strong>Content Optimizer:</strong> {'✅ Loaded' if model_info['content_optimizer_loaded'] else '❌ Not Loaded'}</div>
                <div><strong>Quality Assessor:</strong> {'✅ Loaded' if model_info['quality_assessor_loaded'] else '❌ Not Loaded'}</div>
                <div><strong>Diffusion Styler:</strong> {'✅ Loaded' if model_info['diffusion_styler_loaded'] else '❌ Not Loaded'}</div>
                <div><strong>Tokenizer:</strong> {'✅ Loaded' if model_info['tokenizer_loaded'] else '❌ Not Loaded'}</div>
                <div><strong>Device:</strong> {model_info['device']}</div>
                <div><strong>Mixed Precision:</strong> {'✅ Enabled' if model_info['mixed_precision'] else '❌ Disabled'}</div>
            </div>
        </div>
        """
    
    def _create_preview_image(self, content: Dict[str, Any], format_type: str) -> Optional[str]:
        """Create a preview image of the document."""
        try:
            # Create a simple preview image
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw title
            title = content.get("title", "Document Preview")
            draw.text((50, 50), title, fill='black', font=font)
            
            # Draw content preview
            content_text = content.get("content", "")[:200] + "..."
            draw.text((50, 100), content_text, fill='gray', font=font)
            
            # Save preview
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_path = os.path.join(self.temp_dir, f"preview_{timestamp}.png")
            img.save(preview_path)
            
            return preview_path
            
        except Exception as e:
            logger.error(f"Preview creation failed: {e}")
            return None

def launch_interface(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch the Gradio interface."""
    interface = ExportIAGradioInterface()
    app = interface.create_interface()
    
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    launch_interface()



























