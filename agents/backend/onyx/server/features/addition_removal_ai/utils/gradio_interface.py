"""
Gradio Interface for Addition Removal AI
"""

import gradio as gr
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AdditionRemovalGradioInterface:
    """Gradio interface for Addition Removal AI"""
    
    def __init__(self, editor, enhanced_ai_engine=None):
        """
        Initialize Gradio interface
        
        Args:
            editor: ContentEditor instance
            enhanced_ai_engine: EnhancedAIEngine instance
        """
        self.editor = editor
        self.ai_engine = enhanced_ai_engine
    
    def add_content(self, content: str, addition: str, position: str) -> Dict[str, Any]:
        """Add content through interface"""
        try:
            result = self.editor.add(
                content=content,
                addition=addition,
                position=position
            )
            return {
                "result": result.get("result", ""),
                "success": result.get("success", False),
                "message": "Content added successfully" if result.get("success") else "Failed to add content"
            }
        except Exception as e:
            return {"result": "", "success": False, "message": str(e)}
    
    def remove_content(self, content: str, pattern: str) -> Dict[str, Any]:
        """Remove content through interface"""
        try:
            result = self.editor.remove(content=content, pattern=pattern)
            return {
                "result": result.get("result", ""),
                "success": result.get("success", False),
                "message": "Content removed successfully" if result.get("success") else "Failed to remove content"
            }
        except Exception as e:
            return {"result": "", "success": False, "message": str(e)}
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content with AI"""
        if not self.ai_engine:
            return {"error": "AI engine not available"}
        
        try:
            analysis = self.ai_engine.analyze_content(content)
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    def generate_content(self, prompt: str, max_length: int = 100) -> str:
        """Generate content with AI"""
        if not self.ai_engine:
            return "AI engine not available"
        
        try:
            generated = self.ai_engine.generate_content(prompt, max_length=max_length)
            return generated
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="Addition Removal AI") as interface:
            gr.Markdown("# Addition Removal AI - Content Editor")
            
            with gr.Tabs():
                # Add Content Tab
                with gr.Tab("Add Content"):
                    with gr.Row():
                        with gr.Column():
                            content_input = gr.Textbox(
                                label="Original Content",
                                lines=10,
                                placeholder="Enter your content here..."
                            )
                            addition_input = gr.Textbox(
                                label="Content to Add",
                                lines=5,
                                placeholder="Enter content to add..."
                            )
                            position_select = gr.Dropdown(
                                choices=["start", "end", "before", "after", "replace"],
                                value="end",
                                label="Position"
                            )
                            add_btn = gr.Button("Add Content", variant="primary")
                        
                        with gr.Column():
                            add_result = gr.Textbox(
                                label="Result",
                                lines=15,
                                interactive=False
                            )
                            add_message = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                    
                    add_btn.click(
                        fn=self.add_content,
                        inputs=[content_input, addition_input, position_select],
                        outputs=[add_result, add_message]
                    )
                
                # Remove Content Tab
                with gr.Tab("Remove Content"):
                    with gr.Row():
                        with gr.Column():
                            remove_content_input = gr.Textbox(
                                label="Content",
                                lines=10,
                                placeholder="Enter content..."
                            )
                            pattern_input = gr.Textbox(
                                label="Pattern to Remove",
                                lines=3,
                                placeholder="Enter pattern or text to remove..."
                            )
                            remove_btn = gr.Button("Remove Content", variant="primary")
                        
                        with gr.Column():
                            remove_result = gr.Textbox(
                                label="Result",
                                lines=15,
                                interactive=False
                            )
                            remove_message = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                    
                    remove_btn.click(
                        fn=self.remove_content,
                        inputs=[remove_content_input, pattern_input],
                        outputs=[remove_result, remove_message]
                    )
                
                # AI Analysis Tab
                if self.ai_engine:
                    with gr.Tab("AI Analysis"):
                        with gr.Row():
                            with gr.Column():
                                analyze_input = gr.Textbox(
                                    label="Content to Analyze",
                                    lines=10,
                                    placeholder="Enter content for AI analysis..."
                                )
                                analyze_btn = gr.Button("Analyze", variant="primary")
                            
                            with gr.Column():
                                analyze_result = gr.JSON(
                                    label="Analysis Results"
                                )
                        
                        analyze_btn.click(
                            fn=self.analyze_content,
                            inputs=analyze_input,
                            outputs=analyze_result
                        )
                    
                    # AI Generation Tab
                    with gr.Tab("AI Generation"):
                        with gr.Row():
                            with gr.Column():
                                generate_prompt = gr.Textbox(
                                    label="Prompt",
                                    lines=5,
                                    placeholder="Enter prompt for content generation..."
                                )
                                max_length_slider = gr.Slider(
                                    minimum=10,
                                    maximum=500,
                                    value=100,
                                    step=10,
                                    label="Max Length"
                                )
                                generate_btn = gr.Button("Generate", variant="primary")
                            
                            with gr.Column():
                                generate_result = gr.Textbox(
                                    label="Generated Content",
                                    lines=15,
                                    interactive=False
                                )
                        
                        generate_btn.click(
                            fn=self.generate_content,
                            inputs=[generate_prompt, max_length_slider],
                            outputs=generate_result
                        )
        
        return interface
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False
    ):
        """Launch Gradio interface"""
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )


def create_gradio_app(editor, enhanced_ai_engine=None) -> gr.Blocks:
    """
    Create Gradio app
    
    Args:
        editor: ContentEditor instance
        enhanced_ai_engine: EnhancedAIEngine instance
        
    Returns:
        Gradio Blocks interface
    """
    interface = AdditionRemovalGradioInterface(editor, enhanced_ai_engine)
    return interface.create_interface()

