"""
Text Generation Demo with Gradio
==================================

Interactive demo for text generation using transformers.
"""

import gradio as gr
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ..ml.llm.text_generator import TextGenerator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM modules not available")


# Global generator (lazy loading)
_generator: Optional[TextGenerator] = None


def get_generator(model_name: str = "gpt2") -> TextGenerator:
    """Get or create text generator."""
    global _generator
    if _generator is None or _generator.model_name != model_name:
        _generator = TextGenerator(model_name=model_name)
    return _generator


def generate_text(
    prompt: str,
    model_name: str,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_sequences: int
) -> str:
    """
    Generate text from prompt.
    
    Args:
        prompt: Input prompt
        model_name: Model name
        max_length: Max length
        temperature: Temperature
        top_p: Top-p
        top_k: Top-k
        num_sequences: Number of sequences
    
    Returns:
        Generated text
    """
    if not LLM_AVAILABLE:
        return "LLM modules not available. Please install transformers."
    
    try:
        generator = get_generator(model_name)
        
        generated = generator.generate(
            prompt=prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True
        )
        
        # Format output
        if num_sequences == 1:
            return generated[0]
        else:
            return "\n\n---\n\n".join([f"**Option {i+1}:**\n{g}" for i, g in enumerate(generated)])
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return f"Error: {str(e)}"


def generate_summary(text: str, model_name: str, max_length: int) -> str:
    """
    Generate summary.
    
    Args:
        text: Input text
        model_name: Model name
        max_length: Max length
    
    Returns:
        Summary
    """
    if not LLM_AVAILABLE:
        return "LLM modules not available. Please install transformers."
    
    try:
        generator = get_generator(model_name)
        summary = generator.generate_summary(text, max_length=max_length)
        return summary
    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        return f"Error: {str(e)}"


def create_text_generation_demo() -> gr.Blocks:
    """
    Create Gradio demo for text generation.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Artist Manager AI - Text Generation") as demo:
        gr.Markdown("# 🤖 Artist Manager AI - Text Generation Demo")
        gr.Markdown("Generate text using transformer models.")
        
        with gr.Tabs():
            # Text Generation Tab
            with gr.Tab("Text Generation"):
                gr.Markdown("### Generate Text from Prompt")
                
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["gpt2", "gpt2-medium", "distilgpt2"],
                        label="Model",
                        value="gpt2"
                    )
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=100,
                        step=50,
                        label="Max Length"
                    )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    value="The artist's schedule for today includes",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top-p"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k"
                    )
                
                num_sequences = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Sequences"
                )
                
                gen_btn = gr.Button("Generate", variant="primary")
                gen_output = gr.Textbox(label="Generated Text", lines=10)
                
                gen_btn.click(
                    generate_text,
                    inputs=[prompt, model_name, max_length, temperature, top_p, top_k, num_sequences],
                    outputs=gen_output
                )
            
            # Summary Tab
            with gr.Tab("Text Summarization"):
                gr.Markdown("### Summarize Text")
                
                summary_model = gr.Dropdown(
                    choices=["gpt2", "gpt2-medium"],
                    label="Model",
                    value="gpt2"
                )
                
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to summarize...",
                    lines=10
                )
                
                summary_length = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Summary Length"
                )
                
                summary_btn = gr.Button("Summarize", variant="primary")
                summary_output = gr.Textbox(label="Summary", lines=5)
                
                summary_btn.click(
                    generate_summary,
                    inputs=[input_text, summary_model, summary_length],
                    outputs=summary_output
                )
        
        gr.Markdown("---")
        gr.Markdown("Built with Transformers and Gradio")
    
    return demo


if __name__ == "__main__":
    demo = create_text_generation_demo()
    demo.launch(share=True)




