from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import gradio as gr
import asyncio
import logging
import traceback
from agents.backend.onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService

from typing import Any, List, Dict, Optional
DEBUG = False  # Set to True to show tracebacks in UI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_text_demo")

try:
    finetuning_service = OptimizedFineTuningService()
except Exception as e:
    logger.error(f"Error initializing fine-tuning service: {e}\n{traceback.format_exc()}")
    finetuning_service = None

def run_async(coro) -> Any:
    try:
        return asyncio.get_event_loop().run_until_complete(coro)
    except Exception as e:
        logger.error(f"Async error: {e}\n{traceback.format_exc()}")
        if DEBUG:
            return f"❌ Async error: {e}\n\n{traceback.format_exc()}"
        return "❌ Internal async error. Please try again later."

def generate_ad(prompt, user_id, max_length, temperature) -> Any:
    # Input validation
    if not prompt or not prompt.strip():
        return "❌ Please enter a non-empty prompt."
    try:
        user_id = int(user_id)
        if user_id < 1:
            return "❌ User ID must be a positive integer."
    except Exception:
        return "❌ Invalid User ID."
    try:
        max_length = int(max_length)
        if not (50 <= max_length <= 512):
            return "❌ Max Length must be between 50 and 512."
    except Exception:
        return "❌ Invalid Max Length."
    try:
        temperature = float(temperature)
        if not (0.1 <= temperature <= 1.5):
            return "❌ Temperature must be between 0.1 and 1.5."
    except Exception:
        return "❌ Invalid Temperature."
    # Model inference with error handling
    if finetuning_service is None:
        return "❌ Model service is not available. Check server logs."
    try:
        result = run_async(finetuning_service.generate_with_finetuned_model(
            user_id=user_id,
            prompt=prompt,
            base_model_name="microsoft/DialoGPT-medium",
            max_length=max_length,
            temperature=temperature
        ))
        if not result or not isinstance(result, str):
            return "❌ Model did not return a valid result."
        return result
    except Exception as e:
        logger.error(f"Error during generation: {e}\n{traceback.format_exc()}")
        if DEBUG:
            return f"❌ Error during generation: {e}\n\n{traceback.format_exc()}"
        return "❌ Error during generation. Please try again later."

iface = gr.Interface(
    fn=generate_ad,
    inputs=[
        gr.Textbox(label="Ad Prompt", placeholder="Describe your product or campaign..."),
        gr.Number(label="User ID", value=1),
        gr.Slider(50, 512, value=200, label="Max Length"),
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
    ],
    outputs=gr.Markdown(label="Generated Ad"),
    title="Fine-tuned Ad Generator",
    description="Generate creative ads using your fine-tuned LoRA/P-tuning model. Enter your prompt and parameters below.",
    allow_flagging="never"
)

match __name__:
    case "__main__":
    iface.launch() 