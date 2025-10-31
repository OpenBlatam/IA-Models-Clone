#!/usr/bin/env python3
"""
Comprehensive Error Handling and Input Validation for Gradio Apps
Robust error handling with user-friendly messages and graceful recovery
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import logging
import traceback
import re
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradioErrorHandler:
    """Comprehensive error handling for Gradio applications."""
    
    def __init__(self):
        self.error_log = []
        self.validation_rules = self._setup_validation_rules()
        self.error_messages = self._setup_error_messages()
        
    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup validation rules for different input types."""
        return {
            'text': {
                'min_length': 10,
                'max_length': 10000,
                'allowed_chars': re.compile(r'^[a-zA-Z0-9\s\.,!?;:\'\"()-]+$'),
                'forbidden_patterns': [
                    re.compile(r'<script.*?</script>', re.IGNORECASE),
                    re.compile(r'javascript:', re.IGNORECASE),
                    re.compile(r'<iframe.*?</iframe>', re.IGNORECASE)
                ]
            },
            'url': {
                'pattern': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
                'max_length': 500
            },
            'email': {
                'pattern': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
                'max_length': 254
            },
            'number': {
                'min_value': -1e6,
                'max_value': 1e6,
                'allow_negative': True,
                'allow_zero': True
            },
            'file': {
                'max_size_mb': 50,
                'allowed_extensions': ['.txt', '.csv', '.json', '.md'],
                'forbidden_extensions': ['.exe', '.bat', '.sh', '.py', '.js']
            }
        }
    
    def _setup_error_messages(self) -> Dict[str, str]:
        """Setup user-friendly error messages."""
        return {
            'validation_error': "❌ **Input Validation Error**\n\n{message}\n\n**Please correct the input and try again.**",
            'model_error': "🚫 **Model Error**\n\n{message}\n\n**Troubleshooting:**\n• Check if the model is properly initialized\n• Verify input format and size\n• Try with different parameters",
            'system_error': "💥 **System Error**\n\n{message}\n\n**What happened:** An unexpected error occurred\n**What to do:** Please try again or contact support if the problem persists",
            'memory_error': "💾 **Memory Error**\n\n{message}\n\n**Solutions:**\n• Reduce input size\n• Close other applications\n• Restart the interface",
            'timeout_error': "⏰ **Timeout Error**\n\n{message}\n\n**Solutions:**\n• Reduce input complexity\n• Check system performance\n• Try again later",
            'network_error': "🌐 **Network Error**\n\n{message}\n\n**Solutions:**\n• Check internet connection\n• Verify server status\n• Try again in a moment"
        }
    
    def validate_text_input(self, text: str, input_name: str = "text") -> Tuple[bool, Optional[str]]:
        """Validate text input with comprehensive checks."""
        try:
            # Check if text is provided
            if not text or not text.strip():
                return False, f"'{input_name}' cannot be empty. Please provide some content."
            
            text = text.strip()
            
            # Check length
            if len(text) < self.validation_rules['text']['min_length']:
                return False, f"'{input_name}' is too short. Minimum {self.validation_rules['text']['min_length']} characters required."
            
            if len(text) > self.validation_rules['text']['max_length']:
                return False, f"'{input_name}' is too long. Maximum {self.validation_rules['text']['max_length']} characters allowed."
            
            # Check for forbidden patterns
            for pattern in self.validation_rules['text']['forbidden_patterns']:
                if pattern.search(text):
                    return False, f"'{input_name}' contains forbidden content. Please remove any scripts or unsafe HTML."
            
            # Check character set (optional - can be disabled for international content)
            # if not self.validation_rules['text']['allowed_chars'].match(text):
            #     return False, f"'{input_name}' contains unsupported characters."
            
            return True, None
            
        except Exception as e:
            logger.error(f"Text validation error: {e}")
            return False, f"Text validation failed: {str(e)}"
    
    def validate_url_input(self, url: str) -> Tuple[bool, Optional[str]]:
        """Validate URL input."""
        try:
            if not url or not url.strip():
                return False, "URL cannot be empty."
            
            url = url.strip()
            
            if len(url) > self.validation_rules['url']['max_length']:
                return False, f"URL is too long. Maximum {self.validation_rules['url']['max_length']} characters allowed."
            
            if not self.validation_rules['url']['pattern'].match(url):
                return False, "Invalid URL format. Please provide a valid HTTP/HTTPS URL."
            
            return True, None
            
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False, f"URL validation failed: {str(e)}"
    
    def validate_email_input(self, email: str) -> Tuple[bool, Optional[str]]:
        """Validate email input."""
        try:
            if not email or not email.strip():
                return False, "Email cannot be empty."
            
            email = email.strip()
            
            if len(email) > self.validation_rules['email']['max_length']:
                return False, f"Email is too long. Maximum {self.validation_rules['email']['max_length']} characters allowed."
            
            if not self.validation_rules['email']['pattern'].match(email):
                return False, "Invalid email format. Please provide a valid email address."
            
            return True, None
            
        except Exception as e:
            logger.error(f"Email validation error: {e}")
            return False, f"Email validation failed: {str(e)}"
    
    def validate_number_input(self, number: Union[int, float], input_name: str = "number") -> Tuple[bool, Optional[str]]:
        """Validate number input."""
        try:
            if number is None:
                return False, f"'{input_name}' cannot be empty."
            
            # Check if it's a valid number
            if not isinstance(number, (int, float)) or np.isnan(number) or np.isinf(number):
                return False, f"'{input_name}' must be a valid number."
            
            # Check range
            if number < self.validation_rules['number']['min_value']:
                return False, f"'{input_name}' is too small. Minimum value: {self.validation_rules['number']['min_value']}"
            
            if number > self.validation_rules['number']['max_value']:
                return False, f"'{input_name}' is too large. Maximum value: {self.validation_rules['number']['max_value']}"
            
            # Check negative numbers
            if not self.validation_rules['number']['allow_negative'] and number < 0:
                return False, f"'{input_name}' cannot be negative."
            
            # Check zero
            if not self.validation_rules['number']['allow_zero'] and number == 0:
                return False, f"'{input_name}' cannot be zero."
            
            return True, None
            
        except Exception as e:
            logger.error(f"Number validation error: {e}")
            return False, f"Number validation failed: {str(e)}"
    
    def validate_file_input(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate file input."""
        try:
            if not file_path:
                return False, "No file selected."
            
            file_path = Path(file_path)
            
            # Check file extension
            if file_path.suffix.lower() in self.validation_rules['file']['forbidden_extensions']:
                return False, f"File type '{file_path.suffix}' is not allowed for security reasons."
            
            if file_path.suffix.lower() not in self.validation_rules['file']['allowed_extensions']:
                return False, f"File type '{file_path.suffix}' is not supported. Allowed types: {', '.join(self.validation_rules['file']['allowed_extensions'])}"
            
            # Check file size
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.validation_rules['file']['max_size_mb']:
                    return False, f"File is too large ({file_size_mb:.1f} MB). Maximum size: {self.validation_rules['file']['max_size_mb']} MB"
            
            return True, None
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"File validation failed: {str(e)}"
    
    def validate_model_state(self, model: Any, model_name: str = "model") -> Tuple[bool, Optional[str]]:
        """Validate model state and readiness."""
        try:
            if model is None:
                return False, f"{model_name} is not initialized. Please initialize the model first."
            
            # Check if model has required attributes
            required_attrs = ['forward', 'parameters']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    return False, f"{model_name} is missing required attribute '{attr}'."
            
            # Check if model is on the correct device
            if hasattr(model, 'device'):
                device = next(model.parameters()).device
                if device.type == 'cuda' and not torch.cuda.is_available():
                    return False, f"{model_name} is on CUDA device but CUDA is not available."
            
            return True, None
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False, f"Model validation failed: {str(e)}"
    
    def handle_validation_error(self, error_type: str, message: str) -> str:
        """Handle validation errors with user-friendly messages."""
        error_msg = self.error_messages.get(error_type, self.error_messages['validation_error'])
        return error_msg.format(message=message)
    
    def handle_model_error(self, error: Exception, context: str = "") -> str:
        """Handle model-related errors."""
        error_type = type(error).__name__
        
        if "CUDA out of memory" in str(error):
            return self.error_messages['memory_error'].format(
                message=f"GPU memory is insufficient for this operation.\n**Context**: {context}"
            )
        elif "timeout" in str(error).lower():
            return self.error_messages['timeout_error'].format(
                message=f"Operation timed out.\n**Context**: {context}"
            )
        elif "connection" in str(error).lower() or "network" in str(error).lower():
            return self.error_messages['network_error'].format(
                message=f"Network connection issue.\n**Context**: {context}"
            )
        else:
            return self.error_messages['model_error'].format(
                message=f"**Error Type**: {error_type}\n**Error Message**: {str(error)}\n**Context**: {context}"
            )
    
    def handle_system_error(self, error: Exception, context: str = "") -> str:
        """Handle system-level errors."""
        error_type = type(error).__name__
        error_trace = traceback.format_exc()
        
        # Log the full error for debugging
        logger.error(f"System error in {context}: {error}")
        logger.error(f"Traceback: {error_trace}")
        
        # Store in error log
        self.error_log.append({
            'timestamp': datetime.now(),
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'traceback': error_trace
        })
        
        return self.error_messages['system_error'].format(
            message=f"**Error Type**: {error_type}\n**Error Message**: {str(error)}\n**Context**: {context}"
        )
    
    def safe_execute(self, func: callable, *args, **kwargs) -> Tuple[Any, Optional[str]]:
        """Safely execute a function with comprehensive error handling."""
        try:
            result = func(*args, **kwargs)
            return result, None
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = self.handle_model_error(e, "GPU memory operation")
            return None, error_msg
            
        except torch.cuda.CudaError as e:
            error_msg = self.handle_model_error(e, "CUDA operation")
            return None, error_msg
            
        except TimeoutError as e:
            error_msg = self.handle_model_error(e, "Operation timeout")
            return None, error_msg
            
        except ConnectionError as e:
            error_msg = self.handle_model_error(e, "Network connection")
            return None, error_msg
            
        except ValueError as e:
            if "validation" in str(e).lower():
                error_msg = self.handle_validation_error("validation_error", str(e))
            else:
                error_msg = self.handle_model_error(e, "Value error")
            return None, error_msg
            
        except Exception as e:
            error_msg = self.handle_system_error(e, f"Function execution: {func.__name__}")
            return None, error_msg
    
    def get_error_summary(self) -> str:
        """Get a summary of recent errors."""
        if not self.error_log:
            return "✅ No errors recorded."
        
        recent_errors = self.error_log[-5:]  # Last 5 errors
        
        summary = "📊 **Recent Error Summary**\n\n"
        for error in recent_errors:
            summary += f"**{error['timestamp'].strftime('%H:%M:%S')}** - {error['error_type']}\n"
            summary += f"Context: {error['context']}\n"
            summary += f"Message: {error['error_message'][:100]}...\n\n"
        
        return summary
    
    def clear_error_log(self):
        """Clear the error log."""
        self.error_log.clear()
        logger.info("Error log cleared")

class GradioInputValidator:
    """Input validation wrapper for Gradio components."""
    
    def __init__(self, error_handler: GradioErrorHandler):
        self.error_handler = error_handler
    
    def validate_textbox(self, text: str, input_name: str = "text", required: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate textbox input."""
        if not required and (not text or not text.strip()):
            return True, None  # Optional field can be empty
        
        return self.error_handler.validate_text_input(text, input_name)
    
    def validate_number(self, number: Union[int, float], input_name: str = "number", required: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate number input."""
        if not required and number is None:
            return True, None  # Optional field can be None
        
        return self.error_handler.validate_number_input(number, input_name)
    
    def validate_file(self, file_path: str, required: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate file input."""
        if not required and not file_path:
            return True, None  # Optional field can be empty
        
        return self.error_handler.validate_file_input(file_path)
    
    def validate_dropdown(self, selection: str, choices: List[str], required: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate dropdown selection."""
        if not required and not selection:
            return True, None  # Optional field can be empty
        
        if not selection:
            return False, "Please select an option from the dropdown."
        
        if selection not in choices:
            return False, f"Invalid selection '{selection}'. Please choose from the available options."
        
        return True, None
    
    def validate_slider(self, value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Tuple[bool, Optional[str]]:
        """Validate slider value."""
        if value < min_val or value > max_val:
            return False, f"Slider value {value} is out of range [{min_val}, {max_val}]."
        
        return True, None

class GradioErrorRecovery:
    """Error recovery strategies for Gradio applications."""
    
    def __init__(self, error_handler: GradioErrorHandler):
        self.error_handler = error_handler
    
    def suggest_recovery_action(self, error_type: str, context: str = "") -> str:
        """Suggest recovery actions based on error type."""
        suggestions = {
            'memory_error': [
                "💡 **Reduce input size** - Try with shorter text or smaller batch size",
                "💡 **Close other applications** - Free up system memory",
                "💡 **Restart the interface** - Clear memory cache",
                "💡 **Use CPU mode** - Switch to CPU if GPU memory is insufficient"
            ],
            'validation_error': [
                "💡 **Check input format** - Ensure input meets requirements",
                "💡 **Remove special characters** - Avoid unsupported symbols",
                "💡 **Adjust input length** - Stay within size limits",
                "💡 **Verify file type** - Use supported file formats"
            ],
            'model_error': [
                "💡 **Reinitialize model** - Click the initialize button again",
                "💡 **Check model status** - Verify model is ready",
                "💡 **Try different parameters** - Adjust batch size or model type",
                "💡 **Restart interface** - Fresh start often helps"
            ],
            'system_error': [
                "💡 **Refresh the page** - Reload the interface",
                "💡 **Check system resources** - Ensure sufficient memory/CPU",
                "💡 **Try again later** - System might be temporarily busy",
                "💡 **Contact support** - If problem persists"
            ]
        }
        
        action_list = suggestions.get(error_type, suggestions['system_error'])
        return "\n".join(action_list)
    
    def auto_retry_simple_operations(self, func: callable, max_retries: int = 3, *args, **kwargs) -> Tuple[Any, Optional[str]]:
        """Automatically retry simple operations."""
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result, None
            except Exception as e:
                if attempt == max_retries - 1:
                    return None, str(e)
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                continue
        
        return None, "Max retries exceeded"

def create_error_handling_interface():
    """Create a demo interface showcasing error handling capabilities."""
    
    error_handler = GradioErrorHandler()
    validator = GradioInputValidator(error_handler)
    recovery = GradioErrorRecovery(error_handler)
    
    def validate_demo_input(text: str, number: float, file_path: str, dropdown: str) -> Tuple[str, str]:
        """Demo function showing input validation."""
        errors = []
        
        # Validate text
        is_valid, error_msg = validator.validate_textbox(text, "Demo Text")
        if not is_valid:
            errors.append(f"Text: {error_msg}")
        
        # Validate number
        is_valid, error_msg = validator.validate_number(number, "Demo Number")
        if not is_valid:
            errors.append(f"Number: {error_msg}")
        
        # Validate file
        is_valid, error_msg = validator.validate_file(file_path)
        if not is_valid:
            errors.append(f"File: {error_msg}")
        
        # Validate dropdown
        choices = ["Option A", "Option B", "Option C"]
        is_valid, error_msg = validator.validate_dropdown(dropdown, choices)
        if not is_valid:
            errors.append(f"Dropdown: {error_msg}")
        
        if errors:
            error_summary = "\n".join(errors)
            recovery_actions = recovery.suggest_recovery_action("validation_error")
            return f"❌ **Validation Failed**\n\n{error_summary}", recovery_actions
        else:
            return "✅ **All inputs are valid!**", "🎉 Ready to proceed with processing."
    
    def simulate_error(error_type: str) -> str:
        """Simulate different types of errors."""
        try:
            if error_type == "memory_error":
                # Simulate memory error
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            elif error_type == "validation_error":
                # Simulate validation error
                raise ValueError("Input validation failed: text too long")
            elif error_type == "model_error":
                # Simulate model error
                raise RuntimeError("Model forward pass failed")
            elif error_type == "system_error":
                # Simulate system error
                raise Exception("Unexpected system error")
            else:
                return "Please select an error type to simulate."
                
        except Exception as e:
            if "memory" in error_type:
                return error_handler.handle_model_error(e, "Demo memory operation")
            elif "validation" in error_type:
                return error_handler.handle_validation_error("validation_error", str(e))
            elif "model" in error_type:
                return error_handler.handle_model_error(e, "Demo model operation")
            else:
                return error_handler.handle_system_error(e, "Demo system operation")
    
    def get_error_log() -> str:
        """Get the current error log."""
        return error_handler.get_error_summary()
    
    def clear_logs():
        """Clear error logs."""
        error_handler.clear_error_log()
        return "✅ Error logs cleared successfully!"
    
    # Create the interface
    with gr.Blocks(title="Error Handling & Input Validation Demo", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1>🛡️ Error Handling & Input Validation Demo</h1>
            <p>Comprehensive error handling with user-friendly messages and recovery strategies</p>
        </div>
        """)
        
        with gr.Tabs():
            # Input Validation Tab
            with gr.Tab("🔍 Input Validation"):
                gr.Markdown("### Test Input Validation")
                
                with gr.Row():
                    with gr.Column():
                        demo_text = gr.Textbox(
                            label="Demo Text",
                            placeholder="Enter text to validate (min 10 chars, max 10000 chars)",
                            lines=3
                        )
                        
                        demo_number = gr.Number(
                            label="Demo Number",
                            value=0,
                            info="Enter a number between -1,000,000 and 1,000,000"
                        )
                        
                        demo_file = gr.File(
                            label="Demo File",
                            file_types=['.txt', '.csv', '.json'],
                            info="Upload a text file (max 50MB)"
                        )
                        
                        demo_dropdown = gr.Dropdown(
                            choices=["Option A", "Option B", "Option C"],
                            label="Demo Dropdown",
                            value="Option A"
                        )
                        
                        validate_btn = gr.Button("🔍 Validate Inputs", variant="primary")
                    
                    with gr.Column():
                        validation_result = gr.Markdown("**Ready for validation!**\n\nClick the validate button to check your inputs.")
                        recovery_suggestions = gr.Markdown("")
                
                # Event handler
                validate_btn.click(
                    fn=validate_demo_input,
                    inputs=[demo_text, demo_number, demo_file, demo_dropdown],
                    outputs=[validation_result, recovery_suggestions]
                )
            
            # Error Simulation Tab
            with gr.Tab("🚨 Error Simulation"):
                gr.Markdown("### Simulate Different Error Types")
                
                with gr.Row():
                    with gr.Column():
                        error_type = gr.Radio(
                            choices=["memory_error", "validation_error", "model_error", "system_error"],
                            label="Error Type to Simulate",
                            value="memory_error"
                        )
                        
                        simulate_btn = gr.Button("🚨 Simulate Error", variant="secondary")
                    
                    with gr.Column():
                        error_result = gr.Markdown("**Ready to simulate errors!**\n\nSelect an error type and click simulate to see how it's handled.")
                
                # Event handler
                simulate_btn.click(
                    fn=simulate_error,
                    inputs=[error_type],
                    outputs=[error_result]
                )
            
            # Error Log Tab
            with gr.Tab("📊 Error Log"):
                gr.Markdown("### View Error History and Recovery")
                
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("🔄 Refresh Log", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear Logs", variant="secondary")
                    
                    with gr.Column():
                        error_log_display = gr.Markdown("**Error Log**\n\nClick refresh to view current error history.")
                
                # Event handlers
                refresh_btn.click(
                    fn=get_error_log,
                    outputs=[error_log_display]
                )
                
                clear_btn.click(
                    fn=clear_logs,
                    outputs=[error_log_display]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        ### 🛡️ **Robust Error Handling Features**
        
        This interface demonstrates comprehensive error handling including:
        • **Input Validation** - Comprehensive checks for all input types
        • **Error Classification** - Different handling for different error types
        • **User-Friendly Messages** - Clear explanations and solutions
        • **Recovery Strategies** - Actionable suggestions for common issues
        • **Error Logging** - Track and analyze error patterns
        • **Graceful Degradation** - Continue operation despite errors
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the error handling demo
    demo = create_error_handling_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=True,
        debug=True
    )
