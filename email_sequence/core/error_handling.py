from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import traceback
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import asyncio
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional
"""
Error Handling and Validation System

Comprehensive error handling for data loading, model inference,
and other error-prone operations in the email sequence system.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailSequenceError(Exception):
    """Base exception for email sequence system"""
    pass


class ValidationError(EmailSequenceError):
    """Raised when input validation fails"""
    pass


class ModelError(EmailSequenceError):
    """Raised when model operations fail"""
    pass


class DataError(EmailSequenceError):
    """Raised when data loading or processing fails"""
    pass


class ConfigurationError(EmailSequenceError):
    """Raised when configuration is invalid"""
    pass


class ErrorHandler:
    """Comprehensive error handling for the email sequence system"""
    
    def __init__(self, debug_mode: bool = False):
        
    """__init__ function."""
self.debug_mode = debug_mode
        self.error_log = []
        
    def log_error(self, error: Exception, context: str = "", operation: str = ""):
        """Log error with context and operation information"""
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "operation": operation,
            "traceback": traceback.format_exc() if self.debug_mode else None
        }
        
        self.error_log.append(error_info)
        logger.error(f"Error in {operation}: {error}")
        
        if self.debug_mode:
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def safe_execute(self, func, *args, context: str = "", **kwargs) -> Tuple[Any, Optional[str]]:
        """Safely execute a function with error handling"""
        
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            self.log_error(e, context, func.__name__)
            return None, str(e)
    
    async def safe_async_execute(self, func, *args, context: str = "", **kwargs) -> Tuple[Any, Optional[str]]:
        """Safely execute an async function with error handling"""
        
        try:
            result = await func(*args, **kwargs)
            return result, None
        except Exception as e:
            self.log_error(e, context, func.__name__)
            return None, str(e)
    
    def get_error_summary(self) -> Dict:
        """Get a summary of recent errors"""
        
        if not self.error_log:
            return {"total_errors": 0, "recent_errors": []}
        
        recent_errors = self.error_log[-10:]  # Last 10 errors
        
        error_types = {}
        for error in recent_errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "recent_errors": recent_errors,
            "error_type_distribution": error_types,
            "last_error": self.error_log[-1] if self.error_log else None
        }


class InputValidator:
    """Input validation for user inputs and configuration"""
    
    @staticmethod
    def validate_model_type(model_type: str) -> Tuple[bool, str]:
        """Validate AI model type"""
        
        valid_models = ["GPT-3.5", "GPT-4", "Claude", "Custom", "Custom Model"]
        
        if not model_type:
            return False, "Model type is required"
        
        if model_type not in valid_models:
            return False, f"Invalid model type. Must be one of: {', '.join(valid_models)}"
        
        return True, ""
    
    @staticmethod
    def validate_sequence_length(length: int) -> Tuple[bool, str]:
        """Validate sequence length"""
        
        if not isinstance(length, int):
            return False, "Sequence length must be an integer"
        
        if length < 1 or length > 10:
            return False, "Sequence length must be between 1 and 10"
        
        return True, ""
    
    @staticmethod
    def validate_creativity_level(creativity: float) -> Tuple[bool, str]:
        """Validate creativity level"""
        
        if not isinstance(creativity, (int, float)):
            return False, "Creativity level must be a number"
        
        if creativity < 0.1 or creativity > 1.0:
            return False, "Creativity level must be between 0.1 and 1.0"
        
        return True, ""
    
    @staticmethod
    def validate_subscriber_data(subscriber_data: Dict) -> Tuple[bool, str]:
        """Validate subscriber data"""
        
        required_fields = ["id", "email", "name", "company"]
        
        for field in required_fields:
            if field not in subscriber_data:
                return False, f"Missing required field: {field}"
            
            if not subscriber_data[field]:
                return False, f"Field {field} cannot be empty"
        
        # Validate email format
        email = subscriber_data["email"]
        if "@" not in email or "." not in email:
            return False, "Invalid email format"
        
        return True, ""
    
    @staticmethod
    def validate_training_config(config: Dict) -> Tuple[bool, str]:
        """Validate training configuration"""
        
        required_fields = ["max_epochs", "batch_size", "learning_rate"]
        
        for field in required_fields:
            if field not in config:
                return False, f"Missing required training field: {field}"
        
        if config["max_epochs"] < 1:
            return False, "Max epochs must be at least 1"
        
        if config["batch_size"] < 1:
            return False, "Batch size must be at least 1"
        
        if config["learning_rate"] <= 0:
            return False, "Learning rate must be positive"
        
        return True, ""
    
    @staticmethod
    def validate_evaluation_config(config: Dict) -> Tuple[bool, str]:
        """Validate evaluation configuration"""
        
        if "enable_content_quality" not in config:
            return False, "Content quality setting is required"
        
        if "enable_engagement" not in config:
            return False, "Engagement setting is required"
        
        # Validate weights if provided
        if "content_weight" in config:
            weight = config["content_weight"]
            if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                return False, "Content weight must be between 0 and 1"
        
        return True, ""


class DataLoaderErrorHandler:
    """Error handling for data loading operations"""
    
    def __init__(self, error_handler: ErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
    
    def safe_load_csv(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Safely load CSV file with error handling"""
        
        try:
            if not Path(file_path).exists():
                return None, f"File not found: {file_path}"
            
            df = pd.read_csv(file_path)
            
            if df.empty:
                return None, "CSV file is empty"
            
            return df, None
            
        except pd.errors.EmptyDataError:
            return None, "CSV file is empty or corrupted"
        except pd.errors.ParserError as e:
            return None, f"CSV parsing error: {str(e)}"
        except Exception as e:
            self.error_handler.log_error(e, f"Loading CSV file: {file_path}", "safe_load_csv")
            return None, f"Error loading CSV: {str(e)}"
    
    def safe_load_json(self, file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Safely load JSON file with error handling"""
        
        try:
            if not Path(file_path).exists():
                return None, f"File not found: {file_path}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = json.load(f)
            
            return data, None
            
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON format: {str(e)}"
        except Exception as e:
            self.error_handler.log_error(e, f"Loading JSON file: {file_path}", "safe_load_json")
            return None, f"Error loading JSON: {str(e)}"
    
    def safe_save_data(self, data: Any, file_path: str, file_type: str = "json") -> Tuple[bool, Optional[str]]:
        """Safely save data with error handling"""
        
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            if file_type == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(data, f, indent=2, default=str)
            elif file_type == "csv":
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False)
                else:
                    return False, "Data must be a pandas DataFrame for CSV export"
            else:
                return False, f"Unsupported file type: {file_type}"
            
            return True, None
            
        except Exception as e:
            self.error_handler.log_error(e, f"Saving {file_type} file: {file_path}", "safe_save_data")
            return False, f"Error saving {file_type}: {str(e)}"


class ModelInferenceErrorHandler:
    """Error handling for model inference operations"""
    
    def __init__(self, error_handler: ErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
    
    def safe_model_load(self, model_path: str, model_type: str = "pytorch") -> Tuple[Optional[Any], Optional[str]]:
        """Safely load model with error handling"""
        
        try:
            if not Path(model_path).exists():
                return None, f"Model file not found: {model_path}"
            
            if model_type == "pytorch":
                # Check if CUDA is available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = torch.load(model_path, map_location=device)
            else:
                return None, f"Unsupported model type: {model_type}"
            
            return model, None
            
        except torch.cuda.OutOfMemoryError:
            return None, "GPU out of memory. Try using CPU or reducing batch size"
        except Exception as e:
            self.error_handler.log_error(e, f"Loading model: {model_path}", "safe_model_load")
            return None, f"Error loading model: {str(e)}"
    
    def safe_model_inference(self, model: Any, inputs: Any, **kwargs) -> Tuple[Optional[Any], Optional[str]]:
        """Safely perform model inference with error handling"""
        
        try:
            # Set model to evaluation mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Move to appropriate device
            device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
            
            # Perform inference
            with torch.no_grad():
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                outputs = model(inputs, **kwargs)
            
            return outputs, None
            
        except torch.cuda.OutOfMemoryError:
            return None, "GPU out of memory during inference. Try using CPU or reducing input size"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return None, "Out of memory error. Try reducing batch size or input size"
            else:
                self.error_handler.log_error(e, "Model inference", "safe_model_inference")
                return None, f"Runtime error during inference: {str(e)}"
        except Exception as e:
            self.error_handler.log_error(e, "Model inference", "safe_model_inference")
            return None, f"Error during inference: {str(e)}"
    
    def safe_batch_inference(self, model: Any, batch_inputs: List, batch_size: int = 32) -> Tuple[List, List[str]]:
        """Safely perform batch inference with error handling"""
        
        results = []
        errors = []
        
        try:
            for i in range(0, len(batch_inputs), batch_size):
                batch = batch_inputs[i:i + batch_size]
                
                try:
                    batch_result, error = self.safe_model_inference(model, batch)
                    if error:
                        errors.extend([error] * len(batch))
                        results.extend([None] * len(batch))
                    else:
                        results.extend(batch_result)
                        errors.extend([None] * len(batch))
                        
                except Exception as e:
                    self.error_handler.log_error(e, f"Batch inference batch {i//batch_size}", "safe_batch_inference")
                    errors.extend([str(e)] * len(batch))
                    results.extend([None] * len(batch))
            
            return results, errors
            
        except Exception as e:
            self.error_handler.log_error(e, "Batch inference", "safe_batch_inference")
            return [], [str(e)] * len(batch_inputs)


class GradioErrorHandler:
    """Error handling specifically for Gradio applications"""
    
    def __init__(self, error_handler: ErrorHandler, debug_mode: bool = False):
        
    """__init__ function."""
self.error_handler = error_handler
        self.debug_mode = debug_mode
    
    def safe_gradio_function(self, func, *args, **kwargs) -> Any:
        """Wrapper for Gradio functions with error handling"""
        
        try:
            result = func(*args, **kwargs)
            return result
            
        except ValidationError as e:
            self.error_handler.log_error(e, "Input validation", func.__name__)
            return self._format_gradio_error("Validation Error", str(e))
            
        except ModelError as e:
            self.error_handler.log_error(e, "Model operation", func.__name__)
            return self._format_gradio_error("Model Error", str(e))
            
        except DataError as e:
            self.error_handler.log_error(e, "Data operation", func.__name__)
            return self._format_gradio_error("Data Error", str(e))
            
        except Exception as e:
            self.error_handler.log_error(e, "Unexpected error", func.__name__)
            if self.debug_mode:
                return self._format_gradio_error("Unexpected Error", f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}")
            else:
                return self._format_gradio_error("Unexpected Error", "An unexpected error occurred. Please try again.")
    
    def _format_gradio_error(self, error_type: str, message: str) -> Dict:
        """Format error for Gradio display"""
        
        return {
            "error": True,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_gradio_inputs(self, inputs: Dict) -> Tuple[bool, List[str]]:
        """Validate Gradio inputs before processing"""
        
        errors = []
        validator = InputValidator()
        
        # Validate model type if present
        if "model_type" in inputs:
            is_valid, error_msg = validator.validate_model_type(inputs["model_type"])
            if not is_valid:
                errors.append(error_msg)
        
        # Validate sequence length if present
        if "sequence_length" in inputs:
            is_valid, error_msg = validator.validate_sequence_length(inputs["sequence_length"])
            if not is_valid:
                errors.append(error_msg)
        
        # Validate creativity level if present
        if "creativity_level" in inputs:
            is_valid, error_msg = validator.validate_creativity_level(inputs["creativity_level"])
            if not is_valid:
                errors.append(error_msg)
        
        return len(errors) == 0, errors


# Utility functions for common error handling patterns
def handle_async_operation(func) -> Any:
    """Decorator for handling async operations with error handling"""
    
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in async operation {func.__name__}: {e}")
            raise
    
    return wrapper


def handle_model_operation(func) -> Any:
    """Decorator for handling model operations with error handling"""
    
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            raise ModelError("GPU out of memory. Try using CPU or reducing batch size")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise ModelError("Out of memory error. Try reducing batch size or input size")
            else:
                raise ModelError(f"Runtime error: {str(e)}")
        except Exception as e:
            raise ModelError(f"Model operation failed: {str(e)}")
    
    return wrapper


def handle_data_operation(func) -> Any:
    """Decorator for handling data operations with error handling"""
    
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise DataError(f"File not found: {str(e)}")
        except pd.errors.EmptyDataError:
            raise DataError("Data file is empty")
        except pd.errors.ParserError as e:
            raise DataError(f"Data parsing error: {str(e)}")
        except json.JSONDecodeError as e:
            raise DataError(f"JSON parsing error: {str(e)}")
        except Exception as e:
            raise DataError(f"Data operation failed: {str(e)}")
    
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test error handling system
    error_handler = ErrorHandler(debug_mode=True)
    validator = InputValidator()
    data_handler = DataLoaderErrorHandler(error_handler)
    model_handler = ModelInferenceErrorHandler(error_handler)
    gradio_handler = GradioErrorHandler(error_handler, debug_mode=True)
    
    # Test input validation
    print("Testing input validation...")
    is_valid, error = validator.validate_model_type("GPT-3.5")
    print(f"Model validation: {is_valid}, Error: {error}")
    
    is_valid, error = validator.validate_sequence_length(5)
    print(f"Length validation: {is_valid}, Error: {error}")
    
    # Test error handling
    print("\nTesting error handling...")
    result, error = error_handler.safe_execute(lambda x: x / 0, 10, context="Division test")
    print(f"Safe execution result: {result}, Error: {error}")
    
    # Test error summary
    summary = error_handler.get_error_summary()
    print(f"\nError summary: {summary}") 