from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import logging
import traceback
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import re
import hashlib
from dataclasses import dataclass
from enum import Enum
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import roc_curve, auc
            import psutil
from typing import Any, List, Dict, Optional
import asyncio
"""
Robust Gradio Interfaces with Comprehensive Error Handling and Input Validation

This module provides production-ready Gradio interfaces with:
- Comprehensive input validation and sanitization
- Robust error handling with user-friendly messages
- Detailed logging for debugging and monitoring
- Graceful degradation for edge cases
- Security-focused input validation
- Performance monitoring and optimization
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class InputType(Enum):
    """Input types for validation."""
    FEATURE_VECTOR = "feature_vector"
    CSV_FILE = "csv_file"
    IMAGE = "image"
    TEXT = "text"
    JSON = "json"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Any = None


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self) -> Any:
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_features = 100
        self.max_samples = 10000
        self.allowed_file_types = ['.csv', '.json', '.txt']
        self.feature_range = (-1000, 1000)
        
    def validate_feature_vector(self, features: List[Any]) -> ValidationResult:
        """Validate feature vector input."""
        errors = []
        warnings = []
        sanitized_features = []
        
        try:
            # Check if features is a list
            if not isinstance(features, (list, tuple)):
                raise ValidationError("Features must be a list or tuple")
            
            # Check length
            if len(features) > self.max_features:
                errors.append(f"Too many features. Maximum allowed: {self.max_features}")
                return ValidationResult(False, errors, warnings)
            
            if len(features) == 0:
                errors.append("Feature vector cannot be empty")
                return ValidationResult(False, errors, warnings)
            
            # Validate each feature
            for i, feature in enumerate(features):
                try:
                    # Convert to float
                    feature_float = float(feature)
                    
                    # Check for NaN/Inf
                    if not np.isfinite(feature_float):
                        errors.append(f"Feature {i+1} contains NaN or infinite value")
                        continue
                    
                    # Check range
                    if feature_float < self.feature_range[0] or feature_float > self.feature_range[1]:
                        warnings.append(f"Feature {i+1} ({feature_float}) is outside normal range {self.feature_range}")
                    
                    sanitized_features.append(feature_float)
                    
                except (ValueError, TypeError):
                    errors.append(f"Feature {i+1} must be a valid number")
            
            if len(errors) > 0:
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, sanitized_features)
            
        except Exception as e:
            logger.error(f"Error validating feature vector: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def validate_csv_file(self, file_path: str) -> ValidationResult:
        """Validate CSV file input."""
        errors = []
        warnings = []
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                errors.append("File does not exist")
                return ValidationResult(False, errors, warnings)
            
            # Check file size
            file_size = Path(file_path).stat().st_size
            if file_size > self.max_file_size:
                errors.append(f"File too large. Maximum size: {self.max_file_size / 1024 / 1024:.1f}MB")
                return ValidationResult(False, errors, warnings)
            
            # Check file extension
            if not file_path.lower().endswith('.csv'):
                errors.append("File must be a CSV file")
                return ValidationResult(False, errors, warnings)
            
            # Try to read CSV
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                errors.append(f"Error reading CSV file: {str(e)}")
                return ValidationResult(False, errors, warnings)
            
            # Check dataframe size
            if len(df) > self.max_samples:
                warnings.append(f"Large dataset: {len(df)} samples. Processing may be slow.")
            
            if len(df.columns) > self.max_features:
                errors.append(f"Too many features: {len(df.columns)}. Maximum allowed: {self.max_features}")
                return ValidationResult(False, errors, warnings)
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                warnings.append(f"Dataset contains {missing_count} missing values")
            
            # Check for infinite values
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                warnings.append(f"Dataset contains {inf_count} infinite values")
            
            # Validate data types
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                warnings.append(f"Non-numeric columns detected: {list(non_numeric_cols)}")
            
            return ValidationResult(True, errors, warnings, df)
            
        except Exception as e:
            logger.error(f"Error validating CSV file: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def validate_numeric_input(self, value: Any, min_val: float = None, max_val: float = None) -> ValidationResult:
        """Validate numeric input with optional range constraints."""
        errors = []
        warnings = []
        
        try:
            # Convert to float
            numeric_value = float(value)
            
            # Check for NaN/Inf
            if not np.isfinite(numeric_value):
                errors.append("Value must be finite")
                return ValidationResult(False, errors, warnings)
            
            # Check range constraints
            if min_val is not None and numeric_value < min_val:
                errors.append(f"Value must be >= {min_val}")
                return ValidationResult(False, errors, warnings)
            
            if max_val is not None and numeric_value > max_val:
                errors.append(f"Value must be <= {max_val}")
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, numeric_value)
            
        except (ValueError, TypeError):
            errors.append("Value must be a valid number")
            return ValidationResult(False, errors, warnings)


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self) -> Any:
        self.error_counts = {}
        self.performance_metrics = {}
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors and return user-friendly response."""
        error_id = hashlib.md5(f"{type(error).__name__}:{str(error)}".encode()).hexdigest()[:8]
        
        # Log error details
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Generate user-friendly error message
        if isinstance(error, ValidationError):
            user_message = f"Input validation error: {str(error)}"
            error_level = "warning"
        elif isinstance(error, SecurityError):
            user_message = "Security validation failed. Please check your input."
            error_level = "error"
        elif isinstance(error, (ValueError, TypeError)):
            user_message = "Invalid input format. Please check your data."
            error_level = "warning"
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            user_message = "File access error. Please check file permissions and path."
            error_level = "error"
        elif isinstance(error, MemoryError):
            user_message = "Insufficient memory. Please try with smaller data."
            error_level = "error"
        else:
            user_message = "An unexpected error occurred. Please try again."
            error_level = "error"
        
        return {
            "error": True,
            "error_id": error_id,
            "message": user_message,
            "level": error_level,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": self.error_counts,
            "performance_metrics": self.performance_metrics
        }


class RobustCybersecurityModelInterface:
    """Robust interface for cybersecurity model showcase with comprehensive error handling."""
    
    def __init__(self) -> Any:
        self.model = self._create_demo_model()
        self.model.eval()
        self.validator = InputValidator()
        self.error_handler = ErrorHandler()
        self.prediction_history = []
        self.performance_metrics = {}
        
        logger.info("RobustCybersecurityModelInterface initialized successfully")
    
    def _create_demo_model(self) -> nn.Module:
        """Create a demo model for showcase."""
        try:
            class DemoCybersecurityModel(nn.Module):
                def __init__(self, input_dim=20, num_classes=4) -> Any:
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(32, 16),
                        nn.ReLU()
                    )
                    self.classifier = nn.Linear(16, num_classes)
                    
                def forward(self, x) -> Any:
                    features = self.features(x)
                    return self.classifier(features)
            
            model = DemoCybersecurityModel(input_dim=20, num_classes=4)
            logger.info("Demo model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error creating demo model: {str(e)}")
            raise
    
    def _sanitize_output(self, value: Any) -> Any:
        """Sanitize outputs to handle NaN/Inf values."""
        try:
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
                return value
            elif isinstance(value, (int, float)):
                if not np.isfinite(value):
                    return 0.0
            elif isinstance(value, list):
                return [self._sanitize_output(item) for item in value]
            elif isinstance(value, dict):
                return {k: self._sanitize_output(v) for k, v in value.items()}
            return value
        except Exception as e:
            logger.warning(f"Error sanitizing output: {str(e)}")
            return 0.0
    
    def _safe_model_inference(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Safe model inference with error handling."""
        try:
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
            
            return logits, probabilities, prediction, confidence
            
        except Exception as e:
            logger.error(f"Model inference error: {str(e)}")
            raise
    
    def real_time_inference(self, *feature_values) -> Dict[str, Any]:
        """Real-time inference with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Validate input
            validation_result = self.validator.validate_feature_vector(list(feature_values))
            if not validation_result.is_valid:
                return self.error_handler.handle_error(
                    ValidationError("; ".join(validation_result.errors)),
                    "real_time_inference"
                )
            
            # Show warnings if any
            if validation_result.warnings:
                logger.warning(f"Input warnings: {validation_result.warnings}")
            
            # Convert to tensor
            features = torch.tensor(validation_result.sanitized_data, dtype=torch.float32).unsqueeze(0)
            
            # Get model prediction
            logits, probabilities, prediction, confidence = self._safe_model_inference(features)
            
            # Sanitize outputs
            probs = self._sanitize_output(probabilities.cpu().numpy().flatten())
            pred_class = int(self._sanitize_output(prediction.cpu().numpy()[0]))
            conf_score = float(self._sanitize_output(confidence.cpu().numpy()[0]))
            
            # Threat classification mapping
            threat_types = {
                0: "Normal Traffic",
                1: "Malware Detection",
                2: "Network Intrusion", 
                3: "Data Exfiltration"
            }
            
            # Risk assessment
            if conf_score > 0.8:
                risk_level = "High"
                risk_color = "red"
            elif conf_score > 0.6:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "Low"
                risk_color = "green"
            
            # Store prediction history
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'prediction': pred_class,
                'confidence': conf_score,
                'risk_level': risk_level,
                'features': validation_result.sanitized_data,
                'processing_time': time.time() - start_time
            })
            
            # Update performance metrics
            self.performance_metrics['total_predictions'] = self.performance_metrics.get('total_predictions', 0) + 1
            self.performance_metrics['avg_processing_time'] = (
                (self.performance_metrics.get('avg_processing_time', 0) * 
                 (self.performance_metrics['total_predictions'] - 1) + 
                 (time.time() - start_time)) / self.performance_metrics['total_predictions']
            )
            
            return {
                'error': False,
                'prediction': threat_types[pred_class],
                'confidence': f"{conf_score:.3f}",
                'risk_level': risk_level,
                'risk_color': risk_color,
                'probabilities': {threat_types[i]: f"{prob:.3f}" for i, prob in enumerate(probs)},
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': f"{(time.time() - start_time):.3f}s",
                'warnings': validation_result.warnings
            }
            
        except Exception as e:
            return self.error_handler.handle_error(e, "real_time_inference")
    
    def batch_analysis(self, csv_file) -> Tuple[Union[pd.DataFrame, Dict], Union[go.Figure, Dict], Union[go.Figure, Dict], Union[go.Figure, Dict]]:
        """Batch analysis with comprehensive error handling."""
        start_time = time.time()
        
        try:
            if csv_file is None:
                return self._create_error_response("No file uploaded")
            
            # Validate CSV file
            validation_result = self.validator.validate_csv_file(csv_file.name)
            if not validation_result.is_valid:
                return self._create_error_response("; ".join(validation_result.errors))
            
            df = validation_result.sanitized_data
            
            # Show warnings if any
            if validation_result.warnings:
                logger.warning(f"CSV validation warnings: {validation_result.warnings}")
            
            # Assume last column is target, rest are features
            feature_cols = df.columns[:-1]
            target_col = df.columns[-1]
            
            # Validate target column
            if not df[target_col].dtype in ['int64', 'float64']:
                return self._create_error_response("Target column must contain numeric values")
            
            # Prepare data
            features = df[feature_cols].values
            targets = df[target_col].values
            
            # Sanitize features
            features = self._sanitize_output(features)
            targets = self._sanitize_output(targets)
            
            # Get predictions
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            logits, probabilities, predictions, confidences = self._safe_model_inference(feature_tensor)
            
            # Sanitize outputs
            predictions = self._sanitize_output(predictions.cpu().numpy())
            probabilities = self._sanitize_output(probabilities.cpu().numpy())
            confidences = self._sanitize_output(confidences.cpu().numpy())
            
            # Add predictions to dataframe
            df['Predicted'] = predictions
            df['Confidence'] = confidences
            df['Correct'] = (predictions == targets)
            
            # Create visualizations
            fig1 = self._create_confusion_matrix(targets, predictions)
            fig2 = self._create_roc_curves(targets, probabilities)
            fig3 = self._create_confidence_distribution(df)
            
            logger.info(f"Batch analysis completed in {time.time() - start_time:.3f}s")
            
            return df, fig1, fig2, fig3
            
        except Exception as e:
            return self._create_error_response(str(e))
    
    def _create_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray) -> go.Figure:
        """Create confusion matrix visualization."""
        try:
            cm = confusion_matrix(targets, predictions)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Malware', 'Intrusion', 'Exfiltration'],
                y=['Normal', 'Malware', 'Intrusion', 'Exfiltration'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ))
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=500,
                height=400
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {str(e)}")
            return self._create_error_figure("Error creating confusion matrix")
    
    def _create_roc_curves(self, targets: np.ndarray, probabilities: np.ndarray) -> go.Figure:
        """Create ROC curves visualization."""
        try:
            fig = go.Figure()
            
            for i in range(4):
                fpr, tpr, _ = roc_curve((targets == i).astype(int), probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'Class {i} (AUC = {roc_auc:.3f})',
                    mode='lines'
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=500,
                height=400
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating ROC curves: {str(e)}")
            return self._create_error_figure("Error creating ROC curves")
    
    def _create_confidence_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create confidence distribution visualization."""
        try:
            fig = go.Figure()
            
            for i in range(4):
                class_confidences = df[df['Predicted'] == i]['Confidence']
                if len(class_confidences) > 0:
                    fig.add_trace(go.Box(
                        y=class_confidences,
                        name=f'Class {i}',
                        boxpoints='outliers'
                    ))
            
            fig.update_layout(
                title="Confidence Distribution by Class",
                yaxis_title="Confidence Score",
                width=500,
                height=400
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating confidence distribution: {str(e)}")
            return self._create_error_figure("Error creating confidence distribution")
    
    def _create_error_response(self, message: str) -> Tuple[Dict, Dict, Dict, Dict]:
        """Create error response for batch analysis."""
        error_response = {
            'error': True,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        return error_response, error_response, error_response, error_response
    
    def _create_error_figure(self, message: str) -> go.Figure:
        """Create error figure for visualization failures."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=500,
            height=400
        )
        return fig
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health metrics."""
        try:
            return {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'total_predictions': self.performance_metrics.get('total_predictions', 0),
                'avg_processing_time': self.performance_metrics.get('avg_processing_time', 0),
                'error_summary': self.error_handler.get_error_summary(),
                'memory_usage': self._get_memory_usage(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}


def create_robust_interfaces():
    """Create and configure robust Gradio interfaces."""
    
    # Initialize model interface
    model_interface = RobustCybersecurityModelInterface()
    
    # Custom CSS for better styling and error display
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #c62828;
        margin: 10px 0;
    }
    .warning-message {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ef6c00;
        margin: 10px 0;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2e7d32;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=custom_css, title="Robust Cybersecurity AI Model Showcase") as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ Robust Cybersecurity AI Model Showcase</h1>
            <p>Production-ready machine learning models with comprehensive error handling</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Real-time Inference with Error Handling
            with gr.TabItem("🔍 Real-time Inference"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Features")
                        feature_inputs = []
                        for i in range(20):
                            feature_inputs.append(
                                gr.Number(
                                    label=f"Feature {i+1}",
                                    value=np.random.normal(0, 1),
                                    precision=3,
                                    minimum=-1000,
                                    maximum=1000
                                )
                            )
                        
                        analyze_btn = gr.Button("🚀 Analyze Threat", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Analysis Results")
                        
                        # Error display
                        error_output = gr.HTML(label="Status")
                        
                        with gr.Row():
                            prediction_output = gr.Textbox(label="Threat Type", interactive=False)
                            confidence_output = gr.Textbox(label="Confidence", interactive=False)
                        
                        with gr.Row():
                            risk_output = gr.Textbox(label="Risk Level", interactive=False)
                            processing_time_output = gr.Textbox(label="Processing Time", interactive=False)
                        
                        gr.Markdown("### Class Probabilities")
                        probability_output = gr.JSON(label="Probability Distribution")
                        
                        # Warnings display
                        warnings_output = gr.HTML(label="Warnings")
                
                def analyze_with_error_handling(*features) -> Any:
                    result = model_interface.real_time_inference(*features)
                    
                    if result.get('error', False):
                        error_html = f"""
                        <div class="error-message">
                            <strong>Error:</strong> {result['message']}
                            <br><small>Error ID: {result.get('error_id', 'N/A')}</small>
                        </div>
                        """
                        return (
                            error_html,
                            "Error", "0.000", "Unknown", "N/A",
                            {}, ""
                        )
                    else:
                        error_html = f"""
                        <div class="success-message">
                            <strong>Success:</strong> Analysis completed successfully
                            <br><small>Processing time: {result.get('processing_time', 'N/A')}</small>
                        </div>
                        """
                        
                        warnings_html = ""
                        if result.get('warnings'):
                            warnings_html = f"""
                            <div class="warning-message">
                                <strong>Warnings:</strong><br>
                                {chr(10).join(f"• {w}" for w in result['warnings'])}
                            </div>
                            """
                        
                        return (
                            error_html,
                            result['prediction'],
                            result['confidence'],
                            result['risk_level'],
                            result['processing_time'],
                            result['probabilities'],
                            warnings_html
                        )
                
                analyze_btn.click(
                    fn=analyze_with_error_handling,
                    inputs=feature_inputs,
                    outputs=[error_output, prediction_output, confidence_output, risk_output, processing_time_output, probability_output, warnings_output]
                )
            
            # Tab 2: Batch Analysis with Error Handling
            with gr.TabItem("📊 Batch Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Data")
                        file_input = gr.File(
                            label="Upload CSV file with features and labels",
                            file_types=[".csv"],
                            file_count="single"
                        )
                        analyze_batch_btn = gr.Button("📈 Analyze Batch", variant="primary")
                        
                        # File validation info
                        file_info = gr.HTML(label="File Information")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Analysis Status")
                        batch_status = gr.HTML(label="Status")
                
                with gr.Row():
                    results_table = gr.Dataframe(label="Analysis Results")
                
                with gr.Row():
                    confusion_matrix_plot = gr.Plot(label="Confusion Matrix")
                    roc_plot = gr.Plot(label="ROC Curves")
                
                confidence_dist_plot = gr.Plot(label="Confidence Distribution")
                
                def analyze_batch_with_validation(file) -> Any:
                    if file is None:
                        return (
                            "<div class='error-message'><strong>Error:</strong> No file uploaded</div>",
                            "<div class='error-message'><strong>Error:</strong> No file uploaded</div>",
                            pd.DataFrame(),
                            go.Figure(),
                            go.Figure(),
                            go.Figure()
                        )
                    
                    # Validate file
                    validation_result = model_interface.validator.validate_csv_file(file.name)
                    
                    if not validation_result.is_valid:
                        error_html = f"""
                        <div class="error-message">
                            <strong>Validation Error:</strong><br>
                            {chr(10).join(f"• {e}" for e in validation_result.errors)}
                        </div>
                        """
                        return (
                            error_html,
                            error_html,
                            pd.DataFrame(),
                            go.Figure(),
                            go.Figure(),
                            go.Figure()
                        )
                    
                    # Show file info
                    file_info_html = f"""
                    <div class="success-message">
                        <strong>File Validated:</strong><br>
                        • Rows: {len(validation_result.sanitized_data)}<br>
                        • Columns: {len(validation_result.sanitized_data.columns)}<br>
                        • Size: {Path(file.name).stat().st_size / 1024:.1f} KB
                    </div>
                    """
                    
                    if validation_result.warnings:
                        file_info_html += f"""
                        <div class="warning-message">
                            <strong>Warnings:</strong><br>
                            {chr(10).join(f"• {w}" for w in validation_result.warnings)}
                        </div>
                        """
                    
                    # Perform analysis
                    try:
                        df, fig1, fig2, fig3 = model_interface.batch_analysis(file)
                        
                        if isinstance(df, dict) and df.get('error'):
                            error_html = f"""
                            <div class="error-message">
                                <strong>Analysis Error:</strong> {df['message']}
                            </div>
                            """
                            return (
                                file_info_html,
                                error_html,
                                pd.DataFrame(),
                                go.Figure(),
                                go.Figure(),
                                go.Figure()
                            )
                        
                        success_html = f"""
                        <div class="success-message">
                            <strong>Analysis Completed:</strong> Successfully processed {len(df)} samples
                        </div>
                        """
                        
                        return file_info_html, success_html, df, fig1, fig2, fig3
                        
                    except Exception as e:
                        error_html = f"""
                        <div class="error-message">
                            <strong>Analysis Error:</strong> {str(e)}
                        </div>
                        """
                        return (
                            file_info_html,
                            error_html,
                            pd.DataFrame(),
                            go.Figure(),
                            go.Figure(),
                            go.Figure()
                        )
                
                analyze_batch_btn.click(
                    fn=analyze_batch_with_validation,
                    inputs=file_input,
                    outputs=[file_info, batch_status, results_table, confusion_matrix_plot, roc_plot, confidence_dist_plot]
                )
            
            # Tab 3: System Status and Monitoring
            with gr.TabItem("📈 System Status"):
                with gr.Row():
                    refresh_status_btn = gr.Button("🔄 Refresh Status", variant="primary")
                
                status_output = gr.JSON(label="System Status")
                
                def get_system_status():
                    
    """get_system_status function."""
return model_interface.get_system_status()
                
                refresh_status_btn.click(
                    fn=get_system_status,
                    inputs=[],
                    outputs=status_output
                )
    
    return demo


if __name__ == "__main__":
    # Create and launch the robust interface
    demo = create_robust_interfaces()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        show_tips=True
    ) 