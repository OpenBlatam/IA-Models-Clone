"""
Documentation Generator
=======================

Automatic documentation generation utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def generate_model_docs(
    model: nn.Module,
    output_path: Path,
    input_shape: Optional[tuple] = None,
    include_architecture: bool = True,
    include_parameters: bool = True
) -> Path:
    """
    Generate model documentation.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        input_shape: Input tensor shape
        include_architecture: Include architecture details
        include_parameters: Include parameter count
        
    Returns:
        Path to generated documentation
    """
    docs = []
    docs.append("# Model Documentation\n")
    docs.append(f"**Model Type:** {type(model).__name__}\n\n")
    
    # Architecture
    if include_architecture:
        docs.append("## Architecture\n\n")
        docs.append("```python\n")
        docs.append(str(model))
        docs.append("\n```\n\n")
    
    # Parameters
    if include_parameters:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        docs.append("## Parameters\n\n")
        docs.append(f"- **Total Parameters:** {total_params:,}\n")
        docs.append(f"- **Trainable Parameters:** {trainable_params:,}\n")
        docs.append(f"- **Non-trainable Parameters:** {total_params - trainable_params:,}\n\n")
    
    # Input/Output shapes
    if input_shape:
        docs.append("## Input/Output\n\n")
        docs.append(f"- **Input Shape:** {input_shape}\n")
        
        try:
            model.eval()
            dummy_input = torch.randn((1, *input_shape))
            with torch.no_grad():
                output = model(dummy_input)
            if isinstance(output, torch.Tensor):
                docs.append(f"- **Output Shape:** {tuple(output.shape)}\n")
        except Exception as e:
            docs.append(f"- **Output Shape:** Error computing - {e}\n")
        docs.append("\n")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(docs))
    
    logger.info(f"Model documentation generated: {output_path}")
    return output_path


def generate_training_docs(
    config: Dict[str, Any],
    output_path: Path,
    include_results: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate training documentation.
    
    Args:
        config: Training configuration
        output_path: Output file path
        include_results: Training results (optional)
        
    Returns:
        Path to generated documentation
    """
    docs = []
    docs.append("# Training Documentation\n\n")
    
    # Configuration
    docs.append("## Configuration\n\n")
    docs.append("```yaml\n")
    for key, value in config.items():
        docs.append(f"{key}: {value}\n")
    docs.append("```\n\n")
    
    # Results
    if include_results:
        docs.append("## Results\n\n")
        for key, value in include_results.items():
            if isinstance(value, (int, float)):
                docs.append(f"- **{key}:** {value:.4f}\n")
            else:
                docs.append(f"- **{key}:** {value}\n")
        docs.append("\n")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(docs))
    
    logger.info(f"Training documentation generated: {output_path}")
    return output_path


def generate_api_docs(
    api_code: str,
    output_path: Path,
    api_type: str = 'fastapi'
) -> Path:
    """
    Generate API documentation.
    
    Args:
        api_code: API code
        output_path: Output file path
        api_type: API type ('fastapi', 'flask')
        
    Returns:
        Path to generated documentation
    """
    docs = []
    docs.append(f"# {api_type.upper()} API Documentation\n\n")
    
    if api_type == 'fastapi':
        docs.append("## Endpoints\n\n")
        docs.append("### GET /\n")
        docs.append("Root endpoint.\n\n")
        docs.append("### POST /predict\n")
        docs.append("Model prediction endpoint.\n\n")
        docs.append("**Request:**\n")
        docs.append("- `file`: Uploaded file\n\n")
        docs.append("**Response:**\n")
        docs.append("- `prediction`: Prediction result\n\n")
    
    docs.append("## Usage\n\n")
    docs.append("```bash\n")
    docs.append("python api_fastapi.py\n")
    docs.append("```\n\n")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(docs))
    
    logger.info(f"API documentation generated: {output_path}")
    return output_path


def create_project_readme(
    project_name: str,
    output_path: Path,
    description: str = "",
    features: Optional[List[str]] = None,
    installation: Optional[str] = None,
    usage: Optional[str] = None
) -> Path:
    """
    Create project README.
    
    Args:
        project_name: Project name
        output_path: Output file path
        description: Project description
        features: List of features
        installation: Installation instructions
        usage: Usage examples
        
    Returns:
        Path to generated README
    """
    readme = []
    readme.append(f"# {project_name}\n\n")
    
    if description:
        readme.append(f"{description}\n\n")
    
    # Features
    if features:
        readme.append("## Features\n\n")
        for feature in features:
            readme.append(f"- {feature}\n")
        readme.append("\n")
    
    # Installation
    if installation:
        readme.append("## Installation\n\n")
        readme.append(f"{installation}\n\n")
    else:
        readme.append("## Installation\n\n")
        readme.append("```bash\n")
        readme.append("pip install -r requirements.txt\n")
        readme.append("```\n\n")
    
    # Usage
    if usage:
        readme.append("## Usage\n\n")
        readme.append(f"{usage}\n\n")
    else:
        readme.append("## Usage\n\n")
        readme.append("```python\n")
        readme.append("# Add usage examples here\n")
        readme.append("```\n\n")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(readme))
    
    logger.info(f"Project README created: {output_path}")
    return output_path



