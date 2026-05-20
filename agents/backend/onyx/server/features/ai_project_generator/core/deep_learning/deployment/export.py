"""
Model Export Utilities
======================

Utilities for exporting models to different formats.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...],
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 11,
    device: Optional[torch.device] = None
) -> Path:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        input_shape: Input tensor shape
        input_names: Input names
        output_names: Output names
        dynamic_axes: Dynamic axes for variable-length inputs
        opset_version: ONNX opset version
        device: Device to run on
        
    Returns:
        Path to exported model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    logger.info(f"Model exported to ONNX: {output_path}")
    return output_path


def load_onnx_model(onnx_path: Path) -> Any:
    """
    Load ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        ONNX model
    """
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(onnx_path))
        logger.info(f"ONNX model loaded: {onnx_path}")
        return session
        
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")


def export_to_torchscript(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...],
    method: str = 'trace',
    device: Optional[torch.device] = None
) -> Path:
    """
    Export model to TorchScript.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        input_shape: Input tensor shape
        method: Export method ('trace' or 'script')
        device: Device to run on
        
    Returns:
        Path to exported model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    if method == 'trace':
        # Tracing method
        dummy_input = torch.randn(input_shape).to(device)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path))
        
    elif method == 'script':
        # Scripting method
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(output_path))
        
    else:
        raise ValueError(f"Unknown export method: {method}")
    
    logger.info(f"Model exported to TorchScript: {output_path}")
    return output_path


def create_model_api(
    model: nn.Module,
    api_type: str = 'fastapi',
    output_dir: Path = Path("api"),
    model_path: Optional[Path] = None
) -> Path:
    """
    Generate API code for model serving.
    
    Args:
        model: PyTorch model
        api_type: API framework ('fastapi', 'flask')
        output_dir: Output directory
        model_path: Path to saved model
        
    Returns:
        Path to generated API
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if api_type == 'fastapi':
        api_code = f"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Model API")

# Load model
model_path = Path("{model_path or 'model.pt'}")
model = torch.load(model_path, map_location='cpu')
model.eval()

@app.get("/")
def root():
    return {{"message": "Model API is running"}}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess (adjust based on your model)
        # transform = ...
        # input_tensor = transform(image)
        
        # Predict
        with torch.no_grad():
            # output = model(input_tensor)
            pass
        
        return JSONResponse(content={{"prediction": "result"}})
    except Exception as e:
        return JSONResponse(content={{"error": str(e)}}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    else:
        api_code = f"""
from flask import Flask, request, jsonify
import torch
from pathlib import Path

app = Flask(__name__)

# Load model
model_path = Path("{model_path or 'model.pt'}")
model = torch.load(model_path, map_location='cpu')
model.eval()

@app.route("/", methods=["GET"])
def root():
    return jsonify({{"message": "Model API is running"}})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Process request
        data = request.get_json()
        
        # Predict
        with torch.no_grad():
            # output = model(input_tensor)
            pass
        
        return jsonify({{"prediction": "result"}})
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
"""
    
    api_file = output_dir / f"api_{api_type}.py"
    api_file.write_text(api_code)
    
    logger.info(f"API code generated: {api_file}")
    return api_file



