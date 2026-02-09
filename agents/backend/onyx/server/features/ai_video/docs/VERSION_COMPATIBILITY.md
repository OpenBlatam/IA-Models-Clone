# Version Compatibility Guide

## Ensuring Compatibility with Latest Library Versions

This guide ensures our AI video system remains compatible with the latest stable versions of PyTorch, Transformers, Diffusers, and Gradio.

---

## 📦 Recommended Versions

### Current Stable Versions (as of 2024)

```python
# requirements.txt - Latest stable versions
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
transformers>=4.36.0
diffusers>=0.25.0
gradio>=4.0.0
accelerate>=0.25.0
xformers>=0.0.23
peft>=0.7.0
bitsandbytes>=0.41.0
safetensors>=0.4.0
```

### Version Compatibility Matrix

| Component | Minimum Version | Recommended Version | Latest Version |
|-----------|----------------|-------------------|----------------|
| PyTorch | 2.0.0 | 2.1.0 | 2.2.0 |
| Transformers | 4.30.0 | 4.36.0 | 4.40.0 |
| Diffusers | 0.21.0 | 0.25.0 | 0.28.0 |
| Gradio | 3.40.0 | 4.0.0 | 4.15.0 |
| Accelerate | 0.20.0 | 0.25.0 | 0.28.0 |

---

## 🔄 Migration Guides

### PyTorch 2.0 → 2.1 Migration

#### **New Features to Adopt**
```python
# PyTorch 2.1: Enhanced torch.compile()
import torch

# Old way (PyTorch 2.0)
compiled_model = torch.compile(model, mode="reduce-overhead")

# New way (PyTorch 2.1) - More options
compiled_model = torch.compile(
    model, 
    mode="reduce-overhead",
    fullgraph=True,  # New option
    dynamic=True     # New option
)

# PyTorch 2.1: Enhanced flash attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)  # New in 2.1
```

#### **Breaking Changes**
```python
# PyTorch 2.1: Checkpoint API change
from torch.utils.checkpoint import checkpoint

# Old way (PyTorch 2.0)
x = checkpoint(layer, x)

# New way (PyTorch 2.1) - use_reentrant parameter
x = checkpoint(layer, x, use_reentrant=False)
```

### Transformers 4.30 → 4.36 Migration

#### **New Features to Adopt**
```python
# Transformers 4.36: Enhanced model loading
from transformers import AutoModel

# New device_map options
model = AutoModel.from_pretrained(
    "model_name",
    device_map="auto",  # Enhanced auto mapping
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

# New quantization options
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

#### **API Changes**
```python
# Transformers 4.36: Enhanced tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "model_name",
    trust_remote_code=True,  # New parameter
    use_fast=True  # New parameter
)

# Enhanced generation
outputs = model.generate(
    input_ids,
    return_dict_in_generate=True,  # New default
    output_scores=True,
    output_attentions=True
)
```

### Diffusers 0.21 → 0.25 Migration

#### **New Features to Adopt**
```python
# Diffusers 0.25: Enhanced pipeline loading
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "model_name",
    variant="fp16",  # New parameter
    use_safetensors=True,  # New parameter
    torch_dtype=torch.float16
)

# New optimization methods
pipeline.enable_attention_slicing(slice_size="auto")
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()  # New method
```

#### **API Changes**
```python
# Diffusers 0.25: Enhanced UNet configuration
from diffusers import UNet2DConditionModel

model = UNet2DConditionModel.from_pretrained(
    "model_name",
    use_linear_projection=True,  # New parameter
    only_cross_attention=False,  # New parameter
    upcast_attention=False       # New parameter
)
```

### Gradio 3.40 → 4.0 Migration

#### **Major Changes**
```python
# Gradio 4.0: New component structure
import gradio as gr

# Old way (Gradio 3.40)
textbox = gr.Textbox(label="Input", lines=3)

# New way (Gradio 4.0)
textbox = gr.Textbox(
    label="Input",
    lines=3,
    show_label=True,  # New parameter
    container=True,   # New parameter
    scale=1          # New parameter
)

# New event handling
btn.click(
    fn=function,
    inputs=inputs,
    outputs=outputs,
    api_name="function_name",  # New parameter
    show_progress=True,        # New parameter
    queue=True                 # New parameter
)
```

#### **New Components**
```python
# Gradio 4.0: New components
chatbot = gr.Chatbot(
    show_copy_button=True,  # New feature
    avatar_images=None,     # New feature
    layout="bubble"         # New feature
)

gallery = gr.Gallery(
    allow_preview=True,     # New feature
    selected_index=None,    # New feature
    object_fit="contain"    # New feature
)
```

---

## 🧪 Version Testing

### Automated Version Check

```python
import torch
import transformers
import diffusers
import gradio
import sys

def check_versions():
    """Check if all libraries are at recommended versions."""
    
    version_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "gradio": gradio.__version__
    }
    
    print("📦 Version Information:")
    for lib, version in version_info.items():
        print(f"  {lib}: {version}")
    
    # Check CUDA compatibility
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
    
    return version_info

def test_compatibility():
    """Test basic functionality with current versions."""
    
    print("🧪 Testing Compatibility...")
    
    # Test PyTorch
    try:
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("✅ PyTorch: Basic operations work")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
    
    # Test Transformers
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✅ Transformers: Model loading works")
    except Exception as e:
        print(f"❌ Transformers: {e}")
    
    # Test Diffusers
    try:
        from diffusers import DiffusionPipeline
        print("✅ Diffusers: Pipeline import works")
    except Exception as e:
        print(f"❌ Diffusers: {e}")
    
    # Test Gradio
    try:
        import gradio as gr
        interface = gr.Interface(lambda x: x, "text", "text")
        print("✅ Gradio: Interface creation works")
    except Exception as e:
        print(f"❌ Gradio: {e}")

if __name__ == "__main__":
    check_versions()
    test_compatibility()
```

### Compatibility Test Suite

```python
import unittest
import torch
import transformers
import diffusers
import gradio

class VersionCompatibilityTest(unittest.TestCase):
    """Test suite for version compatibility."""
    
    def test_pytorch_features(self):
        """Test PyTorch 2.1+ features."""
        # Test torch.compile
        if hasattr(torch, 'compile'):
            model = torch.nn.Linear(10, 1)
            compiled_model = torch.compile(model)
            x = torch.randn(5, 10)
            output = compiled_model(x)
            self.assertEqual(output.shape, (5, 1))
        
        # Test flash attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    def test_transformers_features(self):
        """Test Transformers 4.36+ features."""
        from transformers import AutoTokenizer, AutoModel
        
        # Test enhanced model loading
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            trust_remote_code=True
        )
        self.assertIsNotNone(tokenizer)
    
    def test_diffusers_features(self):
        """Test Diffusers 0.25+ features."""
        from diffusers import DiffusionPipeline
        
        # Test pipeline loading with new parameters
        # Note: This is a mock test to avoid downloading large models
        self.assertTrue(hasattr(DiffusionPipeline, 'from_pretrained'))
    
    def test_gradio_features(self):
        """Test Gradio 4.0+ features."""
        import gradio as gr
        
        # Test new component parameters
        textbox = gr.Textbox(
            label="Test",
            show_label=True,
            container=True
        )
        self.assertIsNotNone(textbox)

def run_compatibility_tests():
    """Run all compatibility tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == "__main__":
    run_compatibility_tests()
```

---

## 🔧 Update Scripts

### Automated Update Script

```python
import subprocess
import sys
import os

def update_libraries():
    """Update all libraries to recommended versions."""
    
    print("🔄 Updating libraries to recommended versions...")
    
    # Define target versions
    target_versions = {
        "torch": "2.1.0",
        "torchvision": "0.16.0",
        "torchaudio": "2.1.0",
        "transformers": "4.36.0",
        "diffusers": "0.25.0",
        "gradio": "4.0.0",
        "accelerate": "0.25.0",
        "xformers": "0.0.23",
        "peft": "0.7.0",
        "bitsandbytes": "0.41.0",
        "safetensors": "0.4.0"
    }
    
    for package, version in target_versions.items():
        try:
            print(f"📦 Updating {package} to {version}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package}>={version}"
            ])
            print(f"✅ {package} updated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update {package}: {e}")
    
    print("🎉 Library update completed!")

def create_requirements_file():
    """Create requirements.txt with recommended versions."""
    
    requirements = """# AI Video System Requirements
# Latest stable versions as of 2024

# Core PyTorch
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Transformers and Diffusers
transformers>=4.36.0
diffusers>=0.25.0
accelerate>=0.25.0

# UI and Interface
gradio>=4.0.0

# Optimization
xformers>=0.0.23
peft>=0.7.0
bitsandbytes>=0.41.0
safetensors>=0.4.0

# Additional dependencies
numpy>=1.24.0
pillow>=10.0.0
opencv-python>=4.8.0
scipy>=1.11.0
matplotlib>=3.7.0
tqdm>=4.65.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("📄 requirements.txt created with recommended versions")

if __name__ == "__main__":
    create_requirements_file()
    update_libraries()
```

### Environment Setup Script

```python
import os
import subprocess
import sys

def setup_environment():
    """Set up development environment with correct versions."""
    
    print("🚀 Setting up AI Video development environment...")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    
    # Determine activation script
    if os.name == "nt":  # Windows
        activate_script = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_script = "venv/bin/activate"
    
    # Install requirements
    print("📦 Installing requirements...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    print("✅ Environment setup completed!")
    print(f"🔧 Activate with: source {activate_script}")

if __name__ == "__main__":
    setup_environment()
```

---

## 🚨 Breaking Changes and Workarounds

### Known Issues and Solutions

#### **PyTorch 2.1 Breaking Changes**
```python
# Issue: Checkpoint API change
# Solution: Add use_reentrant=False
from torch.utils.checkpoint import checkpoint

# Before (PyTorch 2.0)
x = checkpoint(layer, x)

# After (PyTorch 2.1)
x = checkpoint(layer, x, use_reentrant=False)
```

#### **Transformers 4.36 Breaking Changes**
```python
# Issue: Tokenizer API changes
# Solution: Add trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(
    "model_name",
    trust_remote_code=True  # Required for some models
)
```

#### **Diffusers 0.25 Breaking Changes**
```python
# Issue: Pipeline loading changes
# Solution: Use new parameters
pipeline = DiffusionPipeline.from_pretrained(
    "model_name",
    variant="fp16",  # Specify variant
    use_safetensors=True  # Use safetensors format
)
```

#### **Gradio 4.0 Breaking Changes**
```python
# Issue: Component API changes
# Solution: Add new parameters
textbox = gr.Textbox(
    label="Input",
    show_label=True,  # Explicitly show label
    container=True    # Use container layout
)
```

---

## 📊 Version Monitoring

### Automated Version Monitoring

```python
import requests
import json
from datetime import datetime

def check_latest_versions():
    """Check latest versions from PyPI."""
    
    packages = [
        "torch", "transformers", "diffusers", "gradio",
        "accelerate", "xformers", "peft", "bitsandbytes"
    ]
    
    latest_versions = {}
    
    for package in packages:
        try:
            response = requests.get(f"https://pypi.org/pypi/{package}/json")
            if response.status_code == 200:
                data = response.json()
                latest_versions[package] = data["info"]["version"]
        except Exception as e:
            print(f"❌ Failed to get version for {package}: {e}")
    
    return latest_versions

def generate_version_report():
    """Generate version compatibility report."""
    
    current_versions = check_versions()
    latest_versions = check_latest_versions()
    
    print("📊 Version Compatibility Report")
    print("=" * 50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for package in current_versions.keys():
        if package in latest_versions:
            current = current_versions[package]
            latest = latest_versions[package]
            
            if current == latest:
                status = "✅ Up to date"
            else:
                status = f"⚠️  Update available: {latest}"
            
            print(f"{package:15} {current:10} {status}")
    
    print()
    print("📋 Recommendations:")
    print("- Update packages marked with ⚠️")
    print("- Test thoroughly after updates")
    print("- Check breaking changes documentation")

if __name__ == "__main__":
    generate_version_report()
```

This version compatibility guide ensures our AI video system remains up-to-date with the latest stable versions while providing clear migration paths and testing procedures. 