from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import json
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import yaml
from pathlib import Path
    from official PyTorch, Transformers, Diffusers, and Gradio documentation.
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from diffusers import DiffusionPipeline
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
import gradio as gr
import gradio as gr
from typing import Any, List, Dict, Optional
import asyncio
"""
Official Documentation Reference System
=====================================

This module provides comprehensive references to official documentation best practices
and up-to-date APIs for PyTorch, Transformers, Diffusers, and Gradio.

Features:
- Official documentation links and references
- Best practices from official sources
- API version compatibility tracking
- Code examples from official documentation
- Migration guides and deprecation warnings
- Performance optimization recommendations
"""


logger = logging.getLogger(__name__)


@dataclass
class LibraryInfo:
    """Information about a library including versions and documentation."""
    name: str
    current_version: str
    min_supported_version: str
    documentation_url: str
    github_url: str
    pip_package: str
    conda_package: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class APIRef:
    """Reference to a specific API or feature."""
    name: str
    description: str
    official_docs_url: str
    code_example: str
    best_practices: List[str]
    deprecation_warning: Optional[str] = None
    migration_guide: Optional[str] = None
    performance_tips: List[str] = field(default_factory=list)


@dataclass
class BestPractice:
    """Best practice from official documentation."""
    title: str
    description: str
    source: str
    code_example: str
    category: str
    importance: str  # "critical", "important", "recommended"


class OfficialDocsReference:
    """
    Comprehensive reference system for official documentation of ML libraries.
    
    Provides access to best practices, API references, and up-to-date information
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        
    """__init__ function."""
self.cache_dir = Path(cache_dir) if cache_dir else Path("./docs_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Library information
        self.libraries = {
            "pytorch": LibraryInfo(
                name="PyTorch",
                current_version="2.1.0",
                min_supported_version="1.13.0",
                documentation_url="https://pytorch.org/docs/stable/",
                github_url="https://github.com/pytorch/pytorch",
                pip_package="torch",
                conda_package="pytorch"
            ),
            "transformers": LibraryInfo(
                name="Transformers",
                current_version="4.35.0",
                min_supported_version="4.20.0",
                documentation_url="https://huggingface.co/docs/transformers/",
                github_url="https://github.com/huggingface/transformers",
                pip_package="transformers"
            ),
            "diffusers": LibraryInfo(
                name="Diffusers",
                current_version="0.24.0",
                min_supported_version="0.18.0",
                documentation_url="https://huggingface.co/docs/diffusers/",
                github_url="https://github.com/huggingface/diffusers",
                pip_package="diffusers"
            ),
            "gradio": LibraryInfo(
                name="Gradio",
                current_version="4.0.0",
                min_supported_version="3.50.0",
                documentation_url="https://gradio.app/docs/",
                github_url="https://github.com/gradio-app/gradio",
                pip_package="gradio"
            )
        }
        
        # Initialize reference data
        self._init_pytorch_refs()
        self._init_transformers_refs()
        self._init_diffusers_refs()
        self._init_gradio_refs()
        
    def _init_pytorch_refs(self) -> Any:
        """Initialize PyTorch API references and best practices."""
        self.pytorch_refs = {
            "mixed_precision": APIRef(
                name="torch.cuda.amp",
                description="Automatic Mixed Precision for faster training",
                official_docs_url="https://pytorch.org/docs/stable/amp.html",
                code_example="""

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
""",
                best_practices=[
                    "Use GradScaler for gradient scaling",
                    "Wrap forward pass in autocast context",
                    "Scale loss before backward pass",
                    "Update scaler after optimizer step"
                ],
                performance_tips=[
                    "Can provide 2-3x speedup on modern GPUs",
                    "Reduces memory usage by ~50%",
                    "Works best with batch sizes >= 32"
                ]
            ),
            "data_loading": APIRef(
                name="DataLoader",
                description="Efficient data loading with multiprocessing",
                official_docs_url="https://pytorch.org/docs/stable/data.html",
                code_example="""

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
""",
                best_practices=[
                    "Use num_workers > 0 for CPU-bound data loading",
                    "Enable pin_memory for GPU training",
                    "Use persistent_workers=True for efficiency",
                    "Set appropriate batch_size based on GPU memory"
                ],
                performance_tips=[
                    "num_workers=4-8 typically optimal",
                    "pin_memory=True can improve GPU transfer speed",
                    "persistent_workers=True reduces worker startup overhead"
                ]
            ),
            "model_checkpointing": APIRef(
                name="Model Checkpointing",
                description="Save and load model states efficiently",
                official_docs_url="https://pytorch.org/docs/stable/checkpoint.html",
                code_example="""
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
""",
                best_practices=[
                    "Save both model and optimizer state",
                    "Include training metadata (epoch, loss, etc.)",
                    "Use torch.save() for PyTorch objects",
                    "Handle device placement when loading"
                ]
            ),
            "distributed_training": APIRef(
                name="Distributed Training",
                description="Multi-GPU and multi-node training",
                official_docs_url="https://pytorch.org/docs/stable/distributed.html",
                code_example="""

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
""",
                best_practices=[
                    "Use DistributedDataParallel for multi-GPU",
                    "Use DistributedSampler for data distribution",
                    "Initialize process group before model creation",
                    "Handle rank-specific operations"
                ]
            )
        }
        
    def _init_transformers_refs(self) -> Any:
        """Initialize Transformers API references and best practices."""
        self.transformers_refs = {
            "model_loading": APIRef(
                name="Model Loading",
                description="Load pre-trained models efficiently",
                official_docs_url="https://huggingface.co/docs/transformers/model_doc/auto",
                code_example="""

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# For specific tasks
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
""",
                best_practices=[
                    "Use AutoModel classes for flexibility",
                    "Load tokenizer from same model",
                    "Specify task-specific model classes",
                    "Use device_map for large models"
                ]
            ),
            "tokenization": APIRef(
                name="Tokenization",
                description="Text tokenization and preprocessing",
                official_docs_url="https://huggingface.co/docs/transformers/preprocessing",
                code_example="""
# Basic tokenization
inputs = tokenizer("Hello world!", return_tensors="pt")

# Batch tokenization
texts = ["Hello world!", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# For training
inputs = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
""",
                best_practices=[
                    "Use padding and truncation for batch processing",
                    "Set appropriate max_length",
                    "Use return_tensors='pt' for PyTorch",
                    "Handle special tokens properly"
                ]
            ),
            "training": APIRef(
                name="Training",
                description="Fine-tuning pre-trained models",
                official_docs_url="https://huggingface.co/docs/transformers/training",
                code_example="""

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
""",
                best_practices=[
                    "Use Trainer class for standard training",
                    "Set appropriate learning rate and warmup",
                    "Use gradient accumulation for large models",
                    "Enable logging and checkpointing"
                ]
            )
        }
        
    def _init_diffusers_refs(self) -> Any:
        """Initialize Diffusers API references and best practices."""
        self.diffusers_refs = {
            "pipeline_usage": APIRef(
                name="Diffusion Pipeline",
                description="Use pre-trained diffusion models",
                official_docs_url="https://huggingface.co/docs/diffusers/using-diffusers/using_diffusion_pipeline",
                code_example="""

# Load pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Generate images
image = pipeline("A beautiful sunset").images[0]
image.save("sunset.png")

# Batch generation
images = pipeline(["prompt1", "prompt2"], num_inference_steps=50)
""",
                best_practices=[
                    "Use torch.float16 for memory efficiency",
                    "Set appropriate num_inference_steps",
                    "Use batch processing for multiple images",
                    "Enable attention slicing for large models"
                ]
            ),
            "custom_training": APIRef(
                name="Custom Training",
                description="Train custom diffusion models",
                official_docs_url="https://huggingface.co/docs/diffusers/training/overview",
                code_example="""

# Initialize model and scheduler
model = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5")

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=1000)

for batch in dataloader:
    noise = torch.randn_like(batch)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch.shape[0],))
    noisy_batch = scheduler.add_noise(batch, noise, timesteps)
    
    noise_pred = model(noisy_batch, timesteps).sample
    loss = F.mse_loss(noise_pred, noise)
    
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
""",
                best_practices=[
                    "Use appropriate noise scheduler",
                    "Implement proper timestep sampling",
                    "Use gradient clipping for stability",
                    "Monitor loss convergence"
                ]
            ),
            "memory_optimization": APIRef(
                name="Memory Optimization",
                description="Optimize memory usage for diffusion models",
                official_docs_url="https://huggingface.co/docs/diffusers/optimization/memory",
                code_example="""
# Enable attention slicing
pipeline.enable_attention_slicing()

# Enable model offloading
pipeline.enable_model_cpu_offload()

# Use sequential CPU offloading
pipeline.enable_sequential_cpu_offload()

# Enable xformers memory efficient attention
pipeline.enable_xformers_memory_efficient_attention()
""",
                best_practices=[
                    "Use attention slicing for large models",
                    "Enable CPU offloading when needed",
                    "Use xformers for memory efficiency",
                    "Consider model sharding for very large models"
                ]
            )
        }
        
    def _init_gradio_refs(self) -> Any:
        """Initialize Gradio API references and best practices."""
        self.gradio_refs = {
            "interface_creation": APIRef(
                name="Interface Creation",
                description="Create Gradio interfaces for ML models",
                official_docs_url="https://gradio.app/docs/interface",
                code_example="""

def predict(text) -> Any:
    # Your model prediction logic here
    return f"Prediction: {text}"

# Create interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Textbox(label="Output"),
    title="My Model Demo",
    description="Enter text to get predictions"
)

# Launch
interface.launch()
""",
                best_practices=[
                    "Use appropriate input/output components",
                    "Provide clear labels and descriptions",
                    "Handle errors gracefully",
                    "Use examples for better UX"
                ]
            ),
            "advanced_components": APIRef(
                name="Advanced Components",
                description="Use advanced Gradio components",
                official_docs_url="https://gradio.app/docs/components",
                code_example="""

def process_image(image, text) -> Any:
    # Process image and text
    return processed_image, f"Processed: {text}"

# Advanced interface
with gr.Blocks() as demo:
    gr.Markdown("# Advanced Demo")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image")
            text_input = gr.Textbox(label="Input Text")
            submit_btn = gr.Button("Process")
        
        with gr.Column():
            image_output = gr.Image(label="Output Image")
            text_output = gr.Textbox(label="Output Text")
    
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, text_input],
        outputs=[image_output, text_output]
    )

demo.launch()
""",
                best_practices=[
                    "Use Blocks for complex layouts",
                    "Organize components in rows and columns",
                    "Use appropriate event handlers",
                    "Provide clear visual hierarchy"
                ]
            ),
            "deployment": APIRef(
                name="Deployment",
                description="Deploy Gradio apps",
                official_docs_url="https://gradio.app/docs/deployment",
                code_example="""
# Local deployment
interface.launch(server_name="0.0.0.0", server_port=7860)

# Hugging Face Spaces deployment
# Create app.py with your interface
# Add requirements.txt
# Push to Hugging Face Spaces

# Docker deployment
# Create Dockerfile
# Build and run container
""",
                best_practices=[
                    "Use appropriate server settings",
                    "Handle CORS for web deployment",
                    "Use environment variables for configuration",
                    "Implement proper error handling"
                ]
            )
        }
    
    def get_library_info(self, library_name: str) -> Optional[LibraryInfo]:
        """Get information about a specific library."""
        return self.libraries.get(library_name.lower())
    
    async def get_api_reference(self, library_name: str, api_name: str) -> Optional[APIRef]:
        """Get API reference for a specific library and API."""
        refs = getattr(self, f"{library_name.lower()}_refs", {})
        return refs.get(api_name)
    
    def get_best_practices(self, library_name: str, category: Optional[str] = None) -> List[BestPractice]:
        """Get best practices for a library."""
        practices = []
        
        if library_name.lower() == "pytorch":
            practices = [
                BestPractice(
                    title="Use torch.cuda.amp for Mixed Precision",
                    description="Enable automatic mixed precision for faster training",
                    source="PyTorch Official Docs",
                    code_example="from torch.cuda.amp import autocast, GradScaler",
                    category="performance",
                    importance="critical"
                ),
                BestPractice(
                    title="Use DataLoader with num_workers",
                    description="Enable multiprocessing for data loading",
                    source="PyTorch Official Docs",
                    code_example="DataLoader(dataset, num_workers=4, pin_memory=True)",
                    category="performance",
                    importance="important"
                ),
                BestPractice(
                    title="Use DistributedDataParallel for Multi-GPU",
                    description="Scale training across multiple GPUs",
                    source="PyTorch Official Docs",
                    code_example="model = DistributedDataParallel(model)",
                    category="scalability",
                    importance="important"
                )
            ]
        elif library_name.lower() == "transformers":
            practices = [
                BestPractice(
                    title="Use AutoModel Classes",
                    description="Use AutoModel for flexible model loading",
                    source="Transformers Official Docs",
                    code_example="AutoModel.from_pretrained('model-name')",
                    category="model_loading",
                    importance="critical"
                ),
                BestPractice(
                    title="Proper Tokenization",
                    description="Use padding and truncation for batch processing",
                    source="Transformers Official Docs",
                    code_example="tokenizer(texts, padding=True, truncation=True)",
                    category="preprocessing",
                    importance="important"
                ),
                BestPractice(
                    title="Use Trainer for Training",
                    description="Use Trainer class for standard training workflows",
                    source="Transformers Official Docs",
                    code_example="Trainer(model, args, train_dataset)",
                    category="training",
                    importance="important"
                )
            ]
        elif library_name.lower() == "diffusers":
            practices = [
                BestPractice(
                    title="Use torch.float16 for Memory Efficiency",
                    description="Reduce memory usage with half precision",
                    source="Diffusers Official Docs",
                    code_example="pipeline = DiffusionPipeline.from_pretrained(..., torch_dtype=torch.float16)",
                    category="memory",
                    importance="critical"
                ),
                BestPractice(
                    title="Enable Attention Slicing",
                    description="Reduce memory usage for large models",
                    source="Diffusers Official Docs",
                    code_example="pipeline.enable_attention_slicing()",
                    category="memory",
                    importance="important"
                ),
                BestPractice(
                    title="Use Appropriate Inference Steps",
                    description="Balance quality vs speed",
                    source="Diffusers Official Docs",
                    code_example="pipeline(prompt, num_inference_steps=50)",
                    category="performance",
                    importance="recommended"
                )
            ]
        elif library_name.lower() == "gradio":
            practices = [
                BestPractice(
                    title="Use Appropriate Components",
                    description="Choose the right input/output components",
                    source="Gradio Official Docs",
                    code_example="gr.Image(), gr.Textbox(), gr.Button()",
                    category="ui",
                    importance="important"
                ),
                BestPractice(
                    title="Handle Errors Gracefully",
                    description="Provide user-friendly error messages",
                    source="Gradio Official Docs",
                    code_example="try: ... except Exception as e: return str(e)",
                    category="user_experience",
                    importance="important"
                ),
                BestPractice(
                    title="Use Examples",
                    description="Provide example inputs for better UX",
                    source="Gradio Official Docs",
                    code_example="interface = gr.Interface(..., examples=[...])",
                    category="user_experience",
                    importance="recommended"
                )
            ]
        
        if category:
            practices = [p for p in practices if p.category == category]
        
        return practices
    
    def check_version_compatibility(self, library_name: str, version: str) -> Dict[str, Any]:
        """Check if a version is compatible with current best practices."""
        lib_info = self.get_library_info(library_name)
        if not lib_info:
            return {"compatible": False, "error": f"Unknown library: {library_name}"}
        
        # Simple version comparison (can be enhanced)
        current_major = int(lib_info.current_version.split('.')[0])
        current_minor = int(lib_info.current_version.split('.')[1])
        version_major = int(version.split('.')[0])
        version_minor = int(version.split('.')[1])
        
        compatible = version_major >= current_major and version_minor >= current_minor
        
        return {
            "compatible": compatible,
            "current_version": lib_info.current_version,
            "requested_version": version,
            "recommendation": "Upgrade" if not compatible else "Current version is good"
        }
    
    def generate_migration_guide(self, library_name: str, from_version: str, to_version: str) -> Dict[str, Any]:
        """Generate migration guide between versions."""
        # This would typically fetch from official migration guides
        # For now, return a template
        return {
            "library": library_name,
            "from_version": from_version,
            "to_version": to_version,
            "breaking_changes": [],
            "deprecations": [],
            "new_features": [],
            "migration_steps": [
                f"1. Update {library_name} to version {to_version}",
                "2. Review breaking changes",
                "3. Update deprecated API calls",
                "4. Test functionality",
                "5. Update dependencies if needed"
            ]
        }
    
    def export_references(self, output_file: str, format: str = "json"):
        """Export all references to a file."""
        data = {
            "libraries": {name: {
                "name": lib.name,
                "current_version": lib.current_version,
                "documentation_url": lib.documentation_url
            } for name, lib in self.libraries.items()},
            "api_references": {
                "pytorch": {name: {
                    "name": ref.name,
                    "description": ref.description,
                    "official_docs_url": ref.official_docs_url
                } for name, ref in self.pytorch_refs.items()},
                "transformers": {name: {
                    "name": ref.name,
                    "description": ref.description,
                    "official_docs_url": ref.official_docs_url
                } for name, ref in self.transformers_refs.items()},
                "diffusers": {name: {
                    "name": ref.name,
                    "description": ref.description,
                    "official_docs_url": ref.official_docs_url
                } for name, ref in self.diffusers_refs.items()},
                "gradio": {name: {
                    "name": ref.name,
                    "description": ref.description,
                    "official_docs_url": ref.official_docs_url
                } for name, ref in self.gradio_refs.items()}
            }
        }
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(data, f, indent=2)
        elif format.lower() == "yaml":
            with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"References exported to {output_file}")
    
    def get_performance_recommendations(self, library_name: str) -> List[str]:
        """Get performance recommendations for a library."""
        recommendations = {
            "pytorch": [
                "Use torch.cuda.amp for automatic mixed precision",
                "Enable pin_memory=True in DataLoader",
                "Use appropriate num_workers for data loading",
                "Use DistributedDataParallel for multi-GPU training",
                "Profile your code with torch.profiler",
                "Use torch.jit.script for model optimization"
            ],
            "transformers": [
                "Use device_map for large model loading",
                "Enable gradient checkpointing for memory efficiency",
                "Use appropriate batch sizes",
                "Use mixed precision training",
                "Cache tokenized datasets",
                "Use model parallelism for very large models"
            ],
            "diffusers": [
                "Use torch.float16 for inference",
                "Enable attention slicing",
                "Use model offloading for large models",
                "Optimize inference steps",
                "Use batch processing when possible",
                "Enable xformers memory efficient attention"
            ],
            "gradio": [
                "Use appropriate queue settings",
                "Optimize model inference time",
                "Use caching for expensive operations",
                "Minimize interface complexity",
                "Use appropriate server settings",
                "Monitor memory usage"
            ]
        }
        
        return recommendations.get(library_name.lower(), [])
    
    def validate_code_snippet(self, code: str, library_name: str) -> Dict[str, Any]:
        """Validate a code snippet against best practices."""
        # This is a simplified validation - in practice, you'd use AST parsing
        issues = []
        recommendations = []
        
        if library_name.lower() == "pytorch":
            if "DataLoader" in code and "num_workers" not in code:
                issues.append("DataLoader without num_workers specified")
                recommendations.append("Add num_workers=4 for better performance")
            
            if "model.train()" in code and "torch.cuda.amp" not in code:
                recommendations.append("Consider using torch.cuda.amp for mixed precision")
        
        elif library_name.lower() == "transformers":
            if "AutoModel" not in code and "from_pretrained" in code:
                recommendations.append("Consider using AutoModel classes for flexibility")
            
            if "tokenizer" in code and "padding" not in code:
                recommendations.append("Consider adding padding for batch processing")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize reference system
    ref_system = OfficialDocsReference()
    
    # Get library information
    pytorch_info = ref_system.get_library_info("pytorch")
    print(f"PyTorch current version: {pytorch_info.current_version}")
    
    # Get API reference
    amp_ref = ref_system.get_api_reference("pytorch", "mixed_precision")
    print(f"AMP description: {amp_ref.description}")
    
    # Get best practices
    practices = ref_system.get_best_practices("pytorch", "performance")
    for practice in practices:
        print(f"- {practice.title}: {practice.description}")
    
    # Check version compatibility
    compat = ref_system.check_version_compatibility("pytorch", "1.12.0")
    print(f"Version compatibility: {compat}")
    
    # Get performance recommendations
    recommendations = ref_system.get_performance_recommendations("pytorch")
    print("Performance recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
    
    # Export references
    ref_system.export_references("official_docs_reference.json") 