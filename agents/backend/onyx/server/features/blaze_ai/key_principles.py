from __future__ import annotations

# Canonical key principles for Blaze AI module (programmatic reference)

KEY_PRINCIPLES: dict[str, list[str]] = {
    "project": [
        "Begin with clear problem definition and dataset analysis",
        "Create modular code structures for models, data, training, evaluation",
        "Use YAML/externally managed configs for hyperparameters and settings",
        "Track experiments and checkpoint models",
        "Use version control for code and configs",
    ],
    "dependencies": [
        "PyTorch (torch)",
        "Transformers",
        "Diffusers",
        "Gradio",
        "NumPy",
        "tqdm",
        "TensorBoard / Weights & Biases",
    ],
    "dl_training": [
        "Custom nn.Module architectures with proper init/normalization",
        "Appropriate loss functions and optimizers",
        "Correct tokenization and sequence handling",
        "Mixed precision (torch.cuda.amp) when applicable",
        "Gradient clipping and non-finite handling",
        "Early stopping and LR scheduling",
        "Efficient DataLoader (workers, pin_memory, prefetch)",
        "DataParallel/DistributedDataParallel for multi-GPU",
        "Gradient accumulation for large batches",
        "Use autograd.detect_anomaly() when debugging",
    ],
    "llms_transformers": [
        "Use pre-trained models/tokenizers from Transformers",
        "Efficient generation with configurable decoding",
        "Optional LoRA/P-tuning for efficient fine-tuning",
        "Implement attention and positional encodings correctly in custom models",
    ],
    "diffusion": [
        "Use Diffusers StableDiffusion/SDXL pipelines",
        "Understand forward (q) and reverse (p) diffusion",
        "Choose appropriate noise schedulers and samplers",
        "AMP on GPU for faster inference",
    ],
    "api_ui": [
        "FastAPI routes with robust error handling",
        "Gradio demos with validation and clear errors",
        "User-friendly interfaces showcasing capabilities",
    ],
    "observability": [
        "Structured logging and exception handling",
        "Profiling (cProfile/Timer) to find bottlenecks",
        "Metrics for model quality and throughput",
    ],
}


