from __future__ import annotations

from typing import Dict


OFFICIAL_REFERENCES: Dict[str, Dict[str, str]] = {
    "pytorch": {
        "docs": "https://pytorch.org/docs/stable/index.html",
        "tutorials": "https://pytorch.org/tutorials/",
        "distributed": "https://pytorch.org/docs/stable/distributed.html",
        "amp": "https://pytorch.org/docs/stable/amp.html",
        "compile": "https://pytorch.org/get-started/pytorch-2.0/",
        "data": "https://pytorch.org/docs/stable/data.html",
    },
    "transformers": {
        "docs": "https://huggingface.co/docs/transformers/index",
        "generation": "https://huggingface.co/docs/transformers/generation_strategies",
        "models": "https://huggingface.co/docs/transformers/main_classes/model",
        "quantization": "https://huggingface.co/docs/transformers/main_classes/quantization",
        "accelerate": "https://huggingface.co/docs/accelerate/",
        "peft": "https://huggingface.co/docs/peft",
    },
    "diffusers": {
        "docs": "https://huggingface.co/docs/diffusers/index",
        "sd_pipelines": "https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion",
        "schedulers": "https://huggingface.co/docs/diffusers/api/schedulers/overview",
        "optimization": "https://huggingface.co/docs/diffusers/optimization/xformers",
    },
    "gradio": {
        "docs": "https://www.gradio.app/docs",
        "guides": "https://www.gradio.app/guides",
        "components": "https://www.gradio.app/docs/components",
        "events": "https://www.gradio.app/docs/events",
    },
}


def get_references() -> Dict[str, Dict[str, str]]:
    return OFFICIAL_REFERENCES


