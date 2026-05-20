"""
Library Manager - Gestor de Librerías
======================================

Gestiona y organiza las dependencias de librerías de manera modular y eficiente.
"""

from typing import Dict, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum


class LibraryCategory(Enum):
    """Categorías de librerías"""
    CORE = "core"
    HTTP = "http"
    AI_ML = "ai_ml"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMERS = "transformers"
    DIFFUSION = "diffusion"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    NLP = "nlp"
    UTILITIES = "utilities"
    TESTING = "testing"
    MONITORING = "monitoring"
    CONFIG = "config"
    OPTIMIZATION = "optimization"
    INTERFACE = "interface"


@dataclass
class Library:
    """Representa una librería con su información"""
    name: str
    version: str
    category: LibraryCategory
    description: str = ""
    optional: bool = False
    dependencies: List[str] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)


class LibraryManager:
    """Gestor centralizado de librerías"""
    
    def __init__(self):
        self._libraries: Dict[str, Library] = {}
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Inicializa el catálogo de librerías"""
        libraries = [
            Library("fastapi", ">=0.115.0", LibraryCategory.CORE, "FastAPI framework"),
            Library("uvicorn", ">=0.32.0", LibraryCategory.CORE, "ASGI server", extras=["standard"]),
            Library("pydantic", ">=2.9.0", LibraryCategory.CORE, "Data validation"),
            Library("pydantic-settings", ">=2.5.0", LibraryCategory.CORE, "Settings management"),
            Library("python-multipart", ">=0.0.12", LibraryCategory.CORE, "File uploads"),
            
            Library("httpx", ">=0.27.0", LibraryCategory.HTTP, "HTTP client"),
            Library("aiofiles", ">=24.1.0", LibraryCategory.HTTP, "Async file I/O"),
            
            Library("openai", ">=1.54.0", LibraryCategory.AI_ML, "OpenAI API"),
            Library("anthropic", ">=0.39.0", LibraryCategory.AI_ML, "Anthropic API"),
            Library("langchain", ">=0.3.0", LibraryCategory.AI_ML, "LangChain framework"),
            
            Library("torch", ">=2.1.0", LibraryCategory.DEEP_LEARNING, "PyTorch"),
            Library("torchvision", ">=0.16.0", LibraryCategory.DEEP_LEARNING, "PyTorch Vision"),
            Library("torchaudio", ">=2.1.0", LibraryCategory.DEEP_LEARNING, "PyTorch Audio"),
            
            Library("transformers", ">=4.36.0", LibraryCategory.TRANSFORMERS, "HuggingFace Transformers"),
            Library("accelerate", ">=0.25.0", LibraryCategory.TRANSFORMERS, "Training acceleration"),
            Library("sentencepiece", ">=0.1.99", LibraryCategory.TRANSFORMERS, "Tokenization"),
            Library("tokenizers", ">=0.15.0", LibraryCategory.TRANSFORMERS, "Fast tokenization"),
            Library("datasets", ">=2.14.0", LibraryCategory.TRANSFORMERS, "Dataset library"),
            Library("safetensors", ">=0.4.0", LibraryCategory.TRANSFORMERS, "Safe tensor storage"),
            Library("peft", ">=0.7.0", LibraryCategory.TRANSFORMERS, "Parameter-efficient fine-tuning"),
            Library("bitsandbytes", ">=0.41.0", LibraryCategory.TRANSFORMERS, "Quantization", optional=True),
            Library("trl", ">=0.7.0", LibraryCategory.TRANSFORMERS, "Transformer RL"),
            
            Library("diffusers", ">=0.25.0", LibraryCategory.DIFFUSION, "Diffusion models"),
            Library("xformers", ">=0.0.23", LibraryCategory.DIFFUSION, "Memory optimization", optional=True),
            
            Library("opencv-python", ">=4.8.0", LibraryCategory.VISION, "Computer vision"),
            Library("pillow", ">=10.0.0", LibraryCategory.VISION, "Image processing"),
            Library("timm", ">=0.9.0", LibraryCategory.VISION, "Pre-trained vision models"),
            Library("albumentations", ">=1.3.0", LibraryCategory.VISION, "Data augmentation"),
            Library("imageio", ">=2.31.0", LibraryCategory.VISION, "Image I/O"),
            
            Library("librosa", ">=0.10.0", LibraryCategory.AUDIO, "Audio analysis"),
            Library("soundfile", ">=0.12.0", LibraryCategory.AUDIO, "Audio I/O"),
            Library("audioread", ">=3.0.0", LibraryCategory.AUDIO, "Audio reading"),
            
            Library("moviepy", ">=1.0.3", LibraryCategory.VIDEO, "Video processing"),
            Library("ffmpeg-python", ">=0.2.0", LibraryCategory.VIDEO, "FFmpeg wrapper"),
            
            Library("nltk", ">=3.8.0", LibraryCategory.NLP, "NLP toolkit"),
            Library("spacy", ">=3.7.0", LibraryCategory.NLP, "NLP library"),
            Library("rouge-score", ">=0.1.2", LibraryCategory.NLP, "ROUGE metric"),
            Library("sacrebleu", ">=2.3.0", LibraryCategory.NLP, "BLEU metric"),
            
            Library("numpy", ">=1.24.0", LibraryCategory.UTILITIES, "Numerical computing"),
            Library("scipy", ">=1.11.0", LibraryCategory.UTILITIES, "Scientific computing"),
            Library("scikit-learn", ">=1.3.0", LibraryCategory.UTILITIES, "Machine learning"),
            Library("pandas", ">=2.0.0", LibraryCategory.UTILITIES, "Data manipulation"),
            Library("tqdm", ">=4.66.0", LibraryCategory.UTILITIES, "Progress bars"),
            Library("einops", ">=0.7.0", LibraryCategory.UTILITIES, "Tensor operations"),
            
            Library("gradio", ">=4.7.0", LibraryCategory.INTERFACE, "Interactive UI"),
            Library("plotly", ">=5.17.0", LibraryCategory.INTERFACE, "Interactive plots"),
            
            Library("tensorboard", ">=2.15.0", LibraryCategory.MONITORING, "Experiment tracking"),
            Library("wandb", ">=0.15.0", LibraryCategory.MONITORING, "Weights & Biases"),
            Library("psutil", ">=5.9.0", LibraryCategory.MONITORING, "System monitoring"),
            
            Library("pyyaml", ">=6.0.1", LibraryCategory.CONFIG, "YAML parser"),
            Library("omegaconf", ">=2.3.0", LibraryCategory.CONFIG, "Configuration management"),
            Library("hydra-core", ">=1.3.0", LibraryCategory.CONFIG, "Configuration framework"),
            
            Library("torchmetrics", ">=1.2.0", LibraryCategory.OPTIMIZATION, "PyTorch metrics"),
            Library("optuna", ">=3.4.0", LibraryCategory.OPTIMIZATION, "Hyperparameter optimization"),
            Library("ray", ">=2.8.0", LibraryCategory.OPTIMIZATION, "Distributed computing", extras=["tune"]),
            
            Library("matplotlib", ">=3.7.0", LibraryCategory.UTILITIES, "Plotting"),
            Library("seaborn", ">=0.12.0", LibraryCategory.UTILITIES, "Statistical plots"),
            Library("h5py", ">=3.9.0", LibraryCategory.UTILITIES, "HDF5 I/O"),
            
            Library("pytest", ">=8.3.0", LibraryCategory.TESTING, "Testing framework"),
            Library("pytest-asyncio", ">=0.23.7", LibraryCategory.TESTING, "Async testing"),
            Library("pytest-cov", ">=4.1.0", LibraryCategory.TESTING, "Coverage"),
            
            Library("python-dotenv", ">=1.0.1", LibraryCategory.UTILITIES, "Environment variables"),
            Library("rich", ">=13.7.0", LibraryCategory.UTILITIES, "Rich terminal output"),
            Library("typer", ">=0.9.0", LibraryCategory.UTILITIES, "CLI framework"),
            Library("structlog", ">=24.2.0", LibraryCategory.UTILITIES, "Structured logging"),
            Library("orjson", ">=3.10.0", LibraryCategory.UTILITIES, "Fast JSON"),
            Library("ujson", ">=5.9.0", LibraryCategory.UTILITIES, "Ultra-fast JSON"),
            Library("aiofiles", ">=24.1.0", LibraryCategory.UTILITIES, "Async file I/O"),
            Library("watchfiles", ">=0.21.0", LibraryCategory.UTILITIES, "File watching"),
            
            Library("python-jose", ">=3.3.0", LibraryCategory.UTILITIES, "JWT tokens", extras=["cryptography"]),
            Library("passlib", ">=1.7.4", LibraryCategory.UTILITIES, "Password hashing", extras=["bcrypt"]),
            
            Library("invisible-watermark", ">=0.2.0", LibraryCategory.DIFFUSION, "Image watermarks", optional=True),
        ]
        
        for lib in libraries:
            self._libraries[lib.name] = lib
    
    def get_libraries_by_category(self, category: LibraryCategory) -> List[Library]:
        """Obtiene librerías por categoría"""
        return [lib for lib in self._libraries.values() if lib.category == category]
    
    def get_libraries_by_keywords(self, keywords: Dict[str, Any]) -> Set[str]:
        """
        Determina qué librerías incluir basándose en keywords del proyecto.
        
        Args:
            keywords: Keywords extraídos del proyecto
            
        Returns:
            Set de nombres de librerías a incluir
        """
        libraries = set()
        
        libraries.update([
            "fastapi", "uvicorn[standard]", "pydantic", "pydantic-settings",
            "python-multipart", "httpx", "aiofiles",
            "python-dotenv", "rich", "structlog", "orjson"
        ])
        
        if keywords.get("is_deep_learning") or keywords.get("requires_pytorch"):
            libraries.update(["torch", "torchvision", "torchaudio"])
            libraries.add("numpy")
            libraries.add("scipy")
            libraries.add("pandas")
            libraries.add("tqdm")
            libraries.add("einops")
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            libraries.update([
                "transformers", "accelerate", "sentencepiece", "tokenizers",
                "datasets", "safetensors", "peft", "trl"
            ])
            if keywords.get("requires_quantization"):
                libraries.add("bitsandbytes")
        
        if keywords.get("is_diffusion"):
            libraries.update(["diffusers", "pillow", "opencv-python", "imageio"])
            if keywords.get("requires_optimization"):
                libraries.add("xformers")
            if keywords.get("requires_watermark"):
                libraries.add("invisible-watermark")
        
        if keywords.get("ai_type") == "vision" and not keywords.get("is_diffusion"):
            libraries.update(["opencv-python", "pillow", "timm", "albumentations", "imageio"])
        
        if keywords.get("ai_type") == "audio":
            libraries.update(["librosa", "soundfile", "audioread"])
        
        if keywords.get("ai_type") == "video":
            libraries.update(["opencv-python", "moviepy", "ffmpeg-python"])
        
        if keywords.get("ai_type") in ["nlp", "chat", "qa"] or keywords.get("is_transformer"):
            libraries.update(["nltk", "spacy", "rouge-score", "sacrebleu"])
        
        if keywords.get("requires_training") or keywords.get("is_deep_learning"):
            libraries.update(["scikit-learn", "matplotlib", "seaborn", "h5py"])
            libraries.add("tensorboard")
            if keywords.get("requires_wandb"):
                libraries.add("wandb")
        
        if keywords.get("requires_gradio"):
            libraries.update(["gradio", "plotly"])
        
        if keywords.get("is_deep_learning") or keywords.get("requires_training"):
            libraries.update(["pyyaml", "omegaconf"])
            libraries.add("torchmetrics")
            libraries.add("optuna")
            if keywords.get("requires_distributed"):
                libraries.add("ray[tune]")
        
        if keywords.get("requires_monitoring"):
            libraries.add("psutil")
        
        libraries.update(["pytest", "pytest-asyncio", "pytest-cov"])
        
        return libraries
    
    def format_requirements(self, libraries: Set[str]) -> str:
        """
        Formatea las librerías como un requirements.txt.
        
        Args:
            libraries: Set de nombres de librerías (pueden incluir extras como "uvicorn[standard]")
            
        Returns:
            String formateado como requirements.txt
        """
        categorized = {}
        misc_libs = []
        
        for lib_name in sorted(libraries):
            base_name = lib_name.split("[")[0].split(">=")[0].split("==")[0].split("~=")[0]
            
            if base_name in self._libraries:
                lib = self._libraries[base_name]
                cat = lib.category.value
                if cat not in categorized:
                    categorized[cat] = []
                
                version = lib.version if lib.version else ">=0.0.0"
                if "[" in lib_name and "]" in lib_name:
                    categorized[cat].append(lib_name)
                elif lib.extras:
                    extras_str = "[" + ",".join(lib.extras) + "]"
                    categorized[cat].append(f"{lib.name}{extras_str}{version}")
                else:
                    categorized[cat].append(f"{lib.name}{version}")
            else:
                misc_libs.append(lib_name)
        
        category_order = [
            "core", "http", "ai_ml", "deep_learning", "transformers",
            "diffusion", "vision", "audio", "video", "nlp",
            "utilities", "interface", "monitoring", "config",
            "optimization", "testing"
        ]
        
        result_lines = []
        
        for cat in category_order:
            if cat in categorized and categorized[cat]:
                result_lines.append(f"\n# {cat.replace('_', ' ').title()}")
                for lib_line in sorted(categorized[cat]):
                    result_lines.append(lib_line)
        
        if misc_libs:
            result_lines.append("\n# Misc")
            result_lines.extend(sorted(misc_libs))
        
        return "\n".join(result_lines).strip() + "\n"

