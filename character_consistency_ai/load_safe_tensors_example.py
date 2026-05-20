"""
Ejemplo: Cómo Cargar y Usar Safe Tensors
==========================================

Este script muestra cómo cargar y usar los safe tensors generados.
"""

from pathlib import Path
from safetensors.torch import load_file
import json
import torch


def load_character_embedding(embedding_path: str, device: str = "cpu"):
    """
    Carga un safe tensor de embedding de personaje.
    
    Args:
        embedding_path: Ruta al archivo .safetensors
        device: Dispositivo donde cargar (cpu/cuda)
    
    Returns:
        Tuple de (embedding tensor, metadata dict)
    """
    embedding_path = Path(embedding_path)
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding no encontrado: {embedding_path}")
    
    print(f"📂 Cargando embedding desde: {embedding_path}")
    
    # Cargar safe tensor
    data = load_file(str(embedding_path))
    embedding = data["character_embedding"]
    
    # Mover a dispositivo
    if device == "cuda" and torch.cuda.is_available():
        embedding = embedding.to("cuda")
        print(f"   ✅ Movido a CUDA")
    else:
        embedding = embedding.to("cpu")
        print(f"   ✅ Movido a CPU")
    
    # Cargar metadata si existe
    metadata_path = embedding_path.with_suffix(".json")
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"   ✅ Metadata cargada")
    
    print(f"\n📊 Información del Embedding:")
    print(f"   - Shape: {embedding.shape}")
    print(f"   - Dtype: {embedding.dtype}")
    print(f"   - Device: {embedding.device}")
    
    if metadata:
        print(f"\n📋 Metadata:")
        for key, value in metadata.items():
            print(f"   - {key}: {value}")
    
    return embedding, metadata


def load_workflow_tensor(workflow_path: str, device: str = "cpu"):
    """
    Carga un workflow tensor completo.
    
    Args:
        workflow_path: Ruta al workflow tensor
        device: Dispositivo donde cargar
    
    Returns:
        Tuple de (embedding, workflow_config, metadata)
    """
    workflow_path = Path(workflow_path)
    
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow tensor no encontrado: {workflow_path}")
    
    print(f"📂 Cargando workflow tensor desde: {workflow_path}")
    
    # Cargar safe tensor
    data = load_file(str(workflow_path))
    embedding = data["character_embedding"]
    workflow_config = data.get("workflow_config", {})
    
    # Mover a dispositivo
    if device == "cuda" and torch.cuda.is_available():
        embedding = embedding.to("cuda")
    else:
        embedding = embedding.to("cpu")
    
    # Cargar metadata
    metadata_path = workflow_path.with_suffix(".json")
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    print(f"\n📊 Workflow Config:")
    for key, value in workflow_config.items():
        print(f"   - {key}: {value}")
    
    return embedding, workflow_config, metadata


def example_usage():
    """Ejemplo completo de uso."""
    print("=" * 70)
    print("Ejemplo: Cargar y Usar Safe Tensors")
    print("=" * 70)
    
    # Ejemplo 1: Cargar embedding simple
    print("\n" + "-" * 70)
    print("Ejemplo 1: Cargar Embedding Simple")
    print("-" * 70)
    
    try:
        # Reemplazar con la ruta real de tu safe tensor
        embedding_path = "./character_embeddings/my_character.safetensors"
        
        if Path(embedding_path).exists():
            embedding, metadata = load_character_embedding(embedding_path)
            
            print(f"\n✅ Embedding cargado exitosamente!")
            print(f"   Puedes usar este embedding en tu pipeline de generación")
            
        else:
            print(f"\n⚠️  Embedding no encontrado: {embedding_path}")
            print(f"   💡 Genera uno primero con: python generate_safe_tensors.py generate <imagenes>")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Ejemplo 2: Cargar workflow tensor
    print("\n" + "-" * 70)
    print("Ejemplo 2: Cargar Workflow Tensor")
    print("-" * 70)
    
    try:
        workflow_path = "./character_embeddings/workflow_*.safetensors"
        
        # Buscar workflow tensors
        from glob import glob
        workflows = glob(str(Path(workflow_path).parent / "workflow_*.safetensors"))
        
        if workflows:
            embedding, config, metadata = load_workflow_tensor(workflows[0])
            
            print(f"\n✅ Workflow tensor cargado!")
            print(f"   Prompt template: {config.get('prompt_template', 'N/A')}")
            print(f"   Negative prompt: {config.get('negative_prompt', 'N/A')}")
            
        else:
            print(f"\n⚠️  No se encontraron workflow tensors")
            print(f"   💡 Crea uno con: python generate_safe_tensors.py workflow <embedding_path>")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Ejemplo 3: Usar embedding en generación
    print("\n" + "-" * 70)
    print("Ejemplo 3: Usar Embedding en Generación")
    print("-" * 70)
    
    print("""
    # Código de ejemplo para usar el embedding:
    
    from safetensors.torch import load_file
    import torch
    
    # Cargar embedding
    data = load_file("character_embeddings/my_character.safetensors")
    character_embedding = data["character_embedding"].to("cuda")
    
    # Usar en tu pipeline (ejemplo con diffusers)
    from diffusers import StableDiffusionPipeline
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to("cuda")
    
    # Inyectar embedding en el proceso de generación
    # (implementación específica depende de tu workflow)
    prompt = "A photo of my character in a forest"
    
    # Generar imagen con el embedding
    # image = pipeline(prompt, character_embedding=character_embedding).images[0]
    """)
    
    print("\n" + "=" * 70)
    print("✅ Ejemplos completados")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()


