"""
Ejemplo de Uso de Workflow JSON
================================

Muestra cómo usar el workflow JSON con safe tensors.
"""

import json
import base64
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file


def load_workflow_json(workflow_path: str) -> dict:
    """Carga un workflow JSON."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding_from_workflow(workflow: dict, device: str = "cuda") -> torch.Tensor:
    """
    Carga el embedding desde el workflow.
    
    Soporta dos formatos:
    1. Embedding incluido en el JSON (base64)
    2. Referencia externa a archivo .safetensors
    """
    embedding_config = workflow.get("embedding", {})
    
    # Opción 1: Embedding incluido en JSON (base64)
    if "data" in embedding_config:
        print("[*] Cargando embedding desde JSON (base64)...")
        array_bytes = base64.b64decode(embedding_config["data"])
        dtype = np.dtype(embedding_config["dtype"])
        shape = tuple(embedding_config["shape"])
        
        embedding = torch.from_numpy(
            np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
        )
        
        if device == "cuda" and torch.cuda.is_available():
            embedding = embedding.to("cuda")
        
        return embedding
    
    # Opción 2: Referencia externa a .safetensors
    elif "path" in embedding_config:
        print(f"[*] Cargando embedding desde: {embedding_config['path']}...")
        data = load_file(embedding_config["path"])
        embedding = data[embedding_config.get("key", "character_embedding")]
        
        if device == "cuda" and torch.cuda.is_available():
            embedding = embedding.to("cuda")
        
        return embedding
    
    else:
        raise ValueError("No se encontró configuración de embedding en el workflow")


def format_prompt(template: str, variables: dict) -> str:
    """Formatea un prompt con variables."""
    return template.format(**variables)


def example_workflow_usage():
    """Ejemplo completo de uso del workflow."""
    print("=" * 70)
    print("EJEMPLO: USO DE WORKFLOW JSON")
    print("=" * 70)
    
    # Cargar workflow
    workflow_path = "workflow_example.json"
    
    if not Path(workflow_path).exists():
        print(f"\n[!] Workflow no encontrado: {workflow_path}")
        print("[*] Crea uno con: python export_to_workflow_json.py export <safe_tensor>")
        return
    
    print(f"\n[*] Cargando workflow: {workflow_path}")
    workflow = load_workflow_json(workflow_path)
    
    # Mostrar información del workflow
    print(f"\n[OK] Workflow cargado:")
    print(f"   Tipo: {workflow['workflow_type']}")
    print(f"   Versión: {workflow['version']}")
    print(f"   Personaje: {workflow['character']['name']}")
    print(f"   Dimensión: {workflow['character']['embedding_dim']}")
    
    # Paso 1: Cargar embedding
    print(f"\n{'='*70}")
    print("PASO 1: Cargar Embedding del Personaje")
    print("=" * 70)
    
    try:
        character_embedding = load_embedding_from_workflow(workflow, device="cuda")
        print(f"[OK] Embedding cargado: shape={character_embedding.shape}, device={character_embedding.device}")
    except Exception as e:
        print(f"[!] Error cargando embedding: {e}")
        return
    
    # Paso 2: Preparar prompt
    print(f"\n{'='*70}")
    print("PASO 2: Preparar Prompt")
    print("=" * 70)
    
    workflow_config = workflow["workflow_config"]
    prompt_template = workflow_config["prompt_template"]
    
    # Variables del prompt
    variables = {
        "character": workflow["character"]["name"],
        "setting": "a beautiful forest"  # Esto vendría del usuario
    }
    
    formatted_prompt = format_prompt(prompt_template, variables)
    print(f"[OK] Prompt formateado: {formatted_prompt}")
    
    # Paso 3: Generar imagen (ejemplo conceptual)
    print(f"\n{'='*70}")
    print("PASO 3: Generar Imagen")
    print("=" * 70)
    
    print("[*] Configuración de generación:")
    print(f"   Modelo: {workflow['model']['model_id']}")
    print(f"   Pasos: {workflow_config['num_inference_steps']}")
    print(f"   Guidance: {workflow_config['guidance_scale']}")
    print(f"   Tamaño: {workflow_config['width']}x{workflow_config['height']}")
    print(f"   Negative prompt: {workflow_config['negative_prompt']}")
    
    print("\n[*] Código de ejemplo para generación:")
    print("""
    from diffusers import FluxPipeline
    
    # Cargar pipeline
    pipeline = FluxPipeline.from_pretrained(
        workflow['model']['model_id'],
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Generar imagen con el embedding del personaje
    # (La implementación específica depende de cómo Flux2 acepta embeddings)
    image = pipeline(
        prompt=formatted_prompt,
        negative_prompt=workflow_config['negative_prompt'],
        character_embedding=character_embedding,
        num_inference_steps=workflow_config['num_inference_steps'],
        guidance_scale=workflow_config['guidance_scale'],
        height=workflow_config['height'],
        width=workflow_config['width'],
    ).images[0]
    
    image.save("generated_character.png")
    """)
    
    print("\n" + "=" * 70)
    print("[OK] Workflow completado")
    print("=" * 70)


def create_workflow_from_safe_tensor(safe_tensor_path: str, output_json: str = None):
    """Crea un workflow JSON desde un safe tensor."""
    from export_to_workflow_json import export_safe_tensor_to_json
    
    if output_json is None:
        output_json = Path(safe_tensor_path).with_suffix('.workflow.json')
    
    print(f"[*] Exportando safe tensor a workflow JSON...")
    workflow = export_safe_tensor_to_json(
        safe_tensor_path=safe_tensor_path,
        output_json_path=str(output_json),
        include_embedding=False,  # Referencia externa (más eficiente)
    )
    
    print(f"[OK] Workflow JSON creado: {output_json}")
    return workflow


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Si se pasa un safe tensor, crear workflow
        safe_tensor = sys.argv[1]
        create_workflow_from_safe_tensor(safe_tensor)
    else:
        # Ejemplo de uso
        example_workflow_usage()


