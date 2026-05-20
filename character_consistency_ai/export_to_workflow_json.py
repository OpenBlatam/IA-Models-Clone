"""
Exportar Safe Tensor a JSON para Workflows
===========================================

Convierte safe tensors a formato JSON listo para usar en workflows.
"""

import json
import base64
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from safetensors.torch import load_file
import torch
import sys

# Fix encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def tensor_to_json_serializable(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Convierte un tensor a formato JSON serializable.
    
    Args:
        tensor: Tensor de PyTorch
        
    Returns:
        Dict con datos del tensor en formato JSON
    """
    # Convertir a numpy
    np_array = tensor.cpu().numpy()
    
    # Opción 1: Base64 (más compacto)
    array_bytes = np_array.tobytes()
    base64_str = base64.b64encode(array_bytes).decode('utf-8')
    
    return {
        "data": base64_str,
        "shape": list(np_array.shape),
        "dtype": str(np_array.dtype),
        "format": "base64"
    }


def export_safe_tensor_to_json(
    safe_tensor_path: str,
    output_json_path: Optional[str] = None,
    include_embedding: bool = True,
    workflow_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Exporta un safe tensor a formato JSON para workflows.
    
    Args:
        safe_tensor_path: Ruta al archivo .safetensors
        output_json_path: Ruta donde guardar el JSON (opcional)
        include_embedding: Si incluir el embedding en el JSON
        workflow_config: Configuración adicional del workflow
        
    Returns:
        Dict con el workflow JSON
    """
    safe_tensor_path = Path(safe_tensor_path)
    
    if not safe_tensor_path.exists():
        raise FileNotFoundError(f"Safe tensor no encontrado: {safe_tensor_path}")
    
    print(f"[*] Cargando safe tensor: {safe_tensor_path}")
    
    # Cargar safe tensor
    data = load_file(str(safe_tensor_path))
    embedding = data["character_embedding"]
    
    # Cargar metadata si existe
    metadata_path = safe_tensor_path.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Crear estructura del workflow
    workflow_json = {
        "workflow_type": "character_consistency",
        "version": "1.0.0",
        "character": {
            "name": metadata.get("character_name", "unknown"),
            "num_images": metadata.get("num_images", 0),
            "embedding_dim": int(embedding.shape[0]),
            "created_at": metadata.get("created_at", ""),
        },
        "model": {
            "model_id": metadata.get("model_id", "black-forest-labs/flux2-dev"),
            "device": metadata.get("device", "cuda"),
        }
    }
    
    # Incluir embedding si se solicita
    if include_embedding:
        print(f"[*] Serializando embedding (shape: {embedding.shape})...")
        workflow_json["embedding"] = tensor_to_json_serializable(embedding)
    
    # Agregar configuración del workflow
    if workflow_config:
        workflow_json["workflow_config"] = workflow_config
    else:
        # Configuración por defecto
        workflow_json["workflow_config"] = {
            "prompt_template": "A photo of {character} in a {setting}",
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
        }
    
    # Agregar metadata adicional
    workflow_json["metadata"] = {
        "source_file": str(safe_tensor_path),
        "export_format": "json",
        "embedding_included": include_embedding,
    }
    
    # Guardar si se especifica ruta de salida
    if output_json_path:
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(workflow_json, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Workflow JSON guardado en: {output_path}")
    
    return workflow_json


def create_workflow_template_json(
    character_name: str,
    embedding_dim: int = 768,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Crea un template de workflow JSON sin embedding (para referenciar externamente).
    
    Args:
        character_name: Nombre del personaje
        embedding_dim: Dimensión del embedding
        output_path: Ruta donde guardar el JSON
        
    Returns:
        Dict con el template del workflow
    """
    workflow_template = {
        "workflow_type": "character_consistency",
        "version": "1.0.0",
        "character": {
            "name": character_name,
            "embedding_dim": embedding_dim,
        },
        "embedding": {
            "source": "external",
            "path": "character_embeddings/{character_name}.safetensors",
            "format": "safetensors"
        },
        "workflow_config": {
            "prompt_template": "A photo of {character} in a {setting}, high quality, detailed",
            "negative_prompt": "blurry, low quality, distorted, deformed",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "seed": None,
        },
        "steps": [
            {
                "step": 1,
                "name": "load_embedding",
                "action": "load_safetensor",
                "params": {
                    "path": "character_embeddings/{character_name}.safetensors",
                    "key": "character_embedding"
                }
            },
            {
                "step": 2,
                "name": "prepare_prompt",
                "action": "format_prompt",
                "params": {
                    "template": "A photo of {character} in a {setting}",
                    "variables": {
                        "character": character_name,
                        "setting": "{user_input}"
                    }
                }
            },
            {
                "step": 3,
                "name": "generate_image",
                "action": "diffusion_generate",
                "params": {
                    "model": "black-forest-labs/flux2-dev",
                    "prompt": "{formatted_prompt}",
                    "negative_prompt": "blurry, low quality",
                    "character_embedding": "{loaded_embedding}",
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                }
            }
        ]
    }
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(workflow_template, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Template de workflow guardado en: {output_path}")
    
    return workflow_template


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Exportar Safe Tensor a JSON para Workflows"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando')
    
    # Comando: export
    export_parser = subparsers.add_parser('export', help='Exportar safe tensor a JSON')
    export_parser.add_argument('safe_tensor', help='Ruta al archivo .safetensors')
    export_parser.add_argument('--output', '-o', help='Ruta de salida del JSON')
    export_parser.add_argument('--no-embedding', action='store_true', help='No incluir embedding en JSON')
    export_parser.add_argument('--prompt', default='A photo of {character} in a {setting}', help='Template del prompt')
    export_parser.add_argument('--negative', default='blurry, low quality', help='Prompt negativo')
    export_parser.add_argument('--steps', type=int, default=50, help='Pasos de inferencia')
    export_parser.add_argument('--guidance', type=float, default=7.5, help='Escala de guía')
    
    # Comando: template
    template_parser = subparsers.add_parser('template', help='Crear template de workflow')
    template_parser.add_argument('--name', required=True, help='Nombre del personaje')
    template_parser.add_argument('--dim', type=int, default=768, help='Dimensión del embedding')
    template_parser.add_argument('--output', '-o', help='Ruta de salida del JSON')
    
    args = parser.parse_args()
    
    if args.command == 'export':
        workflow_config = {
            "prompt_template": args.prompt,
            "negative_prompt": args.negative,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
        }
        
        output_path = args.output or Path(args.safe_tensor).with_suffix('.workflow.json')
        
        workflow_json = export_safe_tensor_to_json(
            safe_tensor_path=args.safe_tensor,
            output_json_path=str(output_path),
            include_embedding=not args.no_embedding,
            workflow_config=workflow_config
        )
        
        # Mostrar resumen
        print("\n" + "=" * 70)
        print("WORKFLOW JSON EXPORTADO")
        print("=" * 70)
        print(f"\nPersonaje: {workflow_json['character']['name']}")
        print(f"Dimensión: {workflow_json['character']['embedding_dim']}")
        print(f"Embedding incluido: {workflow_json['metadata']['embedding_included']}")
        print(f"\nArchivo: {output_path}")
        print("\n[*] Puedes usar este JSON en tu workflow de generación de imágenes.")
        
        # Mostrar ejemplo de uso
        print("\n" + "=" * 70)
        print("EJEMPLO DE USO EN PYTHON:")
        print("=" * 70)
        print(f"""
import json
import base64
import numpy as np
import torch

# Cargar workflow JSON
with open('{output_path}', 'r') as f:
    workflow = json.load(f)

# Decodificar embedding si está incluido
if workflow['metadata']['embedding_included']:
    embedding_data = workflow['embedding']
    array_bytes = base64.b64decode(embedding_data['data'])
    embedding = torch.from_numpy(
        np.frombuffer(array_bytes, dtype=np.dtype(embedding_data['dtype']))
        .reshape(embedding_data['shape'])
    )
    
    print(f"Embedding shape: {{embedding.shape}}")
    
    # Usar en tu pipeline de generación
    # pipeline.generate(prompt, character_embedding=embedding)
""")
    
    elif args.command == 'template':
        template = create_workflow_template_json(
            character_name=args.name,
            embedding_dim=args.dim,
            output_path=args.output or f"workflow_{args.name}.json"
        )
        
        print("\n" + "=" * 70)
        print("TEMPLATE DE WORKFLOW CREADO")
        print("=" * 70)
        print(f"\nPersonaje: {template['character']['name']}")
        print(f"Pasos del workflow: {len(template['steps'])}")
        print(f"\nArchivo: {args.output or f'workflow_{args.name}.json'}")
    
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("EJEMPLOS:")
        print("=" * 70)
        print("\n1. Exportar safe tensor a JSON:")
        print("   python export_to_workflow_json.py export character.safetensors")
        print("\n2. Exportar sin embedding (referencia externa):")
        print("   python export_to_workflow_json.py export character.safetensors --no-embedding")
        print("\n3. Crear template de workflow:")
        print("   python export_to_workflow_json.py template --name MyCharacter")


if __name__ == "__main__":
    main()


