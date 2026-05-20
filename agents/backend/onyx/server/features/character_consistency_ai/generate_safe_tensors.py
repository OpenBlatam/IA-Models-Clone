"""
Script para Generar Safe Tensors de Consistencia de Personaje
==============================================================

Este script demuestra cómo generar safe tensors desde imágenes.
"""

import sys
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.character_consistency_service import CharacterConsistencyService
from config.character_consistency_config import CharacterConsistencyConfig


def generate_safe_tensor_from_images(
    image_paths: list,
    character_name: str = "Character",
    output_dir: str = "./character_embeddings"
):
    """
    Genera un safe tensor desde una o múltiples imágenes.
    
    Args:
        image_paths: Lista de rutas a imágenes
        character_name: Nombre del personaje
        output_dir: Directorio de salida
    """
    print("=" * 70)
    print("Generador de Safe Tensors - Character Consistency AI")
    print("=" * 70)
    print(f"\n📸 Imágenes a procesar: {len(image_paths)}")
    for i, img_path in enumerate(image_paths, 1):
        print(f"   {i}. {img_path}")
    
    # Verificar que las imágenes existan
    missing_images = []
    for img_path in image_paths:
        if not Path(img_path).exists():
            missing_images.append(img_path)
    
    if missing_images:
        print(f"\n❌ Error: Las siguientes imágenes no existen:")
        for img in missing_images:
            print(f"   - {img}")
        print("\n💡 Tip: Asegúrate de que las rutas de las imágenes sean correctas.")
        return None
    
    # Crear configuración
    config = CharacterConsistencyConfig(
        output_dir=output_dir,
        enable_optimizations=True,
    )
    
    # Crear servicio
    print("\n🔧 Inicializando servicio...")
    service = CharacterConsistencyService(config=config)
    
    try:
        # Inicializar modelo
        print("🤖 Inicializando modelo Flux2...")
        print("   (Esto puede tomar unos minutos la primera vez)")
        service.initialize_model()
        
        # Generar embedding
        print(f"\n✨ Generando embedding de consistencia para '{character_name}'...")
        result = service.generate_character_embedding(
            images=image_paths,
            character_name=character_name,
            metadata={
                "source": "script_generation",
                "num_source_images": len(image_paths),
            },
            save_tensor=True,
        )
        
        # Mostrar resultados
        print("\n" + "=" * 70)
        print("✅ Safe Tensor Generado Exitosamente")
        print("=" * 70)
        print(f"\n📊 Información del Embedding:")
        print(f"   - Dimensión: {result['embedding_dim']}")
        print(f"   - Imágenes procesadas: {result['num_images']}")
        print(f"   - Personaje: {result.get('character_name', 'N/A')}")
        
        if result.get('saved'):
            print(f"\n💾 Archivo guardado:")
            print(f"   {result['saved_path']}")
            
            # Mostrar metadata si existe
            metadata_path = Path(result['saved_path']).with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"\n📋 Metadata:")
                for key, value in metadata.items():
                    print(f"   - {key}: {value}")
        
        print("\n" + "=" * 70)
        print("🎉 ¡Proceso completado!")
        print("=" * 70)
        print(f"\n💡 Próximos pasos:")
        print(f"   1. Usa el safe tensor en tu workflow de generación")
        print(f"   2. Carga el tensor con: load_file('{result.get('saved_path', 'path')}')")
        print(f"   3. Consulta WORKFLOW_USAGE.md para más detalles")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error durante la generación: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return None
    
    finally:
        service.close()


def create_workflow_tensor_from_embedding(
    embedding_path: str,
    prompt_template: str = "A photo of {character} in a {setting}",
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    """
    Crea un workflow tensor desde un embedding existente.
    
    Args:
        embedding_path: Ruta al safe tensor del embedding
        prompt_template: Template del prompt
        negative_prompt: Prompt negativo
        num_inference_steps: Pasos de inferencia
        guidance_scale: Escala de guía
    """
    print("=" * 70)
    print("Creando Workflow Tensor")
    print("=" * 70)
    
    if not Path(embedding_path).exists():
        print(f"❌ Error: El embedding no existe: {embedding_path}")
        return None
    
    config = CharacterConsistencyConfig()
    service = CharacterConsistencyService(config=config)
    
    try:
        service.initialize_model()
        
        print(f"\n🔧 Creando workflow tensor desde: {embedding_path}")
        result = service.create_workflow_tensor(
            embedding_path=embedding_path,
            prompt_template=prompt_template,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        print("\n✅ Workflow Tensor Creado:")
        print(f"   {result['workflow_tensor_path']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return None
    
    finally:
        service.close()


def list_existing_tensors(output_dir: str = "./character_embeddings"):
    """
    Lista todos los safe tensors existentes.
    
    Args:
        output_dir: Directorio donde buscar
    """
    print("=" * 70)
    print("Safe Tensors Existentes")
    print("=" * 70)
    
    config = CharacterConsistencyConfig(output_dir=output_dir)
    service = CharacterConsistencyService(config=config)
    
    try:
        embeddings = service.list_embeddings()
        
        if not embeddings:
            print(f"\n📭 No se encontraron safe tensors en: {output_dir}")
            print("\n💡 Tip: Usa generate_safe_tensor_from_images() para crear uno.")
            return
        
        print(f"\n📦 Encontrados {len(embeddings)} safe tensor(s):\n")
        
        for i, emb in enumerate(embeddings, 1):
            print(f"{i}. {emb['filename']}")
            print(f"   📁 Ruta: {emb['path']}")
            print(f"   📏 Tamaño: {emb['size'] / 1024:.2f} KB")
            
            if emb.get('metadata'):
                meta = emb['metadata']
                print(f"   👤 Personaje: {meta.get('character_name', 'N/A')}")
                print(f"   🖼️  Imágenes: {meta.get('num_images', 'N/A')}")
                print(f"   📅 Creado: {meta.get('created_at', 'N/A')}")
            
            print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error: {e}", exc_info=True)
    
    finally:
        service.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generador de Safe Tensors para Character Consistency AI"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: generate
    gen_parser = subparsers.add_parser('generate', help='Generar safe tensor desde imágenes')
    gen_parser.add_argument('images', nargs='+', help='Rutas a las imágenes')
    gen_parser.add_argument('--name', default='Character', help='Nombre del personaje')
    gen_parser.add_argument('--output-dir', default='./character_embeddings', help='Directorio de salida')
    
    # Comando: workflow
    workflow_parser = subparsers.add_parser('workflow', help='Crear workflow tensor')
    workflow_parser.add_argument('embedding', help='Ruta al embedding safe tensor')
    workflow_parser.add_argument('--prompt', default='A photo of {character} in a {setting}', help='Template del prompt')
    workflow_parser.add_argument('--negative', default='blurry, low quality', help='Prompt negativo')
    workflow_parser.add_argument('--steps', type=int, default=50, help='Pasos de inferencia')
    workflow_parser.add_argument('--guidance', type=float, default=7.5, help='Escala de guía')
    
    # Comando: list
    list_parser = subparsers.add_parser('list', help='Listar safe tensors existentes')
    list_parser.add_argument('--output-dir', default='./character_embeddings', help='Directorio a buscar')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_safe_tensor_from_images(
            image_paths=args.images,
            character_name=args.name,
            output_dir=args.output_dir
        )
    
    elif args.command == 'workflow':
        create_workflow_tensor_from_embedding(
            embedding_path=args.embedding,
            prompt_template=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )
    
    elif args.command == 'list':
        list_existing_tensors(output_dir=args.output_dir)
    
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Ejemplos de Uso:")
        print("=" * 70)
        print("\n1. Generar safe tensor desde imágenes:")
        print("   python generate_safe_tensors.py generate image1.jpg image2.jpg --name MyCharacter")
        print("\n2. Crear workflow tensor:")
        print("   python generate_safe_tensors.py workflow character_embeddings/my_character.safetensors")
        print("\n3. Listar safe tensors existentes:")
        print("   python generate_safe_tensors.py list")


