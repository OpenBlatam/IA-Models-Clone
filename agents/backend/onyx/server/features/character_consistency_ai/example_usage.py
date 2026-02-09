"""
Example Usage - Character Consistency AI
=========================================

Ejemplos de uso del modelo de consistencia de personaje.
"""

import asyncio
from pathlib import Path
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService
from character_consistency_ai.config.character_consistency_config import CharacterConsistencyConfig


def example_single_image():
    """Ejemplo: Generar embedding desde una sola imagen."""
    print("=" * 60)
    print("Ejemplo 1: Generar embedding desde una imagen")
    print("=" * 60)
    
    # Crear configuración
    config = CharacterConsistencyConfig(
        output_dir="./example_embeddings",
        enable_optimizations=True,
    )
    
    # Crear servicio
    service = CharacterConsistencyService(config=config)
    
    # Inicializar modelo
    print("Inicializando modelo...")
    service.initialize_model()
    
    # Generar embedding
    print("Generando embedding...")
    result = service.generate_character_embedding(
        images=["path/to/character_image.jpg"],  # Reemplazar con ruta real
        character_name="ExampleCharacter",
        save_tensor=True,
    )
    
    print(f"✓ Embedding generado: {result['embedding_dim']} dimensiones")
    print(f"✓ Guardado en: {result.get('saved_path', 'N/A')}")
    
    service.close()


def example_multiple_images():
    """Ejemplo: Generar embedding desde múltiples imágenes."""
    print("\n" + "=" * 60)
    print("Ejemplo 2: Generar embedding desde múltiples imágenes")
    print("=" * 60)
    
    config = CharacterConsistencyConfig(output_dir="./example_embeddings")
    service = CharacterConsistencyService(config=config)
    service.initialize_model()
    
    # Múltiples imágenes del mismo personaje
    images = [
        "path/to/character_front.jpg",
        "path/to/character_side.jpg",
        "path/to/character_back.jpg",
    ]
    
    print("Generando embedding desde múltiples imágenes...")
    result = service.generate_character_embedding(
        images=images,
        character_name="MultiViewCharacter",
        metadata={
            "source": "game_assets",
            "views": ["front", "side", "back"],
        },
        save_tensor=True,
    )
    
    print(f"✓ Embedding generado desde {result['num_images']} imágenes")
    print(f"✓ Guardado en: {result.get('saved_path', 'N/A')}")
    
    service.close()


def example_workflow_tensor():
    """Ejemplo: Crear workflow tensor."""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Crear workflow tensor")
    print("=" * 60)
    
    config = CharacterConsistencyConfig(output_dir="./example_embeddings")
    service = CharacterConsistencyService(config=config)
    service.initialize_model()
    
    # Primero generar embedding
    result = service.generate_character_embedding(
        images=["path/to/character.jpg"],
        character_name="WorkflowCharacter",
        save_tensor=True,
    )
    
    embedding_path = result["saved_path"]
    
    # Crear workflow tensor
    print("Creando workflow tensor...")
    workflow_result = service.create_workflow_tensor(
        embedding_path=embedding_path,
        prompt_template="A photo of {character} in a {setting}, high quality, detailed",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=50,
        guidance_scale=7.5,
    )
    
    print(f"✓ Workflow tensor creado: {workflow_result['workflow_tensor_path']}")
    
    service.close()


def example_batch_processing():
    """Ejemplo: Procesamiento en batch."""
    print("\n" + "=" * 60)
    print("Ejemplo 4: Procesamiento en batch")
    print("=" * 60)
    
    config = CharacterConsistencyConfig(output_dir="./example_embeddings")
    service = CharacterConsistencyService(config=config)
    service.initialize_model()
    
    # Múltiples grupos de imágenes
    image_groups = [
        ["path/to/hero1_front.jpg", "path/to/hero1_side.jpg"],
        ["path/to/hero2_front.jpg", "path/to/hero2_side.jpg"],
        ["path/to/villain_front.jpg"],
    ]
    
    character_names = ["Hero1", "Hero2", "Villain"]
    
    print("Procesando múltiples personajes...")
    results = service.batch_generate(
        image_groups=image_groups,
        character_names=character_names,
    )
    
    print(f"✓ Procesados {len(results)} personajes")
    for i, result in enumerate(results):
        print(f"  - {character_names[i]}: {result.get('saved_path', 'N/A')}")
    
    service.close()


def example_list_embeddings():
    """Ejemplo: Listar embeddings generados."""
    print("\n" + "=" * 60)
    print("Ejemplo 5: Listar embeddings")
    print("=" * 60)
    
    config = CharacterConsistencyConfig(output_dir="./example_embeddings")
    service = CharacterConsistencyService(config=config)
    
    embeddings = service.list_embeddings()
    
    print(f"✓ Encontrados {len(embeddings)} embeddings:")
    for emb in embeddings:
        print(f"  - {emb['filename']}")
        if emb.get('metadata'):
            char_name = emb['metadata'].get('character_name', 'Unknown')
            print(f"    Personaje: {char_name}")
            print(f"    Imágenes: {emb['metadata'].get('num_images', 'N/A')}")
    
    service.close()


def example_model_info():
    """Ejemplo: Obtener información del modelo."""
    print("\n" + "=" * 60)
    print("Ejemplo 6: Información del modelo")
    print("=" * 60)
    
    config = CharacterConsistencyConfig()
    service = CharacterConsistencyService(config=config)
    service.initialize_model()
    
    info = service.get_model_info()
    
    print("Información del modelo:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    service.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Character Consistency AI - Ejemplos de Uso")
    print("=" * 60)
    print("\nNota: Reemplaza las rutas de imágenes con rutas reales antes de ejecutar.")
    print("\nDescomenta el ejemplo que quieras ejecutar:\n")
    
    # Descomentar el ejemplo que quieras ejecutar:
    
    # example_single_image()
    # example_multiple_images()
    # example_workflow_tensor()
    # example_batch_processing()
    # example_list_embeddings()
    # example_model_info()
    
    print("\n" + "=" * 60)
    print("Ejemplos listos. Descomenta el que quieras ejecutar.")
    print("=" * 60)


