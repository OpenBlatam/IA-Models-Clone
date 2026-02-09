"""
Ejemplo básico de uso de Audio Separation Core.

Este ejemplo muestra cómo:
1. Extraer audio de un video
2. Separar el audio en componentes
3. Mezclar los componentes con volúmenes personalizados
"""

from pathlib import Path
from audio_separation_core import (
    process_video_audio,
    separate_audio,
    mix_audio,
    create_audio_separator,
    create_audio_mixer,
    SeparationConfig,
    MixingConfig
)


def example_1_simple_separation():
    """Ejemplo 1: Separación simple de audio."""
    print("=== Ejemplo 1: Separación Simple ===")
    
    # Separar audio de un archivo
    results = separate_audio(
        "input/audio.wav",
        output_dir="output",
        separator_type="auto",  # Auto-detecta el mejor separador disponible
        components=["vocals", "accompaniment"]
    )
    
    print(f"Voces separadas: {results['vocals']}")
    print(f"Acompañamiento separado: {results['accompaniment']}")


def example_2_video_processing():
    """Ejemplo 2: Procesar video completo."""
    print("\n=== Ejemplo 2: Procesamiento de Video ===")
    
    # Procesar video: extrae audio y lo separa
    result = process_video_audio(
        "input/video.mp4",
        output_dir="output",
        separate=True,
        components=["vocals", "accompaniment"]
    )
    
    print(f"Audio extraído: {result['audio_path']}")
    print(f"Voces: {result['separated']['vocals']}")
    print(f"Acompañamiento: {result['separated']['accompaniment']}")
    print(f"Duración: {result['metadata'].get('duration', 0):.2f} segundos")


def example_3_mixing():
    """Ejemplo 3: Mezclar componentes de audio."""
    print("\n=== Ejemplo 3: Mezcla de Audio ===")
    
    # Mezclar componentes con volúmenes personalizados
    mixed = mix_audio(
        {
            "vocals": "output/vocals.wav",
            "music": "output/accompaniment.wav"
        },
        "output/mixed.wav",
        mixer_type="simple",
        volumes={
            "vocals": 0.9,  # Voces más altas
            "music": 0.6    # Música más baja
        }
    )
    
    print(f"Audio mezclado guardado en: {mixed}")


def example_4_advanced_usage():
    """Ejemplo 4: Uso avanzado con configuración personalizada."""
    print("\n=== Ejemplo 4: Uso Avanzado ===")
    
    # Crear separador con configuración personalizada
    separation_config = SeparationConfig(
        model_type="demucs",
        use_gpu=True,
        components=["vocals", "drums", "bass", "other"],
        overlap=0.25,
        post_process=True
    )
    
    separator = create_audio_separator("demucs", config=separation_config)
    
    # Separar
    separated = separator.separate(
        "input/audio.wav",
        output_dir="output/separated"
    )
    
    print("Componentes separados:")
    for component, path in separated.items():
        print(f"  {component}: {path}")
    
    # Crear mezclador avanzado con efectos
    mixing_config = MixingConfig(
        mixer_type="advanced",
        default_volume=0.8,
        normalize_output=True,
        fade_in=0.5,
        fade_out=2.0,
        apply_reverb=True,
        apply_eq=True,
        apply_compressor=True,
        reverb_params={
            "delay_ms": 50,
            "feedback": 0.3,
            "mix": 0.3
        }
    )
    
    mixer = create_audio_mixer("advanced", config=mixing_config)
    
    # Mezclar con volúmenes personalizados
    final_mix = mixer.mix(
        separated,
        "output/final_mix.wav",
        volumes={
            "vocals": 0.9,
            "drums": 0.7,
            "bass": 0.8,
            "other": 0.6
        }
    )
    
    print(f"\nMezcla final guardada en: {final_mix}")


def example_5_workflow_completo():
    """Ejemplo 5: Workflow completo de video a mezcla final."""
    print("\n=== Ejemplo 5: Workflow Completo ===")
    
    # Paso 1: Extraer y separar audio del video
    print("Paso 1: Extrayendo y separando audio...")
    result = process_video_audio(
        "input/video.mp4",
        output_dir="output",
        separate=True,
        components=["vocals", "accompaniment"]
    )
    
    # Paso 2: Mezclar con volúmenes personalizados
    print("Paso 2: Mezclando componentes...")
    mixed = mix_audio(
        result["separated"],
        "output/final_mix.wav",
        mixer_type="advanced",
        volumes={
            "vocals": 0.85,
            "accompaniment": 0.65
        }
    )
    
    print(f"\n✅ Proceso completo!")
    print(f"   Audio original: {result['audio_path']}")
    print(f"   Voces: {result['separated']['vocals']}")
    print(f"   Acompañamiento: {result['separated']['accompaniment']}")
    print(f"   Mezcla final: {mixed}")


if __name__ == "__main__":
    # Ejecutar ejemplos
    print("Audio Separation Core - Ejemplos de Uso\n")
    
    # Descomentar el ejemplo que quieras ejecutar:
    
    # example_1_simple_separation()
    # example_2_video_processing()
    # example_3_mixing()
    # example_4_advanced_usage()
    # example_5_workflow_completo()
    
    print("\n💡 Nota: Descomenta el ejemplo que quieras ejecutar en el código.")




