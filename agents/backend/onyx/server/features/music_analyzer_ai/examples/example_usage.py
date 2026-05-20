"""
Ejemplo de uso del Music Analyzer AI
"""

import asyncio
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.spotify_service import SpotifyService
from core.music_analyzer import MusicAnalyzer
from services.music_coach import MusicCoach


async def example_basic_analysis():
    """Ejemplo básico de análisis"""
    print("=== Ejemplo de Análisis Musical ===\n")
    
    # Inicializar servicios
    spotify = SpotifyService()
    analyzer = MusicAnalyzer()
    coach = MusicCoach()
    
    try:
        # Buscar canción
        print("1. Buscando canción...")
        tracks = spotify.search_track("Bohemian Rhapsody Queen", limit=1)
        
        if not tracks:
            print("No se encontró la canción")
            return
        
        track = tracks[0]
        print(f"   Encontrada: {track['name']} - {', '.join([a['name'] for a in track['artists']])}")
        
        # Obtener análisis completo
        print("\n2. Obteniendo datos de Spotify...")
        spotify_data = spotify.get_track_full_analysis(track['id'])
        
        # Analizar
        print("3. Analizando música...")
        analysis = analyzer.analyze_track(spotify_data)
        
        # Mostrar resultados
        print("\n=== RESULTADOS DEL ANÁLISIS ===\n")
        
        print(f"Canción: {analysis['track_basic_info']['name']}")
        print(f"Artista: {', '.join(analysis['track_basic_info']['artists'])}")
        print(f"Duración: {analysis['track_basic_info']['duration_seconds']:.1f} segundos")
        
        print("\n--- Análisis Musical ---")
        musical = analysis['musical_analysis']
        print(f"Tonalidad: {musical['key_signature']}")
        print(f"Tempo: {musical['tempo']['bpm']:.1f} BPM ({musical['tempo']['category']})")
        print(f"Compás: {musical['time_signature']}")
        print(f"Escala: {musical['scale']['name']}")
        print(f"Notas de la escala: {', '.join(musical['scale']['notes'])}")
        
        print("\n--- Análisis Técnico ---")
        technical = analysis['technical_analysis']
        print(f"Energía: {technical['energy']['value']:.2f} - {technical['energy']['description']}")
        print(f"Bailabilidad: {technical['danceability']['value']:.2f} - {technical['danceability']['description']}")
        print(f"Valencia: {technical['valence']['value']:.2f} - {technical['valence']['description']}")
        
        print("\n--- Insights Educativos ---")
        educational = analysis['educational_insights']
        print("Puntos de aprendizaje:")
        for point in educational['learning_points']:
            print(f"  • {point}")
        
        print("\nSugerencias de práctica:")
        for suggestion in educational['practice_suggestions'][:3]:
            print(f"  • {suggestion}")
        
        # Coaching
        print("\n=== COACHING MUSICAL ===\n")
        coaching = coach.generate_coaching_analysis(analysis)
        
        print(f"Dificultad: {coaching['overview']['difficulty_level']}")
        print(f"Adecuado para: {', '.join(coaching['overview']['suitable_for'])}")
        
        print("\nRuta de Aprendizaje:")
        for step in coaching['learning_path'][:3]:
            print(f"  Paso {step['step']}: {step['title']}")
            print(f"    {step['description']}")
            print(f"    Duración: {step['duration']}\n")
        
        print("Ejercicios de Práctica:")
        for exercise in coaching['practice_exercises'][:2]:
            print(f"  • {exercise['title']}")
            print(f"    {exercise['description']}")
            print(f"    Tempo: {exercise['tempo']} BPM, Repeticiones: {exercise['repetitions']}\n")
        
        print("Tips de Interpretación:")
        for tip in coaching['performance_tips'][:3]:
            print(f"  • {tip}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Asegúrate de tener configuradas las credenciales de Spotify en el archivo .env")
    print("SPOTIFY_CLIENT_ID y SPOTIFY_CLIENT_SECRET\n")
    
    asyncio.run(example_basic_analysis())

