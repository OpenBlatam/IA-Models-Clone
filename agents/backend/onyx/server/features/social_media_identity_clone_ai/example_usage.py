"""
Ejemplo de uso de Social Media Identity Clone AI
"""

import asyncio
from services.profile_extractor import ProfileExtractor
from services.identity_analyzer import IdentityAnalyzer
from services.content_generator import ContentGenerator


async def main():
    """Ejemplo completo de uso"""
    
    print("🚀 Social Media Identity Clone AI - Ejemplo de Uso\n")
    
    # 1. Inicializar extractor
    print("1️⃣ Inicializando extractor de perfiles...")
    extractor = ProfileExtractor()
    
    # 2. Extraer perfiles (ejemplo)
    print("\n2️⃣ Extrayendo perfiles de redes sociales...")
    
    # TikTok
    tiktok_username = "ejemplo_tiktok"
    print(f"   📱 Extrayendo perfil de TikTok: @{tiktok_username}")
    tiktok_profile = await extractor.extract_tiktok_profile(tiktok_username)
    print(f"   ✅ Perfil de TikTok extraído: {len(tiktok_profile.videos)} videos")
    
    # Instagram
    instagram_username = "ejemplo_instagram"
    print(f"   📸 Extrayendo perfil de Instagram: @{instagram_username}")
    instagram_profile = await extractor.extract_instagram_profile(instagram_username)
    print(f"   ✅ Perfil de Instagram extraído: {len(instagram_profile.posts)} posts")
    
    # YouTube
    youtube_channel_id = "ejemplo_youtube"
    print(f"   🎥 Extrayendo canal de YouTube: {youtube_channel_id}")
    youtube_profile = await extractor.extract_youtube_profile(youtube_channel_id)
    print(f"   ✅ Canal de YouTube extraído: {len(youtube_profile.videos)} videos")
    
    # 3. Construir identidad
    print("\n3️⃣ Construyendo perfil de identidad...")
    analyzer = IdentityAnalyzer()
    identity = await analyzer.build_identity(
        tiktok_profile=tiktok_profile,
        instagram_profile=instagram_profile,
        youtube_profile=youtube_profile
    )
    print(f"   ✅ Identidad construida: {identity.profile_id}")
    print(f"   📊 Estadísticas:")
    print(f"      - Videos: {identity.total_videos}")
    print(f"      - Posts: {identity.total_posts}")
    print(f"      - Temas: {len(identity.content_analysis.topics)}")
    print(f"      - Tono: {identity.content_analysis.tone}")
    
    # 4. Generar contenido
    print("\n4️⃣ Generando contenido basado en identidad...")
    generator = ContentGenerator(identity_profile=identity)
    
    # Generar post de Instagram
    print("\n   📸 Generando post para Instagram...")
    instagram_post = await generator.generate_instagram_post(
        topic="fitness",
        style="motivational"
    )
    print(f"   ✅ Post generado:")
    print(f"      {instagram_post.content[:100]}...")
    print(f"      Hashtags: {', '.join(instagram_post.hashtags[:5])}")
    
    # Generar script de TikTok
    print("\n   📱 Generando script para TikTok...")
    tiktok_script = await generator.generate_tiktok_script(
        topic="cooking",
        duration=60
    )
    print(f"   ✅ Script generado:")
    print(f"      {tiktok_script.content[:100]}...")
    
    # Generar descripción de YouTube
    print("\n   🎥 Generando descripción para YouTube...")
    youtube_description = await generator.generate_youtube_description(
        video_title="Mi Rutina de Mañana",
        tags=["productivity", "morning routine", "self-improvement"]
    )
    print(f"   ✅ Descripción generada:")
    print(f"      {youtube_description.content[:100]}...")
    
    print("\n✨ Proceso completado exitosamente!")


if __name__ == "__main__":
    asyncio.run(main())




