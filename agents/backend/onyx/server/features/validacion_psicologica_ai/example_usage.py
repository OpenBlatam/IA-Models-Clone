"""
Ejemplo de uso del sistema de Validación Psicológica AI
========================================================
"""

import asyncio
from uuid import UUID, uuid4
from agents.backend.onyx.server.features.validacion_psicologica_ai import (
    PsychologicalValidationService,
    SocialMediaPlatform,
)
from agents.backend.onyx.server.features.validacion_psicologica_ai.schemas import (
    SocialMediaConnectRequest,
    ValidationCreate,
)


async def ejemplo_completo():
    """Ejemplo completo de uso del sistema"""
    
    # Inicializar servicio
    service = PsychologicalValidationService()
    
    # Simular ID de usuario
    user_id = uuid4()
    
    print("=" * 60)
    print("Ejemplo de Validación Psicológica AI")
    print("=" * 60)
    
    # 1. Conectar redes sociales
    print("\n1. Conectando redes sociales...")
    
    # Conectar Instagram
    instagram_request = SocialMediaConnectRequest(
        platform=SocialMediaPlatform.INSTAGRAM,
        access_token="instagram_token_123",
        refresh_token="instagram_refresh_123",
        expires_in=3600
    )
    instagram_conn = await service.connect_social_media(user_id, instagram_request)
    print(f"✓ Instagram conectado: {instagram_conn.id}")
    
    # Conectar Twitter
    twitter_request = SocialMediaConnectRequest(
        platform=SocialMediaPlatform.TWITTER,
        access_token="twitter_token_123",
        refresh_token="twitter_refresh_123",
        expires_in=3600
    )
    twitter_conn = await service.connect_social_media(user_id, twitter_request)
    print(f"✓ Twitter conectado: {twitter_conn.id}")
    
    # 2. Verificar conexiones
    print("\n2. Verificando conexiones...")
    connections = await service.get_user_connections(user_id)
    print(f"✓ Total de conexiones: {len(connections)}")
    for conn in connections:
        print(f"  - {conn.platform.value}: {conn.status.value}")
    
    # 3. Crear validación
    print("\n3. Creando validación psicológica...")
    validation_request = ValidationCreate(
        platforms=[SocialMediaPlatform.INSTAGRAM, SocialMediaPlatform.TWITTER],
        include_historical_data=True,
        analysis_depth="deep"
    )
    validation = await service.create_validation(user_id, validation_request)
    print(f"✓ Validación creada: {validation.id}")
    print(f"  Plataformas: {[p.value for p in validation.connected_platforms]}")
    
    # 4. Ejecutar análisis
    print("\n4. Ejecutando análisis psicológico...")
    print("  (Esto puede tomar unos segundos...)")
    validation = await service.run_validation(validation.id)
    print(f"✓ Análisis completado: {validation.status.value}")
    
    # 5. Mostrar resultados
    print("\n5. Resultados del análisis:")
    print("-" * 60)
    
    if validation.profile:
        print("\n📊 Perfil Psicológico:")
        print(f"  Confidence Score: {validation.profile.confidence_score * 100:.1f}%")
        print(f"\n  Rasgos de Personalidad:")
        for trait, score in validation.profile.personality_traits.items():
            print(f"    - {trait.capitalize()}: {score:.2f}")
        
        print(f"\n  Estado Emocional:")
        for key, value in validation.profile.emotional_state.items():
            if isinstance(value, float):
                print(f"    - {key.capitalize()}: {value:.2f}")
            else:
                print(f"    - {key.capitalize()}: {value}")
        
        print(f"\n  Fortalezas:")
        for strength in validation.profile.strengths:
            print(f"    - {strength}")
        
        print(f"\n  Recomendaciones:")
        for rec in validation.profile.recommendations:
            print(f"    - {rec}")
    
    if validation.report:
        print("\n📄 Reporte de Validación:")
        print(f"  Resumen: {validation.report.summary[:200]}...")
        print(f"\n  Insights por Plataforma:")
        for platform, insights in validation.report.social_media_insights.items():
            print(f"    - {platform}:")
            print(f"      Posts: {insights.get('post_count', 0)}")
            print(f"      Engagement: {insights.get('engagement_rate', 0):.1%}")
    
    # 6. Obtener historial
    print("\n6. Historial de validaciones:")
    validations = await service.get_user_validations(user_id)
    print(f"✓ Total de validaciones: {len(validations)}")
    for v in validations:
        print(f"  - {v.id}: {v.status.value} ({v.created_at.strftime('%Y-%m-%d %H:%M')})")
    
    print("\n" + "=" * 60)
    print("Ejemplo completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(ejemplo_completo())




