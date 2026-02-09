"""
Example Usage
=============

Ejemplos de uso del Artist Manager AI.
"""

import asyncio
import os
from datetime import datetime, time, timedelta
from artist_manager_ai import ArtistManager
from artist_manager_ai.core.calendar_manager import CalendarEvent, EventType
from artist_manager_ai.core.routine_manager import RoutineTask, RoutineType
from artist_manager_ai.core.protocol_manager import Protocol, ProtocolCategory, ProtocolPriority
from artist_manager_ai.core.wardrobe_manager import WardrobeItem, Outfit, DressCode, Season


async def main():
    """Ejemplo principal de uso."""
    
    # Configurar API key (o usar variable de entorno)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
    artist_id = "artist_001"
    
    # Crear manager
    async with ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key) as manager:
        
        # 1. Crear evento
        print("📅 Creando evento...")
        event = CalendarEvent(
            id="event_001",
            title="Concierto en el Estadio",
            description="Concierto principal de la gira",
            event_type=EventType.CONCERT,
            start_time=datetime.now() + timedelta(days=3),
            end_time=datetime.now() + timedelta(days=3, hours=3),
            location="Estadio Nacional",
            protocol_requirements=["Llegar 2 horas antes", "No usar teléfono en escenario"]
        )
        manager.calendar.add_event(event)
        print(f"✅ Evento creado: {event.title}")
        
        # 2. Crear rutina
        print("\n🔄 Creando rutina...")
        routine = RoutineTask(
            id="routine_001",
            title="Ejercicio matutino",
            description="30 minutos de ejercicio",
            routine_type=RoutineType.MORNING,
            scheduled_time=time(7, 0),
            duration_minutes=30,
            priority=8,
            days_of_week=[0, 1, 2, 3, 4, 5, 6]  # Todos los días
        )
        manager.routines.add_routine(routine)
        print(f"✅ Rutina creada: {routine.title}")
        
        # 3. Crear protocolo
        print("\n📋 Creando protocolo...")
        protocol = Protocol(
            id="protocol_001",
            title="Protocolo de Redes Sociales",
            description="Guía de comportamiento en redes sociales",
            category=ProtocolCategory.SOCIAL_MEDIA,
            priority=ProtocolPriority.HIGH,
            rules=[
                "No publicar contenido sin aprobación",
                "Responder comentarios de forma profesional",
                "No discutir temas políticos"
            ],
            do_s=["Ser auténtico", "Interactuar con fans"],
            dont_s=["Publicar contenido ofensivo", "Ignorar comentarios"]
        )
        manager.protocols.add_protocol(protocol)
        print(f"✅ Protocolo creado: {protocol.title}")
        
        # 4. Agregar items al guardarropa
        print("\n👔 Agregando items al guardarropa...")
        item1 = WardrobeItem(
            id="item_001",
            name="Camisa negra elegante",
            category="shirt",
            color="black",
            brand="Designer Brand",
            dress_codes=[DressCode.FORMAL, DressCode.SMART_CASUAL],
            season=Season.ALL_SEASON
        )
        manager.wardrobe.add_item(item1)
        print(f"✅ Item agregado: {item1.name}")
        
        # 5. Generar resumen diario con IA
        print("\n🤖 Generando resumen diario con IA...")
        try:
            summary = await manager.generate_daily_summary()
            print(f"✅ Resumen generado:")
            print(f"   - Eventos: {summary.get('events_count', 0)}")
            print(f"   - Rutinas pendientes: {summary.get('pending_routines_count', 0)}")
            if 'summary' in summary:
                print(f"   - Resumen: {summary['summary'][:100]}...")
        except Exception as e:
            print(f"⚠️  Error generando resumen: {e}")
        
        # 6. Generar recomendación de vestimenta con IA
        print("\n👔 Generando recomendación de vestimenta...")
        try:
            recommendation = await manager.generate_wardrobe_recommendation("event_001")
            print(f"✅ Recomendación generada:")
            print(f"   - Código de vestimenta: {recommendation.dress_code.value}")
            print(f"   - Razón: {recommendation.reasoning[:100]}...")
        except Exception as e:
            print(f"⚠️  Error generando recomendación: {e}")
        
        # 7. Verificar cumplimiento de protocolos
        print("\n📋 Verificando cumplimiento de protocolos...")
        try:
            compliance = await manager.check_protocol_compliance("event_001")
            print(f"✅ Verificación completada:")
            print(f"   - Cumplimiento: {compliance.get('compliant', 'N/A')}")
            print(f"   - Protocolos verificados: {len(compliance.get('checked_protocols', []))}")
        except Exception as e:
            print(f"⚠️  Error verificando protocolos: {e}")
        
        # 8. Obtener dashboard
        print("\n📊 Obteniendo dashboard...")
        dashboard = manager.get_dashboard_data()
        print(f"✅ Dashboard obtenido:")
        print(f"   - Eventos próximos: {dashboard['upcoming_events']['count']}")
        print(f"   - Rutinas pendientes: {dashboard['routines']['pending_count']}")
        print(f"   - Protocolos críticos: {dashboard['protocols']['critical_count']}")
        print(f"   - Items en guardarropa: {dashboard['wardrobe']['total_items']}")


if __name__ == "__main__":
    asyncio.run(main())




