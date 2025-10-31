from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
from typing import List
import httpx
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo Script for Template-based Video Generation with AI Avatars
=============================================================

Demonstrates the complete workflow:
1. Template selection
2. AI avatar configuration
3. Image synchronization
4. Script generation
5. Final video composition
"""



class TemplateVideoDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.client = None
        
    async def __aenter__(self) -> Any:
        self.client = httpx.AsyncClient(base_url=self.base_url)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if self.client:
            await self.client.aclose()

    async def list_templates(self) -> dict:
        """Get available templates."""
        response = await self.client.get(
            "/api/v1/templates",
            headers={"Authorization": "Bearer demo-token"}
        )
        return response.json()

    async def get_template_details(self, template_id: str) -> dict:
        """Get template details."""
        response = await self.client.get(
            f"/api/v1/templates/{template_id}",
            headers={"Authorization": "Bearer demo-token"}
        )
        return response.json()

    async def create_avatar_preview(self) -> dict:
        """Create avatar preview."""
        response = await self.client.post(
            "/api/v1/avatar/preview",
            json={
                "avatar_config": {
                    "gender": "female",
                    "style": "realistic",
                    "age_range": "25-35",
                    "ethnicity": "hispanic",
                    "outfit": "business",
                    "voice_settings": {
                        "language": "es",
                        "accent": "neutral",
                        "speed": 1.0,
                        "pitch": 1.0
                    }
                },
                "sample_text": "Hola, soy tu avatar de IA. Voy a presentar información sobre nuestros productos.",
                "preview_duration": 15
            },
            headers={"Authorization": "Bearer demo-token"}
        )
        return response.json()

    async def create_template_video(self) -> dict:
        """Create complete template video with avatar and image sync."""
        response = await self.client.post(
            "/api/v1/videos/template",
            json={
                "template_id": "business_professional",
                "user_id": "demo_user",
                "avatar_config": {
                    "gender": "female",
                    "style": "realistic",
                    "age_range": "25-35",
                    "ethnicity": "hispanic",
                    "outfit": "business",
                    "voice_settings": {
                        "language": "es",
                        "accent": "neutral",
                        "speed": 1.0,
                        "pitch": 1.0
                    }
                },
                "image_sync": {
                    "sync_mode": "auto",
                    "images": [
                        "https://example.com/product1.jpg",
                        "https://example.com/product2.jpg",
                        "https://example.com/chart.jpg",
                        "https://example.com/team.jpg"
                    ],
                    "transition_duration": 0.5,
                    "default_image_duration": 4.0
                },
                "script_config": {
                    "content": "Quiero presentar nuestros productos innovadores que están revolucionando el mercado. Mostraremos las características principales, los beneficios para nuestros clientes y los resultados que hemos logrado este año.",
                    "tone": "professional",
                    "language": "es",
                    "target_duration": 60,
                    "include_pauses": True,
                    "speaking_rate": 1.0,
                    "keywords": ["innovadores", "beneficios", "resultados"]
                },
                "output_format": "mp4",
                "quality": "high",
                "aspect_ratio": "16:9",
                "background_music": "corporate_soft",
                "watermark": "Mi Empresa"
            },
            headers={"Authorization": "Bearer demo-token"}
        )
        return response.json()

    async def get_video_status(self, request_id: str) -> dict:
        """Get template video status."""
        response = await self.client.get(
            f"/api/v1/videos/template/{request_id}",
            headers={"Authorization": "Bearer demo-token"}
        )
        return response.json()

    async def demo_complete_workflow(self) -> Any:
        """Demo the complete template video workflow."""
        print("🎬 Template Video Demo - Complete Workflow")
        print("=" * 60)
        
        try:
            # Step 1: List templates
            print("📋 1. Listando templates disponibles...")
            templates_response = await self.list_templates()
            
            if templates_response.get("success"):
                templates = templates_response["data"]["templates"]
                print(f"✅ Encontrados {len(templates)} templates")
                for template in templates:
                    print(f"  • {template['name']} ({template['category']})")
            
            # Step 2: Get template details
            print("\n🔍 2. Obteniendo detalles del template...")
            template_details = await self.get_template_details("business_professional")
            
            if template_details.get("success"):
                template = template_details["data"]
                print(f"✅ Template: {template['name']}")
                print(f"  📝 Descripción: {template['description']}")
                print(f"  🎯 Características: {', '.join(template['features'])}")
            
            # Step 3: Create avatar preview
            print("\n👤 3. Creando preview del avatar IA...")
            avatar_preview = await self.create_avatar_preview()
            
            if avatar_preview.get("success"):
                preview = avatar_preview["data"]
                print(f"✅ Avatar preview: {preview['preview_id']}")
                print(f"  🎥 URL: {preview['avatar_video_url']}")
                print(f"  ⏰ Expira: {preview['expires_at']}")
            
            # Step 4: Create complete template video
            print("\n🚀 4. Creando video completo con template...")
            video_response = await self.create_template_video()
            
            if video_response.get("success"):
                video = video_response["data"]
                print(f"✅ Video iniciado: {video['request_id']}")
                print(f"  📋 Template: {video['template_id']}")
                print(f"  📊 Estado: {video['status']}")
                print(f"  ⏱️ Estimado: {video['estimated_completion']}")
                
                # Monitor processing stages
                print("\n📊 5. Monitoreando progreso...")
                for i in range(10):  # Monitor for up to 10 iterations
                    await asyncio.sleep(2)
                    
                    status_response = await self.get_video_status(video['request_id'])
                    if status_response.get("success"):
                        status = status_response["data"]
                        
                        print(f"\n  Iteración {i+1}:")
                        print(f"    Estado general: {status['status']}")
                        
                        # Show processing stages
                        stages = status['processing_stages']
                        for stage, stage_status in stages.items():
                            emoji = "✅" if stage_status == "completed" else "🔄" if stage_status == "processing" else "⏳"
                            print(f"    {emoji} {stage}: {stage_status}")
                        
                        # Check if completed
                        if status['status'] == 'completed':
                            print(f"\n🎉 ¡Video completado!")
                            print(f"  🎥 Video final: {status['final_video_url']}")
                            print(f"  👤 Avatar: {status['avatar_video_url']}")
                            print(f"  🖼️ Thumbnail: {status['thumbnail_url']}")
                            print(f"  📝 Script generado: {status['generated_script']}")
                            print(f"  ⏱️ Tiempo procesamiento: {status['processing_time']}s")
                            break
                        elif status['status'] == 'failed':
                            print(f"❌ Error: {status.get('error_message', 'Unknown error')}")
                            break
            
            print("\n" + "=" * 60)
            print("✨ Demo completado - Funcionalidades demostradas:")
            print("  • ✅ Selección de templates")
            print("  • ✅ Configuración de avatar IA")
            print("  • ✅ Sincronización de imágenes")
            print("  • ✅ Generación de script")
            print("  • ✅ Composición de video final")
            print("  • ✅ Monitoreo en tiempo real")
            
        except Exception as e:
            print(f"❌ Error en demo: {e}")
            print("💡 Asegúrate de que la API esté ejecutándose en http://localhost:8000")


async def main():
    """Main demo function."""
    print("🚀 Iniciando Demo de Template Video con Avatar IA...")
    
    async with TemplateVideoDemo() as demo:
        await demo.demo_complete_workflow()


match __name__:
    case "__main__":
    asyncio.run(main()) 