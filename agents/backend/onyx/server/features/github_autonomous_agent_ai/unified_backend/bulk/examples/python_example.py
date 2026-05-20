"""
Ejemplo de uso del SDK Python
=============================
"""

from bul_api_client import create_bul_client, DocumentRequest

def main():
    client = create_bul_client(base_url="http://localhost:8000")
    
    health = client.get_health()
    print(f"✅ API Health: {health['status']}")
    
    request = DocumentRequest(
        query="Crear un plan de marketing digital para una startup tecnológica",
        business_area="marketing",
        document_type="strategy",
        priority=1
    )
    
    def on_progress(status):
        print(f"📊 Progreso: {status.progress}% - Estado: {status.status}")
    
    try:
        print("\n🚀 Generando documento...")
        document = client.generate_document_and_wait(
            request,
            use_websocket=True,
            on_progress=on_progress
        )
        
        print(f"\n✅ Documento generado:")
        print(f"   Título: {document['document']['title']}")
        print(f"   Longitud: {len(document['document']['content'])} caracteres")
        print(f"   Tipo: {document['document']['document_type']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📋 Listando documentos recientes...")
    documents = client.list_documents(limit=5)
    print(f"   Total documentos: {documents.get('total', 0)}")
    
    print("\n📊 Estadísticas de la API...")
    stats = client.get_stats()
    print(f"   Tareas completadas: {stats.get('tasks_completed', 0)}")
    print(f"   Tareas activas: {stats.get('tasks_active', 0)}")

if __name__ == "__main__":
    main()
































