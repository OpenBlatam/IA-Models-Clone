from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import requests
import json
import asyncio
import aiohttp
from typing import Dict, List, Any
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
    import concurrent.futures
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
FastAPI Client - Official Documentation Reference System
=======================================================

Cliente de ejemplo para demostrar el uso de la API FastAPI del sistema
de referencias de documentación oficial.
"""


class OfficialDocsAPIClient:
    """Cliente para la API de referencias de documentación oficial."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = requests.Session()
    
    async def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Realizar request HTTP."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error en request HTTP: {e}")
            return {"success": False, "error": str(e)}
    
    def get_library_info(self, library_name: str) -> Dict:
        """Obtener información de una librería."""
        data = {"library_name": library_name}
        return self._make_request("POST", "/library/info", data)
    
    async def get_api_reference(self, library_name: str, api_name: str) -> Dict:
        """Obtener referencia de una API."""
        data = {"library_name": library_name, "api_name": api_name}
        return self._make_request("POST", "/api/reference", data)
    
    def get_best_practices(self, library_name: str, category: str = None) -> Dict:
        """Obtener mejores prácticas."""
        data = {"library_name": library_name}
        if category:
            data["category"] = category
        return self._make_request("POST", "/best-practices", data)
    
    def check_version_compatibility(self, library_name: str, version: str) -> Dict:
        """Verificar compatibilidad de versiones."""
        data = {"library_name": library_name, "version": version}
        return self._make_request("POST", "/version/compatibility", data)
    
    def validate_code(self, code: str, library_name: str) -> Dict:
        """Validar código."""
        data = {"code": code, "library_name": library_name}
        return self._make_request("POST", "/code/validate", data)
    
    def get_performance_recommendations(self, library_name: str) -> Dict:
        """Obtener recomendaciones de rendimiento."""
        data = {"library_name": library_name}
        return self._make_request("POST", "/performance/recommendations", data)
    
    def generate_migration_guide(self, library_name: str, from_version: str, to_version: str) -> Dict:
        """Generar guía de migración."""
        data = {
            "library_name": library_name,
            "from_version": from_version,
            "to_version": to_version
        }
        return self._make_request("POST", "/migration/guide", data)
    
    def list_libraries(self) -> Dict:
        """Listar librerías disponibles."""
        return self._make_request("GET", "/libraries")
    
    async def list_apis(self, library_name: str) -> Dict:
        """Listar APIs de una librería."""
        return self._make_request("GET", f"/apis/{library_name}")
    
    def analyze_project(self, libraries: List[str]) -> Dict:
        """Analizar un proyecto."""
        return self._make_request("POST", "/analyze/project", {"libraries": libraries})
    
    def export_references(self) -> Dict:
        """Exportar referencias."""
        return self._make_request("POST", "/export/references")

class AsyncOfficialDocsAPIClient:
    """Cliente asíncrono para la API de referencias."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
    
    async async def _make_request(self, session: aiohttp.ClientSession, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Realizar request HTTP asíncrono."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with session.get(url) as response:
                    return await response.json()
            elif method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
                
        except Exception as e:
            print(f"Error en request HTTP asíncrono: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_library_info_async(self, session: aiohttp.ClientSession, library_name: str) -> Dict:
        """Obtener información de librería de forma asíncrona."""
        data = {"library_name": library_name}
        return await self._make_request(session, "POST", "/library/info", data)
    
    async async def get_api_reference_async(self, session: aiohttp.ClientSession, library_name: str, api_name: str) -> Dict:
        """Obtener referencia de API de forma asíncrona."""
        data = {"library_name": library_name, "api_name": api_name}
        return await self._make_request(session, "POST", "/api/reference", data)
    
    async def analyze_multiple_libraries(self, libraries: List[str]) -> Dict:
        """Analizar múltiples librerías de forma asíncrona."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for lib in libraries:
                task = self.get_library_info_async(session, lib)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return {
                "libraries": libraries,
                "results": results,
                "count": len(results)
            }

def demonstrate_sync_client():
    """Demostrar el uso del cliente síncrono."""
    print("🔥 DEMOSTRACIÓN DEL CLIENTE SÍNCRONO")
    print("=" * 50)
    
    client = OfficialDocsAPIClient()
    
    # 1. Obtener información de librerías
    print("\n1. 📚 Información de librerías:")
    libraries = ["pytorch", "transformers", "diffusers", "gradio"]
    
    for lib in libraries:
        result = client.get_library_info(lib)
        if result.get("success"):
            info = result["library"]
            print(f"  {lib.upper()}: v{info['current_version']} - {info['documentation_url']}")
        else:
            print(f"  ❌ Error obteniendo info de {lib}: {result.get('error')}")
    
    # 2. Obtener referencias de API
    print("\n2. 🔧 Referencias de API:")
    api_requests = [
        ("pytorch", "mixed_precision"),
        ("transformers", "model_loading"),
        ("diffusers", "pipeline_usage"),
        ("gradio", "interface_creation")
    ]
    
    for lib, api in api_requests:
        result = client.get_api_reference(lib, api)
        if result.get("success"):
            ref = result["api_reference"]
            print(f"  {lib}/{api}: {ref['description'][:50]}...")
        else:
            print(f"  ❌ Error obteniendo {lib}/{api}: {result.get('error')}")
    
    # 3. Verificar compatibilidad de versiones
    print("\n3. 📊 Compatibilidad de versiones:")
    version_checks = [
        ("pytorch", "2.0.0"),
        ("pytorch", "1.12.0"),
        ("transformers", "4.30.0"),
        ("transformers", "4.15.0")
    ]
    
    for lib, version in version_checks:
        result = client.check_version_compatibility(lib, version)
        if result.get("success"):
            compat = result["compatibility"]
            status = "✅" if compat["compatible"] else "❌"
            print(f"  {status} {lib} {version}: {compat['recommendation']}")
        else:
            print(f"  ❌ Error verificando {lib} {version}: {result.get('error')}")
    
    # 4. Validar código
    print("\n4. 🔍 Validación de código:")
    code_samples = [
        ("pytorch", """
scaler = GradScaler()
with autocast():
    output = model(input)
"""),
        ("transformers", """
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
""")
    ]
    
    for lib, code in code_samples:
        result = client.validate_code(code, lib)
        if result.get("success"):
            validation = result["validation"]
            status = "✅" if validation["valid"] else "❌"
            print(f"  {status} {lib}: {len(validation.get('recommendations', []))} recomendaciones")
        else:
            print(f"  ❌ Error validando código de {lib}: {result.get('error')}")
    
    # 5. Análisis de proyecto
    print("\n5. 📈 Análisis de proyecto:")
    result = client.analyze_project(libraries)
    if result.get("success"):
        analysis = result["analysis"]["project_analysis"]
        print(f"  Total de recomendaciones: {analysis['total_recommendations']}")
        for lib_analysis in analysis["libraries"]:
            print(f"  {lib_analysis['name']}: {lib_analysis['recommendations_count']} recomendaciones")
    else:
        print(f"  ❌ Error en análisis: {result.get('error')}")

async def demonstrate_async_client():
    """Demostrar el uso del cliente asíncrono."""
    print("\n⚡ DEMOSTRACIÓN DEL CLIENTE ASÍNCRONO")
    print("=" * 50)
    
    client = AsyncOfficialDocsAPIClient()
    
    # Analizar múltiples librerías de forma asíncrona
    libraries = ["pytorch", "transformers", "diffusers", "gradio"]
    
    print(f"\n🔄 Analizando {len(libraries)} librerías de forma asíncrona...")
    start_time = time.time()
    
    result = await client.analyze_multiple_libraries(libraries)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Análisis completado en {duration:.2f} segundos")
    print(f"📊 Resultados para {result['count']} librerías:")
    
    for i, (lib, lib_result) in enumerate(zip(libraries, result["results"])):
        if lib_result.get("success"):
            info = lib_result["library"]
            print(f"  {i+1}. {lib.upper()}: v{info['current_version']}")
        else:
            print(f"  {i+1}. ❌ {lib}: {lib_result.get('error')}")

def demonstrate_error_handling():
    """Demostrar manejo de errores."""
    print("\n🛡️ DEMOSTRACIÓN DE MANEJO DE ERRORES")
    print("=" * 50)
    
    client = OfficialDocsAPIClient()
    
    # Casos de error
    error_cases = [
        ("library/info", {"library_name": "invalid_library"}),
        ("api/reference", {"library_name": "pytorch", "api_name": "invalid_api"}),
        ("version/compatibility", {"library_name": "invalid_lib", "version": "1.0.0"}),
    ]
    
    for endpoint, data in error_cases:
        print(f"\n🔍 Probando endpoint: {endpoint}")
        result = client._make_request("POST", f"/{endpoint}", data)
        
        if not result.get("success"):
            print(f"  ❌ Error esperado: {result.get('error')}")
        else:
            print(f"  ⚠️  Resultado inesperado: {result}")

def performance_test():
    """Prueba de rendimiento del cliente."""
    print("\n⚡ PRUEBA DE RENDIMIENTO")
    print("=" * 50)
    
    client = OfficialDocsAPIClient()
    
    # Prueba de múltiples requests secuenciales
    print("\n🔄 Requests secuenciales:")
    start_time = time.time()
    
    for i in range(5):
        result = client.get_library_info("pytorch")
        if result.get("success"):
            print(f"  Request {i+1}: ✅")
        else:
            print(f"  Request {i+1}: ❌")
    
    end_time = time.time()
    sequential_duration = end_time - start_time
    print(f"  ⏱️  Tiempo total: {sequential_duration:.2f} segundos")
    
    # Prueba de requests concurrentes
    print("\n🔄 Requests concurrentes:")
    start_time = time.time()
    
    
    def make_request():
        
    """make_request function."""
return client.get_library_info("pytorch")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [future.result() for future in futures]
    
    end_time = time.time()
    concurrent_duration = end_time - start_time
    
    success_count = sum(1 for r in results if r.get("success"))
    print(f"  ✅ Requests exitosos: {success_count}/5")
    print(f"  ⏱️  Tiempo total: {concurrent_duration:.2f} segundos")
    print(f"  🚀 Mejora: {sequential_duration/concurrent_duration:.1f}x más rápido")

def main():
    """Función principal."""
    print("🎯 CLIENTE FASTAPI - SISTEMA DE REFERENCIAS")
    print("Demostración completa del cliente")
    print("=" * 80)
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor FastAPI detectado y funcionando")
        else:
            print("⚠️  Servidor respondió pero con estado inesperado")
    except requests.exceptions.RequestException:
        print("❌ No se pudo conectar al servidor FastAPI")
        print("💡 Asegúrate de que el servidor esté corriendo en http://localhost:8000")
        print("   Ejecuta: python fastapi_integration.py")
        return
    
    # Ejecutar demostraciones
    demonstrate_sync_client()
    
    # Ejecutar cliente asíncrono
    asyncio.run(demonstrate_async_client())
    
    # Demostrar manejo de errores
    demonstrate_error_handling()
    
    # Prueba de rendimiento
    performance_test()
    
    print("\n🎉 ¡Demostración completada exitosamente!")
    print("El cliente FastAPI está listo para usar en producción.")

match __name__:
    case "__main__":
    main() 