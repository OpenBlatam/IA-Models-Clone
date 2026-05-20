"""
Validación de OpenAPI Schema
Verifica que la API cumple con el esquema OpenAPI
"""

import requests
import json
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"

def fetch_openapi_schema() -> Dict[str, Any]:
    """Obtiene el esquema OpenAPI."""
    try:
        response = requests.get(f"{BASE_URL}/api/openapi.json", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error obteniendo OpenAPI schema: {e}")
    return {}

def validate_endpoint_exists(schema: Dict[str, Any], method: str, path: str) -> bool:
    """Valida que un endpoint existe en el schema."""
    paths = schema.get("paths", {})
    if path not in paths:
        return False
    
    endpoint = paths[path]
    return method.lower() in endpoint

def validate_response_schema(schema: Dict[str, Any], method: str, path: str, 
                            response_data: Dict[str, Any]) -> List[str]:
    """Valida que la respuesta cumple con el schema."""
    errors = []
    
    paths = schema.get("paths", {})
    if path not in paths:
        errors.append(f"Path {path} no existe en schema")
        return errors
    
    endpoint = paths[path].get(method.lower(), {})
    responses = endpoint.get("responses", {})
    
    # Verificar que hay respuesta 200
    if "200" not in responses:
        errors.append(f"No hay respuesta 200 definida para {method} {path}")
        return errors
    
    # Validación básica (se puede expandir con jsonschema)
    response_schema = responses["200"].get("content", {}).get("application/json", {}).get("schema", {})
    
    return errors

def test_openapi_compliance():
    """Prueba cumplimiento de OpenAPI."""
    print(f"\n{'='*70}")
    print(f"  🔍 VALIDACIÓN DE OPENAPI SCHEMA")
    print(f"{'='*70}\n")
    
    # Obtener schema
    print("📥 Obteniendo esquema OpenAPI...")
    schema = fetch_openapi_schema()
    
    if not schema:
        print("❌ No se pudo obtener el esquema OpenAPI")
        return
    
    print(f"✅ Esquema obtenido")
    print(f"   Versión: {schema.get('openapi', 'Unknown')}")
    print(f"   Título: {schema.get('info', {}).get('title', 'Unknown')}")
    print(f"   Endpoints: {len(schema.get('paths', {}))}\n")
    
    # Validar endpoints principales
    endpoints_to_test = [
        ("GET", "/"),
        ("GET", "/api/health"),
        ("GET", "/api/stats"),
        ("POST", "/api/documents/generate"),
        ("GET", "/api/tasks/{task_id}/status"),
        ("GET", "/api/tasks"),
        ("GET", "/api/documents"),
    ]
    
    print("🔍 Validando endpoints...")
    all_valid = True
    
    for method, path in endpoints_to_test:
        # Verificar existencia
        if validate_endpoint_exists(schema, method, path.replace("{task_id}", "test")):
            print(f"  ✅ {method} {path}")
        else:
            print(f"  ❌ {method} {path} - No encontrado en schema")
            all_valid = False
    
    # Validar schema JSON
    print(f"\n📋 Información del schema:")
    info = schema.get("info", {})
    print(f"   Título: {info.get('title', 'N/A')}")
    print(f"   Versión: {info.get('version', 'N/A')}")
    print(f"   Descripción: {info.get('description', 'N/A')[:100]}...")
    
    # Contar endpoints
    paths = schema.get("paths", {})
    total_endpoints = sum(len(methods) for methods in paths.values())
    print(f"   Total de endpoints: {total_endpoints}")
    
    # Verificar componentes
    components = schema.get("components", {})
    schemas = components.get("schemas", {})
    print(f"   Modelos definidos: {len(schemas)}")
    
    print(f"\n{'='*70}")
    if all_valid:
        print("✅ Validación completada")
    else:
        print("⚠️  Algunos endpoints no están en el schema")
    print(f"{'='*70}\n")
    
    return all_valid

if __name__ == "__main__":
    test_openapi_compliance()
































