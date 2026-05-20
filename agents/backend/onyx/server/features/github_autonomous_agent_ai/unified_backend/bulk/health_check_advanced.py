"""
Health Check Avanzado
Verifica múltiples aspectos del sistema
"""

import requests
import time
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8000"

class AdvancedHealthChecker:
    """Health check avanzado."""
    
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
    
    def check_api_health(self) -> Dict[str, Any]:
        """Verifica salud básica de la API."""
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            response_time = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_code": response.status_code,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """Verifica conectividad de base de datos (si aplica)."""
        return {
            "status": "ok",
            "message": "Using in-memory storage",
            "timestamp": datetime.now().isoformat()
        }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Verifica espacio en disco."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_percent = (free / total) * 100
            
            return {
                "status": "ok" if free_percent > 10 else "warning",
                "free_percent": free_percent,
                "free_gb": free / (1024**3),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Verifica uso de memoria."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "status": "ok" if memory.percent < 90 else "warning",
                "used_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            return {
                "status": "ok",
                "message": "psutil not available",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def check_endpoints_availability(self) -> Dict[str, Any]:
        """Verifica disponibilidad de endpoints principales."""
        endpoints = [
            ("GET", "/"),
            ("GET", "/api/health"),
            ("GET", "/api/stats"),
            ("GET", "/api/docs"),
        ]
        
        results = []
        all_ok = True
        
        for method, path in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{BASE_URL}{path}", timeout=5)
                    ok = response.status_code < 500
                    results.append({
                        "endpoint": f"{method} {path}",
                        "status": "ok" if ok else "error",
                        "status_code": response.status_code
                    })
                    if not ok:
                        all_ok = False
            except Exception as e:
                results.append({
                    "endpoint": f"{method} {path}",
                    "status": "error",
                    "error": str(e)
                })
                all_ok = False
        
        return {
            "status": "ok" if all_ok else "error",
            "endpoints": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Ejecuta verificación completa."""
        print("\n" + "="*70)
        print("  🔍 HEALTH CHECK AVANZADO")
        print("="*70 + "\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # API Health
        print("1️⃣ Verificando salud de la API...")
        api_health = self.check_api_health()
        results["checks"]["api_health"] = api_health
        status_icon = "✅" if api_health["status"] == "healthy" else "❌"
        print(f"   {status_icon} API: {api_health['status']}")
        if "response_time" in api_health:
            print(f"      Tiempo de respuesta: {api_health['response_time']:.0f}ms")
        print()
        
        print("2️⃣ Verificando almacenamiento...")
        db_check = self.check_database_connectivity()
        results["checks"]["database"] = db_check
        print(f"   ✅ Almacenamiento: {db_check['status']}")
        print()
        
        print("3️⃣ Verificando espacio en disco...")
        disk_check = self.check_disk_space()
        results["checks"]["disk_space"] = disk_check
        if disk_check["status"] == "ok":
            print(f"   ✅ Espacio disponible: {disk_check.get('free_percent', 0):.1f}%")
        else:
            print(f"   ⚠️ Espacio bajo: {disk_check.get('free_percent', 0):.1f}%")
        print()
        
        print("4️⃣ Verificando memoria...")
        memory_check = self.check_memory_usage()
        results["checks"]["memory"] = memory_check
        if "used_percent" in memory_check:
            print(f"   ✅ Memoria usada: {memory_check['used_percent']:.1f}%")
        print()
        
        print("5️⃣ Verificando endpoints...")
        endpoints_check = self.check_endpoints_availability()
        results["checks"]["endpoints"] = endpoints_check
        ok_count = sum(1 for e in endpoints_check["endpoints"] if e["status"] == "ok")
        print(f"   ✅ Endpoints disponibles: {ok_count}/{len(endpoints_check['endpoints'])}")
        print()
        
        all_healthy = all(
            check.get("status") in ["ok", "healthy"]
            for check in results["checks"].values()
        )
        
        results["overall_status"] = "healthy" if all_healthy else "unhealthy"
        
        print("="*70)
        print(f"  Estado General: {'✅ SALUDABLE' if all_healthy else '❌ PROBLEMAS DETECTADOS'}")
        print("="*70 + "\n")
        
        return results

if __name__ == "__main__":
    checker = AdvancedHealthChecker()
    results = checker.run_comprehensive_check()
    
    import json
    with open("health_check_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ Resultados guardados en health_check_results.json")
































