from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import json
import time
import asyncio
import requests
from pathlib import Path
from typing import List, Dict
from features.seo.service import SEOService
from features.seo.models import SEOScrapeRequest
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Script de prueba optimizado para el servicio de análisis SEO con LangChain
Demuestra todas las nuevas funcionalidades y optimizaciones
"""


# Agregar el directorio padre al path para importar los módulos
sys.path.append(str(Path(__file__).parent.parent.parent))


class SEOTester:
    """Clase para probar el servicio SEO optimizado"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SEO-Tester/1.0'
        })

    def test_basic_analysis(self, url: str = "https://www.google.com") -> Dict:
        """Prueba análisis básico de una URL"""
        print(f"🔍 Probando análisis básico de: {url}")
        print("-" * 50)
        
        try:
            # Usar el endpoint GET simple
            response = self.session.get(f"{self.base_url}/seo/analyze", params={"url": url})
            
            if response.status_code == 200:
                data = response.json()
                self._print_analysis_results(data)
                return data
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error en análisis básico: {str(e)}")
            return None

    def test_advanced_analysis(self, url: str = "https://www.github.com") -> Dict:
        """Prueba análisis avanzado con Selenium"""
        print(f"🚀 Probando análisis avanzado de: {url}")
        print("-" * 50)
        
        try:
            # Usar el endpoint POST con opciones
            payload = {
                "url": url,
                "options": {
                    "use_selenium": True
                }
            }
            
            response = self.session.post(f"{self.base_url}/seo/scrape", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                self._print_analysis_results(data)
                return data
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error en análisis avanzado: {str(e)}")
            return None

    def test_batch_analysis(self, urls: List[str]) -> List[Dict]:
        """Prueba análisis en lote"""
        print(f"📦 Probando análisis en lote de {len(urls)} URLs")
        print("-" * 50)
        
        try:
            response = self.session.post(f"{self.base_url}/seo/batch", json=urls)
            
            if response.status_code == 200:
                results = response.json()
                
                print(f"✅ Análisis completado para {len(results)} URLs")
                
                for i, result in enumerate(results):
                    if result.get('success'):
                        data = result.get('data', {})
                        print(f"  {i+1}. {urls[i]}: {data.get('seo_score', 'N/A')}/100")
                    else:
                        print(f"  {i+1}. {urls[i]}: ❌ {result.get('error', 'Error desconocido')}")
                
                return results
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error en análisis en lote: {str(e)}")
            return []

    def test_url_comparison(self, url1: str, url2: str) -> Dict:
        """Prueba comparación de URLs"""
        print(f"⚖️  Comparando SEO de:")
        print(f"   URL1: {url1}")
        print(f"   URL2: {url2}")
        print("-" * 50)
        
        try:
            response = self.session.get(
                f"{self.base_url}/seo/compare",
                params={"url1": url1, "url2": url2}
            )
            
            if response.status_code == 200:
                comparison = response.json()
                self._print_comparison_results(comparison)
                return comparison
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error en comparación: {str(e)}")
            return None

    def test_cache_functionality(self) -> Dict:
        """Prueba funcionalidades del cache"""
        print("💾 Probando funcionalidades del cache")
        print("-" * 50)
        
        try:
            # Obtener estadísticas del cache
            response = self.session.get(f"{self.base_url}/seo/cache/stats")
            
            if response.status_code == 200:
                stats = response.json()
                print(f"📊 Estadísticas del cache:")
                print(f"   Tamaño: {stats.get('cache_size', 0)} elementos")
                print(f"   Uso de memoria: {stats.get('memory_usage_mb', 0):.2f} MB")
                print(f"   Entrada más antigua: {stats.get('oldest_entry', 'N/A')}")
                
                return stats
            else:
                print(f"❌ Error obteniendo estadísticas: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error en cache: {str(e)}")
            return None

    def test_health_check(self) -> Dict:
        """Prueba el health check del servicio"""
        print("🏥 Probando health check del servicio")
        print("-" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/seo/health")
            
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Estado: {health.get('status', 'unknown')}")
                print(f"📋 Componentes:")
                
                components = health.get('components', {})
                for component, status in components.items():
                    status_icon = "✅" if status else "❌"
                    print(f"   {component}: {status_icon}")
                
                print(f"🚀 Funcionalidades: {len(health.get('features', []))}")
                print(f"💾 Cache: {health.get('cache_size', 0)} elementos")
                
                return health
            else:
                print(f"❌ Error en health check: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error en health check: {str(e)}")
            return None

    def test_performance(self, url: str = "https://www.stackoverflow.com") -> Dict:
        """Prueba el rendimiento del servicio"""
        print(f"⚡ Probando rendimiento con: {url}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # Primera llamada (sin cache)
            response1 = self.session.post(f"{self.base_url}/seo/scrape", json={"url": url})
            first_call_time = time.time() - start_time
            
            # Segunda llamada (con cache)
            start_time = time.time()
            response2 = self.session.post(f"{self.base_url}/seo/scrape", json={"url": url})
            second_call_time = time.time() - start_time
            
            print(f"⏱️  Primera llamada: {first_call_time:.2f}s")
            print(f"⏱️  Segunda llamada (cache): {second_call_time:.2f}s")
            print(f"🚀 Mejora de velocidad: {((first_call_time - second_call_time) / first_call_time * 100):.1f}%")
            
            return {
                "first_call_time": first_call_time,
                "second_call_time": second_call_time,
                "improvement": ((first_call_time - second_call_time) / first_call_time * 100)
            }
            
        except Exception as e:
            print(f"❌ Error en prueba de rendimiento: {str(e)}")
            return None

    def _print_analysis_results(self, data: Dict):
        """Imprime los resultados del análisis de forma optimizada"""
        if not data.get('success'):
            print(f"❌ Error: {data.get('error', 'Error desconocido')}")
            return
        
        seo_data = data.get('data', {})
        
        print(f"✅ Análisis completado exitosamente!")
        print(f"📊 Puntuación SEO: {seo_data.get('seo_score', 'N/A')}/100")
        print(f"⏱️  Tiempo de carga: {seo_data.get('load_time', 'N/A'):.2f}s")
        print(f"📱 Compatible móvil: {'✅' if seo_data.get('mobile_friendly') else '❌'}")
        print(f"🚀 Velocidad: {seo_data.get('page_speed', 'N/A')}")
        print(f"📝 Título: {seo_data.get('title', 'N/A')[:60]}...")
        print(f"📄 Contenido: {seo_data.get('content_length', 0)} caracteres")
        
        # Recomendaciones
        recommendations = seo_data.get('recommendations', [])
        if recommendations:
            print(f"💡 Recomendaciones ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")

    def _print_comparison_results(self, comparison: Dict):
        """Imprime los resultados de la comparación"""
        url1_data = comparison.get('url1', {})
        url2_data = comparison.get('url2', {})
        comp_data = comparison.get('comparison', {})
        
        print(f"📊 Resultados de la comparación:")
        print(f"   URL1 ({url1_data.get('url')}): {url1_data.get('seo_score', 'N/A')}/100")
        print(f"   URL2 ({url2_data.get('url')}): {url2_data.get('seo_score', 'N/A')}/100")
        print(f"   Diferencia SEO: {comp_data.get('seo_score_difference', 'N/A')}")
        print(f"   Ganador: {comp_data.get('winner', 'N/A')}")

def main():
    """Función principal para ejecutar todas las pruebas"""
    print("🚀 Iniciando pruebas del servicio SEO optimizado")
    print("=" * 60)
    
    # Configurar API key de OpenAI (opcional)
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY no configurada. Algunas funcionalidades pueden ser limitadas.")
    
    # Crear tester
    tester = SEOTester()
    
    # Ejecutar pruebas
    tests = [
        ("Health Check", tester.test_health_check),
        ("Análisis Básico", lambda: tester.test_basic_analysis()),
        ("Análisis Avanzado", lambda: tester.test_advanced_analysis()),
        ("Análisis en Lote", lambda: tester.test_batch_analysis([
            "https://www.google.com",
            "https://www.github.com",
            "https://www.stackoverflow.com"
        ])),
        ("Comparación de URLs", lambda: tester.test_url_comparison(
            "https://www.google.com",
            "https://www.github.com"
        )),
        ("Funcionalidades del Cache", tester.test_cache_functionality),
        ("Prueba de Rendimiento", lambda: tester.test_performance())
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = "✅ Exitoso" if result else "❌ Falló"
        except Exception as e:
            print(f"❌ Error en {test_name}: {str(e)}")
            results[test_name] = "❌ Error"
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    for test_name, status in results.items():
        print(f"{test_name}: {status}")
    
    successful_tests = sum(1 for status in results.values() if "✅" in status)
    total_tests = len(results)
    
    print(f"\n🎯 Resultado: {successful_tests}/{total_tests} pruebas exitosas")
    
    if successful_tests == total_tests:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar logs para más detalles.")

match __name__:
    case "__main__":
    main() 