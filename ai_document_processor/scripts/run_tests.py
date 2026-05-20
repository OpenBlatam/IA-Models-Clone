"""
Script para ejecutar pruebas del AI Document Processor
=====================================================
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def run_pytest():
    """Ejecuta pruebas con pytest"""
    print("🧪 Ejecutando pruebas con pytest...")
    
    try:
        # Ejecutar pytest con verbose y coverage
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Todas las pruebas pasaron")
            return True
        else:
            print("❌ Algunas pruebas fallaron")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando pytest: {e}")
        return False

def run_manual_tests():
    """Ejecuta pruebas manuales"""
    print("🔧 Ejecutando pruebas manuales...")
    
    try:
        # Importar y ejecutar ejemplo de uso
        sys.path.append(str(Path.cwd()))
        from example_usage import main, test_different_formats
        
        print("Ejecutando ejemplo principal...")
        asyncio.run(main())
        
        print("Ejecutando pruebas de formatos...")
        asyncio.run(test_different_formats())
        
        print("✅ Pruebas manuales completadas")
        return True
        
    except Exception as e:
        print(f"❌ Error en pruebas manuales: {e}")
        return False

def check_code_quality():
    """Verifica calidad del código"""
    print("🔍 Verificando calidad del código...")
    
    try:
        # Verificar con flake8
        cmd = [sys.executable, "-m", "flake8", ".", "--max-line-length=100"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Código cumple estándares de calidad")
        else:
            print("⚠️  Problemas de calidad encontrados:")
            print(result.stdout)
        
        # Verificar con black
        cmd = [sys.executable, "-m", "black", "--check", "."]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Código está formateado correctamente")
        else:
            print("⚠️  Problemas de formato encontrados:")
            print(result.stdout)
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando calidad: {e}")
        return False

def run_performance_tests():
    """Ejecuta pruebas de rendimiento básicas"""
    print("⚡ Ejecutando pruebas de rendimiento...")
    
    try:
        import time
        from services.document_processor import DocumentProcessor
        from models.document_models import DocumentProcessingRequest, ProfessionalFormat
        
        async def performance_test():
            processor = DocumentProcessor()
            await processor.initialize()
            
            # Crear archivo de prueba
            test_content = "Este es un documento de prueba para medir rendimiento. " * 100
            
            with open("temp_performance_test.md", "w") as f:
                f.write(test_content)
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            request = DocumentProcessingRequest(
                filename="performance_test.md",
                target_format=ProfessionalFormat.CONSULTANCY,
                language="es",
                include_analysis=True
            )
            
            result = await processor.process_document("temp_performance_test.md", request)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Limpiar archivo temporal
            os.unlink("temp_performance_test.md")
            
            print(f"⏱️  Tiempo de procesamiento: {processing_time:.2f} segundos")
            
            if processing_time < 10:  # Menos de 10 segundos
                print("✅ Rendimiento aceptable")
                return True
            else:
                print("⚠️  Rendimiento lento")
                return False
        
        result = asyncio.run(performance_test())
        return result
        
    except Exception as e:
        print(f"❌ Error en pruebas de rendimiento: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 Ejecutando suite de pruebas del AI Document Processor")
    print("=" * 60)
    
    results = []
    
    # Ejecutar diferentes tipos de pruebas
    print("\n1. Pruebas unitarias con pytest")
    results.append(("Pytest", run_pytest()))
    
    print("\n2. Pruebas manuales")
    results.append(("Pruebas Manuales", run_manual_tests()))
    
    print("\n3. Verificación de calidad de código")
    results.append(("Calidad de Código", check_code_quality()))
    
    print("\n4. Pruebas de rendimiento")
    results.append(("Rendimiento", run_performance_tests()))
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        return 0
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar los errores arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


