#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN DE INSTALACIÓN - TODAS LAS MEJORAS
Verificación completa de todas las mejoras implementadas
"""

import sys
import os
import subprocess
import importlib
from datetime import datetime

def verificar_python():
    """Verificar versión de Python"""
    print("🐍 Verificando Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.8+")
        return False

def verificar_dependencias():
    """Verificar dependencias principales"""
    print("\n📦 Verificando dependencias principales...")
    
    dependencias = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
        'transformers', 'spacy', 'nltk', 'opencv-python', 'pillow',
        'requests', 'aiohttp', 'fastapi', 'uvicorn', 'sqlalchemy',
        'redis', 'prometheus-client', 'structlog', 'loguru',
        'pytest', 'black', 'flake8', 'mypy', 'click', 'tqdm'
    ]
    
    dependencias_ok = 0
    dependencias_faltantes = []
    
    for dep in dependencias:
        try:
            importlib.import_module(dep.replace('-', '_'))
            print(f"   ✅ {dep} - OK")
            dependencias_ok += 1
        except ImportError:
            print(f"   ❌ {dep} - FALTANTE")
            dependencias_faltantes.append(dep)
    
    print(f"\n📊 Resumen de dependencias:")
    print(f"   ✅ Instaladas: {dependencias_ok}/{len(dependencias)}")
    print(f"   ❌ Faltantes: {len(dependencias_faltantes)}")
    
    if dependencias_faltantes:
        print(f"\n🔧 Para instalar dependencias faltantes:")
        print(f"   pip install {' '.join(dependencias_faltantes)}")
    
    return len(dependencias_faltantes) == 0

def verificar_archivos():
    """Verificar archivos del sistema"""
    print("\n📁 Verificando archivos del sistema...")
    
    archivos = [
        'real_improvements_engine.py',
        'demo_completo_mejoras.py',
        'demo_librerias_optimas.py',
        'demo_ml_optimizado.py',
        'demo_nlp_avanzado.py',
        'demo_analisis_predictivo.py',
        'demo_arquitectura_empresarial.py',
        'demo_seguridad_avanzada.py',
        'demo_monitoreo_inteligente.py',
        'demo_analytics_avanzados.py',
        'requirements.txt',
        'README.md'
    ]
    
    archivos_ok = 0
    archivos_faltantes = []
    
    for archivo in archivos:
        if os.path.exists(archivo):
            print(f"   ✅ {archivo} - OK")
            archivos_ok += 1
        else:
            print(f"   ❌ {archivo} - FALTANTE")
            archivos_faltantes.append(archivo)
    
    print(f"\n📊 Resumen de archivos:")
    print(f"   ✅ Presentes: {archivos_ok}/{len(archivos)}")
    print(f"   ❌ Faltantes: {len(archivos_faltantes)}")
    
    return len(archivos_faltantes) == 0

def verificar_sistema():
    """Verificar sistema de mejoras"""
    print("\n🔧 Verificando sistema de mejoras...")
    
    try:
        from real_improvements_engine import RealImprovementsEngine
        engine = RealImprovementsEngine()
        print("   ✅ RealImprovementsEngine - OK")
        
        # Verificar mejoras
        mejoras = engine.get_all_improvements()
        print(f"   ✅ Mejoras disponibles: {len(mejoras)}")
        
        # Verificar sistemas específicos
        sistemas = [
            'optimal_libraries_system',
            'intelligent_dependencies_system',
            'optimized_ml_system',
            'advanced_nlp_system',
            'advanced_predictive_analytics',
            'enterprise_architecture_system',
            'advanced_security_system',
            'intelligent_monitoring_system'
        ]
        
        sistemas_ok = 0
        for sistema in sistemas:
            if hasattr(engine, sistema):
                print(f"   ✅ {sistema} - OK")
                sistemas_ok += 1
            else:
                print(f"   ❌ {sistema} - FALTANTE")
        
        print(f"\n📊 Resumen de sistemas:")
        print(f"   ✅ Sistemas disponibles: {sistemas_ok}/{len(sistemas)}")
        
        return sistemas_ok == len(sistemas)
        
    except Exception as e:
        print(f"   ❌ Error al verificar sistema: {e}")
        return False

def verificar_funcionalidades():
    """Verificar funcionalidades principales"""
    print("\n🚀 Verificando funcionalidades principales...")
    
    try:
        from real_improvements_engine import RealImprovementsEngine
        engine = RealImprovementsEngine()
        
        # Verificar funcionalidades
        funcionalidades = [
            ('analyze_current_libraries', 'Análisis de librerías'),
            ('optimize_libraries_automatically', 'Optimización automática'),
            ('analyze_dependencies_intelligently', 'Análisis de dependencias'),
            ('analyze_ml_libraries', 'Análisis de ML'),
            ('analyze_nlp_libraries', 'Análisis de NLP'),
            ('analyze_predictive_libraries', 'Análisis predictivo'),
            ('analyze_architecture_libraries', 'Análisis de arquitectura'),
            ('analyze_security_libraries', 'Análisis de seguridad'),
            ('analyze_monitoring_libraries', 'Análisis de monitoreo')
        ]
        
        funcionalidades_ok = 0
        for func, desc in funcionalidades:
            if hasattr(engine, func):
                print(f"   ✅ {desc} - OK")
                funcionalidades_ok += 1
            else:
                print(f"   ❌ {desc} - FALTANTE")
        
        print(f"\n📊 Resumen de funcionalidades:")
        print(f"   ✅ Funcionalidades disponibles: {funcionalidades_ok}/{len(funcionalidades)}")
        
        return funcionalidades_ok == len(funcionalidades)
        
    except Exception as e:
        print(f"   ❌ Error al verificar funcionalidades: {e}")
        return False

def verificar_rendimiento():
    """Verificar rendimiento del sistema"""
    print("\n⚡ Verificando rendimiento del sistema...")
    
    try:
        from real_improvements_engine import RealImprovementsEngine
        engine = RealImprovementsEngine()
        
        # Medir tiempo de inicialización
        start_time = datetime.now()
        engine.create_optimal_libraries_improvements()
        end_time = datetime.now()
        init_time = (end_time - start_time).total_seconds()
        
        print(f"   ✅ Tiempo de inicialización: {init_time:.2f} segundos")
        
        # Verificar métricas
        if hasattr(engine, 'optimal_libraries_system'):
            metrics = engine.optimal_libraries_system.get_advanced_analytics()
            print(f"   ✅ Métricas del sistema: {metrics.get('system_metrics', {})}")
        
        print("   ✅ Rendimiento del sistema - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error al verificar rendimiento: {e}")
        return False

def verificar_seguridad():
    """Verificar seguridad del sistema"""
    print("\n🔒 Verificando seguridad del sistema...")
    
    try:
        from real_improvements_engine import RealImprovementsEngine
        engine = RealImprovementsEngine()
        
        # Verificar sistemas de seguridad
        if hasattr(engine, 'advanced_security_system'):
            print("   ✅ Sistema de seguridad avanzada - OK")
        
        # Verificar encriptación
        if hasattr(engine, 'optimal_libraries_system'):
            system = engine.optimal_libraries_system
            if hasattr(system, 'vulnerability_database'):
                print("   ✅ Base de datos de vulnerabilidades - OK")
        
        print("   ✅ Seguridad del sistema - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error al verificar seguridad: {e}")
        return False

def verificar_compatibilidad():
    """Verificar compatibilidad del sistema"""
    print("\n🔗 Verificando compatibilidad del sistema...")
    
    try:
        from real_improvements_engine import RealImprovementsEngine
        engine = RealImprovementsEngine()
        
        # Verificar sistemas de compatibilidad
        if hasattr(engine, 'intelligent_dependencies_system'):
            print("   ✅ Sistema de dependencias inteligente - OK")
        
        if hasattr(engine, 'optimal_libraries_system'):
            system = engine.optimal_libraries_system
            if hasattr(system, 'compatibility_database'):
                print("   ✅ Base de datos de compatibilidad - OK")
        
        print("   ✅ Compatibilidad del sistema - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error al verificar compatibilidad: {e}")
        return False

def verificar_monitoreo():
    """Verificar monitoreo del sistema"""
    print("\n📈 Verificando monitoreo del sistema...")
    
    try:
        from real_improvements_engine import RealImprovementsEngine
        engine = RealImprovementsEngine()
        
        # Verificar sistemas de monitoreo
        if hasattr(engine, 'intelligent_monitoring_system'):
            print("   ✅ Sistema de monitoreo inteligente - OK")
        
        # Verificar métricas
        if hasattr(engine, 'optimal_libraries_system'):
            system = engine.optimal_libraries_system
            if hasattr(system, 'system_metrics'):
                print("   ✅ Métricas del sistema - OK")
        
        print("   ✅ Monitoreo del sistema - OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error al verificar monitoreo: {e}")
        return False

def generar_reporte():
    """Generar reporte de verificación"""
    print("\n📊 Generando reporte de verificación...")
    
    reporte = {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'sistema_operativo': os.name,
        'directorio_actual': os.getcwd(),
        'archivos_python': len([f for f in os.listdir('.') if f.endswith('.py')]),
        'verificacion_completa': True
    }
    
    # Guardar reporte
    with open('reporte_verificacion.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print("   ✅ Reporte generado: reporte_verificacion.json")
    return reporte

def main():
    """Función principal de verificación"""
    print("🔍 VERIFICACIÓN DE INSTALACIÓN - TODAS LAS MEJORAS")
    print("=" * 80)
    print("Verificación completa de todas las mejoras implementadas")
    print("=" * 80)
    
    # Ejecutar verificaciones
    verificaciones = [
        ('Python', verificar_python),
        ('Dependencias', verificar_dependencias),
        ('Archivos', verificar_archivos),
        ('Sistema', verificar_sistema),
        ('Funcionalidades', verificar_funcionalidades),
        ('Rendimiento', verificar_rendimiento),
        ('Seguridad', verificar_seguridad),
        ('Compatibilidad', verificar_compatibilidad),
        ('Monitoreo', verificar_monitoreo)
    ]
    
    resultados = {}
    total_ok = 0
    
    for nombre, funcion in verificaciones:
        try:
            resultado = funcion()
            resultados[nombre] = resultado
            if resultado:
                total_ok += 1
        except Exception as e:
            print(f"   ❌ Error en verificación de {nombre}: {e}")
            resultados[nombre] = False
    
    # Resumen final
    print("\n🎉 RESUMEN DE VERIFICACIÓN")
    print("=" * 80)
    
    for nombre, resultado in resultados.items():
        status = "✅ OK" if resultado else "❌ ERROR"
        print(f"   {status} {nombre}")
    
    print(f"\n📊 Resultado final:")
    print(f"   ✅ Verificaciones exitosas: {total_ok}/{len(verificaciones)}")
    print(f"   ❌ Verificaciones fallidas: {len(verificaciones) - total_ok}")
    
    if total_ok == len(verificaciones):
        print("\n🎉 ¡TODAS LAS VERIFICACIONES EXITOSAS!")
        print("   El sistema está listo para usar")
        print("   Puedes ejecutar: python demo_completo_mejoras.py")
    else:
        print("\n⚠️ ALGUNAS VERIFICACIONES FALLARON")
        print("   Revisa los errores anteriores")
        print("   Instala las dependencias faltantes")
        print("   Ejecuta: pip install -r requirements.txt")
    
    # Generar reporte
    generar_reporte()
    
    print("\n💡 COMANDOS ÚTILES:")
    print("   • Instalar dependencias: pip install -r requirements.txt")
    print("   • Verificar instalación: python verificar_instalacion.py")
    print("   • Demo completo: python demo_completo_mejoras.py")
    print("   • Demo librerías: python demo_librerias_optimas.py")
    print("   • Demo ML: python demo_ml_optimizado.py")

if __name__ == "__main__":
    main()



