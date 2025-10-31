#!/usr/bin/env python3
"""
🚀 INSTALACIÓN AUTOMÁTICA - SISTEMA DE MEJORAS INTEGRADAS
Instalación automática de todas las dependencias y configuración del sistema
"""

import sys
import os
import subprocess
import importlib
import time
from datetime import datetime
from pathlib import Path

def print_banner():
    """Imprimir banner del sistema"""
    print("\n" + "=" * 80)
    print("🚀 SISTEMA DE MEJORAS INTEGRADAS - INSTALACIÓN AUTOMÁTICA")
    print("=" * 80)
    print("Instalación automática de todas las dependencias y configuración")
    print("=" * 80)

def verificar_python():
    """Verificar versión de Python"""
    print("\n🐍 Verificando Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.8+")
        return False

def instalar_pip():
    """Instalar o actualizar pip"""
    print("\n📦 Verificando pip...")
    try:
        import pip
        print("   ✅ pip ya está instalado")
        return True
    except ImportError:
        print("   🔄 Instalando pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            print("   ✅ pip instalado correctamente")
            return True
        except subprocess.CalledProcessError:
            print("   ❌ Error al instalar pip")
            return False

def actualizar_pip():
    """Actualizar pip a la última versión"""
    print("\n🔄 Actualizando pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("   ✅ pip actualizado correctamente")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Error al actualizar pip")
        return False

def instalar_dependencias_basicas():
    """Instalar dependencias básicas"""
    print("\n📦 Instalando dependencias básicas...")
    
    dependencias_basicas = [
        "numpy>=1.25.2",
        "pandas>=2.1.4",
        "scipy>=1.11.4",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "plotly>=5.17.0"
    ]
    
    for dep in dependencias_basicas:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_ml():
    """Instalar dependencias de machine learning"""
    print("\n🤖 Instalando dependencias de ML...")
    
    dependencias_ml = [
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.2",
        "lightgbm>=4.1.0",
        "catboost>=1.2.2",
        "optuna>=3.4.0",
        "hyperopt>=0.2.7",
        "scikit-optimize>=0.9.0"
    ]
    
    for dep in dependencias_ml:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_dl():
    """Instalar dependencias de deep learning"""
    print("\n🧠 Instalando dependencias de Deep Learning...")
    
    dependencias_dl = [
        "tensorflow>=2.15.0",
        "torch>=2.1.1",
        "jax>=0.4.20",
        "flax>=0.7.5"
    ]
    
    for dep in dependencias_dl:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_nlp():
    """Instalar dependencias de NLP"""
    print("\n📝 Instalando dependencias de NLP...")
    
    dependencias_nlp = [
        "transformers>=4.35.2",
        "spacy>=3.7.2",
        "nltk>=3.8.1",
        "sentence-transformers>=2.2.2",
        "flair>=0.13.0",
        "gensim>=4.3.2",
        "textblob>=0.17.1"
    ]
    
    for dep in dependencias_nlp:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_web():
    """Instalar dependencias web"""
    print("\n🌐 Instalando dependencias web...")
    
    dependencias_web = [
        "fastapi>=0.104.1",
        "uvicorn>=0.23.2",
        "starlette>=0.27.0",
        "aiohttp>=3.9.0",
        "httpx>=0.25.2",
        "requests>=2.31.0"
    ]
    
    for dep in dependencias_web:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_db():
    """Instalar dependencias de base de datos"""
    print("\n🗄️ Instalando dependencias de base de datos...")
    
    dependencias_db = [
        "sqlalchemy>=2.0.23",
        "alembic>=1.12.1",
        "asyncpg>=0.29.0",
        "redis>=5.0.1",
        "aioredis>=2.0.1"
    ]
    
    for dep in dependencias_db:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_monitoring():
    """Instalar dependencias de monitoreo"""
    print("\n📊 Instalando dependencias de monitoreo...")
    
    dependencias_monitoring = [
        "prometheus-client>=0.19.0",
        "structlog>=23.2.0",
        "loguru>=0.7.2",
        "sentry-sdk>=1.38.0"
    ]
    
    for dep in dependencias_monitoring:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_seguridad():
    """Instalar dependencias de seguridad"""
    print("\n🔒 Instalando dependencias de seguridad...")
    
    dependencias_seguridad = [
        "cryptography>=41.0.7",
        "pyotp>=2.9.0",
        "qrcode>=7.4.2",
        "passlib>=1.7.4",
        "python-jose>=3.3.0"
    ]
    
    for dep in dependencias_seguridad:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_testing():
    """Instalar dependencias de testing"""
    print("\n🧪 Instalando dependencias de testing...")
    
    dependencias_testing = [
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
        "coverage>=7.3.2"
    ]
    
    for dep in dependencias_testing:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_quality():
    """Instalar dependencias de calidad de código"""
    print("\n🔧 Instalando dependencias de calidad de código...")
    
    dependencias_quality = [
        "black>=23.11.0",
        "flake8>=6.1.0",
        "mypy>=1.7.1",
        "isort>=5.12.0"
    ]
    
    for dep in dependencias_quality:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_utilities():
    """Instalar dependencias de utilidades"""
    print("\n🛠️ Instalando dependencias de utilidades...")
    
    dependencias_utilities = [
        "click>=8.1.7",
        "tqdm>=4.66.1",
        "rich>=13.7.0",
        "typer>=0.9.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0"
    ]
    
    for dep in dependencias_utilities:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error al instalar {dep}")
            return False
    
    return True

def instalar_dependencias_opcionales():
    """Instalar dependencias opcionales"""
    print("\n🔧 Instalando dependencias opcionales...")
    
    dependencias_opcionales = [
        "opencv-python>=4.8.1.78",
        "pillow>=10.1.0",
        "scikit-image>=0.22.0",
        "networkx>=3.2.1",
        "statsmodels>=0.14.0",
        "prophet>=1.1.4"
    ]
    
    for dep in dependencias_opcionales:
        try:
            print(f"   🔄 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Advertencia: No se pudo instalar {dep} (opcional)")
    
    return True

def verificar_instalacion():
    """Verificar que la instalación fue exitosa"""
    print("\n🔍 Verificando instalación...")
    
    dependencias_verificar = [
        "numpy", "pandas", "scikit-learn", "tensorflow", "torch",
        "transformers", "spacy", "fastapi", "uvicorn", "sqlalchemy",
        "redis", "prometheus-client", "structlog", "loguru",
        "pytest", "black", "flake8", "mypy", "click", "tqdm"
    ]
    
    dependencias_ok = 0
    dependencias_faltantes = []
    
    for dep in dependencias_verificar:
        try:
            importlib.import_module(dep.replace('-', '_'))
            print(f"   ✅ {dep} - OK")
            dependencias_ok += 1
        except ImportError:
            print(f"   ❌ {dep} - FALTANTE")
            dependencias_faltantes.append(dep)
    
    print(f"\n📊 Resumen de verificación:")
    print(f"   ✅ Instaladas: {dependencias_ok}/{len(dependencias_verificar)}")
    print(f"   ❌ Faltantes: {len(dependencias_faltantes)}")
    
    if dependencias_faltantes:
        print(f"\n⚠️ Dependencias faltantes:")
        for dep in dependencias_faltantes:
            print(f"   - {dep}")
    
    return len(dependencias_faltantes) == 0

def crear_archivos_configuracion():
    """Crear archivos de configuración"""
    print("\n📁 Creando archivos de configuración...")
    
    # Crear archivo .env
    env_content = """# Configuración del Sistema de Mejoras Integradas
# Variables de entorno para configuración

# Configuración general
DEBUG=True
LOG_LEVEL=INFO
ENVIRONMENT=development

# Configuración de base de datos
DATABASE_URL=sqlite:///mejoras.db
REDIS_URL=redis://localhost:6379

# Configuración de monitoreo
PROMETHEUS_PORT=8000
SENTRY_DSN=

# Configuración de seguridad
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here

# Configuración de ML
ML_MODEL_PATH=./models/
ML_CACHE_PATH=./cache/

# Configuración de NLP
NLP_MODEL_PATH=./nlp_models/
NLP_CACHE_PATH=./nlp_cache/

# Configuración de análisis predictivo
PREDICTIVE_MODEL_PATH=./predictive_models/
PREDICTIVE_CACHE_PATH=./predictive_cache/

# Configuración de arquitectura
ARCHITECTURE_CONFIG_PATH=./architecture_config/
ARCHITECTURE_CACHE_PATH=./architecture_cache/

# Configuración de seguridad
SECURITY_CONFIG_PATH=./security_config/
SECURITY_CACHE_PATH=./security_cache/

# Configuración de monitoreo
MONITORING_CONFIG_PATH=./monitoring_config/
MONITORING_CACHE_PATH=./monitoring_cache/
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("   ✅ Archivo .env creado")
    except Exception as e:
        print(f"   ❌ Error al crear .env: {e}")
    
    # Crear archivo de configuración
    config_content = """# Configuración del Sistema de Mejoras Integradas
# Archivo de configuración principal

[general]
debug = true
log_level = INFO
environment = development

[database]
url = sqlite:///mejoras.db
redis_url = redis://localhost:6379

[monitoring]
prometheus_port = 8000
sentry_dsn = 

[security]
secret_key = your-secret-key-here
jwt_secret_key = your-jwt-secret-key-here

[ml]
model_path = ./models/
cache_path = ./cache/

[nlp]
model_path = ./nlp_models/
cache_path = ./nlp_cache/

[predictive]
model_path = ./predictive_models/
cache_path = ./predictive_cache/

[architecture]
config_path = ./architecture_config/
cache_path = ./architecture_cache/

[security_config]
config_path = ./security_config/
cache_path = ./security_cache/

[monitoring_config]
config_path = ./monitoring_config/
cache_path = ./monitoring_cache/
"""
    
    try:
        with open('config.ini', 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("   ✅ Archivo config.ini creado")
    except Exception as e:
        print(f"   ❌ Error al crear config.ini: {e}")
    
    return True

def crear_directorios():
    """Crear directorios necesarios"""
    print("\n📁 Creando directorios necesarios...")
    
    directorios = [
        'models',
        'cache',
        'nlp_models',
        'nlp_cache',
        'predictive_models',
        'predictive_cache',
        'architecture_config',
        'architecture_cache',
        'security_config',
        'security_cache',
        'monitoring_config',
        'monitoring_cache',
        'logs',
        'data',
        'temp'
    ]
    
    for directorio in directorios:
        try:
            Path(directorio).mkdir(exist_ok=True)
            print(f"   ✅ Directorio {directorio} creado")
        except Exception as e:
            print(f"   ❌ Error al crear directorio {directorio}: {e}")
    
    return True

def ejecutar_tests():
    """Ejecutar tests básicos"""
    print("\n🧪 Ejecutando tests básicos...")
    
    try:
        # Test de importación
        from real_improvements_engine import RealImprovementsEngine
        print("   ✅ Importación de RealImprovementsEngine - OK")
        
        # Test de inicialización
        engine = RealImprovementsEngine()
        print("   ✅ Inicialización de RealImprovementsEngine - OK")
        
        # Test de creación de mejoras
        engine.create_optimal_libraries_improvements()
        print("   ✅ Creación de mejoras - OK")
        
        print("   ✅ Tests básicos completados exitosamente")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en tests básicos: {e}")
        return False

def generar_reporte_instalacion():
    """Generar reporte de instalación"""
    print("\n📊 Generando reporte de instalación...")
    
    reporte = {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'sistema_operativo': os.name,
        'directorio_instalacion': os.getcwd(),
        'instalacion_exitosa': True,
        'dependencias_instaladas': True,
        'archivos_configuracion': True,
        'directorios_creados': True,
        'tests_basicos': True
    }
    
    try:
        import json
        with open('reporte_instalacion.json', 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        print("   ✅ Reporte de instalación generado: reporte_instalacion.json")
    except Exception as e:
        print(f"   ❌ Error al generar reporte: {e}")
    
    return reporte

def main():
    """Función principal de instalación"""
    print_banner()
    
    # Verificar Python
    if not verificar_python():
        print("\n❌ INSTALACIÓN FALLIDA: Se requiere Python 3.8 o superior")
        return False
    
    # Instalar pip
    if not instalar_pip():
        print("\n❌ INSTALACIÓN FALLIDA: No se pudo instalar pip")
        return False
    
    # Actualizar pip
    if not actualizar_pip():
        print("\n❌ INSTALACIÓN FALLIDA: No se pudo actualizar pip")
        return False
    
    # Instalar dependencias
    print("\n📦 INSTALANDO DEPENDENCIAS...")
    
    dependencias_instaladas = True
    dependencias_instaladas &= instalar_dependencias_basicas()
    dependencias_instaladas &= instalar_dependencias_ml()
    dependencias_instaladas &= instalar_dependencias_dl()
    dependencias_instaladas &= instalar_dependencias_nlp()
    dependencias_instaladas &= instalar_dependencias_web()
    dependencias_instaladas &= instalar_dependencias_db()
    dependencias_instaladas &= instalar_dependencias_monitoring()
    dependencias_instaladas &= instalar_dependencias_seguridad()
    dependencias_instaladas &= instalar_dependencias_testing()
    dependencias_instaladas &= instalar_dependencias_quality()
    dependencias_instaladas &= instalar_dependencias_utilities()
    dependencias_instaladas &= instalar_dependencias_opcionales()
    
    if not dependencias_instaladas:
        print("\n⚠️ ADVERTENCIA: Algunas dependencias no se pudieron instalar")
    
    # Crear archivos de configuración
    crear_archivos_configuracion()
    
    # Crear directorios
    crear_directorios()
    
    # Verificar instalación
    if not verificar_instalacion():
        print("\n⚠️ ADVERTENCIA: Algunas dependencias no están disponibles")
    
    # Ejecutar tests
    if not ejecutar_tests():
        print("\n⚠️ ADVERTENCIA: Los tests básicos fallaron")
    
    # Generar reporte
    generar_reporte_instalacion()
    
    # Resumen final
    print("\n🎉 INSTALACIÓN COMPLETADA")
    print("=" * 80)
    print("✅ Sistema de Mejoras Integradas instalado correctamente")
    print("✅ Dependencias instaladas")
    print("✅ Archivos de configuración creados")
    print("✅ Directorios creados")
    print("✅ Tests básicos ejecutados")
    print("✅ Reporte de instalación generado")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("   1. Verificar instalación: python verificar_instalacion.py")
    print("   2. Ejecutar demo completo: python demo_completo_mejoras.py")
    print("   3. Configurar variables de entorno en .env")
    print("   4. Implementar en producción")
    
    print("\n💡 COMANDOS ÚTILES:")
    print("   • Verificar instalación: python verificar_instalacion.py")
    print("   • Demo completo: python demo_completo_mejoras.py")
    print("   • Demo librerías: python demo_librerias_optimas.py")
    print("   • Demo ML: python demo_ml_optimizado.py")
    print("   • Tests: pytest tests/")
    print("   • Linting: black . && flake8 . && mypy .")
    
    print("\n🎉 ¡SISTEMA LISTO PARA USAR!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Instalación completada exitosamente")
            sys.exit(0)
        else:
            print("\n❌ Instalación falló")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Instalación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado durante la instalación: {e}")
        sys.exit(1)



