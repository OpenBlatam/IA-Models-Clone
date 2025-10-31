"""
Script de configuración para AI Document Processor
=================================================
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Crea directorios necesarios"""
    directories = [
        "logs",
        "temp", 
        "models",
        "data",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def install_dependencies():
    """Instala dependencias de Python"""
    try:
        print("📦 Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False
    return True

def setup_environment():
    """Configura el archivo de entorno"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creando archivo .env...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Archivo .env creado desde env.example")
        print("⚠️  Recuerda configurar OPENAI_API_KEY en .env")
    elif env_file.exists():
        print("✅ Archivo .env ya existe")
    else:
        print("⚠️  No se encontró env.example")

def check_system_requirements():
    """Verifica requisitos del sistema"""
    print("🔍 Verificando requisitos del sistema...")
    
    # Verificar Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Verificar pip
    try:
        import pip
        print("✅ pip disponible")
    except ImportError:
        print("❌ pip no está disponible")
        return False
    
    return True

def test_installation():
    """Prueba la instalación"""
    print("🧪 Probando instalación...")
    
    try:
        # Importar módulos principales
        from services.document_processor import DocumentProcessor
        from models.document_models import ProfessionalFormat
        print("✅ Módulos principales importados correctamente")
        
        # Probar inicialización básica
        import asyncio
        async def test_init():
            processor = DocumentProcessor()
            await processor.initialize()
            return True
        
        result = asyncio.run(test_init())
        if result:
            print("✅ Inicialización del procesador exitosa")
        else:
            print("❌ Error en inicialización del procesador")
            return False
            
    except Exception as e:
        print(f"❌ Error en prueba de instalación: {e}")
        return False
    
    return True

def main():
    """Función principal de configuración"""
    print("🚀 Configurando AI Document Processor...")
    print("=" * 50)
    
    # Verificar requisitos
    if not check_system_requirements():
        print("❌ Requisitos del sistema no cumplidos")
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error en instalación de dependencias")
        sys.exit(1)
    
    # Configurar entorno
    setup_environment()
    
    # Probar instalación
    if not test_installation():
        print("❌ Error en prueba de instalación")
        sys.exit(1)
    
    print("\n🎉 Configuración completada exitosamente!")
    print("\n📋 Próximos pasos:")
    print("1. Configurar OPENAI_API_KEY en .env (opcional pero recomendado)")
    print("2. Ejecutar: python main.py")
    print("3. Visitar: http://localhost:8001/docs")
    print("4. Probar con: python example_usage.py")

if __name__ == "__main__":
    main()


