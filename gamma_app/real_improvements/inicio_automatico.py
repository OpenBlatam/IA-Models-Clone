#!/usr/bin/env python3
"""
🚀 INICIO AUTOMÁTICO - SISTEMA DE MEJORAS INTEGRADAS
Script de inicio automático para el sistema de mejoras
"""

import sys
import os
import time
import subprocess
from datetime import datetime
from pathlib import Path

def print_banner():
    """Imprimir banner del sistema"""
    print("\n" + "=" * 80)
    print("🚀 SISTEMA DE MEJORAS INTEGRADAS - INICIO AUTOMÁTICO")
    print("=" * 80)
    print("Script de inicio automático para el sistema de mejoras")
    print("=" * 80)

def verificar_instalacion():
    """Verificar que el sistema esté instalado"""
    print("\n🔍 Verificando instalación...")
    
    archivos_requeridos = [
        'real_improvements_engine.py',
        'demo_completo_mejoras.py',
        'verificar_instalacion.py',
        'ejecutar_mejoras.py',
        'instalar_sistema.py',
        'configurar_sistema.py',
        'limpiar_sistema.py',
        'inicio_rapido.py',
        'requirements.txt',
        'README.md'
    ]
    
    archivos_ok = 0
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"   ✅ {archivo} - OK")
            archivos_ok += 1
        else:
            print(f"   ❌ {archivo} - FALTANTE")
    
    if archivos_ok == len(archivos_requeridos):
        print("   ✅ Sistema instalado correctamente")
        return True
    else:
        print("   ❌ Sistema no está completamente instalado")
        return False

def instalar_sistema_automaticamente():
    """Instalar sistema automáticamente"""
    print("\n📦 Instalando sistema automáticamente...")
    
    try:
        # Ejecutar instalación
        result = subprocess.run([sys.executable, 'instalar_sistema.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Sistema instalado correctamente")
            return True
        else:
            print(f"   ❌ Error en instalación: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al ejecutar instalación: {e}")
        return False

def configurar_sistema_automaticamente():
    """Configurar sistema automáticamente"""
    print("\n⚙️ Configurando sistema automáticamente...")
    
    try:
        # Ejecutar configuración
        result = subprocess.run([sys.executable, 'configurar_sistema.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Sistema configurado correctamente")
            return True
        else:
            print(f"   ❌ Error en configuración: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al ejecutar configuración: {e}")
        return False

def limpiar_sistema_automaticamente():
    """Limpiar sistema automáticamente"""
    print("\n🧹 Limpiando sistema automáticamente...")
    
    try:
        # Ejecutar limpieza
        result = subprocess.run([sys.executable, 'limpiar_sistema.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Sistema limpiado correctamente")
            return True
        else:
            print(f"   ❌ Error en limpieza: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al ejecutar limpieza: {e}")
        return False

def ejecutar_mejoras_automaticamente():
    """Ejecutar mejoras automáticamente"""
    print("\n🚀 Ejecutando mejoras automáticamente...")
    
    try:
        # Ejecutar mejoras
        result = subprocess.run([sys.executable, 'ejecutar_mejoras.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Mejoras ejecutadas correctamente")
            return True
        else:
            print(f"   ❌ Error en ejecución: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al ejecutar mejoras: {e}")
        return False

def ejecutar_demo_completo():
    """Ejecutar demo completo"""
    print("\n🎯 Ejecutando demo completo...")
    
    try:
        # Ejecutar demo completo
        result = subprocess.run([sys.executable, 'demo_completo_mejoras.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Demo completo ejecutado correctamente")
            return True
        else:
            print(f"   ❌ Error en demo: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al ejecutar demo: {e}")
        return False

def verificar_instalacion_automaticamente():
    """Verificar instalación automáticamente"""
    print("\n🔍 Verificando instalación automáticamente...")
    
    try:
        # Ejecutar verificación
        result = subprocess.run([sys.executable, 'verificar_instalacion.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Verificación completada correctamente")
            return True
        else:
            print(f"   ❌ Error en verificación: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al ejecutar verificación: {e}")
        return False

def ejecutar_secuencia_completa():
    """Ejecutar secuencia completa de inicio automático"""
    print("\n🔄 Ejecutando secuencia completa de inicio automático...")
    
    # Verificar instalación
    if not verificar_instalacion():
        print("\n📦 Sistema no está instalado, instalando automáticamente...")
        if not instalar_sistema_automaticamente():
            print("❌ Error en instalación automática")
            return False
    
    # Configurar sistema
    if not configurar_sistema_automaticamente():
        print("❌ Error en configuración automática")
        return False
    
    # Limpiar sistema
    if not limpiar_sistema_automaticamente():
        print("❌ Error en limpieza automática")
        return False
    
    # Verificar instalación
    if not verificar_instalacion_automaticamente():
        print("❌ Error en verificación automática")
        return False
    
    # Ejecutar mejoras
    if not ejecutar_mejoras_automaticamente():
        print("❌ Error en ejecución automática de mejoras")
        return False
    
    # Ejecutar demo completo
    if not ejecutar_demo_completo():
        print("❌ Error en demo automático")
        return False
    
    print("\n✅ Secuencia completa de inicio automático ejecutada exitosamente")
    return True

def ejecutar_modo_interactivo():
    """Ejecutar modo interactivo"""
    print("\n🎮 Modo interactivo activado")
    
    while True:
        print("\n📋 OPCIONES DE INICIO AUTOMÁTICO:")
        print("   1. 🔍 Verificar instalación")
        print("   2. 📦 Instalar sistema")
        print("   3. ⚙️ Configurar sistema")
        print("   4. 🧹 Limpiar sistema")
        print("   5. 🚀 Ejecutar mejoras")
        print("   6. 🎯 Demo completo")
        print("   7. 🔄 Secuencia completa")
        print("   8. ❓ Ayuda")
        print("   0. 🚪 Salir")
        
        try:
            opcion = input("\n🔢 Selecciona una opción (0-8): ").strip()
            
            if opcion == "1":
                verificar_instalacion_automaticamente()
            elif opcion == "2":
                instalar_sistema_automaticamente()
            elif opcion == "3":
                configurar_sistema_automaticamente()
            elif opcion == "4":
                limpiar_sistema_automaticamente()
            elif opcion == "5":
                ejecutar_mejoras_automaticamente()
            elif opcion == "6":
                ejecutar_demo_completo()
            elif opcion == "7":
                ejecutar_secuencia_completa()
            elif opcion == "8":
                mostrar_ayuda()
            elif opcion == "0":
                print("\n🚪 Saliendo del modo interactivo...")
                break
            else:
                print("\n❌ Opción no válida. Por favor, selecciona una opción del 0 al 8.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Operación cancelada por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
        
        # Pausa antes de continuar
        input("\n⏸️ Presiona Enter para continuar...")

def mostrar_ayuda():
    """Mostrar ayuda del sistema"""
    print("\n❓ AYUDA DEL INICIO AUTOMÁTICO")
    print("=" * 80)
    
    print("\n📋 DESCRIPCIÓN:")
    print("   El Inicio Automático del Sistema de Mejoras Integradas permite")
    print("   ejecutar automáticamente todas las operaciones necesarias para")
    print("   instalar, configurar, limpiar y ejecutar el sistema completo.")
    
    print("\n🔧 FUNCIONALIDADES:")
    print("   ✅ Verificación automática de instalación")
    print("   ✅ Instalación automática del sistema")
    print("   ✅ Configuración automática del sistema")
    print("   ✅ Limpieza automática del sistema")
    print("   ✅ Ejecución automática de mejoras")
    print("   ✅ Demo automático completo")
    print("   ✅ Secuencia completa automática")
    print("   ✅ Modo interactivo")
    
    print("\n🚀 MODOS DE EJECUCIÓN:")
    print("   🔄 Modo automático: Ejecuta secuencia completa automáticamente")
    print("   🎮 Modo interactivo: Permite seleccionar operaciones específicas")
    print("   🔍 Modo verificación: Solo verifica el estado del sistema")
    print("   📦 Modo instalación: Solo instala el sistema")
    print("   ⚙️ Modo configuración: Solo configura el sistema")
    print("   🧹 Modo limpieza: Solo limpia el sistema")
    print("   🚀 Modo ejecución: Solo ejecuta las mejoras")
    print("   🎯 Modo demo: Solo ejecuta el demo completo")
    
    print("\n💻 COMANDOS ÚTILES:")
    print("   • Inicio automático: python inicio_automatico.py")
    print("   • Modo interactivo: python inicio_automatico.py --interactive")
    print("   • Modo verificación: python inicio_automatico.py --verify")
    print("   • Modo instalación: python inicio_automatico.py --install")
    print("   • Modo configuración: python inicio_automatico.py --configure")
    print("   • Modo limpieza: python inicio_automatico.py --clean")
    print("   • Modo ejecución: python inicio_automatico.py --execute")
    print("   • Modo demo: python inicio_automatico.py --demo")
    
    print("\n🔧 PARÁMETROS:")
    print("   --interactive: Modo interactivo")
    print("   --verify: Solo verificar instalación")
    print("   --install: Solo instalar sistema")
    print("   --configure: Solo configurar sistema")
    print("   --clean: Solo limpiar sistema")
    print("   --execute: Solo ejecutar mejoras")
    print("   --demo: Solo ejecutar demo")
    print("   --help: Mostrar ayuda")
    
    print("\n📚 ARCHIVOS RELACIONADOS:")
    print("   • inicio_automatico.py - Script de inicio automático")
    print("   • instalar_sistema.py - Instalador automático")
    print("   • configurar_sistema.py - Configurador automático")
    print("   • limpiar_sistema.py - Limpiador automático")
    print("   • ejecutar_mejoras.py - Ejecutor de mejoras")
    print("   • demo_completo_mejoras.py - Demo completo")
    print("   • verificar_instalacion.py - Verificador de instalación")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("   1. Ejecutar inicio automático: python inicio_automatico.py")
    print("   2. Seleccionar modo de ejecución")
    print("   3. Seguir las instrucciones en pantalla")
    print("   4. Revisar reportes generados")
    print("   5. Implementar en producción")
    
    print("\n💡 CONSEJOS:")
    print("   • Usa el modo automático para instalación completa")
    print("   • Usa el modo interactivo para operaciones específicas")
    print("   • Siempre verifica la instalación antes de ejecutar")
    print("   • Revisa los reportes generados para análisis detallado")
    print("   • Configura las variables de entorno según necesidades")

def generar_reporte_inicio():
    """Generar reporte de inicio automático"""
    print("\n📊 Generando reporte de inicio automático...")
    
    reporte = {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'sistema_operativo': os.name,
        'directorio_inicio': os.getcwd(),
        'inicio_automatico_exitoso': True,
        'instalacion_automatica': True,
        'configuracion_automatica': True,
        'limpieza_automatica': True,
        'verificacion_automatica': True,
        'ejecucion_automatica': True,
        'demo_automatico': True
    }
    
    try:
        import json
        with open('reporte_inicio_automatico.json', 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        print("   ✅ Reporte de inicio automático generado: reporte_inicio_automatico.json")
    except Exception as e:
        print(f"   ❌ Error al generar reporte: {e}")
    
    return reporte

def main():
    """Función principal del inicio automático"""
    print_banner()
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        argumento = sys.argv[1].lower()
        
        if argumento == "--interactive":
            ejecutar_modo_interactivo()
        elif argumento == "--verify":
            verificar_instalacion_automaticamente()
        elif argumento == "--install":
            instalar_sistema_automaticamente()
        elif argumento == "--configure":
            configurar_sistema_automaticamente()
        elif argumento == "--clean":
            limpiar_sistema_automaticamente()
        elif argumento == "--execute":
            ejecutar_mejoras_automaticamente()
        elif argumento == "--demo":
            ejecutar_demo_completo()
        elif argumento == "--help":
            mostrar_ayuda()
        else:
            print(f"\n❌ Argumento no válido: {argumento}")
            print("💡 Usa --help para ver las opciones disponibles")
    else:
        # Modo automático por defecto
        print("\n🔄 Ejecutando modo automático...")
        
        # Ejecutar secuencia completa
        if ejecutar_secuencia_completa():
            # Generar reporte
            generar_reporte_inicio()
            
            # Resumen final
            print("\n🎉 INICIO AUTOMÁTICO COMPLETADO")
            print("=" * 80)
            print("✅ Sistema de Mejoras Integradas iniciado automáticamente")
            print("✅ Instalación automática completada")
            print("✅ Configuración automática completada")
            print("✅ Limpieza automática completada")
            print("✅ Verificación automática completada")
            print("✅ Ejecución automática completada")
            print("✅ Demo automático completado")
            print("✅ Reporte de inicio automático generado")
            
            print("\n🚀 PRÓXIMOS PASOS:")
            print("   1. Revisar reporte de inicio automático")
            print("   2. Verificar estado del sistema")
            print("   3. Configurar variables de entorno")
            print("   4. Implementar en producción")
            
            print("\n💡 COMANDOS ÚTILES:")
            print("   • Inicio automático: python inicio_automatico.py")
            print("   • Modo interactivo: python inicio_automatico.py --interactive")
            print("   • Verificar instalación: python verificar_instalacion.py")
            print("   • Ejecutar mejoras: python ejecutar_mejoras.py")
            print("   • Demo completo: python demo_completo_mejoras.py")
            
            print("\n🎉 ¡SISTEMA INICIADO AUTOMÁTICAMENTE!")
        else:
            print("\n❌ Error en inicio automático")
            print("💡 Usa el modo interactivo para diagnóstico: python inicio_automatico.py --interactive")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Inicio automático cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado durante el inicio automático: {e}")
        sys.exit(1)



