#!/usr/bin/env python3
"""
BUL Enhanced System Launcher
============================

Script para iniciar el sistema BUL mejorado con API y dashboard.
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def run_api():
    """Ejecutar la API mejorada."""
    print("🚀 Starting Enhanced BUL API...")
    try:
        subprocess.run([
            sys.executable, "bul_enhanced.py", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API: {e}")
    except KeyboardInterrupt:
        print("🛑 API stopped by user")

def run_dashboard():
    """Ejecutar el dashboard."""
    print("📊 Starting BUL Dashboard...")
    try:
        subprocess.run([
            sys.executable, "dashboard.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Dashboard: {e}")
    except KeyboardInterrupt:
        print("🛑 Dashboard stopped by user")

def check_dependencies():
    """Verificar dependencias."""
    print("🔍 Checking dependencies...")
    
    required_files = [
        "bul_enhanced.py",
        "dashboard.py", 
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def install_dependencies():
    """Instalar dependencias."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    """Función principal."""
    print("🚀 BUL Enhanced System Launcher")
    print("=" * 50)
    
    # Verificar archivos
    if not check_dependencies():
        print("❌ Missing required files. Please ensure all files are present.")
        return
    
    # Preguntar si instalar dependencias
    install_deps = input("📦 Install/update dependencies? (y/n): ").lower().strip() == 'y'
    if install_deps:
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually.")
            return
    
    # Crear directorios necesarios
    directories = ["uploads", "downloads", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n🎯 Choose startup mode:")
    print("1. API only")
    print("2. Dashboard only") 
    print("3. Both API and Dashboard")
    print("4. Exit")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting API only...")
        run_api()
    elif choice == "2":
        print("\n📊 Starting Dashboard only...")
        print("⚠️  Note: Dashboard requires API to be running")
        run_dashboard()
    elif choice == "3":
        print("\n🚀 Starting both API and Dashboard...")
        
        # Iniciar API en hilo separado
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Esperar un poco para que la API inicie
        print("⏳ Waiting for API to start...")
        time.sleep(3)
        
        # Iniciar dashboard
        run_dashboard()
    elif choice == "4":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
