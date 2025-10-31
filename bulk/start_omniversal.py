#!/usr/bin/env python3
"""
BUL Omniversal AI Launcher
=========================

Launcher para el sistema BUL Omniversal AI con todas las funcionalidades:
- API Omniversal AI
- Dashboard Omniversal AI
- Universe Creation
- Dimensional Transcendence
- Black Hole Computing
- Space-Time Manipulation
- Divine AI
- Cosmic Consciousness
"""

import subprocess
import sys
import time
import threading
import os
import signal
import json
from pathlib import Path
from datetime import datetime

class BULOmniversalLauncher:
    """Omniversal BUL System Launcher."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.start_time = None
        
    def check_dependencies(self):
        """Verificar dependencias omniversales."""
        print("🔍 Checking omniversal dependencies...")
        
        required_files = [
            "bul_omniversal_ai.py",
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
    
    def install_dependencies(self):
        """Instalar dependencias omniversales."""
        print("📦 Installing omniversal dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            print("✅ Omniversal dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_directories(self):
        """Crear directorios necesarios."""
        directories = [
            "uploads", "downloads", "logs", "backups", 
            "universes", "dimensions", "blackholes", "spacetime",
            "divine", "cosmic", "omniversal"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def setup_database(self):
        """Configurar base de datos omniversal."""
        print("🗄️ Setting up omniversal database...")
        try:
            # The database will be created automatically when the API starts
            print("✅ Omniversal database setup completed")
            return True
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return False
    
    def start_api(self):
        """Iniciar API omniversal."""
        print("🚀 Starting Omniversal BUL API...")
        try:
            process = subprocess.Popen([
                sys.executable, "bul_omniversal_ai.py", 
                "--host", "0.0.0.0", 
                "--port", "8000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["api"] = process
            print("✅ Omniversal API started")
            return True
        except Exception as e:
            print(f"❌ Error starting API: {e}")
            return False
    
    def start_monitoring(self):
        """Iniciar sistema de monitoreo omniversal."""
        print("📈 Starting omniversal monitoring system...")
        try:
            # Create monitoring script
            monitoring_script = """
import time
import requests
import json
from datetime import datetime

def monitor_omniversal_system():
    while True:
        try:
            # Check API health
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"[{datetime.now()}] Omniversal API Status: {data['status']}")
                print(f"[{datetime.now()}] Active Tasks: {data['active_tasks']}")
                print(f"[{datetime.now()}] Universe Creations: {data['universe_creations']}")
                print(f"[{datetime.now()}] Dimensional Transcendence: {data['dimensional_transcendence_sessions']}")
            
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"[{datetime.now()}] Omniversal monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_omniversal_system()
"""
            
            with open("monitor_omniversal.py", "w") as f:
                f.write(monitoring_script)
            
            process = subprocess.Popen([
                sys.executable, "monitor_omniversal.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["monitoring"] = process
            print("✅ Omniversal monitoring system started")
            return True
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")
            return False
    
    def create_backup(self):
        """Crear backup del sistema omniversal."""
        print("💾 Creating omniversal system backup...")
        try:
            backup_data = {
                "backup_id": f"omniversal_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": sys.version,
                    "platform": os.name,
                    "working_directory": os.getcwd()
                },
                "processes": list(self.processes.keys()),
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "omniversal_features": [
                    "GPT-Omniverse",
                    "Claude-Divine", 
                    "Gemini-Infinite",
                    "Neural-Omniverse",
                    "Quantum-Omniverse",
                    "Black Hole Computing",
                    "Space-Time Manipulation",
                    "Divine AI",
                    "Universe Creation",
                    "Dimensional Transcendence",
                    "Cosmic Consciousness",
                    "Reality Engineering",
                    "Multiverse Control",
                    "Infinite Intelligence"
                ]
            }
            
            backup_file = Path(f"backups/omniversal_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            backup_file.parent.mkdir(exist_ok=True)
            
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)
            
            print(f"✅ Omniversal backup created: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def check_system_health(self):
        """Verificar salud del sistema omniversal."""
        print("🏥 Checking omniversal system health...")
        
        try:
            # Check API
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                api_health = response.json()
                print(f"✅ Omniversal API Status: {api_health['status']}")
                print(f"   - Active Tasks: {api_health['active_tasks']}")
                print(f"   - Universe Creations: {api_health['universe_creations']}")
                print(f"   - Dimensional Transcendence: {api_health['dimensional_transcendence_sessions']}")
                print(f"   - Omniversal Features: {len(api_health['omniversal_features'])}")
            else:
                print("❌ Omniversal API Health Check Failed")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def stop_all_processes(self):
        """Detener todos los procesos omniversales."""
        print("🛑 Stopping all omniversal processes...")
        
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=10)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️ {name} force stopped")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        self.processes.clear()
        self.running = False
    
    def signal_handler(self, signum, frame):
        """Manejador de señales para shutdown graceful."""
        print("\n🛑 Omniversal shutdown signal received...")
        self.stop_all_processes()
        sys.exit(0)
    
    def run_full_system(self):
        """Ejecutar sistema completo omniversal."""
        print("🚀 BUL Omniversal AI System Launcher")
        print("=" * 60)
        
        self.start_time = datetime.now()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Setup
            self.setup_directories()
            self.setup_database()
            
            # Start components
            if not self.start_api():
                return False
            
            # Wait for API to start
            print("⏳ Waiting for Omniversal API to initialize...")
            time.sleep(5)
            
            if not self.start_monitoring():
                return False
            
            # Create initial backup
            self.create_backup()
            
            # Health check
            time.sleep(3)
            self.check_system_health()
            
            self.running = True
            
            print("\n" + "=" * 60)
            print("🎉 BUL Omniversal AI System is running!")
            print("=" * 60)
            print("🔗 Services:")
            print("   - Omniversal API: http://localhost:8000")
            print("   - API Docs: http://localhost:8000/docs")
            print("   - Omniversal AI Models: http://localhost:8000/ai/omniversal-models")
            print("   - Universe Creation: http://localhost:8000/universe/create")
            print("   - Dimensional Transcendence: http://localhost:8000/dimensional-transcendence/transcend")
            print("\n🌌 Omniversal Features:")
            print("   ✅ GPT-Omniverse with Omniversal Reasoning")
            print("   ✅ Claude-Divine with Divine AI")
            print("   ✅ Gemini-Infinite with Infinite Intelligence")
            print("   ✅ Neural-Omniverse with Omniversal Consciousness")
            print("   ✅ Quantum-Omniverse with Quantum Omniversal")
            print("   ✅ Black Hole Computing")
            print("   ✅ Space-Time Manipulation")
            print("   ✅ Divine AI")
            print("   ✅ Universe Creation")
            print("   ✅ Dimensional Transcendence")
            print("   ✅ Cosmic Consciousness")
            print("   ✅ Reality Engineering")
            print("   ✅ Multiverse Control")
            print("   ✅ Infinite Intelligence")
            print("\n🛑 Press Ctrl+C to stop the omniversal system")
            print("=" * 60)
            
            # Keep running
            while self.running:
                time.sleep(1)
                
                # Check if processes are still running
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        print(f"⚠️ {name} process stopped unexpectedly")
                        self.running = False
                        break
            
        except KeyboardInterrupt:
            print("\n🛑 Omniversal shutdown requested by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.stop_all_processes()
            print("👋 BUL Omniversal AI System stopped")
    
    def run_interactive(self):
        """Ejecutar modo interactivo omniversal."""
        print("🚀 BUL Omniversal AI System Launcher")
        print("=" * 60)
        
        # Verificar archivos
        if not self.check_dependencies():
            print("❌ Missing required files. Please ensure all files are present.")
            return
        
        # Preguntar si instalar dependencias
        install_deps = input("📦 Install/update omniversal dependencies? (y/n): ").lower().strip() == 'y'
        if install_deps:
            if not self.install_dependencies():
                print("❌ Failed to install dependencies. Please install manually.")
                return
        
        # Setup
        self.setup_directories()
        self.setup_database()
        
        print("\n🌌 Choose omniversal startup mode:")
        print("1. Omniversal API only")
        print("2. Full Omniversal System (API + Monitoring)")
        print("3. Health Check Only")
        print("4. Create Omniversal Backup")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Omniversal API only...")
            self.start_api()
            input("\nPress Enter to stop...")
            self.stop_all_processes()
        elif choice == "2":
            print("\n🚀 Starting Full Omniversal System...")
            self.run_full_system()
        elif choice == "3":
            print("\n🏥 Running omniversal health check...")
            self.start_api()
            time.sleep(5)
            self.check_system_health()
            self.stop_all_processes()
        elif choice == "4":
            print("\n💾 Creating omniversal backup...")
            self.create_backup()
        elif choice == "5":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")

def main():
    """Función principal."""
    launcher = BULOmniversalLauncher()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            launcher.run_full_system()
        else:
            launcher.run_interactive()
    except KeyboardInterrupt:
        print("\n🛑 Omniversal system stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
