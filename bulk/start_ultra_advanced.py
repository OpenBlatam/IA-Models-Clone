#!/usr/bin/env python3
"""
BUL Ultra Advanced System Launcher
==================================

Launcher ultra-avanzado para el sistema BUL con todas las funcionalidades:
- API Ultra Advanced
- Dashboard Ultra Advanced  
- WebSocket Server
- Database Setup
- Backup System
- Monitoring
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

class BULUltraLauncher:
    """Ultra Advanced BUL System Launcher."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.start_time = None
        
    def check_dependencies(self):
        """Verificar dependencias ultra-avanzadas."""
        print("🔍 Checking ultra-advanced dependencies...")
        
        required_files = [
            "bul_ultra_advanced.py",
            "dashboard_ultra.py", 
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
        """Instalar dependencias ultra-avanzadas."""
        print("📦 Installing ultra-advanced dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            print("✅ Ultra-advanced dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_directories(self):
        """Crear directorios necesarios."""
        directories = [
            "uploads", "downloads", "logs", "backups", 
            "templates", "collaboration", "notifications"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def setup_database(self):
        """Configurar base de datos."""
        print("🗄️ Setting up database...")
        try:
            # The database will be created automatically when the API starts
            print("✅ Database setup completed")
            return True
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return False
    
    def start_api(self):
        """Iniciar API ultra-avanzada."""
        print("🚀 Starting Ultra Advanced BUL API...")
        try:
            process = subprocess.Popen([
                sys.executable, "bul_ultra_advanced.py", 
                "--host", "0.0.0.0", 
                "--port", "8000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["api"] = process
            print("✅ Ultra Advanced API started")
            return True
        except Exception as e:
            print(f"❌ Error starting API: {e}")
            return False
    
    def start_dashboard(self):
        """Iniciar dashboard ultra-avanzado."""
        print("📊 Starting Ultra Advanced Dashboard...")
        try:
            process = subprocess.Popen([
                sys.executable, "dashboard_ultra.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["dashboard"] = process
            print("✅ Ultra Advanced Dashboard started")
            return True
        except Exception as e:
            print(f"❌ Error starting Dashboard: {e}")
            return False
    
    def start_monitoring(self):
        """Iniciar sistema de monitoreo."""
        print("📈 Starting monitoring system...")
        try:
            # Create monitoring script
            monitoring_script = """
import time
import requests
import json
from datetime import datetime

def monitor_system():
    while True:
        try:
            # Check API health
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"[{datetime.now()}] API Status: {data['status']}, Tasks: {data['active_tasks']}, Connections: {data['active_connections']}")
            
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"[{datetime.now()}] Monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_system()
"""
            
            with open("monitor.py", "w") as f:
                f.write(monitoring_script)
            
            process = subprocess.Popen([
                sys.executable, "monitor.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["monitoring"] = process
            print("✅ Monitoring system started")
            return True
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")
            return False
    
    def create_backup(self):
        """Crear backup del sistema."""
        print("💾 Creating system backup...")
        try:
            backup_data = {
                "backup_id": f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": sys.version,
                    "platform": os.name,
                    "working_directory": os.getcwd()
                },
                "processes": list(self.processes.keys()),
                "start_time": self.start_time.isoformat() if self.start_time else None
            }
            
            backup_file = Path(f"backups/system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            backup_file.parent.mkdir(exist_ok=True)
            
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)
            
            print(f"✅ Backup created: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def check_system_health(self):
        """Verificar salud del sistema."""
        print("🏥 Checking system health...")
        
        try:
            # Check API
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                api_health = response.json()
                print(f"✅ API Status: {api_health['status']}")
                print(f"   - Active Tasks: {api_health['active_tasks']}")
                print(f"   - WebSocket Connections: {api_health['active_connections']}")
                print(f"   - Collaboration Rooms: {api_health['collaboration_rooms']}")
            else:
                print("❌ API Health Check Failed")
                return False
            
            # Check Dashboard (basic check)
            try:
                response = requests.get("http://localhost:8050", timeout=5)
                if response.status_code == 200:
                    print("✅ Dashboard Status: Online")
                else:
                    print("⚠️ Dashboard Status: Unknown")
            except:
                print("⚠️ Dashboard Status: Not accessible")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def stop_all_processes(self):
        """Detener todos los procesos."""
        print("🛑 Stopping all processes...")
        
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
        print("\n🛑 Shutdown signal received...")
        self.stop_all_processes()
        sys.exit(0)
    
    def run_full_system(self):
        """Ejecutar sistema completo ultra-avanzado."""
        print("🚀 BUL Ultra Advanced System Launcher")
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
            print("⏳ Waiting for API to initialize...")
            time.sleep(5)
            
            if not self.start_dashboard():
                return False
            
            if not self.start_monitoring():
                return False
            
            # Create initial backup
            self.create_backup()
            
            # Health check
            time.sleep(3)
            self.check_system_health()
            
            self.running = True
            
            print("\n" + "=" * 60)
            print("🎉 BUL Ultra Advanced System is running!")
            print("=" * 60)
            print("🔗 Services:")
            print("   - API: http://localhost:8000")
            print("   - Dashboard: http://localhost:8050")
            print("   - API Docs: http://localhost:8000/docs")
            print("   - Metrics: http://localhost:8000/metrics")
            print("   - WebSocket: ws://localhost:8000/ws/{client_id}")
            print("\n📊 Features:")
            print("   ✅ Real-time WebSocket communication")
            print("   ✅ Document templates")
            print("   ✅ Version control")
            print("   ✅ Real-time collaboration")
            print("   ✅ Notification system")
            print("   ✅ Backup & restore")
            print("   ✅ Multi-tenant support")
            print("   ✅ Advanced monitoring")
            print("\n🛑 Press Ctrl+C to stop the system")
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
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.stop_all_processes()
            print("👋 BUL Ultra Advanced System stopped")
    
    def run_interactive(self):
        """Ejecutar modo interactivo."""
        print("🚀 BUL Ultra Advanced System Launcher")
        print("=" * 60)
        
        # Verificar archivos
        if not self.check_dependencies():
            print("❌ Missing required files. Please ensure all files are present.")
            return
        
        # Preguntar si instalar dependencias
        install_deps = input("📦 Install/update ultra-advanced dependencies? (y/n): ").lower().strip() == 'y'
        if install_deps:
            if not self.install_dependencies():
                print("❌ Failed to install dependencies. Please install manually.")
                return
        
        # Setup
        self.setup_directories()
        self.setup_database()
        
        print("\n🎯 Choose startup mode:")
        print("1. API Ultra Advanced only")
        print("2. Dashboard Ultra Advanced only") 
        print("3. Full Ultra Advanced System (API + Dashboard + Monitoring)")
        print("4. Health Check Only")
        print("5. Create Backup")
        print("6. Exit")
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Ultra Advanced API only...")
            self.start_api()
            input("\nPress Enter to stop...")
            self.stop_all_processes()
        elif choice == "2":
            print("\n📊 Starting Ultra Advanced Dashboard only...")
            print("⚠️  Note: Dashboard requires API to be running")
            self.start_dashboard()
            input("\nPress Enter to stop...")
            self.stop_all_processes()
        elif choice == "3":
            print("\n🚀 Starting Full Ultra Advanced System...")
            self.run_full_system()
        elif choice == "4":
            print("\n🏥 Running health check...")
            self.start_api()
            time.sleep(5)
            self.check_system_health()
            self.stop_all_processes()
        elif choice == "5":
            print("\n💾 Creating backup...")
            self.create_backup()
        elif choice == "6":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")

def main():
    """Función principal."""
    launcher = BULUltraLauncher()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            launcher.run_full_system()
        else:
            launcher.run_interactive()
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
