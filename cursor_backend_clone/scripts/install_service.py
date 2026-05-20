"""
Install Service - Instalar como servicio del sistema
====================================================

Script para instalar el agente como servicio del sistema.
"""

import sys
import os
import platform
from pathlib import Path

def install_windows_service():
    """Instalar como servicio de Windows"""
    print("📦 Installing as Windows Service...")
    
    # Usar NSSM (Non-Sucking Service Manager)
    nssm_path = os.getenv("NSSM_PATH", "nssm")
    
    service_name = "CursorAgent24_7"
    python_exe = sys.executable
    script_path = Path(__file__).parent.parent / "main.py"
    
    print(f"Service name: {service_name}")
    print(f"Python: {python_exe}")
    print(f"Script: {script_path}")
    
    # Comando NSSM
    import subprocess
    
    try:
        # Instalar servicio
        subprocess.run([
            nssm_path, "install", service_name,
            python_exe, str(script_path), "--mode", "service"
        ], check=True)
        
        # Configurar servicio
        subprocess.run([nssm_path, "set", service_name, "AppDirectory", str(script_path.parent)], check=True)
        subprocess.run([nssm_path, "set", service_name, "DisplayName", "Cursor Agent 24/7"], check=True)
        subprocess.run([nssm_path, "set", service_name, "Description", "Persistent Cursor Agent"], check=True)
        subprocess.run([nssm_path, "set", service_name, "Start", "SERVICE_AUTO_START"], check=True)
        
        print("✅ Service installed successfully!")
        print(f"💡 Start with: nssm start {service_name}")
        print(f"💡 Stop with: nssm stop {service_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing service: {e}")
        print("💡 Make sure NSSM is installed and in PATH")
        print("💡 Download from: https://nssm.cc/download")


def install_linux_service():
    """Instalar como servicio systemd"""
    print("📦 Installing as systemd service...")
    
    service_name = "cursor-agent-24-7"
    script_path = Path(__file__).parent.parent / "main.py"
    python_exe = sys.executable
    user = os.getenv("USER", "root")
    
    service_content = f"""[Unit]
Description=Cursor Agent 24/7
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={script_path.parent}
ExecStart={python_exe} {script_path} --mode service
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path(f"/etc/systemd/system/{service_name}.service")
    
    print(f"Service file: {service_file}")
    print("\nService configuration:")
    print(service_content)
    
    try:
        # Escribir archivo de servicio (requiere sudo)
        if os.geteuid() == 0:
            service_file.write_text(service_content)
            print(f"✅ Service file created: {service_file}")
            
            # Recargar systemd
            import subprocess
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", service_name], check=True)
            
            print("✅ Service installed and enabled!")
            print(f"💡 Start with: sudo systemctl start {service_name}")
            print(f"💡 Status with: sudo systemctl status {service_name}")
        else:
            print("⚠️  Root privileges required")
            print(f"💡 Run: sudo python {__file__}")
            print(f"💡 Or manually create: {service_file}")
            
    except Exception as e:
        print(f"❌ Error installing service: {e}")


def install_macos_service():
    """Instalar como servicio launchd"""
    print("📦 Installing as launchd service...")
    
    service_name = "com.cursor.agent24-7"
    script_path = Path(__file__).parent.parent / "main.py"
    python_exe = sys.executable
    user_home = Path.home()
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>{script_path}</string>
        <string>--mode</string>
        <string>service</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{script_path.parent}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{user_home}/Library/Logs/{service_name}.log</string>
    <key>StandardErrorPath</key>
    <string>{user_home}/Library/Logs/{service_name}.error.log</string>
</dict>
</plist>
"""
    
    plist_file = user_home / "Library" / "LaunchAgents" / f"{service_name}.plist"
    plist_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Plist file: {plist_file}")
    print("\nService configuration:")
    print(plist_content)
    
    try:
        plist_file.write_text(plist_content)
        print(f"✅ Service file created: {plist_file}")
        print(f"💡 Load with: launchctl load {plist_file}")
        print(f"💡 Unload with: launchctl unload {plist_file}")
        
    except Exception as e:
        print(f"❌ Error installing service: {e}")


def main():
    """Función principal"""
    system = platform.system().lower()
    
    print("🚀 Cursor Agent 24/7 - Service Installer")
    print("=" * 50)
    
    if system == "windows":
        install_windows_service()
    elif system == "linux":
        install_linux_service()
    elif system == "darwin":
        install_macos_service()
    else:
        print(f"❌ Unsupported operating system: {system}")
        print("💡 Manual installation required")


if __name__ == "__main__":
    main()


