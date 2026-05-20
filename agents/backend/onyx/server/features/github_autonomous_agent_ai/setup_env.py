#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para configurar el archivo .env con las credenciales de GitHub OAuth y DeepSeek.
"""

import os
import sys
from pathlib import Path

# Configurar encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def create_env_file():
    """Crear archivo .env con las credenciales configuradas."""
    env_path = Path(__file__).parent / ".env"
    env_example_path = Path(__file__).parent / "env.example"
    
    if env_path.exists():
        print("[!] El archivo .env ya existe.")
        response = input("¿Deseas sobrescribirlo? (s/n): ")
        if response.lower() != 's':
            print("[X] Operacion cancelada.")
            return
    
    # Leer el archivo de ejemplo
    if env_example_path.exists():
        with open(env_example_path, 'r') as f:
            content = f.read()
    else:
        # Crear contenido por defecto
        content = """# API Configuration
API_HOST=0.0.0.0
API_PORT=8025

# GitHub OAuth Configuration
GITHUB_CLIENT_ID=Ov23liSy9XyIj3dD0dQc
GITHUB_CLIENT_SECRET=6ed948f00e7662bbba0eacd356e60747dd12f08f
GITHUB_TOKEN=
GITHUB_REDIRECT_URI=http://localhost:8025/api/github/auth/callback

# DeepSeek LLM Configuration
DEEPSEEK_API_KEY=sk-ae1c47feaa3e483b85a936430d1f494a
DEEPSEEK_API_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_ENABLED=true
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Agent Settings
AGENT_POLL_INTERVAL=5
AGENT_MAX_CONCURRENT_TASKS=3
AGENT_TASK_TIMEOUT=3600

# Storage Settings
STORAGE_PATH=./data
TASKS_DB_PATH=./data/tasks.db

# Logging
LOG_LEVEL=INFO
"""
    
    # Escribir el archivo .env
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] Archivo .env creado exitosamente!")
    print(f"[*] Ubicacion: {env_path.absolute()}")
    print("\n[*] Credenciales configuradas:")
    print("   - GitHub Client ID: Ov23liSy9XyIj3dD0dQc")
    print("   - GitHub Client Secret: 6ed948f00e7662bbba0eacd356e60747dd12f08f")
    print("   - DeepSeek API Key: sk-ae1c47feaa3e483b85a936430d1f494a")
    print("\n[!] IMPORTANTE: Asegurate de que en GitHub OAuth App la callback URL sea:")
    print("   http://localhost:8025/api/github/auth/callback")

if __name__ == "__main__":
    create_env_file()

