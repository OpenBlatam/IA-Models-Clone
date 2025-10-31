from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import subprocess
import time
import os
import sys
import threading

from typing import Any, List, Dict, Optional
import logging
import asyncio
WATCH_TERM = "mejoral"
OPTIMIZE_COMMAND = [sys.executable, "-m", "black", "."]  # Ejemplo: autoformatear con black
CURSOR_TERMINAL_LOG = os.path.expanduser("~/.cursor-terminal.log")  # Ajusta si el log está en otro lado
POLL_INTERVAL = 2  # segundos


def tail_f(filename, callback) -> Any:
    """Sigue el archivo como tail -f y llama a callback con cada nueva línea."""
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(POLL_INTERVAL)
                continue
            callback(line.strip())

def run_optimize():
    
    """run_optimize function."""
print(f"[mejoral_watcher] Ejecutando optimización: {' '.join(OPTIMIZE_COMMAND)}")
    try:
        subprocess.run(OPTIMIZE_COMMAND, check=True)
        print("[mejoral_watcher] Optimización completada.")
    except Exception as e:
        print(f"[mejoral_watcher] Error al optimizar: {e}")

def watcher():
    
    """watcher function."""
print(f"[mejoral_watcher] Observando terminal log: {CURSOR_TERMINAL_LOG}")
    def on_line(line) -> Any:
        if line.lower().endswith(WATCH_TERM):
            print(f"[mejoral_watcher] Detectado '{WATCH_TERM}' en terminal. Ejecutando optimización...")
            run_optimize()
    while not os.path.exists(CURSOR_TERMINAL_LOG):
        print(f"[mejoral_watcher] Esperando a que exista {CURSOR_TERMINAL_LOG}...")
        time.sleep(POLL_INTERVAL)
    tail_f(CURSOR_TERMINAL_LOG, on_line)

if __name__ == "__main__":
    watcher()

"""
USO:
1. Asegúrate de que la terminal de Cursor guarde el output en CURSOR_TERMINAL_LOG (ajusta la ruta si es necesario).
2. Ejecuta este script en segundo plano:
   python mejoral_watcher.py &
3. Cada vez que escribas 'mejoral' al final de una línea en la terminal, se ejecutará el comando de optimización.
4. Puedes cambiar OPTIMIZE_COMMAND por cualquier pipeline de mejora/lint/test que desees.
""" 