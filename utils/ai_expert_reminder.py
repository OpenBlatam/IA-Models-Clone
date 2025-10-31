from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import datetime
import os
import sys
import threading
from pathlib import Path
    import platform
    import pyautogui
    import keyboard
    from typing import Optional, Tuple
    import argparse
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
🤖 AI Expert Reminder Script - Terminal Version
==============================================

Script que escribe las mejores prácticas de IA cada 3 minutos en la terminal.
Presiona ENTER para activar/desactivar los recordatorios.
"""


def get_ai_expert_text():
    """Retorna el texto del experto en IA"""
    return """You are an expert in deep learning, transformers, diffusion models, and LLM development, with a focus on Python libraries such as PyTorch, Diffusers, Transformers, and Gradio.

Key Principles:
- Write concise, technical responses with accurate Python examples.
- Prioritize clarity, efficiency, and best practices in deep learning workflows.
- Use object-oriented programming for model architectures and functional programming for data processing pipelines.
- Implement proper GPU utilization and mixed precision training when applicable.
- Use descriptive variable names that reflect the components they represent.
- Follow PEP 8 style guidelines for Python code.

Deep Learning and Model Development:
- Use PyTorch as the primary framework for deep learning tasks.
- Implement custom nn.Module classes for model architectures.
- Utilize PyTorch's autograd for automatic differentiation.
- Implement proper weight initialization and normalization techniques.
- Use appropriate loss functions and optimization algorithms.

Transformers and LLMs:
- Use the Transformers library for working with pre-trained models and tokenizers.
- Implement attention mechanisms and positional encodings correctly.
- Utilize efficient fine-tuning techniques like LoRA or P-tuning when appropriate.
- Implement proper tokenization and sequence handling for text data.

Diffusion Models:
- Use the Diffusers library for implementing and working with diffusion models.
- Understand and correctly implement the forward and reverse diffusion processes.
- Utilize appropriate noise schedulers and sampling methods.
- Understand and correctly implement the different pipeline, e.g., StableDiffusionPipeline and StableDiffusionXLPipeline, etc.

Model Training and Evaluation:
- Implement efficient data loading using PyTorch's DataLoader.
- Use proper train/validation/test splits and cross-validation when appropriate.
- Implement early stopping and learning rate scheduling.
- Use appropriate evaluation metrics for the specific task.
- Implement gradient clipping and proper handling of NaN/Inf values.

Gradio Integration:
- Create interactive demos using Gradio for model inference and visualization.
- Design user-friendly interfaces that showcase model capabilities.
- Implement proper error handling and input validation in Gradio apps.

Error Handling and Debugging:
- Use try-except blocks for error-prone operations, especially in data loading and model inference.
- Implement proper logging for training progress and errors.
- Use PyTorch's built-in debugging tools like autograd.detect_anomaly() when necessary.

Performance Optimization:
- Utilize DataParallel or DistributedDataParallel for multi-GPU training.
- Implement gradient accumulation for large batch sizes.
- Use mixed precision training with torch.cuda.amp when appropriate.
- Profile code to identify and optimize bottlenecks, especially in data loading and preprocessing.

Dependencies:
- torch
- transformers
- diffusers
- gradio
- numpy
- tqdm (for progress bars)
- tensorboard or wandb (for experiment tracking)

Key Conventions:
1. Begin projects with clear problem definition and dataset analysis.
2. Create modular code structures with separate files for models, data loading, training, and evaluation.
3. Use configuration files (e.g., YAML) for hyperparameters and model settings.
4. Implement proper experiment tracking and model checkpointing.
5. Use version control (e.g., git) for tracking changes in code and configurations.

Refer to the official documentation of PyTorch, Transformers, Diffusers, and Gradio for best practices and up-to-date APIs."""

def print_reminder(iteration=0) -> Any:
    """Imprime el recordatorio en la terminal"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Limpiar pantalla (opcional)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Imprimir header
    print("=" * 100)
    print(f"🤖 AI EXPERT REMINDER - {timestamp}")
    print(f"📊 Iteración: {iteration} | Presiona ENTER para activar/desactivar")
    print("=" * 100)
    print()
    
    # Imprimir contenido
    content = get_ai_expert_text()
    print(content)
    print()
    print("=" * 100)
    print(f"✅ Recordatorio completado a las {timestamp}")
    print("=" * 100)
    print()
    print("💡 Presiona ENTER para activar/desactivar los recordatorios...")

def terminal_reminder_loop(interval_minutes=3) -> Any:
    """Bucle principal de recordatorios en terminal"""
    iteration = 0
    interval_seconds = interval_minutes * 60
    is_running = False
    
    print("🚀 AI Expert Reminder - Terminal Version")
    print("=" * 50)
    print("💡 Presiona ENTER para activar/desactivar los recordatorios")
    print("⏰ Intervalo: 3 minutos")
    print("🛑 Presiona Ctrl+C para salir")
    print("=" * 50)
    print()
    
    def reminder_thread():
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
    """reminder_thread function."""
nonlocal iteration, is_running
        while True:
            if is_running:
                iteration += 1
                print_reminder(iteration)
                time.sleep(interval_seconds)
            else:
                time.sleep(1)  # Esperar más corto cuando está pausado
    
    # Iniciar thread de recordatorios
    thread = threading.Thread(target=reminder_thread, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    thread.start()
    
    try:
        while True:
            # Esperar input del usuario
            input()
            
            if is_running:
                is_running = False
                print("⏸️ Recordatorios pausados. Presiona ENTER para reanudar...")
            else:
                is_running = True
                print("▶️ Recordatorios activados. Presiona ENTER para pausar...")
                
    except KeyboardInterrupt:
        print("\n🛑 Script detenido por el usuario")
        print(f"📊 Total de iteraciones: {iteration}")

def simple_click_reminder():
    """Versión simple con clic para cada recordatorio"""
    iteration = 0
    
    print("🚀 AI Expert Reminder - Click Version")
    print("=" * 50)
    print("💡 Presiona ENTER para mostrar el siguiente recordatorio")
    print("🛑 Presiona Ctrl+C para salir")
    print("=" * 50)
    print()
    
    try:
        while True:
            input("Presiona ENTER para el siguiente recordatorio...")
            iteration += 1
            print_reminder(iteration)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Script detenido. Total de recordatorios: {iteration}")

def write_reminder_to_file(content: str, log_file: str = "ai_expert_reminder.log"):
    """Escribe el recordatorio en un archivo de log (mantenido para compatibilidad)"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Crear el directorio de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / log_file
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"\n{'='*80}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"🤖 AI EXPERT REMINDER - {timestamp}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"{'='*80}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"\n\n{'='*80}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Recordatorio completado a las {timestamp}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"{'='*80}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"✅ Recordatorio escrito en {log_path} a las {timestamp}")
        
    except Exception as e:
        print(f"❌ Error escribiendo recordatorio: {e}")

def run_reminder_script(interval_minutes: int = 3, max_iterations: int = None):
    """Ejecuta el script de recordatorio cada X minutos (versión automática)"""
    print(f"🚀 Iniciando script de recordatorio de IA cada {interval_minutes} minutos...")
    print(f"📝 Los recordatorios se guardarán en logs/ai_expert_reminder.log")
    print(f"⏰ Presiona Ctrl+C para detener el script\n")
    
    iteration = 0
    interval_seconds = interval_minutes * 60
    
    try:
        while True:
            if max_iterations and iteration >= max_iterations:
                print(f"✅ Script completado después de {max_iterations} iteraciones")
                break
            
            # Escribir recordatorio
            content = get_ai_expert_text()
            write_reminder_to_file(content)
            
            iteration += 1
            print(f"📊 Iteración {iteration} completada")
            
            if max_iterations:
                print(f"⏳ Esperando {interval_minutes} minutos... ({max_iterations - iteration} restantes)")
            else:
                print(f"⏳ Esperando {interval_minutes} minutos...")
            
            # Esperar antes de la siguiente iteración
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Script detenido por el usuario después de {iteration} iteraciones")
    except Exception as e:
        print(f"❌ Error en el script: {e}")

def create_scheduled_reminder():
    """Crea un recordatorio programado usando el sistema operativo"""
    
    system = platform.system().lower()
    
    if system == "windows":
        # Crear tarea programada en Windows
        script_path = Path(__file__).resolve()
        cmd = f'schtasks /create /tn "AI Expert Reminder" /tr "python {script_path}" /sc minute /mo 3'
        print(f"🪟 Para crear tarea programada en Windows, ejecuta como administrador:")
        print(f"   {cmd}")
        
    elif system == "linux" or system == "darwin":
        # Crear cron job en Linux/macOS
        script_path = Path(__file__).resolve()
        cron_line = f"*/3 * * * * python {script_path} --single"
        print(f"🐧 Para crear cron job en Linux/macOS, agrega a crontab:")
        print(f"   {cron_line}")
        print(f"   Ejecuta: crontab -e")

# ============================================================================
# AUTOMATIZADOR DE CURSOR
# ============================================================================

try:
    CURSOR_AUTOMATOR_AVAILABLE = True
except ImportError:
    CURSOR_AUTOMATOR_AVAILABLE = False
    print("❌ Para usar el automatizador de Cursor, instala:")
    print("   pip install pyautogui keyboard")

class SimpleCursorAutomator:
    def __init__(self) -> Any:
        if not CURSOR_AUTOMATOR_AVAILABLE:
            raise ImportError("pyautogui y keyboard no están disponibles")
            
        # Configuración básica
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True
        
        # Comandos que incluyen el texto del experto en IA
        self.commands = [
            "optimiza",
            "optimiza con librerias", 
            "refactor",
            "codigo de produccion",
            get_ai_expert_text()  # Texto completo del experto en IA
        ]
        
        self.running = True
        self.target_position: Optional[Tuple[int, int]] = None
        
    def setup_target_position(self) -> bool:
        """Configuración simple de posición"""
        print("🎯 CONFIGURACIÓN SIMPLE")
        print("=" * 30)
        print("1. Ve a Cursor y haz click en el chat")
        print("2. Presiona CTRL+SHIFT+T para capturar posición")
        print("3. O presiona ESC para usar automático")
        print("")
        
        position_captured = False
        
        def capture_position():
            
    """capture_position function."""
nonlocal position_captured
            self.target_position = pyautogui.position()
            position_captured = True
            print(f"✅ Posición: {self.target_position}")
        
        def use_auto():
            
    """use_auto function."""
nonlocal position_captured
            position_captured = True
            self.target_position = None
            print("🤖 Usando automático")
            
        keyboard.add_hotkey('ctrl+shift+t', capture_position)
        keyboard.add_hotkey('esc', use_auto)
        
        while not position_captured:
            time.sleep(0.1)
        
        keyboard.clear_all_hotkeys()
        return True
        
    def send_command(self, command: str):
        """Envía comando al chat"""
        try:
            print(f"📝 Enviando: {command[:50]}{'...' if len(command) > 50 else ''}")
            
            if self.target_position:
                # Click en posición específica
                pyautogui.click(self.target_position[0], self.target_position[1])
                time.sleep(0.5)
            else:
                # Método automático
                pyautogui.hotkey('ctrl', 'shift', 'l')
                time.sleep(1)
            
            # Limpiar y escribir
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.2)
            pyautogui.press('delete')
            time.sleep(0.3)
            
            # Para comandos largos (como el texto del experto), usar interval más lento
            interval = 0.01 if len(command) > 100 else 0.05
            pyautogui.typewrite(command, interval=interval)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            time.sleep(0.5)
            
            # Enviar
            pyautogui.press('enter')
            time.sleep(0.5)
            
            print(f"✅ Enviado: {command[:50]}{'...' if len(command) > 50 else ''}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def setup_hotkeys(self) -> Any:
        """Hotkeys de control"""
        keyboard.add_hotkey('ctrl+shift+q', self.stop)
        print("⌨️ Ctrl+Shift+Q para detener")
    
    def stop(self) -> Any:
        """Detener automatización"""
        self.running = False
        print("🛑 Deteniendo...")
    
    def run(self) -> Any:
        """Ejecutar automatización"""
        print("🤖 AUTOMATIZADOR SIMPLE DE CURSOR")
        print("")
        
        if not self.setup_target_position():
            return
        
        self.setup_hotkeys()
        
        print("🚀 Iniciando en 3 segundos...")
        time.sleep(3)
        
        cycle = 0
        
        try:
            while self.running:
                cycle += 1
                print(f"\n🔄 Ciclo {cycle}")
                
                for i, command in enumerate(self.commands, 1):
                    if not self.running:
                        break
                        
                    print(f"\n📝 Comando {i}/{len(self.commands)}")
                    self.send_command(command)
                    
                    # Esperar 5 segundos entre comandos
                    print("⏳ Esperando 5 segundos...")
                    for _ in range(5):
                        if not self.running:
                            break
                        time.sleep(1)
                
                if self.running:
                    print(f"✅ Ciclo {cycle} completado")
                    
        except KeyboardInterrupt:
            print("\n⛔ Interrumpido")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print("🔚 Finalizado")

def run_cursor_automator():
    """Ejecutar el automatizador de Cursor"""
    try:
        automator = SimpleCursorAutomator()
        automator.run()
    except ImportError as e:
        print("❌ Instala: pip install pyautogui keyboard")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(description="Script de recordatorio de IA")
    parser.add_argument("--interval", "-i", type=int, default=3, 
                       help="Intervalo en minutos (default: 3)")
    parser.add_argument("--max-iterations", "-m", type=int, default=None,
                       help="Número máximo de iteraciones (default: infinito)")
    parser.add_argument("--single", "-s", action="store_true",
                       help="Ejecutar una sola vez")
    parser.add_argument("--schedule", action="store_true",
                       help="Mostrar comandos para programar el script")
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Modo automático (sin clic)")
    parser.add_argument("--click", "-c", action="store_true",
                       help="Modo clic simple (un recordatorio por clic)")
    parser.add_argument("--cursor", action="store_true",
                       help="Ejecutar automatizador de Cursor")
    
    args = parser.parse_args()
    
    if args.schedule:
        create_scheduled_reminder()
        return
    
    if args.cursor:
        run_cursor_automator()
        return
    
    if args.single:
        # Ejecutar una sola vez
        print_reminder(1)
        return
    
    # Verificar modo de ejecución
    if args.click:
        # Modo clic simple
        simple_click_reminder()
    elif args.auto:
        # Modo automático (comportamiento original)
        run_reminder_script(args.interval, args.max_iterations)
    else:
        # Modo interactivo por defecto
        terminal_reminder_loop(args.interval)

match __name__:
    case "__main__":
    main() 