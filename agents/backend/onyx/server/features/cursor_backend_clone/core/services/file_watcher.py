"""
File Watcher - Monitorea archivos para detectar comandos
==========================================================

Monitorea archivos específicos o directorios para detectar cuando se escriben comandos.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime
import time
import sys

# Usar watchdog para monitoreo de archivos
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# Usar aiofiles para lectura async
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

logger = logging.getLogger(__name__)


class CommandFileHandler(FileSystemEventHandler):
    """Handler para eventos de archivos de comandos"""
    
    def __init__(self, callback: Callable, command_file: Path):
        self.callback = callback
        self.command_file = command_file
        self.last_modified = 0
        self.last_size = 0
        
    def on_modified(self, event):
        """Cuando se modifica el archivo"""
        if event.src_path == str(self.command_file):
            try:
                # Evitar procesar el mismo cambio múltiples veces
                current_time = time.time()
                if current_time - self.last_modified < 0.5:
                    return
                    
                self.last_modified = current_time
                
                # Leer el archivo
                if self.command_file.exists():
                    current_size = self.command_file.stat().st_size
                    if current_size != self.last_size:
                        self.last_size = current_size
                        # Ejecutar callback en el event loop
                        asyncio.create_task(self._process_file())
                        
            except Exception as e:
                logger.error(f"Error processing file change: {e}")
    
    async def _process_file(self):
        """Procesar el archivo de comandos"""
        try:
            # Obtener el event loop actual
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.command_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                with open(self.command_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Limpiar el archivo después de leer
            if content.strip():
                # Llamar al callback con el comando
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(content.strip())
                else:
                    # Ejecutar callback síncrono en el loop
                    loop.call_soon_threadsafe(self.callback, content.strip())
                
                # Limpiar el archivo
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(self.command_file, 'w', encoding='utf-8') as f:
                        await f.write('')
                else:
                    with open(self.command_file, 'w', encoding='utf-8') as f:
                        f.write('')
                        
        except Exception as e:
            logger.error(f"Error reading command file: {e}")


class FileWatcher:
    """Monitorea archivos para detectar comandos"""
    
    def __init__(self, command_file: Optional[str] = None, watch_dir: Optional[str] = None):
        self.command_file = Path(command_file) if command_file else None
        self.watch_dir = Path(watch_dir) if watch_dir else None
        self.observer: Optional[Observer] = None
        self.handler: Optional[CommandFileHandler] = None
        self.running = False
        self.callback: Optional[Callable] = None
        
        # Crear archivo de comandos si no existe
        if self.command_file and not self.command_file.exists():
            self.command_file.parent.mkdir(parents=True, exist_ok=True)
            self.command_file.write_text('', encoding='utf-8')
            logger.info(f"Created command file: {self.command_file}")
    
    def set_callback(self, callback: Callable):
        """Establecer callback para comandos"""
        self.callback = callback
    
    async def start(self):
        """Iniciar monitoreo"""
        if self.running:
            logger.warning("File watcher is already running")
            return
        
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, using polling mode")
            self.running = True
            asyncio.create_task(self._poll_loop())
            return
        
        if not self.callback:
            raise ValueError("Callback must be set before starting")
        
        logger.info(f"👀 Starting file watcher...")
        self.running = True
        
        if self.command_file:
            # Monitorear archivo específico
            self.handler = CommandFileHandler(self.callback, self.command_file)
            self.observer = Observer()
            self.observer.schedule(
                self.handler,
                str(self.command_file.parent),
                recursive=False
            )
            self.observer.start()
            logger.info(f"📝 Monitoring command file: {self.command_file}")
        
        elif self.watch_dir:
            # Monitorear directorio
            self.observer = Observer()
            self.observer.schedule(
                self.handler or CommandFileHandler(self.callback, Path(self.watch_dir)),
                str(self.watch_dir),
                recursive=True
            )
            self.observer.start()
            logger.info(f"📁 Monitoring directory: {self.watch_dir}")
    
    async def stop(self):
        """Detener monitoreo"""
        if not self.running:
            return
        
        logger.info("🛑 Stopping file watcher...")
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.handler = None
    
    async def _poll_loop(self):
        """Loop de polling cuando watchdog no está disponible"""
        if not self.command_file:
            return
        
        last_size = 0
        
        while self.running:
            try:
                if self.command_file.exists():
                    current_size = self.command_file.stat().st_size
                    if current_size != last_size and current_size > 0:
                        # Leer el archivo
                        if AIOFILES_AVAILABLE:
                            async with aiofiles.open(self.command_file, 'r', encoding='utf-8') as f:
                                content = await f.read()
                        else:
                            with open(self.command_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                        
                        if content.strip() and self.callback:
                            if asyncio.iscoroutinefunction(self.callback):
                                await self.callback(content.strip())
                            else:
                                self.callback(content.strip())
                            
                            # Limpiar archivo
                            if AIOFILES_AVAILABLE:
                                async with aiofiles.open(self.command_file, 'w', encoding='utf-8') as f:
                                    await f.write('')
                            else:
                                with open(self.command_file, 'w', encoding='utf-8') as f:
                                    f.write('')
                        
                        last_size = current_size
                
                await asyncio.sleep(1.0)  # Poll cada segundo
                
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
                await asyncio.sleep(5)
    
    def get_command_file_path(self) -> Optional[Path]:
        """Obtener ruta del archivo de comandos"""
        return self.command_file

