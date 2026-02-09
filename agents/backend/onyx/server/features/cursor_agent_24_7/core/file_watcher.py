"""
File Watcher - Monitorea archivos para detectar comandos
==========================================================

Monitorea archivos específicos o directorios para detectar cuando se escriben comandos.
Soporta tanto watchdog (eventos del sistema) como polling mode (fallback).
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Union, Awaitable, Any
from datetime import datetime
import time

# Usar watchdog para monitoreo de archivos
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None  # type: ignore
    FileSystemEventHandler = None  # type: ignore
    FileModifiedEvent = None  # type: ignore

# Usar aiofiles para lectura async
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

logger = logging.getLogger(__name__)


class CommandFileHandler(FileSystemEventHandler):
    """
    Handler para eventos de archivos de comandos.
    
    Procesa modificaciones de archivos y ejecuta callbacks cuando
    se detectan nuevos comandos.
    """
    
    def __init__(
        self,
        callback: Callable[[str], Union[None, Awaitable[None]]],
        command_file: Path
    ) -> None:
        """
        Inicializar handler.
        
        Args:
            callback: Función o coroutine a llamar cuando se detecta un comando.
            command_file: Archivo a monitorear.
        
        Raises:
            ValueError: Si callback o command_file son inválidos.
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        if not isinstance(command_file, Path):
            raise ValueError("command_file must be a Path object")
        
        self.callback: Callable[[str], Union[None, Awaitable[None]]] = callback
        self.command_file: Path = command_file
        self.last_modified: float = 0.0
        self.last_size: int = 0
        self._processing: bool = False
        
    def on_modified(self, event: Any) -> None:
        """
        Cuando se modifica el archivo.
        
        Args:
            event: Evento del sistema de archivos.
        """
        if not isinstance(event, FileModifiedEvent):
            return
        
        if event.src_path != str(self.command_file):
            return
        
        try:
            # Evitar procesar el mismo cambio múltiples veces
            current_time = time.time()
            if current_time - self.last_modified < 0.5:
                return
            
            if self._processing:
                return
            
            self.last_modified = current_time
            
            # Leer el archivo
            if not self.command_file.exists():
                return
            
            current_size = self.command_file.stat().st_size
            if current_size != self.last_size and current_size > 0:
                self.last_size = current_size
                self._processing = True
                # Ejecutar callback en el event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._process_file())
                    else:
                        loop.run_until_complete(self._process_file())
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._process_file())
                finally:
                    self._processing = False
                    
        except Exception as e:
            logger.error(f"Error processing file change: {e}", exc_info=True)
            self._processing = False
    
    async def _process_file(self) -> None:
        """
        Procesar el archivo de comandos.
        
        Lee el contenido, ejecuta el callback, y limpia el archivo.
        """
        try:
            # Leer contenido
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(
                    self.command_file,
                    'r',
                    encoding='utf-8'
                ) as f:
                    content = await f.read()
            else:
                with open(self.command_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Procesar comando si hay contenido
            if content.strip():
                # Llamar al callback con el comando
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(content.strip())
                else:
                    # Ejecutar callback síncrono
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(self.callback, content.strip())
                    except RuntimeError:
                        # Si no hay loop, ejecutar directamente
                        self.callback(content.strip())
                
                # Limpiar el archivo
                await self._clear_file()
                        
        except Exception as e:
            logger.error(f"Error reading command file: {e}", exc_info=True)
    
    async def _clear_file(self) -> None:
        """Limpiar el archivo de comandos después de procesarlo"""
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(
                    self.command_file,
                    'w',
                    encoding='utf-8'
                ) as f:
                    await f.write('')
            else:
                with open(self.command_file, 'w', encoding='utf-8') as f:
                    f.write('')
        except Exception as e:
            logger.error(f"Error clearing command file: {e}", exc_info=True)


class FileWatcher:
    """
    Monitorea archivos para detectar comandos.
    
    Soporta monitoreo de archivos específicos o directorios completos
    usando watchdog (eventos del sistema) o polling mode (fallback).
    """
    
    def __init__(
        self,
        command_file: Optional[str] = None,
        watch_dir: Optional[str] = None
    ) -> None:
        """
        Inicializar file watcher.
        
        Args:
            command_file: Ruta del archivo a monitorear (opcional).
            watch_dir: Directorio a monitorear (opcional).
        
        Raises:
            ValueError: Si ambos parámetros son None o si son inválidos.
        """
        if not command_file and not watch_dir:
            raise ValueError("Either command_file or watch_dir must be provided")
        
        self.command_file: Optional[Path] = (
            Path(command_file) if command_file else None
        )
        self.watch_dir: Optional[Path] = (
            Path(watch_dir) if watch_dir else None
        )
        self.observer: Optional[Observer] = None
        self.handler: Optional[CommandFileHandler] = None
        self.running: bool = False
        self.callback: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None
        
        # Crear archivo de comandos si no existe
        if self.command_file:
            if not self.command_file.parent.exists():
                self.command_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.command_file.exists():
                try:
                    self.command_file.write_text('', encoding='utf-8')
                    logger.info(f"Created command file: {self.command_file}")
                except Exception as e:
                    logger.error(f"Failed to create command file: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to create command file: {e}") from e
    
    def set_callback(
        self,
        callback: Callable[[str], Union[None, Awaitable[None]]]
    ) -> None:
        """
        Establecer callback para comandos.
        
        Args:
            callback: Función o coroutine a llamar cuando se detecta un comando.
        
        Raises:
            ValueError: Si callback no es callable.
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.callback = callback
    
    async def start(self) -> None:
        """
        Iniciar monitoreo.
        
        Raises:
            ValueError: Si callback no está establecido.
            RuntimeError: Si hay error al iniciar el observer.
        """
        if self.running:
            logger.warning("File watcher is already running")
            return
        
        if not self.callback:
            raise ValueError("Callback must be set before starting")
        
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, using polling mode")
            self.running = True
            asyncio.create_task(self._poll_loop())
            return
        
        logger.info("👀 Starting file watcher...")
        self.running = True
        
        try:
            if self.command_file:
                # Monitorear archivo específico
                self.handler = CommandFileHandler(
                    self.callback,
                    self.command_file
                )
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
                if not self.watch_dir.exists():
                    raise ValueError(f"Watch directory does not exist: {self.watch_dir}")
                
                self.handler = CommandFileHandler(
                    self.callback,
                    self.watch_dir / "commands.txt"  # Archivo por defecto
                )
                self.observer = Observer()
                self.observer.schedule(
                    self.handler,
                    str(self.watch_dir),
                    recursive=True
                )
                self.observer.start()
                logger.info(f"📁 Monitoring directory: {self.watch_dir}")
        
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}", exc_info=True)
            self.running = False
            raise RuntimeError(f"Failed to start file watcher: {e}") from e
    
    async def stop(self) -> None:
        """Detener monitoreo"""
        if not self.running:
            return
        
        logger.info("🛑 Stopping file watcher...")
        self.running = False
        
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5.0)
            except Exception as e:
                logger.error(f"Error stopping observer: {e}", exc_info=True)
            finally:
                self.observer = None
        
        self.handler = None
    
    async def _poll_loop(self) -> None:
        """
        Loop de polling cuando watchdog no está disponible.
        
        Verifica el archivo periódicamente para detectar cambios.
        """
        if not self.command_file:
            logger.warning("Polling mode requires command_file")
            return
        
        last_size = 0
        
        while self.running:
            try:
                if self.command_file.exists():
                    current_size = self.command_file.stat().st_size
                    if current_size != last_size and current_size > 0:
                        # Leer el archivo
                        if AIOFILES_AVAILABLE:
                            async with aiofiles.open(
                                self.command_file,
                                'r',
                                encoding='utf-8'
                            ) as f:
                                content = await f.read()
                        else:
                            with open(self.command_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                        
                        if content.strip() and self.callback:
                            try:
                                if asyncio.iscoroutinefunction(self.callback):
                                    await self.callback(content.strip())
                                else:
                                    self.callback(content.strip())
                                
                                # Limpiar archivo
                                if AIOFILES_AVAILABLE:
                                    async with aiofiles.open(
                                        self.command_file,
                                        'w',
                                        encoding='utf-8'
                                    ) as f:
                                        await f.write('')
                                else:
                                    with open(self.command_file, 'w', encoding='utf-8') as f:
                                        f.write('')
                            except Exception as e:
                                logger.error(f"Error in callback: {e}", exc_info=True)
                        
                        last_size = current_size
                
                await asyncio.sleep(1.0)  # Poll cada segundo
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in poll loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    def get_command_file_path(self) -> Optional[Path]:
        """
        Obtener ruta del archivo de comandos.
        
        Returns:
            Path del archivo de comandos o None si no está configurado.
        """
        return self.command_file
