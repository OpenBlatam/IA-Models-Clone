"""
Real-time Processing System
===========================
Sistema de procesamiento en tiempo real con WebSockets y streaming
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue


class ProcessingStatus(Enum):
    """Estados del procesamiento"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingUpdate:
    """Actualización de procesamiento"""
    status: ProcessingStatus
    progress: float  # 0-100
    message: str
    timestamp: float
    data: Optional[Dict[str, Any]] = None


class RealTimeProcessor:
    """
    Procesador en tiempo real con actualizaciones de progreso
    """
    
    def __init__(self):
        self.active_processes: Dict[str, Dict] = {}
        self.update_callbacks: Dict[str, List[Callable]] = {}
        self.update_queue: Queue = Queue()
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Iniciar el procesador"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._process_updates, daemon=True)
            self.update_thread.start()
    
    def stop(self):
        """Detener el procesador"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
    
    def register_callback(self, process_id: str, callback: Callable[[ProcessingUpdate], None]):
        """Registrar callback para actualizaciones"""
        if process_id not in self.update_callbacks:
            self.update_callbacks[process_id] = []
        self.update_callbacks[process_id].append(callback)
    
    def unregister_callback(self, process_id: str, callback: Callable):
        """Eliminar callback"""
        if process_id in self.update_callbacks:
            if callback in self.update_callbacks[process_id]:
                self.update_callbacks[process_id].remove(callback)
    
    def process_with_updates(
        self,
        process_id: str,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Procesar con actualizaciones en tiempo real
        
        Args:
            process_id: ID único del proceso
            process_func: Función de procesamiento
            *args: Argumentos para la función
            **kwargs: Argumentos clave para la función
        """
        self.active_processes[process_id] = {
            'start_time': time.time(),
            'status': ProcessingStatus.PENDING
        }
        
        try:
            # Preprocessing
            self._send_update(process_id, ProcessingStatus.PREPROCESSING, 10, "Preparando imagen...")
            
            # Generación con progreso
            self._send_update(process_id, ProcessingStatus.GENERATING, 30, "Generando nueva ropa...")
            
            # Ejecutar función de procesamiento
            result = process_func(*args, **kwargs)
            
            # Postprocessing
            self._send_update(process_id, ProcessingStatus.POSTPROCESSING, 90, "Aplicando mejoras finales...")
            
            # Completado
            self._send_update(process_id, ProcessingStatus.COMPLETED, 100, "Procesamiento completado")
            
            return result
            
        except Exception as e:
            self._send_update(
                process_id,
                ProcessingStatus.FAILED,
                0,
                f"Error: {str(e)}"
            )
            raise
        finally:
            if process_id in self.active_processes:
                del self.active_processes[process_id]
    
    def _send_update(
        self,
        process_id: str,
        status: ProcessingStatus,
        progress: float,
        message: str,
        data: Optional[Dict] = None
    ):
        """Enviar actualización"""
        update = ProcessingUpdate(
            status=status,
            progress=progress,
            message=message,
            timestamp=time.time(),
            data=data
        )
        
        self.update_queue.put((process_id, update))
    
    def _process_updates(self):
        """Procesar actualizaciones en thread separado"""
        while self.running:
            try:
                if not self.update_queue.empty():
                    process_id, update = self.update_queue.get(timeout=0.1)
                    
                    # Actualizar estado
                    if process_id in self.active_processes:
                        self.active_processes[process_id]['status'] = update.status
                        self.active_processes[process_id]['progress'] = update.progress
                    
                    # Llamar callbacks
                    if process_id in self.update_callbacks:
                        for callback in self.update_callbacks[process_id]:
                            try:
                                callback(update)
                            except Exception as e:
                                print(f"Error en callback: {e}")
                
                time.sleep(0.05)  # 20 updates por segundo
            except Exception as e:
                print(f"Error procesando actualizaciones: {e}")
                time.sleep(0.1)
    
    def get_status(self, process_id: str) -> Optional[Dict]:
        """Obtener estado del proceso"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            elapsed = time.time() - process['start_time']
            return {
                'status': process['status'].value,
                'progress': process.get('progress', 0),
                'elapsed_time': elapsed
            }
        return None
    
    def cancel_process(self, process_id: str):
        """Cancelar proceso"""
        if process_id in self.active_processes:
            self._send_update(
                process_id,
                ProcessingStatus.CANCELLED,
                0,
                "Procesamiento cancelado"
            )
            del self.active_processes[process_id]


class WebSocketHandler:
    """
    Manejador de WebSockets para actualizaciones en tiempo real
    """
    
    def __init__(self, processor: RealTimeProcessor):
        self.processor = processor
        self.connections: Dict[str, Any] = {}
    
    async def handle_connection(self, websocket, process_id: str):
        """Manejar conexión WebSocket"""
        self.connections[process_id] = websocket
        
        # Registrar callback para enviar actualizaciones
        def send_update(update: ProcessingUpdate):
            asyncio.create_task(self._send_websocket_update(websocket, update))
        
        self.processor.register_callback(process_id, send_update)
        
        try:
            # Mantener conexión viva
            while True:
                await asyncio.sleep(1)
                # Ping para mantener conexión
                await websocket.ping()
        except Exception:
            pass
        finally:
            self.processor.unregister_callback(process_id, send_update)
            if process_id in self.connections:
                del self.connections[process_id]
    
    async def _send_websocket_update(self, websocket, update: ProcessingUpdate):
        """Enviar actualización por WebSocket"""
        try:
            message = {
                'status': update.status.value,
                'progress': update.progress,
                'message': update.message,
                'timestamp': update.timestamp,
                'data': update.data
            }
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error enviando actualización WebSocket: {e}")


# Instancia global
realtime_processor = RealTimeProcessor()

