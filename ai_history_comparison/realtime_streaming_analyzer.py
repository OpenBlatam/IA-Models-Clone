"""
Advanced Real-time Streaming Data Analysis System
Sistema avanzado de análisis de datos en tiempo real y streaming
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Streaming and real-time imports
try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Data processing imports
import queue
import threading
from collections import deque, defaultdict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Tipos de stream"""
    KAFKA = "kafka"
    REDIS = "redis"
    WEBSOCKET = "websocket"
    HTTP = "http"
    FILE = "file"
    DATABASE = "database"

class ProcessingType(Enum):
    """Tipos de procesamiento"""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    WINDOWED = "windowed"
    EVENT_DRIVEN = "event_driven"

class WindowType(Enum):
    """Tipos de ventana"""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    GLOBAL = "global"

@dataclass
class StreamConfig:
    """Configuración de stream"""
    stream_id: str
    stream_type: StreamType
    processing_type: ProcessingType
    window_size: int = 1000
    window_slide: int = 100
    window_type: WindowType = WindowType.TUMBLING
    batch_size: int = 100
    processing_interval: float = 1.0
    buffer_size: int = 10000
    max_latency: float = 5.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class StreamData:
    """Datos de stream"""
    id: str
    stream_id: str
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False

@dataclass
class StreamMetrics:
    """Métricas de stream"""
    stream_id: str
    total_messages: int = 0
    processed_messages: int = 0
    failed_messages: int = 0
    average_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class StreamAlert:
    """Alerta de stream"""
    id: str
    stream_id: str
    alert_type: str
    severity: str
    message: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedRealtimeStreamingAnalyzer:
    """
    Analizador avanzado de datos en tiempo real y streaming
    """
    
    def __init__(
        self,
        enable_kafka: bool = True,
        enable_redis: bool = True,
        enable_websockets: bool = True,
        max_workers: int = 10,
        default_buffer_size: int = 10000
    ):
        self.enable_kafka = enable_kafka and KAFKA_AVAILABLE
        self.enable_redis = enable_redis and REDIS_AVAILABLE
        self.enable_websockets = enable_websockets and WEBSOCKETS_AVAILABLE
        self.max_workers = max_workers
        self.default_buffer_size = default_buffer_size
        
        # Almacenamiento
        self.stream_configs: Dict[str, StreamConfig] = {}
        self.stream_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=default_buffer_size))
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        self.stream_alerts: Dict[str, List[StreamAlert]] = defaultdict(list)
        
        # Procesadores
        self.stream_processors: Dict[str, Callable] = {}
        self.alert_handlers: Dict[str, Callable] = {}
        
        # Threading y concurrencia
        self.stream_threads: Dict[str, threading.Thread] = {}
        self.processing_queues: Dict[str, queue.Queue] = {}
        self.running_streams: Dict[str, bool] = {}
        
        # Configuración
        self.config = {
            "default_window_size": 1000,
            "default_window_slide": 100,
            "default_batch_size": 100,
            "default_processing_interval": 1.0,
            "max_latency_threshold": 5.0,
            "error_rate_threshold": 0.05,
            "throughput_threshold": 1000.0,
            "alert_cooldown": 60.0  # segundos
        }
        
        # Inicializar componentes
        self._initialize_components()
        
        logger.info("Advanced Realtime Streaming Analyzer inicializado")
    
    def _initialize_components(self):
        """Inicializar componentes del sistema"""
        try:
            # Inicializar Redis si está disponible
            if self.enable_redis:
                try:
                    self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                    self.redis_client.ping()
                    logger.info("Redis conectado exitosamente")
                except Exception as e:
                    logger.warning(f"Redis no disponible: {e}")
                    self.enable_redis = False
            
            # Inicializar Kafka si está disponible
            if self.enable_kafka:
                try:
                    # Verificar conexión a Kafka
                    from kafka.admin import KafkaAdminClient
                    admin_client = KafkaAdminClient(bootstrap_servers=['localhost:9092'])
                    admin_client.list_consumer_groups()
                    logger.info("Kafka conectado exitosamente")
                except Exception as e:
                    logger.warning(f"Kafka no disponible: {e}")
                    self.enable_kafka = False
            
            logger.info("Componentes inicializados")
            
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
    
    async def create_stream(
        self,
        stream_id: str,
        stream_type: StreamType,
        processing_type: ProcessingType,
        config: Optional[Dict[str, Any]] = None
    ) -> StreamConfig:
        """
        Crear stream
        
        Args:
            stream_id: ID del stream
            stream_type: Tipo de stream
            processing_type: Tipo de procesamiento
            config: Configuración adicional
            
        Returns:
            Configuración del stream
        """
        try:
            logger.info(f"Creando stream: {stream_id} ({stream_type.value})")
            
            # Crear configuración
            stream_config = StreamConfig(
                stream_id=stream_id,
                stream_type=stream_type,
                processing_type=processing_type,
                window_size=config.get("window_size", self.config["default_window_size"]) if config else self.config["default_window_size"],
                window_slide=config.get("window_slide", self.config["default_window_slide"]) if config else self.config["default_window_slide"],
                window_type=config.get("window_type", WindowType.TUMBLING) if config else WindowType.TUMBLING,
                batch_size=config.get("batch_size", self.config["default_batch_size"]) if config else self.config["default_batch_size"],
                processing_interval=config.get("processing_interval", self.config["default_processing_interval"]) if config else self.config["default_processing_interval"],
                buffer_size=config.get("buffer_size", self.default_buffer_size) if config else self.default_buffer_size,
                max_latency=config.get("max_latency", self.config["max_latency_threshold"]) if config else self.config["max_latency_threshold"]
            )
            
            # Almacenar configuración
            self.stream_configs[stream_id] = stream_config
            
            # Inicializar métricas
            self.stream_metrics[stream_id] = StreamMetrics(stream_id=stream_id)
            
            # Crear cola de procesamiento
            self.processing_queues[stream_id] = queue.Queue(maxsize=stream_config.buffer_size)
            
            # Inicializar estado
            self.running_streams[stream_id] = False
            
            logger.info(f"Stream creado exitosamente: {stream_id}")
            return stream_config
            
        except Exception as e:
            logger.error(f"Error creando stream: {e}")
            raise
    
    async def start_stream(
        self,
        stream_id: str,
        data_source: str,
        processor: Optional[Callable] = None
    ) -> bool:
        """
        Iniciar stream
        
        Args:
            stream_id: ID del stream
            data_source: Fuente de datos
            processor: Función de procesamiento
            
        Returns:
            True si se inició exitosamente
        """
        try:
            if stream_id not in self.stream_configs:
                raise ValueError(f"Stream {stream_id} no encontrado")
            
            if self.running_streams.get(stream_id, False):
                logger.warning(f"Stream {stream_id} ya está ejecutándose")
                return True
            
            config = self.stream_configs[stream_id]
            
            logger.info(f"Iniciando stream: {stream_id}")
            
            # Configurar procesador
            if processor:
                self.stream_processors[stream_id] = processor
            else:
                self.stream_processors[stream_id] = self._default_processor
            
            # Iniciar thread de stream según el tipo
            if config.stream_type == StreamType.KAFKA and self.enable_kafka:
                thread = threading.Thread(
                    target=self._kafka_stream_worker,
                    args=(stream_id, data_source),
                    daemon=True
                )
            elif config.stream_type == StreamType.REDIS and self.enable_redis:
                thread = threading.Thread(
                    target=self._redis_stream_worker,
                    args=(stream_id, data_source),
                    daemon=True
                )
            elif config.stream_type == StreamType.WEBSOCKET and self.enable_websockets:
                thread = threading.Thread(
                    target=self._websocket_stream_worker,
                    args=(stream_id, data_source),
                    daemon=True
                )
            else:
                # Stream simulado para testing
                thread = threading.Thread(
                    target=self._simulated_stream_worker,
                    args=(stream_id, data_source),
                    daemon=True
                )
            
            # Iniciar thread de procesamiento
            processing_thread = threading.Thread(
                target=self._processing_worker,
                args=(stream_id,),
                daemon=True
            )
            
            # Marcar como ejecutándose
            self.running_streams[stream_id] = True
            
            # Iniciar threads
            thread.start()
            processing_thread.start()
            
            # Almacenar threads
            self.stream_threads[stream_id] = thread
            
            logger.info(f"Stream iniciado exitosamente: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando stream: {e}")
            self.running_streams[stream_id] = False
            return False
    
    def _kafka_stream_worker(self, stream_id: str, topic: str):
        """Worker para stream de Kafka"""
        try:
            config = self.stream_configs[stream_id]
            
            # Crear consumer
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id=f'stream_analyzer_{stream_id}',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            logger.info(f"Kafka consumer iniciado para stream: {stream_id}")
            
            for message in consumer:
                if not self.running_streams.get(stream_id, False):
                    break
                
                # Crear datos de stream
                stream_data = StreamData(
                    id=f"{stream_id}_{int(time.time() * 1000)}",
                    stream_id=stream_id,
                    timestamp=datetime.now(),
                    data=message.value,
                    metadata={
                        "topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset
                    }
                )
                
                # Agregar a cola de procesamiento
                try:
                    self.processing_queues[stream_id].put(stream_data, timeout=1.0)
                except queue.Full:
                    logger.warning(f"Cola llena para stream: {stream_id}")
                
                # Actualizar métricas
                self._update_stream_metrics(stream_id, stream_data)
            
            consumer.close()
            
        except Exception as e:
            logger.error(f"Error en Kafka stream worker: {e}")
        finally:
            self.running_streams[stream_id] = False
    
    def _redis_stream_worker(self, stream_id: str, stream_name: str):
        """Worker para stream de Redis"""
        try:
            config = self.stream_configs[stream_id]
            
            logger.info(f"Redis stream worker iniciado para: {stream_id}")
            
            while self.running_streams.get(stream_id, False):
                try:
                    # Leer del stream de Redis
                    messages = self.redis_client.xread({stream_name: '$'}, count=config.batch_size, block=1000)
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            # Crear datos de stream
                            stream_data = StreamData(
                                id=f"{stream_id}_{msg_id}",
                                stream_id=stream_id,
                                timestamp=datetime.now(),
                                data=fields,
                                metadata={
                                    "stream": stream.decode(),
                                    "message_id": msg_id.decode()
                                }
                            )
                            
                            # Agregar a cola de procesamiento
                            try:
                                self.processing_queues[stream_id].put(stream_data, timeout=1.0)
                            except queue.Full:
                                logger.warning(f"Cola llena para stream: {stream_id}")
                            
                            # Actualizar métricas
                            self._update_stream_metrics(stream_id, stream_data)
                
                except Exception as e:
                    if self.running_streams.get(stream_id, False):
                        logger.error(f"Error leyendo Redis stream: {e}")
                        time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error en Redis stream worker: {e}")
        finally:
            self.running_streams[stream_id] = False
    
    def _websocket_stream_worker(self, stream_id: str, url: str):
        """Worker para stream de WebSocket"""
        try:
            import asyncio
            
            async def websocket_worker():
                try:
                    async with websockets.connect(url) as websocket:
                        logger.info(f"WebSocket conectado para stream: {stream_id}")
                        
                        while self.running_streams.get(stream_id, False):
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                
                                # Crear datos de stream
                                stream_data = StreamData(
                                    id=f"{stream_id}_{int(time.time() * 1000)}",
                                    stream_id=stream_id,
                                    timestamp=datetime.now(),
                                    data=json.loads(message),
                                    metadata={"source": "websocket"}
                                )
                                
                                # Agregar a cola de procesamiento
                                try:
                                    self.processing_queues[stream_id].put(stream_data, timeout=1.0)
                                except queue.Full:
                                    logger.warning(f"Cola llena para stream: {stream_id}")
                                
                                # Actualizar métricas
                                self._update_stream_metrics(stream_id, stream_data)
                                
                            except asyncio.TimeoutError:
                                continue
                            except Exception as e:
                                logger.error(f"Error procesando mensaje WebSocket: {e}")
                
                except Exception as e:
                    logger.error(f"Error conectando WebSocket: {e}")
            
            # Ejecutar worker asíncrono
            asyncio.run(websocket_worker())
            
        except Exception as e:
            logger.error(f"Error en WebSocket stream worker: {e}")
        finally:
            self.running_streams[stream_id] = False
    
    def _simulated_stream_worker(self, stream_id: str, data_type: str):
        """Worker para stream simulado (testing)"""
        try:
            config = self.stream_configs[stream_id]
            
            logger.info(f"Stream simulado iniciado para: {stream_id}")
            
            message_count = 0
            while self.running_streams.get(stream_id, False):
                # Generar datos simulados
                if data_type == "sensor":
                    data = {
                        "sensor_id": f"sensor_{message_count % 10}",
                        "value": np.random.normal(50, 10),
                        "timestamp": datetime.now().isoformat(),
                        "location": f"location_{message_count % 5}"
                    }
                elif data_type == "log":
                    data = {
                        "level": np.random.choice(["INFO", "WARNING", "ERROR"]),
                        "message": f"Log message {message_count}",
                        "timestamp": datetime.now().isoformat(),
                        "service": f"service_{message_count % 3}"
                    }
                else:
                    data = {
                        "id": message_count,
                        "value": np.random.random(),
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Crear datos de stream
                stream_data = StreamData(
                    id=f"{stream_id}_{message_count}",
                    stream_id=stream_id,
                    timestamp=datetime.now(),
                    data=data,
                    metadata={"source": "simulated", "type": data_type}
                )
                
                # Agregar a cola de procesamiento
                try:
                    self.processing_queues[stream_id].put(stream_data, timeout=1.0)
                except queue.Full:
                    logger.warning(f"Cola llena para stream: {stream_id}")
                
                # Actualizar métricas
                self._update_stream_metrics(stream_id, stream_data)
                
                message_count += 1
                time.sleep(config.processing_interval)
            
        except Exception as e:
            logger.error(f"Error en stream simulado: {e}")
        finally:
            self.running_streams[stream_id] = False
    
    def _processing_worker(self, stream_id: str):
        """Worker de procesamiento"""
        try:
            config = self.stream_configs[stream_id]
            processor = self.stream_processors.get(stream_id, self._default_processor)
            
            logger.info(f"Worker de procesamiento iniciado para: {stream_id}")
            
            batch = []
            last_processing_time = time.time()
            
            while self.running_streams.get(stream_id, False):
                try:
                    # Obtener datos de la cola
                    try:
                        stream_data = self.processing_queues[stream_id].get(timeout=1.0)
                        batch.append(stream_data)
                    except queue.Empty:
                        continue
                    
                    # Procesar según el tipo
                    current_time = time.time()
                    
                    if config.processing_type == ProcessingType.STREAMING:
                        # Procesamiento inmediato
                        try:
                            result = processor(stream_data.data, stream_data.metadata)
                            stream_data.processed = True
                            self._handle_processing_result(stream_id, result, stream_data)
                        except Exception as e:
                            logger.error(f"Error procesando datos: {e}")
                            self._update_error_metrics(stream_id)
                    
                    elif config.processing_type == ProcessingType.BATCH:
                        # Procesamiento por lotes
                        if len(batch) >= config.batch_size or (current_time - last_processing_time) >= config.processing_interval:
                            try:
                                batch_data = [item.data for item in batch]
                                batch_metadata = [item.metadata for item in batch]
                                
                                result = processor(batch_data, batch_metadata)
                                
                                # Marcar como procesados
                                for item in batch:
                                    item.processed = True
                                
                                self._handle_processing_result(stream_id, result, batch)
                                batch = []
                                last_processing_time = current_time
                                
                            except Exception as e:
                                logger.error(f"Error procesando lote: {e}")
                                self._update_error_metrics(stream_id)
                    
                    elif config.processing_type == ProcessingType.WINDOWED:
                        # Procesamiento por ventanas
                        if len(batch) >= config.window_size:
                            try:
                                window_data = batch[-config.window_size:]
                                window_batch_data = [item.data for item in window_data]
                                window_batch_metadata = [item.metadata for item in window_data]
                                
                                result = processor(window_batch_data, window_batch_metadata)
                                
                                # Marcar como procesados
                                for item in window_data:
                                    item.processed = True
                                
                                self._handle_processing_result(stream_id, result, window_data)
                                
                                # Mantener solo los datos no procesados
                                batch = [item for item in batch if not item.processed]
                                
                            except Exception as e:
                                logger.error(f"Error procesando ventana: {e}")
                                self._update_error_metrics(stream_id)
                
                except Exception as e:
                    logger.error(f"Error en worker de procesamiento: {e}")
                    time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error en worker de procesamiento: {e}")
    
    def _default_processor(self, data: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Procesador por defecto"""
        try:
            if isinstance(data, list):
                # Procesamiento de lote
                return {
                    "processed_count": len(data),
                    "timestamp": datetime.now().isoformat(),
                    "type": "batch_processing"
                }
            else:
                # Procesamiento individual
                return {
                    "data": data,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat(),
                    "type": "individual_processing"
                }
        except Exception as e:
            logger.error(f"Error en procesador por defecto: {e}")
            return {"error": str(e)}
    
    def _handle_processing_result(self, stream_id: str, result: Any, stream_data: Union[StreamData, List[StreamData]]):
        """Manejar resultado del procesamiento"""
        try:
            # Almacenar datos procesados
            if isinstance(stream_data, list):
                for item in stream_data:
                    self.stream_data[stream_id].append(item)
            else:
                self.stream_data[stream_id].append(stream_data)
            
            # Verificar alertas
            self._check_alerts(stream_id, result)
            
        except Exception as e:
            logger.error(f"Error manejando resultado: {e}")
    
    def _update_stream_metrics(self, stream_id: str, stream_data: StreamData):
        """Actualizar métricas del stream"""
        try:
            metrics = self.stream_metrics[stream_id]
            
            # Actualizar contadores
            metrics.total_messages += 1
            
            # Calcular latencia
            latency = (datetime.now() - stream_data.timestamp).total_seconds()
            metrics.average_latency = (metrics.average_latency * (metrics.total_messages - 1) + latency) / metrics.total_messages
            
            # Calcular throughput (mensajes por segundo)
            time_diff = (datetime.now() - metrics.last_update).total_seconds()
            if time_diff > 0:
                metrics.throughput = metrics.total_messages / time_diff
            
            # Actualizar timestamp
            metrics.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error actualizando métricas: {e}")
    
    def _update_error_metrics(self, stream_id: str):
        """Actualizar métricas de error"""
        try:
            metrics = self.stream_metrics[stream_id]
            metrics.failed_messages += 1
            metrics.error_rate = metrics.failed_messages / max(metrics.total_messages, 1)
            
        except Exception as e:
            logger.error(f"Error actualizando métricas de error: {e}")
    
    def _check_alerts(self, stream_id: str, result: Any):
        """Verificar alertas"""
        try:
            metrics = self.stream_metrics[stream_id]
            config = self.stream_configs[stream_id]
            
            # Verificar latencia
            if metrics.average_latency > config.max_latency:
                self._create_alert(
                    stream_id,
                    "high_latency",
                    "warning",
                    f"Latencia alta: {metrics.average_latency:.2f}s",
                    config.max_latency,
                    metrics.average_latency
                )
            
            # Verificar tasa de error
            if metrics.error_rate > self.config["error_rate_threshold"]:
                self._create_alert(
                    stream_id,
                    "high_error_rate",
                    "critical",
                    f"Tasa de error alta: {metrics.error_rate:.2%}",
                    self.config["error_rate_threshold"],
                    metrics.error_rate
                )
            
            # Verificar throughput
            if metrics.throughput < self.config["throughput_threshold"]:
                self._create_alert(
                    stream_id,
                    "low_throughput",
                    "warning",
                    f"Throughput bajo: {metrics.throughput:.2f} msg/s",
                    self.config["throughput_threshold"],
                    metrics.throughput
                )
            
        except Exception as e:
            logger.error(f"Error verificando alertas: {e}")
    
    def _create_alert(
        self,
        stream_id: str,
        alert_type: str,
        severity: str,
        message: str,
        threshold: float,
        current_value: float
    ):
        """Crear alerta"""
        try:
            # Verificar cooldown
            alerts = self.stream_alerts[stream_id]
            if alerts:
                last_alert = alerts[-1]
                time_diff = (datetime.now() - last_alert.timestamp).total_seconds()
                if time_diff < self.config["alert_cooldown"]:
                    return
            
            # Crear alerta
            alert = StreamAlert(
                id=f"alert_{stream_id}_{int(time.time())}",
                stream_id=stream_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                threshold=threshold,
                current_value=current_value
            )
            
            # Almacenar alerta
            alerts.append(alert)
            
            # Mantener solo las últimas 100 alertas
            if len(alerts) > 100:
                alerts.pop(0)
            
            # Ejecutar handler si existe
            handler = self.alert_handlers.get(stream_id)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error ejecutando alert handler: {e}")
            
            logger.warning(f"Alerta creada: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error creando alerta: {e}")
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Detener stream"""
        try:
            if stream_id not in self.stream_configs:
                raise ValueError(f"Stream {stream_id} no encontrado")
            
            logger.info(f"Deteniendo stream: {stream_id}")
            
            # Marcar como detenido
            self.running_streams[stream_id] = False
            
            # Esperar a que termine el thread
            if stream_id in self.stream_threads:
                thread = self.stream_threads[stream_id]
                thread.join(timeout=5.0)
                del self.stream_threads[stream_id]
            
            logger.info(f"Stream detenido: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deteniendo stream: {e}")
            return False
    
    async def get_stream_metrics(self, stream_id: str) -> Optional[StreamMetrics]:
        """Obtener métricas del stream"""
        try:
            if stream_id not in self.stream_metrics:
                return None
            
            return self.stream_metrics[stream_id]
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return None
    
    async def get_stream_alerts(self, stream_id: str) -> List[StreamAlert]:
        """Obtener alertas del stream"""
        try:
            return self.stream_alerts.get(stream_id, [])
            
        except Exception as e:
            logger.error(f"Error obteniendo alertas: {e}")
            return []
    
    async def set_alert_handler(self, stream_id: str, handler: Callable):
        """Configurar handler de alertas"""
        try:
            self.alert_handlers[stream_id] = handler
            logger.info(f"Alert handler configurado para stream: {stream_id}")
            
        except Exception as e:
            logger.error(f"Error configurando alert handler: {e}")
    
    async def get_stream_data(
        self,
        stream_id: str,
        limit: int = 100,
        processed_only: bool = False
    ) -> List[StreamData]:
        """Obtener datos del stream"""
        try:
            if stream_id not in self.stream_data:
                return []
            
            data = list(self.stream_data[stream_id])
            
            # Filtrar por procesados si se solicita
            if processed_only:
                data = [item for item in data if item.processed]
            
            # Limitar resultados
            return data[-limit:] if limit > 0 else data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del stream: {e}")
            return []
    
    async def get_realtime_summary(self) -> Dict[str, Any]:
        """Obtener resumen en tiempo real"""
        try:
            return {
                "total_streams": len(self.stream_configs),
                "active_streams": sum(1 for running in self.running_streams.values() if running),
                "total_messages": sum(metrics.total_messages for metrics in self.stream_metrics.values()),
                "total_alerts": sum(len(alerts) for alerts in self.stream_alerts.values()),
                "stream_types": {
                    stream_type.value: len([c for c in self.stream_configs.values() if c.stream_type == stream_type])
                    for stream_type in StreamType
                },
                "processing_types": {
                    processing_type.value: len([c for c in self.stream_configs.values() if c.processing_type == processing_type])
                    for processing_type in ProcessingType
                },
                "capabilities": {
                    "kafka": self.enable_kafka,
                    "redis": self.enable_redis,
                    "websockets": self.enable_websockets
                },
                "stream_metrics": {
                    stream_id: {
                        "total_messages": metrics.total_messages,
                        "processed_messages": metrics.processed_messages,
                        "failed_messages": metrics.failed_messages,
                        "average_latency": metrics.average_latency,
                        "throughput": metrics.throughput,
                        "error_rate": metrics.error_rate
                    }
                    for stream_id, metrics in self.stream_metrics.items()
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen en tiempo real: {e}")
            return {}
    
    async def export_realtime_data(self, filepath: str = None) -> str:
        """Exportar datos en tiempo real"""
        try:
            if filepath is None:
                filepath = f"exports/realtime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "stream_configs": {
                    config_id: {
                        "stream_id": config.stream_id,
                        "stream_type": config.stream_type.value,
                        "processing_type": config.processing_type.value,
                        "window_size": config.window_size,
                        "window_slide": config.window_slide,
                        "window_type": config.window_type.value,
                        "batch_size": config.batch_size,
                        "processing_interval": config.processing_interval,
                        "buffer_size": config.buffer_size,
                        "max_latency": config.max_latency,
                        "created_at": config.created_at.isoformat()
                    }
                    for config_id, config in self.stream_configs.items()
                },
                "stream_metrics": {
                    stream_id: {
                        "total_messages": metrics.total_messages,
                        "processed_messages": metrics.processed_messages,
                        "failed_messages": metrics.failed_messages,
                        "average_latency": metrics.average_latency,
                        "throughput": metrics.throughput,
                        "error_rate": metrics.error_rate,
                        "last_update": metrics.last_update.isoformat()
                    }
                    for stream_id, metrics in self.stream_metrics.items()
                },
                "stream_alerts": {
                    stream_id: [
                        {
                            "id": alert.id,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "message": alert.message,
                            "threshold": alert.threshold,
                            "current_value": alert.current_value,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in alerts
                    ]
                    for stream_id, alerts in self.stream_alerts.items()
                },
                "summary": await self.get_realtime_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Datos en tiempo real exportados a {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando datos en tiempo real: {e}")
            raise
























