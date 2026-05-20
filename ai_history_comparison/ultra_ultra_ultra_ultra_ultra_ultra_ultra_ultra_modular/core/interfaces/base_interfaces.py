"""
Base Interfaces - Interfaces Base
================================

Interfaces fundamentales que definen contratos para todos los componentes del sistema.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
import asyncio

T = TypeVar('T')
K = TypeVar('K')


class IComponent(ABC):
    """Interfaz base para todos los componentes del sistema."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Inicializar el componente."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cerrar el componente."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Verificar salud del componente."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del componente."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Versión del componente."""
        pass


class IRepository(Generic[T, K], ABC):
    """Interfaz base para repositorios."""
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Crear una entidad."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: K) -> Optional[T]:
        """Obtener entidad por ID."""
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Obtener todas las entidades."""
        pass
    
    @abstractmethod
    async def update(self, entity_id: K, updates: Dict[str, Any]) -> Optional[T]:
        """Actualizar entidad."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: K) -> bool:
        """Eliminar entidad."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Contar entidades."""
        pass


class IService(ABC):
    """Interfaz base para servicios."""
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Procesar datos."""
        pass
    
    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """Validar datos."""
        pass


class IAnalyzer(IService):
    """Interfaz para analizadores."""
    
    @abstractmethod
    async def analyze(self, content: str) -> Dict[str, Any]:
        """Analizar contenido."""
        pass
    
    @abstractmethod
    async def get_analysis_type(self) -> str:
        """Obtener tipo de análisis."""
        pass


class IComparator(IService):
    """Interfaz para comparadores."""
    
    @abstractmethod
    async def compare(self, item1: Any, item2: Any) -> Dict[str, Any]:
        """Comparar dos elementos."""
        pass
    
    @abstractmethod
    async def get_similarity_score(self, item1: Any, item2: Any) -> float:
        """Obtener puntuación de similitud."""
        pass


class IEvaluator(IService):
    """Interfaz para evaluadores."""
    
    @abstractmethod
    async def evaluate(self, item: Any) -> Dict[str, Any]:
        """Evaluar un elemento."""
        pass
    
    @abstractmethod
    async def get_evaluation_criteria(self) -> List[str]:
        """Obtener criterios de evaluación."""
        pass


class IStorage(ABC):
    """Interfaz para almacenamiento."""
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> bool:
        """Almacenar valor."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Recuperar valor."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Eliminar valor."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verificar si existe."""
        pass


class ICache(IStorage):
    """Interfaz para caché."""
    
    @abstractmethod
    async def set_ttl(self, key: str, ttl_seconds: int) -> bool:
        """Establecer TTL."""
        pass
    
    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """Obtener TTL."""
        pass


class IMessageBroker(ABC):
    """Interfaz para message broker."""
    
    @abstractmethod
    async def publish(self, topic: str, message: Any) -> bool:
        """Publicar mensaje."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: callable) -> str:
        """Suscribirse a tópico."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Cancelar suscripción."""
        pass


class IEventBus(IMessageBroker):
    """Interfaz para event bus."""
    
    @abstractmethod
    async def emit(self, event_type: str, event_data: Any) -> bool:
        """Emitir evento."""
        pass
    
    @abstractmethod
    async def listen(self, event_type: str, handler: callable) -> str:
        """Escuchar evento."""
        pass


class IMiddleware(ABC):
    """Interfaz para middleware."""
    
    @abstractmethod
    async def process_request(self, request: Any) -> Any:
        """Procesar request."""
        pass
    
    @abstractmethod
    async def process_response(self, response: Any) -> Any:
        """Procesar response."""
        pass


class IPlugin(ABC):
    """Interfaz para plugins."""
    
    @abstractmethod
    async def install(self) -> bool:
        """Instalar plugin."""
        pass
    
    @abstractmethod
    async def uninstall(self) -> bool:
        """Desinstalar plugin."""
        pass
    
    @abstractmethod
    async def activate(self) -> bool:
        """Activar plugin."""
        pass
    
    @abstractmethod
    async def deactivate(self) -> bool:
        """Desactivar plugin."""
        pass
    
    @property
    @abstractmethod
    def plugin_info(self) -> Dict[str, Any]:
        """Información del plugin."""
        pass


class IExtension(IPlugin):
    """Interfaz para extensiones."""
    
    @abstractmethod
    async def extend(self, target: Any) -> Any:
        """Extender funcionalidad."""
        pass
    
    @abstractmethod
    async def get_extension_points(self) -> List[str]:
        """Obtener puntos de extensión."""
        pass


class IValidator(ABC):
    """Interfaz para validadores."""
    
    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """Validar datos."""
        pass
    
    @abstractmethod
    async def get_validation_rules(self) -> Dict[str, Any]:
        """Obtener reglas de validación."""
        pass


class ITransformer(ABC):
    """Interfaz para transformadores."""
    
    @abstractmethod
    async def transform(self, data: Any) -> Any:
        """Transformar datos."""
        pass
    
    @abstractmethod
    async def get_transformation_type(self) -> str:
        """Obtener tipo de transformación."""
        pass


class IProcessor(ABC):
    """Interfaz para procesadores."""
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Procesar datos."""
        pass
    
    @abstractmethod
    async def can_process(self, data: Any) -> bool:
        """Verificar si puede procesar."""
        pass


class IHandler(ABC):
    """Interfaz para handlers."""
    
    @abstractmethod
    async def handle(self, event: Any) -> Any:
        """Manejar evento."""
        pass
    
    @abstractmethod
    async def can_handle(self, event: Any) -> bool:
        """Verificar si puede manejar."""
        pass


class IStrategy(ABC):
    """Interfaz para estrategias."""
    
    @abstractmethod
    async def execute(self, context: Any) -> Any:
        """Ejecutar estrategia."""
        pass
    
    @abstractmethod
    async def get_strategy_name(self) -> str:
        """Obtener nombre de estrategia."""
        pass


class IObserver(ABC):
    """Interfaz para observadores."""
    
    @abstractmethod
    async def update(self, subject: Any, event: Any) -> None:
        """Actualizar observador."""
        pass


class ISubject(ABC):
    """Interfaz para sujetos observables."""
    
    @abstractmethod
    async def attach(self, observer: IObserver) -> None:
        """Adjuntar observador."""
        pass
    
    @abstractmethod
    async def detach(self, observer: IObserver) -> None:
        """Desadjuntar observador."""
        pass
    
    @abstractmethod
    async def notify(self, event: Any) -> None:
        """Notificar observadores."""
        pass


class ICommand(ABC):
    """Interfaz para comandos."""
    
    @abstractmethod
    async def execute(self) -> Any:
        """Ejecutar comando."""
        pass
    
    @abstractmethod
    async def undo(self) -> Any:
        """Deshacer comando."""
        pass


class IQuery(ABC):
    """Interfaz para consultas."""
    
    @abstractmethod
    async def execute(self) -> Any:
        """Ejecutar consulta."""
        pass


class ICommandHandler(ABC):
    """Interfaz para command handlers."""
    
    @abstractmethod
    async def handle(self, command: ICommand) -> Any:
        """Manejar comando."""
        pass


class IQueryHandler(ABC):
    """Interfaz para query handlers."""
    
    @abstractmethod
    async def handle(self, query: IQuery) -> Any:
        """Manejar consulta."""
        pass


class IEvent(ABC):
    """Interfaz para eventos."""
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Tipo de evento."""
        pass
    
    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Timestamp del evento."""
        pass
    
    @property
    @abstractmethod
    def data(self) -> Any:
        """Datos del evento."""
        pass


class IEventStore(ABC):
    """Interfaz para event store."""
    
    @abstractmethod
    async def append(self, event: IEvent) -> bool:
        """Agregar evento."""
        pass
    
    @abstractmethod
    async def get_events(self, stream_id: str) -> List[IEvent]:
        """Obtener eventos de stream."""
        pass
    
    @abstractmethod
    async def get_events_since(self, stream_id: str, since: datetime) -> List[IEvent]:
        """Obtener eventos desde timestamp."""
        pass


class IAggregate(ABC):
    """Interfaz para agregados."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """ID del agregado."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> int:
        """Versión del agregado."""
        pass
    
    @abstractmethod
    async def get_uncommitted_events(self) -> List[IEvent]:
        """Obtener eventos no confirmados."""
        pass
    
    @abstractmethod
    async def mark_events_as_committed(self) -> None:
        """Marcar eventos como confirmados."""
        pass


class IProjection(ABC):
    """Interfaz para proyecciones."""
    
    @abstractmethod
    async def project(self, event: IEvent) -> None:
        """Proyectar evento."""
        pass
    
    @abstractmethod
    async def get_state(self) -> Any:
        """Obtener estado."""
        pass


class IReadModel(ABC):
    """Interfaz para read models."""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Any]:
        """Obtener por ID."""
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[Any]:
        """Obtener todos."""
        pass
    
    @abstractmethod
    async def search(self, criteria: Dict[str, Any]) -> List[Any]:
        """Buscar por criterios."""
        pass


class IWriteModel(ABC):
    """Interfaz para write models."""
    
    @abstractmethod
    async def save(self, aggregate: IAggregate) -> bool:
        """Guardar agregado."""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[IAggregate]:
        """Obtener agregado por ID."""
        pass


class ISaga(ABC):
    """Interfaz para sagas."""
    
    @abstractmethod
    async def start(self, event: IEvent) -> None:
        """Iniciar saga."""
        pass
    
    @abstractmethod
    async def handle(self, event: IEvent) -> None:
        """Manejar evento en saga."""
        pass
    
    @abstractmethod
    async def is_completed(self) -> bool:
        """Verificar si está completada."""
        pass


class IWorkflow(ABC):
    """Interfaz para workflows."""
    
    @abstractmethod
    async def start(self, context: Any) -> str:
        """Iniciar workflow."""
        pass
    
    @abstractmethod
    async def execute_step(self, workflow_id: str, step: str) -> Any:
        """Ejecutar paso del workflow."""
        pass
    
    @abstractmethod
    async def get_status(self, workflow_id: str) -> str:
        """Obtener estado del workflow."""
        pass


class IPipeline(ABC):
    """Interfaz para pipelines."""
    
    @abstractmethod
    async def add_stage(self, stage: IProcessor) -> None:
        """Agregar etapa al pipeline."""
        pass
    
    @abstractmethod
    async def remove_stage(self, stage: IProcessor) -> None:
        """Remover etapa del pipeline."""
        pass
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Procesar datos a través del pipeline."""
        pass


class IRegistry(ABC):
    """Interfaz para registros."""
    
    @abstractmethod
    async def register(self, name: str, component: Any) -> bool:
        """Registrar componente."""
        pass
    
    @abstractmethod
    async def unregister(self, name: str) -> bool:
        """Desregistrar componente."""
        pass
    
    @abstractmethod
    async def get(self, name: str) -> Optional[Any]:
        """Obtener componente."""
        pass
    
    @abstractmethod
    async def list_all(self) -> List[str]:
        """Listar todos los componentes."""
        pass


class IFactory(ABC):
    """Interfaz para factories."""
    
    @abstractmethod
    async def create(self, type_name: str, **kwargs) -> Any:
        """Crear instancia."""
        pass
    
    @abstractmethod
    async def register_type(self, type_name: str, type_class: type) -> None:
        """Registrar tipo."""
        pass


class IBuilder(ABC):
    """Interfaz para builders."""
    
    @abstractmethod
    async def build(self) -> Any:
        """Construir objeto."""
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        """Resetear builder."""
        pass


class IAdapter(ABC):
    """Interfaz para adapters."""
    
    @abstractmethod
    async def adapt(self, source: Any, target_type: type) -> Any:
        """Adaptar objeto."""
        pass
    
    @abstractmethod
    async def can_adapt(self, source: Any, target_type: type) -> bool:
        """Verificar si puede adaptar."""
        pass


class IDecorator(ABC):
    """Interfaz para decorators."""
    
    @abstractmethod
    async def decorate(self, target: Any) -> Any:
        """Decorar objeto."""
        pass
    
    @abstractmethod
    async def undecorate(self, target: Any) -> Any:
        """Quitar decoración."""
        pass


class IProxy(ABC):
    """Interfaz para proxies."""
    
    @abstractmethod
    async def proxy(self, target: Any) -> Any:
        """Crear proxy."""
        pass
    
    @abstractmethod
    async def get_target(self) -> Any:
        """Obtener objetivo."""
        pass


class IMonitor(ABC):
    """Interfaz para monitores."""
    
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Iniciar monitoreo."""
        pass
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Detener monitoreo."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas."""
        pass


class ILogger(ABC):
    """Interfaz para loggers."""
    
    @abstractmethod
    async def log(self, level: str, message: str, **kwargs) -> None:
        """Registrar log."""
        pass
    
    @abstractmethod
    async def debug(self, message: str, **kwargs) -> None:
        """Log de debug."""
        pass
    
    @abstractmethod
    async def info(self, message: str, **kwargs) -> None:
        """Log de información."""
        pass
    
    @abstractmethod
    async def warning(self, message: str, **kwargs) -> None:
        """Log de advertencia."""
        pass
    
    @abstractmethod
    async def error(self, message: str, **kwargs) -> None:
        """Log de error."""
        pass


class IConfig(ABC):
    """Interfaz para configuración."""
    
    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Establecer valor de configuración."""
        pass
    
    @abstractmethod
    async def has(self, key: str) -> bool:
        """Verificar si existe configuración."""
        pass




