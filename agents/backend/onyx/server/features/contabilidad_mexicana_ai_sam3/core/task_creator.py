"""
Task Creation Utilities for Contador SAM3 Agent
================================================

Refactored with:
- TaskBuilder pattern for fluent task creation
- TaskParameter dataclass for typed parameters  
- TaskTypeRegistry for extensible task types
- Factory methods for common tasks
"""

from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20


class TaskType(Enum):
    """Available task types."""
    CALCULAR_IMPUESTOS = "calcular_impuestos"
    ASESORIA_FISCAL = "asesoria_fiscal"
    GUIA_FISCAL = "guia_fiscal"
    TRAMITE_SAT = "tramite_sat"
    AYUDA_DECLARACION = "ayuda_declaracion"


@dataclass
class TaskParameters:
    """
    Typed parameters for a task.
    
    Provides type safety and validation for task parameters.
    """
    service_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def with_priority(self, priority: int) -> "TaskParameters":
        """Return new instance with updated priority."""
        return TaskParameters(
            service_type=self.service_type,
            data=self.data.copy(),
            priority=priority,
            metadata=self.metadata.copy(),
        )
    
    def with_metadata(self, key: str, value: Any) -> "TaskParameters":
        """Return new instance with added metadata."""
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        return TaskParameters(
            service_type=self.service_type,
            data=self.data.copy(),
            priority=self.priority,
            metadata=new_metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for task manager."""
        return {
            "service_type": self.service_type,
            "parameters": self.data,
            "priority": self.priority,
            "metadata": self.metadata,
        }


class TaskParameterBuilder(ABC):
    """Abstract base class for task parameter builders."""
    
    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """The task type this builder creates."""
        pass
    
    @abstractmethod
    def build(self) -> TaskParameters:
        """Build the task parameters."""
        pass


class CalcularImpuestosBuilder(TaskParameterBuilder):
    """Builder for tax calculation tasks."""
    
    task_type = TaskType.CALCULAR_IMPUESTOS
    
    def __init__(self):
        self._regimen: Optional[str] = None
        self._tipo_impuesto: Optional[str] = None
        self._datos: Dict[str, Any] = {}
        self._priority: int = 0
    
    def regimen(self, regimen: str) -> "CalcularImpuestosBuilder":
        """Set the fiscal regime."""
        self._regimen = regimen
        return self
    
    def tipo_impuesto(self, tipo: str) -> "CalcularImpuestosBuilder":
        """Set the tax type."""
        self._tipo_impuesto = tipo
        return self
    
    def datos(self, datos: Dict[str, Any]) -> "CalcularImpuestosBuilder":
        """Set the calculation data."""
        self._datos = datos.copy()
        return self
    
    def priority(self, priority: int) -> "CalcularImpuestosBuilder":
        """Set task priority."""
        self._priority = priority
        return self
    
    def build(self) -> TaskParameters:
        """Build the task parameters."""
        if not self._regimen:
            raise ValueError("Regimen is required")
        if not self._tipo_impuesto:
            raise ValueError("Tipo impuesto is required")
        
        return TaskParameters(
            service_type=self.task_type.value,
            data={
                "regimen": self._regimen,
                "tipo_impuesto": self._tipo_impuesto,
                "datos": self._datos,
            },
            priority=self._priority,
        )


class AsesoriaFiscalBuilder(TaskParameterBuilder):
    """Builder for fiscal advice tasks."""
    
    task_type = TaskType.ASESORIA_FISCAL
    
    def __init__(self):
        self._pregunta: Optional[str] = None
        self._contexto: Optional[Dict[str, Any]] = None
        self._priority: int = 0
    
    def pregunta(self, pregunta: str) -> "AsesoriaFiscalBuilder":
        """Set the question."""
        self._pregunta = pregunta
        return self
    
    def contexto(self, contexto: Dict[str, Any]) -> "AsesoriaFiscalBuilder":
        """Set the context."""
        self._contexto = contexto.copy()
        return self
    
    def priority(self, priority: int) -> "AsesoriaFiscalBuilder":
        """Set task priority."""
        self._priority = priority
        return self
    
    def build(self) -> TaskParameters:
        """Build the task parameters."""
        if not self._pregunta:
            raise ValueError("Pregunta is required")
        
        return TaskParameters(
            service_type=self.task_type.value,
            data={
                "pregunta": self._pregunta,
                "contexto": self._contexto,
            },
            priority=self._priority,
        )


class GuiaFiscalBuilder(TaskParameterBuilder):
    """Builder for fiscal guide tasks."""
    
    task_type = TaskType.GUIA_FISCAL
    
    def __init__(self):
        self._tema: Optional[str] = None
        self._nivel_detalle: str = "completo"
        self._priority: int = 0
    
    def tema(self, tema: str) -> "GuiaFiscalBuilder":
        """Set the topic."""
        self._tema = tema
        return self
    
    def nivel_detalle(self, nivel: str) -> "GuiaFiscalBuilder":
        """Set detail level."""
        self._nivel_detalle = nivel
        return self
    
    def priority(self, priority: int) -> "GuiaFiscalBuilder":
        """Set task priority."""
        self._priority = priority
        return self
    
    def build(self) -> TaskParameters:
        """Build the task parameters."""
        if not self._tema:
            raise ValueError("Tema is required")
        
        return TaskParameters(
            service_type=self.task_type.value,
            data={
                "tema": self._tema,
                "nivel_detalle": self._nivel_detalle,
            },
            priority=self._priority,
        )


class TramiteSATBuilder(TaskParameterBuilder):
    """Builder for SAT procedure tasks."""
    
    task_type = TaskType.TRAMITE_SAT
    
    def __init__(self):
        self._tipo_tramite: Optional[str] = None
        self._detalles: Optional[Dict[str, Any]] = None
        self._priority: int = 0
    
    def tipo_tramite(self, tipo: str) -> "TramiteSATBuilder":
        """Set the procedure type."""
        self._tipo_tramite = tipo
        return self
    
    def detalles(self, detalles: Dict[str, Any]) -> "TramiteSATBuilder":
        """Set the details."""
        self._detalles = detalles.copy()
        return self
    
    def priority(self, priority: int) -> "TramiteSATBuilder":
        """Set task priority."""
        self._priority = priority
        return self
    
    def build(self) -> TaskParameters:
        """Build the task parameters."""
        if not self._tipo_tramite:
            raise ValueError("Tipo tramite is required")
        
        return TaskParameters(
            service_type=self.task_type.value,
            data={
                "tipo_tramite": self._tipo_tramite,
                "detalles": self._detalles,
            },
            priority=self._priority,
        )


class AyudaDeclaracionBuilder(TaskParameterBuilder):
    """Builder for declaration assistance tasks."""
    
    task_type = TaskType.AYUDA_DECLARACION
    
    def __init__(self):
        self._tipo_declaracion: Optional[str] = None
        self._periodo: Optional[str] = None
        self._datos: Optional[Dict[str, Any]] = None
        self._priority: int = 0
    
    def tipo_declaracion(self, tipo: str) -> "AyudaDeclaracionBuilder":
        """Set declaration type."""
        self._tipo_declaracion = tipo
        return self
    
    def periodo(self, periodo: str) -> "AyudaDeclaracionBuilder":
        """Set fiscal period."""
        self._periodo = periodo
        return self
    
    def datos(self, datos: Dict[str, Any]) -> "AyudaDeclaracionBuilder":
        """Set the data."""
        self._datos = datos.copy()
        return self
    
    def priority(self, priority: int) -> "AyudaDeclaracionBuilder":
        """Set task priority."""
        self._priority = priority
        return self
    
    def build(self) -> TaskParameters:
        """Build the task parameters."""
        if not self._tipo_declaracion:
            raise ValueError("Tipo declaracion is required")
        if not self._periodo:
            raise ValueError("Periodo is required")
        
        return TaskParameters(
            service_type=self.task_type.value,
            data={
                "tipo_declaracion": self._tipo_declaracion,
                "periodo": self._periodo,
                "datos": self._datos,
            },
            priority=self._priority,
        )


class TaskBuilderRegistry:
    """Registry of task builders."""
    
    _builders: Dict[TaskType, type] = {
        TaskType.CALCULAR_IMPUESTOS: CalcularImpuestosBuilder,
        TaskType.ASESORIA_FISCAL: AsesoriaFiscalBuilder,
        TaskType.GUIA_FISCAL: GuiaFiscalBuilder,
        TaskType.TRAMITE_SAT: TramiteSATBuilder,
        TaskType.AYUDA_DECLARACION: AyudaDeclaracionBuilder,
    }
    
    @classmethod
    def get_builder(cls, task_type: TaskType) -> TaskParameterBuilder:
        """Get a builder for a task type."""
        builder_class = cls._builders.get(task_type)
        if not builder_class:
            raise ValueError(f"No builder registered for {task_type}")
        return builder_class()
    
    @classmethod
    def register(cls, task_type: TaskType, builder_class: type):
        """Register a custom builder."""
        cls._builders[task_type] = builder_class


class TaskCreator:
    """
    Creates tasks with consistent patterns.
    
    Refactored with:
    - Builder pattern for fluent API
    - Registry for extensibility
    - Factory methods for common cases
    - Type-safe parameters
    """
    
    # === Builder Access ===
    
    @staticmethod
    def calcular_impuestos() -> CalcularImpuestosBuilder:
        """Get builder for tax calculation task."""
        return CalcularImpuestosBuilder()
    
    @staticmethod
    def asesoria_fiscal() -> AsesoriaFiscalBuilder:
        """Get builder for fiscal advice task."""
        return AsesoriaFiscalBuilder()
    
    @staticmethod
    def guia_fiscal() -> GuiaFiscalBuilder:
        """Get builder for fiscal guide task."""
        return GuiaFiscalBuilder()
    
    @staticmethod
    def tramite_sat() -> TramiteSATBuilder:
        """Get builder for SAT procedure task."""
        return TramiteSATBuilder()
    
    @staticmethod
    def ayuda_declaracion() -> AyudaDeclaracionBuilder:
        """Get builder for declaration assistance task."""
        return AyudaDeclaracionBuilder()
    
    # === Core Task Creation ===
    
    @staticmethod
    async def create_task(
        task_manager: Any,
        service_type: str,
        parameters: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Create a task with consistent pattern."""
        return await task_manager.create_task(
            service_type=service_type,
            parameters=parameters,
            priority=priority,
        )
    
    @staticmethod
    async def create_from_params(
        task_manager: Any,
        params: TaskParameters
    ) -> str:
        """Create task from TaskParameters object."""
        task_dict = params.to_dict()
        return await task_manager.create_task(
            service_type=task_dict["service_type"],
            parameters=task_dict["parameters"],
            priority=task_dict["priority"],
        )
    
    # === Convenience Factory Methods (Backward Compatible) ===
    
    @staticmethod
    async def create_calcular_impuestos_task(
        task_manager: Any,
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Create tax calculation task."""
        params = (TaskCreator.calcular_impuestos()
            .regimen(regimen)
            .tipo_impuesto(tipo_impuesto)
            .datos(datos)
            .priority(priority)
            .build())
        return await TaskCreator.create_from_params(task_manager, params)
    
    @staticmethod
    async def create_asesoria_fiscal_task(
        task_manager: Any,
        pregunta: str,
        contexto: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create fiscal advice task."""
        builder = TaskCreator.asesoria_fiscal().pregunta(pregunta).priority(priority)
        if contexto:
            builder.contexto(contexto)
        return await TaskCreator.create_from_params(task_manager, builder.build())
    
    @staticmethod
    async def create_guia_fiscal_task(
        task_manager: Any,
        tema: str,
        nivel_detalle: str = "completo",
        priority: int = 0
    ) -> str:
        """Create fiscal guide task."""
        params = (TaskCreator.guia_fiscal()
            .tema(tema)
            .nivel_detalle(nivel_detalle)
            .priority(priority)
            .build())
        return await TaskCreator.create_from_params(task_manager, params)
    
    @staticmethod
    async def create_tramite_sat_task(
        task_manager: Any,
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create SAT procedure task."""
        builder = TaskCreator.tramite_sat().tipo_tramite(tipo_tramite).priority(priority)
        if detalles:
            builder.detalles(detalles)
        return await TaskCreator.create_from_params(task_manager, builder.build())
    
    @staticmethod
    async def create_ayuda_declaracion_task(
        task_manager: Any,
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create declaration assistance task."""
        builder = (TaskCreator.ayuda_declaracion()
            .tipo_declaracion(tipo_declaracion)
            .periodo(periodo)
            .priority(priority))
        if datos:
            builder.datos(datos)
        return await TaskCreator.create_from_params(task_manager, builder.build())
