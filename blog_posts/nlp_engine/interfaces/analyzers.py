from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ..core.entities import AnalysisScore
from ..core.enums import AnalysisType, ProcessingTier
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🔌 ANALYZER INTERFACES - Contratos para Analizadores
===================================================

Interfaces que deben implementar los analizadores de NLP.
"""



class IAnalyzer(ABC):
    """Interface para analizadores NLP específicos."""
    
    @abstractmethod
    async def analyze(self, text: str, context: Dict[str, Any]) -> AnalysisScore:
        """
        Realizar análisis específico del texto.
        
        Args:
            text: Texto a analizar
            context: Contexto adicional para el análisis
            
        Returns:
            AnalysisScore con el resultado del análisis
        """
        pass
    
    @abstractmethod
    def supports(self, analysis_type: AnalysisType) -> bool:
        """
        Verificar si el analizador soporta un tipo de análisis.
        
        Args:
            analysis_type: Tipo de análisis a verificar
            
        Returns:
            True si soporta el tipo de análisis
        """
        pass
    
    @abstractmethod
    def get_performance_tier(self) -> ProcessingTier:
        """
        Obtener el tier de performance del analizador.
        
        Returns:
            ProcessingTier del analizador
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Obtener el nombre identificador del analizador.
        
        Returns:
            Nombre del analizador
        """
        pass
    
    def get_supported_types(self) -> List[AnalysisType]:
        """
        Obtener lista de tipos de análisis soportados.
        
        Returns:
            Lista de AnalysisType soportados
        """
        return [at for at in AnalysisType if self.supports(at)]
    
    def can_handle(self, analysis_type: AnalysisType, tier: ProcessingTier) -> bool:
        """
        Verificar si puede manejar un análisis en un tier específico.
        
        Args:
            analysis_type: Tipo de análisis
            tier: Tier de performance requerido
            
        Returns:
            True si puede manejar el análisis
        """
        return self.supports(analysis_type) and self.get_performance_tier() == tier


class IAnalyzerFactory(ABC):
    """Interface para factory de analizadores."""
    
    @abstractmethod
    def create_analyzer(
        self, 
        analysis_type: AnalysisType, 
        tier: ProcessingTier
    ) -> Optional[IAnalyzer]:
        """
        Crear analizador específico para un tipo y tier.
        
        Args:
            analysis_type: Tipo de análisis requerido
            tier: Tier de performance requerido
            
        Returns:
            IAnalyzer configurado o None si no disponible
        """
        pass
    
    @abstractmethod
    def get_available_analyzers(self, tier: ProcessingTier) -> List[AnalysisType]:
        """
        Obtener analizadores disponibles para un tier.
        
        Args:
            tier: Tier de performance
            
        Returns:
            Lista de AnalysisType disponibles
        """
        pass
    
    def supports_analysis(
        self, 
        analysis_type: AnalysisType, 
        tier: ProcessingTier
    ) -> bool:
        """
        Verificar si puede crear analizador para tipo y tier.
        
        Args:
            analysis_type: Tipo de análisis
            tier: Tier de performance
            
        Returns:
            True si puede crear el analizador
        """
        return self.create_analyzer(analysis_type, tier) is not None
    
    def get_all_supported_types(self) -> Dict[ProcessingTier, List[AnalysisType]]:
        """
        Obtener todos los tipos soportados por tier.
        
        Returns:
            Diccionario de tier -> tipos soportados
        """
        result = {}
        for tier in ProcessingTier:
            result[tier] = self.get_available_analyzers(tier)
        return result


class IAdvancedAnalyzer(IAnalyzer):
    """Interface extendida para analizadores avanzados."""
    
    @abstractmethod
    async def batch_analyze(
        self, 
        texts: List[str], 
        context: Dict[str, Any]
    ) -> List[AnalysisScore]:
        """
        Analizar múltiples textos en lote.
        
        Args:
            texts: Lista de textos a analizar
            context: Contexto compartido
            
        Returns:
            Lista de AnalysisScore
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo utilizado.
        
        Returns:
            Información del modelo
        """
        pass
    
    @abstractmethod
    async def warm_up(self) -> None:
        """
        Preparar el analizador para uso (pre-cargar modelos, etc.).
        """
        pass


class IConfigurableAnalyzer(IAnalyzer):
    """Interface para analizadores configurables."""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configurar el analizador con parámetros específicos.
        
        Args:
            config: Diccionario de configuración
        """
        pass
    
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """
        Obtener configuración actual del analizador.
        
        Returns:
            Configuración actual
        """
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validar una configuración propuesta.
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        pass 