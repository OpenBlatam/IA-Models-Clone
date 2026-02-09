from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from .config import NLPConfig
from .analyzers import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Factory pattern para crear y gestionar analizadores NLP.
"""


    AnalyzerInterface,
    SentimentAnalyzer,
    ReadabilityAnalyzer,
    KeywordAnalyzer,
    LanguageAnalyzer
)

logger = logging.getLogger(__name__)

class AnalyzerFactory:
    """Factory para crear y gestionar analizadores NLP."""
    
    def __init__(self, config: NLPConfig, executor: ThreadPoolExecutor):
        
    """__init__ function."""
self.config = config
        self.executor = executor
        self.analyzers: Dict[str, AnalyzerInterface] = {}
        self.logger = logging.getLogger(f"{__name__}.AnalyzerFactory")
        self._analyzer_registry = self._build_registry()
    
    def _build_registry(self) -> Dict[str, type]:
        """Construir registro de analizadores disponibles."""
        return {
            'sentiment': SentimentAnalyzer,
            'readability': ReadabilityAnalyzer,
            'keywords': KeywordAnalyzer,
            'language': LanguageAnalyzer
        }
    
    async def initialize(self) -> Any:
        """Inicializar factory y crear analizadores."""
        self.logger.info("Initializing AnalyzerFactory...")
        
        # Crear analizadores habilitados
        for analyzer_name, analyzer_class in self._analyzer_registry.items():
            if self._is_analyzer_enabled(analyzer_name):
                try:
                    analyzer = analyzer_class(self.config, self.executor)
                    if analyzer.is_available():
                        self.analyzers[analyzer_name] = analyzer
                        self.logger.info(f"Analyzer '{analyzer_name}' initialized and available")
                    else:
                        self.logger.warning(f"Analyzer '{analyzer_name}' not available")
                except Exception as e:
                    self.logger.error(f"Failed to initialize analyzer '{analyzer_name}': {e}")
        
        self.logger.info(f"AnalyzerFactory initialized with {len(self.analyzers)} analyzers")
    
    def _is_analyzer_enabled(self, analyzer_name: str) -> bool:
        """Verificar si un analizador está habilitado en la configuración."""
        config_mapping = {
            'sentiment': self.config.analysis.enable_sentiment,
            'readability': self.config.analysis.enable_readability,
            'keywords': self.config.analysis.enable_keywords,
            'language': self.config.analysis.enable_language_detection
        }
        return config_mapping.get(analyzer_name, False)
    
    def get_analyzer(self, name: str) -> Optional[AnalyzerInterface]:
        """
        Obtener analizador por nombre.
        
        Args:
            name: Nombre del analizador
            
        Returns:
            Analizador o None si no existe
        """
        return self.analyzers.get(name)
    
    def get_enabled_analyzers(self, options: Optional[Dict[str, Any]] = None) -> List[AnalyzerInterface]:
        """
        Obtener lista de analizadores habilitados para una solicitud.
        
        Args:
            options: Opciones específicas que pueden habilitar/deshabilitar analizadores
            
        Returns:
            Lista de analizadores a usar
        """
        options = options or {}
        enabled_analyzers = []
        
        for name, analyzer in self.analyzers.items():
            # Verificar si está explícitamente deshabilitado en options
            option_key = f"enable_{name}"
            if option_key in options and not options[option_key]:
                continue
            
            # Verificar si está explícitamente habilitado en options (sobrescribe config)
            if option_key in options and options[option_key]:
                enabled_analyzers.append(analyzer)
                continue
            
            # Usar configuración por defecto si no se especifica en options
            if analyzer.is_available():
                enabled_analyzers.append(analyzer)
        
        return enabled_analyzers
    
    def is_analyzer_available(self, name: str) -> bool:
        """
        Verificar si un analizador está disponible.
        
        Args:
            name: Nombre del analizador
            
        Returns:
            True si está disponible
        """
        analyzer = self.analyzers.get(name)
        return analyzer is not None and analyzer.is_available()
    
    def get_analyzer_names(self) -> List[str]:
        """Obtener lista de nombres de analizadores disponibles."""
        return list(self.analyzers.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de todos los analizadores."""
        stats = {
            'total_analyzers': len(self._analyzer_registry),
            'available_analyzers': len(self.analyzers),
            'analyzer_details': {}
        }
        
        for name, analyzer in self.analyzers.items():
            try:
                analyzer_stats = analyzer.get_stats()
                stats['analyzer_details'][name] = analyzer_stats
            except Exception as e:
                stats['analyzer_details'][name] = {'error': str(e)}
        
        return stats
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Obtener capacidades del sistema de análisis."""
        capabilities = {
            'available_analyzers': [],
            'supported_features': {
                'sentiment_analysis': False,
                'readability_analysis': False,
                'keyword_extraction': False,
                'language_detection': False,
                'batch_processing': True,
                'async_processing': True,
                'caching': True
            },
            'model_tiers': ['lightweight', 'standard', 'advanced'],
            'current_tier': self.config.models.type.value
        }
        
        for name, analyzer in self.analyzers.items():
            if analyzer.is_available():
                capabilities['available_analyzers'].append(name)
                
                # Marcar características como disponibles
                if name == 'sentiment':
                    capabilities['supported_features']['sentiment_analysis'] = True
                elif name == 'readability':
                    capabilities['supported_features']['readability_analysis'] = True
                elif name == 'keywords':
                    capabilities['supported_features']['keyword_extraction'] = True
                elif name == 'language':
                    capabilities['supported_features']['language_detection'] = True
        
        return capabilities
    
    def create_custom_analyzer(self, name: str, analyzer_class: type) -> bool:
        """
        Registrar un analizador personalizado.
        
        Args:
            name: Nombre del analizador
            analyzer_class: Clase del analizador (debe implementar AnalyzerInterface)
            
        Returns:
            True si se registró exitosamente
        """
        try:
            # Verificar que implementa la interfaz
            if not issubclass(analyzer_class, AnalyzerInterface):
                self.logger.error(f"Custom analyzer {name} must implement AnalyzerInterface")
                return False
            
            # Crear instancia
            analyzer = analyzer_class(self.config, self.executor)
            
            if analyzer.is_available():
                self.analyzers[name] = analyzer
                self._analyzer_registry[name] = analyzer_class
                self.logger.info(f"Custom analyzer '{name}' registered successfully")
                return True
            else:
                self.logger.warning(f"Custom analyzer '{name}' not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register custom analyzer '{name}': {e}")
            return False
    
    def remove_analyzer(self, name: str) -> bool:
        """
        Remover un analizador.
        
        Args:
            name: Nombre del analizador
            
        Returns:
            True si se removió exitosamente
        """
        if name in self.analyzers:
            del self.analyzers[name]
            self.logger.info(f"Analyzer '{name}' removed")
            return True
        return False 