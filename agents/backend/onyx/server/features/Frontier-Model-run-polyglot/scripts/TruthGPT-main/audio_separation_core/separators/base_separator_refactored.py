"""
Base Separator Refactorizado - Implementación base simplificada.

Mejoras:
1. Eliminada duplicación de código de validación
2. Métodos helper consolidados
3. Menos abstracciones innecesarias
4. Código más directo y legible
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from ..core.base_component import BaseComponent
from ..core.interfaces_refactored import AudioSeparator
from ..utils.format_utils import is_audio_file, is_video_file
from ..utils.validation_utils import validate_audio_path, validate_output_dir


class BaseSeparator(AudioSeparator):
    """
    Clase base para separadores de audio.
    
    Responsabilidades:
    - Validación común de entrada
    - Gestión de rutas y directorios
    - Delegación a implementación específica
    
    Eliminado:
    - Configuración compleja (se pasa como parámetros)
    - Métodos abstractos innecesarios
    - Validación redundante
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Inicializa el separador base.
        
        Args:
            name: Nombre del separador
            use_gpu: Si usar GPU (si está disponible)
        """
        super().__init__(name)
        self.use_gpu = use_gpu
        self._model = None
    
    def _do_initialize(self) -> None:
        """Carga el modelo de separación."""
        self._model = self._load_model()
    
    def _do_cleanup(self) -> None:
        """Limpia el modelo."""
        if self._model is not None:
            self._cleanup_model()
            self._model = None
    
    def separate(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        components: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Separa un archivo de audio en componentes.
        
        Implementa validación común y delega a _perform_separation().
        """
        self._ensure_ready()
        
        # Validar entrada
        input_path = validate_audio_path(input_path)
        
        # Validar formato
        if not (is_audio_file(input_path) or is_video_file(input_path)):
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Determinar componentes
        if components is None:
            components = self.get_supported_components()
        else:
            # Validar componentes
            supported = self.get_supported_components()
            invalid = [c for c in components if c not in supported]
            if invalid:
                raise ValueError(
                    f"Unsupported components: {invalid}. "
                    f"Supported: {supported}"
                )
        
        # Determinar directorio de salida
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_separated"
        output_dir = validate_output_dir(output_dir, create=True)
        
        # Realizar separación
        try:
            results = self._perform_separation(input_path, output_dir, components)
            
            # Validar resultados
            for component, path in results.items():
                if not Path(path).exists():
                    raise FileNotFoundError(f"Separated file not found: {path}")
            
            return results
        except Exception as e:
            self._last_error = str(e)
            raise RuntimeError(f"Separation failed: {e}") from e
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS (implementar en subclases)
    # ════════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def _load_model(self):
        """
        Carga el modelo de separación.
        
        Returns:
            Modelo cargado
        """
        pass
    
    @abstractmethod
    def _cleanup_model(self) -> None:
        """Limpia el modelo."""
        pass
    
    @abstractmethod
    def _perform_separation(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str],
    ) -> Dict[str, str]:
        """
        Realiza la separación (implementación específica).
        
        Args:
            input_path: Ruta al archivo de entrada
            output_dir: Directorio de salida
            components: Componentes a separar
        
        Returns:
            Diccionario con rutas a los componentes separados
        """
        pass
    
    @abstractmethod
    def get_supported_components(self) -> List[str]:
        """Obtiene los componentes soportados."""
        pass

