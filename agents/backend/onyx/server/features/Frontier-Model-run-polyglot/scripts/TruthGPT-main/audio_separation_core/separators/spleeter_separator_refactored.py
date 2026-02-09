"""
Spleeter Separator Refactorizado - Versión simplificada y optimizada.

Mejoras:
- Lógica de modelo consolidada
- Mapeo simplificado
- Menos código duplicado
- Mejor manejo de errores
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .base_separator import BaseSeparator
from ..core.config import SeparationConfig
from ..core.exceptions import AudioSeparationError


class SpleeterSeparator(BaseSeparator):
    """
    Separador de audio usando Spleeter.
    
    Refactorizado para simplificar lógica y eliminar duplicación.
    """
    
    # Mapeo de componentes a nombres de Spleeter
    SPLEETER_COMPONENT_MAP = {
        "vocals": "vocals",
        "accompaniment": "accompaniment",
        "drums": "drums",
        "bass": "bass",
        "other": "other",
        "piano": "piano",
    }
    
    # Mapeo de número de componentes a modelo
    MODEL_BY_COMPONENT_COUNT = {
        2: "spleeter:2stems",
        4: "spleeter:4stems",
        5: "spleeter:5stems-16kHz",
    }
    
    def __init__(
        self,
        config: Optional[SeparationConfig] = None,
        **kwargs
    ):
        """Inicializa el separador Spleeter."""
        if config is None:
            config = SeparationConfig(model_type="spleeter")
        super().__init__(config, **kwargs)
        self._model_name: Optional[str] = None
    
    def _load_model(self, **kwargs):
        """Carga el modelo Spleeter."""
        try:
            from spleeter.separator import Separator
        except ImportError:
            raise AudioSeparationError(
                "Spleeter is not installed. Install with: pip install spleeter",
                component=self.name
            )
        
        # Determinar modelo según componentes o usar el especificado
        self._model_name = self._determine_model_name()
        
        try:
            return Separator(self._model_name)
        except Exception as e:
            raise AudioSeparationError(
                f"Failed to load Spleeter model '{self._model_name}': {e}",
                component=self.name
            ) from e
    
    def _cleanup_model(self) -> None:
        """Limpia el modelo Spleeter."""
        # Spleeter no requiere limpieza explícita
        pass
    
    def _perform_separation(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """Realiza la separación usando Spleeter."""
        if self._model is None:
            raise AudioSeparationError(
                "Model not loaded",
                component=self.name
            )
        
        try:
            # Spleeter separa y guarda automáticamente
            self._model.separate_to_file(
                str(input_path),
                str(output_dir),
                codec="wav"
            )
            
            # Construir rutas de salida
            return self._build_output_paths(input_path, output_dir, components)
        except Exception as e:
            raise AudioSeparationError(
                f"Spleeter separation failed: {e}",
                component=self.name
            ) from e
    
    def _get_default_components(self) -> List[str]:
        """Obtiene los componentes por defecto."""
        return ["vocals", "accompaniment"]
    
    def get_supported_components(self) -> List[str]:
        """Obtiene los componentes soportados según el modelo."""
        if self._model_name is None:
            return self._get_default_components()
        
        if "5stems" in self._model_name:
            return ["vocals", "drums", "bass", "piano", "other"]
        elif "4stems" in self._model_name:
            return ["vocals", "drums", "bass", "other"]
        else:
            return ["vocals", "accompaniment"]
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (consolidados)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _determine_model_name(self) -> str:
        """
        Determina el nombre del modelo según componentes o configuración.
        
        Returns:
            Nombre del modelo Spleeter
        """
        # Usar modelo personalizado si se especifica
        if self._config.model_path:
            return self._config.model_path
        
        # Determinar según número de componentes
        component_count = len(self._config.components)
        return self.MODEL_BY_COMPONENT_COUNT.get(
            component_count,
            "spleeter:2stems"  # Default
        )
    
    def _build_output_paths(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str]
    ) -> Dict[str, str]:
        """
        Construye las rutas de salida para los componentes separados.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_dir: Directorio de salida
            components: Componentes solicitados
        
        Returns:
            Diccionario con rutas a los componentes
        """
        results = {}
        input_stem = input_path.stem
        
        # Spleeter guarda en: output_dir/input_stem/component.wav
        spleeter_dir = output_dir / input_stem
        
        for component in components:
            spleeter_name = self.SPLEETER_COMPONENT_MAP.get(component, component)
            
            # Intentar con subdirectorio (formato estándar de Spleeter)
            output_file = spleeter_dir / f"{spleeter_name}.wav"
            
            if not output_file.exists():
                # Fallback: sin subdirectorio
                output_file = output_dir / f"{spleeter_name}.wav"
            
            if output_file.exists():
                results[component] = str(output_file)
        
        return results

