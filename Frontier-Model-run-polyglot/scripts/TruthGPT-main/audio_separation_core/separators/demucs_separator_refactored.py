"""
Demucs Separator Refactorizado - Versión simplificada y optimizada.

Mejoras:
- Constantes de clase para mapeos
- Métodos helper consolidados
- Mejor manejo de dispositivo
- Código más claro
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .base_separator import BaseSeparator
from ..core.config import SeparationConfig
from ..core.exceptions import AudioSeparationError


class DemucsSeparator(BaseSeparator):
    """
    Separador de audio usando Demucs.
    
    Refactorizado para simplificar lógica y eliminar duplicación.
    """
    
    # Mapeo de componentes a nombres de Demucs
    DEMUCS_COMPONENT_MAP = {
        "vocals": "vocals",
        "drums": "drums",
        "bass": "bass",
        "other": "other",
        "accompaniment": "other",  # Mapear accompaniment a other
    }
    
    # Modelo por defecto
    DEFAULT_MODEL = "htdemucs"
    
    def __init__(
        self,
        config: Optional[SeparationConfig] = None,
        **kwargs
    ):
        """Inicializa el separador Demucs."""
        if config is None:
            config = SeparationConfig(model_type="demucs")
        super().__init__(config, **kwargs)
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
    
    def _load_model(self, **kwargs):
        """Carga el modelo Demucs."""
        try:
            import demucs.api
            import torch
        except ImportError:
            raise AudioSeparationError(
                "Demucs is not installed. Install with: pip install demucs",
                component=self.name
            )
        
        # Determinar dispositivo y modelo
        self._device = self._determine_device(torch)
        self._model_name = self._determine_model_name()
        
        # Demucs API maneja la carga internamente
        return {
            "api": demucs.api,
            "device": self._device,
            "model_name": self._model_name,
        }
    
    def _cleanup_model(self) -> None:
        """Limpia el modelo Demucs."""
        if self._model is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._model = None
    
    def _perform_separation(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """Realiza la separación usando Demucs."""
        if self._model is None:
            raise AudioSeparationError(
                "Model not loaded",
                component=self.name
            )
        
        try:
            from demucs.api import separate
            
            # Realizar separación
            separate(
                model=self._model["model_name"],
                audio=str(input_path),
                out=str(output_dir),
                device=self._model["device"],
                **kwargs
            )
            
            # Construir rutas de salida
            return self._build_output_paths(input_path, output_dir, components)
        except Exception as e:
            raise AudioSeparationError(
                f"Demucs separation failed: {e}",
                component=self.name
            ) from e
    
    def _get_default_components(self) -> List[str]:
        """Obtiene los componentes por defecto."""
        return ["vocals", "drums", "bass", "other"]
    
    def get_supported_components(self) -> List[str]:
        """Obtiene los componentes soportados."""
        return ["vocals", "drums", "bass", "other"]
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (consolidados)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _determine_device(self, torch) -> str:
        """
        Determina el dispositivo a usar (GPU o CPU).
        
        Args:
            torch: Módulo torch importado
        
        Returns:
            "cuda" o "cpu"
        """
        if self._config.use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _determine_model_name(self) -> str:
        """
        Determina el nombre del modelo.
        
        Returns:
            Nombre del modelo Demucs
        """
        return self._config.model_params.get("model_name", self.DEFAULT_MODEL)
    
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
        
        # Demucs guarda en: output_dir/model_name/input_stem/component.wav
        model_output_dir = output_dir / self._model_name / input_stem
        
        for component in components:
            demucs_name = self.DEMUCS_COMPONENT_MAP.get(component, component)
            
            # Intentar con estructura estándar de Demucs
            output_file = model_output_dir / f"{demucs_name}.wav"
            
            if not output_file.exists():
                # Fallback: sin subdirectorio del modelo
                output_file = output_dir / input_stem / f"{demucs_name}.wav"
            
            if output_file.exists():
                results[component] = str(output_file)
        
        return results

