"""
Demucs Separator - Implementación usando Demucs.
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
    
    Demucs es una biblioteca de Facebook Research para separación de audio
    de alta calidad usando deep learning.
    """
    
    def __init__(
        self,
        config: Optional[SeparationConfig] = None,
        **kwargs
    ):
        """
        Inicializa el separador Demucs.
        
        Args:
            config: Configuración
            **kwargs: Parámetros adicionales
        """
        if config is None:
            config = SeparationConfig(model_type="demucs")
        super().__init__(config, **kwargs)
        
        self._demucs = None
        self._model_name = "htdemucs"  # Modelo por defecto
    
    def _load_model(self, **kwargs):
        """
        Carga el modelo Demucs.
        
        Args:
            **kwargs: Parámetros adicionales
        
        Returns:
            Modelo Demucs
        """
        try:
            import demucs.api
            import torch
        except ImportError:
            raise AudioSeparationError(
                "Demucs is not installed. Install with: pip install demucs",
                component=self.name
            )
        
        try:
            # Determinar dispositivo
            device = "cuda" if self._config.use_gpu and torch.cuda.is_available() else "cpu"
            
            # Cargar modelo
            model_name = self._config.model_params.get("model_name", "htdemucs")
            self._model_name = model_name
            
            # Demucs API maneja la carga del modelo internamente
            # Retornamos la API en lugar del modelo directamente
            return {
                "api": demucs.api,
                "device": device,
                "model_name": model_name,
            }
        except Exception as e:
            raise AudioSeparationError(
                f"Failed to load Demucs model: {e}",
                component=self.name
            ) from e
    
    def _cleanup_model(self) -> None:
        """Limpia el modelo Demucs."""
        if self._model is not None:
            # Limpiar memoria GPU si es necesario
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
        """
        Realiza la separación usando Demucs.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_dir: Directorio de salida
            components: Componentes a separar
            **kwargs: Parámetros adicionales
        
        Returns:
            Diccionario con rutas a los componentes separados
        """
        if self._model is None:
            raise AudioSeparationError(
                "Model not loaded",
                component=self.name
            )
        
        try:
            import torch
            from demucs.api import separate
            
            device = self._model["device"]
            model_name = self._model["model_name"]
            
            # Realizar separación
            separate(
                model=model_name,
                audio=str(input_path),
                out=str(output_dir),
                device=device,
                **kwargs
            )
            
            # Demucs separa en: drums, bass, other, vocals
            demucs_mapping = {
                "vocals": "vocals",
                "drums": "drums",
                "bass": "bass",
                "other": "other",
                "accompaniment": "other",  # Mapear accompaniment a other
            }
            
            # Construir rutas de salida
            results = {}
            input_stem = input_path.stem
            
            # Demucs guarda en: output_dir/model_name/input_stem/component.wav
            model_output_dir = output_dir / model_name / input_stem
            
            for component in components:
                demucs_name = demucs_mapping.get(component, component)
                output_file = model_output_dir / f"{demucs_name}.wav"
                
                if output_file.exists():
                    results[component] = str(output_file)
                else:
                    # Intentar sin subdirectorio del modelo
                    output_file = output_dir / input_stem / f"{demucs_name}.wav"
                    if output_file.exists():
                        results[component] = str(output_file)
            
            return results
        except Exception as e:
            raise AudioSeparationError(
                f"Demucs separation failed: {e}",
                component=self.name
            ) from e
    
    def _get_default_components(self) -> List[str]:
        """
        Obtiene los componentes por defecto de Demucs.
        
        Returns:
            Lista de componentes por defecto
        """
        return ["vocals", "drums", "bass", "other"]
    
    def get_supported_components(self) -> List[str]:
        """
        Obtiene los componentes soportados por Demucs.
        
        Returns:
            Lista de componentes soportados
        """
        return ["vocals", "drums", "bass", "other"]




