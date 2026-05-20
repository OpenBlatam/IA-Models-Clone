"""
LALAL Separator - Implementación usando LALAL.AI API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .base_separator import BaseSeparator
from ..core.config import SeparationConfig
from ..core.exceptions import AudioSeparationError


class LALALSeparator(BaseSeparator):
    """
    Separador de audio usando LALAL.AI API.
    
    LALAL.AI es un servicio en la nube para separación de audio de alta calidad.
    Requiere una API key.
    """
    
    # Constantes de clase
    API_BASE_URL = "https://api.lalal.ai/v1"
    API_ENV_KEY = "LALAL_API_KEY"
    DEFAULT_SEPARATION_TYPE = "vocal"
    
    COMPONENT_TO_URL_MAP = {
        "vocals": "vocal_url",
        "accompaniment": "instrumental_url",
    }
    
    COMPONENT_TO_SEPARATION_TYPE = {
        "vocals": "vocal",
        "instrumental": "instrumental",
        "accompaniment": "instrumental",
    }
    
    def __init__(
        self,
        config: Optional[SeparationConfig] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Inicializa el separador LALAL.
        
        Args:
            config: Configuración
            api_key: API key de LALAL.AI (o usar variable de entorno LALAL_API_KEY)
            **kwargs: Parámetros adicionales
        """
        if config is None:
            config = SeparationConfig(model_type="lalal")
        super().__init__(config, **kwargs)
        
        self._api_key = api_key
        self._api_client = None
    
    def _load_model(self, **kwargs):
        """
        Inicializa el cliente de API de LALAL.
        
        Args:
            **kwargs: Parámetros adicionales
        
        Returns:
            Cliente de API
        """
        import os
        import requests
        
        api_key = self._get_api_key()
        
        return {
            "api_key": api_key,
            "base_url": self.API_BASE_URL,
            "session": requests.Session(),
        }
    
    def _get_api_key(self) -> str:
        """
        Obtiene la API key de LALAL.
        
        Returns:
            API key
        
        Raises:
            AudioSeparationError: Si la API key no está disponible
        """
        import os
        
        api_key = self._api_key or os.getenv(self.API_ENV_KEY)
        if not api_key:
            raise AudioSeparationError(
                f"LALAL API key is required. Set {self.API_ENV_KEY} environment variable "
                "or pass api_key parameter.",
                component=self.name
            )
        
        self._api_key = api_key
        return api_key
    
    def _cleanup_model(self) -> None:
        """Limpia el cliente de API."""
        if self._model is not None and "session" in self._model:
            self._model["session"].close()
            self._model = None
    
    def _perform_separation(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """
        Realiza la separación usando LALAL.AI API.
        
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
                "API client not initialized",
                component=self.name
            )
        
        try:
            import requests
            
            api_key = self._model["api_key"]
            base_url = self._model["base_url"]
            session = self._model["session"]
            
            # Subir archivo
            with open(input_path, "rb") as f:
                files = {"file": f}
                headers = {"Authorization": f"Bearer {api_key}"}
                
                # Determinar tipo de separación y subir
                separation_type = self._determine_separation_type(components)
                result = self._upload_and_process(session, files, headers, base_url, separation_type)
                
                # Descargar archivos separados
                return self._download_separated_files(session, result, output_dir, components)
        except Exception as e:
            raise AudioSeparationError(
                f"LALAL separation failed: {e}",
                component=self.name
            ) from e
    
    def _get_default_components(self) -> List[str]:
        """
        Obtiene los componentes por defecto de LALAL.
        
        Returns:
            Lista de componentes por defecto
        """
        return ["vocals", "accompaniment"]
    
    def get_supported_components(self) -> List[str]:
        """
        Obtiene los componentes soportados por LALAL.
        
        Returns:
            Lista de componentes soportados
        """
        return ["vocals", "accompaniment"]
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (consolidados)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _determine_separation_type(self, components: List[str]) -> str:
        """
        Determina el tipo de separación según componentes solicitados.
        
        Args:
            components: Componentes solicitados
        
        Returns:
            Tipo de separación ("vocal" o "instrumental")
        """
        for component in components:
            if component in self.COMPONENT_TO_SEPARATION_TYPE:
                return self.COMPONENT_TO_SEPARATION_TYPE[component]
        return self.DEFAULT_SEPARATION_TYPE
    
    def _upload_and_process(
        self,
        session,
        files: Dict,
        headers: Dict,
        base_url: str,
        separation_type: str
    ) -> Dict:
        """
        Sube el archivo y procesa la separación.
        
        Args:
            session: Sesión de requests
            files: Archivos a subir
            headers: Headers HTTP
            base_url: URL base de la API
            separation_type: Tipo de separación
        
        Returns:
            Resultado de la API con URLs de descarga
        
        Raises:
            AudioSeparationError: Si la subida o procesamiento falla
        """
        upload_url = f"{base_url}/split"
        response = session.post(
            upload_url,
            files=files,
            headers=headers,
            data={"split_type": separation_type}
        )
        response.raise_for_status()
        return response.json()
    
    def _download_separated_files(
        self,
        session,
        result: Dict,
        output_dir: Path,
        components: List[str]
    ) -> Dict[str, str]:
        """
        Descarga los archivos separados desde las URLs de la API.
        
        Args:
            session: Sesión de requests
            result: Resultado de la API con URLs
            output_dir: Directorio de salida
            components: Componentes solicitados
        
        Returns:
            Diccionario con rutas a los archivos descargados
        
        Raises:
            AudioSeparationError: Si la descarga falla
        """
        results = {}
        
        for component in components:
            url_key = self.COMPONENT_TO_URL_MAP.get(component)
            if not url_key or url_key not in result:
                continue
            
            url = result[url_key]
            output_file = output_dir / f"{component}.wav"
            
            try:
                download_response = session.get(url)
                download_response.raise_for_status()
                
                with open(output_file, "wb") as out_file:
                    out_file.write(download_response.content)
                
                results[component] = str(output_file)
            except Exception as e:
                raise AudioSeparationError(
                    f"Failed to download {component} from LALAL: {e}",
                    component=self.name
                ) from e
        
        return results




