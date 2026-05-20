"""
Service Data Builder for Contador AI
=====================================

Centralizes construction of cache_params and service_data dictionaries.
Eliminates repetitive dictionary construction patterns across service methods.

Single Responsibility: Build consistent cache_params and service_data dictionaries.
"""

from typing import Dict, Any, Optional, List


class ServiceDataBuilder:
    """
    Helper class for building cache_params and service_data dictionaries.
    
    Responsibilities:
    - Build cache_params for different services
    - Build service_data for different services
    - Ensure consistency across services
    """
    
    @staticmethod
    def build_calculo_cache_params(
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build cache params for tax calculation."""
        return {
            "regimen": regimen,
            "tipo_impuesto": tipo_impuesto,
            "datos": datos
        }
    
    @staticmethod
    def build_calculo_service_data(
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build service data for tax calculation."""
        return {
            "regimen": regimen,
            "tipo_impuesto": tipo_impuesto,
            "datos_entrada": datos,
        }
    
    @staticmethod
    def build_asesoria_cache_params(
        pregunta: str,
        contexto: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build cache params for fiscal advice."""
        return {
            "pregunta": pregunta,
            "contexto": contexto or {}
        }
    
    @staticmethod
    def build_asesoria_service_data(pregunta: str) -> Dict[str, Any]:
        """Build service data for fiscal advice."""
        return {"pregunta": pregunta}
    
    @staticmethod
    def build_guia_cache_params(
        tema: str,
        nivel_detalle: str
    ) -> Dict[str, Any]:
        """Build cache params for fiscal guide."""
        return {
            "tema": tema,
            "nivel_detalle": nivel_detalle
        }
    
    @staticmethod
    def build_guia_service_data(
        tema: str,
        nivel_detalle: str
    ) -> Dict[str, Any]:
        """Build service data for fiscal guide."""
        return {
            "tema": tema,
            "nivel_detalle": nivel_detalle
        }
    
    @staticmethod
    def build_tramite_cache_params(
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build cache params for SAT procedure."""
        return {
            "tipo_tramite": tipo_tramite,
            "detalles": detalles or {}
        }
    
    @staticmethod
    def build_tramite_service_data(
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build service data for SAT procedure."""
        return {
            "tipo_tramite": tipo_tramite,
            "detalles": detalles or {}
        }
    
    @staticmethod
    def build_declaracion_cache_params(
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build cache params for declaration assistance."""
        return {
            "tipo_declaracion": tipo_declaracion,
            "periodo": periodo,
            "datos": datos or {}
        }
    
    @staticmethod
    def build_declaracion_service_data(
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build service data for declaration assistance."""
        return {
            "tipo_declaracion": tipo_declaracion,
            "periodo": periodo,
            "datos": datos or {}
        }
    
    @staticmethod
    def build_comparacion_service_data(
        regimenes: List[str],
        datos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build service data for regime comparison."""
        return {
            "regimenes": regimenes,
            "datos": datos
        }

