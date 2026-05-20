"""
Contador AI - Main class for Mexican accounting and tax assistance.
Uses OpenRouter to provide fiscal advice, tax calculations, and SAT procedures guidance.

Refactored to:
- Use APIHandler for all API calls with metrics
- Use PromptBuilder for all prompt construction
- Use MessageBuilder for message formatting
- Eliminate duplicate timing and error handling logic
"""

import logging
from typing import Dict, List, Optional, Any

from ..config.contador_config import ContadorConfig
from ..infrastructure.openrouter.openrouter_client import OpenRouterClient
from .prompt_builder import PromptBuilder
from .api_handler import APIHandler
from .system_prompts_builder import SystemPromptsBuilder
from .service_helpers import MessageBuilder
from .response_formatter import ResponseFormatter
from .cache_helper import CacheHelper
from .service_method_helper import ServiceMethodHelper
from .calculator_helper import CalculatorHelper
from .validators import ContadorValidator
from .service_data_builder import ServiceDataBuilder
from .component_initializer import initialize_cache, initialize_metrics

logger = logging.getLogger(__name__)


class ContadorAI:
    """
    AI Contador that uses Open Router to provide Mexican accounting and tax assistance.
    Provides calculations, advice, guides, and SAT procedures support.
    """
    
    def __init__(self, config: Optional[ContadorConfig] = None):
        self.config = config or ContadorConfig()
        self.config.validate()
        
        self.client = OpenRouterClient(self.config.openrouter.api_key)
        self.api_handler = APIHandler(self.client, self.config)
        
        # System prompts for different services
        self.system_prompts = SystemPromptsBuilder.build_all_prompts()
        
        # Initialize optional components using helper
        self.cache = initialize_cache(self.config)
        self.metrics = initialize_metrics()
    
    async def calcular_impuestos(
        self,
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Calcular impuestos para un régimen fiscal específico.
        
        Args:
            regimen: Régimen fiscal (RESICO, PFAE, Sueldos y Salarios, etc.)
            tipo_impuesto: Tipo de impuesto (ISR, IVA, IEPS)
            datos: Datos necesarios para el cálculo (ingresos, gastos, etc.)
            use_cache: Whether to use cache
        
        Returns:
            Response dictionary with calculation and breakdown
        """
        ContadorValidator.validate_calculo_impuestos(regimen, tipo_impuesto, datos)
        
        # Check cache
        cache_params = ServiceDataBuilder.build_calculo_cache_params(
            regimen, tipo_impuesto, datos
        )
        cached = CacheHelper.get_cached_result(
            self.cache, "calcular_impuestos", cache_params, use_cache
        )
        if cached:
            return cached
        
        # Try to use specialized calculator
        resultado_calculadora = CalculatorHelper.try_calcular_impuesto(
            regimen, tipo_impuesto, datos
        )
        
        # For Plataformas, use specialized calculator
        if regimen == "Plataformas" and tipo_impuesto == "ISR":
            try:
                from ..services.calculadora_plataformas import CalculadoraPlataformas
                calc_plataformas = CalculadoraPlataformas()
                ingresos_mensuales = datos.get("ingresos_mensuales", 0)
                tipo_plataforma = datos.get("tipo_plataforma", "general")
                aplicar_iva = datos.get("aplicar_iva", True)
                resultado_calculadora = calc_plataformas.calcular_impuestos_plataforma(
                    ingresos_mensuales, tipo_plataforma, aplicar_iva
                )
            except (ImportError, Exception) as e:
                logger.debug(f"Calculadora plataformas no disponible: {e}")
        
        prompt = PromptBuilder.build_calculation_prompt(regimen, tipo_impuesto, datos)
        
        # Enhance prompt with calculator result if available
        prompt = CalculatorHelper.enhance_prompt_with_calculator_result(
            prompt, resultado_calculadora, result_type="cálculo"
        )
        
        service_data = ServiceDataBuilder.build_calculo_service_data(
            regimen, tipo_impuesto, datos
        )
        
        result = await ServiceMethodHelper.execute_service(
            prompt=prompt,
            system_prompt=self.system_prompts["calculo_impuestos"],
            api_handler=self.api_handler,
            service_name="calcular_impuestos",
            service_data=service_data,
            extract_key="resultado",
            time_field_rename="tiempo_calculo"
        )
        
        # Record metrics
        if self.metrics:
            duration = result.get("tiempo_calculo", 0)
            self.metrics.record_service_call(
                "calcular_impuestos",
                duration,
                success=result.get("success", False)
            )
            self.metrics.record_regimen_usage(regimen)
            self.metrics.record_impuesto_usage(tipo_impuesto)
        
        return result
    
    async def asesoria_fiscal(
        self,
        pregunta: str,
        contexto: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Proporcionar asesoría fiscal personalizada.
        
        Args:
            pregunta: Pregunta o situación fiscal
            contexto: Contexto adicional (regimen, ingresos, etc.)
            use_cache: Whether to use cache
        
        Returns:
            Response dictionary with advice
        """
        prompt = PromptBuilder.build_advice_prompt(pregunta, contexto)
        
        service_data = ServiceDataBuilder.build_asesoria_service_data(pregunta)
        
        result = await ServiceMethodHelper.execute_service(
            prompt=prompt,
            system_prompt=self.system_prompts["asesoria_fiscal"],
            api_handler=self.api_handler,
            service_name="asesoria_fiscal",
            service_data=service_data,
            extract_key="asesoria"
        )
        
        # Cache result if enabled
        cache_params = ServiceDataBuilder.build_asesoria_cache_params(pregunta, contexto)
        CacheHelper.store_result(
            self.cache, "asesoria_fiscal", cache_params, result, use_cache, ttl=1800
        )
        
        return result
    
    async def guia_fiscal(
        self,
        tema: str,
        nivel_detalle: str = "completo",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generar guía fiscal sobre un tema específico.
        
        Args:
            tema: Tema fiscal (ej: "Deducciones RESICO", "Facturación electrónica")
            nivel_detalle: Nivel de detalle (básico, intermedio, completo)
        
        Returns:
            Response dictionary with guide
        """
        ContadorValidator.validate_guia_fiscal(tema, nivel_detalle)
        
        # Build cache params
        cache_params = ServiceDataBuilder.build_guia_cache_params(tema, nivel_detalle)
        
        # Execute with cache
        async def _execute_guia():
            prompt = PromptBuilder.build_guide_prompt(tema, nivel_detalle)
            service_data = ServiceDataBuilder.build_guia_service_data(tema, nivel_detalle)
            
            return await ServiceMethodHelper.execute_service(
                prompt=prompt,
                system_prompt=self.system_prompts["guias_fiscales"],
                api_handler=self.api_handler,
                service_name="guia_fiscal",
                service_data=service_data,
                temperature=0.5,  # Slightly higher for guides
                extract_key="guia",
                time_field_rename="tiempo_generacion"
            )
        
        return await CacheHelper.cached_service_execution(
            self.cache,
            "guia_fiscal",
            cache_params,
            _execute_guia,
            use_cache=use_cache,
            cache_ttl=7200
        )
    
    async def tramite_sat(
        self,
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener información sobre un trámite del SAT.
        
        Args:
            tipo_tramite: Tipo de trámite (ej: "Alta en RFC", "Renovación e.firma")
            detalles: Detalles adicionales del trámite
            use_cache: Whether to use cache
        
        Returns:
            Response dictionary with procedure information
        """
        # Build cache params
        cache_params = ServiceDataBuilder.build_tramite_cache_params(tipo_tramite, detalles)
        
        # Execute with cache (SAT procedures change rarely, cache for 4 hours)
        async def _execute_tramite():
            prompt = PromptBuilder.build_procedure_prompt(tipo_tramite, detalles)
            service_data = ServiceDataBuilder.build_tramite_service_data(tipo_tramite, detalles)
            
            return await ServiceMethodHelper.execute_service(
                prompt=prompt,
                system_prompt=self.system_prompts["tramites_sat"],
                api_handler=self.api_handler,
                service_name="tramite_sat",
                service_data=service_data,
                extract_key="informacion"
            )
        
        return await CacheHelper.cached_service_execution(
            self.cache,
            "tramite_sat",
            cache_params,
            _execute_tramite,
            use_cache=use_cache,
            cache_ttl=14400  # 4 hours
        )
    
    async def ayuda_declaracion(
        self,
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Ayudar con la preparación de una declaración fiscal.
        
        Args:
            tipo_declaracion: Tipo de declaración (mensual, anual, etc.)
            periodo: Período fiscal
            datos: Datos del contribuyente
            use_cache: Whether to use cache
        
        Returns:
            Response dictionary with declaration guidance
        """
        # Build cache params
        cache_params = {
            "tipo_declaracion": tipo_declaracion,
            "periodo": periodo,
            "datos": datos or {}
        }
        
        # Execute with cache
        async def _execute_declaracion():
            prompt = PromptBuilder.build_declaration_prompt(tipo_declaracion, periodo, datos)
            service_data = {
                "tipo_declaracion": tipo_declaracion,
                "periodo": periodo,
                "datos": datos or {}
            }
            
            return await ServiceMethodHelper.execute_service(
                prompt=prompt,
                system_prompt=self.system_prompts["declaraciones"],
                api_handler=self.api_handler,
                service_name="ayuda_declaracion",
                service_data=service_data,
                extract_key="guia"
            )
        
        return await CacheHelper.cached_service_execution(
            self.cache,
            "ayuda_declaracion",
            cache_params,
            _execute_declaracion,
            use_cache=use_cache
        )
    
    async def comparar_regimenes(
        self,
        regimenes: List[str],
        datos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comparar diferentes regímenes fiscales para un contribuyente.
        
        Args:
            regimenes: Lista de regímenes a comparar
            datos: Datos del contribuyente (ingresos, gastos, etc.)
        
        Returns:
            Response dictionary with comparison
        """
        # Validate all regimens
        for regimen in regimenes:
            ContadorValidator.validate_regimen(regimen)
        
        # Try to use specialized comparator
        resultado_calculadora = CalculatorHelper.try_comparar_regimenes(regimenes, datos)
        
        prompt = PromptBuilder.build_comparison_prompt(regimenes, datos)
        
        # Enhance prompt with calculator result if available
        prompt = CalculatorHelper.enhance_prompt_with_calculator_result(
            prompt, resultado_calculadora, result_type="comparación"
        )
        
        service_data = {
            "regimenes": regimenes,
            "datos": datos
        }
        
        result = await ServiceMethodHelper.execute_service(
            prompt=prompt,
            system_prompt=self.system_prompts["asesoria_fiscal"],
            api_handler=self.api_handler,
            service_name="comparar_regimenes",
            service_data=service_data,
            extract_key="comparacion"
        )
        
        # Add calculator results if available
        result = CalculatorHelper.add_calculator_result_to_response(
            result, resultado_calculadora
        )
        
        return result
    
    def exportar_resultado(
        self,
        resultado: Dict[str, Any],
        formato: str = "json"
    ) -> str:
        """
        Exportar resultado a diferentes formatos.
        
        Args:
            resultado: Resultado a exportar
            formato: Formato de exportación (json, markdown, csv)
        
        Returns:
            Contenido exportado como string
        """
        from ..services.exportador_resultados import ExportadorResultados
        exportador = ExportadorResultados()
        
        if formato == "json":
            return exportador.exportar_json(resultado)
        elif formato == "markdown":
            return exportador.exportar_markdown(resultado)
        elif formato == "csv":
            return exportador.exportar_csv(resultado)
        else:
            raise ValueError(f"Formato no soportado: {formato}")
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def get_metrics_stats(self) -> Optional[Dict[str, Any]]:
        """Get metrics statistics."""
        if self.metrics:
            return {
                "services": self.metrics.get_service_stats(),
                "regimenes": self.metrics.get_regimen_stats(),
                "impuestos": self.metrics.get_impuesto_stats(),
                "cache": self.metrics.get_cache_stats(),
                "errors": self.metrics.get_error_stats(),
                "overall": self.metrics.get_overall_stats(),
                "hourly_usage": self.metrics.get_hourly_usage(24)
            }
        return None
    
    async def close(self) -> None:
        """Close client connections."""
        await self.client.close()

