"""
Realtime Market Analysis Service - Análisis de mercado en tiempo real
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RealtimeMarketAnalysisService:
    """Servicio para análisis de mercado en tiempo real"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.market_data: Dict[str, List[Dict[str, Any]]] = {}
        self.insights: Dict[str, Dict[str, Any]] = {}
    
    def record_market_data(
        self,
        store_id: str,
        data_type: str,  # "pricing", "demand", "competition", "trends"
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registrar dato de mercado"""
        
        data_entry = {
            "data_id": f"data_{store_id}_{len(self.market_data.get(store_id, [])) + 1}",
            "store_id": store_id,
            "type": data_type,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        if store_id not in self.market_data:
            self.market_data[store_id] = []
        
        self.market_data[store_id].append(data_entry)
        
        return data_entry
    
    async def analyze_market_trends(
        self,
        store_id: str,
        category: str
    ) -> Dict[str, Any]:
        """Analizar tendencias de mercado"""
        
        store_data = self.market_data.get(store_id, [])
        category_data = [d for d in store_data if d.get("metadata", {}).get("category") == category]
        
        if self.llm_service.client:
            try:
                prompt = f"""Analiza las tendencias de mercado para la categoría {category} en la tienda {store_id}.
                
                Datos disponibles: {category_data[-20:] if category_data else []}
                
                Proporciona:
                1. Tendencias identificadas
                2. Oportunidades de mercado
                3. Amenazas detectadas
                4. Recomendaciones estratégicas"""
                
                result = await self.llm_service.generate_structured(
                    prompt=prompt,
                    system_prompt="Eres un experto en análisis de mercado y tendencias."
                )
                
                return {
                    "store_id": store_id,
                    "category": category,
                    "trends": result if result else self._generate_basic_trends(),
                    "analyzed_at": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error analizando tendencias: {e}")
                return self._generate_basic_trends_response(store_id, category)
        else:
            return self._generate_basic_trends_response(store_id, category)
    
    def _generate_basic_trends(self) -> Dict[str, Any]:
        """Generar tendencias básicas"""
        return {
            "trends": ["Tendencia creciente", "Estacionalidad detectada"],
            "opportunities": ["Oportunidad en segmento X", "Expansión posible"],
            "threats": ["Competencia intensa", "Cambios regulatorios"],
            "recommendations": ["Ajustar estrategia", "Monitorear competencia"]
        }
    
    def _generate_basic_trends_response(
        self,
        store_id: str,
        category: str
    ) -> Dict[str, Any]:
        """Generar respuesta básica"""
        return {
            "store_id": store_id,
            "category": category,
            "trends": self._generate_basic_trends(),
            "analyzed_at": datetime.now().isoformat()
        }
    
    async def get_market_intelligence(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Obtener inteligencia de mercado"""
        
        store_data = self.market_data.get(store_id, [])
        
        # Agrupar por tipo
        by_type = {}
        for entry in store_data:
            data_type = entry["type"]
            if data_type not in by_type:
                by_type[data_type] = []
            by_type[data_type].append(entry)
        
        intelligence = {
            "store_id": store_id,
            "data_sources": list(by_type.keys()),
            "total_data_points": len(store_data),
            "last_updated": store_data[-1]["timestamp"] if store_data else None,
            "summary": {
                "pricing_trends": "stable",
                "demand_level": "moderate",
                "competition_level": "high"
            },
            "generated_at": datetime.now().isoformat()
        }
        
        self.insights[store_id] = intelligence
        
        return intelligence




