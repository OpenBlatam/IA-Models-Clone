"""
Material Search - Sistema de búsqueda de materiales en tiempo real
==================================================================
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class MaterialSearchEngine:
    """Motor de búsqueda de materiales con caché y múltiples fuentes"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=24)
        
    async def search_material(self, material_name: str, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca materiales en múltiples fuentes
        
        Args:
            material_name: Nombre del material a buscar
            location: Ubicación para búsqueda localizada
            
        Returns:
            Lista de resultados con precios y fuentes
        """
        cache_key = f"{material_name.lower()}_{location or 'global'}"
        
        # Verificar caché
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if datetime.now() - cached_result["timestamp"] < self.cache_ttl:
                logger.info(f"Resultado desde caché para: {material_name}")
                return cached_result["results"]
        
        # Búsqueda simulada (en producción se conectaría a APIs reales)
        results = await self._search_multiple_sources(material_name, location)
        
        # Guardar en caché
        self.cache[cache_key] = {
            "results": results,
            "timestamp": datetime.now()
        }
        
        return results
    
    async def _search_multiple_sources(self, material_name: str, location: Optional[str]) -> List[Dict[str, Any]]:
        """Busca en múltiples fuentes de materiales"""
        results = []
        material_lower = material_name.lower()
        
        # Simulación de búsqueda en diferentes proveedores
        sources = [
            {
                "name": "Amazon",
                "url": f"https://www.amazon.com/s?k={material_name.replace(' ', '+')}",
                "location": "Online",
                "price_range": (10.0, 50.0),
                "availability": "Inmediata",
                "rating": 4.5
            },
            {
                "name": "Home Depot",
                "url": f"https://www.homedepot.com/s/{material_name.replace(' ', '%20')}",
                "location": location or "Nacional",
                "price_range": (8.0, 45.0),
                "availability": "En tienda",
                "rating": 4.3
            },
            {
                "name": "Lowe's",
                "url": f"https://www.lowes.com/search?searchTerm={material_name.replace(' ', '+')}",
                "location": location or "Nacional",
                "price_range": (9.0, 48.0),
                "availability": "En tienda",
                "rating": 4.2
            },
            {
                "name": "MercadoLibre",
                "url": f"https://listado.mercadolibre.com.mx/{material_name.replace(' ', '-')}",
                "location": "México" if not location else location,
                "price_range": (7.0, 40.0),
                "availability": "Online",
                "rating": 4.4
            }
        ]
        
        # Agregar resultados según el tipo de material
        for source in sources:
            # Calcular precio estimado basado en el tipo de material
            base_price = self._estimate_price(material_lower)
            price = base_price * (source["price_range"][0] + source["price_range"][1]) / 2 / 30
            
            results.append({
                "source": source["name"],
                "url": source["url"],
                "location": source["location"],
                "estimated_price": round(price, 2),
                "price_range": source["price_range"],
                "availability": source["availability"],
                "rating": source["rating"],
                "delivery_time": "1-3 días" if "Online" in source["availability"] else "Inmediato"
            })
        
        return results
    
    def _estimate_price(self, material_name: str) -> float:
        """Estima el precio base según el tipo de material"""
        price_map = {
            "acero": 2.5,
            "aluminio": 1.8,
            "plastico": 3.0,
            "vidrio": 1.5,
            "motor": 25.0,
            "cable": 0.5,
            "tornillo": 0.02,
            "quemador": 15.0,
            "valvula": 8.0
        }
        
        for key, price in price_map.items():
            if key in material_name:
                return price
        
        return 1.0  # Precio genérico
    
    def clear_cache(self):
        """Limpia el caché"""
        self.cache.clear()
        logger.info("Caché de materiales limpiado")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        return {
            "cache_size": len(self.cache),
            "cached_materials": list(self.cache.keys())
        }




