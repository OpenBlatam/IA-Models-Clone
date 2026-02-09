"""
Intelligent Inventory Service - Sistema de inventario inteligente
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class IntelligentInventoryService:
    """Servicio para inventario inteligente"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.inventory: Dict[str, Dict[str, Any]] = {}
        self.transactions: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: Dict[str, Dict[str, Any]] = {}
    
    def add_product(
        self,
        store_id: str,
        product_name: str,
        sku: str,
        initial_stock: int,
        unit_price: float,
        category: Optional[str] = None,
        reorder_point: Optional[int] = None
    ) -> Dict[str, Any]:
        """Agregar producto al inventario"""
        
        product_id = f"prod_{store_id}_{sku}"
        
        product = {
            "product_id": product_id,
            "store_id": store_id,
            "name": product_name,
            "sku": sku,
            "stock": initial_stock,
            "unit_price": unit_price,
            "category": category,
            "reorder_point": reorder_point or (initial_stock * 0.2),  # 20% por defecto
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        if store_id not in self.inventory:
            self.inventory[store_id] = {}
        
        self.inventory[store_id][product_id] = product
        
        return product
    
    def record_transaction(
        self,
        product_id: str,
        transaction_type: str,  # "sale", "restock", "return", "adjustment"
        quantity: int,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar transacción de inventario"""
        
        # Encontrar producto
        product = None
        for store_inventory in self.inventory.values():
            if product_id in store_inventory:
                product = store_inventory[product_id]
                break
        
        if not product:
            raise ValueError(f"Producto {product_id} no encontrado")
        
        # Actualizar stock
        if transaction_type == "sale":
            product["stock"] -= quantity
        elif transaction_type == "restock":
            product["stock"] += quantity
        elif transaction_type == "return":
            product["stock"] += quantity
        elif transaction_type == "adjustment":
            product["stock"] = quantity
        
        transaction = {
            "transaction_id": f"txn_{product_id}_{len(self.transactions.get(product_id, [])) + 1}",
            "product_id": product_id,
            "type": transaction_type,
            "quantity": quantity,
            "stock_before": product["stock"] - (quantity if transaction_type == "restock" else -quantity if transaction_type == "sale" else 0),
            "stock_after": product["stock"],
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        if product_id not in self.transactions:
            self.transactions[product_id] = []
        
        self.transactions[product_id].append(transaction)
        product["last_updated"] = datetime.now().isoformat()
        
        return transaction
    
    async def predict_demand(
        self,
        product_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Predecir demanda usando ML"""
        
        transactions = self.transactions.get(product_id, [])
        
        if len(transactions) < 10:
            return {
                "product_id": product_id,
                "prediction": "insufficient_data",
                "message": "Se necesitan más transacciones para predecir"
            }
        
        # Análisis básico de tendencia
        recent_transactions = transactions[-30:]  # Últimos 30 días
        sales = [t for t in recent_transactions if t["type"] == "sale"]
        
        if not sales:
            return {
                "product_id": product_id,
                "prediction": "no_sales",
                "message": "No hay ventas recientes"
            }
        
        # Calcular promedio diario
        daily_sales = {}
        for sale in sales:
            date = datetime.fromisoformat(sale["timestamp"]).date()
            daily_sales[date] = daily_sales.get(date, 0) + sale["quantity"]
        
        avg_daily = sum(daily_sales.values()) / len(daily_sales) if daily_sales else 0
        predicted_demand = avg_daily * days
        
        # Usar LLM para análisis más sofisticado si está disponible
        if self.llm_service.client:
            try:
                analysis = await self._analyze_demand_with_llm(transactions, days)
                predicted_demand = analysis.get("predicted_demand", predicted_demand)
            except Exception as e:
                logger.error(f"Error en análisis LLM: {e}")
        
        return {
            "product_id": product_id,
            "prediction_period_days": days,
            "predicted_demand": round(predicted_demand, 2),
            "average_daily_demand": round(avg_daily, 2),
            "confidence": 0.7,  # En producción, calcular confianza real
            "recommendations": self._generate_recommendations(product_id, predicted_demand)
        }
    
    async def _analyze_demand_with_llm(
        self,
        transactions: List[Dict[str, Any]],
        days: int
    ) -> Dict[str, Any]:
        """Analizar demanda usando LLM"""
        prompt = f"""Analiza estas transacciones de inventario y predice la demanda para los próximos {days} días.
        
        Transacciones: {transactions[-20:]}  # Últimas 20
        
        Proporciona:
        - Predicción de demanda
        - Factores considerados
        - Recomendaciones"""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de inventario y predicción de demanda."
        )
        
        return result if result else {}
    
    def _generate_recommendations(
        self,
        product_id: str,
        predicted_demand: float
    ) -> List[str]:
        """Generar recomendaciones"""
        product = None
        for store_inventory in self.inventory.values():
            if product_id in store_inventory:
                product = store_inventory[product_id]
                break
        
        if not product:
            return []
        
        recommendations = []
        current_stock = product["stock"]
        reorder_point = product.get("reorder_point", 0)
        
        if current_stock < reorder_point:
            recommendations.append("Stock bajo - considerar reorden inmediato")
        
        if predicted_demand > current_stock:
            recommendations.append(f"Demanda proyectada ({predicted_demand:.0f}) excede stock actual ({current_stock})")
        
        if current_stock > predicted_demand * 2:
            recommendations.append("Stock excesivo - considerar promoción")
        
        return recommendations
    
    def get_low_stock_alerts(
        self,
        store_id: str
    ) -> List[Dict[str, Any]]:
        """Obtener alertas de stock bajo"""
        
        alerts = []
        inventory = self.inventory.get(store_id, {})
        
        for product_id, product in inventory.items():
            if product["stock"] <= product.get("reorder_point", 0):
                alerts.append({
                    "product_id": product_id,
                    "product_name": product["name"],
                    "current_stock": product["stock"],
                    "reorder_point": product["reorder_point"],
                    "urgency": "high" if product["stock"] == 0 else "medium"
                })
        
        return alerts
    
    def get_inventory_turnover(
        self,
        product_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calcular rotación de inventario"""
        
        transactions = self.transactions.get(product_id, [])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        period_transactions = [
            t for t in transactions
            if start_date <= datetime.fromisoformat(t["timestamp"]) <= end_date
        ]
        
        sales = [t for t in period_transactions if t["type"] == "sale"]
        total_sold = sum(s["quantity"] for s in sales)
        
        product = None
        for store_inventory in self.inventory.values():
            if product_id in store_inventory:
                product = store_inventory[product_id]
                break
        
        if not product:
            return {"error": "Producto no encontrado"}
        
        avg_inventory = product["stock"]  # Simplificado
        turnover = (total_sold / avg_inventory) if avg_inventory > 0 else 0
        
        return {
            "product_id": product_id,
            "period_days": days,
            "total_sold": total_sold,
            "average_inventory": avg_inventory,
            "turnover_rate": round(turnover, 2),
            "days_to_sell": round(days / turnover, 2) if turnover > 0 else float('inf')
        }




