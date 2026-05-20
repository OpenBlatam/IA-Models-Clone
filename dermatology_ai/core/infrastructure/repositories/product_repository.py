from typing import List, Optional

from ...domain.entities import Product
from ...domain.interfaces import IProductRepository
from ..adapters import IDatabaseAdapter
from ..mappers import ProductMapper


class ProductRepository(IProductRepository):
    
    def __init__(self, database: IDatabaseAdapter):
        self.database = database
        self.table_name = "products"
    
    async def get_by_id(self, product_id: str) -> Optional[Product]:
        data = await self.database.get(self.table_name, {"id": product_id})
        if not data:
            return None
        
        return ProductMapper.to_entity(data)
    
    async def search(self, query: str, limit: int = 10) -> List[Product]:
        results = await self.database.query(
            self.table_name,
            filter_conditions={"name": {"$like": f"%{query}%"}},
            limit=limit
        )
        
        return [ProductMapper.to_entity(data) for data in results]
    
    async def get_by_category(self, category: str, limit: int = 10) -> List[Product]:
        results = await self.database.query(
            self.table_name,
            filter_conditions={"category": category},
            limit=limit
        )
        
        return [ProductMapper.to_entity(data) for data in results]

