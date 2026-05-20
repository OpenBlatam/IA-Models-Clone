from typing import Dict, Any

from ...domain.entities import Product


class ProductMapper:
    
    @staticmethod
    def to_dict(product: Product) -> Dict[str, Any]:
        return {
            "id": product.id,
            "name": product.name,
            "category": product.category,
            "description": product.description,
            "ingredients": product.ingredients,
            "price": product.price,
            "rating": product.rating,
            "metadata": product.metadata,
        }
    
    @staticmethod
    def to_entity(data: Dict[str, Any]) -> Product:
        return Product(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            description=data.get("description"),
            ingredients=data.get("ingredients", []),
            price=data.get("price"),
            rating=data.get("rating"),
            metadata=data.get("metadata", {})
        )















