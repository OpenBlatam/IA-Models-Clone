from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from pathlib import Path
import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator, root_validator
from .config import ModelConfig
from agents.backend.onyx.server.features.utils.base_model import OnyxBaseModel
from agents.backend.onyx.server.features.utils.value_objects import Money, Dimensions, SEOData
from agents.backend.onyx.server.features.utils.enums import ProductStatus, ProductType, PriceType, InventoryTracking
from agents.backend.onyx.server.features.utils.validators import not_empty_string, list_or_empty, dict_or_empty
from typing import Any, List, Dict, Optional
import asyncio
"""
Enhanced Product Model - Enterprise Architecture
===============================================

Modelo empresarial mejorado para productos con Clean Architecture,
funcionalidades avanzadas y optimizaciones de rendimiento.
"""



logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED DOMAIN MODELS
# ============================================================================

class ProductStatus(str, Enum):
    """Estados del producto"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


class ProductType(str, Enum):
    """Tipos de producto"""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    BUNDLE = "bundle"


class PriceType(str, Enum):
    """Tipos de precio"""
    FIXED = "fixed"
    VARIABLE = "variable"
    TIERED = "tiered"
    SUBSCRIPTION = "subscription"


class InventoryTracking(str, Enum):
    """Seguimiento de inventario"""
    TRACK = "track"
    NO_TRACK = "no_track"
    BACKORDER = "backorder"


class Money(BaseModel):
    """Value object para representar dinero"""
    amount: Decimal = Field(..., gt=0, description="Monto positivo")
    currency: str = Field("USD", min_length=3, max_length=3, description="Código de moneda ISO 4217")

    @validator('currency')
    def validate_currency(cls, v) -> bool:
        if len(v) != 3:
            raise ValueError("Código de moneda debe tener 3 caracteres")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {"amount": float(self.amount), "currency": self.currency}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Money":
        return cls(amount=Decimal(str(data["amount"])), currency=data["currency"])


class Dimensions(BaseModel):
    """Value object para dimensiones del producto"""
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)
    unit: str = Field("cm")
    weight_unit: str = Field("kg")
    
    def volume(self) -> float:
        return self.length * self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "weight": self.weight,
            "unit": self.unit,
            "weight_unit": self.weight_unit,
            "volume": self.volume()
        }


class SEOData(BaseModel):
    """Value object para datos SEO"""
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    slug: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class EnhancedProductEntity(OnyxBaseModel):
    """
    Entidad de producto empresarial mejorada
    
    Funcionalidades empresariales:
    - Gestión avanzada de inventario
    - Variantes de producto
    - Precios dinámicos y ofertas
    - SEO optimizado
    - Integración con IA
    - Análisis de rentabilidad
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=2, max_length=128, description="Product name")
    description: str = Field(default="", description="Product description")
    short_description: str = Field(default="", description="Short description")
    sku: str = Field(default="", description="SKU")
    product_type: ProductType = Field(default=ProductType.PHYSICAL)
    status: ProductStatus = Field(default=ProductStatus.DRAFT)
    brand_id: Optional[str] = None
    category_id: Optional[str] = None
    base_price: Optional[Money] = None
    sale_price: Optional[Money] = None
    cost_price: Optional[Money] = None
    price_type: PriceType = Field(default=PriceType.FIXED)
    inventory_quantity: int = Field(default=0, ge=0)
    low_stock_threshold: int = Field(default=10, ge=0)
    inventory_tracking: InventoryTracking = Field(default=InventoryTracking.TRACK)
    allow_backorder: bool = Field(default=False)
    dimensions: Optional[Dimensions] = None
    requires_shipping: bool = Field(default=True)
    download_url: Optional[str] = None
    download_limit: Optional[int] = None
    seo_data: SEOData = Field(default_factory=SEOData)
    tags: set[str] = Field(default_factory=set)
    featured: bool = Field(default=False)
    attributes: dict[str, any] = Field(default_factory=dict)
    custom_fields: dict[str, any] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    ai_generated_description: Optional[str] = None
    ai_confidence_score: Optional[float] = None
    ai_last_updated: Optional[datetime] = None

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v) -> Any:
        return not_empty_string(v)

    @field_validator('tags', 'images', 'videos', mode='before')
    @classmethod
    def list_or_empty_validator(cls, v) -> List[Any]:
        return list_or_empty(v)

    @field_validator('attributes', 'custom_fields', mode='before')
    @classmethod
    def dict_or_empty_validator(cls, v) -> Any:
        return dict_or_empty(v)

    @validator('inventory_quantity', 'low_stock_threshold', pre=True, always=True)
    def non_negative(cls, v) -> Any:
        if v < 0:
            raise ValueError('Debe ser no negativo')
        return v

    def set_price(self, price: Money, price_type: PriceType = PriceType.FIXED) -> None:
        """Establece el precio base del producto"""
        if price.amount <= 0:
            raise ValueError("El precio debe ser mayor a 0")
        
        self.base_price = price
        self.price_type = price_type
        self.updated_at = datetime.utcnow()
    
    def set_sale_price(self, sale_price: Money) -> None:
        """Establece precio de oferta"""
        if self.base_price and sale_price.amount >= self.base_price.amount:
            raise ValueError("El precio de oferta debe ser menor al precio base")
        
        self.sale_price = sale_price
        self.updated_at = datetime.utcnow()
    
    def get_effective_price(self) -> Optional[Money]:
        """Obtiene el precio efectivo (oferta o base)"""
        return self.sale_price or self.base_price
    
    def is_on_sale(self) -> bool:
        """Verifica si el producto está en oferta"""
        return self.sale_price is not None
    
    def calculate_discount_percentage(self) -> float:
        """Calcula el porcentaje de descuento"""
        if not self.is_on_sale() or not self.base_price:
            return 0.0
        
        discount = self.base_price.amount - self.sale_price.amount
        return float((discount / self.base_price.amount) * 100)
    
    def update_inventory(self, quantity: int, operation: str = "set") -> None:
        """Actualiza el inventario"""
        if self.inventory_tracking == InventoryTracking.NO_TRACK:
            return
        
        if operation == "set":
            self.inventory_quantity = max(0, quantity)
        elif operation == "add":
            self.inventory_quantity += quantity
        elif operation == "subtract":
            if not self.allow_backorder and quantity > self.inventory_quantity:
                raise ValueError("Cantidad insuficiente en inventario")
            self.inventory_quantity -= quantity
        
        self.updated_at = datetime.utcnow()
        
        # Actualizar estado si está agotado
        if self.inventory_quantity <= 0 and not self.allow_backorder:
            self.status = ProductStatus.OUT_OF_STOCK
    
    def is_low_stock(self) -> bool:
        """Verifica si el stock está bajo"""
        if self.inventory_tracking == InventoryTracking.NO_TRACK:
            return False
        return self.inventory_quantity <= self.low_stock_threshold
    
    def is_in_stock(self) -> bool:
        """Verifica si está en stock"""
        if self.inventory_tracking == InventoryTracking.NO_TRACK:
            return True
        return self.inventory_quantity > 0 or self.allow_backorder
    
    def publish(self) -> None:
        """Publica el producto"""
        if self.status == ProductStatus.DRAFT:
            self.status = ProductStatus.ACTIVE
            self.published_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()
    
    def set_ai_description(self, description: str, confidence: float = 0.0) -> None:
        """Establece descripción generada por IA"""
        self.ai_generated_description = description
        self.ai_confidence_score = confidence
        self.ai_last_updated = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def calculate_profit_margin(self) -> Optional[float]:
        """Calcula el margen de ganancia"""
        effective_price = self.get_effective_price()
        if not effective_price or not self.cost_price:
            return None
        
        profit = effective_price.amount - self.cost_price.amount
        return float((profit / effective_price.amount) * 100)
    
    def validate(self) -> List[str]:
        """Valida la entidad y retorna lista de errores"""
        errors = []
        
        if not self.name or len(self.name.strip()) < 2:
            errors.append("El nombre del producto debe tener al menos 2 caracteres")
        
        if not self.sku or len(self.sku.strip()) < 1:
            errors.append("El SKU es requerido")
        
        if self.base_price and self.base_price.amount <= 0:
            errors.append("El precio base debe ser mayor a 0")
        
        if self.sale_price and self.base_price:
            if self.sale_price.amount >= self.base_price.amount:
                errors.append("El precio de oferta debe ser menor al precio base")
        
        if self.inventory_quantity < 0:
            errors.append("La cantidad en inventario no puede ser negativa")
        
        return errors
    
    def is_valid(self) -> bool:
        """Verifica si la entidad es válida"""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la entidad a diccionario con campos calculados"""
        return self.dict()


class ProductDescriptionModel(nn.Module):
    """
    Advanced Product Description Generation Model
    
    Features:
    - Transformer-based architecture with product-specific enhancements
    - Multi-head attention with context awareness
    - Mixed precision training support
    - Style and tone conditioning
    - SEO optimization
    """
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Load pre-trained model and tokenizer
        self.model_config = AutoConfig.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        
        # Add special tokens
        self._add_special_tokens()
        
        # Enhanced layers
        self.product_context_encoder = nn.Linear(
            self.model_config.hidden_size, 
            self.model_config.hidden_size
        )
        
        self.style_embeddings = nn.Embedding(10, self.model_config.hidden_size)
        self.tone_embeddings = nn.Embedding(5, self.model_config.hidden_size)
        
        # Quality and SEO heads
        self.quality_head = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.model_config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.seo_head = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.model_config.hidden_size // 2, 3),
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _add_special_tokens(self) -> Any:
        """Add special tokens for product description generation."""
        special_tokens = [
            "[PRODUCT]", "[FEATURES]", "[PRICE]", "[BRAND]", 
            "[CATEGORY]", "[DESCRIPTION]", "[LUXURY]", "[TECHNICAL]"
        ]
        
        tokens_to_add = [token for token in special_tokens 
                        if token not in self.tokenizer.get_vocab()]
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            self.base_model.resize_token_embeddings(len(self.tokenizer))
    
    def _init_weights(self) -> Any:
        """Initialize custom layer weights."""
        for module in [self.product_context_encoder, self.quality_head, self.seo_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def generate_description(
        self,
        product_name: str,
        features: List[str],
        category: str = "general",
        brand: str = "unknown",
        style: str = "professional",
        tone: str = "friendly",
        max_length: int = 300,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[Dict]:
        """Generate product description with enhanced features."""
        
        # Create input prompt
        prompt = self._create_generation_prompt(
            product_name, features, category, brand, style, tone
        )
        
        # Tokenize input
        input_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            padding=True,
            truncation=True
        )
        
        # Generation parameters
        generation_kwargs = {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "max_length": len(input_tokens["input_ids"][0]) + max_length,
            "min_length": len(input_tokens["input_ids"][0]) + 50,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1
        }
        
        # Generate
        with torch.no_grad():
            generated_ids = self.base_model.generate(**generation_kwargs)
        
        # Process results
        results = []
        for i in range(num_return_sequences):
            generated_text = self.tokenizer.decode(
                generated_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            description = self._extract_description(generated_text, prompt)
            quality_score, seo_score = self._compute_scores(description)
            
            results.append({
                "description": description,
                "quality_score": quality_score,
                "seo_score": seo_score,
                "metadata": {
                    "product_name": product_name,
                    "category": category,
                    "brand": brand,
                    "style": style,
                    "tone": tone,
                    "word_count": len(description.split()),
                    "char_count": len(description)
                }
            })
        
        return results
    
    def _create_generation_prompt(
        self, product_name: str, features: List[str], 
        category: str, brand: str, style: str, tone: str
    ) -> str:
        """Create structured prompt for generation."""
        return f"[PRODUCT] {product_name} | [CATEGORY] {category} | [BRAND] {brand} | [FEATURES] {', '.join(features)} | Style: {style} | Tone: {tone} | [DESCRIPTION]"
    
    def _extract_description(self, generated_text: str, prompt: str) -> str:
        """Extract description from generated text."""
        if prompt in generated_text:
            description = generated_text.replace(prompt, "").strip()
        else:
            parts = generated_text.split("[DESCRIPTION]")
            description = parts[-1].strip() if len(parts) > 1 else generated_text.strip()
        
        return description.replace("[PAD]", "").replace("[SEP]", "").strip()
    
    def _compute_scores(self, description: str) -> Tuple[float, float]:
        """Compute quality and SEO scores."""
        word_count = len(description.split())
        sentence_count = len([s for s in description.split('.') if s.strip()])
        
        quality_score = min(1.0, (
            0.4 * min(1.0, word_count / 100) +
            0.3 * min(1.0, sentence_count / 5) +
            0.3 * (1 - min(1.0, description.count('!') / 3))
        ))
        
        seo_score = min(1.0, (
            0.5 * min(1.0, word_count / 80) +
            0.3 * (1 if any(char.isupper() for char in description) else 0) +
            0.2 * (1 if ',' in description else 0)
        ))
        
        return quality_score, seo_score
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), save_path / "model.pt")
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        with open(save_path / "config.json", "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, path: str, config: Optional[ModelConfig] = None):
        """Load model from saved state."""
        load_path = Path(path)
        
        if config is None:
            with open(load_path / "config.json", "r") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
                config = ModelConfig(**config_dict)
        
        model = cls(config)
        model.load_state_dict(torch.load(load_path / "model.pt", map_location=config.device))
        model.tokenizer = AutoTokenizer.from_pretrained(load_path / "tokenizer")
        
        logger.info(f"Model loaded from {load_path}")
        return model 