"""
Advanced E-commerce API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal

from ....services.advanced_ecommerce_service import AdvancedEcommerceService, ProductStatus, OrderStatus, PaymentMethodType, SubscriptionStatus
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateProductRequest(BaseModel):
    """Request model for creating a product."""
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    price: Decimal = Field(..., description="Product price")
    category_id: str = Field(..., description="Category ID")
    sku: Optional[str] = Field(default=None, description="Product SKU")
    status: str = Field(default="draft", description="Product status")
    tags: Optional[List[str]] = Field(default=None, description="Product tags")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Product attributes")
    images: Optional[List[str]] = Field(default=None, description="Product images")
    variants: Optional[List[Dict[str, Any]]] = Field(default=None, description="Product variants")
    inventory: Optional[int] = Field(default=None, description="Initial inventory")


class UpdateProductRequest(BaseModel):
    """Request model for updating a product."""
    name: Optional[str] = Field(default=None, description="Product name")
    description: Optional[str] = Field(default=None, description="Product description")
    price: Optional[Decimal] = Field(default=None, description="Product price")
    status: Optional[str] = Field(default=None, description="Product status")
    tags: Optional[List[str]] = Field(default=None, description="Product tags")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Product attributes")


class AddToCartRequest(BaseModel):
    """Request model for adding to cart."""
    product_id: str = Field(..., description="Product ID")
    quantity: int = Field(default=1, ge=1, description="Quantity")
    variant_id: Optional[str] = Field(default=None, description="Variant ID")


class CreateOrderRequest(BaseModel):
    """Request model for creating an order."""
    shipping_address: Dict[str, Any] = Field(..., description="Shipping address")
    payment_method_id: str = Field(..., description="Payment method ID")
    coupon_code: Optional[str] = Field(default=None, description="Coupon code")


class ProcessPaymentRequest(BaseModel):
    """Request model for processing payment."""
    payment_method: str = Field(..., description="Payment method")
    payment_data: Dict[str, Any] = Field(..., description="Payment data")


class CreateSubscriptionRequest(BaseModel):
    """Request model for creating a subscription."""
    plan_id: str = Field(..., description="Subscription plan ID")
    payment_method_id: str = Field(..., description="Payment method ID")


class ProductListRequest(BaseModel):
    """Request model for listing products."""
    category_id: Optional[str] = Field(default=None, description="Category ID")
    status: Optional[str] = Field(default=None, description="Product status")
    user_id: Optional[str] = Field(default=None, description="User ID")
    min_price: Optional[Decimal] = Field(default=None, description="Minimum price")
    max_price: Optional[Decimal] = Field(default=None, description="Maximum price")
    tags: Optional[List[str]] = Field(default=None, description="Product tags")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")


async def get_ecommerce_service(session: DatabaseSessionDep) -> AdvancedEcommerceService:
    """Get e-commerce service instance."""
    return AdvancedEcommerceService(session)


@router.post("/products", response_model=Dict[str, Any])
async def create_product(
    request: CreateProductRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new product."""
    try:
        # Convert status to enum
        try:
            status_enum = ProductStatus(request.status.lower())
        except ValueError:
            raise ValidationError(f"Invalid product status: {request.status}")
        
        result = await ecommerce_service.create_product(
            name=request.name,
            description=request.description,
            price=request.price,
            category_id=request.category_id,
            user_id=str(current_user.id),
            sku=request.sku,
            status=status_enum,
            tags=request.tags,
            attributes=request.attributes,
            images=request.images,
            variants=request.variants,
            inventory=request.inventory
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Product created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create product"
        )


@router.put("/products/{product_id}", response_model=Dict[str, Any])
async def update_product(
    product_id: str,
    request: UpdateProductRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Update an existing product."""
    try:
        # Convert status to enum if provided
        status_enum = None
        if request.status:
            try:
                status_enum = ProductStatus(request.status.lower())
            except ValueError:
                raise ValidationError(f"Invalid product status: {request.status}")
        
        result = await ecommerce_service.update_product(
            product_id=product_id,
            name=request.name,
            description=request.description,
            price=request.price,
            status=status_enum,
            tags=request.tags,
            attributes=request.attributes
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Product updated successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update product"
        )


@router.get("/products/{product_id}", response_model=Dict[str, Any])
async def get_product(
    product_id: str,
    include_variants: bool = Query(default=True, description="Include variants"),
    include_reviews: bool = Query(default=True, description="Include reviews"),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Get product by ID."""
    try:
        result = await ecommerce_service.get_product(
            product_id=product_id,
            include_variants=include_variants,
            include_reviews=include_reviews
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Product retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get product"
        )


@router.post("/products/list", response_model=Dict[str, Any])
async def list_products(
    request: ProductListRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """List products with filtering and pagination."""
    try:
        # Convert status to enum if provided
        status_enum = None
        if request.status:
            try:
                status_enum = ProductStatus(request.status.lower())
            except ValueError:
                raise ValidationError(f"Invalid product status: {request.status}")
        
        result = await ecommerce_service.list_products(
            category_id=request.category_id,
            status=status_enum,
            user_id=request.user_id,
            min_price=request.min_price,
            max_price=request.max_price,
            tags=request.tags,
            page=request.page,
            page_size=request.page_size,
            sort_by=request.sort_by,
            sort_order=request.sort_order
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Products retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list products"
        )


@router.get("/products", response_model=Dict[str, Any])
async def list_products_get(
    category_id: Optional[str] = Query(default=None, description="Category ID"),
    status: Optional[str] = Query(default=None, description="Product status"),
    user_id: Optional[str] = Query(default=None, description="User ID"),
    min_price: Optional[Decimal] = Query(default=None, description="Minimum price"),
    max_price: Optional[Decimal] = Query(default=None, description="Maximum price"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    sort_by: str = Query(default="created_at", description="Sort field"),
    sort_order: str = Query(default="desc", description="Sort order"),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """List products via GET request."""
    try:
        # Convert status to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = ProductStatus(status.lower())
            except ValueError:
                raise ValidationError(f"Invalid product status: {status}")
        
        # Parse tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        result = await ecommerce_service.list_products(
            category_id=category_id,
            status=status_enum,
            user_id=user_id,
            min_price=min_price,
            max_price=max_price,
            tags=tag_list,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Products retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list products"
        )


@router.post("/cart/add", response_model=Dict[str, Any])
async def add_to_cart(
    request: AddToCartRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Add product to cart."""
    try:
        result = await ecommerce_service.add_to_cart(
            user_id=str(current_user.id),
            product_id=request.product_id,
            quantity=request.quantity,
            variant_id=request.variant_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Product added to cart successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add to cart"
        )


@router.get("/cart", response_model=Dict[str, Any])
async def get_cart(
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Get user's cart."""
    try:
        result = await ecommerce_service.get_cart(user_id=str(current_user.id))
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Cart retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cart"
        )


@router.post("/orders", response_model=Dict[str, Any])
async def create_order(
    request: CreateOrderRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Create an order from cart."""
    try:
        result = await ecommerce_service.create_order(
            user_id=str(current_user.id),
            shipping_address=request.shipping_address,
            payment_method_id=request.payment_method_id,
            coupon_code=request.coupon_code
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Order created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create order"
        )


@router.post("/orders/{order_id}/payment", response_model=Dict[str, Any])
async def process_payment(
    order_id: str,
    request: ProcessPaymentRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Process payment for an order."""
    try:
        # Convert payment method to enum
        try:
            payment_method = PaymentMethodType(request.payment_method.lower())
        except ValueError:
            raise ValidationError(f"Invalid payment method: {request.payment_method}")
        
        result = await ecommerce_service.process_payment(
            order_id=order_id,
            payment_method=payment_method,
            payment_data=request.payment_data
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Payment processed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process payment"
        )


@router.post("/subscriptions", response_model=Dict[str, Any])
async def create_subscription(
    request: CreateSubscriptionRequest = Depends(),
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a subscription."""
    try:
        result = await ecommerce_service.create_subscription(
            user_id=str(current_user.id),
            plan_id=request.plan_id,
            payment_method_id=request.payment_method_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Subscription created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_ecommerce_stats(
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Get e-commerce statistics."""
    try:
        result = await ecommerce_service.get_ecommerce_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "E-commerce statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get e-commerce statistics"
        )


@router.get("/product-statuses", response_model=Dict[str, Any])
async def get_product_statuses():
    """Get available product statuses."""
    product_statuses = {
        "draft": {
            "name": "Draft",
            "description": "Product is being worked on and not yet published",
            "visibility": "Private",
            "purchasable": False
        },
        "active": {
            "name": "Active",
            "description": "Product is live and available for purchase",
            "visibility": "Public",
            "purchasable": True
        },
        "inactive": {
            "name": "Inactive",
            "description": "Product is temporarily unavailable",
            "visibility": "Hidden",
            "purchasable": False
        },
        "out_of_stock": {
            "name": "Out of Stock",
            "description": "Product is temporarily out of stock",
            "visibility": "Public",
            "purchasable": False
        },
        "discontinued": {
            "name": "Discontinued",
            "description": "Product is no longer available",
            "visibility": "Hidden",
            "purchasable": False
        }
    }
    
    return {
        "success": True,
        "data": {
            "product_statuses": product_statuses,
            "total_statuses": len(product_statuses)
        },
        "message": "Product statuses retrieved successfully"
    }


@router.get("/order-statuses", response_model=Dict[str, Any])
async def get_order_statuses():
    """Get available order statuses."""
    order_statuses = {
        "pending": {
            "name": "Pending",
            "description": "Order is awaiting payment confirmation",
            "actionable": True,
            "next_statuses": ["confirmed", "cancelled"]
        },
        "confirmed": {
            "name": "Confirmed",
            "description": "Order is confirmed and being processed",
            "actionable": True,
            "next_statuses": ["processing", "cancelled"]
        },
        "processing": {
            "name": "Processing",
            "description": "Order is being prepared for shipment",
            "actionable": True,
            "next_statuses": ["shipped", "cancelled"]
        },
        "shipped": {
            "name": "Shipped",
            "description": "Order has been shipped",
            "actionable": True,
            "next_statuses": ["delivered"]
        },
        "delivered": {
            "name": "Delivered",
            "description": "Order has been delivered",
            "actionable": False,
            "next_statuses": ["refunded"]
        },
        "cancelled": {
            "name": "Cancelled",
            "description": "Order has been cancelled",
            "actionable": False,
            "next_statuses": []
        },
        "refunded": {
            "name": "Refunded",
            "description": "Order has been refunded",
            "actionable": False,
            "next_statuses": []
        }
    }
    
    return {
        "success": True,
        "data": {
            "order_statuses": order_statuses,
            "total_statuses": len(order_statuses)
        },
        "message": "Order statuses retrieved successfully"
    }


@router.get("/payment-methods", response_model=Dict[str, Any])
async def get_payment_methods():
    """Get available payment methods."""
    payment_methods = {
        "credit_card": {
            "name": "Credit Card",
            "description": "Pay with credit card",
            "gateway": "Stripe",
            "fees": "2.9% + $0.30",
            "processing_time": "Instant"
        },
        "debit_card": {
            "name": "Debit Card",
            "description": "Pay with debit card",
            "gateway": "Stripe",
            "fees": "2.9% + $0.30",
            "processing_time": "Instant"
        },
        "paypal": {
            "name": "PayPal",
            "description": "Pay with PayPal account",
            "gateway": "PayPal",
            "fees": "2.9% + $0.30",
            "processing_time": "Instant"
        },
        "stripe": {
            "name": "Stripe",
            "description": "Pay with Stripe",
            "gateway": "Stripe",
            "fees": "2.9% + $0.30",
            "processing_time": "Instant"
        },
        "bank_transfer": {
            "name": "Bank Transfer",
            "description": "Pay via bank transfer",
            "gateway": "Direct",
            "fees": "Free",
            "processing_time": "1-3 business days"
        },
        "cryptocurrency": {
            "name": "Cryptocurrency",
            "description": "Pay with cryptocurrency",
            "gateway": "Crypto",
            "fees": "Variable",
            "processing_time": "10-60 minutes"
        },
        "wallet": {
            "name": "Digital Wallet",
            "description": "Pay with digital wallet",
            "gateway": "Wallet",
            "fees": "1-3%",
            "processing_time": "Instant"
        }
    }
    
    return {
        "success": True,
        "data": {
            "payment_methods": payment_methods,
            "total_methods": len(payment_methods)
        },
        "message": "Payment methods retrieved successfully"
    }


@router.get("/subscription-statuses", response_model=Dict[str, Any])
async def get_subscription_statuses():
    """Get available subscription statuses."""
    subscription_statuses = {
        "active": {
            "name": "Active",
            "description": "Subscription is active and billing",
            "billing": True,
            "access": True
        },
        "inactive": {
            "name": "Inactive",
            "description": "Subscription is temporarily inactive",
            "billing": False,
            "access": False
        },
        "cancelled": {
            "name": "Cancelled",
            "description": "Subscription has been cancelled",
            "billing": False,
            "access": False
        },
        "expired": {
            "name": "Expired",
            "description": "Subscription has expired",
            "billing": False,
            "access": False
        },
        "suspended": {
            "name": "Suspended",
            "description": "Subscription has been suspended",
            "billing": False,
            "access": False
        }
    }
    
    return {
        "success": True,
        "data": {
            "subscription_statuses": subscription_statuses,
            "total_statuses": len(subscription_statuses)
        },
        "message": "Subscription statuses retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_ecommerce_health(
    ecommerce_service: AdvancedEcommerceService = Depends(get_ecommerce_service),
    current_user: CurrentUserDep = Depends()
):
    """Get e-commerce system health status."""
    try:
        # Get e-commerce stats
        stats = await ecommerce_service.get_ecommerce_stats()
        
        # Calculate health metrics
        total_products = stats["data"].get("total_products", 0)
        total_orders = stats["data"].get("total_orders", 0)
        total_revenue = stats["data"].get("total_revenue", 0)
        active_subscriptions = stats["data"].get("active_subscriptions", 0)
        products_by_status = stats["data"].get("products_by_status", {})
        orders_by_status = stats["data"].get("orders_by_status", {})
        
        # Calculate health score
        health_score = 100
        
        # Check product distribution
        active_products = products_by_status.get("active", 0)
        if total_products > 0:
            active_ratio = active_products / total_products
            if active_ratio < 0.5:
                health_score -= 20
            elif active_ratio > 0.9:
                health_score -= 10
        
        # Check order completion rate
        completed_orders = orders_by_status.get("delivered", 0)
        if total_orders > 0:
            completion_rate = completed_orders / total_orders
            if completion_rate < 0.7:
                health_score -= 25
            elif completion_rate > 0.95:
                health_score -= 5
        
        # Check revenue growth (placeholder)
        if total_revenue < 1000:
            health_score -= 15
        
        # Check subscription health
        if active_subscriptions < 10:
            health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_products": total_products,
                "active_products": active_products,
                "total_orders": total_orders,
                "completed_orders": completed_orders,
                "total_revenue": total_revenue,
                "active_subscriptions": active_subscriptions,
                "active_product_ratio": active_ratio if total_products > 0 else 0,
                "order_completion_rate": completion_rate if total_orders > 0 else 0,
                "products_by_status": products_by_status,
                "orders_by_status": orders_by_status,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "E-commerce health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get e-commerce health status"
        )
























