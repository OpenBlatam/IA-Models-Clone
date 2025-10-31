"""
Advanced E-commerce Service for comprehensive e-commerce features
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from dataclasses import dataclass
from enum import Enum
import uuid
from decimal import Decimal
import stripe
import paypalrestsdk
from cryptography.fernet import Fernet
import qrcode
from io import BytesIO
import base64

from ..models.database import (
    Product, ProductCategory, ProductVariant, ProductImage, ProductReview,
    Cart, CartItem, Order, OrderItem, Payment, PaymentMethod, Coupon,
    Subscription, SubscriptionPlan, Invoice, Refund, ShippingAddress,
    Wishlist, WishlistItem, ProductTag, ProductAttribute, Inventory
)
from ..core.exceptions import DatabaseError, ValidationError


class ProductStatus(Enum):
    """Product status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentStatus(Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethodType(Enum):
    """Payment method type enumeration."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    STRIPE = "stripe"
    BANK_TRANSFER = "bank_transfer"
    CRYPTOCURRENCY = "cryptocurrency"
    WALLET = "wallet"


class SubscriptionStatus(Enum):
    """Subscription status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


@dataclass
class ProductMetrics:
    """Product metrics structure."""
    product_id: str
    views_count: int
    sales_count: int
    revenue: Decimal
    average_rating: float
    review_count: int
    conversion_rate: float
    inventory_level: int
    restock_alerts: int


@dataclass
class OrderMetrics:
    """Order metrics structure."""
    total_orders: int
    total_revenue: Decimal
    average_order_value: Decimal
    conversion_rate: float
    cart_abandonment_rate: float
    refund_rate: float
    customer_satisfaction: float


class AdvancedEcommerceService:
    """Service for advanced e-commerce operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.ecommerce_cache = {}
        self.stripe_api_key = None
        self.paypal_client_id = None
        self.paypal_client_secret = None
        self._initialize_payment_gateways()
    
    def _initialize_payment_gateways(self):
        """Initialize payment gateways."""
        try:
            # Initialize Stripe
            stripe.api_key = self.stripe_api_key
            
            # Initialize PayPal
            paypalrestsdk.configure({
                "mode": "sandbox",  # or "live"
                "client_id": self.paypal_client_id,
                "client_secret": self.paypal_client_secret
            })
            
        except Exception as e:
            print(f"Warning: Could not initialize payment gateways: {e}")
    
    async def create_product(
        self,
        name: str,
        description: str,
        price: Decimal,
        category_id: str,
        user_id: str,
        sku: Optional[str] = None,
        status: ProductStatus = ProductStatus.DRAFT,
        tags: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        variants: Optional[List[Dict[str, Any]]] = None,
        inventory: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new product."""
        try:
            # Generate product ID
            product_id = str(uuid.uuid4())
            
            # Generate SKU if not provided
            if not sku:
                sku = f"PROD-{product_id[:8].upper()}"
            
            # Create product
            product = Product(
                product_id=product_id,
                name=name,
                description=description,
                price=price,
                sku=sku,
                category_id=category_id,
                user_id=user_id,
                status=status.value,
                tags=tags or [],
                attributes=attributes or {},
                created_at=datetime.utcnow()
            )
            
            self.session.add(product)
            
            # Create inventory record
            if inventory is not None:
                inventory_record = Inventory(
                    product_id=product_id,
                    quantity=inventory,
                    reserved_quantity=0,
                    low_stock_threshold=10,
                    created_at=datetime.utcnow()
                )
                self.session.add(inventory_record)
            
            # Create product images
            if images:
                for i, image_url in enumerate(images):
                    product_image = ProductImage(
                        product_id=product_id,
                        image_url=image_url,
                        alt_text=f"{name} image {i+1}",
                        sort_order=i,
                        is_primary=(i == 0),
                        created_at=datetime.utcnow()
                    )
                    self.session.add(product_image)
            
            # Create product variants
            if variants:
                for variant_data in variants:
                    variant = ProductVariant(
                        product_id=product_id,
                        name=variant_data.get("name", ""),
                        sku=variant_data.get("sku", f"{sku}-{variant_data.get('name', '').upper()}"),
                        price=variant_data.get("price", price),
                        attributes=variant_data.get("attributes", {}),
                        inventory=variant_data.get("inventory", 0),
                        created_at=datetime.utcnow()
                    )
                    self.session.add(variant)
            
            await self.session.commit()
            
            return {
                "success": True,
                "product_id": product_id,
                "sku": sku,
                "message": "Product created successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create product: {str(e)}")
    
    async def update_product(
        self,
        product_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        price: Optional[Decimal] = None,
        status: Optional[ProductStatus] = None,
        tags: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an existing product."""
        try:
            # Get product
            product_query = select(Product).where(Product.product_id == product_id)
            product_result = await self.session.execute(product_query)
            product = product_result.scalar_one_or_none()
            
            if not product:
                raise ValidationError(f"Product with ID {product_id} not found")
            
            # Update fields
            if name is not None:
                product.name = name
            if description is not None:
                product.description = description
            if price is not None:
                product.price = price
            if status is not None:
                product.status = status.value
            if tags is not None:
                product.tags = tags
            if attributes is not None:
                product.attributes.update(attributes)
            
            product.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            return {
                "success": True,
                "product_id": product_id,
                "message": "Product updated successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update product: {str(e)}")
    
    async def get_product(
        self,
        product_id: str,
        include_variants: bool = True,
        include_reviews: bool = True
    ) -> Dict[str, Any]:
        """Get product by ID."""
        try:
            # Get product
            product_query = select(Product).where(Product.product_id == product_id)
            product_result = await self.session.execute(product_query)
            product = product_result.scalar_one_or_none()
            
            if not product:
                raise ValidationError(f"Product with ID {product_id} not found")
            
            result_data = {
                "product_id": product.product_id,
                "name": product.name,
                "description": product.description,
                "price": float(product.price),
                "sku": product.sku,
                "category_id": product.category_id,
                "user_id": product.user_id,
                "status": product.status,
                "tags": product.tags,
                "attributes": product.attributes,
                "created_at": product.created_at.isoformat(),
                "updated_at": product.updated_at.isoformat()
            }
            
            # Include variants
            if include_variants:
                variants_query = select(ProductVariant).where(
                    ProductVariant.product_id == product_id
                )
                variants_result = await self.session.execute(variants_query)
                variants = variants_result.scalars().all()
                
                result_data["variants"] = [
                    {
                        "variant_id": variant.variant_id,
                        "name": variant.name,
                        "sku": variant.sku,
                        "price": float(variant.price),
                        "attributes": variant.attributes,
                        "inventory": variant.inventory
                    }
                    for variant in variants
                ]
            
            # Include reviews
            if include_reviews:
                reviews_query = select(ProductReview).where(
                    ProductReview.product_id == product_id
                ).order_by(desc(ProductReview.created_at))
                reviews_result = await self.session.execute(reviews_query)
                reviews = reviews_result.scalars().all()
                
                result_data["reviews"] = [
                    {
                        "review_id": review.review_id,
                        "user_id": review.user_id,
                        "rating": review.rating,
                        "title": review.title,
                        "content": review.content,
                        "created_at": review.created_at.isoformat()
                    }
                    for review in reviews
                ]
            
            # Include images
            images_query = select(ProductImage).where(
                ProductImage.product_id == product_id
            ).order_by(ProductImage.sort_order)
            images_result = await self.session.execute(images_query)
            images = images_result.scalars().all()
            
            result_data["images"] = [
                {
                    "image_id": image.image_id,
                    "image_url": image.image_url,
                    "alt_text": image.alt_text,
                    "sort_order": image.sort_order,
                    "is_primary": image.is_primary
                }
                for image in images
            ]
            
            # Include inventory
            inventory_query = select(Inventory).where(
                Inventory.product_id == product_id
            )
            inventory_result = await self.session.execute(inventory_query)
            inventory = inventory_result.scalar_one_or_none()
            
            if inventory:
                result_data["inventory"] = {
                    "quantity": inventory.quantity,
                    "reserved_quantity": inventory.reserved_quantity,
                    "available_quantity": inventory.quantity - inventory.reserved_quantity,
                    "low_stock_threshold": inventory.low_stock_threshold
                }
            
            return {
                "success": True,
                "data": result_data
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get product: {str(e)}")
    
    async def list_products(
        self,
        category_id: Optional[str] = None,
        status: Optional[ProductStatus] = None,
        user_id: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        tags: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """List products with filtering and pagination."""
        try:
            # Build query
            query = select(Product)
            
            if category_id:
                query = query.where(Product.category_id == category_id)
            
            if status:
                query = query.where(Product.status == status.value)
            
            if user_id:
                query = query.where(Product.user_id == user_id)
            
            if min_price:
                query = query.where(Product.price >= min_price)
            
            if max_price:
                query = query.where(Product.price <= max_price)
            
            if tags:
                for tag in tags:
                    query = query.where(Product.tags.contains([tag]))
            
            # Add sorting
            if sort_order.lower() == "desc":
                query = query.order_by(desc(getattr(Product, sort_by)))
            else:
                query = query.order_by(getattr(Product, sort_by))
            
            # Add pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.session.execute(query)
            products = result.scalars().all()
            
            # Get total count
            count_query = select(func.count(Product.id))
            if category_id:
                count_query = count_query.where(Product.category_id == category_id)
            if status:
                count_query = count_query.where(Product.status == status.value)
            if user_id:
                count_query = count_query.where(Product.user_id == user_id)
            if min_price:
                count_query = count_query.where(Product.price >= min_price)
            if max_price:
                count_query = count_query.where(Product.price <= max_price)
            if tags:
                for tag in tags:
                    count_query = count_query.where(Product.tags.contains([tag]))
            
            count_result = await self.session.execute(count_query)
            total_count = count_result.scalar()
            
            # Format results
            formatted_products = []
            for product in products:
                formatted_products.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "description": product.description,
                    "price": float(product.price),
                    "sku": product.sku,
                    "category_id": product.category_id,
                    "user_id": product.user_id,
                    "status": product.status,
                    "tags": product.tags,
                    "created_at": product.created_at.isoformat(),
                    "updated_at": product.updated_at.isoformat()
                })
            
            return {
                "success": True,
                "data": {
                    "products": formatted_products,
                    "total": total_count,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to list products: {str(e)}")
    
    async def add_to_cart(
        self,
        user_id: str,
        product_id: str,
        quantity: int = 1,
        variant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add product to cart."""
        try:
            # Get or create cart
            cart_query = select(Cart).where(
                and_(Cart.user_id == user_id, Cart.status == "active")
            )
            cart_result = await self.session.execute(cart_query)
            cart = cart_result.scalar_one_or_none()
            
            if not cart:
                cart = Cart(
                    user_id=user_id,
                    status="active",
                    created_at=datetime.utcnow()
                )
                self.session.add(cart)
                await self.session.flush()  # Get cart ID
            
            # Check if item already in cart
            existing_item_query = select(CartItem).where(
                and_(
                    CartItem.cart_id == cart.cart_id,
                    CartItem.product_id == product_id,
                    CartItem.variant_id == variant_id
                )
            )
            existing_item_result = await self.session.execute(existing_item_query)
            existing_item = existing_item_result.scalar_one_or_none()
            
            if existing_item:
                # Update quantity
                existing_item.quantity += quantity
                existing_item.updated_at = datetime.utcnow()
            else:
                # Create new cart item
                cart_item = CartItem(
                    cart_id=cart.cart_id,
                    product_id=product_id,
                    variant_id=variant_id,
                    quantity=quantity,
                    created_at=datetime.utcnow()
                )
                self.session.add(cart_item)
            
            await self.session.commit()
            
            return {
                "success": True,
                "cart_id": cart.cart_id,
                "product_id": product_id,
                "quantity": quantity,
                "variant_id": variant_id,
                "message": "Product added to cart successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to add to cart: {str(e)}")
    
    async def get_cart(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get user's cart."""
        try:
            # Get cart
            cart_query = select(Cart).where(
                and_(Cart.user_id == user_id, Cart.status == "active")
            )
            cart_result = await self.session.execute(cart_query)
            cart = cart_result.scalar_one_or_none()
            
            if not cart:
                return {
                    "success": True,
                    "data": {
                        "cart_id": None,
                        "items": [],
                        "total_items": 0,
                        "total_amount": 0.0
                    }
                }
            
            # Get cart items
            items_query = select(CartItem).where(CartItem.cart_id == cart.cart_id)
            items_result = await self.session.execute(items_query)
            items = items_result.scalars().all()
            
            # Format cart items
            formatted_items = []
            total_amount = Decimal('0')
            
            for item in items:
                # Get product details
                product_query = select(Product).where(Product.product_id == item.product_id)
                product_result = await self.session.execute(product_query)
                product = product_result.scalar_one_or_none()
                
                if product:
                    item_total = product.price * item.quantity
                    total_amount += item_total
                    
                    formatted_items.append({
                        "cart_item_id": item.cart_item_id,
                        "product_id": item.product_id,
                        "product_name": product.name,
                        "product_price": float(product.price),
                        "variant_id": item.variant_id,
                        "quantity": item.quantity,
                        "item_total": float(item_total)
                    })
            
            return {
                "success": True,
                "data": {
                    "cart_id": cart.cart_id,
                    "items": formatted_items,
                    "total_items": sum(item.quantity for item in items),
                    "total_amount": float(total_amount)
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get cart: {str(e)}")
    
    async def create_order(
        self,
        user_id: str,
        shipping_address: Dict[str, Any],
        payment_method_id: str,
        coupon_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an order from cart."""
        try:
            # Get cart
            cart_query = select(Cart).where(
                and_(Cart.user_id == user_id, Cart.status == "active")
            )
            cart_result = await self.session.execute(cart_query)
            cart = cart_result.scalar_one_or_none()
            
            if not cart:
                raise ValidationError("No active cart found")
            
            # Get cart items
            items_query = select(CartItem).where(CartItem.cart_id == cart.cart_id)
            items_result = await self.session.execute(items_query)
            items = items_result.scalars().all()
            
            if not items:
                raise ValidationError("Cart is empty")
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Calculate totals
            subtotal = Decimal('0')
            for item in items:
                product_query = select(Product).where(Product.product_id == item.product_id)
                product_result = await self.session.execute(product_query)
                product = product_result.scalar_one_or_none()
                
                if product:
                    subtotal += product.price * item.quantity
            
            # Apply coupon discount
            discount_amount = Decimal('0')
            if coupon_code:
                coupon_query = select(Coupon).where(
                    and_(
                        Coupon.code == coupon_code,
                        Coupon.is_active == True,
                        Coupon.valid_from <= datetime.utcnow(),
                        Coupon.valid_until >= datetime.utcnow()
                    )
                )
                coupon_result = await self.session.execute(coupon_query)
                coupon = coupon_result.scalar_one_or_none()
                
                if coupon:
                    if coupon.discount_type == "percentage":
                        discount_amount = subtotal * (coupon.discount_value / 100)
                    else:
                        discount_amount = coupon.discount_value
            
            # Calculate final total
            total_amount = subtotal - discount_amount
            
            # Create order
            order = Order(
                order_id=order_id,
                user_id=user_id,
                status=OrderStatus.PENDING.value,
                subtotal=subtotal,
                discount_amount=discount_amount,
                total_amount=total_amount,
                shipping_address=shipping_address,
                payment_method_id=payment_method_id,
                created_at=datetime.utcnow()
            )
            
            self.session.add(order)
            
            # Create order items
            for item in items:
                product_query = select(Product).where(Product.product_id == item.product_id)
                product_result = await self.session.execute(product_query)
                product = product_result.scalar_one_or_none()
                
                if product:
                    order_item = OrderItem(
                        order_id=order_id,
                        product_id=item.product_id,
                        variant_id=item.variant_id,
                        quantity=item.quantity,
                        unit_price=product.price,
                        total_price=product.price * item.quantity,
                        created_at=datetime.utcnow()
                    )
                    self.session.add(order_item)
            
            # Update cart status
            cart.status = "completed"
            cart.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            return {
                "success": True,
                "order_id": order_id,
                "total_amount": float(total_amount),
                "message": "Order created successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create order: {str(e)}")
    
    async def process_payment(
        self,
        order_id: str,
        payment_method: PaymentMethodType,
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process payment for an order."""
        try:
            # Get order
            order_query = select(Order).where(Order.order_id == order_id)
            order_result = await self.session.execute(order_query)
            order = order_result.scalar_one_or_none()
            
            if not order:
                raise ValidationError(f"Order with ID {order_id} not found")
            
            # Generate payment ID
            payment_id = str(uuid.uuid4())
            
            # Process payment based on method
            if payment_method == PaymentMethodType.STRIPE:
                payment_result = await self._process_stripe_payment(
                    order, payment_data
                )
            elif payment_method == PaymentMethodType.PAYPAL:
                payment_result = await self._process_paypal_payment(
                    order, payment_data
                )
            else:
                raise ValidationError(f"Unsupported payment method: {payment_method}")
            
            # Create payment record
            payment = Payment(
                payment_id=payment_id,
                order_id=order_id,
                amount=order.total_amount,
                currency="USD",
                payment_method=payment_method.value,
                status=payment_result["status"],
                transaction_id=payment_result.get("transaction_id"),
                gateway_response=payment_result.get("gateway_response", {}),
                created_at=datetime.utcnow()
            )
            
            self.session.add(payment)
            
            # Update order status
            if payment_result["status"] == PaymentStatus.COMPLETED.value:
                order.status = OrderStatus.CONFIRMED.value
            else:
                order.status = OrderStatus.PENDING.value
            
            order.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            return {
                "success": True,
                "payment_id": payment_id,
                "status": payment_result["status"],
                "transaction_id": payment_result.get("transaction_id"),
                "message": "Payment processed successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to process payment: {str(e)}")
    
    async def _process_stripe_payment(
        self,
        order: Order,
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process Stripe payment."""
        try:
            # Create Stripe payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(order.total_amount * 100),  # Convert to cents
                currency='usd',
                metadata={
                    'order_id': order.order_id,
                    'user_id': order.user_id
                }
            )
            
            return {
                "status": PaymentStatus.COMPLETED.value,
                "transaction_id": intent.id,
                "gateway_response": intent
            }
            
        except Exception as e:
            return {
                "status": PaymentStatus.FAILED.value,
                "error": str(e)
            }
    
    async def _process_paypal_payment(
        self,
        order: Order,
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process PayPal payment."""
        try:
            # Create PayPal payment
            payment = paypalrestsdk.Payment({
                "intent": "sale",
                "payer": {
                    "payment_method": "paypal"
                },
                "transactions": [{
                    "amount": {
                        "total": str(order.total_amount),
                        "currency": "USD"
                    },
                    "description": f"Order {order.order_id}"
                }],
                "redirect_urls": {
                    "return_url": payment_data.get("return_url", ""),
                    "cancel_url": payment_data.get("cancel_url", "")
                }
            })
            
            if payment.create():
                return {
                    "status": PaymentStatus.COMPLETED.value,
                    "transaction_id": payment.id,
                    "gateway_response": payment
                }
            else:
                return {
                    "status": PaymentStatus.FAILED.value,
                    "error": payment.error
                }
                
        except Exception as e:
            return {
                "status": PaymentStatus.FAILED.value,
                "error": str(e)
            }
    
    async def create_subscription(
        self,
        user_id: str,
        plan_id: str,
        payment_method_id: str
    ) -> Dict[str, Any]:
        """Create a subscription."""
        try:
            # Get subscription plan
            plan_query = select(SubscriptionPlan).where(SubscriptionPlan.plan_id == plan_id)
            plan_result = await self.session.execute(plan_query)
            plan = plan_result.scalar_one_or_none()
            
            if not plan:
                raise ValidationError(f"Subscription plan with ID {plan_id} not found")
            
            # Generate subscription ID
            subscription_id = str(uuid.uuid4())
            
            # Create subscription
            subscription = Subscription(
                subscription_id=subscription_id,
                user_id=user_id,
                plan_id=plan_id,
                status=SubscriptionStatus.ACTIVE.value,
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=plan.billing_cycle_days),
                payment_method_id=payment_method_id,
                created_at=datetime.utcnow()
            )
            
            self.session.add(subscription)
            await self.session.commit()
            
            return {
                "success": True,
                "subscription_id": subscription_id,
                "plan_name": plan.name,
                "amount": float(plan.price),
                "billing_cycle": plan.billing_cycle,
                "message": "Subscription created successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create subscription: {str(e)}")
    
    async def get_ecommerce_stats(self) -> Dict[str, Any]:
        """Get e-commerce statistics."""
        try:
            # Get total products
            products_query = select(func.count(Product.id))
            products_result = await self.session.execute(products_query)
            total_products = products_result.scalar()
            
            # Get total orders
            orders_query = select(func.count(Order.id))
            orders_result = await self.session.execute(orders_query)
            total_orders = orders_result.scalar()
            
            # Get total revenue
            revenue_query = select(func.sum(Order.total_amount))
            revenue_result = await self.session.execute(revenue_query)
            total_revenue = revenue_result.scalar() or Decimal('0')
            
            # Get active subscriptions
            subscriptions_query = select(func.count(Subscription.id)).where(
                Subscription.status == SubscriptionStatus.ACTIVE.value
            )
            subscriptions_result = await self.session.execute(subscriptions_query)
            active_subscriptions = subscriptions_result.scalar()
            
            # Get products by status
            status_query = select(
                Product.status,
                func.count(Product.id).label('count')
            ).group_by(Product.status)
            
            status_result = await self.session.execute(status_query)
            products_by_status = {row[0]: row[1] for row in status_result}
            
            # Get orders by status
            order_status_query = select(
                Order.status,
                func.count(Order.id).label('count')
            ).group_by(Order.status)
            
            order_status_result = await self.session.execute(order_status_query)
            orders_by_status = {row[0]: row[1] for row in order_status_result}
            
            return {
                "success": True,
                "data": {
                    "total_products": total_products,
                    "total_orders": total_orders,
                    "total_revenue": float(total_revenue),
                    "active_subscriptions": active_subscriptions,
                    "products_by_status": products_by_status,
                    "orders_by_status": orders_by_status,
                    "cache_size": len(self.ecommerce_cache)
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get e-commerce stats: {str(e)}")
























