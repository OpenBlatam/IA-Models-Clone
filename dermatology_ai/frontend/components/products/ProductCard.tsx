'use client';

import React, { memo } from 'react';
import { Card, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { ProductInfo } from '@/lib/types/api';
import { Star, ShoppingBag } from 'lucide-react';
import { clsx } from 'clsx';

interface ProductCardProps {
  product: ProductInfo;
  onViewDetails?: (productId: string) => void;
  className?: string;
}

export const ProductCard: React.FC<ProductCardProps> = memo(({
  product,
  onViewDetails,
  className,
}) => {
  return (
    <Card className={clsx('hover:shadow-lg transition-shadow', className)}>
      <CardContent className="p-6">
        <div className="space-y-4">
          {/* Header */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              {product.name}
            </h3>
            {product.brand && (
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {product.brand}
              </p>
            )}
            {product.category && (
              <Badge variant="default" size="sm" className="mb-2">
                {product.category}
              </Badge>
            )}
          </div>

          {/* Description */}
          {product.description && (
            <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-3">
              {product.description}
            </p>
          )}

          {/* Ingredients */}
          {product.ingredients && product.ingredients.length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Main ingredients:
              </p>
              <div className="flex flex-wrap gap-1">
                {product.ingredients.slice(0, 3).map((ingredient, index) => (
                  <Badge key={index} variant="outline" size="sm">
                    {ingredient}
                  </Badge>
                ))}
                {product.ingredients.length > 3 && (
                  <Badge variant="outline" size="sm">
                    +{product.ingredients.length - 3} more
                  </Badge>
                )}
              </div>
            </div>
          )}

          {/* Price and Rating */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
            <div>
              <p className="text-lg font-bold text-gray-900 dark:text-white">
                ${product.price?.toFixed(2) || 'N/A'}
              </p>
              {product.rating && (
                <div className="flex items-center space-x-1 mt-1">
                  <Star className="h-4 w-4 text-yellow-400 fill-current" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {product.rating.toFixed(1)}
                    {product.reviews_count && ` (${product.reviews_count})`}
                  </span>
                </div>
              )}
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onViewDetails?.(product.product_id)}
            >
              View Details
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

ProductCard.displayName = 'ProductCard';

