'use client';

import { motion } from 'framer-motion';
import { ShoppingCart, Heart, Eye } from 'lucide-react';
import { Button } from './Button';
import { Badge } from './Badge';
import { cn } from '@/lib/utils/cn';
// Note: Using regular img tag instead of Next.js Image for flexibility

interface ProductCardProps {
  id: string;
  title: string;
  description?: string;
  price: number;
  image?: string;
  badge?: string;
  onAddToCart?: () => void;
  onView?: () => void;
  onFavorite?: () => void;
  isFavorite?: boolean;
  className?: string;
  featured?: boolean;
}

export default function ProductCard({
  id,
  title,
  description,
  price,
  image,
  badge,
  onAddToCart,
  onView,
  onFavorite,
  isFavorite = false,
  className,
  featured = false,
}: ProductCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -8, scale: 1.02 }}
      transition={{ duration: 0.3 }}
      className={cn(
        'group relative bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden transition-all duration-300',
        'hover:shadow-tesla-lg hover:border-gray-300',
        featured && 'ring-2 ring-tesla-blue/20',
        className
      )}
    >
      {/* Badge */}
      {badge && (
        <div className="absolute top-4 left-4 z-10">
          <Badge variant="default" className="bg-tesla-blue text-white">
            {badge}
          </Badge>
        </div>
      )}

      {/* Favorite Button */}
      {onFavorite && (
        <button
          onClick={onFavorite}
          className="absolute top-4 right-4 z-10 p-2 bg-white/90 backdrop-blur-sm rounded-full shadow-sm hover:bg-white transition-all min-h-[44px] min-w-[44px] flex items-center justify-center"
          aria-label={isFavorite ? 'Quitar de favoritos' : 'Añadir a favoritos'}
        >
          <Heart
            className={cn(
              'w-5 h-5 transition-colors',
              isFavorite ? 'fill-red-500 text-red-500' : 'text-tesla-gray-dark hover:text-red-500'
            )}
          />
        </button>
      )}

      {/* Image */}
      <div className="relative w-full h-64 bg-gray-100 overflow-hidden">
        {image ? (
          <img
            src={image}
            alt={title}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-tesla-gray-light">
            <Eye className="w-16 h-16" />
          </div>
        )}
        
        {/* Overlay on Hover */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/0 to-black/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        
        {/* Quick View Button */}
        {onView && (
          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <Button
              variant="secondary"
              onClick={onView}
              className="bg-white/95 backdrop-blur-sm hover:bg-white"
            >
              <Eye className="w-4 h-4 mr-2" />
              Ver Detalles
            </Button>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-6">
        <h3 className="text-lg font-semibold text-tesla-black mb-2 line-clamp-2 group-hover:text-tesla-blue transition-colors">
          {title}
        </h3>
        
        {description && (
          <p className="text-sm text-tesla-gray-dark mb-4 line-clamp-2">
            {description}
          </p>
        )}

        <div className="flex items-center justify-between">
          <div>
            <p className="text-2xl font-bold text-tesla-black">
              ${price.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          
          {onAddToCart && (
            <Button
              variant="primary"
              size="sm"
              onClick={onAddToCart}
              className="flex items-center gap-2"
            >
              <ShoppingCart className="w-4 h-4" />
              <span className="hidden sm:inline">Añadir</span>
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

