'use client';

import { motion } from 'framer-motion';
import ProductCard from './ProductCard';
import { cn } from '@/lib/utils/cn';

interface Product {
  id: string;
  title: string;
  description?: string;
  price: number;
  image?: string;
  badge?: string;
  featured?: boolean;
}

interface ProductGridProps {
  products: Product[];
  columns?: 2 | 3 | 4;
  onAddToCart?: (productId: string) => void;
  onView?: (productId: string) => void;
  onFavorite?: (productId: string) => void;
  favorites?: string[];
  className?: string;
}

export default function ProductGrid({
  products,
  columns = 4,
  onAddToCart,
  onView,
  onFavorite,
  favorites = [],
  className,
}: ProductGridProps) {
  const gridCols = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={cn('grid gap-6', gridCols[columns], className)}>
      {products.map((product, index) => (
        <motion.div
          key={product.id}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: index * 0.1 }}
        >
          <ProductCard
            {...product}
            onAddToCart={onAddToCart ? () => onAddToCart(product.id) : undefined}
            onView={onView ? () => onView(product.id) : undefined}
            onFavorite={onFavorite ? () => onFavorite(product.id) : undefined}
            isFavorite={favorites.includes(product.id)}
          />
        </motion.div>
      ))}
    </div>
  );
}



