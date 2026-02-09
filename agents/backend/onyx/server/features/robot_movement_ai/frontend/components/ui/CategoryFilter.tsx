'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';
import { Button } from './Button';

interface Category {
  id: string;
  label: string;
  count?: number;
}

interface CategoryFilterProps {
  categories: Category[];
  selectedCategory?: string;
  onCategoryChange: (categoryId: string) => void;
  className?: string;
  variant?: 'default' | 'pills' | 'tabs';
}

export default function CategoryFilter({
  categories,
  selectedCategory,
  onCategoryChange,
  className,
  variant = 'default',
}: CategoryFilterProps) {
  if (variant === 'pills') {
    return (
      <div className={cn('flex flex-wrap gap-2', className)}>
        {categories.map((category) => (
          <Button
            key={category.id}
            variant={selectedCategory === category.id ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => onCategoryChange(category.id)}
            className={cn(
              'rounded-full',
              selectedCategory === category.id && 'bg-tesla-blue text-white'
            )}
          >
            {category.label}
            {category.count !== undefined && (
              <span className="ml-2 px-2 py-0.5 bg-white/20 rounded-full text-xs">
                {category.count}
              </span>
            )}
          </Button>
        ))}
      </div>
    );
  }

  if (variant === 'tabs') {
    return (
      <div className={cn('flex border-b border-gray-200 overflow-x-auto scrollbar-hide', className)}>
        {categories.map((category) => (
          <button
            key={category.id}
            onClick={() => onCategoryChange(category.id)}
            className={cn(
              'px-6 py-4 font-medium text-sm transition-colors whitespace-nowrap border-b-2 min-h-[44px]',
              selectedCategory === category.id
                ? 'border-tesla-blue text-tesla-blue'
                : 'border-transparent text-tesla-gray-dark hover:text-tesla-black hover:border-gray-300'
            )}
          >
            {category.label}
            {category.count !== undefined && (
              <span className="ml-2 text-xs opacity-70">({category.count})</span>
            )}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className={cn('space-y-2', className)}>
      {categories.map((category) => (
        <motion.button
          key={category.id}
          onClick={() => onCategoryChange(category.id)}
          whileHover={{ x: 4 }}
          whileTap={{ scale: 0.98 }}
          className={cn(
            'w-full text-left px-4 py-3 rounded-md transition-colors min-h-[44px]',
            selectedCategory === category.id
              ? 'bg-tesla-blue text-white'
              : 'bg-gray-50 text-tesla-black hover:bg-gray-100'
          )}
        >
          <div className="flex items-center justify-between">
            <span className="font-medium">{category.label}</span>
            {category.count !== undefined && (
              <span
                className={cn(
                  'text-xs px-2 py-1 rounded-full',
                  selectedCategory === category.id
                    ? 'bg-white/20 text-white'
                    : 'bg-gray-200 text-tesla-gray-dark'
                )}
              >
                {category.count}
              </span>
            )}
          </div>
        </motion.button>
      ))}
    </div>
  );
}



