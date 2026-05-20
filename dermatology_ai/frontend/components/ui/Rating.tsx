'use client';

import React, { useState } from 'react';
import { Star } from 'lucide-react';
import { clsx } from 'clsx';

interface RatingProps {
  value: number;
  onChange?: (value: number) => void;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  readonly?: boolean;
  className?: string;
}

export const Rating: React.FC<RatingProps> = ({
  value,
  onChange,
  max = 5,
  size = 'md',
  readonly = false,
  className,
}) => {
  const [hoverValue, setHoverValue] = useState<number | null>(null);

  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
  };

  const displayValue = hoverValue ?? value;

  return (
    <div className={clsx('flex items-center space-x-1', className)}>
      {Array.from({ length: max }).map((_, index) => {
        const starValue = index + 1;
        const isFilled = starValue <= displayValue;

        return (
          <button
            key={index}
            type="button"
            onClick={() => !readonly && onChange?.(starValue)}
            onMouseEnter={() => !readonly && setHoverValue(starValue)}
            onMouseLeave={() => !readonly && setHoverValue(null)}
            disabled={readonly}
            className={clsx(
              'transition-colors',
              !readonly && 'cursor-pointer hover:scale-110',
              readonly && 'cursor-default'
            )}
            aria-label={`Calificar ${starValue} de ${max}`}
          >
            <Star
              className={clsx(
                sizes[size],
                isFilled
                  ? 'fill-yellow-400 text-yellow-400'
                  : 'fill-gray-200 text-gray-300 dark:fill-gray-700 dark:text-gray-600'
              )}
            />
          </button>
        );
      })}
    </div>
  );
};


