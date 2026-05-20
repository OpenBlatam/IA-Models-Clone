'use client';

import { Star } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import type { StarRatingProps } from '@/lib/types/components';

const sizeClasses = {
  sm: 'h-4 w-4',
  md: 'h-5 w-5',
  lg: 'h-6 w-6',
};

export const StarRating = ({
  rating,
  onRatingChange,
  readonly = false,
  size = 'md',
  className,
}: StarRatingProps): JSX.Element => {
  const handleClick = (value: number): void => {
    if (!readonly && onRatingChange) {
      onRatingChange(value);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent, value: number): void => {
    if (!readonly && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      handleClick(value);
    }
  };

  return (
    <div className={cn('flex space-x-1', className)} role={readonly ? 'img' : undefined} aria-label={`Calificación: ${rating} de 5`}>
      {[1, 2, 3, 4, 5].map((value) => (
        <button
          key={value}
          type="button"
          onClick={() => handleClick(value)}
          onKeyDown={(e) => handleKeyDown(e, value)}
          disabled={readonly}
          className={cn(
            'transition-colors',
            readonly ? 'cursor-default' : 'cursor-pointer hover:scale-110',
            rating >= value
              ? 'text-yellow-500'
              : 'text-gray-300 hover:text-yellow-400'
          )}
          aria-label={readonly ? undefined : `Calificar con ${value} estrellas`}
          tabIndex={readonly ? -1 : 0}
        >
          <Star className={cn(sizeClasses[size], 'fill-current')} />
        </button>
      ))}
    </div>
  );
};

