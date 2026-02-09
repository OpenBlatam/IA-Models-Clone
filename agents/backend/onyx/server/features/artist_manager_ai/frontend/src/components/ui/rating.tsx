'use client';

import { Star } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useState } from 'react';

interface RatingProps {
  value?: number;
  onChange?: (value: number) => void;
  max?: number;
  readonly?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
};

const Rating = ({
  value = 0,
  onChange,
  max = 5,
  readonly = false,
  size = 'md',
  className,
}: RatingProps) => {
  const [hoverValue, setHoverValue] = useState<number | null>(null);

  const handleClick = (newValue: number) => {
    if (!readonly && onChange) {
      onChange(newValue);
    }
  };

  const handleMouseEnter = (newValue: number) => {
    if (!readonly) {
      setHoverValue(newValue);
    }
  };

  const handleMouseLeave = () => {
    if (!readonly) {
      setHoverValue(null);
    }
  };

  const displayValue = hoverValue ?? value;

  return (
    <div className={cn('flex items-center gap-1', className)} role="img" aria-label={`Calificación: ${value} de ${max}`}>
      {Array.from({ length: max }, (_, i) => i + 1).map((star) => (
        <button
          key={star}
          type="button"
          onClick={() => handleClick(star)}
          onMouseEnter={() => handleMouseEnter(star)}
          onMouseLeave={handleMouseLeave}
          disabled={readonly}
          className={cn(
            'transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 rounded',
            readonly ? 'cursor-default' : 'cursor-pointer',
            star <= displayValue ? 'text-yellow-400' : 'text-gray-300'
          )}
          aria-label={`${star} estrella${star > 1 ? 's' : ''}`}
        >
          <Star className={cn(sizeClasses[size], 'fill-current')} />
        </button>
      ))}
    </div>
  );
};

export { Rating };

