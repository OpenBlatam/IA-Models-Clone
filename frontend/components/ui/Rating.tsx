'use client';

import { useState } from 'react';
import { FiStar } from 'react-icons/fi';
import { cn } from '@/utils/classNames';

interface RatingProps {
  value?: number;
  onChange?: (value: number) => void;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  readonly?: boolean;
  className?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
};

export function Rating({
  value = 0,
  onChange,
  max = 5,
  size = 'md',
  readonly = false,
  className,
}: RatingProps) {
  const [hoverValue, setHoverValue] = useState(0);

  const handleClick = (index: number) => {
    if (!readonly && onChange) {
      onChange(index + 1);
    }
  };

  const handleMouseEnter = (index: number) => {
    if (!readonly) {
      setHoverValue(index + 1);
    }
  };

  const handleMouseLeave = () => {
    if (!readonly) {
      setHoverValue(0);
    }
  };

  const displayValue = hoverValue || value;

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {Array.from({ length: max }).map((_, index) => {
        const isFilled = index < displayValue;
        return (
          <button
            key={index}
            type="button"
            onClick={() => handleClick(index)}
            onMouseEnter={() => handleMouseEnter(index)}
            onMouseLeave={handleMouseLeave}
            disabled={readonly}
            className={cn(
              'transition-colors',
              !readonly && 'cursor-pointer hover:scale-110',
              readonly && 'cursor-default'
            )}
            aria-label={`Calificar ${index + 1} de ${max}`}
          >
            <FiStar
              size={size === 'sm' ? 16 : size === 'md' ? 20 : 24}
              className={cn(
                sizeClasses[size],
                isFilled
                  ? 'fill-yellow-400 text-yellow-400'
                  : 'text-gray-300 dark:text-gray-600'
              )}
            />
          </button>
        );
      })}
    </div>
  );
}

