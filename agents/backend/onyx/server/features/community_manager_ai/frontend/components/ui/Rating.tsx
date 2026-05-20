'use client';

import { Star } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

interface RatingProps {
  value?: number;
  onChange?: (value: number) => void;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  readonly?: boolean;
  className?: string;
  showLabel?: boolean;
}

const sizeClasses = {
  sm: 'h-4 w-4',
  md: 'h-5 w-5',
  lg: 'h-6 w-6',
};

export const Rating = ({
  value = 0,
  onChange,
  max = 5,
  size = 'md',
  readonly = false,
  className,
  showLabel = false,
}: RatingProps) => {
  const [hoveredValue, setHoveredValue] = useState<number | null>(null);

  const displayValue = hoveredValue ?? value;

  const handleClick = (newValue: number) => {
    if (!readonly && onChange) {
      onChange(newValue);
    }
  };

  const handleMouseEnter = (newValue: number) => {
    if (!readonly) {
      setHoveredValue(newValue);
    }
  };

  const handleMouseLeave = () => {
    if (!readonly) {
      setHoveredValue(null);
    }
  };

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <div className="flex items-center gap-0.5" role="radiogroup" aria-label="Rating">
        {Array.from({ length: max }).map((_, index) => {
          const starValue = index + 1;
          const isFilled = starValue <= displayValue;

          return (
            <button
              key={index}
              type="button"
              onClick={() => handleClick(starValue)}
              onMouseEnter={() => handleMouseEnter(starValue)}
              onMouseLeave={handleMouseLeave}
              disabled={readonly}
              className={cn(
                'transition-colors',
                !readonly && 'cursor-pointer hover:scale-110',
                readonly && 'cursor-default'
              )}
              aria-label={`${starValue} estrellas`}
              aria-pressed={isFilled}
            >
              <Star
                className={cn(
                  sizeClasses[size],
                  isFilled
                    ? 'fill-yellow-400 text-yellow-400'
                    : 'fill-gray-200 text-gray-300 dark:fill-gray-700 dark:text-gray-600'
                )}
              />
            </button>
          );
        })}
      </div>
      {showLabel && value > 0 && (
        <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
          {value.toFixed(1)}
        </span>
      )}
    </div>
  );
};



