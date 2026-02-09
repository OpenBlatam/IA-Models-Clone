/**
 * Rating component
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils/cn';
import { Star } from 'lucide-react';

export interface RatingProps {
  value: number;
  onChange?: (value: number) => void;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  readonly?: boolean;
  className?: string;
  showValue?: boolean;
}

const sizeClasses = {
  sm: 'h-4 w-4',
  md: 'h-5 w-5',
  lg: 'h-6 w-6',
};

export const Rating: React.FC<RatingProps> = ({
  value,
  onChange,
  max = 5,
  size = 'md',
  disabled = false,
  readonly = false,
  className,
  showValue = false,
}) => {
  const handleClick = (newValue: number) => {
    if (readonly || disabled || !onChange) {
      return;
    }
    onChange(newValue);
  };

  const handleKeyDown = (event: React.KeyboardEvent, newValue: number) => {
    if (readonly || disabled || !onChange) {
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onChange(newValue);
    }
  };

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div
        className="flex items-center gap-1"
        role={readonly ? 'img' : 'radiogroup'}
        aria-label={`Calificación: ${value} de ${max}`}
        aria-readonly={readonly}
      >
        {Array.from({ length: max }, (_, index) => {
          const starValue = index + 1;
          const isFilled = starValue <= value;
          const isHalf = starValue === Math.ceil(value) && value % 1 !== 0;

          return (
            <button
              key={starValue}
              type="button"
              onClick={() => handleClick(starValue)}
              onKeyDown={(e) => handleKeyDown(e, starValue)}
              disabled={readonly || disabled}
              className={cn(
                'transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded',
                !readonly && !disabled && 'cursor-pointer hover:scale-110',
                (readonly || disabled) && 'cursor-default'
              )}
              aria-label={`Calificar ${starValue} de ${max}`}
              tabIndex={readonly || disabled ? -1 : 0}
            >
              <Star
                className={cn(
                  sizeClasses[size],
                  isFilled ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'
                )}
                aria-hidden="true"
              />
            </button>
          );
        })}
      </div>
      {showValue && (
        <span className="text-sm text-muted-foreground">
          {value.toFixed(1)} / {max}
        </span>
      )}
    </div>
  );
};



